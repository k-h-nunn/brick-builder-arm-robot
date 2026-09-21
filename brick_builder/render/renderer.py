"""Offscreen RGB + depth + instance-segmentation rendering.

Two render passes per call, per
section 5: `mujoco.Renderer` cannot do depth and segmentation at once (its two modes
are mutually exclusive), so this module works directly on `GLContext`, `MjvScene`
and `MjrContext` instead. Pass 1 renders normally and reads RGB and depth from the
same framebuffer with a single `mjr_readPixels` call. Pass 2 turns on the
`mjRND_SEGMENT` / `mjRND_IDCOLOR` scene flags, which paint each geom's segmentation
id into the colour buffer as an RGB triple; that triple is decoded back into a
per-pixel body id, then remapped through a user-supplied lookup table to whatever
instance ids the caller cares about.

The GL context is created lazily, on first use, never at import time or in
`__init__` -- this matters for `gymnasium.vector.AsyncVectorEnv`, which forks
worker processes and must not inherit an open GL context from the parent.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Geom group used for collision-only primitives across brick_builder's MJCF (robot and
# brick models both follow the so101 convention: group 2 = visual, group 3 =
# collision). Hidden from every render pass.
COLLISION_GEOM_GROUP = 3


@dataclass
class Frame:
    """One rendered observation."""

    rgb: np.ndarray        # uint8 [H, W, 3]
    depth: np.ndarray      # float32 [H, W], metres
    instance: np.ndarray   # int32 [H, W]; -1 = background, else the caller's instance id
    body_ids: dict         # the body-name -> instance-id map active when this frame was rendered


class SceneRenderer:
    """Renders RGB, metric depth and an instance-segmentation map from an MjModel.

    One `SceneRenderer` owns one offscreen GL context and one `MjrContext`, both
    sized for `width` x `height`. Call `close()` (or use it as a context manager)
    to release them; letting the object get garbage-collected also frees them, but
    that can happen after the GL context is gone, so prefer an explicit close.
    """

    def __init__(self, model: mujoco.MjModel, width: int = 256, height: int = 256):
        self.model = model
        self.width = int(width)
        self.height = int(height)

        self._gl_context = None
        self._mjr_context = None
        self._scene: mujoco.MjvScene | None = None
        self._scene_option: mujoco.MjvOption | None = None
        self._camera = mujoco.MjvCamera()
        self._rect = mujoco.MjrRect(0, 0, self.width, self.height)
        self._lut = np.zeros(model.nbody, dtype=np.int32)
        self._name_to_id: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    def _ensure_context(self) -> None:
        if self._gl_context is not None:
            return

        buffer_w = self.model.vis.global_.offwidth
        buffer_h = self.model.vis.global_.offheight
        if self.width > buffer_w or self.height > buffer_h:
            raise ValueError(
                f"requested {self.width}x{self.height} exceeds the model's offscreen "
                f"framebuffer {buffer_w}x{buffer_h}; raise <visual><global offwidth=.. "
                "offheight=../> in the MJCF."
            )

        self._gl_context = mujoco.GLContext(self.width, self.height)
        self._gl_context.make_current()
        self._mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
        # Reverse-Z depth buffer; matches the conversion used in _depth_to_metres.
        self._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR

        self._scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[COLLISION_GEOM_GROUP] = 0

    def close(self) -> None:
        if self._mjr_context is not None:
            self._mjr_context.free()
            self._mjr_context = None
        if self._gl_context is not None:
            self._gl_context.free()
            self._gl_context = None
        self._scene = None
        self._scene_option = None

    def __enter__(self) -> "SceneRenderer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- configuration -------------------------------------------------

    def instance_lut(self, names_to_ids: dict[str, int]) -> None:
        """Sets the body-name -> instance-id map used by later `render` calls.

        Bodies not present in `names_to_ids` render with instance id 0. Background
        pixels (no geom at all) always render as -1, regardless of this map.
        """
        self._ensure_context()
        self._lut[:] = 0
        for name, instance_id in names_to_ids.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                raise ValueError(f"unknown body name {name!r}")
            self._lut[body_id] = instance_id
        self._name_to_id = dict(names_to_ids)

    # -- rendering -------------------------------------------------------

    def render(self, data: mujoco.MjData, camera: str) -> Frame:
        """Renders one frame from the named fixed camera."""
        self._ensure_context()
        self._gl_context.make_current()

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id == -1:
            raise ValueError(f"unknown camera {camera!r}")
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = cam_id

        rgb, depth_ndc = self._render_pass(data, segment=False)
        depth = self._depth_to_metres(depth_ndc)
        instance = self._render_segmentation(data)

        # EGL/OSMesa frame buffers read out bottom-to-top.
        rgb = np.flipud(rgb).copy()
        depth = np.flipud(depth).copy()
        instance = np.flipud(instance).copy()

        return Frame(rgb=rgb, depth=depth, instance=instance, body_ids=dict(self._name_to_id))

    def _update_scene(self, data: mujoco.MjData) -> None:
        mujoco.mjv_updateScene(
            self.model,
            data,
            self._scene_option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )

    def _render_pass(self, data: mujoco.MjData, segment: bool) -> tuple[np.ndarray, np.ndarray]:
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = segment
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = segment
        self._update_scene(data)
        mujoco.mjr_render(self._rect, self._scene, self._mjr_context)
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        depth = np.empty((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, self._rect, self._mjr_context)
        return rgb, depth

    def _render_segmentation(self, data: mujoco.MjData) -> np.ndarray:
        seg_rgb, _ = self._render_pass(data, segment=True)
        # Undo the flags so a caller inspecting `self._scene` afterward sees the
        # normal render state.
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False

        seg32 = seg_rgb.astype(np.uint32)
        segid_plus_one = seg32[..., 0] + seg32[..., 1] * (2**8) + seg32[..., 2] * (2**16)

        ngeom = self._scene.ngeom
        # table[0] (segid+1 == 0) is background -> -1. Other rows are filled below.
        table = np.full(ngeom + 1, -1, dtype=np.int32)
        for g in self._scene.geoms[:ngeom]:
            if g.segid == -1:
                continue
            pixel_val = g.segid + 1
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM.value:
                body_id = self.model.geom_bodyid[g.objid]
                table[pixel_val] = self._lut[body_id]
            # Non-geom renderables (e.g. skin) have no body to map; leave as background.

        # Clamp out-of-range pixel values defensively; they should not occur.
        safe = np.clip(segid_plus_one, 0, ngeom)
        return table[safe]

    def _depth_to_metres(self, depth_ndc: np.ndarray) -> np.ndarray:
        """Converts the reverse-Z NDC depth buffer to metric depth (metres).

        Mirrors the conversion in `mujoco.Renderer` (reverse-Z / `mjDEPTH_ZEROFAR`):
        undoes the OpenGL perspective divide using the model's near/far clip planes
        (`<visual><map znear/zfar>`, scaled by `model.stat.extent`).
        """
        extent = self.model.stat.extent
        near = np.float32(self.model.vis.map.znear * extent)
        far = np.float32(self.model.vis.map.zfar * extent)

        c = -(far + near) / (far - near)
        d = -(np.float32(2.0) * far * near) / (far - near)
        # Reverse-Z transform of the perspective matrix coefficients.
        c = np.float32(-0.5) * c - np.float32(0.5)
        d = np.float32(-0.5) * d

        depth64 = depth_ndc.astype(np.float64)
        metres = d / (depth64 + c)
        return metres.astype(np.float32)


def plot_contact_sheet(frame: Frame, out_path, title: str = "SceneRenderer contact sheet"):
    """Saves a PNG with RGB | instance | depth side by side for one `Frame`.

    This is the figure-producing function `scripts/make_media.py` calls for this
    package's contact sheet, and what
    `docs/media/render/contact_sheet.png` is built from.
    """
    import matplotlib

    matplotlib.use("Agg")  # no GUI windows in tests or batch media builds
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=150)

    axes[0].imshow(frame.rgb)
    axes[0].set_title("RGB")

    instance_img = axes[1].imshow(frame.instance, cmap="tab20", interpolation="nearest")
    axes[1].set_title("Instance")
    fig.colorbar(instance_img, ax=axes[1], fraction=0.046, pad=0.04)

    depth_img = axes[2].imshow(frame.depth, cmap="viridis")
    axes[2].set_title("Depth (m)")
    fig.colorbar(depth_img, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
