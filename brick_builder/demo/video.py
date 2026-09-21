"""Native-resolution recording for the Brick Builder Arm Robot.

This module renders the picture; :mod:`brick_builder.demo.video_panels` draws every piece of text on
top of it.  Together they are the whole video toolkit, and ``scripts/make_demo_video.py`` is the
one program that drives them.

Three rules decide how this file is written, because breaking any one of them is what made the
earlier attempt unreadable:

1. **Nothing is ever enlarged.**  The simulator renders straight into the output resolution
   (2560x1440 for the final quality), every array that comes back is checked against that exact
   shape, and the only resize in the file is :func:`downscale`, which refuses to make an image
   bigger.  The 1920x1080 file is written frame by frame from the 1440p master, in the same pass,
   so there is no intermediate file of any kind between them.
2. **The studio is light.**  :func:`studio_look` turns the scene's two lamps into directional
   lights, raises the ambient term, lightens the floor and widens it, so there is no black void
   and no hot spotlight.  The backdrop behind the floor is a grey-to-white gradient painted into
   the pixels the camera did not hit, which the instance-segmentation pass identifies exactly.
3. **The subject fills the frame, the arm stays inside it, and the camera keeps off the arm's
   own line.**  :func:`subject_span` measures, on every single frame, how wide the arm, the tray
   and the baseplate are in pixels; :func:`frame_overrun` measures how far outside the frame a
   given set of points reaches; :func:`ray_angle_deg` measures how far the line of sight is from
   the shoulder-to-wrist line.  :class:`Recorder` refuses to write a frame that fails any of the
   three.  The first two pull against each other - this arm, its tray and its baseplate together
   are as tall as they are wide, so a 16:9 frame that holds all of the arm cannot also have it
   span two thirds of the width - and the first version of the video resolved that the wrong way: the
   shoulder and the upper arm were out of shot for the whole take and the machine read as two
   disconnected yellow objects.  The third rule is what the second cut was missing: it sat almost
   on the arm's own line, so the links collapsed into one unreadable vertical stack of yellow and
   the gripper disappeared behind the arm's own body at the moment it closed on the brick.

Everything here only reads the physics modules (``brick_builder.scene``, ``brick_builder.render``); a video
is driven by a motion generator that yields ``(scene, phase, data)`` once per control tick, so the
toolkit never has an opinion about how the arm is moved.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from brick_builder.render.renderer import COLLISION_GEOM_GROUP, Frame, SceneRenderer
from brick_builder.scene.compose import TABLE_GEOM, Scene

from brick_builder.demo.video_panels import Overlay, Readout, Rect

__all__ = [
    "Quality", "QUALITIES", "quality_by_name",
    "studio_look", "FreeCameraRenderer", "CamKey", "interp_camera", "ease",
    "backdrop", "floor_grid", "subject_points", "subject_span", "arm_points", "frame_overrun",
    "ray_angle_deg", "downscale", "VideoWriter", "Recorder", "RecordingResult", "Tick", "ReadoutSpec",
    "CloseUp", "panel_size_for",
]


# ============================================================================ quality presets ===


@dataclass(frozen=True)
class Quality:
    """One recording preset.  Both presets render at their own native size; neither is upscaled.

    ``hd1080`` is the deliverable: 1920x1080 native, and the only encode.
    ``draft`` is the same code path at a smaller native size, for the test suite and for checking
    choreography quickly -- it is a *smaller render*, never a shrunken final blown back up.
    """

    name: str
    width: int
    height: int
    fps: int
    msaa: int
    shadowsize: int
    crf: int
    preset: str
    export_1080: bool
    pip_width: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


QUALITIES: dict[str, Quality] = {
    # 1920x1080 native, and the only encode this video needs: every text element is sized as a
    # fraction of frame height, so the labels stay exactly as readable here as at 1440p while the
    # frame costs about half as much to render.
    "hd1080": Quality("hd1080", 1920, 1080, 30, msaa=4, shadowsize=4096, crf=18, preset="medium",
                      export_1080=False, pip_width=480),
    "draft": Quality("draft", 1280, 720, 30, msaa=4, shadowsize=2048, crf=20, preset="veryfast",
                     export_1080=False, pip_width=320),
}


def quality_by_name(name: str) -> Quality:
    if name not in QUALITIES:
        raise ValueError(f"unknown quality {name!r}; choose from {sorted(QUALITIES)}")
    return QUALITIES[name]


# ============================================================================== the studio look ===


FIELD_OF_VIEW = 35.0
"""The free camera's field of view, degrees.  Wide enough to keep the arm, the tray and the
baseplate in one frame, narrow enough that the nearest brick is not stretched."""


def studio_look(model: mujoco.MjModel, quality: Quality) -> None:
    """Light neutral studio, at full output resolution.  Call before the first render.

    Every write here is a plain numeric field on an already-compiled model -- the same kind of
    write ``brick_builder.scene.randomize`` does for lights and colours -- so nothing needs recompiling.
    The offscreen framebuffer, the MSAA sample count and the shadow map are only read once, when
    :class:`SceneRenderer` builds its GL context on the first ``render()``.
    """

    vis = model.vis
    vis.global_.offwidth = max(int(vis.global_.offwidth), int(quality.width))
    vis.global_.offheight = max(int(vis.global_.offheight), int(quality.height))
    if vis.global_.offwidth < quality.width or vis.global_.offheight < quality.height:
        raise RuntimeError("offscreen framebuffer smaller than the output size")
    vis.global_.fovy = FIELD_OF_VIEW
    vis.quality.offsamples = int(quality.msaa)
    vis.quality.shadowsize = max(int(quality.shadowsize), 8192)
    # A directional light's shadow map is stretched over the whole model extent, and this floor
    # is two metres across for a scene 30 cm wide. At that ratio the depth comparison lands
    # inside its own quantisation step across most of the table and the floor breaks into the
    # dotted moire that used to fill the left of the frame -- shadow acne, not a texture and not
    # the painted grid. Pulling the shadow volume in around the subject puts the same number of
    # shadow texels on a much smaller patch and the floor goes clean.
    vis.map.shadowclip = 0.22
    vis.map.shadowscale = 0.35
    vis.map.znear = min(float(vis.map.znear), 0.004)

    # Moderate ambient everywhere: no surface of the arm is ever black, and the shadows stay grey
    # instead of going to ink.
    vis.headlight.ambient[:] = [0.38, 0.38, 0.40]
    vis.headlight.diffuse[:] = [0.20, 0.20, 0.20]
    vis.headlight.specular[:] = [0.10, 0.10, 0.10]

    # A matte, light floor, made wide enough that its edge is never in shot.  Its brightness and
    # the key light's are set together so that the lit floor lands near 195 of 255: light, but
    # with its shading intact.  At the first, brighter setting two fifths of the frame came back
    # at pure white, which is a blown-out picture however light and airy it looks in a thumbnail.
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    if gid >= 0:
        model.geom_rgba[gid] = [0.80, 0.80, 0.82, 1.0]
        model.geom_size[gid] = [2.0, 2.0, 0.05]
        # And plain, not tiled. A repeating checker on a two-metre floor is under a pixel per
        # tile by the time it reaches the back of the shot, and a texture sampled below one
        # pixel per tile is the dotted moire that used to fill the left of the frame. The
        # painted grid below draws the floor's scale instead, in world space, where it stays a
        # line rather than turning into interference.
        model.geom_matid[gid] = -1

    # Key and fill become directional: a point lamp a hand's width above the table burns a hot
    # spot into the floor and leaves the corners dark, which is exactly what the first attempt
    # looked like.
    for i in range(model.nlight):
        model.light_type[i] = int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        model.light_active[i] = 1
    if model.nlight > 0:
        model.light_dir[0] = _unit([-0.40, 0.35, -1.00])
        model.light_diffuse[0] = [0.50, 0.50, 0.50]
        model.light_specular[0] = [0.16, 0.16, 0.16]
        model.light_castshadow[0] = 1
    if model.nlight > 1:
        model.light_dir[1] = _unit([0.55, -0.50, -0.65])
        model.light_diffuse[1] = [0.30, 0.30, 0.30]
        model.light_specular[1] = [0.05, 0.05, 0.05]
        model.light_castshadow[1] = 0


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / max(float(np.linalg.norm(a)), 1e-12)


# =================================================================================== rendering ===


def assert_shape(array: np.ndarray, width: int, height: int, what: str) -> np.ndarray:
    """Every rendered or composed array passes through here before it is used."""

    want = (height, width, 3) if array.ndim == 3 else (height, width)
    if array.shape != want:
        raise AssertionError(f"{what}: got {array.shape}, expected {want}")
    return array


class FreeCameraRenderer(SceneRenderer):
    """:class:`SceneRenderer` plus a movable camera, and a shape check on every frame.

    The parent renders only from named cameras fixed in the model.  A video needs to move the
    camera, so :meth:`render_free` runs the same pipeline with MuJoCo's free camera
    (lookat / distance / azimuth / elevation) and hands back the same :class:`Frame`.
    """

    def render(self, data, camera: str) -> Frame:
        frame = super().render(data, camera)
        assert_shape(frame.rgb, self.width, self.height, f"{camera} rgb")
        return frame

    def render_free(self, data, lookat, distance: float, azimuth: float, elevation: float) -> Frame:
        self._ensure_context()
        self._gl_context.make_current()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = np.asarray(lookat, dtype=float)
        self._camera.distance = float(distance)
        self._camera.azimuth = float(azimuth)
        self._camera.elevation = float(elevation)

        rgb, depth_ndc = self._render_pass(data, segment=False)
        depth = self._depth_to_metres(depth_ndc)
        instance = self._render_segmentation(data)
        frame = Frame(
            rgb=np.flipud(rgb).copy(),
            depth=np.flipud(depth).copy(),
            instance=np.flipud(instance).copy(),
            body_ids=dict(self._name_to_id),
        )
        assert_shape(frame.rgb, self.width, self.height, "free camera rgb")
        return frame

    def camera_info(self, width: int | None = None, height: int | None = None) -> "CameraInfo":
        """Pose of the free camera as MuJoCo itself computed it for the last ``render_free``."""

        gl = self._scene.camera[0]
        forward = _unit(np.asarray(gl.forward, dtype=float))
        up = _unit(np.asarray(gl.up, dtype=float))
        return CameraInfo(
            pos=np.asarray(gl.pos, dtype=float),
            forward=forward,
            up=up,
            right=np.cross(forward, up),
            fovy_deg=float(self.model.vis.global_.fovy),
            width=int(width or self.width),
            height=int(height or self.height),
        )


@dataclass(frozen=True)
class CameraInfo:
    """Enough of a camera to project a world point onto the pixels of the frame it rendered."""

    pos: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    right: np.ndarray
    fovy_deg: float
    width: int
    height: int


def project(cam: CameraInfo, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(N, 3)`` world points -> pixel x, pixel y, and which of them are in front of the camera."""

    v = np.atleast_2d(np.asarray(points, dtype=float)) - cam.pos[None, :]
    depth = v @ cam.forward
    half_h = np.maximum(depth, 1e-6) * math.tan(math.radians(cam.fovy_deg) / 2.0)
    half_w = half_h * (cam.width / cam.height)
    px = ((v @ cam.right) / np.maximum(half_w, 1e-12) * 0.5 + 0.5) * cam.width
    py = (1.0 - ((v @ cam.up) / np.maximum(half_h, 1e-12) * 0.5 + 0.5)) * cam.height
    return px, py, depth > 1e-6


# ============================================================================== camera keyframes ===


@dataclass(frozen=True)
class CamKey:
    """One keyframe of the main camera's path.  ``t`` is seconds of finished video."""

    t: float
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


def ease(f: float) -> float:
    """Smootherstep: zero velocity *and* zero acceleration at both ends of a move."""

    f = min(max(f, 0.0), 1.0)
    return f * f * f * (f * (f * 6.0 - 15.0) + 10.0)


def interp_camera(keys: Sequence[CamKey], t: float) -> CamKey:
    """Where the camera is at video time ``t``, easing in and out of every move."""

    if not keys:
        raise ValueError("a camera path needs at least one keyframe")
    if t <= keys[0].t:
        return keys[0]
    if t >= keys[-1].t:
        return keys[-1]
    for a, b in zip(keys, keys[1:]):
        if a.t <= t <= b.t:
            f = ease((t - a.t) / max(b.t - a.t, 1e-9))
            turn = ((b.azimuth - a.azimuth + 180.0) % 360.0) - 180.0
            lookat = tuple(np.asarray(a.lookat) + f * (np.asarray(b.lookat) - np.asarray(a.lookat)))
            return CamKey(
                t=t,
                lookat=lookat,
                distance=a.distance + f * (b.distance - a.distance),
                azimuth=a.azimuth + f * turn,
                elevation=a.elevation + f * (b.elevation - a.elevation),
            )
    return keys[-1]


# ================================================================================ picture polish ===


BACKDROP_TOP = (241, 243, 247)
BACKDROP_BOTTOM = (205, 209, 217)


def backdrop(rgb: np.ndarray, instance: np.ndarray) -> np.ndarray:
    """Paint the grey-to-white backdrop into the pixels the camera did not hit.

    The compiled scene has no sky asset, so without this the space above the floor is black.  The
    segmentation pass marks background as ``-1``, which is an exact mask -- no colour keying, no
    edge fringe.
    """

    mask = instance == -1
    if not mask.any():
        return rgb
    height = rgb.shape[0]
    f = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    grad = np.asarray(BACKDROP_BOTTOM, np.float32) * (1.0 - f) + np.asarray(BACKDROP_TOP, np.float32) * f
    out = rgb.copy()
    out[mask] = np.broadcast_to(grad, rgb.shape).astype(np.uint8)[mask]
    return out


def floor_grid(
    rgb: np.ndarray,
    depth: np.ndarray,
    cam: CameraInfo,
    *,
    spacing: float = 0.05,
    extent: float = 0.32,
    colour=(118, 124, 138),
    alpha: float = 0.30,
    tolerance: float = 0.004,
    fade_near: float = 0.34,
    fade_far: float = 0.62,
) -> np.ndarray:
    """A faint square grid on the table top, for scale.

    The grid is sampled in world space and depth-tested against this frame's own depth buffer, so
    the arm and the bricks occlude it instead of it being painted over them.

    Far away the lines crowd into less than a pixel each and the fixed-alpha blend turns them
    into a dotted moire. Two things stop that: the samples along each line are dense enough that
    a line is continuous right to the back of the grid, and each sample's alpha falls off with
    distance from the camera, so the grid dissolves into the backdrop instead of aliasing.
    `fade_near` is where the fade starts and `fade_far` where the grid has gone entirely [m].
    """

    coords = np.arange(-extent, extent + 1e-9, spacing)
    along = np.linspace(-extent, extent, 3000)
    lines = []
    for c in coords:
        lines.append(np.stack([along, np.full_like(along, c), np.zeros_like(along)], axis=1))
        lines.append(np.stack([np.full_like(along, c), along, np.zeros_like(along)], axis=1))
    pts = np.concatenate(lines, axis=0)

    px, py, ahead = project(cam, pts)
    xi = np.rint(px).astype(np.int32)
    yi = np.rint(py).astype(np.int32)
    inside = ahead & (xi >= 0) & (xi < cam.width) & (yi >= 0) & (yi < cam.height)
    if not inside.any():
        return rgb
    xi, yi = xi[inside], yi[inside]
    true_depth = depth[yi, xi]
    v = pts[inside] - cam.pos[None, :]
    along_view = v @ cam.forward
    keep = np.abs(true_depth - along_view) < tolerance
    if not keep.any():
        return rgb
    xi, yi = xi[keep], yi[keep]
    fade = np.clip((fade_far - along_view[keep]) / max(fade_far - fade_near, 1e-6), 0.0, 1.0)
    bright = fade > 0.01
    if not bright.any():
        return rgb
    xi, yi, fade = xi[bright], yi[bright], fade[bright].astype(np.float32)

    out = rgb.astype(np.float32)
    col = np.asarray(colour, dtype=np.float32)
    thickness = max(int(round(cam.width / 1280.0)), 1)
    for oy in range(-thickness, thickness + 1):
        for ox in range(-thickness, thickness + 1):
            a = (alpha if (ox == 0 and oy == 0) else alpha * 0.4) * fade
            yy = np.clip(yi + oy, 0, cam.height - 1)
            xx = np.clip(xi + ox, 0, cam.width - 1)
            out[yy, xx] = out[yy, xx] * (1.0 - a[:, None]) + col[None, :] * a[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================== how big is it ===


def subject_points(scene: Scene) -> np.ndarray:
    """World points that have to be in shot: the arm, the tray, the baseplate and the bricks.

    Every visible geom except the floor contributes the eight corners of its own bounding box,
    which MuJoCo keeps per geom (``geom_aabb``, in the geom's frame).  The bounding *sphere*
    ``geom_rbound`` would be simpler, but it is far too generous for flat things -- the
    baseplate's sphere is 180 mm across where the plate itself is 3 mm thick -- and measuring the
    framing with it makes a well framed shot look badly framed.
    """

    model, data = scene.model, scene.data
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    pts = []
    for g in range(model.ngeom):
        if g == table or int(model.geom_group[g]) == COLLISION_GEOM_GROUP:
            continue
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        corners = centre[None, :] + signs * half[None, :]
        pts.append(data.geom_xpos[g][None, :] + corners @ rot.T)
    if not pts:
        return np.zeros((0, 3))
    out = np.concatenate(pts, axis=0)
    # The arm's base mesh reaches 44 mm below its own origin, under the table top, where the floor
    # hides it.  Framing the shot around parts nobody can see only pushes the camera away.
    return out[out[:, 2] >= 0.0]


def arm_points(scene: Scene, root: str = "base") -> np.ndarray:
    """The same eight-corner boxes as :func:`subject_points`, but only for the arm itself.

    Everything hanging off ``root`` in the body tree: the base, the shoulder, the upper arm, the
    lower arm, the wrist, the gripper and the moving jaw.  This is the set that has to be inside
    the frame - a viewer who cannot see the shoulder cannot tell that the claw at the bottom of
    the picture and the column in the middle of it are one machine.
    """

    model, data = scene.model, scene.data
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    if root_id < 0:
        raise ValueError(f"no body named {root!r} to measure the arm from")
    in_arm = np.zeros(model.nbody, dtype=bool)
    in_arm[root_id] = True
    for b in range(root_id + 1, model.nbody):          # children always follow their parent
        in_arm[b] = in_arm[int(model.body_parentid[b])]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    pts = []
    for g in range(model.ngeom):
        if int(model.geom_group[g]) == COLLISION_GEOM_GROUP or not in_arm[int(model.geom_bodyid[g])]:
            continue
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        pts.append(data.geom_xpos[g][None, :] + (centre[None, :] + signs * half[None, :]) @ rot.T)
    if not pts:
        return np.zeros((0, 3))
    out = np.concatenate(pts, axis=0)
    # The base mesh reaches 44 mm below its own origin, under the table top, where the floor
    # hides it; demanding that it be in shot only pushes the camera away for nothing.
    return out[out[:, 2] >= 0.0]


def ray_angle_deg(cam: CameraInfo, start, end) -> float:
    """Angle between the camera's line of sight and the line from ``start`` to ``end``, degrees.

    Zero means the camera is looking straight down that line, and everything strung along it
    piles up in the same few pixels.  On an arm that is exactly what makes a picture unreadable:
    at a small angle the shoulder, the upper arm, the lower arm and the wrist collapse into one
    another and the machine reads as a single vertical stack with no joints in it.  Ninety
    degrees is broadside, where every link is its own shape.

    The angle is taken at the middle of the line, and it is returned unsigned and folded into
    0-90 degrees, because looking down a line from either end is equally bad.
    """

    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    line = b - a
    norm = float(np.linalg.norm(line))
    if norm < 1e-9:
        return 90.0
    line = line / norm
    ray = 0.5 * (a + b) - cam.pos
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm < 1e-9:
        return 0.0
    cos = abs(float(np.dot(line, ray / ray_norm)))
    return float(math.degrees(math.acos(min(cos, 1.0))))


def subject_span(cam: CameraInfo, points: np.ndarray) -> float:
    """What fraction of the frame width the subject spans, from 0 to (possibly more than) 1."""

    px, _, ahead = project(cam, points)
    if not ahead.any():
        return 0.0
    visible = px[ahead]
    return float(visible.max() - visible.min()) / cam.width


def frame_overrun(cam: CameraInfo, points: np.ndarray) -> float:
    """How far outside the frame ``points`` reach, as a fraction of the frame's own size.

    Zero means every point is inside.  A point behind the camera counts as infinitely outside,
    because it is not in the picture at all.
    """

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.size == 0:
        return 0.0
    px, py, ahead = project(cam, points)
    if not ahead.all():
        return float("inf")
    return max(
        float(-px.min()) / cam.width, float(px.max() - cam.width) / cam.width,
        float(-py.min()) / cam.height, float(py.max() - cam.height) / cam.height,
        0.0,
    )


# ==================================================================================== video I/O ===


def downscale(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Shrink an image.  Refuses to enlarge one -- that is the whole point of this function."""

    if img.shape[1] < width or img.shape[0] < height:
        raise ValueError(
            f"refusing to enlarge {img.shape[1]}x{img.shape[0]} to {width}x{height}; "
            "every picture in this toolkit is rendered at its own native size"
        )
    if (img.shape[1], img.shape[0]) == (width, height):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class VideoWriter:
    """Streaming H.264 writer: one frame in, one frame encoded, nothing held in memory.

    A 2560x1440 clip of half a minute is nine hundred frames of eleven megabytes each; keeping
    them in a list to encode at the end would need ten gigabytes.
    """

    def __init__(self, path, *, width: int, height: int, fps: int, crf: int, preset: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        self.frames = 0
        self._writer = imageio.get_writer(
            str(self.path), fps=int(fps), codec="libx264", quality=None, macro_block_size=1,
            pixelformat="yuv420p",
            output_params=["-crf", str(int(crf)), "-preset", str(preset)],
        )

    def append(self, frame: np.ndarray) -> None:
        assert_shape(frame, self.width, self.height, f"frame for {self.path.name}")
        self._writer.append_data(np.ascontiguousarray(frame))
        self.frames += 1

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ======================================================================================= recording ===


def _export_path(master: Path) -> Path:
    """Where the 1080p export goes, given the master's path.

    A master whose name already says what size it is gets that swapped rather than stacked up, so
    ``..._1440p.mp4`` exports as ``..._1080p.mp4`` and not ``..._1440p_1080p.mp4``.
    """

    stem = master.stem
    if stem.endswith("1440p"):
        stem = stem[: -len("1440p")].rstrip("_-")
    return master.with_name(f"{stem}_1080p.mp4")


@dataclass(frozen=True)
class Tick:
    """One control tick from a motion generator: the live scene, the phase, the measurements."""

    scene: Scene
    phase: str
    data: dict


@dataclass
class RecordingResult:
    """What one recording run produced, in numbers a report can quote without re-measuring."""

    master: Path
    export: Path | None
    frames: int
    seconds: float
    width: int
    height: int
    min_subject_span: float
    max_arm_overrun: float
    """The worst fraction of the frame by which the arm reached outside it.  Zero is the point:
    the whole arm, base included, was in shot on every frame."""

    min_ray_angle: float
    """The smallest angle, over the whole take, between the camera's line of sight and the line
    the video was told to stay off (the shoulder-to-wrist line).  Degrees; 90 is broadside."""

    stills: list[Path]
    still_rects: list[dict[str, Rect]]
    """The element rectangles of each still, in the same order -- what the label crops are cut
    from.  They differ between stills: the title block is only on screen for the first seconds,
    and the phase label takes its place afterwards."""

    rects: dict[str, Rect]
    """The rectangles of the last frame written, for a caller that only wants one example."""


@dataclass(frozen=True)
class ReadoutSpec:
    """One number in the data strip: where its value comes from and how it scales.

    ``label`` is a title, not a remark: a noun phrase with its unit in square brackets.  The
    compact cells are narrow, so a title that does not fit its cell is refused rather than
    drawn over its neighbour.
    """

    key: str
    label: str
    unit: str
    lo: float
    hi: float | None
    """Top of the trace's scale, or `None` to scale it to the biggest value the run has shown --
    which is the only honest choice when the number starts wherever the world happens to be
    wrong and the point of the video is watching it come down."""

    fmt: str = "{:.1f}"

    suffix: str = ""
    """Small dim text beside the big number: the value it is heading for, not a remark."""

    reference: float | None = None
    """Drawn as a dashed line across the plot: the value the number is heading for."""

    references: tuple[float, ...] = ()
    """Several such lines, each labelled with its own value -- an allowable and a strength, say.
    Set this or :attr:`reference`, not both.  With it set, the auto scale also keeps every line
    on screen, because a labelled line the trace has not reached yet is the point of drawing it."""

    text: bool = False
    """The value is a word, not a number: the tick supplies a string under :attr:`key` and the
    cell prints it as it stands.  A text readout carries no trace, so it cannot be the plot."""

    plot: bool = False
    """Draw this one as the strip's full-width time plot.  At most one spec may ask for it;
    the rest are compact title-and-number cells along the bottom of the strip."""


@dataclass(frozen=True)
class CloseUp:
    """Where the picture-in-picture camera looks on one tick.

    A free camera rather than one bolted to the gripper, because a camera bolted to the gripper
    puts the near half of the jaws right up against its lens: the first cut showed two jaw tips
    jammed against the left and right edges of the inset with the right one clipped off.  Aiming
    a free camera at the tool from a fixed direction keeps both jaw faces and the gap between
    them in the middle of the inset for the whole take.
    """

    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


def panel_size_for(quality: Quality, n: int = 4) -> tuple[int, int]:
    """The ``(width, height)`` each of ``n`` panel pictures must be, at this quality: a square.

    The same answer as :attr:`Recorder.panel_size`, available before a :class:`Recorder` exists, for a
    caller that has to build its panel sources first.
    """

    return Overlay(quality.width, quality.height).panel_size(int(n))


class Recorder:
    """Renders a motion generator into a master clip (and a 1920x1080 export beside it).

    The generator is the only source of movement: it yields a :class:`Tick` per control tick, and
    the video decides from the simulator's own clock when the next video frame is due.  The camera
    path, the overlay and the panels are the video's business; the physics is not.
    """

    def __init__(
        self,
        scene: Scene,
        *,
        quality: Quality,
        shot: Sequence[CamKey],
        title: str,
        subtitle: str,
        readouts: Sequence[ReadoutSpec],
        phase_name: Callable[[str], str],
        panel_images: "Callable[[Tick], Sequence[tuple[str, np.ndarray]]] | None" = None,
        n_panels: int = 4,
        pip_camera: "str | Callable[[Tick], CloseUp] | None" = None,
        caption_for: "Callable[[str], str | None] | None" = None,
        title_seconds: float = 4.0,
        trace_seconds: float = 8.0,
        min_span: float = 0.60,
        pip_label: str | Callable[[str], str] | None = None,
        keep_in_frame: Callable[[Scene], np.ndarray] | None = None,
        overrun_tol: float = 0.0,
        off_the_line: Callable[[Scene], tuple] | None = None,
        min_ray_deg: float = 0.0,
        pip_image: "Callable[[Tick], np.ndarray] | None" = None,
    ):
        self.scene = scene
        self.quality = quality
        self.shot = list(shot)
        self.title = title
        self.subtitle = subtitle
        """One compact spec line under the title: settings separated by ``|``, not a sentence.
        It wraps to a second line at a separator when it is too long for one."""

        self.readouts = list(readouts)
        self.n_panels = int(n_panels)
        self.panel_images = panel_images
        """Supplies the right-hand grid: handed the tick, it returns ``n_panels`` pairs of
        ``(title, rgb)``, each array exactly the square :attr:`panel_size`.  With it set, the
        video draws no single inset and needs no ``pip_camera``."""

        self.pip_camera = pip_camera
        self.caption_for = caption_for
        self.phase_name = phase_name
        self.title_seconds = float(title_seconds)
        self.trace_seconds = float(trace_seconds)
        self.min_span = float(min_span)
        self.pip_label = pip_label
        self.keep_in_frame = keep_in_frame
        self.overrun_tol = float(overrun_tol)
        self.off_the_line = off_the_line
        self.min_ray_deg = float(min_ray_deg)
        self.pip_image = pip_image
        """Supplies the single inset picture itself, instead of rendering `pip_camera`. It is
        handed the tick and returns an RGB array at the pip size."""

        if panel_images is None and pip_camera is None and pip_image is None:
            raise ValueError(
                "a video draws either a column of panels (panel_images) or the single inset "
                "(pip_camera / pip_image); it was given neither"
            )

        self.overlay = Overlay(quality.width, quality.height)
        pip_w = int(quality.pip_width)
        pip_h = int(round(pip_w * 9 / 16))
        self._pip_size = (pip_w, pip_h)
        self._main = FreeCameraRenderer(scene.model, width=quality.width, height=quality.height)
        self._main.instance_lut(scene.instance_ids)
        self._close: FreeCameraRenderer | None = None
        if panel_images is None and pip_image is None:
            # Only the path that actually renders an inset pays for a second GL context.
            self._close = FreeCameraRenderer(scene.model, width=pip_w, height=pip_h)
            self._close.instance_lut(scene.instance_ids)
        self._marks: list[float] = []
        """Video times where the step on screen changed -- the vertical rules on the trace."""
        self._last_phase: str | None = None
        self._history: dict[str, list[tuple[float, float | str]]] = {
            r.key: [] for r in self.readouts}

    @property
    def panel_size(self) -> tuple[int, int]:
        """The exact ``(width, height)`` of every panel picture this video will accept."""

        return self.overlay.panel_size(self.n_panels)

    def close(self) -> None:
        self._main.close()
        if self._close is not None:
            self._close.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- one frame ---------------------------------------------------------------------------

    def compose(
        self, tick: Tick, video_t: float
    ) -> tuple[np.ndarray, dict[str, Rect], float, float, float]:
        """Render and label one finished frame at the master resolution.

        Returns the pixels, every element's rectangle, how much of the frame width the subject
        spans, how far the ``keep_in_frame`` points reach outside the frame, and the angle
        between the line of sight and the line the shot is meant to stay off.
        """

        q = self.quality
        key = interp_camera(self.shot, video_t)
        frame = self._main.render_free(
            tick.scene.data, key.lookat, key.distance, key.azimuth, key.elevation
        )
        cam = self._main.camera_info()
        picture = backdrop(frame.rgb, frame.instance)
        picture = floor_grid(picture, frame.depth, cam)
        span = subject_span(cam, subject_points(tick.scene))
        overrun = (
            frame_overrun(cam, self.keep_in_frame(tick.scene))
            if self.keep_in_frame is not None else 0.0
        )
        if self.off_the_line is not None:
            start, end = self.off_the_line(tick.scene)
            ray_deg = ray_angle_deg(cam, start, end)
        else:
            ray_deg = 90.0

        panels: tuple[tuple[str, np.ndarray], ...] = ()
        pip: np.ndarray | None = None
        if self.panel_images is not None:
            panels = tuple(self.panel_images(tick))
            if len(panels) != self.n_panels:
                raise ValueError(
                    f"the video is laid out for {self.n_panels} panels and was handed "
                    f"{len(panels)}"
                )
        elif self.pip_image is not None:
            pip = np.asarray(self.pip_image(tick))
            assert_shape(pip, self._pip_size[0], self._pip_size[1], "close-up")
        else:
            if callable(self.pip_camera):
                shot = self.pip_camera(tick)
                close = self._close.render_free(tick.scene.data, shot.lookat, shot.distance,
                                                shot.azimuth, shot.elevation)
            else:
                close = self._close.render(tick.scene.data, self.pip_camera)
            pip = backdrop(close.rgb, close.instance)
            assert_shape(pip, self._pip_size[0], self._pip_size[1], "close-up")

        readouts = []
        for spec in self.readouts:
            series = self._history[spec.key]
            value = series[-1][1] if series else None
            t_end = series[-1][0] if series else 0.0
            t_start = t_end - self.trace_seconds
            trace = [v for t, v in series if t >= t_start]
            if spec.text:
                readouts.append(Readout(label=spec.label, value=None, unit="",
                                        value_text=value, lo=spec.lo, hi=spec.hi or 1.0,
                                        fmt=spec.fmt))
                continue
            hi = spec.hi
            if hi is None:
                hi = max(max(trace, default=0.0) * 1.08, spec.lo + 1e-6)
                if spec.references:
                    hi = max(hi, max(spec.references) * 1.15)
            marks = ()
            if trace and self._marks:
                # `window`, not `span`: `span` is the subject's share of the frame width, which
                # this loop used to overwrite with a number of seconds -- and the framing guard
                # in `run` then measured the trace's time window instead of the shot.
                t0 = max(t_start, series[0][0])
                window = max(t_end - t0, 1e-9)
                marks = tuple((t - t0) / window for t in self._marks if t >= t0)
            readouts.append(Readout(label=spec.label, value=value, unit=spec.unit, trace=trace,
                                    lo=spec.lo, hi=hi, fmt=spec.fmt, plot=spec.plot,
                                    suffix=spec.suffix, reference=spec.reference,
                                    references=spec.references, marks=marks))

        show_title = video_t < self.title_seconds
        composed, rects = self.overlay.draw(
            picture,
            title=self.title if show_title else None,
            subtitle=self.subtitle if show_title else None,
            phase=None if show_title else self.phase_name(tick.phase),
            caption=self.caption_for(tick.phase) if self.caption_for is not None else None,
            readouts=readouts,
            panels=panels,
            pip=pip,
            pip_label=None if pip is None else (
                self.pip_label(tick.phase) if callable(self.pip_label) else self.pip_label
            ),
        )
        assert_shape(composed, q.width, q.height, "composed frame")
        return composed, rects, span, overrun, ray_deg

    # -- the whole clip ----------------------------------------------------------------------

    def run(
        self,
        motion: Iterable[Tick],
        out_path,
        *,
        still_times: Sequence[float] = (),
        stills_dir=None,
        still_when: Callable[[Tick, float], str | None] | None = None,
    ) -> RecordingResult:
        """Write the clip.  ``still_times`` are clock times to grab a still at.

        ``still_when`` grabs one at a moment nobody can put a time on in advance: it is handed
        every frame's tick and its video time and returns a name to save that frame under, or
        ``None``.  A real motion decides for itself when the jaws have closed and when the studs
        have gone in, so those two frames cannot be picked off a stopwatch beforehand.
        """

        q = self.quality
        out_path = Path(out_path)
        export_path = _export_path(out_path) if q.export_1080 else None
        stills_dir = Path(stills_dir) if stills_dir is not None else None
        if stills_dir is not None:
            stills_dir.mkdir(parents=True, exist_ok=True)

        writer = VideoWriter(out_path, width=q.width, height=q.height, fps=q.fps, crf=q.crf,
                             preset=q.preset)
        export = (
            VideoWriter(export_path, width=1920, height=1080, fps=q.fps, crf=q.crf, preset=q.preset)
            if export_path is not None else None
        )
        wanted = sorted(float(t) for t in still_times)
        timed = 0          # numbers the evenly spaced stills on their own, so a named one
                           # landing between two of them does not leave a gap in the sequence
        stills: list[Path] = []
        still_rects: list[dict[str, Rect]] = []
        rects: dict[str, Rect] = {}
        min_span = 1e9
        max_overrun = 0.0
        min_ray = 1e9
        frames = 0
        sim_t0 = None
        next_frame_t = 0.0

        try:
            for tick in motion:
                t_sim = float(tick.scene.data.time)
                if sim_t0 is None:
                    sim_t0 = t_sim
                video_t = t_sim - sim_t0
                for spec in self.readouts:
                    if spec.key in tick.data:
                        reading = (str(tick.data[spec.key]) if spec.text
                                   else float(tick.data[spec.key]))
                        self._history[spec.key].append((video_t, reading))
                if tick.phase != self._last_phase:
                    if self._last_phase is not None:
                        self._marks.append(video_t)
                    self._last_phase = tick.phase
                if video_t + 1e-9 < next_frame_t:
                    continue
                composed, rects, span, overrun, ray_deg = self.compose(tick, video_t)
                min_span = min(min_span, span)
                max_overrun = max(max_overrun, overrun)
                min_ray = min(min_ray, ray_deg)
                if ray_deg < self.min_ray_deg:
                    raise RuntimeError(
                        f"at {video_t:.2f} s the camera is only {ray_deg:.1f} degrees off the "
                        f"line it was told to stay clear of; the links pile up on one another"
                    )
                if span < self.min_span:
                    raise RuntimeError(
                        f"at {video_t:.2f} s the arm, tray and baseplate span only "
                        f"{span * 100:.0f} percent of the frame width; the camera is too far away"
                    )
                if overrun > self.overrun_tol:
                    raise RuntimeError(
                        f"at {video_t:.2f} s the arm reaches {overrun * 100:.1f} percent of the "
                        f"frame outside it; the camera is too close or pointed wrong"
                    )
                writer.append(composed)
                if export is not None:
                    export.append(downscale(composed, 1920, 1080))
                if wanted and stills_dir is not None and video_t + 1e-9 >= wanted[0]:
                    path = stills_dir / f"still_{timed:02d}_{video_t:05.2f}s.png"
                    imageio.imwrite(str(path), composed)
                    stills.append(path)
                    still_rects.append(dict(rects))
                    timed += 1
                    wanted.pop(0)
                if still_when is not None and stills_dir is not None:
                    moment = still_when(tick, video_t)
                    if moment:
                        path = stills_dir / f"still_{moment}_{video_t:05.2f}s.png"
                        imageio.imwrite(str(path), composed)
                        stills.append(path)
                        still_rects.append(dict(rects))
                frames += 1
                next_frame_t += 1.0 / q.fps
        finally:
            writer.close()
            if export is not None:
                export.close()

        return RecordingResult(
            master=out_path,
            export=export_path,
            frames=frames,
            seconds=frames / q.fps,
            width=q.width,
            height=q.height,
            min_subject_span=0.0 if min_span > 1e8 else min_span,
            max_arm_overrun=max_overrun,
            min_ray_angle=90.0 if min_ray > 1e8 else min_ray,
            stills=stills,
            still_rects=still_rects,
            rects=rects,
        )
