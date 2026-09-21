"""The camera views the robot uses: fixed LOOK poses, and a moving attention window.

Every view here is the wrist camera (`brick_builder.robot.patch_so101.WRIST_CAM` -- the one camera the
real hardware carries) used the same way: move the tool to a pose (or don't -- see the attention
window below), render at a resolution larger than the 256x256 model input, crop to a region of
interest, and resize down to 256x256. Rendering wide and cropping digitally, rather than asking
MuJoCo for a narrower field of view directly, is what a real wide-angle UVC module would also have
to do to get a denser, more central picture of small objects -- the crop is doing the same job a
real firmware's ISP or the host driver would.

Two things are fixed and two are chosen per call:

- **Fixed**: the wrist camera's wide diagonal field of view
  (`patch_so101.WRIST_CAM_DIAGONAL_FOV_DEG`, converted to MuJoCo's vertical `fovy` for whatever
  resolution is being rendered at, by `patch_so101.fovy_from_diagonal_fov`), and the scene
  model's own input resolution (256x256).
- **Chosen**: the render resolution before cropping, and the crop rectangle itself.

Three named views are pre-computed, fixed crops, calibrated once by rendering (the numbers below
are measurements, not guesses):

- `PLATE_VIEW`: the whole 8x8 (128 mm) build plate, hovering at its centre. At this pose and crop
  the jaws take about 7% of the frame, the plate about 25%, and roughly 26 pixels cover one stud
  pitch (16 mm).
- `PLATE_VIEW_DENSE`: a centred 4x4-stud (64 mm) window instead of the whole plate -- roughly
  double the pixel density, at the cost of not showing bricks placed outside that window.
- `TRAY_VIEW`: the loose-brick tray, framing two of its three pockets at high density.

A fourth mode, the **attention window** (`attention_view`), is not a fixed pose at all: given
wherever the wrist camera currently is (from the simulator, or from forward kinematics on a real
arm) and a 3D region of interest -- the target stud cell plus the brick in the gripper, a tray
pocket, whatever the controller's next action is aimed at -- it picks the smallest square crop
that keeps that region in frame with a margin, at the current camera pose, and resizes it the same
way. This is what closed-loop control needs: the wrist camera moves every tick, so the crop that
keeps the target framed has to be recomputed every tick too. `PLATE_VIEW`/`PLATE_VIEW_DENSE`/
`TRAY_VIEW` are exactly what this same mechanism gives at one fixed pose for one fixed ROI (the
whole plate, or a tray pocket) -- they are kept as named, pre-measured constants because a
one-time calibration is worth having, not because the mechanism is different.

Every view, fixed or dynamic, produces the same two things: a 256x256 `Frame` (rgb, instance,
depth) and a `CameraGeometry` already reparametrized for *that* 256x256 image -- a pixel from the
final frame can be handed straight to `brick_builder.perception.backproject`'s functions with no
further bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import mujoco
import numpy as np

from .backproject import CameraGeometry, camera_geometry, project_point
from .scene_model import SceneState
from ..render.renderer import Frame, SceneRenderer
from ..robot.patch_so101 import WRIST_CAM, WRIST_CAM_DIAGONAL_FOV_DEG, fovy_from_diagonal_fov
from ..scene.compose import Scene, reset_scene, tool_down_ik

# --------------------------------------------------------------------------------- the pose ---


@dataclass(frozen=True)
class LookPose:
    """A camera view: where to render from, a wide field of view, and a crop/resize recipe.

    `tcp`/`yaw`, when `tcp` is not `None`, are handed to `tool_down_ik` by `solve_look`/
    `goto_look` to put the arm at a fixed pose before rendering (`PLATE_VIEW`, `PLATE_VIEW_DENSE`,
    `TRAY_VIEW` all do this). `tcp=None` means "wherever the arm already is" -- the attention
    window's own views are built this way, since they exist precisely to follow a moving camera
    rather than to send the arm somewhere.

    `fovy_deg` overrides the camera's compiled-in field of view for `render_size` (see the module
    docstring: MuJoCo's pinhole camera only takes a vertical `fovy`, so the wrist lens's diagonal
    spec has to be converted for whichever resolution is actually rendered). `render_size` is the
    resolution rendered *before* cropping; `crop` is `(x0, y0, x1, y1)` pixel bounds inside that
    render; `output_size` is the final square side (256, the scene model's input resolution).
    """

    name: str
    camera: str
    render_size: tuple[int, int]
    crop: tuple[int, int, int, int]
    fovy_deg: float
    output_size: int = 256
    tcp: tuple[float, float, float] | None = None
    yaw: float = 0.0
    seed_q: tuple[float, float, float, float, float] = (0.0, -0.5, 0.9, 1.2, 0.0)
    """A joint-angle seed for `tool_down_ik`'s search when `tcp` is set; irrelevant otherwise."""

    def crop_wh(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.crop
        return x1 - x0, y1 - y0


# ------------------------------------------------------------------------ the three fixed views ---

#: Plate view: the 8x8 build plate, hovering at its centre -- a low hover well inside the arm's
#: safe, always-reachable band (there is an unreachable dead zone directly over the plate roughly
#: 9-30 cm up; 5.5 cm clears it with margin to spare, not pushed to the minimum). Measured by
#: rendering a marker scene (six calibration bricks on the plate) through the corrected wrist
#: camera: jaws ~7% of frame, plate ~25%, ~26 px per 16 mm stud pitch.
PLATE_TCP = (0.17, 0.0, 0.058)
PLATE_YAW = 0.0
PLATE_RENDER_SIZE = (512, 512)
PLATE_FOVY_DEG = fovy_from_diagonal_fov(WRIST_CAM_DIAGONAL_FOV_DEG, *PLATE_RENDER_SIZE)
#: Square crop containing the *whole* 8x8 (128 mm) plate: the four plate corners, projected at
#: `PLATE_TCP`, bounded and centred in a square (a tilted camera does not project an axis-aligned
#: world square to an axis-aligned pixel rectangle, hence a few pixels of margin).
PLATE_CROP = (159, 138, 369, 348)
#: Denser alternative: a centred 4x4-stud (64 mm) window instead of the whole plate, found the
#: same way -- roughly double `PLATE_CROP`'s pixels-per-stud-pitch, at the cost of not showing a
#: brick placed outside the 4-stud window.
PLATE_CROP_DENSE = (213, 202, 314, 303)

PLATE_VIEW = LookPose(
    name="plate", camera=WRIST_CAM, render_size=PLATE_RENDER_SIZE, crop=PLATE_CROP,
    fovy_deg=PLATE_FOVY_DEG, tcp=PLATE_TCP, yaw=PLATE_YAW,
)
PLATE_VIEW_DENSE = LookPose(
    name="plate_dense", camera=WRIST_CAM, render_size=PLATE_RENDER_SIZE, crop=PLATE_CROP_DENSE,
    fovy_deg=PLATE_FOVY_DEG, tcp=PLATE_TCP, yaw=PLATE_YAW,
)

#: Tray view: the loose-brick tray, retargeted at `SceneConfig.tray_xy`'s default.
TRAY_TCP = (0.13, -0.17, 0.058)
TRAY_YAW = 0.0
TRAY_RENDER_SIZE = (512, 512)
TRAY_FOVY_DEG = fovy_from_diagonal_fov(WRIST_CAM_DIAGONAL_FOV_DEG, *TRAY_RENDER_SIZE)
#: Non-square crop (~1.9:1) framing 2 of the tray's 3 loose-brick pockets at high density and low
#: jaw share, rather than a wider crop that would show all 3 at markedly worse density and jaw
#: intrusion. `output_to_full`/`full_to_output` track the two crop axes independently, so the
#: resulting mild vertical stretch changes how round a stud looks, not backprojection accuracy.
TRAY_CROP = (110, 202, 292, 300)

TRAY_VIEW = LookPose(
    name="tray", camera=WRIST_CAM, render_size=TRAY_RENDER_SIZE, crop=TRAY_CROP,
    fovy_deg=TRAY_FOVY_DEG, tcp=TRAY_TCP, yaw=TRAY_YAW,
)

VIEWS: dict[str, LookPose] = {"plate": PLATE_VIEW, "plate_dense": PLATE_VIEW_DENSE, "tray": TRAY_VIEW}


# ------------------------------------------------------------------------------- driving a pose ---


def solve_look(scene: Scene, pose: LookPose, iters: int = 300) -> np.ndarray:
    """The 5 arm joint angles for `pose`, seeded from `pose.seed_q`. Requires `pose.tcp`."""
    if pose.tcp is None:
        raise ValueError(
            f"LookPose {pose.name!r} has tcp=None ('wherever the arm already is') -- there is "
            "nothing to solve for; use attention_view for a moving camera instead"
        )
    return tool_down_ik(
        scene.kin, np.asarray(pose.tcp, dtype=float), pose.yaw,
        q0=np.asarray(pose.seed_q, dtype=float), iters=iters,
    )


def goto_look(scene: Scene, pose: LookPose, rng: np.random.Generator, iters: int = 300) -> np.ndarray:
    """Drives `scene` to `pose` (a `reset_scene` call) and applies its `fovy_deg`. Requires
    `pose.tcp` -- see `solve_look`."""
    q = solve_look(scene, pose, iters=iters)
    reset_scene(scene, rng, arm_q=q)
    _apply_view_fovy(scene, pose)
    return q


def _apply_view_fovy(scene: Scene, pose: LookPose) -> None:
    cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, pose.camera)
    if cam_id < 0:
        raise ValueError(f"unknown camera {pose.camera!r}")
    scene.model.cam_fovy[cam_id] = pose.fovy_deg
    mujoco.mj_forward(scene.model, scene.data)


# ------------------------------------------------------------------------- crop / resize / map ---


def crop_and_resize(
    array: np.ndarray, crop: tuple[int, int, int, int], output_size: int, *, interpolation: int,
) -> np.ndarray:
    """Crops `array` (`[H, W]` or `[H, W, C]`) to `crop` and resizes to a square.

    `interpolation` is the caller's choice on purpose: `cv2.INTER_NEAREST` for instance ids and
    depth (a label or a metric value is never blended across an object boundary), a smoothing
    kernel (`cv2.INTER_LINEAR` here) for RGB.
    """
    x0, y0, x1, y1 = crop
    cropped = array[y0:y1, x0:x1]
    dtype = cropped.dtype
    if dtype in (np.int32, np.int64):
        # cv2.resize does not support int64/int32 directly for NEAREST on some builds; go via
        # float32 and round-trip, which is exact for nearest-neighbour (no new values are ever
        # synthesized).
        resized = cv2.resize(cropped.astype(np.float32), (output_size, output_size),
                              interpolation=interpolation)
        return np.rint(resized).astype(dtype)
    return cv2.resize(cropped, (output_size, output_size), interpolation=interpolation)


def cropped_camera_geometry(
    geom_full: CameraGeometry, crop: tuple[int, int, int, int], output_size: int,
) -> CameraGeometry:
    """Reparametrizes `geom_full` (the intrinsics/extrinsics of the *pre-crop* render) so its
    pixel functions apply directly to the final `output_size`x`output_size` image.

    A crop-then-resize is an affine map on pixel coordinates, and a pinhole camera's intrinsics
    are exactly the coefficients of an affine map from a local ray direction to pixel coordinates
    -- composing the two gives another pinhole camera, with new `fx`/`fy`/`cx`/`cy` and the same
    `pos`/`rot` (the crop does not move the camera, only which pixels are kept):

        fx' = fx / scale_x,  cx' = (cx - x0) / scale_x     (and the same for fy'/cy' with y)
        scale_x = (x1 - x0) / output_size

    No lens distortion is introduced or removed by this -- MuJoCo's own render is already the
    pinhole model both `geom_full` and this reparametrization assume; real barrel distortion is a
    separate image-space effect (`render.domain_rand.apply_postprocess`).
    """
    x0, y0, x1, y1 = crop
    scale_x = (x1 - x0) / float(output_size)
    scale_y = (y1 - y0) / float(output_size)
    return CameraGeometry(
        fx=geom_full.fx / scale_x,
        fy=geom_full.fy / scale_y,
        cx=(geom_full.cx - x0) / scale_x,
        cy=(geom_full.cy - y0) / scale_y,
        pos=geom_full.pos.copy(),
        rot=geom_full.rot.copy(),
    )


def full_frame_geometry(scene: Scene, pose: LookPose) -> CameraGeometry:
    """Camera intrinsics/extrinsics at `pose`'s *pre-crop* render resolution, read off `scene`'s
    current configuration (not necessarily `pose.tcp` -- the caller may be mid-motion)."""
    return camera_geometry(scene.model, scene.data, pose.camera, *pose.render_size)


def output_to_full(pose: LookPose, px: np.ndarray) -> np.ndarray:
    """Maps pixel coordinate(s) `[..., 2]` (col, row) in the final image back to the pre-crop
    render's pixel coordinates, so `full_frame_geometry`'s intrinsics apply directly."""
    x0, y0, x1, y1 = pose.crop
    scale_x = (x1 - x0) / float(pose.output_size)
    scale_y = (y1 - y0) / float(pose.output_size)
    px = np.asarray(px, dtype=float)
    return np.stack([x0 + px[..., 0] * scale_x, y0 + px[..., 1] * scale_y], axis=-1)


def full_to_output(pose: LookPose, px: np.ndarray) -> np.ndarray:
    """Inverse of `output_to_full`: pre-crop pixel(s) -> final-image pixel coordinate(s).

    Not clamped to `[0, output_size)`: a point outside the crop maps outside that range, which is
    the correct answer ("this world point is not in the final frame"), not an error.
    """
    x0, y0, x1, y1 = pose.crop
    scale_x = (x1 - x0) / float(pose.output_size)
    scale_y = (y1 - y0) / float(pose.output_size)
    px = np.asarray(px, dtype=float)
    return np.stack([(px[..., 0] - x0) / scale_x, (px[..., 1] - y0) / scale_y], axis=-1)


def px_per_stud_pitch(
    scene: Scene, pose: LookPose, plate_xy: tuple[float, float], plane_z: float, pitch: float,
) -> float:
    """Measures pixels-per-stud-pitch (16 mm for DUPLO) in the final `output_size` image, at
    `plate_xy`, by projecting two world points one pitch apart and mapping through the crop.

    A measurement, not a constant: domain-randomized camera jitter and exactly where on the plate
    a brick sits both change it a little.
    """
    geom = full_frame_geometry(scene, pose)
    p0 = project_point(geom, np.array([plate_xy[0], plate_xy[1], plane_z]))
    p1 = project_point(geom, np.array([plate_xy[0] + pitch, plate_xy[1], plane_z]))
    full_px_per_pitch = float(np.linalg.norm(p1 - p0))
    crop_w, _ = pose.crop_wh()
    return full_px_per_pitch * (pose.output_size / crop_w)


# --------------------------------------------------------------------------------- rendering ---


def render_view(scene: Scene, pose: LookPose, renderer: SceneRenderer) -> tuple[Frame, CameraGeometry]:
    """Renders `pose` from `scene`'s *current* configuration and returns the cropped/resized
    256x256 `Frame` (rgb, instance, depth) plus the `CameraGeometry` for that cropped image,
    ready for `backproject.py`.

    Does not move the arm -- call `goto_look` first for a fixed `LookPose` with `tcp` set, or
    just render wherever the arm already is for an attention window. `renderer` must already be
    sized for `pose.render_size` (`SceneRenderer(scene.model, width=pose.render_size[0],
    height=pose.render_size[1])`); reusing one `SceneRenderer` (and so one GL context) across many
    calls at the same `render_size` is the point of taking it as a parameter rather than building
    one internally.
    """
    _apply_view_fovy(scene, pose)
    frame_full = renderer.render(scene.data, pose.camera)
    rgb = crop_and_resize(frame_full.rgb, pose.crop, pose.output_size, interpolation=cv2.INTER_LINEAR)
    instance = crop_and_resize(frame_full.instance, pose.crop, pose.output_size,
                                interpolation=cv2.INTER_NEAREST)
    depth = crop_and_resize(frame_full.depth, pose.crop, pose.output_size,
                             interpolation=cv2.INTER_NEAREST)
    frame = Frame(rgb=rgb, depth=depth, instance=instance, body_ids=frame_full.body_ids)

    geom_full = camera_geometry(scene.model, scene.data, pose.camera, *pose.render_size)
    geometry = cropped_camera_geometry(geom_full, pose.crop, pose.output_size)
    return frame, geometry


# --------------------------------------------------------------------------- attention window ---


def roi_box_corners(center: Sequence[float], half_extents: Sequence[float]) -> np.ndarray:
    """The 8 corners of an axis-aligned world-frame box, `[8, 3]` -- a convenient way to describe
    a 3D region of interest (a stud cell's neighbourhood, the held brick's extent, a tray pocket)
    for `attention_view`."""
    center = np.asarray(center, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    return center + signs * half


def attention_crop(
    geom_full: CameraGeometry, roi_world: np.ndarray, render_size: tuple[int, int], *,
    margin: float = 0.25, min_crop_px: float = 48.0,
) -> tuple[int, int, int, int]:
    """The square crop (pre-resize pixel coordinates) that keeps every point of `roi_world`
    (`[..., 3]`, world/arm-base frame) in frame as seen by `geom_full` -- the full pre-crop
    render's camera geometry, at whatever pose the camera is *currently* at.

    The crop is the ROI's own projected bounding box, padded by `margin` (a fraction of its
    larger side) and grown to at least `min_crop_px` (so a very small or very close ROI still
    gets a sensible amount of context), then centred and clamped to fit inside `render_size` --
    shrinking rather than running off the edge if the render is not big enough to show the full
    padded box. This is the mechanism `PLATE_VIEW`/`PLATE_VIEW_DENSE`/`TRAY_VIEW` are themselves
    an instance of, computed once offline for a fixed ROI at a fixed pose; `attention_view` below
    calls this every time for whatever ROI and pose the caller currently has.

    Raises whatever `project_point` raises if the ROI is behind the camera -- "this region is not
    in front of the camera" is a real answer, not something to clip away silently.
    """
    pts = project_point(geom_full, np.asarray(roi_world, dtype=float).reshape(-1, 3))
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    render_w, render_h = render_size

    side = max(x1 - x0, y1 - y0)
    side = max(side * (1.0 + 2.0 * margin), float(min_crop_px))
    side = min(side, float(render_w), float(render_h))

    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = side / 2.0
    cx = float(np.clip(cx, half, render_w - half))
    cy = float(np.clip(cy, half, render_h - half))
    return (
        int(round(cx - half)), int(round(cy - half)),
        int(round(cx + half)), int(round(cy + half)),
    )


def attention_view(
    scene: Scene, roi_world: np.ndarray, *, camera: str = WRIST_CAM,
    render_size: tuple[int, int] = (512, 512), output_size: int = 256,
    margin: float = 0.25, min_crop_px: float = 48.0, name: str = "attention",
) -> LookPose:
    """A `LookPose` with `tcp=None` (it never moves the arm) whose crop keeps `roi_world` framed
    from wherever the wrist camera currently is.

    `scene.data` must already reflect the pose to look from (after `mj_forward`/`mj_step`, the
    same requirement `backproject.camera_geometry` documents) -- a closed-loop controller calls
    this after every arm motion, with the ROI set to whatever its next action is aimed at (the
    target stud cell plus the held brick, a tray pocket, ...), to get a fresh 256x256 view and its
    exact geometry via `render_view` right after.
    """
    fovy_deg = fovy_from_diagonal_fov(WRIST_CAM_DIAGONAL_FOV_DEG, *render_size)
    cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"unknown camera {camera!r}")
    scene.model.cam_fovy[cam_id] = fovy_deg
    mujoco.mj_forward(scene.model, scene.data)

    geom_full = camera_geometry(scene.model, scene.data, camera, *render_size)
    crop = attention_crop(geom_full, roi_world, render_size, margin=margin, min_crop_px=min_crop_px)
    return LookPose(
        name=name, camera=camera, render_size=render_size, crop=crop, fovy_deg=fovy_deg,
        output_size=output_size, tcp=None,
    )


def window_from_slots(
    state: SceneState, slots_of_interest: Sequence[int], margin: float = 0.25, *,
    min_crop_px: float = 48.0, default_half_extent: float = 0.08,
) -> LookPose:
    """A `LookPose` (`tcp=None`: it does not move the arm) whose crop tightly frames the given
    slots of `state`, computed ONLY from those slots' own published centre and extent (the
    second-moment ellipse's axis lengths) -- no decode, no ground truth. The slot-state
    counterpart to `attention_view`: a controller that only ever reads slot states (never a
    decode) still needs to narrow the next view onto "the target cell and the held brick", and
    this is how it does that, purely from slot numbers.

    `slots_of_interest` empty, or every one of them empty in `state`, returns `state.window`
    unchanged (there is nothing to narrow onto). A slot whose schema has no `moments` field (an
    opaque source) falls back to `default_half_extent` (normalized window units) around its
    centre instead of a measured extent.
    """
    window = state.window
    if not state.schema.has("centre"):
        raise KeyError("schema has no 'centre' field; window_from_slots needs slot centres")

    corners_out = []
    for k in slots_of_interest:
        if state.is_empty(k):
            continue
        cx, cy = state.centre(k)
        if state.schema.has("moments"):
            _, axes = state.orientation_and_axes(k)
            half = max(max(axes), 1e-6)
        else:
            half = default_half_extent
        col = (cx + 1.0) * 0.5 * (window.output_size - 1)
        row = (cy + 1.0) * 0.5 * (window.output_size - 1)
        half_px = half * 0.5 * (window.output_size - 1)
        corners_out.append((col - half_px, row - half_px))
        corners_out.append((col + half_px, row + half_px))

    if not corners_out:
        return window

    pts_full = output_to_full(window, np.array(corners_out))
    render_w, render_h = window.render_size
    x0, y0 = float(pts_full[:, 0].min()), float(pts_full[:, 1].min())
    x1, y1 = float(pts_full[:, 0].max()), float(pts_full[:, 1].max())

    side = max(x1 - x0, y1 - y0)
    side = max(side * (1.0 + 2.0 * margin), min_crop_px)
    side = min(side, float(render_w), float(render_h))
    cx_full, cy_full = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = side / 2.0
    cx_full = float(np.clip(cx_full, half, render_w - half))
    cy_full = float(np.clip(cy_full, half, render_h - half))
    crop = (
        int(round(cx_full - half)), int(round(cy_full - half)),
        int(round(cx_full + half)), int(round(cy_full + half)),
    )
    return replace(window, name="slot_window", crop=crop, tcp=None)
