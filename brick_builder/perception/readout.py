"""VIEWER AND EVALUATION ONLY -- not on the control path.

This project's design (see `scene_model.py`'s own module docstring) is that
the controller consumes `scene_model.SceneState`'s slot rows directly: a scene model's decode is
now optional, and exists only for a human viewer (draw what the model currently believes) or an
accuracy evaluation (compare a decoded, pixel-precise reading against ground truth). This module
is that decode-based reader. Nothing in `brick_builder.control` or a future `brick_builder.action` package
may import it -- `tests/perception/test_readout.py` enforces that with an import-boundary check.

What is built here: every fact below is reconstructed from `SceneState.decode`, the RGB frame, and
the view's camera geometry alone -- plus the "known planes" every real deployment would also have
to know ahead of time: the table top (world `z = 0`), the baseplate's top surface, and each brick
stacking layer's top surface (19.2 mm apart, DUPLO's own `height`). Nothing here ever reads a raw
`instance` array or any other ground-truth channel directly; it reads exactly what `SceneState.
decode` publishes, which for the ideal source happens to be exact, and for a real model would not
be. This is what makes it useful as an *evaluation* tool: comparing this module's pixel-precise
readout against ground truth measures how good a `decode` is, independent of the coarser, faster
slot-state reading (`slot_ops.py`) the controller actually uses.

Three readouts:

- `read_objects`: every brick on the plate (or wherever the view is pointed), as a `BrickReading`
  -- position/yaw/layer in the arm base frame (metres, radians), colour, footprint in studs,
  confidence, pixel count.
- `read_plate_grid`: occupancy and colour per stud cell per stacking layer.
- `read_tray`: which tray pockets hold a brick, and that brick's size/colour.

Two adjacent DUPLO colours pose a real, measured problem that shapes this module's design: the
baseplate's own default green is visually close enough to a green *brick* that colour alone
cannot tell them apart (a plate-sized region check does), and -- much more surprising -- the
stock SO-101's own plastic (`rgba 1 0.82 0.12`, from the vendored MJCF materials) sits almost on
top of DUPLO "yellow" in HSV-cone distance (measured ~0.055 against a separation tolerance of
0.40 that comfortably clears every other colour pair). Colour cannot be trusted to reject robot
structure the way it rejects everything else, so every candidate region here is also checked
geometrically: does its footprint, back-projected onto *some* legal plane, actually come out a
legal brick size at a position on the plate? A patch of yellow robot casing fails that check even
though its colour would have passed -- see `read_objects`'s own docstring for why this geometric
check, not `gripper_region_px`'s analytic jaw-silhouette projection (still provided as a
standalone utility), ended up doing the real work of rejecting the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from .backproject import CameraGeometry, backproject_to_plane, project_point
from .scene_model import SceneState
from .tokens import ObjectToken, grid_queries
from .views import roi_box_corners
from ..bricks.spec import DUPLO, PALETTE as BRICK_PALETTE, BrickSpec
from ..robot.patch_so101 import (
    WRIST_CAM_POS_IN_GRIPPER,
    WRIST_CAM_QUAT_IN_GRIPPER,
    gripper_angle_for_width,
    jaw_geometry,
)

# ------------------------------------------------------------------------------ known geometry ---
#
# Everything in this section is a *calibrated constant*, not something read off ground truth: a
# real deployment would get these numbers from the physical build (where the plate/tray were
# bolted down) and the robot's own CAD, not from a vision system.

DEFAULT_PLATE_XY: tuple[float, float] = (0.17, 0.0)
"""Build-plate centre, arm base frame -- matches `brick_builder.scene.compose.SceneConfig.plate_xy`'s
own default."""

DEFAULT_PLATE_STUDS: tuple[int, int] = (8, 8)

DEFAULT_TRAY_XY: tuple[float, float] = (0.13, -0.17)
"""Tray centre -- matches `SceneConfig.tray_xy`'s own default."""

TRAY_BRICK_SPACING_PITCHES = 4.5
"""Centre-to-centre spacing between tray pockets, in stud pitches -- matches `brick_builder.scene.
compose.TRAY_BRICK_SPACING`."""

TRAY_FLOOR_Z = 0.0025
"""World z of a tray brick's bottom face -- matches `compose.py`'s `_add_tray` (`floor=0.002`
slab thickness plus the `0.0005` clearance gap `add_brick` is placed at)."""

DEFAULT_COLOUR_TOL = 0.40
"""HSV-cone colour distance (`producers.hsv_colour_distance`'s own embedding) below which one
*pixel* counts as a confident match to a palette colour (see `classify_region_colour`). Same
value `producers.BlobProducer` defaults to, chosen there by measuring the worst-case self-distance
and best-case cross-distance of the six DUPLO colours under the full sampled post-process range
(see `producers.py`'s own comment); reused here because it is a property of the paint/rendering
under this per-pixel classification, not of which module is asking."""

BASEPLATE_COLOUR_NAME = "green"
"""The palette name of `brick_builder.bricks.spec.add_baseplate`'s default colour -- the one colour
`read_objects`/`read_plate_grid` cannot use colour alone to separate from a same-coloured brick."""

LEGAL_FOOTPRINT_STUDS = (1, 8)
"""Inclusive (min, max) stud count a fitted rectangle's side may round to and still be considered
a legal brick footprint -- generous on purpose (this module does not hard-code "2x2 or 2x4 only";
it accepts whatever rounds cleanly to an integer stud count in a sane range)."""

SIZE_TOLERANCE_FRAC = 0.30
"""How far a fitted side length may sit from the nearest whole number of stud pitches (as a
fraction of one pitch) and still count as "rounds cleanly" -- generous enough for oblique-view
pixel noise, tight enough to reject a robot part's arbitrary footprint."""

MIN_REGION_QUERY_POINTS = 3
"""A connected component with fewer query points than this is treated as noise, not an object."""

CANDIDATE_LAYERS = (0, 1, 2)
"""Stack layers `read_objects` tries a region against when no explicit `layers` is given -- 0 is
resting on the baseplate; DUPLO layers are `spec.height` (19.2 mm) apart."""

GRIPPER_OPEN_ANGLE = gripper_angle_for_width(0.05)
"""Gripper joint angle for an empty, fully open hand -- what every scene this module's own tests
render with (`SceneConfig(held=None, ...)`; see `compose.gripper_rest_angle`). `gripper_region_
px` uses this to project the jaws' own known footprint; a caller holding a brick should widen
`gripper_margin_px` rather than expect an exact silhouette at a different opening."""


def plate_top_z(spec: BrickSpec = DUPLO) -> float:
    """World z of the baseplate's flat top surface. The plate always sits flush on the table
    (`z = 0`) in every scene this project builds, so this is a constant, not a measurement --
    matches `brick_builder.scene.compose.Scene.plate_top`'s own formula exactly."""
    return 2.0 * spec.roof


def layer_top_z(layer: int, spec: BrickSpec = DUPLO) -> float:
    """World z of a brick's top (roof) face at 0-indexed stack `layer` (0 = resting directly on
    the baseplate) -- `brick_builder.bricks.structures.build_structure`'s own flush-stacking formula,
    read forward: layer `z`'s bottom face is `plate_top + z * height`, so its top is one more
    `height` up."""
    return plate_top_z(spec) + (layer + 1) * spec.height


def tray_brick_top_z(spec: BrickSpec = DUPLO) -> float:
    """World z of a tray brick's top face -- tray bricks never stack, so there is only one."""
    return TRAY_FLOOR_Z + spec.height


def tray_pocket_xy(index: int, n_pockets: int, tray_xy: tuple[float, float] = DEFAULT_TRAY_XY,
                    spec: BrickSpec = DUPLO) -> tuple[float, float]:
    """World (x, y) of tray pocket `index`'s centre -- matches `compose.py`'s `_add_tray`."""
    x = tray_xy[0] + (index - (n_pockets - 1) / 2.0) * TRAY_BRICK_SPACING_PITCHES * spec.pitch
    return x, tray_xy[1]


# ---------------------------------------------------------------------- the gripper's own region ---

_GRIPPER_CAMERA_FRAME_CORNERS: np.ndarray | None = None


def _quat_to_mat(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def _moving_jaw_point_in_gripper(point_local: np.ndarray, angle: float) -> np.ndarray:
    """A point in the moving jaw body's own frame -> the gripper body frame, at gripper joint
    `angle` [rad]. Re-derives (rather than imports) `brick_builder.scene.compose`'s equivalent private
    construction, from only the public `patch_so101.jaw_geometry()` fit -- `compose.py` belongs to
    another package and is read-only here."""
    geo = jaw_geometry()
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    rot_z = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    jaw_rot = _quat_to_mat(geo.jaw_quat)
    return jaw_rot @ (rot_z @ np.asarray(point_local, dtype=float)) + np.asarray(geo.jaw_pos, dtype=float)


def _gripper_region_camera_frame() -> np.ndarray:
    """Corners of the fixed + moving jaw's fitted collision boxes (`patch_so101.jaw_geometry`),
    in the wrist camera's own frame -- independent of render resolution, crop, or the rest of the
    arm's configuration, because the camera is rigidly bolted to the gripper body. Computed once
    (module-level cache) at `GRIPPER_OPEN_ANGLE`.
    """
    global _GRIPPER_CAMERA_FRAME_CORNERS
    if _GRIPPER_CAMERA_FRAME_CORNERS is not None:
        return _GRIPPER_CAMERA_FRAME_CORNERS

    geo = jaw_geometry()
    # The palm/hub boxes (nearest the wrist, farthest from the fingertips) sit close enough to
    # the camera's own mount, at a steep enough angle from its axis, that a fair few of their
    # corners fall well outside any view this module crops to -- see `GRIPPER_REGION_MAX_EXTENT_
    # FRAC`'s docstring. The fingertip pad and finger boxes are what a wrist-camera frame actually
    # shows of the gripper (the private jaws close to the scene, not the mounting hardware behind
    # them), and what the analytic region below needs to cover.
    boxes_gripper_frame = [geo.fixed_pad, *geo.fixed_boxes]
    boxes_moving_local = [geo.moving_pad, *geo.moving_boxes]

    corners = []
    for centre, half in boxes_gripper_frame:
        corners.append(roi_box_corners(centre, half))
    for centre, half in boxes_moving_local:
        local_corners = roi_box_corners(centre, half)
        corners.append(
            np.stack([_moving_jaw_point_in_gripper(c, GRIPPER_OPEN_ANGLE) for c in local_corners])
        )
    corners_gripper = np.concatenate(corners, axis=0)

    r_cg = _quat_to_mat(WRIST_CAM_QUAT_IN_GRIPPER)
    t_cg = np.asarray(WRIST_CAM_POS_IN_GRIPPER, dtype=float)
    # world->local for a fixed rotation/translation, row-vector convention (backproject.py's own):
    # v_gripper = v_camera @ R_cg.T + t_cg  =>  v_camera = (v_gripper - t_cg) @ R_cg
    _GRIPPER_CAMERA_FRAME_CORNERS = (corners_gripper - t_cg) @ r_cg
    return _GRIPPER_CAMERA_FRAME_CORNERS


GRIPPER_REGION_MAX_EXTENT_FRAC = 2.0
"""Some of the jaw assembly's own fitted boxes (the palm/hub nearest the wrist, farthest from the
fingertips) sit far enough off a *cropped* view's own (much narrower than the lens's own 115
degrees) effective field of view that a plain pinhole projection sends their corners to pixel
coordinates thousands of pixels outside the frame -- mathematically correct (nothing says a point
outside the crop's usable cone can't have a well-defined ray), but useless for a bounding box: a
handful of such corners would swamp `gripper_region_px`'s result and make it cover almost the
whole frame. Corners whose projected pixel lands further than this many half-frames from the
principal point cannot appear in the actual 256x256 image and are excluded before the bounding
box is taken; this is a pixel-space test (scaled by the view's own `cx`/`cy`) rather than a fixed
angle, because how much of the lens's own FOV a crop keeps varies a lot between views."""


def gripper_region_px(geometry: CameraGeometry, margin_px: float = 6.0) -> tuple[float, float, float, float]:
    """The wrist camera's own known pixel-space bounding box of its two jaws, for `geometry` (a
    view's resolved camera geometry) -- purely analytic (`patch_so101.jaw_geometry` plus the
    known, rigid camera mount), never measured from a particular render. This is the "known image
    region" half of `read_objects`'s gripper discard; a region's mean colour, checked separately
    per-candidate, would be the other half, except that (see the module docstring) the robot's own
    plastic is a near-exact colour match for DUPLO yellow, so this region check -- not colour --
    is what actually has to do the work for a yellow brick near the jaws.
    """
    corners_cam = _gripper_region_camera_frame()
    corners_world = corners_cam @ geometry.rot.T + geometry.pos
    rel = corners_world - geometry.pos
    local = rel @ geometry.rot
    in_front = local[:, 2] < -1e-9
    if not np.any(in_front):
        return (math.inf, math.inf, -math.inf, -math.inf)  # nothing in view; an empty box

    px_all = project_point(geometry, corners_world[in_front])
    scale = max(geometry.cx, geometry.cy, 1.0)
    limit = GRIPPER_REGION_MAX_EXTENT_FRAC * scale
    plausible = (np.abs(px_all[:, 0] - geometry.cx) <= limit) & (np.abs(px_all[:, 1] - geometry.cy) <= limit)
    if not np.any(plausible):
        return (math.inf, math.inf, -math.inf, -math.inf)

    px = px_all[plausible]
    x0 = float(px[:, 0].min()) - margin_px
    y0 = float(px[:, 1].min()) - margin_px
    x1 = float(px[:, 0].max()) + margin_px
    y1 = float(px[:, 1].max()) + margin_px
    return x0, y0, x1, y1


# --------------------------------------------------------------------------------- colour match ---


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.float64)
    if frame.max(initial=0.0) > 1.0 + 1e-6:
        frame = frame / 255.0
    return frame[..., :3]


def _hsv_cone_pixels(rgb01: np.ndarray, chroma_gain: float = 2.0) -> np.ndarray:
    """Vectorized per-pixel HSV-cone embedding, `[n, 3]` in `[0, 1]` -> `[n, 3]` -- the same
    embedding `producers.hsv_colour_distance`/`_hsv_cone` compare one colour at a time, applied
    here to a whole list of pixels via `cv2.cvtColor` for speed (mirrors `producers._hsv_cone_
    image`'s own vectorization, for an arbitrary pixel list rather than a whole image array)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb01, dtype=np.float32).reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
    theta = np.deg2rad(hsv[:, 0])
    s = hsv[:, 1]
    v = hsv[:, 2]
    return np.stack([chroma_gain * s * np.cos(theta), chroma_gain * s * np.sin(theta), v], axis=-1)


_PALETTE_NAMES = list(BRICK_PALETTE)
_PALETTE_CONES = _hsv_cone_pixels(np.array([BRICK_PALETTE[n] for n in _PALETTE_NAMES], dtype=np.float64))


def classify_region_colour(
    pixel_rgb: np.ndarray, palette: dict[str, tuple[float, float, float]] = BRICK_PALETTE,
    tol: float = DEFAULT_COLOUR_TOL,
) -> tuple[str | None, float]:
    """Classifies a region's colour by a per-pixel plurality vote, not the region's mean colour.

    Under MuJoCo's own Phong-style shading a brick's pixels range from near-white specular
    highlights to near-black shadow (measured on this project's own renders); averaging RGB
    first washes a saturated colour toward grey/white before it is ever compared to a reference
    -- a lit red brick's *mean* pixel measured 0.44+ from "red" in this metric, well outside
    `DEFAULT_COLOUR_TOL`, even though the great majority of its *individual* pixels are a
    confident match. This is exactly `producers.BlobProducer`'s own approach (classify pixels,
    not a region average), applied here to an arbitrary (non-rectangular) set of pixels rather
    than a whole image: each pixel in `pixel_rgb` votes for its own nearest palette colour if that
    distance is within `tol`; the region's colour is the plurality winner among confident votes,
    and the returned "confidence" is that winner's vote share of *all* pixels (not just the
    confident ones), so a region half specular highlight and half a confident colour still scores
    below 1. Returns `(None, 0.0)` if no pixel voted confidently for anything.
    """
    pixel_rgb = np.asarray(pixel_rgb, dtype=np.float64).reshape(-1, 3)
    if pixel_rgb.max(initial=0.0) > 1.0 + 1e-6:
        pixel_rgb = pixel_rgb / 255.0
    if palette is BRICK_PALETTE:
        names, cones = _PALETTE_NAMES, _PALETTE_CONES
    else:
        names = list(palette)
        cones = _hsv_cone_pixels(np.array([palette[n] for n in names], dtype=np.float64))

    pixel_cones = _hsv_cone_pixels(pixel_rgb)
    dists = np.linalg.norm(pixel_cones[:, None, :] - cones[None, :, :], axis=-1)  # [n, len(palette)]
    nearest = np.argmin(dists, axis=1)
    nearest_d = dists[np.arange(len(dists)), nearest]
    confident = nearest_d <= tol
    if not np.any(confident):
        return None, 0.0
    counts = np.bincount(nearest[confident], minlength=len(names))
    winner = int(np.argmax(counts))
    return names[winner], float(counts[winner]) / float(pixel_rgb.shape[0])


# -------------------------------------------------------------------- rectangle pose refinement ---


def fold_yaw_symmetry(yaw: float, nx: int, ny: int) -> float:
    """Folds `yaw` [rad] into the canonical range implied by an `nx`x`ny` rectangular footprint's
    own rotational symmetry: a square footprint (`nx == ny`) looks the same every 90 degrees, any
    other rectangle every 180 -- the same grasp symmetry `AlignFromTokensEnv._grasp_yaw` folds a
    commanded tool spin into (`docs/guide/06-perception.md`), applied here so a fitted
    footprint's yaw and a comparison ground truth's yaw are not spuriously ~180 (or ~90) degrees
    apart for a brick that looks identical either way.
    """
    period = (math.pi / 2.0) if nx == ny else math.pi
    return math.remainder(yaw, period)


def _rect_membership(
    geometry: CameraGeometry, plane_z: float, centre_xy: tuple[float, float], yaw: float,
    size_xy: tuple[float, float], query_px: np.ndarray,
) -> np.ndarray:
    """`[n]` bool: which of `query_px` back-project (onto `z = plane_z`) inside the `size_xy`
    rectangle centred at `centre_xy`, rotated `yaw` -- the geometric core of `refine_rect_pose`."""
    try:
        world = backproject_to_plane(geometry, np.asarray(query_px, dtype=float), plane_z=plane_z)
    except ValueError:
        return np.zeros(len(query_px), dtype=bool)
    dx = world[:, 0] - centre_xy[0]
    dy = world[:, 1] - centre_xy[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    return (np.abs(local_x) <= size_xy[0] / 2.0) & (np.abs(local_y) <= size_xy[1] / 2.0)


def refine_rect_pose(
    geometry: CameraGeometry, plane_z: float, size_xy: tuple[float, float],
    query_px: np.ndarray, target_mask: np.ndarray | None = None, *,
    init_xy: tuple[float, float], init_yaw: float = 0.0,
    iters: int = 40,
    init_step: tuple[float, float, float] = (0.006, 0.006, math.radians(6.0)),
    min_step: tuple[float, float, float] = (0.00005, 0.00005, math.radians(0.02)),
) -> tuple[float, float, float, float]:
    """Coordinate/pattern search for the `(x, y, yaw)` on `z = plane_z` of a known `size_xy`
    rectangle that best explains the observation: which of `query_px` are "on" is given by
    `target_mask` (bool, same length as `query_px`) -- every point in `query_px` if `target_mask`
    is omitted, the common case when `query_px` is already exactly one slot's own claimed points.

    A rotated rectangle's IoU against a fixed point set is a step function (zero gradient almost
    everywhere: it only changes when a query point crosses the rectangle's boundary), so a
    gradient method has nothing to follow -- coordinate/pattern search, which only ever asks "is
    this candidate better than the current one", is the right tool here, not Gauss-Newton. This
    is also `compare.refine_pose`'s underlying primitive: fitting a *believed* entity's pose to an
    *observed* slot is the exact same "known rectangle vs. an observed point set" problem.

    Returns `(x, y, yaw, iou)`; `iou` is the achieved score (plain IoU against `target_mask`, or
    the fraction of `query_px` explained when `target_mask` is omitted).
    """
    query_px = np.asarray(query_px, dtype=float)
    if target_mask is None:
        target_mask = np.ones(len(query_px), dtype=bool)
    else:
        target_mask = np.asarray(target_mask, dtype=bool)

    def score(x: float, y: float, yaw: float) -> float:
        pred = _rect_membership(geometry, plane_z, (x, y), yaw, size_xy, query_px)
        inter = np.logical_and(pred, target_mask).sum()
        union = np.logical_or(pred, target_mask).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    x, y, yaw = float(init_xy[0]), float(init_xy[1]), float(init_yaw)
    step = np.array(init_step, dtype=float)
    floor = np.array(min_step, dtype=float)
    best = score(x, y, yaw)
    for _ in range(iters):
        improved = False
        for axis in range(3):
            for sign in (1.0, -1.0):
                cand = [x, y, yaw]
                cand[axis] = cand[axis] + sign * step[axis]
                s = score(*cand)
                if s > best:
                    x, y, yaw = cand
                    best = s
                    improved = True
        if not improved:
            step *= 0.5
            if np.all(step < floor):
                break
    return x, y, yaw, best


# ------------------------------------------------------------------------------- object tokens ---


@dataclass
class BrickReading(ObjectToken):
    """A single detected brick, extending `tokens.ObjectToken` with what `read_objects` measures
    beyond the base producer interface.

    `x`, `y`, `yaw` (inherited) are now **metric** (metres, radians, arm base frame) -- the later
    back-projection step `docs/guide/06-perception.md` describes `tokens.py`'s own pixel-
    unit fields as waiting for. `size` (inherited) is the footprint's diagonal, in metres.

    `nx`, `ny`: footprint in studs (order matches the fitted rectangle's own local x/y axes, not
    necessarily the world x/y axes -- `yaw` carries the actual orientation).
    `pixel_count`: full-resolution pixel count estimate for the region (query points times the
    query grid's own step squared).
    `raw_centroid_px`: the *un-corrected* pixel centroid of the region's own query points, before
    the oblique-view footprint fit -- kept so a caller (or a test) can measure the fit's own
    improvement over the naive centroid, exactly as the specification asks.
    """

    nx: int = 0
    ny: int = 0
    pixel_count: int = 0
    raw_centroid_px: tuple[float, float] = (0.0, 0.0)


def _legal_footprint(width_m: float, height_m: float, spec: BrickSpec) -> tuple[int, int] | None:
    nx = round(width_m / spec.pitch)
    ny = round(height_m / spec.pitch)
    lo, hi = LEGAL_FOOTPRINT_STUDS
    if not (lo <= nx <= hi and lo <= ny <= hi):
        return None
    tol = SIZE_TOLERANCE_FRAC * spec.pitch
    if abs(width_m - nx * spec.pitch) > tol or abs(height_m - ny * spec.pitch) > tol:
        return None
    return int(nx), int(ny)


def _within_plate_bounds(
    x: float, y: float, plate_xy: tuple[float, float], plate_studs: tuple[int, int],
    spec: BrickSpec, margin: float,
) -> bool:
    half_x = 0.5 * plate_studs[0] * spec.pitch + margin
    half_y = 0.5 * plate_studs[1] * spec.pitch + margin
    return abs(x - plate_xy[0]) <= half_x and abs(y - plate_xy[1]) <= half_y


def _looks_like_baseplate(
    region_px: np.ndarray, geometry: CameraGeometry, plate_xy: tuple[float, float],
    plate_studs: tuple[int, int], spec: BrickSpec,
) -> bool:
    try:
        world = backproject_to_plane(geometry, region_px.astype(float), plane_z=plate_top_z(spec))
    except ValueError:
        return False
    xy = world[:, :2].astype(np.float32)
    if xy.shape[0] < 3:
        return False
    (cx, cy), (w, h), _ = cv2.minAreaRect(xy)
    expected_w = plate_studs[0] * spec.pitch
    expected_h = plate_studs[1] * spec.pitch
    tol = 0.35
    size_ok = (
        abs(w - expected_w) < tol * expected_w and abs(h - expected_h) < tol * expected_h
    ) or (
        abs(w - expected_h) < tol * expected_h and abs(h - expected_w) < tol * expected_w
    )
    centred = math.hypot(cx - plate_xy[0], cy - plate_xy[1]) < 0.5 * spec.pitch * max(plate_studs)
    return bool(size_ok and centred)


def _fit_region(
    full_px: np.ndarray, full_slot: np.ndarray, slot: int, region_px: np.ndarray,
    geometry: CameraGeometry, rgb: np.ndarray, *, spec: BrickSpec,
    plate_xy: tuple[float, float], plate_studs: tuple[int, int], colour_tol: float,
    query_step: int, correct_oblique: bool, layers: tuple[int, ...],
) -> BrickReading | None:
    frame = _as_float_rgb(rgb)
    cols = region_px[:, 0].astype(int)
    rows = region_px[:, 1].astype(int)
    colour_name, colour_conf = classify_region_colour(frame[rows, cols], tol=colour_tol)
    if colour_name is None:
        return None  # not confidently any brick colour: background, robot, whatever else
    centroid_px = (float(region_px[:, 0].mean()), float(region_px[:, 1].mean()))

    if colour_name == BASEPLATE_COLOUR_NAME and _looks_like_baseplate(
        region_px, geometry, plate_xy, plate_studs, spec
    ):
        return None

    best = None
    for layer in layers:
        plane_z = layer_top_z(layer, spec)
        try:
            world = backproject_to_plane(geometry, region_px.astype(float), plane_z=plane_z)
        except ValueError:
            continue
        xy = world[:, :2].astype(np.float32)
        if xy.shape[0] < 3:
            continue
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
        footprint = _legal_footprint(w, h, spec)
        if footprint is None:
            continue
        nx, ny = footprint
        if not _within_plate_bounds(cx, cy, plate_xy, plate_studs, spec, margin=spec.pitch):
            continue
        residual = abs(w - nx * spec.pitch) + abs(h - ny * spec.pitch)
        if best is None or residual < best[0]:
            best = (residual, layer, plane_z, float(cx), float(cy), math.radians(angle_deg), nx, ny)

    if best is None:
        return None
    _, layer, plane_z, cx, cy, yaw0, nx, ny = best

    if correct_oblique:
        x0, y0 = region_px[:, 0].min(), region_px[:, 1].min()
        x1, y1 = region_px[:, 0].max(), region_px[:, 1].max()
        pad = 0.6 * max(x1 - x0, y1 - y0, 4 * query_step)
        local = (
            (full_px[:, 0] >= x0 - pad) & (full_px[:, 0] <= x1 + pad)
            & (full_px[:, 1] >= y0 - pad) & (full_px[:, 1] <= y1 + pad)
        )
        local_px = full_px[local]
        local_target = full_slot[local] == slot
        fit_x, fit_y, fit_yaw, iou = refine_rect_pose(
            geometry, plane_z, (nx * spec.pitch, ny * spec.pitch), local_px, local_target,
            init_xy=(cx, cy), init_yaw=yaw0,
        )
    else:
        fit_x, fit_y, fit_yaw, iou = cx, cy, yaw0, 1.0

    fit_yaw = fold_yaw_symmetry(fit_yaw, nx, ny)
    size_m = math.hypot(nx * spec.pitch, ny * spec.pitch)
    confidence = float(np.clip(iou, 0.0, 1.0)) * float(np.clip(colour_conf, 0.0, 1.0))
    return BrickReading(
        x=fit_x, y=fit_y, yaw=fit_yaw, layer=int(layer), colour=colour_name, size=size_m,
        confidence=confidence, age=0.0, slot=int(slot),
        nx=nx, ny=ny, pixel_count=int(region_px.shape[0]) * query_step * query_step,
        raw_centroid_px=centroid_px,
    )


def read_objects(
    scene_model: SceneState, rgb: np.ndarray, view: CameraGeometry, *,
    query_step: int = 3, plate_xy: tuple[float, float] = DEFAULT_PLATE_XY,
    plate_studs: tuple[int, int] = DEFAULT_PLATE_STUDS, spec: BrickSpec = DUPLO,
    colour_tol: float = DEFAULT_COLOUR_TOL,
    layers: tuple[int, ...] = CANDIDATE_LAYERS, correct_oblique: bool = True,
) -> list[BrickReading]:
    """Every brick visible in `view`, decoded from `scene_model` and `rgb` alone.

    Pipeline: decode an evenly spaced query grid (`query_step` pixels apart); connected components
    per slot (the background slot is never special-cased explicitly -- see the module docstring --
    it just never survives the colour/geometry checks below); for each remaining region, colour-
    match it to the six DUPLO colours (reject anything that doesn't confidently match one -- this
    is what actually discards the background slot and most of the robot), discard it if it looks
    like the baseplate (colour + plate-sized + centred on the plate), then try each of `layers`'
    top planes and keep the one whose back-projected footprint rounds cleanly to a legal brick
    size at a position on the plate -- which is also what rejects the rest of the robot (including
    the wrist, whose casing shares DUPLO yellow's own colour almost exactly: see the module
    docstring), since nothing at the robot's own, much closer, range projects to a plausible
    on-plate brick footprint. `gripper_region_px` (this module's analytic, "known image region"
    projection of the two jaws) is provided as a standalone utility rather than wired into this
    gate: measured on this camera (mounted a few centimetres from a wide-angle lens), the jaws'
    own projected silhouette spans a large enough vertical band of every frame that gating on it
    directly cost real on-plate bricks near the bottom of frame far more often than it caught
    robot structure the colour/geometry checks above were not already catching. The winning
    plane's raw `cv2.minAreaRect` centroid is then refined (`correct_oblique=True`, the default)
    by fitting the *known-size* rectangle to the region's own query points via `refine_rect_pose`
    -- correcting the bias a brick's visible side faces (the camera looks forward and down at ~25
    degrees off vertical) put into the raw centroid.
    """
    res = scene_model.res
    px = grid_queries(query_step, res=res)
    logits = scene_model.decode(px)
    slot_idx = np.argmax(logits, axis=-1)

    side = int(np.arange(query_step // 2, res, query_step).shape[0])
    slot_grid = slot_idx.reshape(side, side)

    readings: list[BrickReading] = []
    for slot in (int(s) for s in np.unique(slot_idx)):
        mask = (slot_grid == slot).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        for label in range(1, n_labels):
            flat = (labels == label).ravel()
            if int(flat.sum()) < MIN_REGION_QUERY_POINTS:
                continue
            region_px = px[flat]
            reading = _fit_region(
                px, slot_idx, slot, region_px, view, rgb, spec=spec, plate_xy=plate_xy,
                plate_studs=plate_studs, colour_tol=colour_tol,
                query_step=query_step, correct_oblique=correct_oblique, layers=layers,
            )
            if reading is not None:
                readings.append(reading)
    return readings


# ------------------------------------------------------------------------------- the plate grid ---


@dataclass
class PlateGrid:
    """`read_plate_grid`'s result: occupancy and colour per stud cell, per requested layer.

    `occupancy[layer]` and `colour[layer]` are `[nx, ny]` arrays (`plate_studs` order), indexed
    `[gx, gy]` exactly like `brick_builder.bricks.structures.Placement.cells()`. `colour[layer][gx,gy]`
    is `""` wherever `occupancy[layer][gx,gy]` is `False`.
    """

    layers: tuple[int, ...]
    plate_studs: tuple[int, int]
    occupancy: dict[int, np.ndarray]
    colour: dict[int, np.ndarray]


def read_plate_grid(
    scene_model: SceneState, rgb: np.ndarray, view: CameraGeometry,
    layers: tuple[int, ...] = (0, 1, 2), *,
    plate_xy: tuple[float, float] = DEFAULT_PLATE_XY, plate_studs: tuple[int, int] = DEFAULT_PLATE_STUDS,
    plate_yaw: float = 0.0, spec: BrickSpec = DUPLO, colour_tol: float = DEFAULT_COLOUR_TOL,
    sample_radius_px: int = 2,
) -> PlateGrid:
    """Occupancy and colour per stud cell per layer, by decoding the projected stud-centre pixels.

    For each `(cell, layer)`, the cell's stud centre is projected onto that layer's top plane and
    decoded (a small `sample_radius_px` neighbourhood, majority-voted, rather than one bare pixel,
    for robustness against antialiasing at an edge). A cell counts as occupied when the sampled
    colour confidently matches one of the six DUPLO colours -- with one deliberate exception: at
    layer 0, a confident *baseplate*-colour match means "the bare plate is showing through", not
    "there is a green brick here" (the same collision `read_objects` handles with a size check;
    here, at layer 0 specifically, colour alone is enough because nothing else at that height
    could be that colour).

    Probing several hypothetical layer heights independently can alias: a cell that is genuinely
    occupied only at layer 0 still decodes to *something* when probed at layer 1's slightly higher
    assumed plane, because the projected pixel only moves a little for one brick-height's worth of
    depth change. When the very same slot is found occupying a cell at more than one probed layer,
    this function keeps only the *lowest* such layer (probing higher than the real surface still
    lands on the real, lower one; the reverse is not physically possible) and clears the rest.
    """
    nx_studs, ny_studs = plate_studs
    res = scene_model.res
    frame = _as_float_rgb(rgb)

    occupancy = {L: np.zeros((nx_studs, ny_studs), dtype=bool) for L in layers}
    colour = {L: np.full((nx_studs, ny_studs), "", dtype=object) for L in layers}
    slot_seen = {L: np.full((nx_studs, ny_studs), -1, dtype=np.int64) for L in layers}

    stud_x0 = -(nx_studs - 1) / 2.0 * spec.pitch
    stud_y0 = -(ny_studs - 1) / 2.0 * spec.pitch
    cos_yaw, sin_yaw = math.cos(plate_yaw), math.sin(plate_yaw)

    for layer in layers:
        plane_z = layer_top_z(layer, spec)
        for gx in range(nx_studs):
            for gy in range(ny_studs):
                lx = stud_x0 + gx * spec.pitch
                ly = stud_y0 + gy * spec.pitch
                wx = plate_xy[0] + cos_yaw * lx - sin_yaw * ly
                wy = plate_xy[1] + sin_yaw * lx + cos_yaw * ly
                try:
                    col, row = project_point(view, np.array([wx, wy, plane_z]))
                except ValueError:
                    continue
                if not (0 <= col < res and 0 <= row < res):
                    continue
                c0 = max(int(round(col)) - sample_radius_px, 0)
                r0 = max(int(round(row)) - sample_radius_px, 0)
                c1 = min(int(round(col)) + sample_radius_px + 1, res)
                r1 = min(int(round(row)) + sample_radius_px + 1, res)
                sc, sr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
                sample_px = np.stack([sc.ravel(), sr.ravel()], axis=-1)
                slots = np.argmax(scene_model.decode(sample_px), axis=-1)
                slot = int(np.bincount(slots).argmax())
                name, _conf = classify_region_colour(frame[sr.ravel(), sc.ravel()], tol=colour_tol)

                slot_seen[layer][gx, gy] = slot
                if name is None:
                    continue
                if layer == 0 and name == BASEPLATE_COLOUR_NAME:
                    continue
                occupancy[layer][gx, gy] = True
                colour[layer][gx, gy] = name

    for gx in range(nx_studs):
        for gy in range(ny_studs):
            occupied_layers = [L for L in layers if occupancy[L][gx, gy]]
            if len(occupied_layers) < 2:
                continue
            slots_here = {slot_seen[L][gx, gy] for L in occupied_layers}
            if len(slots_here) == 1:
                keep = min(occupied_layers)
                for L in occupied_layers:
                    if L != keep:
                        occupancy[L][gx, gy] = False
                        colour[L][gx, gy] = ""

    return PlateGrid(layers=tuple(layers), plate_studs=plate_studs, occupancy=occupancy, colour=colour)


# ------------------------------------------------------------------------------------ the tray ---


@dataclass
class TrayReading:
    """One tray pocket's state."""

    index: int
    occupied: bool
    colour: str = ""
    nx: int = 0
    ny: int = 0


def read_tray(
    scene_model: SceneState, rgb: np.ndarray, view: CameraGeometry, *,
    tray_xy: tuple[float, float] = DEFAULT_TRAY_XY, tray_bricks: int = 2,
    tray_brick: tuple[int, int] = (2, 4), spec: BrickSpec = DUPLO,
    colour_tol: float = DEFAULT_COLOUR_TOL, query_step: int = 3,
) -> list[TrayReading]:
    """Which of the tray's `tray_bricks` pockets hold a brick, with that brick's size and colour.

    Each pocket's known centre (`tray_pocket_xy`) is queried with a small local grid at the tray
    brick's known top-face height; an occupied pocket is refit with `cv2.minAreaRect` (the pocket
    spacing is wide enough, `brick_builder.scene.compose.TRAY_BRICK_SPACING`'s own docstring explains
    why, that neighbouring pockets' regions do not merge).
    """
    res = scene_model.res
    frame = _as_float_rgb(rgb)
    plane_z = tray_brick_top_z(spec)
    half_w = 0.75 * tray_brick[0] * spec.pitch
    half_h = 0.75 * tray_brick[1] * spec.pitch

    out: list[TrayReading] = []
    for index in range(tray_bricks):
        cx, cy = tray_pocket_xy(index, tray_bricks, tray_xy, spec)
        try:
            centre_px = project_point(view, np.array([cx, cy, plane_z]))
        except ValueError:
            out.append(TrayReading(index=index, occupied=False))
            continue
        if not (0 <= centre_px[0] < res and 0 <= centre_px[1] < res):
            out.append(TrayReading(index=index, occupied=False))
            continue

        world_corners = np.array([
            [cx - half_w, cy - half_h, plane_z], [cx - half_w, cy + half_h, plane_z],
            [cx + half_w, cy - half_h, plane_z], [cx + half_w, cy + half_h, plane_z],
        ])
        try:
            corner_px = project_point(view, world_corners)
        except ValueError:
            out.append(TrayReading(index=index, occupied=False))
            continue
        c0 = max(int(np.floor(corner_px[:, 0].min())), 0)
        c1 = min(int(np.ceil(corner_px[:, 0].max())), res)
        r0 = max(int(np.floor(corner_px[:, 1].min())), 0)
        r1 = min(int(np.ceil(corner_px[:, 1].max())), res)
        if c1 <= c0 or r1 <= r0:
            out.append(TrayReading(index=index, occupied=False))
            continue

        cols = np.arange(c0, c1, max(query_step, 1))
        rows = np.arange(r0, r1, max(query_step, 1))
        gc, gr = np.meshgrid(cols, rows)
        pocket_px = np.stack([gc.ravel(), gr.ravel()], axis=-1)
        centre_slot = int(np.argmax(scene_model.decode(np.array([centre_px])), axis=-1)[0])
        slots = np.argmax(scene_model.decode(pocket_px), axis=-1)
        own = slots == centre_slot
        if int(own.sum()) < MIN_REGION_QUERY_POINTS:
            out.append(TrayReading(index=index, occupied=False))
            continue

        region_px = pocket_px[own]
        name, _conf = classify_region_colour(frame[region_px[:, 1], region_px[:, 0]], tol=colour_tol)
        if name is None:
            out.append(TrayReading(index=index, occupied=False))
            continue

        try:
            world = backproject_to_plane(view, region_px.astype(float), plane_z=plane_z)
        except ValueError:
            out.append(TrayReading(index=index, occupied=True, colour=name))
            continue
        xy = world[:, :2].astype(np.float32)
        if xy.shape[0] >= 3:
            (_, _), (w, h), _ = cv2.minAreaRect(xy)
            footprint = _legal_footprint(w, h, spec)
        else:
            footprint = None
        nx, ny = footprint if footprint is not None else (0, 0)
        out.append(TrayReading(index=index, occupied=True, colour=name, nx=nx, ny=ny))
    return out
