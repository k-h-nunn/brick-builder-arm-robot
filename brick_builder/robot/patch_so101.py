"""Load the vendored SO-101 MJCF and patch it in memory so it behaves like the real arm.

The upstream file (``third_party/so101/so101_new_calib.xml``) is never edited.  Everything here is
applied to a :class:`mujoco.MjSpec` at load time, so a new upstream release can be dropped in and the
same patches re-applied.

Three things upstream gets wrong for a brick-pressing task:

1. ``kp=998.22``.  That number comes from a derivation that assumes the servo firmware measures its
   position error in degrees.  It does not.  Rhoban's BAM project put a scope on an STS3215 and
   measured the real law, which gives about 9.4 N*m/rad at the P value LeRobot writes (16).  Press
   force is essentially ``kp * error`` up to saturation, so a gain that is 100x too stiff makes every
   press look easy.
2. ``forcerange=+-3.35`` (and the ``sts3215`` class's +-2.94).  Those are 12 V numbers.  The 7.4 V
   part stalls at about 1.9 N*m.
3. Collision geometry is the full visual mesh.  MuJoCo collides against a mesh's *convex hull*, and
   the hull of ``wrist_roll_follower`` (the fixed jaw) fills the gap between the jaws, so nothing can
   ever be grasped.  We replace all of it with convex primitives fitted to the STL vertices.

Constants below are BAM's identified parameters for the 7.4 V STS3215
(https://github.com/Rhoban/bam, ``bam/params/feetech_sts3215_7_4V/m1.json``, arXiv 2410.08650).
They reproduce, to three decimals, every number the design report quotes: kp 9.4 at P=16 and 18.8 at
P=32, joint damping kt^2/R + viscous = 0.62, frictionloss 0.052, armature 0.026, and the 3.4 N*m the
law implies at full duty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------------------- names ---

ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
"""The six actuated joints, in servo-ID order (this is also the actuator order in the MJCF)."""

TCP_SITE = "tcp_pinch"
"""Site between the fingertips; its z axis is the approach direction (out past the fingertips)."""

WRIST_CAM = "wrist_cam"
"""Camera rigidly attached to the ``gripper`` body -- never to the moving jaw."""

GRIPPER_BODY = "gripper"
MOVING_JAW_BODY = "moving_jaw_so101_v1"
FIXED_PAD_GEOM = "fixed_pad"
MOVING_PAD_GEOM = "moving_pad"

VENDOR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "so101"
VENDOR_XML = VENDOR_DIR / "so101_new_calib.xml"

# ------------------------------------------------------------------------------ BAM constants ---

BAM_KT = 1.1776311631627974
"""Torque constant at the output shaft [N*m/A]."""

BAM_R = 2.478681934101461
"""Motor resistance [Ohm]."""

BAM_ERROR_GAIN = 0.166
"""Firmware factor turning ``P * error_rad`` into a PWM duty cycle; measured with a scope."""

BAM_ERROR_GAIN_RATIO = 1.0049367060648848
"""Per-unit correction on top of the firmware factor, identified by BAM."""

BAM_MAX_DUTY = 0.97
"""Largest duty cycle the firmware will command."""

BAM_ARMATURE = 0.02608291141307205
"""Apparent rotor inertia seen at the output [kg*m^2]."""

BAM_FRICTION_BASE = 0.05152047758105413
"""Dry friction in the gear train [N*m] -> MuJoCo ``frictionloss``."""

BAM_FRICTION_VISCOUS = 0.05989161871380604
"""Mechanical viscous friction [N*m*s/rad]."""

BAM_MAX_VELOCITY = 5.2883481319359
"""Rate limit the firmware applies to its internal target [rad/s]; used by ``servo_model``."""

NOMINAL_VIN = 7.4
"""Supply voltage the torque limit in :class:`ArmConfig` refers to."""

BACKLASH_CTRL_MARGIN_DEG = 1.0
"""Extra control range each actuator gets once the free-play joints exist [deg].

``ServoBus`` writes a motor-side command (``target`` minus the play angle) into ``data.ctrl``, so
the control range has to hold one play's worth more than the joint's own range or MuJoCo clamps the
offset off near the stops.  One degree covers the whole published randomization range for
``dynamics.backlash_deg`` (0.2-1.0), so re-randomizing the play between episodes needs no rebuild.
"""

# ---------------------------------------------------------------------------- fitted geometry ---

PAD_THICKNESS = 0.0012
"""Fingertip pad thickness [m].  The pad is carved out of the fingertip box, not glued on top."""

NOMINAL_GRIP_WIDTH = 0.032
"""Width of the object the TCP is defined for: a DUPLO brick gripped across its 32 mm face."""

GRIP_TIP_OVERHANG = 0.008
"""How far a gripped brick hangs past the fingertips [m].

The stock jaws taper and swing on a hinge, so at a 32 mm opening the gap is widest right at the
tips and narrows going up the finger.  A 32 mm brick therefore cannot be pushed deep into the jaw;
it is held by the fingertips with its lower edge past them.  This is the grasp the TCP is defined
for (design report 01 leaves "does the stock jaw grasp a 32 mm DUPLO wall" as an open question --
it does, near the tips).
"""

_FIXED_JAW_MESH = "wrist_roll_follower_so101_v1"
_MOVING_JAW_MESH = "moving_jaw_so101_v1"

# Split planes read off the meshes: below them the part is a slender finger, above it a palm/hub
# that must not be allowed to reach into the jaw gap.
_FIXED_KNUCKLE_Z = -0.045   # gripper-body frame; the finger points along -z
_MOVING_KNUCKLE_Y = -0.0255  # moving-jaw-body frame; the finger points along -y

_FINGER_BOXES = 2  # boxes per finger; the design report asks for 2-3 per jaw


@dataclass(frozen=True)
class ArmConfig:
    """Everything about the arm a caller may want to randomize or dial in.

    ``torque_limit_nm`` is the stall torque at ``NOMINAL_VIN``; the actual forcerange is scaled by
    ``vin / 7.4`` because a servo's stall torque is proportional to supply voltage.
    """

    p_gain: int = 16               # STS3215 P coefficient (LeRobot writes 16; factory default 32)
    vin: float = 7.4               # supply voltage; scales both gain and torque limit
    torque_limit_nm: float = 1.9   # stall torque at 7.4 V (the 12 V part gives 2.94)
    backlash_deg: float = 0.435    # HALF-range of the free-play joints; 0 disables them
    """Half of the total lost motion at the output shaft [deg].

    The default is the measured 0.87 deg of total play halved, not the datasheet's 0.5 deg: the
    0.5 was being read as a half-range, which put twice the specified play in the model, and the
    real part misses the specification anyway.  See :func:`_add_backlash`.
    """
    pad_friction: float = 1.0
    joint_friction_nm: float = BAM_FRICTION_BASE
    """Dry friction in one servo's gear train [N*m] -> MuJoCo ``frictionloss`` on every actuated
    joint (see :func:`_patch_joints`).  Defaults to the BAM-identified value that used to be
    hard-coded there, so the compiled model is unchanged unless a caller sets this explicitly."""
    print_color_rgba: tuple[float, float, float, float] = (1.0, 0.82, 0.12, 1.0)
    """RGBA applied to every printed-part material (see :func:`_patch_print_color`); leaves the
    two STS3215 servo-casing materials alone, since those are the manufacturer's black plastic,
    not a part this team prints.  Default is the upstream mesh's own colour -- this team's real
    printed arm may be a different colour (unmeasured; see ``docs/guide/10``)."""
    primitive_collision: bool = True
    timestep: float = 0.002


# ------------------------------------------------------------------------------- the servo law ---


def servo_kp(p_gain: int, vin: float) -> float:
    """Position gain in N*m/rad from the measured firmware law (Rhoban BAM).

    The firmware computes ``duty = clip(err * P * 0.166 * ratio, +-0.97)`` and the motor then
    produces ``tau = kt * (Vin * duty - kt * dq) / R``.  Around zero error and zero velocity that is
    a linear spring of stiffness ``(kt / R) * Vin * 0.166 * ratio * P``.

    Gives 9.384 at (16, 7.4) and 18.768 at (32, 7.4), i.e. the 9.4 / 18.8 the design report quotes.
    """
    return (BAM_KT / BAM_R) * vin * BAM_ERROR_GAIN * BAM_ERROR_GAIN_RATIO * float(p_gain)


def servo_damping() -> float:
    """Joint damping [N*m*s/rad] = back-EMF ``kt^2/R`` plus mechanical viscous friction (~0.62).

    The back-EMF term is exactly the ``-kt^2 dq / R`` in BAM's torque law, so putting it in the joint
    instead of the actuator (``kv``) reproduces the same dynamics and keeps the actuator a pure
    position spring, which is what ports to GPU pipelines.
    """
    return BAM_KT * BAM_KT / BAM_R + BAM_FRICTION_VISCOUS


def servo_stall_torque(vin: float = NOMINAL_VIN) -> float:
    """Torque the BAM law reaches at full duty [N*m] -- 3.41 at 7.4 V.

    This is *not* the same as :attr:`ArmConfig.torque_limit_nm` (1.9, the datasheet stall figure).
    The two disagree, which is why the design report randomizes forcerange between them.
    """
    return BAM_KT * vin * BAM_MAX_DUTY / BAM_R


def duty_from_error(error_rad, p_gain: int) -> np.ndarray:
    """Firmware duty cycle for a position error, clipped at +-0.97.  This is ``Present_Load``."""
    duty = np.asarray(error_rad, dtype=float) * p_gain * BAM_ERROR_GAIN * BAM_ERROR_GAIN_RATIO
    return np.clip(duty, -BAM_MAX_DUTY, BAM_MAX_DUTY)


# ------------------------------------------------------------------------------- STL utilities ---


def read_binary_stl(path: str | Path) -> np.ndarray:
    """Return an ``(n_tri * 3, 3)`` float array of triangle vertices from a binary STL.

    Small enough to keep in-tree; pulling in trimesh for three bounding boxes is not worth a
    dependency.  ASCII STLs are rejected loudly rather than parsed.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    n_tri = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if len(raw) != 84 + 50 * n_tri:
        raise ValueError(
            f"{path} is not a binary STL (expected {84 + 50 * n_tri} bytes for {n_tri} triangles, "
            f"got {len(raw)})"
        )
    records = np.frombuffer(raw[84:], dtype="<u1").reshape(n_tri, 50)
    # bytes 0:12 are the facet normal, 12:48 are three float32 vertices, 48:50 the attribute count
    return records[:, 12:48].copy().view("<f4").reshape(n_tri * 3, 3).astype(np.float64)


def _quat_to_mat(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def _mesh_points_in_body(geom) -> np.ndarray:
    """STL vertices of a mesh geom, expressed in its body frame.

    MuJoCo re-centres and re-orients mesh assets at compile time but compensates in ``geom_pos`` and
    ``geom_quat``, so applying the *spec* pos/quat to the raw STL coordinates lands the points
    exactly where the compiled model puts them (verified against ``model.mesh_vert``).
    """
    verts = read_binary_stl(VENDOR_DIR / "assets" / f"{geom.meshname}.stl")
    return verts @ _quat_to_mat(geom.quat).T + np.array(geom.pos, dtype=float)


def _aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def _slab_boxes(points: np.ndarray, axis: int, lo: float, hi: float, n: int):
    """Cut ``points`` into ``n`` equal slabs along ``axis`` and fit an axis-aligned box to each.

    A staircase of boxes follows a tapering finger far better than one box and, unlike a convex
    hull, it cannot bridge the jaw gap.
    """
    boxes = []
    edges = np.linspace(lo, hi, n + 1)
    for k in range(n):
        sel = points[(points[:, axis] >= edges[k]) & (points[:, axis] <= edges[k + 1])]
        if len(sel) < 12:
            continue
        low, high = sel.min(axis=0), sel.max(axis=0)
        low[axis], high[axis] = edges[k], edges[k + 1]
        boxes.append((0.5 * (low + high), 0.5 * (high - low)))
    return boxes


# ------------------------------------------------------------------------------- jaw geometry ---


@dataclass(frozen=True)
class JawGeometry:
    """Fitted jaw numbers shared by the patcher, the TCP site and the grip-width helper.

    ``fixed_inner_x`` / ``moving_inner_x`` are the pad faces that touch the object, in the gripper
    body frame and the moving-jaw body frame respectively.
    """

    fixed_boxes: list
    fixed_palm: list
    moving_boxes: list
    moving_hub: list
    fixed_pad: tuple
    moving_pad: tuple
    fixed_inner_x: float
    moving_inner_point: np.ndarray
    jaw_pos: np.ndarray
    jaw_quat: np.ndarray


_JAW_CACHE: JawGeometry | None = None


def jaw_geometry() -> JawGeometry:
    """Fit the jaw collision boxes from the STL meshes.  Cached: the fit is deterministic."""
    global _JAW_CACHE
    if _JAW_CACHE is not None:
        return _JAW_CACHE

    spec = mujoco.MjSpec.from_file(str(VENDOR_XML))
    clouds = {}
    jaw_pos = jaw_quat = None
    for geom in spec.geoms:
        if geom.classname.name != "collision":
            continue
        if geom.meshname in (_FIXED_JAW_MESH, _MOVING_JAW_MESH):
            clouds[geom.meshname] = _mesh_points_in_body(geom)
    for body in spec.bodies:
        if body.name == MOVING_JAW_BODY:
            # np.asarray on a spec field is a *view* into MjSpec memory: copy, or it dangles
            # once this spec is collected.
            jaw_pos = np.array(body.pos, dtype=float)
            jaw_quat = np.array(body.quat, dtype=float)

    fixed = clouds[_FIXED_JAW_MESH]
    moving = clouds[_MOVING_JAW_MESH]

    # Fixed jaw: finger along -z of the gripper body, inner face at the *largest* x.
    fixed_finger = _slab_boxes(fixed, 2, fixed[:, 2].min(), _FIXED_KNUCKLE_Z, _FINGER_BOXES)
    fixed_palm = _slab_boxes(fixed, 2, _FIXED_KNUCKLE_Z, fixed[:, 2].max(), 1)
    # Moving jaw: finger along -y of the jaw body, inner face at the *smallest* x.
    moving_finger = _slab_boxes(moving, 1, moving[:, 1].min(), _MOVING_KNUCKLE_Y, _FINGER_BOXES)
    moving_hub = _slab_boxes(moving, 1, _MOVING_KNUCKLE_Y, moving[:, 1].max(), 1)

    fixed_pad, fixed_finger = _carve_pad(fixed_finger, axis=0, sign=+1)
    moving_pad, moving_finger = _carve_pad(moving_finger, axis=0, sign=-1)

    _JAW_CACHE = JawGeometry(
        fixed_boxes=fixed_finger,
        fixed_palm=fixed_palm,
        moving_boxes=moving_finger,
        moving_hub=moving_hub,
        fixed_pad=fixed_pad,
        moving_pad=moving_pad,
        fixed_inner_x=float(fixed_pad[0][0] + fixed_pad[1][0]),
        # The jaw swings on a hinge, so the two pad faces are never parallel: the end of the moving
        # pad nearest the hinge is always the narrowest point of the gap.  Measuring there makes
        # `gripper_angle_for_width` a guarantee about the whole pad rather than about one height.
        moving_inner_point=np.array(
            [
                moving_pad[0][0] - moving_pad[1][0],
                moving_pad[0][1] + moving_pad[1][1],
                moving_pad[0][2],
            ]
        ),
        jaw_pos=jaw_pos,
        jaw_quat=jaw_quat,
    )
    return _JAW_CACHE


def _carve_pad(finger_boxes, axis: int, sign: int):
    """Take a pad off the inner face of the fingertip box and shrink that box by the same amount.

    ``sign`` is the direction the inner face points along ``axis``.  Returns ``(pad, boxes)``.
    """
    boxes = [(c.copy(), h.copy()) for c, h in finger_boxes]
    centre, half = boxes[0]
    face = centre[axis] + sign * half[axis]
    pad_half = half.copy()
    pad_half[axis] = 0.5 * PAD_THICKNESS
    pad_centre = centre.copy()
    pad_centre[axis] = face - sign * 0.5 * PAD_THICKNESS
    half[axis] -= 0.5 * PAD_THICKNESS
    centre[axis] -= sign * 0.5 * PAD_THICKNESS
    return (pad_centre, pad_half), boxes


def gripper_angle_for_width(width: float) -> float:
    """Gripper joint angle [rad] at which the *narrowest* gap between the pads is ``width``.

    The moving pad is rigidly attached to the jaw body, which turns about the gripper joint axis, so
    this is a scalar root find on one rotation.  Clamped to the joint range: ask for more than the
    jaws open and you get the fully open angle.
    """
    geo = jaw_geometry()
    jaw_rot = _quat_to_mat(geo.jaw_quat)
    point = geo.moving_inner_point

    def gap(theta: float) -> float:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated = np.array(
            [
                cos_t * point[0] - sin_t * point[1],
                sin_t * point[0] + cos_t * point[1],
                point[2],
            ]
        )
        return float((jaw_rot @ rotated + geo.jaw_pos)[0] - geo.fixed_inner_x)

    lo, hi = -0.18, 1.74
    if gap(lo) > width:
        return lo
    if gap(hi) < width:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gap(mid) < width:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tcp_position() -> np.ndarray:
    """TCP in the gripper body frame: the centre of a nominal 32 mm brick held by the fingertips."""
    geo = jaw_geometry()
    pad_centre, pad_half = geo.fixed_pad
    tip_z = pad_centre[2] - pad_half[2]
    return np.array(
        [
            geo.fixed_inner_x + 0.5 * NOMINAL_GRIP_WIDTH,
            pad_centre[1],
            tip_z - GRIP_TIP_OVERHANG + 0.5 * NOMINAL_GRIP_WIDTH,
        ]
    )


# ------------------------------------------------------------------ wrist camera: mount geometry ---
#
# The real hardware is a 32x32 mm USB UVC camera module on TheRobotStudio's own
# "SO101 Wrist Cam Hex-Nut Mount" -- upstream's own README table (TheRobotStudio/SO-ARM100,
# section "Wrist-Mount Cameras") lists three 32x32-UVC wrist mounts and labels only this one
# "(SO101)"; the other two are "(SO100/101)" (a one-piece replacement for the wrist-roll part
# itself, not an add-on) and SO100-only. Vendored at `third_party/so101/wrist_camera_mount/` (STL,
# STEP, README, photos, `VENDORED.md`).
#
# This pose is **not** reverse-engineered from that bare STL by matching screw holes -- doing that
# blind is exactly how the previous ``WRIST_CAM`` guess (a hand-picked position aimed straight at
# the TCP) ended up filling 80-85% of the frame with the fixed jaw. Instead it is read directly off
# TheRobotStudio's **own** camera-equipped assembly for this exact hardware, vendored at the same
# commit as ``so101_new_calib.xml``: ``Simulation/SO101/so101_new_calib_camera.xml`` (and the
# equivalent ``.urdf``), which that repository's own comment says is "Derived from
# so101_new_calib.xml: adds wrist camera mount + wrist camera (Hex-Nut 32x32 UVC module form
# factor)". That file chains two fixed (jointless) bodies off ``gripper`` -- ``wrist_camera_mount``
# (pos ``(-0.0150, 0.0240, -0.0315)``, the printed bracket) then ``wrist_camera`` (pos
# ``(-0.04941, -0.03813, 0.0175)`` relative to the mount, the 32x32 board) -- both welded, so
# composing them into one direct ``gripper``-child camera pose (below) is exact, not an
# approximation, *for the CAD-nominal design*; see ``docs/guide/03-robot-and-servo-model.md`` for
# the full derivation, the uncertainty budget (mm / degrees) against the real, assembled, 3D-printed
# hardware, and a checkerboard calibration procedure to measure the real thing directly.

WRIST_CAM_POS_IN_GRIPPER = np.array([0.0025, 0.07344, 0.00659])
"""Optical centre of the wrist camera, in the gripper body frame [m] -- approximated as the 32x32
board's own CAD origin, which sits on the sensor plane rather than out at the lens's front element;
the lens's actual entrance pupil is a few mm further out along the view axis, folded into the
uncertainty budget (docs) rather than modelled as a separate offset."""

WRIST_CAM_QUAT_IN_GRIPPER = np.array([0.976158, -0.217060, 0.0, 0.0])
"""Orientation (w, x, y, z) of the wrist camera in the gripper body frame. A MuJoCo camera looks
down its own -z; this points that -z along the camera board mesh's own +z axis, which is the lens
barrel's protrusion direction -- confirmed two independent ways while deriving this (not assumed):
z-profiling ``wrist_camera_so101_v1.stl`` shows a 32x32 mm square cross-section near local z=0 (the
PCB) narrowing to a ~13 mm diameter cylinder reaching out to z=+26.3 mm (the lens), and a probe
camera placed at the board's own origin looking down its +z sees the inside of that same barrel --
exactly what a camera sees looking down its own lens tube. The image "up" direction (this camera's
own +y) is the board's local -y axis: fixed by the rigid mount, not independently chosen -- though
see the derivation notes for the one thing this cannot rule out (the purchased module's sensor die
soldered rotated 180 degrees on its board relative to the CAD would flip the image upside down; a
software/firmware fix, not a pose error)."""

WRIST_CAM_FORWARD_IN_GRIPPER = np.array([0.0, -0.42377, -0.90577])
"""The above quat's viewing axis, spelled out for readers who want the direction without decoding
a quaternion: mostly along -z of the gripper body (the fixed jaw's own pointing direction, i.e.
generally toward the TCP) tilted about 25 degrees toward -y. It passes about 11 degrees from the
TCP itself at the nominal grip -- close, but the real mount is bolted on with a fixed geometric
relationship to the jaw, not aimed at the grasp point the way the old guess was."""

# ------------------------------------------------------------------- wrist camera: lens field of view ---
#
# The specified hardware: a 32x32 mm USB UVC module with a lens in the "~120 degree" class,
# stated range 100-130 degrees measured on the sensor **diagonal** (the number lens vendors always
# quote, since it does not depend on the sensor's aspect ratio). A MuJoCo camera's pinhole model
# only takes a *vertical* ``fovy`` and derives the horizontal extent from whatever width/height the
# caller renders at (assuming square pixels) -- it has no notion of "diagonal" on its own, so a
# diagonal spec has to be converted for a specific resolution before it means anything to MuJoCo.
# ``fovy_from_diagonal_fov`` does that conversion for any resolution/aspect ratio; the constants
# below fix the *nominal* resolution this module compiles the camera's default ``fovy`` for.

WRIST_CAM_DIAGONAL_FOV_DEG = 115.0
"""Nominal diagonal fov [deg]: the midpoint of the stated 100-130 degree range, no more
precise than that range -- there is no datasheet for one specific board; "32x32 mm UVC module,
~120 degree lens" is a class of part, not one SKU."""

WRIST_CAM_DIAGONAL_FOV_RANGE_DEG = (100.0, 130.0)
"""The stated uncertainty range, in diagonal degrees. Use this as the domain-randomization
range for the lens: sample a diagonal fov in this range, then convert with
``fovy_from_diagonal_fov`` for whatever resolution is actually being rendered. Do not sample
``render.domain_rand.CameraJitterConfig.fovy_jitter_deg`` against this range directly -- that
jitter is added to the compiled *vertical* fovy in vertical degrees, a different angle than this
diagonal one."""

WRIST_CAM_NOMINAL_RESOLUTION = (640, 480)
"""Resolution ``WRIST_CAM_FOVY_DEG`` (the camera's compiled-in default) is computed for: 4:3, both
because the Optional-folder README recommends exactly this operating resolution for this class of
UVC module ("set your resolution to 640x480... most models work with lower resolutions") and
because it matches ``brick_builder/envs/align_env.py``'s own wrist-cam render size (160x120, also 4:3).
A caller rendering at a different aspect ratio -- for example a square 512x512 render -- must not
reuse ``WRIST_CAM_FOVY_DEG`` as-is; recompute it for that resolution with
``fovy_from_diagonal_fov``."""


def fovy_from_diagonal_fov(diagonal_deg: float, width: float, height: float) -> float:
    """MuJoCo pinhole vertical fovy [deg] whose corner-to-corner (diagonal) angle over a
    ``width`` x ``height`` render equals ``diagonal_deg``, assuming square pixels and no
    distortion (the pinhole model MuJoCo actually renders; real lens barrel distortion is a
    separate, image-space effect applied afterward -- see ``render.domain_rand.apply_postprocess``
    and its ``k1``/``k2``, and ``WRIST_CAM_DISTORTION_K1_RANGE`` below).

    For a pinhole, each half-angle's tangent is proportional to the corresponding sensor extent,
    so ``tan(diag/2)**2 == tan(fovx/2)**2 + tan(fovy/2)**2`` with
    ``tan(fovx/2) == aspect * tan(fovy/2)`` (``aspect = width / height``), giving
    ``tan(fovy/2) == tan(diag/2) / sqrt(1 + aspect**2)``. Works for any aspect ratio -- 4:3, 16:9,
    or a square render at the fixed look poses `brick_builder.perception.views` defines.
    """
    aspect = float(width) / float(height)
    half_diag = math.radians(diagonal_deg) / 2.0
    half_fovy = math.atan(math.tan(half_diag) / math.sqrt(1.0 + aspect * aspect))
    return math.degrees(2.0 * half_fovy)


def fovx_from_diagonal_fov(diagonal_deg: float, width: float, height: float) -> float:
    """Horizontal fovx [deg] companion to :func:`fovy_from_diagonal_fov`.

    MuJoCo derives this itself from ``fovy`` plus the render's own aspect ratio at render time;
    exposed here so calibration/measurement code (or documentation) can state it directly without
    re-deriving it.
    """
    aspect = float(width) / float(height)
    fovy = fovy_from_diagonal_fov(diagonal_deg, width, height)
    return math.degrees(2.0 * math.atan(aspect * math.tan(math.radians(fovy) / 2.0)))


WRIST_CAM_FOVY_DEG = fovy_from_diagonal_fov(WRIST_CAM_DIAGONAL_FOV_DEG, *WRIST_CAM_NOMINAL_RESOLUTION)
"""Compiled-model default vertical fovy [deg] -- ``WRIST_CAM_DIAGONAL_FOV_DEG`` at
``WRIST_CAM_NOMINAL_RESOLUTION``. Replaces the previous guess's ``41.4`` (an OV5647 camera-module
number that has nothing to do with this hardware)."""

# ---- barrel distortion: represented by the existing post-process, not re-implemented here ----
#
# ``render.domain_rand.PostProcessConfig.k1``/``k2`` (a 2nd/4th-order radial remap, already applied
# to every rendered frame downstream of any camera, not just this one) is the mechanism the rest of
# the codebase uses for lens distortion; this module does not duplicate it (and per the task's own
# scope, `render/domain_rand.py` is not edited here). A lens in the 100-130 degree diagonal class
# typically still shows substantial *barrel* (not pincushion) distortion even after any in-lens
# correction, i.e. predominantly negative ``k1`` in that remap's own convention (see
# ``apply_postprocess``/``_distortion_maps``: ``factor = 1 + k1*r**2 + k2*r**4 < 1`` at the edge
# pulls the source sample inward, which is what makes image content compress toward the edges --
# the barrel look). The ranges below are an engineering estimate for "a wide-angle board lens of
# this class", not a measured number -- no distortion chart exists for "32x32 mm UVC module,
# ~120 degree lens" as a class rather than one SKU; refit them once the calibration procedure
# (docs) has produced real checkerboard images. ``render/domain_rand.py``'s own
# ``DEFAULT_POSTPROCESS_RANGES`` (``k1`` in +-0.15, symmetric about zero) is not tailored to any
# one lens and is left as it is; a caller wanting this camera's distortion represented more
# accurately should pass these two ranges through ``domain_rand.sample_postprocess_config``'s own
# ``ranges`` argument (it fills in the default range for every key it is not given), e.g.
# ``sample_postprocess_config(rng, ranges={"k1": WRIST_CAM_DISTORTION_K1_RANGE, "k2":
# WRIST_CAM_DISTORTION_K2_RANGE})``.

WRIST_CAM_DISTORTION_K1_RANGE = (-0.35, -0.10)
WRIST_CAM_DISTORTION_K2_RANGE = (0.0, 0.08)


# ------------------------------------------------------------------------------- the patching ---


def _apply_options(spec: mujoco.MjSpec, cfg: ArmConfig) -> None:
    """Solver settings from design report 01 section 1.

    Elliptic cone with impratio 10 because the docs warn against a high impratio with a pyramidal
    cone, and a stiff friction cone is what keeps a gripped brick from creeping.  ``noslip`` stays at
    0 because MuJoCo-Warp raises on anything else, so a model that needs it would not port.
    """
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.noslip_iterations = 0


def _patch_actuators(spec: mujoco.MjSpec, cfg: ArmConfig) -> None:
    kp = servo_kp(cfg.p_gain, cfg.vin)
    limit = cfg.torque_limit_nm * cfg.vin / NOMINAL_VIN
    for act in spec.actuators:
        act.set_to_position(kp=kp, kv=0.0)
        act.forcerange = [-limit, limit]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


def _patch_joints(spec: mujoco.MjSpec, cfg: ArmConfig) -> None:
    for joint in spec.joints:
        if joint.name not in ARM_JOINTS:
            continue
        joint.damping = [servo_damping(), 0.0, 0.0]  # hinge uses element 0
        joint.frictionloss = cfg.joint_friction_nm
        joint.armature = BAM_ARMATURE


_PRINTED_MATERIAL_EXCLUDE = frozenset({
    "sts3215_03a_v1_material",
    "sts3215_03a_no_horn_v1_material",
})
"""Vendored materials that are the STS3215 servo's own black plastic casing, not a part this team
prints -- left out of :func:`_patch_print_color`."""


def _patch_print_color(spec: mujoco.MjSpec, cfg: ArmConfig) -> None:
    """Recolour every printed-part material to ``cfg.print_color_rgba``.

    At the default config this reproduces the upstream materials' own rgba exactly, so a caller
    that never sets ``print_color_rgba`` sees no change at all.
    """
    for material in spec.materials:
        if material.name in _PRINTED_MATERIAL_EXCLUDE:
            continue
        material.rgba = list(cfg.print_color_rgba)


def _add_backlash(spec: mujoco.MjSpec, cfg: ArmConfig) -> None:
    """Give every actuated joint a free-play partner on the same axis.

    Two hinges about the same axis through the same point compose to one angle, so this is exactly
    upstream's dummy-body pattern with one fewer body.  The actuator drives the first hinge (the
    motor side) and the link hangs off the second, so motor torque reaches the link only after the
    slack has been taken up.

    The encoder does *not* sit there.  The STS3215's 12-bit magnetic encoder is mounted after the
    1:345 gearbox, directly on the output shaft -- the only arrangement in which one absolute
    0-4095 reading can cover one output turn, and what independent bench measurements of this servo
    state outright.  ``ServoBus`` therefore reports, and closes the firmware's proportional law on,
    the driven joint *plus* this partner.  What that costs the robot is dead time and freedom
    inside the firmware's dead band, not a placement error it cannot see.

    ``cfg.backlash_deg`` is the **half-range**: total lost motion is twice it.  The bench figure of
    0.87 deg is a total -- a dial gauge on an 86 mm lever read 1.3 mm end to end -- so the default
    half-range is 0.435 deg.  The datasheet's own "<= 0.5 deg" is likewise a total, i.e. a
    0.25 deg half-range that the measured part does not meet.

    The actuator's control range is widened here because ``ServoBus`` writes ``target - play`` into
    ``data.ctrl``: without the extra room MuJoCo would clamp that offset away within a degree of
    each joint's stop and quietly weaken the servo there.  The widening is by
    :data:`BACKLASH_CTRL_MARGIN_DEG`, not by this config's own play, so a run that randomizes the
    play upwards between episodes still fits.  Goal positions stay clipped to the joint's own
    range by the bus, so nothing can be commanded past the stop.
    """
    if cfg.backlash_deg <= 0.0:
        return
    play = math.radians(cfg.backlash_deg)
    margin = math.radians(BACKLASH_CTRL_MARGIN_DEG)
    for name in ARM_JOINTS:
        driven = spec.joint(name)
        body = driven.parent
        extra = body.add_joint(
            name=f"{name}_backlash",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=np.array(driven.axis, dtype=float),
            pos=np.array(driven.pos, dtype=float),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[-play, play],
        )
        extra.damping = [0.01, 0.0, 0.0]
        extra.armature = 0.01
        extra.frictionloss = 0.0
    for act in spec.actuators:
        if act.name in ARM_JOINTS:
            lo, hi = float(act.ctrlrange[0]), float(act.ctrlrange[1])
            act.ctrlrange = [lo - margin, hi + margin]


def _replace_collision(spec: mujoco.MjSpec, cfg: ArmConfig) -> None:
    """Swap every collision mesh for convex primitives.

    Arm links get one box per part, in that part's own mesh frame, which is far tighter than a
    capsule on these boxy printed parts.  The jaws get the staircase fit plus a fingertip pad.
    """
    geo = jaw_geometry()
    per_body: dict[str, list] = {}
    doomed = []
    for geom in spec.geoms:
        if geom.classname.name != "collision":
            continue
        doomed.append(geom)
        per_body.setdefault(geom.parent.name, []).append(
            (geom.meshname, np.array(geom.pos, float), np.array(geom.quat, float))
        )
    for geom in doomed:
        spec.delete(geom)

    if not cfg.primitive_collision:
        # Put the meshes back, unchanged, so a test can show that mesh collision cannot grasp.
        for body_name, parts in per_body.items():
            body = spec.body(body_name)
            for k, (mesh, pos, quat) in enumerate(parts):
                body.add_geom(
                    name=f"{body_name}_mesh{k}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh,
                    pos=pos,
                    quat=quat,
                    group=3,
                    condim=3,
                    rgba=[0.4, 0.5, 0.9, 0.3],
                )
        return

    for body_name, parts in per_body.items():
        if body_name in (GRIPPER_BODY, MOVING_JAW_BODY):
            continue
        body = spec.body(body_name)
        for k, (mesh, pos, quat) in enumerate(parts):
            verts = read_binary_stl(VENDOR_DIR / "assets" / f"{mesh}.stl")
            centre, half = _aabb(verts)
            body.add_geom(
                name=f"{body_name}_col{k}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos + _quat_to_mat(quat) @ centre,
                quat=quat,
                size=half,
                group=3,
                condim=3,
                rgba=[0.4, 0.5, 0.9, 0.3],
            )

    gripper = spec.body(GRIPPER_BODY)
    # The gripper body also carries a servo; keep it as a box in its own mesh frame.
    for k, (mesh, pos, quat) in enumerate(p for p in per_body[GRIPPER_BODY]
                                          if p[0] != _FIXED_JAW_MESH):
        verts = read_binary_stl(VENDOR_DIR / "assets" / f"{mesh}.stl")
        centre, half = _aabb(verts)
        gripper.add_geom(
            name=f"gripper_col{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos + _quat_to_mat(quat) @ centre,
            quat=quat,
            size=half,
            group=3,
            condim=3,
            rgba=[0.4, 0.5, 0.9, 0.3],
        )
    _add_boxes(gripper, "fixed_jaw_palm", geo.fixed_palm)
    _add_boxes(gripper, "fixed_jaw", geo.fixed_boxes)
    _add_pad(gripper, FIXED_PAD_GEOM, geo.fixed_pad, cfg)

    jaw = spec.body(MOVING_JAW_BODY)
    _add_boxes(jaw, "moving_jaw_hub", geo.moving_hub)
    _add_boxes(jaw, "moving_jaw", geo.moving_boxes)
    _add_pad(jaw, MOVING_PAD_GEOM, geo.moving_pad, cfg)


def _add_boxes(body, prefix: str, boxes) -> None:
    for k, (centre, half) in enumerate(boxes):
        body.add_geom(
            name=f"{prefix}{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=centre,
            size=half,
            group=3,
            condim=3,
            rgba=[0.4, 0.5, 0.9, 0.3],
        )


def _add_pad(body, name: str, pad, cfg: ArmConfig) -> None:
    """Fingertip pad: condim 4 with priority so its friction wins over whatever it touches."""
    centre, half = pad
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=centre,
        size=half,
        group=3,
        condim=4,
        priority=1,
        friction=[cfg.pad_friction, 0.005, 0.0001],
        solref=[0.004, 1.0],
        solimp=[0.95, 0.999, 0.0005, 0.5, 2.0],
        rgba=[0.9, 0.3, 0.2, 0.6],
    )


def _add_sites_and_camera(spec: mujoco.MjSpec) -> None:
    gripper = spec.body(GRIPPER_BODY)
    tcp = tcp_position()
    gripper.add_site(
        name=TCP_SITE,
        pos=tcp,
        # 180 deg about x: the site's z then points along the gripper body's -z, i.e. outwards past
        # the fingertips, which is the approach direction.
        quat=[0.0, 1.0, 0.0, 0.0],
        size=[0.002, 0.002, 0.002],
        group=4,
        rgba=[0.1, 0.9, 0.2, 1.0],
    )

    gripper.add_camera(
        name=WRIST_CAM,
        pos=WRIST_CAM_POS_IN_GRIPPER,
        quat=WRIST_CAM_QUAT_IN_GRIPPER,
        fovy=WRIST_CAM_FOVY_DEG,
    )


def build_arm_spec(cfg: ArmConfig = ArmConfig()) -> mujoco.MjSpec:
    """Load the vendored MJCF and return a patched, uncompiled :class:`mujoco.MjSpec`.

    Patches: solver options, BAM position actuators, BAM joint damping/friction/armature, printed-
    part colour, optional backlash joints, primitive collision geoms with fingertip pads, the TCP
    site and the wrist camera.  The vendored file on disk is untouched.
    """
    spec = mujoco.MjSpec.from_file(str(VENDOR_XML))
    _apply_options(spec, cfg)
    _patch_actuators(spec, cfg)
    _patch_joints(spec, cfg)
    _patch_print_color(spec, cfg)
    _replace_collision(spec, cfg)
    _add_sites_and_camera(spec)
    _add_backlash(spec, cfg)
    return spec


def attach_arm(
    world: mujoco.MjSpec,
    cfg: ArmConfig = ArmConfig(),
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
) -> None:
    """Attach a patched arm to ``world`` at ``pos``/``quat``.

    Solver options live on the spec being compiled, not on the attached child, so the arm's options
    are copied onto ``world`` as well.  Names are not prefixed: one arm per world.
    """
    arm = build_arm_spec(cfg)
    # Options are set on the world first: MuJoCo keeps the parent's values when attaching and warns
    # about every field where the two disagree.
    _apply_options(world, cfg)
    frame = world.worldbody.add_frame(
        pos=np.asarray(pos, dtype=float), quat=np.asarray(quat, dtype=float)
    )
    # An empty prefix keeps the plain names (`shoulder_pan`, `tcp_pinch`, ...); the default would
    # namespace everything behind "/". One arm per world, so there is nothing to disambiguate.
    world.attach(arm, frame=frame, prefix="")
