"""Assemble one complete world: table, arm, baseplate, placed bricks, a brick in the gripper.

Everything the project simulates goes through here.  A :class:`SceneConfig` says what the world
contains; :func:`build_scene` turns it into a compiled :class:`Scene` with the snap model bound and
the arm already standing at its home pose; :func:`reset_scene` puts an existing scene back to that
state.

Call order matters and is not negotiable:

1. ``attach_arm`` and the brick generators write bodies into a `mujoco.MjSpec`;
2. the snap model's ``declare()`` adds one collet body plus a pile of inactive equalities, which
   must exist *before* compilation;
3. ``world.compile()``;
4. the snap model's ``bind()`` resolves ids against the compiled model.

Doing 2 after 3 silently produces a scene where nothing ever connects, so `build_scene` is the one
place that order is written down.

Frames
------
World x is forward, y left, z up; the arm's base sits at the origin and the table top is z = 0.
The build plate's centre defaults to (0.17, 0.0), which is where the force map (WP-KIN) says the
arm has 12-16 N of downward push at 7.4 V.

The grasp
---------
Both grasp modes hold the brick in the same place, worked out by construction in
:func:`grasp_pose` from the fitted jaw geometry and the brick's own size - never by hand-tuned
numbers.  The brick is gripped across its ``nx``-stud width (31.8 mm for the usual 2-wide brick)
with the tool pointing straight down, so the brick's own +z (studs up) is the gripper body's +z
and the brick's width axis lies along the gripper's jaw-opening axis.

- ``grasp_mode="weld"`` holds it with a *soft* weld equality between the gripper body and the
  brick: compliant in both position and angle, parameters randomizable.  The jaws are commanded
  half a millimetre wider than the brick so the jaw faces never touch it, which is the point - it
  lets Deliverable 1 study the press without also tuning finger contact.
- ``grasp_mode="friction"`` closes the jaws onto the brick with ``grip_squeeze`` metres of
  over-closure and lets friction between the bare printed jaw faces and the brick do the work.
  This is the only mode the real arm can do, because the real arm is a stock SO-101: a fixed jaw,
  a moving jaw, nothing attached to either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from brick_builder.bricks.generator import COLLISION_GROUP, VISUAL_GROUP, BrickHandle, add_baseplate, add_brick
from brick_builder.bricks.snap import ColletSnap
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.snap_detent import DetentSnap
from brick_builder.bricks.spec import DUPLO, BrickSpec
from brick_builder.bricks.structures import Placement, build_structure, validate_structure
from brick_builder.robot.kinematics import ArmKinematics, euler_zyx
from brick_builder.robot.patch_so101 import (
    ARM_JOINTS,
    GRIP_TIP_OVERHANG,
    GRIPPER_BODY,
    TCP_SITE,
    WRIST_CAM,
    ArmConfig,
    attach_arm,
    gripper_angle_for_width,
    jaw_geometry,
    tcp_position,
)

PLATE_BODY = "plate"
"""Body name of the build plate."""

HELD_BODY = "held"
"""Body name of the brick that starts in the gripper."""

OVERVIEW_CAM = "overview_cam"
GRASP_CAM = "grasp_cam"
TABLE_GEOM = "table"
SUPPLY_PREFIX = "supply_brick_"
"""Body-name prefix for a loose brick lying on the table, waiting to be picked up."""
WELD_EQ = "grasp_weld"

WORLD_CAMERAS = (OVERVIEW_CAM, GRASP_CAM)
"""Cameras this module can add.  ``wrist_cam`` comes from the arm patch, not from here."""

_TOOL_DOWN_PITCH = 0.0
_TOOL_DOWN_ROLL = math.pi
"""A tool pointing straight down, in the (pitch, roll) convention `ArmKinematics.ik` uses."""


# --------------------------------------------------------------------------------- the grasp ---


@dataclass(frozen=True)
class GraspPose:
    """Where a brick sits when the gripper is holding it.

    Every number here is derived, not tuned: ``jaw_geometry()`` gives the fitted jaw faces,
    ``gripper_angle_for_width`` inverts the jaw hinge, and the brick's own ``BrickSpec`` gives its
    width and height.  ``pos_in_tcp``/``quat_in_tcp`` is the transform the spec asks the scene to
    carry around - the brick's body origin (centre of its bottom face) expressed in the TCP site's
    frame - and it is what an environment uses to turn a commanded TCP pose into the brick pose it
    expects.
    """

    width: float                 # jaw opening the brick needs [m]
    gripper_angle: float         # gripper joint angle at which the pads just touch it [rad]
    contact_point: np.ndarray    # gripper-frame point where the moving pad meets the brick
    pos_in_gripper: np.ndarray   # brick origin in the gripper body frame
    pos_in_tcp: np.ndarray       # brick origin in the TCP site frame
    quat_in_tcp: np.ndarray      # brick orientation in the TCP site frame (w, x, y, z)
    pad_overlap: float           # height of brick wall actually between the jaw faces [m]
    tip_clearance: float         # how far the brick's bottom face hangs past the jaw tips [m]


# The TCP site is the gripper body rotated 180 degrees about x (see `_add_sites_and_camera` in
# patch_so101), so this matrix converts a vector between the two frames -- it is its own inverse.
_TCP_TO_GRIPPER = np.diag([1.0, -1.0, -1.0])

_BRICK_QUAT_IN_TCP = np.array([0.0, 1.0, 0.0, 0.0])
"""Brick orientation in the TCP frame: the same 180-degree flip about x.

With the tool pointing down the TCP's z axis points down (it is the approach direction), so the
brick's z - the direction its studs point - has to be the TCP's -z for the brick to be held the
right way up.
"""


_JOINT_LO, _JOINT_HI = -0.18, 1.74
"""The gripper joint's range, as `gripper_angle_for_width` uses it."""

TRAY_BRICK_SPACING = 4.5
SUPPLY_SPACING = 4.5
"""Stud pitches between loose supply bricks. The open jaws are about 37.6 mm across the tips and
a 2-wide brick is 31.8 mm, so 4.5 pitches (72 mm) between brick centres leaves the tool roughly
20 mm of clear air either side of the brick it is descending onto."""
"""Centre-to-centre spacing of the tray's bricks, in stud pitches (72 mm).

Measured from the fitted jaw boxes rather than picked to look tidy.  Around a 31.8 mm brick the
open jaws reach about 16 mm beyond the near face and 18 mm beyond the far one, so the tool is
roughly 65 mm wide at the height the bricks stand at.  The old 2.4-pitch (38.4 mm) spacing left
6.6 mm between neighbouring bricks, and the arm could not get the jaws down around one without
knocking the next one over.
"""


def _moving_face(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """A point on the moving pad's working face and its outward normal, in the gripper frame.

    The pad is a box on the moving jaw; its working face is the one at the smallest x in the jaw's
    own frame, so the normal that points across the gap towards the fixed pad is the jaw's -x.
    Both the point and the normal are carried through the joint rotation (about the jaw body's z
    axis) and then through the jaw body's own 90-degree mounting rotation.
    """

    geo = jaw_geometry()
    centre, half = geo.moving_pad
    face = np.array([centre[0] - half[0], centre[1], centre[2]], dtype=float)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    rot_z = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    jaw_rot = np.zeros(9)
    mujoco.mju_quat2Mat(jaw_rot, np.asarray(geo.jaw_quat, dtype=float))
    jaw_rot = jaw_rot.reshape(3, 3)
    point = jaw_rot @ (rot_z @ face) + np.asarray(geo.jaw_pos, dtype=float)
    normal = jaw_rot @ (rot_z @ np.array([-1.0, 0.0, 0.0]))
    return point, normal


def jaw_clearance(point, angle: float) -> float:
    """How far ``point`` (gripper body frame) is clear of the moving pad's face [m].

    Positive means the point is inside the open gap; negative means the pad would be through it.

    The fixed pad's working face is a plane at a constant x, so the interesting geometry is all in
    the moving jaw, which swings on a hinge: its face is a plane tilted against the fixed one, and
    how much room there is depends on *where* you measure.  `gripper_angle_for_width` in
    `patch_so101` deliberately answers only for the narrowest point of the gap, the end of the pad
    nearest the hinge, because that makes it a guarantee for the whole pad.  A grasp needs the
    other question - how much room is there where the object actually is - and gets a very
    different answer: at the angle that leaves 31.8 mm at the narrow end, the fingertips are 39.9
    mm apart.  Closing to 31.8 mm "as measured by `gripper_angle_for_width`" therefore leaves a
    brick held down at the tips completely loose, which is what made the first version of this
    scene drop every brick it tried to hold by friction.
    """

    face_point, normal = _moving_face(angle)
    return float(normal @ (np.asarray(point, dtype=float) - face_point))


def jaw_angle_for_clearance(point, clearance: float) -> float:
    """Joint angle at which :func:`jaw_clearance` at ``point`` equals ``clearance``.

    Clearance grows with the angle (opening the jaws can only make more room), so a bisection over
    the joint range is exact.  Out-of-reach requests are clamped to the joint's own limits, the
    same way `gripper_angle_for_width` clamps.
    """

    lo, hi = _JOINT_LO, _JOINT_HI
    if jaw_clearance(point, lo) > clearance:
        return lo
    if jaw_clearance(point, hi) < clearance:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if jaw_clearance(point, mid) < clearance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grasp_pose(
    nx: int = 2,
    ny: int = 4,
    spec: BrickSpec = DUPLO,
    overhang: float = GRIP_TIP_OVERHANG,
) -> GraspPose:
    """Work out where a brick sits in the stock jaws, from the jaw fit and the brick's size.

    The construction, step by step:

    - the brick is gripped across its ``nx``-stud footprint, so the jaws have to span
      ``nx * pitch - clearance`` (31.8 mm for a 2-wide DUPLO brick);
    - the fixed jaw's working face is a plane at ``jaw_geometry().fixed_inner_x``, and the brick's
      far face is one brick-width out from it, so the brick's centre plane is half a width out;
    - the brick hangs past the jaw tips by ``overhang`` (``GRIP_TIP_OVERHANG``, measured in
      WP-ROBOT).  It has to: the jaw tips reach ``overhang`` below the brick's bottom face
      otherwise, and they would hit the baseplate long before the brick seated on it;
    - that fixes the brick's height in the hand, so the contact with the *moving* jaw is at the
      brick's top outer edge - the highest point of the brick, where the tapering gap is narrowest.
      :func:`jaw_angle_for_clearance` gives the angle at which the jaws just touch it there.

    ``ny`` is accepted so that callers can name the whole brick, but nothing in the grasp depends
    on it: the jaws close across the width, and the length only decides how much of the brick
    sticks out past them.
    """

    geo = jaw_geometry()
    width = nx * spec.pitch - spec.clearance

    pad_centre, pad_half = geo.fixed_pad
    tip_z = float(pad_centre[2] - pad_half[2])       # lowest point of the fixed jaw face
    pad_top_z = float(pad_centre[2] + pad_half[2])
    bottom_z = tip_z - overhang                       # brick's bottom face
    roof_z = bottom_z + spec.height                   # brick's top outer edge: the contact line
    contact_point = np.array([geo.fixed_inner_x + width, float(pad_centre[1]), roof_z])
    angle = jaw_angle_for_clearance(contact_point, 0.0)

    pos_in_gripper = np.array(
        [geo.fixed_inner_x + 0.5 * width, float(pad_centre[1]), bottom_z], dtype=float
    )
    pos_in_tcp = _TCP_TO_GRIPPER @ (pos_in_gripper - tcp_position())

    overlap = min(pad_top_z, roof_z) - max(tip_z, bottom_z)

    return GraspPose(
        width=float(width),
        gripper_angle=float(angle),
        contact_point=contact_point,
        pos_in_gripper=pos_in_gripper,
        pos_in_tcp=pos_in_tcp,
        quat_in_tcp=_BRICK_QUAT_IN_TCP.copy(),
        pad_overlap=float(overlap),
        tip_clearance=float(overhang),
    )


# -------------------------------------------------------------------------------- the config ---


@dataclass(frozen=True)
class HeldBrick:
    """The brick that starts in the gripper.

    ``offset`` and ``yaw`` are the *in-hand* error: where the brick really is relative to where the
    nominal grasp of :func:`grasp_pose` would put it, expressed in the TCP frame (x is the
    jaw-opening direction, y runs along the brick, z is the approach direction).  A press policy
    never sees this directly; it is privileged information for the critic and a thing the scripted
    primitive has to survive.

    ``offset_jitter``/``yaw_jitter`` let :func:`reset_scene` draw a fresh in-hand error from its
    ``rng`` on every reset.  Both are zero by default, which makes a reset a pure function of the
    scene and keeps the default scene bit-reproducible without needing a seed at all.
    """

    nx: int = 2
    ny: int = 4
    colour: str = "red"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    offset_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_jitter: float = 0.0
    grasp_mode: Literal["weld", "friction"] | None = None


@dataclass
class SceneConfig:
    """Everything that decides what a world contains.

    Defaults give the scene the rest of Deliverable 1 is written against: the arm, an 8x8 plate at
    (0.17, 0.0), nothing on the plate yet, and a 2x4 brick soft-welded into the gripper a few
    millimetres above the plate.
    """

    arm: ArmConfig = ArmConfig()
    plate_studs: tuple[int, int] = (8, 8)
    plate_xy: tuple[float, float] = (0.17, 0.0)
    plate_yaw: float = 0.0
    placed: list[Placement] = field(default_factory=list)
    held: HeldBrick | None = field(default_factory=HeldBrick)
    grasp_mode: Literal["weld", "friction"] = "weld"
    snap: SnapParams = SnapParams()
    snap_model: Literal["collet", "detent"] = "collet"
    tray: bool = False
    cameras: tuple[str, ...] = (WRIST_CAM, OVERVIEW_CAM)
    spec: BrickSpec = DUPLO

    # -- things the spec sketch left open, each needed to build a concrete world ----------------
    table: bool = True
    """Add the table plane at z = 0.  Off only for bench-like scenes that want free space."""

    home_tcp: tuple[float, float, float] | None = None
    """Where the TCP stands after a reset; ``None`` = centred over the plate, ``home_hover``
    above it (measured from the brick's bottom face when a brick is held)."""

    home_hover: float = 0.02
    home_yaw: float = 0.0
    """Tool spin at the home pose.  0 lines the held brick's grid up with the plate's."""

    grip_squeeze: float = 0.006
    """How far past the brick's face the jaws are commanded in ``friction`` mode [m]."""

    weld_clearance: float = 0.0005
    """How far *short* of touching the jaws are commanded in ``weld`` mode [m]."""

    weld_solref: tuple[float, float] = (0.008, 1.0)
    weld_solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)
    weld_torquescale: float = 1.0
    """Soft-weld compliance.  ``solref`` is (time constant, damping ratio): 20 ms is about ten
    physics steps, soft enough that the brick can be pushed out of line by a millimetre under a
    few newtons instead of being bolted to the jaw.  ``torquescale`` weights the angular rows
    against the positional ones; small means the grip yields in angle before it yields in
    position, which is what a real pinch grip does."""

    brick_friction: float = 0.35
    """Sliding friction between bricks and between a brick and the plate.

    `bricks/generator.py` leaves its geoms on MuJoCo's default of 1.0, which is a placeholder, not
    a measurement; design report 01 section 4 puts ABS on ABS at 0.35 (0.2-0.5 randomized).  The
    scene is where a world's materials get decided, so it is set here rather than by editing the
    generator, whose own bench tests are calibrated against what it currently produces."""

    tray_xy: tuple[float, float] = (0.13, -0.17)
    tray_brick: tuple[int, int] = (2, 4)
    """Footprint of the spare bricks in the tray, in studs."""

    supply: tuple[tuple[int, int, str], ...] = ()
    """Loose bricks lying on the table beside the plate, each `(nx, ny, colour)`.

    No tray, no pockets, no walls: the deliverable's hardware is a bare arm on a bare table, and
    a brick to be picked up simply lies there. They are laid out in a row along y at
    :attr:`supply_xy`, :data:`SUPPLY_SPACING` stud pitches apart, which is wide enough for the
    open jaws to come down around one without touching its neighbour.
    """

    supply_xy: tuple[float, float] = (0.13, -0.17)
    """Where that row is centred [m] -- the same reachable patch the tray used to occupy."""

    tray_bricks: int = 2
    """How many spare bricks stand in the tray.  Two rather than three: at the spacing the stock
    jaws need (:data:`TRAY_BRICK_SPACING`) a third one would sit 72 mm further out and put the
    arm near the edge of its useful workspace."""

    light_pos: tuple[float, float, float] = (0.25, -0.15, 0.7)
    light_diffuse: tuple[float, float, float] = (0.85, 0.85, 0.85)
    offscreen: tuple[int, int] = (1280, 960)


# --------------------------------------------------------------------------------- the scene ---


@dataclass
class Scene:
    """A compiled world plus everything needed to drive and reset it."""

    model: mujoco.MjModel
    data: mujoco.MjData
    cfg: SceneConfig
    snap: ColletSnap | DetentSnap
    bricks: dict[str, BrickHandle]
    plate: BrickHandle
    kin: ArmKinematics
    arm_qpos_adr: np.ndarray            # int (6,) over ARM_JOINTS
    arm_dof_adr: np.ndarray             # int (6,)
    instance_ids: dict[str, int]        # body name -> instance id, for SceneRenderer.instance_lut
    grasp: GraspPose | None
    held_name: str | None
    home_q: np.ndarray                  # (5,) arm joints at reset, gripper excluded
    weld_eq: int                        # -1 when the scene has no soft weld
    in_hand_offset: np.ndarray          # (4,) dx, dy, dz, dyaw actually used by the last reset
    placed_names: list[str] = field(default_factory=list)

    # -- written by `randomize.apply`, read by `reset_scene` -------------------------------
    randomization: dict = field(default_factory=dict)
    """The last sample `randomize.apply` wrote, verbatim.  Goes into `EpisodeResult.params`."""

    in_hand_override: np.ndarray | None = None
    """In-hand error chosen by randomization; overrides the held brick's own offset/jitter."""

    weld_torquescale: float = 1.0
    """Angular stiffness of the soft weld, re-written into `eq_data` on every reset."""

    nominal: dict = field(default_factory=dict)
    """Copies of the model arrays randomization overwrites, taken once at build time.

    Randomization writes absolute values rather than multiplying what is already there, so
    applying two samples in a row cannot drift; these are the values every sample is measured
    from."""

    # -- convenience --------------------------------------------------------

    @property
    def spec(self) -> BrickSpec:
        return self.cfg.spec

    @property
    def plate_top(self) -> float:
        """World z of the plate's flat top surface, where a seated brick's bottom face lands."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.plate.name)
        return float(self.model.body_pos[bid][2] + 2.0 * self.cfg.spec.roof)

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current TCP site position and rotation matrix, from `data` as it stands."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        return self.data.site_xpos[sid].copy(), self.data.site_xmat[sid].reshape(3, 3).copy()

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def free_joint_adr(self, body: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_free")
        if jid < 0:
            raise KeyError(f"{body!r} has no free joint")
        return int(self.model.jnt_qposadr[jid])

    def warnings(self) -> dict[str, int]:
        """Non-zero MuJoCo warning counters, by name.  Empty means a clean run."""
        out = {}
        for i in range(mujoco.mjtWarning.mjNWARNING):
            n = int(self.data.warning[i].number)
            if n:
                out[mujoco.mjtWarning(i).name] = n
        return out

    def step(self, n: int = 1) -> None:
        """``n`` physics steps with the snap bookkeeping run at its configured rate."""
        every = max(int(self.snap.params.snap_every), 1)
        for k in range(n):
            if k % every == 0:
                self.snap.update(self.model, self.data)
            mujoco.mj_step(self.model, self.data)


# ---------------------------------------------------------------------------------- building ---


def _look_at(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion for a MuJoCo camera at ``pos`` looking at ``target`` (cameras look down -z)."""
    pos = np.asarray(pos, dtype=float)
    z = pos - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


def _add_world_furniture(world: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Table, lights and the render settings every camera in this project depends on."""

    world.visual.global_.offwidth = int(cfg.offscreen[0])
    world.visual.global_.offheight = int(cfg.offscreen[1])
    # The wrist camera sits ~100 mm from the brick, so the near plane has to be well inside that.
    world.stat.extent = 0.6
    world.visual.map.znear = 0.01

    world.worldbody.add_light(
        name="key", pos=list(cfg.light_pos), dir=[-0.35, 0.2, -1.0],
        diffuse=list(cfg.light_diffuse),
    )
    world.worldbody.add_light(
        name="fill", pos=[-0.2, 0.25, 0.5], dir=[0.4, -0.4, -1.0],
        diffuse=[0.35, 0.35, 0.35],
    )
    if cfg.table:
        world.worldbody.add_geom(
            name=TABLE_GEOM,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1.0, 1.0, 0.01],
            pos=[0.0, 0.0, 0.0],
            group=VISUAL_GROUP,
            contype=1,
            conaffinity=1,
            condim=3,
            friction=[0.6, 0.005, 0.0001],
            rgba=[0.55, 0.55, 0.58, 1.0],
        )


def _add_cameras(world: mujoco.MjSpec, cfg: SceneConfig, grasp: GraspPose | None) -> None:
    """Add every camera in ``cfg.cameras`` that this module owns."""

    unknown = [c for c in cfg.cameras if c != WRIST_CAM and c not in WORLD_CAMERAS]
    if unknown:
        raise ValueError(f"unknown camera(s) {unknown}; this scene can build {WORLD_CAMERAS}")

    plate_x, plate_y = cfg.plate_xy
    if OVERVIEW_CAM in cfg.cameras:
        pos = (plate_x + 0.30, plate_y - 0.34, 0.33)
        world.worldbody.add_camera(
            name=OVERVIEW_CAM,
            pos=list(pos),
            quat=list(_look_at(pos, (plate_x - 0.03, plate_y, 0.05))),
            fovy=45.0,
        )
    if GRASP_CAM in cfg.cameras:
        # On the gripper body, looking along the held brick's long axis: one frame shows both jaw
        # faces, the brick's gripped width and how far it hangs past the tips.
        look_at = grasp.pos_in_gripper if grasp is not None else tcp_position()
        centre = np.array([look_at[0], look_at[1], look_at[2] + 0.010])
        pos = centre + np.array([0.0, -0.085, 0.012])
        world.body(GRIPPER_BODY).add_camera(
            name=GRASP_CAM, pos=list(pos), quat=list(_look_at(pos, centre, up=(0.0, 0.0, 1.0))),
            fovy=38.0,
        )


def _add_supply(world: mujoco.MjSpec, cfg: SceneConfig) -> list[BrickHandle]:
    """Loose bricks lying on the table, in a row. Nothing else: no tray, no jig, no fixture.

    They sit one thousandth of a millimetre above the table so the first physics step settles
    them onto it rather than starting with an interpenetration.
    """

    spec = cfg.spec
    sx, sy = cfg.supply_xy
    out = []
    for k, (nx, ny, colour) in enumerate(cfg.supply):
        y = sy + (k - (len(cfg.supply) - 1) / 2.0) * SUPPLY_SPACING * spec.pitch
        out.append(add_brick(
            world, name=f"{SUPPLY_PREFIX}{k}", nx=int(nx), ny=int(ny), colour=str(colour),
            pos=(sx, y, 0.0005), spec=spec,
        ))
    return out


def _add_tray(world: mujoco.MjSpec, cfg: SceneConfig) -> list[BrickHandle]:
    """A shallow open box beside the plate with a few loose bricks in it.

    Deliberately plain: it is a place for spare bricks to sit within reach and in the overview
    camera's view, not a fixture with pockets.  Pockets belong to the hardware fall-back plan
    (PLAN's risk register), and adding them now would bake a jig we have not built into every
    training scene.
    """

    spec = cfg.spec
    inner_x = 0.5 * (cfg.tray_bricks * TRAY_BRICK_SPACING * spec.pitch)
    inner_y = 0.5 * (4 * spec.pitch + 0.01)
    wall, height = 0.004, 0.012
    tx, ty = cfg.tray_xy
    tray = world.worldbody.add_body(name="tray", pos=[tx, ty, 0.0])
    floor = 0.002
    panels = [
        ("floor", (0.0, 0.0, 0.5 * floor), (inner_x + wall, inner_y + wall, 0.5 * floor)),
        ("xp", (inner_x + 0.5 * wall, 0.0, 0.5 * height), (0.5 * wall, inner_y + wall, 0.5 * height)),
        ("xm", (-inner_x - 0.5 * wall, 0.0, 0.5 * height), (0.5 * wall, inner_y + wall, 0.5 * height)),
        ("yp", (0.0, inner_y + 0.5 * wall, 0.5 * height), (inner_x, 0.5 * wall, 0.5 * height)),
        ("ym", (0.0, -inner_y - 0.5 * wall, 0.5 * height), (inner_x, 0.5 * wall, 0.5 * height)),
    ]
    for name, pos, size in panels:
        for group, contype, rgba in (
            (COLLISION_GROUP, 1, [0.35, 0.35, 0.38, 0.6]),
            (VISUAL_GROUP, 0, [0.35, 0.35, 0.38, 1.0]),
        ):
            tray.add_geom(
                name=f"tray_{group}_{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(pos),
                size=list(size),
                group=group,
                contype=contype,
                conaffinity=contype,
                condim=3,
                rgba=rgba,
            )

    colours = ("blue", "yellow", "green", "orange", "white", "red")
    out = []
    for k in range(cfg.tray_bricks):
        x = tx + (k - (cfg.tray_bricks - 1) / 2.0) * TRAY_BRICK_SPACING * spec.pitch
        out.append(
            add_brick(
                world, name=f"tray_brick_{k}", nx=cfg.tray_brick[0], ny=cfg.tray_brick[1],
                colour=colours[k % len(colours)], pos=(x, ty, floor + 0.0005), spec=spec,
            )
        )
    return out


def set_brick_friction(handles, mu: float) -> None:
    """Give every collision geom of these bricks the same sliding friction, before compiling."""

    for handle in handles:
        for geom in handle.body.geoms:
            if int(geom.group) == COLLISION_GROUP:
                geom.friction = [mu, 0.005, 0.0001]


def _rotate_about(point: np.ndarray, centre: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    d = point - centre
    return centre + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def build_scene(cfg: SceneConfig = SceneConfig()) -> Scene:
    """Build, compile and reset one world described by ``cfg``.

    Returns a :class:`Scene` whose `data` is already at the home pose with the placed bricks
    connected to the plate; stepping it is safe straight away.
    """

    problems = validate_structure(cfg.placed, grid_shape=cfg.plate_studs)
    if problems:
        raise ValueError("invalid structure: " + "; ".join(problems))
    if cfg.snap_model not in ("collet", "detent"):
        raise ValueError(f"unknown snap_model {cfg.snap_model!r}")
    if cfg.grasp_mode not in ("weld", "friction"):
        raise ValueError(f"unknown grasp_mode {cfg.grasp_mode!r}")

    spec = cfg.spec
    held_cfg = cfg.held
    grasp = None
    if held_cfg is not None:
        grasp = grasp_pose(held_cfg.nx, held_cfg.ny, spec)

    world = mujoco.MjSpec()
    _add_world_furniture(world, cfg)

    # 1. the arm, then everything that stands on the table.
    attach_arm(world, cfg.arm)

    plate = add_baseplate(
        world, PLATE_BODY, cfg.plate_studs[0], cfg.plate_studs[1],
        pos=(cfg.plate_xy[0], cfg.plate_xy[1], 0.0), spec=spec,
        quat=tuple(_yaw_quat(cfg.plate_yaw)),
    )

    placed = build_structure(world, cfg.placed, plate, spec=spec)
    if cfg.plate_yaw:
        # `build_structure` lays the bricks out for an un-rotated plate; turning them about the
        # plate's centre afterwards keeps the grid arithmetic in one place instead of duplicating
        # it here with a rotation baked in.
        centre = np.array([cfg.plate_xy[0], cfg.plate_xy[1], 0.0])
        for handle in placed:
            pos = np.asarray(handle.body.pos, dtype=float)
            handle.body.pos = list(_rotate_about(pos, centre, cfg.plate_yaw))
            handle.body.quat = list(_yaw_quat(cfg.plate_yaw))

    tray_bricks = _add_tray(world, cfg) if cfg.tray else []
    tray_bricks += _add_supply(world, cfg) if cfg.supply else []

    held: BrickHandle | None = None
    if held_cfg is not None:
        held = add_brick(
            world, HELD_BODY, held_cfg.nx, held_cfg.ny, held_cfg.colour,
            # A harmless parking spot: `reset_scene` puts it in the hand before anything steps.
            pos=(cfg.plate_xy[0], cfg.plate_xy[1], 0.20), spec=spec,
        )
        eq = world.add_equality()
        eq.name = WELD_EQ
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = HELD_BODY
        eq.name2 = GRIPPER_BODY
        eq.active = False
        eq.data = np.zeros(mujoco.mjNEQDATA)
        eq.solref = list(cfg.weld_solref)
        eq.solimp = list(cfg.weld_solimp)

    _add_cameras(world, cfg, grasp)

    # 2. the snap model, on the finished spec and before compiling.
    snap = ColletSnap(cfg.snap) if cfg.snap_model == "collet" else DetentSnap(cfg.snap)
    bricks = list(placed) + list(tray_bricks) + ([held] if held is not None else [])
    set_brick_friction(bricks + [plate], cfg.brick_friction)
    snap.declare(world, bricks, [plate])

    # 3. compile, 4. bind.
    model = world.compile()
    data = mujoco.MjData(model)
    snap.bind(model, data)

    missing = [
        c for c in cfg.cameras
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) < 0
    ]
    if missing:
        raise RuntimeError(f"cameras {missing} are missing from the compiled model")

    kin = ArmKinematics(model)
    arm_qpos_adr, arm_dof_adr = [], []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        arm_qpos_adr.append(int(model.jnt_qposadr[jid]))
        arm_dof_adr.append(int(model.jnt_dofadr[jid]))

    instance_ids: dict[str, int] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            instance_ids[name] = 1  # everything that is not a brick or the plate: the arm, the tray
    instance_ids[plate.name] = 2
    if held is not None:
        instance_ids[held.name] = 3
    for k, handle in enumerate(list(placed) + list(tray_bricks)):
        instance_ids[handle.name] = 4 + k

    scene = Scene(
        model=model,
        data=data,
        cfg=cfg,
        snap=snap,
        bricks={h.name: h for h in bricks},
        plate=plate,
        kin=kin,
        arm_qpos_adr=np.asarray(arm_qpos_adr, dtype=np.int32),
        arm_dof_adr=np.asarray(arm_dof_adr, dtype=np.int32),
        instance_ids=instance_ids,
        grasp=grasp,
        held_name=held.name if held is not None else None,
        home_q=np.zeros(5),
        weld_eq=(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_EQ)
            if held is not None else -1
        ),
        in_hand_offset=np.zeros(4),
        placed_names=[h.name for h in placed],
        weld_torquescale=cfg.weld_torquescale,
    )
    scene.nominal = {
        "body_mass": model.body_mass.copy(),
        "body_inertia": model.body_inertia.copy(),
        "geom_size": model.geom_size.copy(),
        "light_pos": model.light_pos.copy(),
        "light_diffuse": model.light_diffuse.copy(),
        "cam_pos": model.cam_pos.copy(),
        "cam_quat": model.cam_quat.copy(),
        "cam_fovy": model.cam_fovy.copy(),
        "geom_rgba": model.geom_rgba.copy(),
    }
    scene.home_q = home_pose(scene)
    reset_scene(scene, np.random.default_rng(0))
    return scene


# ----------------------------------------------------------------------------------- posing ---


def tool_down_ik(
    kin: ArmKinematics,
    target_pos,
    yaw: float = 0.0,
    q0: np.ndarray | None = None,
    iters: int = 200,
    yaw_tol: float = math.radians(0.05),
) -> np.ndarray:
    """Joint angles putting the TCP at ``target_pos`` with the tool straight down at spin ``yaw``.

    `ArmKinematics.ik` solves position plus tool pitch and roll, which pins the tool axis but
    leaves the spin about that axis free - with the tool pointing down that free number is exactly
    the yaw we care about, and it comes out as whatever ``wrist_roll`` started at (the solver never
    needs to move it).  So the yaw is chased by a fixed-point loop on ``wrist_roll``: solve, read
    the achieved yaw, shift the seed by the error, solve again.  Two or three passes get well
    under a twentieth of a degree.
    """

    q = np.array([0.0, -0.5, 0.9, 1.2, 0.0]) if q0 is None else np.asarray(q0, dtype=float).copy()
    result = None
    for _ in range(8):
        result = kin.ik(np.asarray(target_pos, dtype=float), _TOOL_DOWN_PITCH, _TOOL_DOWN_ROLL,
                        q, iters=iters)
        q = result.q.copy()
        achieved, _, _ = euler_zyx(kin.fk(q)[1])
        err = math.remainder(yaw - achieved, 2.0 * math.pi)
        if abs(err) < yaw_tol:
            break
        q[4] = float(np.clip(q[4] + err, kin.jnt_range[4, 0], kin.jnt_range[4, 1]))
    if result is None or not result.converged:
        raise RuntimeError(
            f"no tool-down pose at {np.asarray(target_pos)} with yaw {math.degrees(yaw):.1f} deg "
            f"(position residual {0.0 if result is None else result.pos_error * 1e3:.2f} mm)"
        )
    return q


def tool_reach_below_tcp(scene: Scene) -> float:
    """How far the lowest part of the tool sits below the TCP, tool pointing down [m].

    With a brick in the hand that is the brick's bottom face, which hangs past the fingertips.
    With an empty gripper it is the fingertips themselves, and they are the thing that will hit
    the plate: they reach 8 mm below the TCP, so a home pose worked out for a held brick puts an
    empty gripper straight through whatever is standing on the plate.
    """

    if scene.grasp is not None:
        return float(scene.grasp.pos_in_tcp[2])
    pad_centre, pad_half = jaw_geometry().fixed_pad
    return float(tcp_position()[2] - (pad_centre[2] - pad_half[2]))


def plate_stack_height(scene: Scene) -> float:
    """Height of the tallest thing standing on the plate, above the plate's top surface [m]."""

    spec = scene.cfg.spec
    top = spec.stud_h
    for p in scene.cfg.placed:
        top = max(top, (p.z + 1) * spec.height + spec.stud_h)
    return top


def home_tcp_target(scene: Scene) -> np.ndarray:
    """Default home TCP: over the plate's centre, the tool hovering clear of everything on it."""

    cfg = scene.cfg
    if cfg.home_tcp is not None:
        return np.asarray(cfg.home_tcp, dtype=float)
    z = scene.plate_top + plate_stack_height(scene) + cfg.home_hover + tool_reach_below_tcp(scene)
    return np.array([cfg.plate_xy[0], cfg.plate_xy[1], z])


def home_pose(scene: Scene) -> np.ndarray:
    """The five arm joint angles a reset starts from."""
    return tool_down_ik(scene.kin, home_tcp_target(scene), scene.cfg.home_yaw)


def gripper_command(scene: Scene, mode: str | None = None) -> float:
    """Gripper joint angle the scene commands for the current grasp mode."""

    cfg = scene.cfg
    if scene.grasp is None:
        return gripper_angle_for_width(0.05)
    grasp = scene.grasp
    mode = mode or _grasp_mode(cfg)
    if mode == "friction":
        return jaw_angle_for_clearance(grasp.contact_point, -cfg.grip_squeeze)
    return jaw_angle_for_clearance(grasp.contact_point, cfg.weld_clearance)


def gripper_rest_angle(scene: Scene) -> float:
    """Jaw angle a reset *starts* at: the pads just touching the brick, with no interference."""

    if scene.grasp is None:
        return gripper_angle_for_width(0.05)
    if _grasp_mode(scene.cfg) == "friction":
        return scene.grasp.gripper_angle
    return gripper_command(scene)


def _grasp_mode(cfg: SceneConfig) -> str:
    if cfg.held is not None and cfg.held.grasp_mode is not None:
        return cfg.held.grasp_mode
    return cfg.grasp_mode


def held_brick_pose(scene: Scene, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the held brick for an in-hand error ``offset`` = (dx, dy, dz, dyaw).

    The error is applied in the TCP frame, so it means the same thing whatever the arm is doing.
    """

    tcp_pos, tcp_rot = scene.tcp_pose()
    grasp = scene.grasp
    local = grasp.pos_in_tcp + np.asarray(offset[:3], dtype=float)
    pos = tcp_pos + tcp_rot @ local
    quat_tcp = np.zeros(4)
    mujoco.mju_mat2Quat(quat_tcp, tcp_rot.flatten())
    # A spin about the tool's own axis, then the fixed brick-in-hand orientation.
    quat = _quat_mul(_quat_mul(quat_tcp, _yaw_quat(float(offset[3]))), grasp.quat_in_tcp)
    return pos, quat


def _write_weld(model: mujoco.MjModel, data: mujoco.MjData, eq: int, body1: int, body2: int,
                torquescale: float) -> None:
    """Freeze the current relative pose of two bodies into a weld's `eq_data`.

    Layout (MuJoCo 3.13, same as `snap_detent`): `[0:3]` is the weld point in body2's frame,
    `[3:6]` the same point in body1's frame, `[6:10]` body2's orientation relative to body1, and
    `[10]` the torque scale.
    """

    point = data.xpos[body1]
    r1 = data.xmat[body1].reshape(3, 3)
    r2 = data.xmat[body2].reshape(3, 3)
    model.eq_data[eq, 0:3] = r2.T @ (point - data.xpos[body2])
    model.eq_data[eq, 3:6] = r1.T @ (point - data.xpos[body1])
    conj = np.empty(4)
    rel = np.empty(4)
    mujoco.mju_negQuat(conj, data.xquat[body1])
    mujoco.mju_mulQuat(rel, conj, data.xquat[body2])
    model.eq_data[eq, 6:10] = rel
    model.eq_data[eq, 10] = torquescale


def reset_scene(scene: Scene, rng: np.random.Generator, arm_q: np.ndarray | None = None) -> None:
    """Put ``scene`` back to a clean start: arm at home, bricks placed, snap state cleared.

    ``rng`` is only consumed when the held brick declares a non-zero ``offset_jitter`` or
    ``yaw_jitter``; with the defaults a reset is a pure function of the scene, and with jitter on
    it is reproducible from the generator's state.  Nothing else here is random - domain
    randomization is `randomize.apply`'s job and belongs *before* a reset, so the bricks are laid
    out against the plate pose that randomization actually chose.
    """

    model, data, cfg = scene.model, scene.data, scene.cfg
    mujoco.mj_resetData(model, data)

    q = scene.home_q if arm_q is None else np.asarray(arm_q, dtype=float)
    data.qpos[scene.arm_qpos_adr[:5]] = q
    data.ctrl[:5] = q
    # The jaws *start* exactly touching the brick and are *commanded* closed: in friction mode the
    # command is `grip_squeeze` past the brick's face, and setting the joint there too would begin
    # the episode with that much interpenetration, which the solver answers by firing the brick out
    # of the hand.  Starting at contact and letting the servo build its pinch force over the first
    # few ticks is both stable and what the real gripper does.
    data.qpos[scene.arm_qpos_adr[5]] = gripper_rest_angle(scene)
    data.ctrl[5] = gripper_command(scene)

    # Placed bricks, read off the plate body as it stands now (so a randomized plate carries them).
    plate_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    plate_pos = model.body_pos[plate_bid].copy()
    plate_quat = model.body_quat[plate_bid].copy()
    plate_yaw = 2.0 * math.atan2(plate_quat[3], plate_quat[0])
    spec = cfg.spec
    stud_x0 = -(scene.plate.nx - 1) / 2.0 * spec.pitch
    stud_y0 = -(scene.plate.ny - 1) / 2.0 * spec.pitch
    top = plate_pos[2] + 2.0 * spec.roof
    for name, p in zip(scene.placed_names, cfg.placed):
        local = np.array([
            stud_x0 + (p.x + (p.w - 1) / 2.0) * spec.pitch,
            stud_y0 + (p.y + (p.h - 1) / 2.0) * spec.pitch,
            0.0,
        ])
        c, s = math.cos(plate_yaw), math.sin(plate_yaw)
        pos = np.array([
            plate_pos[0] + c * local[0] - s * local[1],
            plate_pos[1] + s * local[0] + c * local[1],
            top + p.z * spec.height,
        ])
        adr = scene.free_joint_adr(name)
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = plate_quat

    for k, handle in enumerate(h for h in scene.bricks.values()
                               if h.name.startswith(("tray_brick_", SUPPLY_PREFIX))):
        adr = scene.free_joint_adr(handle.name)
        data.qpos[adr:adr + 3] = np.asarray(handle.body.pos, dtype=float)
        data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

    if scene.held_name is not None:
        held = cfg.held
        if scene.in_hand_override is not None:
            offset = np.asarray(scene.in_hand_override, dtype=float).copy()
        else:
            offset = np.array([*held.offset, held.yaw], dtype=float)
            jitter = np.array([*held.offset_jitter, held.yaw_jitter], dtype=float)
            if np.any(jitter):
                offset = offset + rng.uniform(-jitter, jitter)
        scene.in_hand_offset = offset
        pos, quat = held_brick_pose(scene, offset)
        adr = scene.free_joint_adr(scene.held_name)
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_forward(model, data)

        if scene.weld_eq >= 0:
            mode = _grasp_mode(cfg)
            if mode == "weld":
                b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.held_name)
                b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
                _write_weld(model, data, scene.weld_eq, b1, b2, scene.weld_torquescale)
                data.eq_active[scene.weld_eq] = 1
            else:
                data.eq_active[scene.weld_eq] = 0

    # The snap model owns bookkeeping on both the model and its own side; re-binding is the one
    # call that clears all of it (partners, capture anchors, the clutch schedule).
    scene.snap.bind(model, data)
    mujoco.mj_forward(model, data)
    scene.snap.update(model, data)
    mujoco.mj_forward(model, data)


# -------------------------------------------------------------------------------- measuring ---


@dataclass
class SettleReport:
    """What happened while a scene stood still."""

    seconds: float
    warnings: dict[str, int]
    max_speed: float                 # largest brick linear speed at the end [m/s]
    drift: dict[str, float]          # per brick, how far it moved [m]
    connections: list                # snap connections at the end
    connected_placed: int


def settle(scene: Scene, seconds: float = 2.0) -> SettleReport:
    """Hold the commanded pose for ``seconds`` and report what moved."""

    model, data = scene.model, scene.data
    names = [n for n in scene.bricks]
    start = {n: scene.body_pose(n)[0] for n in names}
    n_steps = int(round(seconds / model.opt.timestep))
    scene.step(n_steps)

    drift = {n: float(np.linalg.norm(scene.body_pose(n)[0] - start[n])) for n in names}
    speeds = []
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")
        if jid < 0:
            continue   # a welded plate has no free joint and cannot be "still moving"
        dof = int(model.jnt_dofadr[jid])
        speeds.append(float(np.linalg.norm(data.qvel[dof:dof + 3])))
    conns = scene.snap.connections()
    connected = {c.upper for c in conns if c.lower == scene.plate.name}
    return SettleReport(
        seconds=seconds,
        warnings=scene.warnings(),
        max_speed=max(speeds) if speeds else 0.0,
        drift=drift,
        connections=conns,
        connected_placed=sum(1 for n in scene.placed_names if n in connected),
    )


def push_held_brick(
    scene: Scene, force: float = 5.0, seconds: float = 5.0, direction=(0.0, 0.0, -1.0)
) -> dict:
    """Pull on the held brick and report how far it slipped in the hand.

    The acceptance test for a grasp is a 5 N downward load held for 5 s with less than 1 mm of
    slip.  Slip is measured in the *TCP frame*, because the arm itself bends under the load: what
    matters is whether the brick moved relative to the gripper, not whether the tool moved.
    """

    model, data = scene.model, scene.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.held_name)
    vec = np.asarray(direction, dtype=float)
    vec = vec / np.linalg.norm(vec) * force

    def in_hand() -> np.ndarray:
        tcp_pos, tcp_rot = scene.tcp_pose()
        return tcp_rot.T @ (data.xpos[bid] - tcp_pos)

    before = in_hand()
    n_steps = int(round(seconds / model.opt.timestep))
    peak = 0.0
    for k in range(n_steps):
        data.xfrc_applied[bid, :3] = vec
        if k % max(int(scene.snap.params.snap_every), 1) == 0:
            scene.snap.update(model, data)
        mujoco.mj_step(model, data)
        peak = max(peak, float(np.linalg.norm(in_hand() - before)))
    data.xfrc_applied[bid, :3] = 0.0
    slip = float(np.linalg.norm(in_hand() - before))
    return {
        "force": force,
        "seconds": seconds,
        "slip_m": slip,
        "peak_slip_m": peak,
        "warnings": scene.warnings(),
        "grasp_mode": _grasp_mode(scene.cfg),
    }


def measure_step_rate(scene: Scene, steps: int = 2000, warmup: int = 200) -> float:
    """Physics steps per second, with the snap bookkeeping running at its configured rate."""

    import time

    scene.step(warmup)
    t0 = time.perf_counter()
    scene.step(steps)
    return steps / (time.perf_counter() - t0)


# ---------------------------------------------------------------------------------- figures ---


def make_scene_figures(out_dir="docs/media/scene", width: int = 900, height: int = 700) -> dict:
    """Render the three PNGs this package owes: overview, wrist view and grasp close-up.

    This is the figure-producing entry point `scripts/make_media.py` calls.  It is also the visual
    check on the grasp construction: `grasp_closeup.png` is where you see how much of the brick's
    wall is actually between the two jaw faces and how far it hangs past the tips, and
    `wrist_view.png` is where you see how much of the wrist camera's frame the fixed jaw takes up.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from brick_builder.render.renderer import SceneRenderer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SceneConfig(
        placed=[Placement(2, 4, 1, 2, 0, "blue"), Placement(2, 2, 5, 3, 0, "yellow")],
        tray=True,
        cameras=(WRIST_CAM, OVERVIEW_CAM, GRASP_CAM),
        # Big enough that the close-up below can be rendered at a size where its caption is
        # readable on a phone without zooming in.
        offscreen=(2200, 1700),
    )
    scene = build_scene(cfg)
    settle(scene, 0.5)

    out = {}
    renderer = SceneRenderer(scene.model, width=width, height=height)
    try:
        renderer.instance_lut(scene.instance_ids)

        frame = renderer.render(scene.data, OVERVIEW_CAM)
        fig, ax = plt.subplots(figsize=(width / 150, height / 150), dpi=150)
        ax.imshow(frame.rgb)
        ax.set_title(
            "Default scene: SO-101, 8x8 plate at (0.17, 0.0), two placed bricks, "
            "a tray, and a 2x4 in the hand",
            fontsize=8,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "overview.png")
        plt.close(fig)
        out["overview.png"] = str(out_dir / "overview.png")

        # Two wrist views: the press pose, and 80 mm higher, where the plate is actually visible.
        home = renderer.render(scene.data, WRIST_CAM)
        lifted_target = home_tcp_target(scene) + np.array([0.02, 0.0, 0.025])
        reset_scene(scene, np.random.default_rng(0),
                    arm_q=tool_down_ik(scene.kin, lifted_target, scene.cfg.home_yaw,
                                       q0=scene.home_q))
        settle(scene, 0.2)
        lifted = renderer.render(scene.data, WRIST_CAM)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=150)
        for ax, frame_, title in (
            (axes[0], home, "wrist_cam at the press pose"),
            (axes[1], lifted, "wrist_cam 25 mm higher, 20 mm further out"),
        ):
            ax.imshow(frame_.rgb)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(
            "The fixed jaw fills the right half of the wrist camera at every pose: "
            "look-then-move, not look-while-pressing",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "wrist_view.png")
        plt.close(fig)
        out["wrist_view.png"] = str(out_dir / "wrist_view.png")
    finally:
        renderer.close()

    # The close-up gets its own, much larger, render: the margins it has to show are fractions of
    # a millimetre, and the caption is the numbers, so both have to survive being looked at on a
    # phone.  2200 x 1700 pixels of render at 200 dpi is 11 x 8.5 inches of figure, which puts the
    # caption at 13 pt -- about 36 pixels tall -- instead of the 8 pt it used to be.
    reset_scene(scene, np.random.default_rng(0))
    settle(scene, 0.3)
    close = SceneRenderer(scene.model, width=2200, height=1700)
    try:
        close.instance_lut(scene.instance_ids)
        frame = close.render(scene.data, GRASP_CAM)
    finally:
        close.close()
    grasp = scene.grasp
    fig, ax = plt.subplots(figsize=(11.0, 9.2), dpi=200)
    ax.imshow(frame.rgb)
    ax.axis("off")
    ax.set_title(
        f"The stock gripper holding a {cfg.held.nx}x{cfg.held.ny} DUPLO brick across its "
        f"{grasp.width * 1e3:.1f} mm width, tool pointing straight down\n"
        f"gripper joint {grasp.gripper_angle:.3f} rad  |  "
        f"{grasp.pad_overlap * 1e3:.1f} mm of the brick's {1e3 * DUPLO.height:.1f} mm wall lies "
        f"between the jaw faces\n"
        f"the brick hangs {grasp.tip_clearance * 1e3:.0f} mm below the jaw tips, so the tips "
        f"clear the baseplate as it seats",
        fontsize=13,
        linespacing=1.5,
    )
    fig.tight_layout()
    # `bbox_inches` because the three-line caption is taller than `tight_layout` reserves for
    # it and the first line was being cropped off the top of the image.
    fig.savefig(out_dir / "grasp_closeup.png", bbox_inches="tight")
    plt.close(fig)
    out["grasp_closeup.png"] = str(out_dir / "grasp_closeup.png")
    return out
