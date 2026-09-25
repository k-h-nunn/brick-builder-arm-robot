"""Pick a DUPLO brick from the supply with the stock jaws and seat it on the baseplate.

Everything here runs on the arm as it comes out of the kit: a base, a shoulder, an upper arm, a
lower arm, a wrist, and a gripper with one fixed jaw and one moving jaw.  Nothing is attached to
either jaw.  The brick is held only by friction between the bare printed jaw faces and its own
two long walls, which is a weak grip and a measured one - see :func:`bench`, and the guide's
"Seating a brick with the stock jaws" for the numbers.

The sequence
------------
``home`` -> ``to_tray`` -> ``open`` -> ``descend`` -> ``close`` -> ``lift`` -> ``carry`` ->
``align`` -> ``place`` -> ``release`` -> ``press`` -> ``retreat`` -> ``home`` -> ``done``.

Two ways of seating the brick are implemented, because the interesting question is which one the
stock gripper can actually do:

``"gripped"``
    keep hold of the brick and drive the tool down until the studs are in.  The reaction pushes
    the brick *up* between the jaws, and the jaw gap narrows going up, so the wedge helps: the
    harder you press, the tighter the brick is pinched.  It is the shorter sequence and it needs
    no second approach.
``"release"``
    set the brick down on the stud tops, open the jaws, bring them together, and push straight
    down on the brick's roof with the tip of the fixed jaw.  No grip force is involved in the
    press at all, so nothing about it depends on how well bare plastic grips bare plastic.
``"gripped_then_push"``
    both, in that order: press it in while holding it, let go, and then push once more on its
    roof with the jaws together.  The second push exists because of what the first one leaves
    behind when it fails - a brick standing at fourteen degrees with one row of studs in its
    sockets and the other row on top of the studs, which a push on the roof can flatten and a
    grip that is holding the brick in that attitude cannot.

Driving the arm
---------------
:class:`ArmDriver` is the same multi-rate loop the environments run: one
:class:`~brick_builder.robot.servo_model.ServoBus` write per 20 ms control tick, ten 2 ms physics
steps, one read, with :class:`~brick_builder.robot.bus_emulator.BusEmulator` supplying the command
latency, read skew and packet loss.  It reuses ``envs.base_env._pose_ik`` rather than keeping a
second copy of the warm-started IK: this module is a scripted sequence for benchmarking and for
recording, not something that has to be transcribed into firmware, so the portability rule that
:mod:`brick_builder.control.primitives` keeps to does not apply to it.

Recording
-------
:meth:`PickPlace.run` is a generator.  It yields one :class:`Step` per control tick, carrying the
live :class:`~brick_builder.scene.compose.Scene` and the phase name, so a video script can render a
frame and draw the caption without knowing anything about the state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Literal

import mujoco
import numpy as np

from brick_builder.control.primitives import (
    CONTROL_DT,
    Feedback,
    MoveTo,
    PressConfig,
    PressScripted,
    feedback_from_readings,
)
from brick_builder.robot.bus_emulator import BusEmulator, BusConfig
from brick_builder.robot.joint_correction import JointCorrection, JointCorrectionConfig
from brick_builder.robot.kinematics import euler_zyx
from brick_builder.robot.patch_so101 import (
    ARM_JOINTS,
    GRIPPER_BODY,
    gripper_angle_for_width,
    jaw_geometry,
    tcp_position,
)
from brick_builder.robot.servo_model import ServoBus, ServoConfig, ServoReadings
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.robot.patch_so101 import ArmConfig
from brick_builder.scene.compose import (
    GraspPose,
    Scene,
    SceneConfig,
    build_scene,
    grasp_pose,
    jaw_angle_for_clearance,
    reset_scene,
    tool_down_ik,
)

PHYSICS_PER_TICK = 10

PHASES: tuple[str, ...] = (
    "home",
    "to_tray",
    "open",
    "descend",
    "close",
    "lift",
    "carry",
    "align",
    "place",
    "release",
    "nudge",
    "press",
    "retreat",
    "go_home",
    "done",
    "failed",
)
"""Every value :attr:`Step.phase` can take, in the order they normally occur."""

PressStrategy = Literal["gripped", "release", "gripped_then_push"]


# --------------------------------------------------------------------------- tool geometry ---


def jaw_tip_offset() -> np.ndarray:
    """Where the fixed jaw's tip sits relative to the TCP, in the TCP frame [m].

    With the jaws brought together the fixed jaw reaches 7.2 mm lower than the moving one, so it
    is the only thing that touches anything when the tool is pushed straight down.  Its tip is a
    box about 14 x 15 mm, offset sideways from the TCP because the TCP sits in the middle of a
    gripped brick and the fixed jaw sits at one edge of it.  ``x`` is that sideways offset, ``z``
    is how far below the TCP the tip reaches (positive, because the TCP frame's z points along
    the approach direction).
    """

    geo = jaw_geometry()
    tip_centre, tip_half = geo.fixed_boxes[0]
    pad_centre, pad_half = geo.fixed_pad
    tcp = tcp_position()
    # The TCP frame is the gripper frame turned 180 degrees about x, so y and z change sign.
    return np.array([
        float(tip_centre[0] - tcp[0]),
        -float(tip_centre[1] - tcp[1]),
        -float((pad_centre[2] - pad_half[2]) - tcp[2]),
    ])


def tcp_for_brick(brick_pos, spin: float, grasp: GraspPose) -> np.ndarray:
    """TCP position that puts a held brick's origin at ``brick_pos``, tool down at yaw ``spin``.

    ``grasp.pos_in_tcp`` is the brick's origin - the centre of its bottom face - in the TCP
    frame; rotating that into the world and subtracting it inverts the relation.
    """

    rot = _tool_down_rot(spin)
    return np.asarray(brick_pos, dtype=float) - rot @ grasp.pos_in_tcp


def tcp_for_foot(foot_xy, foot_z: float, spin: float) -> np.ndarray:
    """TCP position that puts the fixed jaw's tip at ``foot_xy`` and height ``foot_z``."""

    rot = _tool_down_rot(spin)
    offset = rot @ jaw_tip_offset()
    return np.array([foot_xy[0] - offset[0], foot_xy[1] - offset[1], foot_z - offset[2]])


def _tool_down_rot(spin: float) -> np.ndarray:
    """Rotation of the TCP frame when the tool points straight down at yaw ``spin``."""
    c, s = math.cos(spin), math.sin(spin)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def cell_world(scene: Scene, cx: float, cy: float) -> np.ndarray:
    """World xy of a plate cell centre, from the plate's *actual* pose, z at the plate's top."""

    model = scene.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    origin = model.body_pos[bid].copy()
    quat = model.body_quat[bid]
    yaw = 2.0 * math.atan2(quat[3], quat[0])
    pitch = scene.cfg.spec.pitch
    local = np.array([
        (cx - (scene.plate.nx - 1) / 2.0) * pitch,
        (cy - (scene.plate.ny - 1) / 2.0) * pitch,
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        origin[0] + c * local[0] - s * local[1],
        origin[1] + s * local[0] + c * local[1],
        scene.plate_top,
    ])


# ------------------------------------------------------------------------------ the driver ---


class ArmDriver:
    """One 50 Hz control loop over a scene: write, ten physics steps, read.

    Exactly what :class:`~brick_builder.envs.base_env.ArmEnv` does per ``step``, minus the action
    integration, the observation assembly and the reward - a scripted sequence commands a tool
    pose directly.  ``torque`` is the per-joint fraction of the servo's own limit, which is how
    the compliance scalars reach the real ``Torque_Limit`` register.

    ``corrector`` is the optional host-side integral trim on the joint angles the bus reports
    (:class:`~brick_builder.robot.joint_correction.JointCorrection`).  It is off by default, so a
    caller that does not ask for it gets exactly the loop this class has always run.  It is
    frozen automatically on any tick that commands a compliance below 1 - a press - because an
    error the arm is *meant* to have while leaning on a brick must not wind the trim up.
    """

    def __init__(
        self,
        scene: Scene,
        *,
        servo: ServoConfig | None = None,
        bus: BusConfig | None = None,
        rng: np.random.Generator | None = None,
        corrector: JointCorrection | None = None,
    ) -> None:
        self.scene = scene
        self.kin = scene.kin
        self.corrector = corrector
        rng = np.random.default_rng(0) if rng is None else rng
        self.servo = ServoBus(
            scene.model, scene.data, cfg=servo or ServoConfig(),
            rng=np.random.default_rng(int(rng.integers(2**31))),
        )
        self.bus = BusEmulator(
            cfg=bus or BusConfig(), rng=np.random.default_rng(int(rng.integers(2**31)))
        )
        gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.grip_range = scene.model.actuator_ctrlrange[gid].copy()
        self.q_goal = scene.data.qpos[scene.arm_qpos_adr[:5]].copy()
        self.reading: ServoReadings = self.servo.read()
        self.tick = 0
        self.ik_failures = 0
        self.peak_duty = 0.0
        self.peak_torque = 0.0
        self._act_ids = np.array([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS
        ])

    # -- one tick ---------------------------------------------------------------------------

    def step(self, pos, spin: float, grip: float, *, tilt: float = 0.0,
             c_vert: float = 1.0, c_lat: float = 1.0) -> None:
        from brick_builder.envs.base_env import _pose_ik

        pos = np.asarray(pos, dtype=float)
        q5, ok = _pose_ik(self.kin, pos, tilt, spin, self.q_goal)
        if not ok and tilt == 0.0:
            # `_pose_ik` is tuned for an environment, where a tick moves the target a couple of
            # millimetres and two damped-least-squares steps are always enough.  A scripted
            # sequence crosses the whole workspace - the supply is 190 mm from the home pose - and
            # over the supply the short solve stalls several millimetres short and never recovers,
            # because a rejected solve leaves the joint goal exactly where it was.  The full
            # Levenberg-Marquardt solve costs about a millisecond and only runs when the cheap
            # one has already given up, which on a clean run is under 2 % of ticks.
            try:
                q5, ok = tool_down_ik(self.kin, pos, spin, q0=self.q_goal), True
            except RuntimeError:
                ok = False
        if ok:
            self.q_goal = q5
        else:
            self.ik_failures += 1
        grip = float(np.clip(grip, self.grip_range[0], self.grip_range[1]))
        goal = np.concatenate([self._gravity_feedforward(self.q_goal), [grip]])
        if self.corrector is not None:
            # The trim sees the plain tracking error, `q_goal` against the last reading, so it
            # does not spend itself re-deriving the gravity term the feedforward already carries;
            # its output is added to the feedforward instead of replacing the command.
            n = self.corrector.n
            self.corrector.update(
                self.q_goal[:n], self.reading.position[:n],
                freeze=c_vert < 1.0 or c_lat < 1.0,
            )
            goal[:n] += self.corrector.correction

        torque = np.ones(len(ARM_JOINTS))
        torque[[ARM_JOINTS.index(n) for n in ("shoulder_lift", "elbow_flex", "wrist_flex")]] = c_vert
        torque[[ARM_JOINTS.index(n) for n in ("shoulder_pan", "wrist_roll")]] = c_lat

        timing = self.bus.tick(self.tick * CONTROL_DT)
        self.servo.cfg.action_delay_s = float(timing.command_latency_s)
        self.servo.cfg.action_delay_jitter_s = 0.0
        if not timing.write_lost:
            self.servo.write(goal, torque_limit=torque)

        snap_every = max(int(self.scene.snap.params.snap_every), 1)
        for k in range(PHYSICS_PER_TICK):
            self.servo.substep()
            if k % snap_every == 0:
                self.scene.snap.update(self.scene.model, self.scene.data)
            mujoco.mj_step(self.scene.model, self.scene.data)

        self.servo.cfg.read_skew_s = float(timing.read_skew_s) / max(self.servo.n - 1, 1)
        new = self.servo.read()
        stale = np.asarray(timing.reply_lost, dtype=bool)
        if np.any(stale):
            new = ServoReadings(
                position=np.where(stale, self.reading.position, new.position),
                speed=np.where(stale, self.reading.speed, new.speed),
                load=np.where(stale, self.reading.load, new.load),
                current=np.where(stale, self.reading.current, new.current),
                tripped=new.tripped,
            )
        self.reading = new
        self.peak_duty = max(self.peak_duty, float(np.max(np.abs(new.load))))
        self.peak_torque = max(
            self.peak_torque,
            float(np.max(np.abs(self.scene.data.actuator_force[self._act_ids]))),
        )
        self.tick += 1

    def _gravity_feedforward(self, q5: np.ndarray) -> np.ndarray:
        """The command that puts the arm at ``q5``, not ``q5`` itself (see ``ArmEnv``)."""
        kp5 = self.servo.kp[:5]
        comp = q5.copy()
        for _ in range(2):
            comp = q5 + self.kin.gravity_torque(comp) / kp5
        return comp

    def feedback(self, grip: float) -> Feedback:
        return feedback_from_readings(self.kin, self.reading, grip=grip)

    @property
    def tripped(self) -> bool:
        return bool(np.any(self.reading.tripped))


# ------------------------------------------------------------------------------ the config ---


@dataclass(frozen=True)
class PickPlaceConfig:
    """Tuning for :class:`PickPlace`.  Distances in metres, speeds in m/s, times in seconds."""

    press_strategy: PressStrategy = "gripped_then_push"
    """Which of the three ways of seating the brick to use.

    ``"gripped_then_push"`` is the default because it is the one that measures best: 46 out of
    50 runs with no placement error and 45 out of 50 with a millimetre of it, against 43 and 42
    for pressing while gripping alone and 23 and 27 for placing and pushing alone.  It costs
    about five seconds a brick more than pressing while gripping.
    """

    open_clearance: float = 0.004
    """How much wider than the brick the jaws stand while descending onto it.

    Measured at the brick's top outer edge, where the tapering gap is narrowest, so it is a
    guarantee for the whole height of the brick.  4 mm leaves the tool about 37.6 mm wide at tip
    height around a 31.8 mm brick, which clears the neighbouring supply brick.
    """

    squeeze: float = 0.006
    """How far past the brick's face the jaws are commanded once closed [m].

    The same over-closure the scene's own ``friction`` grasp mode uses; with the servo's 1.9 N.m
    limit it settles at about 19 N of pinch.
    """

    approach: float = 0.055
    """Hover height above whatever the tool is about to descend onto [m]."""

    carry_height: float = 0.045
    """Height the brick's bottom face is carried at, above the plate's top surface [m].

    Measured, not guessed: with the tool pointing straight down this arm runs out of reach at a
    TCP height of about 80 mm everywhere it has to go, and the brick hangs 16 mm below the TCP,
    so 45 mm is the most that still clears a standing brick (19.2 mm plus 4.5 mm of stud) by a
    comfortable margin at both ends of the carry.
    """

    descend_speed: float = 0.03
    travel_speed: float = 0.12
    close_time: float = 0.5
    open_time: float = 0.35
    settle_time: float = 0.25

    place_gap: float = 0.0005
    """How far above the stud tops the brick is let go of, in the ``"release"`` strategy [m]."""

    taught_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """What to add to a commanded target so the tool really arrives there [m].  Zero by default.

    This is where a taught grid would go, and the reason one is needed.  Every joint of this arm
    carries half a degree of modelled gear backlash, and the servo's encoder reads the *driven*
    side of that free play, so the link can sit up to half a degree off what the encoder reports
    and nothing the robot can measure will say so.  Measured over the plate: the tool really sits
    about 4 mm short radially and 2 mm low, while the encoders insist it is within 0.3 mm of the
    command.

    Putting the measured 4 mm in here does *not* rescue the success rate - it tightens the runs
    that already worked (0.12 mm off the cell down to 0.04 mm) and makes the ones that failed
    worse (2-3 mm out to 4-6 mm), because the free play settles to whichever stop the last move
    pushed it against and is not the same constant every run.  It is left at zero rather than
    fitted, and the variable part of it is the single biggest thing standing between this
    sequence and a reliable one.  See `docs/guide/02-simulator-and-bricks.md`.
    """

    press: PressConfig = field(default_factory=lambda: PressConfig(press_depth=0.010))
    """Press tuning for the ``"gripped"`` strategy: the project's own, unchanged."""

    release_press: PressConfig = field(default_factory=lambda: PressConfig(
        press_depth=0.014, check_height=0.0010, max_searches=0, max_time=6.0,
    ))
    """Press tuning for the ``"release"`` strategy, which is a different mechanical problem.

    - ``check_height`` 1.0 mm instead of 2.5 mm.  The 2.5 mm threshold was calibrated against the
      soft weld that stands in for the grip, and about 2 mm of it was the weld's own stretch.
      Pushing on a brick with a rigid jaw has no such stretch, so the old threshold called a
      brick that was a millimetre short "seated".
    - ``max_searches`` 0.  The spiral search skims the stud tops and waits for the tool to drop
      into a socket; that is a statement about a brick the tool is *holding*.  Here the brick is
      loose on the plate, the tool only rests on its roof, and sweeping sideways just shoves it
      off the cell.
    - ``press_depth`` 14 mm rather than 10 mm, because the foot is 24 mm out from the tool axis
      and the arm gives up some of its push to the resulting wrist torque.
    """

    nudge: float = 0.0
    """Sideways shove applied to the placed brick with the closed jaws before pressing [m].

    Zero by default, because it was measured and it is worse than doing nothing: twelve runs of
    the ``"release"`` strategy at 1 mm of placement error seat five bricks with no nudge and
    none at all with a 2 mm one, and the run gets five and a half seconds longer.  Squaring a
    loose brick up by pushing on its side with the jaw tip pushes it *off* the studs it was
    resting on more often than it lines it up.  The option stays because "nudge it into line
    first" is the obvious thing to try and this is the number that says not to.
    """

    max_ticks: int = 1800
    """Safety stop: 36 s at 50 Hz, about three times what a clean run takes."""


# ------------------------------------------------------------------------------- one tick ---


@dataclass
class Step:
    """What one control tick of the sequence looked like, for a video or a trace."""

    tick: int
    t: float
    phase: str
    scene: Scene
    tcp_pos: np.ndarray
    cmd_pos: np.ndarray
    grip: float
    peak_duty: float
    peak_torque: float
    press_phase: str | None = None


# --------------------------------------------------------------------------- the sequence ---


class PickPlace:
    """Drive one brick from the supply to a cell on the plate, and report what happened.

    ``brick`` is the body name of the brick to pick (``"supply_brick_0"`` in the default scene),
    ``target`` is ``(cx, cy, layer)`` in plate stud coordinates - the centre of the brick's
    footprint, so a 2-wide footprint sits at a half-integer.

    ``pick_at`` is where the arm *believes* the brick is - the taught supply slot, or whatever a
    camera reported.  Leaving it ``None`` reads the brick's true pose out of the simulator, which
    is fine for a demonstration but is a privileged signal, so anything that measures a success
    rate should pass the slot instead and let the brick be somewhere else.

    ``aim_error`` is what the arm gets *wrong* on top of that: ``(pick_dx, pick_dy, place_dx,
    place_dy)`` in metres, added to where it believes the brick and the cell are.  The simulator
    still knows the truth; the arm does not, which is the only honest way to ask "how much error
    survives".
    """

    def __init__(
        self,
        scene: Scene,
        *,
        brick: str = "supply_brick_0",
        target: tuple[float, float, int] = (3.5, 3.5, 0),
        cfg: PickPlaceConfig = PickPlaceConfig(),
        aim_error: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        pick_at=None,
        driver: ArmDriver | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.scene = scene
        self.brick = brick
        self.target = (float(target[0]), float(target[1]), int(target[2]))
        self.cfg = cfg
        self.aim_error = np.asarray(aim_error, dtype=float)
        self.pick_at = None if pick_at is None else np.asarray(pick_at, dtype=float)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.driver = driver if driver is not None else ArmDriver(scene, rng=self.rng)

        handle = scene.bricks[brick]
        self.grasp = grasp_pose(handle.nx, handle.ny, scene.cfg.spec)
        self.phase = "home"
        self.failure: str | None = None
        self.press: PressScripted | None = None
        self.peak_press_n = 0.0
        self.lifted_by = 0.0
        """Highest the brick's bottom face ever got [m] - the telltale for a failed pick."""
        self.press_forces: list[float] = []
        """Force the arm put into the brick on every tick of the press phase [N].

        Kept as a trace rather than one number because a rigid jaw meeting a rigid stud spikes:
        the peak is an impact, and what seats the brick is what the arm can *hold*."""
        self._grip = gripper_angle_for_width(0.05)
        self._cmd: np.ndarray | None = None

    # -- the two gripper commands ------------------------------------------------------------

    @property
    def grip_open(self) -> float:
        return jaw_angle_for_clearance(self.grasp.contact_point, self.cfg.open_clearance)

    @property
    def grip_closed(self) -> float:
        return jaw_angle_for_clearance(self.grasp.contact_point, -self.cfg.squeeze)

    @property
    def grip_together(self) -> float:
        """Jaws brought together with nothing between them, for the closed-jaw press.

        2 mm of gap at the narrow end rather than the joint's own hard stop: a commanded stop is
        a command the servo will keep leaning on, and there is nothing to be gained from the two
        jaws grinding on each other for the length of a press.
        """
        return gripper_angle_for_width(0.002)

    # -- the run ------------------------------------------------------------------------------

    def run(self) -> Iterator[Step]:
        """Drive the whole sequence, yielding one :class:`Step` per control tick."""

        spec = self.scene.cfg.spec
        cx, cy, layer = self.target
        cell = cell_world(self.scene, cx, cy)
        plate_top = self.scene.plate_top

        # Where the arm *believes* things are.
        believed = (
            self.scene.body_pose(self.brick)[0].copy()
            if self.pick_at is None else self.pick_at
        )
        pick_xy = believed[:2] + self.aim_error[:2]
        pick_bottom = float(believed[2])
        taught = np.asarray(self.cfg.taught_offset, dtype=float)
        place_xy = cell[:2] + self.aim_error[2:] + taught[:2]
        seat_bottom = plate_top + layer * spec.height + taught[2]

        grasp = self.grasp
        # Offer the brick the middle of the open gap rather than the fixed jaw's face.  The
        # nominal grasp has the brick flush against the fixed jaw, which is where it ends up once
        # the jaws close - but it is a terrible place to *arrive*, because the fixed jaw cannot
        # move: a brick a couple of millimetres off towards it is outside the gripper, and the
        # descent pushes it over instead of picking it up.  Starting half the open clearance back
        # means the moving jaw shoves the brick flush as it closes, so an error across the jaws
        # is corrected by the act of gripping.
        centre = _tool_down_rot(0.0) @ np.array([0.5 * self.cfg.open_clearance, 0.0, 0.0])
        pick_tcp = tcp_for_brick([pick_xy[0], pick_xy[1], pick_bottom], 0.0, grasp) - centre
        carry_z = plate_top + self.cfg.carry_height + float(grasp.pos_in_tcp[2])

        yield from self._hold("home", 0.15)
        yield from self._move("to_tray", [pick_tcp[0], pick_tcp[1], carry_z],
                              speed=self.cfg.travel_speed, grip=self.grip_open)
        yield from self._hold("open", self.cfg.open_time, grip=self.grip_open)
        yield from self._move("descend", pick_tcp, speed=self.cfg.descend_speed,
                              grip=self.grip_open)
        if self.failure:
            return
        yield from self._ramp_grip("close", self.grip_open, self.grip_closed, self.cfg.close_time)
        if not self._brick_is_held():
            yield from self._fail("lost the brick while closing")
            return

        yield from self._move("lift", [pick_tcp[0], pick_tcp[1], carry_z],
                              speed=self.cfg.descend_speed, grip=self.grip_closed)
        over_cell = tcp_for_brick([place_xy[0], place_xy[1], seat_bottom], 0.0, grasp)
        yield from self._move("carry", [over_cell[0], over_cell[1], carry_z],
                              speed=self.cfg.travel_speed, grip=self.grip_closed)
        if self.failure:
            return

        if self.cfg.press_strategy == "release":
            yield from self._place_then_press(place_xy, seat_bottom, spec.stud_h)
        else:
            yield from self._press_gripped(place_xy, seat_bottom)
            if self.cfg.press_strategy == "gripped_then_push" and not self.failure:
                yield from self._clear_of_the_brick(place_xy, seat_bottom, spec)
                yield from self._push_with_closed_jaws(place_xy, seat_bottom, spec)

        # Even a run that gave up gets the arm out of the way, so what is on the plate can be
        # verified with nothing resting on it.
        retreat_z = plate_top + self.cfg.carry_height + float(grasp.pos_in_tcp[2])
        yield from self._move("retreat", [self._cmd[0], self._cmd[1], retreat_z],
                              speed=self.cfg.descend_speed, grip=self.grip_open)
        yield from self._move("go_home", self.scene.kin.fk(self.scene.home_q)[0],
                              speed=self.cfg.travel_speed, grip=self.grip_open)
        yield from self._hold("done", self.cfg.settle_time, grip=self.grip_open)

    # -- the two press strategies --------------------------------------------------------------

    def _press_gripped(self, place_xy, seat_bottom: float):
        """Keep hold of the brick and drive it down until the studs are in."""

        grasp = self.grasp
        seat_tcp = tcp_for_brick([place_xy[0], place_xy[1], seat_bottom], 0.0, grasp)
        start = np.array([seat_tcp[0], seat_tcp[1], seat_tcp[2] + self.cfg.approach * 0.25])
        yield from self._move("align", start, speed=self.cfg.descend_speed, grip=self.grip_closed)
        if self.failure:
            return
        self.press = PressScripted(float(seat_tcp[2]), self.cfg.press, grip=self.grip_closed)
        self.press.reset(self.driver.feedback(self.grip_closed))
        yield from self._run_press()
        yield from self._ramp_grip("release", self.grip_closed, self.grip_open, self.cfg.open_time)

    def _place_then_press(self, place_xy, seat_bottom: float, stud_h: float):
        """Set the brick down on the stud tops, let go, and push on its roof with the fixed jaw."""

        grasp = self.grasp
        spec = self.scene.cfg.spec
        rest_bottom = seat_bottom + stud_h + self.cfg.place_gap
        rest_tcp = tcp_for_brick([place_xy[0], place_xy[1], rest_bottom], 0.0, grasp)
        yield from self._move("align", [rest_tcp[0], rest_tcp[1],
                                        rest_tcp[2] + self.cfg.approach * 0.25],
                              speed=self.cfg.descend_speed, grip=self.grip_closed)
        yield from self._move("place", rest_tcp, speed=self.cfg.descend_speed,
                              grip=self.grip_closed)
        yield from self._ramp_grip("release", self.grip_closed, self.grip_open, self.cfg.open_time)
        yield from self._clear_of_the_brick(place_xy, seat_bottom, spec)
        yield from self._push_with_closed_jaws(place_xy, seat_bottom, spec)

    def _clear_of_the_brick(self, place_xy, seat_bottom: float, spec):
        """Lift the open jaws up out of the brick and bring them together above it."""
        clear_z = seat_bottom + 2.0 * spec.height + 0.012
        yield from self._move("release", [self._cmd[0], self._cmd[1], clear_z],
                              speed=self.cfg.descend_speed, grip=self.grip_open)
        yield from self._ramp_grip("release", self.grip_open, self.grip_together,
                                   self.cfg.open_time)

    def _push_with_closed_jaws(self, place_xy, seat_bottom: float, spec):
        """Push straight down on the brick's roof with the tip of the fixed jaw.

        The foot lands on the middle of the roof, which means on the inner half of both stud
        columns - it is 14 mm across and the columns sit 8 mm either side of centre - so a brick
        that is standing on one row gets pushed flat rather than levered further over.  The roof
        height it aims for is the one a seated brick would have; the studs stand ``stud_h`` above
        that, so the push starts by touching those.
        """
        foot_seated = seat_bottom + spec.height + spec.stud_h
        press_top = tcp_for_foot(place_xy, foot_seated + self.cfg.approach * 0.4, 0.0)
        yield from self._move("align", press_top, speed=self.cfg.travel_speed,
                              grip=self.grip_together)
        if self.failure:
            return
        if self.cfg.nudge:
            yield from self._nudge(place_xy, seat_bottom, spec)
        seat_tcp_z = float(tcp_for_foot(place_xy, foot_seated, 0.0)[2])
        self.press = PressScripted(seat_tcp_z, self.cfg.release_press, grip=self.grip_together)
        self.press.reset(self.driver.feedback(self.grip_together))
        yield from self._run_press()

    def _nudge(self, place_xy, seat_bottom: float, spec):
        """Shove the resting brick sideways with the closed jaws before pressing it."""
        side_z = seat_bottom + spec.height * 0.6
        half = 0.5 * (2 * spec.pitch - spec.clearance)
        for sign in (+1.0, -1.0):
            outside = tcp_for_foot(
                [place_xy[0] + sign * (half + 0.010), place_xy[1]], side_z, 0.0
            )
            inside = tcp_for_foot(
                [place_xy[0] + sign * (half - self.cfg.nudge), place_xy[1]], side_z, 0.0
            )
            yield from self._move("nudge", outside, speed=self.cfg.travel_speed,
                                  grip=self.grip_together)
            yield from self._move("nudge", inside, speed=0.02, grip=self.grip_together)
            yield from self._move("nudge", outside, speed=self.cfg.travel_speed,
                                  grip=self.grip_together)

    def _run_press(self):
        press = self.press
        while not press.done:
            target = press.update(self.driver.feedback(press.grip))
            yield self._tick("press", target.pos, press.grip,
                             c_vert=target.c_vert, c_lat=target.c_lat)
            if self.driver.tick > self.cfg.max_ticks:
                yield from self._fail("ran out of time in the press")
                return
        yield from self._hold("press", self.cfg.settle_time, grip=press.grip)

    # -- the small moves ------------------------------------------------------------------------

    def _move(self, phase: str, target, *, speed: float, grip: float):
        move = MoveTo(target, speed=speed, spin=0.0, tilt=0.0, grip=grip, tol=0.0015,
                      settle_s=self.cfg.settle_time)
        move.reset(self.driver.feedback(grip))
        while not move.done:
            out = move.update(self.driver.feedback(grip))
            yield self._tick(phase, out.pos, grip)
            if self.driver.tick > self.cfg.max_ticks:
                yield from self._fail(f"ran out of time in {phase}")
                return

    def _hold(self, phase: str, seconds: float, *, grip: float | None = None):
        grip = self._grip if grip is None else grip
        pos = self._cmd
        if pos is None:
            pos = self.scene.kin.fk(self.driver.q_goal)[0]
        for _ in range(max(int(round(seconds / CONTROL_DT)), 1)):
            yield self._tick(phase, pos, grip)

    def _ramp_grip(self, phase: str, start: float, end: float, seconds: float):
        n = max(int(round(seconds / CONTROL_DT)), 1)
        pos = self._cmd.copy()
        for k in range(n):
            grip = start + (end - start) * (k + 1) / n
            yield self._tick(phase, pos, grip)

    def _fail(self, why: str):
        self.failure = why
        yield from self._hold("failed", 0.1)

    # -- the tick itself -------------------------------------------------------------------------

    def _tick(self, phase: str, pos, grip: float, *, c_vert: float = 1.0,
              c_lat: float = 1.0) -> Step:
        self.phase = phase
        self._cmd = np.asarray(pos, dtype=float).copy()
        self._grip = float(grip)
        self.driver.step(self._cmd, 0.0, grip, c_vert=c_vert, c_lat=c_lat)
        force = gripper_press_force(self.scene, self.brick)
        self.peak_press_n = max(self.peak_press_n, force)
        self.lifted_by = max(self.lifted_by, float(self.scene.body_pose(self.brick)[0][2]))
        if phase == "press":
            self.press_forces.append(force)
        return Step(
            tick=self.driver.tick,
            t=self.driver.tick * CONTROL_DT,
            phase=phase,
            scene=self.scene,
            tcp_pos=self.scene.tcp_pose()[0],
            cmd_pos=self._cmd,
            grip=float(grip),
            peak_duty=self.driver.peak_duty,
            peak_torque=self.driver.peak_torque,
            press_phase=None if self.press is None else self.press.phase,
        )

    def _brick_is_held(self) -> bool:
        """Is the brick still between the jaws, rather than back on the table?"""
        pos = self.scene.body_pose(self.brick)[0]
        tcp_pos, tcp_rot = self.scene.tcp_pose()
        local = tcp_rot.T @ (pos - tcp_pos)
        return bool(np.linalg.norm(local - self.grasp.pos_in_tcp) < 0.012)


# -------------------------------------------------------------------------------- verifying ---


def gripper_press_force(scene: Scene, brick: str) -> float:
    """Downward force the arm is putting into ``brick`` right now [N].

    Every contact between a geom of the arm and a geom of that brick, summed, world -z component.
    This is the number that has to reach about 9 N to seat a 2x4 at the default clutch, and it is
    measured the same way whether the brick is being gripped or pushed on from above, which is
    what makes the two strategies comparable.
    """

    model, data = scene.model, scene.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, brick)
    arm_bodies = _arm_body_ids(model)
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 == bid and b2 in arm_bodies:
            brick_is_second = False
        elif b2 == bid and b1 in arm_bodies:
            brick_is_second = True
        else:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        # `mj_contactForce` gives the force applied to the *second* geom's body, in the contact
        # frame whose rows are (normal, tangent, tangent); checked against this rig by watching
        # the fixed jaw push a gripped brick in +x while the moving jaw pushes it in -x.
        world = c.frame.reshape(3, 3).T @ force[:3]
        on_brick_z = float(world[2]) if brick_is_second else -float(world[2])
        total += -on_brick_z
    return max(total, 0.0)


def _arm_body_ids(model) -> set[int]:
    """Body ids of everything rigidly part of the arm, found by walking up from the gripper."""
    out = set()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    for i in range(model.nbody):
        j, depth = i, 0
        while j > 0 and depth < 32:
            if j == bid:
                out.add(i)
                break
            j = int(model.body_parentid[j])
            depth += 1
    out.add(bid)
    # The moving jaw is a child of the gripper, so the walk above already has it; the forearm and
    # the rest of the arm are not, and nothing above the wrist can touch a brick on the plate.
    return out


@dataclass
class PlacementCheck:
    """What :func:`verify_placement` found, in millimetres and plain booleans."""

    connected: bool
    seated: bool
    at_target_cell: bool
    min_depth_mm: float
    depth_error_mm: float
    lateral_mm: float
    yaw_deg: float
    tilt_deg: float
    n_engaged: int
    success: bool
    at_target_layer: bool = True
    """Target layer check: bottom face within half a brick height of the layer's height.

    Checked on every layer, the baseplate (layer 0) included: a layer-0 brick that ends up on
    top of another brick fails it, and on a stacked target a brick seated on the plate beside
    the lower brick fails it. The lateral check covers the cell.
    """

    height_error_mm: float = 0.0
    """Bottom face, minus the height the target layer puts it at [mm]."""

    lower_bodies: tuple[str, ...] = ()
    """The bodies whose studs this brick's sockets are on, as the snap model reports them."""


def verify_placement(
    scene: Scene,
    brick: str,
    target: tuple[float, float, int],
    *,
    depth_tolerance: float = 0.0003,
    cell_tolerance: float = 0.4,
) -> PlacementCheck:
    """Read the simulator and say whether the brick really went where it was meant to.

    Nothing here looks at what the sequence believed; it reads the snap model's per-stud
    insertion depths, the brick's pose and the live connection list.  ``depth_tolerance`` is the
    specified 0.3 mm: every stud has to be within that of full depth.  Call it *after* the
    arm has left, so "still seated once nothing is holding it" is part of the answer.

    ``target``'s third element is the stacking layer, and it is scored: a brick sent one layer up
    has to be a brick height above the plate's top face, on the studs of whatever is underneath,
    not on the plate beside it.  Every stud reaching full depth says the joint is closed; the
    height says which joint it is.
    """

    spec = scene.cfg.spec
    depths = np.asarray(scene.snap.stud_depths(brick), dtype=float)
    min_depth = float(np.min(depths)) if depths.size else 0.0
    engaged = int(np.count_nonzero(depths > 0.0))

    pos, quat = scene.body_pose(brick)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    yaw, pitch, _ = euler_zyx(rot.reshape(3, 3))

    cx, cy, layer = target
    cell = cell_world(scene, float(cx), float(cy))
    lateral = float(np.linalg.norm(pos[:2] - cell[:2]))

    handle = scene.bricks[brick]
    period = 0.5 * math.pi if handle.nx == handle.ny else math.pi
    connections = [c for c in scene.snap.connections() if c.upper == brick]
    connected = bool(connections)
    depth_error = max(spec.stud_h - min_depth, 0.0)
    at_cell = lateral <= cell_tolerance * spec.pitch
    seated = depth_error <= depth_tolerance
    # The seat the layer names, and how far off it the brick's bottom face ended up. Half a
    # brick height either way, so a seated brick can only be at one layer and a brick sitting on
    # the wrong one is never inside it.
    height_error = float(pos[2]) - (float(cell[2]) + int(layer) * spec.height)
    at_layer = abs(height_error) <= 0.5 * spec.height
    return PlacementCheck(
        connected=connected,
        seated=seated,
        at_target_cell=at_cell,
        min_depth_mm=1e3 * min_depth,
        depth_error_mm=1e3 * depth_error,
        lateral_mm=1e3 * lateral,
        yaw_deg=math.degrees(math.remainder(yaw, period)),
        tilt_deg=math.degrees(abs(pitch)),
        n_engaged=engaged,
        success=bool(connected and seated and at_cell and at_layer),
        at_target_layer=at_layer,
        height_error_mm=1e3 * height_error,
        lower_bodies=tuple(sorted({c.lower for c in connections})),
    )


# ------------------------------------------------------------------------------- one whole run ---


@dataclass
class RunResult:
    """One complete pick and place, from the supply to the plate and back to the home pose."""

    placement: PlacementCheck
    ticks: int
    seconds: float
    peak_duty: float
    peak_torque_nm: float
    """Largest torque any of the six servos was asked for, the gripper included [N.m].

    Read it with that in mind: over a clean run the *arm* joints peak at about 0.9 N.m against
    the 1.9 N.m limit, and the only servo that ever reaches the limit is the gripper, for the
    handful of ticks in which the jaws close on the brick.
    """

    peak_press_n: float
    held_press_n: float
    """The 90th percentile of the press-phase force: what the arm sustained, not its worst spike."""
    lifted_by_mm: float
    """Highest the brick ever got: under 10 mm means the pick, not the press, is what failed."""
    ik_failures: int
    tripped: bool
    failure: str | None
    warnings: dict


def run_once(
    scene: Scene,
    *,
    brick: str = "supply_brick_0",
    target: tuple[float, float, int] = (3.5, 3.5, 0),
    cfg: PickPlaceConfig = PickPlaceConfig(),
    aim_error=(0.0, 0.0, 0.0, 0.0),
    pick_at=None,
    rng: np.random.Generator | None = None,
    on_step=None,
    correction: JointCorrectionConfig | None = None,
) -> RunResult:
    """Run the sequence to the end and verify the result.  ``on_step(step)`` sees every tick.

    ``correction`` switches on the host-side integral trim, with a fresh integrator for this run;
    ``None`` is the loop this function has always run.
    """

    rng = np.random.default_rng(0) if rng is None else rng
    driver = None
    if correction is not None:
        # Built here rather than inside `PickPlace` so the trim starts clean on every run; the
        # draw order off `rng` is the same either way, so seeds still line up.
        driver = ArmDriver(scene, rng=rng, corrector=JointCorrection(5, correction))
    job = PickPlace(scene, brick=brick, target=target, cfg=cfg, aim_error=aim_error,
                    pick_at=pick_at, rng=rng, driver=driver)
    last = None
    for step in job.run():
        last = step
        if on_step is not None:
            on_step(step)
    placement = verify_placement(scene, brick, target)
    return RunResult(
        placement=placement,
        ticks=0 if last is None else last.tick,
        seconds=0.0 if last is None else last.t,
        peak_duty=job.driver.peak_duty,
        peak_torque_nm=job.driver.peak_torque,
        peak_press_n=job.peak_press_n,
        held_press_n=(
            float(np.percentile(job.press_forces, 90)) if job.press_forces else 0.0
        ),
        lifted_by_mm=1e3 * job.lifted_by,
        ik_failures=job.driver.ik_failures,
        tripped=job.driver.tripped,
        failure=job.failure,
        warnings=scene.warnings(),
    )


# ------------------------------------------------------------------------------ the bench ---


SUPPLY_COLOURS = ("blue", "yellow", "green", "orange", "white", "red")
"""The colour cycle spare bricks are drawn from, one per supply slot."""


def make_scene(
    *,
    pad_friction: float = 0.5,
    brick: tuple[int, int] = (2, 4),
    supply_bricks: int = 2,
    snap: SnapParams = SnapParams(),
    cameras: tuple[str, ...] = (),
) -> Scene:
    """The world this sequence runs in: the arm, an empty plate, and loose spare bricks.

    ``pad_friction`` is the sliding friction between the bare jaw faces and the brick, which is
    the number the whole friction grasp turns on and the one nobody has measured on hardware yet.
    ``supply_bricks`` is the number of loose spare bricks on the table (the real hardware has no
    tray; it stays off, :attr:`~brick_builder.scene.compose.SceneConfig.tray`'s own default).
    """

    supply = tuple(
        (brick[0], brick[1], SUPPLY_COLOURS[k % len(SUPPLY_COLOURS)])
        for k in range(supply_bricks)
    )
    return build_scene(SceneConfig(
        held=None,
        placed=[],
        supply=supply,
        snap=snap,
        cameras=cameras,
        arm=ArmConfig(pad_friction=pad_friction),
    ))


def bench(
    *,
    strategy: PressStrategy | None = None,
    pad_friction: float = 0.5,
    place_error: float = 0.0,
    trials: int = 50,
    seed0: int = 0,
    dr_level: float = 0.0,
    brick: tuple[int, int] = (2, 4),
    supply_jitter: float = 0.002,
    cfg: PickPlaceConfig | None = None,
    scene: Scene | None = None,
    correction: JointCorrectionConfig | None = None,
) -> dict:
    """Run ``trials`` randomized pick-and-places and report what fraction of them worked.

    One world is built and reset between trials, which is what makes a few hundred runs
    affordable.  The arm always aims at the supply's *taught* slot, never at where the brick
    actually is, so the jitter below is an error it has to survive rather than one it is told
    about.  Each trial redraws

    - where the brick really is in the supply (``supply_jitter`` metres and a couple of degrees of
      yaw, uniform), which the arm does not know about;
    - ``place_error`` metres of lateral error in where the arm believes the target cell is, at a
      uniformly random direction - a magnitude, not a box, so "2 mm" means 2 mm;
    - this trial's servo and bus parameters at ``dr_level`` (0 is the exactly nominal arm, 1 is
      the full published range of the dynamics and command groups).

    The contact group is deliberately *not* randomized: it owns the jaw friction, which this
    bench is sweeping by hand.

    ``correction`` gives every trial its own host-side integral trim on the reported joint
    angles; ``None``, the default, is the bare proportional servo.
    """

    from brick_builder.scene import randomize

    cfg = cfg or PickPlaceConfig()
    if strategy is not None:
        cfg = PickPlaceConfig(**{**cfg.__dict__, "press_strategy": strategy})
    strategy = cfg.press_strategy
    own_scene = scene is None
    scene = scene if scene is not None else make_scene(pad_friction=pad_friction, brick=brick)
    rows: list[dict] = []
    for k in range(trials):
        rng = np.random.default_rng(seed0 + k)
        if dr_level > 0.0:
            randomize.apply(scene, randomize.sample(
                rng, level=dr_level, groups=("dynamics", "command")
            ))
        reset_scene(scene, rng)
        adr = scene.free_joint_adr("supply_brick_0")
        slot = scene.data.qpos[adr:adr + 3].copy()      # the taught slot, before the jitter
        scene.data.qpos[adr:adr + 2] += rng.uniform(-supply_jitter, supply_jitter, 2)
        yaw = float(rng.uniform(-math.radians(2.0), math.radians(2.0)))
        scene.data.qpos[adr + 3:adr + 7] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        mujoco.mj_forward(scene.model, scene.data)
        scene.snap.bind(scene.model, scene.data)

        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        aim = (0.0, 0.0, place_error * math.cos(angle), place_error * math.sin(angle))
        result = run_once(scene, brick="supply_brick_0", cfg=cfg, aim_error=aim, pick_at=slot,
                          rng=rng, correction=correction)
        rows.append({
            "trial": k,
            "success": result.placement.success,
            "connected": result.placement.connected,
            "at_cell": result.placement.at_target_cell,
            "depth_error_mm": result.placement.depth_error_mm,
            "lateral_mm": result.placement.lateral_mm,
            "tilt_deg": result.placement.tilt_deg,
            "n_engaged": result.placement.n_engaged,
            "peak_press_n": result.peak_press_n,
            "held_press_n": result.held_press_n,
            "peak_torque_nm": result.peak_torque_nm,
            "peak_duty": result.peak_duty,
            "lifted_by_mm": result.lifted_by_mm,
            "seconds": result.seconds,
            "tripped": result.tripped,
            "failure": result.failure or _diagnose(result),
            "warnings": dict(result.warnings),
        })
    if own_scene:
        del scene
    ok = [r for r in rows if r["success"]]
    return {
        "strategy": strategy,
        "pad_friction": pad_friction,
        "place_error_mm": 1e3 * place_error,
        "dr_level": dr_level,
        "brick": list(brick),
        "trials": len(rows),
        "successes": len(ok),
        "success_rate": len(ok) / max(len(rows), 1),
        "median_depth_error_mm": float(np.median([r["depth_error_mm"] for r in rows])),
        "median_seconds": float(np.median([r["seconds"] for r in rows])),
        "max_peak_press_n": float(np.max([r["peak_press_n"] for r in rows])),
        "median_held_press_n": float(np.median([r["held_press_n"] for r in rows])),
        "max_torque_nm": float(np.max([r["peak_torque_nm"] for r in rows])),
        "rows": rows,
    }


def _diagnose(result: RunResult) -> str:
    """A short name for why a run is not a success, read off the verified placement."""
    p = result.placement
    if p.success:
        return "ok"
    if result.tripped:
        return "overload trip"
    if result.lifted_by_mm < 10.0:
        return "pick failed"
    if not p.connected and p.n_engaged == 0:
        return "never engaged"
    if not p.at_target_cell:
        return "wrong cell"
    if p.depth_error_mm > 0.3:
        return f"partial seat ({p.depth_error_mm:.2f} mm short)"
    return "not connected"
