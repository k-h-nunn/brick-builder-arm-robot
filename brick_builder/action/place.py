"""Tracking an ideal sequence: feed-forward action plus one-step feedback, at 10 Hz.

Two things run here.

`Script` is the task, written out once as tool targets: approach, grasp, lift, transport, align,
press, release, retract, verify. It is executed **in the nominal world** by `record_ideal`, and
what that execution saw and did becomes the ideal sequence (see `sequence.py`). Nothing in the
script is corrected, because in the nominal world there is nothing to correct.

`TrackingEpisode` runs the real world against that recording. At vision tick `k` it issues the
recorded action `a*_k` and adds one step of feedback on the difference between the target
scene state `S*_k` and the scene state the camera actually reports:

    u = a*_k + lambda * J_k^+ (features(S*_k) - features(S_k))

`J_k` is measured in the twin, once per step, in two parts: how the reference blobs move when the
tool moves, and how the held brick's blob moves when the brick sits off-centre in the jaws. The
second part is what puts the *brick* on the cell rather than the tool.

The progress index `k` advances only when the world has caught up: while the actual state is
inside a tolerance tube around `S*_k`, where the tube widens in proportion to how fast the
reference is moving. It never runs ahead. Holding for longer than a bound is a failure, and so is
finishing outside tolerance; neither is a timer.

Everything the controller reads comes from the adapter: slot states and what the servo bus
reports. The baseplate's real pose, the brick's real pose and the contact state are not readable
from here and the guard test in `tests/action` checks it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np

from ..control.pick_place import (
    CONTROL_DT, PickPlaceConfig, cell_world, grasp_pose, jaw_angle_for_clearance,
    tcp_for_brick, tcp_for_foot,
)
from ..control.primitives import MoveTo, PressScripted
from ..perception.scene_model import SceneState
from ..robot.patch_so101 import gripper_angle_for_width
from ..scene.compose import HeldBrick, Scene, gripper_command
from .adapter import VISION_EVERY, WorldAdapter
from .sequence import Command, IdealSequence, ReferenceFrame
from .servo import VisualServo
from .twin import BlindStep, DigitalTwin
from .worlds import (
    NOMINAL_PLATE_XY, SPARE_BODY, SUPPLY_BODY, TARGET_CELL, WANTED, nominal_cfg,
    supply_nominal_xy,
)

STEP_TITLES = (
    "Approach", "Grasp", "Lift", "Transport", "Align",
    "Press the brick down", "Release", "Retract", "Verify",
)
"""Every title a step may carry on screen."""


@dataclass(frozen=True)
class PlaceConfig:
    """Tracking gains and tolerances. Distances in metres unless a name says millimetres."""

    lam: float = 0.15
    """Feedback gain on the one-step inverse.

    Small on purpose. The correction is an integrator and the arm takes four or five control
    ticks to arrive, so the loop is commanding into a delay; at 0.4 it overshot and settled into
    a limit cycle that swung ten millimetres either way and never let the progress index move.
    """

    tol_mm: float = 0.4
    """Tracking tolerance at the end of the sequence, and the floor of the tube everywhere."""

    advance_mm: float = 1.0
    """Floor of the tolerance tube for *advancing*. Wider than `tol_mm` because the tracking
    error has a floor of its own: the twin renders a pose the real arm only reaches to within
    the servo's own few tenths of a millimetre, so demanding 0.4 mm before every single frame
    may advance is demanding to be inside the noise. Finishing still needs `tol_mm`."""

    tube_scale: float = 1.6
    """How much of one reference step's own travel the arm may be behind before the index holds."""

    jacobian_every: int = 8
    """Reference frames between Jacobian estimates, and always one at a step boundary. The
    camera geometry changes slowly along a step but a Jacobian measured at the start of a
    200 mm transport is useless by the end of it."""

    freeze_after: str = "Align"
    """Last step whose feedback updates the correction.

    What the correction estimates is static: where the baseplate really is, and where the brick
    really sits in the jaws. Neither changes during the press. After that step every remaining
    move either leans on something -- where the arm is deliberately not where it is commanded --
    or swings the camera through poses where the blobs it lines up on are half out of frame, and
    an estimate updated there gets worse, not better. So the sequence keeps being tracked and
    the error keeps being reported, and the correction stops moving.
    """

    dwell_only: bool | None = None
    """Estimate only where the sequence dwells. `None` means "when picking".

    The pick needs it: estimating while the arm crosses to the supply area integrated tracking
    lag into the correction and walked a brick 116 mm. The placement task does NOT want it --
    it was measured 10/10 estimating throughout, and gating it to dwells alone changed that to
    0/3 with whole-cell errors. Two different tasks, two different answers, and the flag says so
    rather than one of them quietly paying for the other."""

    use_yaw: bool = True
    """Whether the loop corrects tool yaw at all. A knob for isolating it, and an honest switch:
    a yaw estimate that is wrong is worse than none, because a square brick does not care how it
    is turned but the studs under it do."""

    dwell_travel_mm: float = 0.2
    """Tool-target motion per vision tick below which the target counts as standing still."""

    dwell_speed: float = 0.05
    """Joint speed below which the bus says the arm has stopped [rad/s]."""

    min_gain: float = 1.2
    """Smallest singular value the step's Jacobian must have, in units of feature per metre,
    before its inverse is believed. A camera that can barely see the things it is lining up on
    produces a Jacobian whose inverse turns a pixel of noise into centimetres of command; below
    this the tick is pure feed-forward and the index advances anyway, because a loop cannot
    track what it cannot see."""

    match_window: int = 8
    """How many frames back the actual state may be matched to. The reference is compared
    against the frame the world is really at, not the one the clock says it should be at, so a
    lagging arm does not read as a displaced baseplate and get 'corrected' for it."""

    max_hold: int = 70
    """Vision ticks the index may be held before the run is called stalled."""

    settle_ticks: int = 3
    """Consecutive ticks inside tolerance at the last frame that end the run."""

    max_correction: float = 0.015
    """Ceiling on the accumulated correction. Past this the loop is not tracking the plate."""

    closed_loop: bool = True
    """Off: the recorded actions are replayed and nothing is corrected. The control condition."""

    inverse: str = "analytic"
    """Which one-step inverse turns the scene-state difference into a tool correction.

    `"analytic"` is `-lambda * J^+ (s - s*)` with `J` measured in the twin. `"learned"` loads
    `action/weights/inverse.npz` and calls that instead, behind the same gates, the same
    saturation and the same integrator; everything else -- the feed-forward, the progress index,
    the dwells, the end conditions -- is untouched, so the two are measurable against each other
    on the same seeds. The default is the analytic one until the learned one matches it."""

    predictor: str = "analytic"
    """Which one-step prediction the reported prediction error is measured against.

    `"analytic"` is the loop's own rule -- the residual will change by exactly the correction just
    commanded. `"learned"` loads `action/weights/predictor.npz` and reads the predicted residual
    back through the same servo (`models/predictor.py`). Reported only: nothing on the control
    path reads the prediction, so the two are measurable on the same seeds with the same run."""

    hover: float = 0.020
    low: float = 0.006
    align_hold: float = 4.0
    """Seconds the script stands still at the align pose. In the nominal world nothing happens;
    in the real one this is where the feedback does its work."""

    pick: bool = False
    pick_approach: float = 0.055
    pick_settle: float = 0.6
    lift_hold: float = 1.6
    """Seconds the script stands still at the top of the lift [s].

    Where the brick's offset in the jaws is read: the arm is at rest, the brick is clear of the
    table, and the only thing under it is bare table rather than the baseplate and the bricks
    already on it. The reading is taken here, averaged over the dwell, and then held -- it is
    one constant, and the align dwell's job is the baseplate, not this again."""
    pick_clearance: float = 0.016
    """How much wider than the brick the jaws open before the grasp [m], total across the gap.

    A control parameter, not a change of hardware: the same bare jaws, told to open further.
    `press.open_clearance` is 4 mm, two on each side, which is what a task that already knows
    where the brick is needs. This one does not: it is aiming at a blob whose centre has a
    measured floor near 1.7 mm with one small reference in view, so two millimetres of clearance
    is inside its own noise. Measured at 4 mm: a jaw lands on the brick's top face, the arm
    stalls ten millimetres above the commanded grasp height and closes on the brick's top rim.
    At 10 mm the grasp reaches its commanded height in a nominal world but still misses on
    displaced ones; at 16 mm, eight on each side, four of five seeds that had missed twice each
    reached it, and three of them landed on the target cell."""
    pick_low: float = 0.020
    """Height above the brick of the second, low dwell before the grasp [m].

    The approach is two dwells, not one. The high one at `pick_approach` finds the brick from far
    enough away that it is certainly in view; this one is as close above it as the view allows,
    and the estimate that decides the grasp is the one made here. Measured in the twin with the
    jaws open to `pick_clearance`, the brick's blob stays whole all the way down -- border
    fraction 0.000 at every height -- while its mass grows from 0.0074 at the high hover to
    0.0118 here and the weaker singular value of the step's inverse from 4.81 to 5.83 per metre,
    which is a floor of 0.69 mm against 0.83 mm for the same feature noise. The rest of what it
    buys is that the correction is frozen 20 mm above the brick rather than 55."""

    pick_low_settle: float = 1.0
    """Seconds the script stands still at the low dwell [s]."""

    pick_freeze_below: float = 0.034
    """Tool height above the table below which the grasp's correction stops updating [m].

    Just below the low dwell's own tool height, so the last thing the correction sees is that
    dwell and the final descent is a straight drop on a frozen estimate. It also has to stay
    above the height at which the open jaws would straddle the brick and cut its blob in two:
    the centre of a cut blob is the centre of whatever slice is still visible, and acting on
    that shoves the brick sideways out of the gripper -- measured at 4 mm of clearance, where
    the brick ended 232 mm away."""
    stall_load: float = 0.45
    """Gripper load, as a fraction of the servo's limit, that counts as closed on the brick."""

    grasp_yaw_deg: float = 1.5
    """Yaw error the hover estimate must be inside before the jaws are allowed down [deg].

    The open jaws clear the brick by half of `press.open_clearance` on each side -- two
    millimetres -- and a square brick 31.8 mm across turns that much through one and a half
    degrees. A descent started on a yaw estimate that has not settled puts a jaw on top of the
    brick instead of beside it: the arm stalls ten millimetres high, the jaws close on the
    brick's top rim, and the brick slides out of them during the transport."""

    grasp_ticks: int = 3
    """Consecutive vision ticks inside both tolerances before the descent starts."""

    grasp_wait: int = 45
    """Vision ticks the descent waits for that, and no more.

    A brick whose blob barely carries an orientation never settles, and a loop that waits for it
    trips the stall bound and ends the run with the brick still on the table. Past this the
    descent goes ahead on the estimate it has and `grasp_miss` catches what that costs. Wide
    enough for the correction to cross the offsets the task puts there: 15 mm at the 1.2 mm a
    tick the integrator is allowed, plus the arm's own settling."""

    grasp_miss: float = 0.003
    """How far above the commanded grasp height the tool may stop and still count as a grasp [m].

    Read from the bus: forward kinematics on the reported joint angles. Above this the jaws
    closed somewhere other than where they were sent, so the grasp is opened and made again from
    the hover. Once -- a second miss is a failure, not a third attempt."""

    travel_speed: float = 0.10
    descend_speed: float = 0.035
    press: PickPlaceConfig = field(default_factory=PickPlaceConfig)


@dataclass
class TickRecord:
    """One 20 ms control tick, for the video and the log."""

    tick: int
    t: float
    step: str
    step_index: int
    cmd_pos: np.ndarray
    spin: float
    grip: float
    correction: np.ndarray
    k: int = 0
    vision: bool = False
    err_mm: float = float("nan")
    """Scene-state error: the tracking error against `S*_k`, in millimetres of tool motion."""
    tol_mm: float = float("nan")
    predict_mm: float = float("nan")
    """Prediction error: observed minus predicted next scene state, same units."""
    press_n: float = float("nan")
    depth_mm: float = float("nan")
    miou: float = float("nan")
    state: SceneState | None = None
    rgb: np.ndarray | None = None
    target: SceneState | None = None
    """`S*_{k+1}` -- the scene state the controller is asking the world for next."""
    target_label: object = None
    """Whatever the viewer asked to keep for that reference frame (see `sequence.py`)."""
    refs_seen: int = 0


@dataclass
class StepReport:
    name: str
    vision_ticks: int
    final_err_mm: float
    converged: bool


def reference_bodies(twin_scene, state: SceneState, *, exclude: str | None = None) -> dict:
    """Which objects are visible in a reference frame, as `title -> body name`.

    Read off the twin, which knows which of its own slots is which object, and only from the
    frame the twin actually recorded -- so a step over the supply area lines up on loose bricks
    and a step over the plate lines up on the plate. `exclude` drops the brick in the hand: it
    rides with the camera, so lining up on it would be chasing your own hand.
    """
    titles = {
        twin_scene.plate.name: "baseplate",
        SUPPLY_BODY: "brick to pick up",
        SPARE_BODY: "spare brick",
    }
    for i, name in enumerate(twin_scene.placed_names):
        titles.setdefault(name, f"placed brick {i}")
    ids = {}
    for name in list(titles) + list(twin_scene.bricks):
        iid = twin_scene.instance_ids.get(name)
        if iid is not None:
            ids[int(iid)] = name

    out: dict[str, str] = {}
    for k in range(state.K):
        if state.is_empty(k) or state.mass(k) > 0.5 or state.mass(k) < 0.0012:
            continue
        if state.border_fraction(k) > 0.25:
            continue
        name = ids.get(int(state.slot_ids[k]))
        if name is None or name == exclude:
            continue
        out[titles.get(name, name)] = name
    return out


# ================================================================================ the script ===


class Script:
    """The task as tool targets. Executed in the nominal world to make the ideal sequence."""

    def __init__(self, world: WorldAdapter, geometry: "Geometry", cfg: PlaceConfig) -> None:
        self.world = world
        self.g = geometry
        self.cfg = cfg
        self.step = "Approach" if cfg.pick else "Transport"
        self.step_index = 0
        self._cmd = world.feedback(geometry.grip_closed).tcp_pos.copy()
        self._spin = 0.0
        self._grip = geometry.grip_open if cfg.pick else geometry.grip_closed
        self.stall_at: float | None = None

    # -- primitives ---------------------------------------------------------------------------

    def _begin(self, title: str) -> None:
        if title != self.step:
            self.step_index += 1
        self.step = title

    def _emit(self, *, c_vert: float = 1.0, c_lat: float = 1.0) -> Command:
        return Command(pos=np.asarray(self._cmd, dtype=float).copy(), spin=float(self._spin),
                       grip=float(self._grip), c_vert=c_vert, c_lat=c_lat)

    def _move(self, title: str, target, *, speed: float, grip: float | None = None,
              spin: float | None = None):
        self._begin(title)
        if spin is not None:
            self._spin = float(spin)
        if grip is not None:
            self._grip = float(grip)
        move = MoveTo(np.asarray(target, dtype=float), speed=speed, spin=self._spin, tilt=0.0,
                      grip=self._grip, tol=0.0015, settle_s=self.cfg.press.settle_time)
        move.reset(self.world.feedback(self._grip))
        guard = 0
        while not move.done and guard < 1500:
            out = move.update(self.world.feedback(self._grip))
            self._cmd = np.asarray(out.pos, dtype=float)
            yield self._emit()
            guard += 1

    def _hold(self, title: str, seconds: float, *, grip: float | None = None):
        self._begin(title)
        if grip is not None:
            self._grip = float(grip)
        for _ in range(max(int(round(seconds / CONTROL_DT)), 1)):
            yield self._emit()

    def _ramp_grip(self, title: str, start: float, end: float, seconds: float):
        self._begin(title)
        n = max(int(round(seconds / CONTROL_DT)), 1)
        for i in range(n):
            self._grip = start + (end - start) * (i + 1) / n
            yield self._emit()

    def _close_on_the_brick(self, title: str):
        """Close until the bus says the jaws have stopped on something.

        Mechanical state, not a stopwatch: the gripper's own reported load is what says the jaws
        have met the brick rather than each other.
        """
        self._begin(title)
        start, end = self.g.grip_open, self.g.grip_closed
        n = max(int(round(self.cfg.press.close_time / CONTROL_DT)), 1)
        for i in range(n):
            self._grip = start + (end - start) * (i + 1) / n
            yield self._emit()
            if self.stall_at is None and abs(float(self.world.joint_load[-1])) >= self.cfg.stall_load:
                self.stall_at = self.world.t
        yield from self._hold(title, self.cfg.press.settle_time)

    def _press(self, title: str, seat_z: float, press_cfg, grip: float):
        self._begin(title)
        self._grip = float(grip)
        press = PressScripted(float(seat_z), press_cfg, grip=float(grip))
        press.reset(self.world.feedback(grip))
        n = 0
        while not press.done and n < 900:
            target = press.update(self.world.feedback(press.grip))
            self._cmd = np.asarray(target.pos, dtype=float)
            yield self._emit(c_vert=target.c_vert, c_lat=target.c_lat)
            n += 1
        yield from self._hold(title, self.cfg.press.settle_time)

    # -- the task -----------------------------------------------------------------------------

    def run(self) -> Iterator[Command]:
        g, cfg = self.g, self.cfg

        if cfg.pick:
            yield from self._hold("Approach", 0.2, grip=g.grip_open)
            yield from self._move("Approach", g.pick_tcp(cfg.pick_approach),
                                  speed=cfg.travel_speed, grip=g.grip_open, spin=0.0)
            yield from self._hold("Approach", cfg.pick_settle)
            yield from self._move("Approach", g.pick_tcp(cfg.pick_low),
                                  speed=cfg.descend_speed)
            yield from self._hold("Approach", cfg.pick_low_settle)
            yield from self._move("Approach", g.pick_tcp(0.0), speed=cfg.descend_speed)
            yield from self._close_on_the_brick("Grasp")
            yield from self._move("Lift", np.array([self._cmd[0], self._cmd[1], g.carry_z]),
                                  speed=cfg.descend_speed, grip=g.grip_closed)
            yield from self._hold("Lift", cfg.lift_hold)
        else:
            yield from self._hold("Transport", 0.2, grip=g.grip_closed)

        yield from self._move("Transport", g.place_tcp(cfg.hover), speed=cfg.travel_speed,
                              grip=g.grip_closed, spin=0.0)
        yield from self._hold("Align", cfg.align_hold)
        yield from self._move("Press the brick down", g.place_tcp(cfg.low),
                              speed=cfg.descend_speed)
        yield from self._press("Press the brick down", g.place_tcp(0.0)[2], cfg.press.press,
                               g.grip_closed)
        yield from self._ramp_grip("Release", g.grip_closed, g.grip_open, cfg.press.open_time)

        spec = g.spec
        clear_z = g.seat_bottom + 2.0 * spec.height + 0.012
        yield from self._move("Release", np.array([self._cmd[0], self._cmd[1], clear_z]),
                              speed=cfg.descend_speed, grip=g.grip_open)
        yield from self._ramp_grip("Press the brick down", g.grip_open, g.grip_together,
                                   cfg.press.open_time)
        foot_seated = g.seat_bottom + spec.height + spec.stud_h
        top = tcp_for_foot(g.cell_nominal[:2], foot_seated + cfg.press.approach * 0.4, self._spin)
        yield from self._move("Press the brick down", top, speed=cfg.travel_speed,
                              grip=g.grip_together)
        seat_z = float(tcp_for_foot(g.cell_nominal[:2], foot_seated, self._spin)[2])
        yield from self._press("Press the brick down", seat_z,
                               replace(cfg.press.release_press, press_depth=0.018),
                               g.grip_together)

        yield from self._move("Retract", np.array([self._cmd[0], self._cmd[1], g.carry_z]),
                              speed=cfg.descend_speed, grip=g.grip_open)
        yield from self._move("Verify", g.view_tcp, speed=cfg.travel_speed, grip=g.grip_open)
        yield from self._hold("Verify", 1.0)


@dataclass
class Geometry:
    """Nominal task knowledge: where things are supposed to be and how big the tool is.

    All of it comes from the twin -- the nominal world -- and none of it from the world the arm
    is actually standing in.
    """

    twin_scene: Scene
    target: tuple[int, int]
    held: HeldBrick
    cfg: PlaceConfig

    def __post_init__(self) -> None:
        spec = self.twin_scene.cfg.spec
        self.spec = spec
        self.grasp = (self.twin_scene.grasp if self.twin_scene.grasp is not None
                      else grasp_pose(self.held.nx, self.held.ny, spec))
        cx = self.target[0] + (self.held.nx - 1) / 2.0
        cy = self.target[1] + (self.held.ny - 1) / 2.0
        self.cell_nominal = cell_world(self.twin_scene, cx, cy)
        self.seat_bottom = float(self.cell_nominal[2])
        self.carry_z = self.seat_bottom + 0.045 + float(self.grasp.pos_in_tcp[2])
        self.view_tcp = np.array([NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1],
                                  self.seat_bottom + 0.058])
        self.grip_closed = (gripper_command(self.twin_scene)
                            if self.twin_scene.grasp is not None
                            else jaw_angle_for_clearance(self.grasp.contact_point,
                                                         -self.cfg.press.squeeze))
        self.open_clearance = (self.cfg.pick_clearance if self.cfg.pick
                               else self.cfg.press.open_clearance)
        self.grip_open = jaw_angle_for_clearance(self.grasp.contact_point, self.open_clearance)
        self.grip_together = gripper_angle_for_width(0.002)
        self.supply_nominal = np.array([0.0, 0.0, 0.0])

    def place_tcp(self, height: float) -> np.ndarray:
        return tcp_for_brick(
            [self.cell_nominal[0], self.cell_nominal[1], self.seat_bottom + height],
            0.0, self.grasp,
        )

    def pick_tcp(self, height: float) -> np.ndarray:
        """Over the loose brick, jaws open, offered the middle of the gap.

        Arriving flush against the fixed jaw is a bad place to arrive -- the fixed jaw cannot
        move, so a brick a couple of millimetres towards it ends up outside the gripper. Starting
        half the open clearance back lets the moving jaw shove it flush as it closes.
        """
        centre = np.array([0.5 * self.open_clearance, 0.0, 0.0])
        rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        base = tcp_for_brick(
            [self.supply_nominal[0], self.supply_nominal[1], height], 0.0, self.grasp)
        return base - rot @ centre


# =============================================================================== the recording ===


def record_ideal(cfg: PlaceConfig, *, seed: int = 0, target=TARGET_CELL,
                 prepare=None, build_decode: bool = False,
                 source=None, label_hook=None,
                 placed=None) -> tuple[IdealSequence, WorldAdapter]:
    """Run the task in the nominal world and keep what it saw and did.

    Returns the sequence and the adapter it was recorded through; the caller closes the adapter.

    `placed` overrides what is already on the plate in the nominal world; `None` is the two
    reference bricks the single-brick tasks use. A build whose second brick goes down beside
    the first has to record against a plate that already carries it, or the run's own view of
    the brick it just placed matches nothing in the recording.
    """
    from .worlds import build_nominal_world

    held = HeldBrick(nx=WANTED[0], ny=WANTED[1], colour=WANTED[2])
    scene = build_nominal_world(cfg.pick, held=None if cfg.pick else held, prepare=prepare,
                                placed=placed)
    # `source` records the reference sequence through the *same* scene model that will drive
    # the run. A model with a systematic bias then has that bias in both the target state
    # and the observed one, where it cancels instead of steering.
    world = WorldAdapter(scene, rng=np.random.default_rng(seed), build_decode=build_decode,
                         source=source)
    geometry = Geometry(scene, (int(target[0]), int(target[1])), held, cfg)
    if cfg.pick:
        geometry.supply_nominal = supply_nominal_xy(scene)

    script = Script(world, geometry, cfg)
    frames: list[ReferenceFrame] = []
    pending: list[Command] = []
    bodies: dict[int, dict] = {}
    held_by_step: dict[int, str | None] = {}
    held_name = scene.held_name if not cfg.pick else SUPPLY_BODY

    def snapshot(k: int) -> ReferenceFrame:
        state = world.observe()
        return ReferenceFrame(k=k, t=world.t, step=script.step, step_index=script.step_index,
                              state=state,
                              label=None if label_hook is None else label_hook(scene))

    frame = snapshot(0)
    frames.append(frame)
    for command in script.run():
        world.command(command.pos, command.spin, command.grip,
                      c_vert=command.c_vert, c_lat=command.c_lat)
        pending.append(command)
        if len(pending) >= VISION_EVERY:
            frames[-1].commands = pending
            nxt = snapshot(len(frames))
            frames[-1].action = np.array([
                pending[-1].pos[0] - pending[0].pos[0],
                pending[-1].pos[1] - pending[0].pos[1],
                pending[-1].pos[2] - pending[0].pos[2],
                pending[-1].spin - pending[0].spin,
            ])
            frames.append(nxt)
            pending = []
    if pending:
        frames[-1].commands = pending
        frames[-1].action = np.array([
            pending[-1].pos[0] - pending[0].pos[0],
            pending[-1].pos[1] - pending[0].pos[1],
            pending[-1].pos[2] - pending[0].pos[2],
            pending[-1].spin - pending[0].spin,
        ])

    frames = [f for f in frames if f.commands]
    for i, f in enumerate(frames):
        f.k = i
        # The brick is excluded only while it is IN the hand. Once it is let go it is part of
        # the build, and whether it sits where the target puts it relative to the
        # baseplate is exactly what the rest of the sequence is about.
        f.bodies = reference_bodies(
            scene, f.state, exclude=held_name if _in_hand(f.step) else None)
        if cfg.pick and f.step in ("Approach", "Grasp"):
            # Reaching for the brick, the brick is the only thing worth lining up on. Everything
            # else in view -- the spare beside it, the baseplate across the table -- is a
            # landmark that has not moved, and a least-squares fit over the lot answers "where
            # is the arm" when the question is "where is that brick". Measured with the spare in
            # the set: the tool corrected 3.6 mm of a 13.4 mm offset and called itself converged
            # at 1.74 mm, and the yaw, read off the line joining the two blobs, moved 0.26 deg
            # against a brick turned 15.8.
            f.bodies = {t: b for t, b in f.bodies.items() if b == SUPPLY_BODY}
        bodies.setdefault(f.step_index, f.bodies)
        held_by_step.setdefault(f.step_index, held_name if _in_hand(f.step) else None)
    return IdealSequence(frames=frames, bodies_by_step=bodies, held_body=held_name,
                         held_by_step=held_by_step), world


def _in_hand(step: str) -> bool:
    """Steps during which the brick is between the jaws and moving with the camera."""
    return step in ("Lift", "Transport", "Align", "Press the brick down")


# ============================================================================== the real run ===


class TrackingEpisode:
    """Runs the real world against an ideal sequence, one vision tick at a time."""

    def __init__(self, scene: Scene, ideal: IdealSequence, *, cfg: PlaceConfig = PlaceConfig(),
                 seed: int = 0, target=TARGET_CELL, twin_prepare=None,
                 build_decode: bool = False, source=None, inverse=None, predictor=None,
                 observer=None, plan=None, controller=None) -> None:
        self.cfg = cfg
        self.ideal = ideal
        self.seed = seed
        self.plan = plan
        """Steps measured in the twin ahead of the run instead of during it (`hil/plan.py`).
        The loop is unchanged; only where the numbers come from is."""
        self.controller = controller
        """Runs the tick logic somewhere else -- the C library, a native process or the board
        (`hil/episode.py`). `None` is this class's own loop."""
        # `inverse` overrides `cfg.inverse`; `None` means "whatever the config asks for". The
        # learned one is a few thousand multiply-accumulates on slot-row differences and is
        # loaded here, once, so the tick itself only runs the forward pass.
        if inverse is None and cfg.inverse == "learned":
            from .models.inverse import LearnedInverse

            inverse = LearnedInverse.load()
        self.inverse = inverse
        if predictor is None and cfg.predictor == "learned":
            from .models.predictor import LearnedPredictor

            predictor = LearnedPredictor.load()
        self.predictor = predictor
        self.observer = observer
        """Called once per vision tick with everything a tuple log needs. Off the control path:
        the loop's own behaviour does not depend on it (see `action/learn/collect.py`)."""
        self.world = WorldAdapter(scene, rng=np.random.default_rng(seed),
                                  build_decode=build_decode, source=source)
        held = HeldBrick(nx=WANTED[0], ny=WANTED[1], colour=WANTED[2])
        from .worlds import DISTRACTOR
        # Two twins for a pick-and-place, because the nominal world genuinely has two shapes:
        # before the grasp the brick lies on the table and the hand is empty, after it the brick
        # is in the hand and gone from the table. One twin cannot answer both, and the in-hand
        # Jacobian has nothing to measure in the first.
        self.twins = {
            False: DigitalTwin(nominal_cfg(held=None,
                                           supply=(WANTED, DISTRACTOR) if cfg.pick else ()),
                               seed=seed, prepare=twin_prepare),
        }
        if cfg.pick:
            # The wanted brick's row position is left empty rather than dropped, so the spare
            # keeps the body name and the place it has in the world the sequence was recorded
            # in. `_prepare_step` matches the recording's reference bodies to the twin's by
            # name, and renumbering the row silently emptied the reference set.
            self.twins[True] = DigitalTwin(nominal_cfg(held=held, supply=(None, DISTRACTOR)),
                                           seed=seed, prepare=twin_prepare)
        else:
            self.twins[True] = DigitalTwin(nominal_cfg(held=held), seed=seed,
                                           prepare=twin_prepare)
        self.twin = self.twins[not cfg.pick]
        self.reports: list[StepReport] = []
        self.timed_out: list[str] = []
        self.failure: str | None = None
        self.correction = np.zeros(3)
        self.k = 0
        self.holds = 0
        self.blind_holds = 0
        """Ticks of the current hold where nothing was measured at all."""
        self.max_hold_seen = 0
        self.matched = 0
        self.lag = 0
        self._in_hand_reset = False
        self.dwelling = False
        self.dwell_ticks = 0
        self.frozen = False
        self.pick_frozen = False
        self._descending = False
        self.frozen_correction = np.zeros(3)
        self.regrasps = 0
        """Times the grasp was opened and made again because the tool stopped short."""
        self.yaw_err = float("nan")
        self._grasp_settled = 0
        self._gating_grasp = False
        self._grasp_checked = False
        self._regrasping = False
        # Where the descent onto the brick begins, and how low it is told to go. Both are read
        # off the recording, which is nominal task knowledge, not the world.
        self._descent_k = next(
            (i for i, f in enumerate(ideal.frames)
             if f.commands and f.commands[0].pos[2] < cfg.pick_freeze_below), None)
        self._grasp_k = max(
            (i for i, f in enumerate(ideal.frames) if f.step == "Grasp"), default=-1)
        self._grasp_z = min(
            (c.pos[2] for f in ideal.frames if f.step in ("Approach", "Grasp")
             for c in f.commands), default=0.0)
        self._freeze_index = max(
            (f.step_index for f in ideal.frames if f.step == cfg.freeze_after), default=10 ** 6)
        self._step_index = -1
        self._jacobian_at = -10 ** 6
        self.gain = 0.0
        self.trusted = False
        self.blind_frames = 0
        self._servo: VisualServo | None = None
        self._step = None
        self._predicted: np.ndarray | None = None
        self._pred_mm = float("nan")
        self._state: SceneState | None = None
        self._rgb: np.ndarray | None = None
        self.errors: list[float] = []
        self.absolute: list[float] = []
        self.absolute_mm = float("nan")
        self.unmatched = 0
        """Vision ticks where the placed brick could not be found, so no relative error exists."""
        self.pred_errors: list[float] = []
        self.step_errors: dict[int, list[float]] = {}

    def close(self) -> None:
        self.world.close()
        for twin in set(self.twins.values()):
            twin.close()

    # -- one vision tick ------------------------------------------------------------------------

    def _prepare_step(self, frame: ReferenceFrame) -> None:
        """Measure the Jacobians in the twin, at this reference frame's own pose.

        Re-measured at every step boundary and every `jacobian_every` frames in between. Six
        renders each, a few milliseconds; measuring once per step instead was the difference
        between the loop tracking and the loop walking away, because a Jacobian taken at the
        start of a 200 mm transport describes a camera that is nowhere near where it ends up.
        """
        bodies = dict(frame.bodies or self.ideal.bodies_by_step.get(frame.step_index, {}))
        held = self.ideal.held_by_step.get(frame.step_index)
        self.twin = self.twins[held is not None]
        if held is not None and self.twin.scene.held_name is not None:
            held = self.twin.scene.held_name
        bodies = {k: v for k, v in bodies.items()
                  if v in self.twin.scene.instance_ids}
        if not bodies:
            self._step = None
            self.trusted = False
            return
        pose = frame.commands[0]
        try:
            self._step = self._build(frame, bodies, held, pose)
        except BlindStep:
            self._step = None
            self.trusted = False
            return
        self._settle_servo(held, frame, bodies)

    def _build(self, frame, bodies, held, pose):
        if self.plan is not None:
            return self.plan.step_for(frame.k)
        return self.twin.build_step(
            frame.step, pose.pos, pose.spin, bodies,
            brick_body=held, tol_mm=self.cfg.tol_mm, min_refs=1,
            angle_from_blob=(len(bodies) == 1), square=(WANTED[0] == WANTED[1]),
            grip=pose.grip,
            # Two in-hand unknowns, from the brick's blob centre. The offsets in the plane are
            # what a tool move can answer; how deep the brick sits in the jaws is the grasp's
            # business and is checked against the commanded grasp height instead. Reading the
            # whole eight-number row to solve them measured worse, not better: mass, the second
            # moments and `orient4` also move with the brick's tilt and with how much of it the
            # jaws cut into, the fit has no column for either, and it spreads them over the two
            # it does have. At the lift dwell, in a nominal world with the brick centred in the
            # jaws to 0.3 mm, the row read 6.2 mm and the centre read 1.1 mm.
            brick_dof=2, brick_row=False,
        )

    def _settle_servo(self, held, frame, bodies) -> None:
        if self._servo is None:
            self._servo = VisualServo(step=self._step, lam=self.cfg.lam,
                                      settle_ticks=self.cfg.settle_ticks, min_refs=1,
                                      require_brick=held is not None)
            if not self.cfg.use_yaw:
                self._servo.max_spin = 0.0
            if self.cfg.pick:
                self._servo.max_in_hand = 0.012
        else:
            # Swapped, not rebuilt. The in-hand reading is an average over many frames of a
            # quantity that does not change; building a new servo at every Jacobian estimate
            # threw that average away every eight frames and left the loop steering on one
            # noisy frame's worth of it, which is about two millimetres.
            self._servo.step = self._step
            self._servo.brick_ref = self._step.brick_ref
            self._servo.require_brick = held is not None
        if frame.step == self.cfg.freeze_after and not self._in_hand_reset:
            self._in_hand_reset = True
            # Not on a pick. There the reading was already taken at the lift dwell, against the
            # bare table and with the arm at rest, and it is one constant: starting it again
            # over the baseplate measured 3.2 mm where the jaws hold the brick to 0.3 mm, and
            # the align dwell turned that into 5.5 mm of tool correction. The placement task
            # starts with the brick already held and has no lift dwell to read it at, so it
            # reads it here, which is what it was measured 10/10 with.
            if not self.cfg.pick:
                self._servo.reset_in_hand()
        sv = np.linalg.svd(self._step.J[:, :2], compute_uv=False)
        self.gain = float(sv[-1]) if sv.size else 0.0
        # Two blobs, or the one blob the whole step is about. Reaching for the brick there is
        # nothing else to line up on by design, and the wider single-blob dead band is what
        # pays for the noisier reading.
        alone = self.cfg.pick and frame.step in ("Approach", "Grasp")
        self.trusted = self.gain >= self.cfg.min_gain and (len(bodies) >= 2 or alone)

    def run(self) -> Iterator[TickRecord]:
        if self.controller is not None:
            yield from self.controller.run(self)
            return
        cfg = self.cfg
        frames = self.ideal.frames
        settled = 0
        while self.k < len(frames):
            frame = frames[self.k]
            if (frame.step_index != self._step_index
                    or self.k - self._jacobian_at >= cfg.jacobian_every):
                self._step_index = frame.step_index
                self._jacobian_at = self.k
                self._prepare_step(frame)

            # A held index issues the frame's last write only, so the world runs a fifth as
            # fast; on the way back up to the hover after a missed grasp it issues the whole
            # frame, because the arm has 55 mm to travel and the stall bound is counted in
            # vision ticks.
            commands = (frame.commands if self.holds == 0 or self._regrasping
                        or self._gating_grasp else frame.commands[-1:])
            for command in commands:
                pos = np.array([command.pos[0] + self.correction[0],
                                command.pos[1] + self.correction[1], command.pos[2]])
                self.world.command(pos, command.spin + self.correction[2], command.grip,
                                   c_vert=command.c_vert, c_lat=command.c_lat)
                yield self._record(frame, pos, command, vision=False)

            # While the arm is leaning on something the reference blobs are half behind the
            # tool, the tool is deliberately not where it is commanded, and the tracking error
            # is an error the arm is *meant* to have. Correcting on it winds the estimate up and
            # pushes the brick off the cell it was just pressed onto, so the correction freezes
            # and the sequence plays forward.
            self._pressing = any(c.c_vert < 1.0 or c.c_lat < 1.0 for c in frame.commands)
            if not self.frozen and frame.step_index > self._freeze_index:
                self.frozen = True
                self.frozen_correction = self.correction.copy()
            low = frame.commands[0].pos[2] < cfg.pick_freeze_below
            if cfg.pick and not self.frozen and low and not self.pick_frozen:
                self.pick_frozen = True
            if self.pick_frozen and self.ideal.held_by_step.get(frame.step_index) is None \
                    and frame.step in ("Approach", "Grasp"):
                self._descending = True
            else:
                self._descending = False
            self._state, self._rgb = self.world.observe_rgb()
            match, err_mm, delta, seen = self._feedback(frame)
            self.errors.append(err_mm)
            if self.dwelling:
                self.dwell_ticks += 1
            self.step_errors.setdefault(frame.step_index, []).append(err_mm)
            rec = self._record(frame, pos, commands[-1], vision=True, err_mm=err_mm, seen=seen)
            yield rec

            if self._grasp_missed(frame, commands[-1]):
                continue

            # The tube can never be tighter than the reading is trustworthy. With one small
            # blob in view the feature floor is about 1.7 mm; asking the index to wait for
            # 1.0 mm there is asking it to wait for noise to go away, and it does not -- the
            # grasp descent stopped 10 mm short doing exactly that, gripped the brick near its
            # top, and the 12 mm of vertical offset that left in the jaws came back as a phantom
            # 9 mm sideways in the in-hand estimate.
            floor = (self._servo.dead_band_single_mm if seen < 2 else self._servo.dead_band_mm
                     ) if self._servo is not None else cfg.advance_mm
            tube = max(cfg.advance_mm, floor, cfg.tube_scale * 1e3 * frame.travel)
            last = self.k == len(frames) - 1
            # The tube is the whole test. Whether the best match came out a frame or two back
            # is worth logging, but it is not a second veto: reference frames a couple of
            # millimetres apart and a tracking floor of a few tenths mean the nearest frame
            # wanders by one either way, and vetoing on it deadlocks the index.
            self.lag = self.k - match
            if err_mm != err_mm:
                self.blind_frames += 1
            # A loop that is not going to act cannot catch up, so it does not wait. Holding the
            # index would stall the run on an error nothing is going to reduce -- which is what
            # the camera-off control condition did: in a world identical to the recording it
            # still held at a step whose measurement floor (one small blob, about 1.7 mm) sits
            # above the advance floor, and tripped the stall bound. The error is still measured
            # and still reported for every one of these frames.
            # `_descending` too: the grasp's correction is already held below `pick_freeze_below`
            # because the jaws cut the brick's blob there, so holding the INDEX as well parks the
            # tool part-way down. Measured: the tool stopped 10 mm short, gripped the brick near
            # its top, and that 12 mm of vertical offset in the jaws aliased into the in-hand
            # estimate as 9 mm sideways -- which the loop then dutifully corrected for.
            # Waiting only ever makes sense where the estimate is being updated, which is at a
            # dwell. While the target is moving there is nothing to wait for: the residual is
            # tracking lag, the correction is deliberately held constant, and holding the index
            # as well simply parks the tool part-way through the move. That is what stopped the
            # grasp descent 10 mm short and left 12 mm of vertical offset in the jaws.
            dwell_only = cfg.pick if cfg.dwell_only is None else cfg.dwell_only
            playing_forward = (self.frozen or self._descending or not cfg.closed_loop
                               or not self.trusted
                               or (cfg.pick and frame.step == "Lift")
                               or (dwell_only and not self.dwelling))
            # Two places the index waits for something other than the tube. On the way back up
            # to the hover after a missed grasp there is nothing to compare against until the
            # arm is there. And the descent itself waits for the hover estimate to settle in
            # yaw as well as in position, because a square brick turned a degree and a half
            # fills the two millimetres of clearance the open jaws leave it.
            if self._regrasping:
                playing_forward = False
                if (self.k == self._descent_k - 1
                        and float(np.max(np.abs(self.world.joint_speed))) < cfg.dwell_speed):
                    self._regrasping = False
            self._gating_grasp = self._holding_for_grasp(frame, err_mm, floor)
            if not self._gating_grasp and (err_mm != err_mm or err_mm <= tube or playing_forward):
                if last:
                    # Finishing is the tight test, not the advancing one -- unless the
                    # correction is frozen, in which case nothing is going to move the error and
                    # waiting for it would be waiting forever. The number reached is reported
                    # either way.
                    inside = err_mm != err_mm or err_mm <= cfg.tol_mm or self.frozen
                    settled = settled + 1 if inside else 0
                    if settled >= cfg.settle_ticks:
                        break
                else:
                    self.k += 1
                self.holds = 0
                self.blind_holds = 0
            else:
                settled = 0
                self.holds += 1
                if err_mm != err_mm:
                    self.blind_holds += 1
                self.max_hold_seen = max(self.max_hold_seen, self.holds)
                if self.holds > cfg.max_hold:
                    # Two different faults wear the same number of held ticks. One is an error
                    # the loop can see and cannot reduce; the other is a reference that has left
                    # the window, where there is nothing to reduce it with. They want different
                    # answers, so the record says which it was.
                    if self.blind_holds * 2 > self.holds:
                        self.failure = (
                            f"stalled at {frame.step}: nothing to line up on for "
                            f"{self.blind_holds} of {self.holds} ticks, "
                            f"{seen} reference blob(s) matched")
                    else:
                        self.failure = (
                            f"stalled at {frame.step} with {err_mm:.2f} mm to make up, "
                            f"{seen} reference blob(s) matched")
                    break
        for index, errs in self.step_errors.items():
            title = next(f.step for f in frames if f.step_index == index)
            self.reports.append(StepReport(title, len(errs), errs[-1],
                                           errs[-1] <= self.cfg.tol_mm * 4))

    # -- the grasp -------------------------------------------------------------------------------

    def _holding_for_grasp(self, frame: ReferenceFrame, err_mm: float, floor: float) -> bool:
        """Whether the descent onto the brick is still waiting for the hover estimate to settle.

        Only at the last frame before the descent, and only for a pick. Both halves of the
        estimate have to have arrived -- where the brick is and which way it is turned -- because
        the descent freezes the correction and the jaws then go wherever it left them. Measured
        without this, on a brick lying 13.4 mm and 15.8 deg off: the index ran through the hover
        while the reading still said 21 mm, the correction stopped at 1.6 mm, and the grasp
        stopped short twice.
        """
        cfg = self.cfg
        if not (cfg.pick and cfg.closed_loop) or self.frozen or self._descent_k is None:
            return False
        if self.k != self._descent_k - 1 or self._regrasping:
            return False
        close = err_mm == err_mm and err_mm <= max(floor, cfg.advance_mm)
        turned = (self.yaw_err == self.yaw_err
                  and abs(self.yaw_err) <= math.radians(cfg.grasp_yaw_deg))
        self._grasp_settled = self._grasp_settled + 1 if (close and turned) else 0
        return self._grasp_settled < cfg.grasp_ticks and self.holds < cfg.grasp_wait

    def _grasp_missed(self, frame: ReferenceFrame, command: Command) -> bool:
        """Did the jaws close somewhere other than where they were sent? If so, go back up.

        The reading is forward kinematics on the joint angles the bus reports, which is
        mechanical state the controller is allowed to have. A tool that stopped short gripped
        the brick near its top rim, where it does not stay: the brick slid 32 mm out of the jaws
        between the lift and the press. So the grasp is opened and made again from the hover,
        once.
        """
        cfg = self.cfg
        if not (cfg.pick and cfg.closed_loop) or self._descent_k is None:
            return False
        if self._grasp_checked or self.k != self._grasp_k or self._regrasping:
            return False
        self._grasp_checked = True
        tcp_z = float(self.world.feedback(command.grip).tcp_pos[2])
        if tcp_z - self._grasp_z <= cfg.grasp_miss:
            return False
        short_mm = 1e3 * (tcp_z - self._grasp_z)
        if self.regrasps:
            self.failure = f"the grasp stopped {short_mm:.1f} mm short twice"
            return False
        self.regrasps += 1
        self.k = self._descent_k - 1
        self.holds = 0
        self._step_index = -1
        self._jacobian_at = -10 ** 6
        self._grasp_settled = 0
        self._grasp_checked = False
        self._regrasping = True
        self.pick_frozen = False
        self._descending = False
        if self._servo is not None:
            self._servo.reset_in_hand()
        return True

    # -- the inverse ------------------------------------------------------------------------------

    def _feedback(self, frame: ReferenceFrame) -> tuple[int, float, np.ndarray, int]:
        """One step of the inverse, plus the next-frame prediction it implies.

        The comparison frame is chosen first: the best match among the last few reference frames,
        never one ahead. A lagging arm then reads as lag, which the progress index answers by
        holding, instead of reading as a displaced world, which the feedback would answer by
        driving the tool somewhere it should not go.
        """
        self.yaw_err = float("nan")
        # The lift dwell is the one place a tick is worth taking on an untrusted reading: what
        # is read there is the brick's own blob against the twin's, which needs no reference
        # geometry at all, and over the supply area there is only one reference blob in view
        # anyway. Nothing is steered by it -- the correction integrates only on trusted ticks.
        in_hand_only = (self.cfg.pick and self._step is not None
                        and self._step.J_brick is not None and frame.step == "Lift")
        if (self._servo is None or self._step is None
                or (not self.trusted and not in_hand_only)
                or getattr(self, "_pressing", False) or getattr(self, "_descending", False)):
            return self.k, float("nan"), np.zeros(3), 0
        if self.frozen:
            # Still measured and still reported; simply no longer acted on.
            pass

        # Relative from the moment the brick is let go: until then it is in the hand and the
        # question really is "is the tool where the recording had it"; afterwards it is part of
        # the build and the question is "does it sit where the target puts it, relative
        # to the baseplate and the bricks already there".
        # Estimate only where the sequence dwells. Everything this loop estimates is a STATIC
        # unknown -- where the brick lies, where the plate is, how the brick sits in the jaws --
        # but a residual measured while the arm is moving is dominated by tracking lag and by
        # the bias of a single small blob, not by the world. Integrating those frames is what
        # walked a brick 116 mm in a world where nothing was wrong. On moving frames the run is
        # feed-forward plus the last constant estimate, and nothing else.
        dwell_only = self.cfg.pick if self.cfg.dwell_only is None else self.cfg.dwell_only
        self.dwelling = (not dwell_only) or (
            1e3 * frame.travel < self.cfg.dwell_travel_mm
            and float(np.max(np.abs(self.world.joint_speed))) < self.cfg.dwell_speed)
        if self.cfg.pick:
            # One dwell, one unknown. The brick's offset in the jaws is read at the lift dwell
            # and nowhere else; every other tick reads the stored value.
            self._servo.hold_in_hand = not (self.dwelling and frame.step == "Lift")
        rel = self.frozen and self.ideal.held_by_step.get(frame.step_index) is None
        # In relative mode the placed brick is the whole point: measured over the baseplate and
        # the bricks that never moved, the relative error is near zero whatever the arm did. If
        # the brick itself is not among the blobs that matched, the number is not small, it is
        # unknown -- and it is reported as unknown.
        need = self.ideal.held_body if rel else None
        best, out = self.k, None
        best_err = float("inf")
        lo = max(0, self.k - self.cfg.match_window)
        for j in range(lo, self.k + 1):
            trial = self._servo.update(self._state, self._step.refs,
                                       reference=self.ideal.frames[j].state, dry_run=True,
                                       relative=rel)
            if trial["blind"]:
                continue
            e = float(np.linalg.norm(
                np.array([trial["d_r"][0] + trial["d_b"][0],
                          trial["d_r"][1] + trial["d_b"][1]])))
            if e < best_err:
                best_err, best, out = e, j, trial
        if out is None:
            return self.k, float("nan"), np.zeros(3), 0
        # Once the frame is chosen, take the reading again for real so the in-hand average sees
        # exactly one measurement per tick.
        out = self._servo.update(self._state, self._step.refs,
                                 reference=self.ideal.frames[best].state, relative=rel)
        absolute = self._servo.update(self._state, self._step.refs,
                                      reference=self.ideal.frames[best].state, dry_run=True)
        self.absolute_mm = (float("nan") if absolute["blind"]
                            else 1e3 * float(np.linalg.norm(
                                np.array([absolute["d_r"][0] + absolute["d_b"][0],
                                          absolute["d_r"][1] + absolute["d_b"][1]]))))
        self.absolute.append(self.absolute_mm)
        if out["blind"]:
            return self.k, float("nan"), np.zeros(3), 0
        if need is not None:
            titles = [t for t, body in (frame.bodies or {}).items() if body == need]
            if not titles or out["found"].centres.get(titles[0]) is None:
                self.unmatched += 1
                return self.k, float("nan"), np.zeros(3), int(out["seen"])
        self.matched = best
        self.yaw_err = float(out["d_r"][2])

        measured = np.array([out["d_r"][0] + out["d_b"][0],
                             out["d_r"][1] + out["d_b"][1], out["d_r"][2]])
        # Prediction error: what the last tick said this tick would look like, against what it
        # does. The prediction is the reference's own step forward from where the world actually
        # was, plus the correction that was commanded -- no model beyond that, and no hindsight.
        if self._predicted is not None:
            self._pred_mm = 1e3 * float(np.linalg.norm(measured[:2] - self._predicted[:2]))
            self.pred_errors.append(self._pred_mm)

        analytic = -self.cfg.lam * measured
        if self.cfg.closed_loop and not self.frozen and self.dwelling and self.trusted:
            # The one substitution: where the correction increment comes from. The saturation,
            # the integrator and its ceiling are the same either way, and so is everything that
            # reads `measured` -- the reported tracking error, the tube the progress index waits
            # on, and the prediction error.
            delta = analytic.copy() if self.inverse is None else self._learned(frame, out, analytic)
            n = float(np.linalg.norm(delta[:2]))
            if n > 0.0012:
                delta[:2] *= 0.0012 / n
            delta[2] = float(np.clip(delta[2], -math.radians(0.7), math.radians(0.7)))
            self.correction = self.correction + delta
            n_c = float(np.linalg.norm(self.correction[:2]))
            if n_c > self.cfg.max_correction:
                self.correction[:2] *= self.cfg.max_correction / n_c
        else:
            delta = np.zeros(3)
        if self.observer is not None:
            self.observer(self, frame, best, out, measured, analytic, delta)

        self._predicted = (measured + delta if self.predictor is None
                           else self.predictor.predict(self, frame, out, measured, delta))
        return best, 1e3 * float(np.linalg.norm(measured[:2])), delta, int(out["seen"])

    def _learned(self, frame: ReferenceFrame, out: dict, analytic: np.ndarray) -> np.ndarray:
        """`G(observed scene state, target scene state, mechanical state)`, in metres and radians.

        Slot-row differences and what the servo bus reports, and nothing else: the same two
        readings the analytic inverse takes, minus the Jacobian.
        """
        mech = np.concatenate([self.world.joint_position, self.world.joint_speed])
        return np.asarray(self.inverse.delta(
            state=self._state, target=self.ideal.frames[self.matched].state,
            found=out["found"], want=out["want"], mech=mech,
            in_hand=self.ideal.held_by_step.get(frame.step_index) is not None,
            analytic=analytic), dtype=float)

    # -- bookkeeping -------------------------------------------------------------------------------

    def _record(self, frame: ReferenceFrame, pos, command: Command, *, vision: bool,
                err_mm: float = float("nan"), seen: int = 0) -> TickRecord:
        nxt = self.ideal.frames[min(self.k + 1, len(self.ideal.frames) - 1)]
        rec = TickRecord(
            tick=self.world.tick, t=self.world.t, step=frame.step,
            step_index=frame.step_index, cmd_pos=np.asarray(pos, dtype=float).copy(),
            spin=float(command.spin + self.correction[2]), grip=float(command.grip),
            correction=self.correction.copy(), k=self.k, vision=vision,
            err_mm=err_mm if vision else float("nan"),
            tol_mm=self.cfg.tol_mm, predict_mm=self._pred_mm if vision else float("nan"),
            refs_seen=seen,
        )
        if vision:
            rec.state = self._state
            rec.rgb = self._rgb
            rec.target = nxt.state
            rec.target_label = nxt.label
        return rec
