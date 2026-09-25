"""Sequencing the scripted skills: ``LOOK -> ALIGN -> PRESS -> VERIFY``, with timeouts.

:class:`SkillFSM` is what actually places a brick.  It walks the four states of the placement
(section 3), each with its own deadline, and hands whatever the current state wants to the same
:class:`~brick_builder.control.primitives.ToolTarget` interface the environments consume:

``look``    hold a pose from which the wrist camera can see the plate, so a perception producer
            has a still frame to work from.  Skipped when no viewing pose is configured -- the press
            environment is blind by design and starts straight at ``align``.
``align``   a straight-line move to the hover point above the target cell.
``press``   :class:`~brick_builder.control.primitives.PressScripted`.
``verify``  the press lets go, the tool settles, and then it is tugged a few millimetres upward
            at a reduced torque limit.  Two verdicts come out of that, and on this arm they do
            not agree.

**What the specification asks for, and what it actually measures.**  The specification defines
VERIFY as the tug plus a check that the servos' ``Present_Load`` rises, on the reasoning that a
snapped-in brick stops the tool following its own command.  Measured here, that signal does not
work, and the reason is this arm rather than the idea: ``Present_Load`` *is* the joint position
error, scaled, and this arm's own joint friction is enough that a tug this small does not
reliably move the tool whether or not anything is holding it -- the load rise is stiction rather
than attachment: over 40 trials the verdict was right 7-8 times in 20, i.e. worse than a coin.
Tugging hard enough to beat the stiction (8 mm) reverses the problem -- it pulls a genuinely
seated 2x4 straight back off the plate, which makes the test destroy what it is testing.

**What does work** is the height the tool settles at once the press has let go
(:attr:`SkillFSM.seated_by_height`): measured on this rig (2026-09-25, seed 7, 40 trials), a
real seat leaves it -0.44 to +0.34 mm from the seated height and a brick resting on the stud
tops 3.87-4.40 mm, against ``verify_height_tol`` = 2.5 mm.  It is not perfect: measured (seed 7,
20 seated-group trials, single attempt each, no case rejection -- of which 9/20 actually landed a
real seat, see :func:`verify_trials`), the verdict was right 37 times out of 40 overall -- Wilson
interval [0.801, 0.974].  Every disagreement was inspected, and they are not one shape but two.
Two of the three are the snap model's own connection gate, not a height problem: every stud of the
held brick reads 4.70-4.74 mm of raw insertion depth (a fully seated depth) with the tool settled
accordingly, but the brick sits 0.47-0.53 mm laterally off the stud axis -- just outside
``SnapParams.capture_lateral`` (0.30 mm, ``bricks/snap_api.py``), the gate the snap model itself
requires before it will report that depth as engaged or form a connection at all
(``bricks/snap_detent.py``).  Below that gate the model reports zero depth and welds nothing, so
the "really seated" ground truth this module's tests use calls it not seated
(:func:`ground_truth_occupancy`, below, reads it occupied); the tool, which only ever feels where
the brick physically stopped it, cannot tell the two cases apart by height, because a brick 0.5 mm off the stud axis stops the tool
at the same height a centred one would.  The third disagreement is an ordinary shallow press, not
a gate rejection: every stud engaged well inside the gate (lateral <= 0.08 mm) but stopped at
3.90 mm, 0.15 mm short of the 4.05 mm ``really_seated`` threshold (0.6 mm short of the full 4.5 mm
stud height).  So the height verdict's failure mode has two distinct causes, and neither is caught
by a second reading of height alone (see ``occupancy_check`` below).  Both numbers are reported;
:attr:`SkillFSM.verified` stays the spec's load-rise verdict so the finding is visible rather than
quietly redefined.

A second, independent cross-check is available whenever a scene model is: ``occupancy_check``,
an optional zero-argument callable a caller supplies (closing over whatever vision producer it
has bound to the target cell), returning ``True``/``False`` for "occupied"/"empty" or ``None``
when no reading is available this tick.  It is read once, at the same moment the settled height
is (right after the press releases and the tool stops moving), and stored verbatim in
:attr:`SkillFSM.seated_by_occupancy` -- never used to override the height verdict, only reported
alongside it, since a producer's occupancy call has its own failure modes (dropout, wrong cell)
that this module cannot itself audit.  With no callable given (the default), the field stays
``None`` and nothing about the height-only behaviour changes.  :func:`ground_truth_occupancy` is
the placeholder binding for a sim-only caller (a test, or a scripted demo with no scene model
wired up yet) -- simulator ground truth, the same source the rest of this project's placeholder
perception stands in with until a trained scene model replaces it.  It deliberately reads the raw
insertion depth of whichever stud sits nearest the held brick's own sockets -- the same per-socket
pairing the snap model itself uses, but without that model's ``capture_lateral``/tilt connection
gates -- so that it answers a coarser question than seating on purpose: "is anything resting in
the cell at all," not "is it seated" or "is it connected," because that is the question a real
occupancy sensor could answer too.  A brick pressed to a seating depth but rejected by the
connection gate (the two-of-three false positives traced above) reads occupied here -- unlike the
connection model's own gated depth, which reports zero for that case.  Known defect: the
threshold (0.1 x stud height, 0.45 mm) falls inside a resting brick's own raw depth (0.11-0.55 mm
measured, seed 7), so a brick resting on the stud tops reads empty in 24 of 27 measured runs; the
"anything in the cell" reading is not achieved yet.

Everything here obeys the same two rules as ``primitives.py``: deployable signals only, and
arithmetic simple enough to port to C.

:func:`run_skill` is the plain control loop this runs in outside reinforcement learning: write
the servo goals once per 20 ms tick, step physics ten times, read back.  It is the same multi-rate
tick ``envs/base_env.py`` implements, minus the observation assembly, reward and episode
bookkeeping that make that one a gymnasium environment -- a scripted demo, or a hardware bring-up
script, should not have to construct an RL environment to press a brick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from brick_builder.control.primitives import (
    CONTROL_DT,
    Feedback,
    MoveTo,
    PressConfig,
    PressScripted,
    ToolTarget,
    feedback_from_readings,
)
from brick_builder.robot.kinematics import euler_zyx

STATES: tuple[str, ...] = ("idle", "look", "align", "press", "verify", "done", "failed")


@dataclass(frozen=True)
class FsmConfig:
    """Deadlines and the verify tug.

    ``verify_torque`` is the torque-limit fraction the tug runs at, and it has a floor that is
    not a matter of taste.  Measured on this rig: at 0.6 the arm cannot climb at all -- gravity
    at the press pose eats the whole budget -- so a loose brick and a seated one both refuse to
    move and the tug tells them apart not at all (rise 0.019 seated against 0.043 loose, the
    wrong way round).  At 0.8 the arm can climb, and a seated brick is the only thing that stops
    it.  ``verify_lift`` is likewise larger than it looks it needs to be: the soft weld that
    stands in for the grip stretches a couple of millimetres before the arm feels the plate at
    all, so a 4 mm tug mostly measures the weld and an 8 mm one measures the brick.
    """

    look_pos: tuple[float, float, float] | None = None
    look_hold: float = 0.30
    approach_speed: float = 0.06
    hover: float = 0.005
    align_tol: float = 0.0015

    verify_release: float = 0.25
    verify_lift: float = 0.004
    verify_torque: float = 0.8
    verify_settle: float = 0.10
    verify_time: float = 0.30
    verify_load_rise: float = 0.010
    verify_height_tol: float = 0.0025
    """How far above the seated height the tool may stand, once the press has let go, and still
    count as having seated the brick.  Measured after the servo encoder correction (2026-09-25,
    seed 7, 40 trials): a real seat settles -0.44 to +0.34 mm from the seated height, a brick
    resting on the stud tops 3.87-4.40 mm.  (Before the correction: 2.0-2.3 mm and 4.6-6.5 mm.)
    Needs recalibrating for the friction grasp and again on hardware."""

    timeouts: dict[str, float] = field(
        default_factory=lambda: {"look": 2.0, "align": 4.0, "press": 8.0, "verify": 1.5}
    )


class SkillFSM:
    """Place one brick on cell ``target_xy`` at seated tool height ``seat_z``.

    Read :attr:`state` for where it is, :attr:`done`/:attr:`failed` for whether it has finished,
    :attr:`verified` for the verify tug's verdict (``None`` until the tug has run) and
    :attr:`timed_out` for which state ran out of time, if any.
    """

    def __init__(
        self,
        seat_z: float,
        target_xy,
        *,
        press: PressConfig = PressConfig(),
        cfg: FsmConfig = FsmConfig(),
        spin: float = 0.0,
        grip: float = 0.0,
        dt: float = CONTROL_DT,
        occupancy_check: Callable[[], bool | None] | None = None,
    ) -> None:
        self.seat_z = float(seat_z)
        self.target_xy = np.asarray(target_xy, dtype=float)[:2].copy()
        self.press_cfg = press
        self.cfg = cfg
        self.spin = float(spin)
        self.grip = float(grip)
        self.dt = float(dt)
        self.occupancy_check = occupancy_check

        self.state = "idle"
        self.done = False
        self.failed = False
        self.timed_out: str | None = None
        self.verified: bool | None = None
        self.verify_load_rise = 0.0
        self.verify_lift_follow = 0.0
        self.verify_height = float("nan")
        self.seated_by_height: bool | None = None
        self.seated_by_occupancy: bool | None = None
        self.press: PressScripted | None = None
        self.history: list[tuple[str, float]] = []

        self._move: MoveTo | None = None
        self._t = 0.0
        self._base_load = 0.0
        self._base_n = 0
        self._tug_load = 0.0
        self._tug_n = 0
        self._verify_from = np.zeros(3)
        self._settled_at: np.ndarray | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    def reset(self, fb: Feedback) -> None:
        self.state = "idle"
        self.done = False
        self.failed = False
        self.timed_out = None
        self.verified = None
        self.verify_load_rise = 0.0
        self.verify_lift_follow = 0.0
        self.verify_height = float("nan")
        self.seated_by_height = None
        self.seated_by_occupancy = None
        self.history = []
        self._t = 0.0
        if self.cfg.look_pos is not None:
            self._enter("look", fb)
        else:
            self._enter("align", fb)

    def _enter(self, state: str, fb: Feedback) -> None:
        self.history.append((state, self._t))
        self.state = state
        self._t = 0.0
        cfg = self.cfg
        if state == "look":
            self._move = MoveTo(cfg.look_pos, cfg.approach_speed, spin=self.spin,
                                grip=self.grip, tol=cfg.align_tol, dt=self.dt)
            self._move.reset(fb)
        elif state == "align":
            above = np.array([self.target_xy[0], self.target_xy[1], self.seat_z + cfg.hover])
            self._move = MoveTo(above, cfg.approach_speed, tilt=0.0, spin=self.spin,
                                grip=self.grip, tol=cfg.align_tol, dt=self.dt)
            self._move.reset(fb)
        elif state == "press":
            self.press = PressScripted(self.seat_z, self.press_cfg, grip=self.grip, dt=self.dt)
            self.press.reset(fb)
        elif state == "verify":
            self._verify_from = np.asarray(fb.tcp_pos, dtype=float).copy()
            self._settled_at = None
            self._base_load = 0.0
            self._base_n = 0
            self._tug_load = 0.0
            self._tug_n = 0
        elif state in ("done", "failed"):
            self.done = True
            self.failed = state == "failed"

    # -- per tick ----------------------------------------------------------------------------

    def update(self, fb: Feedback) -> ToolTarget:
        cfg = self.cfg
        limit = cfg.timeouts.get(self.state)
        if limit is not None and self._t > limit and self.state not in ("done", "failed"):
            self.timed_out = self.state
            target = self._hold(fb)
            self._enter("failed", fb)
            return target

        if self.state == "look":
            target = self._move.update(fb)
            if self._move.done and self._t >= cfg.look_hold:
                self._enter("align", fb)
        elif self.state == "align":
            target = self._move.update(fb)
            if self._move.done:
                self._enter("press", fb)
        elif self.state == "press":
            target = self.press.update(fb)
            if self.press.done:
                self._enter("verify", fb)
        elif self.state == "verify":
            target = self._verify(fb)
        else:
            target = self._hold(fb)

        self._t += self.dt
        return target

    def _hold(self, fb: Feedback) -> ToolTarget:
        pos = self._verify_from if self.state == "verify" else np.asarray(fb.tcp_pos, dtype=float)
        return ToolTarget(pos=pos.copy(), tilt=0.0, spin=self.spin, grip=self.grip)

    def _verify(self, fb: Feedback) -> ToolTarget:
        """Release, settle, tug, decide.

        The release phase is not optional.  A press ends with the tool commanded a centimetre
        below where it actually is, so the servos are carrying their full press error and the
        load register is pinned near the top of its range; measuring a "baseline" there and then
        tugging would read the press *relaxing*, which is a large load **drop** whether or not
        the brick is attached.  Commanding the pose the tool is genuinely at takes that error out
        first, and the baseline is then the load of simply standing still.

        The tug load is averaged over the second half of ``verify_time``; the first half is the
        servo accelerating into the new command and says nothing about what holds the brick.
        """
        cfg = self.cfg
        pos = self._verify_from.copy()
        c_vert = 1.0
        after_release = self._t - cfg.verify_release
        if after_release < 0.0:
            pass                                    # holding the pose the press left the tool at
        elif self._settled_at is None:
            # Re-latch: a press ends with the tool held a few millimetres below where it would
            # rest unloaded, so it springs up as the command is taken away. Measuring the tug
            # from the pose at *entry* would therefore count that spring-back as the tug moving,
            # which is exactly backwards -- the seated brick would look like the free one.
            self._settled_at = np.asarray(fb.tcp_pos, dtype=float).copy()
            self._verify_from = self._settled_at.copy()
            # The second, cheaper verdict: once the press has let go, is the tool standing at the
            # height a seated brick would put it at? On this arm that turns out to be the signal
            # that works -- see the module docstring.
            self.verify_height = float(self._settled_at[2] - self.seat_z)
            self.seated_by_height = self.verify_height <= cfg.verify_height_tol
            # Third, independent cross-check, only when a caller gave us one: a scene model's
            # own occupancy reading of the target cell, taken at this same settled moment. Never
            # used to override the height verdict -- see the module docstring on why.
            if self.occupancy_check is not None:
                self.seated_by_occupancy = self.occupancy_check()
            pos = self._verify_from.copy()
            self._base_load += fb.peak_load
            self._base_n += 1
        elif after_release < cfg.verify_settle:
            self._base_load += fb.peak_load
            self._base_n += 1
        else:
            pos[2] += cfg.verify_lift
            c_vert = cfg.verify_torque
            tug = after_release - cfg.verify_settle
            if tug >= 0.5 * cfg.verify_time:
                self._tug_load += fb.peak_load
                self._tug_n += 1
            if tug >= cfg.verify_time:
                base_mean = self._base_load / max(self._base_n, 1)
                tug_mean = self._tug_load / max(self._tug_n, 1)
                self.verify_load_rise = float(tug_mean - base_mean)
                self.verify_lift_follow = float(fb.tcp_pos[2] - self._verify_from[2])
                self.verified = self.verify_load_rise >= cfg.verify_load_rise
                self._enter("done", fb)
        return ToolTarget(pos=pos, tilt=0.0, spin=self.spin, grip=self.grip,
                          c_vert=c_vert, c_lat=1.0)


# ------------------------------------------------------------------------------- the plain loop ---


def run_skill(
    scene,
    fsm: SkillFSM,
    *,
    max_ticks: int = 400,
    servo_cfg=None,
    seed: int = 0,
    record: list | None = None,
) -> dict:
    """Drive ``fsm`` against a built :class:`~brick_builder.scene.compose.Scene` and report what happened.

    One tick is one ``ServoBus.write``, ten physics steps with the snap bookkeeping run at its
    configured rate, and one ``ServoBus.read``; the joint goal comes from inverse kinematics of
    the tool target plus a gravity feedforward term, because the servo model is a plain position
    servo with no gravity compensation of its own and would otherwise sag tens of millimetres
    (the static-sag finding).

    ``record`` is an optional list that gets one dict per tick appended to it.
    """
    from brick_builder.robot.patch_so101 import ARM_JOINTS
    from brick_builder.robot.servo_model import ServoBus, ServoConfig
    from brick_builder.scene.compose import gripper_command

    model, data, kin = scene.model, scene.data, scene.kin
    bus = ServoBus(model, data, cfg=servo_cfg or ServoConfig(), rng=np.random.default_rng(seed))
    kp5 = bus.kp[:5]
    vert = [ARM_JOINTS.index(n) for n in ("shoulder_lift", "elbow_flex", "wrist_flex")]
    lat = [ARM_JOINTS.index(n) for n in ("shoulder_pan", "wrist_roll")]

    q_goal = data.qpos[scene.arm_qpos_adr[:5]].copy()
    readings = bus.read()
    fsm.reset(feedback_from_readings(kin, readings, grip=float(data.ctrl[5])))

    snap_every = max(int(scene.snap.params.snap_every), 1)
    before_verify = None
    before_depths = np.zeros(0)
    ticks = 0
    for ticks in range(1, max_ticks + 1):
        fb = feedback_from_readings(kin, readings, grip=float(data.ctrl[5]))
        target = fsm.update(fb)

        q_goal = _ik(kin, target, q_goal)
        goal = q_goal.copy()
        for _ in range(2):
            goal = q_goal + kin.gravity_torque(goal) / kp5
        torque = np.ones(len(ARM_JOINTS))
        torque[vert] = target.c_vert
        torque[lat] = target.c_lat
        bus.write(np.concatenate([goal, [gripper_command(scene)]]), torque_limit=torque)

        for k in range(10):
            bus.substep()
            if k % snap_every == 0:
                scene.snap.update(model, data)
            mujoco.mj_step(model, data)
        readings = bus.read()
        if before_verify is None and fsm.state == "verify":
            # Ground truth has to be taken before the tug, not after: a tug hard enough to be
            # worth measuring can itself pull a seated brick back off, and then "was it seated"
            # answered afterwards is a question about the tug, not about the press.
            before_verify = scene.snap.connections()
            if scene.held_name is not None:
                before_depths = np.asarray(scene.snap.stud_depths(scene.held_name)).copy()

        if record is not None:
            record.append({
                "state": fsm.state,
                "phase": fsm.press.phase if fsm.press is not None else "",
                "cmd": np.asarray(target.pos).copy(),
                "tcp": scene.tcp_pose()[0].copy(),
                "load": float(np.max(np.abs(readings.load))),
                "c_vert": target.c_vert,
            })
        if fsm.done:
            break

    return {
        "ticks": ticks,
        "seconds": ticks * fsm.dt,
        "state": fsm.state,
        "failed": fsm.failed,
        "timed_out": fsm.timed_out,
        "verified": fsm.verified,
        "verify_load_rise": fsm.verify_load_rise,
        "verify_lift_follow": fsm.verify_lift_follow,
        "verify_height": fsm.verify_height,
        "seated_by_height": fsm.seated_by_height,
        "seated_by_occupancy": fsm.seated_by_occupancy,
        "connections": scene.snap.connections(),
        "connections_before_verify": before_verify if before_verify is not None else [],
        "depths_before_verify": before_depths,
        "warnings": scene.warnings(),
    }


def ground_truth_occupancy(scene, brick: str | None = None) -> bool | None:
    """A placeholder ``occupancy_check`` binding for a sim-only caller: simulator ground truth
    stands in for a scene model's own occupancy readout, the same way it does everywhere else in
    this project until a trained model replaces it.  ``brick`` defaults to ``scene.held_name``;
    returns ``None`` if there is nothing held to ask about (matching a real reading that never
    arrived).

    Deliberately coarser than :attr:`SkillFSM.seated_by_height`: "is anything resting in the cell
    at all" (the nearest stud to each of the brick's own sockets reading more than a tenth of the
    stud height of raw insertion depth), not "is it seated to depth" or "is it connected."  This
    reads :meth:`~brick_builder.bricks.snap_api.SnapGeometry.pair_metrics` directly -- the same
    per-socket, nearest-stud pairing the snap model itself uses -- rather than
    ``SnapModel.stud_depths``, which is gated to zero for any pairing outside the connection
    model's own ``capture_lateral``/tilt tolerances (``bricks/snap_detent.py``) and so answers
    "is it connected," a strictly narrower question a real occupancy sensor has no way to answer
    either.  The two questions can disagree in both directions -- a brick resting shallowly on the
    stud tops is occupied but not seated, and a brick pressed to a seating depth but rejected by
    the connection gate is occupied despite the snap model's own bookkeeping calling it empty (see
    the module docstring's VERIFY section for a traced case of the second) -- which is exactly why
    this cross-check does not get to override :attr:`SkillFSM.seated_by_height`.
    """
    brick = scene.held_name if brick is None else brick
    if brick is None:
        return None
    geom = scene.snap.geom
    bi = geom.brick_names.index(brick)
    sockets = np.nonzero(geom.socket_brick == bi)[0]
    if sockets.size == 0:
        return None
    depth, lateral, _tilt, valid = geom.pair_metrics(scene.data, rows=sockets)
    lateral = np.where(valid, lateral, np.inf)
    nearest = np.argmin(lateral, axis=1)
    best_depth = depth[np.arange(sockets.size), nearest]
    return bool(best_depth.size and float(best_depth.max()) > 0.1 * geom.stud_h)


def _ik(kin, target: ToolTarget, q0: np.ndarray) -> np.ndarray:
    """Warm-started inverse kinematics for one tool target; keeps ``q0`` if it does not converge."""
    q = np.asarray(q0, dtype=float).copy()
    result = None
    for _ in range(2):
        result = kin.ik(np.asarray(target.pos, dtype=float), float(target.tilt), math.pi, q,
                        iters=6, pos_tol=2e-4, ang_tol=math.radians(0.2))
        q = result.q.copy()
        achieved, _, _ = euler_zyx(kin.fk(q)[1])
        err = math.remainder(target.spin - achieved, 2.0 * math.pi)
        if abs(err) < math.radians(0.1):
            break
        q[4] = float(np.clip(q[4] + err, kin.jnt_range[4, 0], kin.jnt_range[4, 1]))
    return q if (result is not None and result.converged) else np.asarray(q0, dtype=float)


# ----------------------------------------------------------------------------------- figures ---


def verify_trials(
    trials: int = 12,
    seed: int = 0,
    seated_error: float = 0.0005,
    loose_error: float = 0.008,
    cfg: FsmConfig = FsmConfig(),
    press: PressConfig = PressConfig(max_searches=0),
    occupancy: bool = False,
) -> dict:
    """Run the whole skill on cases that do seat and cases that cannot, and report both verdicts.

    The two groups differ only in how wrong the *believed* cell is.  ``seated_error`` is half a
    millimetre, which always goes in; ``loose_error`` is 8 mm, outside the spiral's own 4 mm
    reach, so the press sweeps, gives up and hands VERIFY a brick that is resting on the stud
    tops rather than snapped onto them.  Both groups use exactly the same configuration, which
    is what makes the comparison a test of VERIFY and not of the press.

    Each draw is a single attempt -- this function does not redraw or reject a case whose actual
    outcome disagrees with the group it was drawn for (``out["runs"][i]["really_seated"]`` says
    which, from the snap model's own stud depths, so the disagreement is visible rather than
    hidden).  At ``seated_error``, comfortably inside the 3 mm capture basin on paper, a single
    attempt only actually seats a 2x4 well under half the time on this rig (2026-09-25, seed 7, 20
    trials: 9/20), purely from run-to-run servo noise near the capture gate's own edge.  That is
    expected, not a bug to sample away: the "seated" group's *name* describes the believed error
    the arm is given, not a guarantee every draw lands a real seat, and the accuracy this function
    measures is the height verdict's own accuracy on exactly the mix of outcomes a fixed 0.5 mm
    belief error actually produces -- see the module docstring for what that mix does to the
    verdict.

    ``occupancy=True`` binds each trial's :attr:`SkillFSM.occupancy_check` to
    :func:`ground_truth_occupancy` on this same scene, so ``out["runs"][i]["seated_by_occupancy"]``
    is populated instead of left ``None`` -- this does not consume any extra draws from ``rng``,
    so it does not change which cases the two groups land on.
    """
    from brick_builder.scene.compose import (
        HeldBrick,
        SceneConfig,
        build_scene,
        reset_scene,
        tool_down_ik,
        tool_reach_below_tcp,
    )

    scene = build_scene(SceneConfig(held=HeldBrick(nx=2, ny=4), cameras=(), home_hover=0.005))
    seat_z = float(scene.plate_top + tool_reach_below_tcp(scene))
    cell = np.array(scene.cfg.plate_xy, dtype=float)
    rng = np.random.default_rng(seed)

    out: dict[str, list] = {"seated": [], "loose": [], "runs": []}
    for tag, error in (("seated", seated_error), ("loose", loose_error)):
        for _ in range(trials):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            believed = cell + error * np.array([math.cos(angle), math.sin(angle)])
            start = np.array([believed[0], believed[1], seat_z + cfg.hover])
            reset_scene(scene, rng, arm_q=tool_down_ik(scene.kin, start, 0.0, q0=scene.home_q))
            fsm = SkillFSM(seat_z, believed, press=press, cfg=cfg)
            if occupancy:
                fsm.occupancy_check = lambda: ground_truth_occupancy(scene)
            report = run_skill(scene, fsm, max_ticks=500, seed=int(rng.integers(2**31)))
            report["group"] = tag
            depths = report["depths_before_verify"]
            report["really_seated"] = bool(
                depths.size and depths.min() >= 0.9 * scene.cfg.spec.stud_h
            )
            out[tag].append(report["verify_load_rise"])
            out["runs"].append(report)
    return out


def make_verify_figure(out_dir="docs/media/control", trials: int = 12, seed: int = 0) -> dict:
    """Load rise during the verify tug, for a seated brick and for one that only looks seated.

    This figure is the evidence behind the ``verify_load_rise`` threshold: two clouds with a gap
    between them, and the threshold drawn through the gap.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = verify_trials(trials=trials, seed=seed)
    jitter = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for x, tag, colour, label in (
        (0.0, "seated", "#1a7f37", "snapped onto the plate"),
        (1.0, "loose", "#d1242f", "resting on the stud tops"),
    ):
        values = np.asarray(data[tag])
        ax.scatter(x + 0.04 * jitter.normal(size=values.size), values, color=colour, s=20,
                   label=f"{label} (n={values.size})")
    ax.axhline(FsmConfig().verify_load_rise, color="#57606a", linestyle="--",
               label=f"threshold {FsmConfig().verify_load_rise}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["seated", "not seated"])
    ax.set_ylabel("rise in peak |load| during the tug (duty)")
    ax.set_title("VERIFY: a 1.5 mm upward tug at half the torque limit", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "verify_tug.png"
    fig.savefig(path)
    plt.close(fig)
    return {"verify_tug.png": str(path)}
