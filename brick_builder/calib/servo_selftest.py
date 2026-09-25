"""Arm bring-up self-test: ping, identity and safety registers, a torque-off range read, one
small commanded move per joint -- a pass/fail table.

Runs against the virtual bus today and the real bus the day the arm arrives, the same code and the
same table either way. This is the longer report a teammate reads once, by hand, before trusting
the bus at all; it does not go around :meth:`~brick_builder.hardware.bus.Bus.enable_torque`'s own
gate (voltage window, temperature, a missing or unhealthy servo) -- it calls that gate, and reports
what it found alongside everything else.

``Bus.disable_torque()`` is bus-wide: every call (before the first range read, and after each
joint's own commanded move) drops holding torque on all six servos at once, not just the one under
test. Against the virtual bus this costs at most one 20 ms free-fall tick, because nothing here
steps physics except a written goal; on the real arm every one of those calls leaves the whole arm
limp. Support it for the whole run, not only during the range reads.

Five checks per joint
---------------------
``ping``
    The servo answers its own id.
``registers``
    Model number (``servo_major``/``servo_minor``), firmware version (``firmware_major``/
    ``firmware_minor``), supply voltage, temperature, the P gain, both dead-band registers and
    both angle-limit registers -- read once each, nothing inferred.
``gate`` (voltage, temperature, status)
    The same bounds :meth:`Bus.enable_torque` checks before it will energize anything --
    :attr:`~brick_builder.hardware.bus.BusSettings.min_voltage_v`/``max_voltage_v``,
    ``max_temperature_c``, a zero status byte. A joint that fails here is marked FAIL and the
    commanded-move step below does not run for any joint (torque enables all six together).
``range read`` (torque off)
    ``Present_Position`` tracks the joint smoothly across its own calibrated travel while torque
    is disabled, i.e. the encoder and the protocol work over the whole mechanical range and not
    just near the servo's own zero. Torque is written off before this step even starts, so a
    servo left energized from a previous session does not fight the read. Against the virtual bus
    this is driven directly (the equivalent of a technician back-driving the joint by hand): the
    scene's joint is set, not commanded, and
    :meth:`~brick_builder.hardware.virtual_bus.SimulatedServos.resync` pushes the result into the
    register file the way a real encoder would. Against real hardware there is no way to move the
    joint by hand from software, so the check instead polls ``Present_Position`` with torque off
    for :attr:`SelfTestConfig.range_poll_s` while a person moves the joint (announced by name), and
    passes when the bus stays healthy, every reading is inside the joint's own calibrated limits,
    and the readings actually span at least :attr:`SelfTestConfig.range_poll_min_span_deg` --
    coverage of the range is the person's job, the span check is this script's.
``commanded move``
    Torque on, then one small step (:attr:`SelfTestConfig.move_amplitude_rad`, about 5 degrees) per
    joint, *from wherever the range read left it* -- the range read can leave a joint anywhere in
    its travel, so the baseline this checks against is the position actually read back just before
    the step, not an assumed common pose, and the step is aimed back toward the middle of the
    joint's own travel so it never asks a joint already near one stop to run further past it. The
    reading has to have moved between :attr:`SelfTestConfig.move_min_fraction` and
    :attr:`SelfTestConfig.move_max_fraction` of the commanded delta, settle there (the last few
    ticks' spread under :attr:`SelfTestConfig.move_settle_tol_rad`), and never trip the servo
    model's overload latch. This is "does the loop respond, settle and stay within its rating", not
    a settling-accuracy measurement; that is ``servo_workspace_map.py``'s job. Afterward (not
    scored) the joint is driven the rest of the way to the centre of its own travel before the next
    joint's range read starts, so a joint already checked is not still sitting near its own
    mechanical stop while a neighbour is swept through its full range -- see ``run_self_test``'s
    docstring.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..hardware.bus import Bus, BusError, BusState
from ..hardware.sts3215 import DEG_PER_TICK, JOINT_NAMES, ProtocolError, describe_error

N_JOINTS = 6


@dataclass(frozen=True)
class SelfTestConfig:
    """Amplitudes and tolerances for the active checks. Nothing here is a datasheet limit --
    they are this script's own pass/fail thresholds, generous on purpose: this is a smoke check,
    not the settling-error measurement (``servo_workspace_map.py``)."""

    move_amplitude_rad: float = 0.08          # ~4.6 deg
    move_settle_ticks: int = 40                # 0.8 s at 50 Hz
    move_settle_avg_ticks: int = 10            # ticks at the end the settled reading is over
    move_min_fraction: float = 0.10            # of the commanded delta, lower bound
    move_max_fraction: float = 1.5             # of the commanded delta, upper bound (no runaway)
    move_settle_tol_rad: float = math.radians(1.0)   # std of the last few ticks, "has it stopped"
    range_points: int = 5                      # per joint, torque-off virtual-bus range read
    range_margin_rad: float = math.radians(2.0)     # stay this far off each mechanical stop
    range_tol_ticks: float = 3.0               # encoder counts the readback may be off by
    range_poll_s: float = 3.0                  # real hardware: how long to watch Present_Position
    range_poll_min_span_deg: float = 5.0       # real hardware: least motion the poll has to see


@dataclass
class JointSelfTest:
    """One servo's row."""

    servo_id: int
    joint: str
    present: bool = False
    model_number: tuple[int, int] | None = None
    firmware_version: tuple[int, int] | None = None
    voltage_v: float = math.nan
    temperature_c: int = -1
    voltage_ok: bool | None = None
    temperature_ok: bool | None = None
    p_gain: int = -1
    angle_limit_ticks: tuple[int, int] | None = None
    dead_band_ticks: tuple[int, int] | None = None
    range_read_ok: bool | None = None
    range_read_detail: str = ""
    commanded_move_ok: bool | None = None
    commanded_move_detail: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if not self.present or self.problems:
            return False
        if self.voltage_ok is False or self.temperature_ok is False:
            return False
        return self.range_read_ok is not False and self.commanded_move_ok is not False


@dataclass
class SelfTestReport:
    joints: list[JointSelfTest]
    config: SelfTestConfig = field(default_factory=SelfTestConfig)

    @property
    def ok(self) -> bool:
        return bool(self.joints) and all(j.ok for j in self.joints)

    def table(self) -> str:
        def mark(v: bool | None) -> str:
            return "-" if v is None else ("pass" if v else "FAIL")

        lines = [
            "arm self-test", "",
            "| id | joint | ping | model | firmware | Supply [V] | Temperature [°C] | "
            "P gain | dead band [ticks] | angle limit [ticks] | range read | commanded move | "
            "pass |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for j in self.joints:
            model = "-" if j.model_number is None else f"{j.model_number[0]}.{j.model_number[1]}"
            fw = ("-" if j.firmware_version is None
                  else f"{j.firmware_version[0]}.{j.firmware_version[1]}")
            db = "-" if j.dead_band_ticks is None else f"{j.dead_band_ticks[0]}/{j.dead_band_ticks[1]}"
            limits = "-" if j.angle_limit_ticks is None else f"{j.angle_limit_ticks[0]}/{j.angle_limit_ticks[1]}"
            volt = "-" if math.isnan(j.voltage_v) else f"{j.voltage_v:.1f}"
            lines.append(
                f"| {j.servo_id} | {j.joint} | {'yes' if j.present else 'NO'} | {model} | {fw} | "
                f"{volt} | {j.temperature_c} | {j.p_gain} | {db} | {limits} | "
                f"{mark(j.range_read_ok)} | {mark(j.commanded_move_ok)} | "
                f"{'PASS' if j.ok else 'FAIL'} |"
            )
        problems = [(j, p) for j in self.joints for p in j.problems]
        if problems:
            lines.append("")
            for j, p in problems:
                lines.append(f"- id {j.servo_id} ({j.joint}): {p}")
        lines.append("")
        lines.append(f"overall: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- the five checks ---


def _read_identity(bus: Bus, servo_id: int, row: JointSelfTest) -> None:
    row.model_number = (bus.read_register(servo_id, "servo_major"),
                        bus.read_register(servo_id, "servo_minor"))
    row.firmware_version = (bus.read_register(servo_id, "firmware_major"),
                            bus.read_register(servo_id, "firmware_minor"))
    state = bus.read_state(servo_id)
    row.voltage_v = state.voltage_v
    row.temperature_c = state.temperature_c
    row.p_gain = bus.read_register(servo_id, "p_coefficient")
    row.angle_limit_ticks = (bus.read_register(servo_id, "min_angle_limit"),
                             bus.read_register(servo_id, "max_angle_limit"))
    row.dead_band_ticks = (bus.read_register(servo_id, "cw_dead_band"),
                           bus.read_register(servo_id, "ccw_dead_band"))
    row.voltage_ok = bus.settings.min_voltage_v <= state.voltage_v <= bus.settings.max_voltage_v
    row.temperature_ok = state.temperature_c <= bus.settings.max_temperature_c
    if not row.voltage_ok:
        row.problems.append(
            f"{state.voltage_v:.1f} V outside {bus.settings.min_voltage_v:.1f}-"
            f"{bus.settings.max_voltage_v:.1f} V")
    if not row.temperature_ok:
        row.problems.append(
            f"{state.temperature_c} C above {bus.settings.max_temperature_c} C")
    if state.status:
        row.problems.append(f"status byte {describe_error(state.status)}")


def _range_read_virtual(bus: Bus, index: int, row: JointSelfTest, *, vbus, scene,
                        cfg: SelfTestConfig) -> None:
    """Torque-off range read against the virtual bus: pose the scene directly, read it back."""
    cal = bus.calibration[index]
    lo, hi = min(cal.lo, cal.hi), max(cal.lo, cal.hi)
    lo += cfg.range_margin_rad
    hi -= cfg.range_margin_rad
    if not hi > lo:
        row.range_read_ok = None
        row.range_read_detail = "joint range too narrow for the margin"
        return
    servo = vbus.servos.servo
    qadr = int(servo.qpos_adr[index])
    padr = int(servo.play_qpos_adr[index]) if servo.has_play else None
    worst_ticks = 0.0
    tested = np.linspace(lo, hi, cfg.range_points)
    import mujoco

    for angle in tested:
        scene.data.qpos[qadr] = float(angle)
        if padr is not None:
            scene.data.qpos[padr] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        vbus.servos.resync()
        pos_ticks, _, _, valid = bus.read_ticks()
        if not valid[index]:
            row.range_read_ok = False
            row.range_read_detail = f"no reply at {math.degrees(angle):.1f} deg"
            return
        want_ticks = cal.to_ticks(angle)
        worst_ticks = max(worst_ticks, abs(pos_ticks[index] - want_ticks))
    row.range_read_ok = bool(worst_ticks <= cfg.range_tol_ticks)
    row.range_read_detail = (f"{cfg.range_points} points over "
                             f"{math.degrees(hi - lo):.0f} deg, worst {worst_ticks:.1f} ticks")


def _range_read_real(bus: Bus, index: int, row: JointSelfTest, cfg: SelfTestConfig,
                     announce: Callable[[str], None] | None = None) -> None:
    """Torque-off range read against real hardware: watch, do not drive.

    Nothing here moves the joint -- that is the point, torque is off -- so it announces the joint
    by name, then polls ``Present_Position`` for :attr:`SelfTestConfig.range_poll_s` while a person
    moves it by hand. It passes when the bus stays healthy, every reading lands inside the joint's
    own calibrated limits, and the readings span at least
    :attr:`SelfTestConfig.range_poll_min_span_deg` -- a PASS with no motion seen is refused rather
    than silently accepted, because "the person's job" still needs a way to fail if it did not
    happen.
    """
    if announce is not None:
        announce(f"move {row.joint} through its range now "
                 f"({cfg.range_poll_s:.0f} s, torque is off)")
    cal = bus.calibration[index]
    lo, hi = min(cal.lo_ticks, cal.hi_ticks), max(cal.lo_ticks, cal.hi_ticks)
    deadline = time.monotonic() + cfg.range_poll_s
    seen_lo = seen_hi = None
    misses = 0
    samples = 0
    while time.monotonic() < deadline:
        pos_ticks, _, _, valid = bus.read_ticks()
        samples += 1
        if not valid[index]:
            misses += 1
        else:
            t = pos_ticks[index]
            seen_lo = t if seen_lo is None else min(seen_lo, t)
            seen_hi = t if seen_hi is None else max(seen_hi, t)
            if not (lo - cfg.range_tol_ticks <= t <= hi + cfg.range_tol_ticks):
                row.range_read_ok = False
                row.range_read_detail = f"reading {t} outside calibrated limits [{lo}, {hi}]"
                return
        time.sleep(0.02)
    if samples == 0 or misses == samples:
        row.range_read_ok = False
        row.range_read_detail = "no valid reply during the poll"
        return
    span_deg = 0.0 if seen_lo is None else (seen_hi - seen_lo) * DEG_PER_TICK
    row.range_read_ok = bool(span_deg >= cfg.range_poll_min_span_deg)
    row.range_read_detail = f"{samples} reads over {cfg.range_poll_s:.1f} s, moved {span_deg:.1f} deg"
    if not row.range_read_ok:
        row.range_read_detail += f" (< {cfg.range_poll_min_span_deg:.0f} deg required -- move it further)"


def _commanded_move(bus: Bus, index: int, row: JointSelfTest, cfg: SelfTestConfig) -> None:
    """One small step on *one* joint, from its own settled reading, every other joint held fixed.

    Torque is bus-wide (one sync-write moves whatever the goal vector says for all six), so every
    other joint's goal is its own current reading, unchanged, for the whole call -- this is one
    joint's check, not a synchronized whole-arm move.

    The range read leaves this joint at the last point of its own sweep, close to one mechanical
    stop, not at a moderate pose -- so *this* check always starts a gravity-loaded joint
    (``shoulder_lift``, ``elbow_flex``, ``wrist_flex``) from wherever its own travel is hardest to
    hold with no gravity feed-forward, with the rest of the arm still wherever it was left. That is
    by design (a range-off-the-extreme check is the point), and it means this check can genuinely
    FAIL here, honestly, on those three joints when the arm's own weight at that particular extreme
    is close to what a proportional-only loop can serve: against the virtual bus at seed 0,
    ``shoulder_lift`` and ``elbow_flex`` both do (see guide 08's table); ``wrist_flex`` passes but
    close to the check's own upper bound. :func:`_park_after_check`, right after this, keeps that
    from compounding onto the *next* joint's check: it drives this joint back toward the centre of
    its own travel (when it still can) before joint ``k + 1``'s range read starts, so a joint
    already checked is not left sitting near its own extreme while a neighbour is swept through its
    full range right next to it -- without that step, one joint saturated near its own stop could
    drag a downstream joint past its own calibrated limit through the kinematic chain during what
    was meant to be someone else's small, isolated step. Parking does not, and cannot, rescue a
    joint that is already saturated at its own extreme by the arm's own weight; it only stops that
    joint from also poisoning the check after it.

    The range read can leave a joint somewhere torque was never applied to hold it (a pose set
    directly on the virtual bus, or wherever a person's hand left it on real hardware), so the
    first thing that happens once torque comes on is that joint arresting whatever fall gravity
    alone would otherwise give it -- a real transient, not an artifact of this test, and not what
    the small step below should be measured through. A first settle phase (goal held at the reading
    torque came on with) runs before the baseline is read, so that transient is caught and counted
    toward the overload check without being read as "the commanded move."
    """
    catch_ticks, _, _, valid0 = bus.read_ticks()
    catch_goal = np.array([bus.calibration[k].to_rad(t) for k, t in enumerate(catch_ticks)])
    tripped = False
    state = None
    for _ in range(cfg.move_settle_ticks):
        state = bus.cycle(catch_goal)
        tripped = tripped or bool(state.tripped[index])
    if not valid0[index] or not state.valid[index]:
        row.commanded_move_ok = False
        row.commanded_move_detail = "no reply before the step"
        return
    start = float(state.position[index])

    cal = bus.calibration[index]
    lo, hi = min(cal.lo, cal.hi), max(cal.lo, cal.hi)
    centre = 0.5 * (lo + hi)
    direction = 1.0 if start <= centre else -1.0
    goal = catch_goal.copy()
    goal[index] = start + direction * cfg.move_amplitude_rad

    history: list[BusState] = []
    for tick in range(cfg.move_settle_ticks):
        state = bus.cycle(goal)
        tripped = tripped or bool(state.tripped[index])
        if tick >= cfg.move_settle_ticks - cfg.move_settle_avg_ticks:
            history.append(state)
    if not state.valid[index]:
        row.commanded_move_ok = False
        row.commanded_move_detail = "no reply while commanded"
        return

    settled = float(np.mean([h.position[index] for h in history]))
    settled_std = float(np.std([h.position[index] for h in history]))
    moved = direction * (settled - start)
    fraction = moved / cfg.move_amplitude_rad if cfg.move_amplitude_rad else 0.0
    settled_ok = settled_std <= cfg.move_settle_tol_rad
    overload_ok = not tripped
    row.commanded_move_ok = bool(
        cfg.move_min_fraction <= fraction <= cfg.move_max_fraction
        and settled_ok and overload_ok)
    detail = (f"target {math.degrees(cfg.move_amplitude_rad):.1f} deg from the measured start, "
             f"moved {math.degrees(moved):.2f} deg ({100.0 * fraction:.0f} % of commanded)")
    if not settled_ok:
        detail += f", not settled (spread {math.degrees(settled_std):.3f} deg)"
    if not overload_ok:
        detail += ", overload protection tripped"
    row.commanded_move_detail = detail


def _park_after_check(bus: Bus, index: int, cfg: SelfTestConfig) -> None:
    """Drive joint ``index`` toward the centre of its own travel, every other joint's goal held at
    its own current reading -- the same one-joint-moves shape as :func:`_commanded_move`, just
    aimed at the middle of the range instead of a small step off wherever the range read left it.

    Not one of the five checks (nothing here is scored, and a joint that has not reached the centre
    by ``move_settle_ticks`` is not reported as a problem): the goal is only to leave a joint that
    can hold it somewhere moderate before the next joint's range read starts, not to guarantee every
    joint gets there. A joint whose own commanded-move check already found it saturated at its own
    extreme (:func:`_commanded_move`'s docstring) generally stays saturated here too -- this does
    not rescue that joint, it only stops asking it to travel further once the servo model's own
    overload latch trips, the same protection every other move in this module respects.
    """
    ticks, _, _, valid = bus.read_ticks()
    if not valid[index]:
        return
    goal = np.array([bus.calibration[k].to_rad(t) for k, t in enumerate(ticks)])
    cal = bus.calibration[index]
    lo, hi = min(cal.lo, cal.hi), max(cal.lo, cal.hi)
    goal[index] = 0.5 * (lo + hi)
    for _ in range(cfg.move_settle_ticks):
        state = bus.cycle(goal)
        if bool(state.tripped[index]):
            return


# ------------------------------------------------------------------------------------- the run ---


def run_self_test(bus: Bus, *, vbus=None, scene=None,
                  config: SelfTestConfig = SelfTestConfig(),
                  announce: Callable[[str], None] | None = None) -> SelfTestReport:
    """Ping, registers, torque-off range read, one commanded move -- every joint, one report.

    ``vbus``/``scene`` (the :class:`~brick_builder.hardware.virtual_bus.VirtualBus` and the scene
    behind it) are given only when ``bus`` is talking to the virtual bus; without them the range
    check falls back to the real-hardware poll, which uses ``announce`` (if given) to name the
    joint to move and how long. Torque is written off before anything else runs, so a servo left
    energized from a previous session does not fight the range read.

    The range read and the commanded move run **one joint at a time**, interleaved: joint ``k``'s
    range read, then -- through :meth:`~brick_builder.hardware.bus.Bus.enable_torque`'s own gate,
    exactly the way every other caller goes through it, not around it -- its commanded move, then
    :func:`_park_after_check` drives it back toward the centre of its own travel, before joint
    ``k + 1``'s range read starts.

    That park step stops an already-checked joint from *also* sitting near its own extreme while a
    later joint is swept through its full range next to it, which used to let one joint's control
    loop saturate, drag a neighbour through the kinematic chain, and drive it past its own
    calibrated limit during what was meant to be a small, isolated step. It does not, and cannot,
    make every joint's own check pass: a gravity-loaded joint (``shoulder_lift``, ``elbow_flex``,
    ``wrist_flex``) is tested at its own range extreme *before* it has been parked, with whatever
    weight the rest of the arm hangs on it there, and no gravity feed-forward is applied anywhere in
    this check (:func:`_commanded_move`'s docstring) -- so that joint's own commanded-move check can
    genuinely fail, honestly, on the arm's own weight at that particular extreme, independent of
    joint order. Against the virtual bus at seed 0 this happens on two of the three gravity-loaded
    joints (guide 08's table has the numbers); it is not a bug in the check, and on real hardware it
    means that joint sagging or driving toward its own stop under load it cannot hold -- support the
    arm through each gravity-loaded joint's range read and commanded move rather than treating a
    FAIL there as routine.
    """
    rows: list[JointSelfTest] = []
    for k, (servo_id, joint) in enumerate(zip(bus.ids, JOINT_NAMES)):
        row = JointSelfTest(servo_id=servo_id, joint=joint)
        row.present = bus.ping(servo_id)
        if not row.present:
            rows.append(row)
            continue
        try:
            _read_identity(bus, servo_id, row)
        except (BusError, ProtocolError) as exc:
            row.problems.append(f"register read failed: {exc}")
        rows.append(row)

    # Torque off before anything else touches the bus: the range read assumes the joint is free to
    # move (by hand, or by the virtual bus's direct scene poke), and a servo can come up with
    # torque still latched from a previous session.
    bus.disable_torque()

    for k, row in enumerate(rows):
        if not row.present or row.problems:
            continue
        try:
            if vbus is not None and scene is not None:
                _range_read_virtual(bus, k, row, vbus=vbus, scene=scene, cfg=config)
            else:
                _range_read_real(bus, k, row, config, announce=announce)
        except (BusError, ProtocolError) as exc:
            row.range_read_ok = False
            row.range_read_detail = str(exc)

        # The commanded move needs torque, gated exactly the way `Bus.enable_torque` always gates
        # it (voltage, temperature, a missing or unhealthy servo anywhere on the bus -- torque is
        # bus-wide, so one bad servo blocks every joint's move, the same as it would on the wire).
        try:
            bus.enable_torque()
        except BusError as exc:
            for other in rows:
                if other.present:
                    other.problems.append(f"torque not enabled: {exc}")
            return SelfTestReport(rows, config)
        try:
            _commanded_move(bus, k, row, config)
            _park_after_check(bus, k, config)
        finally:
            bus.disable_torque()

    return SelfTestReport(rows, config)
