"""Arm bring-up self-test: ping, identity and safety registers, a torque-off range read, one
small commanded move per joint -- a pass/fail table.

Runs against the virtual bus today and the real bus the day the arm arrives, the same code and the
same table either way. This is the longer report a teammate reads once, by hand, before trusting
the bus at all; it does not go around :meth:`~brick_builder.hardware.bus.Bus.enable_torque`'s own
gate (voltage window, temperature, a missing or unhealthy servo) -- it calls that gate, and reports
what it found alongside everything else.

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
    a settling-accuracy measurement; that is ``servo_workspace_map.py``'s job.
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
    joint's check, not a synchronized whole-arm move. That matters because this runs right after
    that one joint's own torque-off range read, while every joint still to be range-read is still
    wherever the scene reset left it (its own moderate rest pose) and every joint already checked
    has already settled near its own moderate pose too: at most one joint is ever away from a
    moderate pose when this runs, which is what keeps the load on it inside what its own P gain and
    torque budget can serve. Running the whole arm's range read first and only then trying to bring
    six simultaneously-extended joints back at once is not safe -- it can leave more than one joint
    unable to hold anywhere near its target, saturated at the torque limit indefinitely.

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

    The range read and the commanded move run **one joint at a time**, interleaved (`_commanded_
    move`'s docstring says why): joint ``k``'s range read, then -- through
    :meth:`~brick_builder.hardware.bus.Bus.enable_torque`'s own gate, exactly the way every other
    caller goes through it, not around it -- its commanded move, before joint ``k + 1``'s range
    read starts. At most one joint is ever away from a moderate pose at a time.
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
        finally:
            bus.disable_torque()

    return SelfTestReport(rows, config)
