"""Arm bring-up self-test: ping, identity and safety registers, a torque-off range read, one
small commanded move per joint -- a pass/fail table.

Runs against the virtual bus today and the real bus the day the arm arrives, the same code and the
same table either way.  This is a longer report than :meth:`~brick_builder.hardware.bus.Bus.self_test`
(the gate ``enable_torque`` runs before energizing anything): it is what a teammate runs once, by
hand, before trusting the bus at all.

Six checks per joint
---------------------
``ping``
    The servo answers its own id.
``registers``
    Model number (``servo_major``/``servo_minor``), firmware version (``firmware_major``/
    ``firmware_minor``), supply voltage, temperature, the P gain, both dead-band registers and
    both angle-limit registers -- read once each, nothing inferred.
``range read`` (torque off)
    ``Present_Position`` tracks the joint smoothly across its own calibrated travel while torque
    is disabled, i.e. the encoder and the protocol work over the whole mechanical range and not
    just near the servo's own zero.  Against the virtual bus this is driven directly (the
    equivalent of a technician backdrifting the joint by hand): the scene's joint is set, not
    commanded, and :meth:`~brick_builder.hardware.virtual_bus.SimulatedServos.resync` pushes the result
    into the register file the way a real encoder would.  Against real hardware there is no way to
    move the joint by hand from software, so the check instead polls ``Present_Position`` with
    torque off for :attr:`SelfTestConfig.range_poll_s` while a person moves the joint, and passes
    when the bus stays healthy and every reading is inside the joint's own calibrated limits --
    coverage of the range is the person's job, not the script's.
``commanded move``
    Torque on, every joint driven to 0 rad and settled (a known baseline -- the range check above
    can leave a joint anywhere in its travel), then one small step
    (:attr:`SelfTestConfig.move_amplitude_rad`, about 5 degrees) per joint, and the reading has to
    have moved at least :attr:`SelfTestConfig.move_min_fraction` of the way there in the commanded
    direction.  That bar is deliberately low (10 %, not "close to the target"): with no gravity
    feed-forward (this drives ``Bus.cycle`` directly, the way ``servo_workspace_map.py`` does, not
    ``RealArmWorld``, which adds it) a gravity-loaded joint's own P-gain droop can be several times
    a 5-degree step even at 0 rad -- ``shoulder_lift`` reaches only 16 % of a 4.6 degree step there
    on the virtual bus.  This check is "does the loop respond at all", not a settling-accuracy
    measurement; that is ``servo_workspace_map.py``'s job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..hardware.bus import Bus, BusError
from ..hardware.sts3215 import DEG_PER_TICK, JOINT_NAMES, ProtocolError, describe_error

N_JOINTS = 6


@dataclass(frozen=True)
class SelfTestConfig:
    """Amplitudes and tolerances for the two active checks.  Nothing here is a datasheet limit --
    they are this script's own pass/fail thresholds, generous on purpose: this is a smoke check,
    not the settling-error measurement (``servo_workspace_map.py``)."""

    move_amplitude_rad: float = 0.08          # ~4.6 deg
    move_settle_ticks: int = 40                # 0.8 s at 50 Hz
    move_min_fraction: float = 0.10            # of the commanded delta; see the module docstring
    range_points: int = 5                      # per joint, torque-off virtual-bus range read
    range_margin_rad: float = math.radians(2.0)     # stay this far off each mechanical stop
    range_tol_ticks: float = 3.0               # encoder counts the readback may be off by
    range_poll_s: float = 3.0                  # real hardware: how long to watch Present_Position


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
            "| id | joint | ping | model | firmware | V | C | P gain | dead band [ticks] | "
            "range read | commanded move | pass |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for j in self.joints:
            model = "-" if j.model_number is None else f"{j.model_number[0]}.{j.model_number[1]}"
            fw = ("-" if j.firmware_version is None
                  else f"{j.firmware_version[0]}.{j.firmware_version[1]}")
            db = "-" if j.dead_band_ticks is None else f"{j.dead_band_ticks[0]}/{j.dead_band_ticks[1]}"
            volt = "-" if math.isnan(j.voltage_v) else f"{j.voltage_v:.1f}"
            lines.append(
                f"| {j.servo_id} | {j.joint} | {'yes' if j.present else 'NO'} | {model} | {fw} | "
                f"{volt} | {j.temperature_c} | {j.p_gain} | {db} | {mark(j.range_read_ok)} | "
                f"{mark(j.commanded_move_ok)} | {'PASS' if j.ok else 'FAIL'} |"
            )
        problems = [(j, p) for j in self.joints for p in j.problems]
        if problems:
            lines.append("")
            for j, p in problems:
                lines.append(f"- id {j.servo_id} ({j.joint}): {p}")
        lines.append("")
        lines.append(f"overall: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------ the six checks ---


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


def _range_read_real(bus: Bus, index: int, row: JointSelfTest, cfg: SelfTestConfig) -> None:
    """Torque-off range read against real hardware: watch, do not drive.

    Nothing here moves the joint -- that is the point, torque is off -- so it polls
    ``Present_Position`` for :attr:`SelfTestConfig.range_poll_s` while a person moves it by hand,
    and passes when the bus stays healthy and every reading lands inside the joint's own
    calibrated limits.  How much of the range that covers is on the person, not the script.
    """
    import time

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
    row.range_read_ok = True
    span_deg = 0.0 if seen_lo is None else (seen_hi - seen_lo) * DEG_PER_TICK
    row.range_read_detail = f"{samples} reads over {cfg.range_poll_s:.1f} s, moved {span_deg:.1f} deg"


def _commanded_move(bus: Bus, home: np.ndarray, index: int, row: JointSelfTest,
                    cfg: SelfTestConfig) -> None:
    start = float(home[index])
    goal = home.copy()
    goal[index] = start + cfg.move_amplitude_rad
    state = None
    for _ in range(cfg.move_settle_ticks):
        state = bus.cycle(goal)
    if state is None or not state.valid[index]:
        row.commanded_move_ok = False
        row.commanded_move_detail = "no reply while commanded"
        return
    moved = float(state.position[index]) - start
    fraction = moved / cfg.move_amplitude_rad if cfg.move_amplitude_rad else 0.0
    row.commanded_move_ok = bool(fraction >= cfg.move_min_fraction)
    row.commanded_move_detail = (
        f"target {math.degrees(cfg.move_amplitude_rad):.1f} deg, "
        f"moved {math.degrees(moved):.2f} deg ({100.0 * fraction:.0f} % of commanded)")


# ------------------------------------------------------------------------------------- the run ---


def run_self_test(bus: Bus, *, vbus=None, scene=None,
                  config: SelfTestConfig = SelfTestConfig()) -> SelfTestReport:
    """Ping, registers, torque-off range read, one commanded move -- every joint, one report.

    ``vbus``/``scene`` (the :class:`~brick_builder.hardware.virtual_bus.VirtualBus` and the scene
    behind it) are given only when ``bus`` is talking to the virtual bus; without them the range
    check falls back to the real-hardware poll.  Nothing here enables torque on its own beyond the
    one small move -- :meth:`~brick_builder.hardware.bus.Bus.enable_torque` is the caller's job,
    same as it always is.
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
            continue
        try:
            if vbus is not None and scene is not None:
                _range_read_virtual(bus, k, row, vbus=vbus, scene=scene, cfg=config)
            else:
                _range_read_real(bus, k, row, config)
        except (BusError, ProtocolError) as exc:
            row.range_read_ok = False
            row.range_read_detail = str(exc)
        rows.append(row)

    # The commanded move needs torque, which every joint gets together (a sync-write).  The range
    # check (virtual or real) can leave a joint anywhere in its travel, including poses with
    # substantial gravity torque, and this check is a smoke test of the closed loop, not the
    # settling-accuracy measurement (`servo_workspace_map.py`) -- so torque comes on, and every
    # joint is driven to 0 rad and given a full settle before the small moves start, the same
    # known, moderate-torque baseline every run.
    result = bus.enable_torque(check=False)
    if not result.ok:
        for row in rows:
            if row.present:
                row.problems.append("torque not enabled: " + "; ".join(result.problems))
        return SelfTestReport(rows, config)
    home = np.zeros(6)
    for _ in range(config.move_settle_ticks):
        bus.cycle(home)
    for k, row in enumerate(rows):
        if not row.present:
            continue
        try:
            _commanded_move(bus, home, k, row, config)
        except (BusError, ProtocolError) as exc:
            row.commanded_move_ok = False
            row.commanded_move_detail = str(exc)
    bus.disable_torque()
    return SelfTestReport(rows, config)
