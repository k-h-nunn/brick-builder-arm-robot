"""Serial-bus driver for six STS3215 servos: one sync-write and one sync-read per control tick.

``Bus`` owns a port and the protocol in :mod:`brick_builder.hardware.sts3215`.  It is deliberately
free of MuJoCo and of the rest of the package: the only things it needs are a serial port, the
memory table and a per-joint calibration, so the same object drives the virtual bus over a
pseudo-terminal and the real arm over the kit's USB adapter board.

One tick
--------
:meth:`Bus.cycle` is the 50 Hz loop body::

    state = bus.cycle(goal_rad)     # sync-write Goal_Position, sync-read 56-70

It clips every goal to that joint's travel limits, converts to encoder counts, sends one
sync-write, then one sync-read of addresses 56-70 and decodes the six replies.  ``state.rtt_s``
is the measured wall-clock time from the first byte of the sync-write to the last byte of the
last reply, which is the number to compare against the 20 ms control period.

Safety
------
Four layers, all in the driver rather than in the caller:

* **Angle limits.**  Every goal is clipped to the joint's own travel range before it reaches the
  wire; nothing the caller does can command a servo past its stop.
* **Torque limit.**  ``Torque_Limit`` (48) is a RAM register and is written as a fraction of the
  servo's own limit, per joint, every tick a caller asks for compliance.
* **Soft stop.**  :meth:`disable_torque` writes ``Torque_Enable`` (40) = 0 to all six, which drops
  the arm's holding torque immediately without cutting power to the bus.
* **Startup self-test.**  :meth:`self_test` pings all six, reads voltage and temperature from each
  and refuses to enable torque when a servo is missing, the supply is outside its window or a
  servo is already hot.  :meth:`enable_torque` runs it first and raises rather than energizing an
  arm whose bus is half-connected.

Function-pointer contract
-------------------------
The microcontroller's own bus stub is the pair ``bus_write(targets[6])`` and
``bus_read(pos[6], speed[6], load[6])``, both in encoder counts.  :meth:`write_ticks` and
:meth:`read_ticks` are those two functions on this side of the line; everything in radians is a
layer above them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

import numpy as np

from . import sts3215 as proto
from .sts3215 import (
    JointCalibration,
    ProtocolError,
    SERVO_IDS,
    ServoState,
    StatusError,
    so101_calibration,
)

N_SERVOS = 6
DEFAULT_BAUD = 1_000_000
CONTROL_PERIOD_S = 0.02
"""50 Hz, the same control period ``control/pick_place.py`` runs the simulator at."""


class Transport(Protocol):
    """What :class:`Bus` needs from a port.  ``serial.Serial`` satisfies it as it stands."""

    def write(self, data: bytes) -> int | None: ...
    def read(self, size: int) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class BusSettings:
    """Port, timing and safety limits.  Every field is a documented measurement or a datasheet."""

    baud: int = DEFAULT_BAUD
    """Line rate.  1 Mbps is the STS3215 factory default and what the wire budget in
    ``robot/bus_emulator.py`` assumes (260 us sync-write, 1.4 ms sync-read for six servos)."""

    timeout_s: float = 0.006
    """How long one reply may take before it counts as lost.  Six replies plus turnarounds cost
    about 3.1 ms at 1 Mbps, so 6 ms leaves room without spilling past a 20 ms tick."""

    retries: int = 2
    """Re-sends of a *read* after a timeout or a bad checksum.  A sync-write is never retried:
    the next tick carries a fresher goal anyway."""

    sync_read: bool = True
    """Use SYNC READ (0x82).  ``False`` falls back to one READ per servo, for firmware that does
    not implement it -- six round trips instead of one, measured by ``measure_cycle_time``."""

    echo: bool = False
    """Set when the adapter board echoes what the host transmits back into the receiver.  The
    kit's board does not; some half-duplex adapters do, and this discards the echo."""

    min_voltage_v: float = 6.0
    max_voltage_v: float = 8.4
    """Supply window the self-test enforces.  The arm is specified at 7.4 V; below 6.0 V the
    servo's own low-voltage protection is near, and 8.4 V is a fully charged 2S pack."""

    max_temperature_c: int = 55
    """Refuse to energize above this.  The servo's own limit (register 13) ships at 70."""

    torque_limit: float = 1.0
    """Default ``Torque_Limit`` as a fraction of the servo's own limit."""

    goal_speed_counts: int = 0
    """``Goal_Speed`` (46) written at startup.  0 is the register's "no limit" value, which is
    what a position loop driven at 50 Hz wants: the host's own trajectory sets the speed."""

    settle_s: float = 0.0
    """Extra delay after a write before the next instruction, for a bus with a slow adapter."""


@dataclass
class BusCounters:
    """Everything that can go wrong on the wire, counted rather than raised."""

    instructions: int = 0
    replies: int = 0
    timeouts: int = 0
    checksum_errors: int = 0
    framing_errors: int = 0
    retries: int = 0
    resync_bytes: int = 0
    status_errors: int = 0
    missing_replies: int = 0

    @property
    def errors(self) -> int:
        return (self.timeouts + self.checksum_errors + self.framing_errors
                + self.missing_replies)

    def as_dict(self) -> dict[str, int]:
        return {
            "instructions": self.instructions,
            "replies": self.replies,
            "timeouts": self.timeouts,
            "checksum_errors": self.checksum_errors,
            "framing_errors": self.framing_errors,
            "retries": self.retries,
            "resync_bytes": self.resync_bytes,
            "status_errors": self.status_errors,
            "missing_replies": self.missing_replies,
            "errors": self.errors,
        }


@dataclass
class BusState:
    """One sync-read, in engineering units, plus how long the exchange took."""

    position: np.ndarray        # [rad], per joint
    speed: np.ndarray           # [rad/s]
    load: np.ndarray            # PWM duty in [-1, 1] -- not a torque
    current: np.ndarray         # [A]
    voltage: np.ndarray         # [V]
    temperature: np.ndarray     # [deg C]
    moving: np.ndarray          # bool
    status: np.ndarray          # the servo's own error byte
    valid: np.ndarray           # bool: False where the reply was lost and the value is held
    ticks: np.ndarray           # raw Present_Position counts
    rtt_s: float = 0.0
    t_mono: float = 0.0

    def copy(self) -> "BusState":
        return BusState(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in self.__dict__.items()})

    @property
    def tripped(self) -> np.ndarray:
        """Per joint: has the servo latched its own overload protection (status bit 5)?

        Named to match ``ServoReadings.tripped``, so ``control.primitives.feedback_from_readings``
        takes a :class:`BusState` and a simulated reading interchangeably."""
        return (self.status & 0x20) != 0


@dataclass
class SelfTestResult:
    """What :meth:`Bus.self_test` found.  ``ok`` is the only thing torque enabling looks at."""

    ok: bool
    present: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    voltage_v: dict[int, float] = field(default_factory=dict)
    temperature_c: dict[int, int] = field(default_factory=dict)
    p_gain: dict[int, int] = field(default_factory=dict)
    dead_band: dict[int, tuple[int, int]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"self-test: {'pass' if self.ok else 'FAIL'}",
                 f"  present: {self.present}  missing: {self.missing}"]
        for servo_id in self.present:
            cw, ccw = self.dead_band.get(servo_id, (0, 0))
            lines.append(
                f"  id {servo_id}: {self.voltage_v.get(servo_id, float('nan')):.1f} V, "
                f"{self.temperature_c.get(servo_id, -1)} C, P={self.p_gain.get(servo_id, -1)}, "
                f"dead band {cw}/{ccw} counts"
            )
        lines.extend(f"  problem: {p}" for p in self.problems)
        return "\n".join(lines)


class BusError(RuntimeError):
    """The bus could not do what was asked after every retry."""


class Bus:
    """The driver.  ``port`` is a device path; ``transport`` replaces pyserial in a test."""

    def __init__(
        self,
        port: str | None = None,
        settings: BusSettings = BusSettings(),
        *,
        ids: Sequence[int] = SERVO_IDS,
        calibration: Sequence[JointCalibration] | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.settings = settings
        self.ids = tuple(int(i) for i in ids)
        self.calibration = tuple(calibration if calibration is not None else so101_calibration())
        if len(self.calibration) != len(self.ids):
            raise ValueError("one calibration entry per servo id")
        self.counters = BusCounters()
        self.port = port
        self._buf = bytearray()
        self._torque_on = False
        self._last: BusState | None = None
        if transport is not None:
            self._io: Transport | None = transport
        elif port is not None:
            self._io = _open_serial(port, settings)
        else:
            raise ValueError("Bus needs either a port or a transport")

    # ------------------------------------------------------------------------- lifecycle ---

    def close(self) -> None:
        if self._io is not None:
            try:
                self.disable_torque()
            except Exception:
                pass
            self._io.close()
            self._io = None

    def __enter__(self) -> "Bus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def n(self) -> int:
        return len(self.ids)

    # -------------------------------------------------------------------- raw transactions ---

    def _send(self, frame: bytes) -> None:
        if self._io is None:
            raise BusError("the bus is closed")
        self.counters.instructions += 1
        self._io.write(frame)
        if self.settings.echo:
            self._drain(len(frame))
        if self.settings.settle_s:
            time.sleep(self.settings.settle_s)

    def _drain(self, count: int) -> None:
        """Swallow ``count`` echoed bytes without parsing them."""
        deadline = time.perf_counter() + self.settings.timeout_s
        got = 0
        while got < count and time.perf_counter() < deadline:
            chunk = self._io.read(count - got)
            got += len(chunk)

    def _recv(self, deadline: float) -> proto.Packet:
        """One status packet, or :class:`BusError` at ``deadline``."""
        while True:
            try:
                packet, used = proto.take_packet(bytes(self._buf))
            except proto.ChecksumError:
                self.counters.checksum_errors += 1
                # Drop the header that started the bad frame and keep looking: a checksum error
                # is one corrupted frame, not a dead port.
                del self._buf[:2]
                continue
            if packet is not None:
                del self._buf[:used]
                self.counters.replies += 1
                return packet
            if used:
                self.counters.resync_bytes += used
                del self._buf[:used]
            if time.perf_counter() >= deadline:
                self.counters.timeouts += 1
                raise BusError("timed out waiting for a status packet")
            # `read(1)` blocks until one byte or the port's own timeout, which is the cheap way
            # to wait; whatever else has already landed comes with it in the same call.
            chunk = self._io.read(max(int(getattr(self._io, "in_waiting", 0) or 0), 1))
            if chunk:
                self._buf.extend(chunk)

    def transact(self, frame: bytes, *, expect: bool = True) -> proto.Packet | None:
        """Send one instruction and read one status packet, retrying on a lost or bad reply."""
        last: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            if attempt:
                self.counters.retries += 1
                self._buf.clear()
            self._send(frame)
            if not expect:
                return None
            try:
                return self._recv(time.perf_counter() + self.settings.timeout_s)
            except BusError as exc:
                last = exc
        raise BusError(f"no reply after {self.settings.retries + 1} attempts") from last

    # -------------------------------------------------------------- single-servo registers ---

    def ping(self, servo_id: int) -> bool:
        """True when the servo answers.  A missing servo is a ``False``, not an exception."""
        try:
            packet = self.transact(proto.ping(servo_id))
        except BusError:
            return False
        return packet is not None and packet.servo_id == servo_id

    def read_register(self, servo_id: int, name: str) -> int:
        """Read one named register, decoded (sign-magnitude registers come back signed)."""
        reg = proto.REGISTERS[name]
        packet = self.transact(proto.read(servo_id, reg.address, reg.size))
        if packet is None or len(packet.params) != reg.size:
            raise BusError(f"servo {servo_id}: short reply reading {name}")
        if packet.error:
            self.counters.status_errors += 1
        value = packet.params[0] if reg.size == 1 else proto.from_bytes16(*packet.params[:2])
        if reg.signed_bit is not None:
            value = proto.to_signed(value, reg.signed_bit)
        return value

    def write_register(self, servo_id: int, name: str, value: int) -> None:
        """Write one named register.  EPROM registers are unlocked and re-locked around the write."""
        reg = proto.REGISTERS[name]
        if not reg.writable:
            raise ValueError(f"{name} is read-only")
        raw = proto.from_signed(value, reg.signed_bit) if reg.signed_bit is not None else value
        data = [raw & 0xFF] if reg.size == 1 else list(proto.to_bytes16(raw))
        eprom = reg.area == "eprom"
        if eprom:
            self.transact(proto.write(servo_id, proto.REGISTERS["lock"].address, [proto.LOCK_OPEN]))
        packet = self.transact(proto.write(servo_id, reg.address, data))
        if packet is not None and packet.error:
            self.counters.status_errors += 1
            raise StatusError(servo_id, packet.error)
        if eprom:
            self.transact(proto.write(servo_id, proto.REGISTERS["lock"].address,
                                      [proto.LOCK_CLOSED]))

    def read_state(self, servo_id: int) -> ServoState:
        """One servo's 56-70 block."""
        packet = self.transact(proto.read(servo_id, proto.STATE_BLOCK_ADDRESS,
                                          proto.STATE_BLOCK_LENGTH))
        if packet is None:
            raise BusError(f"servo {servo_id}: no state reply")
        return proto.decode_state(packet.params)

    # --------------------------------------------------------------- the microcontroller pair ---

    def write_ticks(self, targets: Sequence[int]) -> None:
        """``bus_write(targets[6])``: one sync-write of Goal_Position, in encoder counts.

        Counts are clipped to each joint's tick limits here as well as in :meth:`cycle`, so the
        guard holds for a caller that works in counts and never touches radians.
        """
        if len(targets) != self.n:
            raise ValueError(f"expected {self.n} targets, got {len(targets)}")
        clipped = [int(min(max(int(t), cal.lo_ticks), cal.hi_ticks))
                   for t, cal in zip(targets, self.calibration)]
        self._send(proto.sync_write_goals(self.ids, clipped))

    def read_ticks(self) -> tuple[list[int], list[int], list[int], list[bool]]:
        """``bus_read(pos[6], speed[6], load[6])``: counts, counts/s and 0.1 % duty.

        The fourth return is the per-servo reply-valid flag; a lost reply leaves the previous
        value in place rather than stalling the loop, which is what the emulated bus in
        ``robot/bus_emulator.py`` models and what the real driver has to do.
        """
        states = self._read_block()
        pos, speed, load, valid = [], [], [], []
        held = self._last
        for k, state in enumerate(states):
            if state is None:
                valid.append(False)
                pos.append(int(held.ticks[k]) if held is not None else self.calibration[k].centre)
                speed.append(0)
                load.append(0)
            else:
                valid.append(True)
                pos.append(state.position_ticks)
                speed.append(state.speed_counts)
                load.append(state.load_counts)
        return pos, speed, load, valid

    def _read_block(self) -> list[ServoState | None]:
        """Sync-read (or six plain reads) of 56-70.  ``None`` where a reply did not come back."""
        out: list[ServoState | None] = [None] * self.n
        if not self.settings.sync_read:
            for k, servo_id in enumerate(self.ids):
                try:
                    out[k] = self.read_state(servo_id)
                except (BusError, ProtocolError):
                    self.counters.missing_replies += 1
            return out
        index = {servo_id: k for k, servo_id in enumerate(self.ids)}
        for attempt in range(self.settings.retries + 1):
            if attempt:
                self.counters.retries += 1
                self._buf.clear()
            self._send(proto.sync_read_state(self.ids))
            deadline = time.perf_counter() + self.settings.timeout_s
            wanted = [k for k in range(self.n) if out[k] is None]
            for _ in wanted:
                try:
                    packet = self._recv(deadline)
                except BusError:
                    break
                k = index.get(packet.servo_id)
                if k is None or len(packet.params) != proto.STATE_BLOCK_LENGTH:
                    self.counters.framing_errors += 1
                    continue
                if packet.error:
                    self.counters.status_errors += 1
                out[k] = proto.decode_state(packet.params)
            if all(s is not None for s in out):
                return out
        self.counters.missing_replies += sum(1 for s in out if s is None)
        return out

    # ------------------------------------------------------------------------------ one tick ---

    def cycle(self, goal_rad: Sequence[float] | np.ndarray,
              torque_limit: Sequence[float] | np.ndarray | None = None) -> BusState:
        """One 50 Hz tick: sync-write the goals, sync-read 56-70, return the decoded state.

        ``torque_limit`` is a per-joint fraction of the servo's own limit.  It is written only
        when it changes, because ``Torque_Limit`` is a two-byte RAM register per servo and a
        sync-write of it costs another 260 us of a 20 ms budget for nothing.
        """
        goal = np.asarray(goal_rad, dtype=float).reshape(self.n)
        ticks = [cal.to_ticks(cal.clamp(g)) for g, cal in zip(goal, self.calibration)]
        t0 = time.perf_counter()
        if torque_limit is not None:
            self.set_torque_limit(torque_limit)
        self.write_ticks(ticks)
        state = self._decode_cycle(self._read_block(), time.perf_counter() - t0)
        self._last = state
        return state

    def _decode_cycle(self, states: list[ServoState | None], rtt: float) -> BusState:
        held = self._last
        n = self.n
        out = BusState(
            position=np.zeros(n), speed=np.zeros(n), load=np.zeros(n), current=np.zeros(n),
            voltage=np.zeros(n), temperature=np.zeros(n), moving=np.zeros(n, dtype=bool),
            status=np.zeros(n, dtype=int), valid=np.zeros(n, dtype=bool),
            ticks=np.zeros(n, dtype=int), rtt_s=float(rtt), t_mono=time.monotonic(),
        )
        for k, (state, cal) in enumerate(zip(states, self.calibration)):
            if state is None:
                if held is not None:
                    for name in ("position", "speed", "load", "current", "voltage",
                                 "temperature", "status", "ticks"):
                        getattr(out, name)[k] = getattr(held, name)[k]
                continue
            out.valid[k] = True
            out.ticks[k] = state.position_ticks
            out.position[k] = cal.to_rad(state.position_ticks)
            out.speed[k] = cal.speed_to_rad_s(state.speed_counts)
            out.load[k] = state.duty
            out.current[k] = state.current_a
            out.voltage[k] = state.voltage_v
            out.temperature[k] = state.temperature_c
            out.moving[k] = state.moving
            out.status[k] = state.status
        return out

    # -------------------------------------------------------------------------------- safety ---

    def set_torque_limit(self, fraction: Sequence[float] | np.ndarray) -> None:
        """``Torque_Limit`` (48) as a per-joint fraction of the servo's own limit, in one frame."""
        frac = np.clip(np.asarray(fraction, dtype=float).reshape(self.n), 0.0, 1.0)
        counts = [int(round(f * 1000)) for f in frac]
        self._send(proto.sync_write(proto.REGISTERS["torque_limit"].address,
                                    {i: list(proto.to_bytes16(c))
                                     for i, c in zip(self.ids, counts)}))

    def disable_torque(self) -> None:
        """Soft stop: ``Torque_Enable`` (40) = 0 on all six, in one broadcast sync-write."""
        self._send(proto.sync_write(proto.REGISTERS["torque_enable"].address,
                                    {i: [proto.TORQUE_DISABLED] for i in self.ids}))
        self._torque_on = False

    def enable_torque(self, *, check: bool = True) -> SelfTestResult:
        """Run the self-test and energize only if it passes.

        The servos are seeded with their own present position as the goal before torque comes on,
        so an arm that has been pushed by hand does not snap back to a stale target.
        """
        result = self.self_test() if check else SelfTestResult(ok=True, present=list(self.ids))
        if not result.ok:
            raise BusError("self-test failed, torque not enabled:\n" + result.report())
        states = self._read_block()
        seed = [s.position_ticks if s is not None else cal.centre
                for s, cal in zip(states, self.calibration)]
        self.write_ticks(seed)
        self._send(proto.sync_write(proto.REGISTERS["goal_speed"].address,
                                    {i: list(proto.to_bytes16(
                                        proto.from_signed(self.settings.goal_speed_counts)))
                                     for i in self.ids}))
        self.set_torque_limit(np.full(self.n, self.settings.torque_limit))
        self._send(proto.sync_write(proto.REGISTERS["torque_enable"].address,
                                    {i: [proto.TORQUE_ENABLED] for i in self.ids}))
        self._torque_on = True
        return result

    @property
    def torque_enabled(self) -> bool:
        return self._torque_on

    def self_test(self) -> SelfTestResult:
        """Ping all six, read voltage/temperature/P/dead band, and decide whether it is safe."""
        result = SelfTestResult(ok=True)
        for servo_id in self.ids:
            if not self.ping(servo_id):
                result.missing.append(servo_id)
                continue
            result.present.append(servo_id)
            try:
                state = self.read_state(servo_id)
                result.voltage_v[servo_id] = state.voltage_v
                result.temperature_c[servo_id] = state.temperature_c
                result.p_gain[servo_id] = self.read_register(servo_id, "p_coefficient")
                result.dead_band[servo_id] = (self.read_register(servo_id, "cw_dead_band"),
                                              self.read_register(servo_id, "ccw_dead_band"))
            except (BusError, ProtocolError) as exc:
                result.problems.append(f"id {servo_id}: {exc}")
                continue
            if state.status:
                result.problems.append(
                    f"id {servo_id}: status byte {proto.describe_error(state.status)}")
            if not (self.settings.min_voltage_v <= state.voltage_v <= self.settings.max_voltage_v):
                result.problems.append(
                    f"id {servo_id}: {state.voltage_v:.1f} V outside "
                    f"{self.settings.min_voltage_v:.1f}-{self.settings.max_voltage_v:.1f} V")
            if state.temperature_c > self.settings.max_temperature_c:
                result.problems.append(
                    f"id {servo_id}: {state.temperature_c} C above "
                    f"{self.settings.max_temperature_c} C")
        if result.missing:
            result.problems.append(f"no reply from servo id(s) {result.missing}")
        result.ok = not result.problems
        return result

    # ----------------------------------------------------------------------------- measuring ---

    def measure_cycle_time(self, goal_rad: Sequence[float] | np.ndarray,
                           ticks: int = 200) -> dict[str, float]:
        """Time ``ticks`` sync-write/sync-read pairs and report the round trip in milliseconds."""
        samples = np.zeros(int(ticks))
        before = self.counters.errors
        for k in range(int(ticks)):
            samples[k] = self.cycle(goal_rad).rtt_s
        return {
            "ticks": float(ticks),
            "mean_ms": float(samples.mean() * 1e3),
            "median_ms": float(np.median(samples) * 1e3),
            "p95_ms": float(np.percentile(samples, 95) * 1e3),
            "max_ms": float(samples.max() * 1e3),
            "errors": float(self.counters.errors - before),
            "fits_in_period": float(np.percentile(samples, 95) < CONTROL_PERIOD_S),
        }


def _open_serial(port: str, settings: BusSettings):
    """Open a pyserial port, imported here so the protocol is usable without the extra."""
    try:
        import serial
    except ImportError as exc:  # pragma: no cover - the hardware extra is not installed
        raise ImportError(
            "the serial driver needs pyserial: install the 'hardware' extra"
        ) from exc
    return serial.Serial(
        port=port, baudrate=settings.baud, bytesize=8, parity="N", stopbits=1,
        timeout=settings.timeout_s, write_timeout=settings.timeout_s,
    )


def with_port(port: str, **kwargs) -> Bus:
    """``Bus`` on a device path, with :class:`BusSettings` fields passed as keywords."""
    settings = replace(BusSettings(), **{k: v for k, v in kwargs.items()
                                         if k in BusSettings.__dataclass_fields__})
    rest = {k: v for k, v in kwargs.items() if k not in BusSettings.__dataclass_fields__}
    return Bus(port, settings, **rest)
