"""A virtual servo bus: the STS3215 protocol answered from the simulator, over a pseudo-terminal.

``os.openpty()` gives two file descriptors joined by the kernel's terminal driver.  The virtual
bus owns one end and answers instruction packets; ``Bus("/dev/pts/N")`` opens the other with
pyserial and cannot tell the difference between it and the kit's USB adapter board.  That is the
point: the driver under test is the same object either way, and every byte crosses a real file
descriptor, so framing, timeouts, resynchronization and partial reads are exercised rather than
mocked.

Two backends
------------
:class:`RegisterServos` is a plain register file with the STS3215 power-on defaults.  It answers
every instruction from memory and runs no physics, which is what a protocol-conformance run needs:
millions of packets in a couple of minutes.

:class:`SimulatedServos` puts a MuJoCo :class:`~brick_builder.scene.compose.Scene` behind the same
register file.  A sync-write of ``Goal_Position`` becomes one
:meth:`~brick_builder.robot.servo_model.ServoBus.write` and ten 2 ms physics steps -- one 20 ms
control tick -- after which ``Present_Position``, ``Present_Speed`` and ``Present_Load`` are
refreshed from ``ServoBus.read()``.  The whole existing servo model is therefore behind the wire:
encoder quantization, the firmware's rate limit and dead band, free play, and the duty cycle that
``Present_Load`` really reports.

Wire imperfections
------------------
``packet_loss``, ``turnaround_s`` and ``servo_latency_s`` mean what they mean in
``robot/bus_emulator.py``: a lost request takes every reply of that tick with it, a lost reply
leaves one servo's entry stale, and the turnaround is the direction-switch dead time before each
reply.  The defaults are lossless and instantaneous so a conformance run measures the protocol and
nothing else; a run that wants the emulator's 0.5 % loss asks for it.
"""

from __future__ import annotations

import math
import os
import select
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import sts3215 as proto
from .sts3215 import JointCalibration, so101_calibration

MEM_SIZE = 71
"""Highest documented address is 70 (``Present_Current`` high byte)."""

PHYSICS_PER_TICK = 10
"""2 ms physics steps per 20 ms control tick, the same multiple ``control/pick_place.py`` uses."""


@dataclass
class ServoDefaults:
    """Power-on register values the virtual servos come up with."""

    p_gain: int = 16
    """``P_Coefficient`` (21).  16 is what ``ArmConfig`` ships with; the factory value is 32."""
    d_gain: int = 0
    i_gain: int = 0
    cw_dead_band: int = 0
    ccw_dead_band: int = 0
    """Registers 26/27.  0 matches ``ServoConfig.error_deadband_ticks``'s shipped default."""
    voltage_v: float = 7.4
    temperature_c: int = 25
    max_temperature_c: int = 70
    min_voltage_v: float = 4.5
    max_voltage_v: float = 12.0
    torque_limit: int = 1000
    status_return_level: int = 1
    """0 answers PING only; 1 answers every addressed instruction.  Broadcast frames are never
    answered at any level, which is what makes a sync-write silent and a sync-read the exception."""


class RegisterServos:
    """Six STS3215 register files.  Answers the protocol from memory; runs no physics."""

    def __init__(
        self,
        ids: Sequence[int] = proto.SERVO_IDS,
        *,
        defaults: ServoDefaults = ServoDefaults(),
        calibration: Sequence[JointCalibration] | None = None,
    ) -> None:
        self.ids = tuple(int(i) for i in ids)
        self.defaults = defaults
        self.calibration = tuple(calibration if calibration is not None else so101_calibration())
        self.mem: dict[int, bytearray] = {}
        self.staged: dict[int, list[tuple[int, bytes]]] = {i: [] for i in self.ids}
        for k, servo_id in enumerate(self.ids):
            self.mem[servo_id] = self._power_on(servo_id, self.calibration[k])

    def _power_on(self, servo_id: int, cal: JointCalibration) -> bytearray:
        mem = bytearray(MEM_SIZE)
        d = self.defaults
        put = self._put
        put(mem, "id", servo_id)
        put(mem, "baud_rate", proto.BAUD_CODES[1_000_000])
        put(mem, "return_delay", 0)
        put(mem, "status_return_level", d.status_return_level)
        put(mem, "min_angle_limit", cal.lo_ticks)
        put(mem, "max_angle_limit", cal.hi_ticks)
        put(mem, "max_temperature", d.max_temperature_c)
        put(mem, "max_voltage", int(round(d.max_voltage_v / proto.VOLT_PER_COUNT)))
        put(mem, "min_voltage", int(round(d.min_voltage_v / proto.VOLT_PER_COUNT)))
        put(mem, "max_torque", 1000)
        put(mem, "p_coefficient", d.p_gain)
        put(mem, "d_coefficient", d.d_gain)
        put(mem, "i_coefficient", d.i_gain)
        put(mem, "cw_dead_band", d.cw_dead_band)
        put(mem, "ccw_dead_band", d.ccw_dead_band)
        put(mem, "lock", proto.LOCK_CLOSED)
        put(mem, "torque_enable", proto.TORQUE_DISABLED)
        put(mem, "torque_limit", d.torque_limit)
        put(mem, "goal_position", cal.centre)
        put(mem, "present_position", cal.centre)
        put(mem, "present_voltage", int(round(d.voltage_v / proto.VOLT_PER_COUNT)))
        put(mem, "present_temperature", d.temperature_c)
        return mem

    @staticmethod
    def _put(mem: bytearray, name: str, value: int) -> None:
        reg = proto.REGISTERS[name]
        if reg.size == 1:
            mem[reg.address] = int(value) & 0xFF
        else:
            lo, hi = proto.to_bytes16(int(value))
            mem[reg.address] = lo
            mem[reg.address + 1] = hi

    def get(self, servo_id: int, name: str) -> int:
        reg = proto.REGISTERS[name]
        mem = self.mem[servo_id]
        raw = (mem[reg.address] if reg.size == 1
               else proto.from_bytes16(mem[reg.address], mem[reg.address + 1]))
        return proto.to_signed(raw, reg.signed_bit) if reg.signed_bit is not None else raw

    def set(self, servo_id: int, name: str, value: int) -> None:
        reg = proto.REGISTERS[name]
        raw = proto.from_signed(value, reg.signed_bit) if reg.signed_bit is not None else value
        self._put(self.mem[servo_id], name, raw)

    # -- the instruction handlers --------------------------------------------------------------

    def present(self, servo_id: int) -> bool:
        return servo_id in self.mem

    def read(self, servo_id: int, address: int, length: int) -> bytes:
        mem = self.mem[servo_id]
        if address + length > len(mem):
            raise IndexError(f"read of {length} at {address} runs past the memory table")
        return bytes(mem[address:address + length])

    def write(self, servo_id: int, address: int, data: bytes, *, notify: bool = True) -> None:
        mem = self.mem[servo_id]
        mem[address:address + len(data)] = data
        if notify:
            self.on_write(servo_id, address, data)

    def on_write(self, servo_id: int, address: int, data: bytes) -> None:
        """Hook: a subclass reacts to a single-servo write that lands on a register it cares about.

        Not called for the per-servo slices of a sync-write; those land together and are announced
        once through :meth:`sync_written`, so one sync-write is one control tick and not six.
        """

    def stage(self, servo_id: int, address: int, data: bytes) -> None:
        self.staged[servo_id].append((address, bytes(data)))

    def commit(self, servo_id: int | None = None) -> None:
        targets = self.ids if servo_id is None else (servo_id,)
        for i in targets:
            for address, data in self.staged[i]:
                self.write(i, address, data)
            self.staged[i].clear()

    def sync_written(self, address: int, servo_ids: Sequence[int]) -> None:
        """Hook: one sync-write has landed on every servo in ``servo_ids``."""

    def answers(self, servo_id: int, instruction: int) -> bool:
        if instruction == proto.INST_PING:
            return True
        return self.get(servo_id, "status_return_level") >= 1

    def error_byte(self, servo_id: int) -> int:
        return self.get(servo_id, "servo_status")


class SimulatedServos(RegisterServos):
    """Register files backed by a MuJoCo scene through the existing ``ServoBus`` emulation.

    One sync-write of ``Goal_Position`` is one control tick: the goals go to ``ServoBus.write``,
    ten 2 ms physics steps run, and ``ServoBus.read`` refreshes 56-70.  A plain (single-servo)
    write of ``Goal_Position`` does the same, so a driver that talks to one servo at a time --
    ``servo_sysid.py`` does -- still moves the arm.
    """

    def __init__(
        self,
        scene,
        *,
        ids: Sequence[int] = proto.SERVO_IDS,
        defaults: ServoDefaults | None = None,
        calibration: Sequence[JointCalibration] | None = None,
        servo_cfg=None,
        rng: np.random.Generator | None = None,
        step_physics: bool = True,
    ) -> None:
        from ..robot.servo_model import ServoBus, ServoConfig

        cfg = servo_cfg if servo_cfg is not None else ServoConfig()
        if defaults is None:
            defaults = ServoDefaults(voltage_v=cfg.vin,
                                     cw_dead_band=int(round(cfg.error_deadband_ticks)),
                                     ccw_dead_band=int(round(cfg.error_deadband_ticks)))
        super().__init__(ids, defaults=defaults, calibration=calibration)
        self.scene = scene
        self.servo = ServoBus(scene.model, scene.data, cfg=cfg,
                              rng=rng if rng is not None else np.random.default_rng(0))
        self.step_physics = bool(step_physics)
        self.ticks = 0
        self.lock = threading.RLock()
        self._seed_from_scene()

    def _seed_from_scene(self) -> None:
        reading = self.servo.read()
        for k, servo_id in enumerate(self.ids):
            cal = self.calibration[k]
            self.set(servo_id, "goal_position", cal.to_ticks(reading.position[k]))
        self._refresh(reading)

    def resync(self) -> None:
        """Push the plant's *current* state into 56-70, without a control tick.

        Nothing here writes ``data.ctrl`` or steps physics: this is for a caller that moved the
        scene's joints directly (the self-test's torque-off range check pokes ``data.qpos`` the
        way a technician backdrives a real joint by hand) and wants ``Present_Position`` -- what a
        sync-read over the wire actually reports -- to reflect it, the way it would on hardware.
        """
        self._refresh(self.servo.read())

    # -- the control tick ----------------------------------------------------------------------

    def on_write(self, servo_id: int, address: int, data: bytes) -> None:
        goal = proto.REGISTERS["goal_position"]
        if address <= goal.address < address + len(data):
            self.step()

    def sync_written(self, address: int, servo_ids: Sequence[int]) -> None:
        goal = proto.REGISTERS["goal_position"]
        if address == goal.address:
            self.step()

    def step(self) -> None:
        """Apply the register file's goals and run one 20 ms control tick of physics."""
        with self.lock:
            goal = np.array([self.calibration[k].to_rad(self.get(i, "goal_position"))
                             for k, i in enumerate(self.ids)])
            enabled = np.array([self.get(i, "torque_enable") for i in self.ids], dtype=float)
            limit = np.array([self.get(i, "torque_limit") / 1000.0 for i in self.ids])
            self.servo.write(goal, torque_limit=np.clip(limit * enabled, 0.0, 1.0))
            if self.step_physics:
                import mujoco

                snap_every = max(int(self.scene.snap.params.snap_every), 1)
                for k in range(PHYSICS_PER_TICK):
                    self.servo.substep()
                    if k % snap_every == 0:
                        self.scene.snap.update(self.scene.model, self.scene.data)
                    mujoco.mj_step(self.scene.model, self.scene.data)
            self.ticks += 1
            self._refresh(self.servo.read())

    def _refresh(self, reading) -> None:
        for k, servo_id in enumerate(self.ids):
            if not self.present(servo_id):
                continue   # a test can delete an id from `mem` to simulate a servo never present
            cal = self.calibration[k]
            self.set(servo_id, "present_position", cal.to_ticks(reading.position[k]))
            self.set(servo_id, "present_speed", cal.speed_to_counts(reading.speed[k]))
            self.set(servo_id, "present_load",
                     int(round(reading.load[k] / proto.DUTY_PER_COUNT)))
            self.set(servo_id, "present_current",
                     int(round(reading.current[k] / proto.AMP_PER_COUNT)))
            self.set(servo_id, "moving", int(abs(reading.speed[k]) > 0.5 * proto.RAD_PER_TICK))
            # An overload trip is the servo's own protection latching, which is exactly what the
            # status byte's overload bit reports on real hardware.
            self.set(servo_id, "servo_status", 0x20 if bool(reading.tripped[k]) else 0)


@dataclass
class WireConfig:
    """Imperfections the virtual bus injects, with the same meaning as ``BusConfig``'s fields."""

    packet_loss: float = 0.0
    """Probability that any single packet is lost -- the request, or one reply."""
    turnaround_s: float = 0.0
    """Direction-switch dead time before each reply."""
    servo_latency_s: float = 0.0
    """Extra time a servo takes to start answering, on top of the turnaround."""
    seed: int = 0


@dataclass
class VirtualBusCounters:
    requests: int = 0
    replies: int = 0
    dropped_requests: int = 0
    dropped_replies: int = 0
    bad_frames: int = 0

    @property
    def packets(self) -> int:
        """Every frame that crossed the line, in both directions."""
        return self.requests + self.replies

    def as_dict(self) -> dict[str, int]:
        return {**self.__dict__, "packets": self.packets}


class VirtualBus:
    """A pseudo-terminal with :class:`RegisterServos` (or :class:`SimulatedServos`) behind it.

    ``bus.port`` is the device path to hand to :class:`~brick_builder.hardware.bus.Bus`.  The
    responder runs in its own thread; :meth:`start` and :meth:`stop` bracket it, and the object is
    a context manager.
    """

    def __init__(
        self,
        servos: RegisterServos | None = None,
        wire: WireConfig = WireConfig(),
        *,
        poll_s: float = 0.002,
    ) -> None:
        self.servos = servos if servos is not None else RegisterServos()
        self.wire = wire
        self.counters = VirtualBusCounters()
        self.rng = np.random.default_rng(wire.seed)
        self._poll_s = float(poll_s)
        self._master, self._slave = os.openpty()
        _make_raw(self._slave)
        _make_raw(self._master)
        os.set_blocking(self._master, False)
        self._buf = bytearray()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def port(self) -> str:
        """The device path the driver opens, e.g. ``/dev/pts/7``."""
        return os.ttyname(self._slave)

    # ------------------------------------------------------------------------------ lifecycle ---

    def start(self) -> "VirtualBus":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._serve, name="virtual-bus", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        for fd in (self._master, self._slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "VirtualBus":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def raise_if_failed(self) -> None:
        """Re-raise anything the responder thread hit, so a test does not pass on a dead bus."""
        if self._error is not None:
            raise self._error

    # -------------------------------------------------------------------------------- serving ---

    def _serve(self) -> None:
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([self._master], [], [], self._poll_s)
                if not ready:
                    continue
                try:
                    chunk = os.read(self._master, 4096)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    return
                if not chunk:
                    continue
                self._buf.extend(chunk)
                self._drain()
        except BaseException as exc:  # surfaced by raise_if_failed
            self._error = exc

    def _drain(self) -> None:
        while True:
            try:
                packet, used = proto.take_packet(bytes(self._buf))
            except proto.ChecksumError:
                self.counters.bad_frames += 1
                del self._buf[:2]
                continue
            if packet is None:
                if used:
                    del self._buf[:used]
                return
            del self._buf[:used]
            self.counters.requests += 1
            for reply in self.handle(packet):
                if self.wire.turnaround_s or self.wire.servo_latency_s:
                    time.sleep(self.wire.turnaround_s + self.wire.servo_latency_s)
                if self.wire.packet_loss and self.rng.random() < self.wire.packet_loss:
                    self.counters.dropped_replies += 1
                    continue
                self.counters.replies += 1
                os.write(self._master, reply)

    def handle(self, packet: proto.Packet) -> list[bytes]:
        """Everything one instruction packet produces, in the order it goes back on the wire."""
        if self.wire.packet_loss and self.rng.random() < self.wire.packet_loss:
            self.counters.dropped_requests += 1
            return []
        servos, code, params = self.servos, packet.instruction, packet.params
        target = packet.servo_id

        if code == proto.INST_SYNC_WRITE:
            address, size = params[0], params[1]
            body, touched = params[2:], []
            for offset in range(0, len(body), size + 1):
                servo_id = body[offset]
                if servos.present(servo_id):
                    servos.write(servo_id, address, bytes(body[offset + 1:offset + 1 + size]),
                                 notify=False)
                    touched.append(servo_id)
            servos.sync_written(address, touched)
            return []

        if code == proto.INST_SYNC_READ:
            address, length = params[0], params[1]
            out = []
            for servo_id in params[2:]:
                if not servos.present(servo_id):
                    continue
                out.append(proto.build_status(servo_id, servos.error_byte(servo_id),
                                              servos.read(servo_id, address, length)))
            return out

        if target == proto.BROADCAST_ID:
            if code == proto.INST_ACTION:
                servos.commit()
            elif code == proto.INST_WRITE:
                for servo_id in servos.ids:
                    servos.write(servo_id, params[0], bytes(params[1:]), notify=False)
                servos.sync_written(params[0], servos.ids)
            return []

        if not servos.present(target):
            return []
        if code == proto.INST_PING:
            return [proto.build_status(target, servos.error_byte(target))]
        if code == proto.INST_READ:
            data = servos.read(target, params[0], params[1])
            return [proto.build_status(target, servos.error_byte(target), data)]
        if code == proto.INST_WRITE:
            servos.write(target, params[0], bytes(params[1:]))
        elif code == proto.INST_REG_WRITE:
            servos.stage(target, params[0], bytes(params[1:]))
        elif code == proto.INST_ACTION:
            servos.commit(target)
        else:
            self.counters.bad_frames += 1
            return []
        if not servos.answers(target, code):
            return []
        return [proto.build_status(target, servos.error_byte(target))]


def _make_raw(fd: int) -> None:
    """Turn off echo and line discipline so the pty carries bytes, not lines."""
    import termios
    import tty

    try:
        tty.setraw(fd, termios.TCSANOW)
    except termios.error:  # pragma: no cover - the master end on some kernels
        pass


# ------------------------------------------------------------------------------ helper scenes ---


def simulated_bus(scene, *, wire: WireConfig = WireConfig(), servo_cfg=None,
                  step_physics: bool = True, seed: int = 0) -> VirtualBus:
    """A virtual bus with ``scene``'s arm behind it, ready to :meth:`VirtualBus.start`."""
    servos = SimulatedServos(scene, servo_cfg=servo_cfg, step_physics=step_physics,
                             rng=np.random.default_rng(seed))
    return VirtualBus(servos, wire)


def conformance_run(packets: int = 1_000_000, *, wire: WireConfig = WireConfig(),
                    timeout_s: float = 600.0) -> dict:
    """Drive the register-file backend for ``packets`` frames and count protocol errors.

    One control cycle is eight frames on the wire: one sync-write, one sync-read request and six
    replies.  Returns the driver's and the bus's own counters plus the measured round trip, which
    is the evidence the protocol implementation is exercised end to end rather than unit-tested.
    """
    from .bus import Bus, BusSettings

    per_cycle = 2 + len(proto.SERVO_IDS)
    cycles = max(int(math.ceil(packets / per_cycle)), 1)
    goal = np.zeros(len(proto.SERVO_IDS))
    with VirtualBus(RegisterServos(), wire) as vbus:
        bus = Bus(vbus.port, BusSettings(timeout_s=0.05, retries=2))
        t0 = time.perf_counter()
        rtt = np.zeros(cycles)
        for k in range(cycles):
            rtt[k] = bus.cycle(goal).rtt_s
            if time.perf_counter() - t0 > timeout_s:
                rtt = rtt[:k + 1]
                cycles = k + 1
                break
        elapsed = time.perf_counter() - t0
        bus.close()
        vbus.raise_if_failed()
        return {
            "cycles": cycles,
            "packets": cycles * per_cycle,
            "seconds": elapsed,
            "driver": bus.counters.as_dict(),
            "bus": vbus.counters.as_dict(),
            "rtt_mean_ms": float(rtt.mean() * 1e3),
            "rtt_p95_ms": float(np.percentile(rtt, 95) * 1e3),
            "rtt_max_ms": float(rtt.max() * 1e3),
        }
