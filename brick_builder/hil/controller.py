"""The three ways of reaching the same C controller.

`SilController` calls it in this process through ctypes. `ProcessController` runs it as a native
process and speaks the wire protocol over a pipe or a TCP socket. `SerialController` speaks the
same protocol over a UART to the microcontroller. All three present the same four calls, and the
host loop in `episode.py` cannot tell them apart.

Lock step is the host's job and is the same for every transport: send TICK_BEGIN, wait for
TARGETS, run the plant for exactly those writes, send SCENE_STATE and BUS_READINGS, wait for
TICK_LOG. Nothing advances on the controller's own clock.
"""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np

from . import protocol as P
from .library import CBusReadings, CSlots, CTickOut, ControlLib, load


class ControllerError(RuntimeError):
    pass


class SilController:
    """Software in the loop: the library called inside the Python process."""

    kind = "c"

    def __init__(self, lib: ControlLib | None = None) -> None:
        if lib is None:
            lib, _ = load()
        self.lib = lib
        self._ctrl = ctypes.create_string_buffer(lib.sizeof("bb_controller"))
        self._out = CTickOut()
        self._slots = CSlots()
        self._bus = CBusReadings()
        self._blob = None
        self.ticks = 0
        self.tick_seconds = 0.0
        self.worst_seconds = 0.0
        self._begin_seconds = 0.0

    def start(self, blob: bytes, q_home) -> None:
        self._blob = ctypes.create_string_buffer(blob, len(blob))
        q = (ctypes.c_double * 5)(*[float(v) for v in q_home])
        rc = self.lib._lib.bb_control_init(
            ctypes.cast(self._ctrl, ctypes.c_void_p), self._blob, len(blob), q)
        if rc != 0:
            raise ControllerError(f"bb_control_init refused the bundle (code {rc})")

    def tick_begin(self) -> P.Targets:
        t0 = time.perf_counter()
        n = self.lib._lib.bb_tick_begin(ctypes.cast(self._ctrl, ctypes.c_void_p),
                                        ctypes.byref(self._out))
        self._begin_seconds = time.perf_counter() - t0
        out = self._out
        commands = [
            P.TickCommand(pos=np.array(out.cmd[i].pos), spin=out.cmd[i].spin,
                          grip=out.cmd[i].grip, c_vert=out.cmd[i].c_vert,
                          c_lat=out.cmd[i].c_lat, q=np.array(out.cmd[i].q),
                          ik_ok=bool(out.cmd[i].ik_ok))
            for i in range(n)
        ]
        return P.Targets(k=out.k, step_index=out.step_index, step_id=out.step_id,
                         commands=commands)

    def tick_end(self, state, bus) -> P.TickLog:
        slots = np.ascontiguousarray(state.slots, dtype=np.float32)
        depth = np.ascontiguousarray(state.depth, dtype=np.float32)
        ids = np.ascontiguousarray(state.slot_ids, dtype=np.int32)
        ctypes.memmove(self._slots.slots, slots.ctypes.data, slots.nbytes)
        ctypes.memmove(self._slots.depth, depth.ctypes.data, depth.nbytes)
        ctypes.memmove(self._slots.ids, ids.ctypes.data, ids.nbytes)
        for i in range(6):
            self._bus.position[i] = float(bus[0][i])
            self._bus.speed[i] = float(bus[1][i])
            self._bus.load[i] = float(bus[2][i])
        t0 = time.perf_counter()
        self.lib._lib.bb_tick_end(ctypes.cast(self._ctrl, ctypes.c_void_p),
                                  ctypes.byref(self._slots), ctypes.byref(self._bus),
                                  ctypes.byref(self._out))
        self._account(time.perf_counter() - t0 + self._begin_seconds)
        self._begin_seconds = 0.0
        out = self._out
        return P.TickLog(
            k=out.k, step_index=out.step_index, step_id=out.step_id, holds=out.holds,
            refs_seen=out.refs_seen, done=bool(out.done), failure=out.failure,
            regrasps=out.regrasps, matched=out.matched, ik_failures=0,
            correction=np.array(out.correction), err_mm=out.err_mm, tol_mm=out.tol_mm,
            absolute_mm=out.absolute_mm, predict_mm=out.predict_mm, yaw_err=out.yaw_err)

    def _account(self, dt: float) -> None:
        """Vision tick time: `bb_tick_begin` plus `bb_tick_end`, as `bb_session` counts it."""
        self.ticks += 1
        self.tick_seconds += dt
        self.worst_seconds = max(self.worst_seconds, dt)

    def status(self) -> P.Status:
        mean = 1e6 * self.tick_seconds / max(self.ticks, 1)
        return P.Status(version=1, n_frames=0, ticks=self.ticks, bad_crc=0, resyncs=0,
                        ik_failures=0, tick_us_mean=mean, tick_us_max=1e6 * self.worst_seconds)

    def close(self) -> None:
        self._blob = None


class _WireController:
    """Shared half of the two transports that speak the protocol."""

    kind = "wire"

    def __init__(self, *, timeout: float = 20.0) -> None:
        self.dec = P.Decoder()
        self.seq = 0
        self.timeout = timeout
        self.pending: list[P.Frame] = []
        self.corruptions = 0
        self._damage: tuple[int, str] | None = None

    # -- transport, provided by the subclass -------------------------------------------------

    def _send(self, data: bytes) -> None:
        raise NotImplementedError

    def _recv(self) -> bytes:
        raise NotImplementedError

    # -- framing ------------------------------------------------------------------------------

    def _write(self, msg_type: int, payload: bytes = b"") -> None:
        self.seq = (self.seq + 1) & 0xFFFF
        frame = P.encode(msg_type, payload, seq=self.seq)
        if self._damage is not None and self._damage[0] == msg_type:
            kind = self._damage[1]
            self._damage = None
            self.corruptions += 1
            if kind == "lose":
                return
            frame = bytearray(frame)
            frame[len(frame) // 2] ^= 0xFF
            frame = bytes(frame)
        self._send(frame)

    def _await(self, want: int, *, deadline: float | None = None) -> P.Frame:
        end = time.monotonic() + self.timeout if deadline is None else deadline
        while True:
            for i, frame in enumerate(self.pending):
                if frame.type == want:
                    return self.pending.pop(i)
                if frame.type == P.MSG_ERROR:
                    code, detail = P.parse_error(frame.payload)
                    self.pending.pop(i)
                    raise ControllerError(
                        f"controller reported {P.ERRORS.get(code, code)} (detail {detail})")
            if time.monotonic() > end:
                raise ControllerError(
                    f"timed out waiting for {P.NAMES.get(want, want)} after {self.timeout:.0f} s")
            data = self._recv()
            if data:
                self.pending.extend(self.dec.feed(data))

    def start(self, blob: bytes, q_home) -> None:
        self._await(P.MSG_HELLO)
        for chunk in P.bundle_chunks(blob):
            self._write(P.MSG_TARGET_BUNDLE, chunk)
            status, received = P.parse_bundle_ack(self._await(P.MSG_BUNDLE_ACK).payload)
            if status != 0:
                raise ControllerError(f"controller refused the bundle (code {status})")
        if received != len(blob):
            raise ControllerError(f"controller took {received} of {len(blob)} bundle bytes")
        # The same seed `SilController` hands `bb_control_init`: the arm's own starting joint
        # angles, so the first inverse-kinematics solve starts where the arm is.
        self._write(P.MSG_START, P.start(q_home))
        status, received = P.parse_bundle_ack(self._await(P.MSG_BUNDLE_ACK).payload)
        if status != 0:
            raise ControllerError(f"controller refused to start (code {status})")

    def tick_begin(self) -> P.Targets:
        self._write(P.MSG_TICK_BEGIN)
        return P.parse_targets(self._await(P.MSG_TARGETS).payload)

    def tick_end(self, state, bus) -> P.TickLog:
        self._write(P.MSG_BUS_READINGS, P.bus_readings(bus[0], bus[1], bus[2]))
        self._write(P.MSG_SCENE_STATE,
                    P.scene_state(getattr(state, "frame_id", 0), float(state.t), state.slots,
                                  state.depth, state.slot_ids))
        return P.parse_tick_log(self._await(P.MSG_TICK_LOG).payload)

    def status(self) -> P.Status:
        self._write(P.MSG_STATUS)
        return P.parse_status(self._await(P.MSG_STATUS).payload)

    def damage_next(self, msg_type: int, kind: str = "crc") -> None:
        """Fault injection on the next outgoing frame of `msg_type`.

        `crc` flips the byte in the middle of the frame; `lose` does not send it. Either way the
        controller never acts on it and the reply wait ends in `ControllerError`."""
        if kind not in ("crc", "lose"):
            raise ValueError(f"unknown damage {kind!r}")
        self._damage = (msg_type, kind)

    def inject_corruption(self, kind: str = "crc") -> None:
        """Send a deliberately damaged frame. `crc` flips a payload byte after the CRC is
        computed; `drop` sends a frame with its tail cut off."""
        self.corruptions += 1
        frame = bytearray(P.encode(P.MSG_STATUS, b"\0" * 8, seq=0xFFFF))
        if kind == "crc":
            frame[10] ^= 0xFF
            self._send(bytes(frame))
        elif kind == "drop":
            self._send(bytes(frame[:-3]))
        else:
            raise ValueError(f"unknown corruption {kind!r}")


class ProcessController(_WireController):
    """The library in a native process, over a pipe (default) or a loopback TCP socket."""

    kind = "process"

    def __init__(self, runner: Path, *, tcp_port: int = 0, timeout: float = 20.0) -> None:
        super().__init__(timeout=timeout)
        self.runner = Path(runner)
        self.tcp_port = int(tcp_port)
        self.socket = None
        if self.tcp_port:
            import socket

            self.proc = subprocess.Popen([str(self.runner), "--tcp", str(self.tcp_port)],
                                         stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    self.socket = socket.create_connection(("127.0.0.1", self.tcp_port), 1.0)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise ControllerError("the runner never accepted a connection")
                    time.sleep(0.05)
            # Lock step sends two small frames back to back; without this the second waits
            # for the first one's delayed acknowledgement, about 40 ms a tick.
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.settimeout(0.5)
        else:
            self.proc = subprocess.Popen([str(self.runner), "--stdio"],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, bufsize=0)

    def _send(self, data: bytes) -> None:
        if self.socket is not None:
            self.socket.sendall(data)
        else:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def _recv(self) -> bytes:
        if self.socket is not None:
            try:
                return self.socket.recv(65536)
            except OSError:
                return b""
        # Wait with a bound, so a reply that never comes ends in the timeout `_await` reports
        # rather than in a read that blocks for ever.
        ready, _, _ = select.select([self.proc.stdout], [], [], 0.5)
        if not ready:
            return b""
        data = os.read(self.proc.stdout.fileno(), 65536)
        if not data:
            raise ControllerError("the runner closed its output stream")
        return data

    def close(self) -> None:
        try:
            self._write(P.MSG_RESET)
        except Exception:
            pass
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
        elif self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5.0)


class SerialController(_WireController):
    """The same protocol over a UART to the microcontroller."""

    kind = "serial"

    def __init__(self, port: str, *, baud: int = 921600, timeout: float = 20.0) -> None:
        super().__init__(timeout=timeout)
        import serial  # pyserial, the `hardware` extra

        self.port = serial.Serial(port, baudrate=baud, timeout=0.2)
        # The board reboots when the bridge asserts DTR; give it a moment and drop the banner.
        time.sleep(0.5)
        self.port.reset_input_buffer()

    def start(self, blob: bytes, q_home) -> None:
        # The greeting the board sent at boot went out with the banner; ask for it again.
        self._write(P.MSG_HELLO)
        super().start(blob, q_home)

    def _send(self, data: bytes) -> None:
        self.port.write(data)
        self.port.flush()

    def _recv(self) -> bytes:
        return self.port.read(4096) or b""

    def close(self) -> None:
        try:
            self._write(P.MSG_RESET)
        except Exception:
            pass
        self.port.close()


def free_port() -> int:
    """Unused loopback TCP port number, as the kernel assigns one to port 0."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def make(spec: str, *, lib: ControlLib | None = None, build=None, tcp_port: int = 0):
    """`c`, `process`, `process:tcp` or `serial:/dev/ttyACM0`."""
    if spec == "c":
        return SilController(lib)
    if spec in ("process", "process:pipe"):
        return ProcessController(build.runner)
    if spec == "process:tcp":
        return ProcessController(build.runner, tcp_port=tcp_port or free_port())
    if spec.startswith("serial:"):
        return SerialController(spec.split(":", 1)[1])
    raise ValueError(f"unknown controller {spec!r}")
