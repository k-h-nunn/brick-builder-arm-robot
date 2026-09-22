"""The wire protocol, host side: framing, CRC and the message payloads.

Mirrors `firmware/control_lib/src/bb_proto.c` exactly -- one frame is

    B1 C5 | type u8 | flags u8 | len u16 | seq u16 | payload[len] | crc32 u32

with the CRC over `type` through the end of the payload. The decoder is incremental, keeps a
count of bad CRCs and resynchronisations, and never blocks: it is fed whatever bytes arrived.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

SYNC = b"\xb1\xc5"
HDR_BYTES = 6   # type, flags, len, seq -- after the two sync bytes
CRC_BYTES = 4
MAX_PAYLOAD = 2048

MSG_HELLO = 0x01
MSG_TARGET_BUNDLE = 0x02
MSG_BUNDLE_ACK = 0x03
MSG_SCENE_STATE = 0x04
MSG_BUS_READINGS = 0x05
MSG_TARGETS = 0x06
MSG_TICK_LOG = 0x07
MSG_STATUS = 0x08
MSG_ERROR = 0x09
MSG_TICK_BEGIN = 0x0A
MSG_RESET = 0x0B

NAMES = {
    MSG_HELLO: "HELLO", MSG_TARGET_BUNDLE: "TARGET_BUNDLE", MSG_BUNDLE_ACK: "BUNDLE_ACK",
    MSG_SCENE_STATE: "SCENE_STATE", MSG_BUS_READINGS: "BUS_READINGS", MSG_TARGETS: "TARGETS",
    MSG_TICK_LOG: "TICK_LOG", MSG_STATUS: "STATUS", MSG_ERROR: "ERROR",
    MSG_TICK_BEGIN: "TICK_BEGIN", MSG_RESET: "RESET",
}

ERRORS = {1: "bad CRC", 2: "bad length", 3: "bad sequence", 4: "bundle too large",
          5: "wrong state"}

NJOINT = 6
K = 8
W = 14

#: Bytes of bundle payload per TARGET_BUNDLE frame, leaving room for the 8-byte chunk header.
CHUNK = MAX_PAYLOAD - 8


class ProtocolError(RuntimeError):
    pass


@dataclass
class Frame:
    type: int
    flags: int
    seq: int
    payload: bytes

    @property
    def name(self) -> str:
        return NAMES.get(self.type, f"0x{self.type:02x}")


def encode(msg_type: int, payload: bytes = b"", *, seq: int = 0, flags: int = 0) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload of {len(payload)} B exceeds {MAX_PAYLOAD} B")
    head = struct.pack("<BBHH", msg_type, flags, len(payload), seq & 0xFFFF)
    crc = zlib.crc32(head + payload) & 0xFFFFFFFF
    return SYNC + head + payload + struct.pack("<I", crc)


class Decoder:
    """Incremental frame decoder. `feed` returns however many whole frames the bytes completed."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.bad_crc = 0
        self.resyncs = 0

    def feed(self, data: bytes) -> list[Frame]:
        self.buf.extend(data)
        out: list[Frame] = []
        while True:
            frame = self._one()
            if frame is None:
                return out
            out.append(frame)

    def _one(self) -> Frame | None:
        while True:
            at = self.buf.find(SYNC)
            if at < 0:
                # Keep the last byte: it may be the first half of a sync pair.
                if len(self.buf) > 1:
                    del self.buf[:-1]
                return None
            if at > 0:
                del self.buf[:at]
                self.resyncs += 1
            if len(self.buf) < 2 + HDR_BYTES:
                return None
            msg_type, flags, length, seq = struct.unpack_from("<BBHH", self.buf, 2)
            if length > MAX_PAYLOAD:
                del self.buf[:2]
                self.resyncs += 1
                continue
            total = 2 + HDR_BYTES + length + CRC_BYTES
            if len(self.buf) < total:
                return None
            head = bytes(self.buf[2:2 + HDR_BYTES])
            payload = bytes(self.buf[2 + HDR_BYTES:2 + HDR_BYTES + length])
            (want,) = struct.unpack_from("<I", self.buf, 2 + HDR_BYTES + length)
            if (zlib.crc32(head + payload) & 0xFFFFFFFF) != want:
                self.bad_crc += 1
                del self.buf[:2]
                self.resyncs += 1
                continue
            del self.buf[:total]
            return Frame(type=msg_type, flags=flags, seq=seq, payload=payload)


# ------------------------------------------------------------------------------ payloads ---


def bundle_chunks(blob: bytes, *, chunk: int = CHUNK):
    """TARGET_BUNDLE payloads: total size, this chunk's offset, then the bytes."""
    total = len(blob)
    for off in range(0, total, chunk):
        yield struct.pack("<II", total, off) + blob[off:off + chunk]


def scene_state(frame_id: int, host_t: float, slots, depth, slot_ids) -> bytes:
    body = struct.pack("<Id", int(frame_id), float(host_t))
    body += np.ascontiguousarray(slots, dtype="<f4").tobytes()
    body += np.ascontiguousarray(depth, dtype="<f4").tobytes()
    body += np.ascontiguousarray(slot_ids, dtype="<i4").tobytes()
    return body


def bus_readings(position, speed, load) -> bytes:
    return (np.ascontiguousarray(position, dtype="<f8").tobytes()
            + np.ascontiguousarray(speed, dtype="<f8").tobytes()
            + np.ascontiguousarray(load, dtype="<f8").tobytes())


@dataclass
class Targets:
    k: int
    step_index: int
    step_id: int
    commands: list


@dataclass
class TickCommand:
    pos: np.ndarray
    spin: float
    grip: float
    c_vert: float
    c_lat: float
    q: np.ndarray
    ik_ok: bool


def parse_targets(payload: bytes) -> Targets:
    n, k, step_index, step_id = struct.unpack_from("<4i", payload, 0)
    commands = []
    off = 16
    for _ in range(n):
        values = struct.unpack_from("<13d", payload, off)
        (ik_ok,) = struct.unpack_from("<i", payload, off + 104)
        commands.append(TickCommand(
            pos=np.array(values[0:3]), spin=values[3], grip=values[4],
            c_vert=values[5], c_lat=values[6], q=np.array(values[7:13]), ik_ok=bool(ik_ok)))
        off += 112
    return Targets(k=k, step_index=step_index, step_id=step_id, commands=commands)


@dataclass
class TickLog:
    k: int
    step_index: int
    step_id: int
    holds: int
    refs_seen: int
    done: bool
    failure: int
    regrasps: int
    matched: int
    ik_failures: int
    correction: np.ndarray
    err_mm: float
    tol_mm: float
    absolute_mm: float
    predict_mm: float
    yaw_err: float


def parse_tick_log(payload: bytes) -> TickLog:
    ints = struct.unpack_from("<10i", payload, 0)
    doubles = struct.unpack_from("<8d", payload, 40)
    return TickLog(
        k=ints[0], step_index=ints[1], step_id=ints[2], holds=ints[3], refs_seen=ints[4],
        done=bool(ints[5]), failure=ints[6], regrasps=ints[7], matched=ints[8],
        ik_failures=ints[9], correction=np.array(doubles[0:3]), err_mm=doubles[3],
        tol_mm=doubles[4], absolute_mm=doubles[5], predict_mm=doubles[6], yaw_err=doubles[7])


@dataclass
class Status:
    version: int
    n_frames: int
    ticks: int
    bad_crc: int
    resyncs: int
    ik_failures: int
    tick_us_mean: float
    tick_us_max: float


def parse_status(payload: bytes) -> Status:
    version, n_frames, ticks, bad_crc, resyncs = struct.unpack_from("<5I", payload, 0)
    (ik_failures,) = struct.unpack_from("<i", payload, 20)
    mean, worst = struct.unpack_from("<2d", payload, 24)
    return Status(version=version, n_frames=n_frames, ticks=ticks, bad_crc=bad_crc,
                  resyncs=resyncs, ik_failures=ik_failures, tick_us_mean=mean,
                  tick_us_max=worst)


def parse_error(payload: bytes) -> tuple[int, int]:
    return struct.unpack_from("<2i", payload, 0)


def parse_bundle_ack(payload: bytes) -> tuple[int, int]:
    status, received = struct.unpack_from("<iI", payload, 0)
    return status, received
