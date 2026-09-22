"""Feetech STS3215 serial protocol: packet framing, the memory table, and unit conversion.

Pure Python over ``bytes``; no serial port, no MuJoCo, no numpy dependency on the hot path, so
the same code runs in a unit test, behind a pseudo-terminal and against real hardware.

Protocol
--------
The STS/SMS series speaks the same frame layout as Dynamixel 1.0::

    instruction   FF FF  ID  LEN  INSTR  PARAM...  CHK
    status        FF FF  ID  LEN  ERROR  PARAM...  CHK

``LEN`` counts the bytes after it (parameters + instruction/error + checksum), and
``CHK = (~(ID + LEN + INSTR + sum(PARAM))) & 0xFF``.  Instructions used here: PING (0x01),
READ (0x02), WRITE (0x03), REG WRITE (0x04), ACTION (0x05), SYNC READ (0x82), SYNC WRITE (0x83),
addressed to one id or to the broadcast id 0xFE.

Two STS-specific details separate this from a Dynamixel driver:

* **Little-endian 16-bit registers.**  The STS/SMS series stores the low byte first; the older
  SCS series stores the high byte first.  Feetech's own SDK carries this as the ``END`` flag,
  0 for ``sms_sts`` and 1 for ``scscl``.
* **Sign-magnitude, not two's complement.**  ``Goal_Speed``, ``Present_Speed`` and
  ``Present_Current`` put the sign in bit 15 and the magnitude in the remaining bits;
  ``Present_Load`` puts it in bit 10 over a 0-1000 magnitude.  Feetech's SDK spells this
  ``SCS_TOHOST(value, bit)``.

Source for the memory table and the two conventions above: Feetech's published STS3215 memory
table (STS/SMS series datasheet), as reproduced in Feetech's SCServo SDK (``SCServo_Python``,
``sms_sts.py`` and its register constants).  Register semantics this project already depended on
-- P at 21, the dead band at 26/27, ``Torque_Limit`` at 48, ``Present_Position`` 56,
``Present_Speed`` 58, ``Present_Load`` 60, ``Present_Current`` 69 -- match
``brick_builder/robot/servo_model.py``, which is the cross-check that this table is the same part.

Units
-----
The encoder is 12 bit over one output-shaft turn, so one count is ``2*pi/4096`` rad of *link*
angle (the encoder sits past the 1:345 gearbox; see guide 03).  ``Present_Load`` is a PWM duty
cycle in 0.1 % steps, not a torque.  ``Present_Voltage`` is 0.1 V per count and
``Present_Temperature`` is 1 degree Celsius per count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------------- framing ---

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE
MAX_ID = 0xFD

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_REG_WRITE = 0x04
INST_ACTION = 0x05
INST_RESET = 0x06
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83

INSTRUCTION_NAMES = {
    INST_PING: "PING",
    INST_READ: "READ",
    INST_WRITE: "WRITE",
    INST_REG_WRITE: "REG_WRITE",
    INST_ACTION: "ACTION",
    INST_RESET: "RESET",
    INST_SYNC_READ: "SYNC_READ",
    INST_SYNC_WRITE: "SYNC_WRITE",
}

ERROR_BITS: tuple[tuple[int, str], ...] = (
    (0x01, "voltage"),
    (0x02, "angle sensor"),
    (0x04, "temperature"),
    (0x08, "current"),
    (0x10, "angle"),
    (0x20, "overload"),
)
"""Bit layout of the status packet's error byte.

The same layout the memory table gives for ``Unloading_Condition`` (19) and
``LED_Alarm_Condition`` (20), whose factory values 44 and 47 decode as temperature+current+overload
and voltage+angle-sensor+temperature+current respectively.
"""


class ProtocolError(RuntimeError):
    """A frame that does not parse: bad header, bad length, bad checksum, or a short read."""


class ChecksumError(ProtocolError):
    """A frame whose checksum does not match its contents."""


class StatusError(RuntimeError):
    """A servo answered, and its error byte is non-zero."""

    def __init__(self, servo_id: int, error: int) -> None:
        self.servo_id = int(servo_id)
        self.error = int(error)
        super().__init__(f"servo {self.servo_id}: {describe_error(error)}")


def describe_error(error: int) -> str:
    """Human-readable decode of a status packet's error byte."""
    error = int(error)
    if error == 0:
        return "no error"
    names = [name for bit, name in ERROR_BITS if error & bit]
    unknown = error & ~sum(bit for bit, _ in ERROR_BITS)
    if unknown:
        names.append(f"unknown bits 0x{unknown:02x}")
    return ", ".join(names)


def checksum(body: Sequence[int]) -> int:
    """``(~sum(body)) & 0xFF`` over everything between the header and the checksum itself."""
    return (~sum(body)) & 0xFF


def _frame(body: list[int]) -> bytes:
    return HEADER + bytes(body) + bytes([checksum(body)])


def build_instruction(servo_id: int, instruction: int, params: Sequence[int] = ()) -> bytes:
    """One instruction packet.  ``params`` is the instruction's payload, without the checksum."""
    params = [int(p) & 0xFF for p in params]
    return _frame([int(servo_id) & 0xFF, len(params) + 2, int(instruction) & 0xFF, *params])


def build_status(servo_id: int, error: int = 0, params: Sequence[int] = ()) -> bytes:
    """One status packet, which is what the servo (or the virtual bus) sends back."""
    params = [int(p) & 0xFF for p in params]
    return _frame([int(servo_id) & 0xFF, len(params) + 2, int(error) & 0xFF, *params])


@dataclass(frozen=True)
class Packet:
    """A parsed frame.  ``code`` is the instruction byte on the way out, the error byte back."""

    servo_id: int
    code: int
    params: bytes

    @property
    def error(self) -> int:
        return self.code

    @property
    def instruction(self) -> int:
        return self.code


def parse(buf: bytes) -> Packet:
    """Parse exactly one frame.  Raises :class:`ProtocolError` on anything that is not one."""
    packet, used = take_packet(buf)
    if packet is None:
        raise ProtocolError(f"not a complete packet: {buf.hex(' ')}")
    if used != len(buf):
        raise ProtocolError(f"{len(buf) - used} trailing byte(s) after the packet")
    return packet


def take_packet(buf: bytes) -> tuple[Packet | None, int]:
    """Pull the first complete frame out of a byte stream.

    Returns ``(packet, bytes_consumed)``.  ``(None, n)`` means no complete frame yet and ``n``
    leading bytes are junk that can be discarded -- which is how the driver resynchronizes after
    a collision on the half-duplex line rather than giving up on the port.  A frame whose
    checksum fails raises, because a resync would silently hide a wiring fault.
    """
    start = buf.find(HEADER)
    if start < 0:
        # Keep the last byte: it may be the first 0xFF of a header still arriving.
        return None, max(len(buf) - 1, 0)
    if len(buf) < start + 4:
        return None, start
    length = buf[start + 3]
    if length < 2:
        raise ProtocolError(f"length byte {length} is below the minimum of 2")
    end = start + 4 + length
    if len(buf) < end:
        return None, start
    body = buf[start + 2:end - 1]
    if checksum(body) != buf[end - 1]:
        raise ChecksumError(
            f"checksum 0x{buf[end - 1]:02x}, expected 0x{checksum(body):02x}, "
            f"frame {buf[start:end].hex(' ')}"
        )
    return Packet(servo_id=buf[start + 2], code=buf[start + 4], params=bytes(buf[start + 5:end - 1])), end


# ------------------------------------------------------------------------- word conventions ---


def to_bytes16(value: int) -> tuple[int, int]:
    """Split a 16-bit register value into (low, high) -- the STS/SMS byte order."""
    value = int(value) & 0xFFFF
    return value & 0xFF, (value >> 8) & 0xFF


def from_bytes16(low: int, high: int) -> int:
    """Join (low, high) back into a 16-bit register value."""
    return ((int(high) & 0xFF) << 8) | (int(low) & 0xFF)


def to_signed(value: int, sign_bit: int = 15) -> int:
    """Sign-magnitude register value to a signed integer (Feetech's ``SCS_TOHOST``)."""
    value = int(value)
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


def from_signed(value: int, sign_bit: int = 15) -> int:
    """Signed integer to a sign-magnitude register value (Feetech's ``SCS_TOSCS``)."""
    value = int(value)
    if value < 0:
        return (-value & ((1 << sign_bit) - 1)) | (1 << sign_bit)
    return value & ((1 << sign_bit) - 1)


# --------------------------------------------------------------------------- memory table ---


@dataclass(frozen=True)
class Register:
    """One entry of the STS3215 memory table."""

    address: int
    size: int
    name: str
    area: str            # "eprom" or "sram"
    unit: str = "-"
    signed_bit: int | None = None
    """Bit holding the sign when the register is sign-magnitude; ``None`` when it is unsigned."""

    @property
    def writable(self) -> bool:
        """Can the host write it?  The ``Present_*`` block and the version bytes cannot."""
        if self.address in _READ_ONLY:
            return False
        return self.area == "sram" or self.address in _WRITABLE_EPROM


_WRITABLE_EPROM = {5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28,
                   30, 31, 33, 34, 35, 36, 37, 38, 39}

_READ_ONLY = {0, 1, 3, 4, 56, 58, 60, 62, 63, 65, 66, 69}

REGISTERS: dict[str, Register] = {
    # -- EPROM -----------------------------------------------------------------------------
    "firmware_major": Register(0, 1, "firmware_major", "eprom"),
    "firmware_minor": Register(1, 1, "firmware_minor", "eprom"),
    "servo_major": Register(3, 1, "servo_major", "eprom"),
    "servo_minor": Register(4, 1, "servo_minor", "eprom"),
    "id": Register(5, 1, "id", "eprom"),
    "baud_rate": Register(6, 1, "baud_rate", "eprom"),
    "return_delay": Register(7, 1, "return_delay", "eprom", "2 us"),
    "status_return_level": Register(8, 1, "status_return_level", "eprom"),
    "min_angle_limit": Register(9, 2, "min_angle_limit", "eprom", "counts"),
    "max_angle_limit": Register(11, 2, "max_angle_limit", "eprom", "counts"),
    "max_temperature": Register(13, 1, "max_temperature", "eprom", "deg C"),
    "max_voltage": Register(14, 1, "max_voltage", "eprom", "0.1 V"),
    "min_voltage": Register(15, 1, "min_voltage", "eprom", "0.1 V"),
    "max_torque": Register(16, 2, "max_torque", "eprom", "0.1 %"),
    "phase": Register(18, 1, "phase", "eprom"),
    "unloading_condition": Register(19, 1, "unloading_condition", "eprom"),
    "led_alarm_condition": Register(20, 1, "led_alarm_condition", "eprom"),
    "p_coefficient": Register(21, 1, "p_coefficient", "eprom"),
    "d_coefficient": Register(22, 1, "d_coefficient", "eprom"),
    "i_coefficient": Register(23, 1, "i_coefficient", "eprom"),
    "min_startup_force": Register(24, 2, "min_startup_force", "eprom", "0.1 %"),
    "cw_dead_band": Register(26, 1, "cw_dead_band", "eprom", "counts"),
    "ccw_dead_band": Register(27, 1, "ccw_dead_band", "eprom", "counts"),
    "protection_current": Register(28, 2, "protection_current", "eprom", "6.5 mA"),
    "angular_resolution": Register(30, 1, "angular_resolution", "eprom"),
    "position_correction": Register(31, 2, "position_correction", "eprom", "counts", signed_bit=11),
    "operation_mode": Register(33, 1, "operation_mode", "eprom"),
    "protective_torque": Register(34, 1, "protective_torque", "eprom", "%"),
    "protection_time": Register(35, 1, "protection_time", "eprom", "10 ms"),
    "overload_torque": Register(36, 1, "overload_torque", "eprom", "%"),
    "speed_p_coefficient": Register(37, 1, "speed_p_coefficient", "eprom"),
    "overcurrent_time": Register(38, 1, "overcurrent_time", "eprom", "10 ms"),
    "speed_i_coefficient": Register(39, 1, "speed_i_coefficient", "eprom"),
    # -- SRAM ------------------------------------------------------------------------------
    "torque_enable": Register(40, 1, "torque_enable", "sram"),
    "acceleration": Register(41, 1, "acceleration", "sram", "100 counts/s^2"),
    "goal_position": Register(42, 2, "goal_position", "sram", "counts"),
    "running_time": Register(44, 2, "running_time", "sram", "ms"),
    "goal_speed": Register(46, 2, "goal_speed", "sram", "counts/s", signed_bit=15),
    "torque_limit": Register(48, 2, "torque_limit", "sram", "0.1 %"),
    "lock": Register(55, 1, "lock", "sram"),
    "present_position": Register(56, 2, "present_position", "sram", "counts"),
    "present_speed": Register(58, 2, "present_speed", "sram", "counts/s", signed_bit=15),
    "present_load": Register(60, 2, "present_load", "sram", "0.1 %", signed_bit=10),
    "present_voltage": Register(62, 1, "present_voltage", "sram", "0.1 V"),
    "present_temperature": Register(63, 1, "present_temperature", "sram", "deg C"),
    "async_write_flag": Register(64, 1, "async_write_flag", "sram"),
    "servo_status": Register(65, 1, "servo_status", "sram"),
    "moving": Register(66, 1, "moving", "sram"),
    "present_current": Register(69, 2, "present_current", "sram", "6.5 mA", signed_bit=15),
}
"""The STS3215 memory table, by name.  See the module docstring for the source."""

REGISTERS_BY_ADDRESS: dict[int, Register] = {r.address: r for r in REGISTERS.values()}

TORQUE_DISABLED = 0
TORQUE_ENABLED = 1
LOCK_OPEN = 0
"""``lock`` (55) open: EPROM writes take effect.  1 locks EPROM and is the power-on state."""
LOCK_CLOSED = 1

BAUD_CODES: dict[int, int] = {
    1_000_000: 0, 500_000: 1, 250_000: 2, 128_000: 3,
    115_200: 4, 76_800: 5, 57_600: 6, 38_400: 7,
}
"""``baud_rate`` register (6) code for each supported line rate."""

STATE_BLOCK_ADDRESS = REGISTERS["present_position"].address
STATE_BLOCK_LENGTH = (
    REGISTERS["present_current"].address + REGISTERS["present_current"].size - STATE_BLOCK_ADDRESS
)
"""One sync-read covering 56-70: position, speed, load, voltage, temperature, status, moving,
current.  15 bytes, which is the ``read_payload`` ``robot/bus_emulator.py`` times the wire with."""


# ------------------------------------------------------------------------------ conversions ---

TICKS_PER_REV = 4096
"""12-bit encoder over one *output-shaft* turn (guide 03: the encoder is past the gearbox)."""

RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REV
DEG_PER_TICK = 360.0 / TICKS_PER_REV
VOLT_PER_COUNT = 0.1
AMP_PER_COUNT = 0.0065
DUTY_PER_COUNT = 0.001
"""``Present_Load`` is duty in 0.1 % steps -- matching ``ServoConfig.load_lsb``."""

CENTRE_TICKS = 2048
"""Mid-scale, and the tick a calibrated SO-101 joint reads at its own zero angle."""


@dataclass(frozen=True)
class JointCalibration:
    """Tick-to-radian mapping and travel limits for one servo.

    ``centre`` is the count the joint reads at 0 rad, ``direction`` is +1 when counts rise with
    the MJCF joint angle and -1 when they fall, and ``lo``/``hi`` are the joint's own travel
    limits in radians.  The shipped values are the simulator's: every joint centred at 2048 with
    ``direction`` +1, and the limits of ``third_party/so101/so101_new_calib.xml``.  The real arm's
    per-unit homing offsets and drive directions are a measurement (guide 08), not a constant.
    """

    name: str
    centre: int = CENTRE_TICKS
    direction: int = 1
    lo: float = -math.pi
    hi: float = math.pi

    def to_ticks(self, rad: float) -> int:
        return int(round(self.centre + self.direction * float(rad) / RAD_PER_TICK))

    def to_rad(self, ticks: float) -> float:
        return self.direction * (float(ticks) - self.centre) * RAD_PER_TICK

    def speed_to_rad_s(self, counts_per_s: float) -> float:
        return self.direction * float(counts_per_s) * RAD_PER_TICK

    def speed_to_counts(self, rad_s: float) -> int:
        return int(round(self.direction * float(rad_s) / RAD_PER_TICK))

    def clamp(self, rad: float) -> float:
        return min(max(float(rad), self.lo), self.hi)

    @property
    def lo_ticks(self) -> int:
        return min(self.to_ticks(self.lo), self.to_ticks(self.hi))

    @property
    def hi_ticks(self) -> int:
        return max(self.to_ticks(self.lo), self.to_ticks(self.hi))


SO101_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.9198621771937616, 1.9198621771937634),
    "shoulder_lift": (-1.7453292519943224, 1.7453292519943366),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.6580628494556928, 1.6580627293335335),
    "wrist_roll": (-2.7438472969992493, 2.841206309382605),
    "gripper": (-0.17453297762778586, 1.7453291995659765),
}
"""Joint travel limits [rad] of the vendored SO-101, in servo-id order when zipped with
:data:`SERVO_IDS`.  Copied from the compiled model rather than re-derived; a test in
``tests/hardware/`` compiles the arm and checks every bound still matches, so an upstream
geometry change cannot leave the driver enforcing a stale limit.
"""

SERVO_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
"""Bus ids of the six servos, in the order of ``patch_so101.ARM_JOINTS``."""

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)


def so101_calibration() -> tuple[JointCalibration, ...]:
    """The six joints' shipped calibration, in servo-id order."""
    return tuple(
        JointCalibration(name=name, centre=CENTRE_TICKS, direction=1,
                         lo=SO101_JOINT_LIMITS[name][0], hi=SO101_JOINT_LIMITS[name][1])
        for name in JOINT_NAMES
    )


# ------------------------------------------------------------------------------ packet helpers ---


def ping(servo_id: int) -> bytes:
    return build_instruction(servo_id, INST_PING)


def read(servo_id: int, address: int, length: int) -> bytes:
    return build_instruction(servo_id, INST_READ, [address, length])


def write(servo_id: int, address: int, data: Sequence[int]) -> bytes:
    return build_instruction(servo_id, INST_WRITE, [address, *data])


def reg_write(servo_id: int, address: int, data: Sequence[int]) -> bytes:
    """Stage a write; every staged write lands at the next :func:`action` (register 64)."""
    return build_instruction(servo_id, INST_REG_WRITE, [address, *data])


def action(servo_id: int = BROADCAST_ID) -> bytes:
    return build_instruction(servo_id, INST_ACTION)


def write_word(servo_id: int, address: int, value: int) -> bytes:
    return write(servo_id, address, list(to_bytes16(value)))


def sync_write(address: int, payload: Mapping[int, Sequence[int]]) -> bytes:
    """One sync-write: the same register block to several servos in one frame.

    ``payload`` maps servo id to that servo's data bytes; every entry must be the same length,
    because the frame carries one data length for all of them.
    """
    items = list(payload.items())
    if not items:
        raise ValueError("sync_write needs at least one servo")
    size = len(items[0][1])
    if any(len(data) != size for _, data in items):
        raise ValueError("every servo in a sync-write must carry the same number of bytes")
    params: list[int] = [address, size]
    for servo_id, data in items:
        params.append(servo_id)
        params.extend(data)
    return build_instruction(BROADCAST_ID, INST_SYNC_WRITE, params)


def sync_read(address: int, length: int, ids: Iterable[int]) -> bytes:
    """One sync-read request; each addressed servo answers with its own status packet, in order."""
    ids = list(ids)
    if not ids:
        raise ValueError("sync_read needs at least one servo")
    return build_instruction(BROADCAST_ID, INST_SYNC_READ, [address, length, *ids])


def sync_write_goals(ids: Sequence[int], ticks: Sequence[int]) -> bytes:
    """Goal_Position (42) for every servo in one frame."""
    return sync_write(REGISTERS["goal_position"].address,
                      {i: list(to_bytes16(t)) for i, t in zip(ids, ticks)})


def sync_read_state(ids: Sequence[int]) -> bytes:
    """Sync-read of addresses 56-70 -- the block :data:`STATE_BLOCK_LENGTH` describes."""
    return sync_read(STATE_BLOCK_ADDRESS, STATE_BLOCK_LENGTH, ids)


@dataclass(frozen=True)
class ServoState:
    """One servo's 56-70 block, decoded into engineering units."""

    position_ticks: int
    speed_counts: int
    load_counts: int
    voltage_v: float
    temperature_c: int
    status: int
    moving: bool
    current_a: float

    @property
    def duty(self) -> float:
        """``Present_Load`` as a PWM duty cycle in [-1, 1].  Not a torque (guide 03)."""
        return self.load_counts * DUTY_PER_COUNT


def decode_state(block: bytes) -> ServoState:
    """Decode the 15-byte 56-70 block a sync-read returns."""
    if len(block) != STATE_BLOCK_LENGTH:
        raise ProtocolError(f"state block is {len(block)} bytes, expected {STATE_BLOCK_LENGTH}")
    position = from_bytes16(block[0], block[1])
    speed = to_signed(from_bytes16(block[2], block[3]), 15)
    load = to_signed(from_bytes16(block[4], block[5]), 10)
    return ServoState(
        position_ticks=position,
        speed_counts=speed,
        load_counts=load,
        voltage_v=block[6] * VOLT_PER_COUNT,
        temperature_c=block[7],
        status=block[9],
        moving=bool(block[10]),
        current_a=to_signed(from_bytes16(block[13], block[14]), 15) * AMP_PER_COUNT,
    )


def encode_state(state: ServoState) -> bytes:
    """Inverse of :func:`decode_state` -- what the virtual bus puts on the wire."""
    out = bytearray(STATE_BLOCK_LENGTH)
    out[0], out[1] = to_bytes16(int(state.position_ticks))
    out[2], out[3] = to_bytes16(from_signed(int(state.speed_counts), 15))
    out[4], out[5] = to_bytes16(from_signed(int(state.load_counts), 10))
    out[6] = int(round(state.voltage_v / VOLT_PER_COUNT)) & 0xFF
    out[7] = int(state.temperature_c) & 0xFF
    out[9] = int(state.status) & 0xFF
    out[10] = 1 if state.moving else 0
    out[13], out[14] = to_bytes16(from_signed(int(round(state.current_a / AMP_PER_COUNT)), 15))
    return bytes(out)
