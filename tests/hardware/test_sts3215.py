"""Protocol framing, the memory table and the tick/radian conversion."""

import math

import numpy as np
import pytest

from brick_builder.hardware import sts3215 as proto


def test_instruction_frame_matches_the_documented_layout():
    frame = proto.ping(1)
    assert frame[:2] == proto.HEADER
    assert frame[2] == 1 and frame[3] == 2 and frame[4] == proto.INST_PING
    assert frame[-1] == proto.checksum(frame[2:-1])
    packet = proto.parse(frame)
    assert packet.servo_id == 1 and packet.instruction == proto.INST_PING and packet.params == b""


def test_sync_write_carries_one_length_for_every_servo():
    frame = proto.sync_write_goals([1, 2, 3], [2048, 1024, 3072])
    packet = proto.parse(frame)
    assert packet.servo_id == proto.BROADCAST_ID
    assert packet.instruction == proto.INST_SYNC_WRITE
    assert packet.params[0] == proto.REGISTERS["goal_position"].address
    assert packet.params[1] == 2
    body = packet.params[2:]
    assert [body[0], body[3], body[6]] == [1, 2, 3]
    assert proto.from_bytes16(body[1], body[2]) == 2048
    assert proto.from_bytes16(body[7], body[8]) == 3072


def test_sync_write_refuses_ragged_payloads():
    with pytest.raises(ValueError):
        proto.sync_write(42, {1: [0, 0], 2: [0]})


def test_a_corrupted_checksum_is_not_silently_accepted():
    frame = bytearray(proto.read(3, 56, 15))
    frame[-1] ^= 0xFF
    with pytest.raises(proto.ChecksumError):
        proto.parse(bytes(frame))


def test_the_parser_resynchronizes_past_junk_on_the_line():
    frame = proto.build_status(4, 0, b"\x01\x02")
    packet, used = proto.take_packet(b"\x00\x11\x22" + frame)
    assert packet.servo_id == 4 and packet.params == b"\x01\x02"
    assert used == 3 + len(frame)


def test_a_partial_frame_asks_for_more_bytes_rather_than_failing():
    frame = proto.build_status(4, 0, b"\x01\x02")
    packet, used = proto.take_packet(frame[:-1])
    assert packet is None and used == 0


@pytest.mark.parametrize("value", [-1000, -1, 0, 1, 1000])
def test_sign_magnitude_round_trips(value):
    assert proto.to_signed(proto.from_signed(value, 15), 15) == value
    if abs(value) < 1024:
        assert proto.to_signed(proto.from_signed(value, 10), 10) == value


def test_the_state_block_is_the_fifteen_bytes_of_addresses_56_to_70():
    assert proto.STATE_BLOCK_ADDRESS == 56
    assert proto.STATE_BLOCK_LENGTH == 15
    state = proto.ServoState(position_ticks=2047, speed_counts=-321, load_counts=-450,
                             voltage_v=7.4, temperature_c=31, status=0x20, moving=True,
                             current_a=0.234)
    back = proto.decode_state(proto.encode_state(state))
    assert back.position_ticks == 2047
    assert back.speed_counts == -321
    assert back.load_counts == -450
    assert back.temperature_c == 31 and back.status == 0x20 and back.moving
    assert back.voltage_v == pytest.approx(7.4, abs=0.05)
    assert back.current_a == pytest.approx(0.234, abs=proto.AMP_PER_COUNT)
    assert back.duty == pytest.approx(-0.45)


def test_present_load_is_a_duty_cycle_in_tenths_of_a_percent():
    assert proto.DUTY_PER_COUNT == 0.001
    assert proto.REGISTERS["present_load"].signed_bit == 10


def test_registers_the_driver_names_sit_where_the_memory_table_puts_them():
    expected = {
        "torque_enable": 40, "goal_position": 42, "goal_speed": 46, "torque_limit": 48,
        "lock": 55, "present_position": 56, "present_speed": 58, "present_load": 60,
        "present_voltage": 62, "present_temperature": 63, "moving": 66,
        "p_coefficient": 21, "cw_dead_band": 26, "ccw_dead_band": 27, "present_current": 69,
    }
    for name, address in expected.items():
        assert proto.REGISTERS[name].address == address


def test_one_encoder_count_is_one_four_thousand_and_ninety_sixth_of_a_turn():
    assert proto.TICKS_PER_REV == 4096
    assert proto.DEG_PER_TICK == pytest.approx(0.087890625)
    assert proto.RAD_PER_TICK == pytest.approx(2.0 * math.pi / 4096)


def test_calibration_maps_zero_radians_to_mid_scale_and_back():
    for cal in proto.so101_calibration():
        assert cal.to_ticks(0.0) == proto.CENTRE_TICKS
        for angle in np.linspace(cal.lo, cal.hi, 9):
            ticks = cal.to_ticks(angle)
            assert abs(cal.to_rad(ticks) - angle) <= 0.5 * proto.RAD_PER_TICK + 1e-12
            assert cal.lo_ticks <= ticks <= cal.hi_ticks


def test_the_shipped_joint_limits_still_match_the_compiled_arm():
    import mujoco

    from brick_builder.robot.patch_so101 import ArmConfig, build_arm_spec

    model = build_arm_spec(ArmConfig(backlash_deg=0.0)).compile()
    for name, (lo, hi) in proto.SO101_JOINT_LIMITS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, name
        assert model.jnt_range[jid][0] == pytest.approx(lo)
        assert model.jnt_range[jid][1] == pytest.approx(hi)


def test_error_bits_decode_the_way_the_factory_alarm_registers_do():
    # Unloading_Condition ships at 44 = temperature + current + overload.
    assert proto.describe_error(44) == "temperature, current, overload"
    assert proto.describe_error(0) == "no error"
