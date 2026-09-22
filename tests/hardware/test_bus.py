"""The driver against the virtual bus: framing over a real file descriptor, and the safety rules.

Every test here opens a pseudo-terminal and drives it with the same ``Bus`` object the real arm
will be driven with, so the bytes, the timeouts and the resynchronization are exercised rather
than mocked.  The register-file backend is used wherever physics is not the point.
"""

import numpy as np
import pytest

from brick_builder.hardware import sts3215 as proto
from brick_builder.hardware.bus import Bus, BusError, BusSettings
from brick_builder.hardware.virtual_bus import (
    RegisterServos,
    ServoDefaults,
    VirtualBus,
    WireConfig,
    conformance_run,
)

SETTINGS = BusSettings(timeout_s=0.25, retries=2)


def flush(driver):
    """Wait for the far end to have applied a fire-and-forget write.

    A sync-write draws no reply, so the only way to know the virtual bus has processed one is to
    follow it with something that does: a ping the far end must answer after it."""
    assert driver.ping(driver.ids[0])


@pytest.fixture
def bus():
    vbus = VirtualBus(RegisterServos(), WireConfig()).start()
    driver = Bus(vbus.port, SETTINGS)
    try:
        yield driver, vbus
    finally:
        driver.close()
        vbus.stop()
        vbus.raise_if_failed()


def test_every_servo_answers_a_ping_and_a_missing_one_does_not(bus):
    driver, _ = bus
    assert all(driver.ping(i) for i in driver.ids)
    assert not driver.ping(9)


def test_one_cycle_writes_the_goals_and_reads_the_state_block_back(bus):
    driver, vbus = bus
    state = driver.cycle(np.zeros(6))
    assert state.valid.all()
    assert state.ticks.tolist() == [proto.CENTRE_TICKS] * 6
    assert state.voltage.tolist() == [pytest.approx(7.4, abs=0.05)] * 6
    assert driver.counters.errors == 0
    for servo_id in driver.ids:
        assert vbus.servos.get(servo_id, "goal_position") == proto.CENTRE_TICKS


def test_goals_are_clipped_to_each_joint_s_own_travel(bus):
    driver, vbus = bus
    driver.cycle(np.full(6, 10.0))
    for servo_id, cal in zip(driver.ids, driver.calibration):
        assert vbus.servos.get(servo_id, "goal_position") == cal.hi_ticks
    driver.write_ticks([99999] * 6)
    flush(driver)
    for servo_id, cal in zip(driver.ids, driver.calibration):
        assert vbus.servos.get(servo_id, "goal_position") == cal.hi_ticks


def test_the_torque_limit_register_takes_a_fraction_of_the_servo_s_own_limit(bus):
    driver, vbus = bus
    driver.set_torque_limit(np.full(6, 0.6))
    flush(driver)
    assert [vbus.servos.get(i, "torque_limit") for i in driver.ids] == [600] * 6


def test_the_soft_stop_clears_torque_enable_on_every_servo(bus):
    driver, vbus = bus
    driver.enable_torque()
    flush(driver)
    assert all(vbus.servos.get(i, "torque_enable") == 1 for i in driver.ids)
    driver.disable_torque()
    flush(driver)
    assert all(vbus.servos.get(i, "torque_enable") == 0 for i in driver.ids)
    assert not driver.torque_enabled


def test_the_self_test_passes_on_a_healthy_bus_and_reads_the_four_registers(bus):
    driver, _ = bus
    result = driver.self_test()
    assert result.ok and not result.missing
    assert set(result.present) == set(driver.ids)
    assert all(v == pytest.approx(7.4, abs=0.05) for v in result.voltage_v.values())
    assert set(result.p_gain.values()) == {ServoDefaults().p_gain}
    assert "self-test: pass" in result.report()


def test_torque_is_refused_when_a_servo_is_missing():
    vbus = VirtualBus(RegisterServos(ids=(1, 2, 3, 4, 5)), WireConfig()).start()
    driver = Bus(vbus.port, BusSettings(timeout_s=0.05, retries=0))
    try:
        result = driver.self_test()
        assert not result.ok and result.missing == [6]
        with pytest.raises(BusError):
            driver.enable_torque()
    finally:
        driver.close()
        vbus.stop()


def test_torque_is_refused_when_the_supply_is_outside_its_window():
    servos = RegisterServos(defaults=ServoDefaults(voltage_v=5.2))
    vbus = VirtualBus(servos, WireConfig()).start()
    driver = Bus(vbus.port, SETTINGS)
    try:
        result = driver.self_test()
        assert not result.ok
        assert any("outside" in p for p in result.problems)
        with pytest.raises(BusError):
            driver.enable_torque()
    finally:
        driver.close()
        vbus.stop()


def test_torque_is_refused_when_a_servo_is_already_hot():
    servos = RegisterServos(defaults=ServoDefaults(temperature_c=70))
    vbus = VirtualBus(servos, WireConfig()).start()
    driver = Bus(vbus.port, SETTINGS)
    try:
        assert not driver.self_test().ok
    finally:
        driver.close()
        vbus.stop()


def test_enabling_torque_seeds_the_goal_from_where_the_arm_actually_is():
    servos = RegisterServos()
    for servo_id in servos.ids:
        servos.set(servo_id, "present_position", 2100)
    vbus = VirtualBus(servos, WireConfig()).start()
    driver = Bus(vbus.port, SETTINGS)
    try:
        driver.enable_torque()
        flush(driver)
        assert [servos.get(i, "goal_position") for i in servos.ids] == [2100] * 6
    finally:
        driver.close()
        vbus.stop()


def test_a_lost_reply_holds_the_previous_reading_instead_of_stalling():
    vbus = VirtualBus(RegisterServos(), WireConfig(packet_loss=0.25, seed=3)).start()
    driver = Bus(vbus.port, BusSettings(timeout_s=0.02, retries=1))
    try:
        driver.cycle(np.zeros(6))
        held = 0
        for _ in range(60):
            state = driver.cycle(np.zeros(6))
            held += int((~state.valid).sum())
            # Whatever the wire did, every entry is a finite number the controller can use.
            assert np.isfinite(state.position).all()
        assert held > 0, "25 % packet loss should have cost at least one reply"
        assert driver.counters.errors > 0
    finally:
        driver.close()
        vbus.stop()


def test_a_named_register_round_trips_through_read_and_write(bus):
    driver, _ = bus
    driver.write_register(1, "p_coefficient", 32)
    assert driver.read_register(1, "p_coefficient") == 32
    driver.write_register(2, "goal_speed", -600)
    assert driver.read_register(2, "goal_speed") == -600


def test_a_read_only_register_is_refused(bus):
    driver, _ = bus
    with pytest.raises(ValueError):
        driver.write_register(1, "present_position", 0)


def test_the_microcontroller_pair_moves_encoder_counts_in_both_directions(bus):
    driver, vbus = bus
    driver.write_ticks([2000, 2100, 2200, 1900, 1800, 2048])
    flush(driver)
    assert [vbus.servos.get(i, "goal_position") for i in driver.ids] == \
        [2000, 2100, 2200, 1900, 1800, 2048]
    for servo_id, value in zip(driver.ids, (10, 20, 30, 40, 50, 60)):
        vbus.servos.set(servo_id, "present_position", 1000 + value)
        vbus.servos.set(servo_id, "present_speed", -value)
        vbus.servos.set(servo_id, "present_load", value)
    pos, speed, load, valid = driver.read_ticks()
    assert pos == [1010, 1020, 1030, 1040, 1050, 1060]
    assert speed == [-10, -20, -30, -40, -50, -60]
    assert load == [10, 20, 30, 40, 50, 60]
    assert all(valid)


def test_reading_one_servo_at_a_time_gives_the_same_answer_as_a_sync_read():
    vbus = VirtualBus(RegisterServos(), WireConfig()).start()
    fast = Bus(vbus.port, BusSettings(timeout_s=0.25, sync_read=True))
    try:
        a = fast.cycle(np.zeros(6))
        fast.settings = BusSettings(timeout_s=0.25, sync_read=False)
        b = fast.cycle(np.zeros(6))
        assert a.ticks.tolist() == b.ticks.tolist()
    finally:
        fast.close()
        vbus.stop()


def test_the_measured_cycle_fits_inside_a_twenty_millisecond_control_period(bus):
    driver, _ = bus
    timing = driver.measure_cycle_time(np.zeros(6), ticks=100)
    assert timing["errors"] == 0
    assert timing["fits_in_period"] == 1.0
    assert timing["p95_ms"] < 20.0


def test_a_short_conformance_run_makes_no_protocol_errors():
    result = conformance_run(packets=8000, timeout_s=60.0)
    assert result["packets"] >= 8000
    assert result["driver"]["errors"] == 0
    assert result["bus"]["bad_frames"] == 0
    assert result["driver"]["replies"] == result["bus"]["replies"]
