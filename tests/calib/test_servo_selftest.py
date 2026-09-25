"""The self-test against the virtual bus: every joint answers, registers read back the virtual
servos' own defaults, the torque-off range read and the commanded move both pass."""

from __future__ import annotations

import pytest

from brick_builder.calib.servo_bench import bare_virtual_arm
from brick_builder.calib.servo_selftest import SelfTestConfig, run_self_test
from brick_builder.hardware.bus import Bus, BusSettings
from brick_builder.hardware.sts3215 import JOINT_NAMES


@pytest.fixture
def virtual_bus():
    scene, vbus = bare_virtual_arm(seed=0)
    vbus.start()
    bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
    try:
        yield bus, vbus, scene
    finally:
        bus.close()
        vbus.stop()
        vbus.raise_if_failed()


def test_every_joint_passes_against_a_clean_virtual_arm(virtual_bus):
    bus, vbus, scene = virtual_bus
    report = run_self_test(bus, vbus=vbus, scene=scene,
                           config=SelfTestConfig(range_points=3, move_settle_ticks=30))
    assert report.ok, report.table()
    assert [j.joint for j in report.joints] == list(JOINT_NAMES)
    for j in report.joints:
        assert j.present
        assert j.model_number is not None and j.firmware_version is not None
        assert j.dead_band_ticks == (0, 0)   # ServoDefaults' dead band, nominal
        assert j.range_read_ok is True
        assert j.commanded_move_ok is True


def test_a_missing_servo_is_reported_present_false_and_not_ok(virtual_bus):
    """`Bus.ping` (and so a self-test row) for an id the far end never answers -- a fast check on
    its own settings, not through the full per-joint self-test (whose shared bus.cycle/read_ticks
    calls sync-read every id together, so a permanently silent one would cost every other joint's
    check a retry timeout too; that is a property of the wire, not of this self-test, and is
    already covered where it belongs, `tests/hardware/test_bus.py`)."""
    bus, vbus, scene = virtual_bus
    missing_id = bus.ids[2]
    del vbus.servos.mem[missing_id]
    assert bus.ping(missing_id) is False


def test_the_table_renders_a_pass_fail_row_per_joint(virtual_bus):
    bus, vbus, scene = virtual_bus
    report = run_self_test(bus, vbus=vbus, scene=scene,
                           config=SelfTestConfig(range_points=3, move_settle_ticks=30))
    table = report.table()
    assert table.count("PASS") + table.count("FAIL") >= len(JOINT_NAMES)
    assert "overall: PASS" in table
