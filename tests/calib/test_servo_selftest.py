"""The self-test against the virtual bus: every joint answers, registers read back the virtual
servos' own defaults, the torque-off range read tracks the joint, and the commanded move is
measured from where the joint actually was, not an assumed baseline.

Against the nominal virtual arm (seed 0, no gravity feed-forward -- this drives `Bus.cycle`
directly, the same raw loop `servo_workspace_map.py` and real hardware get, not `RealArmWorld`)
four of six joints pass the commanded-move check; `shoulder_lift` and `elbow_flex` do not, because
the range read leaves each of them at its own calibrated extreme, and at that extreme the rest of
the arm's weight is close to or beyond what a proportional-only loop with no gravity feed-forward
can hold or recover a small step from. That is a true, repeatable property of this control law at
these two joints' own mechanical limits -- not an artifact of testing order, and not fixed by it --
and is exactly what this check is for: catching a joint whose loop does not behave as expected,
honestly, rather than reporting PASS through a wrong measurement. See `servo_selftest.py`'s module
docstring and guide 08."""

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


def test_every_joint_reports_registers_and_a_working_range_read(virtual_bus):
    bus, vbus, scene = virtual_bus
    report = run_self_test(bus, vbus=vbus, scene=scene,
                           config=SelfTestConfig(range_points=3, move_settle_ticks=30))
    assert [j.joint for j in report.joints] == list(JOINT_NAMES)
    for j in report.joints:
        assert j.present
        assert j.model_number is not None and j.model_number != (0, 0)
        assert j.firmware_version is not None and j.firmware_version != (0, 0)
        assert j.dead_band_ticks == (0, 0)   # ServoDefaults' dead band, nominal
        assert j.voltage_ok is True and j.temperature_ok is True
        assert j.range_read_ok is True
        assert j.angle_limit_ticks is not None


def test_the_commanded_move_is_measured_from_the_reading_not_the_commanded_home(virtual_bus):
    """The bug this replaces: `moved` computed against a commanded 0 baseline instead of the
    reading just before the step, which let three joints read triple-digit percentages and still
    say PASS. A joint that clearly can (shoulder_pan, a joint with negligible gravity torque about
    its own axis) must land close to the commanded amount, settled, no overload -- and every
    present joint's own detail string, PASS or FAIL, must report a fraction inside the pass
    window's own generous bounds (nothing near the old 1160 %/2562 %), because those bounds are
    exactly what `commanded_move_ok` is computed from."""
    bus, vbus, scene = virtual_bus
    cfg = SelfTestConfig(range_points=3, move_settle_ticks=30)
    report = run_self_test(bus, vbus=vbus, scene=scene, config=cfg)
    by_joint = {j.joint: j for j in report.joints}
    shoulder_pan = by_joint["shoulder_pan"]
    assert shoulder_pan.commanded_move_ok is True
    pan_pct = float(shoulder_pan.commanded_move_detail.split("(")[1].split(" %")[0])
    assert 80.0 <= pan_pct <= 120.0, shoulder_pan.commanded_move_detail   # close to commanded
    lo_pct, hi_pct = 100.0 * cfg.move_min_fraction, 100.0 * cfg.move_max_fraction
    # An honest FAIL on a gravity-loaded joint at its own extreme (shoulder_lift, elbow_flex) reads
    # a genuinely large negative fraction -- seed 0 measures -219 % on elbow_flex -- so the shared
    # bound below only needs to rule out the old bug's magnitude (1160 %/2562 %), not the pass
    # window itself; a FAIL's own fraction is checked against the pass window only when it claims
    # to pass.
    for j in report.joints:
        assert "% of commanded" in j.commanded_move_detail
        pct = float(j.commanded_move_detail.split("(")[1].split(" %")[0])
        assert abs(pct) < 300.0, j.commanded_move_detail   # nothing near the old 1160 %/2562 %
        if j.commanded_move_ok:
            assert lo_pct <= pct <= hi_pct, j.commanded_move_detail


def test_a_joint_that_cannot_hold_its_range_extreme_fails_honestly_not_silently(virtual_bus):
    """`shoulder_lift` and `elbow_flex` genuinely cannot hold or recover from their own range-read
    extreme under a proportional-only loop with no gravity feed-forward (see the module docstring
    and `servo_selftest.py`'s); the report has to say FAIL for each, with a detail that is not the
    default empty string, not paper over it with a PASS."""
    bus, vbus, scene = virtual_bus
    report = run_self_test(bus, vbus=vbus, scene=scene,
                           config=SelfTestConfig(range_points=3, move_settle_ticks=30))
    by_joint = {j.joint: j for j in report.joints}
    for name in ("shoulder_lift", "elbow_flex"):
        row = by_joint[name]
        assert row.commanded_move_ok is False, name
        assert row.commanded_move_detail, name
        assert row.ok is False, name
    assert "FAIL" in report.table()


def test_a_missing_servo_fails_its_own_self_test_row(virtual_bus):
    """A servo the far end never answers is `present=False` and the row (not just `bus.ping`) is
    not ok. `Bus.enable_torque`'s own gate scans every id on the bus, not just the joint currently
    under test, so it refuses torque from the very first joint's own turn onward -- the first
    joint still gets its own torque-off range read (torque is not needed for that), but never its
    commanded move, and `run_self_test` returns as soon as that first refusal happens: no later
    joint, present or missing, gets a range read or a move, and every present row (not only the
    missing one) carries a "torque not enabled" problem and reads FAIL."""
    bus, vbus, scene = virtual_bus
    missing_id = bus.ids[2]  # elbow_flex, the third joint tested
    del vbus.servos.mem[missing_id]
    report = run_self_test(bus, vbus=vbus, scene=scene,
                           config=SelfTestConfig(range_points=3, move_settle_ticks=30))
    by_id = {j.servo_id: j for j in report.joints}
    assert by_id[missing_id].present is False
    assert by_id[missing_id].ok is False
    assert report.ok is False
    first = by_id[bus.ids[0]]
    assert first.present is True
    assert first.range_read_ok is True    # the one range read that runs before the first refusal
    assert first.commanded_move_ok is None
    assert any("torque not enabled" in p for p in first.problems)
    for later_id in bus.ids[1:]:
        if later_id == missing_id:
            continue
        row = by_id[later_id]
        assert row.present is True
        assert row.range_read_ok is None   # never reached: torque was refused for the whole bus
        assert any("torque not enabled" in p for p in row.problems)
        assert row.ok is False


def test_the_table_renders_a_pass_fail_row_per_joint(virtual_bus):
    bus, vbus, scene = virtual_bus
    report = run_self_test(bus, vbus=vbus, scene=scene,
                           config=SelfTestConfig(range_points=3, move_settle_ticks=30))
    table = report.table()
    assert table.count("PASS") + table.count("FAIL") >= len(JOINT_NAMES)
    assert "Supply [V]" in table and "Temperature" in table
    # shoulder_lift and elbow_flex fail honestly at their own range extreme; see the module
    # docstring.
    assert "overall: FAIL" in table


def test_torque_is_off_for_the_first_joints_range_read_even_if_left_on(virtual_bus, monkeypatch):
    """A servo left energized from a previous session must not fight the range read: torque is
    written off at the very start of `run_self_test`, before the first joint's range read.

    Checking `bus.torque_enabled` only after `run_self_test` returns is vacuous against the virtual
    bus: every joint's own `finally` disables torque when its turn ends, so the flag reads `False`
    at the end regardless of whether the initial disable ran, and the virtual range read pokes
    `qpos` directly rather than stepping physics, so a stale torque state would not perturb its
    reading either. This instead captures `bus.torque_enabled` at the moment the first joint's
    range read actually begins, with torque forced on beforehand -- the one place a dropped initial
    `disable_torque()` call would show up."""
    import brick_builder.calib.servo_selftest as selftest_mod

    bus, vbus, scene = virtual_bus
    bus.enable_torque()
    assert bus.torque_enabled is True

    seen: list[bool] = []
    original = selftest_mod._range_read_virtual

    def spy(bus_, index, row, *, vbus, scene, cfg):
        if not seen:
            seen.append(bus_.torque_enabled)
        return original(bus_, index, row, vbus=vbus, scene=scene, cfg=cfg)

    monkeypatch.setattr(selftest_mod, "_range_read_virtual", spy)
    run_self_test(bus, vbus=vbus, scene=scene,
                 config=SelfTestConfig(range_points=3, move_settle_ticks=10))
    assert seen == [False]
