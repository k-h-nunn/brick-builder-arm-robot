"""Unit tests for the host-side outer correction, and the measurement it exists for.

The integrator tests are arithmetic on six numbers and run in milliseconds.  The two that drive
the arm over a workspace grid are marked slow: they are the before-and-after the correction was
written for, and they are the only place the encoder change is measured as millimetres at the tool
rather than as counts at a joint.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from brick_builder.robot.joint_correction import (
    CONTROL_DT,
    JointCorrection,
    JointCorrectionConfig,
    measure_settling,
    random_servo,
    summarize,
    workspace_grid,
)
from brick_builder.robot.servo_model import ServoConfig

TICK = 2.0 * math.pi / 4096


# ------------------------------------------------------------------------------ the integrator ---


def test_a_standing_error_is_walked_out_at_the_configured_rate():
    """``ki`` is per second: one second of a fixed error moves the target by ``ki * error``."""
    corr = JointCorrection(3, JointCorrectionConfig(ki=2.0, deadzone_ticks=0.0))
    desired = np.array([0.1, 0.2, 0.3])
    measured = desired - 0.01          # 10 mrad short on every joint
    for _ in range(int(round(1.0 / CONTROL_DT))):
        out = corr.update(desired, measured)
    assert corr.correction == pytest.approx(np.full(3, 2.0 * 0.01), rel=1e-12)
    assert out == pytest.approx(desired + corr.correction, rel=1e-12)


def test_the_clamp_is_the_anti_windup():
    """The correction is the integrator state, so clamping it stops the windup at source."""
    cfg = JointCorrectionConfig(ki=50.0, clamp_ticks=5.0, deadzone_ticks=0.0)
    corr = JointCorrection(2, cfg)
    desired = np.zeros(2)
    for _ in range(500):
        corr.update(desired, desired - 1.0)     # a full radian of error, forever
    assert corr.correction == pytest.approx(np.full(2, 5.0 * TICK), rel=1e-12)

    # and it comes straight back out of the clamp when the error reverses, with no lag
    for _ in range(500):
        corr.update(desired, desired + 1.0)
    assert corr.correction == pytest.approx(np.full(2, -5.0 * TICK), rel=1e-12)


def test_freeze_holds_the_trim_and_does_not_lose_it():
    """A press or a grip is an error the arm is meant to have; it must not wind the trim up."""
    corr = JointCorrection(2, JointCorrectionConfig(ki=4.0, deadzone_ticks=0.0))
    desired = np.zeros(2)
    for _ in range(25):
        corr.update(desired, desired - 0.01)
    held = corr.correction.copy()
    assert np.all(held > 0.0)

    for _ in range(500):
        out = corr.update(desired, desired - 0.5, freeze=True)
    assert corr.correction == pytest.approx(held, rel=1e-12)
    assert out == pytest.approx(desired + held, rel=1e-12)


def test_reset_drops_the_trim():
    corr = JointCorrection(4, JointCorrectionConfig(ki=4.0, deadzone_ticks=0.0))
    for _ in range(50):
        corr.update(np.zeros(4), np.full(4, -0.02))
    assert np.any(corr.correction != 0.0)
    corr.reset()
    assert np.all(corr.correction == 0.0)
    assert corr.update(np.zeros(4), np.zeros(4)) == pytest.approx(np.zeros(4))


def test_nothing_is_integrated_inside_the_dead_zone():
    """Below one count the reading does not even carry the sign of the error."""
    corr = JointCorrection(1, JointCorrectionConfig(ki=10.0, deadzone_ticks=1.0))
    for _ in range(200):
        corr.update(np.zeros(1), np.array([-0.9 * TICK]))
    assert corr.correction[0] == 0.0

    for _ in range(200):
        corr.update(np.zeros(1), np.array([-3.0 * TICK]))
    assert corr.correction[0] > 0.0


def test_the_update_allocates_nothing_and_reuses_its_output():
    """Written for a C99 port: fixed arrays, one loop, and the same buffer every call."""
    corr = JointCorrection(5)
    first = corr.update(np.zeros(5), np.zeros(5))
    second = corr.update(np.zeros(5), np.zeros(5))
    assert first is second
    assert first.shape == (5,)


# --------------------------------------------------------------------- what it is worth at the tool ---


@pytest.mark.slow
def test_the_encoder_correction_is_what_removes_the_millimetres():
    """Before: several millimetres, and the encoders cannot see any of it.  After: well under one.

    This is the measurement the whole servo model turns on.  ``encoder_on_output=False`` is the
    motor-side model this project ran before the correction; over the tool-down workspace grid it
    stops 3-4 mm from where it was sent and forward kinematics of its own readings agrees with the
    command, so nothing on the robot knows.  With the encoder where the hardware puts it the error
    collapses and what is left is visible.
    """
    before = measure_settling(servo_cfg=ServoConfig(encoder_on_output=False))
    after = measure_settling(servo_cfg=ServoConfig(encoder_on_output=True))
    b, a = summarize(before), summarize(after)
    assert b["n"] == a["n"] >= 40

    assert 2.0 < b["error_mm"]["mean"] < 6.0, b["error_mm"]
    assert b["encoder_gap_mm"]["mean"] > 0.7 * b["error_mm"]["mean"], b
    assert a["error_mm"]["mean"] < 1.0, a["error_mm"]
    assert a["error_mm"]["mean"] < 0.25 * b["error_mm"]["mean"]
    assert a["encoder_gap_mm"]["p95"] < 0.5, a["encoder_gap_mm"]


@pytest.mark.slow
def test_the_outer_trim_earns_its_place_when_the_firmware_has_a_dead_band():
    """With nothing for it to correct the trim is a wash; with a dead band it halves the error."""
    cfg = ServoConfig(error_deadband_ticks=7.0)
    off = summarize(measure_settling(servo_cfg=cfg))
    on = summarize(measure_settling(servo_cfg=cfg, correction=JointCorrectionConfig()))
    assert on["error_mm"]["mean"] < 0.6 * off["error_mm"]["mean"], (off, on)
    # the price is a slower arrival and a visible buzz, both bounded
    assert on["settle_s"]["mean"] > off["settle_s"]["mean"]
    assert on["chatter_mm"]["worst"] < 2.0, on["chatter_mm"]


# ------------------------------------------------------------------------------- the harness ---


def test_the_grid_is_inside_the_tool_down_workspace():
    grid = workspace_grid()
    assert len(grid) == 60
    for pos, spin in grid:
        assert 0.12 < float(np.hypot(pos[0], pos[1])) < 0.23
        assert 0.02 < pos[2] < 0.09
        assert abs(spin) <= 0.6


def test_random_servo_stays_inside_the_published_ranges():
    from brick_builder.robot.patch_so101 import servo_kp
    from brick_builder.scene import randomize

    for seed in range(6):
        arm_cfg, servo_cfg = random_servo(seed)
        kp = servo_kp(arm_cfg.p_gain, arm_cfg.vin)
        lo, hi = randomize.RANGES["dynamics.kp"]
        assert lo <= kp <= hi
        lo, hi = randomize.RANGES["dynamics.backlash_deg"]
        assert lo <= arm_cfg.backlash_deg <= hi
        assert isinstance(servo_cfg, ServoConfig)
        assert servo_cfg.vin == pytest.approx(arm_cfg.vin)


def test_summarize_reports_every_statistic_the_report_quotes():
    samples = measure_settling(targets=workspace_grid(reaches=(0.17,), heights=(0.05,),
                                                      bearings=(0.0,)), seconds=1.0)
    out = summarize(samples)
    assert out["n"] == 1
    for field in ("error_mm", "encoder_gap_mm", "settle_s", "chatter_mm"):
        assert set(out[field]) == {"mean", "median", "p95", "worst", "min"}
    assert summarize([]) == {"n": 0}


def test_the_correction_config_is_a_plain_frozen_record():
    """A C99 port copies these four numbers; nothing here may be a Python object."""
    cfg = JointCorrectionConfig()
    for field in dataclasses.fields(cfg):
        assert isinstance(getattr(cfg, field.name), (int, float))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.ki = 2.0
