"""The fitting math in scripts/hardware/servo_sysid.py, on synthetic data, plus (slow) the
5-draw recovery validation against the virtual bus."""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.hardware.sts3215 import DEG_PER_TICK, RAD_PER_TICK
from brick_builder.robot.patch_so101 import ArmConfig, BAM_FRICTION_BASE, servo_kp
from scripts.hardware import servo_sysid as sysid


def test_fit_droop_recovers_a_known_linear_law():
    kp_true, offset_true = 9.4, 0.001
    torques = np.linspace(0.0, 0.3, 6)
    errors = torques / kp_true + offset_true
    kp, offset, residual = sysid.fit_droop(torques, errors)
    assert kp == pytest.approx(kp_true, rel=1e-6)
    assert offset == pytest.approx(offset_true, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_fit_droop_rejects_a_line_that_does_not_sag_with_load():
    torques = np.linspace(0.0, 0.3, 6)
    errors = -torques  # negative slope: not physical
    with pytest.raises(ValueError):
        sysid.fit_droop(torques, errors)


def test_fit_lost_motion_reads_off_a_synthetic_hysteresis_width():
    n = 240
    width = 0.05  # rad
    up = np.linspace(0.0, 1.0, n // 2)
    down = np.linspace(1.0, 0.0, n - n // 2)
    command = np.concatenate([up, down])
    # Reading lags the command by `width/2` on the way up, leads by `width/2` on the way down --
    # a clean rectangular loop of the width `fit_lost_motion` is supposed to read off.
    reading = np.concatenate([up - width / 2.0, down + width / 2.0])
    lost = sysid.fit_lost_motion(command, reading)
    assert lost == pytest.approx(width, abs=1e-3)


def test_fit_dead_band_ticks_reads_a_synthetic_zero_load_pocket():
    # Fine enough sampling that several samples land inside a small (encoder-tick-scale) pocket --
    # the ramp test itself is coarser than this (one sample per 50 Hz control tick), which is
    # exactly `--validate`'s finding: at the ramp's real sampling rate this does not work.
    n = 4000
    peak = n // 2
    command = np.concatenate([np.linspace(0.0, 1.0, peak), np.linspace(1.0, 0.0, n - peak)])
    load = np.full(n, 0.3)
    pocket_ticks_true = 20.0
    half_width_rad = pocket_ticks_true * RAD_PER_TICK
    # A dead-band freeze centres on wherever the output was when it engaged, not on the ramp's own
    # geometric peak -- pick a centre comfortably below the peak so both bounds of the pocket are
    # actually reached (the peak itself is a one-sided extreme of `command`, and centring the
    # pocket there would make only one of the two bounds reachable).
    center = command[peak] - 3.0 * half_width_rad
    load[np.abs(command - center) <= half_width_rad] = 0.0
    fitted = sysid.fit_dead_band_ticks(command, load)
    assert fitted == pytest.approx(pocket_ticks_true, rel=0.15)


def test_fit_dead_band_ticks_is_zero_with_no_pocket():
    n = 200
    command = np.concatenate([np.linspace(0.0, 1.0, n // 2), np.linspace(1.0, 0.0, n - n // 2)])
    load = np.full(n, 0.3)  # never zero: no dead band to find
    assert sysid.fit_dead_band_ticks(command, load) == 0.0


def test_known_values_match_the_default_arm_config():
    known = sysid.known_values(ArmConfig(), deadband_counts=3.0)
    assert known["kp"] == pytest.approx(servo_kp(16, 7.4))
    assert known["friction"] == pytest.approx(BAM_FRICTION_BASE)
    assert known["backlash_deg"] == pytest.approx(2.0 * ArmConfig().backlash_deg)
    assert known["dead_band_ticks"] == pytest.approx(3.0)
    assert known["deadband_and_play"] == pytest.approx(
        2.0 * ArmConfig().backlash_deg + 2.0 * 3.0 * DEG_PER_TICK)


@pytest.mark.slow
def test_recovery_over_five_randomized_draws():
    """Task 2's validation: 5 draws of the simulator's servo/joint parameters, perturbed inside
    the randomization ranges, identified through the virtual bus.

    kp recovers well (a few percent).  Friction is known to be biased low (the settling transient
    crosses the friction dead zone -- guide 08).  Neither the dead-band/free-play *split*
    (`fit_dead_band_ticks`'s zero-``Present_Load`` pocket) nor even the *combined*
    dead-band-plus-play figure the split comes from is a trustworthy measurement of the firmware
    dead band: the pocket the split looks for is shorter than one 20 ms control tick at these
    gains, so the split recovers 0 on every draw, and the ramp's own position hysteresis does not
    linearly add up the dead band's true contribution either (its mean error against the true
    combined figure is large and draw-dependent -- see guide 08).  This is the regression guard on
    that finding, not a claim that the combined figure is precise: only that kp is, and that the
    dead-band split reliably reports 0.
    """
    table, records = sysid.validate_recovery("shoulder_lift", 5, seed0=1000)
    assert "| draw | parameter |" in table
    assert len(records) == 5
    failed = [r for r in records if "error" in r]
    assert not failed, failed

    def mean_err(name: str) -> float:
        errs = [100.0 * abs(r["fitted"][name] - r["truth"][name]) / max(abs(r["truth"][name]), 1e-9)
                for r in records]
        return float(np.mean(errs))

    assert mean_err("kp") < 15.0
    # The known failure mode, asserted rather than just printed: every draw here has a non-zero
    # true dead band (seed0=1000 draws all land above a few ticks), and the split recovers 0 for
    # all of them.
    assert all(r["fitted"]["dead_band_ticks"] == 0.0 for r in records)
    assert all(r["truth"]["dead_band_ticks"] > 1.0 for r in records)
    # The combined figure is computed and finite -- a sanity bound, not a precision claim.
    assert 0.0 < mean_err("deadband_and_play_deg") < 200.0
