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
    # The fit itself works on a synthetic pocket -- `--validate`'s finding (guide 08) is not that
    # this function is broken, but that a gravity-loaded joint's ramp never produces a pocket for
    # it to find in the first place (the position error never enters the dead band at all).
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


def test_identify_refuses_a_joint_with_negligible_gravity_torque():
    """`shoulder_pan` (a vertical axis, zero gravity torque at every angle) has to raise, not
    return a number a near-zero-signal droop fit happened to produce."""
    scene, vbus = sysid.virtual_arm()
    vbus.start()
    from brick_builder.hardware.bus import Bus, BusSettings

    bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
    try:
        bus.enable_torque(check=False)
        with pytest.raises(ValueError, match="gravity torque"):
            sysid.identify(bus, "shoulder_pan", kin=scene.kin)
        with pytest.raises(ValueError, match="gravity torque"):
            sysid.identify(bus, "wrist_roll", kin=scene.kin)
    finally:
        bus.close()
        vbus.stop()


@pytest.mark.slow
def test_recovery_over_five_randomized_draws():
    """5 draws of the simulator's servo/joint parameters, perturbed inside the randomization
    ranges, identified through the virtual bus with the *nominal* kinematic model (no live link
    masses -- what identification on real hardware would use too, so `dynamics.link_mass_scale`'s
    confound with kp is not hidden by this validation).

    kp recovers to within about 10 %, including that confound.  Friction is known to be biased low
    (the settling transient crosses the friction dead zone -- guide 08).  On this gravity-loaded
    joint the dead-band *split* (`fit_dead_band_ticks`'s zero-`Present_Load` pocket) finds nothing
    on any draw, because the joint's own gravity droop keeps its ramp's position error outside the
    dead band at every point -- not because the pocket is too brief to sample. The *combined*
    dead-band-plus-play figure is, on this same gravity-loaded joint, a direct (unsplit) estimate
    of free play alone, recovering to within about 20 %, because the dead band never contributed to
    it in the first place -- this is the regression guard on that finding.
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

    assert mean_err("kp") < 10.0
    assert mean_err("friction") > 40.0   # the known low bias, asserted so a fix would be noticed
    # The known failure mode, asserted rather than just printed: every draw here has a non-zero
    # true dead band, and the split recovers 0 for all of them.
    assert all(r["fitted"]["dead_band_ticks"] == 0.0 for r in records)
    assert all(r["truth"]["dead_band_ticks"] > 1.0 for r in records)
    # `backlash_deg` here is the *raw combined* ramp figure (servo_params.py does not subtract the
    # registered dead band on a gravity-loaded joint): a real recovery of free play, not a sanity
    # bound on a number nobody trusts.
    assert mean_err("backlash_deg") < 25.0
