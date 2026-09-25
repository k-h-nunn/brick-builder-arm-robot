"""The parameter-file fragment builder: no bus, just a self-test report and a sysid fit in,
the `arm:` YAML block out, in `configs/params/default.yaml`'s own field format."""

from __future__ import annotations

import types

import pytest

from brick_builder.calib.servo_params import build_fields, render_param_fragment
from brick_builder.calib.servo_selftest import JointSelfTest, SelfTestConfig, SelfTestReport
from brick_builder.hardware.sts3215 import JOINT_NAMES


def _report(p_gains, dead_bands, voltages) -> SelfTestReport:
    joints = [
        JointSelfTest(servo_id=i + 1, joint=name, present=True, p_gain=p, dead_band_ticks=db,
                     voltage_v=v)
        for i, (name, p, db, v) in enumerate(zip(JOINT_NAMES, p_gains, dead_bands, voltages))
    ]
    return SelfTestReport(joints, SelfTestConfig())


def _fit(**kwargs):
    base = dict(joint="shoulder_lift", kp_nm_per_rad=9.3, friction_nm=0.015,
               peak_speed_rad_s=2.8, lost_motion_deg=1.4, deadband_and_play_deg=1.2,
               dead_band_ticks=0.0, backlash_deg=1.2, droop_points=6, droop_residual_deg=0.1)
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def test_fields_read_off_uniform_registers():
    report = _report([16] * 6, [(0, 0)] * 6, [7.4] * 6)
    fields = build_fields(report)
    assert fields["servo_p_gain"].value == 16
    assert fields["servo_dead_band_counts"].value == 0
    assert fields["supply_voltage_V"].value == pytest.approx(7.4)
    assert "all 6" in fields["servo_p_gain"].notes


def test_disagreeing_registers_are_flagged_not_averaged():
    p_gains = [16, 16, 16, 16, 16, 32]
    report = _report(p_gains, [(0, 0)] * 6, [7.4] * 6)
    fields = build_fields(report)
    assert fields["servo_p_gain"].value == 16   # the mode
    assert "disagree" in fields["servo_p_gain"].notes


def test_backlash_on_a_gravity_loaded_joint_is_the_raw_combined_figure_not_subtracted():
    """`shoulder_lift` is gravity-loaded (`GRAVITY_LOADED_JOINTS`): its ramp never sees the
    firmware dead band (guide 08), so the registered dead band is not subtracted back out of the
    combined ramp figure -- subtracting it would take out something the ramp never measured."""
    report = _report([16] * 6, [(4, 4)] * 6, [7.4] * 6)
    fit = _fit(joint="shoulder_lift", deadband_and_play_deg=1.2, dead_band_ticks=0.0,
              backlash_deg=1.2)
    fields = build_fields(report, fit)
    assert fields["backlash_total_deg"].value == pytest.approx(1.2, abs=1e-3)
    assert fields["servo_dead_band_counts"].value == 4
    assert "gravity-loaded" in fields["backlash_total_deg"].notes


def test_backlash_off_a_gravity_loaded_joint_does_subtract_the_registered_dead_band():
    report = _report([16] * 6, [(4, 4)] * 6, [7.4] * 6)
    fit = _fit(joint="wrist_roll", deadband_and_play_deg=1.2, dead_band_ticks=0.0, backlash_deg=1.2)
    fields = build_fields(report, fit)
    # 4 registered ticks each way -> 8 ticks total -> 8 * (360/4096) deg taken out of 1.2.
    expected = 1.2 - 8.0 * (360.0 / 4096.0)
    assert fields["backlash_total_deg"].value == pytest.approx(expected, abs=1e-3)
    assert "unvalidated" in fields["backlash_total_deg"].notes


def test_joint_friction_is_always_derived_never_measured():
    """The friction fit is biased low by a fixed mechanism (guide 08) whether the bus is virtual
    or real, so pasting a `--real` fragment must not claim `status: measured` for it."""
    report = _report([16] * 6, [(0, 0)] * 6, [7.4] * 6)
    fit = _fit()
    fields = build_fields(report, fit)
    assert fields["joint_friction_Nm"].status == "derived"
    text = render_param_fragment(fields, virtual=False)
    friction_block = text.split("joint_friction_Nm:")[1]
    assert "status: derived" in friction_block
    assert "status: measured" not in friction_block


def test_fragment_loads_through_brick_builder_params(tmp_path):
    """The printed fragment is not just YAML-shaped text: it has to load through the real params
    loader and reach `to_arm_config` with the values it printed."""
    from brick_builder.params import load_params, to_arm_config

    report = _report([16] * 6, [(0, 0)] * 6, [7.4] * 6)
    fit = _fit(joint="shoulder_lift", kp_nm_per_rad=9.32, friction_nm=0.015,
              deadband_and_play_deg=1.22)
    fields = build_fields(report, fit)
    text = render_param_fragment(fields, measured_on="2026-09-25")
    path = tmp_path / "fragment.yaml"
    path.write_text(text)

    params = load_params(path)
    assert params.value("arm.servo_p_gain") == 16
    assert params.value("arm.backlash_total_deg") == pytest.approx(1.22, abs=1e-3)
    arm_cfg = to_arm_config(params)
    assert arm_cfg.p_gain == 16


def test_render_param_fragment_is_pasteable_yaml_shaped_text():
    report = _report([16] * 6, [(0, 0)] * 6, [7.4] * 6)
    fields = build_fields(report)
    text = render_param_fragment(fields, measured_on="2026-09-25")
    assert text.startswith("arm:\n")
    assert "servo_p_gain:" in text
    assert "value: 16" in text
    assert "measured_on: 2026-09-25" in text
    assert "self-consistency check" in text  # virtual=True by default


def test_render_param_fragment_marks_real_hardware_measured_not_assumed():
    report = _report([16] * 6, [(0, 0)] * 6, [7.4] * 6)
    fields = build_fields(report)
    text = render_param_fragment(fields, virtual=False)
    assert "status: measured" in text
    assert "self-consistency check" not in text
