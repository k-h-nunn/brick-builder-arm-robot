"""Tests for brick_builder.params' conversion into the simulator's own config objects.

Each converter is checked two ways: does it read the right parameter file field into the right
dataclass field (with the right unit conversion), and does a partial override change *only* the
simulator field it should -- the acceptance test the task description asks for directly
("a partial override changes exactly the intended simulator fields").
"""

from __future__ import annotations

import dataclasses
import textwrap

import pytest

from brick_builder import params as P
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import DUPLO
from brick_builder.robot.patch_so101 import ArmConfig
from brick_builder.robot.servo_model import ServoConfig


def _override(tmp_path, text: str):
    path = tmp_path / "override.yaml"
    path.write_text(textwrap.dedent(text))
    return P.load_params(path)


# ---------------------------------------------------------------------------
# filter_dataclass_kwargs: the safety net every converter is built on
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Toy:
    a: int = 1
    b: str = "x"


def test_filter_dataclass_kwargs_keeps_only_existing_fields():
    kept, skipped = P.filter_dataclass_kwargs(_Toy, {"a": 5, "b": "y", "c": 999})
    assert kept == {"a": 5, "b": "y"}
    assert skipped == ["c"]


def test_filter_dataclass_kwargs_with_no_unknowns_reports_nothing_skipped():
    kept, skipped = P.filter_dataclass_kwargs(_Toy, {"a": 5})
    assert kept == {"a": 5}
    assert skipped == []


def test_filter_dataclass_kwargs_actually_protects_a_construction_call():
    """The exact scenario the task warns about: a field that does not exist must not raise."""
    kept, _ = P.filter_dataclass_kwargs(_Toy, {"a": 2, "field_removed_by_another_agent": True})
    built = dataclasses.replace(_Toy(), **kept)
    assert built == _Toy(a=2)


# ---------------------------------------------------------------------------
# to_brick_spec
# ---------------------------------------------------------------------------


def test_to_brick_spec_matches_duplo_at_defaults():
    spec = P.to_brick_spec(P.load_params())
    assert spec == DUPLO


def test_to_brick_spec_converts_millimetres_to_metres(tmp_path):
    pf = _override(tmp_path, """
        bricks:
          pitch_mm: 20.0
    """)
    spec = P.to_brick_spec(pf)
    assert spec.pitch == pytest.approx(0.020)
    # Nothing else moved.
    assert spec.stud_d == DUPLO.stud_d
    assert spec.height == DUPLO.height


def test_to_brick_spec_density_has_no_unit_conversion(tmp_path):
    pf = _override(tmp_path, """
        bricks:
          density_kg_m3: 1100.0
    """)
    spec = P.to_brick_spec(pf)
    assert spec.density == 1100.0


# ---------------------------------------------------------------------------
# to_snap_params
# ---------------------------------------------------------------------------


def test_to_snap_params_at_defaults_matches_the_files_shipped_values():
    """The shipped default matches the simulator's own bare placeholder (status assumed): the
    first-principles estimate (4.79 N, derived) is kept in guide 02c as an upper scenario, not
    used here."""
    sp = P.to_snap_params(P.load_params())
    assert sp.f_clutch == pytest.approx(1.0)
    assert sp.beta == pytest.approx(0.15)
    assert sp.curve == "ramp"
    assert sp.exponent == pytest.approx(1.91)


def test_to_snap_params_partial_override_changes_only_f_clutch(tmp_path):
    pf = _override(tmp_path, """
        bricks:
          clutch_force_per_stud_N: 2.5
    """)
    sp = P.to_snap_params(pf)
    default_sp = P.to_snap_params(P.load_params())
    assert sp.f_clutch == pytest.approx(2.5)
    for f in dataclasses.fields(SnapParams):
        if f.name == "f_clutch":
            continue
        assert getattr(sp, f.name) == getattr(default_sp, f.name), f.name


def test_to_snap_params_mm_fields_convert_correctly(tmp_path):
    pf = _override(tmp_path, """
        bricks:
          lateral_capture_tolerance_mm: 0.6
          press_curve_detent_depth_mm: 2.0
          press_curve_detent_width_mm: 1.2
    """)
    sp = P.to_snap_params(pf)
    assert sp.capture_lateral == pytest.approx(0.0006)
    assert sp.detent_depth == pytest.approx(0.002)
    assert sp.detent_width == pytest.approx(0.0012)


def test_to_snap_params_curve_choice_reaches_the_dataclass(tmp_path):
    pf = _override(tmp_path, """
        bricks:
          press_curve_shape: detent
    """)
    assert P.to_snap_params(pf).curve == "detent"


# ---------------------------------------------------------------------------
# to_arm_config
# ---------------------------------------------------------------------------


def test_to_arm_config_at_defaults():
    ac = P.to_arm_config(P.load_params())
    assert ac.vin == pytest.approx(7.4)
    assert ac.p_gain == 16
    assert ac.torque_limit_nm == pytest.approx(1.9)
    # arm.backlash_total_deg ships 0.87 (a TOTAL); ArmConfig.backlash_deg is a half-range.
    assert ac.backlash_deg == pytest.approx(0.435)
    assert ac.pad_friction == pytest.approx(0.40)


def test_to_arm_config_backlash_total_deg_is_halved(tmp_path):
    """arm.backlash_total_deg is the TOTAL free play a teammate measures; ArmConfig.backlash_deg
    is a half-range, so the conversion must divide by two."""
    pf = _override(tmp_path, """
        arm:
          backlash_total_deg: 1.0
    """)
    ac = P.to_arm_config(pf)
    assert ac.backlash_deg == pytest.approx(0.5)


def test_to_arm_config_partial_override_changes_only_vin(tmp_path):
    pf = _override(tmp_path, """
        arm:
          supply_voltage_V: 6.9
    """)
    ac = P.to_arm_config(pf)
    default_ac = P.to_arm_config(P.load_params())
    assert ac.vin == pytest.approx(6.9)
    for f in dataclasses.fields(ArmConfig):
        if f.name == "vin":
            continue
        assert getattr(ac, f.name) == getattr(default_ac, f.name), f.name


def test_to_arm_config_pad_friction_comes_from_the_bricks_section(tmp_path):
    """jaw_on_brick_friction lives under `bricks.*` in the file but sets ArmConfig.pad_friction."""
    pf = _override(tmp_path, """
        bricks:
          jaw_on_brick_friction: 0.9
    """)
    assert P.to_arm_config(pf).pad_friction == pytest.approx(0.9)


def test_to_arm_config_joint_friction_has_no_field_to_land_in():
    """The recorded-but-unreachable parameter: ArmConfig has no matching field today."""
    valid = {f.name for f in dataclasses.fields(ArmConfig)}
    assert "joint_friction_nm" not in valid
    assert "joint_friction_Nm" not in valid
    # And building an ArmConfig from the params file must not raise over it.
    P.to_arm_config(P.load_params())


# ---------------------------------------------------------------------------
# to_servo_config
# ---------------------------------------------------------------------------


def test_to_servo_config_only_touches_vin(tmp_path):
    pf = _override(tmp_path, """
        arm:
          supply_voltage_V: 6.9
    """)
    sc = P.to_servo_config(pf)
    default_sc = ServoConfig()
    assert sc.vin == pytest.approx(6.9)
    for f in dataclasses.fields(ServoConfig):
        if f.name == "vin":
            continue
        assert getattr(sc, f.name) == getattr(default_sc, f.name), f.name


def test_to_servo_config_error_deadband_ticks_comes_from_dead_band_counts(tmp_path):
    pf = _override(tmp_path, """
        arm:
          servo_dead_band_counts: 5
    """)
    sc = P.to_servo_config(pf)
    default_sc = ServoConfig()
    assert sc.error_deadband_ticks == pytest.approx(5.0)
    for f in dataclasses.fields(ServoConfig):
        if f.name in ("vin", "error_deadband_ticks"):
            continue
        assert getattr(sc, f.name) == getattr(default_sc, f.name), f.name


# ---------------------------------------------------------------------------
# to_camera_settings / to_postprocess_config
# ---------------------------------------------------------------------------


def test_to_camera_settings_computes_fovy_from_diagonal():
    from brick_builder.robot.patch_so101 import fovy_from_diagonal_fov

    cam = P.to_camera_settings(P.load_params())
    expected = fovy_from_diagonal_fov(115.0, 640, 480)
    assert cam.fovy_deg == pytest.approx(expected)
    assert cam.width_px == 640
    assert cam.height_px == 480


def test_to_camera_settings_recomputes_fovy_for_a_different_resolution(tmp_path):
    pf = _override(tmp_path, """
        camera:
          resolution_width_px: 512
          resolution_height_px: 512
    """)
    cam = P.to_camera_settings(pf)
    from brick_builder.robot.patch_so101 import fovy_from_diagonal_fov

    assert cam.fovy_deg == pytest.approx(fovy_from_diagonal_fov(115.0, 512, 512))
    # A square render has a larger vertical fovy than the 4:3 default, for the same diagonal.
    default_cam = P.to_camera_settings(P.load_params())
    assert cam.fovy_deg > default_cam.fovy_deg


def test_to_postprocess_config_sets_only_k1_k2(tmp_path):
    from brick_builder.render.domain_rand import PostProcessConfig

    pf = _override(tmp_path, """
        camera:
          distortion_k1: -0.3
          distortion_k2: 0.05
    """)
    pp = P.to_postprocess_config(pf)
    default_pp = PostProcessConfig()
    assert pp.k1 == pytest.approx(-0.3)
    assert pp.k2 == pytest.approx(0.05)
    for f in dataclasses.fields(PostProcessConfig):
        if f.name in ("k1", "k2"):
            continue
        assert getattr(pp, f.name) == getattr(default_pp, f.name), f.name


# ---------------------------------------------------------------------------
# to_scene_config
# ---------------------------------------------------------------------------


def test_to_scene_config_workspace_fields(tmp_path):
    import math

    pf = _override(tmp_path, """
        workspace:
          baseplate_x_m: 0.20
          baseplate_yaw_deg: 5.0
    """)
    cfg, skipped = P.to_scene_config(pf)
    assert cfg.plate_xy[0] == pytest.approx(0.20)
    assert cfg.plate_yaw == pytest.approx(math.radians(5.0))
    assert skipped == []


def test_to_scene_config_never_raises_on_an_extra_override_field():
    """Simulates a caller (e.g. an experiment) passing a kwarg that does not exist on
    SceneConfig -- this must be reported as skipped, not raised."""
    cfg, skipped = P.to_scene_config(P.load_params(), definitely_not_a_real_field=123)
    assert "definitely_not_a_real_field" in skipped


def test_to_scene_config_accepts_legitimate_overrides_like_held_and_tray():
    from brick_builder.scene.compose import HeldBrick

    cfg, skipped = P.to_scene_config(P.load_params(), held=HeldBrick(nx=2, ny=2), tray=True)
    assert cfg.held.nx == 2 and cfg.held.ny == 2
    assert cfg.tray is True
    assert skipped == []


# ---------------------------------------------------------------------------
# expected_brick_mass_g
# ---------------------------------------------------------------------------


def test_expected_brick_mass_g_matches_the_shipped_defaults():
    spec = P.to_brick_spec(P.load_params())
    assert P.expected_brick_mass_g(spec, 2, 4) == pytest.approx(12.52, abs=0.05)
    assert P.expected_brick_mass_g(spec, 2, 2) == pytest.approx(6.60, abs=0.05)


def test_expected_brick_mass_g_scales_with_density():
    spec = P.to_brick_spec(P.load_params())
    heavier = dataclasses.replace(spec, density=spec.density * 2.0)
    assert P.expected_brick_mass_g(heavier, 2, 4) == pytest.approx(
        2.0 * P.expected_brick_mass_g(spec, 2, 4), rel=1e-6
    )
