"""Tests for the `seating-feasibility` experiment."""

from __future__ import annotations

import argparse
import json

import pytest

from brick_builder import params as P
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import DUPLO
from brick_builder.experiments import seating_feasibility as sf
from brick_builder.experiments.common import ExperimentContext, make_out_dir


# ---------------------------------------------------------------------------
# push_on_needed / verdict: pure functions
# ---------------------------------------------------------------------------


def test_push_on_needed_matches_the_simulators_own_engagement_mechanism():
    """Cross-checked against brick_builder.bricks.snap_api.engagement directly: at full seat every
    curve family gives g=1, so push-on-needed must equal n * f_clutch * (1 + beta) exactly for
    'ramp' and 'plateau'."""
    from brick_builder.bricks.snap_api import engagement

    params = SnapParams(f_clutch=3.0, beta=0.2, curve="ramp")
    g_full = engagement(DUPLO.stud_h, params, DUPLO.stud_h)
    assert g_full == pytest.approx(1.0)
    assert sf.push_on_needed(params, 8, False, DUPLO) == pytest.approx(8 * 3.0 * 1.2)


def test_push_on_needed_tilted_only_fights_row_studs():
    params = SnapParams(f_clutch=1.0, beta=0.15, curve="ramp")
    straight = sf.push_on_needed(params, 8, False, DUPLO)
    tilted = sf.push_on_needed(params, 8, True, DUPLO, row_studs=2)
    assert straight / tilted == pytest.approx(4.0)


def test_push_on_needed_detent_uses_the_peak_gain():
    from brick_builder.bricks.materials import detent_peak_gain

    ramp = SnapParams(f_clutch=2.0, beta=0.15, curve="ramp")
    detent = SnapParams(f_clutch=2.0, beta=0.15, curve="detent")
    gain = detent_peak_gain(DUPLO)
    assert sf.push_on_needed(detent, 8, False, DUPLO) == pytest.approx(
        gain * sf.push_on_needed(ramp, 8, False, DUPLO)
    )
    assert gain > 1.0


def test_push_on_needed_is_not_inflated_by_one_over_one_minus_beta():
    """The discrepancy this module's docstring calls out: push-on-needed must NOT equal
    n * f_clutch * (1+beta) / (1-beta) (materials.push_on_force's convention when fed a raw
    f_clutch), only n * f_clutch * (1+beta)."""
    params = SnapParams(f_clutch=4.79, beta=0.15, curve="ramp")
    needed = sf.push_on_needed(params, 8, False, DUPLO)
    wrong_convention = 8 * 4.79 * 1.15 / 0.85
    correct_convention = 8 * 4.79 * 1.15
    assert needed == pytest.approx(correct_convention)
    assert needed != pytest.approx(wrong_convention, rel=0.01)


@pytest.mark.parametrize(
    "needed, lo, hi, expected",
    [
        (5.0, 9.0, 14.0, "possible everywhere on the patch"),
        (11.0, 9.0, 14.0, "possible only in the best part of the patch"),
        (20.0, 9.0, 14.0, "not possible anywhere on the patch"),
        (9.0, 9.0, 14.0, "possible everywhere on the patch"),  # boundary: <=
        (14.0, 9.0, 14.0, "possible only in the best part of the patch"),  # boundary: <=
    ],
)
def test_verdict_boundaries(needed, lo, hi, expected):
    assert sf.verdict(needed, lo, hi) == expected


# ---------------------------------------------------------------------------
# run(): the real force map, kept small with a tiny grid
# ---------------------------------------------------------------------------


def _ctx(tmp_path, params_file=None, grid_nx=3, grid_ny=3):
    pf = P.load_params(params_file)
    out_dir = make_out_dir(tmp_path, "seating-feasibility")
    extra = argparse.Namespace(grid_nx=grid_nx, grid_ny=grid_ny)
    return ExperimentContext(name="seating-feasibility", params=pf, out_dir=out_dir, extra=extra,
                              argv=["python", "-m", "brick_builder.experiments", "seating-feasibility"])


def test_run_writes_report_data_and_figure(tmp_path):
    ctx = _ctx(tmp_path)
    sf.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "data.json").exists()
    assert (ctx.out_dir / "seating_feasibility.png").exists()


def test_data_json_has_all_four_configurations_times_two_scenarios(tmp_path):
    ctx = _ctx(tmp_path)
    sf.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    names = {r["config"] for r in data["rows"]}
    assert names == {"2x4 straight", "2x4 tilted (1 row)", "2x2 straight", "2x2 tilted (1 row)"}
    scenarios = {r["scenario"] for r in data["rows"]}
    assert scenarios == {"your parameters", "first-principles upper scenario"}
    assert len(data["rows"]) == len(names) * len(scenarios)
    for r in data["rows"]:
        assert r["verdict"] in (
            "possible everywhere on the patch",
            "possible only in the best part of the patch",
            "not possible anywhere on the patch",
        )


def test_first_principles_scenario_is_independent_of_the_params_file(tmp_path):
    """The second scenario is a fixed reference point (guide 02c section 2.3), not something a
    member's own parameter file can move -- that is the whole point of showing it side by side."""
    ctx = _ctx(tmp_path)
    sf.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    fp_rows = [r for r in data["rows"] if r["scenario"] == "first-principles upper scenario"]
    expected = data["first_principles_clutch_force_per_stud_N"]
    assert all(r["clutch_force_per_stud_N"] == pytest.approx(expected) for r in fp_rows)


def test_tilted_always_needs_less_force_than_straight_for_the_same_brick(tmp_path):
    ctx = _ctx(tmp_path)
    sf.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    by_key = {(r["config"], r["scenario"]): r for r in data["rows"]}
    for scenario in ("your parameters", "first-principles upper scenario"):
        assert (by_key[("2x4 tilted (1 row)", scenario)]["needed_N"]
                < by_key[("2x4 straight", scenario)]["needed_N"])
        assert (by_key[("2x2 tilted (1 row)", scenario)]["needed_N"]
                < by_key[("2x2 straight", scenario)]["needed_N"])


def test_reflects_the_datasheet_torque_band_from_the_guide(tmp_path):
    """docs/guide/03-robot-and-servo-model.md quotes 9.5-14.0 N at 7.4 V / 1.9 N.m over a
    similar patch; the default parameter file ships exactly those arm settings."""
    ctx = _ctx(tmp_path, grid_nx=7, grid_ny=5)
    sf.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    band = data["force_band"]
    assert band["lo_N"] == pytest.approx(9.5, abs=0.5)
    assert band["hi_N"] == pytest.approx(14.0, abs=0.5)


def test_a_lower_clutch_value_makes_seating_easier(tmp_path):
    import textwrap

    easy = tmp_path / "easy.yaml"
    easy.write_text(textwrap.dedent("""
        bricks:
          clutch_force_per_stud_N: 0.5
    """))
    ctx = _ctx(tmp_path / "out", easy)
    sf.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    straight = next(
        r for r in data["rows"]
        if r["config"] == "2x4 straight" and r["scenario"] == "your parameters"
    )
    assert straight["verdict"] == "possible everywhere on the patch"
    # The first-principles scenario is untouched by the override: still the same fixed risk.
    fp_straight = next(
        r for r in data["rows"]
        if r["config"] == "2x4 straight" and r["scenario"] == "first-principles upper scenario"
    )
    assert fp_straight["verdict"] == "not possible anywhere on the patch"
