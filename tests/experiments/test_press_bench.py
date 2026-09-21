"""Tests for the `press-bench` experiment."""

from __future__ import annotations

import argparse
import json
import textwrap

import pytest

from brick_builder import params as P
from brick_builder.experiments import press_bench
from brick_builder.experiments.common import ExperimentContext, make_out_dir


# ---------------------------------------------------------------------------
# fit_clutch_force_and_beta: pure function, no simulator, fast
# ---------------------------------------------------------------------------


def test_fit_recovers_a_known_f_clutch_and_beta_exactly():
    """Construct measurements from a known (f_clutch, beta) and check the fit inverts them."""
    f_clutch, beta = 3.0, 0.2
    measured = {
        "push_on_2x2_N": 4 * f_clutch * (1 + beta),
        "pull_off_2x2_N": 4 * f_clutch * (1 - beta),
        "push_on_2x4_N": 8 * f_clutch * (1 + beta),
        "pull_off_2x4_N": 8 * f_clutch * (1 - beta),
    }
    fit = press_bench.fit_clutch_force_and_beta(measured)
    assert fit["f_clutch_N"] == pytest.approx(f_clutch, rel=1e-6)
    assert fit["beta"] == pytest.approx(beta, rel=1e-6)
    assert fit["consistency_check"]["2x4_over_2x2_ratio"] == pytest.approx(2.0)
    # Both routes to the clutch force must agree: the fitted value and the direct
    # (push + pull) / (2 * n_studs) average, per brick.
    for nx, ny in ((2, 2), (2, 4)):
        n = nx * ny
        direct = (measured[f"push_on_{nx}x{ny}_N"] + measured[f"pull_off_{nx}x{ny}_N"]) / (2 * n)
        assert direct == pytest.approx(f_clutch, rel=1e-6)


def test_fit_with_only_pull_off_readings_skips_beta():
    fit = press_bench.fit_clutch_force_and_beta({
        "push_on_2x2_N": None, "pull_off_2x2_N": 12.0,
        "push_on_2x4_N": None, "pull_off_2x4_N": 24.0,
    })
    assert fit["beta"] is None
    # Without beta, f_clutch is just pull_off / n (no (1-beta) correction available).
    assert fit["f_clutch_N"] == pytest.approx((12.0 / 4 + 24.0 / 8) / 2)


def test_fit_with_nothing_measured_returns_all_none():
    fit = press_bench.fit_clutch_force_and_beta({
        "push_on_2x2_N": None, "pull_off_2x2_N": None,
        "push_on_2x4_N": None, "pull_off_2x4_N": None,
    })
    assert fit["f_clutch_N"] is None
    assert fit["beta"] is None
    assert fit["consistency_check"] is None


def test_fmt_handles_none():
    assert press_bench._fmt(None) == "-"
    assert press_bench._fmt(1.23456, 2) == "1.23"


# ---------------------------------------------------------------------------
# run(): the real simulator, kept small with a fast ram speed
# ---------------------------------------------------------------------------


def _ctx(tmp_path, params_file=None, speed=0.02, pull_travel=0.006):
    pf = P.load_params(params_file)
    out_dir = make_out_dir(tmp_path, "press-bench")
    extra = argparse.Namespace(speed=speed, pull_travel=pull_travel)
    return ExperimentContext(name="press-bench", params=pf, out_dir=out_dir, extra=extra,
                              argv=["python", "-m", "brick_builder.experiments", "press-bench"])


def test_run_writes_report_data_and_figure(tmp_path):
    ctx = _ctx(tmp_path)
    press_bench.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "data.json").exists()
    assert (ctx.out_dir / "press_bench.png").exists()


def test_both_bricks_seat_and_peaks_are_positive(tmp_path):
    ctx = _ctx(tmp_path)
    press_bench.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    results = {r["nx"] * 10 + r["ny"]: r for r in data["results"]}
    for key in (22, 24):
        r = results[key]
        assert r["seated"] is True
        assert r["peak_push_on_N"] > 0.0
        assert r["peak_pull_off_N"] > 0.0
    # A 2x4 has twice the studs of a 2x2, so it should need roughly twice the push, at the
    # same snap parameters.
    assert results[24]["peak_push_on_N"] / results[22]["peak_push_on_N"] == pytest.approx(2.0, rel=0.15)


def test_report_says_not_enough_measurements_when_file_has_none(tmp_path):
    ctx = _ctx(tmp_path)
    press_bench.run(ctx)
    text = (ctx.out_dir / "report.md").read_text()
    assert "all null" in text


def test_report_fits_and_prints_values_when_measurements_are_present(tmp_path, capsys):
    path = tmp_path / "measured.yaml"
    path.write_text(textwrap.dedent("""
        bricks:
          measured:
            push_on_2x2_N: 12.0
            pull_off_2x2_N: 9.0
            push_on_2x4_N: 24.0
            pull_off_2x4_N: 18.0
    """))
    ctx = _ctx(tmp_path / "out", path)
    press_bench.run(ctx)
    text = (ctx.out_dir / "report.md").read_text()
    assert "Put these in your parameter file" in text
    assert "clutch_force_per_stud_N" in text
    out = capsys.readouterr().out
    assert "Fit from your measured" in out


def test_data_json_has_both_curves(tmp_path):
    ctx = _ctx(tmp_path)
    press_bench.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    for r in data["results"]:
        assert len(r["press_depth_mm"]) == len(r["press_force_N"])
        assert len(r["pull_depth_mm"]) == len(r["pull_force_N"])
        assert len(r["press_depth_mm"]) > 10


@pytest.mark.slow
def test_run_at_the_bench_default_speed(tmp_path):
    """The bench's own default (slower, more realistic) speed; slow-marked since it is the
    heavier of the two paths this experiment can take."""
    ctx = _ctx(tmp_path, speed=0.002, pull_travel=0.006)
    press_bench.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
