"""Tests for brick_builder.eval.report."""

import json

import numpy as np
import pytest

from brick_builder.eval.report import (
    AcceptanceThresholds,
    check_acceptance,
    group_by_params,
    plot_success_heatmap,
    render_markdown_report,
    success_rate,
    success_table,
    write_json_report,
    write_markdown_report,
)
from brick_builder.eval.results import EpisodeResult, FailureClass


def _synthetic_results(seed: int = 0, n_per_cell: int = 40) -> list[EpisodeResult]:
    """Deterministic synthetic sweep: misalignment (mm) x yaw error (deg), success decays with both."""
    rng = np.random.default_rng(seed)
    misaligns = [0, 2, 4, 6, 8]
    yaws = [0, 5, 10, 15]
    results = []
    for m in misaligns:
        for y in yaws:
            # success probability drops as misalignment/yaw grow; deterministic given seed
            p = max(0.02, 0.97 - 0.06 * m - 0.03 * y)
            for _ in range(n_per_cell):
                success = bool(rng.random() < p)
                peak_load = float(rng.uniform(0.1, 0.95))
                structure_intact = bool(rng.random() > 0.02)
                seat_depth_error = None if success else float(rng.uniform(0.0, 0.02))
                failure = None
                if not success:
                    failure = FailureClass.WRONG_CELL if seat_depth_error and seat_depth_error > 0.003 else FailureClass.PARTIAL_SEAT
                results.append(
                    EpisodeResult(
                        success=success,
                        steps=int(rng.integers(20, 200)),
                        seat_depth_error=seat_depth_error,
                        peak_load=peak_load,
                        structure_intact=structure_intact,
                        failure=failure,
                        ret=float(rng.normal()),
                        params={"misalign_mm": m, "yaw_deg": y},
                    )
                )
    return results


def test_group_by_params_splits_on_key_values():
    results = _synthetic_results()
    groups = group_by_params(results, ("misalign_mm",))
    assert set(k[0] for k in groups) == {0, 2, 4, 6, 8}
    assert sum(len(v) for v in groups.values()) == len(results)


def test_success_rate_matches_manual_count():
    results = _synthetic_results()
    n = len(results)
    successes = sum(1 for r in results if r.success)
    iv = success_rate(results)
    assert iv.n == n
    assert iv.p_hat == pytest.approx(successes / n)


def test_success_table_sorted_and_covers_all_groups():
    results = _synthetic_results()
    table = success_table(results, ("misalign_mm", "yaw_deg"))
    assert len(table) == 5 * 4
    keys = [row.key for row in table]
    assert keys == sorted(keys, key=lambda k: tuple(str(x) for x in k))
    # success rate should trend downward as misalignment grows, at fixed yaw=0
    zero_yaw = [row for row in table if row.key[1] == 0]
    zero_yaw.sort(key=lambda row: row.key[0])
    rates = [row.interval.p_hat for row in zero_yaw]
    assert rates[0] > rates[-1]


def test_write_json_report_roundtrips(tmp_path):
    results = _synthetic_results(n_per_cell=10)
    out = tmp_path / "report.json"
    payload = write_json_report(results, out)

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert on_disk["n_episodes"] == len(results)
    assert len(on_disk["episodes"]) == len(results)
    assert 0.0 <= on_disk["overall"]["success_rate"] <= 1.0
    assert on_disk["overall"]["wilson_low"] <= on_disk["overall"]["success_rate"]
    # failure classes serialize as their string value, not repr
    failure_values = {e["failure"] for e in on_disk["episodes"] if e["failure"] is not None}
    assert failure_values <= {"wrong_cell", "partial_seat"}


def test_render_markdown_report_contains_key_sections():
    results = _synthetic_results(n_per_cell=10)
    text = render_markdown_report(results, sweep_keys=("misalign_mm", "yaw_deg"))
    assert "# Evaluation report" in text
    assert "Overall success rate" in text
    assert "95% Wilson CI" in text
    assert "## Failure taxonomy" in text
    assert "wrong_cell" in text or "partial_seat" in text
    assert "## Success by misalign_mm, yaw_deg" in text
    assert "misalign_mm" in text and "yaw_deg" in text


def test_write_markdown_report_writes_file(tmp_path):
    results = _synthetic_results(n_per_cell=5)
    out = tmp_path / "report.md"
    text = write_markdown_report(results, out, sweep_keys=("misalign_mm",))
    assert out.exists()
    assert out.read_text() == text


def test_plot_success_heatmap_writes_png(tmp_path):
    results = _synthetic_results(n_per_cell=10)
    out = tmp_path / "heatmap.png"
    returned = plot_success_heatmap(results, "misalign_mm", "yaw_deg", out)
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 500  # a real PNG, not an empty stub


def test_plot_success_heatmap_raises_on_missing_keys(tmp_path):
    results = [EpisodeResult(success=True, steps=1, params={})]
    with pytest.raises(ValueError):
        plot_success_heatmap(results, "missing_x", "missing_y", tmp_path / "h.png")


def test_check_acceptance_flags_a_bad_policy():
    # A deliberately bad "student": mostly fails, high loads, damages structure often.
    rng = np.random.default_rng(1)
    bad = [
        EpisodeResult(
            success=bool(rng.random() < 0.3),
            steps=int(rng.integers(50, 200)),
            seat_depth_error=float(rng.uniform(0.001, 0.01)),
            peak_load=float(rng.uniform(0.5, 1.0)),
            structure_intact=bool(rng.random() > 0.1),
            params={},
        )
        for _ in range(200)
    ]
    rows = check_acceptance(bad, AcceptanceThresholds())
    by_name = {r.name: r for r in rows}
    assert by_name["Wilson lower bound"].passed is False
    assert by_name["Peak load within budget"].passed is False


def test_check_acceptance_passes_a_good_policy_against_scripted():
    rng = np.random.default_rng(2)

    def make(p_success: float, n: int) -> list[EpisodeResult]:
        return [
            EpisodeResult(
                success=bool(rng.random() < p_success),
                steps=int(rng.integers(20, 80)),
                seat_depth_error=0.0001,
                peak_load=float(rng.uniform(0.1, 0.4)),
                structure_intact=True,
                params={},
            )
            for _ in range(n)
        ]

    student = make(0.97, 500)
    scripted = make(0.75, 500)

    rows = check_acceptance(student, AcceptanceThresholds(), scripted_results=scripted)
    by_name = {r.name: r for r in rows}
    assert by_name["Wilson lower bound"].passed is True
    assert by_name["Beats scripted baseline"].passed is True
    assert by_name["Seat depth error (median)"].passed is True
    assert by_name["Peak load within budget"].passed is True
    assert by_name["Structure intact"].passed is True


def test_check_acceptance_without_scripted_marks_that_row_failed():
    results = [EpisodeResult(success=True, steps=10, peak_load=0.1, params={})]
    rows = check_acceptance(results)
    by_name = {r.name: r for r in rows}
    assert by_name["Beats scripted baseline"].passed is False
    assert "no scripted_results" in by_name["Beats scripted baseline"].measured


# --- rl-eval-export-2: peak_load/structure_intact must fail loud when not reported -----------


def test_epsiode_result_defaults_peak_load_and_structure_intact_to_none():
    """Sanity check on the dataclass itself: both fields must default to `None` ("not
    reported"), exactly like `seat_depth_error` already does -- not to a "safe"-looking
    0.0/True that would let an environment silently stop reporting them."""
    r = EpisodeResult(success=True, steps=1)
    assert r.peak_load is None
    assert r.structure_intact is None
    assert r.seat_depth_error is None


def test_check_acceptance_fails_loud_when_peak_load_never_reported():
    """50 episodes with no `peak_load` at all (an env that stopped setting it): this must be a
    clear, failed 'not reported' row -- the exact vacuous-pass rl-eval-export-2 describes,
    where the old 0.0 default made this read '100% within budget'."""
    results = [EpisodeResult(success=True, steps=10, structure_intact=True, params={}) for _ in range(50)]
    rows = check_acceptance(results, AcceptanceThresholds())
    by_name = {r.name: r for r in rows}
    row = by_name["Peak load within budget"]
    assert row.passed is False
    assert "not reported" in row.measured
    assert "50/50" in row.measured


def test_check_acceptance_fails_loud_when_structure_intact_never_reported():
    """Same failure mode as above, for `structure_intact` -- the old True default made this
    read '100% intact' with zero actual evidence."""
    results = [EpisodeResult(success=True, steps=10, peak_load=0.1, params={}) for _ in range(30)]
    rows = check_acceptance(results, AcceptanceThresholds())
    by_name = {r.name: r for r in rows}
    row = by_name["Structure intact"]
    assert row.passed is False
    assert "not reported" in row.measured
    assert "30/30" in row.measured


def test_check_acceptance_fails_loud_when_only_some_episodes_are_missing_the_field():
    """The bar is "missing in any episode", not "missing in every episode": a policy/env that
    reports peak_load on 99 of 100 episodes and silently drops it on the 100th (e.g. an
    early-exit code path) must still fail loud, not quietly average over the 99 that did
    report -- that partial-drop is exactly the kind of regression this exists to catch."""
    results = [
        EpisodeResult(success=True, steps=10, peak_load=0.1, structure_intact=True, params={})
        for _ in range(99)
    ]
    results.append(EpisodeResult(success=True, steps=10, structure_intact=True, params={}))  # no peak_load
    rows = check_acceptance(results, AcceptanceThresholds())
    by_name = {r.name: r for r in rows}
    row = by_name["Peak load within budget"]
    assert row.passed is False
    assert "1/100" in row.measured
    # structure_intact was reported on every episode, so that row is unaffected.
    assert by_name["Structure intact"].passed is True


def test_check_acceptance_still_scores_normally_when_fully_reported():
    """Regression guard: a batch where every episode reports both fields must be scored
    exactly as before (this is `test_check_acceptance_passes_a_good_policy_against_scripted`'s
    own scenario, re-asserted narrowly on just these two rows' `measured` text so a future
    change cannot quietly reintroduce a 'not reported' row when nothing is actually missing)."""
    results = [
        EpisodeResult(success=True, steps=10, peak_load=0.2, structure_intact=True, params={})
        for _ in range(20)
    ]
    rows = check_acceptance(results, AcceptanceThresholds())
    by_name = {r.name: r for r in rows}
    assert "not reported" not in by_name["Peak load within budget"].measured
    assert "not reported" not in by_name["Structure intact"].measured
    assert by_name["Peak load within budget"].passed is True
    assert by_name["Structure intact"].passed is True
