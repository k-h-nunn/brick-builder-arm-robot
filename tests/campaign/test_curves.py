"""The figures and the tables, drawn from a campaign state that was never run.

Matplotlib on the Agg backend and a fabricated `summary.csv` -- enough to check that every figure
is produced, that a missing series does not take a figure down with it, and that the tables carry
the fixed baselines the report is read against.
"""

from __future__ import annotations

import csv
import json

import pytest

from brick_builder.campaign import curves, tables

ROWS = [
    # round, task, inverse, predictor, seated, low, high
    (0, "placement", "analytic", "analytic", "10/10", 1.0, 0.72, 1.0),
    (0, "placement", "learned", "learned", "6/10", 0.6, 0.31, 0.83),
    (0, "pick_and_place", "analytic", "analytic", "7/10", 0.7, 0.40, 0.89),
    (0, "pick_and_place", "learned", "learned", "3/10", 0.3, 0.11, 0.60),
    (0, "build_2", "analytic", "analytic", "5/10", 0.5, 0.24, 0.76),
    (1, "placement", "learned", "learned", "8/10", 0.8, 0.49, 0.94),
    (1, "placement", "learned", "analytic", "8/10", 0.8, 0.49, 0.94),
    (1, "pick_and_place", "learned", "learned", "5/10", 0.5, 0.24, 0.76),
    (2, "placement", "learned", "learned", "9/10", 0.9, 0.60, 0.98),
    (2, "pick_and_place", "learned", "learned", "6/10", 0.6, 0.31, 0.83),
]


@pytest.fixture()
def campaign_dir(tmp_path):
    rows = []
    for i, (rnd, task, inverse, predictor, seated, rate, low, high) in enumerate(ROWS):
        rows.append({
            "round": rnd, "version": f"r{rnd}", "ticks_trained_on": 5000 * rnd,
            "episodes_trained_on": 20 * rnd, "task": task, "inverse": inverse,
            "predictor": predictor, "inverse_version": f"r{rnd}",
            "predictor_version": f"r{rnd}", "episodes": 10, "seated_on_target": seated,
            "success_rate": rate, "wilson_low": low, "wilson_high": high,
            "bricks_on_target": seated, "lateral_mm_median": 0.1 + 0.01 * i,
            "lateral_mm_min": 0.02, "lateral_mm_max": 0.3,
            "median_tracking_mm": 1.2 + 0.1 * i, "median_prediction_mm": 0.4 - 0.02 * rnd,
            "ticks_align": 12, "ticks_approach": 40, "replanned": 0, "damage": "intact",
            "failures": "" if rate == 1.0 else "seated on the wrong cell=2",
            "wall_s_median": 30.0,
        })
    with (tmp_path / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    state = {"config": {"name": "test"}, "rounds": {}}
    for rnd in (0, 1, 2):
        state["rounds"][str(rnd)] = {
            "index": rnd, "version": f"r{rnd}", "ticks": 5000 * rnd,
            "episodes_logged": 20 * rnd, "scores": {},
            "metrics": ({} if rnd == 0 else {
                "action_rmse_mm_x": 0.5 - 0.05 * rnd, "action_rmse_mm_y": 0.6 - 0.05 * rnd,
                "action_rmse_deg": 0.45, "analytic_rms_mm_x": 0.55,
                "action_held_out_ticks": 1234}),
            "wall_s": {"perform": 600.0, "train": 400.0, "evaluate": 900.0},
        }
    (tmp_path / "campaign.json").write_text(json.dumps(state, indent=2))
    return tmp_path


def test_every_figure_is_drawn(campaign_dir) -> None:
    paths = curves.draw_all(campaign_dir)
    assert len(paths) == 4
    for path in paths:
        assert path.exists() and path.stat().st_size > 8_000, path


def test_a_missing_series_does_not_take_a_figure_down(campaign_dir) -> None:
    """`build_2` has no learned column and no round has an action error in round 0."""
    rows = curves.read_summary(campaign_dir / "summary.csv")
    assert curves.success_figure(rows, campaign_dir / "s.png").exists()
    assert curves.action_figure({"rounds": {}}, campaign_dir / "a.png").exists()


def test_one_row_per_round_task_and_inverse_in_the_success_figure(campaign_dir) -> None:
    """The prediction rule is not a column of the success figure: the two are the same run."""
    rows = curves._primary(curves.read_summary(campaign_dir / "summary.csv"))
    keys = [(r["round"], r["task"], r["inverse"]) for r in rows]
    assert len(keys) == len(set(keys))
    paired = [r for r in rows if r["round"] == 1 and r["task"] == "placement"]
    assert len(paired) == 1 and paired[0]["predictor"] == "learned"


def test_the_tables_carry_the_fixed_baselines(campaign_dir) -> None:
    text = tables.render(campaign_dir)
    assert "| fixed | analytic |" in text
    assert "| published | ground truth | -- | 10/10 |" in text
    assert "| published | ground truth | -- | 8/10 |" in text
    assert "### Cost per round" in text
    assert "### Failures" in text
    assert "Build 2" in text
