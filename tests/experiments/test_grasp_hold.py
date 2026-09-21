"""Tests for the `grasp-hold` experiment."""

from __future__ import annotations

import argparse
import json

import pytest

from brick_builder import params as P
from brick_builder.experiments import grasp_hold as gh
from brick_builder.experiments.common import ExperimentContext, make_out_dir


def test_max_holding_force_picks_the_largest_force_that_still_holds():
    rows = [
        {"direction": "d", "force_N": 2.0, "holds": True},
        {"direction": "d", "force_N": 5.0, "holds": True},
        {"direction": "d", "force_N": 8.0, "holds": False},
        {"direction": "other", "force_N": 20.0, "holds": True},  # different direction, ignored
    ]
    assert gh._max_holding_force(rows, "d") == 5.0


def test_max_holding_force_none_when_nothing_holds():
    rows = [{"direction": "d", "force_N": 2.0, "holds": False}]
    assert gh._max_holding_force(rows, "d") is None


def _ctx(tmp_path, params_file=None, forces=None, hold_seconds=0.6, settle_steps=250):
    pf = P.load_params(params_file)
    out_dir = make_out_dir(tmp_path, "grasp-hold")
    extra = argparse.Namespace(forces=forces or [5.0], hold_seconds=hold_seconds,
                               settle_steps=settle_steps)
    return ExperimentContext(name="grasp-hold", params=pf, out_dir=out_dir, extra=extra,
                              argv=["python", "-m", "brick_builder.experiments", "grasp-hold"])


def test_run_writes_report_data_and_figure(tmp_path):
    ctx = _ctx(tmp_path)
    gh.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "data.json").exists()
    assert (ctx.out_dir / "grasp_hold.png").exists()


def test_data_json_covers_both_bricks_and_both_directions(tmp_path):
    ctx = _ctx(tmp_path)
    gh.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert set(data["rows"]) == {"2x2", "2x4"}
    for rows in data["rows"].values():
        directions = {r["direction"] for r in rows}
        assert directions == {label for label, _ in gh.DIRECTIONS}


def test_slip_is_reported_in_millimetres_and_is_non_negative(tmp_path):
    ctx = _ctx(tmp_path)
    gh.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    for rows in data["rows"].values():
        for r in rows:
            assert r["slip_mm"] >= 0.0
            assert r["peak_slip_mm"] >= r["slip_mm"] - 1e-9


def test_no_scene_fields_are_skipped_under_the_current_dataclasses(tmp_path):
    """Right now SceneConfig/HeldBrick have every field this module asks for; if a later
    edit removes one, this test is exactly what would catch it (and the experiment itself
    degrades gracefully rather than crashing -- see the params_mod.filter_dataclass_kwargs
    tests)."""
    ctx = _ctx(tmp_path)
    gh.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["skipped_fields"] == []


@pytest.mark.slow
def test_full_default_force_sweep(tmp_path):
    ctx = _ctx(tmp_path, forces=[2.0, 5.0, 8.0, 12.0, 17.0], hold_seconds=2.0, settle_steps=500)
    gh.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    for rows in data["rows"].values():
        assert len(rows) == 2 * 5  # 2 directions x 5 forces
