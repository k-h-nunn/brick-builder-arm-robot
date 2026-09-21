"""Tests for the `force-map` experiment."""

from __future__ import annotations

import argparse
import json

import pytest

from brick_builder import params as P
from brick_builder.experiments import force_map as fm
from brick_builder.experiments.common import ExperimentContext, make_out_dir


def _ctx(tmp_path, params_file=None, **extra_kwargs):
    pf = P.load_params(params_file)
    out_dir = make_out_dir(tmp_path, "force-map")
    defaults = dict(x_min=0.14, x_max=0.20, y_min=-0.04, y_max=0.04, z=0.02, grid_nx=3, grid_ny=3)
    defaults.update(extra_kwargs)
    extra = argparse.Namespace(**defaults)
    return ExperimentContext(name="force-map", params=pf, out_dir=out_dir, extra=extra,
                              argv=["python", "-m", "brick_builder.experiments", "force-map"])


def test_run_writes_report_data_and_figure(tmp_path):
    ctx = _ctx(tmp_path)
    fm.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "data.json").exists()
    assert (ctx.out_dir / "force_map.png").exists()


def test_data_json_has_a_sane_force_range(tmp_path):
    ctx = _ctx(tmp_path)
    fm.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert 0.0 < data["force_lo_N"] <= data["force_median_N"] <= data["force_hi_N"]
    assert data["n_reachable"] <= data["n_total"]
    assert data["n_reachable"] > 0


def test_baseplate_position_is_reported_when_in_range(tmp_path):
    ctx = _ctx(tmp_path)  # 0.17, 0.0 default baseplate is inside 0.14-0.20 x -0.04-0.04
    fm.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["baseplate"]["x_m"] == pytest.approx(0.17)
    assert data["baseplate"]["reachable"] is True
    assert data["baseplate"]["force_N"] is not None


def test_baseplate_outside_grid_is_reported_as_such(tmp_path):
    ctx = _ctx(tmp_path, x_min=0.05, x_max=0.09)  # excludes the default 0.17 baseplate
    fm.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["baseplate"]["force_N"] is None
    assert data["baseplate"]["reachable"] is None


def test_higher_supply_voltage_increases_available_force(tmp_path):
    import textwrap

    hot = tmp_path / "hot.yaml"
    hot.write_text(textwrap.dedent("""
        arm:
          supply_voltage_V: 12.0
    """))
    ctx_default = _ctx(tmp_path / "a")
    ctx_hot = _ctx(tmp_path / "b", hot)
    fm.run(ctx_default)
    fm.run(ctx_hot)
    base = json.loads((ctx_default.out_dir / "data.json").read_text())
    hi = json.loads((ctx_hot.out_dir / "data.json").read_text())
    assert hi["force_lo_N"] > base["force_lo_N"]
    assert hi["force_hi_N"] > base["force_hi_N"]
