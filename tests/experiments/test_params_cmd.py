"""Tests for the `params` experiment: print and check a parameter file."""

from __future__ import annotations

import json
import textwrap

from brick_builder import params as P
from brick_builder.experiments import params_cmd
from brick_builder.experiments.common import ExperimentContext, make_out_dir


def _ctx(tmp_path, params_file=None):
    pf = P.load_params(params_file)
    out_dir = make_out_dir(tmp_path, "params")
    return ExperimentContext(name="params", params=pf, out_dir=out_dir,
                              argv=["python", "-m", "brick_builder.experiments", "params"])


def test_run_writes_report_data_and_figure(tmp_path):
    ctx = _ctx(tmp_path)
    params_cmd.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "data.json").exists()
    assert (ctx.out_dir / "params_headline.png").exists()


def test_report_contains_every_section_and_the_status_table(tmp_path):
    ctx = _ctx(tmp_path)
    params_cmd.run(ctx)
    text = (ctx.out_dir / "report.md").read_text()
    for section in ("bricks", "arm", "camera", "workspace"):
        assert f"### {section}" in text
    assert "Still assumptions" in text
    assert "Hooks this loader could not reach" in text
    assert "clutch_force_per_stud_N" in text


def test_data_json_reports_mass_check_and_assumption_count(tmp_path):
    ctx = _ctx(tmp_path)
    params_cmd.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["n_parameters"] == len(P.SCHEMA)
    assert data["n_assumptions"] > 0
    assert len(data["mass_check"]) == 2
    for row in data["mass_check"]:
        assert row["agree_within_10pct"] is True  # the shipped defaults are self-consistent


def test_mass_check_flags_a_deliberately_wrong_mass(tmp_path):
    path = tmp_path / "wrong_mass.yaml"
    path.write_text(textwrap.dedent("""
        bricks:
          mass_2x4_g: 20.0
    """))
    ctx = _ctx(tmp_path / "out", path)
    params_cmd.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    row = next(r for r in data["mass_check"] if r["brick"] == "2x4")
    assert row["agree_within_10pct"] is False


def test_hooks_missing_names_the_camera_gap_but_not_joint_friction(tmp_path):
    ctx = _ctx(tmp_path)
    params_cmd.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    joined = " ".join(data["hooks_missing"])
    assert "joint_friction" not in joined  # ArmConfig.joint_friction_nm closes this gap
    assert "CameraConfig" in joined


def test_headline_figure_reflects_measured_status_when_overridden(tmp_path):
    path = tmp_path / "measured.yaml"
    path.write_text(textwrap.dedent("""
        bricks:
          clutch_force_per_stud_N:
            value: 3.0
            status: measured
    """))
    ctx = _ctx(tmp_path / "out", path)
    params_cmd.run(ctx)
    # headline_unmeasured should no longer include the now-measured field
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assumed_paths = {a["path"] for a in data["assumptions"]}
    assert "bricks.clutch_force_per_stud_N" not in assumed_paths
