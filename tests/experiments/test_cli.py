"""Tests for `python -m brick_builder.experiments <name> ...`: the dispatcher itself."""

from __future__ import annotations

import json
import textwrap

import pytest

from brick_builder import experiments as exp_pkg


def test_every_registered_experiment_module_imports_and_has_run():
    import importlib

    for name in exp_pkg.EXPERIMENTS:
        module = importlib.import_module(f".{exp_pkg.EXPERIMENTS[name]}", package="brick_builder.experiments")
        assert callable(module.run), name


def test_params_experiment_runs_end_to_end_through_main(tmp_path, capsys):
    rc = exp_pkg.main(["params", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wrote" in out
    reports = list(tmp_path.glob("params/*/report.md"))
    assert len(reports) == 1


def test_unknown_experiment_name_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        exp_pkg.main(["not-a-real-experiment"])
    assert excinfo.value.code != 0


def test_invalid_params_file_exits_2_and_prints_the_problem(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(textwrap.dedent("""
        arm:
          supply_voltage_V: 999.0
    """))
    rc = exp_pkg.main(["params", "--params", str(bad), "--out", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "999" in err
    assert not (tmp_path / "out").exists()  # never even got to making an output directory


def test_valid_override_file_is_used(tmp_path):
    override = tmp_path / "mine.yaml"
    override.write_text(textwrap.dedent("""
        arm:
          supply_voltage_V: 6.9
    """))
    rc = exp_pkg.main(["params", "--params", str(override), "--out", str(tmp_path / "out")])
    assert rc == 0
    data_files = list((tmp_path / "out").glob("params/*/data.json"))
    assert len(data_files) == 1
    data = json.loads(data_files[0].read_text())
    assert data["parameters"]["arm"]["supply_voltage_V"]["value"] == 6.9
    assert data["parameters"]["arm"]["supply_voltage_V"]["status"] == "measured"


def test_experiment_specific_flags_reach_the_experiment(tmp_path):
    rc = exp_pkg.main([
        "press-bench", "--out", str(tmp_path), "--speed", "0.05", "--pull-travel", "0.004",
    ])
    assert rc == 0
    data_files = list(tmp_path.glob("press-bench/*/data.json"))
    data = json.loads(data_files[0].read_text())
    assert data["speed_m_s"] == 0.05
    assert data["pull_travel_m"] == 0.004


def test_bogus_experiment_specific_flag_exits_cleanly(capsys):
    rc = exp_pkg.main(["press-bench", "--not-a-real-flag", "3"])
    assert rc != 0


def test_default_out_dir_is_named_results():
    import inspect

    src = inspect.getsource(exp_pkg.main)
    assert '"results"' in src


def test_report_paths_are_named_by_experiment_and_timestamp(tmp_path):
    exp_pkg.main(["params", "--out", str(tmp_path)])
    entries = list((tmp_path / "params").iterdir())
    assert len(entries) == 1
    # YYYYMMDDTHHMMSSZ
    name = entries[0].name
    assert len(name) == 16 and name.endswith("Z") and name[8] == "T"
