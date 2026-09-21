"""Tests for brick_builder.experiments.common: the plumbing every experiment module shares."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from brick_builder import params as P
from brick_builder.experiments import common


def test_make_out_dir_is_base_name_timestamp(tmp_path):
    when = datetime(2026, 9, 21, 3, 4, 5, tzinfo=timezone.utc)
    out = common.make_out_dir(tmp_path, "press-bench", when=when)
    assert out == tmp_path / "press-bench" / "20260921T030405Z"
    assert out.is_dir()


def test_make_out_dir_is_idempotent_and_fresh(tmp_path):
    when = datetime(2026, 9, 21, 3, 4, 5, tzinfo=timezone.utc)
    out1 = common.make_out_dir(tmp_path, "params", when=when)
    (out1 / "marker.txt").write_text("x")
    out2 = common.make_out_dir(tmp_path, "params", when=when)
    assert out1 == out2
    assert (out2 / "marker.txt").exists()  # mkdir(exist_ok=True): does not wipe the directory


def test_two_calls_a_second_apart_get_different_directories(tmp_path):
    t0 = datetime(2026, 9, 21, 3, 4, 5, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 21, 3, 4, 6, tzinfo=timezone.utc)
    out1 = common.make_out_dir(tmp_path, "params", when=t0)
    out2 = common.make_out_dir(tmp_path, "params", when=t1)
    assert out1 != out2


def test_numpy_json_encoder_handles_numpy_scalars_and_arrays():
    payload = {
        "a": np.float64(1.5),
        "b": np.int32(3),
        "c": np.array([1.0, 2.0, 3.0]),
        "d": np.bool_(True),
        "e": True,
        "f": None,
    }
    text = json.dumps(payload, cls=common.NumpyJSONEncoder)
    back = json.loads(text)
    assert back == {"a": 1.5, "b": 3, "c": [1.0, 2.0, 3.0], "d": True, "e": True, "f": None}


def test_write_data_json_round_trips(tmp_path):
    path = common.write_data_json(tmp_path, {"x": np.float32(2.0)})
    assert path == tmp_path / "data.json"
    assert json.loads(path.read_text()) == {"x": 2.0}


def test_parameter_table_md_lists_every_requested_path_and_its_status():
    pf = P.load_params()
    md = common.parameter_table_md(pf, ["bricks.clutch_force_per_stud_N", "arm.supply_voltage_V"])
    assert "bricks.clutch_force_per_stud_N" in md
    assert "1.0" in md
    assert "assumed" in md
    assert "arm.supply_voltage_V" in md


def test_parameter_table_md_shows_not_set_for_null_values():
    pf = P.load_params()
    md = common.parameter_table_md(pf, ["bricks.measured.push_on_2x4_N"])
    assert "not set" in md


def test_assumptions_md_can_be_filtered_by_prefix():
    pf = P.load_params()
    bricks_only = common.assumptions_md(pf, prefixes=("bricks.",))
    assert "arm.supply_voltage_V" not in bricks_only
    assert "bricks.clutch_force_per_stud_N" in bricks_only


def test_assumptions_md_all_measured_message(tmp_path):
    import yaml

    nested = {}
    for path, spec in P.SCHEMA.items():
        value = spec.default if spec.default is not None else 0.0
        parts = path.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"value": value, "status": "measured"}
    path = tmp_path / "all.yaml"
    path.write_text(yaml.safe_dump(nested))
    pf = P.load_params(path)
    assert "measured" in common.assumptions_md(pf).lower()


def test_report_header_includes_command_and_params_source():
    pf = P.load_params()
    ctx = common.ExperimentContext(name="params", params=pf, out_dir=None, argv=["python", "-m", "x", "params"])
    header = common.report_header(ctx, "Title", "what ran")
    assert "# Title" in header
    assert "python -m x params" in header
    assert str(pf.source) in header


def test_report_header_notes_a_merged_override_source():
    defaults = P.load_params()
    ctx = common.ExperimentContext(name="params", params=defaults, out_dir=None)
    header = common.report_header(ctx, "T", "w")
    assert "merged over" not in header  # source == base_source: no override in play


def test_write_report_joins_sections_with_blank_lines(tmp_path):
    path = common.write_report(tmp_path, ["# a", "b", None, "c"])
    text = path.read_text()
    assert text == "# a\n\nb\n\nc\n"


def test_status_marker_covers_all_three_statuses():
    assert common.status_marker("assumed") == "[assumed]"
    assert common.status_marker("derived") == "[derived]"
    assert common.status_marker("measured") == "[MEASURED]"
    with pytest.raises(KeyError):
        common.status_marker("not_a_status")


def test_new_figure_and_save_figure_write_a_real_png(tmp_path):
    fig, ax = common.new_figure((3.0, 2.0))
    ax.plot([0, 1], [0, 1])
    path = common.save_figure(fig, tmp_path, "x.png")
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.suffix == ".png"


def test_relevant_paths_filters_by_prefix():
    paths = common.relevant_paths(("workspace.",))
    assert paths
    assert all(p.startswith("workspace.") for p in paths)
    assert "arm.supply_voltage_V" not in paths
