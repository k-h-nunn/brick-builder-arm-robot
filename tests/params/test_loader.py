"""Tests for brick_builder.params: loading, validation and merging.

These are about the parameter file mechanism itself, not about physics -- brick_builder.bricks and
brick_builder.robot already have their own tests for the numbers. What matters here is that a member
editing configs/params/default.yaml (or their own copy of it) gets a clear error for a mistake
and a correct, minimal change for a good edit.
"""

from __future__ import annotations

import dataclasses
import json
import textwrap

import pytest

from brick_builder import params as P


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


# ---------------------------------------------------------------------------
# The shipped default file
# ---------------------------------------------------------------------------


def test_default_file_matches_the_schema_exactly():
    """No schema/file drift: every SCHEMA path is in the file, and nothing extra is."""
    pf = P.load_params()
    assert set(pf.values) == set(P.SCHEMA)


def test_default_file_loads_with_no_problems():
    pf = P.load_params()
    assert len(pf.values) == len(P.SCHEMA)
    assert pf.source == P.DEFAULT_PARAMS_PATH
    assert pf.base_source == P.DEFAULT_PARAMS_PATH


def test_every_default_value_is_inside_its_own_schema_range():
    pf = P.load_params()
    for path, spec in P.SCHEMA.items():
        pv = pf.get(path)
        if pv.value is None:
            assert spec.nullable, f"{path} is null but not nullable"
            continue
        if spec.kind in (float, int):
            lo = -float("inf") if spec.lo is None else spec.lo
            hi = float("inf") if spec.hi is None else spec.hi
            assert lo <= pv.value <= hi, path
        if spec.kind is str and spec.choices is not None:
            assert pv.value in spec.choices, path


def test_headline_unknowns_are_not_status_measured_by_default():
    """The whole point of the file: nobody has measured these yet, and it says so."""
    pf = P.load_params()
    for path in (
        "bricks.clutch_force_per_stud_N",
        "bricks.brick_on_brick_friction",
        "bricks.jaw_on_brick_friction",
        "arm.supply_voltage_V",
        "arm.torque_limit_Nm",
        "camera.diagonal_fov_deg",
    ):
        assert pf.get(path).status != "measured", path


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_through_yaml_preserves_every_value_and_status():
    pf = P.load_params()
    text = pf.to_yaml()
    import yaml

    reparsed_raw = yaml.safe_load(text)
    flat = P._flatten(reparsed_raw)
    values, problems = P._parse_leaves(flat, require_all=True)
    assert problems == []
    for path, pv in pf.values.items():
        again = values[path]
        assert again.value == pv.value, path
        assert again.status == pv.status, path
        assert again.measured_by == pv.measured_by, path
        assert again.notes == pv.notes, path


def test_as_nested_dict_round_trips_through_a_file(tmp_path):
    pf = P.load_params()
    path = tmp_path / "roundtrip.yaml"
    path.write_text(pf.to_yaml())
    reloaded = P.load_params(path)
    for key in pf.values:
        assert reloaded.value(key) == pf.value(key), key
        assert reloaded.get(key).status == pf.get(key).status, key


# ---------------------------------------------------------------------------
# Partial override: the central promise of the loader
# ---------------------------------------------------------------------------


def test_partial_override_changes_exactly_the_intended_fields(tmp_path):
    path = _write(tmp_path, "partial.yaml", """
        arm:
          supply_voltage_V: 6.9
    """)
    pf = P.load_params(path)
    defaults = P.load_params()

    assert pf.value("arm.supply_voltage_V") == 6.9
    assert pf.get("arm.supply_voltage_V").status == "measured"

    # Every other parameter -- including every other arm.* field -- is untouched, status and all.
    for key in P.SCHEMA:
        if key == "arm.supply_voltage_V":
            continue
        assert pf.values[key] == defaults.values[key], key


def test_partial_override_with_full_block_form_keeps_its_own_metadata(tmp_path):
    path = _write(tmp_path, "partial.yaml", """
        bricks:
          clutch_force_per_stud_N:
            value: 2.3
            status: measured
            measured_by: kurt
            measured_on: 2026-09-20
            notes: kitchen scale, five repeats, averaged push-on and pull-off readings
    """)
    pf = P.load_params(path)
    pv = pf.get("bricks.clutch_force_per_stud_N")
    assert pv.value == 2.3
    assert pv.status == "measured"
    assert pv.measured_by == "kurt"
    assert pv.measured_on == "2026-09-20"
    assert "kitchen scale" in pv.notes


def test_an_unquoted_yaml_date_in_measured_on_becomes_a_plain_string(tmp_path):
    """PyYAML parses an unquoted 2026-09-20 as datetime.date, not str -- which then fails to
    JSON-encode wherever a report writes a ParamValue out. Must come back as a plain string."""
    path = _write(tmp_path, "partial.yaml", """
        arm:
          supply_voltage_V:
            value: 6.9
            status: measured
            measured_on: 2026-09-20
    """)
    pv = P.load_params(path).get("arm.supply_voltage_V")
    assert pv.measured_on == "2026-09-20"
    assert isinstance(pv.measured_on, str)
    json.dumps(dataclasses.asdict(pv))  # must not raise


def test_loading_the_default_path_explicitly_is_the_same_as_loading_none():
    a = P.load_params(None)
    b = P.load_params(P.DEFAULT_PARAMS_PATH)
    assert a.values == b.values


def test_empty_override_file_is_identical_to_the_defaults(tmp_path):
    path = _write(tmp_path, "empty.yaml", "{}")
    pf = P.load_params(path)
    defaults = P.load_params()
    assert pf.values == defaults.values
    assert pf.source != defaults.source  # the source path is still recorded correctly
    assert pf.source == path


# ---------------------------------------------------------------------------
# Validation errors are clear
# ---------------------------------------------------------------------------


def test_unknown_key_is_rejected_with_a_suggestion(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltag_V: 6.9
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    msg = str(excinfo.value)
    assert "arm.supply_voltag_V" in msg
    assert "arm.supply_voltage_V" in msg  # the suggestion


def test_completely_unrelated_unknown_key_gets_no_bogus_suggestion(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          zzz_not_a_real_parameter_at_all: 1.0
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    assert "zzz_not_a_real_parameter_at_all" in str(excinfo.value)


def test_out_of_range_value_is_rejected_with_the_bound(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltage_V: 999.0
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    msg = str(excinfo.value)
    assert "999" in msg
    assert "15" in msg  # the upper bound


def test_wrong_type_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltage_V: "seven point four"
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    assert "arm.supply_voltage_V" in str(excinfo.value)


def test_bad_enum_choice_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        bricks:
          press_curve_shape: bouncy
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    assert "bouncy" in str(excinfo.value)


def test_multiple_problems_are_all_reported_together(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltag_V: 999.0
          backlash_total_deg: -5.0
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    problems = excinfo.value.problems
    assert len(problems) >= 2


def test_block_without_a_value_key_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltage_V:
            status: measured
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    msg = str(excinfo.value)
    assert "arm.supply_voltage_V" in msg
    assert "value" in msg


def test_block_with_an_unknown_field_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltage_V:
            value: 7.4
            certainty: high
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    assert "certainty" in str(excinfo.value)


def test_bad_status_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.yaml", """
        arm:
          supply_voltage_V:
            value: 7.4
            status: pretty_sure
    """)
    with pytest.raises(P.ParamsError) as excinfo:
        P.load_params(path)
    assert "pretty_sure" in str(excinfo.value)


def test_not_yaml_is_rejected_cleanly(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("arm: [this is not: a mapping")
    with pytest.raises(P.ParamsError):
        P.load_params(path)


def test_top_level_list_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(P.ParamsError):
        P.load_params(path)


def test_missing_file_is_rejected_cleanly(tmp_path):
    with pytest.raises(P.ParamsError):
        P.load_params(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Nullable measured.* fields
# ---------------------------------------------------------------------------


def test_measured_fields_default_to_null_and_status_assumed():
    pf = P.load_params()
    for leaf in ("push_on_2x2_N", "push_on_2x4_N", "pull_off_2x2_N", "pull_off_2x4_N"):
        pv = pf.get(f"bricks.measured.{leaf}")
        assert pv.value is None
        assert pv.status == "assumed"


def test_a_bare_null_override_does_not_crash_and_stays_unmeasured(tmp_path):
    path = _write(tmp_path, "partial.yaml", """
        bricks:
          measured:
            push_on_2x4_N: null
    """)
    pf = P.load_params(path)
    pv = pf.get("bricks.measured.push_on_2x4_N")
    assert pv.value is None
    assert pv.status == "assumed"


def test_filling_in_a_measured_field_works(tmp_path):
    path = _write(tmp_path, "partial.yaml", """
        bricks:
          measured:
            pull_off_2x4_N: 22.5
    """)
    pf = P.load_params(path)
    pv = pf.get("bricks.measured.pull_off_2x4_N")
    assert pv.value == 22.5
    assert pv.status == "measured"


# ---------------------------------------------------------------------------
# list_assumptions
# ---------------------------------------------------------------------------


def test_list_assumptions_excludes_measured_parameters(tmp_path):
    path = _write(tmp_path, "partial.yaml", """
        arm:
          supply_voltage_V: 6.9
    """)
    pf = P.load_params(path)
    assumptions = P.list_assumptions(pf)
    assert "arm.supply_voltage_V" not in {a.path for a in assumptions}
    assert len(assumptions) == len(P.SCHEMA) - 1


def test_list_assumptions_include_derived_false_only_gives_pure_guesses():
    pf = P.load_params()
    only_assumed = P.list_assumptions(pf, include_derived=False)
    everything = P.list_assumptions(pf)
    assert all(a.status == "assumed" for a in only_assumed)
    assert len(only_assumed) < len(everything)
    assert any(a.status == "derived" for a in everything)


def test_a_fully_measured_file_has_no_assumptions_left(tmp_path):
    """Measuring one field removes exactly it (covered above); this checks the empty case, by
    overriding every SCHEMA leaf as status 'measured' in one file."""
    import yaml

    nested: dict = {}
    for path, spec in P.SCHEMA.items():
        value = spec.default if spec.default is not None else 0.0
        parts = path.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"value": value, "status": "measured"}

    path = tmp_path / "all_measured.yaml"
    path.write_text(yaml.safe_dump(nested))
    pf = P.load_params(path)
    assert P.list_assumptions(pf) == []
