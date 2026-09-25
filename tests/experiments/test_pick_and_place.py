"""Tests for the `pick-and-place` experiment.

The module it wraps, brick_builder/control/pick_place.py, is still under active development
elsewhere in the codebase. Every "not available" detection path is tested against a fake stand-in module
(see `_check_pick_place_module`'s docstring for why: fighting Python's import cache to simulate
a real import failure is not worth it when the check logic can be tested directly), so these
stay fast and correct regardless of that module's state. One slow-marked test exercises the real
module end to end and accepts either outcome, checking internal consistency of whichever branch
actually ran -- see its own docstring.
"""

from __future__ import annotations

import argparse
import json
import time
import types

import pytest

from brick_builder import params as P
from brick_builder.experiments import pick_and_place as pap
from brick_builder.experiments.common import ExperimentContext, make_out_dir


def _ctx(tmp_path, params_file=None, trials=2, place_error_mm=1.0, timeout_s=180):
    pf = P.load_params(params_file)
    out_dir = make_out_dir(tmp_path, "pick-and-place")
    extra = argparse.Namespace(trials=trials, place_error_mm=place_error_mm, timeout_s=timeout_s)
    return ExperimentContext(name="pick-and-place", params=pf, out_dir=out_dir, extra=extra,
                              argv=["python", "-m", "brick_builder.experiments", "pick-and-place"])


def test_failure_category_strips_the_per_trial_numeric_detail():
    assert pap._failure_category("partial seat (2.20 mm short)") == "partial seat"
    assert pap._failure_category("partial seat (0.63 mm short)") == "partial seat"
    # Categories with no parenthetical detail pass through unchanged.
    assert pap._failure_category("never engaged") == "never engaged"
    assert pap._failure_category("wrong cell") == "wrong cell"
    assert pap._failure_category("overload trip") == "overload trip"


def test_run_reports_not_available_when_result_has_an_unexpected_shape(tmp_path, monkeypatch):
    """A bench() that imports fine, has the right signature, and returns *something* without
    crashing -- but not the dict shape this module reads -- must still be caught, not crash the
    CLI with an uncaught KeyError. This is what the widened try/except in run() is for."""
    def fake_bench(**kwargs):
        return {"totally": "not what this module expects"}

    fake_pp = types.SimpleNamespace(bench=fake_bench, PickPlaceConfig=object())
    monkeypatch.setattr(pap, "_load_pick_place", lambda: fake_pp)
    monkeypatch.setattr(pap, "_build_scene", lambda params, pp: (None, []))
    ctx = _ctx(tmp_path)
    pap.run(ctx)  # must not raise
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["available"] is False
    assert "KeyError" in data["reason"]


# ---------------------------------------------------------------------------
# _check_pick_place_module: the pure signature/shape check
# ---------------------------------------------------------------------------


def test_check_accepts_a_module_with_everything_needed():
    def bench(*, trials, scene, pad_friction, strategy, seed0, place_error, dr_level, cfg=None):
        ...

    fake = types.SimpleNamespace(bench=bench, PickPlaceConfig=object())
    pap._check_pick_place_module(fake)  # must not raise


def test_check_rejects_a_module_missing_bench():
    fake = types.SimpleNamespace(PickPlaceConfig=object())
    with pytest.raises(pap._NotAvailable, match="bench"):
        pap._check_pick_place_module(fake)


def test_check_rejects_a_module_missing_pickplaceconfig():
    fake = types.SimpleNamespace(bench=lambda **kw: None)
    with pytest.raises(pap._NotAvailable, match="PickPlaceConfig"):
        pap._check_pick_place_module(fake)


def test_check_rejects_a_bench_missing_the_scene_parameter():
    """The specific incompatible-signature case this module's docstring calls out: an older
    `bench` with no way to hand it a pre-built scene."""
    def bench(*, trials, pad_friction, strategy, seed0, place_error, dr_level):
        ...

    fake = types.SimpleNamespace(bench=bench, PickPlaceConfig=object())
    with pytest.raises(pap._NotAvailable, match="scene"):
        pap._check_pick_place_module(fake)


def test_check_rejects_a_bench_that_is_not_even_callable_with_a_signature():
    fake = types.SimpleNamespace(bench=object(), PickPlaceConfig=object())
    with pytest.raises(pap._NotAvailable):
        pap._check_pick_place_module(fake)


# ---------------------------------------------------------------------------
# run(): behaviour when _load_pick_place fails, raises, or hangs
# ---------------------------------------------------------------------------


def test_run_reports_not_available_when_load_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "_load_pick_place",
                         lambda: (_ for _ in ()).throw(pap._NotAvailable("pretend: missing bench")))
    ctx = _ctx(tmp_path)
    pap.run(ctx)
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "pick_and_place.png").exists()
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["available"] is False
    assert "pretend: missing bench" in data["reason"]
    text = (ctx.out_dir / "report.md").read_text()
    assert "not available yet" in text.lower()


def test_run_reports_not_available_when_bench_raises_at_call_time(tmp_path, monkeypatch):
    """The module imports fine and has the right shape, but blows up when actually run -- e.g.
    a half-finished function body. Must be caught, not crash the whole experiments CLI."""
    def broken_bench(**kwargs):
        raise RuntimeError("half-implemented, sorry")

    fake_pp = types.SimpleNamespace(bench=broken_bench, PickPlaceConfig=object())
    monkeypatch.setattr(pap, "_load_pick_place", lambda: fake_pp)
    monkeypatch.setattr(pap, "_build_scene", lambda params, pp: (None, []))
    ctx = _ctx(tmp_path)
    pap.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["available"] is False
    assert "half-implemented" in data["reason"]


def test_run_reports_not_available_on_timeout(tmp_path, monkeypatch):
    def slow_bench(**kwargs):
        time.sleep(5.0)
        return {}

    fake_pp = types.SimpleNamespace(bench=slow_bench, PickPlaceConfig=object())
    monkeypatch.setattr(pap, "_load_pick_place", lambda: fake_pp)
    monkeypatch.setattr(pap, "_build_scene", lambda params, pp: (None, []))
    ctx = _ctx(tmp_path, timeout_s=1)
    pap.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["available"] is False
    assert "1 s" in data["reason"] or "did not finish" in data["reason"]


def test_run_reports_skipped_scene_fields_when_present(tmp_path, monkeypatch):
    def fake_bench(**kwargs):
        return {
            "trials": 3, "successes": 2,
            "rows": [
                {"success": True, "failure": "ok"},
                {"success": True, "failure": "ok"},
                {"success": False, "failure": "partial seat (0.4 mm short)"},
            ],
            "median_held_press_n": 7.0, "max_peak_press_n": 15.0, "max_torque_nm": 1.2,
        }

    fake_pp = types.SimpleNamespace(bench=fake_bench, PickPlaceConfig=object())
    monkeypatch.setattr(pap, "_load_pick_place", lambda: fake_pp)
    monkeypatch.setattr(pap, "_build_scene", lambda params, pp: (None, ["unknown_field"]))
    ctx = _ctx(tmp_path)
    pap.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert data["available"] is True
    assert data["skipped_fields"] == ["unknown_field"]
    assert data["successes"] == 2
    # Grouped by category, not by the per-trial exact shortfall -- see _failure_category.
    assert data["failure_counts"] == {"partial seat": 1}
    assert 0.0 <= data["wilson_low"] <= data["success_rate"] <= data["wilson_high"] <= 1.0


def test_report_and_figure_exist_when_every_trial_succeeds(tmp_path, monkeypatch):
    def fake_bench(**kwargs):
        return {
            "trials": 2, "successes": 2,
            "rows": [{"success": True, "failure": "ok"}, {"success": True, "failure": "ok"}],
            "median_held_press_n": 7.0, "max_peak_press_n": 15.0, "max_torque_nm": 1.2,
        }

    fake_pp = types.SimpleNamespace(bench=fake_bench, PickPlaceConfig=object())
    monkeypatch.setattr(pap, "_load_pick_place", lambda: fake_pp)
    monkeypatch.setattr(pap, "_build_scene", lambda params, pp: (None, []))
    ctx = _ctx(tmp_path)
    pap.run(ctx)
    assert (ctx.out_dir / "pick_and_place.png").exists()
    text = (ctx.out_dir / "report.md").read_text()
    assert "2/2" in text


# ---------------------------------------------------------------------------
# The real module, end to end -- accepts either outcome
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_against_the_real_pick_place_module(tmp_path):
    """brick_builder/control/pick_place.py is still under active development and may be
    mid-edit at any given moment, so this does not assert which branch runs -- only that
    whichever branch does run is internally consistent. When this project's own CI or a report
    run wants to know the actual current state, brick_builder.experiments.pick_and_place.run's
    printed output says so explicitly ("NOT AVAILABLE YET" or a success rate)."""
    ctx = _ctx(tmp_path, trials=1)
    pap.run(ctx)
    data = json.loads((ctx.out_dir / "data.json").read_text())
    assert (ctx.out_dir / "report.md").exists()
    assert (ctx.out_dir / "pick_and_place.png").exists()
    if data["available"]:
        assert data["trials"] == 1
        assert data["successes"] in (0, 1)
        assert 0.0 <= data["wilson_low"] <= data["wilson_high"] <= 1.0
    else:
        assert isinstance(data["reason"], str) and data["reason"]
