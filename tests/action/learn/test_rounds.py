"""The round API a campaign driver calls: version ids, resuming, the record, and the gate.

Everything here runs on the toy linear world and a temporary weights directory, so it never
renders, never touches the shipped checkpoints and finishes in a few seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from brick_builder.action.learn import rounds
from brick_builder.action.models.mlp import Mlp

from .conftest import toy_episodes


@pytest.fixture
def logged(tmp_path):
    """The toy world written out in the on-disk episode format `dataset.load` reads."""
    data = toy_episodes(n_episodes=6, per_episode=40)
    out = tmp_path / "set"
    out.mkdir()
    for e in np.unique(data.episode):
        sel = data.episode == e
        np.savez(out / f"ep_{int(e):05d}.npz",
                 **{k: v[sel] for k, v in data.arrays.items()})
    return out


def _config(logged, tmp_path, **kw) -> rounds.RoundConfig:
    args = dict(data=(str(logged),), seed=0, minutes=0.3, epochs=3, root=str(tmp_path / "rounds"))
    args.update(kw)
    return rounds.RoundConfig(**args)


def test_the_version_id_is_a_function_of_the_data_and_the_hyperparameters(logged, tmp_path):
    a = _config(logged, tmp_path)
    assert a.fingerprint() == _config(logged, tmp_path).fingerprint()
    assert a.fingerprint() != _config(logged, tmp_path, seed=1).fingerprint()
    assert a.fingerprint() != _config(logged, tmp_path, teacher=0.0).fingerprint()
    assert a.fingerprint() != _config(logged, tmp_path, data=(str(logged), "other")).fingerprint()
    # The budget is not part of it: a round stopped by its clock and continued is the same round.
    assert a.fingerprint() == _config(logged, tmp_path, minutes=99.0).fingerprint()


def test_a_round_writes_both_checkpoints_a_report_and_its_optimizer_state(logged, tmp_path):
    result = rounds.train_round(_config(logged, tmp_path), log=lambda *_: None)

    assert result.predictor.exists() and result.inverse.exists() and result.report.exists()
    assert (result.directory / "state" / "predictor.pt").exists()
    assert (result.directory / "state" / "inverse.pt").exists()
    assert result.episodes == 6 and result.ticks == 240
    assert Mlp.load(result.inverse).sizes[-1] == 3
    assert np.isfinite(result.metrics["inverse_held_out_mse"])
    assert json.loads(result.report.read_text())["version"] == result.version
    assert rounds.versions(tmp_path / "rounds") == [result.version]
    assert rounds.load_round(result.version, tmp_path / "rounds").ticks == result.ticks


def test_the_same_round_called_twice_resumes_rather_than_starting_again(logged, tmp_path):
    cfg = _config(logged, tmp_path, epochs=2)
    first = rounds.train_round(cfg, log=lambda *_: None)
    lines: list[str] = []
    again = rounds.train_round(cfg, log=lines.append)
    assert again.version == first.version and again.directory == first.directory
    assert any("resumed at epoch 2" in line for line in lines)


def test_a_round_can_carry_the_predictor_over_instead_of_refitting_it(logged, tmp_path):
    source = rounds.train_round(_config(logged, tmp_path), log=lambda *_: None)
    carried = rounds.train_round(
        _config(logged, tmp_path, train_predictor=False, predictor_from=str(source.predictor),
                version="carried"), log=lambda *_: None)
    a, b = Mlp.load(source.predictor), Mlp.load(carried.predictor)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a.layers, b.layers))


def test_deploying_writes_the_pair_and_a_record_of_where_it_came_from(logged, tmp_path):
    result = rounds.train_round(_config(logged, tmp_path), log=lambda *_: None)
    weights = tmp_path / "weights"
    record = rounds.deploy(result, weights=weights, scores={"placement": {}}, note="a test")

    assert (weights / "predictor.npz").exists() and (weights / "inverse.npz").exists()
    assert record["version"] == result.version
    assert rounds.deployed(weights)["note"] == "a test"
    assert rounds.deployed(tmp_path / "empty") is None


def _scores(placement: str, pick: str) -> dict:
    return {task: {"analytic": {"seated_on_target_cell": "10/10" if task == "placement"
                                else "7/10"},
                   "learned": {"seated_on_target_cell": value}}
            for task, value in (("placement", placement), ("pick", pick))}


def test_the_gate_refuses_a_round_that_loses_on_either_task(logged, tmp_path):
    result = rounds.train_round(_config(logged, tmp_path), log=lambda *_: None)
    weights = tmp_path / "weights"

    assert rounds.deploy_if_better(result, _scores("10/10", "6/10"), weights=weights,
                                   log=lambda *_: None) is None
    assert rounds.deploy_if_better(result, _scores("9/10", "8/10"), weights=weights,
                                   log=lambda *_: None) is None
    assert not (weights / "inverse.npz").exists()

    record = rounds.deploy_if_better(result, _scores("10/10", "8/10"), weights=weights,
                                     log=lambda *_: None)
    assert record is not None and (weights / "inverse.npz").exists()
    assert "pick: learned 8/10" in record["note"]


def test_seated_fraction_reads_the_published_form():
    assert rounds.seated_fraction({"seated_on_target_cell": "7/10"}) == pytest.approx(0.7)
    assert rounds.seated_fraction({}) == 0.0
