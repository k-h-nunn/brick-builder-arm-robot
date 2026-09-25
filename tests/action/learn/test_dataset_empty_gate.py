"""`hindsight_pairs`/`teacher_pairs` when nothing passes the `trusted & ~frozen` gate.

A short `--source cnn` round can log ticks that are all already `frozen` by the time they are
recorded (see `docs/guide/16b-campaign-on-the-scene-model.md`). Before this test, both functions
crashed on `np.stack([])` (`ValueError: need at least one array to stack`) partway through
`train_inverse`, after the predictor had already trained and been written to disk. They now
return zero-row arrays and log one line saying why, instead of raising.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.action.learn import dataset as ds
from brick_builder.action.models.features import INVERSE_IN, INVERSE_OUT, N_SLATE, ROW_W


def _episodes(*, frozen: bool, n_episodes: int = 3, per_episode: int = 10) -> ds.Episodes:
    """The same shape `dataset.load` produces, with every tick's `trusted`/`frozen` flags fixed
    rather than read from a real run -- only the two flags `hindsight_pairs`/`teacher_pairs` gate
    on need to vary for this test."""
    rng = np.random.default_rng(0)
    keys = ("obs_rows", "obs_valid", "obs_weight", "tgt_rows", "tgt_valid", "mech", "cmd", "ff",
           "analytic", "step_index", "trusted", "frozen", "in_hand")
    chunks: dict[str, list] = {key: [] for key in keys}
    episode = []
    for e in range(n_episodes):
        for t in range(per_episode):
            chunks["obs_rows"].append(rng.normal(0.0, 0.05, size=(N_SLATE, ROW_W)))
            chunks["obs_valid"].append(np.ones(N_SLATE))
            chunks["obs_weight"].append(np.full(N_SLATE, 0.9))
            chunks["tgt_rows"].append(rng.normal(0.0, 0.05, size=(N_SLATE, ROW_W)))
            chunks["tgt_valid"].append(np.ones(N_SLATE))
            chunks["mech"].append(rng.normal(0.0, 0.1, size=18))
            chunks["cmd"].append(rng.normal(0.0, 0.01, size=4))
            chunks["ff"].append(rng.normal(0.0, 0.01, size=4))
            chunks["analytic"].append(rng.normal(0.0, 0.01, size=3))
            chunks["step_index"].append(0)
            chunks["trusted"].append(True)
            chunks["frozen"].append(frozen)
            chunks["in_hand"].append(True)
        episode.append(np.full(per_episode, e, dtype=np.int64))
    arrays = {}
    for key, values in chunks.items():
        if key in ("trusted", "frozen", "in_hand"):
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return ds.Episodes(arrays=arrays, episode=np.concatenate(episode))


def test_all_frozen_returns_empty_pairs_without_raising(capsys) -> None:
    data = _episodes(frozen=True)
    hind = ds.hindsight_pairs(data, np.random.default_rng(1))
    assert hind["x"].shape == (0, INVERSE_IN)
    assert hind["y"].shape == (0, INVERSE_OUT)

    teach = ds.teacher_pairs(data)
    assert teach["x"].shape == (0, INVERSE_IN)
    assert teach["y"].shape == (0, INVERSE_OUT)

    err = capsys.readouterr().out
    assert "hindsight_pairs" in err and "0 of" in err
    assert "teacher_pairs" in err and "0 of" in err


def test_some_unfrozen_still_returns_real_pairs() -> None:
    """The gate itself is unchanged -- this is not a blanket bypass."""
    data = _episodes(frozen=False)
    hind = ds.hindsight_pairs(data, np.random.default_rng(1))
    assert hind["x"].shape[0] > 0
    assert hind["x"].shape[1] == INVERSE_IN

    teach = ds.teacher_pairs(data)
    assert teach["x"].shape[0] > 0


def test_log_callback_used_instead_of_default_print() -> None:
    lines = []
    data = _episodes(frozen=True)
    ds.hindsight_pairs(data, np.random.default_rng(1), log=lines.append)
    ds.teacher_pairs(data, log=lines.append)
    assert len(lines) == 2
    assert all("0 of" in line for line in lines)
