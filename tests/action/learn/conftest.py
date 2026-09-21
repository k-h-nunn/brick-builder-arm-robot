"""A toy linear world in the logged-tuple format, for the fast learning tests.

The slate rows move by a fixed matrix times the action, the correction increment is proportional
to the difference from the target, and nothing renders. That is enough for the predictor to have
something to learn, for the distal-teacher term to have a gradient, and for every shape and
normalization check to run in under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.action.learn import dataset as ds
from brick_builder.action.models.features import ACTION_W, N_SLATE, ROW_W


def toy_episodes(n_episodes: int = 12, per_episode: int = 60, seed: int = 0) -> ds.Episodes:
    rng = np.random.default_rng(seed)
    gain = rng.normal(0.0, 0.4, size=(ACTION_W, N_SLATE * ROW_W))

    keys = ("obs_rows", "obs_valid", "obs_weight", "tgt_rows", "tgt_valid", "slot_ids", "mech",
            "cmd", "ff", "analytic", "delta", "measured", "err_mm", "k", "step_index",
            "dwelling", "trusted", "frozen", "in_hand", "seen", "t", "step")
    chunks: dict[str, list] = {key: [] for key in keys}
    episode = []
    for e in range(n_episodes):
        rows = rng.normal(0.0, 0.05, size=(N_SLATE, ROW_W))
        target = rows + rng.normal(0.0, 0.02, size=rows.shape)
        corr = np.zeros(3)
        ff = np.array([0.2, 0.0, 0.05, 0.0])
        for t in range(per_episode):
            diff = (rows - target).reshape(-1)
            measured = np.array([diff[1], diff[2], diff[7]]) * 0.05
            analytic = -0.15 * measured
            delta = np.clip(analytic, -0.0012, 0.0012)
            chunks["obs_rows"].append(rows.copy())
            chunks["obs_valid"].append(np.ones(N_SLATE))
            chunks["obs_weight"].append(np.full(N_SLATE, 0.9))
            chunks["tgt_rows"].append(target.copy())
            chunks["tgt_valid"].append(np.ones(N_SLATE))
            chunks["slot_ids"].append(np.arange(N_SLATE))
            chunks["mech"].append(rng.normal(0.0, 0.1, size=18))
            chunks["cmd"].append(ff + np.array([corr[0], corr[1], 0.0, corr[2]]))
            chunks["ff"].append(ff.copy())
            chunks["analytic"].append(analytic)
            chunks["delta"].append(delta)
            chunks["measured"].append(measured)
            chunks["err_mm"].append(1e3 * float(np.linalg.norm(measured[:2])))
            chunks["k"].append(t)
            chunks["step_index"].append(0)
            chunks["dwelling"].append(True)
            chunks["trusted"].append(True)
            chunks["frozen"].append(False)
            chunks["in_hand"].append(True)
            chunks["seen"].append(3)
            chunks["t"].append(0.1 * t)
            chunks["step"].append("Align")

            action = np.array([delta[0], delta[1], 0.0, delta[2]])
            rows = rows + (action @ gain).reshape(N_SLATE, ROW_W)
            corr = corr + delta
            ff = ff + np.array([0.0005, 0.0003, 0.0, 0.0])
        episode.append(np.full(per_episode, e, dtype=np.int64))

    flags = ("dwelling", "trusted", "frozen", "in_hand")
    arrays = {}
    for key, values in chunks.items():
        if key == "step":
            arrays[key] = np.asarray(values, dtype="<U24")
        elif key in flags:
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    return ds.Episodes(arrays=arrays, episode=np.concatenate(episode))


@pytest.fixture(scope="session")
def toy() -> ds.Episodes:
    return toy_episodes()
