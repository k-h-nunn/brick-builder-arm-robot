"""A tiny, real `BrickBuilder/Reach-v0` PPO checkpoint, shared by tests/scripts/*.

Trained just enough (a couple of tiny PPO updates, `SyncVectorEnv`) for the actor's weights to
move off their random init and for every downstream tool (`scripts/eval.py`, the `policy` demo,
the C export path) to have a real checkpoint to load -- session-scoped so every test in this
directory reuses the one checkpoint instead of retraining it.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import gymnasium as gym
import pytest

import brick_builder.envs  # noqa: F401 -- registers BrickBuilder/* gymnasium ids
from brick_builder.rl.ppo_asym import PPOConfig, train


def _make_reach_env(seed: int) -> gym.Env:
    del seed
    return gym.make("BrickBuilder/Reach-v0")


@pytest.fixture(scope="session")
def tiny_reach_checkpoint(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("reach_ckpt") / "run"
    cfg = PPOConfig(n_envs=4, n_steps=32, epochs=2, minibatches=4, total_steps=4 * 32 * 2,
                     seed=0, async_envs=False)
    return train(_make_reach_env, cfg, run_dir)
