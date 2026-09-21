"""Tiny PPO and SAC checkpoints, shared by every export test.

Spec section 5 says: "Until a BrickBuilder checkpoint exists, test with a checkpoint produced by
a few PPO updates on the Pendulum wrapper used in tests/rl." This rebuilds that exact wrapper
(rather than importing `tests.rl.test_ppo_asym`, which is not a package the export tests
should depend on) and trains just enough updates for the actor's weights to move off their
random init, session-scoped so every test in this directory reuses one checkpoint instead of
re-training it. `sac_checkpoint` is the same idea for the "parity tests
cover both learners" -- a few SAC updates on the same Pendulum wrapper.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from brick_builder.rl.ppo_asym import PPOConfig, train
from brick_builder.rl.sac_asym import SACConfig
from brick_builder.rl.sac_asym import train as sac_train


class DictPendulum(gym.ObservationWrapper):
    """Same wrapper as tests/rl/test_ppo_asym.py: Pendulum-v1's Box observation duplicated
    into {"actor": obs, "critic": obs}, since Pendulum has no privileged state to separate."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Dict({"actor": obs_space, "critic": obs_space})

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return {"actor": obs, "critic": obs}


def make_pendulum(seed: int) -> gym.Env:
    del seed
    return DictPendulum(gym.make("Pendulum-v1"))


@pytest.fixture(scope="session")
def pendulum_checkpoint(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("export_ckpt")
    cfg = PPOConfig(
        n_envs=4, n_steps=64, epochs=2, minibatches=4, total_steps=4 * 64 * 5,
        seed=0, async_envs=False,
    )
    return train(make_pendulum, cfg, run_dir / "run")


@pytest.fixture(scope="session")
def sac_checkpoint(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("export_ckpt_sac")
    cfg = SACConfig(
        n_envs=4, total_steps=4 * 40, seed=0, async_envs=False,
        warmup_steps=0, batch_size=16, buffer_size=1000, utd_ratio=1.0,
    )
    return sac_train(make_pendulum, cfg, run_dir / "run")
