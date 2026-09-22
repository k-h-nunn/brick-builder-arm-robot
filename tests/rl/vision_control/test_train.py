"""The policy's shapes and gradients, and five PPO updates end to end.

The smoke test is deliberately tiny -- one environment, four steps an update -- because the point
is that the rollout, the advantage estimate, the alignment term and the checkpoint round trip all
fit together, not that anything is learned in twenty transitions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from brick_builder.rl.vision_control.policy import (  # noqa: E402
    POLICY_IN, PolicyConfig, VisionControlPolicy, match_target, policy_features,
)

K, W = 8, 14


def _obs(rng: np.random.Generator) -> dict:
    return {"image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "mech": rng.normal(size=12).astype(np.float32),
            "target": rng.normal(0.0, 0.1, size=(K, W)).astype(np.float32),
            "progress": rng.random(3).astype(np.float32)}


def test_the_matcher_pairs_a_row_with_its_own_copy():
    rng = np.random.default_rng(0)
    target = torch.as_tensor(rng.normal(0.0, 0.3, size=(K, W)), dtype=torch.float32)
    rows = target[torch.as_tensor([2, 0, 5, 1, 3, 4, 6, 7])].clone()
    matched, _ = match_target(rows, torch.ones(K), target, tau=0.02)
    assert torch.max(torch.abs(matched - rows)) < 1e-3


def test_an_empty_target_slot_is_never_matched_to():
    rng = np.random.default_rng(1)
    target = torch.as_tensor(rng.normal(0.0, 0.3, size=(K, W)), dtype=torch.float32)
    target[3:] = 0.0
    rows = target[:1].repeat(K, 1)
    matched, _ = match_target(rows, torch.ones(K), target, tau=0.05)
    assert torch.max(torch.abs(matched)) > 0.0


def test_the_feature_vector_has_the_documented_width():
    rng = np.random.default_rng(2)
    rows = torch.as_tensor(rng.normal(0.0, 0.2, size=(K, W)), dtype=torch.float32)
    target = torch.as_tensor(rng.normal(0.0, 0.2, size=(K, W)), dtype=torch.float32)
    x = policy_features(rows, torch.ones(K), target, torch.zeros(12), torch.zeros(3),
                        in_hand=1.0, tau=0.25)
    assert x.shape == (POLICY_IN,)
    assert torch.all(torch.isfinite(x))


@pytest.mark.slow
def test_an_untrained_policy_commands_exactly_zero():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    policy = VisionControlPolicy(PolicyConfig(), critic_dim=30)
    obs = _obs(np.random.default_rng(3))
    with torch.no_grad():
        mean = policy.distribution(policy.features(obs)[None]).mean
    assert torch.max(torch.abs(mean)) == 0.0


@pytest.mark.slow
def test_the_gradient_reaches_the_convolutional_encoder():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    policy = VisionControlPolicy(PolicyConfig(), critic_dim=30)
    # The output layer starts at zero, so its own gradient is what carries the signal back; the
    # check is that the encoder's first convolution sees something at all.
    features = policy.features(_obs(np.random.default_rng(4)))
    features.sum().backward()
    stem = policy.encoder.stem[0].weight
    assert stem.grad is not None and float(stem.grad.abs().sum()) > 0.0


@pytest.mark.slow
def test_the_encoder_stays_in_evaluation_mode():
    policy = VisionControlPolicy(PolicyConfig(), critic_dim=30)
    policy.train()
    assert not policy.encoder.training
    assert policy.head.training


@pytest.mark.slow
def test_five_ppo_updates_run_and_checkpoint(tmp_path):
    from brick_builder.rl.vision_control.env import VisionControlConfig
    from brick_builder.rl.vision_control.train import VisionTrainConfig, train

    cfg = VisionTrainConfig(
        n_envs=1, n_steps=4, minibatches=2, epochs=1, minutes=20.0, total_steps=20,
        async_envs=False, threads=2, seed=0,
        env=VisionControlConfig(max_ticks=8, seeds=(10,)),
    )
    lines: list[str] = []
    path = train(cfg, tmp_path / "run", log=lines.append)

    assert path.exists()
    rows = (tmp_path / "run" / "log.csv").read_text().strip().splitlines()
    assert len(rows) == 6  # a header and five updates
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["update"] == 5
    assert state["global_step"] == 20
    for key in ("pg_loss", "v_loss", "align_loss"):
        assert key in rows[0]


@pytest.mark.slow
def test_training_resumes_from_its_own_checkpoint(tmp_path):
    from brick_builder.rl.vision_control.env import VisionControlConfig
    from brick_builder.rl.vision_control.train import VisionTrainConfig, train

    def run(total: int, lines: list[str]):
        cfg = VisionTrainConfig(
            n_envs=1, n_steps=4, minibatches=2, epochs=1, minutes=20.0, total_steps=total,
            async_envs=False, threads=2, seed=0,
            env=VisionControlConfig(max_ticks=8, seeds=(10,)),
        )
        return train(cfg, tmp_path / "run", log=lines.append)

    run(8, [])
    lines: list[str] = []
    path = run(16, lines)
    assert any("resumed at update 2" in line for line in lines)
    assert torch.load(path, map_location="cpu", weights_only=False)["update"] == 4
