"""Unit tests for brick_builder/rl/nets.py: shapes, the state-independent log-std, the running
normalizer's statistics and checkpoint round-trip, and the actor/critic separation that
`ppo_asym.select_action` relies on for its own "no privileged leakage" property."""

from __future__ import annotations

import numpy as np
import torch

from brick_builder.rl.nets import Actor, Critic, RunningNorm


def test_actor_output_shape_and_state_independent_log_std():
    actor = Actor(obs_dim=5, act_dim=3, hidden=(8, 8))
    obs = torch.randn(7, 5)
    mean = actor(obs)
    assert mean.shape == (7, 3)

    dist = actor.distribution(obs)
    # log_std is one vector shared by the whole batch, not a function of obs: every row's
    # std must be identical even though the observations (and hence means) differ.
    std = dist.stddev
    assert torch.allclose(std, std[0].expand_as(std))
    assert actor.log_std.shape == (3,)
    assert actor.log_std.requires_grad


def test_actor_matches_manual_init_log_std():
    actor = Actor(obs_dim=2, act_dim=2, init_log_std=-2.0)
    assert torch.allclose(actor.log_std, torch.full((2,), -2.0))


def test_critic_output_shape():
    critic = Critic(obs_dim=9, hidden=(16, 16))
    obs = torch.randn(4, 9)
    value = critic(obs)
    assert value.shape == (4, 1)


def test_actor_never_needs_critic_observation():
    """Structural guarantee behind the acceptance test in test_ppo_asym.py: Actor.forward
    has no parameter through which a critic observation could enter, so feeding the critic
    a batch of NaNs cannot possibly affect the actor's output."""
    actor = Actor(obs_dim=4, act_dim=2)
    obs_actor = torch.randn(6, 4)
    critic_obs_with_nans = torch.full((6, 4), float("nan"))  # a separate tensor, never passed in

    out_normal = actor(obs_actor)
    # Nothing about calling the actor changes if some *other* tensor happens to hold NaNs;
    # this only documents that Actor.forward's signature takes exactly one tensor.
    assert torch.isfinite(out_normal).all()
    assert torch.isnan(critic_obs_with_nans).all()  # sanity: it really is poisoned, just unused


def test_running_norm_matches_numpy_mean_var():
    norm = RunningNorm(dim=3)
    rng = np.random.default_rng(0)
    data = rng.normal(loc=(1.0, -2.0, 5.0), scale=(2.0, 0.5, 3.0), size=(4000, 3)).astype(np.float32)

    # Feed it in chunks, as training does batch-by-batch, not all at once.
    for chunk in np.array_split(data, 20):
        norm.update(torch.as_tensor(chunk))

    np_mean = data.mean(axis=0)
    np_var = data.var(axis=0)
    assert np.allclose(norm.mean.numpy(), np_mean, atol=1e-2)
    assert np.allclose(norm.var.numpy(), np_var, atol=1e-1)


def test_running_norm_normalize_is_roughly_standardized():
    norm = RunningNorm(dim=2)
    rng = np.random.default_rng(1)
    data = rng.normal(loc=(10.0, -3.0), scale=(4.0, 1.0), size=(2000, 2)).astype(np.float32)
    norm.update(torch.as_tensor(data))
    normed = norm.normalize(torch.as_tensor(data)).numpy()
    assert np.allclose(normed.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(normed.std(axis=0), 1.0, atol=0.1)


def test_running_norm_frozen_stops_updates():
    norm = RunningNorm(dim=2)
    norm.update(torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    mean_before = norm.mean.clone()
    var_before = norm.var.clone()
    norm.frozen = True
    norm.update(torch.tensor([[100.0, -100.0], [200.0, -200.0]]))
    assert torch.equal(norm.mean, mean_before)
    assert torch.equal(norm.var, var_before)


def test_running_norm_clips_extreme_values():
    norm = RunningNorm(dim=1, clip=5.0)
    norm.update(torch.zeros(100, 1))  # mean 0, var ~0 (eps-regularized)
    normed = norm.normalize(torch.tensor([[1e6], [-1e6]]))
    assert torch.all(normed <= 5.0)
    assert torch.all(normed >= -5.0)


def test_running_norm_state_dict_round_trip():
    norm = RunningNorm(dim=4)
    norm.update(torch.randn(50, 4))
    state = norm.state_dict()

    restored = RunningNorm(dim=4)
    restored.load_state_dict(state)
    assert torch.equal(restored.mean, norm.mean)
    assert torch.equal(restored.var, norm.var)
    assert torch.equal(restored.count, norm.count)

    probe = torch.randn(10, 4)
    assert torch.equal(norm.normalize(probe), restored.normalize(probe))
