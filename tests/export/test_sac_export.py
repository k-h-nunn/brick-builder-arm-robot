"""Tests for exporting a SAC actor, against its acceptance criterion:
"extend brick_builder/export so the header records the squashing mode and parity tests cover both
learners." Uses the `sac_checkpoint` fixture (tests/export/conftest.py): a few SAC updates on
the same Pendulum wrapper the PPO export tests use.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from brick_builder.export.actor_to_c import load_actor_arrays, write_header
from brick_builder.export.build import build_library
from brick_builder.export.c_policy import CPolicy
from brick_builder.export.parity import _torch_actions, check_parity, sample_observations


def test_ppo_checkpoint_header_records_clamp_squash_mode(pendulum_checkpoint, tmp_path):
    """A checkpoint written before this field existed (any PPOConfig checkpoint: its cfg dict
    has no "squash" key) must still export exactly as before -- backward compatibility for
    the PPO export tests."""
    header = write_header(pendulum_checkpoint, tmp_path)
    text = header.read_text()
    assert '#define POLICY_SQUASH_MODE "clamp"' in text
    assert "#define POLICY_SQUASH_TANH 0" in text

    arrays = load_actor_arrays(pendulum_checkpoint)
    assert arrays["squash"] == "clamp"


def test_sac_checkpoint_header_records_tanh_squash_mode(sac_checkpoint, tmp_path):
    header = write_header(sac_checkpoint, tmp_path)
    text = header.read_text()
    assert '#define POLICY_SQUASH_MODE "tanh"' in text
    assert "#define POLICY_SQUASH_TANH 1" in text

    arrays = load_actor_arrays(sac_checkpoint)
    assert arrays["squash"] == "tanh"


@pytest.fixture(scope="module")
def sac_built_policies(sac_checkpoint, tmp_path_factory):
    build_dir_f = tmp_path_factory.mktemp("sac_build_float")
    build_dir_t = tmp_path_factory.mktemp("sac_build_table")
    lib_f, _ = build_library(sac_checkpoint, build_dir=build_dir_f, tanh_table=False)
    lib_t, _ = build_library(sac_checkpoint, build_dir=build_dir_t, tanh_table=True)
    cp_f = CPolicy(lib_f, obs_dim=3, act_dim=1)
    cp_t = CPolicy(lib_t, obs_dim=3, act_dim=1)
    return {"float": cp_f, "table": cp_t}


def test_sac_torch_reference_action_is_tanh_of_mean_not_clamp(sac_checkpoint):
    """`_torch_actions` (brick_builder/export/parity.py) must take the tanh branch for a SAC
    checkpoint, not silently fall back to the PPO clamp convention -- otherwise this whole
    parity check would validate the wrong reference action."""
    obs = sample_observations(sac_checkpoint, 5, seed=0)
    tanh_ref = _torch_actions(sac_checkpoint, obs)

    from brick_builder.rl.ppo_asym import load_actor, select_action

    actor, norm_actor, obs_spec = load_actor(sac_checkpoint)
    act_low = torch.as_tensor(obs_spec["act_low"], dtype=torch.float32)
    act_high = torch.as_tensor(obs_spec["act_high"], dtype=torch.float32)
    with torch.no_grad():
        clamp_action, _ = select_action(actor, norm_actor, torch.as_tensor(obs), deterministic=True)
        clamp_ref = torch.clamp(clamp_action, act_low, act_high).numpy()

    # The two conventions must actually differ here (unsquashed means routinely land outside
    # (-1, 1) for an undertrained actor) -- otherwise this test would not be exercising
    # anything different from the PPO case.
    assert not np.allclose(tanh_ref, clamp_ref, atol=1e-4)
    assert np.all(tanh_ref >= act_low.numpy() - 1e-6) and np.all(tanh_ref <= act_high.numpy() + 1e-6)


@pytest.mark.parametrize("variant", ["float", "table"])
def test_sac_parity_within_1e_5_on_10000_observations(sac_checkpoint, sac_built_policies, variant):
    """Same acceptance bound as the PPO export test, against the SAC (tanh-squashed) actor."""
    report = check_parity(sac_checkpoint, sac_built_policies[variant], n=10_000, seed=0)
    assert report.n == 10_000
    assert report.max_abs_error <= 1e-5, (
        f"{variant} tanh: max abs error {report.max_abs_error} exceeds the 1e-5 acceptance bound"
    )


def test_sac_c_policy_respects_action_bounds(sac_checkpoint, sac_built_policies):
    obs = sample_observations(sac_checkpoint, 200, seed=3)
    for cp in sac_built_policies.values():
        acts = cp.act_batch(obs)
        assert np.all(np.isfinite(acts))
        assert np.all(acts >= -2.0 - 1e-6) and np.all(acts <= 2.0 + 1e-6)  # Pendulum-v1 action space
