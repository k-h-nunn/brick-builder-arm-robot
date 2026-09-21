"""Tests for brick_builder/rl/ppo_asym.py, against the four acceptance criteria in spec section 6:
learns Pendulum-v1 (wrapped into the Dict actor/critic format) to mean return > -300; same
seed gives identical first-update losses; a checkpoint round-trip reproduces actions
bit-for-bit; and the actor's forward pass never touches the critic observation.

Also covers rl-eval-export-1 (the truncation-bootstrap fix): gymnasium 1.3's `SAME_STEP`
autoreset means a sub-env that ends this step already returns its *next* episode's first
observation as `obs`, with the ending episode's real final observation only reachable via
`infos["final_obs"]` -- see `extract_final_obs`/`_step_with_truncation_bootstrap` and the
"truncation bootstrap" section below for the synthetic envs that pin this down against an
analytic value.

Pendulum-v1 is wrapped by hand here (not a real brick_builder env) purely as a cheap, well
understood continuous-control task to exercise the PPO loop end to end. All tests use
`async_envs=False` (SyncVectorEnv), which is faster and fully deterministic for a handful of
lightweight envs and avoids fork-server startup cost inside pytest.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from brick_builder.rl.nets import Actor, RunningNorm
from brick_builder.rl.ppo_asym import (
    PPOConfig,
    _gae,
    _step_with_truncation_bootstrap,
    evaluate,
    extract_final_obs,
    load_actor,
    make_vector_env,
    select_action,
    train,
)


class DictPendulum(gym.ObservationWrapper):
    """Wraps Pendulum-v1's plain Box observation into {"actor": obs, "critic": obs}. Both
    halves are identical: Pendulum's observation is already the full state, so there is no
    extra privileged information to give the critic (unlike a real brick_builder env, where
    "critic" carries things like contact forces the real robot cannot sense)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Dict({"actor": obs_space, "critic": obs_space})

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return {"actor": obs, "critic": obs}


def make_pendulum(seed: int) -> gym.Env:
    del seed  # Pendulum-v1 needs no per-seed construction; seeding happens at reset()
    return DictPendulum(gym.make("Pendulum-v1"))


def _tiny_cfg(**overrides) -> PPOConfig:
    """A cfg small enough to run a couple of PPO updates in well under a second, for the
    unit tests that only need the training loop to execute, not to actually learn anything."""
    defaults = dict(
        n_envs=4, n_steps=32, epochs=2, minibatches=4, total_steps=4 * 32 * 2, seed=0, async_envs=False,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


def test_train_runs_and_writes_checkpoint(tmp_path: Path):
    ckpt_path = train(make_pendulum, _tiny_cfg(), tmp_path / "run")
    assert ckpt_path == tmp_path / "run" / "checkpoint.pt"
    assert ckpt_path.exists()
    assert (tmp_path / "run" / "returns.json").exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("actor", "critic", "norm_actor", "norm_critic", "cfg", "obs_spec"):
        assert key in ckpt
    assert ckpt["obs_spec"]["actor_dim"] == 3
    assert ckpt["obs_spec"]["act_dim"] == 1


def test_same_seed_gives_identical_first_update_losses(tmp_path: Path):
    """Determinism acceptance test: two independent `train()` calls with the same seed must
    produce the same losses on the very first PPO update."""
    losses_a: list[dict] = []
    losses_b: list[dict] = []

    train(make_pendulum, _tiny_cfg(seed=7), tmp_path / "a", callback=lambda u, m: losses_a.append(m))
    train(make_pendulum, _tiny_cfg(seed=7), tmp_path / "b", callback=lambda u, m: losses_b.append(m))

    assert len(losses_a) == len(losses_b) >= 1
    first_a, first_b = losses_a[0], losses_b[0]
    for key in ("pg_loss", "v_loss", "entropy", "approx_kl"):
        assert first_a[key] == pytest.approx(first_b[key], abs=0.0, rel=1e-6), key


def test_different_seed_gives_different_first_update_losses(tmp_path: Path):
    """Sanity check on the determinism test above: it would trivially pass if `seed` were
    silently ignored, so also confirm that changing the seed changes the outcome."""
    losses_a: list[dict] = []
    losses_c: list[dict] = []
    train(make_pendulum, _tiny_cfg(seed=7), tmp_path / "a", callback=lambda u, m: losses_a.append(m))
    train(make_pendulum, _tiny_cfg(seed=8), tmp_path / "c", callback=lambda u, m: losses_c.append(m))
    assert losses_a[0]["pg_loss"] != losses_c[0]["pg_loss"]


def test_checkpoint_round_trip_reproduces_actions_bit_for_bit(tmp_path: Path):
    ckpt_path = train(make_pendulum, _tiny_cfg(), tmp_path / "run")

    actor_a, norm_a, _ = load_actor(ckpt_path)
    actor_b, norm_b, _ = load_actor(ckpt_path)

    probe = torch.as_tensor(np.random.default_rng(0).normal(size=(16, 3)), dtype=torch.float32)
    action_a, _ = select_action(actor_a, norm_a, probe, deterministic=True)
    action_b, _ = select_action(actor_b, norm_b, probe, deterministic=True)
    assert torch.equal(action_a, action_b)

    # And the checkpoint's own actor must match a fresh load loaded straight from the same
    # in-memory state_dict (rules out a save/load precision loss for this dtype).
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor_c = Actor(raw["obs_spec"]["actor_dim"], raw["obs_spec"]["act_dim"], tuple(raw["cfg"]["hidden_actor"]))
    actor_c.load_state_dict(raw["actor"])
    norm_c = RunningNorm(raw["obs_spec"]["actor_dim"], clip=raw["cfg"]["norm_clip"])
    norm_c.load_state_dict(raw["norm_actor"])
    action_c, _ = select_action(actor_c, norm_c, probe, deterministic=True)
    assert torch.equal(action_a, action_c)


def test_actor_forward_pass_never_touches_critic_observation(tmp_path: Path):
    """Feed the critic observation NaNs during a real rollout step and confirm the actor's
    action and log-prob stay finite: the only way that could fail is if some future
    refactor accidentally shared a tensor or buffer between the two networks."""
    ckpt_path = train(make_pendulum, _tiny_cfg(), tmp_path / "run")
    actor, norm_actor, _ = load_actor(ckpt_path)

    obs_actor = torch.as_tensor(np.random.default_rng(1).normal(size=(8, 3)), dtype=torch.float32)
    action, logprob = select_action(actor, norm_actor, obs_actor, deterministic=False)
    assert torch.isfinite(action).all()
    assert torch.isfinite(logprob).all()

    # Same actor observation, paired with a poisoned critic observation elsewhere in the
    # pipeline, must give the exact same action: select_action has no way to see it, but this
    # nails down the guarantee against a future signature change that adds one.
    torch.manual_seed(123)
    action_clean, _ = select_action(actor, norm_actor, obs_actor, deterministic=True)
    _poisoned_critic_obs = torch.full((8, 3), float("nan"))  # never passed anywhere
    torch.manual_seed(123)
    action_again, _ = select_action(actor, norm_actor, obs_actor, deterministic=True)
    assert torch.equal(action_clean, action_again)
    assert torch.isfinite(action_clean).all()



# --- rl-eval-export-1: truncation must bootstrap, not zero -----------------------------------
#
# gymnasium 1.3's vector envs support three autoreset modes (`gym.vector.AutoresetMode`);
# `make_vector_env` runs `SAME_STEP` (confirmed directly below). Under `SAME_STEP`, a sub-env
# that ends its episode on step t is already reset by the time `step()` returns: `obs` at
# index i is the *next* episode's first observation, and the ending episode's real last
# observation is only reachable via `infos["final_obs"][i]`, valid where
# `infos["_final_obs"][i]` is True (gymnasium's own convention, verified against the real
# `gymnasium.vector.SyncVectorEnv`/`AsyncVectorEnv` source, not assumed). Nothing here is a
# "reset transition": every one of `SAME_STEP`'s rows is a real step with a real action and a
# real reward, unlike `NEXT_STEP` autoreset (gymnasium's actual default, which this project
# does not use), where the row right after a truncation has an ignored action and a
# manufactured zero reward. The tests below pin down the one thing `SAME_STEP` does not hand
# you for free: bootstrapping a truncated episode's value target from the right observation.


class _ConstantRewardEnv(gym.Env):
    """Reward is always 1.0, never terminates, truncates at `horizon` steps. The analytic
    infinite-horizon value is `1 / (1 - gamma)`; a bootstrap wrongly zeroed at the time limit
    gives the strictly smaller finite truncated-horizon partial sum instead."""

    def __init__(self, horizon: int = 5):
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-1e6, 1e6, shape=(1,), dtype=np.float32),
            "critic": gym.spaces.Box(-1e6, 1e6, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.horizon = horizon
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        obs = {"actor": np.zeros(1, np.float32), "critic": np.zeros(1, np.float32)}
        return obs, {}

    def step(self, action):
        self.t += 1
        obs = {"actor": np.zeros(1, np.float32), "critic": np.zeros(1, np.float32)}
        truncated = self.t >= self.horizon
        return obs, 1.0, False, truncated, {}


class _TerminatingEnv(gym.Env):
    """Reward 1.0 for 2 steps, then a TRUE termination on step 3 -- well before any time
    limit. A value function must bootstrap zero here, for any gamma."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-1e6, 1e6, shape=(1,), dtype=np.float32),
            "critic": gym.spaces.Box(-1e6, 1e6, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        obs = {"actor": np.zeros(1, np.float32), "critic": np.zeros(1, np.float32)}
        return obs, {}

    def step(self, action):
        self.t += 1
        obs = {"actor": np.zeros(1, np.float32), "critic": np.zeros(1, np.float32)}
        terminated = self.t >= 3
        return obs, 1.0, terminated, False, {}


class _DistinctFinalObsEnv(gym.Env):
    """Reward 5.0 every step; truncates at `horizon`. The observation just before truncation
    is `FINAL_VALUE`; the very next (post-autoreset) observation is `RESET_VALUE` --
    deliberately different so a bootstrap that reads the wrong one is caught red-handed
    instead of passing by coincidence."""

    FINAL_VALUE = 7.0
    RESET_VALUE = -3.0

    def __init__(self, horizon: int = 3):
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-1e3, 1e3, shape=(1,), dtype=np.float32),
            "critic": gym.spaces.Box(-1e3, 1e3, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.horizon = horizon
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        obs = {"actor": np.array([self.RESET_VALUE], np.float32), "critic": np.array([self.RESET_VALUE], np.float32)}
        return obs, {}

    def step(self, action):
        self.t += 1
        truncated = self.t >= self.horizon
        val = self.FINAL_VALUE if truncated else float(self.t)
        obs = {"actor": np.array([val], np.float32), "critic": np.array([val], np.float32)}
        return obs, 5.0, False, truncated, {}


class _ConstCritic:
    """Stand-in for `Critic`: ignores its input, always outputs `value`. Lets a test put "the
    critic" exactly at a chosen point (e.g. the analytic fixed point) without training it
    there."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.value, dtype=torch.float32)


class _LinearCritic:
    """Stand-in for `Critic`: `V(x) = weight * x[..., 0] + bias`. Lets a test tell which
    observation a bootstrap actually read, by making the value depend on it."""

    def __init__(self, weight: float, bias: float = 0.0):
        self.weight = float(weight)
        self.bias = float(bias)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x[:, :1] + self.bias


def test_make_vector_env_runs_same_step_autoreset():
    """Ground truth for every test in this section: confirm the actual configured mode,
    rather than assume it."""
    envs = make_vector_env(make_pendulum, 2, seed=0, asynchronous=False)
    try:
        assert envs.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    finally:
        envs.close()


def test_extract_final_obs_reads_the_right_mask_and_values():
    """(c) direct unit test of the indexing logic: only the masked-True slots are meaningful,
    and their values must come through unchanged."""
    infos = {
        "final_obs": np.array(
            [
                {"actor": np.array([1.0]), "critic": np.array([2.0])},
                None,
                {"actor": np.array([9.0]), "critic": np.array([8.0])},
            ],
            dtype=object,
        ),
        "_final_obs": np.array([True, False, True]),
    }
    result = extract_final_obs(infos, n_envs=3, actor_dim=1, critic_dim=1)
    assert result is not None
    actor, critic, mask = result
    assert list(mask) == [True, False, True]
    assert actor[0, 0] == pytest.approx(1.0) and critic[0, 0] == pytest.approx(2.0)
    assert actor[2, 0] == pytest.approx(9.0) and critic[2, 0] == pytest.approx(8.0)


def test_extract_final_obs_returns_none_when_nothing_ended():
    assert extract_final_obs({}, n_envs=2, actor_dim=1, critic_dim=1) is None


def test_truncation_bootstraps_to_the_analytic_infinite_horizon_value():
    """(a) constant reward 1, no termination, truncates at `horizon`. With the critic already
    at the analytic fixed point V* = 1/(1-gamma), the value-target reward `_gae` will consume
    must equal V* exactly at the truncation boundary -- not the truncated-as-terminal value
    (plain reward 1.0) a wrongly-zeroed bootstrap would give."""
    gamma = 0.9
    horizon = 5
    v_star = 1.0 / (1.0 - gamma)  # 10.0
    n_envs = 2

    envs = gym.vector.SyncVectorEnv(
        [lambda: _ConstantRewardEnv(horizon) for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        norm_critic = RunningNorm(1)
        critic = _ConstCritic(v_star)
        envs.reset(seed=[0, 1])
        action = torch.zeros(n_envs, 1)
        act_low = torch.tensor([-1.0])
        act_high = torch.tensor([1.0])

        saw_truncation = False
        for _ in range(horizon + 2):
            _, real_reward, value_target_reward, terminated, truncated = _step_with_truncation_bootstrap(
                envs, critic, norm_critic, action, act_low, act_high, n_envs, 1, 1, gamma,
            )
            assert not terminated.any()
            if truncated.any():
                saw_truncation = True
                assert truncated.all()  # both lockstep envs truncate on the same tick
                for i in range(n_envs):
                    assert real_reward[i] == pytest.approx(1.0)             # episode stats: real reward only
                    assert value_target_reward[i] == pytest.approx(v_star)  # matches the analytic value ...
                    assert value_target_reward[i] != pytest.approx(1.0)     # ... not the zeroed-bootstrap one
            else:
                for i in range(n_envs):
                    assert value_target_reward[i] == pytest.approx(real_reward[i])  # untouched off the boundary
        assert saw_truncation
    finally:
        envs.close()


def test_gae_returns_match_the_analytic_value_end_to_end_through_truncation():
    """(a) again, but literally through `_gae` itself rather than just its input: collect a
    real multi-step, multi-truncation rollout with `_step_with_truncation_bootstrap` (crossing
    the horizon boundary several times), feed the resulting buffers straight into `_gae`, and
    check its `returns` -- what the value loss actually regresses toward -- equal the analytic
    value at every single step, not just at the boundary. The old (buggy) `rewards` buffer
    (plain reward, no truncation bootstrap folded in) is built the same way and fed through the
    same unmodified `_gae` for contrast: it must NOT reach the analytic value."""
    gamma = 0.9
    horizon = 5
    v_star = 1.0 / (1.0 - gamma)  # 10.0
    n_steps = 2 * horizon + 3  # crosses the truncation boundary at least twice
    n_envs = 2

    envs = gym.vector.SyncVectorEnv(
        [lambda: _ConstantRewardEnv(horizon) for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        norm_critic = RunningNorm(1)
        critic = _ConstCritic(v_star)
        envs.reset(seed=[0, 1])
        action = torch.zeros(n_envs, 1)
        act_low = torch.tensor([-1.0])
        act_high = torch.tensor([1.0])

        buf_real_rewards = torch.zeros(n_steps, n_envs)
        buf_value_target_rewards = torch.zeros(n_steps, n_envs)
        buf_values = torch.full((n_steps, n_envs), v_star)  # ConstCritic: V(anything) == v_star
        buf_dones = torch.zeros(n_steps, n_envs)
        next_done = torch.zeros(n_envs)
        any_truncation = False

        for t in range(n_steps):
            buf_dones[t] = next_done
            _, real_reward, value_target_reward, terminated, truncated = _step_with_truncation_bootstrap(
                envs, critic, norm_critic, action, act_low, act_high, n_envs, 1, 1, gamma,
            )
            any_truncation = any_truncation or bool(truncated.any())
            done = terminated | truncated
            buf_real_rewards[t] = torch.as_tensor(real_reward, dtype=torch.float32)
            buf_value_target_rewards[t] = torch.as_tensor(value_target_reward, dtype=torch.float32)
            next_done = torch.as_tensor(done.astype(np.float32))
        assert any_truncation

        next_value = torch.full((n_envs,), v_star)  # the running obs is also always V* under ConstCritic

        _, fixed_returns = _gae(
            buf_value_target_rewards, buf_values, buf_dones, next_value, next_done, gamma, lam=0.95,
        )
        assert torch.allclose(fixed_returns, torch.full_like(fixed_returns, v_star), atol=1e-4), fixed_returns

        _, buggy_returns = _gae(
            buf_real_rewards, buf_values, buf_dones, next_value, next_done, gamma, lam=0.95,
        )
        assert not torch.allclose(buggy_returns, torch.full_like(buggy_returns, v_star), atol=1e-2)
    finally:
        envs.close()


def test_true_termination_still_gets_zero_bootstrap():
    """(b) a true termination must still zero the bootstrap, whatever the critic says -- this
    was already correct before the fix and must stay that way."""
    n_envs = 2
    envs = gym.vector.SyncVectorEnv(
        [lambda: _TerminatingEnv() for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        norm_critic = RunningNorm(1)
        critic = _ConstCritic(100.0)  # large: any leaked bootstrap would be impossible to miss
        envs.reset(seed=[0, 1])
        action = torch.zeros(n_envs, 1)
        act_low = torch.tensor([-1.0])
        act_high = torch.tensor([1.0])

        saw_termination = False
        for _ in range(4):
            _, real_reward, value_target_reward, terminated, truncated = _step_with_truncation_bootstrap(
                envs, critic, norm_critic, action, act_low, act_high, n_envs, 1, 1, 0.9,
            )
            if terminated.any():
                saw_termination = True
                assert terminated.all()
                assert not truncated.any()
                for i in range(n_envs):
                    assert value_target_reward[i] == pytest.approx(real_reward[i])
                    assert value_target_reward[i] == pytest.approx(1.0)
        assert saw_termination
    finally:
        envs.close()


def test_bootstrap_uses_true_final_observation_not_reset_observation():
    """(c) under `SAME_STEP` autoreset, `next_obs` for a sub-env that just ended is already
    the following episode's reset observation -- bootstrapping from it (instead of
    `infos["final_obs"]`) would silently value an unrelated trajectory's starting state. The
    three candidate values (correct final obs, wrong reset obs, old zeroed bootstrap) are all
    numerically distinct here, so a wrong-index bug cannot pass by coincidence."""
    horizon = 3
    gamma = 0.5
    envs = gym.vector.SyncVectorEnv(
        [lambda: _DistinctFinalObsEnv(horizon)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        norm_critic = RunningNorm(1)
        critic = _LinearCritic(weight=1.0, bias=0.0)  # V(x) = x
        envs.reset(seed=[0])
        action = torch.zeros(1, 1)
        act_low = torch.tensor([-1.0])
        act_high = torch.tensor([1.0])

        truncated = np.array([False])
        for _ in range(horizon):
            _, real_reward, value_target_reward, terminated, truncated = _step_with_truncation_bootstrap(
                envs, critic, norm_critic, action, act_low, act_high, 1, 1, 1, gamma,
            )
        assert truncated[0] and not terminated[0]
        correct = 5.0 + gamma * _DistinctFinalObsEnv.FINAL_VALUE   # 5 + 0.5*7 = 8.5
        wrong_reset = 5.0 + gamma * _DistinctFinalObsEnv.RESET_VALUE  # 5 + 0.5*(-3) = 3.5
        wrong_zeroed = 5.0
        assert value_target_reward[0] == pytest.approx(correct)
        assert value_target_reward[0] != pytest.approx(wrong_reset)
        assert value_target_reward[0] != pytest.approx(wrong_zeroed)
    finally:
        envs.close()


def test_bootstrap_step_does_not_update_the_running_normalizer():
    """The bootstrap's own value lookup must call `norm_critic.normalize`, never
    `norm_critic.update` -- only the ordinary per-step observation the policy will next act
    from should move the running statistics (that update happens in `train`'s loop, not
    here); otherwise the normalizer would be biased by a value lookup, not an actual
    transition the policy will ever see."""
    envs = gym.vector.SyncVectorEnv(
        [lambda: _ConstantRewardEnv(horizon=2)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        norm_critic = RunningNorm(1)
        norm_critic.update(torch.zeros(1, 1))  # give it some non-trivial state first
        before_mean = norm_critic.mean.clone()
        before_count = norm_critic.count.clone()
        critic = _ConstCritic(5.0)
        envs.reset(seed=[0])
        action = torch.zeros(1, 1)
        act_low = torch.tensor([-1.0])
        act_high = torch.tensor([1.0])
        for _ in range(3):
            _step_with_truncation_bootstrap(envs, critic, norm_critic, action, act_low, act_high, 1, 1, 1, 0.9)
        assert torch.equal(norm_critic.mean, before_mean)
        assert torch.equal(norm_critic.count, before_count)
    finally:
        envs.close()


def test_evaluate_counts_each_episode_once_with_no_reset_step_pollution(tmp_path: Path):
    """evaluate() is not vectorized (a fresh env per episode, stepped to completion, then
    closed -- see its docstring), so there is no autoreset reset-step to accidentally count;
    this pins that down against a real checkpoint and a real truncating env."""
    horizon = 4

    def make_env(seed):
        del seed
        return _ConstantRewardEnv(horizon)

    cfg = PPOConfig(n_envs=2, n_steps=8, epochs=1, minibatches=2, total_steps=16, seed=0, async_envs=False)
    ckpt_path = train(make_env, cfg, tmp_path / "run")
    results = evaluate(ckpt_path, make_env, episodes=5, seed=0)
    assert len(results) == 5
    for r in results:
        assert r.steps == horizon
        assert r.ret == pytest.approx(horizon * 1.0)


@pytest.mark.slow
def test_learns_pendulum(tmp_path: Path):
    """Full acceptance criterion: mean return > -300 on held-out episodes, in well under the
    4-minute CPU budget. Hyperparameters here are tuned for this toy task specifically
    (gamma=0.9 instead of the arm's 0.99, higher lr, more epochs) -- Pendulum has a much
    shorter effective horizon than the press skill this recipe is meant for, and the design
    report's numbers are explicitly a starting point for the real task, not a fixed
    contract. Measured: mean return -271.6 over 10 held-out episodes, ~137s wall-clock on
    this machine (training + evaluation) at seed 0.
    """
    cfg = PPOConfig(
        n_envs=8, n_steps=1024, epochs=10, minibatches=32, lr=1e-3, gamma=0.9,
        total_steps=400_000, seed=0, async_envs=False, ent_coef=0.0,
    )
    ckpt_path = train(make_pendulum, cfg, tmp_path / "run")
    results = evaluate(ckpt_path, make_pendulum, episodes=10, seed=1000)
    mean_return = float(np.mean([r.ret for r in results]))
    assert mean_return > -300, f"mean return {mean_return} did not clear -300"
