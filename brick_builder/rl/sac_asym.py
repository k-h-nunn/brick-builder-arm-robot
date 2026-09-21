"""Soft Actor-Critic with an asymmetric twin critic (Pinto 2018 for the asymmetry; Haarnoja
et al. 2018 for SAC itself): the off-policy fallback recipe called for once
`brick_builder/rl/ppo_asym.py`'s on-policy throughput falls below about 2,000 env steps/s (the
spec section 1's own measured numbers trigger this).

Reuses `brick_builder.rl.ppo_asym`/`brick_builder.rl.nets` pieces directly rather than re-implementing
them: `Actor` and `RunningNorm` (imported, not copied) are exactly the classes PPO's actor
uses, and this module's checkpoint has the same top-level keys (`actor`, `norm_actor`, `cfg`,
`obs_spec`) with the same meaning, so `brick_builder.rl.ppo_asym.load_actor` and
`brick_builder.export.actor_to_c.load_actor_arrays` can both reconstruct a SAC actor's trunk
without change -- a SAC checkpoint is a drop-in for the C export pipeline. `make_vector_env`
is also reused as-is (vector envs, `SAME_STEP` autoreset, forkserver by default).

What is genuinely new here, because SAC needs it and PPO does not:

- **Twin critics with LayerNorm** (`QCritic`, defined in this file, not `nets.py`): plain
  `Linear -> LayerNorm -> Tanh` blocks over the concatenated (critic observation, action).
  LayerNorm is what keeps the critic loss stable under this recipe's high update-to-data
  ratio (Q-values would otherwise drift under repeated updates on a mostly-stationary
  replay buffer); PPO's on-policy `Critic` never needs this because it is only ever updated a
  few epochs against on-policy data. Twin critics + their own target networks (Polyak
  averaged) fight overestimation bias exactly as in the original SAC/TD3 papers. Never
  exported: `brick_builder/export` only ever reads a checkpoint's `actor` key.
- **A squashed Gaussian action.** The actor trunk (`nets.Actor`) is unchanged -- a plain MLP
  mean plus a state-independent learned `log_std`, exactly PPO's architecture, which is what
  keeps the checkpoint's `actor` state dict byte-for-byte compatible with the C exporter's
  expected layer-key layout (`net.{2i}.weight`/`net.{2i}.bias`). What differs is what is done
  with that mean and std: the deployed (and sampled) action is `tanh(u) * scale + bias`, `u`
  drawn from `Normal(mean, std)` (or `u = mean` for the deterministic/eval path), where
  `scale = (act_high - act_low) / 2` and `bias = (act_high + act_low) / 2` map the (-1, 1)
  tanh range onto the environment's actual action bounds. `brick_builder/export` records this as
  the checkpoint's `"squash"` mode (`"tanh"` here, `"clamp"` for a PPO checkpoint that has no
  such field) so `c/policy.c` applies the matching final step.
- **A replay buffer with 3-step returns** (`ReplayBuffer`, `_flush_n_step`): plain
  numpy arrays, CPU only. Vector-env transitions are accumulated per sub-environment into a
  short sliding window; once a window fills (or an episode ends, which flushes every
  still-open partial window at whatever length it has) one `(obs, action, n_step_return,
  next_obs, done)` row is written to the buffer, discounted by `gamma` inside the window and
  bootstrapped by the target critics outside it. `done` records a true MDP termination only,
  never a time-limit truncation (rl-eval-export-1) -- a truncated episode still gets `done=0`
  and `next_obs` set to the environment's real final observation (from the vector env's
  `infos["final_obs"]`, not the post-autoreset observation of the following episode), so
  `sac_update`'s target still bootstraps through it instead of learning that running out the
  clock ends the world.
- **Automatic entropy tuning** (`log_alpha`, an SGD'd scalar): the temperature that trades
  off reward against action entropy is itself learned, target entropy `-act_dim` by default
  (Haarnoja et al. 2018's own heuristic), rather than fixed as a hyperparameter.

CPU only, per project policy: nothing here calls `.cuda()`; the caller sets
`CUDA_VISIBLE_DEVICES=""` before this module (or `torch`) is ever imported, exactly as
`ppo_asym.py` requires.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from brick_builder.eval.results import EpisodeResult
from brick_builder.rl.nets import Actor, RunningNorm
from brick_builder.rl.ppo_asym import extract_final_obs
from brick_builder.rl.ppo_asym import load_actor as _load_ppo_style_actor
from brick_builder.rl.ppo_asym import make_vector_env, plot_learning_curve  # noqa: F401 -- re-exported

try:  # pragma: no cover - exercised whenever tensorboard is present, which it is in this venv
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


@dataclass(kw_only=True)
class SACConfig:
    """Starting hyperparameters for the off-policy fallback. Like
    `PPOConfig`, `total_steps` has no sensible default and `seed` defaults to 0."""

    n_envs: int = 8
    n_step: int = 3                    # n-step return length
    gamma: float = 0.99
    tau: float = 0.005                 # Polyak averaging rate for the target critics
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None  # None -> -act_dim, the standard SAC heuristic
    buffer_size: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 1_000          # global env-steps of uniform-random action before learning starts
    utd_ratio: float = 0.5             # gradient updates per environment step, aggregated over n_envs
    init_log_std: float = -0.5
    total_steps: int
    seed: int = 0

    hidden_actor: tuple[int, ...] = (64, 64)
    hidden_critic: tuple[int, ...] = (256, 256)
    norm_clip: float = 10.0
    async_envs: bool = True            # False selects SyncVectorEnv (used by the test suite)
    vector_context: str = "forkserver"
    log_every: int = 1                 # updates between TensorBoard/callback scalar logs
    squash: str = "tanh"               # recorded in the checkpoint; brick_builder/export reads this
    device: str = "cpu"                # kept for interface parity; the GPU belongs to another job


def _layer_init(layer: nn.Linear, gain: float, bias: float = 0.0) -> nn.Linear:
    """Same orthogonal-init convention as `brick_builder.rl.nets._layer_init`, kept as a small
    local copy rather than importing a name that module marks private with a leading
    underscore."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class QCritic(nn.Module):
    """One member of the twin critic: `Q(critic_obs, action) -> scalar`, `Linear -> LayerNorm
    -> Tanh` per hidden layer. Never exported (`brick_builder/export` only reads a checkpoint's
    `actor` key), so it does not need `nets.Critic`'s export-friendly plain-MLP shape."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.in_dim = in_dim
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(_layer_init(nn.Linear(d, h), gain=2**0.5))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            d = h
        layers.append(_layer_init(nn.Linear(d, 1), gain=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        """Q-value, shape (B, 1), for a batch of concatenated (normalized critic obs, action)."""
        return self.net(obs_action)


def action_scale_bias(act_low: torch.Tensor, act_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`tanh(u) * scale + bias` maps `(-1, 1)` onto `[act_low, act_high]`."""
    return (act_high - act_low) / 2.0, (act_high + act_low) / 2.0


def squashed_action(
    actor: Actor,
    norm_actor: RunningNorm,
    obs_actor: torch.Tensor,
    act_scale: torch.Tensor,
    act_bias: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`(action, logprob)`. The deployed action is always `tanh(u) * scale + bias`, `u` the
    actor's raw (pre-squash) output -- `u = mean` when `deterministic` (this is the "the
    deployed action is tanh(mean)" convention this project uses, which
    `brick_builder/export`'s C actor also implements), else the reparameterized sample `mean +
    std * eps` so gradients flow through `actor`'s parameters for the actor-loss term.

    `logprob` is `None` when `deterministic` (eval never needs it); otherwise it is the
    log-density of the *squashed* action, i.e. the base Gaussian log-density minus the
    standard tanh change-of-variables correction (Haarnoja et al. 2018, appendix C) and the
    constant `log(scale)` term the additional affine map contributes.
    """
    normed = norm_actor.normalize(obs_actor)
    mean = actor(normed)
    std = actor.log_std.exp().expand_as(mean)
    if deterministic:
        u = mean
    else:
        u = mean + std * torch.randn_like(mean)
    squashed = torch.tanh(u)
    action = squashed * act_scale + act_bias

    if deterministic:
        return action, None

    normal_logprob = (-0.5 * ((u - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
    correction = torch.log(act_scale * (1.0 - squashed.pow(2)) + 1e-6).sum(-1)
    logprob = normal_logprob - correction
    return action, logprob


class ReplayBuffer:
    """Plain numpy circular buffer of n-step transitions. CPU only, per project policy; a
    numpy.random.Generator (not torch's global RNG) draws the sample indices, per the
    project's randomness convention."""

    def __init__(self, capacity: int, actor_dim: int, critic_dim: int, act_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.obs_actor = np.zeros((capacity, actor_dim), dtype=np.float32)
        self.obs_critic = np.zeros((capacity, critic_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward_n = np.zeros(capacity, dtype=np.float32)
        self.next_obs_actor = np.zeros((capacity, actor_dim), dtype=np.float32)
        self.next_obs_critic = np.zeros((capacity, critic_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.gamma_n = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs_actor, obs_critic, action, reward_n, next_obs_actor, next_obs_critic, done, gamma_n) -> None:
        i = self.ptr
        self.obs_actor[i] = obs_actor
        self.obs_critic[i] = obs_critic
        self.action[i] = action
        self.reward_n[i] = reward_n
        self.next_obs_actor[i] = next_obs_actor
        self.next_obs_critic[i] = next_obs_critic
        self.done[i] = done
        self.gamma_n[i] = gamma_n
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs_actor": torch.as_tensor(self.obs_actor[idx]),
            "obs_critic": torch.as_tensor(self.obs_critic[idx]),
            "action": torch.as_tensor(self.action[idx]),
            "reward_n": torch.as_tensor(self.reward_n[idx]),
            "next_obs_actor": torch.as_tensor(self.next_obs_actor[idx]),
            "next_obs_critic": torch.as_tensor(self.next_obs_critic[idx]),
            "done": torch.as_tensor(self.done[idx]),
            "gamma_n": torch.as_tensor(self.gamma_n[idx]),
        }


def _flush_n_step(
    buf: deque, replay: ReplayBuffer, gamma: float, next_obs_actor: np.ndarray, next_obs_critic: np.ndarray,
    n_step: int, terminal: bool, terminated: bool = True,
) -> None:
    """Turn the contents of one env's sliding window into replay rows.

    Not `terminal`: `buf` has exactly reached `n_step` entries; emit the single full n-step
    transition starting at `buf[0]` and pop it, sliding the window by one. `next_obs_actor`/
    `next_obs_critic` here are the ordinary continuing next observation (bootstrap enabled,
    `done=0`) -- this branch never touches an episode boundary, so it needed no change for
    rl-eval-export-1.

    `terminal`: the episode just ended (`terminated or truncated`) with `buf` holding
    1..n_step entries (it may be shorter than `n_step` near the start of a short episode);
    emit one transition per start position still in `buf` (longest first), each summing
    rewards to the end of the episode, then clear `buf`. What `done` (and hence the target's
    bootstrap term in `sac_update`) gets stored as depends on *why* the episode ended, which
    is what `terminated` (only meaningful when `terminal` is True) says:

    - `terminated=True` (a true MDP termination): `done=1`, zeroing the bootstrap -- correct,
      unchanged from before this fix. `next_obs_actor`/`next_obs_critic` are never read in
      this case (multiplied by `1 - done = 0`), so what the caller passes for them does not
      matter.
    - `terminated=False` (a time-limit truncation, the common case here): `done=0`, so
      `sac_update` bootstraps through the target critics from `next_obs_actor`/
      `next_obs_critic` -- the caller MUST pass the environment's true final observation here
      (e.g. from the vector env's `infos["final_obs"]` under `SAME_STEP` autoreset, see
      `extract_final_obs`), never the post-autoreset observation of the following episode:
      that belongs to an unrelated trajectory and bootstrapping from it would be exactly the
      wrong-index bug this fix exists to avoid, not merely the old zeroed-bootstrap one.
    """
    if terminal:
        done_value = 1.0 if terminated else 0.0
        rewards = [r for (_, _, _, r) in buf]
        k = len(rewards)
        for start in range(k):
            ret = 0.0
            g = 1.0
            for r in rewards[start:]:
                ret += g * r
                g *= gamma
            obs_a0, obs_c0, act0, _ = buf[start]
            replay.add(obs_a0, obs_c0, act0, ret, next_obs_actor, next_obs_critic, done_value, g)
        buf.clear()
        return

    if len(buf) == n_step:
        ret = 0.0
        g = 1.0
        for (_, _, _, r) in buf:
            ret += g * r
            g *= gamma
        obs_a0, obs_c0, act0, _ = buf[0]
        replay.add(obs_a0, obs_c0, act0, ret, next_obs_actor, next_obs_critic, 0.0, g)
        buf.popleft()


def _collect_transition(
    envs: gym.vector.VectorEnv, action: np.ndarray, n_envs: int, actor_dim: int, critic_dim: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Step every sub-env once with `action`; return `(next_obs, reward, terminated,
    truncated, final_obs_actor, final_obs_critic)`.

    The last two are `None` unless at least one sub-env ended this step (the common case is
    that none did), in which case they are `(n_envs, dim)` arrays -- valid only at the indices
    where `terminated | truncated` is True -- of that sub-env's real final observation.

    This is rl-eval-export-1's fix for SAC's data-collection side: `make_vector_env` runs
    `AutoresetMode.SAME_STEP`, under which `next_obs` for a sub-env that ends this step is
    already the *following* episode's first observation, not the ending episode's real last
    one (see `extract_final_obs`). Reading `next_obs` there instead -- the bug this replaces --
    would bootstrap a truncated episode from an unrelated trajectory's starting state, which is
    a strictly worse mistake than the old zeroed bootstrap it was silently standing in for.
    """
    next_obs, reward, terminated, truncated, infos = envs.step(action)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    reward = np.asarray(reward, dtype=np.float64)
    done = terminated | truncated

    final_obs_actor = final_obs_critic = None
    if done.any():
        final = extract_final_obs(infos, n_envs, actor_dim, critic_dim)
        if final is None:
            raise RuntimeError(
                "a sub-environment ended but envs.step()'s info carried no 'final_obs' -- "
                "this requires AutoresetMode.SAME_STEP (see make_vector_env)"
            )
        final_obs_actor, final_obs_critic, final_mask = final
        if not np.array_equal(final_mask, done):
            raise RuntimeError(
                "infos['_final_obs'] did not match terminated|truncated -- refusing to guess "
                "which sub-envs actually have a final observation"
            )
    return next_obs, reward, terminated, truncated, final_obs_actor, final_obs_critic


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.mul_(1.0 - tau).add_(s_param, alpha=tau)


def sac_update(
    batch: dict[str, torch.Tensor],
    actor: Actor, critic1: QCritic, critic2: QCritic, critic1_targ: QCritic, critic2_targ: QCritic,
    norm_actor: RunningNorm, norm_critic: RunningNorm,
    log_alpha: torch.Tensor, target_entropy: float,
    act_scale: torch.Tensor, act_bias: torch.Tensor,
    actor_opt: torch.optim.Optimizer, critic_opt: torch.optim.Optimizer, alpha_opt: torch.optim.Optimizer,
    gamma: float, tau: float,
) -> dict[str, float]:
    """One SAC gradient step (critics, actor, temperature, target Polyak update) against one
    sampled `batch` (see `ReplayBuffer.sample`). A free function (not a method) so both
    `train()` and the throughput benchmark in `tests/rl/test_sac_asym.py` can call it against
    arbitrary network sizes without going through the environment-stepping loop.
    """
    obs_actor = batch["obs_actor"]
    obs_critic = batch["obs_critic"]
    action = batch["action"]
    reward_n = batch["reward_n"]
    next_obs_actor = batch["next_obs_actor"]
    next_obs_critic = batch["next_obs_critic"]
    done = batch["done"]
    gamma_n = batch["gamma_n"]

    alpha = log_alpha.exp()

    with torch.no_grad():
        next_action, next_logprob = squashed_action(actor, norm_actor, next_obs_actor, act_scale, act_bias)
        next_normed_critic = norm_critic.normalize(next_obs_critic)
        next_q1 = critic1_targ(torch.cat([next_normed_critic, next_action], dim=-1)).squeeze(-1)
        next_q2 = critic2_targ(torch.cat([next_normed_critic, next_action], dim=-1)).squeeze(-1)
        next_q = torch.minimum(next_q1, next_q2) - alpha * next_logprob
        target = reward_n + gamma_n * (1.0 - done) * next_q

    normed_critic = norm_critic.normalize(obs_critic)
    critic_input = torch.cat([normed_critic, action], dim=-1)
    q1 = critic1(critic_input).squeeze(-1)
    q2 = critic2(critic_input).squeeze(-1)
    critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    new_action, new_logprob = squashed_action(actor, norm_actor, obs_actor, act_scale, act_bias)
    q1_pi = critic1(torch.cat([normed_critic.detach(), new_action], dim=-1)).squeeze(-1)
    q2_pi = critic2(torch.cat([normed_critic.detach(), new_action], dim=-1)).squeeze(-1)
    q_pi = torch.minimum(q1_pi, q2_pi)
    actor_loss = (alpha.detach() * new_logprob - q_pi).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    alpha_loss = -(log_alpha * (new_logprob.detach() + target_entropy)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    _soft_update(critic1_targ, critic1, tau)
    _soft_update(critic2_targ, critic2, tau)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha_loss": alpha_loss.item(),
        "alpha": alpha.item(),
        "q_mean": q1.mean().item(),
    }


def train(
    make_env: Callable[[int], gym.Env],
    cfg: SACConfig,
    run_dir: Path,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> Path:
    """Train and checkpoint an asymmetric-critic SAC policy. Returns the checkpoint path.

    Determinism: `torch.manual_seed(cfg.seed)` at the top (network init and action sampling
    both draw from torch's global generator, as in `ppo_asym.train`); replay-buffer sampling
    uses an explicit `numpy.random.Generator(cfg.seed)`. Two calls with the same seed and
    `n_envs` produce identical weights, samples and, hence, identical losses on the first
    update.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    envs = make_vector_env(make_env, cfg.n_envs, cfg.seed, asynchronous=cfg.async_envs, context=cfg.vector_context)
    try:
        obs_space = envs.single_observation_space
        act_space = envs.single_action_space
        actor_dim = int(np.prod(obs_space["actor"].shape))
        critic_dim = int(np.prod(obs_space["critic"].shape))
        act_dim = int(np.prod(act_space.shape))
        act_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        act_high = torch.as_tensor(act_space.high, dtype=torch.float32)
        act_scale, act_bias = action_scale_bias(act_low, act_high)
        target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)

        actor = Actor(actor_dim, act_dim, cfg.hidden_actor, cfg.init_log_std)
        critic1 = QCritic(critic_dim + act_dim, cfg.hidden_critic)
        critic2 = QCritic(critic_dim + act_dim, cfg.hidden_critic)
        critic1_targ = QCritic(critic_dim + act_dim, cfg.hidden_critic)
        critic2_targ = QCritic(critic_dim + act_dim, cfg.hidden_critic)
        critic1_targ.load_state_dict(critic1.state_dict())
        critic2_targ.load_state_dict(critic2.state_dict())
        for p in critic1_targ.parameters():
            p.requires_grad_(False)
        for p in critic2_targ.parameters():
            p.requires_grad_(False)

        norm_actor = RunningNorm(actor_dim, clip=cfg.norm_clip)
        norm_critic = RunningNorm(critic_dim, clip=cfg.norm_clip)
        log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=True)

        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
        critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.lr_critic)
        alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr_alpha)

        replay = ReplayBuffer(cfg.buffer_size, actor_dim, critic_dim, act_dim, rng)
        n_step_bufs = [deque() for _ in range(cfg.n_envs)]

        writer = SummaryWriter(log_dir=str(run_dir / "tb")) if SummaryWriter is not None else None
        history: list[tuple[int, float]] = []

        obs, _ = envs.reset(seed=[cfg.seed + i for i in range(cfg.n_envs)])
        obs_actor = torch.as_tensor(np.asarray(obs["actor"]), dtype=torch.float32)
        obs_critic = torch.as_tensor(np.asarray(obs["critic"]), dtype=torch.float32)
        ep_return = np.zeros(cfg.n_envs, dtype=np.float64)

        n_iters = max(1, cfg.total_steps // cfg.n_envs)
        global_step = 0
        update_idx = 0
        update_debt = 0.0
        last_metrics: dict[str, float] = {}

        for it in range(n_iters):
            with torch.no_grad():
                norm_actor.update(obs_actor)
                norm_critic.update(obs_critic)
                if global_step < cfg.warmup_steps:
                    action = act_low + (act_high - act_low) * torch.rand(cfg.n_envs, act_dim)
                else:
                    action, _ = squashed_action(actor, norm_actor, obs_actor, act_scale, act_bias)

            action_np = torch.clamp(action, act_low, act_high).numpy()
            next_obs, reward, terminated, truncated, final_obs_actor, final_obs_critic = _collect_transition(
                envs, action_np, cfg.n_envs, actor_dim, critic_dim,
            )
            done = terminated | truncated
            ep_return += reward

            next_obs_actor = np.asarray(next_obs["actor"], dtype=np.float32)
            next_obs_critic = np.asarray(next_obs["critic"], dtype=np.float32)
            obs_actor_np = obs_actor.numpy()
            obs_critic_np = obs_critic.numpy()

            finished_returns: list[float] = []
            for i in range(cfg.n_envs):
                n_step_bufs[i].append((obs_actor_np[i], obs_critic_np[i], action_np[i], float(reward[i])))
                if done[i]:
                    _flush_n_step(
                        n_step_bufs[i], replay, cfg.gamma, final_obs_actor[i], final_obs_critic[i],
                        cfg.n_step, terminal=True, terminated=bool(terminated[i]),
                    )
                    finished_returns.append(float(ep_return[i]))
                    ep_return[i] = 0.0
                else:
                    _flush_n_step(
                        n_step_bufs[i], replay, cfg.gamma, next_obs_actor[i], next_obs_critic[i],
                        cfg.n_step, terminal=False,
                    )

            obs_actor = torch.as_tensor(next_obs_actor, dtype=torch.float32)
            obs_critic = torch.as_tensor(next_obs_critic, dtype=torch.float32)
            global_step += cfg.n_envs

            update_debt += cfg.utd_ratio * cfg.n_envs
            n_updates = int(update_debt)
            update_debt -= n_updates
            did_update = False
            if global_step >= cfg.warmup_steps and replay.size >= cfg.batch_size:
                for _ in range(n_updates):
                    batch = replay.sample(cfg.batch_size)
                    last_metrics = sac_update(
                        batch, actor, critic1, critic2, critic1_targ, critic2_targ,
                        norm_actor, norm_critic, log_alpha, target_entropy, act_scale, act_bias,
                        actor_opt, critic_opt, alpha_opt, cfg.gamma, cfg.tau,
                    )
                    update_idx += 1
                    did_update = True

            if finished_returns:
                history.append((global_step, float(np.mean(finished_returns))))

            if did_update or finished_returns:
                mean_return = float(np.mean(finished_returns)) if finished_returns else float("nan")
                metrics = {"global_step": float(global_step), "mean_return": mean_return, **last_metrics}
                if writer is not None and update_idx % max(1, cfg.log_every) == 0:
                    for key, val in metrics.items():
                        if key != "global_step" and not np.isnan(val):
                            writer.add_scalar(f"train/{key}", val, global_step)
                if callback is not None:
                    callback(update_idx, metrics)

        if writer is not None:
            writer.close()

        obs_spec = {
            "actor_dim": actor_dim,
            "critic_dim": critic_dim,
            "act_dim": act_dim,
            "act_low": act_space.low.tolist(),
            "act_high": act_space.high.tolist(),
        }
        checkpoint = {
            "actor": actor.state_dict(),
            "critic1": critic1.state_dict(),
            "critic2": critic2.state_dict(),
            "norm_actor": norm_actor.state_dict(),
            "norm_critic": norm_critic.state_dict(),
            "log_alpha": log_alpha.detach().item(),
            "cfg": asdict(cfg),
            "obs_spec": obs_spec,
        }
        ckpt_path = run_dir / "checkpoint.pt"
        torch.save(checkpoint, ckpt_path)
        import json

        with open(run_dir / "returns.json", "w") as f:
            json.dump(history, f)
        return ckpt_path
    finally:
        envs.close()


def load_actor(checkpoint: Path) -> tuple[Actor, RunningNorm, dict]:
    """Rebuild a frozen actor + input normalizer from a SAC checkpoint.

    Identical in every respect to `brick_builder.rl.ppo_asym.load_actor` -- reused directly,
    since a SAC checkpoint's `actor`/`norm_actor`/`cfg`/`obs_spec` keys are laid out exactly
    like a PPO checkpoint's (this is what "checkpoint layout" reuse means in practice). The
    returned actor's raw `forward()` is *not* yet squashed; callers that want the deployed
    action must go through `select_action` below (or `evaluate`), not `ppo_asym.select_action`,
    which does not know about the tanh squash.
    """
    return _load_ppo_style_actor(checkpoint)


def select_action(
    actor: Actor, norm_actor: RunningNorm, obs_actor: torch.Tensor, act_low: torch.Tensor, act_high: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The SAC counterpart of `ppo_asym.select_action`: same call shape (actor, normalizer,
    actor-only observation, deterministic flag in, `(action, logprob)` out) but the returned
    action is squashed (`tanh(mean)` when `deterministic`), as `brick_builder/export`'s C actor
    also computes for a checkpoint whose `cfg["squash"] == "tanh"`."""
    act_scale, act_bias = action_scale_bias(act_low, act_high)
    return squashed_action(actor, norm_actor, obs_actor, act_scale, act_bias, deterministic)


def evaluate(checkpoint: Path, make_env: Callable[[int], gym.Env], episodes: int, seed: int) -> list[EpisodeResult]:
    """Roll out `episodes` deterministic (`tanh(mean)`) episodes from a SAC checkpoint.

    Same shape and same `EpisodeResult` fallbacks as `brick_builder.rl.ppo_asym.evaluate`, so
    swapping which module trained a checkpoint does not change how it is evaluated. Also the
    same non-vectorized, one-fresh-env-per-episode rollout, so it counts each episode exactly
    once and has no autoreset reset-step to exclude in the first place -- see the longer note
    on this in `ppo_asym.evaluate`.
    """
    actor, norm_actor, obs_spec = load_actor(checkpoint)
    act_low = torch.as_tensor(obs_spec["act_low"], dtype=torch.float32)
    act_high = torch.as_tensor(obs_spec["act_high"], dtype=torch.float32)

    results = []
    for i in range(episodes):
        env = make_env(seed + i)
        obs, info = env.reset(seed=seed + i)
        terminated = truncated = False
        steps = 0
        ret = 0.0
        while not (terminated or truncated):
            with torch.no_grad():
                obs_actor = torch.as_tensor(np.asarray(obs["actor"]), dtype=torch.float32).unsqueeze(0)
                action, _ = select_action(actor, norm_actor, obs_actor, act_low, act_high, deterministic=True)
                action = torch.clamp(action, act_low, act_high).squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            steps += 1
        env.close()
        # rl-eval-export-2: pass peak_load/structure_intact through as reported, `None` when
        # absent -- never substitute a "safe"-looking 0.0/True here, or an env that stops
        # reporting one would silently look like it passed every acceptance check that reads it.
        peak_load = info.get("peak_load")
        structure_intact = info.get("structure_intact")
        results.append(
            EpisodeResult(
                success=bool(info.get("success", False)),
                steps=steps,
                seat_depth_error=info.get("seat_depth_error"),
                peak_load=None if peak_load is None else float(peak_load),
                structure_intact=None if structure_intact is None else bool(structure_intact),
                failure=info.get("failure"),
                ret=ret,
                params={"seed": seed + i, **info.get("params", {})},
            )
        )
    return results
