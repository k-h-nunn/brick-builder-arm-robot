"""PPO with an asymmetric critic (Pinto 2018), CleanRL-style: one file, one algorithm,
no framework underneath it (see
section 3 for why this replaced a DAgger-first plan, and why not Stable-Baselines3).

Every environment used with this module must expose `observation_space = Dict({"actor":
Box, "critic": Box})`. The actor network only ever sees the "actor" half (what the real
robot could provide); the critic sees "critic" (privileged simulator state, dropped at
export). Action spaces are a single continuous `Box`.

CPU only, per project policy: nothing here calls `.cuda()`, and the caller is responsible
for `CUDA_VISIBLE_DEVICES=""` so an accidental `import torch` elsewhere in the process does
not touch a GPU another job may be using.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from brick_builder.eval.results import EpisodeResult
from brick_builder.rl.nets import Actor, Critic, RunningNorm

try:  # pragma: no cover - exercised whenever tensorboard is present, which it is in this venv
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


@dataclass(kw_only=True)
class PPOConfig:
    """Starting hyperparameters from design report 02 section 3. `total_steps` has no
    sensible default (it depends entirely on the skill being trained); `seed` defaults to 0
    only for convenience since the spec text leaves it unset."""

    n_envs: int = 16
    n_steps: int = 512
    epochs: int = 5
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.02
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    init_log_std: float = -1.0
    total_steps: int
    seed: int = 0

    # Not in the spec's field list, added for a compact-but-complete implementation:
    hidden_actor: tuple[int, ...] = (64, 64)
    hidden_critic: tuple[int, ...] = (256, 256)
    norm_clip: float = 10.0
    async_envs: bool = True          # False selects SyncVectorEnv (used by the test suite)
    vector_context: str = "forkserver"
    log_every: int = 1               # updates between TensorBoard/callback scalar logs


def make_vector_env(
    make_env: Callable[[int], gym.Env],
    n_envs: int,
    seed: int,
    asynchronous: bool,
    context: str = "forkserver",
) -> gym.vector.VectorEnv:
    """Build the vectorized env, running gymnasium 1.3's `SAME_STEP` autoreset (not the
    library default, which is `NEXT_STEP`): a sub-env that ends its episode on step t is
    already reset by the time `step()` returns, so `obs` for that index is the *next*
    episode's first observation and every one of the `n_steps` rows this module collects is a
    real, learnable transition -- unlike `NEXT_STEP`, there is no dummy reset row (ignored
    action, reward forced to 0) that would need to be filtered out of the loss, and no
    off-by-one in `finished_returns`/the running normalizer from one.

    The one thing `SAME_STEP` does not hand you for free is the value bootstrap: the true
    final observation of the episode that just ended is not `obs` (that is the new episode's
    first observation) but `infos["final_obs"][i]`, valid where `infos["_final_obs"][i]` is
    True (see `extract_final_obs`). `_step_with_truncation_bootstrap` is the one place that
    reads it, so `train()`'s rollout loop never has to reason about autoreset itself.

    No GL context exists yet at this point (this module never creates one), so it is always
    safe to fork here even for environments that will create one later, per spec 0.
    """
    env_fns = [(lambda i=i: make_env(seed + i)) for i in range(n_envs)]
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
    if asynchronous:
        return gym.vector.AsyncVectorEnv(env_fns, context=context, autoreset_mode=autoreset_mode)
    return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)


def select_action(
    actor: Actor, norm_actor: RunningNorm, obs_actor: torch.Tensor, deterministic: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (or, if `deterministic`, take the mean of) the actor's action distribution.

    Takes only the actor-half observation by construction, which is what makes the
    "actor never touches the critic observation" acceptance test true: there is no
    parameter here to smuggle privileged state through.
    """
    normed = norm_actor.normalize(obs_actor)
    dist = actor.distribution(normed)
    action = dist.mean if deterministic else dist.sample()
    logprob = dist.log_prob(action).sum(-1)
    return action, logprob


def compute_value(critic: Critic, norm_critic: RunningNorm, obs_critic: torch.Tensor) -> torch.Tensor:
    """Value estimate for a batch of privileged observations, shape (B,)."""
    return critic(norm_critic.normalize(obs_critic)).squeeze(-1)


def extract_final_obs(
    infos: dict, n_envs: int, actor_dim: int, critic_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pull the true final observation of every sub-env whose episode ended this step out of
    a `SAME_STEP`-autoreset vector env's `infos` (see `make_vector_env`).

    `infos["final_obs"]` (when present) is a length-`n_envs` object array holding each ended
    sub-env's real last `{"actor": ..., "critic": ...}` observation dict where
    `infos["_final_obs"]` is True, and `None` elsewhere; both are gymnasium's own vector-env
    convention, not something this module invents. Returns `None` when nothing ended this
    step (`infos` has no `"final_obs"` key at all -- the common case, since most steps do not
    cross an episode boundary).

    Returned arrays are `(n_envs, dim)`; only the rows where the returned mask is True are
    meaningful, the rest are zero filler that must never be used (callers gate on the mask, or
    on `terminated | truncated`, which is exactly the same mask by construction).
    """
    if "final_obs" not in infos:
        return None
    mask = np.asarray(infos["_final_obs"], dtype=bool)
    final_obs_actor = np.zeros((n_envs, actor_dim), dtype=np.float32)
    final_obs_critic = np.zeros((n_envs, critic_dim), dtype=np.float32)
    for i in np.nonzero(mask)[0]:
        fo = infos["final_obs"][i]
        final_obs_actor[i] = np.asarray(fo["actor"], dtype=np.float32).reshape(-1)
        final_obs_critic[i] = np.asarray(fo["critic"], dtype=np.float32).reshape(-1)
    return final_obs_actor, final_obs_critic, mask


def _step_with_truncation_bootstrap(
    envs: gym.vector.VectorEnv,
    critic: Critic,
    norm_critic: RunningNorm,
    action: torch.Tensor,
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    n_envs: int,
    actor_dim: int,
    critic_dim: int,
    gamma: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step every sub-env once; return `(obs, real_reward, value_target_reward, terminated,
    truncated)`.

    This is the fix for rl-eval-export-1: a time-limit truncation is not a true termination,
    so the value function must still bootstrap through it, from the value of the episode's
    actual final observation -- not zero (the old bug: `done = terminated | truncated` zeroed
    the bootstrap on every truncation, which is nearly every episode here) and not
    `V(next_obs)` (the tempting-looking but wrong fix: under `SAME_STEP` autoreset, `next_obs`
    for a sub-env that just ended is already the *following* episode's first observation, an
    unrelated trajectory with no business bootstrapping this one).

    `real_reward` is exactly what the environment paid out this step; episode statistics
    (`ep_return`/`finished_returns`) must be built only from this, never from
    `value_target_reward`. `value_target_reward` is `real_reward` plus `gamma * V(final_obs)`
    added on at indices that truncated without terminating -- this, not `real_reward`, is what
    belongs in the rollout buffer `_gae` consumes, so the value target assumes the episode
    would have continued past the time limit instead of assuming it just ended. A true
    termination (with or without simultaneous truncation) gets no addition, matching the
    already-correct behaviour of zeroing the bootstrap there.
    """
    clipped = torch.clamp(action, act_low, act_high).numpy()
    obs, reward, terminated, truncated, infos = envs.step(clipped)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    real_reward = np.asarray(reward, dtype=np.float64)
    value_target_reward = real_reward.copy()

    truncated_only = truncated & ~terminated
    if truncated_only.any():
        final = extract_final_obs(infos, n_envs, actor_dim, critic_dim)
        if final is None:
            raise RuntimeError(
                "a sub-environment truncated but envs.step()'s info carried no 'final_obs' -- "
                "this bootstrap requires AutoresetMode.SAME_STEP (see make_vector_env); "
                "something changed the vector env's autoreset mode"
            )
        _, final_obs_critic, final_mask = final
        done = terminated | truncated
        if not np.array_equal(final_mask, done):
            raise RuntimeError(
                "infos['_final_obs'] did not match terminated|truncated -- refusing to guess "
                "which sub-envs actually have a final observation to bootstrap from"
            )
        with torch.no_grad():
            v_final = compute_value(
                critic, norm_critic, torch.as_tensor(final_obs_critic, dtype=torch.float32)
            )
        value_target_reward[truncated_only] += gamma * v_final.numpy()[truncated_only]

    return obs, real_reward, value_target_reward, terminated, truncated


def _gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor,
    next_done: torch.Tensor, gamma: float, lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation, CleanRL-indexed: `dones[t]` is the done flag that
    was already true when `rewards[t]`'s step began (i.e. whether the transition at t starts
    from a just-reset observation), matching `SAME_STEP` autoreset semantics.

    `dones` (and `next_done`) still combine `terminated | truncated`: cutting the lambda
    recursion and the bootstrap at *every* episode boundary is correct regardless of why the
    episode ended, because the next slot's value belongs to an unrelated trajectory either
    way. What must differ by *why* the episode ended is not handled here -- `rewards` must
    already carry it: `_step_with_truncation_bootstrap` adds `gamma * V(final_obs)` onto a
    truncated (not terminated) step's reward before it ever reaches this function, so a
    plain zeroed bootstrap here is only ever applied to a true termination, never to a
    time-limit truncation."""
    n_steps, n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(n_envs)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def train(
    make_env: Callable[[int], gym.Env],
    cfg: PPOConfig,
    run_dir: Path,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> Path:
    """Train and checkpoint an asymmetric-critic PPO policy. Returns the checkpoint path.

    `callback(update_index, metrics)` is called after every PPO update (metrics: pg_loss,
    v_loss, entropy, approx_kl, mean_return over episodes that finished this update, and
    global_step). It exists mainly so tests can inspect per-update losses without parsing
    TensorBoard event files; training itself does not depend on it.

    Determinism: `torch.manual_seed(cfg.seed)` is called once at the top (network init and
    Gaussian sampling both draw from torch's global generator, which has no way to thread an
    explicit `torch.Generator` through `nn.init.orthogonal_` or `Distribution.sample()`
    without materially more code). Minibatch shuffling instead uses an explicit
    `numpy.random.Generator(cfg.seed)`, per the project's randomness convention. Two calls
    with the same seed and the same `n_envs` produce identical weights, samples and shuffles,
    hence identical losses on the first update.
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

        actor = Actor(actor_dim, act_dim, cfg.hidden_actor, cfg.init_log_std)
        critic = Critic(critic_dim, cfg.hidden_critic)
        norm_actor = RunningNorm(actor_dim, clip=cfg.norm_clip)
        norm_critic = RunningNorm(critic_dim, clip=cfg.norm_clip)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr, eps=1e-5
        )

        writer = SummaryWriter(log_dir=str(run_dir / "tb")) if SummaryWriter is not None else None
        history: list[tuple[int, float]] = []  # (global_step, mean episodic return) for plotting

        obs, _ = envs.reset(seed=[cfg.seed + i for i in range(cfg.n_envs)])
        obs_actor = torch.as_tensor(np.asarray(obs["actor"]), dtype=torch.float32)
        obs_critic = torch.as_tensor(np.asarray(obs["critic"]), dtype=torch.float32)
        next_done = torch.zeros(cfg.n_envs)
        ep_return = np.zeros(cfg.n_envs, dtype=np.float64)

        batch_size = cfg.n_envs * cfg.n_steps
        minibatch_size = batch_size // cfg.minibatches
        n_updates = max(1, cfg.total_steps // batch_size)
        global_step = 0

        for update in range(n_updates):
            buf_obs_actor = torch.zeros(cfg.n_steps, cfg.n_envs, actor_dim)
            buf_obs_critic = torch.zeros(cfg.n_steps, cfg.n_envs, critic_dim)
            buf_actions = torch.zeros(cfg.n_steps, cfg.n_envs, act_dim)
            buf_logprobs = torch.zeros(cfg.n_steps, cfg.n_envs)
            buf_rewards = torch.zeros(cfg.n_steps, cfg.n_envs)
            buf_dones = torch.zeros(cfg.n_steps, cfg.n_envs)
            buf_values = torch.zeros(cfg.n_steps, cfg.n_envs)
            finished_returns: list[float] = []

            for t in range(cfg.n_steps):
                global_step += cfg.n_envs
                buf_obs_actor[t] = obs_actor
                buf_obs_critic[t] = obs_critic
                buf_dones[t] = next_done

                with torch.no_grad():
                    norm_actor.update(obs_actor)
                    norm_critic.update(obs_critic)
                    action, logprob = select_action(actor, norm_actor, obs_actor)
                    value = compute_value(critic, norm_critic, obs_critic)
                buf_actions[t] = action
                buf_logprobs[t] = logprob
                buf_values[t] = value

                obs, real_reward, value_target_reward, terminated, truncated = _step_with_truncation_bootstrap(
                    envs, critic, norm_critic, action, act_low, act_high,
                    cfg.n_envs, actor_dim, critic_dim, cfg.gamma,
                )
                done = terminated | truncated
                ep_return += real_reward  # real reward only: statistics must not see the bootstrap
                for i, d in enumerate(done):
                    if d:
                        finished_returns.append(float(ep_return[i]))
                        ep_return[i] = 0.0

                buf_rewards[t] = torch.as_tensor(value_target_reward, dtype=torch.float32)
                obs_actor = torch.as_tensor(np.asarray(obs["actor"]), dtype=torch.float32)
                obs_critic = torch.as_tensor(np.asarray(obs["critic"]), dtype=torch.float32)
                next_done = torch.as_tensor(done.astype(np.float32))

            with torch.no_grad():
                next_value = compute_value(critic, norm_critic, obs_critic)
                advantages, returns = _gae(
                    buf_rewards, buf_values, buf_dones, next_value, next_done, cfg.gamma, cfg.lam
                )

            b_obs_actor = buf_obs_actor.reshape(-1, actor_dim)
            b_obs_critic = buf_obs_critic.reshape(-1, critic_dim)
            b_actions = buf_actions.reshape(-1, act_dim)
            b_logprobs = buf_logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_norm_actor = norm_actor.normalize(b_obs_actor)
            b_norm_critic = norm_critic.normalize(b_obs_critic)

            indices = np.arange(batch_size)
            last_pg_loss = last_v_loss = last_entropy = last_kl = 0.0
            for epoch in range(cfg.epochs):
                rng.shuffle(indices)
                kls = []
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start : start + minibatch_size]
                    dist = actor.distribution(b_norm_actor[mb])
                    new_logprob = dist.log_prob(b_actions[mb]).sum(-1)
                    entropy = dist.entropy().sum(-1)
                    logratio = new_logprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                    kls.append(approx_kl.item())

                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    new_value = critic(b_norm_critic[mb]).squeeze(-1)
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm
                    )
                    optimizer.step()
                    last_pg_loss, last_v_loss = pg_loss.item(), v_loss.item()
                    last_entropy = entropy_loss.item()

                last_kl = float(np.mean(kls))
                if cfg.target_kl is not None and last_kl > 1.5 * cfg.target_kl:
                    break  # early stop this update's epochs, as in the SB3/CleanRL recipe

            mean_return = float(np.mean(finished_returns)) if finished_returns else float("nan")
            if finished_returns:
                history.append((global_step, mean_return))
            metrics = {
                "pg_loss": last_pg_loss,
                "v_loss": last_v_loss,
                "entropy": last_entropy,
                "approx_kl": last_kl,
                "mean_return": mean_return,
                "global_step": float(global_step),
            }
            if writer is not None and update % cfg.log_every == 0:
                for key, val in metrics.items():
                    if key != "global_step" and not np.isnan(val):
                        writer.add_scalar(f"train/{key}", val, global_step)
            if callback is not None:
                callback(update, metrics)

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
            "critic": critic.state_dict(),
            "norm_actor": norm_actor.state_dict(),
            "norm_critic": norm_critic.state_dict(),
            "cfg": asdict(cfg),
            "obs_spec": obs_spec,
        }
        ckpt_path = run_dir / "checkpoint.pt"
        torch.save(checkpoint, ckpt_path)
        with open(run_dir / "returns.json", "w") as f:
            json.dump(history, f)
        return ckpt_path
    finally:
        envs.close()


def load_actor(checkpoint: Path) -> tuple[Actor, RunningNorm, dict]:
    """Rebuild a frozen actor + input normalizer from a checkpoint, for evaluation or
    export. The critic is intentionally not reconstructed here: nothing downstream of
    training is supposed to need it."""
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    obs_spec = ckpt["obs_spec"]
    cfg = ckpt["cfg"]
    actor = Actor(obs_spec["actor_dim"], obs_spec["act_dim"], tuple(cfg["hidden_actor"]))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    norm_actor = RunningNorm(obs_spec["actor_dim"], clip=cfg["norm_clip"])
    norm_actor.load_state_dict(ckpt["norm_actor"])
    norm_actor.frozen = True
    return actor, norm_actor, obs_spec


def evaluate(
    checkpoint: Path, make_env: Callable[[int], gym.Env], episodes: int, seed: int
) -> list[EpisodeResult]:
    """Roll out `episodes` deterministic (mean-action) episodes from a checkpoint.

    Generic over any env exposing the actor/critic Dict contract: envs that report richer
    outcomes (seat depth, load, structure) should put them in `info` under the matching
    `EpisodeResult` field names, which are passed through when present; envs that do not
    (e.g. Pendulum) fall back to the dataclass defaults plus the raw return.

    Deliberately not a vector env: each episode gets its own freshly made, freshly reset
    `env`, stepped in a plain `while not (terminated or truncated)` loop and closed before the
    next one starts. That makes "one `EpisodeResult` per completed episode, never a partial or
    doubled one" true by construction -- there is no autoreset here at all, so there is no
    `SAME_STEP`/`NEXT_STEP` reset-step ambiguity of the kind `_step_with_truncation_bootstrap`
    has to handle for training's vectorized rollout, and nothing to exclude from these counts.
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
                action, _ = select_action(actor, norm_actor, obs_actor, deterministic=True)
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


def plot_learning_curve(history: list[tuple[int, float]], out_path: Path, title: str = "Learning curve") -> Path:
    """Save a PNG of mean episodic return vs. environment step. `history` is exactly the
    list `train()` writes to `<run_dir>/returns.json`. Used both by the docs-media build for
    this package and is the figure-producing function `scripts/make_media.py`
    can call for any RL run.
    """
    import matplotlib

    matplotlib.use("Agg")  # no GUI windows in tests or batch media builds
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [h[0] for h in history]
    returns = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(steps, returns, color="#1f6feb", linewidth=1.5)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episodic return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
