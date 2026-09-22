"""Asymmetric PPO over the pixels-to-correction policy, with an alignment term through `F`.

The algorithm is the one in `brick_builder/rl/ppo_asym.py` and the generalized advantage
estimator, the vector-environment construction and the learning-curve figure are imported from it.
The rollout and update loops are written out again here for one reason: that module's actor is an
MLP over a flat `Box`, and this policy's observation is an image plus three small vectors, so
there is no flat actor half to hand it. Everything else -- the `SAME_STEP` autoreset, the
truncation bootstrap through the critic, the clipped surrogate, the early stop on the approximate
KL -- is the same recipe, and the critic and its normalizer are that module's own classes.

Three things are specific to this stack.

- **The encoder moves slowly.** `encoder_lr_scale` puts the convolutional scene model on a
  hundredth of the head's learning rate, and it stays in evaluation mode so its batch-norm
  statistics never move (see `policy.py`).
- **The alignment term.** The frozen predictor `F` is fed the policy's own slot rows and the tool
  motion the world saw, and its predicted change is compared against the next frame's rows as the
  policy itself read them during the rollout. That target is a frame the encoder has already
  moved past by the time the update runs, which makes it a target-network-style constraint: it
  asks the representation to stay predictable, not to match a fixed label.
- **Budgeted and resumable.** Training stops on its own wall-clock budget and writes a checkpoint
  after every update, so a run cut off by its budget continues from where it stopped. Logging is a
  plain CSV plus a PNG; nothing writes TensorBoard event files.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ...action.models.features import N_SLATE, ROW_W
from ...action.models.inverse import WEIGHTS
from ...action.models.mlp import Mlp
from ..ppo_asym import _gae, make_vector_env, plot_learning_curve
from .env import VisionControlConfig, make_env
from .policy import PolicyConfig, VisionControlPolicy, alignment_input

__all__ = ["VisionTrainConfig", "train", "evaluate"]

LOG_FIELDS = ("update", "global_step", "elapsed_s", "episodes", "mean_return", "seated_rate",
              "mean_lateral_mm", "pg_loss", "v_loss", "entropy", "approx_kl", "align_loss")


@dataclass
class VisionTrainConfig:
    """Batch shape, optimizer and budget. Defaults are sized for two processes on a shared CPU."""

    n_envs: int = 2
    """Environment processes. Two is the ceiling on this machine: each one holds a MuJoCo world, a
    digital twin and an EGL context, and the policy's own forward pass runs in the parent."""

    n_steps: int = 128
    epochs: int = 1
    """One pass over each batch. The gradient step is a forward and a backward through a
    2.4-million-weight convolutional model per sample, so a second epoch costs as much as another
    whole batch of experience and experience is the scarcer of the two here."""

    minibatches: int = 4
    lr: float = 3e-4
    encoder_lr_scale: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    target_kl: float | None = 0.03
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    align_coef: float = 0.1
    """Weight on the next-state prediction term."""

    max_grad_norm: float = 0.5
    minutes: float = 120.0
    total_steps: int = 10 ** 9
    seed: int = 0
    async_envs: bool = True
    threads: int = 3
    env: VisionControlConfig = field(default_factory=VisionControlConfig)


# ------------------------------------------------------------------------------- the predictor ---


def _frozen_predictor(model: Mlp):
    """`F` as a torch function of a batch, in `F`'s own normalized output units."""
    layers = [(torch.as_tensor(w), torch.as_tensor(b)) for w, b in model.layers]
    in_mean = torch.as_tensor(model.in_mean)
    in_std = torch.as_tensor(model.in_std)

    def forward(x: torch.Tensor) -> torch.Tensor:
        h = ((x - in_mean) / in_std).clamp(-model.clip, model.clip)
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < len(layers) - 1:
                h = torch.tanh(h)
        return h

    return forward


# ------------------------------------------------------------------------------------ training ---


def train(cfg: VisionTrainConfig, run_dir: Path, log=print) -> Path:
    """Train, checkpointing after every update. Returns the checkpoint path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(cfg.threads))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    envs = make_vector_env(lambda s: make_env(s, cfg.env), cfg.n_envs, cfg.seed,
                           asynchronous=cfg.async_envs)
    try:
        critic_dim = int(np.prod(envs.single_observation_space["critic"].shape))
        policy = VisionControlPolicy(PolicyConfig(), critic_dim=critic_dim)
        opt = torch.optim.Adam(policy.groups(cfg.lr, cfg.encoder_lr_scale), lr=cfg.lr, eps=1e-5)
        predictor_model = Mlp.load(WEIGHTS / "predictor.npz")
        predict = _frozen_predictor(predictor_model)
        p_out_mean = torch.as_tensor(predictor_model.out_mean)
        p_out_std = torch.as_tensor(predictor_model.out_std)

        start_update, global_step, history = 0, 0, []
        ckpt_path = run_dir / "checkpoint.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            policy.load_state_dict(state["policy"])
            opt.load_state_dict(state["opt"])
            start_update = int(state["update"])
            global_step = int(state["global_step"])
            history = [tuple(h) for h in state["history"]]
            log(f"resumed at update {start_update}, {global_step} steps")

        csv_path = run_dir / "log.csv"
        if not csv_path.exists():
            with csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(LOG_FIELDS)

        obs, _ = envs.reset(seed=[cfg.seed + i for i in range(cfg.n_envs)])
        next_done = torch.zeros(cfg.n_envs)
        ep_return = np.zeros(cfg.n_envs)
        t_start = time.perf_counter()
        update = start_update

        while (time.perf_counter() - t_start) / 60.0 < cfg.minutes and global_step < cfg.total_steps:
            roll = _Rollout(cfg, critic_dim)
            finished, seated, lateral = [], [], []
            for t in range(cfg.n_steps):
                global_step += cfg.n_envs
                feats, rows = [], []
                with torch.no_grad():
                    for i in range(cfg.n_envs):
                        f, r, _ = policy.forward_obs(_actor_at(obs, i))
                        feats.append(f)
                        rows.append(r)
                    feats_t = torch.stack(feats)
                    policy.norm_features.update(feats_t)
                    critic_t = torch.as_tensor(np.asarray(obs["critic"]), dtype=torch.float32)
                    policy.norm_critic.update(critic_t)
                    dist = policy.distribution(feats_t)
                    action = dist.sample()
                    logprob = dist.log_prob(action).sum(-1)
                    value = policy.value(critic_t)

                roll.store(t, obs, torch.stack(rows), critic_t, action, logprob, value, next_done)
                clipped = torch.clamp(action, -1.0, 1.0).numpy()
                obs, reward, terminated, truncated, infos = envs.step(clipped)
                done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
                roll.tool_delta[t] = _info_array(infos, "tool_delta", cfg.n_envs, 4)

                reward = np.asarray(reward, dtype=np.float64)
                value_target = reward.copy()
                only_truncated = np.asarray(truncated, dtype=bool) & ~np.asarray(terminated, bool)
                if only_truncated.any():
                    final = _final_critic(infos, cfg.n_envs, critic_dim)
                    if final is not None:
                        with torch.no_grad():
                            v_final = policy.value(torch.as_tensor(final, dtype=torch.float32))
                        value_target[only_truncated] += cfg.gamma * v_final.numpy()[only_truncated]

                ep_return += reward
                for i, d in enumerate(done):
                    if not d:
                        continue
                    finished.append(float(ep_return[i]))
                    ep_return[i] = 0.0
                    try:
                        ok, lat = _episode_outcome(infos, i)
                    except (KeyError, IndexError, TypeError):
                        ok, lat = None, None
                    if ok is not None:
                        seated.append(float(ok))
                    if lat is not None:
                        lateral.append(lat)
                roll.rewards[t] = torch.as_tensor(value_target, dtype=torch.float32)
                next_done = torch.as_tensor(done.astype(np.float32))

            with torch.no_grad():
                next_value = policy.value(torch.as_tensor(np.asarray(obs["critic"]),
                                                          dtype=torch.float32))
                advantages, returns = _gae(roll.rewards, roll.values, roll.dones, next_value,
                                           next_done, cfg.gamma, cfg.lam)

            metrics = _update(policy, opt, cfg, roll, advantages, returns, rng, predict,
                              p_out_mean, p_out_std)

            update += 1
            mean_return = float(np.mean(finished)) if finished else float("nan")
            if finished:
                history.append((global_step, mean_return))
            row = {
                "update": update, "global_step": global_step,
                "elapsed_s": round(time.perf_counter() - t_start, 1),
                "episodes": len(finished), "mean_return": mean_return,
                "seated_rate": float(np.mean(seated)) if seated else float("nan"),
                "mean_lateral_mm": float(np.mean(lateral)) if lateral else float("nan"),
                **metrics,
            }
            with csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in LOG_FIELDS])
            log("  update {update:4d}  step {global_step:7d}  {elapsed_s:7.0f} s  "
                "episodes {episodes:2d}  return {mean_return:8.3f}  seated {seated_rate:5.2f}  "
                "pg {pg_loss:8.4f}  v {v_loss:9.3f}  kl {approx_kl:.4f}  "
                "align {align_loss:.4f}".format(**row))
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                        "update": update, "global_step": global_step, "history": history,
                        "cfg": asdict(cfg), "critic_dim": critic_dim}, ckpt_path)

        (run_dir / "returns.json").write_text(json.dumps(history))
        if history:
            plot_learning_curve(history, run_dir / "learning_curve.png",
                                title="Vision-control PPO, placement align phase")
        return ckpt_path
    finally:
        envs.close()


class _Rollout:
    """One update's worth of transitions. The frames dominate: 196 kB each, kept as `uint8`."""

    def __init__(self, cfg: VisionTrainConfig, critic_dim: int) -> None:
        t, n = cfg.n_steps, cfg.n_envs
        self.image = np.zeros((t, n, 256, 256, 3), dtype=np.uint8)
        self.mech = np.zeros((t, n, 12), dtype=np.float32)
        self.target = np.zeros((t, n, 8, 14), dtype=np.float32)
        self.progress = np.zeros((t, n, 3), dtype=np.float32)
        self.rows = torch.zeros(t, n, 8, 14)
        self.tool_delta = np.zeros((t, n, 4), dtype=np.float32)
        self.critic = torch.zeros(t, n, critic_dim)
        self.actions = torch.zeros(t, n, 3)
        self.logprobs = torch.zeros(t, n)
        self.values = torch.zeros(t, n)
        self.rewards = torch.zeros(t, n)
        self.dones = torch.zeros(t, n)

    def store(self, t, obs, rows, critic, action, logprob, value, done) -> None:
        actor = obs["actor"]
        self.image[t] = np.asarray(actor["image"])
        self.mech[t] = np.asarray(actor["mech"])
        self.target[t] = np.asarray(actor["target"])
        self.progress[t] = np.asarray(actor["progress"])
        self.rows[t] = rows
        self.critic[t] = critic
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.values[t] = value
        self.dones[t] = done

    def actor_at(self, flat: int, n_envs: int) -> dict:
        t, i = divmod(flat, n_envs)
        return {"image": self.image[t, i], "mech": self.mech[t, i],
                "target": self.target[t, i], "progress": self.progress[t, i]}


def _update(policy, opt, cfg, roll, advantages, returns, rng, predict, p_out_mean, p_out_std):
    """One PPO update over the rollout, plus the alignment term. Returns the logged losses."""
    batch = cfg.n_steps * cfg.n_envs
    b_actions = roll.actions.reshape(-1, 3)
    b_logprobs = roll.logprobs.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_critic = roll.critic.reshape(-1, roll.critic.shape[-1])
    align_target, align_mask, align_action = _alignment_targets(roll, cfg)

    indices = np.arange(batch)
    last = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "align_loss": 0.0}
    for _ in range(cfg.epochs):
        rng.shuffle(indices)
        kls = []
        for start in range(0, batch, max(batch // cfg.minibatches, 1)):
            mb = indices[start:start + max(batch // cfg.minibatches, 1)]
            if not len(mb):
                continue
            feats, rows, valids = [], [], []
            for flat in mb:
                f, r, v = policy.forward_obs(roll.actor_at(int(flat), cfg.n_envs))
                feats.append(f)
                rows.append(r)
                valids.append(v)
            dist = policy.distribution(torch.stack(feats))
            new_logprob = dist.log_prob(b_actions[mb]).sum(-1)
            entropy = dist.entropy().sum(-1)
            logratio = new_logprob - b_logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                kls.append(float(((ratio - 1) - logratio).mean()))

            adv = b_adv[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = torch.max(-adv * ratio,
                                -adv * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
            value = policy.value(b_critic[mb])
            v_loss = 0.5 * ((value - b_returns[mb]) ** 2).mean()

            align = torch.zeros(())
            keep = [j for j, flat in enumerate(mb) if align_mask[int(flat)]]
            if keep and cfg.align_coef > 0.0:
                x = torch.stack([
                    alignment_input(rows[j], valids[j],
                                    torch.as_tensor(roll.actor_at(int(mb[j]), cfg.n_envs)["mech"]),
                                    torch.as_tensor(align_action[int(mb[j])]), 1.0)
                    for j in keep])
                pred = predict(x)
                tgt = (torch.as_tensor(np.stack([align_target[int(mb[j])] for j in keep]))
                       - p_out_mean) / p_out_std
                # Huber, not a square. The policy's slots are filled in the frame's own candidate
                # order, so a brick appearing or leaving the list shuffles a whole row between two
                # ticks and the squared error of that one pair reaches the hundreds; squared, it
                # would swamp the policy gradient it is meant to sit beside.
                align = torch.nn.functional.smooth_l1_loss(pred, tgt)

            loss = (pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                    + cfg.align_coef * align)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
            last = {"pg_loss": float(pg_loss.detach()), "v_loss": float(v_loss.detach()),
                    "entropy": float(entropy.mean().detach()),
                    "approx_kl": float(np.mean(kls)), "align_loss": float(align.detach())}
        if cfg.target_kl is not None and kls and float(np.mean(kls)) > 1.5 * cfg.target_kl:
            break
    return last


def _alignment_targets(roll, cfg):
    """`(change [B, N_SLATE * ROW_W], usable [B], action [B, 4])` for the next-state term.

    A pair is usable when the next tick is in this rollout, in the same environment column, and
    the episode did not end between them.
    """
    t, n = cfg.n_steps, cfg.n_envs
    change = np.zeros((t * n, N_SLATE * ROW_W), dtype=np.float32)
    mask = np.zeros(t * n, dtype=bool)
    action = np.zeros((t * n, 4), dtype=np.float32)
    rows = roll.rows.detach().numpy()
    for a in range(t - 1):
        for i in range(n):
            if roll.dones[a + 1, i] > 0.5:
                continue
            flat = a * n + i
            delta = (rows[a + 1, i, :N_SLATE] - rows[a, i, :N_SLATE])
            change[flat] = np.concatenate([delta[:, 0:1], delta[:, 1:3], delta[:, 3:6],
                                           delta[:, 12:14]], axis=1).reshape(-1)
            action[flat] = roll.tool_delta[a + 1, i]
            mask[flat] = True
    return change, mask, action


def _actor_at(obs: dict, i: int) -> dict:
    return {k: np.asarray(v)[i] for k, v in obs["actor"].items()}


def _info_array(infos: dict, key: str, n: int, width: int) -> np.ndarray:
    out = np.zeros((n, width), dtype=np.float32)
    values = infos.get(key)
    if values is None:
        return out
    for i in range(n):
        v = values[i]
        if v is not None and np.size(v) == width:
            out[i] = np.asarray(v, dtype=np.float32).reshape(width)
    return out


def _final_critic(infos: dict, n: int, dim: int) -> np.ndarray | None:
    if "final_obs" not in infos:
        return None
    out = np.zeros((n, dim), dtype=np.float32)
    for i in np.nonzero(np.asarray(infos["_final_obs"], dtype=bool))[0]:
        out[i] = np.asarray(infos["final_obs"][i]["critic"], dtype=np.float32).reshape(-1)
    return out


def _episode_outcome(infos: dict, i: int):
    """The scored result of the episode that ended at index `i`, or `(None, None)`.

    Under `SAME_STEP` autoreset the sub-environment is already reset by the time `step` returns,
    so the info dict that comes back is the *new* episode's; the one that just ended is kept in
    `infos["final_info"]`, which is where the placement was scored.
    """
    return _at(infos.get("final_info"), "success", i), _at(infos.get("final_info"), "lateral_mm", i)


def _at(final, key: str, i: int):
    """One field of one sub-environment's final info, whichever shape the vector env returns.

    Gymnasium aggregates sub-environment infos into a dict of arrays with a `_key` mask beside
    each, and nests the ended episodes' infos under `final_info` the same way; older builds hand
    back a sequence of dicts instead. Both are read here rather than guessed at, and anything else
    reads as "not reported" rather than as a zero.
    """
    if final is None:
        return None
    value = None
    if isinstance(final, dict):
        mask = final.get(f"_{key}")
        column = final.get(key)
        if column is not None and (mask is None or (i < len(mask) and bool(mask[i]))):
            value = column[i]
    elif i < len(final) and isinstance(final[i], dict):
        value = final[i].get(key)
    if value is None:
        return None
    return bool(value) if key == "success" else float(value)


# ---------------------------------------------------------------------------------- evaluation ---


def evaluate(checkpoint: Path | None, seeds, env_cfg: VisionControlConfig, *,
             deterministic: bool = True, log=print) -> dict:
    """Roll out one episode per seed and score it. `checkpoint=None` is the analytic baseline.

    With no checkpoint the action is exactly zero at every tick, which is today's controller; the
    two columns are otherwise the same code, the same seeds and the same recorded sequences.
    """
    env = make_env(0, env_cfg)
    policy = None
    if checkpoint is not None:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        policy = VisionControlPolicy(PolicyConfig(), critic_dim=int(state["critic_dim"]))
        policy.load_state_dict(state["policy"])
        policy.eval()
        policy.norm_features.frozen = True
        policy.norm_critic.frozen = True

    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed), options={"world_seed": int(seed)})
            done = False
            steps, ret = 0, 0.0
            t0 = time.perf_counter()
            info: dict = {}
            while not done:
                if policy is None:
                    action = np.zeros(3, dtype=np.float32)
                else:
                    with torch.no_grad():
                        f, _, _ = policy.forward_obs(obs["actor"])
                        dist = policy.distribution(f[None])
                        a = dist.mean if deterministic else dist.sample()
                        action = torch.clamp(a, -1.0, 1.0)[0].numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                ret += float(reward)
                steps += 1
                done = terminated or truncated
            rows.append({"seed": int(seed), "steps": steps, "ret": ret,
                         "wall_s": round(time.perf_counter() - t0, 1),
                         **{k: v for k, v in info.items()
                            if k != "params" and np.size(v) == 1}})
            log(f"  seed {seed:3d} seated={rows[-1].get('success')} "
                f"lateral={rows[-1].get('lateral_mm', float('nan')):6.3f} mm  "
                f"track={rows[-1].get('median_tracking_mm', float('nan')):5.2f} mm  "
                f"pred={rows[-1].get('median_prediction_mm', float('nan')):5.2f} mm  "
                f"{rows[-1].get('failure', '')}")
    finally:
        env.close()

    seated = [r for r in rows if r.get("success")]
    lat = sorted(float(r["lateral_mm"]) for r in seated)
    return {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "seeds": [int(s) for s in seeds],
        "seated_on_target_cell": f"{len(seated)}/{len(rows)}",
        "lateral_mm_range": [lat[0], lat[-1]] if lat else None,
        "median_tracking_mm": float(np.median([r["median_tracking_mm"] for r in rows])),
        "median_prediction_mm": float(np.median([r["median_prediction_mm"] for r in rows])),
        "median_ticks": float(np.median([r["steps"] for r in rows])),
        "rows": rows,
    }
