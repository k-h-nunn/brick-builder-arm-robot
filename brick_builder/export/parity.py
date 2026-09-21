"""Parity and timing checks between the torch actor and the exported `CPolicy`.

`check_parity` is the function spec section 5's acceptance criterion names ("a function that
checks parity on N random observations"): parity must hold to <= 1e-5 max absolute error on
10 000 random observations, for a checkpoint from either learner.
`benchmark_inference` measures nanoseconds per call, for the float-`tanhf` and lookup-table
variants, on this machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from brick_builder.export.c_policy import CPolicy
from brick_builder.rl.ppo_asym import load_actor, select_action


@dataclass
class ParityReport:
    n: int
    max_abs_error: float
    mean_abs_error: float


def _torch_actions(checkpoint: Path, obs: np.ndarray) -> np.ndarray:
    """The torch actor's deterministic action for a batch of raw (unnormalized) actor
    observations -- exactly what `c/policy.c` (via `CPolicy`) is supposed to reproduce.

    Branches on the checkpoint's own `cfg["squash"]` (absent means PPO's convention):
    `"clamp"` normalizes, takes the Gaussian mean, and clamps to the action bounds
    (`brick_builder.rl.ppo_asym.evaluate`); `"tanh"` normalizes, takes the mean, and maps
    `tanh(mean)` onto the bounds (`brick_builder.rl.sac_asym.select_action`) -- this is why the
    check does not simply call `brick_builder.rl.ppo_asym.select_action` unconditionally: that
    function knows only the clamp convention, and would give the wrong reference action for a
    SAC checkpoint's tanh one.
    """
    actor, norm_actor, obs_spec = load_actor(checkpoint)
    act_low = torch.as_tensor(obs_spec["act_low"], dtype=torch.float32)
    act_high = torch.as_tensor(obs_spec["act_high"], dtype=torch.float32)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    squash = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)["cfg"].get("squash", "clamp")
    with torch.no_grad():
        if squash == "tanh":
            mean = actor(norm_actor.normalize(obs_t))
            scale, bias = (act_high - act_low) / 2.0, (act_high + act_low) / 2.0
            action = torch.tanh(mean) * scale + bias
        else:
            action, _ = select_action(actor, norm_actor, obs_t, deterministic=True)
            action = torch.clamp(action, act_low, act_high)
    return action.numpy()


def sample_observations(checkpoint: Path, n: int, seed: int = 0) -> np.ndarray:
    """`n` random actor observations drawn from the checkpoint's own frozen normalizer
    statistics (mean, sqrt(var)), so the sample lands in the range the policy actually saw
    during training rather than an arbitrary one that mostly gets clipped."""
    from brick_builder.rl.ppo_asym import load_actor as _load  # local import: torch is heavy

    _, norm_actor, obs_spec = _load(checkpoint)
    mean = norm_actor.mean.numpy()
    std = np.sqrt(norm_actor.var.numpy())
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=np.maximum(std, 1e-6), size=(n, obs_spec["actor_dim"])).astype(np.float32)


def check_parity(checkpoint: Path, cpolicy: CPolicy, n: int = 10_000, seed: int = 0) -> ParityReport:
    """Compare `cpolicy.act` against the torch actor on `n` random observations."""
    obs = sample_observations(checkpoint, n, seed)
    torch_act = _torch_actions(checkpoint, obs)
    c_act = cpolicy.act_batch(obs)
    err = np.abs(torch_act - c_act)
    return ParityReport(n=n, max_abs_error=float(err.max()), mean_abs_error=float(err.mean()))


def plot_parity_and_speed(
    checkpoint: Path,
    cp_float: CPolicy,
    cp_table: CPolicy,
    out_path: Path,
    n_parity: int = 10_000,
    n_calls: int = 3_000,
) -> Path:
    """Save a two-panel PNG: parity (max abs error vs. torch) and inference speed, float
    `tanhf` vs. table tanh. Used by `docs/guide/07-export-and-firmware.md` and, per spec
    section 0, is the figure-producing function `scripts/make_media.py` can call for this
    module."""
    import matplotlib

    matplotlib.use("Agg")  # no GUI windows in tests or batch media builds
    import matplotlib.pyplot as plt

    rep_float = check_parity(checkpoint, cp_float, n=n_parity)
    rep_table = check_parity(checkpoint, cp_table, n=n_parity)
    obs = sample_observations(checkpoint, 1, seed=1)[0]
    ns_float = benchmark_inference(cp_float, obs, n_calls=n_calls, n_warmup=n_calls // 10)
    ns_table = benchmark_inference(cp_table, obs, n_calls=n_calls, n_warmup=n_calls // 10)

    variants = ["float tanhf", "table tanh"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)

    axes[0].bar(variants, [rep_float.max_abs_error, rep_table.max_abs_error], color=["#1f6feb", "#d9822b"])
    axes[0].axhline(1e-5, color="red", linestyle="--", linewidth=1, label="1e-5 acceptance bound")
    axes[0].set_ylabel(f"max abs error vs torch (n={n_parity})")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("C actor parity")

    axes[1].bar(variants, [ns_float, ns_table], color=["#1f6feb", "#d9822b"])
    axes[1].set_ylabel("median ns / inference")
    axes[1].set_title("C actor inference time")

    fig.suptitle("brick_builder/export: policy_theta.h forward pass")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def benchmark_inference(cpolicy: CPolicy, obs: np.ndarray, n_calls: int = 50_000, n_warmup: int = 1_000) -> float:
    """Median nanoseconds per `cpolicy.act(obs)` call, timed from Python with
    `time.perf_counter_ns`. This includes ctypes call overhead, which the real firmware
    (calling `policy_act` directly from C) will not pay; it is still the number this
    machine can actually report without a separate C benchmark harness."""
    for _ in range(n_warmup):
        cpolicy.act(obs)
    samples = np.empty(n_calls, dtype=np.float64)
    for i in range(n_calls):
        t0 = time.perf_counter_ns()
        cpolicy.act(obs)
        samples[i] = time.perf_counter_ns() - t0
    return float(np.median(samples))
