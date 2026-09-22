"""Training the predictor and the inverse. CPU, deterministic, time-budgeted and resumable.

The predictor is plain regression: one step of the scene, given the action. The inverse is
trained in reverse, two ways at once.

- **Hindsight.** The state `j` ticks later is used as the target and the correction increment the
  controller actually commanded at that tick is the label. With the network's inputs clipped at
  three standard deviations, a target far outside the range the loop ever sees arrives saturated,
  so a distant goal reads as a direction and not as a magnitude.
- **Distal teacher.** The predictor is frozen and the inverse's output is fed through it: the loss
  is the slot distance between `F(state, mech, a* + G(state, target, mech))` and the target state,
  plus a penalty on the size of the action (Jordan and Rumelhart, 1992). No label is needed for
  this term at all -- the target state is its own supervision.

A third, optional term regresses on the analytic inverse's own answer. It is off by default: the
point of the exercise is an inverse learned from the arm's motion, not a distillation of the
Jacobian. It is what the DAgger-style remedy turns on.

Everything is written as one `.npz` per model (`brick_builder/action/weights/`), small enough to
ship; the optimizer state needed to resume lives beside the dataset, which is git-ignored.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..models.features import (
    ACTION_W, INVERSE_IN, INVERSE_OUT, MECH_W, N_SLATE, PER_ROW, PREDICTOR_IN, PREDICTOR_OUT,
    ROW_W,
)
from ..models.mlp import Mlp, from_torch, std_floor
from . import dataset as ds

ACTION_AT = N_SLATE * PER_ROW + MECH_W
"""Where the action sits inside the predictor's input vector."""


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (64, 64)
    batch: int = 256
    lr: float = 1e-3
    epochs: int = 60
    seed: int = 0
    holdout: float = 0.15
    clip: float = 3.0
    distal: float = 0.3
    """Weight on the distal-teacher term."""
    action_penalty: float = 0.02
    teacher: float = 0.0
    """Weight on regression against the analytic inverse. The DAgger-style remedy raises it."""
    hindsight_repeats: int = 2
    max_ff_mm: float | None = None
    """Keep hindsight windows over which the recorded target moves no further than this [mm].
    `None` is every acting tick; a small value is "more data near the dwells"."""
    minutes: float = 30.0


def _torch():
    import torch

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    return torch


def _stack(sizes, torch):
    from torch import nn

    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x.mean(axis=0).astype(np.float32),
            std_floor(x.std(axis=0), 1e-6))


def _norm_in(t, mean, std, clip: float):
    return ((t - mean) / std).clamp(-clip, clip)


# ---------------------------------------------------------------------------------- predictor ---


def train_predictor(data: ds.Episodes, cfg: TrainConfig, out: Path, *,
                    resume: Path | None = None, log=print) -> Mlp:
    torch = _torch()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    pairs = ds.predictor_pairs(data)
    held = ds.episode_split(data, np.random.default_rng(cfg.seed + 1), cfg.holdout)[pairs["i"]]
    x, y, w = pairs["x"], pairs["y"], pairs["w"]
    in_mean, in_std = _stats(x[~held])
    out_mean, out_std = _stats(y[~held])
    log(f"predictor: {int((~held).sum())} train / {int(held.sum())} held-out pairs, "
        f"in {x.shape[1]} -> out {y.shape[1]}")

    net = _stack((PREDICTOR_IN, *cfg.hidden, PREDICTOR_OUT), torch)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    start = 0
    if resume is not None and Path(resume).exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        start = int(state["epoch"])
        log(f"predictor: resumed at epoch {start}")

    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    wt = torch.as_tensor(w, dtype=torch.float32)
    m_in = torch.as_tensor(in_mean)
    s_in = torch.as_tensor(in_std)
    m_out = torch.as_tensor(out_mean)
    s_out = torch.as_tensor(out_std)
    train_idx = np.nonzero(~held)[0]
    held_idx = np.nonzero(held)[0]

    t0 = time.perf_counter()
    val = float("nan")
    # A resume that is already at `epochs` runs no epoch at all and simply republishes the
    # weights it resumed, which is what a round called twice with the same budget should do.
    for epoch in range(start, cfg.epochs):
        order = rng.permutation(train_idx)
        total = 0.0
        for b in range(0, len(order), cfg.batch):
            sel = torch.as_tensor(order[b:b + cfg.batch])
            pred = net(_norm_in(xt[sel], m_in, s_in, cfg.clip))
            target = (yt[sel] - m_out) / s_out
            loss = (wt[sel] * (pred - target) ** 2).sum() / wt[sel].sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        with torch.no_grad():
            sel = torch.as_tensor(held_idx)
            pred = net(_norm_in(xt[sel], m_in, s_in, cfg.clip))
            target = (yt[sel] - m_out) / s_out
            val = float((wt[sel] * (pred - target) ** 2).sum() / wt[sel].sum().clamp(min=1.0))
        log(f"  predictor epoch {epoch:3d}  train {total / max(len(order), 1):.5f}  "
            f"held-out {val:.5f}  ({time.perf_counter() - t0:.0f} s)")
        if resume is not None:
            Path(resume).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1},
                       resume)
        if (time.perf_counter() - t0) / 60.0 > cfg.minutes:
            log(f"  predictor: time budget reached at epoch {epoch}")
            break

    model = from_torch(net, in_mean, in_std, out_mean, out_std, clip=cfg.clip,
                       meta={"kind": "predictor", "pairs": int(x.shape[0]),
                             "held_out_mse": float(val), "seed": cfg.seed})
    model.save(out)
    return model


# ------------------------------------------------------------------------------------ inverse ---


def train_inverse(data: ds.Episodes, predictor: Mlp, cfg: TrainConfig, out: Path, *,
                  resume: Path | None = None, log=print) -> Mlp:
    torch = _torch()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    hind = ds.hindsight_pairs(data, np.random.default_rng(cfg.seed + 2),
                              repeats=cfg.hindsight_repeats, max_ff_mm=cfg.max_ff_mm)
    teach = ds.teacher_pairs(data)
    held_tick = ds.episode_split(data, np.random.default_rng(cfg.seed + 1), cfg.holdout)
    held = held_tick[hind["i"]]

    x, y = hind["x"], hind["y"]
    in_mean, in_std = _stats(x[~held])
    out_mean, out_std = _stats(y[~held])
    log(f"inverse: {int((~held).sum())} train / {int(held.sum())} held-out hindsight pairs, "
        f"{teach['x'].shape[0]} teacher pairs")

    # Everything the distal-teacher term needs, precomputed: the predictor's input with a zero
    # action in it, the feed-forward the action is added to, and the change of the slate rows the
    # target asks for.
    obs, val = data["obs_rows"], data["obs_valid"]
    i, j = hind["i"], hind["j"]
    both = (val[i] > 0) & (val[j] > 0)
    target_change = ((obs[j] - obs[i]) * both[:, :, None]).reshape(len(i), -1)
    weight = np.repeat(both.astype(np.float32), ROW_W, axis=1)
    base = np.stack([
        __import__("brick_builder.action.models.features", fromlist=["x"]).encode_predictor(
            obs[a], val[a], data["obs_weight"][a], data["mech"][a], np.zeros(ACTION_W),
            data["in_hand"][a])
        for a in i]).astype(np.float32)
    ff = ds.feed_forward(data, i).astype(np.float32)

    net = _stack((INVERSE_IN, *cfg.hidden, INVERSE_OUT), torch)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    start = 0
    if resume is not None and Path(resume).exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        start = int(state["epoch"])
        log(f"inverse: resumed at epoch {start}")

    f_layers = [(torch.as_tensor(w), torch.as_tensor(b)) for w, b in predictor.layers]

    def predict(xp):
        h = ((xp - torch.as_tensor(predictor.in_mean)) / torch.as_tensor(predictor.in_std)
             ).clamp(-predictor.clip, predictor.clip)
        for n, (w, b) in enumerate(f_layers):
            h = h @ w.T + b
            if n < len(f_layers) - 1:
                h = torch.tanh(h)
        return h  # in the predictor's own normalized output units

    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    bt = torch.as_tensor(base)
    fft = torch.as_tensor(ff)
    wt = torch.as_tensor(weight)
    ct = torch.as_tensor(((target_change - predictor.out_mean) / predictor.out_std
                          ).astype(np.float32))
    m_in, s_in = torch.as_tensor(in_mean), torch.as_tensor(in_std)
    m_out, s_out = torch.as_tensor(out_mean), torch.as_tensor(out_std)
    zero = torch.zeros(1)

    tx = torch.as_tensor(teach["x"], dtype=torch.float32)
    ty = torch.as_tensor(teach["y"], dtype=torch.float32)
    teach_held = held_tick[teach["i"]]
    teach_train = np.nonzero(~teach_held)[0]

    train_idx = np.nonzero(~held)[0]
    held_idx = np.nonzero(held)[0]
    t0 = time.perf_counter()
    val_loss = float("nan")
    for epoch in range(start, cfg.epochs):
        order = rng.permutation(train_idx)
        totals = np.zeros(3)
        for b in range(0, len(order), cfg.batch):
            sel = torch.as_tensor(order[b:b + cfg.batch])
            g = net(_norm_in(xt[sel], m_in, s_in, cfg.clip))
            l_hind = ((g - (yt[sel] - m_out) / s_out) ** 2).mean()

            action = fft[sel] + torch.cat(
                [g[:, :2] * s_out[:2] + m_out[:2], zero.expand(len(sel), 1),
                 (g[:, 2:3] * s_out[2:3] + m_out[2:3])], dim=1)
            xp = torch.cat([bt[sel][:, :ACTION_AT], action,
                            bt[sel][:, ACTION_AT + ACTION_W:]], dim=1)
            change = predict(xp)
            l_distal = ((wt[sel] * (change - ct[sel]) ** 2).sum()
                        / wt[sel].sum().clamp(min=1.0))
            l_size = (g ** 2).mean()
            loss = l_hind + cfg.distal * l_distal + cfg.action_penalty * l_size

            if cfg.teacher > 0.0 and len(teach_train):
                pick = torch.as_tensor(rng.choice(teach_train, size=min(cfg.batch,
                                                                        len(teach_train)),
                                                  replace=False))
                gt = net(_norm_in(tx[pick], m_in, s_in, cfg.clip))
                loss = loss + cfg.teacher * ((gt - (ty[pick] - m_out) / s_out) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            totals += np.array([float(l_hind.detach()), float(l_distal.detach()),
                                float(l_size.detach())]) * len(sel)
        with torch.no_grad():
            sel = torch.as_tensor(held_idx)
            g = net(_norm_in(xt[sel], m_in, s_in, cfg.clip))
            val_loss = float(((g - (yt[sel] - m_out) / s_out) ** 2).mean())
        n = max(len(order), 1)
        log(f"  inverse epoch {epoch:3d}  hindsight {totals[0] / n:.5f}  "
            f"distal {totals[1] / n:.5f}  size {totals[2] / n:.5f}  "
            f"held-out {val_loss:.5f}  ({time.perf_counter() - t0:.0f} s)")
        if resume is not None:
            Path(resume).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1},
                       resume)
        if (time.perf_counter() - t0) / 60.0 > cfg.minutes:
            log(f"  inverse: time budget reached at epoch {epoch}")
            break

    model = from_torch(net, in_mean, in_std, out_mean, out_std, clip=cfg.clip,
                       meta={"kind": "inverse", "pairs": int(x.shape[0]),
                             "held_out_mse": float(val_loss), "seed": cfg.seed,
                             "distal": cfg.distal, "teacher": cfg.teacher})
    model.save(out)
    return model


def write_report(path: Path, cfg: TrainConfig, predictor: Mlp, inverse: Mlp, extra: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "config": asdict(cfg),
        "predictor": {"sizes": list(predictor.sizes), "params": predictor.n_params,
                      "macs": predictor.macs, "bytes": predictor.weight_bytes,
                      **predictor.meta},
        "inverse": {"sizes": list(inverse.sizes), "params": inverse.n_params,
                    "macs": inverse.macs, "bytes": inverse.weight_bytes, **inverse.meta},
        **extra,
    }, indent=2, default=float) + "\n")
    return path
