"""Training of `F`: multi-step open-loop rollout loss, deterministic by seed, exact resume.

    L = sum_{h=1..H} 0.9^h [ Huber_1((z^_{t+h} - z_{t+h}) / s) . w_field . w_slot
                             + 0.2 BCE(present_{t+h}) + 0.1 MSE(q_{t+h}) ]  +  1e-4 |theta|^2

- `s`, `w_field`: the layout's scales and weights (centre 1, other dynamic components 0.25).
- `w_slot = 1 - border` of the observed row, over objects of tick t still present at t+h.
- `F`'s slot set is the objects of tick t (valid, slot id >= 0), the set every numpy entry
  point of `ScenePredictor` uses; the background slot is never an input.
- Teacher forcing for the first 20 % of epochs (the observed state at t+h is the input of tick
  h+1 where present), open loop afterwards. Contexts come from the log at every tick.
- AdamW, lr 1e-3, batch 256. Initialization and batch order depend on the seed alone.
- State file after every epoch: parameters, optimizer, batch-order generator, epoch, history.
  A run started again with the same data and configuration continues from it and ends
  bit-identical to an uninterrupted run (same torch thread count).

Reports: `b_cosine`, `one_step_report` and `action_response` run through the numpy entry points
(`ScenePredictor.affine`, `predict`, `action_jacobian`) on the logged frames, background
included; `prediction_report` is the multi-step open-loop error on windows.

Command line:

    python -m brick_builder.rl.scene_target.train_model synthetic --transitions 20000 \
        --epochs 60 --threads 2 --out results/scene_target/predictor_synthetic
    python -m brick_builder.rl.scene_target.train_model episodes \
        --dir data_cache/scene_target/<run>/<level> --out <dir> --threads 2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as Fn

from .codec import MM_PER_WINDOW_UNIT
from .dataset import ToyConfig, Windows, load_episodes, split_logs, synthetic_logs
from .model import Q_W, ModelConfig, ScenePredictor, SlotLayout, fit_input_norm

__all__ = [
    "TrainConfig", "TrainResult", "action_response", "b_cosine", "evaluate_loss",
    "one_step_report", "prediction_report", "rollout_loss", "train",
]


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters of `F`."""

    seed: int = 0
    epochs: int = 60
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    """AdamW decoupled decay; the 1e-4 |theta|^2 term is `l2`, in the loss."""
    l2: float = 1e-4
    discount: float = 0.9
    huber_delta: float = 1.0
    w_valid: float = 0.2
    w_q: float = 0.1
    teacher_frac: float = 0.2

    @property
    def teacher_epochs(self) -> int:
        return int(round(self.teacher_frac * self.epochs))


@dataclass
class TrainResult:
    model: ScenePredictor
    history: list[dict]
    completed: bool
    """All `epochs` ran (False when the time budget stopped the run; the state file resumes it)."""
    resumed_from: int
    """Epoch the run continued from (0 for a fresh run)."""
    seconds: float
    """Wall time of this call's epochs [s]."""
    version: str
    fingerprint: str
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------- loss ---


def rollout_loss(model: ScenePredictor, b: dict, cfg: TrainConfig, *, teacher: bool):
    """`(loss, parts)` of one batch of windows (tensors from `Windows.tensors`)."""
    L = model.layout
    rows0, depth0 = b["rows"][:, 0], b["depth"][:, 0]
    obj0 = b["present"][:, 0]
    tf = (b["rows"][:, 1:], b["depth"][:, 1:], b["present"][:, 1:]) if teacher else None
    steps = model.rollout_t(rows0, depth0, obj0, b["ctx"], b["action"], teacher=tf)
    total = torch.zeros((), dtype=rows0.dtype)
    m0 = obj0.to(rows0.dtype)
    n0 = m0.sum() + 1e-8
    parts = {"state": 0.0, "valid": 0.0, "q": 0.0}
    for h, s in enumerate(steps, start=1):
        tr, td, pres = b["rows"][:, h], b["depth"][:, h], b["present"][:, h]
        pf = pres.to(rows0.dtype)
        r = (s.dyn - L.dyn(tr, td)) / model.dyn_scale
        hub = Fn.huber_loss(r, torch.zeros_like(r), delta=cfg.huber_delta, reduction="none")
        per = (hub * model.dyn_weight).sum(dim=-1)
        w = pf * (1.0 - L.border(tr))
        state = (per * w).sum() / (w.sum() + 1e-8)
        bce = (Fn.binary_cross_entropy_with_logits(s.valid_logit, pf, reduction="none")
               * m0).sum() / n0
        qt = (b["q_next"][:, h - 1] - model.ctx_mean[:Q_W]) / model.ctx_std[:Q_W]
        qmse = Fn.mse_loss(s.q_norm, qt)
        g = cfg.discount ** h
        total = total + g * (state + cfg.w_valid * bce + cfg.w_q * qmse)
        parts["state"] += g * float(state.detach())
        parts["valid"] += g * float(bce.detach())
        parts["q"] += g * float(qmse.detach())
    l2 = sum((p * p).sum() for p in model.net.parameters())
    total = total + cfg.l2 * l2
    parts["l2"] = float(l2.detach())
    return total, parts


@torch.no_grad()
def evaluate_loss(model: ScenePredictor, windows: Windows, cfg: TrainConfig,
                  batch: int = 2048) -> float:
    """Open-loop rollout loss over `windows`, mean over batches weighted by size."""
    was = model.training
    model.eval()
    T = windows.tensors()
    tot = n = 0.0
    for s in range(0, len(windows), batch):
        b = {k: v[s:s + batch] for k, v in T.items()}
        loss, _ = rollout_loss(model, b, cfg, teacher=False)
        m = b["rows"].shape[0]
        tot += float(loss) * m
        n += m
    model.train(was)
    return tot / max(n, 1.0)


# ----------------------------------------------------------------------------- training ---


def _json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _param_hash(model: ScenePredictor) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _save_state(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def train(train_w: Windows, cfg: TrainConfig = TrainConfig(), *, layout: SlotLayout,
          model_cfg: ModelConfig = ModelConfig(), val_w: Windows | None = None,
          state_path=None, out_path=None, init=None, meta: dict | None = None,
          minutes: float | None = None, threads: int | None = None,
          log: Callable[[str], None] | None = print,
          on_step: Callable[[int, int], None] | None = None) -> TrainResult:
    """Trains `F` on `train_w`; resumes from `state_path` when it exists.

    `init` (a `ScenePredictor` or a checkpoint path) warm-starts the parameters and keeps its
    input normalization; otherwise the normalization is fitted on `train_w`. `minutes` stops
    at the first epoch boundary past the budget. `on_step(epoch, step)` runs after every
    optimizer step. The state file refuses a different data set or configuration."""
    if threads is not None:
        torch.set_num_threads(int(threads))
    if train_w.W != layout.W:
        raise ValueError(f"windows are W={train_w.W}; layout is W={layout.W}")
    fp = train_w.fingerprint()
    if init is not None:
        base = init if isinstance(init, ScenePredictor) else ScenePredictor.load(init)
        if base.layout != layout or base.cfg != model_cfg:
            raise ValueError("warm start: the initial model has a different layout or architecture")
        model = copy.deepcopy(base)
        model.meta = dict(meta or base.meta)
        init_hash = _param_hash(base)
    else:
        norm = fit_input_norm(layout, train_w.rows[:, 0], train_w.depth[:, 0],
                              train_w.present[:, 0], train_w.ctx[:, 0])
        model = ScenePredictor.create(layout, model_cfg, seed=cfg.seed, norm=norm, meta=meta)
        init_hash = ""
    key = {"train": fp, "val": val_w.fingerprint() if val_w is not None else "",
           "cfg": asdict(cfg), "model": model_cfg.to_dict(), "layout": layout.to_dict(),
           "init": init_hash}
    key_json = json.dumps(key, sort_keys=True)
    version = _json_hash({"key": key, "meta": {k: model.meta.get(k) for k in
                                                ("source", "weights_sha256", "W")}})[:12]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []
    start = 0
    state_path = Path(state_path) if state_path is not None else None
    if state_path is not None and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        if st["key"] != key_json:
            raise ValueError(f"{state_path} was written for a different data set or configuration")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = json.loads(st["rng"])
        history = json.loads(st["history"])
        start = int(st["epoch"])
        if log:
            log(f"predictor: resumed at epoch {start}")

    T = train_w.tensors()
    model.train()
    t_run = time.perf_counter()
    for epoch in range(start, cfg.epochs):
        if minutes is not None and time.perf_counter() - t_run > 60.0 * minutes:
            break
        teacher = epoch < cfg.teacher_epochs
        t0 = time.perf_counter()
        acc = {"loss": 0.0, "state": 0.0, "valid": 0.0, "q": 0.0}
        n = 0
        for step, idx in enumerate(train_w.batches(cfg.batch, rng)):
            ti = torch.from_numpy(idx)
            b = {k: v[ti] for k, v in T.items()}
            loss, parts = rollout_loss(model, b, cfg, teacher=teacher)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m = len(idx)
            acc["loss"] += float(loss.detach()) * m
            for k in ("state", "valid", "q"):
                acc[k] += parts[k] * m
            n += m
            if on_step is not None:
                on_step(epoch, step)
        rec = {"epoch": epoch + 1, "teacher": bool(teacher),
               **{k: v / max(n, 1) for k, v in acc.items()}}
        if val_w is not None and len(val_w):
            rec["val_loss"] = evaluate_loss(model, val_w, cfg)
            model.train()
        rec["seconds"] = time.perf_counter() - t0
        history.append(rec)
        if state_path is not None:
            _save_state(state_path, {
                "key": key_json, "model": model.state_dict(), "opt": opt.state_dict(),
                "rng": json.dumps(rng.bit_generator.state), "history": json.dumps(history),
                "epoch": epoch + 1})
        if log:
            v = f" val {rec['val_loss']:.4f}" if "val_loss" in rec else ""
            log(f"predictor: epoch {epoch + 1}/{cfg.epochs} loss {rec['loss']:.4f}{v} "
                f"({rec['seconds']:.1f} s{', teacher-forced' if teacher else ''})")
    seconds = time.perf_counter() - t_run
    model.eval()
    completed = len(history) >= cfg.epochs
    model.meta["train"] = {"version": version, "fingerprint": fp, "cfg": asdict(cfg),
                           "epochs": len(history), "windows": len(train_w),
                           "horizon": train_w.horizon, "completed": completed}
    if out_path is not None and completed:
        model.save(out_path)
    return TrainResult(model=model, history=history, completed=completed, resumed_from=start,
                       seconds=seconds, version=version, fingerprint=fp)


# ------------------------------------------------------------------------------- reports ---


@torch.no_grad()
def prediction_report(model: ScenePredictor, windows: Windows, *,
                      mm_scale: float = MM_PER_WINDOW_UNIT, batch: int = 2048) -> dict:
    """Open-loop error per horizon step on `windows`, against copy-last.

    Centre RMSE [mm-eq] (named schema): RMS over present objects weighted by `1 - border`,
    times `mm_scale`. Per-component RMSE in scaled units (`s`). The windows' first tick is the
    logged frame, so this is the error of `ScenePredictor.rollout` on those frames."""
    L = model.layout
    T = windows.tensors()
    H = windows.horizon
    num = np.zeros((H, L.D))
    num_copy = np.zeros((H, L.D))
    den = np.zeros(H)
    for s in range(0, len(windows), batch):
        b = {k: v[s:s + batch] for k, v in T.items()}
        obj0 = b["present"][:, 0]
        steps = model.rollout_t(b["rows"][:, 0], b["depth"][:, 0], obj0, b["ctx"], b["action"])
        z0 = L.dyn(b["rows"][:, 0], b["depth"][:, 0])
        for h, st in enumerate(steps, start=1):
            tr = b["rows"][:, h]
            tgt = L.dyn(tr, b["depth"][:, h])
            w = (b["present"][:, h].to(tr.dtype) * (1.0 - L.border(tr)))[..., None]
            num[h - 1] += ((((st.dyn - tgt) / model.dyn_scale) ** 2) * w).sum(dim=(0, 1)).numpy()
            num_copy[h - 1] += ((((z0 - tgt) / model.dyn_scale) ** 2) * w).sum(dim=(0, 1)).numpy()
            den[h - 1] += float(w.sum())
    return _error_report(L, num, num_copy, den, mm_scale,
                         {"horizon": H, "n_windows": len(windows)})


def _error_report(L: SlotLayout, num, num_copy, den, mm_scale: float, head: dict) -> dict:
    rmse = np.sqrt(num / np.maximum(den, 1e-12)[:, None])
    rmse_copy = np.sqrt(num_copy / np.maximum(den, 1e-12)[:, None])
    out = {**head, "slot_weight": den.tolist(),
           "field_rmse": {n: rmse[:, i].tolist() for i, n in enumerate(L.dyn_names)},
           "field_rmse_copy_last": {n: rmse_copy[:, i].tolist() for i, n in enumerate(L.dyn_names)}}
    if L.centre:
        ci = list(L.centre)
        sc = float(np.asarray(L.dyn_scale)[ci[0]])

        def centre(x):
            return (mm_scale * sc * np.sqrt(np.sum(x[:, ci] ** 2, axis=1))).tolist()

        out["centre_rmse_mm"] = centre(rmse)
        out["centre_rmse_copy_last_mm"] = centre(rmse_copy)
        out["mm_scale"] = float(mm_scale)
    return out


def _log_ticks(logs, stride: int):
    """`(log, t, frame, context, action)` for every `stride`-th transition of every log."""
    for log in logs:
        if log.n_ticks < 2:
            continue
        ctxs = log.contexts()
        for t in range(0, log.n_ticks - 1, max(int(stride), 1)):
            a = log.exec_action[t + 1]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(ctxs[t]))):
                continue
            yield log, t, log.frame(t), ctxs[t], a


def one_step_report(model: ScenePredictor, logs, *, mm_scale: float = MM_PER_WINDOW_UNIT,
                    stride: int = 1) -> dict:
    """One-tick error of `ScenePredictor.predict` on the logged frames (background included),
    against copy-last.

    Objects of tick t paired by slot id at t+1, weighted by `1 - border` of the observed row.
    Centre RMSE [mm-eq] and per-component RMSE in scaled units, as in `prediction_report`."""
    L = model.layout
    s = np.asarray(L.dyn_scale)
    num, num_copy, den = np.zeros((1, L.D)), np.zeros((1, L.D)), np.zeros(1)
    n_ticks = 0
    for log, t, fr, ctx, a in _log_ticks(logs, stride):
        pred = model.predict(fr, ctx, a)
        nxt = log.frame(t + 1)
        dp = L.dyn(pred.rows.astype(np.float64), pred.depth.astype(np.float64))
        d0 = L.dyn(fr.rows.astype(np.float64), fr.depth.astype(np.float64))
        dn = L.dyn(nxt.rows.astype(np.float64), nxt.depth.astype(np.float64))
        n_ticks += 1
        for i in np.nonzero(fr.objects)[0]:
            j = np.nonzero(nxt.objects & (nxt.slot_ids == fr.slot_ids[i]))[0]
            if not j.size:
                continue
            j = int(j[0])
            w = 1.0 - float(L.border(nxt.rows[j].astype(np.float64)))
            num[0] += w * ((dp[i] - dn[j]) / s) ** 2
            num_copy[0] += w * ((d0[i] - dn[j]) / s) ** 2
            den[0] += w
    return _error_report(L, num, num_copy, den, mm_scale,
                         {"horizon": 1, "n_ticks": n_ticks, "path": "predict on logged frames"})


def _cos_stats(a: np.ndarray, b: np.ndarray) -> dict:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-30)
    gain = na / np.maximum(nb, 1e-30)
    return {"n": int(len(cos)), "cos_median": float(np.median(cos)),
            "cos_p10": float(np.percentile(cos, 10)), "cos_min": float(np.min(cos)),
            "cos_mean": float(np.mean(cos)), "gain_median": float(np.median(gain)),
            "gain_p10": float(np.percentile(gain, 10)), "gain_p90": float(np.percentile(gain, 90))}


def b_cosine(model: ScenePredictor, logs, truth, *, stride: int = 1) -> dict:
    """Cosine and gain ratio between `ScenePredictor.affine`'s `B` and a known `B`, on the logged
    frames (background included) of `logs`, every `stride`-th tick.

    `truth(rows [K, W], depth [K], ctx [43]) -> (f [K, D], B [K, D, n_a])` in physical units.
    Per object: `B` flattened in scaled units (`B_dj s_a,j / s_d`). Also reported for the
    centre rows alone."""
    scale = np.asarray(model.layout.dyn_scale)[:, None] / np.asarray(model.cfg.action_scale)[None, :]
    ci = list(model.layout.centre)
    Bm_all, Bt_all = [], []
    for _, _, fr, ctx, _ in _log_ticks(logs, stride):
        o = fr.objects
        if not o.any():
            continue
        _, Bm = model.affine(fr, ctx)
        _, Bt = truth(fr.rows, fr.depth, ctx)
        Bm_all.append(Bm[o] / scale)
        Bt_all.append(np.asarray(Bt)[o] / scale)
    Bm_all = np.concatenate(Bm_all)
    Bt_all = np.concatenate(Bt_all)
    out = {"path": "affine on logged frames",
           "all": _cos_stats(Bm_all.reshape(len(Bm_all), -1), Bt_all.reshape(len(Bt_all), -1))}
    if ci:
        out["centre"] = _cos_stats(Bm_all[:, ci].reshape(len(Bm_all), -1),
                                   Bt_all[:, ci].reshape(len(Bt_all), -1))
    return out


def action_response(model: ScenePredictor, logs, action_range, *, factor: float = 10.0,
                    n: int = 64, truth=None) -> dict:
    """Autograd `d z' / d a` at `factor` x the training action range against `B`, on the first
    `n` logged frames (background included) of `logs`.

    Returns the smallest ratio `|J| / |B|` and the largest `|J - B|` over the objects, the
    prediction's deviation from linearity in `a`, and with `truth` (as in `b_cosine`) the
    range of `|J| / |B_true|` in scaled units."""
    a_big = factor * np.asarray(action_range, dtype=np.float64)
    L = model.layout
    scale = np.asarray(L.dyn_scale)[:, None] / np.asarray(model.cfg.action_scale)[None, :]
    ratios, diffs, lin, true_ratios = [], [], [], []
    for _, _, frame, ctx, _ in _log_ticks(logs, 1):
        if len(ratios) >= n:
            break
        obj = frame.objects
        if not obj.any():
            continue
        J = model.action_jacobian(frame, ctx, a_big)
        _, Bm = model.affine(frame, ctx)
        ratios.append(float(np.linalg.norm(J[obj]) / max(np.linalg.norm(Bm[obj]), 1e-30)))
        diffs.append(float(np.max(np.abs(J[obj] - Bm[obj]))))
        if truth is not None:
            _, Bt = truth(frame.rows, frame.depth, ctx)
            true_ratios.append(float(np.linalg.norm(J[obj] / scale)
                                     / max(np.linalg.norm(np.asarray(Bt)[obj] / scale), 1e-30)))

        def dyn(a):
            p = model.predict(frame, ctx, a)
            return L.dyn(p.rows.astype(np.float64), p.depth.astype(np.float64))

        z0, z1, zb = dyn(np.zeros_like(a_big)), dyn(a_big / factor), dyn(a_big)
        dev = np.abs((zb - z0) - factor * (z1 - z0))[obj] / np.asarray(L.dyn_scale)
        lin.append(float(dev.max()))
    out = {"path": "action_jacobian on logged frames", "factor": factor,
           "action": a_big.tolist(), "n": len(ratios),
           "jacobian_over_B_min": float(min(ratios)), "jacobian_minus_B_max": float(max(diffs)),
           "nonlinearity_max_scaled": float(max(lin))}
    if true_ratios:
        out["jacobian_over_true_B_min"] = float(min(true_ratios))
        out["jacobian_over_true_B_max"] = float(max(true_ratios))
    return out


# ---------------------------------------------------------------------------------- CLI ---


def _loadavg() -> float:
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return float("nan")


def _main_synthetic(args) -> dict:
    toy_cfg = ToyConfig()
    t0 = time.perf_counter()
    logs, toy = synthetic_logs(args.transitions, seed=args.seed, cfg=toy_cfg)
    windows = Windows.from_logs(logs, horizon=args.horizon)
    tr, va = windows.split()
    _, held_logs = split_logs(logs)
    t_data = time.perf_counter() - t0
    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch=args.batch)
    out = Path(args.out)
    load0 = _loadavg()
    res = train(tr, cfg, layout=toy.layout, val_w=va, state_path=out / "train_state.pt",
                out_path=out / "model.pt",
                meta={"source": "synthetic", "weights_sha256": "", "W": toy.layout.W},
                minutes=args.minutes, threads=args.threads)
    load1 = _loadavg()
    ep_s = [h["seconds"] for h in res.history[res.resumed_from:]]
    report = {
        "command": "train_model synthetic", "transitions": int(sum(l.n_transitions for l in logs)),
        "episodes": len(logs), "windows_train": len(tr), "windows_held_out": len(va),
        "horizon": args.horizon, "seed": args.seed, "epochs": cfg.epochs, "batch": cfg.batch,
        "completed": res.completed, "resumed_from": res.resumed_from,
        "torch_threads": torch.get_num_threads(), "params": res.model.n_params,
        "data_seconds": t_data, "train_seconds": res.seconds,
        "epoch_seconds_median": float(np.median(ep_s)) if ep_s else float("nan"),
        "loadavg_start": load0, "loadavg_end": load1, "version": res.version,
        "final": res.history[-1] if res.history else {},
    }
    if res.completed:
        report["b_cosine_held_out"] = b_cosine(res.model, held_logs, toy.affine)
        report["action_response_10x"] = action_response(res.model, held_logs,
                                                        toy_cfg.action_range, truth=toy.affine)
        report["one_step_held_out"] = one_step_report(res.model, held_logs,
                                                      mm_scale=args.mm_scale)
        report["prediction_held_out"] = prediction_report(res.model, va, mm_scale=args.mm_scale)
    return report


def _main_episodes(args) -> dict:
    logs = load_episodes(args.dir)
    if not logs:
        raise SystemExit(f"no episode logs in {args.dir}")
    from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, opaque_schema
    from .codec import Calibration

    W = int(logs[0].z.shape[2])
    meta0 = logs[0].meta
    for log in logs:
        if log.z.shape[2] != W or log.meta.get("source") != meta0.get("source") or \
                log.meta.get("weights_sha256") != meta0.get("weights_sha256"):
            raise SystemExit("episode logs mix sources, weights or widths")
    if W == PLACEHOLDER_SLOT_SCHEMA.W and not args.opaque:
        layout = SlotLayout.for_schema(PLACEHOLDER_SLOT_SCHEMA)
    else:
        cal = Calibration.load(args.calibration) if args.calibration else None
        layout = SlotLayout.for_schema(opaque_schema(W),
                                       feature_std=cal.feature_std if cal else None,
                                       depth_std=cal.depth_std if cal else None)
    windows = Windows.from_logs(logs, horizon=args.horizon)
    tr, va = windows.split()
    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch=args.batch)
    out = Path(args.out)
    meta = {"source": meta0.get("source"), "weights_sha256": meta0.get("weights_sha256", ""),
            "W": W}
    res = train(tr, cfg, layout=layout, val_w=va if len(va) else None,
                state_path=out / "train_state.pt", out_path=out / "model.pt", meta=meta,
                init=args.init, minutes=args.minutes, threads=args.threads)
    report = {"command": "train_model episodes", "episodes": len(logs),
              "transitions": int(sum(l.n_transitions for l in logs)), "windows_train": len(tr),
              "windows_held_out": len(va), "completed": res.completed,
              "train_seconds": res.seconds, "version": res.version,
              "torch_threads": torch.get_num_threads(),
              "final": res.history[-1] if res.history else {}}
    if res.completed and len(va):
        _, held_logs = split_logs(logs)
        report["one_step_held_out"] = one_step_report(res.model, held_logs,
                                                      mm_scale=args.mm_scale)
        report["prediction_held_out"] = prediction_report(res.model, va, mm_scale=args.mm_scale)
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Next-scene-state predictor training")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("synthetic", "episodes"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--epochs", type=int, default=60)
        s.add_argument("--batch", type=int, default=256)
        s.add_argument("--horizon", type=int, default=3)
        s.add_argument("--threads", type=int, default=2)
        s.add_argument("--minutes", type=float, default=None)
        s.add_argument("--mm-scale", type=float, default=MM_PER_WINDOW_UNIT)
        if name == "synthetic":
            s.add_argument("--transitions", type=int, default=20000)
        else:
            s.add_argument("--dir", required=True)
            s.add_argument("--init", default=None)
            s.add_argument("--opaque", action="store_true")
            s.add_argument("--calibration", default=None)
    args = p.parse_args(argv)
    report = _main_synthetic(args) if args.cmd == "synthetic" else _main_episodes(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
