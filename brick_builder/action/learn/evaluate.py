"""Measuring the two models: offline against the log, and in the closed loop against the analytic
inverse on the same seeds and the same worlds.

Offline the predictor is scored one step ahead and five steps ahead, per slot field, against two
naive predictors -- "nothing changes" and "the last change repeats" -- and the inverse is scored
against the analytic inverse's own answer on held-out ticks. The closed-form check feeds the
inverse's action through the frozen predictor and asks whether the predicted next state is nearer
the target than the current one.

In the closed loop the only thing that differs between the two columns is where the correction
increment comes from. Everything else -- the recorded target sequence, the gates, the dwells, the
progress index, the end conditions, the seeds and the displaced worlds -- is the same, and the
scorer is `control.pick_place.verify_placement`, which reads the simulator, which is why this
module is not on the control path.

    python -m brick_builder.action.learn.evaluate closed-loop --task placement --inverse analytic
    python -m brick_builder.action.learn.evaluate offline --data data_cache/action/placement
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..models.features import ACTION_W, N_SLATE, ROW_W
from ..models.mlp import Mlp
from . import dataset as ds

FIELDS = ("mass", "centre x", "centre y", "sxx", "sxy", "syy", "orient4 re", "orient4 im")
DWELL_STEPS = ("Approach", "Lift", "Align")
"""Steps where the sequence stands still and a static unknown is estimated."""


# ------------------------------------------------------------------------------ offline: F ---


def _rows_from_flat(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(-1, N_SLATE, ROW_W)


def predictor_metrics(data: ds.Episodes, model: Mlp, *, held_only: bool = True,
                      seed: int = 0, horizon: int = 5) -> dict:
    """One-step and `horizon`-step error per slot field, against the two naive predictors."""
    pairs = ds.predictor_pairs(data)
    held = ds.episode_split(data, np.random.default_rng(seed + 1), 0.15)[pairs["i"]]
    sel = np.nonzero(held if held_only else np.ones_like(held))[0]
    x, y, w = pairs["x"][sel], pairs["y"][sel], pairs["w"][sel]
    i = pairs["i"][sel]

    pred = model.forward(x)
    obs = data["obs_rows"]
    prev_change = np.zeros_like(y)
    ok_prev = np.zeros(len(i), dtype=bool)
    for t, a in enumerate(i):
        if a - 1 >= 0 and data.episode[a - 1] == data.episode[a] \
                and data["step_index"][a - 1] == data["step_index"][a]:
            prev_change[t] = (obs[a] - obs[a - 1]).reshape(-1)
            ok_prev[t] = True

    def per_field(err: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        e = _rows_from_flat(err * w)
        m = _rows_from_flat(w)
        if mask is not None:
            e, m = e[mask], m[mask]
        num = (e ** 2).sum(axis=(0, 1))
        den = np.maximum(m.sum(axis=(0, 1)), 1.0)
        return np.sqrt(num / den)

    one = {
        "model": per_field(pred - y),
        "no change": per_field(0.0 - y),
        "last change repeats": per_field(prev_change - y, mask=ok_prev),
    }

    # Five steps, the model fed its own output, with the actions the run really commanded.
    roll_model, roll_none, target, keep = _rollout(data, model, sel, pairs, horizon)
    many = {"model": per_field(roll_model - target, mask=keep),
            "no change": per_field(roll_none - target, mask=keep)}

    # The conversion was calibrated against the norm of the whole stacked centre difference, so
    # a per-entry root-mean-square is scaled by the square root of how many entries a tick
    # typically has before it is quoted in millimetres.
    mm = ds.mm_per_centre_unit(data)
    entries = math.sqrt(float(_rows_from_flat(w)[:, :, 1].sum(axis=1).mean()))
    return {
        "n_pairs": int(len(sel)),
        "mm_per_centre_unit": mm,
        "slate_entries_per_tick": entries ** 2,
        "one_step": {k: v.tolist() for k, v in one.items()},
        "one_step_centre_mm": {k: float(mm * entries * math.hypot(v[1], v[2]))
                               for k, v in one.items()},
        f"{horizon}_step": {k: v.tolist() for k, v in many.items()},
        f"{horizon}_step_centre_mm": {k: float(mm * entries * math.hypot(v[1], v[2]))
                                      for k, v in many.items()},
        "fields": list(FIELDS),
        "analytic_one_step_mm": analytic_prediction_mm(data),
    }


def _rollout(data: ds.Episodes, model: Mlp, sel: np.ndarray, pairs: dict, horizon: int):
    """Feed the predictor its own output for `horizon` ticks, with the logged actions."""
    i = pairs["i"][sel]
    obs, val, wgt = data["obs_rows"], data["obs_valid"], data["obs_weight"]
    from ..models.features import encode_predictor

    n = len(i)
    rows = obs[i].astype(np.float64).copy()
    start = rows.copy()
    keep = np.ones(n, dtype=bool)
    for h in range(horizon):
        a = i + h
        b = a + 1
        ok = (b < data.n) & (data.episode[np.minimum(b, data.n - 1)] == data.episode[a]) \
            & (data["step_index"][np.minimum(b, data.n - 1)] == data["step_index"][a])
        keep &= ok
        action = np.zeros((n, ACTION_W))
        valid_b = np.minimum(b, data.n - 1)
        action[ok] = (data["cmd"][valid_b][ok] - data["cmd"][a][ok])[:, :ACTION_W]
        x = np.stack([encode_predictor(rows[t], val[a[t]], wgt[a[t]], data["mech"][a[t]],
                                       action[t], data["in_hand"][a[t]]) for t in range(n)])
        rows = rows + model.forward(x).reshape(n, N_SLATE, ROW_W)
    end = np.minimum(i + horizon, data.n - 1)
    target = (obs[end] - obs[i]).reshape(n, -1)
    del start
    return (rows - obs[i]).reshape(n, -1), np.zeros_like(target), target, keep


def analytic_prediction_mm(data: ds.Episodes) -> float:
    """Median of the prediction error the controller already logs, over the same log [mm].

    The controller predicts the next tick's residual as this tick's plus the correction it
    commanded; this is that prediction's error, recomputed from the log.
    """
    i = np.arange(data.n - 1)
    trusted = data.flag("trusted")
    keep = (data.episode[i] == data.episode[i + 1]) & trusted[i] & trusted[i + 1]
    i = i[keep]
    pred = data["measured"][i][:, :2] + data["delta"][i][:, :2]
    err = 1e3 * np.linalg.norm(data["measured"][i + 1][:, :2] - pred, axis=1)
    return float(np.median(err)) if err.size else float("nan")


# ------------------------------------------------------------------------------ offline: G ---


def inverse_metrics(data: ds.Episodes, inverse: Mlp, predictor: Mlp, *, seed: int = 0) -> dict:
    """Action error against the analytic inverse, and the closed-form check through `F`."""
    teach = ds.teacher_pairs(data)
    held_tick = ds.episode_split(data, np.random.default_rng(seed + 1), 0.15)
    sel = np.nonzero(held_tick[teach["i"]])[0]
    sel = sel[teach["i"][sel] + 1 < data.n]
    x, y = teach["x"][sel], teach["y"][sel]
    g = inverse.forward(x).astype(np.float64)
    err = g - y
    scale = np.array([1e3, 1e3, math.degrees(1.0)])

    # Closed form: does F(state, mech, a* + G) land nearer the target than the state already is?
    i = teach["i"][sel]
    from ..models.features import encode_predictor

    obs, val, wgt = data["obs_rows"], data["obs_valid"], data["obs_weight"]
    tgt, tval = data["tgt_rows"], data["tgt_valid"]
    ff = ds.feed_forward(data, i)
    both = (val[i] > 0) & (tval[i] > 0)
    mm = ds.mm_per_centre_unit(data)

    def centre_mm(rows_delta: np.ndarray) -> np.ndarray:
        """The stacked centre difference of one tick, in millimetres of tool motion."""
        d = rows_delta[:, :, ds.CENTRE] * both[:, :, None]
        return mm * np.sqrt((d ** 2).sum(axis=(1, 2)))

    def landed(action_xy_spin: np.ndarray) -> np.ndarray:
        action = ff.copy()
        action[:, 0] += action_xy_spin[:, 0]
        action[:, 1] += action_xy_spin[:, 1]
        action[:, 3] += action_xy_spin[:, 2]
        x_pred = np.stack([encode_predictor(obs[a], val[a], wgt[a], data["mech"][a], action[t],
                                            data["in_hand"][a]) for t, a in enumerate(i)])
        nxt = obs[i] + predictor.forward(x_pred).reshape(len(i), N_SLATE, ROW_W)
        return centre_mm(nxt - tgt[i])

    before = centre_mm(obs[i] - tgt[i])
    after_learned = landed(g)
    after_analytic = landed(data["analytic"][i])
    after_none = landed(np.zeros_like(g))
    move = before > 0.5
    return {
        "n_held_out": int(len(sel)),
        "action_rmse": [float(v) for v in (np.sqrt((err ** 2).mean(axis=0)) * scale)],
        "action_mae": [float(v) for v in (np.abs(err).mean(axis=0) * scale)],
        "action_bias": [float(v) for v in (err.mean(axis=0) * scale)],
        "analytic_rms": [float(v) for v in (np.sqrt((y ** 2).mean(axis=0)) * scale)],
        "units": ["mm", "mm", "deg"],
        "correlation": [float(np.corrcoef(g[:, c], y[:, c])[0, 1]) for c in range(3)],
        "closed_form_mm": {
            "distance before": float(np.median(before[move])),
            "after learned action": float(np.median(after_learned[move])),
            "after analytic action": float(np.median(after_analytic[move])),
            "after no action": float(np.median(after_none[move])),
            "n": int(move.sum()),
        },
    }


# --------------------------------------------------------------------------------- closed loop ---


def run_episode(task: str, seed: int, *, inverse: str = "analytic", residual: bool = False,
                weights: Path | None = None, source=None, ideal=None) -> dict:
    """One closed-loop episode, scored. `ideal` lets a caller reuse a recorded sequence."""
    from ..place import PlaceConfig, TrackingEpisode, record_ideal
    from ..worlds import SUPPLY_BODY, build_pick_world, build_world
    from ...control.pick_place import verify_placement

    cfg = PlaceConfig(pick=task == "pick", inverse=inverse)
    if ideal is None:
        ideal, nominal = record_ideal(cfg, seed=seed, source=source)
        nominal.close()

    g = None
    if inverse == "learned":
        from ..models.inverse import LearnedInverse

        g = LearnedInverse.load(weights, residual=residual)

    if task == "pick":
        scene, d_xy, d_yaw, b_xy, _ = build_pick_world(seed)
        brick = SUPPLY_BODY
    else:
        scene, d_xy, d_yaw = build_world(seed)
        brick = scene.held_name
        b_xy = np.zeros(2)

    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, source=source, inverse=g)
    t0 = time.perf_counter()
    per_step: dict[str, list[float]] = {}
    try:
        for rec in episode.run():
            if rec.vision:
                per_step.setdefault(f"{rec.step_index:02d} {rec.step}", []).append(rec.err_mm)
        check = verify_placement(scene, brick, (4.5, 3.5, 0))
    finally:
        episode.close()
    depth_mm = float(check.depth_error_mm)
    seated, at_cell, lateral = bool(check.seated), bool(check.at_target_cell), float(check.lateral_mm)

    converge = {}
    for key, errs in per_step.items():
        name = key[3:]
        if name not in DWELL_STEPS:
            continue
        hit = next((n for n, e in enumerate(errs) if e == e and e <= 1.0), None)
        converge[key] = {"ticks": len(errs),
                         "ticks_to_converge": hit if hit is not None else -1,
                         "final_mm": errs[-1]}
    preds = [p for p in episode.pred_errors if p == p]
    errs = [e for e in episode.errors if e == e]
    failure, ticks = episode.failure or "", len(episode.errors)
    del episode, scene, check
    gc.collect()
    return {
        "task": task, "seed": seed, "inverse": inverse, "residual": residual,
        "seated": seated, "at_target_cell": at_cell,
        "lateral_mm": lateral, "depth_error_mm": depth_mm,
        "plate_offset_mm": float(1e3 * np.linalg.norm(d_xy)),
        "plate_yaw_deg": float(math.degrees(d_yaw)),
        "brick_offset_mm": float(1e3 * np.linalg.norm(b_xy)),
        "dwells": converge,
        "median_prediction_mm": float(statistics.median(preds)) if preds else float("nan"),
        "median_tracking_mm": float(statistics.median(errs)) if errs else float("nan"),
        "vision_ticks": ticks,
        "failure": failure,
        "wall_s": float(time.perf_counter() - t0),
    }


def compare(task: str, seeds: Sequence[int], variants: Sequence[dict], *, log=print) -> dict:
    """Every variant on every seed, against the same recorded target sequence per seed.

    The recording is nominal and costs as much as an episode, so it is made once per seed and
    handed to each variant: the columns then differ only in where the correction comes from.
    """
    from ..place import PlaceConfig, record_ideal

    out = {v["name"]: [] for v in variants}
    for seed in seeds:
        ideal, nominal = record_ideal(PlaceConfig(pick=task == "pick"), seed=seed)
        nominal.close()
        for v in variants:
            kw = {k: val for k, val in v.items() if k != "name"}
            row = run_episode(task, seed, ideal=ideal, **kw)
            out[v["name"]].append(row)
            log(f"  {task} seed {seed} {v['name']:16s} seated={row['seated']} "
                f"cell={row['at_target_cell']} lateral={row['lateral_mm']:7.2f} mm  "
                f"pred={row['median_prediction_mm']:.2f} mm  track={row['median_tracking_mm']:.2f} mm"
                f"  {row['failure']}")
    return {"task": task, "seeds": list(seeds),
            "columns": {name: _summary(task, name, rows) for name, rows in out.items()}}


def _summary(task: str, name: str, rows: list[dict]) -> dict:
    seated = sum(r["seated"] and r["at_target_cell"] for r in rows)
    on_cell = sum(r["at_target_cell"] for r in rows)
    lat = sorted(r["lateral_mm"] for r in rows if r["seated"] and r["at_target_cell"])
    on_lat = sorted(r["lateral_mm"] for r in rows if r["at_target_cell"])
    preds = [r["median_prediction_mm"] for r in rows
             if r["median_prediction_mm"] == r["median_prediction_mm"]]
    dwell = {}
    for row in rows:
        for key, v in row["dwells"].items():
            dwell.setdefault(key[3:], []).append(v["ticks_to_converge"])
    return {
        "name": name, "task": task,
        "seated_on_target_cell": f"{seated}/{len(rows)}",
        "on_target_cell": f"{on_cell}/{len(rows)}",
        "lateral_mm_range": [lat[0], lat[-1]] if lat else None,
        "lateral_mm_median_on_cell": float(statistics.median(on_lat)) if on_lat else None,
        "median_prediction_mm": float(statistics.median(preds)) if preds else float("nan"),
        "ticks_to_converge": {k: (float(statistics.median([t for t in v if t >= 0]))
                                  if any(t >= 0 for t in v) else None)
                              for k, v in dwell.items()},
        "rows": rows,
    }


def closed_loop(task: str, seeds: Sequence[int], **kw) -> dict:
    rows = [run_episode(task, seed, **kw) for seed in seeds]
    seated = sum(r["seated"] and r["at_target_cell"] for r in rows)
    lat = sorted(r["lateral_mm"] for r in rows if r["seated"] and r["at_target_cell"])
    preds = [r["median_prediction_mm"] for r in rows if r["median_prediction_mm"] == r["median_prediction_mm"]]
    return {
        "task": task, "inverse": kw.get("inverse", "analytic"),
        "residual": kw.get("residual", False),
        "seeds": list(seeds), "rows": rows,
        "seated_on_target_cell": f"{seated}/{len(rows)}",
        "lateral_mm_range": [lat[0], lat[-1]] if lat else None,
        "lateral_mm_median": float(statistics.median(lat)) if lat else None,
        "median_prediction_mm": float(statistics.median(preds)) if preds else float("nan"),
    }


# --------------------------------------------------------------------------------------- CLI ---


def _variant(name: str) -> dict:
    """`analytic`, `learned`, `residual`, or `<name>:<checkpoint>` for a named alternative."""
    label, _, weights = name.partition(":")
    if label == "analytic":
        return {"name": name, "inverse": "analytic"}
    return {"name": name, "inverse": "learned", "residual": label == "residual",
            "weights": Path(weights) if weights else None}


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MUJOCO_GL", "egl")

    parser = argparse.ArgumentParser(prog="evaluate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    cl = sub.add_parser("closed-loop")
    cl.add_argument("--task", choices=("placement", "pick"), default="placement")
    cl.add_argument("--variants", default="analytic,learned",
                    help="comma-separated: analytic, learned, residual, or learned:<weights.npz>")
    cl.add_argument("--seeds", default="0-9")
    cl.add_argument("--out", default=None)

    off = sub.add_parser("offline")
    off.add_argument("--data", nargs="+", default=["data_cache/action/placement"])
    off.add_argument("--predictor", default="brick_builder/action/weights/predictor.npz")
    off.add_argument("--inverse", default="brick_builder/action/weights/inverse.npz")
    off.add_argument("--limit", type=int, default=None)
    off.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "closed-loop":
        lo, _, hi = args.seeds.partition("-")
        seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
        variants = [_variant(name) for name in args.variants.split(",") if name]
        report = compare(args.task, seeds, variants,
                         log=lambda line: print(line, flush=True))
        print(json.dumps({name: {k: v for k, v in col.items() if k != "rows"}
                          for name, col in report["columns"].items()}, indent=2))
    else:
        data = ds.load(args.data, limit=args.limit)
        predictor = Mlp.load(args.predictor)
        inverse = Mlp.load(args.inverse)
        report = {"ticks": data.n, "episodes": int(data.episode.max() + 1),
                  "predictor": predictor_metrics(data, predictor),
                  "inverse": inverse_metrics(data, inverse, predictor)}
        print(json.dumps(report, indent=2, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
