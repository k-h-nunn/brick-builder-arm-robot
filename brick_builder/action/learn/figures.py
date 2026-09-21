"""The two figures the closed-loop guide page carries.

    python -m brick_builder.action.learn.figures --out docs/media/action
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_trace(task: str = "placement", seed: int = 1, inverse: str = "analytic") -> dict:
    """One closed-loop run, kept per vision tick: the prediction error and the step it was in."""
    from ..place import PlaceConfig, TrackingEpisode, record_ideal
    from ..worlds import build_pick_world, build_world

    cfg = PlaceConfig(pick=task == "pick", inverse=inverse)
    ideal, nominal = record_ideal(cfg, seed=seed)
    nominal.close()
    g = None
    if inverse == "learned":
        from ..models.inverse import LearnedInverse

        g = LearnedInverse.load()
    scene = (build_pick_world(seed)[0] if task == "pick" else build_world(seed)[0])
    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, inverse=g)
    predict, track, steps = [], [], []
    try:
        for rec in episode.run():
            if rec.vision:
                predict.append(rec.predict_mm)
                track.append(rec.err_mm)
                steps.append(rec.step)
    finally:
        episode.close()
    return {"predict": np.array(predict), "track": np.array(track), "steps": steps}


def prediction_error_figure(trace: dict, out_path: Path) -> Path:
    plt = _plt()
    # A tick where the loop took no reading -- pressing, frozen below the grasp height, or with
    # too little in view to trust -- carries the previous tick's number in the record. Plotting
    # that would draw a flat line where in fact nothing was measured, so those ticks are gaps.
    predict = np.where(np.isnan(trace["track"]), np.nan, trace["predict"])
    steps = trace["steps"]
    x = np.arange(len(predict))

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    ax.plot(x, predict, color="#1f6feb", linewidth=1.1)
    top = float(np.nanmax(predict)) if np.any(np.isfinite(predict)) else 1.0
    ax.set_ylim(0.0, 1.15 * max(top, 0.1))
    bounds = [i for i in range(1, len(steps)) if steps[i] != steps[i - 1]]
    for b in bounds:
        ax.axvline(b, color="#999999", linewidth=0.7, linestyle="--")
    seen = set()
    for start, name in zip([0] + bounds, [steps[0]] + [steps[b] for b in bounds]):
        if name in seen:
            continue
        seen.add(name)
        ax.text(start + 1, ax.get_ylim()[1] * 0.96, name, fontsize=7, rotation=90,
                va="top", color="#555555")
    ax.set_xlabel("Vision tick [10 Hz]")
    ax.set_ylabel("Prediction error [mm]")
    ax.set_title("Prediction error over one closed-loop placement [mm]")
    ax.set_xlim(0, max(len(predict) - 1, 1))
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def action_figure(x: np.ndarray, analytic: np.ndarray, learned: np.ndarray,
                  out_path: Path) -> Path:
    """Learned against analytic correction, per component, on held-out ticks."""
    plt = _plt()
    scale = (1e3, 1e3, math.degrees(1.0))
    names = ("x", "y", "spin")
    units = ("mm/tick", "mm/tick", "deg/tick")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=150)
    for c, ax in enumerate(axes):
        a = analytic[:, c] * scale[c]
        b = learned[:, c] * scale[c]
        ax.scatter(a, b, s=3, alpha=0.18, color="#1f6feb", linewidths=0)
        lim = float(np.percentile(np.abs(np.concatenate([a, b])), 99.5))
        lim = lim if lim > 0 else 1.0
        ax.plot([-lim, lim], [-lim, lim], color="#d9822b", linewidth=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Analytic [{units[c]}]")
        ax.set_ylabel(f"Learned [{units[c]}]")
        r = float(np.corrcoef(a, b)[0, 1])
        ax.set_title(f"Correction in {names[c]},  r = {r:.2f}", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
    fig.suptitle("Learned and analytic correction on held-out ticks", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    del x
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MUJOCO_GL", "egl")

    parser = argparse.ArgumentParser(prog="figures", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="docs/media/action")
    parser.add_argument("--data", nargs="+", default=["data_cache/action/placement"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", default="brick_builder/action/weights/inverse.npz")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    trace = run_trace("placement", seed=args.seed)
    print(prediction_error_figure(trace, out / "prediction_error.png"))

    from ..models.mlp import Mlp
    from . import dataset as ds

    data = ds.load(args.data, limit=args.limit)
    teach = ds.teacher_pairs(data)
    held = ds.episode_split(data, np.random.default_rng(1), 0.15)[teach["i"]]
    sel = np.nonzero(held)[0]
    net = Mlp.load(args.weights)
    print(action_figure(teach["x"][sel], teach["y"][sel], net.forward(teach["x"][sel]),
                        out / "learned_vs_analytic_action.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
