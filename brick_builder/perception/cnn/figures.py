"""Figures for the guide page, built from a finished evaluation.

- `samples_figure`: held-out frames, ground truth and prediction.
- `failures_figure`: the frames with the most missed bricks, each labelled with its cause.
- `error_vs_size_figure`: recall and centre error against apparent brick size.
- `class_share_figure`: per-class intersection over union.

Titles and axis labels are short technical captions with units; type is at least 12 pt and every
axis is labelled. Segmentation panels are drawn by `source.CnnSceneModel.colourize_instances`, so
every object appears in the mean colour its own slot row publishes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .evaluate import brick_ids, gt_masks, green_ids, touching_same_colour_ids
from .labels import CLASS_NAMES

MEDIA_DIR = Path("docs/media/perception_cnn")

def _predicted_image(source, rgb: np.ndarray) -> np.ndarray:
    """One `feature_maps` call, drawn as a segmentation."""
    maps = source.feature_maps(rgb, window=None)
    return source.colourize_instances(maps["instance_map"], maps["state"])


def _truth_image(source, sample) -> np.ndarray:
    """The renderer's instance map, coloured by the same rule as the prediction."""
    from .demo import ground_truth_instance_image

    return ground_truth_instance_image(sample, source)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def samples_figure(samples, source, out_path: Path | str = MEDIA_DIR / "samples.png",
                   n: int = 6) -> Path:
    """Held-out frames: what the camera saw, the true objects, and the model's objects."""
    plt = _plt()
    chosen = _one_per_kind(samples, n)
    fig, axes = plt.subplots(3, len(chosen), figsize=(2.9 * len(chosen), 9.6), dpi=140)
    for col, sample in enumerate(chosen):
        source.reset()
        axes[0, col].imshow(sample.rgb)
        axes[1, col].imshow(_truth_image(source, sample))
        axes[2, col].imshow(_predicted_image(source, sample.rgb))
        axes[0, col].set_title(str(sample.meta.get("kind", "frame")), fontsize=13)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
    for r, label in enumerate(("Arm camera POV", "Ground truth", "Prediction")):
        axes[r, 0].set_ylabel(label, fontsize=13)
    fig.suptitle("Held-out frames - ground truth and prediction", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _save(fig, out_path)


def _one_per_kind(samples, n: int) -> list:
    """One frame of each scene kind first, then whatever else is needed to fill the row -- a strip
    of six frames of the same kind shows less than a strip covering the set."""
    by_kind: dict[str, list] = {}
    for sample in samples:
        by_kind.setdefault(str(sample.meta.get("kind", "frame")), []).append(sample)
    chosen: list = []
    round_index = 0
    while len(chosen) < n and any(len(v) > round_index for v in by_kind.values()):
        for kind in sorted(by_kind):
            if len(chosen) >= n:
                break
            if len(by_kind[kind]) > round_index:
                chosen.append(by_kind[kind][round_index])
        round_index += 1
    return chosen or list(samples[:n])


def failures_figure(samples, source, out_path: Path | str = MEDIA_DIR / "failures.png",
                    n: int = 6) -> Path:
    """The frames the model does worst on, with the reason written out in words."""
    plt = _plt()
    scored = []
    for sample in samples:
        source.reset()
        source.observe(sample.rgb, window=None)
        truth = gt_masks(sample)
        ids = brick_ids(sample)
        if not ids:
            continue
        pred_bricks = [m for m, k in zip(source.last_masks, source.last_kinds) if k == "brick"]
        missed, small, occluded = 0, 0, 0
        for inst_id in ids:
            mask = truth[inst_id]
            best = max((float(np.logical_and(mask, p).sum()) /
                        max(float(np.logical_or(mask, p).sum()), 1.0) for p in pred_bricks),
                       default=0.0)
            if best < 0.5:
                missed += 1
                if mask.sum() < 800:
                    small += 1
                if sample.instance[np.logical_not(mask)].size and _is_occluded(sample, inst_id):
                    occluded += 1
        if missed:
            greens = green_ids(sample)
            touching = touching_same_colour_ids(sample)
            reason = "small brick" if small else "jaw occlusion"
            if occluded and not small:
                reason = "arm occlusion"
            if greens & set(ids):
                reason = "green on green plate"
            elif touching:
                reason = "touching same colour"
            scored.append((missed / len(ids), missed, len(ids), reason, sample))
    scored.sort(key=lambda r: -r[0])
    chosen = scored[:n]
    if not chosen:
        chosen = [(0.0, 0, 0, "no misses in batch", samples[0])]

    cols = min(n, len(chosen))
    fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 7.2), dpi=140, squeeze=False)
    for col, (_frac, missed, total, reason, sample) in enumerate(chosen):
        source.reset()
        axes[0][col].imshow(_truth_image(source, sample))
        axes[1][col].imshow(_predicted_image(source, sample.rgb))
        axes[0][col].set_title(f"{missed}/{total} bricks missed\n{reason}", fontsize=12)
        for r in range(2):
            axes[r][col].set_xticks([])
            axes[r][col].set_yticks([])
    axes[0][0].set_ylabel("Ground truth", fontsize=13)
    axes[1][0].set_ylabel("Prediction", fontsize=13)
    fig.suptitle("Failure cases - missed bricks by cause", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.0)
    return _save(fig, out_path)


def _is_occluded(sample, inst_id: int) -> bool:
    """A brick counts as hidden when the arm's pixels surround more than a third of the brick's
    own outline -- a cheap, honest proxy for "the jaws are in the way"."""
    import cv2

    from .labels import ARM_CLASS

    mask = (sample.instance == inst_id).astype(np.uint8)
    grown = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    ring = grown & ~mask.astype(bool)
    if not ring.any():
        return False
    return float((sample.classes[ring] == ARM_CLASS).mean()) > 0.33


def error_vs_size_figure(report: dict, per_brick: list[dict],
                         out_path: Path | str = MEDIA_DIR / "error_vs_size.png") -> Path:
    """Recall and centre error against how many pixels the brick covers."""
    plt = _plt()
    rows = report.get("by_size", [])
    if not rows:
        raise ValueError("the report carries no size breakdown")
    labels = [_size_label(r["area_lo"], r["area_hi"]) for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), dpi=140)
    axes[0].bar(x, [r["recall"] for r in rows], color="#3b7dd8")
    for i, r in enumerate(rows):
        axes[0].annotate(f"n={r['n']}", (i, r["recall"]), ha="center", va="bottom", fontsize=11)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Recall @ IoU 0.5", fontsize=13)
    axes[0].set_title("Brick recall", fontsize=15)

    axes[1].bar(x, [r["centroid_mm"] for r in rows], color="#d8743b")
    axes[1].set_ylabel("Centre error at plate (mm)", fontsize=13)
    axes[1].set_title("Centre error", fontsize=15)

    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlabel("Brick area (% of frame)", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=11)

    axes[2].set_xlabel("Brick area (% of frame, log scale)", fontsize=13)
    axes[2].set_ylabel("Centre error at plate (mm)", fontsize=13)
    axes[2].set_title("Centre error per brick", fontsize=15)
    axes[2].grid(alpha=0.3)
    axes[2].tick_params(labelsize=11)
    found = [b for b in per_brick if b.get("found") and "centroid_mm" in b]
    if found:
        areas = np.array([b["area_px"] for b in found])
        errs = np.array([b["centroid_mm"] for b in found])
        kinds = [("all", np.ones(len(found), dtype=bool), "#6b7280"),
                 ("touching same colour", np.array([b["touching"] for b in found]), "#d84b4b"),
                 ("green on green plate", np.array([b["green"] for b in found]), "#2f9e44")]
        for label, sel, colour in kinds:
            if sel.any():
                axes[2].scatter(areas[sel] / 65536.0 * 100.0, errs[sel], s=14, alpha=0.55,
                                label=label, color=colour)
        axes[2].set_xscale("log")
        axes[2].legend(fontsize=11)
    fig.suptitle("Brick detection vs apparent size", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_path)


def _size_label(lo: int, hi: int) -> str:
    def pct(v):
        return v / 65536.0 * 100.0

    if hi >= 10 ** 9:
        return f"over {pct(lo):.0f}%"
    return f"{pct(lo):.1f} to {pct(hi):.0f}%"


def class_share_figure(report: dict, out_path: Path | str = MEDIA_DIR / "class_overlap.png") -> Path:
    """Per-class overlap as a bar chart, for the guide's summary."""
    plt = _plt()
    values = [report[f"iou_{name}"] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=140)
    ax.barh(list(CLASS_NAMES)[::-1], values[::-1], color="#3b7dd8")
    for i, v in enumerate(values[::-1]):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=12)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("IoU", fontsize=13)
    ax.set_title("Per-class IoU, 300 held-out frames", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def make_all(checkpoint=None, cache=None, split: str = "val", limit: int = 300,
             out_dir: Path | str = MEDIA_DIR) -> dict:
    """Run the evaluation and write every figure the guide page embeds."""
    from .data import DEFAULT_CACHE, load_samples
    from .evaluate import evaluation_report
    from .source import CnnSceneModel

    cache = cache or DEFAULT_CACHE
    report = evaluation_report(checkpoint=checkpoint, cache=cache, split=split, limit=limit)
    per_brick = list(np.load(Path("results/vision_model/per_brick.npy"), allow_pickle=True))
    samples = load_samples(cache, split, limit=min(limit, 60))
    source = CnnSceneModel(checkpoint=checkpoint)
    out_dir = Path(out_dir)
    paths = {
        "samples": samples_figure(samples, source, out_dir / "samples.png"),
        "failures": failures_figure(samples, source, out_dir / "failures.png"),
        "error_vs_size": error_vs_size_figure(report, per_brick, out_dir / "error_vs_size.png"),
        "class_overlap": class_share_figure(report, out_dir / "class_overlap.png"),
    }
    return {"report": report, "figures": {k: str(v) for k, v in paths.items()}}
