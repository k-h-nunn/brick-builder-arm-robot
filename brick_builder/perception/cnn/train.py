"""Training the convolutional scene model: targets, losses, and a time-budgeted loop.

Runs on a CPU by default and is meant to: the whole point of this model is that a laptop can
build and run it without a cluster. A GPU is used if one is asked for, nothing more.

The four targets, all built from the renderer's own labels
---------------------------------------------------------
``class``
    the four-way class map (`labels.class_map_from_instance`), trained with cross-entropy. The
    classes are very unbalanced -- roughly half the pixels are background and a tenth are bricks --
    so each class is weighted by the inverse square root of its share, measured once over the
    training split rather than guessed.

``centre``
    a Gaussian bump at each brick's centroid, with a width that follows the brick's own apparent
    size, trained with mean-squared error. This is how CenterNet and Panoptic-DeepLab train a
    centre map, and it is what makes a peak mean "one brick here".

``offset``
    for every brick pixel, the vector to its own brick's centroid, trained with L1 over brick
    pixels only (an offset at a non-brick pixel means nothing and is not supervised).

``depth``
    the renderer's metric depth, normalized per frame by its own median and median absolute
    deviation, trained with L1. Normalizing per frame is what makes the target scale-invariant:
    the network is never asked to guess how far away the table is in metres, only which parts of
    this frame are nearer than which, which is all the slot contract's ordinal depth uses.

Augmentation
------------
Horizontal and vertical flips (applied to image and labels together, before the targets are
built, so an offset can never disagree with the image it belongs to) and a small photometric
jitter: brightness, contrast and a per-channel gain kept small enough that a red brick cannot
become an orange one. Nothing is rotated by an arbitrary angle: that would resample the label
map and blur instance boundaries for no gain.

Budget and resuming
-------------------
`--minutes` is a wall-clock budget, not an epoch count. The loop checkpoints after every epoch and
stops when the budget runs out or the epochs do, whichever comes first; re-running with the same
`--out` picks the checkpoint back up, so a run interrupted at 40 minutes continues rather than
restarts.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DEFAULT_CACHE, read_shard, shard_paths
from .labels import CLASS_NAMES, FIRST_BRICK_INSTANCE, N_CLASSES, class_map_from_instance
from .model import OFFSET_SCALE, SceneModelNet, pack_weights, unpack_weights

DEFAULT_OUT = Path("results/vision_model")
WEIGHTS_DIR = Path("brick_builder/perception/weights")
WEIGHTS_NAME = "vision_model.pt"


@dataclass
class TrainConfig:
    """Everything a run is decided by. Saved into the checkpoint, so a checkpoint explains itself."""

    cache: str = str(DEFAULT_CACHE)
    out: str = str(DEFAULT_OUT)
    minutes: float = 120.0
    epochs: int = 200
    batch_size: int = 8
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    threads: int = 6
    crop: int = 160
    """Side of the random crop a training step is taken on (0 = the whole 256x256 window).

    The network is fully convolutional, so training on a crop and running on the whole window is
    consistent: the object scale, the lens and the labels are identical, there is simply less of
    the frame per step. It costs about 40 per cent of a full-window step, which is what makes a
    useful number of passes over the set fit in a wall-clock budget on a shared machine, and it
    is an honest augmentation in its own right -- a crop cuts objects at its edge, which is
    exactly the case the slot row's border fraction exists for."""

    steps_per_epoch: int = 220
    """Cap on optimizer steps between checkpoints. With a large set this makes an "epoch" a fixed
    slice of work rather than a full pass, so checkpoints, validation and the loss curve arrive at
    a predictable rate whatever the machine is doing."""

    device: str = "cpu"
    width: float = 1.0
    val_frames: int = 256
    w_class: float = 1.0
    w_centre: float = 120.0
    w_offset: float = 0.6
    w_depth: float = 0.4
    class_weights: tuple[float, ...] = ()
    """Filled in from the training split's own class frequencies when left empty."""

    extra_splits: tuple[str, ...] = ()
    """Extra splits mixed into the training set in memory, on top of `cache`/`train`. For adding
    targeted data (for example a fix aimed at one failure mode) without touching the shared
    `train` split on disk: generate the new frames into their own split name under `cache` and
    name it here. The held-out split is never affected by this field."""

    init_weights: str | None = None
    """A shipped weights snapshot (`_save_weights`'s format) to initialize the model from, on a
    fresh run only (`checkpoint.pt` still wins if `out` already has one, so re-running the same
    fine-tune command resumes it rather than restarting from this snapshot again)."""

    save_shipped: bool = True
    """Whether a new best-held-out epoch also overwrites the live
    `brick_builder/perception/weights/vision_model.pt` (the file `CnnSceneModel` loads by
    default), on top of always writing `out/best.pt`. Leave this on for an ordinary training run;
    turn it off for an experimental run (a speed probe, a fine-tune whose numbers have not been
    gated yet) so a script interrupted or still under evaluation can never leave the shipped
    checkpoint in an unvetted, partially-trained state -- shipping is then a deliberate, separate
    step: copy `out/best.pt` over the shipped file once the gate in the guide passes."""

    photometric_strength: float = 1.0
    """Scales the photometric jitter `_augment` applies, relative to the original fixed ranges
    (brightness +/-0.06, contrast 0.88-1.14, per-channel gain +/-0.10, pixel noise std 0.012):
    1.0 reproduces them exactly, 2.0 doubles each jitter's half-width. Widening this is the
    "photometric augmentation" half of a data-level fix for a lighting- or contrast-driven
    failure; a `class`/`centre`/`offset` target never depends on pixel values, so it is
    unaffected either way."""


# --------------------------------------------------------------------------------- targets ---

_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _grid(res: int) -> tuple[np.ndarray, np.ndarray]:
    if res not in _GRID_CACHE:
        ys, xs = np.indices((res, res))
        _GRID_CACHE[res] = (xs.astype(np.float32), ys.astype(np.float32))
    return _GRID_CACHE[res]


def make_targets(instance: np.ndarray, depth: np.ndarray) -> dict[str, np.ndarray]:
    """The four label arrays for one frame, from its instance map and metric depth."""
    res = int(instance.shape[0])
    xs, ys = _grid(res)
    flat = instance.ravel().astype(np.int64)
    n_bins = 256
    counts = np.bincount(flat, minlength=n_bins).astype(np.float64)
    sum_x = np.bincount(flat, weights=xs.ravel(), minlength=n_bins)
    sum_y = np.bincount(flat, weights=ys.ravel(), minlength=n_bins)

    centre_x = np.zeros(n_bins, dtype=np.float32)
    centre_y = np.zeros(n_bins, dtype=np.float32)
    live = counts > 0
    centre_x[live] = (sum_x[live] / counts[live]).astype(np.float32)
    centre_y[live] = (sum_y[live] / counts[live]).astype(np.float32)

    heat = np.zeros((res, res), dtype=np.float32)
    for inst_id in np.nonzero(live)[0]:
        if inst_id < FIRST_BRICK_INSTANCE:
            continue
        area = float(counts[inst_id])
        sigma = float(np.clip(math.sqrt(area) / 6.0, 1.8, 10.0))
        cx, cy = float(centre_x[inst_id]), float(centre_y[inst_id])
        r = int(math.ceil(3.0 * sigma))
        x0, x1 = max(0, int(cx) - r), min(res, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(res, int(cy) + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        gx = xs[y0:y1, x0:x1] - cx
        gy = ys[y0:y1, x0:x1] - cy
        bump = np.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma))
        np.maximum(heat[y0:y1, x0:x1], bump, out=heat[y0:y1, x0:x1])

    brick = instance >= FIRST_BRICK_INSTANCE
    offset = np.zeros((2, res, res), dtype=np.float32)
    offset[0] = (centre_x[instance] - xs) / OFFSET_SCALE
    offset[1] = (centre_y[instance] - ys) / OFFSET_SCALE
    offset *= brick[None]

    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d)
    if valid.any():
        med = float(np.median(d[valid]))
        mad = float(np.median(np.abs(d[valid] - med)))
        scale = max(1.4826 * mad, 1e-3)
        d_norm = np.where(valid, (d - med) / scale, 0.0).astype(np.float32)
    else:
        d_norm = np.zeros_like(d)

    return {
        "classes": class_map_from_instance(instance).astype(np.int64),
        "heat": heat,
        "offset": offset,
        "offset_mask": brick.astype(np.float32),
        "depth": np.clip(d_norm, -6.0, 6.0),
        "depth_mask": valid.astype(np.float32),
    }


# ---------------------------------------------------------------------------------- dataset ---


@dataclass
class FrameSet:
    """A whole split held in memory: about 0.4 MB per frame, so a few thousand fit comfortably."""

    rgb: np.ndarray        # uint8 [N, 256, 256, 3]
    instance: np.ndarray   # uint8 [N, 256, 256]
    depth: np.ndarray      # float16 [N, 256, 256]
    kinds: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.rgb.shape[0])


def load_split(cache: Path | str, split: str, limit: int | None = None) -> FrameSet:
    rgb, inst, depth, kinds = [], [], [], []
    for shard in shard_paths(cache, split):
        samples = read_shard(shard)
        for s in samples:
            rgb.append(s.rgb)
            inst.append(s.instance)
            depth.append(s.depth)
            kinds.append(str(s.meta.get("kind", "?")))
        if limit is not None and len(rgb) >= limit:
            break
    if not rgb:
        raise FileNotFoundError(
            f"no shards under {Path(cache) / split} -- generate the dataset first "
            "(`python scripts/train_vision_model.py --generate`)"
        )
    n = len(rgb) if limit is None else min(limit, len(rgb))
    return FrameSet(np.stack(rgb[:n]), np.stack(inst[:n]), np.stack(depth[:n]), kinds[:n])


def class_frequencies(frames: FrameSet, sample: int = 400) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for i in idx:
        counts += np.bincount(class_map_from_instance(frames.instance[i]).ravel(), minlength=N_CLASSES)
    return counts / counts.sum()


def _augment(rgb: np.ndarray, instance: np.ndarray, depth: np.ndarray, rng: np.random.Generator,
             crop: int = 0, strength: float = 1.0):
    """Crop and flips first (image and labels together), then a photometric jitter on the image.

    `strength` scales the jitter's half-widths around the original fixed ranges (see
    `TrainConfig.photometric_strength`); 1.0 reproduces brightness +/-0.06, contrast 0.88-1.14,
    gain +/-0.10, noise std 0.012 exactly.
    """
    if crop and crop < rgb.shape[0]:
        res = rgb.shape[0]
        r0 = int(rng.integers(0, res - crop + 1))
        c0 = int(rng.integers(0, res - crop + 1))
        sl = (slice(r0, r0 + crop), slice(c0, c0 + crop))
        rgb, instance, depth = rgb[sl], instance[sl], depth[sl]
    if rng.random() < 0.5:
        rgb, instance, depth = rgb[:, ::-1], instance[:, ::-1], depth[:, ::-1]
    if rng.random() < 0.5:
        rgb, instance, depth = rgb[::-1], instance[::-1], depth[::-1]
    img = rgb.astype(np.float32) / 255.0
    gain_half = 0.10 * strength
    gain = 1.0 + rng.uniform(-gain_half, gain_half, size=3).astype(np.float32)  # small: never a hue swap
    brightness = float(rng.uniform(-0.06 * strength, 0.06 * strength))
    contrast = float(rng.uniform(1.0 - 0.12 * strength, 1.0 + 0.14 * strength))
    img = (img - 0.5) * contrast + 0.5 + brightness
    img = np.clip(img * gain, 0.0, 1.0)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0.0, 0.012 * strength, size=img.shape).astype(np.float32),
                      0.0, 1.0)
    return img, np.ascontiguousarray(instance), np.ascontiguousarray(depth)


def make_batch(frames: FrameSet, idx: np.ndarray, rng: np.random.Generator | None,
               crop: int = 0, photometric_strength: float = 1.0):
    """One training (or validation, with `rng=None`) batch as tensors.

    Validation never crops: it is measured on the whole 256x256 window, which is what the model
    is handed at run time.
    """
    images, targets = [], []
    for i in idx:
        rgb, instance, depth = frames.rgb[i], frames.instance[i], frames.depth[i]
        if rng is None:
            img = rgb.astype(np.float32) / 255.0
        else:
            img, instance, depth = _augment(rgb, instance, depth, rng, crop, photometric_strength)
        images.append(img.transpose(2, 0, 1))
        targets.append(make_targets(instance, depth))
    batch = {
        "image": torch.from_numpy(np.stack(images)),
        "classes": torch.from_numpy(np.stack([t["classes"] for t in targets])),
        "heat": torch.from_numpy(np.stack([t["heat"] for t in targets]))[:, None],
        "offset": torch.from_numpy(np.stack([t["offset"] for t in targets])),
        "offset_mask": torch.from_numpy(np.stack([t["offset_mask"] for t in targets]))[:, None],
        "depth": torch.from_numpy(np.stack([t["depth"] for t in targets]))[:, None],
        "depth_mask": torch.from_numpy(np.stack([t["depth_mask"] for t in targets]))[:, None],
    }
    return batch


# ----------------------------------------------------------------------------------- losses ---


def compute_loss(out, batch: dict, cfg: TrainConfig, class_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    ce = F.cross_entropy(out.class_logits, batch["classes"], weight=class_weight)
    heat = torch.sigmoid(out.centre_logits)
    centre = F.mse_loss(heat, batch["heat"])
    om = batch["offset_mask"]
    denom = om.sum().clamp_min(1.0) * 2.0
    offset = (torch.abs(out.offset - batch["offset"]) * om).sum() / denom
    dm = batch["depth_mask"]
    depth = (torch.abs(out.depth - batch["depth"]) * dm).sum() / dm.sum().clamp_min(1.0)
    total = cfg.w_class * ce + cfg.w_centre * centre + cfg.w_offset * offset + cfg.w_depth * depth
    return total, {
        "loss": float(total.detach()), "ce": float(ce.detach()), "centre": float(centre.detach()),
        "offset": float(offset.detach()), "depth": float(depth.detach()),
    }


@torch.no_grad()
def evaluate_split(model, frames: FrameSet, cfg: TrainConfig, class_weight: torch.Tensor,
                   limit: int, seed: int = 0) -> dict:
    """Validation loss plus the per-class intersection-over-union the curve is read from."""
    model.eval()
    n = min(limit, len(frames))
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, n, cfg.batch_size):
        chunk = idx[start:start + cfg.batch_size]
        batch = make_batch(frames, chunk, rng=None)
        out = model(batch["image"])
        _, parts = compute_loss(out, batch, cfg, class_weight)
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        batches += 1
        pred = out.class_logits.argmax(dim=1).numpy()
        true = batch["classes"].numpy()
        for c in range(N_CLASSES):
            p, t = pred == c, true == c
            inter[c] += float((p & t).sum())
            union[c] += float((p | t).sum())
    model.train()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    out_dict = {f"val_{k}": v / max(batches, 1) for k, v in totals.items()}
    for c, name in enumerate(CLASS_NAMES):
        out_dict[f"val_iou_{name}"] = float(iou[c])
    out_dict["val_miou"] = float(np.nanmean(iou))
    return out_dict


# ------------------------------------------------------------------------------- the loop ---


def train(cfg: TrainConfig, verbose: bool = True) -> dict:
    """Run (or resume) one training job and return its final summary."""
    torch.set_num_threads(int(cfg.threads))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    csv_path = out_dir / "training_log.csv"

    train_frames = load_split(cfg.cache, "train")
    for extra in cfg.extra_splits:
        more = load_split(cfg.cache, extra)
        train_frames = FrameSet(
            rgb=np.concatenate([train_frames.rgb, more.rgb]),
            instance=np.concatenate([train_frames.instance, more.instance]),
            depth=np.concatenate([train_frames.depth, more.depth]),
            kinds=train_frames.kinds + more.kinds,
        )
        if verbose:
            print(f"extra split {extra!r}: +{len(more)} frames")
    val_frames = load_split(cfg.cache, "val")
    if verbose:
        print(f"train frames {len(train_frames)}  val frames {len(val_frames)}")

    if not cfg.class_weights:
        freq = class_frequencies(train_frames)
        weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
        weights = weights / weights.mean()
        cfg = TrainConfig(**{**asdict(cfg), "class_weights": tuple(float(w) for w in weights)})
        if verbose:
            print("class share  " + "  ".join(f"{n}={f:.3f}" for n, f in zip(CLASS_NAMES, freq)))
            print("class weight " + "  ".join(f"{n}={w:.2f}" for n, w in zip(CLASS_NAMES, weights)))
    class_weight = torch.tensor(cfg.class_weights, dtype=torch.float32)

    model = SceneModelNet(width=cfg.width).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict] = []
    start_epoch, elapsed_before, best = 0, 0.0, float("inf")

    if cfg.init_weights and not ckpt_path.exists():
        # A fine-tune from a shipped snapshot, not a from-scratch run: the optimizer starts
        # fresh (this is a new `out`, so there is no `checkpoint.pt` to resume it from either),
        # only the weights carry over.
        init_state = torch.load(cfg.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(unpack_weights(init_state["model"]))
        if verbose:
            print(f"initialized from {cfg.init_weights}")

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        elapsed_before = float(state.get("elapsed_s", 0.0))
        history = list(state.get("history", []))
        best = float(state.get("best_val", float("inf")))
        if verbose:
            print(f"resumed from {ckpt_path} at epoch {start_epoch} "
                  f"({elapsed_before / 60:.1f} min already spent)")

    budget_s = cfg.minutes * 60.0
    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed + 1)
    steps_per_epoch = max(1, min(cfg.steps_per_epoch, len(train_frames) // cfg.batch_size))
    model.train()
    stop_reason = "epochs"

    for epoch in range(start_epoch, cfg.epochs):
        spent = elapsed_before + (time.perf_counter() - t_start)
        if spent >= budget_s:
            stop_reason = "time budget"
            break
        # One cosine cycle over whatever the budget allows, estimated from the epochs done so far.
        progress = min(1.0, spent / budget_s)
        lr = cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in opt.param_groups:
            group["lr"] = lr

        order = rng.permutation(len(train_frames))[:steps_per_epoch * cfg.batch_size]
        running: dict[str, float] = {}
        t_epoch = time.perf_counter()
        for step in range(steps_per_epoch):
            chunk = order[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            batch = make_batch(train_frames, chunk, rng, cfg.crop, cfg.photometric_strength)
            out = model(batch["image"])
            loss, parts = compute_loss(out, batch, cfg, class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            if elapsed_before + (time.perf_counter() - t_start) >= budget_s:
                stop_reason = "time budget"
                break

        row = {"epoch": epoch, "lr": lr, "seconds": round(time.perf_counter() - t_epoch, 1)}
        row.update({k: v / max(step + 1, 1) for k, v in running.items()})
        row.update(evaluate_split(model, val_frames, cfg, class_weight, cfg.val_frames))
        row["elapsed_min"] = round((elapsed_before + time.perf_counter() - t_start) / 60.0, 2)
        history.append(row)
        if verbose:
            print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  val {row['val_loss']:.4f}  "
                  f"mIoU {row['val_miou']:.3f}  brick IoU {row['val_iou_brick']:.3f}  "
                  f"{row['elapsed_min']:.1f} min", flush=True)

        if row["val_loss"] < best:
            best = row["val_loss"]
            _save_weights(model, cfg, row, out_dir / "best.pt")
            if cfg.save_shipped:
                _save_weights(model, cfg, row, WEIGHTS_DIR / WEIGHTS_NAME)
        torch.save({
            "model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
            "elapsed_s": elapsed_before + (time.perf_counter() - t_start), "history": history,
            "best_val": best, "config": asdict(cfg),
        }, ckpt_path)
        _write_csv(csv_path, history)
        if stop_reason == "time budget":
            break

    _write_csv(csv_path, history)
    plot_curves(history, out_dir / "loss_curves.png")
    summary = {
        "epochs_done": len(history), "stop_reason": stop_reason,
        "minutes_used": round((elapsed_before + time.perf_counter() - t_start) / 60.0, 2),
        "best_val_loss": best,
        "final": history[-1] if history else {},
        "parameters": model.n_parameters(),
        "weights": str(WEIGHTS_DIR / WEIGHTS_NAME),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _save_weights(model, cfg: TrainConfig, row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": pack_weights(model.state_dict()),
        "width": cfg.width,
        "class_names": list(CLASS_NAMES),
        "offset_scale": OFFSET_SCALE,
        "val": {k: v for k, v in row.items() if k.startswith("val_")},
    }, path)


def _write_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = sorted({k for row in history for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_curves(history: list[dict], out_path: Path) -> Path | None:
    """Training and validation loss, the four loss terms, and the per-class overlap."""
    if not history:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=130)

    axes[0].plot(epochs, [r["loss"] for r in history], label="train", linewidth=2)
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="held-out", linewidth=2)
    axes[0].set_title("Total loss", fontsize=14)
    axes[0].legend(fontsize=12)

    for key, label in (("ce", "class CE"), ("centre", "centre MSE"), ("offset", "offset L1"),
                       ("depth", "depth L1")):
        axes[1].plot(epochs, [r[key] for r in history], label=label, linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_title("Train loss by term", fontsize=14)
    axes[1].legend(fontsize=11)

    for name in CLASS_NAMES:
        axes[2].plot(epochs, [r[f"val_iou_{name}"] for r in history], label=name, linewidth=2)
    axes[2].plot(epochs, [r["val_miou"] for r in history], "k--", label="mIoU", linewidth=2)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Held-out IoU", fontsize=14)
    axes[2].legend(fontsize=11, loc="lower right")

    for ax in axes:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
