"""Query-set builders and slot statistics for a per-pixel scene-model decode.

A decode answers "which of the K slots does this pixel belong to" only at pixels it is asked
about; this module builds the pixel sets to ask at (`grid_queries`, `region_queries`) and turns a
batch of decode logits back into per-slot statistics (`slot_statistics`) -- mass, centroid,
spread, mean colour -- which is the raw material a token producer turns into `ObjectToken`s.

Everything here is in pixel units. Metric (millimetre) positions require the camera's known
pose and belong to a later package (`perception/backproject.py`); nothing here assumes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def grid_queries(step: int, res: int = 256) -> np.ndarray:
    """An evenly spaced (col, row) pixel query grid, `step` pixels apart, centred in the frame.

    Used for task-aligned readouts that need broad coverage (a baseplate stud grid, a coarse
    "is anything here" scan) rather than a dense per-pixel decode. Returns `[n, 2]` int64,
    `n = (res // step) ** 2` (approximately; the grid is centred so it may be one row/col short).
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if res <= 0:
        raise ValueError("res must be positive")
    coords = np.arange(step // 2, res, step, dtype=np.int64)
    cols, rows = np.meshgrid(coords, coords, indexing="xy")
    return np.stack([cols.ravel(), rows.ravel()], axis=-1)


def region_queries(mask: np.ndarray) -> np.ndarray:
    """Every (col, row) pixel coordinate where a boolean mask `[H, W]` is True.

    Used for a region-specific readout: the gripper's footprint (in-hand brick + its offset), a
    tray pocket (inventory), or any other region-of-interest a LOOK skill has already picked.
    Returns `[n, 2]` int64, `n` = the number of True pixels (0 if none).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W]; got shape {mask.shape}")
    rows, cols = np.nonzero(mask)
    return np.stack([cols, rows], axis=-1).astype(np.int64)


@dataclass
class ObjectToken:
    """One object, as the control code (skills, RL policy, FSM) actually consumes it.

    Positions and `size` are in pixel units here (see module docstring); a later back-projection
    step turns them into metres using the calibrated camera pose. `slot` records which of the K
    slots this token came from in the frame it was decoded from -- purely diagnostic, since slot
    order carries no meaning frame to frame.
    """

    x: float
    y: float
    yaw: float
    layer: int
    colour: str
    size: float
    confidence: float
    age: float
    slot: int


@dataclass
class SlotStats:
    """Per-slot statistics computed from a batch of decode logits and the pixels they were at.

    All arrays are `[B, K, ...]`. `mass` is the soft pixel count assigned to each slot (the
    column sums of the softmax over slots); a slot with `mass` near zero was not decoded at any
    query pixel and its `centroid`/`var`/`mean_colour` are `nan`. `var` is the (col, row)
    variance of the query pixels under the slot's soft assignment (a 2x2x2... simplification:
    the two axis variances plus their covariance, not a full second-moment tensor per PLAN's
    "second moments", since that is all `producers.py` needs).
    """

    mass: np.ndarray        # [B, K]
    centroid: np.ndarray    # [B, K, 2]  (col, row)
    var: np.ndarray         # [B, K, 2]  (var_col, var_row)
    covar: np.ndarray       # [B, K]     cov(col, row)
    mean_colour: np.ndarray  # [B, K, 3]  in [0, 1]


def slot_statistics(logits: torch.Tensor, px: torch.Tensor, frames: torch.Tensor) -> SlotStats:
    """Turn decode logits at a set of query pixels into per-slot mass/centroid/spread/colour.

    `logits` [B, n, K] (a decode's output), `px` [B, n, 2] (col, row) the same
    queries were decoded at, `frames` [B, H, W, 3] float in [0, 1] the source images (for the
    mean-colour readout). The soft assignment weight of query `i` to slot `j` is
    `softmax_j(logits[..., j])`; a slot's statistics are the weighted moments of the query pixels
    under that assignment.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    px = torch.as_tensor(px, dtype=torch.float32)
    frames = torch.as_tensor(frames, dtype=torch.float32)
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,n,K]; got {tuple(logits.shape)}")
    if px.shape[:2] != logits.shape[:2]:
        raise ValueError(f"px {tuple(px.shape)} and logits {tuple(logits.shape)} disagree on [B,n]")

    weights = torch.softmax(logits, dim=-1)  # [B,n,K]
    mass = weights.sum(dim=1)  # [B,K]
    mass_safe = mass.clamp_min(1e-8)

    centroid = torch.einsum("bnk,bnd->bkd", weights, px) / mass_safe.unsqueeze(-1)  # [B,K,2]
    diff = px.unsqueeze(2) - centroid.unsqueeze(1)  # [B,n,K,2]
    var = torch.einsum("bnk,bnkd->bkd", weights, diff * diff) / mass_safe.unsqueeze(-1)  # [B,K,2]
    covar = torch.einsum("bnk,bnk->bk", weights, diff[..., 0] * diff[..., 1]) / mass_safe  # [B,K]

    b, h, w, _ = frames.shape
    col = px[..., 0].round().long().clamp(0, w - 1)
    row = px[..., 1].round().long().clamp(0, h - 1)
    batch_idx = torch.arange(b).view(b, 1).expand_as(col)
    colour = frames[batch_idx, row, col, :]  # [B,n,3] in [0,1]
    mean_colour = torch.einsum("bnk,bnc->bkc", weights, colour) / mass_safe.unsqueeze(-1)  # [B,K,3]

    present = (mass > 1e-6).unsqueeze(-1)
    nan2 = torch.full_like(centroid, float("nan"))
    nan3 = torch.full_like(mean_colour, float("nan"))
    centroid = torch.where(present, centroid, nan2)
    var = torch.where(present, var, nan2)
    covar = torch.where(mass > 1e-6, covar, torch.full_like(covar, float("nan")))
    mean_colour = torch.where(present, mean_colour, nan3)

    return SlotStats(
        mass=mass.numpy(),
        centroid=centroid.numpy(),
        var=var.numpy(),
        covar=covar.numpy(),
        mean_colour=mean_colour.numpy(),
    )


def plot_decode_overlay(frame: np.ndarray, px: np.ndarray, logits: torch.Tensor, save_path: str | None = None):
    """The decode argmax overlay figure: query pixels coloured by their most likely slot.

    `frame` [H,W,3] in [0,1] or uint8, `px` [n,2] (col,row) the query pixels decoded, `logits`
    [n,K] (or [1,n,K]) the raw decode logits at those pixels. Returns a matplotlib `Figure`
    (Agg backend, no GUI window) with the frame as background and one scatter point per query,
    coloured by `argmax_k logits[..., k]`; saves to `save_path` when given. This is the figure
    function `scripts/make_media.py` calls for this package's contact sheet.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = np.asarray(frame)
    if frame.dtype != np.uint8 and frame.max() <= 1.0 + 1e-6:
        frame = (frame * 255).astype(np.uint8)
    logits = torch.as_tensor(logits)
    if logits.dim() == 3:
        logits = logits[0]
    px = np.asarray(px)
    assign = torch.argmax(logits, dim=-1).numpy()
    k = int(logits.shape[-1])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=128)
    ax.imshow(frame)
    scatter = ax.scatter(px[:, 0], px[:, 1], c=assign, cmap="tab10", vmin=0, vmax=max(k - 1, 1), s=10)
    ax.set_title("Decode argmax overlay")
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    fig.colorbar(scatter, ax=ax, label="slot", ticks=range(k))
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
