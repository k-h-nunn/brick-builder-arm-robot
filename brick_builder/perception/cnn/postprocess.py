"""From the network's four heads to a `SceneState`: grouping, then the ideal source's own code.

Two steps, and only the first one is new.

**Grouping.** The class head says which pixels are bricks; the centre head says where each brick's
centre is; the offset head says, for each brick pixel, where its own brick's centre is. Peaks of
the centre heat map (a 3x3 non-maximum suppression, a probability floor, and a cap on how many
survive) become the instance centres, and every brick pixel joins whichever centre its own offset
points nearest to. That is the bottom-up grouping of Panoptic-DeepLab: no proposals, no
region-of-interest pooling, no box-level non-maximum suppression, and -- the reason it is here --
no dependence on colour, so two identical bricks touching each other come apart.

The arm and the baseplate are not grouped: the renderer gives every arm link and both jaws one
instance id and the baseplate another, so the whole class mask *is* the instance, exactly as the
ideal source sees it.

**Slots.** From there on, nothing is reimplemented. `ideal_scene_model._blob_row` turns a mask
plus the frame into one `PLACEHOLDER_SLOT_SCHEMA` row, `scene_model.TemporalSlotAssigner` hands
out slots and keeps identity across frames (by Hungarian matching on the rows, since a vision
model has no simulator id), `ideal_scene_model._background_mask` collects whatever no kept object
claimed, and `scene_model.rank_scale` turns per-object depths into the ordinal `[0, 1]` field.

Those two `ideal_scene_model` helpers are private names. They are called rather than copied
because a second copy of the blob statistics is a second thing to keep in step with the schema --
see this package's guide page for the one-line refactor (renaming them, or better, lifting the
whole "masks plus a frame plus a depth map -> SceneState" block out of `IdealSceneModel.observe`
into a shared function) that would make the dependency a public one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..ideal_scene_model import _background_mask, _blob_row
from ..scene_model import (
    LOGIT_MAGNITUDE,
    PLACEHOLDER_SLOT_SCHEMA,
    SceneState,
    TemporalSlotAssigner,
    rank_scale,
)
from .labels import ARM_CLASS, BASEPLATE_CLASS, BRICK_CLASS
from .model import OFFSET_SCALE

__all__ = ["GroupingConfig", "Prediction", "SlotAssembly", "decode_heads", "instances_from_heads",
           "slots_from_masks"]


@dataclass(frozen=True)
class GroupingConfig:
    """The three numbers the grouping needs, and one cap.

    `centre_threshold` is a probability, so it means the same thing whatever the frame looks like.
    `min_area_px` throws away specks that would otherwise take a slot from a real object -- the
    smallest brick this camera ever sees spans a few hundred pixels at the working distance, so 60
    is well below anything real. `nms_kernel` is the neighbourhood a peak has to be the maximum of.
    """

    centre_threshold: float = 0.30
    min_area_px: int = 60
    nms_kernel: int = 7
    max_instances: int = 12


@dataclass
class Prediction:
    """One frame's decoded heads, as plain numpy."""

    classes: np.ndarray       # int64 [H, W], values in `labels.CLASS_NAMES`
    class_prob: np.ndarray    # float32 [4, H, W]
    centre: np.ndarray        # float32 [H, W], probability
    offset: np.ndarray        # float32 [2, H, W], pixels (already multiplied by OFFSET_SCALE)
    depth: np.ndarray         # float32 [H, W], normalized relative depth


def decode_heads(out, cfg: GroupingConfig = GroupingConfig()) -> Prediction:
    """`model.HeadOutputs` for a batch of one -> `Prediction` (numpy, no gradients)."""
    del cfg
    with torch.no_grad():
        prob = F.softmax(out.class_logits[0], dim=0)
        return Prediction(
            classes=prob.argmax(dim=0).numpy().astype(np.int64),
            class_prob=prob.numpy().astype(np.float32),
            centre=torch.sigmoid(out.centre_logits[0, 0]).numpy().astype(np.float32),
            offset=(out.offset[0] * OFFSET_SCALE).numpy().astype(np.float32),
            depth=out.depth[0, 0].numpy().astype(np.float32),
        )


def _peaks(centre: np.ndarray, cfg: GroupingConfig) -> np.ndarray:
    """Local maxima of the heat map above the threshold, `[n, 2]` as (col, row), strongest first."""
    t = torch.from_numpy(centre)[None, None]
    pooled = F.max_pool2d(t, kernel_size=cfg.nms_kernel, stride=1, padding=cfg.nms_kernel // 2)
    keep = (t == pooled) & (t >= cfg.centre_threshold)
    rows, cols = np.nonzero(keep[0, 0].numpy())
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    order = np.argsort(-centre[rows, cols])[: cfg.max_instances]
    return np.stack([cols[order], rows[order]], axis=-1).astype(np.float64)


def instances_from_heads(
    pred: Prediction, cfg: GroupingConfig = GroupingConfig()
) -> tuple[list[np.ndarray], list[str]]:
    """Boolean instance masks and a short label for each, largest-first within each kind.

    The order is: the arm (if any pixel is arm), the baseplate (if any), then one mask per brick.
    `TemporalSlotAssigner` re-sorts by mass anyway, so this order only decides tie-breaks.
    """
    masks: list[np.ndarray] = []
    kinds: list[str] = []

    arm = pred.classes == ARM_CLASS
    if int(arm.sum()) >= cfg.min_area_px:
        masks.append(arm)
        kinds.append("arm")
    plate = pred.classes == BASEPLATE_CLASS
    if int(plate.sum()) >= cfg.min_area_px:
        masks.append(plate)
        kinds.append("baseplate")

    brick = pred.classes == BRICK_CLASS
    n_brick = int(brick.sum())
    if n_brick < cfg.min_area_px:
        return masks, kinds

    peaks = _peaks(pred.centre, cfg)
    ys, xs = np.nonzero(brick)
    if peaks.shape[0] == 0:
        # No centre survived: report the brick pixels as one object rather than losing them.
        masks.append(brick)
        kinds.append("brick")
        return masks, kinds

    votes = np.stack([xs + pred.offset[0][ys, xs], ys + pred.offset[1][ys, xs]], axis=-1)
    d = np.linalg.norm(votes[:, None, :] - peaks[None, :, :], axis=-1)
    owner = np.argmin(d, axis=1)

    for k in range(peaks.shape[0]):
        sel = owner == k
        if int(sel.sum()) < cfg.min_area_px:
            continue
        m = np.zeros(pred.classes.shape, dtype=bool)
        m[ys[sel], xs[sel]] = True
        masks.append(m)
        kinds.append("brick")
    return masks, kinds


@dataclass
class SlotAssembly:
    """The `SceneState` plus the bookkeeping an evaluation (or a viewer) needs to say which
    predicted mask ended up in which slot."""

    state: SceneState
    slot_of: np.ndarray       # int64 [n_masks]; -1 = over the K-1 cap, folded into the background
    background_slot: int


def slots_from_masks(
    masks: list[np.ndarray],
    frame01: np.ndarray,
    depth_map: np.ndarray,
    assigner: TemporalSlotAssigner,
    *,
    K: int,
    window,
    source: str,
    t: float,
    frame_id: int,
    build_decode: bool = True,
) -> SlotAssembly:
    """Masks plus the frame they came from -> one `SceneState`.

    This mirrors, call for call, what `IdealSceneModel.observe` does once it has its ground-truth
    masks: one `_blob_row` per object, `TemporalSlotAssigner.step` to hand out slots and keep the
    `K - 1` largest, `_background_mask` for everything unclaimed, and `rank_scale` on the per-slot
    median depth. The only difference is where the masks and the depths came from.
    """
    res = int(frame01.shape[0])
    ids = list(range(len(masks)))
    masses = np.array([m.sum() for m in masks], dtype=np.float64)
    rows = np.zeros((len(masks), PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    for i, m in enumerate(masks):
        rows[i] = _blob_row(m, frame01, res, amodal=False, age=0.0)

    # Called even when nothing was found: a frame with no objects is still a frame, and the
    # assigner has to age its slots and pick a background slot for it, exactly as the ideal
    # source's unconditional call does.
    slot_of = assigner.step(masses, rows, external_ids=None)
    kept = slot_of >= 0
    background_slot = assigner.background_slot
    mask_by_id = {i: masks[i] for i in ids}
    background_pixels = _background_mask(mask_by_id, kept, ids, (res, res))

    slots = np.zeros((K, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    slot_ids = np.full(K, -1, dtype=np.int64)
    for i in ids:
        if not kept[i]:
            continue
        k = int(slot_of[i])
        row = rows[i].copy()
        row[PLACEHOLDER_SLOT_SCHEMA.fields["age"]] = float(np.tanh(assigner.slot_age(k) / 10.0))
        slots[k] = row
        slot_ids[k] = k  # no simulator id available: the slot index is the identity token
    bg_age = float(np.tanh(assigner.slot_age(background_slot) / 10.0))
    slots[background_slot] = _blob_row(background_pixels, frame01, res, amodal=False, age=bg_age)

    present = np.zeros(K, dtype=bool)
    median_depth = np.full(K, np.nan)
    for i in ids:
        if not kept[i]:
            continue
        k = int(slot_of[i])
        d = depth_map[masks[i]]
        d = d[np.isfinite(d)]
        if d.size:
            present[k] = True
            median_depth[k] = float(np.median(d))
    d_bg = depth_map[background_pixels]
    d_bg = d_bg[np.isfinite(d_bg)]
    if d_bg.size:
        present[background_slot] = True
        median_depth[background_slot] = float(np.median(d_bg))
    slot_depth = rank_scale(median_depth, present)

    decode_fn = None
    if build_decode:
        slot_map = np.full((res, res), background_slot, dtype=np.int64)
        for i in ids:
            if kept[i]:
                slot_map[masks[i]] = int(slot_of[i])

        def decode_fn(px: np.ndarray) -> np.ndarray:  # noqa: E306
            px = np.asarray(px, dtype=np.int64)
            n = px.shape[0]
            logits = np.full((n, K), -LOGIT_MAGNITUDE, dtype=np.float32)
            col, row = px[:, 0], px[:, 1]
            inside = (col >= 0) & (col < res) & (row >= 0) & (row < res)
            idx = np.full(n, background_slot, dtype=np.int64)
            idx[inside] = slot_map[row[inside], col[inside]]
            logits[np.arange(n), idx] = LOGIT_MAGNITUDE
            return logits

    state = SceneState(
        slots=slots, depth=slot_depth, schema=PLACEHOLDER_SLOT_SCHEMA, t=t, source=source,
        window=window, slot_ids=slot_ids, frame_id=frame_id, _decode_fn=decode_fn,
    )
    return SlotAssembly(state=state, slot_of=np.asarray(slot_of, dtype=np.int64),
                        background_slot=int(background_slot))
