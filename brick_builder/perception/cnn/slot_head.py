"""A differentiable pixel-to-slot head: the four per-pixel maps to `K x W` slot rows, in torch.

`postprocess.py` turns the same four maps into a `SceneState` by hard decisions -- an argmax class
map, an argmin over predicted centres, boolean masks, a median depth. Those decisions are what the
shipped scene model publishes and they are not differentiable, so nothing downstream of them can
send a gradient back into the network.

This module does the same arithmetic with soft assignments instead, so the whole path

    pixels -> class/centre/offset/depth maps -> slot rows -> the learned inverse `G`

carries a gradient. Every step mirrors the hard one:

| Hard (`postprocess.py`) | Soft (here) |
|---|---|
| `classes = argmax(class_logits)` | `p = softmax(class_logits)`, used as a per-pixel weight |
| brick pixel joins its nearest predicted centre | `softmax` over `-d^2 / (2 tau^2) + log p_centre(peak)` |
| boolean mask, then `_blob_row` | the same statistics, weighted by the soft mask |
| whatever no object claimed is background | `clamp(1 - sum of the kept soft masks, 0, 1)` |
| `rank_scale` on per-slot median depth | the same ranks through a logistic comparison |

The peaks themselves are still chosen by the hard non-maximum suppression of `postprocess._peaks`,
so the head reports the same objects the shipped model reports; a peak's own strength enters the
grouping as a logit bias, which is the centre head's gradient path. Object selection (which peaks
survive, which soft masses clear `min_area_px`) stays discrete -- a differentiable relaxation of
*how many objects there are* is a different problem and is not what the controller needs.

`tau` is the one knob. At `tau -> 0` the grouping softmax becomes the argmin and the head
reproduces `postprocess.slots_from_masks` exactly; at the shipped value it is soft over about a
pixel of vote distance, which is what `parity` below measures.

Identity across frames comes from the same `scene_model.TemporalSlotAssigner` the shipped source
uses, run on the detached rows, so a head swapped in under `CnnSceneModel` hands out the same
slots. Nothing here reads the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState, TemporalSlotAssigner
from .labels import ARM_CLASS, BASEPLATE_CLASS, BRICK_CLASS
from .model import OFFSET_SCALE, HeadOutputs
from .postprocess import GroupingConfig, _peaks

__all__ = ["SlotHeadConfig", "SoftSlots", "SlotHead", "soft_masks", "soft_blob_rows", "parity"]

FIELDS = PLACEHOLDER_SLOT_SCHEMA.fields
ROW_W = PLACEHOLDER_SLOT_SCHEMA.W


@dataclass(frozen=True)
class SlotHeadConfig:
    """The soft head's own parameters, on top of the grouping the shipped model already has."""

    tau_px: float = 1.0
    """Width of the grouping softmax, in pixels of vote distance. Small enough that a pixel whose
    vote lands two pixels from one peak and ten from another is assigned to the first with weight
    1 - 4e-9, so the soft masks are hard almost everywhere and differ from the argmin only in the
    thin band between two bricks -- which is exactly the band a gradient should be able to move."""

    tau_depth: float = 0.01
    """Width of the logistic that compares two slots' depths, in the depth head's own normalized
    units. The depth map is normalized per frame to roughly unit range, so 0.01 orders two slots
    a hundredth of that range apart with weight 0.73 and anything further apart essentially hard."""

    class_gate: str = "straight_through"
    """How a pixel's class decides whether it belongs to an object.

    `"soft"` weights every pixel by its class probability. That is the plainest soft assignment
    and it does not reproduce the hard rows: the class head gives a table pixel a small but
    non-zero brick probability, and summed over sixty thousand pixels a small number is not
    small -- measured, it moves a blob's centre by three pixels and its mean colour by 0.05.

    `"straight_through"` keeps the argmax mask in the forward pass and the probability's gradient
    in the backward one (Bengio et al., 2013), so the rows *are* the hard rows and the class head
    still receives a gradient. It is the default because the head's job is to differentiate the
    published representation, not a smoothed relative of it. The grouping between bricks stays
    genuinely soft either way -- that is where two objects compete and where a gradient has
    something to move."""

    eps: float = 1e-6
    """Floor on a soft mask's own total weight, so an empty object does not divide by zero."""


@dataclass
class SoftSlots:
    """One frame's slot state, differentiable, plus the bookkeeping `postprocess` also returns."""

    slots: torch.Tensor
    """`[K, W]` float32, `PLACEHOLDER_SLOT_SCHEMA` rows. Carries a gradient to the four maps."""
    depth: torch.Tensor
    """`[K]` float32, the ordinal depth field, soft-ranked. Carries a gradient to the depth map."""
    masks: torch.Tensor
    """`[n, H, W]` float32, the soft mask per candidate object, in the order they were built."""
    slot_of: np.ndarray
    """`[n]` int64: which slot each candidate landed in (`-1` = over the `K - 1` cap)."""
    background_slot: int
    kinds: list[str]

    def state(self, *, window, source: str = "conv_net_soft", t: float = 0.0,
              frame_id: int = 0) -> SceneState:
        """The same rows as a plain `SceneState`, detached -- for a consumer that wants no torch."""
        return SceneState(
            slots=self.slots.detach().numpy(), depth=self.depth.detach().numpy(),
            schema=PLACEHOLDER_SLOT_SCHEMA, t=float(t), source=source, window=window,
            slot_ids=np.where(np.arange(self.slots.shape[0]) == self.background_slot, -1,
                              np.arange(self.slots.shape[0])).astype(np.int64),
            frame_id=int(frame_id),
        )


# ------------------------------------------------------------------------------- the soft masks ---


def soft_masks(out: HeadOutputs, cfg: SlotHeadConfig = SlotHeadConfig(),
               grouping: GroupingConfig = GroupingConfig()) -> tuple[torch.Tensor, list[str]]:
    """`HeadOutputs` for a batch of one -> `([n, H, W]` soft masks, one kind per mask).

    The order is the shipped decoder's: the arm, the baseplate, then one mask per surviving centre
    peak. A candidate whose soft area is under `min_area_px` is dropped, as it is there.
    """
    prob = F.softmax(out.class_logits[0], dim=0)            # [4, H, W]
    centre = torch.sigmoid(out.centre_logits[0, 0])          # [H, W]
    offset = out.offset[0] * OFFSET_SCALE                    # [2, H, W], pixels
    h, w = centre.shape
    winner = prob.detach().argmax(dim=0)

    def gate(cls: int) -> torch.Tensor:
        p = prob[cls]
        if cfg.class_gate == "soft":
            return p
        hard = (winner == cls).to(p.dtype)
        return hard + p - p.detach()

    masks: list[torch.Tensor] = []
    kinds: list[str] = []
    for cls, kind in ((ARM_CLASS, "arm"), (BASEPLATE_CLASS, "baseplate")):
        m = gate(cls)
        if float(m.detach().sum()) >= grouping.min_area_px:
            masks.append(m)
            kinds.append(kind)

    brick = gate(BRICK_CLASS)
    if float(brick.detach().sum()) < grouping.min_area_px:
        return _stack(masks, h, w), kinds

    peaks = _peaks(centre.detach().numpy(), grouping)
    if peaks.shape[0] == 0:
        # No centre survived: one object, exactly as the hard decoder falls back to.
        masks.append(brick)
        kinds.append("brick")
        return _stack(masks, h, w), kinds

    rows = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    cols = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    vote_x = cols + offset[0]
    vote_y = rows + offset[1]

    px = torch.as_tensor(peaks[:, 0], dtype=torch.float32).view(-1, 1, 1)
    py = torch.as_tensor(peaks[:, 1], dtype=torch.float32).view(-1, 1, 1)
    d2 = (vote_x[None] - px) ** 2 + (vote_y[None] - py) ** 2           # [m, H, W]

    # The peak's own strength biases the competition: two peaks a pixel apart are resolved towards
    # the stronger one, which is the only place the centre head sees a gradient from this head.
    peak_p = centre[torch.as_tensor(peaks[:, 1], dtype=torch.long),
                    torch.as_tensor(peaks[:, 0], dtype=torch.long)].clamp_min(1e-6)
    logits = -d2 / (2.0 * cfg.tau_px ** 2) + torch.log(peak_p).view(-1, 1, 1)
    share = torch.softmax(logits, dim=0)                                # [m, H, W]

    for k in range(peaks.shape[0]):
        m = brick * share[k]
        if float(m.detach().sum()) < grouping.min_area_px:
            continue
        masks.append(m)
        kinds.append("brick")
    return _stack(masks, h, w), kinds


def _stack(masks: list[torch.Tensor], h: int, w: int) -> torch.Tensor:
    return torch.stack(masks) if masks else torch.zeros((0, h, w))


# -------------------------------------------------------------------------- the blob statistics ---


def soft_blob_rows(masks: torch.Tensor, frame01: torch.Tensor,
                   cfg: SlotHeadConfig = SlotHeadConfig()) -> torch.Tensor:
    """`[n, H, W]` soft masks and a `[3, H, W]` frame -> `[n, W]` schema rows, `age` left at zero.

    Every field is the weighted form of what `ideal_scene_model._blob_row` computes with a boolean
    mask: a sum over pixels weighted by the mask, divided by the mask's own total weight. With a
    0/1 mask the two agree exactly.
    """
    n, h, w = masks.shape
    res = h
    if masks.numel() == 0:
        return torch.zeros((0, ROW_W))

    lin = torch.linspace(-1.0, 1.0, res)
    x = lin.view(1, res).expand(res, res)
    y = lin.view(res, 1).expand(res, res)

    total = masks.sum(dim=(1, 2)).clamp_min(cfg.eps)                    # [n]

    def wmean(field: torch.Tensor) -> torch.Tensor:
        """`[H, W]` -> `[n]`, the mask-weighted mean of a per-pixel field."""
        return (masks * field[None]).sum(dim=(1, 2)) / total

    mx, my = wmean(x), wmean(y)
    dx = x[None] - mx.view(-1, 1, 1)
    dy = y[None] - my.view(-1, 1, 1)

    def wmean_n(field: torch.Tensor) -> torch.Tensor:
        """As `wmean`, for a field that already has the object axis."""
        return (masks * field).sum(dim=(1, 2)) / total

    sxx, sxy, syy = wmean_n(dx * dx), wmean_n(dx * dy), wmean_n(dy * dy)

    colour = 2.0 * frame01 - 1.0                                        # [3, H, W]
    mean_colour = torch.stack([wmean(colour[c]) for c in range(3)], dim=1)   # [n, 3]
    dev = colour[None] - mean_colour.view(n, 3, 1, 1)
    colour_spread = (masks[:, None] * dev ** 2).sum(dim=(1, 2, 3)) / (3.0 * total)

    border = torch.zeros((res, res))
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = 1.0
    border_frac = wmean(border)

    # The fourth-order complex moment, `fourth_moment`'s formula with the same weights.
    r2 = wmean_n(dx * dx + dy * dy).clamp_min(cfg.eps)
    a2 = dx * dx - dy * dy
    b2 = 2.0 * dx * dy
    re4 = a2 * a2 - b2 * b2
    im4 = 2.0 * a2 * b2
    c4_re = wmean_n(re4) / (r2 * r2)
    c4_im = wmean_n(im4) / (r2 * r2)

    out = [
        (FIELDS["mass"], (total / float(res * res)).view(n, 1)),
        (FIELDS["centre"], torch.stack([mx, my], dim=1)),
        (FIELDS["moments"], torch.stack([sxx, sxy, syy], dim=1)),
        (FIELDS["colour"], mean_colour),
        (FIELDS["colour_spread"], colour_spread.view(n, 1)),
        (FIELDS["border"], border_frac.view(n, 1)),
        (FIELDS["age"], torch.zeros((n, 1))),
        (FIELDS["orient4"], torch.stack([c4_re, c4_im], dim=1)),
    ]
    return torch.cat([value for _, value in sorted(out, key=lambda kv: kv[0].start)], dim=1)


def _weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The weighted median of `values` under `weights`, differentiable in the value it picks.

    The hard path takes `np.median(depth[mask])`; with a 0/1 mask this is the same number, and
    with a soft one it is the depth at which half the mask's weight lies on either side. The
    *choice* of which pixel that is is discrete -- as a median always is -- and the gradient goes
    to that pixel's own depth.
    """
    finite = torch.isfinite(values)
    v = torch.where(finite, values, torch.zeros_like(values))
    wt = weights * finite
    if float(wt.detach().sum()) <= 0.0:
        return torch.zeros((), dtype=values.dtype)
    order = torch.argsort(v.detach())
    cum = torch.cumsum(wt[order], dim=0)
    half = 0.5 * cum[-1]
    idx = int(torch.searchsorted(cum.detach(), half.detach().reshape(1)).item())
    return v[order[min(idx, v.numel() - 1)]]


def _soft_rank(depths: torch.Tensor, present: torch.Tensor, tau: float) -> torch.Tensor:
    """`scene_model.rank_scale` with the step function replaced by a logistic.

    Rank 0 is the nearest (smallest) present slot, rank 1 the furthest; an absent slot is exactly
    0.5, as every producer's depth field is. At `tau -> 0` this is `rank_scale` term for term.
    """
    K = depths.shape[0]
    out = torch.full((K,), 0.5)
    idx = torch.nonzero(present, as_tuple=False).reshape(-1)
    m = int(idx.numel())
    if m == 0:
        return out
    if m == 1:
        out = out.clone()
        out[idx] = torch.zeros((), dtype=depths.dtype) + 0.0 * depths[idx].sum()
        return out
    d = depths[idx]
    diff = (d.view(-1, 1) - d.view(1, -1)) / tau
    rank = (torch.sigmoid(diff).sum(dim=1) - 0.5) / float(m - 1)
    return out.clone().index_copy(0, idx, rank.clamp(0.0, 1.0))


# ------------------------------------------------------------------------------------- the head ---


class SlotHead(torch.nn.Module):
    """`HeadOutputs` + the frame -> `SoftSlots`. No parameters of its own.

    It is a module rather than a function so it can hold the slot assigner (one per episode, like
    every other producer's) and be dropped into a policy next to the network whose maps it reads.
    """

    def __init__(self, *, K: int = 8, vanish_after: int = 5,
                 cfg: SlotHeadConfig = SlotHeadConfig(),
                 grouping: GroupingConfig = GroupingConfig(), temporal: bool = True) -> None:
        super().__init__()
        if K < 2:
            raise ValueError(f"K must be at least 2 (1 background + >=1 object slot); got {K}")
        self.K = int(K)
        self.cfg = cfg
        self.grouping = grouping
        self.temporal = bool(temporal)
        """Hand slots out through the temporal assigner, so a slot means the same object from one
        frame to the next. `False` fills slots in the frame's own candidate order (the arm, the
        baseplate, then bricks by peak strength, background last) and keeps no state at all, which
        is what a policy trained on stored frames needs: a forward pass replayed in an update has
        to reproduce the one taken during the rollout, and a slot index that depends on the frames
        before it does not."""
        self.assigner = TemporalSlotAssigner(K=self.K, vanish_after=int(vanish_after))

    def reset(self) -> None:
        """New episode: forget slot identity, exactly as `CnnSceneModel.reset` does."""
        self.assigner.reset()

    def forward(self, out: HeadOutputs, frame01: torch.Tensor,
                depth_map: torch.Tensor | None = None, *, with_depth: bool = True) -> SoftSlots:
        """One frame. `frame01` is `[3, H, W]` in `[0, 1]`; `depth_map` defaults to the head's own.

        `depth_map` is separate so the same head can be scored against a source that supplies a
        metric depth map, which is what the parity check below does. `with_depth=False` skips the
        per-slot median and returns the empty-slot value 0.5 everywhere: the median is the one
        expensive step here (a sort per object over the whole window) and no consumer of the slot
        rows on the control path reads the depth field -- `models.features.ROW_FIELDS` does not
        include it.
        """
        masks, kinds = soft_masks(out, self.cfg, self.grouping)
        rows = soft_blob_rows(masks, frame01, self.cfg)
        depth = out.depth[0, 0] if depth_map is None else depth_map

        masses = masks.detach().sum(dim=(1, 2)).numpy().astype(np.float64)
        if self.temporal:
            slot_of = self.assigner.step(masses, rows.detach().numpy().astype(np.float64),
                                         external_ids=None)
            background_slot = int(self.assigner.background_slot)
        else:
            n = int(masks.shape[0])
            slot_of = np.where(np.arange(n) < self.K - 1, np.arange(n), -1).astype(np.int64)
            background_slot = min(n, self.K - 1)
        kept = slot_of >= 0

        claimed = (masks[torch.as_tensor(np.nonzero(kept)[0], dtype=torch.long)].sum(dim=0)
                   if kept.any() else torch.zeros(masks.shape[-2:]))
        background = (1.0 - claimed).clamp(0.0, 1.0)
        bg_row = soft_blob_rows(background[None], frame01, self.cfg)[0]

        slots = torch.zeros((self.K, ROW_W))
        age_col = FIELDS["age"].start
        index: list[int] = []
        values: list[torch.Tensor] = []
        for i in range(masks.shape[0]):
            if not kept[i]:
                continue
            k = int(slot_of[i])
            age = float(np.tanh(self.assigner.slot_age(k) / 10.0))
            row = rows[i].clone()
            row = torch.cat([row[:age_col], torch.full((1,), age), row[age_col + 1:]])
            index.append(k)
            values.append(row)
        bg_age = float(np.tanh(self.assigner.slot_age(background_slot) / 10.0))
        index.append(background_slot)
        values.append(torch.cat([bg_row[:age_col], torch.full((1,), bg_age),
                                 bg_row[age_col + 1:]]))
        slots = slots.index_copy(0, torch.as_tensor(index, dtype=torch.long),
                                 torch.stack(values))

        if not with_depth:
            return SoftSlots(slots=slots, depth=torch.full((self.K,), 0.5), masks=masks,
                             slot_of=np.asarray(slot_of, dtype=np.int64),
                             background_slot=background_slot, kinds=list(kinds))

        present = torch.zeros(self.K, dtype=torch.bool)
        depths = torch.zeros(self.K)
        flat_depth = depth.reshape(-1)
        d_index, d_values = [], []
        for i in range(masks.shape[0]):
            if not kept[i]:
                continue
            k = int(slot_of[i])
            value = _weighted_median(flat_depth, masks[i].reshape(-1))
            if float(masks[i].detach().sum()) > 0.0:
                present[k] = True
                d_index.append(k)
                d_values.append(value)
        bg_value = _weighted_median(flat_depth, background.reshape(-1))
        if float(background.detach().sum()) > 0.0:
            present[background_slot] = True
            d_index.append(background_slot)
            d_values.append(bg_value)
        if d_index:
            depths = depths.index_copy(0, torch.as_tensor(d_index, dtype=torch.long),
                                       torch.stack(d_values))

        return SoftSlots(slots=slots, depth=_soft_rank(depths, present, self.cfg.tau_depth),
                         masks=masks, slot_of=np.asarray(slot_of, dtype=np.int64),
                         background_slot=background_slot, kinds=list(kinds))


# ------------------------------------------------------------------------------------- parity ---


def parity(frames, checkpoint=None, *, cfg: SlotHeadConfig = SlotHeadConfig(),
           device: str = "cpu", grouping: GroupingConfig | None = None) -> dict:
    """Per-field error of the soft head against the hard post-processing it mirrors.

    `frames` is a sequence of `[H, W, 3]` images (`uint8` or float), or of anything with an `rgb`
    attribute. Both paths run on the same network output and the same frame.

    Two comparisons, because two different things can disagree.

    - **Per object.** Both decoders build their candidates from the same centre peaks in the same
      order, so mask `i` is the same object on both sides and the rows can be compared field for
      field. This is the head's own accuracy and is what "reproduces the hard rows" means.
    - **Per slot.** The whole assembled state, slot index for slot index, including the temporal
      assigner and the background slot. This one is only meaningful on a **sequence** of frames:
      slot identity is carried across frames, so feeding unrelated frames makes the two assigners
      diverge for reasons that have nothing to do with the head. `frames_are_a_sequence` in the
      result says which was measured.

    A frame where the two decoders kept a different number of candidates is counted rather than
    scored, since there is then no object-for-object correspondence to score.
    """
    import time

    from ..ideal_scene_model import _blob_row
    from .postprocess import decode_heads, instances_from_heads
    from .source import CnnSceneModel

    source = CnnSceneModel(checkpoint=checkpoint, device=device, build_decode=False)
    group = grouping or source.grouping
    head = SlotHead(K=source.K, vanish_after=source.vanish_after, cfg=cfg, grouping=group)

    names = list(PLACEHOLDER_SLOT_SCHEMA.names())
    obj_err = {name: [] for name in names}
    slot_err = {name: [] for name in names}
    depth_err: list[float] = []
    objects = slots_scored = mismatched = one_sided = 0
    t_hard = t_soft = 0.0
    images = [np.asarray(getattr(f, "rgb", f)) for f in frames]

    for image in images:
        frame01 = np.asarray(image, dtype=np.float32)
        if frame01.max(initial=0.0) > 1.0 + 1e-6:
            frame01 = frame01 / 255.0
        frame01 = np.ascontiguousarray(frame01[..., :3])
        res = int(frame01.shape[0])
        x = torch.from_numpy(np.ascontiguousarray(frame01.transpose(2, 0, 1)))[None]
        with torch.no_grad():
            out = source.net(x)

        t0 = time.perf_counter()
        pred = decode_heads(out, group)
        hard_masks, hard_kinds = instances_from_heads(pred, group)
        hard = source.observe(image, window=None)
        t_hard += time.perf_counter() - t0

        t0 = time.perf_counter()
        soft = head(out, x[0], depth_map=out.depth[0, 0])
        t_soft += time.perf_counter() - t0

        if len(hard_masks) == len(soft.kinds) and hard_kinds == soft.kinds:
            soft_rows = soft_blob_rows(soft.masks, x[0], cfg)
            for i, mask in enumerate(hard_masks):
                want = _blob_row(mask, frame01.astype(np.float64), res, amodal=False, age=0.0)
                got = soft_rows[i].detach().numpy()
                objects += 1
                for name in names:
                    sl = PLACEHOLDER_SLOT_SCHEMA.fields[name]
                    obj_err[name].append(float(np.abs(want[sl] - got[sl]).max()))
        else:
            mismatched += 1

        for k in range(source.K):
            a_empty, b_empty = hard.is_empty(k), bool(not torch.any(soft.slots[k]))
            if a_empty != b_empty:
                one_sided += 1
                continue
            if a_empty:
                continue
            slots_scored += 1
            for name in names:
                a = np.asarray(PLACEHOLDER_SLOT_SCHEMA.get(hard.slots[k], name), dtype=np.float64)
                b = PLACEHOLDER_SLOT_SCHEMA.get(soft.slots[k], name).detach().numpy()
                slot_err[name].append(float(np.abs(a - b).max()))
            depth_err.append(abs(float(hard.depth[k]) - float(soft.depth[k])))

    def summarize(err: dict) -> dict:
        return {
            "max_abs": {n: float(np.max(v)) if v else float("nan") for n, v in err.items()},
            "mean_abs": {n: float(np.mean(v)) if v else float("nan") for n, v in err.items()},
        }

    n = max(len(images), 1)
    return {
        "frames": len(images),
        "per_object": {"scored": objects, "frames_with_a_different_candidate_count": mismatched,
                       **summarize(obj_err)},
        "per_slot": {"scored": slots_scored, "one_sided": one_sided,
                     "depth_mean_abs": float(np.mean(depth_err)) if depth_err else float("nan"),
                     "depth_max_abs": float(np.max(depth_err)) if depth_err else float("nan"),
                     **summarize(slot_err)},
        "hard_ms_per_frame": 1e3 * t_hard / n,
        "soft_ms_per_frame": 1e3 * t_soft / n,
        "objects_per_frame": objects / n,
    }
