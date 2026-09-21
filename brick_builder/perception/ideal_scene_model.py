"""`IdealSceneModel`: the simulator's default scene-model producer, built from ground truth.

The reference source: it publishes *exactly*
`scene_model.PLACEHOLDER_SLOT_SCHEMA` (mass, centre, second moments, colour, colour spread,
border fraction, age -- see `scene_model.py`'s own docstring for the full layout) so the rest of
the perception/control stack can be built and measured today, computed from `brick_builder.render.
renderer.Frame`'s instance segmentation and metric depth instead of a trained model's weights.

What counts as "a labelled object", concretely: every distinct instance id in the frame is a
candidate object, **except** `-1` (the renderer's own "no geom hit at all" background) and `0`
(`SceneRenderer.instance_lut`'s own default for a body with no explicit name-based override --
concretely, in `brick_builder.scene.compose`'s scenes, this is the table plane). Both are "unlabelled"
by the renderer's own convention and always fall into one shared background slot -- the "table,
anything unlabelled" the deliverable asks for. Every other id -- the baseplate, the gripper, every
brick -- is a candidate object competing for one of the remaining `K - 1` slots.

Identity across frames comes for free here: the simulator's own instance id *is* the stable
external id `scene_model.TemporalSlotAssigner` wants, so the same brick keeps the same slot for as
long as it is visible, with no matching needed (unlike `conventional_scene_model.
ColourSegmentationSceneModel`, which has no such id and has to match blobs frame to frame).

An **optional** per-pixel ground-truth label map is also built and exposed through `SceneState.
decode` -- viewer/evaluation only (see `readout.py`'s own banner), never part of the slot-state
contract a controller reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scene_model import (
    LOGIT_MAGNITUDE, PLACEHOLDER_SLOT_SCHEMA, SceneState, TemporalSlotAssigner, fourth_moment,
    rank_scale, to_window_xy,
)

BACKGROUND_IDS = frozenset({-1, 0})
"""Instance ids that are always "unlabelled" -- see the module docstring."""


@dataclass
class IdealSceneModel:
    """Builds a `SceneState` from ground-truth `instance`/`depth` arrays (see `observe`).

    The four degradation knobs below are **off by default** and exist only for later robustness
    studies (measuring how a downstream reader degrades as the ideal source is made to look more
    like a real, imperfect vision model): they are never enabled by the accuracy tests in
    `tests/perception/test_readout.py`.

    - `amodal`: report each object's statistics from the convex hull of its *visible* pixels
      rather than the visible pixels themselves -- an approximate upper bound on what a reader
      could achieve if occlusion did not fragment or truncate an object's own silhouette. This is
      an approximation (a convex hull can paper over a modest occluded notch, not recover a
      largely hidden object), not true unoccluded ground truth, which a single rendered instance
      map cannot provide by construction; used for debugging/comparison only, off by default.
    - `boundary_noise_px`: erode or dilate each object's mask by a random amount up to this many
      pixels (direction and amount drawn per object, per call) before computing its statistics.
    - `merge_same_colour_touching`: two objects of the same nearest palette colour whose masks
      touch (or would touch after a 1px dilation) are reported as a single slot -- modelling a
      real segmentation's structural limit that touching same-colour objects are hard to tell
      apart by appearance alone.
    - `min_object_px`: objects with fewer than this many ground-truth pixels are dropped entirely
      into the background slot instead of getting their own.
    - `label_noise_p`: each pixel is independently reassigned to a different object's group (or
      to background) with this probability, before statistics are computed.

    After each `observe` call, `last_object_count`/`last_kept_count`/`last_overflow_count` report
    how many distinct labelled objects were found, how many got their own slot, and how many were
    folded into the background slot for exceeding the `K - 1` cap.
    """

    K: int = 8
    vanish_after: int = 5
    amodal: bool = False
    boundary_noise_px: int = 0
    merge_same_colour_touching: bool = False
    min_object_px: int = 0
    label_noise_p: float = 0.0
    palette: dict[str, tuple[float, float, float]] | None = None
    build_decode: bool = True
    """Whether to also build the optional per-pixel ground-truth decode (`SceneState.decode`).
    On by default (cheap: an array lookup, no model to run) since it is what `readout.py` and the
    figures need; a caller that only wants slot states can set this `False` to skip building it."""

    assigner: TemporalSlotAssigner = field(init=False, repr=False)
    last_object_count: int = field(default=0, init=False, repr=False)
    last_kept_count: int = field(default=0, init=False, repr=False)
    last_overflow_count: int = field(default=0, init=False, repr=False)
    _frame_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.palette is None:
            from ..bricks.spec import PALETTE

            self.palette = PALETTE
        if self.K < 2:
            raise ValueError(f"K must be at least 2 (1 background + >=1 object slot); got {self.K}")
        self.assigner = TemporalSlotAssigner(K=self.K, vanish_after=self.vanish_after)

    def reset(self) -> None:
        """Clears slot identity (a new episode/scene -- old ids should not be "remembered")."""
        self.assigner.reset()

    def observe(
        self,
        rgb: np.ndarray,
        *,
        window,
        instance: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        present_mask: np.ndarray | None = None,
        t: float | None = None,
        seed: int | None = None,
    ) -> SceneState:
        if instance is None or depth is None:
            raise ValueError(
                "IdealSceneModel.observe needs both `instance` and `depth` -- it builds the "
                "scene state FROM the renderer's ground truth, unlike a real vision model which "
                "only ever sees `rgb`"
            )
        rgb_arr = np.asarray(rgb)
        instance = np.asarray(instance)
        depth = np.asarray(depth, dtype=np.float64)
        if instance.ndim != 2 or instance.shape[0] != instance.shape[1]:
            raise ValueError(f"instance must be a square [res,res] array; got shape {instance.shape}")
        res = int(instance.shape[0])
        if depth.shape != instance.shape:
            raise ValueError(f"instance {instance.shape} and depth {depth.shape} must match")
        if rgb_arr.shape[:2] != instance.shape:
            raise ValueError(f"rgb {rgb_arr.shape[:2]} and instance {instance.shape} must match")

        rng = np.random.default_rng(seed)
        frame = _as_float_rgb(rgb_arr)

        effective = instance.copy()
        if present_mask is not None:
            pm = np.asarray(present_mask, dtype=bool)
            if pm.shape != instance.shape:
                raise ValueError(f"present_mask {pm.shape} must match instance {instance.shape}")
            effective[~pm] = -1

        if self.label_noise_p > 0.0:
            ids_present = np.unique(effective)
            flip = rng.random(effective.shape) < self.label_noise_p
            if np.any(flip) and ids_present.size:
                effective = effective.copy()
                effective[flip] = rng.choice(ids_present, size=int(flip.sum()))

        ids, counts = np.unique(effective, return_counts=True)
        counts_by_id = {int(i): int(c) for i, c in zip(ids, counts)}
        object_ids = [i for i in counts_by_id if i not in BACKGROUND_IDS]
        if self.min_object_px > 0:
            object_ids = [i for i in object_ids if counts_by_id[i] >= self.min_object_px]

        merges: dict[int, int] = {}
        if self.merge_same_colour_touching and len(object_ids) > 1:
            merges = _merge_touching_same_colour(effective, object_ids, frame, self.palette)

        def _root(i: int) -> int:
            while i in merges:
                i = merges[i]
            return i

        groups: dict[int, list[int]] = {}
        for i in object_ids:
            groups.setdefault(_root(i), []).append(i)
        self.last_object_count = len(groups)

        group_ids = list(groups)
        masks = {g: np.isin(effective, groups[g]) for g in group_ids}
        if self.boundary_noise_px > 0:
            masks = _jitter_masks(masks, self.boundary_noise_px, rng)

        masses_full = np.array([masks[g].sum() for g in group_ids], dtype=np.float64)
        rows_full = np.zeros((len(group_ids), PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
        for i, g in enumerate(group_ids):
            rows_full[i] = _blob_row(masks[g], frame, res, amodal=self.amodal, age=0.0)

        slot_of = self.assigner.step(masses_full, rows_full, external_ids=group_ids)
        kept_mask = slot_of >= 0
        self.last_kept_count = int(kept_mask.sum())
        self.last_overflow_count = int((~kept_mask).sum())

        background_slot = self.assigner.background_slot
        background_pixels = _background_mask(masks, kept_mask, group_ids, effective.shape)

        slots = np.zeros((self.K, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
        slot_ids = np.full(self.K, -1, dtype=np.int64)
        for i, g in enumerate(group_ids):
            if not kept_mask[i]:
                continue
            k = int(slot_of[i])
            age_val = float(np.tanh(self.assigner.slot_age(k) / 10.0))
            row = rows_full[i].copy()
            row[PLACEHOLDER_SLOT_SCHEMA.fields["age"]] = age_val
            slots[k] = row
            slot_ids[k] = g

        bg_age = float(np.tanh(self.assigner.slot_age(background_slot) / 10.0))
        slots[background_slot] = _blob_row(background_pixels, frame, res, amodal=False, age=bg_age)
        slot_ids[background_slot] = -1

        present = np.zeros(self.K, dtype=bool)
        median_depth = np.full(self.K, np.nan)
        for i, g in enumerate(group_ids):
            if not kept_mask[i]:
                continue
            k = int(slot_of[i])
            d = depth[masks[g]]
            d = d[np.isfinite(d)]
            if d.size:
                present[k] = True
                median_depth[k] = float(np.median(d))
        d_bg = depth[background_pixels]
        d_bg = d_bg[np.isfinite(d_bg)]
        if d_bg.size:
            present[background_slot] = True
            median_depth[background_slot] = float(np.median(d_bg))
        slot_depth = rank_scale(median_depth, present)

        decode_fn = None
        if self.build_decode:
            slot_map = np.full(instance.shape, background_slot, dtype=np.int64)
            for i, g in enumerate(group_ids):
                if not kept_mask[i]:
                    continue
                slot_map[masks[g]] = int(slot_of[i])

            def decode_fn(px: np.ndarray) -> np.ndarray:  # noqa: E306
                px = np.asarray(px, dtype=np.int64)
                n = px.shape[0]
                logits = np.full((n, self.K), -LOGIT_MAGNITUDE, dtype=np.float32)
                col, row = px[:, 0], px[:, 1]
                inside = (col >= 0) & (col < res) & (row >= 0) & (row < res)
                idx = np.full(n, background_slot, dtype=np.int64)
                idx[inside] = slot_map[row[inside], col[inside]]
                logits[np.arange(n), idx] = LOGIT_MAGNITUDE
                return logits

        frame_id = self._frame_counter
        self._frame_counter += 1
        return SceneState(
            slots=slots, depth=slot_depth, schema=PLACEHOLDER_SLOT_SCHEMA,
            t=float(t) if t is not None else float(frame_id), source="ideal", window=window,
            slot_ids=slot_ids, frame_id=frame_id, _decode_fn=decode_fn,
        )


def _background_mask(masks: dict[int, np.ndarray], kept_mask: np.ndarray, group_ids: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Boolean `[res,res]`: the background slot's own pixels -- everything NOT claimed by a kept
    object (background ids, plus any dropped/overflowed object)."""
    claimed = np.zeros(shape, dtype=bool)
    for i, g in enumerate(group_ids):
        if kept_mask[i]:
            claimed |= masks[g]
    return ~claimed


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.float64)
    if frame.max(initial=0.0) > 1.0 + 1e-6:
        frame = frame / 255.0
    return frame[..., :3]


def _blob_row(mask: np.ndarray, frame: np.ndarray, res: int, *, amodal: bool, age: float) -> np.ndarray:
    """One `PLACEHOLDER_SLOT_SCHEMA` row for the object occupying boolean `mask` `[res,res]`."""
    row = np.zeros(PLACEHOLDER_SLOT_SCHEMA.W, dtype=np.float32)
    n = int(mask.sum())
    if n == 0:
        return row

    stat_mask = _amodal_fill(mask) if amodal else mask
    ys, xs = np.nonzero(stat_mask)
    x, y = to_window_xy(xs, ys, res)
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    sxx, sxy, syy = float(np.mean(dx * dx)), float(np.mean(dx * dy)), float(np.mean(dy * dy))

    # Colour and border always come from the *actually visible* pixels (`mask`, not the amodal
    # fill) -- an occluded pixel's true colour is not observed, and it is never on the crop's own
    # border by definition of being interior to the fill.
    ys_v, xs_v = np.nonzero(mask)
    rgb01 = frame[ys_v, xs_v]
    colour = 2.0 * rgb01 - 1.0
    mean_colour = colour.mean(axis=0)
    colour_spread = float(np.mean((colour - mean_colour) ** 2))
    on_border = (xs_v == 0) | (xs_v == res - 1) | (ys_v == 0) | (ys_v == res - 1)
    border_frac = float(np.count_nonzero(on_border)) / len(xs_v)

    fields = PLACEHOLDER_SLOT_SCHEMA.fields
    row[fields["mass"]] = len(xs) / (res * res)  # `len(xs)`, not `n`: amodal fill grows the mass too
    row[fields["centre"]] = [mx, my]
    row[fields["moments"]] = [sxx, sxy, syy]
    row[fields["colour"]] = mean_colour
    row[fields["colour_spread"]] = colour_spread
    row[fields["border"]] = border_frac
    row[fields["age"]] = age
    row[fields["orient4"]] = fourth_moment(dx, dy)
    return row


def _amodal_fill(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return mask
    pts = np.stack([xs, ys], axis=-1).astype(np.int32)
    hull = cv2.convexHull(pts)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool) | mask


def _merge_touching_same_colour(
    effective: np.ndarray, object_ids: list[int], frame: np.ndarray, palette: dict[str, tuple[float, float, float]],
) -> dict[int, int]:
    from .producers import hsv_colour_distance

    mask = {i: (effective == i) for i in object_ids}
    colour_name = {}
    names = list(palette)
    for i in object_ids:
        mean_rgb = tuple(frame[mask[i]][:, :3].mean(axis=0))
        dists = [hsv_colour_distance(mean_rgb, palette[n]) for n in names]
        colour_name[i] = names[int(np.argmin(dists))]

    kernel = np.ones((3, 3), np.uint8)
    dilated = {i: cv2.dilate(mask[i].astype(np.uint8), kernel, iterations=1).astype(bool) for i in object_ids}

    parent = {i: i for i in object_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for a_idx in range(len(object_ids)):
        for b_idx in range(a_idx + 1, len(object_ids)):
            a, b = object_ids[a_idx], object_ids[b_idx]
            if colour_name[a] != colour_name[b]:
                continue
            if np.any(dilated[a] & mask[b]) or np.any(dilated[b] & mask[a]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    return {i: find(i) for i in object_ids if find(i) != i}


def _jitter_masks(masks: dict[int, np.ndarray], max_px: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Erodes or dilates each object's own mask independently by a random amount (direction and
    magnitude drawn per object). Unlike the old per-pixel slot map this project used before the
    slot-state redesign, these masks are only ever used to recompute *this object's own*
    statistics, so two objects' masks growing into each other is not a bookkeeping problem here
    (there is no shared label map to corrupt) -- it simply, correctly, makes their reported
    blobs overlap a little, exactly what boundary noise should do."""
    out = {}
    for gid, m in masks.items():
        amount = int(rng.integers(1, max_px + 1))
        kernel = np.ones((2 * amount + 1, 2 * amount + 1), np.uint8)
        op = cv2.dilate if rng.random() < 0.5 else cv2.erode
        out[gid] = op(m.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out
