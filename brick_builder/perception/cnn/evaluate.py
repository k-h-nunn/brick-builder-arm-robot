"""How good is it, measured against what the controller actually needs?

The controller reads slot rows. So the headline numbers here are slot-row numbers: how far the
model's reported centre, mass, second moments and colour are from the ideal source's, on frames
neither model has seen, and how often a brick keeps its slot from one frame to the next. The
pixel-level numbers (class overlap, instance precision and recall) are here too, because they
explain the slot-row numbers when something goes wrong.

Every comparison is against `ideal_scene_model.IdealSceneModel` fed the *same* instance map the
labels were generated from (`data.perception_instance_lut`), so both producers are answering the
same question about the same objects.

The colour-threshold source (`conventional_scene_model.ColourSegmentationSceneModel`) is run on
the same frames as the baseline. It publishes no class and no masks, only slot rows, so the
comparison between the two is made at slot level -- detection by centroid match, then the field
errors -- and the pixel-level rows of the table are marked as not applicable for it rather than
filled with a number that would mean something different.

Sections
--------
`frame_metrics`      one held-out frame: masks, instances, slot rows, depth order.
`evaluate_frames`    the whole held-out split, aggregated, plus the per-size and green-on-green
                     breakdowns.
`evaluate_episode`   a `control.pick_place` run, frames in order: slot identity switches.
`speed`              milliseconds per frame at one and four threads, parameters, multiply-
                     accumulates, checkpoint size.
`evaluation_report`  all of the above into one dictionary, and the table the guide prints.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..ideal_scene_model import IdealSceneModel
from .data import DEFAULT_CACHE, Sample, load_samples
from .labels import CLASS_NAMES, FIRST_BRICK_INSTANCE, N_CLASSES

IOU_MATCH = 0.5
CENTRE_MATCH = 0.25
"""Slot-level detection gate, in normalized window units (`[-1, 1]` across the frame): 0.25 is
32 pixels of a 256-pixel window, about two thirds of a 2x4 brick's long side at the usual working
distance. Loose on purpose -- it is a "did it find this object at all" test, and the centroid
error is reported separately and in full."""


# ------------------------------------------------------------------------------- small tools ---


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def gt_masks(sample: Sample) -> dict[int, np.ndarray]:
    """Ground-truth mask per instance id (background id 0 excluded)."""
    return {int(i): (sample.instance == i) for i in np.unique(sample.instance) if i != 0}


def brick_ids(sample: Sample) -> list[int]:
    return [int(i) for i in np.unique(sample.instance) if i >= FIRST_BRICK_INSTANCE]


def touching_same_colour_ids(sample: Sample) -> set[int]:
    """Ground-truth bricks that touch another brick of the same palette colour.

    Uses `ideal_scene_model._merge_touching_same_colour`, the same test the ideal source's own
    "merge" degradation knob uses, so "touching and the same colour" means here exactly what it
    means there.
    """
    from ..ideal_scene_model import _merge_touching_same_colour
    from ...bricks.spec import PALETTE

    ids = brick_ids(sample)
    if len(ids) < 2:
        return set()
    frame = sample.rgb.astype(np.float64) / 255.0
    merges = _merge_touching_same_colour(sample.instance.astype(np.int32), ids, frame, PALETTE)
    out: set[int] = set()
    for a, b in merges.items():
        out.add(int(a))
        out.add(int(b))
    return out


def green_ids(sample: Sample) -> set[int]:
    """Ground-truth bricks whose mean colour is nearest to green in the palette."""
    from ..producers import hsv_colour_distance
    from ...bricks.spec import PALETTE

    frame = sample.rgb.astype(np.float64) / 255.0
    names = list(PALETTE)
    out: set[int] = set()
    for i in brick_ids(sample):
        mean_rgb = tuple(frame[sample.instance == i].mean(axis=0))
        d = [hsv_colour_distance(mean_rgb, PALETTE[n]) for n in names]
        if names[int(np.argmin(d))] == "green":
            out.add(i)
    return out


def _match_by_iou(pred: list[np.ndarray], true: list[np.ndarray]) -> dict[int, int]:
    """Best one-to-one assignment of predicted masks to true masks by overlap; pairs below
    `IOU_MATCH` are dropped. Returns `{true index: predicted index}`."""
    if not pred or not true:
        return {}
    cost = np.zeros((len(true), len(pred)))
    for a, t in enumerate(true):
        for b, p in enumerate(pred):
            cost[a, b] = -_iou(t, p)
    rows, cols = linear_sum_assignment(cost)
    return {int(a): int(b) for a, b in zip(rows, cols) if -cost[a, b] >= IOU_MATCH}


def _present_slots(state, background_slot: int) -> list[int]:
    return [k for k in range(state.K) if k != background_slot and not state.is_empty(k)]


def _match_slots_by_centre(pred_state, pred_slots, true_state, true_slots) -> dict[int, int]:
    """One-to-one match of a source's slots to the ideal source's, on centre distance alone --
    the only thing a producer that publishes no mask can be matched on."""
    if not pred_slots or not true_slots:
        return {}
    cost = np.zeros((len(true_slots), len(pred_slots)))
    for a, kt in enumerate(true_slots):
        ct = np.asarray(true_state.centre(kt))
        for b, kp in enumerate(pred_slots):
            cost[a, b] = float(np.linalg.norm(ct - np.asarray(pred_state.centre(kp))))
    rows, cols = linear_sum_assignment(cost)
    return {true_slots[a]: pred_slots[b] for a, b in zip(rows, cols) if cost[a, b] <= CENTRE_MATCH}


def _angle_error_deg(a: float, b: float) -> float:
    """Difference between two ellipse orientations, in degrees, folded into [0, 90]: an ellipse's
    axis has no head or tail, so 179 degrees and 1 degree are the same orientation."""
    d = math.degrees(a - b) % 180.0
    return float(min(d, 180.0 - d))


# --------------------------------------------------------------------------- one frame's row ---


@dataclass
class FrameResult:
    """Everything measured on one frame, before aggregation."""

    rows: dict
    per_brick: list[dict]


def frame_metrics(sample: Sample, source, ideal: IdealSceneModel, colour_source=None,
                  window=None) -> FrameResult:
    """Run every producer on one frame and measure them against the renderer's ground truth."""
    instance = sample.instance.astype(np.int32)
    depth_true = sample.depth.astype(np.float64)
    truth = gt_masks(sample)
    touching = touching_same_colour_ids(sample)
    greens = green_ids(sample)

    ideal_state = ideal.observe(sample.rgb, window=window, instance=instance, depth=depth_true)
    ideal_bg = ideal.assigner.background_slot
    slot_of_id = {int(g): k for k, g in enumerate(ideal_state.slot_ids) if g != -1}

    state = source.observe(sample.rgb, window=window)
    pred = source.last_prediction
    pred_masks = list(source.last_masks)
    kinds = list(source.last_kinds)

    row: dict = {}

    # -- per-class overlap -------------------------------------------------------------------
    true_class = sample.classes
    for c, name in enumerate(CLASS_NAMES):
        p, t = pred.classes == c, true_class == c
        row[f"inter_{name}"] = float(np.logical_and(p, t).sum())
        row[f"union_{name}"] = float(np.logical_or(p, t).sum())

    # -- brick instances ---------------------------------------------------------------------
    brick_index = [i for i, k in enumerate(kinds) if k == "brick"]
    pred_brick = [pred_masks[i] for i in brick_index]
    true_brick_ids = brick_ids(sample)
    true_brick = [truth[i] for i in true_brick_ids]
    matched = _match_by_iou(pred_brick, true_brick)
    row["brick_true"] = float(len(true_brick))
    row["brick_pred"] = float(len(pred_brick))
    row["brick_tp"] = float(len(matched))
    row["brick_touch_true"] = float(sum(1 for i in true_brick_ids if i in touching))
    row["brick_touch_tp"] = float(sum(1 for a, i in enumerate(true_brick_ids)
                                      if i in touching and a in matched))
    row["brick_green_true"] = float(sum(1 for i in true_brick_ids if i in greens))
    row["brick_green_tp"] = float(sum(1 for a, i in enumerate(true_brick_ids)
                                      if i in greens and a in matched))

    # -- centroid, orientation, size, in pixels and millimetres ------------------------------
    fx = float(sample.meta.get("fx", 0.0)) or None
    per_brick: list[dict] = []
    res = int(sample.instance.shape[0])
    for a, inst_id in enumerate(true_brick_ids):
        gt_mask = truth[inst_id]
        area = float(gt_mask.sum())
        d_here = depth_true[gt_mask]
        d_here = d_here[np.isfinite(d_here)]
        depth_m = float(np.median(d_here)) if d_here.size else float("nan")
        entry = {
            "area_px": area, "depth_m": depth_m, "touching": inst_id in touching,
            "green": inst_id in greens, "found": a in matched,
        }
        if a in matched:
            pm = pred_brick[matched[a]]
            entry["iou"] = _iou(gt_mask, pm)
            ty, tx = np.nonzero(gt_mask)
            py, px = np.nonzero(pm)
            err_px = float(np.hypot(px.mean() - tx.mean(), py.mean() - ty.mean()))
            entry["centroid_px"] = err_px
            if fx and math.isfinite(depth_m):
                entry["centroid_mm"] = err_px * depth_m / fx * 1e3
            k_true = slot_of_id.get(inst_id)
            if k_true is not None:
                axes = ideal_state.orientation_and_axes(k_true)[1]
                ratio = axes[0] / max(axes[1], 1e-9)
                if ratio >= 1.5:  # an elongated brick: a 2x4 looks like this, a 2x2 does not
                    ang_true = ideal_state.orientation_and_axes(k_true)[0]
                    k_pred = int(source.last_slot_of[brick_index[matched[a]]])
                    if k_pred >= 0:
                        entry["orient_deg"] = _angle_error_deg(
                            state.orientation_and_axes(k_pred)[0], ang_true)
        per_brick.append(entry)

    # -- slot rows, field by field, against the ideal source ---------------------------------
    row.update(_slot_field_errors(state, source.assigner.background_slot,
                                  ideal_state, ideal_bg, prefix="cnn"))
    if colour_source is not None:
        colour_state = colour_source.observe(sample.rgb, window=window)
        row.update(_slot_field_errors(colour_state, colour_source.assigner.background_slot,
                                      ideal_state, ideal_bg, prefix="colour"))
        row.update(_slot_detection(colour_state, colour_source.assigner.background_slot,
                                   ideal_state, ideal_bg, slot_of_id, true_brick_ids,
                                   prefix="colour", res=res, depth_true=depth_true, fx=fx,
                                   sample=sample))

    # -- depth order -------------------------------------------------------------------------
    conc, total = 0, 0
    order_items = []
    for a, inst_id in enumerate(true_brick_ids):
        if a not in matched:
            continue
        pm = pred_brick[matched[a]]
        d_true = depth_true[truth[inst_id]]
        d_true = d_true[np.isfinite(d_true)]
        d_pred = pred.depth[pm]
        if d_true.size and d_pred.size:
            order_items.append((float(np.median(d_true)), float(np.median(d_pred))))
    for i in range(len(order_items)):
        for j in range(i + 1, len(order_items)):
            if order_items[i][0] == order_items[j][0]:
                continue
            total += 1
            same = (order_items[i][0] < order_items[j][0]) == (order_items[i][1] < order_items[j][1])
            conc += int(same)
    row["depth_pairs"] = float(total)
    row["depth_concordant"] = float(conc)
    return FrameResult(rows=row, per_brick=per_brick)


def _slot_field_errors(state, background_slot: int, ideal_state, ideal_bg: int, prefix: str) -> dict:
    """Absolute error per slot-row field, averaged over the slots that matched."""
    pred_slots = _present_slots(state, background_slot)
    true_slots = _present_slots(ideal_state, ideal_bg)
    pairs = _match_slots_by_centre(state, pred_slots, ideal_state, true_slots)
    out: dict = {f"{prefix}_slots_true": float(len(true_slots)),
                 f"{prefix}_slots_pred": float(len(pred_slots)),
                 f"{prefix}_slots_matched": float(len(pairs))}
    acc: dict[str, list[float]] = {"mass": [], "centre": [], "moments": [], "colour": []}
    for kt, kp in pairs.items():
        acc["mass"].append(abs(state.mass(kp) - ideal_state.mass(kt)))
        acc["centre"].append(float(np.linalg.norm(
            np.asarray(state.centre(kp)) - np.asarray(ideal_state.centre(kt)))))
        acc["moments"].append(float(np.linalg.norm(
            np.asarray(state.moments(kp)) - np.asarray(ideal_state.moments(kt)))))
        acc["colour"].append(float(np.linalg.norm(
            np.asarray(state.colour(kp)) - np.asarray(ideal_state.colour(kt)))))
    for name, values in acc.items():
        out[f"{prefix}_field_{name}_sum"] = float(np.sum(values))
        out[f"{prefix}_field_{name}_n"] = float(len(values))
    return out


def _slot_detection(state, background_slot, ideal_state, ideal_bg, slot_of_id, true_brick_ids,
                    prefix, res, depth_true, fx, sample) -> dict:
    """Slot-level brick detection and centroid error for a producer that publishes no masks."""
    pred_slots = _present_slots(state, background_slot)
    brick_slots = [slot_of_id[i] for i in true_brick_ids if i in slot_of_id]
    pairs = _match_slots_by_centre(state, pred_slots, ideal_state, brick_slots)
    out = {f"{prefix}_brick_true": float(len(brick_slots)),
           f"{prefix}_brick_tp": float(len(pairs))}
    errs_px, errs_mm = [], []
    lin_half = 0.5 * (res - 1)
    for kt, kp in pairs.items():
        d = float(np.linalg.norm(
            np.asarray(state.centre(kp)) - np.asarray(ideal_state.centre(kt)))) * lin_half
        errs_px.append(d)
        inst_id = next((i for i in true_brick_ids if slot_of_id.get(i) == kt), None)
        if inst_id is not None and fx:
            dd = depth_true[sample.instance == inst_id]
            dd = dd[np.isfinite(dd)]
            if dd.size:
                errs_mm.append(d * float(np.median(dd)) / fx * 1e3)
    out[f"{prefix}_centroid_px_sum"] = float(np.sum(errs_px))
    out[f"{prefix}_centroid_px_n"] = float(len(errs_px))
    out[f"{prefix}_centroid_mm_sum"] = float(np.sum(errs_mm))
    out[f"{prefix}_centroid_mm_n"] = float(len(errs_mm))
    return out


# ------------------------------------------------------------------------ the held-out split ---


def evaluate_frames(samples: list[Sample], source, colour_source=None, progress: bool = False) -> dict:
    """Aggregate `frame_metrics` over a list of frames. Each frame is independent, so every
    producer's slot identity is reset before each one."""
    totals: dict[str, float] = {}
    per_brick: list[dict] = []
    ideal = IdealSceneModel()
    for n, sample in enumerate(samples):
        source.reset()
        ideal.reset()
        if colour_source is not None:
            colour_source.reset()
        result = frame_metrics(sample, source, ideal, colour_source)
        for k, v in result.rows.items():
            totals[k] = totals.get(k, 0.0) + v
        per_brick.extend(result.per_brick)
        if progress and (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(samples)} frames", flush=True)

    out: dict = {"frames": len(samples)}
    iou = []
    for name in CLASS_NAMES:
        value = totals[f"inter_{name}"] / max(totals[f"union_{name}"], 1.0)
        out[f"iou_{name}"] = value
        iou.append(value)
    out["miou"] = float(np.mean(iou))

    out["brick_recall"] = totals["brick_tp"] / max(totals["brick_true"], 1.0)
    out["brick_precision"] = totals["brick_tp"] / max(totals["brick_pred"], 1.0)
    out["brick_touching_recall"] = totals["brick_touch_tp"] / max(totals["brick_touch_true"], 1.0)
    out["brick_touching_n"] = totals["brick_touch_true"]
    out["brick_green_recall"] = totals["brick_green_tp"] / max(totals["brick_green_true"], 1.0)
    out["brick_green_n"] = totals["brick_green_true"]
    out["depth_order_accuracy"] = totals["depth_concordant"] / max(totals["depth_pairs"], 1.0)
    out["depth_order_pairs"] = totals["depth_pairs"]

    for prefix in ("cnn", "colour"):
        if f"{prefix}_slots_matched" not in totals:
            continue
        out[f"{prefix}_slot_match_rate"] = (totals[f"{prefix}_slots_matched"]
                                            / max(totals[f"{prefix}_slots_true"], 1.0))
        for field_name in ("mass", "centre", "moments", "colour"):
            n = max(totals[f"{prefix}_field_{field_name}_n"], 1.0)
            out[f"{prefix}_field_{field_name}"] = totals[f"{prefix}_field_{field_name}_sum"] / n
    if "colour_brick_true" in totals:
        out["colour_brick_recall"] = totals["colour_brick_tp"] / max(totals["colour_brick_true"], 1.0)
        out["colour_centroid_px"] = (totals["colour_centroid_px_sum"]
                                     / max(totals["colour_centroid_px_n"], 1.0))
        out["colour_centroid_mm"] = (totals["colour_centroid_mm_sum"]
                                     / max(totals["colour_centroid_mm_n"], 1.0))

    found = [b for b in per_brick if b.get("found")]
    out["centroid_px"] = float(np.mean([b["centroid_px"] for b in found])) if found else float("nan")
    mm = [b["centroid_mm"] for b in found if "centroid_mm" in b]
    out["centroid_mm"] = float(np.mean(mm)) if mm else float("nan")
    out["centroid_mm_p90"] = float(np.percentile(mm, 90)) if mm else float("nan")
    orient = [b["orient_deg"] for b in found if "orient_deg" in b]
    out["orientation_deg"] = float(np.mean(orient)) if orient else float("nan")
    out["orientation_deg_p90"] = float(np.percentile(orient, 90)) if orient else float("nan")
    out["orientation_n"] = len(orient)
    out["mean_iou_matched"] = float(np.mean([b["iou"] for b in found])) if found else float("nan")

    out["by_size"] = _by_size(per_brick)
    out["_per_brick"] = per_brick
    return out


SIZE_BINS = [(0, 400), (400, 1200), (1200, 3000), (3000, 8000), (8000, 10**9)]


def _by_size(per_brick: list[dict]) -> list[dict]:
    """Recall and centroid error against apparent size -- the honest way to show that a small,
    far brick is harder than a near one."""
    rows = []
    for lo, hi in SIZE_BINS:
        sel = [b for b in per_brick if lo <= b["area_px"] < hi]
        if not sel:
            continue
        found = [b for b in sel if b.get("found")]
        mm = [b["centroid_mm"] for b in found if "centroid_mm" in b]
        rows.append({
            "area_lo": lo, "area_hi": hi, "n": len(sel),
            "recall": len(found) / len(sel),
            "centroid_px": float(np.mean([b["centroid_px"] for b in found])) if found else float("nan"),
            "centroid_mm": float(np.mean(mm)) if mm else float("nan"),
        })
    return rows


# ----------------------------------------------------------------------------- one episode ---


def evaluate_episode(source, colour_source=None, seed: int = 11, max_frames: int = 90,
                     every: int = 6) -> dict:
    """Slot identity across a real pick-and-place run.

    Frames arrive in order, as they would at run time, so this is the only part of the evaluation
    that can say anything about identity: how often an object that stays visible changes slot.
    The ideal source is measured the same way and is the floor -- it carries the simulator's own
    object id, so it never switches.
    """
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from ...control.pick_place import PickPlace, make_scene
    from ...render.renderer import SceneRenderer
    from .data import (
        RENDER_SIZE, jitter_plate, perception_instance_lut, randomize_appearance, render_sample,
        roi_around, sample_camera_model, window_for_roi,
    )

    rng = np.random.default_rng(seed)
    scene = make_scene(supply_bricks=2, cameras=())
    jitter_plate(scene, rng, max_xy=0.01, max_yaw_deg=5.0)
    randomize_appearance(scene, rng)
    diagonal, post = sample_camera_model(rng)          # one camera model for the whole run
    runner = PickPlace(scene, brick="supply_brick_0", target=(3.5, 3.5, 0),
                       rng=np.random.default_rng(seed + 1))
    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    plate_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    plate_xy = scene.model.body_pos[plate_bid][:2].copy()

    samples: list[Sample] = []
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        for step in runner.run():
            if len(samples) >= max_frames:
                break
            if step.tick % every:
                continue
            brick_pos = scene.body_pose(runner.brick)[0]
            focus = brick_pos if step.phase in ("to_tray", "open", "descend", "close", "lift") \
                else np.array([plate_xy[0], plate_xy[1], scene.plate_top])
            pose = window_for_roi(scene, roi_around(focus, (0.06, 0.06, 0.04)), diagonal, None,
                                  name="episode")
            if pose is None:
                continue
            samples.append(render_sample(scene, pose, renderer, post, rng,
                                         {"kind": "episode", "phase": step.phase}))
    finally:
        renderer.close()

    ideal = IdealSceneModel()
    results = {"episode_frames": len(samples)}
    for name, producer in (("cnn", source), ("colour", colour_source), ("ideal", ideal)):
        if producer is None:
            continue
        producer.reset()
        switches, tracked = _identity_switches(samples, producer, ideal_reference=(name == "ideal"))
        results[f"{name}_switches_per_100"] = 100.0 * switches / max(tracked, 1)
        results[f"{name}_tracked_frames"] = tracked

    # The same accuracy numbers as the held-out split, but on this episode's own frames: an
    # episode is a harder, more correlated sample than a set of independent scenes, and the
    # controller only ever sees episodes.
    if samples:
        frames_report = evaluate_frames(samples, source, colour_source)
        frames_report.pop("_per_brick", None)
        results["accuracy_on_this_episode"] = {
            k: frames_report[k] for k in
            ("miou", "iou_brick", "iou_arm", "iou_baseplate", "brick_recall", "brick_precision",
             "centroid_px", "centroid_mm", "depth_order_accuracy", "cnn_slot_match_rate")
            if k in frames_report
        }
    return results


def _identity_switches(samples: list[Sample], producer, ideal_reference: bool) -> tuple[int, int]:
    """Count how often a ground-truth object's slot changes between consecutive frames.

    An object is followed by matching each frame's slots to the ideal source's (by centroid), so
    "which slot is this brick in" is answered the same way for every producer.
    """
    ideal = IdealSceneModel()
    previous: dict[int, int] = {}
    switches, tracked = 0, 0
    for sample in samples:
        instance = sample.instance.astype(np.int32)
        depth = sample.depth.astype(np.float64)
        ideal_state = ideal.observe(sample.rgb, window=None, instance=instance, depth=depth)
        ideal_bg = ideal.assigner.background_slot
        slot_of_id = {int(g): k for k, g in enumerate(ideal_state.slot_ids) if g != -1}
        if ideal_reference:
            state, bg = ideal_state, ideal_bg
        else:
            state = producer.observe(sample.rgb, window=None)
            bg = producer.assigner.background_slot
        pairs = _match_slots_by_centre(state, _present_slots(state, bg),
                                       ideal_state, _present_slots(ideal_state, ideal_bg))
        current = {int(ideal_state.slot_ids[kt]): kp for kt, kp in pairs.items()
                   if ideal_state.slot_ids[kt] != -1}
        for inst_id, slot in current.items():
            if inst_id in previous:
                tracked += 1
                switches += int(previous[inst_id] != slot)
        previous = current
    return switches, tracked


# ---------------------------------------------------------------------------------- speed ---


def speed(source, frames: list[Sample], threads: tuple[int, ...] = (1, 3), repeats: int = 15) -> dict:
    """Milliseconds per frame, end to end (`observe`, so grouping and slot assembly included).

    The *median* of the repeats, not the mean, and the machine's load average is recorded next to
    it: this project's machine is shared, and a mean over a run that happened to collide with
    somebody else's job says more about that job than about this model.
    """
    import os

    import torch

    from .model import count_macs

    original = torch.get_num_threads()
    out: dict = {"load_average_1min": float(os.getloadavg()[0]), "cpu_count": os.cpu_count()}
    try:
        for n in threads:
            torch.set_num_threads(n)
            source.reset()
            for i in range(min(3, len(frames))):
                source.observe(frames[i].rgb, window=None)
            times = []
            for i in range(repeats):
                t0 = time.perf_counter()
                source.observe(frames[i % len(frames)].rgb, window=None)
                times.append((time.perf_counter() - t0) * 1e3)
            out[f"ms_per_frame_{n}_threads"] = float(np.median(times))
            out[f"ms_per_frame_{n}_threads_best"] = float(np.min(times))
    finally:
        torch.set_num_threads(original)
    out["parameters"] = source.n_parameters
    out["macs_per_frame"] = count_macs(source.net, 256)
    path = Path(source.checkpoint)
    out["checkpoint_bytes"] = int(path.stat().st_size) if path.exists() else 0
    return out


def speed_of(producer, frames: list[Sample], repeats: int = 15) -> float:
    """Median milliseconds per `observe` for any producer, on the same frames -- so the colour
    threshold's cost can be put next to the network's rather than left unsaid."""
    producer.reset()
    for i in range(min(3, len(frames))):
        producer.observe(frames[i].rgb, window=None)
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        producer.observe(frames[i % len(frames)].rgb, window=None)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# ------------------------------------------------------------------------------- the report ---


def evaluation_report(
    checkpoint: str | Path | None = None,
    cache: str | Path = DEFAULT_CACHE,
    split: str = "val",
    limit: int | None = 300,
    out_dir: str | Path = "results/vision_model",
    with_episode: bool = True,
    progress: bool = True,
) -> dict:
    """Everything, into `results/vision_model/evaluation.json`, and the printable table."""
    from ..conventional_scene_model import ColourSegmentationSceneModel
    from .source import CnnSceneModel

    samples = load_samples(cache, split, limit=limit)
    source = CnnSceneModel(checkpoint=checkpoint)
    colour = ColourSegmentationSceneModel()
    if progress:
        print(f"evaluating {len(samples)} held-out frames", flush=True)
    report = evaluate_frames(samples, source, colour, progress=progress)
    per_brick = report.pop("_per_brick")
    report["speed"] = speed(source, samples[:8])
    report["speed"]["colour_ms_per_frame"] = speed_of(ColourSegmentationSceneModel(), samples[:8])
    if with_episode:
        if progress:
            print("running one pick-and-place episode for identity", flush=True)
        report["episode"] = evaluate_episode(source, ColourSegmentationSceneModel())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=1, default=float),
                                             encoding="utf-8")
    np.save(out_dir / "per_brick.npy", np.array(per_brick, dtype=object), allow_pickle=True)
    (out_dir / "evaluation_table.md").write_text(format_table(report), encoding="utf-8")
    return report


def format_table(report: dict) -> str:
    """The table the guide page shows. One row per thing the controller depends on."""
    s = report.get("speed", {})
    e = report.get("episode", {})

    def f(key, fmt="{:.3f}", src=report, missing="n/a"):
        v = src.get(key)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return missing
        return fmt.format(v)

    lines = [
        "| metric | CNN | colour threshold |",
        "| --- | --- | --- |",
        f"| IoU background/table | {f('iou_background')} | n/a |",
        f"| IoU baseplate | {f('iou_baseplate')} | n/a |",
        f"| IoU arm | {f('iou_arm')} | n/a |",
        f"| IoU brick | {f('iou_brick')} | n/a |",
        f"| mIoU, 4 classes | {f('miou')} | n/a |",
        f"| Brick recall @ IoU 0.5 | {f('brick_recall')} | {f('colour_brick_recall')} (centroid match) |",
        f"| Brick precision @ IoU 0.5 | {f('brick_precision')} | n/a |",
        f"| Brick recall, touching same colour (n={int(report.get('brick_touching_n', 0))}) | {f('brick_touching_recall')} | n/a |",
        f"| Brick recall, green on green plate (n={int(report.get('brick_green_n', 0))}) | {f('brick_green_recall')} | n/a |",
        f"| Centre error (px) | {f('centroid_px', '{:.2f}')} | {f('colour_centroid_px', '{:.2f}')} |",
        f"| Centre error at plate (mm) | {f('centroid_mm', '{:.2f}')} | {f('colour_centroid_mm', '{:.2f}')} |",
        f"| Centre error at plate, p90 (mm) | {f('centroid_mm_p90', '{:.2f}')} | n/a |",
        f"| Orientation error, elongated bricks (deg, n={report.get('orientation_n', 0)}) | {f('orientation_deg', '{:.1f}')} | n/a |",
        f"| Depth-order accuracy, brick pairs | {f('depth_order_accuracy')} | n/a |",
        f"| Slot match rate vs ideal source | {f('cnn_slot_match_rate')} | {f('colour_slot_match_rate')} |",
        f"| Slot-row error: mass | {f('cnn_field_mass', '{:.4f}')} | {f('colour_field_mass', '{:.4f}')} |",
        f"| Slot-row error: centre | {f('cnn_field_centre', '{:.4f}')} | {f('colour_field_centre', '{:.4f}')} |",
        f"| Slot-row error: second moments | {f('cnn_field_moments', '{:.4f}')} | {f('colour_field_moments', '{:.4f}')} |",
        f"| Slot-row error: colour | {f('cnn_field_colour', '{:.4f}')} | {f('colour_field_colour', '{:.4f}')} |",
        f"| Slot changes per 100 tracked frames | {f('cnn_switches_per_100', '{:.1f}', e)} | {f('colour_switches_per_100', '{:.1f}', e)} |",
    ]
    for n in sorted(int(k.split("_")[3]) for k in s if re.fullmatch(r"ms_per_frame_\d+_threads", k)):
        lines.append(
            f"| Latency, {n} CPU thread{'' if n == 1 else 's'} (ms/frame, median and best) | "
            f"{f(f'ms_per_frame_{n}_threads', '{:.0f}', s)} / "
            f"{f(f'ms_per_frame_{n}_threads_best', '{:.0f}', s)} | - |"
        )
    lines += [
        f"| Latency, 1 CPU thread (ms/frame) | - | {f('colour_ms_per_frame', '{:.0f}', s)} |",
        f"| Parameters (M) | {s.get('parameters', 0) / 1e6:.2f} | 0 |",
        f"| MAC per frame (G) | {s.get('macs_per_frame', 0) / 1e9:.2f} | - |",
        f"| Checkpoint (MB) | {s.get('checkpoint_bytes', 0) / 1e6:.1f} | - |",
    ]
    return "\n".join(lines) + "\n"
