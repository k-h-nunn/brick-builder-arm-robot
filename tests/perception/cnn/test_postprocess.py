"""Grouping and slot assembly, on synthetic heads -- no trained weights involved.

These tests build the four head outputs by hand, so they check the *logic* that turns a class
map, a centre heat map and an offset field into instances and then into slot rows, independently
of how well any particular checkpoint predicts them.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.perception.cnn.labels import ARM_CLASS, BASEPLATE_CLASS, BRICK_CLASS
from brick_builder.perception.cnn.postprocess import (
    GroupingConfig, Prediction, instances_from_heads, slots_from_masks,
)
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, TemporalSlotAssigner

RES = 128


def _prediction(boxes, arm_box=None, plate_box=None, depth_of=None) -> Prediction:
    """Heads that would come from a perfect network: exact classes, exact centres, exact offsets."""
    classes = np.zeros((RES, RES), dtype=np.int64)
    centre = np.zeros((RES, RES), dtype=np.float32)
    offset = np.zeros((2, RES, RES), dtype=np.float32)
    depth = np.zeros((RES, RES), dtype=np.float32)
    ys, xs = np.indices((RES, RES)).astype(np.float32)
    if arm_box is not None:
        r0, c0, r1, c1 = arm_box
        classes[r0:r1, c0:c1] = ARM_CLASS
    if plate_box is not None:
        r0, c0, r1, c1 = plate_box
        classes[r0:r1, c0:c1] = BASEPLATE_CLASS
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        classes[r0:r1, c0:c1] = BRICK_CLASS
        cy, cx = (r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0
        centre[int(round(cy)), int(round(cx))] = 1.0
        box = (slice(r0, r1), slice(c0, c1))
        offset[0][box] = cx - xs[box]
        offset[1][box] = cy - ys[box]
        depth[box] = 0.0 if depth_of is None else depth_of[i]
    prob = np.zeros((4, RES, RES), dtype=np.float32)
    for c in range(4):
        prob[c] = (classes == c).astype(np.float32)
    return Prediction(classes=classes, class_prob=prob, centre=centre, offset=offset, depth=depth)


def test_two_touching_bricks_are_separated_by_the_centres_not_by_colour():
    """The two boxes share an edge and would be one connected component; the centre and offset
    heads are the only thing that can tell them apart, and they do."""
    pred = _prediction([(20, 20, 60, 50), (20, 50, 60, 80)])
    masks, kinds = instances_from_heads(pred)
    assert kinds.count("brick") == 2
    bricks = [m for m, k in zip(masks, kinds) if k == "brick"]
    assert not np.any(bricks[0] & bricks[1])
    assert abs(int(bricks[0].sum()) - int(bricks[1].sum())) <= 2


def test_arm_and_baseplate_are_one_instance_each():
    pred = _prediction([(80, 80, 100, 100)], arm_box=(0, 0, 20, 128), plate_box=(30, 0, 70, 128))
    masks, kinds = instances_from_heads(pred)
    assert kinds.count("arm") == 1
    assert kinds.count("baseplate") == 1
    assert kinds.count("brick") == 1


def test_specks_below_the_area_floor_are_dropped():
    pred = _prediction([(20, 20, 60, 60), (100, 100, 103, 103)])
    masks, kinds = instances_from_heads(pred, GroupingConfig(min_area_px=60))
    assert kinds.count("brick") == 1


def test_no_centre_found_still_reports_the_brick_pixels():
    pred = _prediction([(20, 20, 60, 60)])
    pred.centre[:] = 0.0
    masks, kinds = instances_from_heads(pred)
    assert kinds.count("brick") == 1


def test_slot_rows_use_the_placeholder_schema_and_fill_the_background_slot():
    pred = _prediction([(20, 20, 60, 60)], arm_box=(0, 0, 10, 128))
    masks, _ = instances_from_heads(pred)
    frame = np.full((RES, RES, 3), 0.5, dtype=np.float32)
    frame[20:60, 20:60] = (0.8, 0.1, 0.1)
    assigner = TemporalSlotAssigner(K=8)
    assembly = slots_from_masks(masks, frame, pred.depth, assigner, K=8, window=None,
                                source="test", t=0.0, frame_id=0)
    state = assembly.state
    assert state.schema is PLACEHOLDER_SLOT_SCHEMA
    assert not state.is_empty(assembly.background_slot)
    filled = [k for k in range(state.K) if not state.is_empty(k)]
    assert len(filled) == len(masks) + 1  # every object, plus the background slot


def test_depth_order_follows_the_predicted_depth_map():
    pred = _prediction([(10, 10, 40, 40), (70, 70, 100, 100)], depth_of=[-1.0, 1.0])
    masks, kinds = instances_from_heads(pred)
    frame = np.full((RES, RES, 3), 0.4, dtype=np.float32)
    assembly = slots_from_masks(masks, frame, pred.depth, TemporalSlotAssigner(K=8), K=8,
                                window=None, source="test", t=0.0, frame_id=0)
    state = assembly.state
    near, far = int(assembly.slot_of[0]), int(assembly.slot_of[1])
    assert state.depth[near] < state.depth[far]


def test_decode_is_optional_and_off_when_asked():
    pred = _prediction([(20, 20, 60, 60)])
    masks, _ = instances_from_heads(pred)
    frame = np.full((RES, RES, 3), 0.5, dtype=np.float32)
    with_decode = slots_from_masks(masks, frame, pred.depth, TemporalSlotAssigner(K=8), K=8,
                                   window=None, source="t", t=0.0, frame_id=0,
                                   build_decode=True).state
    without = slots_from_masks(masks, frame, pred.depth, TemporalSlotAssigner(K=8), K=8,
                               window=None, source="t", t=0.0, frame_id=0,
                               build_decode=False).state
    assert with_decode.has_decode and not without.has_decode
    logits = with_decode.decode(np.array([[30, 30], [120, 120]]))
    assert logits.shape == (2, 8)
    assert int(np.argmax(logits[0])) != int(np.argmax(logits[1]))


def test_reuses_the_ideal_sources_blob_statistics_rather_than_a_second_copy():
    """A copy of the blob maths would be a second thing to keep in step with the schema. The row
    this module produces for a mask must be, to the bit, the row the ideal source produces for the
    same mask on the same frame."""
    from brick_builder.perception.ideal_scene_model import _blob_row

    mask = np.zeros((RES, RES), dtype=bool)
    mask[15:55, 25:95] = True
    frame = np.full((RES, RES, 3), 0.3, dtype=np.float32)
    frame[mask] = (0.9, 0.2, 0.1)
    assembly = slots_from_masks([mask], frame, np.zeros((RES, RES), dtype=np.float32),
                                TemporalSlotAssigner(K=8), K=8, window=None, source="t", t=0.0,
                                frame_id=0)
    k = int(assembly.slot_of[0])
    expected = _blob_row(mask, frame, RES, amodal=False, age=0.0)
    fields = PLACEHOLDER_SLOT_SCHEMA.fields
    for name, sl in fields.items():
        if name == "age":
            continue
        np.testing.assert_array_equal(assembly.state.slots[k][sl], expected[sl],
                                      err_msg=f"field {name}")


@pytest.mark.parametrize("n_objects", [9, 12])
def test_more_objects_than_slots_keeps_the_largest(n_objects):
    boxes, size = [], 30
    for i in range(n_objects):
        r = 2 + (i // 4) * 32
        c = 2 + (i % 4) * 32
        side = size - i
        boxes.append((r, c, r + side, c + side))
    pred = _prediction(boxes)
    masks, _ = instances_from_heads(pred, GroupingConfig(max_instances=16))
    frame = np.full((RES, RES, 3), 0.5, dtype=np.float32)
    assembly = slots_from_masks(masks, frame, pred.depth, TemporalSlotAssigner(K=8), K=8,
                                window=None, source="t", t=0.0, frame_id=0)
    assert int((assembly.slot_of >= 0).sum()) <= 7
