"""Tests for `brick_builder.perception.conventional_scene_model`: the classical colour-segmentation
producer, and the two documented stubs."""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.perception.conventional_scene_model import (
    DEFAULT_PALETTE,
    ColourSegmentationSceneModel,
    ConventionalGpuSceneModel,
    LearnedSceneModel,
)
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA


def _synthetic_scene(res: int = 128):
    img = np.full((res, res, 3), 0.5, dtype=np.float32)  # a plain grey background
    boxes = {
        "red": (10, 10, 40, 40),
        "blue": (80, 10, 110, 40),
        "yellow": (10, 80, 40, 110),
    }
    for colour, (r0, c0, r1, c1) in boxes.items():
        img[r0:r1, c0:c1] = DEFAULT_PALETTE[colour]
    return img, boxes


def test_finds_each_coloured_box_with_a_sensible_centroid():
    img, boxes = _synthetic_scene()
    src = ColourSegmentationSceneModel()
    st = src.observe(img, window=None, t=0.0)
    assert src.last_object_count == 3

    present = [k for k in range(st.K) if not st.is_empty(k)]
    found_colours = 0
    for colour, (r0, c0, r1, c1) in boxes.items():
        ref = np.array([2 * v - 1 for v in DEFAULT_PALETTE[colour]])
        best_k, best_d = None, np.inf
        for k in present:
            d = float(np.linalg.norm(np.array(st.colour(k)) - ref))
            if d < best_d:
                best_k, best_d = k, d
        assert best_d < 0.3, f"{colour}: nearest slot colour distance {best_d}"
        expected_x = (c0 + c1 - 1) / 2.0 / (128 - 1) * 2.0 - 1.0
        expected_y = (r0 + r1 - 1) / 2.0 / (128 - 1) * 2.0 - 1.0
        cx, cy = st.centre(best_k)
        assert abs(cx - expected_x) < 0.05
        assert abs(cy - expected_y) < 0.05
        found_colours += 1
    assert found_colours == 3


def test_background_slot_covers_the_rest_of_the_frame():
    img, boxes = _synthetic_scene()
    src = ColourSegmentationSceneModel()
    st = src.observe(img, window=None)
    bg = src.assigner.background_slot
    assert not st.is_empty(bg)
    assert st.mass(bg) > 0.5  # most of a 128x128 frame is the plain grey background


def test_identity_persists_across_frames_via_hungarian_matching():
    img, boxes = _synthetic_scene()
    src = ColourSegmentationSceneModel()
    st1 = src.observe(img, window=None, t=0.0)
    st2 = src.observe(img, window=None, t=1.0)  # identical frame -> identical blobs
    np.testing.assert_array_equal(st1.slot_ids, st2.slot_ids)


def test_identity_tracks_a_slowly_moving_blob():
    res = 128
    src = ColourSegmentationSceneModel()
    prev_slot = None
    for shift in range(0, 20, 4):
        img = np.full((res, res, 3), 0.5, dtype=np.float32)
        img[40:70, 40 + shift : 70 + shift] = DEFAULT_PALETTE["red"]
        st = src.observe(img, window=None, t=float(shift))
        red_slots = [k for k in range(st.K) if not st.is_empty(k) and st.mass(k) > 0.05
                     and np.linalg.norm(np.array(st.colour(k)) - np.array([2 * v - 1 for v in DEFAULT_PALETTE["red"]])) < 0.3]
        assert len(red_slots) == 1
        if prev_slot is not None:
            assert red_slots[0] == prev_slot, "the moving red blob should keep its own slot"
        prev_slot = red_slots[0]


def test_present_mask_excludes_pixels():
    img, boxes = _synthetic_scene()
    mask = np.ones(img.shape[:2], dtype=bool)
    mask[10:40, 10:40] = False  # hide the red box entirely
    src = ColourSegmentationSceneModel()
    st = src.observe(img, window=None, present_mask=mask)
    ref_red = np.array([2 * v - 1 for v in DEFAULT_PALETTE["red"]])
    for k in range(st.K):
        if st.is_empty(k):
            continue
        d = float(np.linalg.norm(np.array(st.colour(k)) - ref_red))
        assert d > 0.3, "the masked-out red box must not be found as its own slot"


def test_schema_and_shape():
    img, _ = _synthetic_scene()
    src = ColourSegmentationSceneModel()
    st = src.observe(img, window=None)
    assert st.schema is PLACEHOLDER_SLOT_SCHEMA
    assert st.slots.shape == (8, 14)
    assert not st.has_decode  # colour segmentation publishes no ground-truth decode


def test_overflow_when_many_more_than_k_minus_1_blobs():
    res = 200
    img = np.full((res, res, 3), 0.5, dtype=np.float32)
    colours = list(DEFAULT_PALETTE)
    k = 0
    for row in range(0, res - 20, 20):
        for col in range(0, res - 20, 20):
            img[row : row + 10, col : col + 10] = DEFAULT_PALETTE[colours[k % len(colours)]]
            k += 1
    src = ColourSegmentationSceneModel(K=8, min_area_px=4)
    st = src.observe(img, window=None)
    assert src.last_object_count > 7
    assert src.last_kept_count == 7
    assert src.last_overflow_count == src.last_object_count - 7


# ------------------------------------------------------------------------------------- stubs ---


def test_learned_scene_model_forwards_to_the_trained_convolutional_model():
    """The stub this class used to be is now a real producer: it forwards to
    `perception.cnn.source.CnnSceneModel`, which ships trained weights. Skipped where those
    weights are absent (a checkout that has not trained one)."""
    from pathlib import Path

    from brick_builder.perception.cnn.source import DEFAULT_WEIGHTS, CnnSceneModel

    if not Path(DEFAULT_WEIGHTS).exists():
        pytest.skip(f"no trained checkpoint at {DEFAULT_WEIGHTS}")
    assert ConventionalGpuSceneModel is LearnedSceneModel     # the old name still resolves
    model = LearnedSceneModel()
    assert isinstance(model.impl, CnnSceneModel)
    state = model.observe(np.zeros((256, 256, 3), dtype=np.uint8), window=None)
    assert state.schema is PLACEHOLDER_SLOT_SCHEMA
    assert state.slots.shape == (8, PLACEHOLDER_SLOT_SCHEMA.W)
    assert state.source == "conv_net"
    model.reset()

