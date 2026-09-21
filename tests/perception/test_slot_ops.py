"""Tests for `brick_builder.perception.slot_ops`: comparing, selecting, editing and logging
`SceneState`s. All pure functions of arrays; no render needed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState
from brick_builder.perception.slot_ops import (
    LogRecord,
    read_log,
    select_slot,
    set_slot_blob,
    set_slot_offset,
    slot_delta,
    slot_distance,
    write_log,
)


def _state(rows: dict[int, list[float]], K: int = 8) -> SceneState:
    slots = np.zeros((K, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    slot_ids = np.full(K, -1, dtype=np.int64)
    for k, row in rows.items():
        slots[k, : len(row)] = row
        slot_ids[k] = k
    return SceneState(slots=slots, depth=np.full(K, 0.5, dtype=np.float32), schema=PLACEHOLDER_SLOT_SCHEMA,
                       t=0.0, source="test", window=None, slot_ids=slot_ids)


RED = [0.2, 0.1, -0.1, 0.01, 0.0, 0.01, 0.5, -0.9, -0.8, 0.02, 0.0, 0.3]  # mass, centre, moments, colour(2*rgb-1)...


def test_slot_delta_zero_for_identical_states():
    st = _state({0: RED})
    d = slot_delta(st, st)
    np.testing.assert_allclose(d, 0.0)
    assert slot_distance(st, st) == pytest.approx(0.0)


def test_slot_delta_empty_vs_object_is_large_not_masked():
    intended = _state({0: RED})
    actual = _state({})
    d = slot_delta(intended, actual)
    assert np.any(d[0] != 0.0)
    assert slot_distance(intended, actual) > 0.0


def test_slot_delta_rejects_mismatched_shapes():
    from brick_builder.perception.scene_model import opaque_schema

    a = _state({0: RED})
    slots = np.zeros((8, 5), dtype=np.float32)
    b = SceneState(slots=slots, depth=np.full(8, 0.5, dtype=np.float32), schema=opaque_schema(5),
                   t=0.0, source="t", window=None, slot_ids=np.full(8, -1))
    with pytest.raises(ValueError):
        slot_delta(a, b)


def test_slot_distance_weights_down_weight_a_column():
    intended = _state({0: RED})
    actual = _state({0: [0.2, 0.1, -0.1, 0.01, 0.0, 0.01, -0.5, 0.9, 0.8, 0.02, 0.0, 0.3]})  # colour flipped
    w_full = slot_distance(intended, actual)
    weights = np.ones(PLACEHOLDER_SLOT_SCHEMA.W)
    weights[PLACEHOLDER_SLOT_SCHEMA.fields["colour"]] = 0.0
    w_zeroed = slot_distance(intended, actual, weights=weights)
    assert w_zeroed < w_full


# --------------------------------------------------------------------------------- select_slot ---


def test_select_slot_finds_matching_colour_and_size():
    # slot 3: red-ish, small; slot 5: blue-ish, bigger
    red = [0.05, 0.2, 0.2, 0.02, 0.0, 0.02, 0.54, -0.82, -0.82, 0.01, 0.0, 0.1]
    blue = [0.10, -0.3, -0.1, 0.08, 0.0, 0.08, -0.8, -0.36, 0.24, 0.01, 0.0, 0.1]
    st = _state({3: red, 5: blue})
    assert select_slot(st, colour=(0.77, 0.09, 0.09), colour_tol=0.5) == 3
    assert select_slot(st, colour=(0.0, 0.32, 0.62), colour_tol=0.5) == 5
    assert select_slot(st, colour=(0.0, 0.46, 0.22), colour_tol=0.3) is None  # green: no match


def test_select_slot_size_constraint():
    small = [0.05, 0, 0, 0.01, 0.0, 0.01, 0, 0, 0, 0, 0, 0]  # axes ~ (0.2, 0.2)
    big = [0.2, 0, 0, 0.09, 0.0, 0.01, 0, 0, 0, 0, 0, 0]     # axes ~ (0.6, 0.2)
    st = _state({0: small, 1: big})
    assert select_slot(st, size=(0.2, 0.2), size_tol=0.15) == 0
    assert select_slot(st, size=(0.6, 0.2), size_tol=0.15) == 1


def test_select_slot_excludes_and_respects_mass_bounds():
    st = _state({0: [0.5] + [0.0] * 11, 1: [0.01] + [0.0] * 11})
    assert select_slot(st, min_mass=0.1) == 0
    assert select_slot(st, min_mass=0.1, exclude=[0]) is None
    assert select_slot(st, max_mass=0.05) == 1


def test_select_slot_none_when_schema_lacks_field():
    from brick_builder.perception.scene_model import opaque_schema

    st = SceneState(slots=np.zeros((4, 3), dtype=np.float32), depth=np.full(4, 0.5, dtype=np.float32),
                     schema=opaque_schema(3), t=0.0, source="t", window=None, slot_ids=np.full(4, -1))
    assert select_slot(st, colour=(1, 0, 0)) is None


# ------------------------------------------------------------------------------------ editing ---


def test_set_slot_blob_overwrites_only_given_fields():
    st = _state({0: RED})
    edited = set_slot_blob(st, 0, centre=(0.5, -0.5))
    assert edited.centre(0) == pytest.approx((0.5, -0.5))
    assert edited.mass(0) == pytest.approx(st.mass(0))  # untouched
    # original is not mutated
    assert st.centre(0) == pytest.approx((0.1, -0.1))


def test_set_slot_blob_rejects_field_not_in_schema():
    from brick_builder.perception.scene_model import opaque_schema

    st = SceneState(slots=np.zeros((4, 3), dtype=np.float32), depth=np.full(4, 0.5, dtype=np.float32),
                     schema=opaque_schema(3), t=0.0, source="t", window=None, slot_ids=np.full(4, -1))
    with pytest.raises(KeyError):
        set_slot_blob(st, 0, centre=(0.0, 0.0))


def test_set_slot_offset_axis_aligned_reference():
    # reference slot 0: centred at (0,0), axis-aligned ellipse, major axis length 0.4 along x
    ref = [0.1, 0.0, 0.0, 0.04, 0.0, 0.0001, 0, 0, 0, 0, 0, 0]
    st = _state({0: ref, 1: [0.0] * 12})
    edited = set_slot_offset(st, 1, reference_slot=0, offset=(1.0, 0.0))
    cx, cy = edited.centre(1)
    assert cx == pytest.approx(0.4, abs=1e-3)
    assert cy == pytest.approx(0.0, abs=1e-3)


def test_set_slot_offset_rotated_reference():
    # reference ellipse rotated 90 degrees: major axis along y
    ref = [0.1, 0.0, 0.0, 0.0001, 0.0, 0.04, 0, 0, 0, 0, 0, 0]
    st = _state({0: ref, 1: [0.0] * 12})
    edited = set_slot_offset(st, 1, reference_slot=0, offset=(1.0, 0.0))
    angle, axes = st.orientation_and_axes(0)
    assert abs(abs(math.degrees(angle)) - 90.0) < 1.0
    cx, cy = edited.centre(1)
    # offset along the reference's own major axis (now world-y) should move mostly in y
    assert abs(cy) > abs(cx)


# ------------------------------------------------------------------------------------ logging ---


def test_log_write_read_roundtrip(tmp_path):
    st = _state({0: RED})
    records = [
        LogRecord.from_state(st, joint_state=np.array([0.1, 0.2, 0.3]), action=np.array([0.0, 1.0])),
        LogRecord.from_state(st, joint_state=np.array([0.4, 0.5, 0.6]), action=np.array([1.0, 0.0])),
    ]
    path = tmp_path / "log.npz"
    write_log(records, path)
    assert path.is_file()
    back = read_log(path)
    assert len(back) == 2
    for a, b in zip(records, back):
        assert a.t == pytest.approx(b.t)
        np.testing.assert_allclose(a.slots, b.slots)
        np.testing.assert_allclose(a.depth, b.depth)
        np.testing.assert_array_equal(a.slot_ids, b.slot_ids)
        np.testing.assert_allclose(a.joint_state, b.joint_state)
        np.testing.assert_allclose(a.action, b.action)


def test_write_log_refuses_empty():
    with pytest.raises(ValueError):
        write_log([], "/tmp/unused.npz")
