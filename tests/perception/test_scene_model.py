"""Tests for `brick_builder.perception.scene_model`: the schema, `SceneState`'s named accessors, the
shared `rank_scale` helper, and `TemporalSlotAssigner`'s identity bookkeeping.

None of this needs a render or a scene -- these are pure functions/dataclasses, so they run
unconditionally (no skip).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from brick_builder.perception.scene_model import (
    PLACEHOLDER_SLOT_SCHEMA,
    SceneState,
    SlotSchema,
    TemporalSlotAssigner,
    opaque_schema,
    rank_scale,
    to_window_xy,
)


def _state(rows: dict[int, list[float]], depth: dict[int, float] | None = None, K: int = 8) -> SceneState:
    slots = np.zeros((K, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    d = np.full(K, 0.5, dtype=np.float32)
    slot_ids = np.full(K, -1, dtype=np.int64)
    for k, row in rows.items():
        slots[k, : len(row)] = row
        slot_ids[k] = k
    if depth:
        for k, v in depth.items():
            d[k] = v
    return SceneState(
        slots=slots, depth=d, schema=PLACEHOLDER_SLOT_SCHEMA, t=0.0, source="test",
        window=None, slot_ids=slot_ids,
    )


# --------------------------------------------------------------------------------- SlotSchema ---


def test_schema_get_and_has():
    row = np.arange(PLACEHOLDER_SLOT_SCHEMA.W, dtype=np.float32)
    np.testing.assert_allclose(PLACEHOLDER_SLOT_SCHEMA.get(row, "centre"), [1, 2])
    assert PLACEHOLDER_SLOT_SCHEMA.has("mass")
    assert not PLACEHOLDER_SLOT_SCHEMA.has("nope")
    with pytest.raises(KeyError):
        PLACEHOLDER_SLOT_SCHEMA.get(row, "nope")


def test_opaque_schema_declares_nothing():
    s = opaque_schema(72)
    assert s.W == 72
    assert not s.has("mass")
    assert s.names() == ()


def test_placeholder_schema_width_and_field_names():
    assert PLACEHOLDER_SLOT_SCHEMA.W == 14
    assert set(PLACEHOLDER_SLOT_SCHEMA.names()) == {
        "orient4",
        "mass", "centre", "moments", "colour", "colour_spread", "border", "age",
    }


# --------------------------------------------------------------------------------- SceneState ---


def test_scene_state_rejects_shape_mismatches():
    slots = np.zeros((8, 14), dtype=np.float32)
    with pytest.raises(ValueError):
        SceneState(slots=slots, depth=np.zeros(7), schema=PLACEHOLDER_SLOT_SCHEMA, t=0.0,
                   source="t", window=None, slot_ids=np.full(8, -1))
    with pytest.raises(ValueError):
        SceneState(slots=slots, depth=np.full(8, 0.5), schema=opaque_schema(3), t=0.0,
                   source="t", window=None, slot_ids=np.full(8, -1))


def test_named_accessors_read_the_declared_columns():
    st = _state({0: [0.2, 0.1, -0.3, 0.04, 0.0, 0.09, 0.5, -0.5, 0.0, 0.02, 0.1, 0.6]})
    assert st.mass(0) == pytest.approx(0.2)
    assert st.centre(0) == pytest.approx((0.1, -0.3))
    assert st.moments(0) == pytest.approx((0.04, 0.0, 0.09))
    assert st.colour(0) == pytest.approx((0.5, -0.5, 0.0))
    assert st.colour_spread(0) == pytest.approx(0.02)
    assert st.border_fraction(0) == pytest.approx(0.1)
    assert st.age(0) == pytest.approx(0.6)


def test_is_empty_true_only_for_all_zero_row_with_depth_half():
    st = _state({0: [0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    assert not st.is_empty(0)
    assert st.is_empty(1)


def test_orientation_and_axes_axis_aligned_ellipse():
    # sxx=0.04 (std 0.2 along x), syy=0.01 (std 0.1 along y), no correlation -> angle 0, axes (0.4,0.2)
    st = _state({0: [0.1, 0, 0, 0.04, 0.0, 0.01, 0, 0, 0, 0, 0, 0]})
    angle, axes = st.orientation_and_axes(0)
    assert abs(math.remainder(angle, math.pi)) < 1e-6
    assert axes[0] == pytest.approx(0.4, abs=1e-6)
    assert axes[1] == pytest.approx(0.2, abs=1e-6)


def test_orientation_and_axes_rotated_ellipse():
    # A diagonal ellipse: equal variance along x/y with positive covariance -> major axis at 45deg
    st = _state({0: [0.1, 0, 0, 0.02, 0.015, 0.02, 0, 0, 0, 0, 0, 0]})
    angle, axes = st.orientation_and_axes(0)
    assert abs(abs(math.degrees(angle)) - 45.0) < 5.0
    assert axes[0] > axes[1]


def test_decode_raises_when_absent_and_works_when_present():
    st = _state({0: [0.1] + [0] * 11})
    with pytest.raises(RuntimeError):
        st.decode(np.array([[0, 0]]))
    assert not st.has_decode

    def fn(px):
        return np.zeros((px.shape[0], st.K), dtype=np.float32)

    st2 = SceneState(slots=st.slots, depth=st.depth, schema=st.schema, t=0.0, source="t",
                      window=None, slot_ids=st.slot_ids, _decode_fn=fn)
    assert st2.has_decode
    out = st2.decode(np.array([[1, 2], [3, 4]]))
    assert out.shape == (2, st2.K)
    with pytest.raises(ValueError):
        st2.decode(np.array([1, 2]))  # not [n,2]


def test_res_reads_window_output_size_or_raises():
    st = _state({})
    with pytest.raises(AttributeError):
        _ = st.res


# --------------------------------------------------------------------------------- rank_scale ---


def test_rank_scale_orders_present_slots_nearest_to_smallest():
    depth = np.array([5.0, 1.0, 3.0, np.nan])
    present = np.array([True, True, True, False])
    out = rank_scale(depth, present)
    assert out[3] == pytest.approx(0.5)
    assert out[1] < out[2] < out[0]
    assert out[1] == pytest.approx(0.0)
    assert out[0] == pytest.approx(1.0)


def test_rank_scale_lone_present_slot_is_zero():
    out = rank_scale(np.array([3.0, np.nan]), np.array([True, False]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)


def test_rank_scale_nothing_present_is_all_half():
    out = rank_scale(np.array([np.nan, np.nan]), np.array([False, False]))
    np.testing.assert_allclose(out, [0.5, 0.5])


def test_to_window_xy_endpoints():
    x, y = to_window_xy(np.array([0, 255]), np.array([0, 255]), res=256)
    np.testing.assert_allclose(x, [-1.0, 1.0])
    np.testing.assert_allclose(y, [-1.0, 1.0])


# ------------------------------------------------------------------------- TemporalSlotAssigner ---


def test_identity_persists_with_external_ids_across_frames():
    a = TemporalSlotAssigner(K=4, vanish_after=2)
    masses = np.array([10.0, 5.0])
    rows = np.zeros((2, 3))
    s1 = a.step(masses, rows, external_ids=[100, 200])
    s2 = a.step(masses, rows, external_ids=[100, 200])
    s3 = a.step(masses, rows, external_ids=[200, 100])  # order shuffled -- identity must not care
    assert list(s1) == list(s2) == list(s3[::-1])


def test_vanished_slot_decays_and_is_then_reused():
    a = TemporalSlotAssigner(K=2, vanish_after=2)
    a.step(np.array([10.0]), np.zeros((1, 3)), external_ids=[1])
    # object 1 stops appearing for `vanish_after` frames -- its slot must NOT be reused yet
    for _ in range(2):
        s = a.step(np.array([]), np.zeros((0, 3)), external_ids=[])
        assert list(s) == []
    # one frame further (missing count now exceeds vanish_after): a new object may claim the slot
    a.step(np.array([]), np.zeros((0, 3)), external_ids=[])
    s = a.step(np.array([20.0]), np.zeros((1, 3)), external_ids=[999])
    assert s[0] == 0  # the only slot, now free


def test_background_slot_is_whatever_nothing_claims():
    a = TemporalSlotAssigner(K=3, vanish_after=1)
    s = a.step(np.array([10.0, 5.0]), np.zeros((2, 3)), external_ids=[1, 2])
    claimed = set(int(x) for x in s)
    assert a.background_slot not in claimed
    assert 0 <= a.background_slot < 3


def test_overflow_beyond_k_minus_1_reports_minus_one():
    a = TemporalSlotAssigner(K=4, vanish_after=1)  # room for 3 objects + background
    masses = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
    s = a.step(masses, np.zeros((5, 3)), external_ids=list(range(5)))
    kept = s[s >= 0]
    overflowed = s[s < 0]
    assert len(kept) == 3
    assert len(overflowed) == 2
    # the largest three are kept
    kept_indices = np.nonzero(s >= 0)[0]
    assert set(kept_indices.tolist()) == {0, 1, 2}


def test_hungarian_matching_without_external_ids_tracks_moving_rows():
    a = TemporalSlotAssigner(K=4, vanish_after=2, match_max_distance=2.0)
    rows_a = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    s1 = a.step(np.array([10.0, 9.0]), rows_a, external_ids=None)
    rows_b = rows_a + 0.2  # small drift, well under match_max_distance
    s2 = a.step(np.array([10.0, 9.0]), rows_b, external_ids=None)
    assert list(s1) == list(s2)


def test_hungarian_matching_large_jump_does_not_match():
    a = TemporalSlotAssigner(K=4, vanish_after=2, match_max_distance=0.5)
    rows_a = np.array([[0.0, 0.0, 0.0]])
    s1 = a.step(np.array([10.0]), rows_a, external_ids=None)
    rows_far = np.array([[100.0, 100.0, 100.0]])
    s2 = a.step(np.array([10.0]), rows_far, external_ids=None)
    # too far to match the old row -- either a different slot, or (if none free) unassigned
    assert s1[0] != s2[0] or s2[0] == -1


def test_reset_clears_identity():
    a = TemporalSlotAssigner(K=4, vanish_after=5)
    s1 = a.step(np.array([10.0]), np.zeros((1, 3)), external_ids=[1])
    a.reset()
    s2 = a.step(np.array([10.0]), np.zeros((1, 3)), external_ids=[1])
    assert list(s1) == list(s2)  # both claim slot 0 fresh, but via independent, unrelated resets
    assert a.slot_age(int(s2[0])) == 1
