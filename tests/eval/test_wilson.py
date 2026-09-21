"""Tests for brick_builder.eval.wilson."""

import math

import pytest

from brick_builder.eval.wilson import intervals_disjoint, wilson_interval


def test_published_value_180_of_200():
    # Acceptance example: 180/200 -> [0.851, 0.934].
    iv = wilson_interval(180, 200)
    assert iv.p_hat == pytest.approx(0.9)
    assert iv.low == pytest.approx(0.851, abs=1e-3)
    assert iv.high == pytest.approx(0.934, abs=1e-3)


def test_interval_contains_point_estimate():
    iv = wilson_interval(37, 50)
    assert iv.low <= iv.p_hat <= iv.high


def test_bounds_stay_in_unit_interval():
    iv_zero = wilson_interval(0, 10)
    iv_full = wilson_interval(10, 10)
    assert 0.0 <= iv_zero.low <= iv_zero.high <= 1.0
    assert 0.0 <= iv_full.low <= iv_full.high <= 1.0
    assert iv_zero.p_hat == 0.0
    assert iv_full.p_hat == 1.0


def test_interval_narrows_with_more_data_same_rate():
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert (large.high - large.low) < (small.high - small.low)


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_successes_out_of_range_raises():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def test_intervals_disjoint_true_for_far_apart_rates():
    a = wilson_interval(95, 100)   # high success rate, tight interval
    b = wilson_interval(20, 100)   # low success rate, tight interval
    assert intervals_disjoint(a, b)
    assert intervals_disjoint(b, a)


def test_intervals_disjoint_false_for_overlapping():
    a = wilson_interval(50, 100)
    b = wilson_interval(55, 100)
    assert not intervals_disjoint(a, b)


def test_matches_hand_computed_wilson_formula():
    # Cross-check against the textbook formula directly for an arbitrary case.
    successes, n, z = 42, 60, 1.96
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    expected_low = (center - margin) / denom
    expected_high = (center + margin) / denom
    iv = wilson_interval(successes, n, z=z)
    assert iv.low == pytest.approx(expected_low)
    assert iv.high == pytest.approx(expected_high)
