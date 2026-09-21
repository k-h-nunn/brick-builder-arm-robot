"""Tests for brick_builder.eval.taxonomy."""

from brick_builder.eval.results import EpisodeResult, FailureClass
from brick_builder.eval.taxonomy import failure_counts, label, label_all


def _ep(**kwargs) -> EpisodeResult:
    defaults = dict(success=False, steps=50, peak_load=0.1, structure_intact=True)
    defaults.update(kwargs)
    return EpisodeResult(**defaults)


def test_success_labels_none():
    ep = _ep(success=True)
    assert label(ep) is None


def test_explicit_failure_is_preserved():
    ep = _ep(failure=FailureClass.SLIP)
    assert label(ep) is FailureClass.SLIP


def test_structure_damage_takes_priority():
    ep = _ep(structure_intact=False, peak_load=0.99, steps=200)
    assert label(ep) is FailureClass.STRUCTURE_DAMAGE


def test_overload_trip_from_peak_load():
    ep = _ep(peak_load=0.98)
    assert label(ep) is FailureClass.OVERLOAD_TRIP


def test_timeout_at_max_steps():
    ep = _ep(steps=200, peak_load=0.1)
    assert label(ep, max_steps=200) is FailureClass.TIMEOUT


def test_jam_from_sustained_load_without_timeout():
    ep = _ep(steps=50, peak_load=0.85)
    assert label(ep, max_steps=200) is FailureClass.JAM


def test_partial_seat_for_small_residual():
    ep = _ep(steps=50, peak_load=0.1, seat_depth_error=0.001)
    assert label(ep) is FailureClass.PARTIAL_SEAT


def test_wrong_cell_for_large_seat_residual():
    ep = _ep(steps=50, peak_load=0.1, seat_depth_error=0.02)
    assert label(ep) is FailureClass.WRONG_CELL


def test_kinematic_is_the_fallback():
    ep = _ep(steps=50, peak_load=0.1, seat_depth_error=None)
    assert label(ep) is FailureClass.KINEMATIC


def test_label_all_fills_only_unset_failures():
    ep_unset = _ep(steps=50, peak_load=0.1, seat_depth_error=None)
    ep_set = _ep(failure=FailureClass.JAM)
    ep_success = _ep(success=True, peak_load=0.0)

    labelled = label_all([ep_unset, ep_set, ep_success])

    assert labelled[0].failure is FailureClass.KINEMATIC
    assert labelled[1].failure is FailureClass.JAM
    assert labelled[2].failure is None
    assert labelled[2].success is True


def test_failure_counts_ignores_successes_and_counts_by_class():
    results = [
        _ep(success=True),
        _ep(failure=FailureClass.JAM),
        _ep(failure=FailureClass.JAM),
        _ep(failure=FailureClass.SLIP),
    ]
    counts = failure_counts(results)
    assert counts["jam"] == 2
    assert counts["slip"] == 1
    assert "success" not in counts
    assert sum(counts.values()) == 3


def test_failure_counts_flags_unlabelled_records():
    results = [_ep(success=False, failure=None, peak_load=0.0, steps=1, seat_depth_error=None)]
    # Force an unlabelled record by bypassing label(): failure stays None and success is False.
    counts = failure_counts(results)
    assert counts["unlabelled"] == 1


# --- rl-eval-export-2: label() must not crash or misread a None peak_load/structure_intact ---


def test_label_does_not_crash_when_peak_load_and_structure_intact_are_unreported():
    """`peak_load`/`structure_intact` are `None` ("not reported"), not comparable to a float
    or usable as `not None`-is-truthy -- `label` must not raise, and with nothing else to go
    on for a short episode it falls all the way through to the kinematic catch-all, exactly as
    it already does when `seat_depth_error` alone is missing."""
    ep = _ep(peak_load=None, structure_intact=None, seat_depth_error=None, steps=50)
    assert label(ep, max_steps=200) is FailureClass.KINEMATIC


def test_label_does_not_treat_unreported_structure_intact_as_damaged():
    """`not None` is `True` in plain Python -- a naive `if not result.structure_intact:` (the
    pre-fix code) would misread "not reported" as "definitely broken" and always return
    STRUCTURE_DAMAGE. It must instead fall through to whatever the other, actually-reported
    fields say."""
    ep = _ep(structure_intact=None, peak_load=0.98)  # peak_load alone should win here
    assert label(ep) is FailureClass.OVERLOAD_TRIP
    assert label(ep) is not FailureClass.STRUCTURE_DAMAGE


def test_label_still_detects_structure_damage_when_explicitly_reported():
    """Regression guard: an explicit `structure_intact=False` must still win, unchanged."""
    ep = _ep(structure_intact=False, peak_load=0.99, steps=200)
    assert label(ep) is FailureClass.STRUCTURE_DAMAGE


def test_label_does_not_treat_unreported_peak_load_as_overload_or_jam():
    """A `None` peak_load must skip both load-based heuristics (overload trip and jam) rather
    than crashing on `None >= threshold`, falling through to the next real signal."""
    ep = _ep(peak_load=None, seat_depth_error=0.001, steps=50)
    assert label(ep) is FailureClass.PARTIAL_SEAT
