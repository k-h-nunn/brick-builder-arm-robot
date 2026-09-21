"""Rule-based failure-class labeller.

`FailureClass` itself lives in `brick_builder/eval/results.py`, alongside `EpisodeResult`, because both
`rl` and `envs` need to construct labelled episodes. This module only adds the logic that turns raw
simulator-derived fields on a finished, unsuccessful `EpisodeResult` into one of those classes, plus
a counting helper used by the report.

The rule order matters: some conditions overlap (a jam can also show as an overload trip), so the
checks below run most-specific-and-most-severe first. An env's own termination logic may already set
`.failure` explicitly; `label` only fills gaps or re-derives it for records built without one.
"""

from __future__ import annotations

from collections import Counter

from brick_builder.eval.results import EpisodeResult, FailureClass

# Termination-tick budget shared with press_env.py's episode length. Kept here
# so this module works standalone from result.steps without needing an env import.
DEFAULT_MAX_STEPS = 200

# A closed hand grasp band; below this the gripper reports "empty" in sim telemetry. Used only for
# heuristics when a record has no explicit `failure` set and steps ran out with the brick missing.
_LOAD_JAM_THRESHOLD = 0.8
_OVERLOAD_LOAD_THRESHOLD = 0.97  # matches the emulated overload trip band in the servo model


def label(result: EpisodeResult, max_steps: int = DEFAULT_MAX_STEPS) -> FailureClass | None:
    """Return the failure class for `result`, deriving one if `.failure` is unset.

    Returns `None` for a successful episode. If `result.failure` is already set, it is returned
    unchanged (the environment that produced the episode is the authority on its own termination
    reason); this function is a fallback labeller for records that only carry raw state.
    """
    if result.success:
        return None
    if result.failure is not None:
        return result.failure

    # `structure_intact`/`peak_load` are `None` when an environment did not report them
    # (rl-eval-export-2) -- that is "no evidence either way", not "definitely broken"/"definitely
    # overloaded", so each check below only fires on an explicit measured value and otherwise
    # falls through to the next signal, same as the pre-existing `seat_depth_error is None` gap.
    if result.structure_intact is False:
        return FailureClass.STRUCTURE_DAMAGE
    if result.peak_load is not None and result.peak_load >= _OVERLOAD_LOAD_THRESHOLD:
        return FailureClass.OVERLOAD_TRIP
    if result.steps >= max_steps:
        return FailureClass.TIMEOUT
    if result.peak_load is not None and result.peak_load >= _LOAD_JAM_THRESHOLD:
        return FailureClass.JAM
    if result.seat_depth_error is not None:
        # Seated but shallow/off-target rather than fully unseated: distinguish a wedge/tilt
        # (partial seat) from a clean miss onto the wrong studs by how large the residual is.
        if 0.0 < result.seat_depth_error <= 0.003:
            return FailureClass.PARTIAL_SEAT
        return FailureClass.WRONG_CELL
    return FailureClass.KINEMATIC


def label_all(results: list[EpisodeResult], max_steps: int = DEFAULT_MAX_STEPS) -> list[EpisodeResult]:
    """Return a copy of `results` with `.failure` filled in on every failed episode."""
    labelled = []
    for r in results:
        if r.success or r.failure is not None:
            labelled.append(r)
            continue
        labelled.append(
            EpisodeResult(
                success=r.success,
                steps=r.steps,
                seat_depth_error=r.seat_depth_error,
                peak_load=r.peak_load,
                structure_intact=r.structure_intact,
                failure=label(r, max_steps=max_steps),
                ret=r.ret,
                params=r.params,
            )
        )
    return labelled


def failure_counts(results: list[EpisodeResult]) -> Counter:
    """Count failures by class over a batch of (already-labelled) results.

    Successful episodes are not counted. Any failed episode with `.failure is None` (a record that
    was never labelled) is counted under the string key `"unlabelled"` so it is visible in reports
    rather than silently dropped.
    """
    counts: Counter = Counter()
    for r in results:
        if r.success:
            continue
        counts[r.failure.value if r.failure is not None else "unlabelled"] += 1
    return counts
