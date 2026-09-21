"""The loop in a world with nothing wrong with it.

A nominal world is the one case where the answer is known without measuring anything: the
recording was made in it, so the feed-forward sequence alone already puts the brick on the cell
and every correction the loop computes is a correction to nothing. These check that switching
the camera corrections ON does not move the brick -- for the placement task, which starts with
the brick already held, and for the whole pick-and-place, which fetches it off the table first.

Slow by the suite's standards: each one records a target sequence and then tracks it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

TOLERANCE_MM = 0.5
"""Lateral error a nominal run may end with."""

CORRECTION_MM = 0.5
CORRECTION_DEG = 0.5
"""Accumulated correction a nominal run may reach at a dwell, where the estimate is updated."""


def _track(pick: bool, seed: int = 0):
    """Record in the nominal world, track the same nominal world, score the result."""
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import SUPPLY_BODY, build_pick_world, build_world
    from brick_builder.control.pick_place import verify_placement

    cfg = PlaceConfig(pick=pick, closed_loop=True)
    ideal, nominal = record_ideal(cfg, seed=seed)
    nominal.close()

    if pick:
        scene, *_ = build_pick_world(seed, displaced=False)
        brick = SUPPLY_BODY
    else:
        scene, _, _ = build_world(seed, displaced=False)
        brick = scene.held_name

    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed)
    at_dwell = []
    try:
        for record in episode.run():
            if record.vision and episode.dwelling:
                at_dwell.append(record.correction.copy())
        check = verify_placement(scene, brick, (4.5, 3.5, 0))
    finally:
        episode.close()
    return check, np.asarray(at_dwell) if at_dwell else np.zeros((1, 3)), episode


@pytest.mark.slow
def test_a_nominal_placement_lands_on_the_cell() -> None:
    check, corrections, episode = _track(pick=False)
    assert episode.failure is None, episode.failure
    assert check.at_target_cell and check.seated, (
        f"lateral {check.lateral_mm:.2f} mm, depth error {check.depth_error_mm:.2f} mm"
    )
    assert check.lateral_mm < TOLERANCE_MM, f"lateral {check.lateral_mm:.2f} mm"


@pytest.mark.slow
@pytest.mark.xfail(reason="lands 15.9 mm from the cell, one stud pitch: the in-hand estimate "
                          "reads 3.2 mm over the baseplate where the brick is centred in the "
                          "jaws to 0.3 mm, and the align dwell turns that into 5.5 mm of tool "
                          "correction", strict=False)
def test_a_nominal_pick_and_place_lands_on_the_cell() -> None:
    check, corrections, episode = _track(pick=True)
    assert episode.failure is None, episode.failure
    assert check.at_target_cell and check.seated, (
        f"lateral {check.lateral_mm:.2f} mm, depth error {check.depth_error_mm:.2f} mm"
    )
    assert check.lateral_mm < TOLERANCE_MM, f"lateral {check.lateral_mm:.2f} mm"


@pytest.mark.slow
@pytest.mark.xfail(reason="the correction reaches 5.5 mm and 1.5 deg at the align dwell",
                   strict=False)
def test_a_nominal_pick_and_place_never_corrects_for_anything() -> None:
    """Every unknown the loop estimates is already zero here, so the correction must stay zero.

    The interesting failure is not the final millimetre but the integrator walking away while
    the sequence dwells, which is where the estimate is updated.
    """
    _, corrections, _ = _track(pick=True)
    worst_xy = 1e3 * float(np.max(np.linalg.norm(corrections[:, :2], axis=1)))
    worst_yaw = math.degrees(float(np.max(np.abs(corrections[:, 2]))))
    assert worst_xy < CORRECTION_MM, f"correction reached {worst_xy:.2f} mm at a dwell"
    assert worst_yaw < CORRECTION_DEG, f"correction reached {worst_yaw:.2f} deg at a dwell"


@pytest.mark.slow
def test_a_nominal_grasp_closes_where_it_was_sent() -> None:
    """The jaws have to reach the commanded grasp height, or what they hold slides out of them.

    Measured with four millimetres of clearance: a jaw lands on the brick's top face, the arm
    stalls 9.6 mm high, and the brick slid 32 mm out of the jaws between the lift and the press.
    """
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import SUPPLY_BODY, build_pick_world

    cfg = PlaceConfig(pick=True, closed_loop=True)
    ideal, nominal = record_ideal(cfg, seed=0)
    nominal.close()
    scene, *_ = build_pick_world(0, displaced=False)
    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=0)
    in_jaw = None
    try:
        for record in episode.run():
            if in_jaw is None and record.step == "Lift":
                pos, _ = scene.body_pose(SUPPLY_BODY)
                tcp_pos, tcp_rot = scene.tcp_pose()
                in_jaw = tcp_rot.T @ (np.asarray(pos) - np.asarray(tcp_pos))
    finally:
        episode.close()
    assert in_jaw is not None, "the run never reached the lift"
    lateral = 1e3 * float(np.linalg.norm(in_jaw[:2]))
    assert lateral < 2.0, f"the brick sits {lateral:.2f} mm off-centre between the jaws"
