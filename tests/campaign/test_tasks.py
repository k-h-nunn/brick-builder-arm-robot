"""The task set: what the command compiles to, which seeds each phase draws, and the record.

Fast, except the two that build worlds.
"""

from __future__ import annotations

import pytest

from brick_builder.campaign.tasks import (
    BASE_PLACED, BUILD_COMMAND, CSV_FIELDS, EVAL_SEEDS, FIRST_PERFORM_SEED, TASKS, BrickRecord,
    EpisodeRecord, build_goal, failure_class, perform_seeds,
)


def _brick(index: int = 0, *, seated: bool = True, at_cell: bool = True,
           lateral: float = 0.1) -> BrickRecord:
    return BrickRecord(index=index, cell=(4, 3), seated=seated, at_target_cell=at_cell,
                       connected=seated, lateral_mm=lateral, depth_error_mm=0.0, tilt_deg=0.0)


def _record(**kw) -> EpisodeRecord:
    base = dict(task="placement", seed=0, phase="evaluate", round_index=0, inverse="analytic",
                predictor="analytic", inverse_version="", predictor_version="", source="ideal")
    base.update(kw)
    return EpisodeRecord(**base)


# ------------------------------------------------------------------------------- the build task ---


def test_the_build_command_compiles_to_two_bricks_on_the_bottom_layer() -> None:
    """One English command, two placements, and both reachable by the existing geometry.

    The action layer places on the plate's own top face, so a stacked second brick would need a
    seat height it has no way to express; the campaign's build is two bricks side by side. Both
    cells are inside the 100-200 mm band the gripper's downward force was characterised over,
    which the right-hand neighbour at 218 mm is not.
    """
    goal = build_goal(BUILD_COMMAND)
    placements = goal.placements
    assert len(placements) == 2
    assert [(p.x, p.y, p.z) for p in placements] == [(4, 3, 0), (2, 3, 0)]
    assert {p.colour for p in placements} == {"blue"}
    assert {(p.h, p.w) for p in placements} == {(2, 2)}


def test_the_build_command_is_refused_when_it_does_not_compile() -> None:
    with pytest.raises(ValueError):
        build_goal("put the blue 2x2 at column 5 row 4 then the red 2x2 on top of it")


def test_remaining_steps_drives_the_re_plan() -> None:
    """A brick that is not on its cell leaves its own step outstanding, and that is the signal."""
    goal = build_goal(BUILD_COMMAND)
    first = goal.steps[0].placement
    assert len(goal.remaining_steps([])) == 2
    assert len(goal.remaining_steps([first])) == 1
    assert goal.remaining_steps([first])[0].placement == goal.steps[1].placement


def test_the_task_set_is_three_tasks() -> None:
    assert list(TASKS) == ["placement", "pick_and_place", "build_2"]
    assert TASKS["placement"].cycles == 1 and not TASKS["placement"].pick
    assert TASKS["pick_and_place"].cycles == 1 and TASKS["pick_and_place"].pick
    assert TASKS["build_2"].cycles == 2 and TASKS["build_2"].pick
    assert TASKS["build_2"].command == BUILD_COMMAND
    assert BASE_PLACED and all(p.z == 0 for p in BASE_PLACED)


# ------------------------------------------------------------------------------------- the seeds ---


def test_performing_seeds_never_touch_the_evaluation_set() -> None:
    every = set()
    for task in TASKS:
        for round_index in range(12):
            seeds = perform_seeds(task, round_index, 10)
            assert len(set(seeds)) == 10
            assert not set(seeds) & set(EVAL_SEEDS)
            assert min(seeds) >= FIRST_PERFORM_SEED
            assert not set(seeds) & every, "two rounds drew the same world"
            every |= set(seeds)


def test_performing_seeds_are_deterministic() -> None:
    assert perform_seeds("placement", 3, 4) == perform_seeds("placement", 3, 4)
    assert perform_seeds("placement", 3, 4) != perform_seeds("pick_and_place", 3, 4)


# ------------------------------------------------------------------------------------ the record ---


def test_an_episode_succeeds_only_when_every_brick_is_on_its_cell() -> None:
    assert _record(bricks=[_brick()]).success
    assert _record(bricks=[_brick(), _brick(1)]).success
    assert not _record(bricks=[_brick(), _brick(1, seated=False)]).success
    assert not _record(bricks=[_brick(at_cell=False)]).success
    assert not _record(bricks=[]).success


def test_the_csv_row_has_the_same_columns_whatever_the_task() -> None:
    one = _record(bricks=[_brick()]).row()
    two = _record(task="build_2", bricks=[_brick(), _brick(1)]).row()
    assert list(one) == list(two) == list(CSV_FIELDS)
    assert one["brick1_cell"] == ""
    assert two["brick1_on_target"] is True


def test_the_failure_class_folds_in_what_the_scorer_saw() -> None:
    assert failure_class("", [_brick()]) == ""
    assert failure_class("", [_brick(at_cell=False)]) == "seated on the wrong cell"
    assert failure_class("", [_brick(seated=False)]) == "partial seat"
    assert failure_class("the grasp stopped 4.2 mm short twice", []) == "grasp short"
    assert "stalled" in failure_class("held past max_hold with an error", [])


# ------------------------------------------------------------------------------------ the worlds ---


@pytest.mark.slow
def test_the_first_cycle_builds_the_world_the_pick_task_measures_on() -> None:
    """`cycle_world` is `build_pick_world` with a plate that may already carry bricks. With
    nothing extra on it the two have to be the same world, or the campaign is not measuring the
    published task."""
    import numpy as np

    from brick_builder.action.worlds import build_pick_world
    from brick_builder.campaign.tasks import cycle_world

    mine, d_xy, d_yaw, b_xy = cycle_world(3, 0, BASE_PLACED)
    theirs, t_xy, t_yaw, tb_xy, _ = build_pick_world(3)
    assert np.allclose(d_xy, t_xy) and d_yaw == pytest.approx(t_yaw)
    assert np.allclose(b_xy, tb_xy)
    assert np.allclose(mine.data.qpos, theirs.data.qpos, atol=1e-9)


@pytest.mark.slow
def test_the_second_cycle_puts_the_first_brick_on_the_plate() -> None:
    from brick_builder.bricks.structures import Placement
    from brick_builder.campaign.tasks import cycle_world

    placed = list(BASE_PLACED) + [Placement(h=2, w=2, x=4, y=3, z=0, colour="blue")]
    scene, *_ = cycle_world(3, 1, placed)
    assert len(scene.placed_names) == 3
