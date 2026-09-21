"""Tests for `brick_builder.language.goal`: the `Goal` type in isolation, without the compiler.

These build `Goal`s directly out of `structures.Placement`s (the same way a hand-written test
fixture or another component further down the pipeline would), rather than through
`compile_command` - see `test_compiler.py` for compiler-level (and compiler-plus-goal,
end-to-end) tests.
"""

from __future__ import annotations

import pytest

from brick_builder.bricks.structures import Placement, format_structure, parse_structure
from brick_builder.language.goal import (
    GRID_SHAPE,
    KNOWN_SIZES,
    CellExpectation,
    Goal,
    Inventory,
    Step,
    brick_kind,
)


def test_known_sizes_are_the_three_duplo_footprints():
    assert set(KNOWN_SIZES) == {(2, 2), (2, 4), (4, 2)}


def test_brick_kind_is_symmetric_and_orientation_independent():
    assert brick_kind(2, 4) == brick_kind(4, 2) == (2, 4)
    assert brick_kind(2, 2) == (2, 2)


# --------------------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------------------- #


def test_inventory_default_is_unlimited():
    inventory = Inventory()
    assert inventory.available(2, 4, "red", needed=1000)
    assert inventory.shortages([Placement(2, 4, 0, 0, 0, "red")] * 50) == []


def test_inventory_available_counts_by_orientation_independent_kind():
    inventory = Inventory(counts={((2, 4), "red"): 2})
    assert inventory.available(2, 4, "red")
    assert inventory.available(4, 2, "red")  # same kind, rotated
    assert not inventory.available(2, 4, "blue")
    assert not inventory.available(2, 2, "red")


def test_inventory_demand_counts_across_orientations():
    placements = [Placement(2, 4, 0, 0, 0, "red"), Placement(4, 2, 4, 0, 0, "red")]
    inventory = Inventory()
    assert inventory.demand(placements) == {((2, 4), "red"): 2}


def test_inventory_shortages_reports_missing_and_short_stock():
    placements = [Placement(2, 2, 0, 0, 0, "red"), Placement(2, 2, 2, 0, 0, "red"), Placement(2, 2, 4, 0, 0, "blue")]
    inventory = Inventory(counts={((2, 2), "red"): 1})
    problems = inventory.shortages(placements)
    assert any("red 2x2" in p and "needs 2" in p for p in problems)
    assert any("blue 2x2" in p and "0" in p for p in problems)


def test_inventory_shortages_empty_when_stock_is_sufficient():
    placements = [Placement(2, 2, 0, 0, 0, "red")] * 3
    inventory = Inventory(counts={((2, 2), "red"): 3})
    assert inventory.shortages(placements) == []


# --------------------------------------------------------------------------------------- #
# Goal construction, `to_text`, and the Goal -> text -> Goal round trip
# --------------------------------------------------------------------------------------- #


def _sample_goal() -> Goal:
    placements = [
        Placement(2, 2, 0, 0, 0, "red"),
        Placement(2, 4, 2, 0, 0, "blue"),
        Placement(2, 2, 0, 0, 1, "yellow"),
    ]
    return Goal.from_placements(placements, grid_shape=GRID_SHAPE)


def test_goal_placements_property_matches_step_order():
    goal = _sample_goal()
    assert [(p.h, p.w, p.colour) for p in goal.placements] == [(2, 2, "red"), (2, 4, "blue"), (2, 2, "yellow")]


def test_goal_to_text_matches_structures_format():
    goal = _sample_goal()
    assert goal.to_text() == format_structure(goal.placements)
    assert goal.to_text() == "2x2 (0,0,0) red\n2x4 (2,0,0) blue\n2x2 (0,0,1) yellow"


def test_goal_round_trips_through_text():
    goal = _sample_goal()
    reparsed = parse_structure(goal.to_text())
    assert reparsed == goal.placements
    rebuilt = Goal.from_placements(reparsed, grid_shape=goal.grid_shape)
    assert rebuilt.placements == goal.placements
    assert rebuilt.to_text() == goal.to_text()


# --------------------------------------------------------------------------------------- #
# expected_grid_after / final_grid
# --------------------------------------------------------------------------------------- #


def test_expected_grid_after_before_any_step_is_empty():
    goal = _sample_goal()
    assert goal.expected_grid_after(-1) == {}


def test_expected_grid_after_accumulates_one_step_at_a_time():
    goal = _sample_goal()

    after0 = goal.expected_grid_after(0)
    assert set(after0) == {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)}
    assert after0[(0, 0, 0)] == CellExpectation(size="2x2", colour="red", step_index=0)

    after1 = goal.expected_grid_after(1)
    assert (2, 0, 0) in after1 and (3, 1, 0) in after1
    assert after1[(2, 0, 0)] == CellExpectation(size="2x4", colour="blue", step_index=1)
    # step 0's cells are still there too - this is cumulative, not a diff
    assert after1[(0, 0, 0)].colour == "red"


def test_expected_grid_after_final_step_equals_final_grid():
    goal = _sample_goal()
    last = len(goal.steps) - 1
    assert goal.expected_grid_after(last) == goal.final_grid
    assert goal.final_grid[(0, 0, 1)] == CellExpectation(size="2x2", colour="yellow", step_index=2)


def test_final_grid_of_an_empty_goal_is_empty():
    assert Goal(steps=()).final_grid == {}


def test_expected_grid_after_rejects_out_of_range_index():
    goal = _sample_goal()
    with pytest.raises(IndexError):
        goal.expected_grid_after(len(goal.steps))
    with pytest.raises(IndexError):
        goal.expected_grid_after(-2)


# --------------------------------------------------------------------------------------- #
# remaining_steps: replanning from what the controller actually observes
# --------------------------------------------------------------------------------------- #


def test_remaining_steps_is_empty_when_everything_observed():
    goal = _sample_goal()
    assert goal.remaining_steps(goal.placements) == []


def test_remaining_steps_is_everything_when_nothing_observed():
    goal = _sample_goal()
    assert goal.remaining_steps(None) == list(goal.steps)
    assert goal.remaining_steps([]) == list(goal.steps)


def test_remaining_steps_after_a_brick_is_removed():
    """The disturbance case: the top (yellow) brick is knocked off after steps 0-1 were done."""

    goal = _sample_goal()
    observed_after_disturbance = goal.placements[:2]  # red and blue still there, yellow gone
    remaining = goal.remaining_steps(observed_after_disturbance)
    assert [s.placement.colour for s in remaining] == ["yellow"]
    assert remaining[0] == goal.steps[2]


def test_remaining_steps_redoes_a_step_whose_support_was_removed():
    """If the base brick a later step depends on disappears, that later step is judged purely by
    its own footprint's colour - a documented simplification, not a physics simulation - so it
    is *not* automatically added back just because its support vanished."""

    goal = _sample_goal()
    observed_missing_base = [goal.placements[1], goal.placements[2]]  # red (step 0) is gone
    remaining = goal.remaining_steps(observed_missing_base)
    assert [s.placement.colour for s in remaining] == ["red"]


def test_remaining_steps_preserves_original_order():
    placements = [Placement(2, 2, x, 0, 0, "red") for x in (0, 2, 4, 6)]
    goal = Goal.from_placements(placements)
    # only the 2nd and 4th are observed
    observed = [placements[1], placements[3]]
    remaining = goal.remaining_steps(observed)
    assert [s.placement.x for s in remaining] == [0, 4]


# --------------------------------------------------------------------------------------- #
# problems / is_valid: reused structures.py checks, plus inventory
# --------------------------------------------------------------------------------------- #


def test_goal_problems_empty_for_a_valid_goal():
    assert _sample_goal().problems() == []
    assert _sample_goal().is_valid()


def test_goal_problems_flags_overlap():
    goal = Goal.from_placements([Placement(2, 4, 0, 0, 0, "red"), Placement(2, 2, 1, 0, 0, "blue")])
    problems = goal.problems()
    assert any("overlap" in p for p in problems)
    assert not goal.is_valid()


def test_goal_problems_flags_out_of_grid():
    goal = Goal.from_placements([Placement(2, 4, 7, 0, 0, "red")], grid_shape=(8, 8))
    assert any("out of the" in p for p in goal.problems())


def test_goal_problems_flags_unsupported():
    goal = Goal.from_placements([Placement(2, 2, 0, 0, 1, "red")])  # layer 1, nothing at layer 0
    assert any("unsupported" in p for p in goal.problems())


def test_goal_problems_flags_inventory_shortage():
    goal = Goal.from_placements([Placement(2, 2, 0, 0, 0, "red")])
    inventory = Inventory(counts={((2, 2), "blue"): 5})  # no red at all
    problems = goal.problems(inventory)
    assert any("red 2x2" in p for p in problems)
    assert not goal.is_valid(inventory)


def test_goal_default_grid_shape_is_the_stock_eight_by_eight_plate():
    assert GRID_SHAPE == (8, 8)
    assert Goal(steps=()).grid_shape == (8, 8)


def test_step_defaults_source_to_tray_and_relation_to_none():
    step = Step(placement=Placement(2, 2, 0, 0, 0, "red"))
    assert step.source == "tray"
    assert step.relation is None
