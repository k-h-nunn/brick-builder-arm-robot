"""Tests for `brick_builder.language.compiler.compile_command`.

`ACCEPTED`, `AMBIGUOUS`, and `INVALID` below are the bulk command-case coverage (60+ commands
between them); the functions after them are end-to-end tests that need more than one line to set
up - a `current_grid`, an `Inventory`, or checking something about the `Goal` that comes back
rather than just whether compilation succeeded.

`AMBIGUOUS` and `INVALID` both mean `understood=False`: the split just documents *why* - a
genuinely ambiguous reading (two matching bricks, a target that is already taken) versus a
vocabulary or completeness problem (an unknown colour, a missing size, an off-the-plate cell).
Either way, `compile_command` never guesses: `goal` is `None` and `paraphrase` is empty.
"""

from __future__ import annotations

import pytest

from brick_builder.bricks.structures import Placement, parse_structure
from brick_builder.language.compiler import compile_command
from brick_builder.language.goal import Goal, Inventory, brick_kind

# --------------------------------------------------------------------------------------- #
# Accepted commands: (command text, expected placement text - structures.py's own format).
# --------------------------------------------------------------------------------------- #

ACCEPTED: list[tuple[str, str]] = [
    # Absolute placement: digits, "N by M", "(x,y)", 1-based column/row, explicit layer.
    ("put the red 2x4 brick at column 3 row 2", "2x4 (2,1,0) red"),
    ("place a blue 2 by 2 on the plate at (4,5)", "2x2 (4,5,0) blue"),
    ("put a yellow 4x2 brick at column 1 row 1", "4x2 (0,0,0) yellow"),
    ("put the green 2x2 at column 7 row 7", "2x2 (6,6,0) green"),
    ("put a white 2x4 brick at (0,0)", "2x4 (0,0,0) white"),
    ("put an orange 2x2 at column 1 row 1 layer 1", "2x2 (0,0,0) orange"),
    ("put a red 4 by 2 brick at column 1 row 1", "4x2 (0,0,0) red"),
    # Sequences: "then", "and then", a numbered list, ";".
    (
        "put a red 2x2 at column 1 row 1 then put a blue 2x2 at column 3 row 1",
        "2x2 (0,0,0) red\n2x2 (2,0,0) blue",
    ),
    (
        "put a red 2x2 at column 1 row 1 and then put a blue 2x2 at column 3 row 1",
        "2x2 (0,0,0) red\n2x2 (2,0,0) blue",
    ),
    (
        "1. put a red 2x2 at column 1 row 1. 2. put a blue 2x2 at column 3 row 1.",
        "2x2 (0,0,0) red\n2x2 (2,0,0) blue",
    ),
    (
        "put a red 2x2 at column 1 row 1; put a blue 2x2 at column 3 row 1",
        "2x2 (0,0,0) red\n2x2 (2,0,0) blue",
    ),
    # Orientation.
    ("put a red 2x4 brick at column 1 row 1 lengthwise", "2x4 (0,0,0) red"),
    ("put a red 2x4 brick at column 1 row 1 across", "4x2 (0,0,0) red"),
    ("put a red 2x4 brick at column 1 row 1 rotated", "4x2 (0,0,0) red"),
    # Colour/size synonyms and quantities.
    ("put a red two by four brick at column 1 row 1", "2x4 (0,0,0) red"),
    ("put a red two-by-four brick at column 1 row 1", "2x4 (0,0,0) red"),
    ("put a red square brick at column 1 row 1", "2x2 (0,0,0) red"),
    ("put a red long brick at column 1 row 1", "2x4 (0,0,0) red"),
    # Polite filler.
    ("could you please put the red 2x2 brick at column 1 row 1, thanks", "2x2 (0,0,0) red"),
    ("please put a red 2x2 at column 1 row 1", "2x2 (0,0,0) red"),
    # The middle of the plate, and every corner.
    ("put a red 2x2 in the middle of the plate", "2x2 (3,3,0) red"),
    ("put a red 2x2 in the corner", "2x2 (0,0,0) red"),
    ("put a red 2x2 in the far right corner", "2x2 (6,6,0) red"),
    ("put a red 2x2 in the near left corner", "2x2 (0,0,0) red"),
    ("put a red 2x2 in the far left corner", "2x2 (0,6,0) red"),
    ("put a red 2x2 in the near right corner", "2x2 (6,0,0) red"),
    # Named structures.
    (
        "a tower of four blue 2x2 bricks at column 1 row 1",
        "2x2 (0,0,0) blue\n2x2 (0,0,1) blue\n2x2 (0,0,2) blue\n2x2 (0,0,3) blue",
    ),
    ("a tower of three 2x2 bricks", "2x2 (3,3,0) red\n2x2 (3,3,1) red\n2x2 (3,3,2) red"),
    (
        "a tower of five 2x2 bricks",
        "2x2 (3,3,0) red\n2x2 (3,3,1) red\n2x2 (3,3,2) red\n2x2 (3,3,3) red\n2x2 (3,3,4) red",
    ),
    (
        "a staircase of two blue 2x2 steps at column 1 row 1",
        "2x2 (0,0,0) blue\n2x2 (2,0,0) blue\n2x2 (2,0,1) blue",
    ),
    (
        "a staircase with three steps",
        "2x2 (1,3,0) red\n2x2 (3,3,0) red\n2x2 (3,3,1) red\n2x2 (5,3,0) red\n2x2 (5,3,1) red\n2x2 (5,3,2) red",
    ),
    ("a wall of three red 2x2 bricks at column 1 row 1", "2x2 (0,0,0) red\n2x2 (2,0,0) red\n2x2 (4,0,0) red"),
    ("a wall of two red 2x4 bricks at column 1 row 1 rotated", "4x2 (0,0,0) red\n4x2 (2,0,0) red"),
    ("a wall of two 2x4 bricks", "2x4 (0,3,0) red\n2x4 (4,3,0) red"),
    # Relative placements resolved against a brick placed earlier in the *same* command.
    ("put a red 2x2 at column 1 row 1 then put a blue 2x2 on top of the red brick", "2x2 (0,0,0) red\n2x2 (0,0,1) blue"),
    ("put a red 2x2 at column 4 row 4 then put a blue 2x2 next to the red brick", "2x2 (3,3,0) red\n2x2 (5,3,0) blue"),
    ("put a red 2x2 at column 4 row 4 then put a blue 2x2 left of the red brick", "2x2 (3,3,0) red\n2x2 (1,3,0) blue"),
    ("put a red 2x2 at column 4 row 4 then put a blue 2x2 right of the red brick", "2x2 (3,3,0) red\n2x2 (5,3,0) blue"),
    ("put a red 2x2 at column 4 row 4 then put a blue 2x2 behind the red brick", "2x2 (3,3,0) red\n2x2 (3,5,0) blue"),
    (
        "put a red 2x2 at column 4 row 4 then put a blue 2x2 in front of the red brick",
        "2x2 (3,3,0) red\n2x2 (3,1,0) blue",
    ),
]

assert len(ACCEPTED) >= 30


@pytest.mark.parametrize("text, expected_text", ACCEPTED, ids=[t for t, _ in ACCEPTED])
def test_accepted_command_compiles_to_expected_placements(text, expected_text):
    result = compile_command(text)
    assert result.understood, f"expected understood=True, problems={result.problems}"
    assert result.goal is not None
    assert result.goal.to_text() == expected_text
    assert result.problems == []
    assert result.paraphrase  # never empty when understood
    assert result.goal.is_valid()


# --------------------------------------------------------------------------------------- #
# Ambiguous commands: more than one honest reading, or a target that conflicts with something
# else in the very same command. `compile_command` refuses rather than picking one.
# --------------------------------------------------------------------------------------- #

AMBIGUOUS: list[str] = [
    "put the red 2x2 at column 1 row 1 and then put the red 2x2 at column 1 row 1",
    "put a blue 2x2 on top of the brick",
    "put a blue 2x2 left of the brick",
    "put a blue 2x2 behind the brick",
    "put the second red 2x2",
    "1) put a red 2x2 at column 1 row 1 2) put a blue 2x2 at column 3 row 1",
]


@pytest.mark.parametrize("text", AMBIGUOUS)
def test_ambiguous_command_is_not_understood(text):
    result = compile_command(text)
    assert not result.understood
    assert result.goal is None
    assert result.paraphrase == ""
    assert result.problems


# --------------------------------------------------------------------------------------- #
# Invalid commands: an unknown word, a missing required piece, or a structurally impossible
# placement (off the plate, unsupported).
# --------------------------------------------------------------------------------------- #

INVALID: list[str] = [
    "",
    "asdf jkl nonsense",
    "put the purple 2x2 at column 1 row 1",
    "put the brick at column 1 row 1",
    "put the red brick at column 1 row 1",
    "put the red 3x3 brick at column 1 row 1",
    "put the red 2x2 at column 1",
    "put the red 2x2 at row 1",
    "put a green 2x2 at column 0 row 1",
    "put a red 2x2 at column 9 row 1",
    "put a red 2x2 at column 1 row 9",
    "put a red 2x2 at column 1 row 1 layer 2",
    "a tower of 2x2 bricks",
    "a staircase with steps",
    "a wall of bricks",
]


@pytest.mark.parametrize("text", INVALID)
def test_invalid_command_is_not_understood(text):
    result = compile_command(text)
    assert not result.understood
    assert result.goal is None
    assert result.paraphrase == ""
    assert result.problems


def test_the_command_case_lists_cover_at_least_sixty_cases():
    assert len(ACCEPTED) + len(AMBIGUOUS) + len(INVALID) >= 60


# --------------------------------------------------------------------------------------- #
# Specific problem wording, so a regression can't silently swap in a vaguer message.
# --------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, needle",
    [
        ("put the purple 2x2 at column 1 row 1", "purple"),
        ("put the brick at column 1 row 1", "colour"),
        ("put the red brick at column 1 row 1", "size"),
        ("put the red 3x3 brick at column 1 row 1", "not one of the available brick sizes"),
        ("put the red 2x2 at column 1", "column and a row"),
        ("put a red 2x2 at column 9 row 1", "would overhang the plate"),
        ("put a red 2x2 at column 1 row 1 layer 2", "unsupported"),
        ("a tower of 2x2 bricks", "missing count"),
        ("put the red 2x2 at column 1 row 1 and then put the red 2x2 at column 1 row 1", "occupied"),
        ("put a blue 2x2 on top of the brick", "which brick"),
    ],
)
def test_problem_messages_are_specific(text, needle):
    result = compile_command(text)
    assert not result.understood
    assert any(needle in p for p in result.problems), result.problems


# --------------------------------------------------------------------------------------- #
# Determinism: same input, same output, every time.
# --------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", [t for t, _ in ACCEPTED[:8]] + AMBIGUOUS[:3] + INVALID[:3])
def test_compile_command_is_deterministic(text):
    results = [compile_command(text) for _ in range(5)]
    first = results[0]
    for other in results[1:]:
        assert other.understood == first.understood
        assert other.problems == first.problems
        assert other.paraphrase == first.paraphrase
        if first.goal is not None:
            assert other.goal == first.goal
        else:
            assert other.goal is None


def test_compile_command_is_deterministic_with_inventory_and_current_grid():
    current = parse_structure("2x2 (0,0,0) red")
    inventory = Inventory(counts={((2, 2), "blue"): 5})
    text = "put a blue 2x2 on top of the red brick"
    r1 = compile_command(text, inventory=inventory, current_grid=current)
    r2 = compile_command(text, inventory=inventory, current_grid=current)
    assert r1 == r2


# --------------------------------------------------------------------------------------- #
# Round trip: Goal -> placement text -> Goal, for a multi-step compiled command.
# --------------------------------------------------------------------------------------- #


def test_goal_round_trips_through_its_own_text_format():
    result = compile_command(
        "put a red 2x2 at column 1 row 1 then put a blue 2x4 at column 3 row 1 then "
        "put a yellow 2x2 on top of the red brick"
    )
    assert result.understood
    text = result.goal.to_text()
    reparsed = parse_structure(text)
    assert reparsed == result.goal.placements
    rebuilt = Goal.from_placements(reparsed, grid_shape=result.goal.grid_shape)
    assert rebuilt.to_text() == text


# --------------------------------------------------------------------------------------- #
# The expected-grid sequence of a multi-step command, read back off the compiled Goal.
# --------------------------------------------------------------------------------------- #


def test_expected_grid_sequence_of_a_multi_step_command():
    result = compile_command(
        "put a red 2x2 at column 1 row 1 then put a blue 2x2 at column 3 row 1 then "
        "put a yellow 2x2 on top of the red brick"
    )
    assert result.understood
    goal = result.goal
    assert len(goal.steps) == 3

    after0 = goal.expected_grid_after(0)
    assert set(after0) == {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)}
    assert all(cell.colour == "red" for cell in after0.values())

    after1 = goal.expected_grid_after(1)
    assert {(2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0)} <= set(after1)
    assert (0, 0, 0) in after1  # cumulative: step 0's brick is still there

    final = goal.final_grid
    assert final == goal.expected_grid_after(2)
    assert final[(0, 0, 1)].colour == "yellow"
    # the plate footprint under the yellow brick is unchanged by adding it on top
    assert final[(0, 0, 0)].colour == "red"


# --------------------------------------------------------------------------------------- #
# remaining_steps after a simulated disturbance, for a command compiled end-to-end.
# --------------------------------------------------------------------------------------- #


def test_remaining_steps_after_a_brick_is_removed_from_a_compiled_goal():
    result = compile_command(
        "put a red 2x2 at column 1 row 1 then put a blue 2x2 at column 3 row 1 then "
        "put a yellow 2x2 on top of the red brick"
    )
    assert result.understood
    goal = result.goal

    # nothing has been disturbed: everything the goal asked for is observed
    assert goal.remaining_steps(goal.placements) == []

    # the yellow brick (step 2) gets knocked off; red and blue (steps 0-1) are still there
    observed = goal.placements[:2]
    remaining = goal.remaining_steps(observed)
    assert len(remaining) == 1
    assert remaining[0].placement.colour == "yellow"
    assert remaining[0] == goal.steps[2]


# --------------------------------------------------------------------------------------- #
# Relative placements resolved against `current_grid` (bricks the command did not itself place).
# --------------------------------------------------------------------------------------- #


def test_relative_placement_resolves_against_current_grid():
    current = parse_structure("2x2 (0,0,0) red")
    result = compile_command("put a blue 2x2 on top of the red brick", current_grid=current)
    assert result.understood
    assert result.goal.to_text() == "2x2 (0,0,1) blue"


def test_ambiguous_reference_with_two_matching_bricks_on_current_grid():
    current = parse_structure("2x2 (0,0,0) red\n2x2 (5,5,0) red")
    result = compile_command("put a yellow 2x2 on top of the red brick", current_grid=current)
    assert not result.understood
    assert any("more than one red brick" in p for p in result.problems)


def test_no_matching_reference_brick_on_current_grid():
    result = compile_command("put a yellow 2x2 on top of the red brick", current_grid=[])
    assert not result.understood
    assert any("no" in p and "red brick" in p for p in result.problems)


def test_ambiguous_reference_disambiguated_by_size():
    current = parse_structure("2x2 (0,0,0) red\n2x4 (4,4,0) red")
    result = compile_command("put a blue 2x2 on top of the red 2x4 brick", current_grid=current)
    assert result.understood
    assert result.goal.to_text() == "2x2 (4,4,1) blue"


def test_target_already_occupied_on_current_grid_is_refused():
    current = parse_structure("2x2 (0,0,0) blue")
    result = compile_command("put a red 2x2 at column 1 row 1", current_grid=current)
    assert not result.understood
    assert any("occupied" in p for p in result.problems)


def test_placement_supported_by_current_grid_is_not_falsely_flagged_unsupported():
    """A brick resting on something from `current_grid` (not on another step in this command)
    must not be flagged unsupported just because `current_grid` itself is not part of the Goal's
    own `steps` - a regression test for exactly that bug."""

    current = parse_structure("2x2 (0,0,0) blue")
    result = compile_command("put a red 2x2 on top of the blue brick", current_grid=current)
    assert result.understood, result.problems
    assert result.goal.to_text() == "2x2 (0,0,1) red"
    # `Goal.is_valid()` with no current_grid correctly says the *goal alone* would not stand up
    # (nothing in its own steps is at layer 0) - it has no way to know it is meant to rest on
    # something from current_grid. That is expected: current_grid is compile_command's concern,
    # not Goal's own, and is exactly why compile_command does not lean on `goal.problems()` for
    # support/bounds once current_grid is involved (see its comment on the point).
    assert not result.goal.is_valid()


def test_named_structure_can_be_placed_relative_to_current_grid():
    current = parse_structure("2x2 (0,0,0) red")
    result = compile_command("a tower of two blue 2x2 bricks on top of the red brick", current_grid=current)
    assert result.understood
    assert result.goal.to_text() == "2x2 (0,0,1) blue\n2x2 (0,0,2) blue"


# --------------------------------------------------------------------------------------- #
# Inventory: bricks that are not (or are not sufficiently) in stock.
# --------------------------------------------------------------------------------------- #


def test_inventory_shortage_blocks_a_single_placement():
    inventory = Inventory(counts={(brick_kind(2, 4), "red"): 0})
    result = compile_command("put the red 2x4 at column 1 row 1", inventory=inventory)
    assert not result.understood
    assert any("red 2x4" in p for p in result.problems)


def test_inventory_shortage_blocks_a_structure_needing_more_than_is_in_stock():
    inventory = Inventory(counts={(brick_kind(2, 2), "red"): 3})
    result = compile_command("a tower of four red 2x2 bricks", inventory=inventory)
    assert not result.understood
    assert any("needs 4" in p for p in result.problems)


def test_inventory_with_sufficient_stock_succeeds():
    inventory = Inventory(counts={(brick_kind(2, 2), "red"): 3})
    result = compile_command("a tower of three red 2x2 bricks", inventory=inventory)
    assert result.understood
    assert result.goal.is_valid(inventory)


def test_no_inventory_given_means_unlimited_stock():
    result = compile_command("a tower of twenty red 2x2 bricks")
    assert result.understood
    assert len(result.goal.steps) == 20


# --------------------------------------------------------------------------------------- #
# Structures default colour and size (disclosed in the paraphrase); plain placements never do.
# --------------------------------------------------------------------------------------- #


def test_structure_without_colour_or_size_defaults_to_red_2x2_and_says_so():
    result = compile_command("a wall of two bricks")
    assert result.understood
    # two 2x2 bricks side by side span 4 studs, centred on an 8-wide plate: columns 3 and 5
    # (grid x=2 and x=4), row 4 (grid y=3).
    assert result.goal.to_text() == "2x2 (2,3,0) red\n2x2 (4,3,0) red"
    assert "assuming red" in result.paraphrase
    assert "assuming 2x2" in result.paraphrase


def test_plain_placement_never_defaults_colour_or_size():
    assert not compile_command("put a brick at column 1 row 1").understood
    assert not compile_command("put a red brick at column 1 row 1").understood
    assert not compile_command("put a 2x2 brick at column 1 row 1").understood


# --------------------------------------------------------------------------------------- #
# The paraphrase discloses the assumptions this compiler is allowed to make (never colour/size).
# --------------------------------------------------------------------------------------- #


def test_paraphrase_discloses_one_based_column_row_assumption():
    result = compile_command("put a red 2x2 at column 3 row 2")
    assert result.understood
    assert "counted from 1" in result.paraphrase
    assert "x=2, y=1" in result.paraphrase


def test_paraphrase_discloses_default_location_assumption():
    result = compile_command("put a red 2x2")
    assert result.understood
    assert "no location was given" in result.paraphrase
