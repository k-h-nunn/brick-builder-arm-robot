"""The `Goal` type: what one compiled command asks the closed-loop controller to build.

A `Goal` is an ordered list of `Step`s. Each `Step` wraps one `Placement` (see
`brick_builder.bricks.structures`, which owns the brick-on-a-stud-grid representation, the placement
text format, and the structural validity checks - this module reuses all three rather than
inventing new ones) plus bookkeeping: where the brick comes from, and, if the compiler worked the
target cell out from something already on the plate ("on top of the red brick"), a plain-English
note of how.

A `Goal` is inert data. It does not simulate anything, look at a camera, or talk to the
controller. The controller (outside this package) reads `steps` in order, calls
`expected_grid_after` to know what the plate should look like at each point along the way, and
calls `remaining_steps` against what it actually observes to re-plan around a disturbance -
see `docs/guide/11-commands.md` for the placeholder/interface story this is part of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..bricks.structures import Placement, format_structure, occupancy_grid, validate_structure

# The stock hardware is one fixed 8x8-stud baseplate; the language layer does not need to make
# this configurable, only the CLI's `--grid` (what is already on the plate) is a variable.
GRID_SHAPE: tuple[int, int] = (8, 8)

# The three DUPLO footprints this placeholder compiler knows how to talk about, per the
# design statement. Keyed exactly as `(h, w)` appear in the "{h}x{w}" placement text - `h` is the
# row/y extent, `w` is the column/x extent (see `structures.Placement`'s own docstring).
KNOWN_SIZES: tuple[tuple[int, int], ...] = ((2, 2), (2, 4), (4, 2))


def brick_kind(h: int, w: int) -> tuple[int, int]:
    """The orientation-independent "kind" of a brick footprint.

    A 2x4 and a 4x2 are the same physical brick lying two different ways, so they draw from the
    same tray pocket; `brick_kind(2, 4) == brick_kind(4, 2) == (2, 4)` (the smaller number
    first). Two bricks of the same kind can still be placed at different orientations.
    """

    return (h, w) if h <= w else (w, h)


@dataclass(frozen=True)
class Inventory:
    """What is available in the tray: a count of each brick kind/colour combination.

    Keyed by `(brick_kind(h, w), colour)`. `counts is None` (the default) means "unlimited" - a
    placeholder tray that never runs out, which is the right default for a compiler that mostly
    cannot see the real tray. Pass an explicit `counts` mapping to make "no such brick in the
    inventory" a real, checkable problem instead. A kind/colour combination missing from
    `counts` means zero on hand, not unlimited - an explicit `counts` mapping is the complete
    stock list, not a set of overrides.
    """

    counts: Mapping[tuple[tuple[int, int], str], int] | None = None

    def available(self, h: int, w: int, colour: str, needed: int = 1) -> bool:
        if self.counts is None:
            return True
        have = self.counts.get((brick_kind(h, w), colour), 0)
        return have >= needed

    def demand(self, placements: Sequence[Placement]) -> dict[tuple[tuple[int, int], str], int]:
        """How many of each kind/colour a list of placements would use in total."""

        out: dict[tuple[tuple[int, int], str], int] = {}
        for p in placements:
            key = (brick_kind(p.h, p.w), p.colour)
            out[key] = out.get(key, 0) + 1
        return out

    def shortages(self, placements: Sequence[Placement]) -> list[str]:
        """Human-readable problems, one per kind/colour these placements need more of than the
        tray has. Always empty when `counts is None` (an unlimited tray has no shortages)."""

        if self.counts is None:
            return []
        problems: list[str] = []
        for (kind, colour), need in self.demand(placements).items():
            have = self.counts.get((kind, colour), 0)
            if need > have:
                h, w = kind
                plural = "" if have == 1 else "s"
                problems.append(
                    f"the tray only has {have} {colour} {h}x{w} brick{plural}, "
                    f"but this command needs {need}"
                )
        return problems


@dataclass(frozen=True)
class Step:
    """One brick placement, plus where it comes from and, if it was derived from something
    already on the plate, a plain-English note of how the target cell was worked out."""

    placement: Placement
    source: str = "tray"
    relation: str | None = None


@dataclass(frozen=True)
class CellExpectation:
    """What one stud cell should hold, per `Goal.expected_grid_after`."""

    size: str
    colour: str
    step_index: int


@dataclass(frozen=True)
class Goal:
    """An ordered plan: the bricks one compiled command asks for, in the order to place them."""

    steps: tuple[Step, ...]
    grid_shape: tuple[int, int] = GRID_SHAPE

    @property
    def placements(self) -> list[Placement]:
        return [step.placement for step in self.steps]

    @classmethod
    def from_placements(
        cls,
        placements: Sequence[Placement],
        *,
        grid_shape: tuple[int, int] = GRID_SHAPE,
        source: str = "tray",
    ) -> "Goal":
        """Build a `Goal` out of bare placements (e.g. bricks read back with
        `structures.parse_structure`), with no relation notes."""

        return cls(steps=tuple(Step(placement=p, source=source) for p in placements), grid_shape=grid_shape)

    def expected_grid_after(self, step_index: int) -> dict[tuple[int, int, int], CellExpectation]:
        """The occupancy grid - size and colour per stud cell, on every layer - once steps
        `0 .. step_index` (inclusive, 0-based) have all been carried out.

        `step_index == -1` is the empty plate, before the first step; `step_index == len(steps)
        - 1` is the finished structure (also available as `final_grid`).
        """

        if not (-1 <= step_index < len(self.steps)):
            raise IndexError(f"step_index {step_index} out of range for {len(self.steps)} step(s)")
        grid: dict[tuple[int, int, int], CellExpectation] = {}
        for idx, step in enumerate(self.steps[: step_index + 1]):
            p = step.placement
            for gx, gy in p.cells():
                grid[(gx, gy, p.z)] = CellExpectation(size=f"{p.h}x{p.w}", colour=p.colour, step_index=idx)
        return grid

    @property
    def final_grid(self) -> dict[tuple[int, int, int], CellExpectation]:
        return self.expected_grid_after(len(self.steps) - 1) if self.steps else {}

    def remaining_steps(self, observed: Sequence[Placement] | None) -> list[Step]:
        """Which steps are not done yet, judged against what is actually on the plate.

        `observed` is a list of `Placement`s - the same representation `current_grid` uses
        elsewhere in this package - describing every brick the controller currently sees. A step
        counts as already done only if *every* cell of its footprint, at its own layer, is
        covered by a brick of the right colour in `observed`; a step whose brick was knocked off,
        removed, or never finished comes back out of this list, in its original order, so the
        controller can redo it.
        """

        colour_at = _colour_grid(observed or [])
        return [step for step in self.steps if not _placement_is_present(step.placement, colour_at)]

    def to_text(self) -> str:
        return format_structure(self.placements)

    def problems(self, inventory: Inventory | None = None) -> list[str]:
        """Every reason this goal could not actually be built: overlaps, off the plate,
        unsupported bricks (all from `structures.validate_structure`), plus, when an inventory is
        given, any brick the tray does not have enough of."""

        problems = list(validate_structure(self.placements, grid_shape=self.grid_shape))
        if inventory is not None:
            problems.extend(inventory.shortages(self.placements))
        return problems

    def is_valid(self, inventory: Inventory | None = None) -> bool:
        return not self.problems(inventory)


def _colour_grid(placements: Sequence[Placement]) -> dict[tuple[int, int, int], str]:
    """Cell -> colour, for a list of placements (reuses `structures.occupancy_grid`'s indices
    rather than re-walking every footprint by hand)."""

    placements = list(placements)
    idx_grid = occupancy_grid(placements)
    return {cell: placements[idx].colour for cell, idx in idx_grid.items()}


def _placement_is_present(p: Placement, colour_at: Mapping[tuple[int, int, int], str]) -> bool:
    return all(colour_at.get((gx, gy, p.z)) == p.colour for gx, gy in p.cells())
