"""BrickGPT-format structure descriptions: parsing, formatting, validity checks, and building.

A structure is a list of `Placement`s. Each placement gives a brick's footprint (`h` x `w`,
studs), its position on an integer stud grid (`x`, `y` for the footprint's minimum corner, `z`
for the stacking layer, 0 = resting on the baseplate), and a colour word from
`brick_builder.bricks.spec.PALETTE`. The on-disk text format is one placement per line:

    "{h}x{w} ({x},{y},{z}) {colour}"

e.g. "2x4 (0,0,0) red" is a 2x4 brick (h=2 studs along the grid's row/y axis, w=4 along the
column/x axis - see `Placement`'s docstring) with its minimum corner at grid column 0, row 0,
sitting on the baseplate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

from brick_builder.bricks.generator import BrickHandle, ContactModel, add_brick
from brick_builder.bricks.spec import DUPLO, PALETTE, BrickSpec

_LINE_RE = re.compile(
    r"^\s*(\d+)x(\d+)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*([A-Za-z]+)\s*$"
)


@dataclass(frozen=True)
class Placement:
    """One brick in a structure, on an integer stud grid.

    `h` is the footprint extent along the grid's row axis (y, in studs), `w` along the column
    axis (x, in studs) - matching the order they appear in the "{h}x{w}" text. `x`, `y` is the
    grid cell of the footprint's minimum corner (so the brick occupies grid cells
    `x..x+w-1` by `y..y+h-1`), `z` is the stacking layer (0 = resting directly on the
    baseplate). `colour` must be a key of `brick_builder.bricks.spec.PALETTE`.
    """

    h: int
    w: int
    x: int
    y: int
    z: int
    colour: str

    def cells(self) -> list[tuple[int, int]]:
        """Every `(grid_x, grid_y)` cell this placement's footprint covers."""

        return [(gx, gy) for gx in range(self.x, self.x + self.w) for gy in range(self.y, self.y + self.h)]


def format_placement(p: Placement) -> str:
    return f"{p.h}x{p.w} ({p.x},{p.y},{p.z}) {p.colour}"


def parse_placement(text: str) -> Placement:
    m = _LINE_RE.match(text)
    if not m:
        raise ValueError(f"not a valid BrickGPT placement line: {text!r}")
    h, w, x, y, z, colour = m.groups()
    return Placement(h=int(h), w=int(w), x=int(x), y=int(y), z=int(z), colour=colour.lower())


def format_structure(placements: list[Placement]) -> str:
    return "\n".join(format_placement(p) for p in placements)


def parse_structure(text: str) -> list[Placement]:
    lines = [line for line in text.splitlines() if line.strip()]
    return [parse_placement(line) for line in lines]


def occupancy_grid(placements: list[Placement]) -> dict[tuple[int, int, int], int]:
    """Map every occupied `(grid_x, grid_y, layer_z)` cell to the index of the placement there.

    A dict, not a dense array, because a structure's grid extent is not known in advance and
    most cells of a large baseplate are empty.
    """

    grid: dict[tuple[int, int, int], int] = {}
    for idx, p in enumerate(placements):
        for gx, gy in p.cells():
            grid[(gx, gy, p.z)] = idx
    return grid


def validate_structure(
    placements: list[Placement], grid_shape: tuple[int, int] | None = None
) -> list[str]:
    """Return a list of human-readable problems; an empty list means the structure is valid.

    Checks: overlap (two placements claim the same
    cell), out-of-grid (only checked when `grid_shape` is given, as `(nx, ny)`), and support
    (every placement above layer 0 must have every one of its footprint cells occupied by
    some placement in the layer directly below - a brick cannot float).
    """

    problems: list[str] = []
    seen: dict[tuple[int, int, int], int] = {}
    for idx, p in enumerate(placements):
        if p.colour not in PALETTE:
            problems.append(f"placement {idx}: unknown colour {p.colour!r}")
        for gx, gy in p.cells():
            if grid_shape is not None:
                nx, ny = grid_shape
                if not (0 <= gx < nx and 0 <= gy < ny):
                    problems.append(f"placement {idx}: cell ({gx},{gy}) is out of the {nx}x{ny} grid")
            key = (gx, gy, p.z)
            if key in seen:
                problems.append(f"placement {idx}: overlaps placement {seen[key]} at cell {key}")
            else:
                seen[key] = idx

    for idx, p in enumerate(placements):
        if p.z == 0:
            continue
        for gx, gy in p.cells():
            below = seen.get((gx, gy, p.z - 1))
            if below is None:
                problems.append(f"placement {idx}: cell ({gx},{gy}) at layer {p.z} is unsupported")

    return problems


def build_structure(
    world: mujoco.MjSpec,
    placements: list[Placement],
    plate: BrickHandle,
    spec: BrickSpec = DUPLO,
    contact: ContactModel | None = None,
) -> list[BrickHandle]:
    """Instantiate every placement as a brick body on top of `plate`.

    Grid cell `(0, 0)` sits at the baseplate's stud `(0, 0)` (its minimum corner, in
    `add_baseplate`'s own coordinates); a placement's world-frame centre position follows from
    its footprint size the same way `add_brick` positions studs within a brick. Layer `z` maps
    to world height `plate_top + z * spec.height`, matching the flush-stacking behaviour the
    generator's geometry produces (see `generator.py`'s module docstring).
    """

    handles = []
    for idx, p in enumerate(placements):
        handle = add_brick(
            world, name=f"struct_{idx}", nx=p.w, ny=p.h, colour=p.colour,
            pos=placement_pose(p, plate, spec), spec=spec, contact=contact,
        )
        handles.append(handle)
    return handles


def placement_pose(
    p: Placement, plate: BrickHandle, spec: BrickSpec = DUPLO
) -> tuple[float, float, float]:
    """World position of one placement's brick body origin, in the plate's own frame.

    Split out of :func:`build_structure` so a caller that builds its bricks some other way -
    :mod:`brick_builder.bricks.breakable`, through
    :func:`brick_builder.scene.compose.build_scene` - lays them out on exactly the same grid.
    """

    plate_stud_x0 = -(plate.nx - 1) / 2.0 * spec.pitch
    plate_stud_y0 = -(plate.ny - 1) / 2.0 * spec.pitch
    plate_top = plate.body.pos[2] + spec.roof * 2.0  # matches add_baseplate's slab thickness
    return (
        plate_stud_x0 + (p.x + (p.w - 1) / 2.0) * spec.pitch,
        plate_stud_y0 + (p.y + (p.h - 1) / 2.0) * spec.pitch,
        plate_top + p.z * spec.height,
    )
