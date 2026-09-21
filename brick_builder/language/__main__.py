"""`python -m brick_builder.language "<command>"`: compile one command and show what it means.

Prints the paraphrase (what the compiler thinks you asked for, in unambiguous words), the
placement text (`brick_builder.bricks.structures`' "{h}x{w} (x,y,z) colour" format, one line per
brick), and an ASCII picture of the expected baseplate after every step, one picture per layer
that ends up with anything on it. If the command was not understood, it prints the specific
problems instead and exits with a non-zero status - it never guesses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bricks.spec import PALETTE
from ..bricks.structures import Placement, parse_structure
from .compiler import CompileResult, compile_command
from .goal import CellExpectation, Goal, Inventory, brick_kind


def _parse_inventory_file(path: str) -> Inventory:
    """One brick per line: "<count> <colour> <size>", e.g. "3 red 2x2". Blank lines and lines
    starting with "#" are ignored; repeated lines for the same colour/size add up."""

    counts: dict[tuple[tuple[int, int], str], int] = {}
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or "x" not in parts[2]:
            raise ValueError(f"{path}:{lineno}: expected '<count> <colour> <size>', e.g. '3 red 2x2', got {raw!r}")
        count_text, colour, size_text = parts
        h_text, w_text = size_text.split("x", 1)
        key = (brick_kind(int(h_text), int(w_text)), colour.lower())
        counts[key] = counts.get(key, 0) + int(count_text)
    return Inventory(counts=counts)


def _parse_grid_file(path: str) -> list[Placement]:
    return parse_structure(Path(path).read_text())


_COLOUR_CODES = {name: name[0].upper() for name in PALETTE}


def _render_layer(cells: dict[tuple[int, int, int], CellExpectation], z: int, grid_shape: tuple[int, int]) -> str:
    """One layer of the expected grid as a character picture, `grid_shape` studs on a side.

    Row 1 (grid y=0, the edge nearest the arm's base - see the coordinate diagram in
    docs/guide/11-commands.md) is printed at the *bottom*, so the picture reads the way the
    plate looks from above with the arm below the page. Column and row labels are 1-based,
    matching how a person would count studs on the physical plate.
    """

    nx, ny = grid_shape
    lines = [f"layer {z + 1} (z={z}):", "    " + " ".join(f"{x + 1:>2}" for x in range(nx))]
    for y in range(ny - 1, -1, -1):
        row = []
        for x in range(nx):
            cell = cells.get((x, y, z))
            row.append(f" {_COLOUR_CODES.get(cell.colour, '?')}" if cell else " .")
        lines.append(f"{y + 1:>3} " + " ".join(row))
    return "\n".join(lines)


def render_goal(goal: Goal) -> str:
    """An ASCII picture of `goal`'s finished plate, one layer at a time, bottom layer first."""

    grid = goal.final_grid
    layers = sorted({z for (_, _, z) in grid})
    if not layers:
        return "(nothing to place)"
    pictures = [_render_layer(grid, z, goal.grid_shape) for z in layers]
    key = ", ".join(f"{code}={name}" for name, code in sorted(_COLOUR_CODES.items(), key=lambda kv: kv[1]))
    return "\n\n".join(pictures) + f"\n\nkey: {key}, .=empty"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m brick_builder.language",
        description="Compile one ordinary-English building command into a goal, once, and show what it means.",
    )
    parser.add_argument("command", nargs="+", help='the command, e.g. "put the red 2x4 at column 3 row 2"')
    parser.add_argument("--inventory", metavar="PATH", help="tray stock file: lines of '<count> <colour> <size>'")
    parser.add_argument("--grid", metavar="PATH", help="what is already on the plate, in the placement text format")
    args = parser.parse_args(argv)

    text = " ".join(args.command)
    inventory = _parse_inventory_file(args.inventory) if args.inventory else None
    current_grid = _parse_grid_file(args.grid) if args.grid else None

    result: CompileResult = compile_command(text, inventory=inventory, current_grid=current_grid)

    if not result.understood:
        print(f"error: command not understood: {text!r}")
        print("problems:")
        for problem in result.problems:
            print(f"  - {problem}")
        return 1

    print(f"paraphrase: {result.paraphrase}")
    print()
    print("placement text:")
    print(result.goal.to_text())
    print()
    print(render_goal(result.goal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
