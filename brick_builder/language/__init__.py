"""A placeholder feedforward compiler from ordinary English building commands to a `Goal`.

One command compiles once, outside the control loop, into a `Goal` the closed-loop controller
then pursues; nothing here is a language model, nothing here runs per control tick, and nothing
here talks to a network. See `docs/guide/11-commands.md` for what the command language covers,
the coordinate convention, and what a more capable language component would have to implement to
replace this one (it is exactly `compile_command`'s signature and the `Goal` type below).

The public interface:

- `compile_command(text, *, inventory=None, current_grid=None) -> CompileResult`
- `Goal`, `Step`, `Inventory`, `CellExpectation` - the data `compile_command` produces and
  consumes.

`python -m brick_builder.language "<command>"` is a small command-line front end over the same
function; see `__main__.py`.
"""

from .compiler import CompileResult, compile_command
from .goal import GRID_SHAPE, KNOWN_SIZES, CellExpectation, Goal, Inventory, Step, brick_kind

__all__ = [
    "compile_command",
    "CompileResult",
    "Goal",
    "Step",
    "Inventory",
    "CellExpectation",
    "GRID_SHAPE",
    "KNOWN_SIZES",
    "brick_kind",
]
