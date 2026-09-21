# Commands

**Status: built.** `brick_builder/language/` is a small, self-contained package: `compiler.py` (the
grammar), `goal.py` (the `Goal` type it produces), and `__main__.py` (`python -m
brick_builder.language`). Every number on this page comes from the tests in `tests/language/` and
from running the tool directly.

## Purpose

This is the part of the project that turns one line of ordinary written English into a `Goal` -
an ordered list of bricks to place, each with a size, a colour, and a target cell on the
baseplate - which the closed-loop controller then carries out. It runs once, before the arm
moves, and never again during that build: see [Placeholder scope](#placeholder-scope)
below for what that means and why.

It is a hand-written grammar over a small, fixed vocabulary, not a language model. It does not
learn, does not call out to anything, and does not try to be clever about a command it was not
taught. When a command uses a word or a construction it does not recognise, or asks for
something that is genuinely ambiguous (two red bricks on the plate and no way to tell which one
"the red brick" means), it says so specifically and builds nothing, rather than guessing.

## Command coverage

All of the examples below are real - they are drawn from `tests/language/test_compiler.py`,
which checks the exact placement text each one produces.

**Absolute placement**, by column and row (see [the coordinate convention](#the-coordinate-convention)),
by an `(x, y)` pair, and by layer:

```
put the red 2x4 brick at column 3 row 2
place a blue 2 by 2 on the plate at (4,5)
put an orange 2x2 at column 1 row 1 layer 1
put a red 2x2 at column 1 row 1, then put a blue 2x2 at column 1 row 1 on the second layer
```

(a layer above the baseplate has to actually rest on something, just like on real bricks - see
[unsupported](#when-a-command-is-not-understood) below - which is why that last example needs
the first brick before the second one can go on layer 2.)

**Orientation** - `lengthwise` is the default (a brick's size is written and read as `h x w`,
`h` along rows, `w` along columns; see `brick_builder/bricks/structures.py`); `across` and `rotated`
both swap the two:

```
put a red 2x4 brick at column 1 row 1            (2x4: 2 rows tall, 4 columns wide)
put a red 2x4 brick at column 1 row 1 rotated     (4x2: 4 rows tall, 2 columns wide)
```

**Sequences** - "then", "and then", ";", and a numbered list written as "1.", "2.", ...:

```
put a red 2x2 at column 1 row 1, then put a blue 2x2 at column 3 row 1
1. put a red 2x2 at column 1 row 1.
2. put a blue 2x2 at column 3 row 1.
```

Later steps in the same command see the earlier ones: "on top of the red brick" below refers to
the brick the first half of this same command just placed, not something that has to already be
on the plate.

**Relative placement**, resolved against whatever is already on the plate (`current_grid`) plus
anything the command has placed so far:

```
put a red 2x2 at column 1 row 1, then put a yellow 2x2 on top of the red brick
put a red 2x2 at column 4 row 4, then put a blue 2x2 left of the red brick
put a red 2x2 at column 4 row 4, then put a blue 2x2 right of the red brick
put a red 2x2 at column 4 row 4, then put a blue 2x2 in front of the red brick
put a red 2x2 at column 4 row 4, then put a blue 2x2 behind the red brick
put a red 2x2 at column 4 row 4, then put a blue 2x2 next to the red brick   (same as "right of")
```

"Left"/"right" and "in front of"/"behind" follow the same convention as the coordinate axes
below: left/right is toward column 1 or column 8, front/behind is toward row 1 or row 8. If more
than one brick matches the description ("the red brick" and there are two), or none does, the
compiler refuses and says which is wrong, rather than picking one:

```
put a yellow 2x2 on top of the red brick     -> "there is more than one red brick on the plate ..."
put a yellow 2x2 on top of the red brick     -> "there is no red brick on the plate ..."
```

Naming a size along with the colour resolves it: "on top of the red 2x4 brick" when there is a
red 2x2 *and* a red 2x4 on the plate picks the 2x4 one.

**The middle of the plate, and a corner:**

```
put a red 2x2 in the middle of the plate
put a red 2x2 in the corner                 (defaults to the corner at column 1, row 1)
put a red 2x2 in the far right corner
put a red 2x2 in the near left corner
```

**Small named structures**, built from a count, an optional size, and an optional colour:

```
a tower of three 2x2 bricks
a staircase with three steps
a wall of two 2x4 bricks
a tower of four blue 2x2 bricks at column 1 row 1
```

A tower stacks straight up; a wall stands bricks side by side in a row; a staircase is columns of
increasing height, one stud-width apart, so each step is fully supported by the baseplate rather
than by the step before it (no overhang - see [Known limits](#known-limits) for why that matters).
Leaving out the colour or the size on a *named structure* assumes red 2x2 bricks and says so in
the paraphrase; a plain, one-brick placement never does this (see the next section).

**Quantities and synonyms:** digits or number words ("3" or "three") everywhere a count is
needed; "2x4", "2 by 4", and "two-by-four" are the same size; "square" means 2x2 and "long" means
2x4. Ordinary politeness ("please", "could you", "thanks") is ignored.

## No default colour or size

The compiler never invents a colour or a size. "Put the brick at column 1 row 1" and "put the red
brick at column 1 row 1" are both refused - not defaulted to some brick - because which brick to
pick up from the tray is exactly the kind of decision this placeholder is not allowed to make on
its own. (Compare: leaving out the *location* of a command does default, to the middle of the
plate, and the paraphrase says so - a location default cannot cause the wrong brick to be
picked, so it is a safe place for this compiler to fill in a reasonable answer instead of
refusing.)

## The coordinate convention

`x` is the column, `0` to `7`; `y` is the row, `0` to `7`; both count in studs from the corner of
the baseplate nearest the arm's base, on the arm's *left*. A command may give column/row as the
numbers a person would use standing at the plate - "column 1" through "column 8" - and the
compiler converts them to the grid's own 0-based `x`/`y` and says so in the paraphrase ("columns
and rows counted from 1, so grid x=2, y=1"). An `(x, y)` pair written in parentheses, like
`(4,5)`, is already 0-based, matching `brick_builder/bricks/structures.py`'s own placement text
format - there is no separate conversion or disclosure for that form.

```
        arm's base
             |
             v
      +----+----+----+----+----+----+----+----+
  r8  |    |    |    |    |    |    |    |    |   far edge (row 8, y=7)
      +----+----+----+----+----+----+----+----+
  r7  |    |    |    |    |    |    |    |    |
      +----+----+----+----+----+----+----+----+
      :                                        :
      +----+----+----+----+----+----+----+----+
  r2  |    |    |    |    |    |    |    |    |
      +----+----+----+----+----+----+----+----+
  r1  | O  |    |    |    |    |    |    |    |   near edge (row 1, y=0): nearest the arm
      +----+----+----+----+----+----+----+----+
       c1   c2   c3   c4   c5   c6   c7   c8
       |                                    |
       arm's left (column 1, x=0)           arm's right (column 8, x=7)

  O = column 1, row 1 = grid (x=0, y=0): the corner nearest the arm's base, on its left.
```

"Left of"/"right of" and "in front of"/"behind" (used in relative placements, above) follow this
same picture: left and right run along the row toward column 1 or column 8, "in front of" moves
toward row 1 (nearer the arm) and "behind" moves toward row 8.

Layers count up from the baseplate: layer 1 (or "the first layer") is `z=0`, resting directly on
the baseplate; layer 2 ("the second layer") is `z=1`, and so on - the same 1-based-to-0-based
conversion, and the same disclosure, as column and row.

`python -m brick_builder.language` draws the plate the same way: column 1 on the left, row 1 at the
*bottom* of the picture (so it reads like looking down at the plate from above, with the arm
below the page) - see [the command-line tool](#the-command-line-tool) below.

## When a command is not understood

`compile_command` returns a `CompileResult` with four fields: `goal`, `understood`, `problems`,
and `paraphrase`. Whenever `understood` is `False`, `goal` is `None` and `paraphrase` is the
empty string - there is no partial commitment and nothing to confirm, because nothing was built.
`problems` is a list of specific, plain-English reasons, one for each step of the command that
had one:

- an unrecognised colour or brick size ("unknown colour 'purple' (available: blue, green, orange,
  red, white, yellow)");
- a missing colour or size ("missing colour (colour and size are required)");
- an ambiguous reference ("there is more than one red brick on the plate ...; say which one, for
  example by adding its size");
- a target that is already occupied, off the plate, or unsupported (the last three reuse
  `brick_builder.bricks.structures.validate_structure` directly, so the wording matches whatever that
  module already says elsewhere in the project);
- the tray not holding enough of some colour/size (when an `Inventory` is given).

`python -m brick_builder.language` prints exactly this list under `problems:` and exits with status 1;
nothing is printed as if it were going to happen.

## Placeholder scope

This package is a placeholder feed-forward compiler from ordinary human language commands. That
is a specific, deliberate design, not an apology:

- **One-shot.** `compile_command` runs once per command, before the controller starts moving,
  and produces a `Goal` - inert data, not a running process. It is never called again while that
  `Goal` is being carried out. If the plate does not end up looking the way the `Goal` expected
  (a brick slips, is knocked away, was never placed), that is the controller's problem to notice
  and react to via `Goal.remaining_steps`, not this compiler's - it does not watch, and it is not
  asked to explain itself mid-build.
- **Outside the control loop.** Nothing here runs at control-tick rate, touches the servo bus, or
  has a notion of time. It has no access to a camera, the network, or anything outside its own
  arguments.
- **No language model.** The grammar in `compiler.py` is hand-written and fixed: a tokeniser plus
  a small set of rules, not a model with weights, a prompt, or a context window. Every accepted
  command in [Command coverage](#command-coverage) works because a rule for it exists in that file
  and is exercised by a test in `tests/language/test_compiler.py` - not because the compiler
  generalised to it.
- **Deterministic.** The same command, the same inventory, and the same plate state always
  produce the exact same `Goal` (or the exact same refusal) - `tests/language/test_compiler.py`
  checks this directly, including with an inventory and a current grid supplied.

The interface a more capable language component would have to fill is exactly this:

```python
def compile_command(
    text: str,
    *,
    inventory: Inventory | None = None,
    current_grid: Sequence[Placement] | None = None,
) -> CompileResult:
    ...
```

and the `Goal` it has to produce - `brick_builder/language/goal.py`'s `Goal`: an ordered tuple of
`Step`s (a `Placement`, a source, and an optional note of what it was derived from), plus
`expected_grid_after`, `remaining_steps`, `to_text`, and `problems`/`is_valid`. Nothing about the
controller, `__main__.py`, or the tests would need to change if `compile_command` one day parsed
with a language model instead of the grammar in this package, provided it kept returning the same
`CompileResult` - understood or not, and never a guess it cannot stand behind.

## Known limits

- **Colour and size are never assumed**, even when only one brick in the whole palette could
  plausibly be meant. This is the one place the compiler is stricter than it has to be, on
  purpose (see [No default colour or size](#no-default-colour-or-size)).
- **No pronouns.** "Put a blue brick on it" does not refer back to a brick just discussed; every
  relative placement names the reference brick's colour directly, every time.
- **A stray number or ordinal is refused, not ignored.** "Put the *second* red 2x2 brick" is
  refused ("unrecognized reference 'second'") rather than silently placing *a* red
  2x2 brick - there is no notion of "the second one" for a plain placement to mean.
- **A numbered list needs a period after the number** ("1. ...", "2. ..."), not a closing
  parenthesis ("1) ...", "2) ..."). The two are genuinely ambiguous to tell apart in a
  hand-written grammar without lookahead: a row number followed by the next item's bare "2)"
  cannot be told apart from an ordinary trailing number, since nothing separates them. "then",
  "and then", and ";" are unambiguous and always work.
- **A relation only ever names one reference brick.** "Put a brick between the red brick and the
  blue brick" is not supported.
- **A relative placement copies the reference brick's layer**, even if the command also names an
  explicit layer; the explicit layer is silently ignored rather than checked for a conflict.
- **A structure's own bricks are always identical** - the same size and colour throughout a
  tower, staircase, or wall. There is no "a staircase that changes colour partway up."
- **The plate is always the stock 8x8**, per the hardware; only what is *on* the plate
  (`current_grid`) is a variable.

## The command-line tool

```
$ .venv/bin/python -m brick_builder.language "put the red 2x4 at column 3 row 2"
paraphrase: a red 2x4 brick, at column 3, row 2 (columns and rows counted from 1, so grid x=2, y=1), on the baseplate

placement text:
2x4 (2,1,0) red

layer 1 (z=0):
     1  2  3  4  5  6  7  8
  8  .  .  .  .  .  .  .  .
  7  .  .  .  .  .  .  .  .
  6  .  .  .  .  .  .  .  .
  5  .  .  .  .  .  .  .  .
  4  .  .  .  .  .  .  .  .
  3  .  .  R  R  R  R  .  .
  2  .  .  R  R  R  R  .  .
  1  .  .  .  .  .  .  .  .

key: B=blue, G=green, O=orange, R=red, W=white, Y=yellow, .=empty
```

A command that is not understood exits with status 1 and prints the specific problems instead of
a picture:

```
$ .venv/bin/python -m brick_builder.language "put the purple 2x2 at column 1 row 1"
error: command not understood: 'put the purple 2x2 at column 1 row 1'
problems:
  - step 1: unknown colour 'purple' (available: blue, green, orange, red, white, yellow)
```

`--inventory PATH` reads tray stock from a text file, one brick per line, `<count> <colour>
<size>` (`#` starts a comment):

```
3 red 2x2
2 blue 2x4
```

`--grid PATH` reads what is already on the plate from a file in `brick_builder.bricks.structures`'s
own placement text format (the same format `Goal.to_text()` produces), so a relative command can
resolve against it:

```
2x2 (0,0,0) red
```

```
$ .venv/bin/python -m brick_builder.language --grid grid.txt "put a blue 2x2 on top of the red brick"
paraphrase: a blue 2x2 brick, on top of the red 2x2 brick at column 1, row 1

placement text:
2x2 (0,0,1) blue
```
