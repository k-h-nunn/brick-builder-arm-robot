# Commands

**Status: built.** `brick_builder/language/` is a small, self-contained package: `compiler.py` (the
grammar), `goal.py` (the `Goal` type it produces), and `__main__.py` (`python -m
brick_builder.language`). Every number on this page comes from the tests in `tests/language/` and
from running the tool directly.

## Purpose

Turns one line of ordinary written English into a `Goal` — an ordered list of bricks to place,
each with a size, a colour, and a target cell on the baseplate — for the closed-loop controller to
carry out. Runs once, before the arm moves, never again during that build (see
[Placeholder scope](#placeholder-scope)).

A hand-written grammar over a small, fixed vocabulary, not a language model: no learning, no
external calls. A command using an unrecognised word or construction, or asking for something
genuinely ambiguous (two red bricks on the plate, no way to tell which "the red brick" means), is
refused with a specific reason rather than guessed.

## Command coverage

Examples below are drawn from `tests/language/test_compiler.py`, which checks the exact placement
text each one produces.

**Absolute placement**, by column and row (see [the coordinate convention](#the-coordinate-convention)),
by an `(x, y)` pair, and by layer:

```
put the red 2x4 brick at column 3 row 2
place a blue 2 by 2 on the plate at (4,5)
put an orange 2x2 at column 1 row 1 layer 1
put a red 2x2 at column 1 row 1, then put a blue 2x2 at column 1 row 1 on the second layer
```

(A layer above the baseplate has to rest on something, as with real bricks — see
[unsupported](#when-a-command-is-not-understood) — so the last example needs the first brick
before the second one can go on layer 2.)

**Orientation** — `lengthwise` is the default (a brick's size is written and read as `h x w`, `h`
along rows, `w` along columns; see `brick_builder/bricks/structures.py`); `across` and `rotated`
both swap the two:

```
put a red 2x4 brick at column 1 row 1            (2x4: 2 rows tall, 4 columns wide)
put a red 2x4 brick at column 1 row 1 rotated     (4x2: 4 rows tall, 2 columns wide)
```

**Sequences** — "then", "and then", ";", and a numbered list written as "1.", "2.", ...:

```
put a red 2x2 at column 1 row 1, then put a blue 2x2 at column 3 row 1
1. put a red 2x2 at column 1 row 1.
2. put a blue 2x2 at column 3 row 1.
```

Later steps see the earlier ones: "on top of the red brick" below refers to the brick the first
half of this same command just placed, not something that has to already be on the plate.

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

Left/right and front/behind follow the coordinate axes below: left/right toward column 1 or
column 8, front/behind toward row 1 or row 8. An ambiguous reference (more than one match, or
none) is refused, naming which is wrong rather than picking one:

```
put a yellow 2x2 on top of the red brick     -> "there is more than one red brick on the plate ..."
put a yellow 2x2 on top of the red brick     -> "there is no red brick on the plate ..."
```

Naming a size along with the colour resolves it: "on top of the red 2x4 brick" picks the 2x4 when
a red 2x2 *and* a red 2x4 are both on the plate.

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
than by the step before it (no overhang — see [Known limits](#known-limits)). Leaving out the
colour or size on a *named structure* defaults to red 2x2 and says so in the paraphrase; a plain
one-brick placement never defaults (next section).

**Quantities and synonyms:** digits or number words ("3" or "three") everywhere a count is needed;
"2x4", "2 by 4", and "two-by-four" are the same size; "square" means 2x2 and "long" means 2x4.
Ordinary politeness ("please", "could you", "thanks") is ignored.

## No default colour or size

The compiler never invents a colour or a size. "Put the brick at column 1 row 1" and "put the red
brick at column 1 row 1" are both refused, not defaulted — which brick to pick up from the tray is
exactly the kind of decision this placeholder cannot make on its own. A missing *location*, by
contrast, defaults to the middle of the plate and says so in the paraphrase: a location default
cannot pick the wrong brick, so it is a safe place to fill in an answer instead of refusing.

## The coordinate convention

`x` is the column, `0` to `7`; `y` is the row, `0` to `7`; both count in studs from the corner of
the baseplate nearest the arm's base, on the arm's *left*. A command may give column/row as the
numbers a person would use standing at the plate — "column 1" through "column 8" — and the
compiler converts them to the grid's own 0-based `x`/`y` and says so in the paraphrase ("columns
and rows counted from 1, so grid x=2, y=1"). An `(x, y)` pair written in parentheses, like
`(4,5)`, is already 0-based, matching `brick_builder/bricks/structures.py`'s own placement text
format — no conversion or disclosure for that form.

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

"Left of"/"right of" and "in front of"/"behind" (relative placements, above) follow this same
picture: left and right run along the row toward column 1 or column 8; "in front of" moves toward
row 1 (nearer the arm), "behind" moves toward row 8.

Layers count up from the baseplate: layer 1 ("the first layer") is `z=0`, resting directly on the
baseplate; layer 2 ("the second layer") is `z=1`, and so on — the same 1-based-to-0-based
conversion and disclosure as column and row.

`python -m brick_builder.language` draws the plate the same way: column 1 on the left, row 1 at
the *bottom* of the picture (looking down at the plate from above, arm below the page) — see
[the command-line tool](#the-command-line-tool) below.

## When a command is not understood

`compile_command` returns a `CompileResult` with four fields: `goal`, `understood`, `problems`,
and `paraphrase`. Whenever `understood` is `False`, `goal` is `None` and `paraphrase` is the empty
string — no partial commitment, nothing was built. `problems` is a list of specific, plain-English
reasons, one for each step of the command that had one:

- an unrecognised colour or brick size ("unknown colour 'purple' (available: blue, green, orange,
  red, white, yellow)");
- a missing colour or size ("missing colour (colour and size are required)");
- an ambiguous reference ("there is more than one red brick on the plate ...; say which one, for
  example by adding its size");
- a target that is already occupied, off the plate, or unsupported (the last three reuse
  `brick_builder.bricks.structures.validate_structure` directly, so the wording matches whatever that
  module already says elsewhere in the project);
- the tray not holding enough of some colour/size (when an `Inventory` is given).

`python -m brick_builder.language` prints exactly this list under `problems:` and exits with
status 1; nothing is printed as if it were going to happen.

## Placeholder scope

This package is a placeholder feed-forward compiler from ordinary human language commands — a
specific, deliberate design:

- **One-shot.** `compile_command` runs once per command, before the controller starts moving, and
  produces a `Goal` — inert data, not a running process — and is never called again while that
  `Goal` is carried out. If the plate ends up not matching the `Goal` (a brick slips, is knocked
  away, was never placed), that is the controller's problem via `Goal.remaining_steps`; the
  compiler does not watch and is not asked mid-build.
- **Outside the control loop.** Nothing here runs at control-tick rate, touches the servo bus, or
  has a notion of time. No access to a camera, the network, or anything outside its own arguments.
- **No language model.** `compiler.py`'s grammar is hand-written and fixed: a tokeniser plus a
  small rule set, not weights, a prompt, or a context window. Every command in
  [Command coverage](#command-coverage) works because a rule for it exists and is tested
  (`tests/language/test_compiler.py`), not because the compiler generalised to it.
- **Deterministic.** The same command, inventory, and plate state always produce the same `Goal`
  or the same refusal — `tests/language/test_compiler.py` checks this directly, including with an
  inventory and grid supplied.

The interface a more capable language component would have to fill:

```python
def compile_command(
    text: str,
    *,
    inventory: Inventory | None = None,
    current_grid: Sequence[Placement] | None = None,
) -> CompileResult:
    ...
```

and the `Goal` it has to produce — `brick_builder/language/goal.py`'s `Goal`: an ordered tuple of
`Step`s (a `Placement`, a source, an optional derivation note), plus `expected_grid_after`,
`remaining_steps`, `to_text`, `problems`/`is_valid`. Nothing about the controller, `__main__.py`,
or the tests would need to change if `compile_command` parsed with a language model instead of
this grammar, provided it returns the same `CompileResult` — understood or not, never a guess it
cannot stand behind.

## Known limits

- **Colour and size are never assumed**, even when only one brick in the palette could plausibly
  be meant — deliberately stricter than necessary (see
  [No default colour or size](#no-default-colour-or-size)).
- **No pronouns.** "Put a blue brick on it" does not refer back to a brick just discussed; a
  relative placement names the reference brick's colour directly, every time.
- **A stray number or ordinal is refused, not ignored.** "Put the *second* red 2x2 brick" is
  refused ("unrecognized reference 'second'"), not silently placing *a* red 2x2 — there is no
  "the second one" for a plain placement.
- **A numbered list needs a period after the number** ("1. ..."), not a closing parenthesis
  ("1) ..."): the two are ambiguous to a hand-written grammar without lookahead (a row number
  followed by "2)" cannot be told from an ordinary trailing number). "then", "and then", ";" are
  unambiguous.
- **A relation names only one reference brick.** "Put a brick between the red brick and the blue
  brick" is not supported.
- **A relative placement copies the reference brick's layer**, even with an explicit layer named;
  the explicit layer is silently ignored, not checked for conflict.
- **A structure's bricks are always identical** — one size and colour throughout a tower,
  staircase, or wall; no colour change partway up.
- **The plate is always the stock 8x8**; only what is *on* it (`current_grid`) varies.

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
