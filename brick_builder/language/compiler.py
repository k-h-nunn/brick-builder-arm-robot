"""`compile_command`: turn one ordinary-English building command into a `Goal`, once.

This is the placeholder feedforward compiler described in `docs/guide/11-commands.md`: no
language model, nothing in the control loop, no network access. `compile_command` is a pure
function - tokenise, parse with a small hand-written grammar, resolve against the tray and
whatever is already on the plate, and hand back a `Goal` or a specific list of reasons it could
not. It never guesses a brick's colour or size, and it never picks between two ambiguous
readings; the closed-loop controller (elsewhere in this project) pursues whatever `Goal` comes
back, and never calls back into this module while doing so.

The pipeline, top to bottom:

1. **Tokenise** (`_scan`): a left-to-right character scan - no regular expressions - into words,
   integers, and single-character punctuation.
2. **Normalise**: spelled-out numbers and ordinals become number/ordinal tokens
   (`_words_to_numbers`), "2x4" / "2 by 4" / "two-by-four" / "square" / "long" all become one
   `size` token (`_merge_sizes`), and "(4,5)" becomes one `coord` token (`_merge_coords`).
3. **Strip filler** (`_drop_filler`): articles, politeness, verbs ("put", "build", ...) and
   connective prepositions carry no information in this grammar and are dropped.
4. **Split into steps** (`_split_segments`): on "then", ";", "." and a leading list number
   ("1.", "2."). A number followed by ")" instead of "." is not supported - see that function's
   docstring for why.
5. **Parse each step**: a structure keyword ("tower", "staircase", "wall") dispatches to its own
   small parser; anything else is one placement (`_parse_single`). Both extract a colour, a
   size, an optional orientation and layer, and a location - absolute, relative to a brick
   already on the plate, the middle of the plate, a corner, or (if nothing else was said) a
   disclosed default of the middle.
6. **Resolve**: relative and default locations are worked out against `current_grid` plus
   whatever earlier steps in the same command have already added (so "put a red 2x2 at (0,0),
   then a blue 2x2 on top of it" resolves "it" against a brick this same command just placed).
   Every resolved placement is checked for a collision with what is already there
   (`structures.occupancy_grid`); the finished goal is checked as a whole with
   `structures.validate_structure` (overlap, off the plate, unsupported) and, if an `Inventory`
   was given, against the tray's stock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..bricks.spec import PALETTE
from ..bricks.structures import Placement, occupancy_grid, validate_structure
from .goal import GRID_SHAPE, Goal, Inventory, KNOWN_SIZES, Step, brick_kind

# ---------------------------------------------------------------------------------------------
# 1. Tokenising - a hand-scanned character stream, not a regular expression.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    kind: str  # "word" | "number" | "ordinal" | "size" | "coord" | "punct"
    text: str = ""
    value: object = None


_PUNCT_CHARS = set("(),.;:!?-")


def _scan(text: str) -> list[_Token]:
    """Split raw text into word, integer, and single-character punctuation tokens."""

    tokens: list[_Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            digits = text[i:j]
            tokens.append(_Token("number", text=digits, value=int(digits)))
            i = j
        elif ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            tokens.append(_Token("word", text=text[i:j].lower()))
            i = j
        elif ch in _PUNCT_CHARS:
            tokens.append(_Token("punct", text=ch))
            i += 1
        else:
            i += 1  # stray symbol (quotes, "#", ...): not part of this grammar, ignore it
    return tokens


# ---------------------------------------------------------------------------------------------
# 2. Normalising - number words, size phrases, coordinate tuples.
# ---------------------------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10,
}
_ORDINAL_NAMES = {v: k for k, v in _ORDINAL_WORDS.items()}


def _words_to_numbers(tokens: list[_Token]) -> list[_Token]:
    out = []
    for t in tokens:
        if t.kind == "word" and t.text in _NUMBER_WORDS:
            out.append(_Token("number", value=_NUMBER_WORDS[t.text]))
        elif t.kind == "word" and t.text in _ORDINAL_WORDS:
            out.append(_Token("ordinal", value=_ORDINAL_WORDS[t.text]))
        else:
            out.append(t)
    return out


def _merge_sizes(tokens: list[_Token]) -> list[_Token]:
    """"2x4", "2 by 4", "two-by-four" -> one `size` token; "square"/"long" are size synonyms."""

    out: list[_Token] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind == "word" and t.text == "square":
            out.append(_Token("size", value=(2, 2)))
            i += 1
            continue
        if t.kind == "word" and t.text == "long":
            out.append(_Token("size", value=(2, 4)))
            i += 1
            continue
        if t.kind == "number":
            j = i + 1
            if j < n and tokens[j].kind == "punct" and tokens[j].text == "-":
                j += 1
            if j < n and tokens[j].kind == "word" and tokens[j].text in ("x", "by"):
                j += 1
                if j < n and tokens[j].kind == "punct" and tokens[j].text == "-":
                    j += 1
                if j < n and tokens[j].kind == "number":
                    out.append(_Token("size", value=(t.value, tokens[j].value)))
                    i = j + 1
                    continue
        out.append(t)
        i += 1
    return out


def _merge_coords(tokens: list[_Token]) -> list[_Token]:
    """"(4,5)" -> one `coord` token holding `(4, 5)`."""

    out: list[_Token] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if (
            t.kind == "punct" and t.text == "(" and i + 4 < n
            and tokens[i + 1].kind == "number"
            and tokens[i + 2].kind == "punct" and tokens[i + 2].text == ","
            and tokens[i + 3].kind == "number"
            and tokens[i + 4].kind == "punct" and tokens[i + 4].text == ")"
        ):
            out.append(_Token("coord", value=(tokens[i + 1].value, tokens[i + 3].value)))
            i += 5
            continue
        out.append(t)
        i += 1
    return out


# ---------------------------------------------------------------------------------------------
# 3-4. Filler and sequencing.
# ---------------------------------------------------------------------------------------------

# Words that carry no information for this grammar: articles, politeness, verbs (every command
# here means "place a brick"), and prepositions that are only ever glue ("at column 3", "of
# three bricks") once the words that matter (column, tower, ...) are matched directly.
_FILLER_WORDS = {
    "a", "an", "the",
    "please", "kindly", "could", "would", "can", "you", "thanks", "thank", "now", "just",
    "okay", "ok",
    "and",
    "brick", "bricks", "step", "steps",
    "put", "place", "build", "make", "add", "set", "position", "construct",
    "of", "with", "to", "at", "in", "on", "onto", "up",
    "is", "there", "it", "that",
}

# Punctuation kept past the filler pass because `_split_segments` still needs it. Only "." and
# ";" - not ")" - split a segment: "1. ...  2. ..." works (the "2" that starts the second item
# is cleanly on the far side of the first period), but "1) ... 2) ..." cannot be told apart from
# an ordinary number falling right before a ")" with nothing separating them (a placement's own
# last number and the next item's marker would run together); see docs/guide/11-commands.md.
_SEGMENT_PUNCT = {".", ";"}


def _drop_filler(tokens: list[_Token]) -> list[_Token]:
    out = []
    for t in tokens:
        if t.kind == "word" and t.text in _FILLER_WORDS:
            continue
        if t.kind == "punct" and t.text not in _SEGMENT_PUNCT:
            continue
        out.append(t)
    return out


def _split_segments(tokens: list[_Token]) -> list[list[_Token]]:
    """One command may describe several steps: "...; then ...", "1. ... 2. ...". Split on
    "then" and sentence-ish punctuation, and drop anything that is left holding only a bare
    number - a list marker, not a step."""

    segments: list[list[_Token]] = []
    current: list[_Token] = []
    for t in tokens:
        is_separator = (t.kind == "word" and t.text == "then") or (t.kind == "punct" and t.text in _SEGMENT_PUNCT)
        if is_separator:
            segments.append(current)
            current = []
        else:
            current.append(t)
    segments.append(current)
    return [seg for seg in segments if seg and not (len(seg) == 1 and seg[0].kind in ("number", "ordinal"))]


def _tokenize_and_segment(text: str) -> list[list[_Token]]:
    tokens = _scan(text)
    tokens = _words_to_numbers(tokens)
    tokens = _merge_sizes(tokens)
    tokens = _merge_coords(tokens)
    tokens = _drop_filler(tokens)
    return _split_segments(tokens)


# ---------------------------------------------------------------------------------------------
# 5. Locations - absolute, relative to a brick already on the plate, the middle, or a corner.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _AbsoluteLoc:
    x: int
    y: int
    note: str


@dataclass(frozen=True)
class _RelativeLoc:
    ref_colour: str
    ref_size: tuple[int, int] | None
    relation: str


@dataclass(frozen=True)
class _MiddleLoc:
    pass


@dataclass(frozen=True)
class _CornerLoc:
    left: bool
    near: bool


@dataclass(frozen=True)
class _DefaultLoc:
    pass


_Location = _AbsoluteLoc | _RelativeLoc | _MiddleLoc | _CornerLoc | _DefaultLoc

# Checked in this order so "in the left corner" is read as a corner (with a qualifier), not as
# "left of" a brick that was never named; whichever anchor word appears anywhere in the segment
# wins, regardless of where in the sentence it falls.
_RELATION_ANCHORS: tuple[tuple[str, str], ...] = (
    ("corner", "corner"),
    ("top", "on_top_of"),
    ("front", "in_front_of"),
    ("behind", "behind"),
    ("left", "left_of"),
    ("right", "right_of"),
    ("next", "next_to"),
    ("middle", "middle"),
)

_RELATION_PHRASES = {
    "on_top_of": "on top of",
    "left_of": "left of",
    "right_of": "right of",
    "in_front_of": "in front of",
    "behind": "behind",
    "next_to": "next to",
}


def _find_anchor(tokens: list[_Token]) -> tuple[str | None, int | None]:
    for word, kind in _RELATION_ANCHORS:
        for i, t in enumerate(tokens):
            if t.kind == "word" and t.text == word:
                return kind, i
    return None, None


def _next_number(tokens: list[_Token], idx: int) -> int | None:
    for t in tokens[idx + 1:]:
        if t.kind == "number":
            return t.value
    return None


def _read_absolute(tokens: list[_Token]) -> tuple[_AbsoluteLoc | None, str | None]:
    """`(location, error)`; `(None, None)` if no absolute location is written here at all."""

    col = row = None
    for i, t in enumerate(tokens):
        if t.kind == "word" and t.text == "column":
            col = _next_number(tokens, i)
        elif t.kind == "word" and t.text == "row":
            row = _next_number(tokens, i)
    if col is not None or row is not None:
        if col is None or row is None:
            return None, "a placement needs both a column and a row; only one was given"
        x, y = col - 1, row - 1
        return _AbsoluteLoc(x, y, note=f"columns and rows counted from 1, so grid x={x}, y={y}"), None
    for t in tokens:
        if t.kind == "coord":
            x, y = t.value
            return _AbsoluteLoc(x, y, note=f"grid x={x}, y={y}"), None
    return None, None


def _read_corner(tokens: list[_Token]) -> _CornerLoc:
    right = any(t.kind == "word" and t.text == "right" for t in tokens)
    far = any(t.kind == "word" and t.text == "far" for t in tokens)
    return _CornerLoc(left=not right, near=not far)


def _read_location(tokens: list[_Token]) -> tuple[_Location | None, list[_Token], str | None]:
    """`(location, main_tokens, error)`. `main_tokens` excludes a relation's own reference-brick
    phrase, so a caller reading the *new* brick's colour/size never picks up the reference
    brick's colour/size by mistake."""

    kind, idx = _find_anchor(tokens)
    if kind == "corner":
        return _read_corner(tokens), tokens, None
    if kind == "middle":
        return _MiddleLoc(), tokens, None
    if kind is not None:
        main_tokens, ref_tokens = tokens[:idx], tokens[idx + 1:]
        ref_colour = _find_colour(ref_tokens)
        if ref_colour is None:
            phrase = _RELATION_PHRASES[kind]
            return None, main_tokens, (
                f"'{phrase}' needs to know which brick, by colour - for example "
                f"'{phrase} the red brick'"
            )
        ref_size = _find_size(ref_tokens)
        return _RelativeLoc(ref_colour=ref_colour, ref_size=ref_size, relation=kind), main_tokens, None
    abs_loc, err = _read_absolute(tokens)
    if err:
        return None, tokens, err
    if abs_loc is not None:
        return abs_loc, tokens, None
    return _DefaultLoc(), tokens, None


def _resolve_location(
    location: _Location, h: int, w: int, grid_shape: tuple[int, int], working_grid: list[Placement],
) -> tuple[int | None, int | None, int | None, str | None, str | None]:
    """`(x, y, z_override, note, error)`. `z_override` is `None` unless the location is relative
    to another brick, in which case it fixes the layer too (an explicit "layer" word is ignored
    in favour of the reference, rather than raising a conflict - a documented limitation)."""

    nx, ny = grid_shape
    if isinstance(location, _AbsoluteLoc):
        note = f"at column {location.x + 1}, row {location.y + 1} ({location.note})"
        return location.x, location.y, None, note, None

    if isinstance(location, _MiddleLoc):
        x, y = (nx - w) // 2, (ny - h) // 2
        return x, y, None, f"in the middle of the plate (column {x + 1}, row {y + 1})", None

    if isinstance(location, _DefaultLoc):
        x, y = (nx - w) // 2, (ny - h) // 2
        note = (
            f"in the middle of the plate (column {x + 1}, row {y + 1}) - "
            f"no location was given, so this is assumed"
        )
        return x, y, None, note, None

    if isinstance(location, _CornerLoc):
        x = 0 if location.left else nx - w
        y = 0 if location.near else ny - h
        side = ("near " if location.near else "far ") + ("left" if location.left else "right")
        return x, y, None, f"in the {side} corner (column {x + 1}, row {y + 1})", None

    if isinstance(location, _RelativeLoc):
        matches = [
            p for p in working_grid
            if p.colour == location.ref_colour
            and (location.ref_size is None or brick_kind(p.h, p.w) == brick_kind(*location.ref_size))
        ]
        phrase = _RELATION_PHRASES[location.relation]
        if not matches:
            size_note = f" {location.ref_size[0]}x{location.ref_size[1]}" if location.ref_size else ""
            return None, None, None, None, (
                f"there is no{size_note} {location.ref_colour} brick on the plate to place "
                f"anything {phrase}"
            )
        if len(matches) > 1:
            where = ", ".join(f"column {p.x + 1} row {p.y + 1} layer {p.z + 1}" for p in matches)
            return None, None, None, None, (
                f"there is more than one {location.ref_colour} brick on the plate ({where}); "
                f"say which one, for example by adding its size"
            )
        ref = matches[0]
        rel = location.relation
        if rel == "on_top_of":
            x, y, z = ref.x, ref.y, ref.z + 1
        elif rel == "left_of":
            x, y, z = ref.x - w, ref.y, ref.z
        elif rel in ("right_of", "next_to"):
            x, y, z = ref.x + ref.w, ref.y, ref.z
        elif rel == "in_front_of":
            x, y, z = ref.x, ref.y - h, ref.z
        elif rel == "behind":
            x, y, z = ref.x, ref.y + ref.h, ref.z
        else:  # pragma: no cover - _RELATION_PHRASES and this chain are kept in lockstep
            raise AssertionError(f"unhandled relation {rel!r}")
        note = f"{phrase} the {ref.colour} {ref.h}x{ref.w} brick at column {ref.x + 1}, row {ref.y + 1}"
        return x, y, z, note, None

    raise AssertionError(f"unhandled location {location!r}")  # pragma: no cover


# ---------------------------------------------------------------------------------------------
# Attribute extraction shared by every kind of segment.
# ---------------------------------------------------------------------------------------------


def _find_colour(tokens: list[_Token]) -> str | None:
    for t in tokens:
        if t.kind == "word" and t.text in PALETTE:
            return t.text
    return None


# Colour words a person might reach for that are not in this tray's palette (`brick_builder.bricks.
# spec.PALETTE`), so an unknown colour gets a specific problem instead of "which colour brick?".
_OFF_PALETTE_COLOURS = {
    "purple", "pink", "black", "brown", "gray", "grey", "cyan", "magenta", "teal", "violet",
    "silver", "gold", "tan", "beige", "maroon", "navy", "lime", "indigo",
}


def _find_off_palette_colour(tokens: list[_Token]) -> str | None:
    for t in tokens:
        if t.kind == "word" and t.text in _OFF_PALETTE_COLOURS:
            return t.text
    return None


def _colour_problem(tokens: list[_Token]) -> str:
    off = _find_off_palette_colour(tokens)
    if off:
        palette = ", ".join(sorted(PALETTE))
        return f"unknown colour '{off}' (available: {palette})"
    return "missing colour (colour and size are required)"


def _find_size(tokens: list[_Token]) -> tuple[int, int] | None:
    for t in tokens:
        if t.kind == "size":
            return t.value
    return None


_KNOWN_KINDS = {brick_kind(*s) for s in KNOWN_SIZES}


def _size_problem(size: tuple[int, int]) -> str | None:
    if brick_kind(*size) not in _KNOWN_KINDS:
        sizes = ", ".join(f"{h}x{w}" for h, w in KNOWN_SIZES)
        return f"a {size[0]}x{size[1]} brick is not one of the available brick sizes ({sizes})"
    return None


def _find_orientation(tokens: list[_Token]) -> str:
    for t in tokens:
        if t.kind == "word" and t.text in ("rotated", "across"):
            return "rotated"
    return "lengthwise"


def _find_count(tokens: list[_Token]) -> int | None:
    for t in tokens:
        if t.kind == "number":
            return t.value
    return None


def _read_layer(tokens: list[_Token]) -> tuple[int | None, str | None]:
    """`(z, note)` for an explicit "layer 2" / "second layer"; `(None, None)` if none is given."""

    for i, t in enumerate(tokens):
        if t.kind == "word" and t.text in ("layer", "layers") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.kind == "number" and nxt.value >= 1:
                n = nxt.value
                return n - 1, f"layer {n} counted from 1, so grid z={n - 1}"
        if (
            t.kind == "ordinal" and i + 1 < len(tokens)
            and tokens[i + 1].kind == "word" and tokens[i + 1].text in ("layer", "layers")
        ):
            n = t.value
            name = _ORDINAL_NAMES.get(n, f"{n}th")
            return n - 1, f"the {name} layer is grid z={n - 1}"
    return None, None


def _find_unclaimed_quantity(tokens: list[_Token]) -> _Token | None:
    """A number or ordinal a single placement's grammar has no use for - e.g. "the *second* red
    2x2" - so the caller can say it does not understand rather than silently drop it. A number
    right after "column"/"row"/"layer" is claimed by that phrase, and an ordinal right before
    "layer" is claimed by "the *second* layer"; anything else left over is unclaimed.
    """

    for i, t in enumerate(tokens):
        if t.kind == "number":
            prev = tokens[i - 1] if i > 0 else None
            if prev and prev.kind == "word" and prev.text in ("column", "row", "layer", "layers"):
                continue
            return t
        if t.kind == "ordinal":
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt and nxt.kind == "word" and nxt.text in ("layer", "layers"):
                continue
            return t
    return None


def _find_occupied_conflict(
    working_grid: list[Placement], new_placement: Placement,
) -> tuple[Placement | None, tuple[int, int] | None]:
    """The existing placement (and cell) `new_placement` would collide with, if any - reuses
    `structures.occupancy_grid` rather than re-walking footprints by hand."""

    occ = occupancy_grid(working_grid)
    for gx, gy in new_placement.cells():
        idx = occ.get((gx, gy, new_placement.z))
        if idx is not None:
            return working_grid[idx], (gx, gy)
    return None, None


def _occupied_problem(cell: tuple[int, int], z: int, occupant: Placement) -> str:
    return (
        f"column {cell[0] + 1}, row {cell[1] + 1}, layer {z + 1} is already occupied "
        f"by a {occupant.colour} brick"
    )


def _validate_against(
    working_grid: list[Placement], new_placement: Placement, grid_shape: tuple[int, int],
) -> list[str]:
    """Off-the-plate and unsupported problems `new_placement` would have if it joined
    `working_grid` right now.

    Reuses `structures.validate_structure` on the combined list rather than re-implementing
    bounds/support checking, and keeps only the messages about the placement under test - always
    the last one in the combined list, so its index is known rather than parsed back out of a
    message. (Overlap is checked separately, by `_find_occupied_conflict`, so its message can
    name the colour already there; `validate_structure` would otherwise catch it here too.)
    """

    combined = working_grid + [new_placement]
    prefix = f"placement {len(working_grid)}:"
    raw_problems = validate_structure(combined, grid_shape=grid_shape)
    mine = [p for p in raw_problems if p.startswith(prefix) and "overlaps" not in p]
    off_plate = [p for p in mine if "is out of the" in p]
    if not off_plate:
        return mine
    # One sentence in the counted-from-1 columns and rows the person typed, instead of one
    # message per overhanging cell in grid coordinates.
    return [p for p in mine if p not in off_plate] + [_off_plate_problem(new_placement, grid_shape)]


def _off_plate_problem(placement: Placement, grid_shape: tuple[int, int]) -> str:
    cols = (placement.x + 1, placement.x + placement.w)
    rows = (placement.y + 1, placement.y + placement.h)
    return (
        f"a {placement.h}x{placement.w} brick placed there would overhang the plate: it needs "
        f"columns {cols[0]}-{cols[1]} and rows {rows[0]}-{rows[1]}, and the plate has "
        f"columns 1-{grid_shape[0]} and rows 1-{grid_shape[1]}"
    )


# ---------------------------------------------------------------------------------------------
# 6. Parsing + resolving one segment. Each parser returns a `_SegmentOutcome`.
# ---------------------------------------------------------------------------------------------


@dataclass
class _SegmentOutcome:
    placements: list[Placement]
    relations: list[str | None]
    paraphrase: str | None
    error: str | None


def _parse_single(tokens: list[_Token], working_grid: list[Placement], grid_shape: tuple[int, int]) -> _SegmentOutcome:
    location, main_tokens, err = _read_location(tokens)
    if err:
        return _SegmentOutcome([], [], None, err)

    colour = _find_colour(main_tokens)
    if colour is None:
        return _SegmentOutcome([], [], None, _colour_problem(main_tokens))

    size = _find_size(main_tokens)
    if size is None:
        return _SegmentOutcome([], [], None, "missing size (colour and size are required)")
    problem = _size_problem(size)
    if problem:
        return _SegmentOutcome([], [], None, problem)

    orientation = _find_orientation(main_tokens)
    h, w = size
    if orientation == "rotated":
        h, w = w, h

    layer, layer_note = _read_layer(main_tokens)

    stray = _find_unclaimed_quantity(main_tokens)
    if stray is not None:
        word = stray.text if stray.kind == "number" else _ORDINAL_NAMES.get(stray.value, str(stray.value))
        return _SegmentOutcome([], [], None, f"unrecognized reference '{word}'")

    x, y, z_override, loc_note, loc_err = _resolve_location(location, h, w, grid_shape, working_grid)
    if loc_err:
        return _SegmentOutcome([], [], None, loc_err)
    z = z_override if z_override is not None else (layer if layer is not None else 0)

    placement = Placement(h=h, w=w, x=x, y=y, z=z, colour=colour)
    conflict, cell = _find_occupied_conflict(working_grid, placement)
    if conflict is not None:
        return _SegmentOutcome([], [], None, _occupied_problem(cell, z, conflict))
    extra_problems = _validate_against(working_grid, placement, grid_shape)
    if extra_problems:
        return _SegmentOutcome([], [], None, "; ".join(extra_problems))

    bits = [f"a {colour} {h}x{w} brick", loc_note]
    if orientation == "rotated":
        bits.append("rotated")
    if z_override is None:
        bits.append(layer_note if layer_note else ("on the baseplate" if z == 0 else f"on layer {z + 1}"))
    relation = loc_note if isinstance(location, _RelativeLoc) else None
    return _SegmentOutcome([placement], [relation], ", ".join(bits), None)


@dataclass
class _StructureAttrs:
    count: int
    h: int
    w: int
    colour: str
    orientation: str
    layer: int | None
    layer_note: str | None
    location: _Location
    disclosures: list[str]


def _parse_structure_attrs(tokens: list[_Token], struct_noun: str) -> tuple[_StructureAttrs | None, str | None]:
    location, main_tokens, err = _read_location(tokens)
    if err:
        return None, err

    count = _find_count(main_tokens)
    if count is None or count < 1:
        return None, f"missing count (e.g. 'a {struct_noun} of three 2x2 bricks')"

    disclosures: list[str] = []
    size = _find_size(main_tokens)
    if size is None:
        size = (2, 2)
        disclosures.append("assuming 2x2 bricks, since no size was given")
    else:
        problem = _size_problem(size)
        if problem:
            return None, problem

    colour = _find_colour(main_tokens)
    if colour is None:
        off = _find_off_palette_colour(main_tokens)
        if off:
            return None, _colour_problem(main_tokens)
        colour = "red"
        disclosures.append("assuming red, since no colour was given")

    orientation = _find_orientation(main_tokens)
    h, w = size
    if orientation == "rotated":
        h, w = w, h

    layer, layer_note = _read_layer(main_tokens)
    return _StructureAttrs(count, h, w, colour, orientation, layer, layer_note, location, disclosures), None


def _structure_bits(
    head: str, attrs: _StructureAttrs, loc_note: str | None, z_override: int | None, base_z: int,
) -> list[str]:
    bits = [head, loc_note]
    if attrs.orientation == "rotated":
        bits.append("rotated")
    if z_override is None:
        bits.append(
            attrs.layer_note if attrs.layer_note
            else ("starting on the baseplate" if base_z == 0 else f"starting on layer {base_z + 1}")
        )
    bits.extend(attrs.disclosures)
    return bits


def _parse_tower(tokens: list[_Token], working_grid: list[Placement], grid_shape: tuple[int, int]) -> _SegmentOutcome:
    attrs, err = _parse_structure_attrs(tokens, "tower")
    if err:
        return _SegmentOutcome([], [], None, err)
    x, y, z_override, loc_note, loc_err = _resolve_location(attrs.location, attrs.h, attrs.w, grid_shape, working_grid)
    if loc_err:
        return _SegmentOutcome([], [], None, loc_err)
    base_z = z_override if z_override is not None else (attrs.layer if attrs.layer is not None else 0)

    placements: list[Placement] = []
    relations: list[str | None] = []
    trial_grid = list(working_grid)
    for k in range(attrs.count):
        p = Placement(h=attrs.h, w=attrs.w, x=x, y=y, z=base_z + k, colour=attrs.colour)
        conflict, cell = _find_occupied_conflict(trial_grid, p)
        if conflict is not None:
            return _SegmentOutcome([], [], None, _occupied_problem(cell, p.z, conflict))
        extra_problems = _validate_against(trial_grid, p, grid_shape)
        if extra_problems:
            return _SegmentOutcome([], [], None, "; ".join(extra_problems))
        placements.append(p)
        relations.append(f"layer {k + 1} of a tower of {attrs.count} {attrs.colour} {attrs.h}x{attrs.w} bricks")
        trial_grid.append(p)

    head = f"a tower of {attrs.count} {attrs.colour} {attrs.h}x{attrs.w} bricks"
    bits = _structure_bits(head, attrs, loc_note, z_override, base_z)
    return _SegmentOutcome(placements, relations, ", ".join(bits), None)


def _parse_staircase(tokens: list[_Token], working_grid: list[Placement], grid_shape: tuple[int, int]) -> _SegmentOutcome:
    attrs, err = _parse_structure_attrs(tokens, "staircase")
    if err:
        return _SegmentOutcome([], [], None, err)
    total_w = attrs.count * attrs.w
    x, y, z_override, loc_note, loc_err = _resolve_location(attrs.location, attrs.h, total_w, grid_shape, working_grid)
    if loc_err:
        return _SegmentOutcome([], [], None, loc_err)
    base_z = z_override if z_override is not None else (attrs.layer if attrs.layer is not None else 0)

    placements: list[Placement] = []
    relations: list[str | None] = []
    trial_grid = list(working_grid)
    for i in range(attrs.count):
        for level in range(i + 1):
            p = Placement(h=attrs.h, w=attrs.w, x=x + i * attrs.w, y=y, z=base_z + level, colour=attrs.colour)
            conflict, cell = _find_occupied_conflict(trial_grid, p)
            if conflict is not None:
                return _SegmentOutcome([], [], None, _occupied_problem(cell, p.z, conflict))
            extra_problems = _validate_against(trial_grid, p, grid_shape)
            if extra_problems:
                return _SegmentOutcome([], [], None, "; ".join(extra_problems))
            placements.append(p)
            relations.append(
                f"step {i + 1} of {attrs.count} (layer {level + 1}) in a staircase of "
                f"{attrs.colour} {attrs.h}x{attrs.w} bricks"
            )
            trial_grid.append(p)

    head = f"a staircase of {attrs.count} steps, {attrs.colour} {attrs.h}x{attrs.w} bricks"
    bits = _structure_bits(head, attrs, loc_note, z_override, base_z)
    return _SegmentOutcome(placements, relations, ", ".join(bits), None)


def _parse_wall(tokens: list[_Token], working_grid: list[Placement], grid_shape: tuple[int, int]) -> _SegmentOutcome:
    attrs, err = _parse_structure_attrs(tokens, "wall")
    if err:
        return _SegmentOutcome([], [], None, err)
    total_w = attrs.count * attrs.w
    x, y, z_override, loc_note, loc_err = _resolve_location(attrs.location, attrs.h, total_w, grid_shape, working_grid)
    if loc_err:
        return _SegmentOutcome([], [], None, loc_err)
    base_z = z_override if z_override is not None else (attrs.layer if attrs.layer is not None else 0)

    placements: list[Placement] = []
    relations: list[str | None] = []
    trial_grid = list(working_grid)
    for k in range(attrs.count):
        p = Placement(h=attrs.h, w=attrs.w, x=x + k * attrs.w, y=y, z=base_z, colour=attrs.colour)
        conflict, cell = _find_occupied_conflict(trial_grid, p)
        if conflict is not None:
            return _SegmentOutcome([], [], None, _occupied_problem(cell, p.z, conflict))
        extra_problems = _validate_against(trial_grid, p, grid_shape)
        if extra_problems:
            return _SegmentOutcome([], [], None, "; ".join(extra_problems))
        placements.append(p)
        relations.append(f"brick {k + 1} of {attrs.count} in a wall of {attrs.colour} {attrs.h}x{attrs.w} bricks")
        trial_grid.append(p)

    head = f"a wall of {attrs.count} {attrs.colour} {attrs.h}x{attrs.w} bricks"
    bits = _structure_bits(head, attrs, loc_note, z_override, base_z)
    return _SegmentOutcome(placements, relations, ", ".join(bits), None)


_STRUCTURE_PARSERS = {"tower": _parse_tower, "staircase": _parse_staircase, "wall": _parse_wall}


def _parse_segment(tokens: list[_Token], working_grid: list[Placement], grid_shape: tuple[int, int]) -> _SegmentOutcome:
    if not tokens:
        return _SegmentOutcome([], [], None, "empty step")
    for word, parser in _STRUCTURE_PARSERS.items():
        if any(t.kind == "word" and t.text == word for t in tokens):
            return parser(tokens, working_grid, grid_shape)
    return _parse_single(tokens, working_grid, grid_shape)


# ---------------------------------------------------------------------------------------------
# Public interface.
# ---------------------------------------------------------------------------------------------


@dataclass
class CompileResult:
    """What `compile_command` hands back.

    `goal` is `None` whenever `understood` is `False` - this compiler never guesses, so a
    command it cannot fully resolve produces no goal at all, only `problems`. `paraphrase` is
    empty in that case too, for the same reason: there is nothing confirmed to restate.
    """

    goal: Goal | None
    understood: bool
    problems: list[str]
    paraphrase: str


def compile_command(
    text: str,
    *,
    inventory: Inventory | None = None,
    current_grid: Sequence[Placement] | None = None,
) -> CompileResult:
    """Compile one ordinary-English command into a `Goal`, once.

    `inventory` (a `goal.Inventory`) limits which bricks are available to place; left as `None`,
    the tray is treated as never running out. `current_grid` (a list of `structures.Placement`,
    the same format `Goal.remaining_steps` takes) describes what is already on the plate, both
    for validity checking and for resolving relative phrases ("on top of the red brick"). Earlier
    steps within the same command are added to that picture as they are resolved, so a multi-step
    command can refer to a brick it just placed.
    """

    segments = _tokenize_and_segment(text)
    if not segments:
        return CompileResult(None, False, ["no command found in the input"], "")

    working_grid: list[Placement] = list(current_grid) if current_grid else []
    steps: list[Step] = []
    problems: list[str] = []
    paraphrases: list[str] = []

    for seg_index, seg_tokens in enumerate(segments, start=1):
        outcome = _parse_segment(seg_tokens, working_grid, GRID_SHAPE)
        if outcome.error:
            problems.append(f"step {seg_index}: {outcome.error}")
            continue
        for placement, relation in zip(outcome.placements, outcome.relations):
            steps.append(Step(placement=placement, source="tray", relation=relation))
            working_grid.append(placement)
        if outcome.paraphrase:
            paraphrases.append(outcome.paraphrase)

    if problems:
        return CompileResult(None, False, problems, "")

    # Overlap, bounds, and support were already checked for every step as it was resolved, above
    # (via `_find_occupied_conflict` and `_validate_against`), each time against `current_grid`
    # plus every step already added - so, unlike `Goal.problems()` on its own, that checking
    # knows about a brick this command relies on for support that came from `current_grid`
    # rather than from an earlier step. All that is left is the tray's stock, which is a whole-
    # command question (three requested bricks can be a problem even if any one of them alone
    # would not be).
    goal = Goal(steps=tuple(steps), grid_shape=GRID_SHAPE)
    shortages = inventory.shortages(goal.placements) if inventory is not None else []
    if shortages:
        return CompileResult(None, False, shortages, "")

    return CompileResult(goal, True, [], "; then ".join(paraphrases))
