"""The one text system for the Brick Builder Arm Robot videos: every string, drawn with PIL.

There is exactly one way text reaches a frame in this toolkit, and it is :meth:`Overlay.draw`.
The frame arrives at its finished size and every glyph, box and line is drawn at that size --
nothing is drawn small and scaled up, which is what made the earlier attempt unreadable.

Four elements exist, and no others:

============  ==========================================================================
title block   top left, over the opening seconds: the film's name and one spec line
phase label   top left, once the title has gone: the step the arm is on
panel grid    right-hand column: `n` equal square panels, each titled above itself
data strip    along the bottom of the main view: five readouts, one a full-width plot
============  ==========================================================================

The panels are square because what goes in them is square -- a camera view, a scene model, a
feature map -- and :meth:`Overlay.clear_rect` is the picture left over for the arm: everything
left of the grid and above the strip.

Every string drawn here is a title: a noun phrase, or a short imperative for a step, with units
in square brackets.  No sentences, no captions, no commentary.  The strings are the caller's;
this module supplies no wording of its own and refuses anything that does not fit its box.

Sizes are fractions of the frame height, so they hold at any output size.  The panel column and
the data strip are right-aligned and together leave the left :data:`MAIN_CLEAR_FRAC` of the frame
free for the arm; everything sits inside a 4 percent safe margin.

:func:`check_layout` is run on every frame: it takes the rectangle of every element that was
drawn and raises :class:`LayoutError` if any two of them touch or if any one leaves the safe
area.  Overlapping text cannot reach a finished file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["Rect", "LayoutError", "check_layout", "Readout", "Overlay", "fit_caption"]


# DejaVu Sans ships with matplotlib, which this project already depends on.
def _font_dir() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


# Fractions of the frame height.
TITLE_FRAC = 0.045
SUBTITLE_FRAC = 0.024
"""The spec line under the title.  It is one dense line of settings rather than a sentence, so it
gets its own smaller face and is allowed to run wider than the title itself."""

PHASE_FRAC = 0.034
VALUE_FRAC = 0.050
NUMBER_FRAC = 0.030
LABEL_FRAC = 0.025
SMALL_FRAC = 0.022

CAPTION_FRAC = 0.030
"""Kept so older imports still resolve.  Nothing draws a caption any more."""

MARGIN_FRAC = 0.04

# The right-hand column holds the panel grid; the data strip runs along the bottom of what is
# left, under the main view.  The panels are square -- the model's window is square, and a
# feature map squeezed into a letterbox cannot be read -- so their side falls out of the column
# width, the grid shape and the number of panels.  :meth:`Overlay.panel_size` computes it once.
PANEL_COL_W_FRAC = 0.41
PANEL_COLS = 2
PANEL_GAP_FRAC = 0.012
STRIP_H_FRAC = 0.37

MAIN_CLEAR_FRAC = 0.55
"""Fraction of the frame width, on the left, that the panel grid may not reach into.

Checked when the overlay is built, so a change to the fractions above that would crowd the arm
fails at once rather than on the first frame of a take."""

MAX_SPEC_LINES = 3
"""Lines the spec line under the title may take.  The panel grid takes two fifths of the width,
so a full settings line needs three at 1920 wide."""

MAX_READOUTS = 5
COMPACT_COLS = 2
"""Columns of compact cells in the data strip.  Two, not four: a title with its unit in brackets
needs about a third of the strip, and a title clipped to fit is not a title."""

TRACE_HEADROOM = 0.18
"""Fraction of a trace box left empty above the top of its scale.

Without it a reading sitting at the top of its range draws a line flush against the label above,
and the panel reads as two stray horizontal rules instead of a graph."""

TEXT = (247, 249, 252, 255)
DIM_TEXT = (215, 221, 232, 255)
BOX = (17, 20, 27, 179)         # 70 percent opaque: title, phase label
STRIP_BOX = (17, 20, 27, 240)   # the data strip carries numbers, so it gets its own solid panel
BORDER = (232, 236, 243, 255)
TRACE_COLOURS = ((99, 178, 255, 255), (247, 160, 86, 255))
BASELINE = (150, 158, 175, 170)
REFERENCE = (198, 206, 222, 200)   # the dashed line at the value the run is heading for


@dataclass(frozen=True)
class Rect:
    """A drawn element's pixel rectangle, in the usual left, top, right, bottom order."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def intersects(self, other: "Rect") -> bool:
        return not (self.x1 <= other.x0 or other.x1 <= self.x0
                    or self.y1 <= other.y0 or other.y1 <= self.y0)

    def inside(self, other: "Rect") -> bool:
        return (self.x0 >= other.x0 and self.y0 >= other.y0
                and self.x1 <= other.x1 and self.y1 <= other.y1)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


class LayoutError(RuntimeError):
    """Two elements touched, or one left the safe area.  Raised before the frame is written."""


def check_layout(rects: dict[str, Rect], safe: Rect) -> None:
    problems = []
    for name, r in rects.items():
        if not r.inside(safe):
            problems.append(f"{name} {r.as_tuple()} leaves the safe area {safe.as_tuple()}")
    names = sorted(rects)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if rects[a].intersects(rects[b]):
                problems.append(f"{a} {rects[a].as_tuple()} overlaps {b} {rects[b].as_tuple()}")
    if problems:
        raise LayoutError("; ".join(problems))


def fit_caption(text: str, measure, max_px: float, *, max_lines: int = 2) -> list[str]:
    """Break a line of text into as few lines as fit ``max_px`` pixels, measured, not guessed.

    A newline in ``text`` is a break the author asked for and is kept as written.  Otherwise:
    one line if the whole string fits, and failing that as even a split as the word boundaries
    allow.  Raises if it cannot be done in ``max_lines``.
    """

    if "\n" in text:
        lines = [" ".join(part.split()) for part in text.split("\n")]
        lines = [line for line in lines if line]
        widest = max((measure(line) for line in lines), default=0.0)
        if len(lines) > max_lines or widest > max_px:
            raise ValueError(
                f"the given line breaks give {len(lines)} lines up to {widest:.0f} px; "
                f"the limit is {max_lines} lines of {max_px:.0f} px: {text!r}"
            )
        return lines

    text = " ".join(text.split())
    if not text:
        return []
    if measure(text) <= max_px:
        return [text]
    words = text.split(" ")
    for lines_wanted in range(2, max_lines + 1):
        best: list[str] | None = None
        best_worst = None
        for cuts in _splits(len(words), lines_wanted):
            lines = [" ".join(words[a:b]) for a, b in zip((0,) + cuts, cuts + (len(words),))]
            widths = [measure(line) for line in lines]
            if max(widths) > max_px:
                continue
            worst = max(widths) - min(widths)
            if best_worst is None or worst < best_worst:
                best, best_worst = lines, worst
        if best is not None:
            return best
    raise ValueError(
        f"text does not fit {max_px:.0f} px in {max_lines} lines at this size: {text!r}"
    )


def _splits(n: int, parts: int) -> list[tuple[int, ...]]:
    """Every way of cutting ``n`` words into ``parts`` non-empty runs, as cut positions."""

    if parts == 2:
        return [(i,) for i in range(1, n)]
    out: list[tuple[int, ...]] = []
    for i in range(1, n - parts + 2):
        for rest in _splits(n - i, parts - 1):
            out.append((i,) + tuple(i + r for r in rest))
    return out


@dataclass(frozen=True)
class Readout:
    """One number in the data strip: its title, its value and the history behind it."""

    label: str
    value: float | None
    """`None` means nothing measured yet, which prints as a dash rather than a fake zero."""
    unit: str
    trace: Sequence[float] = field(default_factory=tuple)
    lo: float = 0.0
    hi: float = 1.0
    fmt: str = "{:.1f}"

    suffix: str = ""
    """Small dim text beside the number: a target value, not a remark."""

    reference: float | None = None
    """A value worth a dashed line across the plot, in the plot's own units."""

    marks: Sequence[float] = field(default_factory=tuple)
    """Fractions along the trace (0..1) worth a thin vertical rule -- where one step ended and
    the next began."""

    plot: bool = False
    """Draw this one as the strip's full-width time plot.  At most one readout may ask for it;
    the rest are drawn as compact title-and-number cells."""

    def text(self) -> str:
        if self.value is None:
            return "—"
        return f"{self.fmt.format(self.value)} {self.unit}".strip()


class Overlay:
    """Draws the four elements onto a finished frame, and checks the layout every time."""

    def __init__(self, width: int, height: int):
        self.width, self.height = int(width), int(height)
        d = _font_dir()
        self.f_title = ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), self._px(TITLE_FRAC))
        self.f_subtitle = ImageFont.truetype(str(d / "DejaVuSans.ttf"), self._px(SUBTITLE_FRAC))
        self.f_phase = ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), self._px(PHASE_FRAC))
        self.f_value = ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), self._px(VALUE_FRAC))
        self.f_number = ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), self._px(NUMBER_FRAC))
        self.f_label = ImageFont.truetype(str(d / "DejaVuSans.ttf"), self._px(LABEL_FRAC))
        self.f_small = ImageFont.truetype(str(d / "DejaVuSans.ttf"), self._px(SMALL_FRAC))

        self.margin = int(round(MARGIN_FRAC * self.height))
        self.margin_x = int(round(MARGIN_FRAC * self.width))
        self.pad = int(round(0.018 * self.height))
        self.radius = int(round(0.014 * self.height))
        self.safe = Rect(self.margin_x, self.margin,
                         self.width - self.margin_x, self.height - self.margin)
        self.column_w = int(round(PANEL_COL_W_FRAC * self.width))
        """Width reserved on the right for the panel grid, whatever the grid does with it."""

        self.column_x = self.safe.x1 - self.column_w
        self.panel_gap = int(round(PANEL_GAP_FRAC * self.height))

        if self.column_x < MAIN_CLEAR_FRAC * self.width:
            raise LayoutError(
                f"the panel grid starts at x={self.column_x} and would leave only "
                f"{self.column_x / self.width:.0%} of the frame clear for the arm; "
                f"{MAIN_CLEAR_FRAC:.0%} is the floor"
            )

    def _px(self, frac: float) -> int:
        return max(int(round(frac * self.height)), 8)

    # -- geometry of the fixed boxes -------------------------------------------------------

    def strip_rect(self) -> Rect:
        """The data strip: along the bottom of the main view, left of the panel grid."""

        h = int(round(STRIP_H_FRAC * self.height))
        return Rect(self.margin_x, self.height - self.margin - h,
                    self.column_x - self.pad, self.height - self.margin)

    def clear_rect(self) -> Rect:
        """The picture no panel and no data strip covers: what the shot has to be framed in.

        The title block and the phase label sit in its top left corner over the opening, and
        nothing else is drawn inside it.
        """

        return Rect(0, 0, self.column_x - self.pad, self.strip_rect().y0 - self.pad)

    def _panel_bar(self) -> int:
        """Height of one panel's title bar, above its picture.

        The bar carries the same face as the data strip's own titles: a four-word panel title
        has to fit the width of a square panel, which 27 px type does not leave room for.
        """

        return self._line_height(self.f_small) + max(self.pad // 2, 4)

    def _grid(self, n: int) -> tuple[int, int]:
        """Columns and rows of the panel grid for ``n`` panels."""

        n = int(n)
        if n < 1:
            raise ValueError(f"a panel grid holds at least one panel, got {n}")
        cols = min(n, PANEL_COLS)
        return cols, -(-n // cols)

    def panel_size(self, n: int) -> tuple[int, int]:
        """The exact ``(width, height)`` of the picture in each of ``n`` equal panels: a square.

        The title bar is drawn above the picture and is not part of this size: hand :meth:`draw`
        an array of exactly these pixels.
        """

        cols, rows = self._grid(n)
        by_width = (self.column_w - (cols - 1) * self.panel_gap) // cols
        room = self.safe.height - rows * self._panel_bar() - (rows - 1) * self.panel_gap
        side = min(by_width, room // rows)
        if side < 24:
            raise LayoutError(
                f"{n} panels in a {cols}x{rows} grid leave {side} px a side; the column is "
                f"{self.column_w} px wide and {self.safe.height} px tall"
            )
        return side, side

    def panels_rect(self, n: int) -> Rect:
        """The block the whole grid of ``n`` panels occupies, title bars included."""

        cols, rows = self._grid(n)
        side, _ = self.panel_size(n)
        width = cols * side + (cols - 1) * self.panel_gap
        height = rows * (side + self._panel_bar()) + (rows - 1) * self.panel_gap
        return Rect(self.safe.x1 - width, self.margin, self.safe.x1, self.margin + height)

    def panel_rects(self, n: int) -> list[Rect]:
        """Every panel's rectangle, title bar included, in the order they are drawn."""

        cols, _ = self._grid(n)
        side, _ = self.panel_size(n)
        bar = self._panel_bar()
        grid = self.panels_rect(n)
        out = []
        for i in range(int(n)):
            x0 = grid.x0 + (i % cols) * (side + self.panel_gap)
            y0 = grid.y0 + (i // cols) * (side + bar + self.panel_gap)
            out.append(Rect(x0, y0, x0 + side, y0 + bar + side))
        return out

    def pip_rect(self, size: tuple[int, int], *, label: bool = False) -> Rect:
        """Where the single inset goes.  With a label it is taller by one line plus its padding."""

        w, h = int(size[0]), int(size[1])
        if label:
            h += self._line_height(self.f_label) + self.pad
        return Rect(self.width - self.margin_x - w, self.margin,
                    self.width - self.margin_x, self.margin + h)

    # -- drawing ---------------------------------------------------------------------------

    def draw(
        self,
        picture: np.ndarray,
        *,
        title: str | None = None,
        subtitle: str | None = None,
        phase: str | None = None,
        readouts: Sequence[Readout] = (),
        panels: Sequence[tuple[str, np.ndarray]] = (),
        pip: np.ndarray | None = None,
        pip_label: str | None = None,
        caption: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Rect]]:
        """Compose one finished frame.  Returns the pixels and every element's rectangle.

        ``panels`` is a sequence of ``(title, rgb)`` pairs drawn as a grid of equal square
        panels in the right-hand column; each array must be exactly :meth:`panel_size`.  ``pip``
        and
        ``pip_label`` are the older single inset, kept for the scripted film.  ``caption`` is
        gone: the layout has no caption element and a non-``None`` one is refused.
        """

        if caption is not None:
            raise ValueError(
                "the caption element was removed from the layout; the title block takes a title "
                "and one spec line, and every other string is a title on its own box"
            )
        if panels and pip is not None:
            raise ValueError("pass either a panel column or the single inset, not both")
        if picture.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"overlay built for {self.width}x{self.height}, got a "
                f"{picture.shape[1]}x{picture.shape[0]} frame"
            )
        base = Image.fromarray(np.ascontiguousarray(picture), mode="RGB").convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rects: dict[str, Rect] = {}

        if panels:
            rects.update(self._draw_panels(base, draw, panels))
        elif pip is not None:
            rects["close_up"] = self._draw_pip(base, draw, pip, pip_label)
        limit_x = min([r.x0 - self.pad for r in rects.values()] or [self.safe.x1])
        if title:
            rects["title"] = self._draw_title(draw, title, subtitle, limit_x=limit_x)
        elif phase:
            rects["phase"] = self._draw_phase(draw, phase, limit_x=limit_x)
        if readouts:
            rects["data_strip"] = self._draw_strip(draw, readouts)

        check_layout(rects, self.safe)
        out = Image.alpha_composite(base, layer).convert("RGB")
        return np.asarray(out, dtype=np.uint8), rects

    def _box(self, draw: ImageDraw.ImageDraw, rect: Rect, fill=BOX) -> None:
        draw.rounded_rectangle(rect.as_tuple(), radius=self.radius, fill=fill)

    def _line_height(self, font: ImageFont.FreeTypeFont) -> int:
        ascent, descent = font.getmetrics()
        return int(round((ascent + descent) * 1.02))

    # -- title block and phase label ---------------------------------------------------------

    def _spec_lines(self, spec: str, max_px: float) -> list[str]:
        """The spec line, on one line if it fits and otherwise split at its ``|`` separators.

        The fewest lines that fit win, and among those the split with the narrowest widest line,
        so the block comes out as even as the separators allow.  A spec with no separators falls
        back to breaking on words.
        """

        spec = " ".join(spec.split())
        measure = self.f_subtitle.getlength
        if measure(spec) <= max_px:
            return [spec]
        parts = [p.strip() for p in spec.split("|") if p.strip()]
        if len(parts) > 1:
            for wanted in range(2, MAX_SPEC_LINES + 1):
                best: tuple[float, list[str]] | None = None
                for cuts in _splits(len(parts), wanted):
                    lines = [" | ".join(parts[a:b])
                             for a, b in zip((0,) + cuts, cuts + (len(parts),))]
                    widest = max(measure(line) for line in lines)
                    if widest <= max_px and (best is None or widest < best[0]):
                        best = (widest, lines)
                if best is not None:
                    return best[1]
            raise LayoutError(
                f"the spec line does not fit {MAX_SPEC_LINES} lines of {max_px:.0f} px: {spec!r}"
            )
        try:
            return fit_caption(spec, measure, max_px, max_lines=MAX_SPEC_LINES)
        except ValueError as exc:
            raise LayoutError(str(exc)) from exc

    def _draw_title(self, draw, title: str, subtitle: str | None, *, limit_x: int) -> Rect:
        max_px = limit_x - self.margin_x - 2 * self.pad
        if self.f_title.getlength(title) > max_px:
            raise LayoutError(f"the title does not fit {max_px:.0f} px: {title!r}")
        lines = [(title, self.f_title, TEXT)]
        if subtitle:
            lines += [(line, self.f_subtitle, DIM_TEXT)
                      for line in self._spec_lines(subtitle, max_px)]
        w = max(int(round(font.getlength(text))) for text, font, _ in lines)
        h = sum(self._line_height(font) for _, font, _ in lines)
        rect = Rect(self.margin_x, self.margin,
                    self.margin_x + w + 2 * self.pad, self.margin + h + 2 * self.pad)
        self._box(draw, rect)
        y = rect.y0 + self.pad
        for text, font, colour in lines:
            draw.text((rect.x0 + self.pad, y), text, font=font, fill=colour)
            y += self._line_height(font)
        return rect

    def _draw_phase(self, draw, phase: str, *, limit_x: int) -> Rect:
        max_px = limit_x - self.margin_x - 2 * self.pad
        if self.f_phase.getlength(phase) > max_px:
            raise LayoutError(f"the phase label does not fit {max_px:.0f} px: {phase!r}")
        w = int(round(self.f_phase.getlength(phase)))
        h = self._line_height(self.f_phase)
        rect = Rect(self.margin_x, self.margin,
                    self.margin_x + w + 2 * self.pad, self.margin + h + 2 * self.pad)
        self._box(draw, rect)
        draw.text((rect.x0 + self.pad, rect.y0 + self.pad), phase, font=self.f_phase, fill=TEXT)
        return rect

    # -- the panel column ----------------------------------------------------------------------

    def _draw_panels(self, base: Image.Image, draw,
                     panels: Sequence[tuple[str, np.ndarray]]) -> dict[str, Rect]:
        """A grid of equal square panels in the right-hand column, each titled above itself."""

        n = len(panels)
        side, _ = self.panel_size(n)
        bar_h = self._panel_bar()
        thickness = max(int(round(0.0022 * self.height)), 2)
        room = side - 2 * (self.pad // 2)
        rects: dict[str, Rect] = {}

        for (title, image), rect in zip(panels, self.panel_rects(n)):
            i = len(rects)
            array = np.asarray(image)
            if array.shape != (side, side, 3):
                raise ValueError(
                    f"panel {i} {title!r} has shape {array.shape}; panel_size({n}) asks for "
                    f"exactly {(side, side, 3)} -- {side} by {side} pixels, RGB"
                )
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if self.f_small.getlength(title) > room:
                raise LayoutError(
                    f"the panel's title does not fit its bar of {room} px: {title!r}"
                )

            bar = Rect(rect.x0, rect.y0, rect.x1, rect.y0 + bar_h)
            base.paste(Image.fromarray(np.ascontiguousarray(array), mode="RGB"),
                       (rect.x0, bar.y1))
            draw.rectangle(bar.as_tuple(), fill=STRIP_BOX)
            draw.text((bar.x0 + self.pad // 2,
                       bar.y0 + (bar_h - self._line_height(self.f_small)) // 2), title,
                      font=self.f_small, fill=DIM_TEXT)
            draw.rectangle(rect.as_tuple(), outline=BORDER, width=thickness)
            rects[f"panel_{i}"] = rect
        return rects

    def _draw_pip(self, base: Image.Image, draw, pip: np.ndarray,
                  label: str | None = None) -> Rect:
        """The older single inset, with its title bar under it."""

        h, w = pip.shape[:2]
        rect = self.pip_rect((w, h), label=bool(label))
        if label and self.f_label.getlength(label) > rect.width - 2 * self.pad:
            raise LayoutError(f"the close-up's label does not fit its panel: {label!r}")
        base.paste(Image.fromarray(np.ascontiguousarray(pip), mode="RGB"), (rect.x0, rect.y0))
        thickness = max(int(round(0.0022 * self.height)), 2)
        if label:
            bar = Rect(rect.x0, rect.y0 + h, rect.x1, rect.y1)
            draw.rectangle(bar.as_tuple(), fill=STRIP_BOX)
            draw.text((bar.x0 + self.pad, bar.y0 + self.pad // 2), label,
                      font=self.f_label, fill=DIM_TEXT)
        draw.rectangle(rect.as_tuple(), outline=BORDER, width=thickness)
        return rect

    # -- the data strip ------------------------------------------------------------------------

    def _draw_strip(self, draw, readouts: Sequence[Readout]) -> Rect:
        """Up to five readouts: one full-width plot, the rest as compact title-and-number cells."""

        if len(readouts) > MAX_READOUTS:
            raise LayoutError(
                f"the data strip holds {MAX_READOUTS} readouts, got {len(readouts)}"
            )
        plotted = [r for r in readouts if r.plot]
        if len(plotted) > 1:
            raise LayoutError(
                f"{len(plotted)} readouts ask for the plot; the strip draws one"
            )
        compact = [r for r in readouts if not r.plot]
        rect = self.strip_rect()
        self._box(draw, rect, fill=STRIP_BOX)

        x0, x1 = rect.x0 + self.pad, rect.x1 - self.pad
        top, bottom = rect.y0 + self.pad, rect.y1 - self.pad
        cols = min(len(compact), COMPACT_COLS) or 1
        rows = -(-len(compact) // cols)
        cell_w = (x1 - x0) // cols
        room = cell_w - self.pad // 2
        # Title beside the number where every cell has the room for it, and title over number
        # where one of them has not: one shape for the whole strip, decided once.
        beside = all(self._cell_width(r) + self.pad <= room for r in compact)
        cell_h = self._line_height(self.f_number) + self.pad // 3
        if not beside:
            cell_h += self._line_height(self.f_small)
        if plotted:
            block = rows * cell_h + (self.pad if rows else 0)
            self._draw_plot(draw, Rect(x0, top, x1, bottom - block), plotted[0])
            cells_top = bottom - rows * cell_h
        else:
            cells_top = top
            cell_h = (bottom - top) // max(rows, 1)
        for i, readout in enumerate(compact):
            cx = x0 + (i % cols) * cell_w
            cy = cells_top + (i // cols) * cell_h
            self._draw_cell(draw, Rect(cx, cy, cx + cell_w, cy + cell_h), readout,
                            beside=beside)
        return rect

    def _cell_width(self, readout: Readout) -> float:
        """What one compact cell's title and number need side by side, without the gap."""

        return (self.f_small.getlength(readout.label)
                + self._number_width(readout, self.f_number, self.f_small)
                + 2 * self.pad)

    def _number_width(self, readout: Readout, font, suffix_font) -> float:
        width = font.getlength(readout.text())
        if readout.suffix:
            width += self.pad // 3 + suffix_font.getlength(readout.suffix)
        return width

    def _draw_cell(self, draw, rect: Rect, readout: Readout, *, beside: bool) -> None:
        """One compact readout: its title in small caps, the number beside it or under it."""

        label = readout.label
        value = readout.text()
        # A real gutter when the number sits at the right of its cell: with only a few pixels
        # there, a right-aligned number lands against the next cell's title and the two read as
        # one phrase -- "86 Prediction error [mm]".
        room = rect.width - (2 * self.pad if beside else self.pad // 2)
        label_w = self.f_small.getlength(label)
        number_w = self.f_number.getlength(value)
        used = self._number_width(readout, self.f_number, self.f_small)
        if label_w > room:
            raise LayoutError(
                f"the data strip's label does not fit its cell of {room} px: {label!r}"
            )
        if used > room:
            raise LayoutError(
                f"the data strip's number does not fit its cell of {room} px: {value!r}"
            )
        # The suffix sits on the number's baseline, so the two read as one phrase.
        drop = self.f_number.getmetrics()[0] - self.f_small.getmetrics()[0]
        if beside:
            number_x = rect.x0 + room - used
            draw.text((rect.x0, rect.y0 + drop), label, font=self.f_small, fill=DIM_TEXT)
            draw.text((number_x, rect.y0), value, font=self.f_number, fill=TEXT)
            suffix_y = rect.y0 + drop
        else:
            draw.text((rect.x0, rect.y0), label, font=self.f_small, fill=DIM_TEXT)
            number_x = rect.x0
            number_y = rect.y0 + self._line_height(self.f_small)
            draw.text((number_x, number_y), value, font=self.f_number, fill=TEXT)
            suffix_y = number_y + drop
        if readout.suffix:
            draw.text((number_x + number_w + self.pad // 3, suffix_y), readout.suffix,
                      font=self.f_small, fill=DIM_TEXT)

    def _draw_plot(self, draw, rect: Rect, readout: Readout) -> None:
        """The plotted readout: its title, the big number under it, the trace under that."""

        label = readout.label
        value = readout.text()
        gap = self.pad // 3
        label_w = self.f_label.getlength(label)
        number_w = self.f_value.getlength(value)
        used = self._number_width(readout, self.f_value, self.f_label)
        if label_w > rect.width:
            raise LayoutError(
                f"the data strip's label does not fit its panel of {rect.width} px: {label!r}"
            )
        if used > rect.width:
            raise LayoutError(
                f"the data strip's number does not fit its panel of {rect.width} px: {value!r}"
            )
        # Title beside the big number where the strip is wide enough, over it where it is not;
        # either way the plot takes whatever height the header leaves.
        drop = self.f_value.getmetrics()[0] - self.f_label.getmetrics()[0]
        if label_w + self.pad + used <= rect.width:
            header_h = self._line_height(self.f_value)
            value_x, value_y = rect.x1 - used, rect.y0
            draw.text((rect.x0, rect.y0 + drop), label, font=self.f_label, fill=DIM_TEXT)
        else:
            header_h = self._line_height(self.f_label) + self._line_height(self.f_value)
            value_x = rect.x0
            value_y = rect.y0 + self._line_height(self.f_label)
            draw.text((rect.x0, rect.y0), label, font=self.f_label, fill=DIM_TEXT)
        draw.text((value_x, value_y), value, font=self.f_value, fill=TEXT)
        if readout.suffix:
            # Sits on the big number's baseline, so the two read as one phrase.
            draw.text((value_x + number_w + gap, value_y + drop), readout.suffix,
                      font=self.f_label, fill=DIM_TEXT)

        # Before the first measurement the scale is meaningless -- an axis reading 0.00 at both
        # ends says the plot is broken rather than empty -- so it is simply not drawn.
        has_data = len(readout.trace) >= 2
        top_text = f"{readout.fmt.format(readout.hi)} {readout.unit}".strip() if has_data else ""
        bottom_text = readout.fmt.format(readout.lo) if has_data else ""
        gutter = int(round(max(self.f_small.getlength(top_text),
                               self.f_small.getlength(bottom_text)))) + self.pad // 2
        plot = Rect(rect.x0, rect.y0 + header_h + self.pad // 3, rect.x1 - gutter, rect.y1)
        if plot.height < 24 or plot.width < 64:
            raise LayoutError(
                f"the plot has {plot.width}x{plot.height} px to draw in"
            )
        self._draw_trace(draw, plot, readout, TRACE_COLOURS[0])
        if has_data:
            self._draw_axis(draw, plot, rect.x1, top_text, bottom_text, readout)

    def _draw_axis(self, draw, plot: Rect, right: int, top_text: str, bottom_text: str,
                   readout: Readout) -> None:
        """Top and bottom of the scale, and the dashed line's own value, in the gutter."""

        draw.text((right, plot.y1 - self._usable_height(plot)), top_text,
                  font=self.f_small, fill=DIM_TEXT, anchor="rm")
        draw.text((right, plot.y1), bottom_text, font=self.f_small, fill=DIM_TEXT, anchor="rm")
        if readout.reference is None:
            return
        span = max(readout.hi - readout.lo, 1e-9)
        frac = float(np.clip((readout.reference - readout.lo) / span, 0.0, 1.0))
        y = plot.y1 - frac * self._usable_height(plot)
        # Nudged clear of the scale ends rather than dropped: the dashed line is the tolerance
        # and the number on it is the point of the plot, so it is always drawn.
        room = self._line_height(self.f_small)
        top_y = plot.y1 - self._usable_height(plot)
        if abs(y - plot.y1) < room:
            y = plot.y1 - room
        elif abs(y - top_y) < room:
            y = top_y + room
        draw.text((right, y), readout.fmt.format(readout.reference),
                  font=self.f_small, fill=REFERENCE, anchor="rm")

    def _trace_width(self) -> int:
        return max(int(round(0.0028 * self.height)), 3)

    def _usable_height(self, rect: Rect) -> float:
        return (rect.height - self._trace_width()) * (1.0 - TRACE_HEADROOM)

    def _draw_trace(self, draw, rect: Rect, readout: Readout, colour) -> None:
        """A time trace: one line, a baseline, the step rules and one dashed reference.

        The plot is inset from the top of its box by :data:`TRACE_HEADROOM`, so a reading at the
        very top of its range does not draw flush against the line above it.
        """

        width = self._trace_width()
        plot_h = self._usable_height(rect)
        span = max(readout.hi - readout.lo, 1e-9)

        def y_of(value: float) -> float:
            return rect.y1 - float(np.clip((value - readout.lo) / span, 0.0, 1.0)) * plot_h

        draw.line([(rect.x0, rect.y1), (rect.x1, rect.y1)], fill=BASELINE, width=max(width - 1, 2))
        for f in readout.marks:
            x = rect.x0 + float(np.clip(f, 0.0, 1.0)) * (rect.x1 - rect.x0)
            draw.line([(x, rect.y1), (x, rect.y1 - plot_h)], fill=BASELINE, width=1)
        values = np.asarray(list(readout.trace), dtype=float)
        if values.size >= 2:
            xs = np.linspace(rect.x0, rect.x1, values.size)
            ys = [y_of(float(v)) for v in values]
            draw.line([(float(x), float(y)) for x, y in zip(xs, ys)], fill=colour, width=width,
                      joint="curve")
        # Drawn last, so that a trace which has arrived at the reference does not hide it: the
        # dashes stay visible over the line, which is what "it got there" looks like.
        if readout.reference is not None:
            y = y_of(readout.reference)
            dash, gap = 4 * width, 3 * width
            x = rect.x0
            while x < rect.x1:
                draw.line([(x, y), (min(x + dash, rect.x1), y)], fill=REFERENCE,
                          width=max(width - 1, 2))
                x += dash + gap
