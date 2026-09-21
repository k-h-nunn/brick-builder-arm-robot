"""Shared plumbing every experiment module uses: output directories, report writing, tables.

Nothing here builds a scene or touches MuJoCo; it is pure file/text/JSON handling so it can be
imported cheaply by every experiment (including ``params``, which never opens a simulator).
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..params import Assumption, ParamFile, list_assumptions  # noqa: E402

# Readable-labels-on-a-phone defaults for every figure an experiment saves. A member is going to
# look at these full-resolution, not squint at a thumbnail (the task's own instruction: no GIFs,
# no thumbnails, large readable labels, light background, nothing overlapping).
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 11,
})

FIGURE_DPI = 170
"""Full resolution, not a thumbnail: every experiment figure is saved at this dpi or higher."""

# The Okabe-Ito colourblind-safe categorical palette already used by brick_builder.bricks.materials'
# own figures, reused here so every figure in the project reads consistently.
SERIES_COLOURS: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#111111",
)
STATUS_COLOURS: dict[str, str] = {"assumed": "#5a5a5a", "derived": "#D55E00", "measured": "#009E73"}
INK = "#1a1a1a"
MUTED = "#5a5a5a"


# ---------------------------------------------------------------------------
# Output directory and the run context every experiment gets
# ---------------------------------------------------------------------------


def make_out_dir(base: str | Path, name: str, *, when: datetime | None = None) -> Path:
    """``<base>/<name>/<timestamp>/``, created fresh. UTC, sortable, filesystem-safe."""
    when = when or datetime.now(timezone.utc)
    ts = when.strftime("%Y%m%dT%H%M%SZ")
    out = Path(base) / name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class ExperimentContext:
    """Everything an experiment module's ``run(ctx)`` gets from the CLI."""

    name: str
    params: ParamFile
    out_dir: Path
    seed: int = 0
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    extra: Any = None  # argparse.Namespace from the experiment's own add_arguments, or None

    @property
    def command_line(self) -> str:
        return " ".join(self.argv)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class NumpyJSONEncoder(json.JSONEncoder):
    """``json.dump`` chokes on numpy scalars/arrays; this project produces them constantly."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_data_json(out_dir: Path, data: dict) -> Path:
    path = Path(out_dir) / "data.json"
    path.write_text(json.dumps(data, indent=2, cls=NumpyJSONEncoder, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# The parameter table and assumption list every report.md includes
# ---------------------------------------------------------------------------


def relevant_paths(prefixes: tuple[str, ...]) -> list[str]:
    """Every schema path starting with one of ``prefixes`` (e.g. ``("bricks.",)``), sorted."""
    from .. import params as params_mod

    return sorted(p for p in params_mod.SCHEMA if any(p.startswith(pre) for pre in prefixes))


def parameter_table_md(params: ParamFile, paths: list[str] | None = None) -> str:
    """A markdown table: parameter, value, unit, status, notes -- for the report's 'what was run'.

    Every value's status is right there in the table, which is the whole point: a reader can see
    at a glance which numbers in a report came from a measurement and which came from a guess.
    """
    paths = sorted(params.values) if paths is None else paths
    lines = [
        "| parameter | value | unit | status | notes |",
        "|---|---|---|---|---|",
    ]
    for path in paths:
        pv = params.get(path)
        value = pv.value if pv.value is not None else "*(not set)*"
        notes = (pv.notes or "").strip().replace("\n", " ")
        if len(notes) > 140:
            notes = notes[:137] + "..."
        badge = {"assumed": "assumed", "derived": "derived", "measured": "**measured**"}[pv.status]
        lines.append(f"| `{path}` | {value} | {pv.unit} | {badge} | {notes} |")
    return "\n".join(lines)


def assumptions_md(params: ParamFile, prefixes: tuple[str, ...] | None = None) -> str:
    """A short 'still assumptions' section, optionally restricted to a set of path prefixes."""
    items: list[Assumption] = list_assumptions(params)
    if prefixes is not None:
        items = [a for a in items if any(a.path.startswith(p) for p in prefixes)]
    if not items:
        return "Every parameter this experiment used is status `measured`."
    lines = [
        "| parameter | value | status | what it is |",
        "|---|---|---|---|",
    ]
    for a in items:
        lines.append(f"| `{a.path}` | {a.value} {a.unit} | {a.status} | {a.doc} |")
    return "\n".join(lines)


def plain_language_status(params: ParamFile, key_paths: tuple[str, ...]) -> str:
    """One sentence per key parameter, in plain words, for a report's 'what this is based on'."""
    out = []
    for path in key_paths:
        pv = params.get(path)
        out.append(f"- **{path}** = {pv.value} {pv.unit} ({pv.status}).")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


def report_header(ctx: ExperimentContext, title: str, what_ran: str) -> str:
    """The top of every report.md: title, when, command, what this experiment does."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {title}",
        "",
        f"Run: {now}  ",
        f"Command: `{ctx.command_line}`  ",
        f"Parameters: `{ctx.params.source}`" + (
            "" if ctx.params.source == ctx.params.base_source
            else f" (merged over `{ctx.params.base_source}`)"
        ),
        f"Host: {platform.node() or 'unknown'}, Python {platform.python_version()}",
        "",
        what_ran,
        "",
    ]
    return "\n".join(lines)


def write_report(out_dir: Path, sections: list[str]) -> Path:
    """Join ``sections`` with blank lines and write ``report.md``."""
    text = "\n\n".join(s.rstrip() for s in sections if s is not None) + "\n"
    path = Path(out_dir) / "report.md"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def new_figure(figsize: tuple[float, float] = (9.0, 6.0)):
    """A figure/axes pair with this project's readability defaults already applied."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(alpha=0.22, color=MUTED, linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def save_figure(fig, out_dir: Path, filename: str, *, rect: tuple[float, float, float, float] | None = None) -> Path:
    """Save ``fig`` at full resolution and close it. ``rect`` is passed to ``tight_layout``
    (left, bottom, right, top, in figure fractions) for a figure that reserves its own space for
    a manually-placed suptitle/caption outside the axes."""
    path = Path(out_dir) / filename
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    return path


def status_marker(status: str) -> str:
    """A short bracketed tag for embedding a status inline in a plot label or console line."""
    return {"assumed": "[assumed]", "derived": "[derived]", "measured": "[MEASURED]"}[status]
