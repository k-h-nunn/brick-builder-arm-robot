"""Evaluation report generation: JSON dump, markdown summary, sweep heatmaps, acceptance check.

Consumes a `list[EpisodeResult]` (each episode's sweep/randomization values live in `.params`,
the evaluation guide) and produces the artefacts the evaluation suite owes:

- a JSON file with every episode plus aggregate stats,
- a markdown report with success tables (Wilson 95 % intervals) and per-failure-class counts,
- sweep heatmaps (e.g. misalignment in mm vs yaw error in degrees) as PNG,
- an "acceptance" check against the section 6.4 thresholds, printed as a pass/fail table.

Matplotlib is used with the non-interactive `Agg` backend so this module is safe to import in
tests and headless scripts; no figure window is ever opened.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brick_builder.eval.results import EpisodeResult
from brick_builder.eval.taxonomy import failure_counts, label_all
from brick_builder.eval.wilson import WilsonInterval, intervals_disjoint, wilson_interval


# ---------------------------------------------------------------------------
# Grouping and success tables
# ---------------------------------------------------------------------------


def group_by_params(results: list[EpisodeResult], keys: tuple[str, ...]) -> dict[tuple, list[EpisodeResult]]:
    """Group episodes by the values of `keys` in `.params`.

    Missing keys are grouped under `None`. Returns a dict keyed by a tuple of values in the same
    order as `keys`, e.g. `group_by_params(results, ("misalign_mm", "yaw_deg"))`.
    """
    groups: dict[tuple, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        key = tuple(r.params.get(k) for k in keys)
        groups[key].append(r)
    return dict(groups)


def success_rate(results: list[EpisodeResult]) -> WilsonInterval:
    """Wilson interval over a batch of episodes (successes = `.success`)."""
    n = len(results)
    successes = sum(1 for r in results if r.success)
    return wilson_interval(successes, n)


@dataclass(frozen=True)
class SuccessRow:
    """One row of a success table: a group's key, its Wilson interval, and n."""

    key: tuple
    interval: WilsonInterval


def success_table(results: list[EpisodeResult], keys: tuple[str, ...]) -> list[SuccessRow]:
    """Build a success table, one row per distinct value combination of `keys`, sorted by key."""
    groups = group_by_params(results, keys)
    rows = [SuccessRow(key=k, interval=success_rate(v)) for k, v in groups.items()]
    rows.sort(key=lambda row: tuple(str(x) for x in row.key))
    return rows


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def _episode_to_dict(r: EpisodeResult) -> dict[str, Any]:
    d = asdict(r)
    if d["failure"] is not None:
        d["failure"] = d["failure"].value if hasattr(d["failure"], "value") else str(d["failure"])
    return d


def write_json_report(results: list[EpisodeResult], path: str | Path) -> dict[str, Any]:
    """Write a JSON report (every episode plus overall aggregate) and return the same dict."""
    overall = success_rate(results)
    payload = {
        "n_episodes": len(results),
        "overall": {
            "success_rate": overall.p_hat,
            "wilson_low": overall.low,
            "wilson_high": overall.high,
        },
        "failure_counts": dict(failure_counts(results)),
        "episodes": [_episode_to_dict(r) for r in results],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_key(keys: tuple[str, ...], values: tuple) -> str:
    return ", ".join(f"{k}={v}" for k, v in zip(keys, values))


def render_markdown_report(
    results: list[EpisodeResult],
    sweep_keys: tuple[str, ...] = (),
    title: str = "Evaluation report",
) -> str:
    """Build the markdown text of a report (no file I/O). `write_markdown_report` wraps this."""
    lines = [f"# {title}", ""]
    overall = success_rate(results)
    lines.append(f"Episodes: {overall.n}")
    lines.append(
        f"Overall success rate: {overall.p_hat:.3f} "
        f"(95% Wilson CI [{overall.low:.3f}, {overall.high:.3f}])"
    )
    lines.append("")

    lines.append("## Failure taxonomy")
    lines.append("")
    counts = failure_counts(results)
    if counts:
        lines.append("| Failure class | Count |")
        lines.append("|---|---|")
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {n} |")
    else:
        lines.append("No failures recorded.")
    lines.append("")

    if sweep_keys:
        lines.append(f"## Success by {', '.join(sweep_keys)}")
        lines.append("")
        lines.append(f"| {' | '.join(sweep_keys)} | n | success rate | 95% Wilson CI |")
        lines.append(f"|{'---|' * len(sweep_keys)}---|---|---|")
        for row in success_table(results, sweep_keys):
            key_cells = " | ".join(str(v) for v in row.key)
            iv = row.interval
            lines.append(
                f"| {key_cells} | {iv.n} | {iv.p_hat:.3f} | [{iv.low:.3f}, {iv.high:.3f}] |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown_report(
    results: list[EpisodeResult],
    path: str | Path,
    sweep_keys: tuple[str, ...] = (),
    title: str = "Evaluation report",
) -> str:
    """Render and write the markdown report; returns the text."""
    text = render_markdown_report(results, sweep_keys=sweep_keys, title=title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# Sweep heatmaps
# ---------------------------------------------------------------------------


def plot_success_heatmap(
    results: list[EpisodeResult],
    x_key: str,
    y_key: str,
    path: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "Success rate",
) -> Path:
    """Save a 2D success-rate heatmap over `params[x_key]` vs `params[y_key]` as a PNG.

    Cells with no episodes are left blank (NaN, drawn as a distinct color by the colormap's `bad`
    setting) rather than shown as 0 % success, so an unswept combination is visually distinct from
    one that was swept and always failed.
    """
    groups = group_by_params(results, (x_key, y_key))
    xs = sorted({k[0] for k in groups if k[0] is not None})
    ys = sorted({k[1] for k in groups if k[1] is not None})
    if not xs or not ys:
        raise ValueError(f"No episodes have both '{x_key}' and '{y_key}' set in params")

    grid = np.full((len(ys), len(xs)), np.nan)
    for (xv, yv), eps in groups.items():
        if xv is None or yv is None:
            continue
        iv = success_rate(eps)
        grid[ys.index(yv), xs.index(xv)] = iv.p_hat

    fig, ax = plt.subplots(figsize=(1.2 * len(xs) + 2, 1.0 * len(ys) + 2))
    cmap = matplotlib.colormaps["RdYlGn"].with_extremes(bad="#dddddd")
    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap=cmap, origin="lower", aspect="auto")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(y) for y in ys])
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(y_label or y_key)
    ax.set_title(title)
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceThresholds:
    """The Definition-of-Done numbers.

    All fields are the "must be at least this good" targets; `None` skips that check when a
    results set has no matching data (e.g. no seat-depth episodes yet).
    """

    min_wilson_low: float = 0.85          # student Wilson lower bound at <=4mm/8deg misalignment
    min_margin_over_scripted: float = 0.15  # student beats scripted by >=15 points
    max_seat_depth_error: float | None = 0.0003   # metres (0.3 mm)
    max_peak_load: float = 0.9
    min_peak_load_fraction: float = 0.99  # fraction of episodes that must respect max_peak_load
    min_structure_intact_fraction: float = 0.98
    max_median_steps: float | None = None  # set from a time budget when tick rate is known


@dataclass(frozen=True)
class AcceptanceResult:
    name: str
    passed: bool
    measured: str
    threshold: str


def check_acceptance(
    results: list[EpisodeResult],
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    scripted_results: list[EpisodeResult] | None = None,
) -> list[AcceptanceResult]:
    """Check `results` against `thresholds` and return one row per criterion.

    `scripted_results`, if given, enables the "beats scripted by >=X points with disjoint CIs"
    check; without it that row is skipped (returned as not applicable, marked failed with a note,
    so a caller cannot silently drop the comparison).
    """
    rows: list[AcceptanceResult] = []

    overall = success_rate(results)
    rows.append(
        AcceptanceResult(
            name="Wilson lower bound",
            passed=overall.low >= thresholds.min_wilson_low,
            measured=f"{overall.low:.3f} (p_hat={overall.p_hat:.3f}, n={overall.n})",
            threshold=f">= {thresholds.min_wilson_low:.3f}",
        )
    )

    if scripted_results is not None:
        scripted = success_rate(scripted_results)
        margin = overall.p_hat - scripted.p_hat
        disjoint = intervals_disjoint(overall, scripted)
        rows.append(
            AcceptanceResult(
                name="Beats scripted baseline",
                passed=(margin >= thresholds.min_margin_over_scripted) and disjoint,
                measured=f"margin={margin:+.3f}, disjoint_CIs={disjoint}",
                threshold=f">= {thresholds.min_margin_over_scripted:.3f} pts, disjoint CIs",
            )
        )
    else:
        rows.append(
            AcceptanceResult(
                name="Beats scripted baseline",
                passed=False,
                measured="no scripted_results given",
                threshold=f">= {thresholds.min_margin_over_scripted:.3f} pts, disjoint CIs",
            )
        )

    if thresholds.max_seat_depth_error is not None:
        depths = [r.seat_depth_error for r in results if r.seat_depth_error is not None]
        if depths:
            med = float(np.median(depths))
            rows.append(
                AcceptanceResult(
                    name="Seat depth error (median)",
                    passed=med <= thresholds.max_seat_depth_error,
                    measured=f"{med * 1000:.3f} mm",
                    threshold=f"<= {thresholds.max_seat_depth_error * 1000:.3f} mm",
                )
            )
        else:
            rows.append(
                AcceptanceResult(
                    name="Seat depth error (median)",
                    passed=False,
                    measured="no seat_depth_error data",
                    threshold=f"<= {thresholds.max_seat_depth_error * 1000:.3f} mm",
                )
            )

    n = len(results)

    # rl-eval-export-2: peak_load/structure_intact default to None ("not reported"), not a
    # safe-looking 0.0/True, precisely so a task or environment that stops populating them
    # cannot make this check pass vacuously. If *any* episode in the batch is missing the
    # field, that is exactly the silent-regression case this exists to catch, so the whole
    # criterion fails loud with a row that says so, rather than quietly scoring only the
    # episodes that happened to report it.
    missing_load = sum(1 for r in results if r.peak_load is None)
    if missing_load:
        rows.append(
            AcceptanceResult(
                name="Peak load within budget",
                passed=False,
                measured=f"peak_load not reported for {missing_load}/{n} episodes",
                threshold=f">= {thresholds.min_peak_load_fraction:.3f} fraction",
            )
        )
    else:
        n_within_load = sum(1 for r in results if r.peak_load <= thresholds.max_peak_load)
        frac_within_load = n_within_load / n if n else 0.0
        rows.append(
            AcceptanceResult(
                name="Peak load within budget",
                passed=frac_within_load >= thresholds.min_peak_load_fraction,
                measured=f"{frac_within_load:.3f} of episodes <= {thresholds.max_peak_load}",
                threshold=f">= {thresholds.min_peak_load_fraction:.3f} fraction",
            )
        )

    missing_structure = sum(1 for r in results if r.structure_intact is None)
    if missing_structure:
        rows.append(
            AcceptanceResult(
                name="Structure intact",
                passed=False,
                measured=f"structure_intact not reported for {missing_structure}/{n} episodes",
                threshold=f">= {thresholds.min_structure_intact_fraction:.3f}",
            )
        )
    else:
        n_intact = sum(1 for r in results if r.structure_intact)
        frac_intact = n_intact / n if n else 0.0
        rows.append(
            AcceptanceResult(
                name="Structure intact",
                passed=frac_intact >= thresholds.min_structure_intact_fraction,
                measured=f"{frac_intact:.3f}",
                threshold=f">= {thresholds.min_structure_intact_fraction:.3f}",
            )
        )

    if thresholds.max_median_steps is not None:
        med_steps = float(np.median([r.steps for r in results])) if results else float("nan")
        rows.append(
            AcceptanceResult(
                name="Median episode length",
                passed=med_steps <= thresholds.max_median_steps,
                measured=f"{med_steps:.1f} steps",
                threshold=f"<= {thresholds.max_median_steps:.1f} steps",
            )
        )

    return rows


def format_acceptance_table(rows: list[AcceptanceResult]) -> str:
    """Render `check_acceptance`'s rows as a plain-text pass/fail table."""
    lines = []
    header = f"{'criterion':<30} {'result':<8} {'measured':<40} {'threshold'}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        mark = "PASS" if row.passed else "FAIL"
        lines.append(f"{row.name:<30} {mark:<8} {row.measured:<40} {row.threshold}")
    lines.append("-" * len(header))
    n_pass = sum(1 for r in rows if r.passed)
    lines.append(f"{n_pass}/{len(rows)} criteria passed")
    return "\n".join(lines)
