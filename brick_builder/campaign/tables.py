"""The campaign's tables, in the documentation's own style.

    python -m brick_builder.campaign.tables --campaign results/campaign/closed_loop

One table per task: a row per round and inverse, with the analytic baseline and the published
numbers repeated in every one of them so a column is never read without what it is against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .curves import read_summary

PUBLISHED_TEXT = {"placement": "10/10", "pick_and_place": "8/10"}
"""What the closed-loop page reports with ground-truth driving. `build_2` has no published
number: this campaign is the first measurement of it."""

SHIP_NOTE = """`Would ship` applies the same rule `action.learn.rounds.deploy_if_better` uses --
`yes` when the round's learned inverse matched the fixed analytic baseline on every task it was
measured on, `meets` when it matched on this task but not on all of them, `no` when it did not
match here -- to whichever tasks the round measured. `deploy_if_better` itself is not called by
the campaign, and its own default task pair does not cover `build_2`, which this column does. The
campaign deploys every round to its own weights directory whatever this says."""


def _fmt(value: float, digits: int = 2) -> str:
    return "--" if value != value else f"{value:.{digits}f}"


def task_table(rows: Sequence[dict], task: str) -> str:
    """One markdown table for one task."""
    keep = [r for r in rows if r["task"] == task]
    if not keep:
        return ""
    analytic = [r for r in keep if r["inverse"] == "analytic"]
    learned = sorted([r for r in keep if r["inverse"] == "learned"], key=lambda r: r["round"])
    learned = [r for r in learned if r["predictor"] == "learned"] or learned

    head = ("| Round | Inverse | Version | Seated on target cell | Wilson 95 % | Bricks | "
            "Lateral error [mm] | Median tracking [mm] | Median prediction [mm] | "
            "Ticks at Align | Tuples trained on | Would ship |")
    rule = "|" + "---|" * 12
    lines = [head, rule]
    if analytic:
        a = analytic[0]
        lines.append(
            f"| fixed | analytic | -- | {a['seated_on_target']} | "
            f"{100 * a['wilson_low']:.0f}-{100 * a['wilson_high']:.0f} % | "
            f"{a['bricks_on_target']} | "
            f"{_fmt(a['lateral_mm_median'])} | {_fmt(a['median_tracking_mm'])} | "
            f"{_fmt(a['median_prediction_mm'])} | {_fmt(a['ticks_align'], 0)} | -- | -- |")
    if task in PUBLISHED_TEXT:
        lines.append(f"| published | ground truth | -- | {PUBLISHED_TEXT[task]} | -- | -- | "
                     "-- | -- | -- | -- | -- | -- |")
    recheck = next((r for r in analytic[1:] if r["predictor"] == "analytic"
                    and r["round"] != analytic[0]["round"]), None) if analytic else None
    if recheck:
        lines.append(
            f"| recheck (round {recheck['round']}) | analytic | -- | "
            f"{recheck['seated_on_target']} | "
            f"{100 * recheck['wilson_low']:.0f}-{100 * recheck['wilson_high']:.0f} % | "
            f"{recheck['bricks_on_target']} | "
            f"{_fmt(recheck['lateral_mm_median'])} | {_fmt(recheck['median_tracking_mm'])} | "
            f"{_fmt(recheck['median_prediction_mm'])} | {_fmt(recheck['ticks_align'], 0)} | "
            "-- | -- |")
    for r in learned:
        lines.append(
            f"| {r['round']} | learned | {r['version']} | {r['seated_on_target']} | "
            f"{100 * r['wilson_low']:.0f}-{100 * r['wilson_high']:.0f} % | "
            f"{r['bricks_on_target']} | "
            f"{_fmt(r['lateral_mm_median'])} | {_fmt(r['median_tracking_mm'])} | "
            f"{_fmt(r['median_prediction_mm'])} | {_fmt(r['ticks_align'], 0)} | "
            f"{r['ticks_trained_on'] or '--'} | {r.get('would_ship') or '--'} |")
    return "\n".join(lines)


def cost_table(state: dict) -> str:
    """Wall clock per phase per round, in minutes."""
    lines = ["| Round | Perform [min] | Train [min] | Evaluate [min] | Episodes logged | "
             "Tuples |", "|---|---|---|---|---|---|"]
    for index in sorted(int(k) for k in state.get("rounds", {})):
        r = state["rounds"][str(index)]
        wall = r.get("wall_s", {}) or {}
        lines.append(
            f"| {index} | {wall.get('perform', 0) / 60.0:.1f} | "
            f"{wall.get('train', 0) / 60.0:.1f} | {wall.get('evaluate', 0) / 60.0:.1f} | "
            f"{r.get('episodes_logged', 0)} | {r.get('ticks', 0)} |")
    return "\n".join(lines)


def failure_table(rows: Sequence[dict]) -> str:
    """What the failures were, per task and round, for the learned inverse.

    Round 0 carries a `placement` row for both prediction rules (the two share a driving run);
    keeping only the `predictor == "learned"` row, the same restriction every other round already
    gets, is what keeps a round from printing twice.
    """
    lines = ["| Round | Task | Seated on target cell | Failures |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["round"], r["task"])):
        if r["inverse"] != "learned" or r["predictor"] != "learned":
            continue
        lines.append(f"| {r['round']} | {r['task'].replace('_', ' ')} | "
                     f"{r['seated_on_target']} | {r['failures'] or 'none'} |")
    return "\n".join(lines)


def render(campaign_dir: Path) -> str:
    campaign_dir = Path(campaign_dir)
    rows = read_summary(campaign_dir / "summary.csv")
    state = json.loads((campaign_dir / "campaign.json").read_text())
    blocks = []
    for task in ("placement", "pick_and_place", "build_2", "stack_2"):
        table = task_table(rows, task)
        if table:
            blocks.append(f"### {task.replace('_', ' ').capitalize()}, seeds 0-9\n\n{table}")
    blocks.append(SHIP_NOTE)
    blocks.append("### Cost per round\n\n" + cost_table(state))
    blocks.append("### Failures\n\n" + failure_table(rows))
    return "\n\n".join(blocks) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tables", description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", default="results/campaign/closed_loop")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    text = render(Path(args.campaign))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
