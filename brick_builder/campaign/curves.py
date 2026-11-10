"""The campaign's four figures, drawn from `summary.csv` and `campaign.json`.

    python -m brick_builder.campaign.curves --campaign results/campaign/closed_loop

Success per task against the round, for both inverses; the median prediction error of the two
prediction rules; the learned inverse's action error against the analytic one's own magnitude;
and how much the log grew. Nothing is computed here -- these read what the rounds measured.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

TASK_COLOUR = {"placement": "#1f6feb", "pick_and_place": "#d9822b", "build_2": "#2a9d55"}
PUBLISHED = {"placement": 1.0, "pick_and_place": 0.8}
"""The published closed-loop numbers with ground-truth driving: 10/10 and 8/10."""


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_summary(path: Path) -> list[dict]:
    """`summary.csv` as rows, with the numbers parsed."""
    rows = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            out = dict(row)
            out["round"] = int(row["round"])
            out["episodes"] = int(row["episodes"] or 0)
            out["ticks_trained_on"] = int(row["ticks_trained_on"] or 0)
            out["episodes_trained_on"] = int(row["episodes_trained_on"] or 0)
            for key in ("success_rate", "wilson_low", "wilson_high", "lateral_mm_median",
                        "median_prediction_mm", "median_tracking_mm", "ticks_align"):
                out[key] = _f(row.get(key))
            rows.append(out)
    return rows


def _f(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _primary(rows: Sequence[dict]) -> list[dict]:
    """One row per round, task and inverse: the column scored with the learned prediction rule
    when there is one, so a task is not counted twice."""
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["round"], row["task"], row["inverse"])
        if key not in best or row["predictor"] == "learned":
            best[key] = row
    return list(best.values())


def _runs(values: Sequence[int]) -> list[list[int]]:
    """Split ascending round numbers into consecutive runs, so a sparse task's rounds are never
    joined by a line across the rounds it was not evaluated in."""
    out: list[list[int]] = []
    for v in values:
        if out and v - out[-1][-1] == 1:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def success_figure(rows: Sequence[dict], out_path: Path) -> Path:
    """Seated on the target cell against the round, one panel per task, both inverses.

    The analytic inverse is a fixed baseline measured in round 0, so it is drawn as a level line
    across every round rather than as a series that happens to be flat. A sparse task (evaluated
    in some rounds only) draws no line across the rounds in between: a straight segment there
    would show a trend across rounds that were never measured.
    """
    plt = _plt()
    rows = _primary(rows)
    tasks = [t for t in ("placement", "pick_and_place", "build_2")
             if any(r["task"] == t for r in rows)]
    rounds = sorted({r["round"] for r in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 4.3), dpi=150,
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        learned = sorted([r for r in rows if r["task"] == task and r["inverse"] == "learned"],
                         key=lambda r: r["round"])
        analytic = [r for r in rows if r["task"] == task and r["inverse"] == "analytic"]
        colour = TASK_COLOUR.get(task, "#1f6feb")
        if learned:
            by_round = {r["round"]: r for r in learned}
            for i, run in enumerate(_runs(sorted(by_round))):
                run_rows = [by_round[v] for v in run]
                x = [r["round"] for r in run_rows]
                y = [100.0 * r["success_rate"] for r in run_rows]
                lo = [100.0 * (r["success_rate"] - r["wilson_low"]) for r in run_rows]
                hi = [100.0 * (r["wilson_high"] - r["success_rate"]) for r in run_rows]
                ax.errorbar(x, y, yerr=[lo, hi], color=colour, marker="o", markersize=4,
                            linestyle="-" if len(run) > 1 else "none", linewidth=1.4,
                            elinewidth=1.2, capsize=3,
                            label="learned inverse" if i == 0 else None)
        if analytic:
            base = 100.0 * analytic[0]["success_rate"]
            ax.axhline(base, color="#444444", linewidth=1.2, linestyle="--",
                       label=f"analytic inverse ({analytic[0]['seated_on_target']})")
            ax.axhspan(100.0 * analytic[0]["wilson_low"], 100.0 * analytic[0]["wilson_high"],
                       color="#444444", alpha=0.08)
        if task in PUBLISHED:
            ax.axhline(100.0 * PUBLISHED[task], color="#d9822b", linewidth=1.0,
                       linestyle=":", label="published")
        ax.set_title({"placement": "Placement", "pick_and_place": "Pick-and-Place",
                      "build_2": "Side-by-Side Assembly"}[task], fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylim(-5, 105)
        ax.set_xticks(rounds)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                 borderaxespad=0.0, frameon=False)
    axes[0][0].set_ylabel("Seated on target cell [%]")
    fig.suptitle("Assembly Success", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path)


def prediction_figure(rows: Sequence[dict], out_path: Path) -> Path:
    """Median prediction error against the round, per prediction rule.

    A sparse task's rounds are split into consecutive runs, the same as in `success_figure`, so
    no line is drawn across rounds it was not evaluated in.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    styles = {"learned": "-", "analytic": "--"}
    markers = {"learned": "o", "analytic": "s"}
    drawn = False
    for task in ("placement", "pick_and_place", "build_2"):
        for predictor in ("learned", "analytic"):
            series = sorted([r for r in rows if r["task"] == task
                             and r["predictor"] == predictor
                             and r["inverse"] == "learned"], key=lambda r: r["round"])
            series = [r for r in series if r["median_prediction_mm"] == r["median_prediction_mm"]]
            if not series:
                continue
            by_round = {r["round"]: r for r in series}
            for i, run in enumerate(_runs(sorted(by_round))):
                run_rows = [by_round[v] for v in run]
                ax.plot([r["round"] for r in run_rows],
                        [r["median_prediction_mm"] for r in run_rows],
                        marker=markers[predictor], markersize=4, linewidth=1.3,
                        linestyle=styles[predictor] if len(run) > 1 else "none",
                        color=TASK_COLOUR.get(task, "#1f6feb"),
                        label=f"{task.replace('_', ' ')}, F {predictor}" if i == 0 else None)
            drawn = True
    ax.set_xlabel("Round")
    ax.set_ylabel("Median prediction error [mm]")
    ax.set_title("Prediction Error", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    if drawn:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                 borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    return _save(fig, out_path)


def action_figure(state: dict, out_path: Path) -> Path:
    """The learned inverse's error against the analytic answer, and that answer's own size."""
    plt = _plt()
    rounds = sorted(int(k) for k in state.get("rounds", {}))
    x, rmse_x, rmse_y, rmse_s, base_x = [], [], [], [], []
    for index in rounds:
        metrics = state["rounds"][str(index)].get("metrics") or {}
        if "action_rmse_mm_x" not in metrics:
            continue
        x.append(index)
        rmse_x.append(float(metrics["action_rmse_mm_x"]))
        rmse_y.append(float(metrics["action_rmse_mm_y"]))
        rmse_s.append(float(metrics["action_rmse_deg"]))
        base_x.append(float(metrics.get("analytic_rms_mm_x", float("nan"))))

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    if x:
        ax.plot(x, rmse_x, "-o", markersize=4, linewidth=1.3, color="#1f6feb",
                label="x [mm/tick]")
        ax.plot(x, rmse_y, "-s", markersize=4, linewidth=1.3, color="#d9822b",
                label="y [mm/tick]")
        ax.plot(x, rmse_s, "-^", markersize=4, linewidth=1.3, color="#2a9d55",
                label="spin [deg/tick]")
        ax.plot(x, base_x, ":", linewidth=1.1, color="#888888",
                label="analytic RMS, x [mm/tick]")
        ax.legend(fontsize=7)
        ax.set_xticks(x)
    ax.set_xlabel("Round")
    ax.set_ylabel("RMS error [mm/tick, deg/tick]")
    ax.set_title("Inverse Model Error", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return _save(fig, out_path)


def data_figure(state: dict, out_path: Path) -> Path:
    """Episodes and tuples the round trained on."""
    plt = _plt()
    rounds = sorted(int(k) for k in state.get("rounds", {}))
    x = [i for i in rounds if state["rounds"][str(i)].get("ticks")]
    ticks = [int(state["rounds"][str(i)]["ticks"]) for i in x]
    episodes = [int(state["rounds"][str(i)].get("episodes_logged", 0)) for i in x]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(x, ticks, "-o", markersize=4, linewidth=1.4, color="#1f6feb", label="tuples [ticks]")
    ax.set_xlabel("Round")
    ax.set_ylabel("Tuples trained on [ticks]")
    ax.grid(alpha=0.25, linewidth=0.5)
    twin = ax.twinx()
    twin.plot(x, episodes, "--s", markersize=4, linewidth=1.2, color="#d9822b",
              label="episodes")
    twin.set_ylabel("Episodes trained on")
    if x:
        ax.set_xticks(x)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="lower right")
    ax.set_title("Training Data", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.clf()
    _plt().close(fig)
    return out_path


def draw_all(campaign_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """Every figure the guide page carries, from one finished or running campaign."""
    campaign_dir = Path(campaign_dir)
    out_dir = Path(out_dir) if out_dir else campaign_dir / "curves"
    rows = read_summary(campaign_dir / "summary.csv")
    state = json.loads((campaign_dir / "campaign.json").read_text())
    return [
        success_figure(rows, out_dir / "campaign_success.png"),
        prediction_figure(rows, out_dir / "campaign_prediction_error.png"),
        action_figure(state, out_dir / "campaign_action_error.png"),
        data_figure(state, out_dir / "campaign_data.png"),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="curves", description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", default="results/campaign/closed_loop")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    for path in draw_all(Path(args.campaign), Path(args.out) if args.out else None):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
