"""``press-bench``: push a 2x2 and a 2x4 on and pull them off with the virtual force gauge.

    python -m brick_builder.experiments press-bench --params my_params.yaml

Builds ``brick_builder.bricks.bench.PressBench`` for a 2x2 and a 2x4, using the snap-fit parameters
and brick-on-brick friction from the parameter file, presses each one home at a commanded speed,
then pulls it back off, and reports the peak push-on force, the peak pull-off force, and the
whole force-against-depth trace for both.

If ``bricks.measured.push_on_*_N`` / ``bricks.measured.pull_off_*_N`` are filled in (not null),
this also fits ``clutch_force_per_stud_N`` and ``push_pull_asymmetry_beta`` to them the way
docs/guide/02c-materials-and-damage.md section 7.0's worksheet does, and prints the values to
paste back into your parameter file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..bricks.bench import BenchConfig, PressBench
from ..bricks.snap import ColletSnap
from .. import params as params_mod
from .common import (
    ExperimentContext,
    SERIES_COLOURS,
    parameter_table_md,
    report_header,
    save_figure,
    write_data_json,
    write_report,
)

BRICKS = ((2, 2), (2, 4))
RELEVANT_PREFIXES = ("bricks.",)


def _fmt(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--speed", type=float, default=0.002,
                         help="ram speed, m/s (default 0.002; the bench's own default)")
    parser.add_argument("--pull-travel", type=float, default=0.006,
                         help="how far to pull back up, m (default 0.006)")


def _run_one(nx: int, ny: int, params_file, speed: float, pull_travel: float) -> dict:
    spec = params_mod.to_brick_spec(params_file)
    snap_params = params_mod.to_snap_params(params_file)
    friction = float(params_file.value("bricks.brick_on_brick_friction"))

    cfg = BenchConfig(nx=nx, ny=ny, spec=spec, friction=friction)
    bench = PressBench(ColletSnap(snap_params), cfg)
    press_trace = bench.press(speed=speed)
    seated_after_press = bench.seated()  # must be read before pull() backs the brick out again
    pull_trace = bench.pull(speed=speed, travel=pull_travel)

    return {
        "nx": nx, "ny": ny, "n_studs": nx * ny,
        "peak_push_on_N": press_trace.peak_force,
        "peak_pull_off_N": pull_trace.peak_pull,
        "seated": seated_after_press,
        "warnings": press_trace.warnings + pull_trace.warnings,
        "press_depth_mm": (press_trace.depth * 1e3).tolist(),
        "press_force_N": press_trace.force.tolist(),
        "pull_depth_mm": (pull_trace.depth * 1e3).tolist(),
        "pull_force_N": pull_trace.force.tolist(),
    }


def fit_clutch_force_and_beta(measured: dict[str, float | None], bricks: tuple = BRICKS) -> dict:
    """Fit ``f_clutch`` (N/stud, the average of push-on and pull-off) and ``beta`` from
    whole-brick push-on/pull-off readings.

    Follows docs/guide/02c-materials-and-damage.md section 7.0's worksheet exactly:
    ``f_clutch = pull_off / n_studs / (1 - beta)``, and ``beta`` from the push/pull ratio
    ``r = push_on / pull_off`` via ``beta = (r - 1) / (r + 1)`` (the inverse of
    ``push = f_clutch * n * (1+beta)``, ``pull = f_clutch * n * (1-beta)``). Algebraically the
    same as the more direct ``f_clutch = (push_on + pull_off) / (2 * n_studs)`` and
    ``beta = (push_on - pull_off) / (push_on + pull_off)`` -- guide 10 spells out that form for
    the by-hand arithmetic; this function uses the pull-off-and-ratio route because it still
    produces a ``beta`` from a single brick's pair of readings before averaging across bricks.
    Uses whichever of the four readings are present; returns ``None`` for a value that cannot be
    computed from what is given.
    """
    per_brick = {}
    betas = []
    for nx, ny in bricks:
        n = nx * ny
        push = measured.get(f"push_on_{nx}x{ny}_N")
        pull = measured.get(f"pull_off_{nx}x{ny}_N")
        entry = {"n_studs": n, "push_on_N": push, "pull_off_N": pull,
                  "f_clutch_from_pull_N": None, "beta": None}
        if pull:
            entry["f_clutch_from_pull_N"] = pull / n
        if push and pull:
            r = push / pull
            entry["beta"] = (r - 1.0) / (r + 1.0)
            betas.append(entry["beta"])
        per_brick[f"{nx}x{ny}"] = entry

    beta = float(np.mean(betas)) if betas else None
    f_estimates = []
    for nx, ny in bricks:
        pull = measured.get(f"pull_off_{nx}x{ny}_N")
        if pull and beta is not None:
            f_estimates.append(pull / (nx * ny) / (1.0 - beta))
        elif pull:
            f_estimates.append(pull / (nx * ny))
    f_clutch = float(np.mean(f_estimates)) if f_estimates else None

    consistency = None
    p22, p24 = per_brick["2x2"]["pull_off_N"], per_brick["2x4"]["pull_off_N"]
    if p22 and p24:
        consistency = {
            "2x4_over_2x2_ratio": p24 / p22,
            "expected_ratio": (2 * 4) / (2 * 2),
            "note": "close to 2.0 means the force really is per-stud, as the model assumes",
        }
    return {
        "per_brick": per_brick,
        "f_clutch_N": f_clutch,
        "beta": beta,
        "consistency_check": consistency,
    }


def _figure(results: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, res, colour in zip(axes, results, SERIES_COLOURS):
        ax.plot(res["press_depth_mm"], res["press_force_N"], color=colour, lw=2.2, label="pressing on")
        ax.plot(res["pull_depth_mm"], res["pull_force_N"], color=colour, lw=2.2, ls="--",
                label="pulling off")
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.axvline(0.0, color="#888888", lw=0.8, ls=":")
        ax.scatter([res["press_depth_mm"][int(np.argmax(res["press_force_N"]))]],
                   [res["peak_push_on_N"]], color=colour, zorder=5, s=60, marker="^")
        ax.scatter([res["pull_depth_mm"][int(np.argmin(res["pull_force_N"]))]],
                   [-res["peak_pull_off_N"]], color=colour, zorder=5, s=60, marker="v")
        ax.text(0.02, 0.97,
                 f"peak push-on: {res['peak_push_on_N']:.2f} N\n"
                 f"peak pull-off: {res['peak_pull_off_N']:.2f} N",
                 transform=ax.transAxes, va="top", ha="left", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=colour, alpha=0.9))
        ax.set_xlabel("insertion depth [mm]")
        ax.set_ylabel("ram force [N] (positive = pressing down)")
        ax.set_title(f"{res['nx']}x{res['ny']} brick", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("Press-bench force vs. insertion depth", fontsize=14)
    return save_figure(fig, out_dir, "press_bench.png")


def run(ctx: ExperimentContext) -> None:
    params = ctx.params
    speed = getattr(ctx.extra, "speed", 0.002)
    pull_travel = getattr(ctx.extra, "pull_travel", 0.006)

    results = [_run_one(nx, ny, params, speed, pull_travel) for nx, ny in BRICKS]

    for res in results:
        print(f"{res['nx']}x{res['ny']}: peak push-on {res['peak_push_on_N']:.2f} N, "
              f"peak pull-off {res['peak_pull_off_N']:.2f} N, seated={res['seated']}")

    measured = {
        "push_on_2x2_N": params.value("bricks.measured.push_on_2x2_N"),
        "push_on_2x4_N": params.value("bricks.measured.push_on_2x4_N"),
        "pull_off_2x2_N": params.value("bricks.measured.pull_off_2x2_N"),
        "pull_off_2x4_N": params.value("bricks.measured.pull_off_2x4_N"),
    }
    have_measurements = any(v is not None for v in measured.values())
    fit = fit_clutch_force_and_beta(measured) if have_measurements else None
    if fit is not None:
        print("Fit from your measured push-on/pull-off values:")
        print(f"  clutch_force_per_stud_N -> {fit['f_clutch_N']}")
        print(f"  push_pull_asymmetry_beta -> {fit['beta']}")

    fig_path = _figure(results, ctx.out_dir)

    sim_rows = "| brick | studs | peak push-on (N) | peak pull-off (N) | seated |\n|---|---|---|---|---|\n"
    for res in results:
        sim_rows += (
            f"| {res['nx']}x{res['ny']} | {res['n_studs']} | {res['peak_push_on_N']:.2f} | "
            f"{res['peak_pull_off_N']:.2f} | {res['seated']} |\n"
        )

    fit_section = ""
    if fit is not None:
        fit_lines = ["| brick | your push-on (N) | your pull-off (N) | implied f_clutch (N) | implied beta |",
                     "|---|---|---|---|---|"]
        for name, e in fit["per_brick"].items():
            fit_lines.append(
                f"| {name} | {_fmt(e['push_on_N'])} | {_fmt(e['pull_off_N'])} | "
                f"{_fmt(e['f_clutch_from_pull_N'])} | {_fmt(e['beta'], 3)} |"
            )
        fit_lines.append("")
        if fit["f_clutch_N"] is not None:
            fit_lines.append(
                f"**Put these in your parameter file:** "
                f"`bricks.clutch_force_per_stud_N` = **{fit['f_clutch_N']:.3f} N** (status: measured)."
            )
        else:
            fit_lines.append(
                "Not enough measurements to fit clutch_force_per_stud_N yet -- fill in at least "
                "one pull-off reading."
            )
        if fit["beta"] is not None:
            fit_lines.append(
                f"`bricks.push_pull_asymmetry_beta` = **{fit['beta']:.3f}** (status: measured)."
            )
        if fit["consistency_check"] is not None:
            cc = fit["consistency_check"]
            close = abs(cc["2x4_over_2x2_ratio"] - cc["expected_ratio"]) < 0.6
            fit_lines.append(
                f"\nConsistency check: your 2x4 pull-off is {cc['2x4_over_2x2_ratio']:.2f}x your "
                f"2x2 pull-off (expected {cc['expected_ratio']:.1f}x if the force really is "
                "per-stud). " + ("Consistent with a per-stud force." if close else
                                  "Further off than expected: something besides the studs may be "
                                  "carrying load, or one of the two readings is off.")
            )
        fit_section = "## Fit from your measurements\n\n" + "\n".join(fit_lines)
    else:
        fit_section = (
            "## Fit from your measurements\n\n"
            "`bricks.measured.push_on_*_N` / `pull_off_*_N` are all null in this parameter file. "
            "Do the kitchen-scale test in docs/guide/10-measure-and-enter-parameters.md, fill "
            "those four fields in, and re-run this experiment to get a fitted "
            "`clutch_force_per_stud_N` and `push_pull_asymmetry_beta`."
        )

    reading = [
        f"At the current parameters, a 2x4 needs **{results[1]['peak_push_on_N']:.1f} N** to press "
        f"home and holds with **{results[1]['peak_pull_off_N']:.1f} N** once seated; a 2x2 needs "
        f"**{results[0]['peak_push_on_N']:.1f} N** and holds with **{results[0]['peak_pull_off_N']:.1f} N**.",
        "Compare the push-on numbers against what the arm can actually deliver with the "
        "`seating-feasibility` experiment.",
    ]

    sections = [
        report_header(
            ctx, "press-bench: push-on and pull-off force",
            "Presses a 2x2 and a 2x4 brick home on the virtual force gauge (brick_builder.bricks.bench) "
            "at the parameter file's snap-fit settings, then pulls each back off, and reports the "
            "peak forces and the whole force-against-depth trace."
        ),
        "## Parameters used\n\n" + parameter_table_md(
            params, sorted(p for p in params.values if any(p.startswith(pre) for pre in RELEVANT_PREFIXES))
        ),
        "## Results (simulated)\n\n" + sim_rows,
        f"![Force against depth, pressing on and pulling off]({fig_path.name})\n",
        fit_section,
        "## Reading\n\n" + "\n".join(f"- {r}" for r in reading),
    ]
    write_report(ctx.out_dir, sections)

    write_data_json(ctx.out_dir, {
        "speed_m_s": speed,
        "pull_travel_m": pull_travel,
        "results": results,
        "measured_inputs": measured,
        "fit": fit,
    })
