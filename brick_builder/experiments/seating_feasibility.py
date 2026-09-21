"""``seating-feasibility``: can the arm's downward force actually seat a brick?

    python -m brick_builder.experiments seating-feasibility --params my_params.yaml

The single most consequential question in the project (docs/guide/02c-materials-and-damage.md,
"Can the stock arm seat a brick?"): does the arm have enough downward force, at the member's
supply voltage and torque limit, to push a brick's studs all the way home at the member's
clutch-force-per-stud -- for a straight press of a whole row of studs at once, and for a
tilted press of one stud row at a time?

**A note on where the "push-on force needed" number comes from.** ``brick_builder.bricks.materials``
also has a ``push_on_force()``, which this experiment does not call. Both take a clutch force
per stud (the average of push-on and pull-off, what ``SnapParams.f_clutch`` and this file's
``bricks.clutch_force_per_stud_N`` mean) and both now compute ``n_studs * f_clutch * (1 + beta)``
(times the shape's peak gain), matching ``SnapParams``/``snap_api.engagement()`` -- the mechanism
the simulator actually runs -- and checked directly against ``brick_builder.bricks.bench.PressBench``
for a straight press (``tests/bricks/test_materials.py`` pins the agreement for 2x2 and 2x4).
This module still computes it independently, through :func:`push_on_needed`, rather than
importing ``brick_builder.bricks.materials``: that module pulls in the whole sourced-material
database for one formula, and keeping this experiment's own copy means a change there cannot
silently change what this report says the arm needs. Everything about the *arm's available
force* side (this module's other half) comes straight from ``brick_builder.robot.force_map``,
unrelated to either.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .. import params as params_mod
from ..bricks.materials import clutch_force, detent_peak_gain
from ..bricks.snap_api import SnapParams
from ..bricks.spec import DUPLO, BrickSpec
from ..robot.force_map import press_force_map
from ..robot.kinematics import ArmKinematics
from ..robot.patch_so101 import ArmConfig, attach_arm
from .common import (
    ExperimentContext,
    parameter_table_md,
    report_header,
    save_figure,
    write_data_json,
    write_report,
)

BUILD_PLATE_X_RANGE = (0.12, 0.24)
BUILD_PLATE_Y_RANGE = (-0.06, 0.06)
BUILD_PLATE_Z = 0.02
RELEVANT_PREFIXES = ("bricks.", "arm.")

CONFIGS = (
    ("2x4 straight", 2, 4, False),
    ("2x4 tilted (1 row)", 2, 4, True),
    ("2x2 straight", 2, 2, False),
    ("2x2 tilted (1 row)", 2, 2, True),
)

#: The first-principles upper scenario (docs/guide/02c-materials-and-damage.md section 2.3): an
#: estimate from press-fit geometry and ABS material properties, kept as a
#: scenario to show side by side with whatever the parameter file actually ships -- NOT as the
#: default, because it rests on an assumed interference that nobody has confirmed on a real
#: brick. Read from brick_builder.bricks.materials so this stays in sync with that derivation.
FIRST_PRINCIPLES_CLUTCH_N = clutch_force(DUPLO).per_stud


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grid-nx", type=int, default=7, help="workspace grid points along x (default 7)")
    parser.add_argument("--grid-ny", type=int, default=5, help="workspace grid points along y (default 5)")


def push_on_needed(
    snap_params: SnapParams, n_studs: int, tilted: bool, spec: BrickSpec, row_studs: int = 2
) -> float:
    """Force [N] the tool must reach to seat the brick, computed the way the simulator would.

    ``n_studs * f_clutch * (1 + beta)`` is exactly what ``snap_api.engagement()`` /
    ``ColletSnap`` produce at full seat (``engagement(stud_h) == 1`` for every curve shape), and
    it is what ``brick_builder.bricks.bench.predicted_press_force`` uses too. A tilted press only
    fights ``row_studs`` studs at a time. The ``detent`` shape peaks above its seated value
    (``detent_peak_gain``); ``ramp`` and ``plateau`` both peak at the seated value.
    """
    gain = detent_peak_gain(spec) if snap_params.curve == "detent" else 1.0
    n = min(row_studs, n_studs) if tilted else n_studs
    return n * snap_params.f_clutch * (1.0 + snap_params.beta) * gain


def arm_force_band(arm_cfg: ArmConfig, grid: tuple[int, int] = (7, 5)) -> dict:
    """The arm's downward force at ``arm_cfg.vin``/``torque_limit_nm``, over the build-plate patch."""
    world = mujoco.MjSpec()
    attach_arm(world, arm_cfg)
    model = world.compile()
    kin = ArmKinematics(model)
    xs = np.linspace(*BUILD_PLATE_X_RANGE, grid[0])
    ys = np.linspace(*BUILD_PLATE_Y_RANGE, grid[1])
    result = press_force_map(
        model, kin, xs, ys, z=BUILD_PLATE_Z, vin_list=(arm_cfg.vin,),
        torque_limit_nm=arm_cfg.torque_limit_nm,
    )
    reach = result["reachable"]
    forces = result["force"][arm_cfg.vin][reach]
    lo = float(np.nanmin(forces)) if forces.size else 0.0
    hi = float(np.nanmax(forces)) if forces.size else 0.0
    return {"lo_N": lo, "hi_N": hi, "n_reachable": int(reach.sum()), "n_total": int(reach.size),
            "xs": xs, "ys": ys, "result": result}


def verdict(needed: float, lo: float, hi: float) -> str:
    if needed <= lo:
        return "possible everywhere on the patch"
    if needed <= hi:
        return "possible only in the best part of the patch"
    return "not possible anywhere on the patch"


def _figure(rows: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    colours = {"possible everywhere on the patch": "#009E73",
               "possible only in the best part of the patch": "#E69F00",
               "not possible anywhere on the patch": "#D55E00"}
    max_x = max(r["needed_N"] for r in rows) * 1.0
    max_x = max(max_x, max(r["hi_N"] for r in rows)) * 1.15
    for i, r in enumerate(rows):
        ax.barh(i, r["hi_N"], color="#0072B2", alpha=0.18, height=0.55, zorder=1)
        ax.barh(i, r["lo_N"], color="#0072B2", alpha=0.35, height=0.55, zorder=1)
        colour = colours[r["verdict"]]
        ax.plot([r["needed_N"]], [i], marker="D", ms=15, color=colour, zorder=5,
                markeredgecolor="white", markeredgewidth=1.3)
        ax.text(max_x, i, f"  {r['verdict']}", va="center", ha="left", fontsize=10.5, color=colour,
                fontweight="bold")
        # A rule between one configuration's pair of scenario rows and the next configuration's,
        # so "your parameters" and "first-principles" read as one visually grouped pair.
        if i % 2 == 1 and i + 1 < len(rows):
            ax.axhline(i + 0.5, color="#c9c9c9", lw=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlim(0.0, max_x * 1.75)
    ax.set_xlabel("force [N]", fontsize=12)
    fig.suptitle("Push-on force needed vs. what the arm can deliver over the build plate",
                 fontsize=15, y=0.99)
    fig.text(
        0.5, 0.955,
        "blue bar = arm's downward force (dark = worst cell, light = best cell)    "
        "diamond = force needed to seat\neach pair: your parameters, then the first-principles upper scenario",
        ha="center", va="top", fontsize=10.5, color="#3a3a3a",
    )
    ax.invert_yaxis()
    return save_figure(fig, out_dir, "seating_feasibility.png", rect=(0.0, 0.0, 1.0, 0.87))


#: The two scenarios shown side by side in every report, so the first-principles risk stays
#: visible even when a member's own parameter file already has a nicer number in it. The second
#: scenario always uses the *current* file's beta and curve -- only the clutch itself is swapped
#: -- because beta and the curve shape are separate unknowns this scenario says nothing about.
SCENARIOS = (
    ("your parameters", None),
    ("first-principles upper scenario", FIRST_PRINCIPLES_CLUTCH_N),
)


def run(ctx: ExperimentContext) -> None:
    params = ctx.params
    grid = (getattr(ctx.extra, "grid_nx", 7), getattr(ctx.extra, "grid_ny", 5))

    spec = params_mod.to_brick_spec(params)
    snap_params = params_mod.to_snap_params(params)
    arm_cfg = params_mod.to_arm_config(params)

    band = arm_force_band(arm_cfg, grid=grid)
    print(f"Arm at {arm_cfg.vin:g} V, {arm_cfg.torque_limit_nm:g} N.m: "
          f"{band['lo_N']:.1f} to {band['hi_N']:.1f} N over the build plate "
          f"({band['n_reachable']}/{band['n_total']} cells reachable)")

    rows = []
    for name, nx, ny, tilted in CONFIGS:
        # The clutch (clutch force per stud) at which this configuration stops being possible at
        # all, i.e. where "needed" would equal the arm's best cell -- computed once per config,
        # from a unit clutch, so it does not depend on which scenario is being reported.
        per_unit = push_on_needed(
            SnapParams(f_clutch=1.0, beta=snap_params.beta, curve=snap_params.curve),
            nx * ny, tilted, spec,
        )
        ceiling = band["hi_N"] / per_unit if per_unit > 0 else float("inf")
        for scenario, override_clutch in SCENARIOS:
            scenario_params = (
                snap_params if override_clutch is None
                else SnapParams(f_clutch=override_clutch, beta=snap_params.beta, curve=snap_params.curve)
            )
            needed = push_on_needed(scenario_params, nx * ny, tilted, spec)
            v = verdict(needed, band["lo_N"], band["hi_N"])
            rows.append({
                "config": name, "scenario": scenario, "clutch_force_per_stud_N": scenario_params.f_clutch,
                "label": f"{name}\n{scenario} ({scenario_params.f_clutch:.2f} N/stud)",
                "n_studs": nx * ny, "tilted": tilted,
                "needed_N": needed, "lo_N": band["lo_N"], "hi_N": band["hi_N"], "verdict": v,
                "ceiling_clutch_force_per_stud_N": ceiling,
            })
            print(f"  {name} [{scenario}, {scenario_params.f_clutch:.2f} N/stud]: "
                  f"needs {needed:.1f} N -> {v}")

    fig_path = _figure(rows, ctx.out_dir)

    table = ("| press | scenario | clutch (N/stud) | studs fought | force needed (N) | "
             "arm's range (N) | verdict | clutch ceiling (N/stud) |\n"
             "|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        fought = min(2, r["n_studs"]) if r["tilted"] else r["n_studs"]
        table += (
            f"| {r['config']} | {r['scenario']} | {r['clutch_force_per_stud_N']:.2f} | {fought} | "
            f"{r['needed_N']:.1f} | {r['lo_N']:.1f} to {r['hi_N']:.1f} | {r['verdict']} | "
            f"{r['ceiling_clutch_force_per_stud_N']:.2f} |\n"
        )

    by_key = {(r["config"], r["scenario"]): r for r in rows}
    yours_straight = by_key[("2x4 straight", "your parameters")]
    yours_tilted = by_key[("2x4 tilted (1 row)", "your parameters")]
    fp_straight = by_key[("2x4 straight", "first-principles upper scenario")]
    fp_tilted = by_key[("2x4 tilted (1 row)", "first-principles upper scenario")]
    straight_ok = yours_straight["verdict"] != "not possible anywhere on the patch"
    tilted_ok = yours_tilted["verdict"] != "not possible anywhere on the patch"
    reading = [
        f"At this file's clutch_force_per_stud_N = {snap_params.f_clutch:.2f} N and "
        f"push_pull_asymmetry_beta = {snap_params.beta:.2f}, a straight press of a 2x4 is "
        f"**{yours_straight['verdict']}**, and a tilted press of a 2x4 is "
        f"**{yours_tilted['verdict']}**.",
        f"At the first-principles upper scenario ({FIRST_PRINCIPLES_CLUTCH_N:.2f} N/stud, "
        "guide 02c section 2.3 -- not this file's default, kept visible here so the risk does "
        f"not disappear when the shipped default looks fine), a straight press of a 2x4 is "
        f"**{fp_straight['verdict']}**, and a tilted press is **{fp_tilted['verdict']}**.",
    ]
    if not straight_ok and tilted_ok:
        reading.append(
            "A tilted press (one stud row at a time) is not an optimisation here -- it is the "
            "only press that works at all, matching guide 02c's own conclusion."
        )
    if not tilted_ok:
        reading.append(
            "Even a tilted press is out of reach at these parameters. Check "
            "`arm.supply_voltage_V` and `arm.torque_limit_Nm` first -- more voltage buys more "
            "push directly -- and re-measure `bricks.clutch_force_per_stud_N` if you have not "
            "already; it is the single most load-bearing number here."
        )
    reading.append(
        "The 'clutch ceiling' column is the highest clutch_force_per_stud_N each press could "
        "still manage at the arm's best cell; compare it against your own measured or derived "
        "value in `press-bench`'s report."
    )

    sections = [
        report_header(
            ctx, "seating-feasibility: can the arm press a brick home?",
            "Computes the force needed to seat a 2x4 and a 2x2, straight and tilted (one stud "
            "row at a time), from the parameter file's snap-fit settings, and compares it "
            "against the arm's own downward force over the build plate at the file's supply "
            "voltage and torque limit (brick_builder.robot.force_map). Every configuration is shown "
            "for this file's own parameters AND for the first-principles upper scenario, side "
            "by side, so a shipped default that happens to look fine never hides the risk."
        ),
        "## Parameters used\n\n" + parameter_table_md(
            params, sorted(p for p in params.values if any(p.startswith(pre) for pre in RELEVANT_PREFIXES))
        ),
        f"## Arm's downward force\n\n{band['lo_N']:.1f} to {band['hi_N']:.1f} N over the build "
        f"plate patch (x in {BUILD_PLATE_X_RANGE} m, y in {BUILD_PLATE_Y_RANGE} m, z = "
        f"{BUILD_PLATE_Z} m), {band['n_reachable']}/{band['n_total']} grid cells reachable.\n",
        "## Verdict\n\n" + table,
        f"![Push-on force needed against what the arm can deliver, your parameters and the "
        f"first-principles scenario side by side]({fig_path.name})\n",
        "## Reading\n\n" + "\n".join(f"- {r}" for r in reading),
    ]
    write_report(ctx.out_dir, sections)

    write_data_json(ctx.out_dir, {
        "arm": {"vin": arm_cfg.vin, "torque_limit_Nm": arm_cfg.torque_limit_nm,
                "p_gain": arm_cfg.p_gain},
        "snap": {"f_clutch": snap_params.f_clutch, "beta": snap_params.beta,
                 "curve": snap_params.curve},
        "first_principles_clutch_force_per_stud_N": FIRST_PRINCIPLES_CLUTCH_N,
        "force_band": {"lo_N": band["lo_N"], "hi_N": band["hi_N"],
                        "n_reachable": band["n_reachable"], "n_total": band["n_total"]},
        "rows": rows,
    })
