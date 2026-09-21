"""``grasp-hold``: can the stock jaws hold a brick by friction alone?

    python -m brick_builder.experiments grasp-hold --params my_params.yaml

The real SO-101 gripper is a fixed jaw and a moving jaw, nothing attached to either -- so a
grasp holds a brick only by friction between the bare printed jaw faces and the brick's own
walls (``bricks.jaw_on_brick_friction``). This builds a scene with a 2x4 and a 2x2 gripped in
friction mode, pulls on each one with a sweep of loads while it hangs (representing the brick's
own weight and whatever else is being carried) and while something pushes up on it (the reaction
of pressing the brick onto the baseplate), and reports how far it slips.

Scene construction goes entirely through ``brick_builder.params.filter_dataclass_kwargs`` (via
``to_scene_config``): ``SceneConfig`` (and possibly ``HeldBrick``) is still being extended
elsewhere in this codebase, so every dataclass this module builds is constructed from only the
fields that exist on it *right now*, never a fixed kwarg list.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from .. import params as params_mod
from ..bricks.materials import ARM_GRIP_FORCE_N
from .common import ExperimentContext, SERIES_COLOURS, parameter_table_md, report_header, save_figure, write_data_json, write_report

RELEVANT_PREFIXES = ("bricks.jaw_on_brick_friction", "bricks.brick_on_brick_friction", "arm.")
BRICKS = ((2, 2), (2, 4))
DIRECTIONS = (("hanging (own weight / carried load)", (0.0, 0.0, -1.0)),
              ("pushed up (press reaction)", (0.0, 0.0, 1.0)))
SLIP_LIMIT_M = 0.001  # 1 mm, the acceptance figure compose.py's own docstring uses


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--forces", type=float, nargs="+", default=[2.0, 5.0, 8.0, 12.0, 17.0],
                         help=f"newtons to sweep (default: 2 5 8 12 17, the last being "
                              f"ARM_GRIP_FORCE_N={ARM_GRIP_FORCE_N})")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--settle-steps", type=int, default=500,
                         help="physics steps to let the gripper close before pulling (default "
                              "500 = 1 s at dt=2 ms, matching tests/scene/test_compose.py)")


def _build_scene(params, nx: int, ny: int):
    from ..robot.patch_so101 import ArmConfig
    from ..scene.compose import HeldBrick, SceneConfig, build_scene

    held_kwargs = {"nx": nx, "ny": ny, "colour": "red", "grasp_mode": "friction"}
    held_valid = {f.name for f in dataclasses.fields(HeldBrick)}
    held = HeldBrick(**{k: v for k, v in held_kwargs.items() if k in held_valid})
    skipped_held = sorted(set(held_kwargs) - held_valid)

    scene_cfg, skipped_scene = params_mod.to_scene_config(
        params, held=held, grasp_mode="friction", tray=False, cameras=(),
    )
    scene = build_scene(scene_cfg)
    return scene, skipped_held + skipped_scene


def _sweep(scene, forces: list[float], seconds: float, settle_steps: int) -> list[dict]:
    from ..scene.compose import push_held_brick, reset_scene

    rows = []
    for label, direction in DIRECTIONS:
        for force in forces:
            reset_scene(scene, np.random.default_rng(0))
            # The gripper is commanded closed at reset but needs time to actually get there and
            # build up pinch force; every existing grasp test in tests/scene/test_compose.py
            # settles 500 steps (1 s) before pulling, and skipping this makes the brick fly out
            # of an effectively still-open gripper regardless of friction.
            scene.step(settle_steps)
            out = push_held_brick(scene, force=force, seconds=seconds, direction=direction)
            row = {
                "direction": label, "force_N": force, "slip_mm": out["slip_m"] * 1e3,
                "peak_slip_mm": out["peak_slip_m"] * 1e3, "holds": out["slip_m"] < SLIP_LIMIT_M,
                "warnings": out["warnings"],
            }
            # The damage monitor is attached to every scene `build_scene` makes, and `reset_scene`
            # clears it, so these two numbers are this hold's own and nobody had to drive them.
            if scene.monitor is not None:
                row["damage_state"] = scene.monitor.state
                row["peak_wall_stress_MPa"] = scene.monitor.peak_wall_stress_Pa / 1e6
            rows.append(row)
    return rows


def _max_holding_force(rows: list[dict], direction: str) -> float | None:
    """The largest swept force that still held (slip < 1 mm), for one direction."""
    ok = [r["force_N"] for r in rows if r["direction"] == direction and r["holds"]]
    return max(ok) if ok else None


def _figure(all_rows: dict[str, list[dict]], out_dir) -> "object":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, (label, direction) in zip(axes, DIRECTIONS):
        for (nx, ny), colour in zip(BRICKS, SERIES_COLOURS):
            rows = [r for r in all_rows[f"{nx}x{ny}"] if r["direction"] == label]
            forces = [r["force_N"] for r in rows]
            slips = [max(r["slip_mm"], 1e-4) for r in rows]
            ax.plot(forces, slips, "o-", color=colour, lw=2.0, ms=7, label=f"{nx}x{ny}")
        ax.axhline(SLIP_LIMIT_M * 1e3, color="#D55E00", lw=1.4, ls="--")
        ax.text(0.98, SLIP_LIMIT_M * 1e3, "1 mm slip limit ", color="#D55E00", va="bottom",
                ha="right", fontsize=9.5, transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel("applied force [N]", fontsize=12)
        ax.set_title(label, fontsize=12.5)
        ax.legend(fontsize=10, loc="upper left")
    axes[0].set_ylabel("slip after the hold [mm] (log scale)", fontsize=12)
    fig.suptitle("Friction-only grasp, slip vs. applied force", fontsize=14.5)
    return save_figure(fig, out_dir, "grasp_hold.png")


def run(ctx: ExperimentContext) -> None:
    params = ctx.params
    forces = sorted(getattr(ctx.extra, "forces", [2.0, 5.0, 8.0, 12.0, 17.0]))
    seconds = getattr(ctx.extra, "hold_seconds", 2.0)

    settle_steps = getattr(ctx.extra, "settle_steps", 500)
    all_rows: dict[str, list[dict]] = {}
    skipped_all: list[str] = []
    summary = []
    for nx, ny in BRICKS:
        scene, skipped = _build_scene(params, nx, ny)
        skipped_all.extend(skipped)
        rows = _sweep(scene, forces, seconds, settle_steps)
        all_rows[f"{nx}x{ny}"] = rows
        for label, _ in DIRECTIONS:
            max_ok = _max_holding_force(rows, label)
            summary.append({"brick": f"{nx}x{ny}", "direction": label, "holds_up_to_N": max_ok})
            print(f"{nx}x{ny}, {label}: holds up to "
                  f"{'nothing tried' if max_ok is None else f'{max_ok:g} N'} "
                  f"(of {forces} N tried)")

    if skipped_all:
        print(f"Note: these SceneConfig/HeldBrick fields do not exist right now and were "
              f"skipped: {sorted(set(skipped_all))}")

    fig_path = _figure(all_rows, ctx.out_dir)

    table = "| brick | direction | " + " | ".join(f"{f:g} N" for f in forces) + " |\n"
    table += "|---|---|" + "---|" * len(forces) + "\n"
    for name, rows in all_rows.items():
        for label, _ in DIRECTIONS:
            cells = [r for r in rows if r["direction"] == label]
            table += f"| {name} | {label} | " + " | ".join(
                f"{c['slip_mm']:.3f} mm{'  OK' if c['holds'] else '  SLIPPED'}" for c in cells
            ) + " |\n"

    summary_md = "| brick | direction | holds up to |\n|---|---|---|\n"
    for s in summary:
        summary_md += f"| {s['brick']} | {s['direction']} | " + (
            "nothing tried held" if s["holds_up_to_N"] is None else f"{s['holds_up_to_N']:g} N"
        ) + " |\n"

    damage_md = ""
    if any("damage_state" in r for rows in all_rows.values() for r in rows):
        damage_md = (
            "| brick | direction | force [N] | damage state | peak wall stress [MPa] |\n"
            "|---|---|---|---|---|\n"
        )
        for name, rows in all_rows.items():
            for r in rows:
                if "damage_state" not in r:
                    continue
                damage_md += (f"| {name} | {r['direction']} | {r['force_N']:g} | "
                              f"{r['damage_state']} | {r['peak_wall_stress_MPa']:.2f} |\n")

    friction = params.value("bricks.jaw_on_brick_friction")
    reading = [
        f"At jaw_on_brick_friction = {friction:g}, see the summary table for how much load each "
        f"brick/direction combination survives with under 1 mm of slip.",
        "The stock SO-101 jaws have nothing attached to grip with, so this is a hard ceiling on "
        "how much can be carried, or how hard the brick can be pushed against the baseplate, "
        "before it starts sliding out of the gripper.",
        "One thing worth knowing before changing `jaw_on_brick_friction` to try to fix a low "
        "number here: `tests/scene/test_compose.py` shows the *downward* (hanging) hold is "
        "dominated by the jaw's wedge geometry, not by friction -- it stays marginal even at a "
        "friction coefficient of 1.0, and the upward (press-reaction) hold barely changes "
        "between a coefficient of 0.3 and 1.0 either, because the closing gap wedges onto the "
        "brick either way. So a low number in the 'hanging' column especially may not move much "
        "even after a real friction measurement; it is closer to a fact about the stock jaw "
        "shape than about this parameter.",
    ]
    if skipped_all:
        reading.append(
            f"Some requested scene fields do not exist on the current `SceneConfig`/`HeldBrick` "
            f"(still being extended elsewhere in the codebase): {sorted(set(skipped_all))}. The "
            f"scene was still built with everything else."
        )

    sections = [
        report_header(
            ctx, "grasp-hold: friction-only grasp under load",
            "Grips a 2x4 and a 2x2 with the stock jaws in friction mode (no weld), pulls on the "
            "held brick with a sweep of loads while it hangs and while it is pushed up, and "
            "reports the slip after each hold."
        ),
        "## Parameters used\n\n" + parameter_table_md(
            params, sorted(p for p in params.values if any(p.startswith(pre) for pre in RELEVANT_PREFIXES))
        ),
        "## Summary\n\n" + summary_md,
        "## Full sweep\n\n" + table,
        *(["## Damage\n\n" + damage_md] if damage_md else []),
        f"![Slip against applied load]({fig_path.name})\n",
        "## Reading\n\n" + "\n".join(f"- {r}" for r in reading),
    ]
    write_report(ctx.out_dir, sections)

    write_data_json(ctx.out_dir, {
        "forces_N": forces,
        "hold_seconds": seconds,
        "settle_steps": settle_steps,
        "jaw_on_brick_friction": friction,
        "rows": all_rows,
        "summary": summary,
        "skipped_fields": sorted(set(skipped_all)),
    })
