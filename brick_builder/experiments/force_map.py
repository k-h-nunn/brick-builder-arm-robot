"""``force-map``: available downward force over the workspace, at your arm's settings.

    python -m brick_builder.experiments force-map --params my_params.yaml

Grids the arm's reachable workspace at a fixed height and reports the static downward force it
can deliver, tool-down, at the parameter file's supply voltage and torque limit -- the same
computation ``brick_builder.robot.force_map`` uses for the guide's own force-map figure, just at your
own arm settings instead of the three fixed reference voltages. Also reports the force at your
configured baseplate position (``workspace.baseplate_x_m``/``_y_m``) specifically.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from .. import params as params_mod
from ..robot.force_map import figure_force_map, press_force_map
from ..robot.kinematics import ArmKinematics
from ..robot.patch_so101 import attach_arm
from .common import ExperimentContext, parameter_table_md, report_header, write_data_json, write_report

RELEVANT_PREFIXES = ("arm.", "workspace.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x-min", type=float, default=0.08)
    parser.add_argument("--x-max", type=float, default=0.30)
    parser.add_argument("--y-min", type=float, default=-0.16)
    parser.add_argument("--y-max", type=float, default=0.16)
    parser.add_argument("--z", type=float, default=0.02, help="tool height, m (default 0.02)")
    parser.add_argument("--grid-nx", type=int, default=23)
    parser.add_argument("--grid-ny", type=int, default=21)


def _nearest_cell(result: dict, x: float, y: float) -> tuple[int, int]:
    ix = int(np.argmin(np.abs(result["xs"] - x)))
    iy = int(np.argmin(np.abs(result["ys"] - y)))
    return iy, ix


def run(ctx: ExperimentContext) -> None:
    params = ctx.params
    extra = ctx.extra
    x_range = (getattr(extra, "x_min", 0.08), getattr(extra, "x_max", 0.30))
    y_range = (getattr(extra, "y_min", -0.16), getattr(extra, "y_max", 0.16))
    z = getattr(extra, "z", 0.02)
    grid = (getattr(extra, "grid_nx", 23), getattr(extra, "grid_ny", 21))

    arm_cfg = params_mod.to_arm_config(params)

    world = mujoco.MjSpec()
    attach_arm(world, arm_cfg)
    model = world.compile()
    kin = ArmKinematics(model)
    xs = np.linspace(*x_range, grid[0])
    ys = np.linspace(*y_range, grid[1])
    result = press_force_map(
        model, kin, xs, ys, z=z, vin_list=(arm_cfg.vin,), torque_limit_nm=arm_cfg.torque_limit_nm,
    )
    reach = result["reachable"]
    forces = result["force"][arm_cfg.vin]
    reachable_forces = forces[reach]

    n_reach, n_total = int(reach.sum()), int(reach.size)
    lo, hi = float(np.nanmin(reachable_forces)), float(np.nanmax(reachable_forces))
    med = float(np.nanmedian(reachable_forces))

    bx = float(params.value("workspace.baseplate_x_m"))
    by = float(params.value("workspace.baseplate_y_m"))
    in_range = x_range[0] <= bx <= x_range[1] and y_range[0] <= by <= y_range[1]
    plate_force = plate_reachable = None
    if in_range:
        iy, ix = _nearest_cell(result, bx, by)
        plate_reachable = bool(reach[iy, ix])
        if plate_reachable:
            plate_force = float(forces[iy, ix])

    print(f"Arm at {arm_cfg.vin:g} V, {arm_cfg.torque_limit_nm:g} N.m: "
          f"{lo:.1f} to {hi:.1f} N over the grid ({n_reach}/{n_total} cells reachable, "
          f"median {med:.1f} N)")
    if plate_force is not None:
        print(f"  at your configured baseplate ({bx:.3f}, {by:.3f}) m: "
              f"~{plate_force:.1f} N (nearest grid cell)")
    elif in_range and not plate_reachable:
        print(f"  your configured baseplate ({bx:.3f}, {by:.3f}) m is NOT reachable tool-down at z={z} m")
    else:
        print(f"  your configured baseplate ({bx:.3f}, {by:.3f}) m is outside this grid's range; "
              "widen --x-min/--x-max/--y-min/--y-max to check it")

    fig_path = figure_force_map(result, ctx.out_dir / "force_map.png")

    reading = [
        f"Over the grid, the arm delivers **{lo:.1f} to {hi:.1f} N** tool-down "
        f"({n_reach}/{n_total} cells reachable), median {med:.1f} N.",
    ]
    if plate_force is not None:
        reading.append(f"At your configured baseplate position, that is about **{plate_force:.1f} N**.")
    reading.append(
        "Compare this against the `press-bench` and `seating-feasibility` reports' push-on "
        "numbers: if the baseplate force is below what a brick needs, either move the baseplate "
        "closer to the high-force region visible in the figure, or increase "
        "`arm.supply_voltage_V`."
    )

    sections = [
        report_header(
            ctx, "force-map: downward force over the workspace",
            "Grids the workspace at a fixed tool height and reports the static downward press "
            "force the arm can deliver tool-down, at the parameter file's supply voltage and "
            "torque limit (brick_builder.robot.force_map.press_force_map)."
        ),
        "## Parameters used\n\n" + parameter_table_md(
            params, sorted(p for p in params.values if any(p.startswith(pre) for pre in RELEVANT_PREFIXES))
        ),
        f"## Grid\n\nx in {x_range} m, y in {y_range} m, z = {z} m, {grid[0]}x{grid[1]} points.\n",
        f"![Downward force over the workspace]({fig_path.name})\n",
        "## Reading\n\n" + "\n".join(f"- {r}" for r in reading),
    ]
    write_report(ctx.out_dir, sections)

    write_data_json(ctx.out_dir, {
        "arm": {"vin": arm_cfg.vin, "torque_limit_Nm": arm_cfg.torque_limit_nm},
        "grid": {"x_range": x_range, "y_range": y_range, "z": z, "nx": grid[0], "ny": grid[1]},
        "n_reachable": n_reach,
        "n_total": n_total,
        "force_lo_N": lo,
        "force_hi_N": hi,
        "force_median_N": med,
        "baseplate": {"x_m": bx, "y_m": by, "force_N": plate_force, "reachable": plate_reachable},
    })
