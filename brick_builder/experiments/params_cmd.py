"""``params``: print and check a parameter file.

    python -m brick_builder.experiments params --params my_params.yaml

Loads the file (already validated by the time this module runs -- an invalid file never gets
here, ``brick_builder.experiments.main`` prints the validation error and exits first), prints every
parameter grouped by section with its status, cross-checks the two brick masses against what
the entered dimensions and density actually generate, and lists which parameters are still
assumptions, worst first. No simulator is built; this is the cheapest of the six experiments and
the one to run right after editing a parameter file, before anything else.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt

from .. import params as params_mod
from ..params import ParamFile
from ..robot.patch_so101 import ArmConfig
from .common import (
    ExperimentContext,
    MUTED,
    STATUS_COLOURS,
    assumptions_md,
    parameter_table_md,
    report_header,
    save_figure,
    write_data_json,
    write_report,
)

SECTIONS = ("bricks", "arm", "camera", "workspace")

# The four headline unknowns, plus a plausible/reference band for each where one
# exists, so the figure can show "here is the current value, here is the range it plausibly
# sits in" rather than just a bare number.
HEADLINE = (
    ("bricks.clutch_force_per_stud_N", "N", (0.5, 5.0), "plausible range"),
    ("bricks.brick_on_brick_friction", "-", (0.2, 0.5), "guide 02c randomization range"),
    ("bricks.jaw_on_brick_friction", "-", (0.3, 1.0), "guide 02c randomization range"),
    ("arm.supply_voltage_V", "V", (5.0, 12.0), "force-map reference range"),
    ("arm.torque_limit_Nm", "N*m", (1.9, 3.41), "datasheet vs. identified-law estimates"),
    ("camera.diagonal_fov_deg", "deg", (100.0, 130.0), "stated lens range"),
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def _mass_check(params: ParamFile) -> list[dict]:
    spec = params_mod.to_brick_spec(params)
    rows = []
    for nx, ny, path in ((2, 2, "bricks.mass_2x2_g"), (2, 4, "bricks.mass_2x4_g")):
        recorded = params.value(path)
        expected = params_mod.expected_brick_mass_g(spec, nx, ny)
        rows.append({
            "brick": f"{nx}x{ny}",
            "recorded_g": recorded,
            "expected_from_density_g": expected,
            "agree_within_10pct": (
                recorded is not None
                and abs(recorded - expected) <= 0.10 * expected
            ),
        })
    return rows


def _hooks_missing(params: ParamFile) -> list[str]:
    """Parameters this file records that currently have nowhere to land in the simulator."""
    notes = []
    valid = {f.name for f in dataclasses.fields(ArmConfig)}
    if "joint_friction_nm" not in valid and "joint_friction" not in valid:
        notes.append(
            "arm.joint_friction_Nm: ArmConfig has no joint-friction field. patch_so101._patch_joints "
            "hard-codes the BAM-identified dry friction (0.0515 N.m) on every joint, so this value is "
            "recorded but never reaches the simulator."
        )
    notes.append(
        "camera.*: there is no CameraConfig dataclass to build. The wrist camera's fovy and pose are "
        "module-level constants in patch_so101.py computed once at import time; this loader's "
        "CameraSettings is a local stand-in, not something you can hand to attach_arm()."
    )
    return notes


def _figure_headline(params: ParamFile, out_dir) -> str:
    fig, axes = plt.subplots(len(HEADLINE), 1, figsize=(9.5, 1.55 * len(HEADLINE)))
    for ax, (path, unit, band, band_label) in zip(axes, HEADLINE):
        pv = params.get(path)
        lo, hi = band
        pad = 0.15 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.15
        ax.axvspan(lo, hi, color="#0072B2", alpha=0.12)
        ax.axvline(lo, color="#0072B2", lw=1.0, ls=":")
        ax.axvline(hi, color="#0072B2", lw=1.0, ls=":")
        colour = STATUS_COLOURS[pv.status]
        ax.plot([pv.value], [0.0], marker="o", ms=13, color=colour, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        unit_text = "" if unit == "-" else f" {unit}"
        ax.text(pv.value, 0.55, f"{pv.value:g}{unit_text}  ({pv.status})", ha="center", va="bottom",
                fontsize=10.5, color=colour, fontweight="bold")
        ax.text(lo - pad * 0.1, -0.65, band_label, ha="left", va="top", fontsize=8.5, color=MUTED)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([])
        ax.set_title(path, fontsize=10.5, loc="left", color="#1a1a1a")
        ax.grid(alpha=0.18, axis="x")
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Headline unknowns vs. plausible range", fontsize=13)
    path = save_figure(fig, out_dir, "params_headline.png")
    return str(path)


def run(ctx: ExperimentContext) -> None:
    params = ctx.params

    mass_rows = _mass_check(params)
    assumptions = params_mod.list_assumptions(params)
    n_measured = sum(1 for pv in params.values.values() if pv.status == "measured")
    hooks = _hooks_missing(params)

    print(f"Loaded {len(params.values)} parameters from {params.source}")
    print(f"  measured: {n_measured}   derived: "
          f"{sum(1 for pv in params.values.values() if pv.status == 'derived')}   "
          f"assumed: {sum(1 for pv in params.values.values() if pv.status == 'assumed')}")
    for row in mass_rows:
        flag = "OK" if row["agree_within_10pct"] else "CHECK"
        print(f"  [{flag}] {row['brick']} mass: recorded {row['recorded_g']} g, "
              f"density+dimensions predict {row['expected_from_density_g']:.2f} g")
    headline_unmeasured = [p for p, *_ in HEADLINE if params.get(p).status != "measured"]
    if not assumptions:
        print("  every parameter is status 'measured'.")
    else:
        print(f"  {len(assumptions)} parameter(s) are not yet measured.")
        if headline_unmeasured:
            print("  of the headline unknowns, still not measured: "
                  + ", ".join(headline_unmeasured))

    fig_path = _figure_headline(params, ctx.out_dir)
    cam = params_mod.to_camera_settings(params)

    sections_md = []
    for section in SECTIONS:
        paths = sorted(p for p in params.values if p.startswith(section + "."))
        text = f"### {section}\n\n" + parameter_table_md(params, paths)
        if section == "camera":
            text += (
                f"\n\nAt this diagonal field of view and resolution, MuJoCo's own vertical "
                f"`fovy` (the number that actually reaches the simulator's camera) is "
                f"**{cam.fovy_deg:.1f} deg**."
            )
        sections_md.append(text)

    mass_md = "| brick | recorded (g) | density+dims predict (g) | agree within 10% |\n|---|---|---|---|\n"
    for row in mass_rows:
        mass_md += (
            f"| {row['brick']} | {row['recorded_g']} | {row['expected_from_density_g']:.2f} | "
            f"{'yes' if row['agree_within_10pct'] else 'no -- check density_kg_m3 or the dimensions'} |\n"
        )

    reading = [
        f"{len(params.values)} parameters loaded, {n_measured} of them `measured`, "
        f"{len(assumptions)} still `assumed` or `derived`.",
    ]
    if headline_unmeasured:
        reading.append(
            "Of the headline unknowns (the ones that decide the most), still not "
            "measured on this team's own hardware: " + ", ".join(f"`{p}`" for p in headline_unmeasured)
            + ". These are worth measuring before any of the others, whatever their status label."
        )
    elif assumptions:
        reading.append(
            "All of the headline unknowns are measured; what remains assumed is lower-impact."
        )
    reading.append(
        "See docs/guide/10-measure-and-enter-parameters.md for the measurement procedure for "
        "every parameter still marked `assumed`."
    )

    sections = [
        report_header(
            ctx, "params: parameter file check",
            "Loads a parameter file, validates it against the schema in brick_builder/params.py, "
            "cross-checks the two recorded brick masses against what the density and dimensions "
            "predict, and lists everything still unmeasured."
        ),
        "## Parameters, by section\n\n" + "\n\n".join(sections_md),
        "## Brick mass cross-check\n\n" + mass_md,
        f"![Headline unknowns against their plausible range]({Path(fig_path).name})\n",
        "## Still assumptions\n\n" + assumptions_md(params),
        "## Hooks this loader could not reach\n\n" + "\n".join(f"- {h}" for h in hooks),
        "## Reading\n\n" + "\n".join(f"- {r}" for r in reading),
    ]
    write_report(ctx.out_dir, sections)

    write_data_json(ctx.out_dir, {
        "source": str(params.source),
        "n_parameters": len(params.values),
        "n_measured": n_measured,
        "n_assumptions": len(assumptions),
        "mass_check": mass_rows,
        "camera_fovy_deg": cam.fovy_deg,
        "assumptions": [
            {"path": a.path, "value": a.value, "status": a.status, "unit": a.unit, "doc": a.doc}
            for a in assumptions
        ],
        "hooks_missing": hooks,
        "parameters": params.as_nested_dict(),
    })
