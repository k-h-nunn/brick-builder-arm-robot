"""Figure-producing functions for brick_builder.bricks, for scripts/make_media.py.

Kept separate from generator.py so the generator itself stays free of matplotlib/rendering
concerns (generator.py builds MjSpecs; this module renders and plots them).
"""

from __future__ import annotations

import mujoco

import numpy as np

from dataclasses import replace

from brick_builder.bricks.bench import (
    BenchConfig,
    PressBench,
    measure_rocking_ratio,
    predicted_press_force,
    quasi_static_band,
)
from brick_builder.bricks.generator import COLLISION_GROUP, VISUAL_GROUP, add_brick
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import DUPLO, BrickSpec

# One fixed colour per curve shape, used by both figures so the same shape is the same colour
# wherever it appears. Checked for colour-vision separation rather than picked by eye.
CURVE_COLOURS = {"ramp": "#1f6fb2", "plateau": "#7a5195", "detent": "#c1272d"}
CURVES = ("ramp", "plateau", "detent")


def make_exploded_collision_vs_visual_figure(out_path: str, spec: BrickSpec = DUPLO, nx: int = 2, ny: int = 4) -> None:
    """Render collision geoms and visual geoms of one brick side by side, save as PNG.

    Produces `docs/media/bricks/exploded_collision_vs_visual.png`. Uses `mujoco.Renderer` with
    two passes over the same scene, one showing only `COLLISION_GROUP` and one showing only
    `VISUAL_GROUP`, so the two panels show exactly what the physics engine sees and what a
    camera would see - the whole point of keeping the two geometry sets separate.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    world = mujoco.MjSpec()
    world.worldbody.add_light(name="key", pos=[0.1, -0.1, 0.2], dir=[-0.3, 0.3, -1.0])
    add_brick(world, "brick", nx, ny, "red", pos=(0.0, 0.0, 0.05), free=False, spec=spec)
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=480)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, group, title in ((axes[0], COLLISION_GROUP, "collision"), (axes[1], VISUAL_GROUP, "visual")):
        scene_option = mujoco.MjvOption()
        for g in range(mujoco.mjNGROUP):
            scene_option.geomgroup[g] = 1 if g == group else 0
        renderer.update_scene(data, camera=-1, scene_option=scene_option)
        img = renderer.render()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{nx}x{ny} DUPLO brick: collision primitives vs. visual mesh")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    renderer.close()


def make_force_vs_depth_figure(
    out_path,
    params: "SnapParams | None" = None,
    cfg: "BenchConfig | None" = None,
    speed: float = 0.002,
    curves: tuple = CURVES,
) -> dict:
    """Press one 2x4 brick on with each curve shape and plot measured against configured.

    One line per shape in `curves` (the measured trace) over the curve the parameters asked
    for (the pale thick line underneath). The three shapes are the same in only one place, the
    right-hand edge: a seated stud holds `f_clutch * (1 - beta)` whichever shape is in use.

    Returns one entry per shape with its relative RMS error over `quasi_static_band`, plus the
    default shape's numbers at the top level so older callers keep working.
    """

    params = params or SnapParams()
    cfg = cfg or BenchConfig()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.bricks.snap import ColletSnap

    n = cfg.nx * cfg.ny
    h = cfg.spec.stud_h
    grid = np.linspace(0.0, h, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    out: dict = {"curves": {}}
    for curve in curves:
        p = replace(params, curve=curve)
        colour = CURVE_COLOURS.get(curve, "#444444")
        bench = PressBench(ColletSnap(p), cfg)
        trace = bench.press(speed=speed)
        model = predicted_press_force(grid, n, p, h)
        ax.plot(grid * 1e3, model, lw=3.2, color=colour, alpha=0.25, solid_capstyle="round")
        # the measured trace in the order it happened, not resampled onto the depth grid: a
        # press through a detent is not monotone in depth and sorting it would hide that.
        stop = int(np.argmax(trace.depth)) + 1
        ax.plot(
            trace.depth[:stop] * 1e3, trace.force[:stop], lw=1.8, color=colour,
            solid_capstyle="round",
        )
        measured = trace.at_depths(grid)
        # direct label on the configured curve, so the shapes are told apart by more than
        # their colour and the labels cannot land on top of each other
        anchor, offset = {
            "ramp": (0.85 * h, (0, -16)),
            "plateau": (0.45 * h, (0, 7)),
        }.get(curve, (p.detent_depth, (5, 5)))
        j = int(np.argmin(np.abs(grid - anchor)))
        ax.annotate(
            curve,
            xy=(grid[j] * 1e3, model[j]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            color=colour,
        )
        lo, hi = quasi_static_band(p, h)
        band = (grid >= lo) & (grid <= hi)
        out["curves"][curve] = {
            "rms_relative": float(
                np.sqrt(np.mean((measured[band] - model[band]) ** 2))
                / max(np.max(model[band]), 1e-9)
            ),
            "peak_measured": trace.peak_force,
            "peak_model": float(np.max(model)),
            "seated_force": float(model[-1]),
        }

    ax.axvline(h * 1e3, color="#888888", lw=0.8, ls=":")
    ax.set_xlim(-0.35, h * 1e3 + 0.35)
    ax.annotate(
        "all three end here: a seated stud holds the same force",
        xy=(h * 1e3, n * params.f_clutch * (1.0 + params.beta)),
        xytext=(-6, 14),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color="#555555",
    )
    ax.set_xlabel("insertion depth (mm)")
    ax.set_ylabel("press force (N)")
    ax.set_title(f"{cfg.nx}x{cfg.ny} DUPLO press at {speed * 1e3:.0f} mm/s: three curve shapes")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], color="#444444", lw=2.0, label="simulated (collet model)"),
        plt.Line2D([], [], color="#444444", lw=3.2, alpha=0.25, label="configured  n·f·(1+β)·g(d)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    first = out["curves"].get(params.curve) or next(iter(out["curves"].values()))
    out.update(
        {
            "rms_relative": first["rms_relative"],
            "peak_measured": first["peak_measured"],
            "peak_model": first["peak_model"],
        }
    )
    return out


def make_rocking_vs_flat_figure(
    out_path,
    params: "SnapParams | None" = None,
    tilt_deg: float = 2.0,
    curves: tuple = CURVES,
) -> dict:
    """Press the same brick flat and rocked under each curve shape, and plot all three.

    Every run is the same brick, the same studs and the same gauge; only the tool's grip, the
    motion and `SnapParams.curve` differ. The top row is the pair of force traces for each
    shape, the bottom row is the three peak ratios, which is bench test 4. The point the
    figure makes is that rocking only pays when the curve has an entry bump to stage: with
    "ramp" and "plateau" the press ends with every stud sliding at once whatever the tool did,
    and the ratio is about 1.

    Returns one entry per shape, plus the default shape's numbers at the top level so older
    callers keep working.
    """

    params = params or SnapParams()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {c: measure_rocking_ratio(replace(params, curve=c), tilt_deg=tilt_deg)
               for c in curves}

    fig = plt.figure(figsize=(7.6, 5.4), dpi=150)
    gs = fig.add_gridspec(2, len(curves), height_ratios=[3, 2], hspace=0.45, wspace=0.28)
    top = max(r["flat_peak"] for r in results.values()) * 1.12
    axes = []
    for i, curve in enumerate(curves):
        r = results[curve]
        colour = CURVE_COLOURS.get(curve, "#444444")
        ax = fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(r["flat"].t, r["flat"].force, color="#999999", lw=1.6)
        ax.plot(r["rock"].t, r["rock"].force, color=colour, lw=1.8)
        ax.set_ylim(-2.0, top)
        ax.set_title(f"{curve}   ratio {r['ratio']:.2f}", fontsize=10, color=colour)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("press force at the tool (N)", fontsize=9)
            ax.annotate("flat", xy=(r["flat"].t[int(np.argmax(r["flat"].force))],
                                    r["flat"].peak_force),
                        xytext=(4, 2), textcoords="offset points", fontsize=8, color="#777777")
            ax.annotate("rocking", xy=(r["rock"].t[int(np.argmax(r["rock"].force))],
                                       r["rocking_peak"]),
                        xytext=(4, -10), textcoords="offset points", fontsize=8, color=colour)
        else:
            ax.tick_params(labelleft=False)

    bx = fig.add_subplot(gs[1, :])
    names = list(curves)
    ratios = [results[c]["ratio"] for c in names]
    ypos = np.arange(len(names))[::-1]
    bx.barh(ypos, ratios, height=0.55, color=[CURVE_COLOURS.get(c, "#444") for c in names])
    bx.axvline(0.75, color="#333333", ls="--", lw=1.0)
    bx.set_ylim(ypos.min() - 0.6, ypos.max() + 0.9)
    bx.annotate(
        "0.75: what the spec asks a staged press to reach",
        xy=(0.75, ypos.max() + 0.55), xytext=(5, 0), textcoords="offset points",
        fontsize=8, color="#333333", va="center", annotation_clip=False,
    )
    for y, c, v in zip(ypos, names, ratios):
        bx.annotate(f"{v:.2f}", xy=(v, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#333333")
    bx.set_yticks(ypos, names, fontsize=9)
    bx.set_xlim(0.0, max(1.25, max(ratios) * 1.2))
    bx.set_xlabel("rocking peak force / flat peak force", fontsize=9)
    bx.grid(alpha=0.25, axis="x")
    bx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.tick_params(labelsize=8, left=False)
    fig.suptitle(
        f"Rocking a 2x4 on at {tilt_deg:g}\u00b0 of tilt: only an entry bump makes it pay",
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    out = {
        "curves": {
            c: {k: v for k, v in r.items() if k not in ("flat", "rock")}
            for c, r in results.items()
        }
    }
    first = results.get(params.curve) or next(iter(results.values()))
    out.update(
        {
            "flat_peak": first["flat_peak"],
            "rocking_peak": first["rocking_peak"],
            "ratio": first["ratio"],
            "rocking_depth_spread": first["rocking_depth_spread"],
            "flat_depth_spread": first["flat_depth_spread"],
            "peak_wrist_torque": first["peak_wrist_torque"],
        }
    )
    return out
