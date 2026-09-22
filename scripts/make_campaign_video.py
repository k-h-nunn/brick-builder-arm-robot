#!/usr/bin/env python3
"""Record the two-brick build at 1920x1080, in one continuous take.

    python scripts/make_campaign_video.py
    python scripts/make_campaign_video.py --quality draft --seed 2
    python scripts/make_campaign_video.py --inverse learned \
        --weights results/campaign/closed_loop/weights/inverse.npz --version r5

One English command, two bricks, one simulator: the arm places the brick it is holding on the
cell the command names, then fetches a second brick off the table and places it beside the first.
Both cycles run in the same world and the same recording, so nothing is cut together.

The layout is the closed-loop video's: four square panels (the arm camera, the scene model's
labels, the target frame's labels, the renderer's labels) and the data strip, every string a
title with its units.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brick_builder.control.pick_place import CONTROL_DT, verify_placement  # noqa: E402
from brick_builder.demo.video import (  # noqa: E402
    CamKey, Recorder, Tick, arm_points, panel_size_for, quality_by_name, studio_look,
)
from make_demo_video import (  # noqa: E402
    CL_AZIMUTH, CL_DISTANCE, CL_ELEVATION, CL_LOOKAT, CL_MIN_SPAN, CL_READOUTS, CL_TITLE,
    FROZEN_SHA, Labeller, Panels, _cnn, _recording_labeller,
)

TITLE = CL_TITLE


def spec_line(*, task: str, driving: str, displayed: str, inverse: str, version: str,
              offset_mm: float, offset_deg: float) -> str:
    """The one line under the title. Titles and values, no sentences."""
    model = (f"scene model: {driving}" if driving == displayed
             else f"control: {driving} | panel: {displayed}")
    tag = f"{inverse} inverse" + (f" {version}" if version else "")
    return (f"{task} | {model} | {tag} | target-sequence tracking, 10 Hz | "
            f"DUPLO 2x2 | baseplate offset: {offset_mm:.1f} mm, {offset_deg:.1f} deg")


def build_scene_for(seed: int, prepare):
    """One world holding everything both cycles need: a brick in the jaws and two on the table.

    The first cycle places the held brick; the second fetches one off the table. Both in this
    world, because a video that swaps simulators mid-take is two videos.
    """
    from brick_builder.action.worlds import (
        DISTRACTOR, NOMINAL_PLATE_XY, REF_RED, REF_YELLOW, SUPPLY_BODY, SUPPLY_XY, WANTED,
        brick_displacement, displacement, nudge_brick, scene_cfg,
    )
    from brick_builder.scene.compose import HeldBrick, build_scene, reset_scene

    d_xy, d_yaw = displacement(seed)
    held = HeldBrick(nx=2, ny=2, colour="blue", offset_jitter=(0.0, 0.002, 0.0))
    cfg = scene_cfg(
        plate_xy=np.asarray(NOMINAL_PLATE_XY) + d_xy, plate_yaw=d_yaw,
        placed=[REF_YELLOW, REF_RED], held=held,
        home_tcp=(NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075), cameras=(),
    )
    cfg.supply = (WANTED, DISTRACTOR)
    cfg.supply_xy = SUPPLY_XY
    scene = build_scene(cfg)
    if prepare is not None:
        prepare(scene.model)
    reset_scene(scene, np.random.default_rng(seed))
    b_xy, b_yaw = brick_displacement(seed)
    nudge_brick(scene, SUPPLY_BODY, b_xy, b_yaw)
    return scene, d_xy, d_yaw


def cycle_motion(scene, episode, panels, truth, brick, *, lead_seconds: float,
                 hold_seconds: float, last: dict):
    """One cycle's per-tick records, turned into ticks the video understands."""
    from brick_builder.control.pick_place import gripper_press_force

    hold_tcp = episode.world.feedback(episode.ideal.frames[0].commands[0].grip).tcp_pos.copy()
    grip0 = episode.ideal.frames[0].commands[0].grip
    state, rgb = episode.world.observe_rgb()
    panels.update(rgb, episode.ideal.frames[0].label, truth(scene), window=truth.window)
    if panels.miou == panels.miou:
        last["miou"] = panels.miou
    del state
    for _ in range(max(int(round(lead_seconds / CONTROL_DT)), 1)):
        episode.world.command(hold_tcp, 0.0, grip0)
        yield Tick(scene=scene, phase=episode.ideal.frames[0].step, data=dict(last))

    for rec in episode.run():
        if rec.vision and rec.state is not None:
            panels.update(rec.rgb, rec.target_label, truth(scene), window=truth.window)
            if panels.miou == panels.miou:
                last["miou"] = panels.miou
        if rec.err_mm == rec.err_mm:
            last["err_mm"] = float(rec.err_mm)
        if rec.predict_mm == rec.predict_mm:
            last["predict_mm"] = float(rec.predict_mm)
        force = gripper_press_force(scene, brick)
        if force > 0.05:
            last["press_n"] = float(force)
        depths = np.asarray(scene.snap.stud_depths(brick), dtype=float)
        if depths.size:
            last["depth_mm"] = float(1e3 * np.min(depths))
        yield Tick(scene=scene, phase=rec.step, data=dict(last))

    park = episode.ideal.frames[-1].commands[-1]
    park_pos = np.array([park.pos[0] + episode.correction[0],
                         park.pos[1] + episode.correction[1], park.pos[2]])
    for _ in range(max(int(round(hold_seconds / CONTROL_DT)), 1)):
        episode.world.command(park_pos, park.spin + episode.correction[2], park.grip)
        yield Tick(scene=scene, phase="Verify", data=dict(last))


def build_video(out: Path, *, quality_name: str = "hd1080", seed: int = 1,
                lead_seconds: float = 2.0, gap_seconds: float = 1.5,
                hold_seconds: float = 3.0, stills: int = 6, stills_dir: Path | None = None,
                cnn_panels: bool = True, cnn_drives: bool = False,
                inverse: str = "analytic", weights: Path | None = None, version: str = ""):
    """Both cycles of the build, one world, one recording."""
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import REF_RED, REF_YELLOW, SUPPLY_BODY
    from brick_builder.bricks.structures import Placement
    from brick_builder.campaign.tasks import TASKS

    spec = TASKS["build_2"]
    cells = spec.cells
    quality = quality_by_name(quality_name)
    prepare = lambda model: studio_look(model, quality)   # noqa: E731

    source = None
    driving = "ground truth"
    if cnn_drives:
        source = _cnn()
        driving = f"CNN {FROZEN_SHA}"

    # Cycle one places the brick already in the jaws; cycle two fetches the next one off the
    # table. Two configurations, one world.
    cfg_one = PlaceConfig(pick=False, inverse=inverse)
    cfg_two = PlaceConfig(pick=True, inverse=inverse)
    placed_two = [REF_YELLOW, REF_RED,
                  Placement(h=2, w=2, x=cells[0][0], y=cells[0][1], z=0, colour="blue")]

    ideal_one, nominal_one = record_ideal(cfg_one, seed=seed, target=cells[0], prepare=prepare,
                                          source=source,
                                          label_hook=_recording_labeller(cfg_one, prepare, seed))
    nominal_one.close()
    ideal_two, nominal_two = record_ideal(cfg_two, seed=seed, target=cells[1], prepare=prepare,
                                          source=source, placed=placed_two,
                                          label_hook=_recording_labeller(cfg_two, prepare, seed))
    nominal_two.close()

    scene, d_xy, d_yaw = build_scene_for(seed, prepare)
    studio_look(scene.model, quality)
    truth = Labeller(scene)
    cnn = _cnn() if (cnn_panels and not cnn_drives) else source
    panels = Panels(panel_size_for(quality, 4), cnn=cnn)

    g = None
    if inverse == "learned":
        from brick_builder.action.models.inverse import LearnedInverse

        g = LearnedInverse.load(weights)

    first = TrackingEpisode(scene, ideal_one, cfg=cfg_one, seed=seed, target=cells[0],
                            twin_prepare=prepare, build_decode=True, source=source, inverse=g)
    held_name = scene.held_name

    length = (lead_seconds + len(ideal_one) * 5 * CONTROL_DT + gap_seconds
              + len(ideal_two) * 5 * CONTROL_DT + hold_seconds)
    still_times = list(np.linspace(0.04 * length, 0.97 * length, stills)) if stills else []
    subtitle = spec_line(
        task="Two-brick build, one command", driving=driving,
        displayed=(f"CNN {FROZEN_SHA}" if cnn is not None else "ground truth"),
        inverse=inverse, version=version,
        offset_mm=1e3 * float(np.linalg.norm(d_xy)), offset_deg=abs(math.degrees(d_yaw)))

    state: dict = {"second": None, "checks": []}

    def motion():
        last: dict = {}
        yield from cycle_motion(scene, first, panels, truth, held_name,
                                lead_seconds=lead_seconds, hold_seconds=gap_seconds, last=last)
        state["checks"].append(verify_placement(scene, held_name,
                                               (cells[0][0] + 0.5, cells[0][1] + 0.5, 0)))
        first.close()
        second = TrackingEpisode(scene, ideal_two, cfg=cfg_two, seed=seed, target=cells[1],
                                 twin_prepare=prepare, build_decode=True, source=source,
                                 inverse=g)
        state["second"] = second
        yield from cycle_motion(scene, second, panels, truth, SUPPLY_BODY,
                                lead_seconds=0.6, hold_seconds=hold_seconds, last=last)
        state["checks"].append(verify_placement(scene, SUPPLY_BODY,
                                               (cells[1][0] + 0.5, cells[1][1] + 0.5, 0)))

    with Recorder(
        scene, quality=quality,
        shot=[CamKey(0.0, CL_LOOKAT, CL_DISTANCE, CL_AZIMUTH, CL_ELEVATION)],
        title=TITLE, subtitle=subtitle, readouts=CL_READOUTS,
        panel_images=lambda tick: panels.entries(), n_panels=4,
        phase_name=lambda phase: phase,
        title_seconds=lead_seconds, min_span=CL_MIN_SPAN, keep_in_frame=arm_points,
        overrun_tol=0.0, trace_seconds=1e6,
    ) as recorder:
        result = recorder.run(motion(), out, still_times=still_times, stills_dir=stills_dir)

    truth.close()
    if state["second"] is not None:
        state["second"].close()
    return result, state["checks"], (d_xy, d_yaw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_campaign_video.py")
    parser.add_argument("--out", default="media_out/hd/"
                        "brick_builder_arm_robot_two_brick_build_1080p.mp4")
    parser.add_argument("--quality", default="hd1080", choices=("draft", "hd1080"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lead", type=float, default=2.0)
    parser.add_argument("--gap", type=float, default=1.5,
                        help="seconds held between the two cycles")
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--stills", type=int, default=6)
    parser.add_argument("--stills-dir", default=None)
    parser.add_argument("--cnn-drives", action="store_true")
    parser.add_argument("--inverse", choices=("analytic", "learned"), default="analytic")
    parser.add_argument("--weights", default=None, help="a learned inverse checkpoint")
    parser.add_argument("--version", default="", help="version id for the spec line")
    args = parser.parse_args(argv)

    out = Path(args.out)
    stills_dir = Path(args.stills_dir) if args.stills_dir else out.parent / "stills"
    result, checks, disp = build_video(
        out, quality_name=args.quality, seed=args.seed, lead_seconds=args.lead,
        gap_seconds=args.gap, hold_seconds=args.hold, stills=args.stills,
        stills_dir=stills_dir if args.stills else None, cnn_drives=args.cnn_drives,
        inverse=args.inverse, weights=Path(args.weights) if args.weights else None,
        version=args.version)
    print(f"master {result.master} {result.frames} frames {result.seconds:.1f}s "
          f"{result.width}x{result.height}")
    print(f"baseplate offset {1e3 * disp[0][0]:+.2f}, {1e3 * disp[0][1]:+.2f} mm, "
          f"{math.degrees(disp[1]):+.2f} deg")
    ok = True
    for i, check in enumerate(checks):
        print(f"brick {i}: seated={check.seated} at_cell={check.at_target_cell} "
              f"lateral={check.lateral_mm:.2f} mm depth_error={check.depth_error_mm:.2f} mm")
        ok &= bool(check.seated and check.at_target_cell)
    print("RESULT: built" if ok else "RESULT: not built")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
