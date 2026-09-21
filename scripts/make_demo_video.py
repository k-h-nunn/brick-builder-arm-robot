"""Film one closed-loop pick and place at 1920x1080, in one continuous take.

Usage:

    python scripts/make_demo_video.py --cnn-drives
    python scripts/make_demo_video.py --pick --out media_out/hd/brick_builder_arm_robot_closed_loop_pick_and_place_1080p.mp4

The motion comes from `brick_builder.action`: a reference sequence recorded in a nominal
world, tracked in a world nobody described to the controller. The layout is four square
panels (the arm camera, the scene model's labels, the target frame's labels, the
renderer's labels) and a data strip. Every string on screen is a title with its units; the
old scripted film and its narration are gone.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brick_builder.control.pick_place import (
    CONTROL_DT,
    PickPlace,
    PickPlaceConfig,
    PlacementCheck,
    cell_world,
    gripper_press_force,
    make_scene,
    verify_placement,
)
from brick_builder.robot.patch_so101 import WRIST_CAM
from brick_builder.demo.cinema import (
    CamKey,
    Film,
    ReadoutSpec,
    Tick,
    arm_points,
    panel_size_for,
    quality_by_name,
    studio_look,
)
from brick_builder.robot.patch_so101 import WRIST_CAM, gripper_angle_for_width
from brick_builder.scene.compose import Scene, reset_scene

TITLE = "Brick Builder Arm Robot"
SUBTITLE = "pick and place in simulation"
CAPTION = ("scripted controller, stock SO-101 gripper,\n"
           "simulated DUPLO press fit")
"""The line break is deliberate: it falls where the sense falls, and it keeps the caption box
inside the left column of the frame instead of reaching across the arm."""

CAPTION_PHASES = frozenset({"home", "to_tray", "open"})
"""The phases the caption is up for: the opening, until the arm arrives over the tray.

It says what the viewer is watching and none of it changes during the take, so once the arm is
at work it is three lines of dead type sitting on the picture.  In the first cut it stayed up to
the last frame and covered the upper arm and the wrist during the closing hold on the seated
brick, which is the one shot in the film with nothing else in it."""

BRICK = "tray_brick_0"
"""Which of the two spare bricks in the tray the arm picks."""

TARGET = (3.5, 3.5, 0)
"""Where it goes: the middle of the baseplate, bottom layer.  A 2-wide footprint straddles two
stud columns, which is why the cell coordinates are halves."""

# ================================================================== the closed-loop film ===
#
# The second thing this script can film: one closed-loop placement driven by `brick_builder.
# action`'s ideal-sequence tracking. Same toolkit; what changes is where the motion comes from
# and what the panels show.

CL_TITLE = "Brick Builder Arm Robot"

PANEL_TITLES = ("Arm camera POV", "Observed scene state",
                "Target scene state", "Ground truth segmentation")

CL_READOUTS = (
    ReadoutSpec(key="err_mm", label="Scene-state error [mm]", unit="", lo=0.0, hi=None,
                fmt="{:.2f}", reference=0.4, plot=True),
    ReadoutSpec(key="miou", label="Segmentation mIoU [%]", unit="", lo=0.0, hi=100.0,
                fmt="{:.0f}"),
    ReadoutSpec(key="predict_mm", label="Prediction error [mm]", unit="", lo=0.0, hi=5.0,
                fmt="{:.2f}"),
    ReadoutSpec(key="press_n", label="Press force [N]", unit="", lo=0.0, hi=20.0, fmt="{:.1f}"),
    ReadoutSpec(key="depth_mm", label="Insertion depth [mm]", unit="", lo=0.0, hi=5.0,
                fmt="{:.2f}"),
)

# Aimed into the part of the frame nothing is drawn over, and far enough from the top left that
# the step label never lands on the arm: measured, the whole arm projects to (539, 138)-(1003,
# 587) with the clear box at (0, 0)-(1037, 618). The subject spans 24 % of the frame width,
# against 55 % in the old single-inset layout. That is what four square feature-map panels and
# a full-width plot cost, and it is the layout the work is being judged on.
CL_LOOKAT = (-0.06, 0.14, 0.06)
CL_DISTANCE = 1.00
CL_AZIMUTH = 268.0
CL_ELEVATION = -24.0
CL_MIN_SPAN = 0.21


def spec_line(*, driving: str, displayed: str, offset_mm: float, offset_deg: float,
              pick: bool) -> str:
    """The one line under the title. Titles and values, no sentences.

    When the same model both drives the loop and is shown in the panels there is one field, not
    two: naming it twice would read as two different things.
    """
    job = "Closed-loop pick and place" if pick else "Closed-loop placement"
    model = (f"scene model: {driving}" if driving == displayed
             else f"control input: {driving} | panel: {displayed}")
    return (f"{job} | {model} | controller: ideal-sequence tracking, 10 Hz | "
            f"brick: DUPLO 2x2 | baseplate offset: {offset_mm:.1f} mm, {offset_deg:.1f} deg")


DARK = (20, 22, 26)
ARM_TINT = (176, 158, 106)
PLATE_TINT = (62, 150, 96)
"""One palette for all three label panels. Side by side they are only comparable if the same
thing is the same colour in each: background and table dark, the arm tan, the baseplate green,
and every brick painted in its own measured colour."""


def paint_labels(class_map, instance_map, rgb) -> np.ndarray:
    """One label image, in the convention all three panels share.

    `class_map` says background / baseplate / arm / brick per pixel; `instance_map` separates one
    brick from the next. Bricks take the mean colour of their own pixels in `rgb`, lifted towards
    white so a dark brick is still visible -- so the same brick is the same colour whether the
    labels came from the network, from the reference sequence or from the renderer.
    """
    from brick_builder.perception.cnn.labels import ARM_CLASS, BASEPLATE_CLASS, BRICK_CLASS

    class_map = np.asarray(class_map)
    instance_map = np.asarray(instance_map)
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.empty((*class_map.shape, 3), np.uint8)
    out[:] = DARK
    out[class_map == ARM_CLASS] = ARM_TINT
    out[class_map == BASEPLATE_CLASS] = PLATE_TINT
    bricks = class_map == BRICK_CLASS
    for i in np.unique(instance_map[bricks]):
        mask = bricks & (instance_map == i)
        if not mask.any():
            continue
        mean = rgb[mask].mean(axis=0) / 255.0
        out[mask] = np.clip(60.0 + 195.0 * mean, 0, 255).astype(np.uint8)
    return out


def _recording_labeller(cfg, prepare, seed):
    """A `Labeller` for the recording world, built on its first frame.

    `record_ideal` owns that world and only ever hands it out through this hook, so the renderer
    is made the first time it is needed and kept for the rest of the recording.
    """
    box: dict = {}

    def hook(scene):
        if "labeller" not in box:
            box["labeller"] = Labeller(scene)
        return box["labeller"](scene)

    return hook


class Labeller:
    """The renderer's own labels for one world, in the convention the accuracy is measured in."""

    def __init__(self, scene):
        from brick_builder.action.adapter import WRIST_FULL, WRIST_RENDER
        from brick_builder.perception.cnn.data import perception_instance_lut
        from brick_builder.render.renderer import SceneRenderer

        self.window = WRIST_FULL
        self.renderer = SceneRenderer(scene.model, width=WRIST_RENDER[0], height=WRIST_RENDER[1])
        self.renderer.instance_lut(perception_instance_lut(scene))

    def __call__(self, scene) -> dict:
        from brick_builder.perception.cnn.labels import class_map_from_instance
        from brick_builder.perception.views import render_view

        frame, _ = render_view(scene, self.window, self.renderer)
        instance = np.asarray(frame.instance)
        return {"instance": instance, "class": class_map_from_instance(instance),
                "rgb": np.asarray(frame.rgb)}

    def close(self):
        self.renderer.close()


class Panels:
    """The four pictures, rebuilt on every vision tick and held between them."""

    def __init__(self, size, cnn=None):
        self.w, self.h = int(size[0]), int(size[1])
        self.cnn = cnn
        blank = np.zeros((self.h, self.w, 3), np.uint8)
        blank[:] = DARK
        self.images = [blank.copy() for _ in PANEL_TITLES]
        self.miou = float("nan")

    def _fit(self, img, nearest: bool) -> np.ndarray:
        import cv2

        mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(np.asarray(img), (self.w, self.h), interpolation=mode)

    def update(self, rgb, target_label, truth_label, *, window=None) -> None:
        """`target_label` and `truth_label` are `Labeller` outputs; `rgb` is this frame's
        camera image, which the network sees and the brick colours are measured from."""
        self.images[0] = self._fit(rgb, nearest=False)

        if self.cnn is not None:
            maps = self.cnn.feature_maps(np.asarray(rgb), window=window)
            shown = paint_labels(maps["class_map"], maps["instance_map"], rgb)
            if truth_label is not None:
                try:
                    self.miou = 100.0 * float(self.cnn.accuracy(
                        maps["instance_map"], truth_label["instance"],
                        truth_label["class"])["miou"])
                except Exception:
                    self.miou = float("nan")
        else:
            shown = (paint_labels(truth_label["class"], truth_label["instance"], rgb)
                     if truth_label is not None else self.images[1])
        self.images[1] = self._fit(shown, nearest=True)

        if target_label is not None:
            self.images[2] = self._fit(
                paint_labels(target_label["class"], target_label["instance"],
                             target_label["rgb"]), nearest=True)
        if truth_label is not None:
            self.images[3] = self._fit(
                paint_labels(truth_label["class"], truth_label["instance"], rgb), nearest=True)

    def entries(self):
        return list(zip(PANEL_TITLES, self.images))


def closed_loop_motion(scene, episode, panels, truth, brick, *, lead_seconds=2.0,
                       hold_seconds=3.0):
    """`brick_builder.action`'s per-tick records, turned into ticks the film understands."""
    from brick_builder.control.pick_place import gripper_press_force

    last: dict = {}
    hold_tcp = episode.world.feedback(episode.ideal.frames[0].commands[0].grip).tcp_pos.copy()
    grip0 = episode.ideal.frames[0].commands[0].grip
    # Fill the panels before the first frame is written. They are otherwise four black squares
    # for the whole of the lead-in, which reads as four broken panels rather than as a title
    # card, and the camera is already looking at the scene.
    state, rgb = episode.world.observe_rgb()
    panels.update(rgb, episode.ideal.frames[0].label, truth(scene), window=truth.window)
    if panels.miou == panels.miou:
        last["miou"] = panels.miou
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
    park_spin = park.spin + episode.correction[2]
    park_grip = park.grip
    for _ in range(max(int(round(hold_seconds / CONTROL_DT)), 1)):
        episode.world.command(park_pos, park_spin, park_grip)
        yield Tick(scene=scene, phase="Verify", data=dict(last))


def closed_loop_film(out: Path, *, quality_name: str = "hd1080", seed: int = 1,
                     lead_seconds: float = 2.0, hold_seconds: float = 3.0,
                     stills: int = 6, stills_dir: Path | None = None, pick: bool = False,
                     cnn_panels: bool = True, cnn_drives: bool = False):
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import SUPPLY_BODY, build_pick_world, build_world
    from brick_builder.control.pick_place import verify_placement

    quality = quality_by_name(quality_name)
    cfg = PlaceConfig(pick=pick)
    prepare = lambda model: studio_look(model, quality)   # noqa: E731

    # The reference sequence keeps a label image per frame, rendered through the same look-up
    # table the accuracy is measured with, so "Target scene state" is drawn in the same
    # convention as the other two -- arm and all.
    ideal, nominal = record_ideal(cfg, seed=seed, prepare=prepare, build_decode=False,
                                  label_hook=_recording_labeller(cfg, prepare, seed))
    nominal.close()

    if pick:
        scene, d_xy, d_yaw, _, _ = build_pick_world(seed)
        brick = SUPPLY_BODY
    else:
        scene, d_xy, d_yaw = build_world(seed)
        brick = scene.held_name
    studio_look(scene.model, quality)
    source = None
    driving = "ground-truth scene model"
    if cnn_drives:
        source = _cnn()
        driving = f"CNN, frozen {FROZEN_SHA}"
    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, twin_prepare=prepare,
                              build_decode=True, source=source)
    truth = Labeller(scene)
    cnn = _cnn() if (cnn_panels and not cnn_drives) else source
    panels = Panels(panel_size_for(quality, 4), cnn=cnn)

    length = lead_seconds + len(ideal) * 5 * CONTROL_DT + hold_seconds
    still_times = list(np.linspace(0.05 * length, 0.97 * length, stills)) if stills else []
    subtitle = spec_line(driving=driving,
                         displayed=(f"CNN, frozen {FROZEN_SHA}" if cnn is not None
                                    else "ground-truth scene model"),
                         offset_mm=1e3 * float(np.linalg.norm(d_xy)),
                         offset_deg=abs(math.degrees(d_yaw)), pick=pick)

    with Film(
        scene, quality=quality,
        shot=[CamKey(0.0, CL_LOOKAT, CL_DISTANCE, CL_AZIMUTH, CL_ELEVATION)],
        title=CL_TITLE, subtitle=subtitle, readouts=CL_READOUTS,
        panel_images=lambda tick: panels.entries(), n_panels=4,
        phase_name=lambda phase: phase,
        title_seconds=lead_seconds, min_span=CL_MIN_SPAN, keep_in_frame=arm_points,
        overrun_tol=0.0, trace_seconds=1e6,
    ) as f:
        result = f.run(closed_loop_motion(scene, episode, panels, truth, brick,
                                          lead_seconds=lead_seconds, hold_seconds=hold_seconds),
                       out, still_times=still_times, stills_dir=stills_dir)
    check = verify_placement(scene, brick, (4.5, 3.5, 0))
    truth.close()
    episode.close()
    return result, episode, check, (d_xy, d_yaw)


FROZEN = Path(__file__).resolve().parents[1] / "results" / "film" / "vision_model_frozen.pt"
FROZEN_SHA = "3e07a382"


def _cnn():
    from brick_builder.perception.cnn.source import CnnSceneModel

    return CnnSceneModel(checkpoint=FROZEN) if FROZEN.exists() else CnnSceneModel()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_demo_video.py")
    parser.add_argument("--out", default="media_out/hd/"
                        "brick_builder_arm_robot_closed_loop_placement_1080p.mp4")
    parser.add_argument("--quality", default="hd1080", choices=("draft", "hd1080"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lead", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--stills", type=int, default=6)
    parser.add_argument("--stills-dir", default=None)
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--cnn-drives", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    stills_dir = Path(args.stills_dir) if args.stills_dir else out.parent / "stills"
    result, episode, check, disp = closed_loop_film(
        out, quality_name=args.quality, seed=args.seed, lead_seconds=args.lead,
        hold_seconds=args.hold, stills=args.stills,
        stills_dir=stills_dir if args.stills else None, pick=args.pick,
        cnn_drives=args.cnn_drives,
    )
    print(f"master {result.master} {result.frames} frames {result.seconds:.1f}s "
          f"{result.width}x{result.height}")
    print(f"baseplate offset {1e3 * disp[0][0]:+.2f}, {1e3 * disp[0][1]:+.2f} mm, "
          f"{math.degrees(disp[1]):+.2f} deg")
    for r in episode.reports:
        print(f"  {r.name:24s} {r.vision_ticks:4d} ticks  final {r.final_err_mm:6.2f} mm")
    print(f"placement seated={check.seated} at_cell={check.at_target_cell} "
          f"lateral={check.lateral_mm:.2f} mm depth_error={check.depth_error_mm:.2f} mm")
    return 0 if check.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
