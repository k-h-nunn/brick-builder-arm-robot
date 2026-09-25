#!/usr/bin/env python3
"""Record a two-brick build at 1920x1080, in one continuous take.

    python scripts/make_campaign_video.py --task stack_2 --seed 0
    python scripts/make_campaign_video.py --task stack_2 --seed 0 --cnn-drives --targets truth \
        --probe
    python scripts/make_campaign_video.py --qa-only

One English command, two bricks, one simulator: the arm places the brick it is holding on the
cell the command names, then fetches a second brick off the table and places it where the
command's second step puts it -- on top of the first for `stack_2`, beside it for `build_2`.
Both cycles run in the same world and the same recording, so nothing is cut together.

`--probe` runs the same world, the same recordings and the same control ticks without rendering
the take, and reports whether both bricks seated: the seed is chosen with it before the one
1080p recording is made.

The layout is the closed-loop video's: four square panels (the arm camera, the target frame's
labels, the scene model's labels and the renderer's labels, the last two side by side) and the
data strip, plus one title line under the panels carrying the English command. After the take
the QA checks run on the file and on the stills: frame shape, no two text boxes intersecting,
the smallest font, no code identifiers on screen, step names from the list, every stud coloured
like its brick, and a main view that is a render rather than noise; the text areas are cropped
at 100 % beside the stills.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from itertools import combinations
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
from brick_builder.demo.video_panels import (  # noqa: E402
    BOX, LABEL_FRAC, TEXT, LayoutError, Rect, check_layout, fit_caption,
)
from make_demo_video import (  # noqa: E402
    CL_AZIMUTH, CL_DISTANCE, CL_ELEVATION, CL_LOOKAT, CL_MIN_SPAN, CL_READOUTS, CL_TITLE,
    FROZEN_SHA, PANEL_TITLES, Labeller, Panels, _cnn, _recording_labeller,
)

TITLE = CL_TITLE

JOBS = {"stack_2": "Two-brick stack", "build_2": "Two-brick build, side by side"}
"""The title each task goes under on the title card."""

PANEL_ORDER = ("Arm camera POV", "Target scene state",
               "Observed scene state", "Ground truth segmentation")
"""Top row: what the camera sees and what the target sequence wants. Bottom row: the scene
model's dense output beside the renderer's ground truth it is scored against."""

STEP_NAMES = frozenset({"Approach", "Grasp", "Lift", "Transport", "Align",
                        "Press the brick down", "Release", "Retract", "Verify"})
MAX_MAIN_VIEW_GRADIENT = 5.0
"""Mean absolute difference between horizontally adjacent pixels of the main view [0-255]. A
clean render of the studio measures 0.9-2.7 on the closed-loop stills; frames read from a freed
framebuffer measured 7.0-53.6."""

MIN_FONT_PX = 22
"""Smallest glyph size any on-screen string may use at 1080 lines [px]."""

IDENTIFIER = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b|\w\(|\.py\b|\b[a-z]+[A-Z][A-Za-z]*\b")
"""snake_case, a call, a file name, camelCase: what a code identifier looks like on screen."""
NOT_IDENTIFIERS = frozenset({"mIoU"})


def spec_line(*, job: str, driving: str, displayed: str, inverse: str, version: str,
              offset_mm: float, offset_deg: float) -> str:
    """The one line under the title. Titles and values, no sentences.

    Job, perception source, controller, bricks, and the world's offset; the command has its own
    title line under the panels, which is on screen with the title card and after it.
    """
    model = (f"scene model: {driving}" if driving == displayed
             else f"control: {driving} | panel: {displayed}")
    tag = f"{inverse} inverse" + (f" {version}" if version else "")
    return (f"{job} | {model} | target-sequence tracking, 10 Hz | {tag} | 2 x DUPLO 2x2 | "
            f"baseplate offset: {offset_mm:.1f} mm, {offset_deg:.1f} deg")


def command_lines(command: str, measure, max_px: float) -> list[str]:
    """Command title line, split at its "then" when it needs two lines."""
    text = f"Command: {command}"
    if measure(text) <= max_px:
        return [text]
    head, sep, tail = command.partition(" then ")
    if sep:
        lines = [f"Command: {head}", f"then {tail}"]
        if max(measure(line) for line in lines) <= max_px:
            return lines
    return fit_caption(text, measure, max_px, max_lines=2)


class CommandRecorder(Recorder):
    """Closed-loop video recorder with the English command under the panel grid."""

    def __init__(self, scene, *, command: str, **kw):
        from PIL import ImageFont

        super().__init__(scene, **kw)
        ov = self.overlay
        font_dir = Path(ov.f_label.path).parent
        self.command_font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"),
                                               ov._px(LABEL_FRAC))
        grid = ov.panels_rect(self.n_panels)
        self.command_lines = command_lines(command, self.command_font.getlength,
                                           grid.width - 2 * ov.pad)
        self._line_h = ov._line_height(self.command_font)
        y0 = grid.y1 + ov.panel_gap
        self.command_rect = Rect(grid.x0, y0, grid.x1,
                                 y0 + len(self.command_lines) * self._line_h + 2 * ov.pad)
        if not self.command_rect.inside(ov.safe):
            raise LayoutError(f"the command line {self.command_rect.as_tuple()} leaves the safe "
                              f"area {ov.safe.as_tuple()}")

    def font_sizes(self) -> dict[str, int]:
        ov = self.overlay
        return {"title": ov.f_title.size, "spec line": ov.f_subtitle.size,
                "step": ov.f_phase.size, "value": ov.f_value.size, "number": ov.f_number.size,
                "label": ov.f_label.size, "panel title": ov.f_small.size,
                "command": self.command_font.size}

    def compose(self, tick, video_t):
        from PIL import Image, ImageDraw

        composed, rects, span, overrun, ray = super().compose(tick, video_t)
        ov = self.overlay
        base = Image.fromarray(np.ascontiguousarray(composed), mode="RGB").convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        r = self.command_rect
        draw.rounded_rectangle(r.as_tuple(), radius=ov.radius, fill=BOX)
        y = r.y0 + ov.pad
        for line in self.command_lines:
            draw.text((r.x0 + ov.pad, y), line, font=self.command_font, fill=TEXT)
            y += self._line_h
        out = np.asarray(Image.alpha_composite(base, layer).convert("RGB"), dtype=np.uint8)
        rects = dict(rects)
        rects["command"] = r
        check_layout(rects, ov.safe)
        return out, rects, span, overrun, ray


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
                 hold_seconds: float, last: dict, on_hold=None):
    """One cycle's per-tick records, turned into ticks the video understands.

    With `panels` set to `None` nothing is drawn, and every control tick, every observation the
    loop makes and every physics step is the same as with it: that is the probe.
    """
    from brick_builder.control.pick_place import gripper_press_force

    hold_tcp = episode.world.feedback(episode.ideal.frames[0].commands[0].grip).tcp_pos.copy()
    grip0 = episode.ideal.frames[0].commands[0].grip
    state, rgb = episode.world.observe_rgb()
    if panels is not None:
        panels.update(rgb, episode.ideal.frames[0].label, truth(scene), window=truth.window)
        if panels.miou == panels.miou:
            last["miou"] = panels.miou
    del state
    for _ in range(max(int(round(lead_seconds / CONTROL_DT)), 1)):
        episode.world.command(hold_tcp, 0.0, grip0)
        yield Tick(scene=scene, phase=episode.ideal.frames[0].step, data=dict(last))

    for rec in episode.run():
        if panels is not None and rec.vision and rec.state is not None:
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

    if on_hold is not None:
        on_hold()
    park = episode.ideal.frames[-1].commands[-1]
    park_pos = np.array([park.pos[0] + episode.correction[0],
                         park.pos[1] + episode.correction[1], park.pos[2]])
    for _ in range(max(int(round(hold_seconds / CONTROL_DT)), 1)):
        episode.world.command(park_pos, park.spin + episode.correction[2], park.grip)
        yield Tick(scene=scene, phase="Verify", data=dict(last))


class DisplayPanels(Panels):
    """The four panels in `PANEL_ORDER`: scene model output beside ground truth."""

    def entries(self):
        by_title = dict(zip(PANEL_TITLES, self.images))
        return [(title, by_title[title]) for title in PANEL_ORDER]


def still_picker(state: dict, lead_seconds: float, hold_seconds: float):
    """Named stills at key moments of the build, timed from the run itself.

    `(name, cycle, step, seconds into that step)`; the last one is taken in the closing hold.
    """
    wanted = [("1_title", 0, None, 0.6 * lead_seconds),
              ("2_first_press", 0, "Press the brick down", 0.4),
              ("3_second_grasp", 1, "Grasp", 0.3),
              ("4_second_align", 1, "Align", 2.0),
              ("5_second_press", 1, "Press the brick down", 0.4),
              ("6_verify", 1, "hold", max(hold_seconds - 0.4, 0.0))]
    started: dict = {}

    def pick(tick, video_t):
        state["phases"].add(tick.phase)
        cycle = state["cycle"]
        step = "hold" if state.get("holding") and cycle == 1 else tick.phase
        started.setdefault((cycle, step), video_t)
        started.setdefault((cycle, None), video_t)
        for item in list(wanted):
            name, c, s, after = item
            if c == cycle and (cycle, s) in started and video_t - started[(cycle, s)] >= after:
                wanted.remove(item)
                return name
        return None

    return pick


def build_video(out: Path | None, *, task: str = "stack_2", quality_name: str = "hd1080",
                seed: int = 0, lead_seconds: float = 2.0, gap_seconds: float = 1.5,
                hold_seconds: float = 3.0, stills_dir: Path | None = None,
                cnn_drives: bool = False, inverse: str = "analytic",
                weights: Path | None = None, version: str = "", probe: bool = False,
                targets: str = "driving"):
    """Both cycles of the build in one world: recorded, or run headless with `probe`."""
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import REF_RED, REF_YELLOW, SUPPLY_BODY
    from brick_builder.campaign.tasks import TASKS, cell_layer

    spec = TASKS[task]
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
    placed_two = [REF_YELLOW, REF_RED, spec.placement(0)]

    # `targets="driving"` records the target sequences through the scene model that drives the
    # run, as the campaign's `--source cnn` does, so target and observation are in the same
    # representation. `"truth"` records them with the simulator's own scene model, as the
    # closed-loop placement and pick-and-place videos do.
    target_source = source if targets == "driving" else None
    ideal_one, nominal_one = record_ideal(cfg_one, seed=seed, target=cells[0], prepare=prepare,
                                          source=target_source,
                                          label_hook=_recording_labeller(cfg_one, prepare, seed))
    nominal_one.close()
    ideal_two, nominal_two = record_ideal(cfg_two, seed=seed, target=cells[1], prepare=prepare,
                                          source=target_source, placed=placed_two,
                                          label_hook=_recording_labeller(cfg_two, prepare, seed))
    nominal_two.close()

    scene, d_xy, d_yaw = build_scene_for(seed, prepare)
    studio_look(scene.model, quality)
    truth = panels = None
    displayed = driving
    if not probe:
        truth = Labeller(scene)
        # The panels get their own copy of the scene model. The one that drives keeps a slot
        # tracker across frames, and a display that fed it extra observations would make the
        # take a different run from the probe that chose its seed.
        panels = DisplayPanels(panel_size_for(quality, 4), cnn=_cnn())
        displayed = f"CNN {FROZEN_SHA}"

    g = None
    if inverse == "learned":
        from brick_builder.action.models.inverse import LearnedInverse

        g = LearnedInverse.load(weights)

    first = TrackingEpisode(scene, ideal_one, cfg=cfg_one, seed=seed, target=cells[0],
                            twin_prepare=prepare, build_decode=True, source=source, inverse=g)
    held_name = scene.held_name
    subtitle = spec_line(
        job=JOBS[task], driving=driving, displayed=displayed, inverse=inverse, version=version,
        offset_mm=1e3 * float(np.linalg.norm(d_xy)), offset_deg=abs(math.degrees(d_yaw)))

    state: dict = {"second": None, "checks": [], "cycle": 0, "holding": False,
                   "phases": set(), "episodes": []}

    def centre(cell):
        return (cell[0] + 0.5, cell[1] + 0.5, cell_layer(cell))

    def motion():
        last: dict = {}
        yield from cycle_motion(scene, first, panels, truth, held_name,
                                lead_seconds=lead_seconds, hold_seconds=gap_seconds, last=last)
        state["checks"].append(verify_placement(scene, held_name, centre(cells[0])))
        state["episodes"].append(first)
        # `first` is closed after the take, not here: `SceneRenderer.close` frees its GL objects
        # in whichever context is current, which mid-take is the recorder's, and the main view
        # then reads a freed framebuffer for the rest of the recording.
        state["cycle"] = 1
        second = TrackingEpisode(scene, ideal_two, cfg=cfg_two, seed=seed, target=cells[1],
                                 twin_prepare=prepare, build_decode=True, source=source,
                                 inverse=g, placed=placed_two)
        state["second"] = second
        yield from cycle_motion(scene, second, panels, truth, SUPPLY_BODY,
                                lead_seconds=0.6, hold_seconds=hold_seconds, last=last,
                                on_hold=lambda: state.update(holding=True))
        state["checks"].append(verify_placement(scene, SUPPLY_BODY, centre(cells[1])))
        state["episodes"].append(second)

    result = None
    info: dict = {"title": TITLE, "subtitle": subtitle, "command": spec.command}
    if probe:
        for tick in motion():
            state["phases"].add(tick.phase)
    else:
        picker = still_picker(state, lead_seconds, hold_seconds)
        with CommandRecorder(
            scene, command=spec.command, quality=quality,
            shot=[CamKey(0.0, CL_LOOKAT, CL_DISTANCE, CL_AZIMUTH, CL_ELEVATION)],
            title=TITLE, subtitle=subtitle, readouts=CL_READOUTS,
            panel_images=lambda tick: panels.entries(), n_panels=4,
            phase_name=lambda phase: phase,
            title_seconds=lead_seconds, min_span=CL_MIN_SPAN, keep_in_frame=arm_points,
            overrun_tol=0.0, trace_seconds=1e6,
        ) as recorder:
            result = recorder.run(motion(), out, still_times=(), stills_dir=stills_dir,
                                  still_when=picker)
            info.update(command_lines=recorder.command_lines, fonts=recorder.font_sizes(),
                        panel_bar=recorder.overlay._panel_bar(),
                        safe=recorder.overlay.safe.as_tuple(),
                        clear=recorder.overlay.clear_rect().as_tuple())
        truth.close()
    for episode in state["episodes"] or [first]:
        episode.close()
    info["cycles"] = [{"failure": e.failure or "",
                       "steps": [(r.name, int(r.vision_ticks), float(r.final_err_mm))
                                 for r in e.reports]} for e in state["episodes"]]
    info.update(phases=sorted(state["phases"]),
                panel_titles=list(PANEL_ORDER), readouts=[r.label for r in CL_READOUTS],
                studs=stud_colour_problems(scene))
    return result, state["checks"], (d_xy, d_yaw), info


# ------------------------------------------------------------------------------------------ QA ---


def stud_colour_problems(scene) -> list[str]:
    """Bricks and baseplate whose visual geoms, studs included, are not one colour."""
    model = scene.model
    problems = []
    for name in list(scene.bricks) + [scene.plate.name]:
        if "_collet_" in name:
            continue
        body = model.body(name).id
        colours = set()
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body or model.geom_contype[gid] != 0:
                continue
            mat = int(model.geom_matid[gid])
            rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[gid]
            colours.add(tuple(np.round(np.asarray(rgba, dtype=float), 4)))
        if len(colours) != 1:
            problems.append(f"{name}: {len(colours)} colours among its visual geoms")
    return problems


def screen_strings(info: dict) -> list[str]:
    return ([info["title"], info["subtitle"]] + list(info.get("command_lines", []))
            + list(info["phases"]) + list(info["panel_titles"]) + list(info["readouts"]))


def run_qa(video: Path, stills_dir: Path) -> list[str]:
    """QA checks on the finished file and its stills; 100 % crops of every text area."""
    import imageio.v2 as imageio

    manifest = json.loads((stills_dir / "manifest.json").read_text())
    info = manifest["info"]
    failures: list[str] = []

    reader = imageio.get_reader(str(video))
    try:
        size = tuple(reader.get_meta_data()["size"])
        first = reader.get_data(0)
    finally:
        reader.close()
    if size != (1920, 1080) or first.shape != (1080, 1920, 3):
        failures.append(f"video frame is {size}, {first.shape}; wanted 1920x1080 RGB")

    safe = Rect(*info["safe"])
    crops = stills_dir / "crops"
    crops.mkdir(exist_ok=True)
    for still in manifest["stills"]:
        path = Path(still["path"])
        img = imageio.imread(str(path))
        if img.shape[:2] != (1080, 1920):
            failures.append(f"{path.name} is {img.shape}")
        rects = {k: Rect(*v) for k, v in still["rects"].items()}
        clear = Rect(*info["clear"])
        view = img[clear.y0:clear.y1, clear.x0:clear.x1].astype(np.int16)
        gradient = float(np.mean(np.abs(np.diff(view, axis=1))))
        if gradient > MAX_MAIN_VIEW_GRADIENT:
            failures.append(f"{path.name}: main view gradient {gradient:.1f}, over "
                            f"{MAX_MAIN_VIEW_GRADIENT}: the render is corrupt")
        for (a, ra), (b, rb) in combinations(rects.items(), 2):
            if ra.intersects(rb):
                failures.append(f"{path.name}: {a} {ra.as_tuple()} intersects {b} "
                                f"{rb.as_tuple()}")
        try:
            check_layout(rects, safe)
        except LayoutError as exc:
            failures.append(f"{path.name}: {exc}")
        for key, r in rects.items():
            if key.startswith("panel_"):
                if still is not manifest["stills"][0]:
                    continue
                r = Rect(r.x0, r.y0, r.x1, r.y0 + int(info["panel_bar"]))
            imageio.imwrite(str(crops / f"{path.stem}_{key}.png"), img[r.y0:r.y1, r.x0:r.x1])

    small = {k: v for k, v in info["fonts"].items() if v < MIN_FONT_PX}
    if small:
        failures.append(f"fonts under {MIN_FONT_PX} px: {small}")
    for text in screen_strings(info):
        for match in IDENTIFIER.findall(text):
            if match not in NOT_IDENTIFIERS:
                failures.append(f"code identifier on screen: {match!r} in {text!r}")
    odd = sorted(set(info["phases"]) - STEP_NAMES)
    if odd:
        failures.append(f"step names outside the list: {odd}")
    failures += [f"stud colour: {p}" for p in info["studs"]]
    return failures


def write_manifest(stills_dir: Path, result, info: dict, checks) -> None:
    stills = [{"path": str(p), "rects": {k: list(r.as_tuple()) for k, r in rects.items()}}
              for p, rects in zip(result.stills, result.still_rects)]
    bricks = [{"seated": bool(c.seated), "at_cell": bool(c.at_target_cell),
               "at_layer": bool(c.at_target_layer), "lateral_mm": float(c.lateral_mm),
               "depth_error_mm": float(c.depth_error_mm),
               "height_error_mm": float(c.height_error_mm),
               "lower_bodies": list(c.lower_bodies)} for c in checks]
    (stills_dir / "manifest.json").write_text(json.dumps(
        {"video": str(result.master), "frames": result.frames, "seconds": result.seconds,
         "width": result.width, "height": result.height, "stills": stills, "info": info,
         "bricks": bricks}, indent=1, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_campaign_video.py")
    parser.add_argument("--out", default="media_out/hd/"
                        "brick_builder_arm_robot_two_brick_build_1080p.mp4")
    parser.add_argument("--task", choices=tuple(JOBS), default="stack_2")
    parser.add_argument("--quality", default="hd1080", choices=("draft", "hd1080"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lead", type=float, default=2.0)
    parser.add_argument("--gap", type=float, default=1.5,
                        help="seconds held between the two cycles")
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--stills-dir", default="media_out/hd/two_brick_build_stills")
    parser.add_argument("--cnn-drives", action="store_true")
    parser.add_argument("--inverse", choices=("analytic", "learned"), default="analytic")
    parser.add_argument("--weights", default=None, help="a learned inverse checkpoint")
    parser.add_argument("--version", default="", help="version id for the spec line")
    parser.add_argument("--targets", choices=("driving", "truth"), default="driving",
                        help="scene model the target sequences are recorded through")
    parser.add_argument("--probe", action="store_true",
                        help="run the take's control without rendering it, and score it")
    parser.add_argument("--qa-only", action="store_true",
                        help="re-run the checks on an existing take and its stills")
    args = parser.parse_args(argv)

    out = Path(args.out)
    stills_dir = Path(args.stills_dir)
    if args.qa_only:
        failures = run_qa(out, stills_dir)
        for line in failures:
            print(f"QA FAIL {line}")
        print("QA: pass" if not failures else f"QA: {len(failures)} failure(s)")
        return 1 if failures else 0

    if not args.probe:
        stills_dir.mkdir(parents=True, exist_ok=True)
    result, checks, disp, info = build_video(
        None if args.probe else out, task=args.task, quality_name=args.quality,
        seed=args.seed, lead_seconds=args.lead, gap_seconds=args.gap, hold_seconds=args.hold,
        stills_dir=None if args.probe else stills_dir, cnn_drives=args.cnn_drives,
        inverse=args.inverse, weights=Path(args.weights) if args.weights else None,
        version=args.version, probe=args.probe, targets=args.targets)
    if result is not None:
        print(f"master {result.master} {result.frames} frames {result.seconds:.1f}s "
              f"{result.width}x{result.height}")
    print(f"task {args.task} seed {args.seed} "
          f"control {'CNN ' + FROZEN_SHA if args.cnn_drives else 'ground truth'}")
    print(f"baseplate offset {1e3 * disp[0][0]:+.2f}, {1e3 * disp[0][1]:+.2f} mm, "
          f"{math.degrees(disp[1]):+.2f} deg")
    ok = len(checks) == 2
    for i, check in enumerate(checks):
        print(f"brick {i}: seated={check.seated} at_cell={check.at_target_cell} "
              f"at_layer={check.at_target_layer} lateral={check.lateral_mm:.2f} mm "
              f"depth_error={check.depth_error_mm:.2f} mm "
              f"height_error={check.height_error_mm:.2f} mm on={','.join(check.lower_bodies)}")
        ok &= bool(check.seated and check.at_target_cell and check.at_target_layer)
    for i, cycle in enumerate(info["cycles"]):
        steps = "  ".join(f"{n} {t} ticks {e:.2f} mm" for n, t, e in cycle["steps"])
        print(f"cycle {i + 1}: {steps}  failure: {cycle['failure'] or 'none'}")
    print("RESULT: built" if ok else "RESULT: not built")
    if result is not None:
        write_manifest(stills_dir, result, info, checks)
        failures = run_qa(out, stills_dir)
        for line in failures:
            print(f"QA FAIL {line}")
        print("QA: pass" if not failures else f"QA: {len(failures)} failure(s)")
        ok &= not failures
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
