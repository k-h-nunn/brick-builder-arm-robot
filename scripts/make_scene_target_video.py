#!/usr/bin/env python3
"""Record a scene-target take at 1920x1080, in one continuous open-loop replay.

    python scripts/make_scene_target_video.py --seed 1000 --source ideal
    python scripts/make_scene_target_video.py --qa-only --out media_out/hd/scene_target_l0a_gt_1080p.mp4

This wave's first video (5.2, A5): the arm plays back a cached L0a (viewing-pose) target
sequence open loop, `u = 0` on the recording clock -- there is no controller or predictor yet
(WP4/WP5), so what this take shows is the *environment*: the arm reproducing the recorded path,
the scene model's own dense decode of what it sees, and the target sequence it is being scored
against. Four panels, titled exactly "Arm camera POV", "Observed scene state", "Target scene
state" and "Predicted next scene state" (the last one copy-last, honestly labelled as a
placeholder in the spec line); a data strip of scene-state distance, prediction error and
ground-truth tool position error, all in mm-eq/mm; a title card naming the project and the run.

This file and `brick_builder/rl/scene_target/viewer.py` are WP8's own files. Everything it reads
from `brick_builder/rl/scene_target/` (`targets`, `world`, `codec`, `rules`, `wrappers`, `config`)
is that package's public surface, used exactly as `env.py` itself uses it -- this script builds
its own small open-loop driver instead of stepping `SceneTargetEnv`, because the dense decode
this video needs (`SceneState.decode`) comes from the raw per-tick state, which the environment's
observation (slot rows only, by design -- decode is viewer-only, see `perception/scene_model.py`)
does not carry.

After a take, `--qa-only` (or the same run right after recording) checks the file and its stills:
frame shape, no two text boxes touching or leaving the safe area, no drawn box over the arm's own
pixels, the smallest glyph size, and no code identifier or hash on screen -- the same checks
`scripts/make_campaign_video.py` runs, reusing `viewer.py`'s versions of them.
"""

from __future__ import annotations

import argparse
import json
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

from brick_builder.demo.video import (  # noqa: E402
    CamKey, Tick, arm_points, panel_size_for, quality_by_name, studio_look,
)
from brick_builder.demo.video_panels import LayoutError, Rect  # noqa: E402
from brick_builder.rl.scene_target import targets as TG  # noqa: E402
from brick_builder.rl.scene_target import viewer as V  # noqa: E402
from brick_builder.rl.scene_target.codec import SceneDistance  # noqa: E402
from brick_builder.rl.scene_target.config import SceneTargetConfig  # noqa: E402
from brick_builder.rl.scene_target.rules import pose_error  # noqa: E402
from brick_builder.rl.scene_target.wrappers import centre_error_mm  # noqa: E402
from brick_builder.rl.scene_target.world import (  # noqa: E402
    NOMINAL_PLATE_XY, ViewingPoseWorld, WRIST_FULL, build_source, observe, reset_source,
)

# Camera framing: the same lookat/distance/azimuth/elevation `scripts/make_campaign_video.py`
# measured for the two-brick build (span 24% of frame width, whole arm in shot) -- the same
# SO-101 arm at the same base, and the same baseplate nominal position (`NOMINAL_PLATE_XY`);
# the viewing-pose world is a strict subset of that scene (no tray, no supply bricks), so the
# same shot frames it at least as generously. Not re-measured for this scene; see the report.
CAM_LOOKAT = (-0.06, 0.14, 0.06)
CAM_DISTANCE = 1.00
CAM_AZIMUTH = 268.0
CAM_ELEVATION = -24.0
CAM_MIN_SPAN = 0.21

STILL_NAMES = (("start", 0), ("mid", None), ("end", None))
"""Named stills: the settled start pose, the midpoint tick, and the final tick. `mid`/`end`'s
target index is filled in once `T` is known (`_still_picker`)."""


def _still_picker(progress: dict):
    wanted = list(STILL_NAMES)
    wanted[1] = ("mid", progress["T"] // 2)
    wanted[2] = ("end", progress["T"] - 1)
    picked: set[str] = set()

    def pick(tick, video_t):
        del tick, video_t
        for name, target_k in wanted:
            if progress["k"] == target_k and name not in picked:
                picked.add(name)
                return name
        return None

    return pick


def build_video(out: Path | None, *, seed: int, source: str = "ideal",
                source_kwargs: dict | None = None, quality_name: str = "hd1080",
                cache_dir: str | None = None, stills_dir: Path | None = None,
                lead_seconds: float = 1.0):
    """One open-loop L0a take (or, with `out=None`, the same run headless -- for a fast check of
    the numbers before paying for an encode)."""
    quality = quality_by_name(quality_name)
    cache_dir = Path(cache_dir) if cache_dir is not None else TG.default_cache_dir()
    source_kwargs = dict(source_kwargs or {})

    seq, seq_stats = TG.load_or_generate(seed, source_name=source, source_kwargs=source_kwargs,
                                         cache_dir=cache_dir, family="viewing_pose")
    target_decode = V.load_or_render_target_decode(seq, source_name=source,
                                                   source_kwargs=source_kwargs,
                                                   cache_dir=cache_dir)

    world = ViewingPoseWorld(render_samples=quality.msaa)
    studio_look(world.scene.model, quality)
    settle_ticks = SceneTargetConfig().settle_control_ticks
    start = seq.tool_pose[0]
    world.reset(plate_offset_xy=np.zeros(2), plate_yaw=0.0, arm_tcp=start[:3], arm_yaw=start[3],
               seed=seed, settle_ticks=settle_ticks)

    obs_source, obs_info = build_source(source, source_kwargs, build_decode=True)
    reset_source(obs_source)
    distance = SceneDistance()

    panel_size = panel_size_for(quality, 4)
    panels = V.Panels(panel_size)
    last = {"d_mm": 0.0, "pred_err_mm": float("nan"), "tool_err_mm": 0.0}
    progress = {"k": 0, "T": seq.T}
    box = {"prev_rows": None, "prev_ids": None}

    def _record(state, target_k: int) -> None:
        target = seq.frame(target_k)
        res = distance(state, target)
        last["d_mm"] = res.d_mm
        pos, yaw = world.tool_pose_truth()
        pe = pose_error(pos, yaw, seq.tool_pose[min(target_k, seq.T - 1)],
                        plate_nominal_xy=NOMINAL_PLATE_XY, plate_offset_xy=np.zeros(2),
                        plate_yaw=0.0)
        last["tool_err_mm"] = pe.xyz_mm
        if box["prev_rows"] is not None:
            last["pred_err_mm"] = centre_error_mm(box["prev_rows"], box["prev_ids"], state.slots,
                                                  state.slot_ids, mm_scale=distance.mm_scale)
        box["prev_rows"], box["prev_ids"] = state.slots.copy(), state.slot_ids.copy()

    def motion():
        # Tick -1: the settled start pose, before any recorded command -- the same moment
        # `SceneTargetEnv.reset()` scores against `seq.frame(1)` (2.2).
        frame0 = world.render()
        state0 = observe(obs_source, frame0, t=0.0)
        _record(state0, 1)
        panels.update(wrist_rgb=frame0.rgb, observed_rgb=V.dense_decode_rgb(state0, res=WRIST_FULL.output_size),
                     target_rgb=target_decode[0])
        for k in range(seq.T):
            progress["k"] = k
            cmds = seq.frame_commands(k)
            for c in cmds:
                world.command(c[:3], c[3], c[4], c_vert=c[5], c_lat=c[6])
                yield Tick(scene=world.scene, phase="Viewing pose", data=dict(last))
            frame = world.render()
            state = observe(obs_source, frame, t=0.1 * (k + 1))
            target_k = min(k + 1, seq.T - 1)
            _record(state, target_k)
            observed_decode = V.dense_decode_rgb(state, res=WRIST_FULL.output_size)
            panels.update(wrist_rgb=frame.rgb, observed_rgb=observed_decode,
                         target_rgb=target_decode[target_k])
            progress["k"] = k + 1

    subtitle = V.spec_line(controller="open-loop replay",
                           predictor="copy-last",
                           scene_model=("ground truth" if source == "ideal" else source),
                           level="Viewing pose", offset_mm=0.0, offset_deg=0.0)

    result = None
    with V.ArmAwareRecorder(
        world.scene, quality=quality, shot=[CamKey(0.0, CAM_LOOKAT, CAM_DISTANCE, CAM_AZIMUTH,
                                                  CAM_ELEVATION)],
        title=V.PROJECT_TITLE, subtitle=subtitle, readouts=V.DEFAULT_READOUTS,
        panel_images=lambda tick: panels.entries(), n_panels=4,
        phase_name=lambda phase: phase, title_seconds=lead_seconds, min_span=CAM_MIN_SPAN,
        keep_in_frame=arm_points, overrun_tol=0.0, trace_seconds=1e6,
    ) as recorder:
        result = recorder.run(motion(), out, still_times=(),
                              stills_dir=stills_dir, still_when=_still_picker(progress))
        info = {
            "title": V.PROJECT_TITLE, "subtitle": subtitle, "seed": seed, "source": source,
            "T": seq.T, "phases": ["Viewing pose"], "panel_titles": list(V.PANEL_TITLES),
            "readouts": [r.label for r in V.DEFAULT_READOUTS],
            "fonts": {"title": recorder.overlay.f_title.size, "spec line": recorder.overlay.f_subtitle.size,
                     "phase": recorder.overlay.f_phase.size, "value": recorder.overlay.f_value.size,
                     "number": recorder.overlay.f_number.size, "label": recorder.overlay.f_label.size,
                     "panel title": recorder.overlay.f_small.size},
            "safe": recorder.overlay.safe.as_tuple(), "clear": recorder.overlay.clear_rect().as_tuple(),
            "panel_bar": recorder.overlay._panel_bar(),
            "arm_bodies": recorder.arm_bodies, "arm_px_under_boxes": dict(recorder.arm_under_text),
            "seq_stats": {k: v for k, v in seq_stats.items()},
            "final_d_mm": last["d_mm"], "final_tool_err_mm": last["tool_err_mm"],
        }
    world.close()
    return result, info


# ------------------------------------------------------------------------------------------ QA ---


def screen_strings(info: dict) -> list[str]:
    return [info["title"], info["subtitle"]] + list(info["phases"]) + list(info["panel_titles"]) \
        + list(info["readouts"])


def run_qa(video: Path, stills_dir: Path) -> list[str]:
    """QA checks on the finished file and its stills (mirrors `make_campaign_video.py`'s pass,
    through `viewer.py`'s versions of the same checks)."""
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
    clear = Rect(*info["clear"])
    for still in manifest["stills"]:
        path = Path(still["path"])
        img = imageio.imread(str(path))
        if img.shape[:2] != (1080, 1920):
            failures.append(f"{path.name} is {img.shape}")
        rects = {k: Rect(*v) for k, v in still["rects"].items()}
        gradient = V.main_view_gradient(img, clear)
        if gradient > V.MAX_MAIN_VIEW_GRADIENT:
            failures.append(f"{path.name}: main view gradient {gradient:.1f}, over "
                            f"{V.MAX_MAIN_VIEW_GRADIENT}: the render is corrupt")
        try:
            V.check_stills_layout(rects, safe)
        except LayoutError as exc:
            failures.append(f"{path.name}: {exc}")

    if not info.get("arm_bodies"):
        failures.append("the arm's pixels were not identified in the main view")
    for key, count in sorted(info.get("arm_px_under_boxes", {"(none)": -1}).items()):
        if count != 0:
            failures.append(f"{key}: {count} px of the arm under the box on some frame")
    small = {k: v for k, v in info["fonts"].items() if v < V.MIN_FONT_PX}
    if small:
        failures.append(f"fonts under {V.MIN_FONT_PX} px: {small}")
    hits = V.screen_identifiers(screen_strings(info))
    failures += [f"code identifier on screen: {h!r}" for h in hits]
    if tuple(info["panel_titles"]) != V.PANEL_TITLES:
        failures.append(f"panel titles are {info['panel_titles']}, not the exact four required")
    return failures


def write_manifest(stills_dir: Path, result, info: dict) -> None:
    stills = [{"path": str(p), "rects": {k: list(r.as_tuple()) for k, r in rects.items()}}
              for p, rects in zip(result.stills, result.still_rects)]
    (stills_dir / "manifest.json").write_text(json.dumps(
        {"video": str(result.master), "frames": result.frames, "seconds": result.seconds,
         "width": result.width, "height": result.height,
         "min_subject_span": result.min_subject_span, "max_arm_overrun": result.max_arm_overrun,
         "stills": stills, "info": info}, indent=1, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_scene_target_video.py")
    parser.add_argument("--out", default="media_out/hd/scene_target_l0a_gt_1080p.mp4")
    parser.add_argument("--quality", default="hd1080", choices=("draft", "hd1080"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--source", default="ideal", choices=("ideal", "learned"))
    parser.add_argument("--weights", default=None, help="a learned-source checkpoint (explicit path)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--stills-dir", default="media_out/hd/scene_target_l0a_stills")
    parser.add_argument("--lead", type=float, default=1.0)
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

    stills_dir.mkdir(parents=True, exist_ok=True)
    source_kwargs = {"threads": 2}
    if args.source == "learned" and args.weights:
        source_kwargs["weights"] = args.weights
    result, info = build_video(out, seed=args.seed, source=args.source,
                               source_kwargs=source_kwargs if args.source == "learned" else {},
                               quality_name=args.quality, cache_dir=args.cache_dir,
                               stills_dir=stills_dir, lead_seconds=args.lead)
    print(f"master {result.master} {result.frames} frames {result.seconds:.2f}s "
          f"{result.width}x{result.height}")
    print(f"min subject span {result.min_subject_span:.2%}  max arm overrun "
          f"{result.max_arm_overrun:.4f}  min ray angle {result.min_ray_angle:.1f} deg")
    print(f"seed {args.seed}  source {args.source}  T {info['T']}  "
          f"final scene-state distance {info['final_d_mm']:.2f} mm-eq  "
          f"final tool position error {info['final_tool_err_mm']:.2f} mm")
    write_manifest(stills_dir, result, info)
    failures = run_qa(out, stills_dir)
    for line in failures:
        print(f"QA FAIL {line}")
    print("QA: pass" if not failures else f"QA: {len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
