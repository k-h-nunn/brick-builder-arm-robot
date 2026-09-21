#!/usr/bin/env python
"""Evaluate a trained checkpoint: one command, one JSON + one markdown report.

    .venv/bin/python scripts/eval.py --checkpoint runs/reach/checkpoint.pt --env BrickBuilder/Reach-v0

Rolls out `--episodes` held-out episodes (seeded `--seed, --seed+1, ...`, deterministic mean
action, no exploration noise), writes `<out-dir>/report.json` and `report.md`
(`brick_builder.eval.report`: Wilson intervals, the failure taxonomy) and records `--videos` of them
(the first few, plus one more per distinct failure class actually seen) to `<out-dir>/episode_*
.mp4`. `--actor c` runs the exported C actor (`brick_builder.export`) instead of the original torch
one -- built fresh from the checkpoint with the system `gcc` -- so the same command checks that
the exported policy reproduces the torch one's episode outcomes under the same seed (the export
section 5's acceptance criterion). `--actor scripted` needs no `--checkpoint` at all: it runs the
zero-residual scripted baseline ("setting the residual scale to 0 gives the
scripted baseline, which every evaluation also reports") through this same report path, so the
baseline number does not need a separate hand-written script.

`--sweep press-misalign` instead runs `BrickBuilder/PressFromHeld-v0`'s own misalignment sweep
(`brick_builder.envs.press_env.baseline_jobs`/`sweep`) and writes the sweep's own JSON and
success-curve figure -- this is how "beats the scripted baseline by >=15 points at 4-6 mm"
(Deliverable 1) actually gets measured. It supports `--actor torch` (the sweep's own
`evaluate_point` loads a torch checkpoint directly, one process per sweep point) and `--actor
scripted` (no checkpoint, `evaluate_point`'s own scripted-baseline path); not `--actor c` (the
sweep never exports a C actor per point).

CPU only: `CUDA_VISIBLE_DEVICES` is cleared before `torch` is ever imported in this process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium as gym
import numpy as np

import brick_builder.envs  # noqa: F401 -- registers BrickBuilder/* gymnasium ids
from brick_builder.demo.common import load_actor_fn, pick_camera, policy_action_fn, run_episode, write_video
from brick_builder.eval.report import success_rate, write_json_report, write_markdown_report
from brick_builder.eval.results import EpisodeResult

# Cameras a scene needs added to actually show anything on video. Reach-v0 and
# AlignFromTokens-v0 both already default to a usable camera (and neither one accepts a
# `cameras` constructor kwarg at all); PressFromHeld-v0 is blind by default (`cameras=()`,
# see the environments guide) and needs one added to record anything.
_VIDEO_CAMERA_KWARGS = {
    "BrickBuilder/PressFromHeld-v0": {"cameras": ("overview_cam",)},
}


def _scripted_action_fn(env_id: str):
    """`action_fn(env) -> np.ndarray` for the checkpoint-free scripted baseline.

    The five positional/grip channels are residuals: a policy that always
    outputs zero there leaves the executed target exactly equal to the scripted primitive's own
    target, by construction (`brick_builder/envs/base_env.py`'s own docstring). `c_vert`/`c_lat` are
    not residuals -- they map straight to a servo torque-limit fraction -- so an all-zero array
    only reproduces the scripted baseline for a task that freezes compliance (Reach,
    AlignFromTokens both do). `PressFromHeld-v0` does not freeze it, and its own scripted primitive
    runs at a specific, non-midpoint torque split (`brick_builder.envs.press_env.scripted_action`), so
    it needs that function instead of a literal zero.
    """
    if env_id == "BrickBuilder/PressFromHeld-v0":
        from brick_builder.envs.press_env import scripted_action

        return scripted_action

    def zero_action(env) -> np.ndarray:
        return np.zeros(env.action_space.shape, dtype=np.float32)

    return zero_action


def _episode_result(info: dict, seed: int) -> EpisodeResult:
    result = info.get("episode_result")
    if result is None:
        return EpisodeResult(success=bool(info.get("success", False)), steps=0, params={"seed": seed})
    result.params.setdefault("seed", seed)
    return result


def run_flat_eval(
    env_id: str, env_kwargs: dict, checkpoint: Path | None, actor_kind: str, episodes: int, seed0: int,
    n_videos: int, out_dir: Path, fps: int,
) -> list[EpisodeResult]:
    if actor_kind == "scripted":
        action_fn = _scripted_action_fn(env_id)
    else:
        act_fn, _ = load_actor_fn(checkpoint, actor_kind)
        action_fn = policy_action_fn(act_fn)

    env = gym.make(env_id, **env_kwargs)
    results: list[EpisodeResult] = []
    try:
        for i in range(episodes):
            _, info = run_episode(env, action_fn, seed=seed0 + i)
            results.append(_episode_result(info, seed0 + i))
    finally:
        env.close()

    # First `n_videos` episodes, plus the first episode of every distinct failure class actually
    # observed that is not already in that set -- a small, fixed-cost way to see one clip of each
    # way this checkpoint fails, not just its most common outcome.
    seeds_to_record = list(range(min(n_videos, episodes)))
    seen_failures: set[str] = set()
    for i, r in enumerate(results):
        if r.failure is not None and r.failure.value not in seen_failures:
            seen_failures.add(r.failure.value)
            if i not in seeds_to_record:
                seeds_to_record.append(i)

    if seeds_to_record:
        video_kwargs = {**env_kwargs, **_VIDEO_CAMERA_KWARGS.get(env_id, {})}
        video_env = gym.make(env_id, **video_kwargs)
        try:
            camera = pick_camera(video_env.unwrapped.scene)
            for i in seeds_to_record:
                tag = results[i].failure.value if results[i].failure is not None else "success"
                frames, _ = run_episode(
                    video_env, action_fn, seed=seed0 + i, camera=camera, width=480, height=360,
                )
                if frames and camera is not None:
                    write_video(frames, out_dir / f"episode_{i:03d}_{tag}.mp4", fps=fps)
        finally:
            video_env.close()

    write_json_report(results, out_dir / "report.json")
    write_markdown_report(results, out_dir / "report.md", title=f"{env_id} evaluation ({actor_kind} actor)")
    overall = success_rate(results)
    print(
        f"{env_id}: {overall.p_hat:.3f} success (n={overall.n}, "
        f"95% CI [{overall.low:.3f}, {overall.high:.3f}]) -> {out_dir}"
    )
    return results


def run_press_sweep(
    checkpoint: Path | None, env_kwargs: dict, episodes: int, workers: int, out_dir: Path,
) -> list[dict]:
    from brick_builder.envs.press_env import baseline_jobs, make_press_figures, sweep

    level = env_kwargs.get("level", 3)
    dr = env_kwargs.get("dr", False)
    jobs = baseline_jobs(episodes=episodes, checkpoint=str(checkpoint) if checkpoint else None)
    for job in jobs:
        job["level"] = level
        job["dr"] = dr

    results = sweep(jobs, workers=workers)
    (out_dir / "press_sweep.json").write_text(json.dumps(results, indent=1))
    make_press_figures(out_dir=str(out_dir), results=results, workers=workers)
    total = sum(j["n"] for j in results)
    total_ok = sum(j["successes"] for j in results)
    print(f"press sweep: {total_ok}/{total} across {len(results)} points -> {out_dir}")
    return results


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, type=Path,
                         help="required unless --actor scripted")
    parser.add_argument("--env", required=True, help="a registered gymnasium id, e.g. BrickBuilder/Reach-v0")
    parser.add_argument("--env-kwargs", default="{}", help="JSON object merged into gym.make(...)")
    parser.add_argument("--producer", choices=["surrogate", "blob"], default=None,
                         help="BrickBuilder/AlignFromTokens-v0 only: overrides env-kwargs['producer']")
    parser.add_argument("--actor", choices=["torch", "c", "scripted"], default="torch",
                         help="'scripted' runs the checkpoint-free zero-residual baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000, help="held-out seeds start here")
    parser.add_argument("--videos", type=int, default=3, help="record this many of the first episodes")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="default: <checkpoint dir>/eval, or runs/eval_scripted/<env> for --actor scripted")
    parser.add_argument("--sweep", choices=["press-misalign"], default=None,
                         help="run BrickBuilder/PressFromHeld-v0's misalignment sweep instead")
    parser.add_argument("--sweep-episodes", type=int, default=20, help="episodes per sweep point")
    parser.add_argument("--workers", type=int, default=1, help="sweep only: forkserver pool size")
    args = parser.parse_args(argv)

    if args.actor != "scripted" and args.checkpoint is None:
        parser.error("--checkpoint is required unless --actor scripted")

    env_kwargs = json.loads(args.env_kwargs)
    if args.producer is not None:
        env_kwargs["producer"] = args.producer

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.checkpoint is not None:
        out_dir = args.checkpoint.resolve().parent / "eval"
    else:
        out_dir = Path("runs") / "eval_scripted" / args.env.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep == "press-misalign":
        if args.actor not in ("torch", "scripted"):
            raise SystemExit("--sweep press-misalign only supports --actor torch or scripted")
        checkpoint = None if args.actor == "scripted" else args.checkpoint
        return run_press_sweep(checkpoint, env_kwargs, args.sweep_episodes, args.workers, out_dir)

    return run_flat_eval(
        args.env, env_kwargs, args.checkpoint, args.actor, args.episodes, args.seed,
        args.videos, out_dir, args.fps,
    )


if __name__ == "__main__":
    main()
