"""`python -m brick_builder.demo press`: one press, recorded from the overview camera.

Defaults to `PressScripted`'s own scripted baseline (`brick_builder.envs.press_env.scripted_action`):
zero residual, the primitive's own compliance -- the same thing `press_baseline_curve.png`
measures in bulk. Pass `--checkpoint <path> [--actor torch|c]` to watch a trained residual policy
instead, `--level 0..5` for the curriculum rung, and `--misalign-mm` to force an exact lateral
error rather than the rung's own random draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import brick_builder.envs  # noqa: F401
from brick_builder.demo.common import load_actor_fn, policy_action_fn, run_episode, write_video
from brick_builder.envs.press_env import scripted_action


def run(record: str | None = None, seed: int = 0, extra=()) -> Path:
    parser = argparse.ArgumentParser(prog="press", add_help=False)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--misalign-mm", type=float, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--actor", choices=["torch", "c"], default="torch")
    args = parser.parse_args(list(extra))

    misalign = None if args.misalign_mm is None else (args.misalign_mm * 1e-3, 0.0)
    env = gym.make(
        "BrickBuilder/PressFromHeld-v0", level=args.level, misalign=misalign, dr=False,
        cameras=("overview_cam",),
    )
    if args.checkpoint is not None:
        act_fn, _ = load_actor_fn(args.checkpoint, args.actor)
        action_fn = policy_action_fn(act_fn)
        tag = f"learned residual ({args.actor})"
    else:
        action_fn = scripted_action
        tag = "scripted baseline"

    frames, info = run_episode(env, action_fn, seed=seed, camera="overview_cam", width=640, height=480)
    env.close()
    out = write_video(frames, record or "docs/media/demos/press.mp4", fps=25)
    print(f"wrote {out} ({tag}, level {args.level}: "
          f"{'seated' if info.get('success', False) else 'not seated'})")
    return out
