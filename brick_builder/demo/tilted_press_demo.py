"""`python -m brick_builder.demo tilted-press`: the tilted-press strategy
(`PressConfig(tilted_press=True)`).

Same scripted baseline as `press`, but with the tool held tilted until one stud row engages and
then levelled out (`brick_builder.control.primitives.PressScripted`, `tilted_press=True`); level 2
(+-4 mm, 8 deg) by default, since that is large enough to sometimes need the search this demo is
meant to show off. See `docs/media/control/press_recovery.png` for the same mechanism as a static
figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import brick_builder.envs  # noqa: F401
from brick_builder.control.primitives import PressConfig
from brick_builder.demo.common import run_episode, write_video
from brick_builder.envs.press_env import scripted_action


def run(record: str | None = None, seed: int = 0, extra=()) -> Path:
    parser = argparse.ArgumentParser(prog="tilted-press", add_help=False)
    parser.add_argument("--level", type=int, default=2)
    args = parser.parse_args(list(extra))

    env = gym.make(
        "BrickBuilder/PressFromHeld-v0", level=args.level, dr=False,
        press=PressConfig(tilted_press=True), cameras=("overview_cam",),
    )
    frames, info = run_episode(env, scripted_action, seed=seed, camera="overview_cam", width=640, height=480)
    env.close()
    out = write_video(frames, record or "docs/media/demos/tilted_press.mp4", fps=25)
    print(f"wrote {out} (level {args.level}: {'seated' if info.get('success', False) else 'not seated'})")
    return out
