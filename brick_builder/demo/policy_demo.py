"""`python -m brick_builder.demo policy`: roll out a trained checkpoint and record it.

Defaults to `runs/reach_smoke/checkpoint.pt` against `BrickBuilder/Reach-v0`, the one skill this
project has trained all the way to its acceptance target (200/200 held-out successes -- see
`docs/guide/05-training-and-evaluation.md`). Point `--checkpoint`/`--env` at any other run
(`scripts/train.py` writes one `runs/<name>/checkpoint.pt` per config); `--actor c` plays the
exported C actor (`brick_builder.export`) instead of the original torch one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import brick_builder.envs  # noqa: F401
from brick_builder.demo.common import load_actor_fn, pick_camera, policy_action_fn, run_episode, write_video

DEFAULT_CHECKPOINT = "runs/reach_smoke/checkpoint.pt"
DEFAULT_ENV = "BrickBuilder/Reach-v0"


def run(record: str | None = None, seed: int = 0, extra=()) -> Path:
    parser = argparse.ArgumentParser(prog="policy", add_help=False)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--actor", choices=["torch", "c"], default="torch")
    args = parser.parse_args(list(extra))

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"no checkpoint at {args.checkpoint}. Train one first, e.g.:\n"
            f"  .venv/bin/python scripts/train.py --config configs/train/reach.yaml"
        )

    # PressFromHeld-v0 is blind by design (cameras=() by default): give this demo something to
    # actually show. Reach and AlignFromTokens already default to a usable camera.
    env_kwargs = {"cameras": ("overview_cam",)} if args.env == "BrickBuilder/PressFromHeld-v0" else {}
    env = gym.make(args.env, **env_kwargs)
    camera = pick_camera(env.unwrapped.scene)

    act_fn, _ = load_actor_fn(args.checkpoint, args.actor)
    frames, info = run_episode(
        env, policy_action_fn(act_fn), seed=seed, camera=camera, width=640, height=480,
    )
    env.close()
    out = write_video(frames, record or "docs/media/demos/policy.mp4", fps=25)
    print(f"wrote {out} ({args.env}, actor={args.actor}, success={info.get('success', False)})")
    return out
