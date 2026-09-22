#!/usr/bin/env python3
"""Train the pixels-to-correction policy with asymmetric PPO, or measure one against the baseline.

    .venv/bin/python scripts/train_vision_control.py --minutes 120 --run runs/vision_control/a
    .venv/bin/python scripts/train_vision_control.py evaluate --run runs/vision_control/a \
        --seeds 0-9

The policy is the shipped convolutional scene model, the differentiable slot head and a
`G`-shaped MLP over slot-row differences; its action is a residual on the analytic controller, so
an untrained policy is today's controller exactly. Training is CPU-only, deterministic given
`--seed`, budgeted by `--minutes` and resumable: a checkpoint is written after every update, so a
run stopped by its own budget continues from where it left off when started again with the same
`--run`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train_vision_control.py",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=("train", "evaluate"), default="train")
    parser.add_argument("--run", default="runs/vision_control/default")
    parser.add_argument("--task", choices=("placement", "pick"), default="placement")
    parser.add_argument("--start-step", default="Align",
                        help="first step of an episode; 'none' starts at the first frame")
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--n-envs", type=int, default=2, help="environment processes; 2 is the "
                                                              "ceiling on a shared 16-thread CPU")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.01)
    parser.add_argument("--align-coef", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sync", action="store_true", help="one process, for a smoke test")
    parser.add_argument("--seeds", default="0-9", help="evaluation seeds")
    parser.add_argument("--baseline", action="store_true",
                        help="evaluate the analytic controller (a zero residual) instead")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from brick_builder.rl.vision_control.env import VisionControlConfig
    from brick_builder.rl.vision_control.train import VisionTrainConfig, evaluate, train

    env_cfg = VisionControlConfig(
        task=args.task,
        start_step=None if args.start_step.lower() == "none" else args.start_step,
    )
    run_dir = Path(args.run)

    if args.command == "evaluate":
        checkpoint = None if args.baseline else run_dir / "checkpoint.pt"
        report = evaluate(checkpoint, _seeds(args.seeds), env_cfg,
                          log=lambda line: print(line, flush=True))
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
        out = Path(args.out) if args.out else run_dir / (
            "baseline.json" if args.baseline else "evaluation.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=float) + "\n")
        return 0

    cfg = VisionTrainConfig(
        n_envs=args.n_envs, n_steps=args.n_steps, minibatches=args.minibatches,
        epochs=args.epochs, lr=args.lr, encoder_lr_scale=args.encoder_lr_scale,
        align_coef=args.align_coef, minutes=args.minutes, seed=args.seed,
        async_envs=not args.sync, threads=args.threads, env=env_cfg,
    )
    path = train(cfg, run_dir, log=lambda line: print(line, flush=True))
    print(f"checkpoint -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
