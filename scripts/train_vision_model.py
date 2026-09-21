#!/usr/bin/env python
"""Build and train the conventional convolutional scene model.

Two steps, either of which can be run on its own:

    # 1. render the training set (about 15 minutes; writes to data_cache/, git-ignored)
    .venv/bin/python scripts/train_vision_model.py --generate

    # 2. train inside a wall-clock budget (default two hours), resumable
    .venv/bin/python scripts/train_vision_model.py --minutes 120

    # both, plus the contact sheet to check the labels before training starts
    .venv/bin/python scripts/train_vision_model.py --generate --contact-sheet --minutes 90

Everything a run produces lands in `--out` (default `results/vision_model/`, git-ignored):
`checkpoint.pt` (model, optimizer, history -- re-running the same command resumes from it),
`training_log.csv`, `loss_curves.png` and `train_summary.json`. The best held-out checkpoint is
also written to `brick_builder/perception/weights/vision_model.pt`, which is the one
`brick_builder.perception.cnn.source.CnnSceneModel` loads by default and the one that ships.

CPU only by default, per project policy: the GPU may be busy with someone else's job, so
`CUDA_VISIBLE_DEVICES` is cleared before torch is imported. Pass `--device cuda` on a machine
where the GPU is yours to use.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--generate", action="store_true",
                        help="render the training and held-out splits before training")
    parser.add_argument("--scenes", type=int, default=170, help="training scenes to render")
    parser.add_argument("--val-scenes", type=int, default=34, help="held-out scenes to render")
    parser.add_argument("--frames-per-scene", type=int, default=16)
    parser.add_argument("--contact-sheet", action="store_true",
                        help="write a 48-frame contact sheet with label overlays and stop if "
                             "--no-train was also given")
    parser.add_argument("--no-train", action="store_true", help="generate only, do not train")
    parser.add_argument("--cache", type=Path, default=None, help="default: data_cache/vision_model")
    parser.add_argument("--out", type=Path, default=None, help="default: results/vision_model")
    parser.add_argument("--minutes", type=float, default=120.0, help="wall-clock training budget")
    parser.add_argument("--epochs", type=int, default=200, help="cap on epochs (budget usually wins)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--width", type=float, default=1.0, help="channel-count multiplier")
    parser.add_argument("--threads", type=int, default=6, help="torch intra-op threads")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from brick_builder.perception.cnn import data as data_mod
    from brick_builder.perception.cnn.train import DEFAULT_OUT, TrainConfig, train

    cache = args.cache or data_mod.DEFAULT_CACHE
    out = args.out or DEFAULT_OUT
    Path(out).mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    if args.generate:
        summary["train_data"] = data_mod.generate(
            cache, "train", n_scenes=args.scenes, frames_per_scene=args.frames_per_scene)
        print(json.dumps(summary["train_data"]))
        summary["val_data"] = data_mod.generate(
            cache, "val", n_scenes=args.val_scenes, frames_per_scene=args.frames_per_scene)
        print(json.dumps(summary["val_data"]))

    if args.contact_sheet:
        import numpy as np

        samples = data_mod.load_samples(cache, "train", limit=1200)
        rng = np.random.default_rng(0)
        pick = [samples[i] for i in rng.permutation(len(samples))[:48]]
        path = data_mod.contact_sheet(pick, Path(out) / "contact_sheet.png")
        print(f"contact sheet: {path}")
        summary["contact_sheet"] = str(path)

    if not args.no_train:
        cfg = TrainConfig(
            cache=str(cache), out=str(out), minutes=args.minutes, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, width=args.width, threads=args.threads,
            device=args.device, seed=args.seed,
        )
        print(json.dumps(asdict(cfg), indent=1))
        summary["training"] = train(cfg)
        print(json.dumps(summary["training"], indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
