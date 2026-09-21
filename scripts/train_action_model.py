#!/usr/bin/env python3
"""Train the next-scene-state predictor and the learned one-step inverse from logged tuples.

    .venv/bin/python scripts/train_action_model.py --data data_cache/action/placement \
        data_cache/action/pick --minutes 20

The predictor is trained first and then frozen; the inverse is trained through it (the distal
teacher) as well as on hindsight-relabelled tuples. Both run on the CPU, both are deterministic
given `--seed`, and both are time-budgeted and resumable: the optimizer state is written after
every epoch to `--state-dir` (git-ignored), so a run interrupted by its own budget continues
from where it stopped. The checkpoints themselves are two small `.npz` files under
`brick_builder/action/weights/`.

Collect the tuples first:

    python -m brick_builder.action.learn.collect --task placement --minutes 60 \
        --out data_cache/action/placement
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train_action_model.py",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", nargs="+", default=["data_cache/action/placement"])
    parser.add_argument("--weights", default="brick_builder/action/weights")
    parser.add_argument("--state-dir", default="data_cache/action/train")
    parser.add_argument("--report", default="data_cache/action/train/report.json")
    parser.add_argument("--minutes", type=float, default=20.0,
                        help="budget per model; both are resumable")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--distal", type=float, default=0.3)
    parser.add_argument("--teacher", type=float, default=0.0,
                        help="weight on regression against the analytic inverse (DAgger remedy)")
    parser.add_argument("--action-penalty", type=float, default=0.02)
    parser.add_argument("--max-ff-mm", type=float, default=None,
                        help="keep only hindsight windows where the target moves this little")
    parser.add_argument("--limit", type=int, default=None, help="episodes read per directory")
    parser.add_argument("--only", choices=("predictor", "inverse", "both"), default="both")
    parser.add_argument("--suffix", default="", help="appended to both checkpoint names")
    args = parser.parse_args(argv)

    from brick_builder.action.learn import dataset as ds
    from brick_builder.action.learn.train import TrainConfig, train_inverse, train_predictor, write_report
    from brick_builder.action.models.mlp import Mlp

    t0 = time.perf_counter()
    data = ds.load(args.data, limit=args.limit)
    print(f"{data.n} ticks from {int(data.episode.max() + 1)} episodes "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)

    cfg = TrainConfig(hidden=tuple(args.hidden), epochs=args.epochs, seed=args.seed,
                      distal=args.distal, teacher=args.teacher,
                      action_penalty=args.action_penalty, minutes=args.minutes,
                      max_ff_mm=args.max_ff_mm)
    weights = Path(args.weights)
    state = Path(args.state_dir)
    p_path = weights / f"predictor{args.suffix}.npz"
    i_path = weights / f"inverse{args.suffix}.npz"

    if args.only in ("predictor", "both"):
        predictor = train_predictor(data, cfg, p_path,
                                    resume=state / f"predictor{args.suffix}.pt")
    else:
        predictor = Mlp.load(p_path)
    print(f"predictor: {predictor.sizes} {predictor.n_params} params "
          f"{predictor.macs} MAC/tick {predictor.weight_bytes} bytes", flush=True)

    if args.only in ("inverse", "both"):
        inverse = train_inverse(data, predictor, cfg, i_path,
                                resume=state / f"inverse{args.suffix}.pt")
    else:
        inverse = Mlp.load(i_path)
    print(f"inverse:   {inverse.sizes} {inverse.n_params} params "
          f"{inverse.macs} MAC/tick {inverse.weight_bytes} bytes", flush=True)

    write_report(Path(args.report), cfg, predictor, inverse,
                 {"data": list(args.data), "ticks": int(data.n),
                  "episodes": int(data.episode.max() + 1),
                  "mm_per_centre_unit": ds.mm_per_centre_unit(data),
                  "wall_s": time.perf_counter() - t0})
    print(f"done in {(time.perf_counter() - t0) / 60.0:.1f} min -> {p_path}, {i_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
