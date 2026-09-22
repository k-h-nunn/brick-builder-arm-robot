"""One episode process: read a job file, run the episodes in it, append a record per episode.

The campaign driver runs at most two of these at once. Each owns its own simulator, its own
renderer and, when the convolutional scene model drives, its own copy of it, so the only thing
they share is the directory of recorded target sequences -- and the driver partitions the seeds
so that two processes never want the same one.

    python -m brick_builder.campaign.worker jobs.json
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


def _source(name: str):
    """`None` for simulator ground truth; the trained convolutional model otherwise."""
    if name == "ideal":
        return None
    from ..perception.cnn.source import CnnSceneModel

    return CnnSceneModel()


def run_jobs(jobs: Sequence[dict], out: Path, *, log=print) -> int:
    from .tasks import TASKS, run_episode

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sources: dict[str, object] = {}
    failed = 0
    for job in jobs:
        job = dict(job)
        spec = TASKS[job.pop("task")]
        name = job.pop("source_name", "ideal")
        if name not in sources:
            sources[name] = _source(name)
        for key in ("weights", "predictor_weights", "cache", "tuples"):
            if job.get(key):
                job[key] = Path(job[key])
            else:
                job[key] = None
        try:
            record = run_episode(spec, int(job.pop("seed")), source=sources[name],
                                 source_name=name, log=log, **job)
        except Exception:  # one bad episode must not lose the batch
            failed += 1
            log(f"episode failed: {spec.name} {job}\n{traceback.format_exc()}")
            continue
        with out.open("a") as fh:
            fh.write(json.dumps(asdict(record), default=float) + "\n")
        marks = "".join("+" if b.on_target else "." for b in record.bricks)
        log(f"  {record.task:14s} seed {record.seed:7d} {record.inverse:8s} [{marks}] "
            f"track {record.median_tracking_mm:5.2f} mm  pred "
            f"{record.median_prediction_mm:5.2f} mm  {record.wall_s:5.1f} s  {record.failure}")
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MUJOCO_GL", "egl")
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m brick_builder.campaign.worker <jobs.json>", file=sys.stderr)
        return 2
    payload = json.loads(Path(argv[0]).read_text())
    failed = run_jobs(payload["jobs"], Path(payload["out"]),
                      log=lambda line: print(line, flush=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
