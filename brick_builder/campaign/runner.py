"""One episode process: read a job file, run the episodes in it, append a record per episode.

The campaign driver runs at most two of these at once. Each owns its own simulator, its own
renderer and, when the convolutional scene model drives, its own copy of it, so the only thing
they share is the directory of recorded target sequences -- and the driver partitions the seeds
so that two processes never want the same one.

    python -m brick_builder.campaign.runner jobs.json
    python -m brick_builder.campaign.runner --tasks stack_2 build_2 --seeds 0-9 --no-replan \
        --out results/campaign/stacked_builds/evaluate.jsonl

The second form evaluates tasks on held-out seeds outside a campaign: it builds the job list
itself, one evaluation job per task and seed, analytic inverse and prediction rule, the
target sequences cached in `ideals/` beside `--out`.
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


def evaluation_jobs(tasks: Sequence[str], seeds: Sequence[int], *, source: str = "ideal",
                    cache: Path | None = None, replan: bool = True) -> list[dict]:
    """One `evaluate` job per task and seed, analytic inverse and analytic prediction rule."""
    return [{"task": task, "seed": int(seed), "source_name": source,
             "cache": None if cache is None else str(cache), "replan": bool(replan),
             "phase": "evaluate", "round_index": 0, "inverse": "analytic", "weights": None,
             "inverse_version": "analytic", "predictor": "analytic",
             "predictor_weights": None, "predictor_version": "analytic"}
            for task in tasks for seed in seeds]


def _parse(argv: Sequence[str]):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m brick_builder.campaign.runner")
    parser.add_argument("jobs", nargs="?", help="a job file: {\"jobs\": [...], \"out\": ...}")
    parser.add_argument("--tasks", nargs="+", help="evaluate these tasks instead of a job file")
    parser.add_argument("--seeds", default="0-9", help="evaluation seeds, 'a-b' or 'a'")
    parser.add_argument("--source", choices=("ideal", "cnn"), default="ideal")
    parser.add_argument("--no-replan", dest="replan", action="store_false")
    parser.add_argument("--out", help="records file, one JSON line per episode")
    parser.add_argument("--cache", help="recorded target sequences (default: ideals/ beside "
                                        "--out)")
    args = parser.parse_args(list(argv))
    if (args.jobs is None) == (args.tasks is None):
        parser.error("give a job file or --tasks, not both")
    if args.tasks is not None and not args.out:
        parser.error("--tasks needs --out")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.jobs is not None:
        payload = json.loads(Path(args.jobs).read_text())
        jobs, out = payload["jobs"], Path(payload["out"])
    else:
        lo, _, hi = args.seeds.partition("-")
        seeds = range(int(lo), int(hi) + 1) if hi else (int(lo),)
        out = Path(args.out)
        cache = Path(args.cache) if args.cache else out.parent / "ideals"
        jobs = evaluation_jobs(args.tasks, seeds, source=args.source, cache=cache,
                               replan=args.replan)
    failed = run_jobs(jobs, out, log=lambda line: print(line, flush=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
