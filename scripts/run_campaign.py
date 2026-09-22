#!/usr/bin/env python3
"""Run the closed-loop campaign: perform the tasks, train on what happened, redeploy, measure.

    .venv/bin/python scripts/run_campaign.py --rounds 5 --episodes 5 --budget-minutes 210
    .venv/bin/python scripts/run_campaign.py --resume
    .venv/bin/python scripts/run_campaign.py --evaluate-only --round 0 --source cnn \
        --name closed_loop_cnn

Rounds of: perform `N` episodes of every task on fresh seeds with the models the campaign has
deployed, train the next-scene-state predictor and the one-step inverse on everything logged so
far, redeploy them under a new version id, and score every task on the held-out seeds 0-9 with
the learned inverse and with the analytic one.

Results land under `results/campaign/<name>/`: `campaign.json` (the resumable state), a CSV per
round, `summary.csv` over all of them, and the curves. The campaign never writes the checkpoints
the repository ships -- its own deployed pair lives under `results/campaign/<name>/weights/`.

Two episode processes at a time, three threads each, no GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_campaign.py",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="closed_loop", help="campaign directory name")
    parser.add_argument("--root", default="results/campaign")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5,
                        help="episodes per task per performing round")
    parser.add_argument("--tasks", nargs="+",
                        default=["placement", "pick_and_place", "build_2"])
    parser.add_argument("--seeds", default="0-9", help="the held-out evaluation seeds")
    parser.add_argument("--source", choices=("ideal", "cnn"), default="ideal",
                        help="which scene model drives the loop")
    parser.add_argument("--processes", type=int, default=2,
                        help="episode processes at once")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--budget-minutes", type=float, default=240.0)
    parser.add_argument("--train-minutes", type=float, default=4.0, help="budget per model")
    parser.add_argument("--sparse-tasks", nargs="*", default=[],
                        help="tasks evaluated in the first and last round only")
    parser.add_argument("--round-minutes", type=float, default=38.0,
                        help="expected cost of a round, checked against the budget")
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing campaign (the default when one is there)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="score one round's deployed models and stop")
    parser.add_argument("--round", type=int, default=0,
                        help="which round --evaluate-only scores")
    parser.add_argument("--no-baseline-learned", dest="baseline_learned",
                        action="store_false",
                        help="skip the shipped pair in round 0")
    parser.add_argument("--curves-only", action="store_true",
                        help="redraw the figures and tables from what is already measured")
    args = parser.parse_args(argv)

    from brick_builder.campaign.campaign import Campaign, CampaignConfig
    from brick_builder.campaign.curves import draw_all
    from brick_builder.campaign.tables import render

    lo, _, hi = args.seeds.partition("-")
    seeds = tuple(range(int(lo), int(hi) + 1)) if hi else (int(lo),)

    cfg = CampaignConfig(
        name=args.name, root=args.root, rounds=args.rounds, episodes=args.episodes,
        tasks=tuple(args.tasks), eval_seeds=seeds, source=args.source, processes=args.processes,
        threads=args.threads, budget_minutes=args.budget_minutes,
        train_minutes=args.train_minutes, train_epochs=args.train_epochs, seed=args.seed,
        round_minutes=args.round_minutes, sparse_tasks=tuple(args.sparse_tasks),
        baseline_learned=args.baseline_learned)
    campaign = Campaign(cfg, log=lambda line: print(line, flush=True))

    if args.curves_only:
        campaign.summarize()
    elif args.evaluate_only:
        inverse, predictor, version = campaign.deployed_paths()
        if args.round == 0 and inverse is None:
            inverse, predictor = campaign.shipped_paths()
            version = "shipped"
        records = campaign.evaluate(args.round,
                                    campaign.columns_for(args.round, version, inverse, predictor))
        campaign.score_round(args.round, version, records)
        campaign.summarize()
    else:
        campaign.run()

    for path in draw_all(campaign.dir):
        print(path)
    (campaign.dir / "tables.md").write_text(render(campaign.dir))
    print(campaign.dir / "tables.md")
    print(f"elapsed {campaign.elapsed_minutes:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
