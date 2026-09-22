"""Run the same placement through two controllers and report every difference.

    python -m brick_builder.hil.equivalence --seeds 0-9
    python -m brick_builder.hil.equivalence --seeds 0-4 --pick
    python -m brick_builder.hil.equivalence --seeds 0 --against process

Both sides are given the same target bundle, so the twin's numbers are identical and what is
compared is the control logic: the progress index frame by frame, where the step boundaries fall,
the six joint targets per 50 Hz tick, and whether the brick ended up seated on the target cell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


@dataclass
class SideResult:
    ks: list = field(default_factory=list)
    step_index: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    err_mm: list = field(default_factory=list)
    seconds: float = 0.0
    seated: bool = False
    at_cell: bool = False
    lateral_mm: float = float("nan")
    depth_error_mm: float = float("nan")
    failure: str | None = None
    ticks: int = 0
    tick_us_mean: float = float("nan")
    tick_us_max: float = float("nan")
    ik_overrides: int = 0


@dataclass
class SeedComparison:
    seed: int
    index_identical: bool
    boundaries_identical: bool
    n_vision_ticks: tuple
    max_target_diff: float
    first_divergent_tick: int
    max_err_mm_diff: float
    reference: dict
    other: dict


def _worlds(cfg, seed: int):
    from ..action.worlds import SUPPLY_BODY, build_pick_world, build_world

    if cfg.pick:
        scene, _, _, _, _ = build_pick_world(seed)
        return scene, SUPPLY_BODY
    scene, _, _ = build_world(seed)
    return scene, scene.held_name


def _score(scene, brick):
    from ..control.pick_place import verify_placement

    return verify_placement(scene, brick, (4.5, 3.5, 0))


def run_side(ideal, plan, blob, cfg, seed: int, controller_spec: str, *, lib=None,
             build=None) -> SideResult:
    """One episode with one controller. `python` is this repository's own loop."""
    from ..action.place import TrackingEpisode
    from . import controller as hil_controller
    from . import episode as hil_episode

    scene, brick = _worlds(cfg, seed)
    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, plan=plan)
    out = SideResult()
    driver = None
    started = time.perf_counter()
    try:
        if controller_spec == "python":
            goals = []
            ks = []
            steps = []
            errs = []
            for record in episode.run():
                goals.append(episode.world.joint_goal.copy())
                if record.vision:
                    ks.append(record.k)
                    steps.append(record.step_index)
                    errs.append(record.err_mm)
            out.ks, out.step_index, out.err_mm = ks, steps, errs
            # The Python path's joint targets are what the driver solved, plus the gripper.
            out.targets = [np.concatenate([g, [0.0]]) for g in goals]
            out.ticks = len(ks)
        else:
            driver = hil_controller.make(controller_spec, lib=lib, build=build)
            log = hil_episode.RunLog()
            episode.controller = hil_episode.ControllerEpisode(driver, blob, log=log)
            for record in episode.run():
                if record.vision:
                    out.step_index.append(record.step_index)
            out.ks = log.ks
            out.err_mm = log.err_mm
            out.targets = log.targets
            out.ik_overrides = log.ik_overrides
            out.ticks = len(log.ks)
            status = driver.status()
            out.tick_us_mean = status.tick_us_mean
            out.tick_us_max = status.tick_us_max
        out.seconds = time.perf_counter() - started
        out.failure = episode.failure
        check = _score(scene, brick)
        out.seated = bool(check.seated)
        out.at_cell = bool(check.at_target_cell)
        out.lateral_mm = float(check.lateral_mm)
        out.depth_error_mm = float(check.depth_error_mm)
    finally:
        episode.close()
        if driver is not None:
            driver.close()
    return out


def compare(seed: int, *, pick: bool, against: str, lib, build) -> SeedComparison:
    from ..action.place import PlaceConfig, record_ideal
    from . import bundle as B
    from . import plan as planning

    cfg = PlaceConfig(pick=pick)
    ideal, nominal = record_ideal(cfg, seed=seed)
    arm = planning.arm_constants(nominal.kinematics)
    grip = planning.grip_range(nominal.kinematics.model)
    nominal.close()

    made = planning.build_plan(ideal, cfg, seed=seed)
    blob = B.write_bundle(ideal, made, cfg, arm, grip_range=grip)

    reference = run_side(ideal, made, blob, cfg, seed, "python", lib=lib, build=build)
    other = run_side(ideal, made, blob, cfg, seed, against, lib=lib, build=build)

    n = min(len(reference.targets), len(other.targets))
    worst = 0.0
    first_bad = -1
    for i in range(n):
        d = float(np.max(np.abs(np.asarray(reference.targets[i])[:5]
                                - np.asarray(other.targets[i])[:5])))
        if d > worst:
            worst = d
        if first_bad < 0 and d > 1e-6:
            first_bad = i
    m = min(len(reference.err_mm), len(other.err_mm))
    err_diff = 0.0
    for i in range(m):
        a, b = reference.err_mm[i], other.err_mm[i]
        if a != a and b != b:
            continue
        err_diff = max(err_diff, abs(a - b))

    return SeedComparison(
        seed=seed,
        index_identical=reference.ks == other.ks,
        boundaries_identical=(_boundaries(reference.ks) == _boundaries(other.ks)
                              and reference.step_index == other.step_index),
        n_vision_ticks=(len(reference.ks), len(other.ks)),
        max_target_diff=worst,
        first_divergent_tick=first_bad,
        max_err_mm_diff=err_diff,
        reference=asdict(reference) | {"targets": [], "ks": [], "err_mm": [], "step_index": []},
        other=asdict(other) | {"targets": [], "ks": [], "err_mm": [], "step_index": []},
    )


def _boundaries(ks) -> list:
    out = []
    for i in range(1, len(ks)):
        if ks[i] != ks[i - 1]:
            out.append(i)
    return out


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def table(rows) -> str:
    head = ("| seed | index | boundaries | vision ticks | max joint target diff [rad] | "
            "max error diff [mm] | seated on cell (python / other) |")
    line = "|---|---|---|---|---|---|---|"
    body = []
    for r in rows:
        ref, oth = r.reference, r.other
        body.append(
            f"| {r.seed} | {'same' if r.index_identical else 'DIFFERENT'} | "
            f"{'same' if r.boundaries_identical else 'DIFFERENT'} | "
            f"{r.n_vision_ticks[0]} / {r.n_vision_ticks[1]} | {r.max_target_diff:.3e} | "
            f"{r.max_err_mm_diff:.3e} | "
            f"{_mark(ref)} / {_mark(oth)} |")
    return "\n".join([head, line, *body])


def _mark(side: dict) -> str:
    return "yes" if (side["seated"] and side["at_cell"]) else "no"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="brick_builder.hil.equivalence",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--against", default="c",
                        help="c (default), process, process:tcp, serial:/dev/ttyACM0")
    parser.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    args = parser.parse_args(argv)

    from .library import load

    lib, build = load()
    rows = []
    for seed in parse_seeds(args.seeds):
        row = compare(seed, pick=args.pick, against=args.against, lib=lib, build=build)
        rows.append(row)
        print(f"seed {seed}: index {'same' if row.index_identical else 'DIFFERENT'}, "
              f"ticks {row.n_vision_ticks}, max |dq| {row.max_target_diff:.3e} rad, "
              f"seated {_mark(row.reference)}/{_mark(row.other)}", flush=True)
    print()
    print(table(rows))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(r) for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
