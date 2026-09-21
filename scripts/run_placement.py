#!/usr/bin/env python3
"""Run one closed-loop placement, headless, and print the result.

    .venv/bin/python scripts/run_placement.py
    .venv/bin/python scripts/run_placement.py --seed 3 --cnn
    .venv/bin/python scripts/run_placement.py --pick          # fetch the brick first

The sequence is recorded once in the digital twin from the goal and nominal parameters, then
tracked in a world whose baseplate has been moved somewhere the controller was never told about
(`brick_builder.action.worlds.displacement`). Feedback is on target minus observed scene state,
at the vision rate; the correction freezes after Align, and the press runs open loop from the
converged pose. `verify_placement` then reads the answer back out of the simulator.

`--cnn` drives the loop from the trained convolutional scene model instead of simulator ground
truth, which is the same path the film uses. Exit status is 0 when the brick seated at the
target cell, 1 when it did not -- the run itself is not a failure either way.

No display and no GPU: `MUJOCO_GL` defaults to `egl` (set `osmesa` instead where there is no
EGL device) and CUDA is switched off before torch is imported.
"""

from __future__ import annotations

import argparse
import math
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

import numpy as np


def _scene_model(name: str):
    """`None` for simulator ground truth; the trained convolutional model otherwise."""
    if name == "ideal":
        return None
    from brick_builder.perception.cnn.source import CnnSceneModel

    return CnnSceneModel()


def run(seed: int = 1, pick: bool = False, model: str = "ideal") -> int:
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import SUPPLY_BODY, build_pick_world, build_world
    from brick_builder.control.pick_place import verify_placement

    cfg = PlaceConfig(pick=pick)
    source = _scene_model(model)

    t0 = time.perf_counter()
    ideal, nominal = record_ideal(cfg, seed=seed, source=source)
    nominal.close()
    print(f"target sequence: {len(ideal)} reference frames "
          f"({time.perf_counter() - t0:.1f} s)")

    if pick:
        scene, d_xy, d_yaw, _, _ = build_pick_world(seed)
        brick = SUPPLY_BODY
    else:
        scene, d_xy, d_yaw = build_world(seed)
        brick = scene.held_name
    print(f"baseplate offset from nominal: {1e3 * float(np.linalg.norm(d_xy)):.2f} mm, "
          f"{abs(math.degrees(d_yaw)):.2f} deg (unknown to the controller)")

    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, source=source)
    t0 = time.perf_counter()
    try:
        for _ in episode.run():
            pass
        for report in episode.reports:
            mark = "ok " if report.converged else "   "
            print(f"  {mark}{report.name:24s} {report.vision_ticks:4d} vision ticks  "
                  f"final {report.final_err_mm:6.2f} mm")
        if episode.failure:
            print(f"episode failure: {episode.failure}")
        check = verify_placement(scene, brick, (4.5, 3.5, 0))
    finally:
        episode.close()

    print(f"scene model: {'ground truth' if source is None else 'convolutional'}  "
          f"seed {seed}  {time.perf_counter() - t0:.1f} s")
    if getattr(scene, "monitor", None) is not None:
        print(scene.monitor.line())
    print(f"placement: seated={check.seated} at_target_cell={check.at_target_cell} "
          f"lateral={check.lateral_mm:.2f} mm depth_error={check.depth_error_mm:.2f} mm")
    print("RESULT: placed" if check.success else "RESULT: not placed")
    return 0 if check.success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_placement.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=1,
                        help="draws the baseplate offset and the brick's offset in the jaws")
    parser.add_argument("--pick", action="store_true",
                        help="fetch the brick from the table first instead of starting with it held")
    parser.add_argument("--cnn", dest="model", action="store_const", const="cnn", default="ideal",
                        help="drive the loop from the trained convolutional scene model")
    args = parser.parse_args(argv)
    return run(seed=args.seed, pick=args.pick, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
