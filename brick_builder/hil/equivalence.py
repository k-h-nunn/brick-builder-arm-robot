"""Run the same placement through several controllers and report every difference.

    python -m brick_builder.hil.equivalence --seeds 0-9
    python -m brick_builder.hil.equivalence --seeds 0-9 --pick
    python -m brick_builder.hil.equivalence --seeds 0-9 --against c,process,process:tcp

Every side is given the same target bundle, so the twin's numbers are identical and what is
compared is the control logic: the progress index frame by frame, where the step boundaries fall,
the six joint targets per 50 Hz tick, and whether the brick ended up seated on the target cell.
The C library in this process (`c`) is compared against the repository's own Python loop; the
native process over a pipe (`process`) or loopback TCP (`process:tcp`) is compared against `c`
bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

TARGET_BOUND = 1e-6
"""Radians a joint target may differ from the Python loop's on the same 50 Hz tick."""


@dataclass
class SideResult:
    ks: list = field(default_factory=list)
    step_index: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    """Per 50 Hz tick: the joint targets the side published (five arm joints first)."""
    err_mm: list = field(default_factory=list)
    corrections: list = field(default_factory=list)
    seconds: float = 0.0
    seated: bool = False
    at_cell: bool = False
    lateral_mm: float = float("nan")
    depth_error_mm: float = float("nan")
    failure: str | None = None
    error: str | None = None
    """A transport or protocol failure that ended the side early, verbatim."""
    ticks: int = 0
    tick_us_mean: float = float("nan")
    tick_us_max: float = float("nan")
    ik_overrides: int = 0
    bad_crc: int = 0
    resyncs: int = 0

    @property
    def on_cell(self) -> bool:
        return bool(self.seated and self.at_cell)


@dataclass
class SeedComparison:
    task: str
    seed: int
    controller: str
    against: str
    """What this side was compared with: `python` for `c`, `c` for the wire transports."""
    index_identical: bool
    boundaries_identical: bool
    n_vision_ticks: tuple
    n_control_ticks: tuple
    max_target_diff: float
    first_divergent_tick: int
    max_err_mm_diff: float
    same_outcome: bool
    bit_identical: bool
    passed: bool
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
            for record in episode.run():
                if record.vision:
                    out.ks.append(record.k)
                    out.step_index.append(record.step_index)
                    out.err_mm.append(record.err_mm)
                    out.corrections.append(episode.correction.copy())
                else:
                    # What the driver solved for this write, plus a gripper column so the rows
                    # line up with the C side's six.
                    out.targets.append(np.concatenate([episode.world.joint_goal, [0.0]]))
            out.ticks = len(out.ks)
        else:
            driver = hil_controller.make(controller_spec, lib=lib, build=build)
            log = hil_episode.RunLog()
            episode.controller = hil_episode.ControllerEpisode(driver, blob, log=log)
            try:
                for record in episode.run():
                    if record.vision:
                        out.step_index.append(record.step_index)
            except (hil_controller.ControllerError, OSError) as exc:
                out.error = f"{type(exc).__name__}: {exc}"
            out.ks = log.ks
            out.err_mm = log.err_mm
            out.corrections = log.corrections
            out.targets = log.targets
            out.ik_overrides = log.ik_overrides
            out.ticks = len(log.ks)
            if out.error is None:
                status = driver.status()
                out.tick_us_mean = status.tick_us_mean
                out.tick_us_max = status.tick_us_max
                out.bad_crc = status.bad_crc
                out.resyncs = status.resyncs
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


def failure_class(text: str | None) -> int:
    """The Python loop's failure message as the C library's failure class (`library.FAILURES`)."""
    if text is None:
        return 0
    if text.startswith("stalled"):
        return 1 if "nothing to line up on" in text else 2
    if text.startswith("the grasp stopped"):
        return 3
    return -1


def _same_floats(a, b) -> bool:
    """Equal to the last bit, NaN equal to NaN."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def diff(task: str, seed: int, controller: str, against: str, ref: SideResult,
         other: SideResult) -> SeedComparison:
    n = min(len(ref.targets), len(other.targets))
    worst = 0.0
    first_bad = -1
    for i in range(n):
        d = float(np.max(np.abs(np.asarray(ref.targets[i])[:5]
                                - np.asarray(other.targets[i])[:5])))
        worst = max(worst, d)
        if first_bad < 0 and d > TARGET_BOUND:
            first_bad = i
    if first_bad < 0 and len(ref.targets) != len(other.targets):
        first_bad = n
    m = min(len(ref.err_mm), len(other.err_mm))
    err_diff = 0.0
    for i in range(m):
        a, b = ref.err_mm[i], other.err_mm[i]
        if a != a and b != b:
            continue
        err_diff = max(err_diff, abs(a - b))

    index_same = ref.ks == other.ks
    bounds_same = (_boundaries(ref.ks) == _boundaries(other.ks)
                   and ref.step_index == other.step_index)
    same_outcome = (ref.on_cell == other.on_cell
                    and failure_class(ref.failure) == failure_class(other.failure)
                    and other.error is None)
    bit = (index_same and _same_floats(ref.targets, other.targets)
           and _same_floats(ref.err_mm, other.err_mm)
           and _same_floats(ref.corrections, other.corrections))
    if against == "python":
        passed = (index_same and bounds_same and same_outcome and first_bad < 0
                  and worst <= TARGET_BOUND)
    else:
        passed = bit and same_outcome
    strip = {"targets": [], "ks": [], "err_mm": [], "step_index": [], "corrections": []}
    return SeedComparison(
        task=task, seed=seed, controller=controller, against=against,
        index_identical=index_same, boundaries_identical=bounds_same,
        n_vision_ticks=(len(ref.ks), len(other.ks)),
        n_control_ticks=(len(ref.targets), len(other.targets)),
        max_target_diff=worst, first_divergent_tick=first_bad, max_err_mm_diff=err_diff,
        same_outcome=same_outcome, bit_identical=bit, passed=passed,
        reference=asdict(ref) | strip, other=asdict(other) | strip,
    )


def compare(seed: int, *, pick: bool, against, lib, build) -> list[SeedComparison]:
    """One seed: the Python loop, then every controller in `against` on the same bundle."""
    from ..action.place import PlaceConfig, record_ideal
    from . import bundle as B
    from . import plan as planning

    if isinstance(against, str):
        against = [against]
    task = "pick" if pick else "placement"
    cfg = PlaceConfig(pick=pick)
    ideal, nominal = record_ideal(cfg, seed=seed)
    arm = planning.arm_constants(nominal.kinematics)
    grip = planning.grip_range(nominal.kinematics.model)
    nominal.close()

    made = planning.build_plan(ideal, cfg, seed=seed)
    blob = B.write_bundle(ideal, made, cfg, arm, grip_range=grip)

    sides = {"python": run_side(ideal, made, blob, cfg, seed, "python", lib=lib, build=build)}
    rows = []
    for spec in against:
        sides[spec] = run_side(ideal, made, blob, cfg, seed, spec, lib=lib, build=build)
        base = "python" if spec == "c" or "c" not in sides else "c"
        rows.append(diff(task, seed, spec, base, sides[base], sides[spec]))
    return rows


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
    head = ("| task | seed | controller | against | index | boundaries | vision ticks | "
            "max joint target diff [rad] | max error diff [mm] | on cell (ref / side) | "
            "bit-identical | pass |")
    line = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    body = []
    for r in rows:
        body.append(
            f"| {r.task} | {r.seed} | {r.controller} | {r.against} | "
            f"{'same' if r.index_identical else 'DIFFERENT'} | "
            f"{'same' if r.boundaries_identical else 'DIFFERENT'} | "
            f"{r.n_vision_ticks[0]} / {r.n_vision_ticks[1]} | {r.max_target_diff:.3e} | "
            f"{r.max_err_mm_diff:.3e} | {_mark(r.reference)} / {_mark(r.other)} | "
            f"{'yes' if r.bit_identical else 'no'} | {'PASS' if r.passed else 'FAIL'} |")
    return "\n".join([head, line, *body])


def _mark(side: dict) -> str:
    if side.get("error"):
        return "error"
    return "yes" if (side["seated"] and side["at_cell"]) else "no"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="brick_builder.hil.equivalence",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--against", default="c",
                        help="comma list of c, process, process:tcp, serial:/dev/ttyACM0")
    parser.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    args = parser.parse_args(argv)

    from .library import load

    lib, build = load()
    against = [a.strip() for a in args.against.split(",") if a.strip()]
    rows = []
    for seed in parse_seeds(args.seeds):
        started = time.perf_counter()
        got = compare(seed, pick=args.pick, against=against, lib=lib, build=build)
        rows.extend(got)
        for row in got:
            print(f"{row.task} seed {seed} {row.controller} vs {row.against}: "
                  f"index {'same' if row.index_identical else 'DIFFERENT'}, "
                  f"ticks {row.n_vision_ticks}, max |dq| {row.max_target_diff:.3e} rad, "
                  f"bit-identical {row.bit_identical}, on cell "
                  f"{_mark(row.reference)}/{_mark(row.other)}, tick "
                  f"{row.other['tick_us_mean']:.0f}/{row.other['tick_us_max']:.0f} us, "
                  f"{'PASS' if row.passed else 'FAIL'}", flush=True)
            if row.other.get("error"):
                print(f"  {row.controller}: {row.other['error']}", flush=True)
        print(f"  seed {seed}: {time.perf_counter() - started:.0f} s", flush=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps([asdict(r) for r in rows], indent=1))
    print()
    print(table(rows))
    return 0 if all(r.passed for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
