"""Tuple logging at 10 Hz, from the arm's own motion.

One row per vision tick: the slate of slot rows the loop lined itself up on, the same slate read
off the target frame, the mechanical state the servo bus reported, the tool target actually
commanded, the analytic inverse's answer, the correction increment applied, and which step and
gate the tick belonged to. `TrackingEpisode.observer` is the only hook -- the loop's own behaviour
does not depend on it -- so the logged run is the run the controller would have made anyway.

Three things are varied across episodes, because a log of successful runs only ever shows states
on the target sequence:

- **seed**, which moves the baseplate 4-8 mm and 2-3 degrees and, for the pick, the brick on the
  table 8-15 mm and 10-20 degrees;
- a **deliberate offset**, an initial correction of up to 7 mm and 2 degrees, so the loop starts
  somewhere it would never start by itself;
- a **probe**, a smooth bounded perturbation added to the commanded tool motion every vision tick
  (a mean-reverting first-order process, not white noise, so consecutive ticks are a motion
  rather than a jitter).

Collection seeds start at 100. Seeds 0-9 are what the closed-loop tables are measured on and are
never trained on.

    python -m brick_builder.action.learn.collect --task placement --minutes 60 \
        --out data_cache/action/placement
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..models.features import HELD_NAME, N_SLATE, ROW_W
from ..models.inverse import tick_slate

EVAL_SEEDS = tuple(range(10))
"""Never collected from: the closed-loop tables are measured on these."""

FIRST_COLLECT_SEED = 100

MODES = ("clean", "offset", "probe", "both")
MODE_WEIGHTS = (0.35, 0.25, 0.25, 0.15)


@dataclass
class ProbeConfig:
    """A smooth, bounded perturbation of the commanded tool motion."""

    rho: float = 0.85
    """First-order retention per vision tick. At 0.85 the perturbation's correlation time is
    about six ticks, which is a motion the loop can chase rather than noise it averages out."""
    sigma_xy: float = 0.00035
    """Driving noise on the increment [m per tick]."""
    sigma_spin: float = math.radians(0.25)
    max_xy: float = 0.012
    """Ceiling on the accumulated correction while probing [m], a little under the loop's own."""
    max_spin: float = math.radians(4.0)
    offset_xy: float = 0.007
    """Largest deliberate starting offset [m]."""
    offset_spin: float = math.radians(2.0)


@dataclass
class Collector:
    """The observer: one appended row per vision tick, and the probe that perturbs the next one."""

    rng: np.random.Generator
    probe: bool = False
    probe_cfg: ProbeConfig = field(default_factory=ProbeConfig)

    rows: dict = field(default_factory=lambda: {k: [] for k in (
        "obs_rows", "obs_valid", "obs_weight", "tgt_rows", "tgt_valid", "slot_ids",
        "mech", "cmd", "ff", "analytic", "delta", "measured", "err_mm", "k",
        "step_index", "dwelling", "trusted", "frozen", "in_hand", "seen", "t")})
    steps: list = field(default_factory=list)
    _p: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __call__(self, episode, frame, matched, out, measured, analytic, delta) -> None:
        state = episode._state
        target = episode.ideal.frames[matched].state
        found, want = out["found"], out.get("want")
        if want is None:
            return
        names, obs_rows, obs_valid, obs_weight, tgt_rows, tgt_valid = tick_slate(
            state, target, found, want)

        ids = np.full(N_SLATE, -1, dtype=np.int64)
        for i, name in enumerate(list(names) + [HELD_NAME]):
            k = found.slots.get(name)
            if k is not None and i < N_SLATE:
                ids[i] = int(state.slot_ids[k])

        # The correction in force while this tick's commands were issued, which is what makes
        # `cmd` the tool target the world actually saw.
        before = np.asarray(episode.correction, dtype=float) - np.asarray(delta, dtype=float)
        last = frame.commands[-1]
        ff = np.array([last.pos[0], last.pos[1], last.pos[2], last.spin], dtype=float)
        cmd = ff + np.array([before[0], before[1], 0.0, before[2]])

        mech = np.concatenate([episode.world.joint_position, episode.world.joint_speed,
                               episode.world.joint_load])
        in_hand = episode.ideal.held_by_step.get(frame.step_index) is not None

        append = self.rows
        append["obs_rows"].append(obs_rows)
        append["obs_valid"].append(obs_valid)
        append["obs_weight"].append(obs_weight)
        append["tgt_rows"].append(tgt_rows)
        append["tgt_valid"].append(tgt_valid)
        append["slot_ids"].append(ids)
        append["mech"].append(mech)
        append["cmd"].append(cmd)
        append["ff"].append(ff)
        append["analytic"].append(np.asarray(analytic, dtype=float))
        append["delta"].append(np.asarray(delta, dtype=float))
        append["measured"].append(np.asarray(measured, dtype=float))
        append["err_mm"].append(float(1e3 * np.linalg.norm(np.asarray(measured)[:2])))
        append["k"].append(int(episode.k))
        append["step_index"].append(int(frame.step_index))
        append["dwelling"].append(bool(episode.dwelling))
        append["trusted"].append(bool(episode.trusted))
        append["frozen"].append(bool(episode.frozen))
        append["in_hand"].append(bool(in_hand))
        append["seen"].append(int(out["seen"]))
        append["t"].append(float(episode.world.t))
        self.steps.append(frame.step)

        if self.probe and not episode.frozen and not getattr(episode, "_descending", False):
            self._perturb(episode)

    def _perturb(self, episode) -> None:
        c = self.probe_cfg
        self._p = c.rho * self._p + np.array([
            self.rng.normal(0.0, c.sigma_xy), self.rng.normal(0.0, c.sigma_xy),
            self.rng.normal(0.0, c.sigma_spin)])
        corr = np.asarray(episode.correction, dtype=float) + self._p
        n = float(np.linalg.norm(corr[:2]))
        if n > c.max_xy:
            corr[:2] *= c.max_xy / n
        corr[2] = float(np.clip(corr[2], -c.max_spin, c.max_spin))
        episode.correction = corr

    # -- storage ---------------------------------------------------------------------------------

    def n_ticks(self) -> int:
        return len(self.rows["k"])

    def arrays(self, **extra) -> dict:
        out = {}
        for key, values in self.rows.items():
            if not values:
                return {}
            arr = np.asarray(values)
            out[key] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
        out["step"] = np.asarray(self.steps, dtype="<U24")
        out.update(extra)
        return out


# ------------------------------------------------------------------------------- one episode ---


def _ideal_pool(cfg, seeds: Sequence[int]) -> list:
    """Record the target sequence once per pool seed and reuse it; recording is nominal, so it
    depends on the task and on the driver's own noise, not on the world the run happens in."""
    from ..place import record_ideal

    pool = []
    for seed in seeds:
        ideal, world = record_ideal(cfg, seed=seed)
        world.close()
        pool.append(ideal)
    return pool


def collect_episode(task: str, seed: int, ideal, *, mode: str, rng: np.random.Generator,
                    probe_cfg: ProbeConfig, cfg) -> dict:
    """Run one episode with the observer attached and return its arrays.

    `cfg.inverse` decides which inverse drives it. Driving with the learned one and keeping the
    analytic answer as the label is the DAgger-style relabelling: the states are the learned
    policy's own, and the teacher says what it would have done there.
    """
    from ..place import TrackingEpisode
    from ..worlds import SUPPLY_BODY, build_pick_world, build_world
    from ...control.pick_place import verify_placement

    if task == "pick":
        scene, d_xy, d_yaw, b_xy, b_yaw = build_pick_world(seed)
        brick = SUPPLY_BODY
    else:
        scene, d_xy, d_yaw = build_world(seed)
        brick = scene.held_name
        b_xy, b_yaw = np.zeros(2), 0.0

    collector = Collector(rng=rng, probe=mode in ("probe", "both"), probe_cfg=probe_cfg)
    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, observer=collector)
    if mode in ("offset", "both"):
        r = probe_cfg.offset_xy * math.sqrt(rng.uniform(0.15, 1.0))
        th = rng.uniform(0.0, 2.0 * math.pi)
        episode.correction = np.array([r * math.cos(th), r * math.sin(th),
                                       rng.uniform(-1.0, 1.0) * probe_cfg.offset_spin])
    t0 = time.perf_counter()
    try:
        for _ in episode.run():
            pass
        check = verify_placement(scene, brick, (4.5, 3.5, 0))
    finally:
        episode.close()
    return collector.arrays(
        seed=np.int64(seed), mode=np.asarray(mode, dtype="<U8"),
        task=np.asarray(task, dtype="<U12"),
        plate_offset_mm=np.float32(1e3 * float(np.linalg.norm(d_xy))),
        plate_yaw_deg=np.float32(math.degrees(d_yaw)),
        brick_offset_mm=np.float32(1e3 * float(np.linalg.norm(b_xy))),
        seated=np.bool_(check.seated), at_cell=np.bool_(check.at_target_cell),
        lateral_mm=np.float32(check.lateral_mm),
        failure=np.asarray(episode.failure or "", dtype="<U96"),
        wall_s=np.float32(time.perf_counter() - t0),
    )


# --------------------------------------------------------------------------------------- CLI ---


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MUJOCO_GL", "egl")

    parser = argparse.ArgumentParser(prog="collect", description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=("placement", "pick"), default="placement")
    parser.add_argument("--out", default="data_cache/action/placement")
    parser.add_argument("--minutes", type=float, default=60.0, help="wall-clock budget")
    parser.add_argument("--episodes", type=int, default=10_000, help="ceiling on episodes")
    parser.add_argument("--seed", type=int, default=0, help="offsets the collection seeds")
    parser.add_argument("--pool", type=int, default=3, help="target sequences recorded and reused")
    parser.add_argument("--inverse", choices=("analytic", "learned"), default="analytic",
                        help="which inverse drives the logged runs; 'learned' is the DAgger set")
    args = parser.parse_args(argv)

    from ..place import PlaceConfig

    cfg = PlaceConfig(pick=args.task == "pick", inverse=args.inverse)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    base = FIRST_COLLECT_SEED + 1000 * args.seed
    pool = _ideal_pool(cfg, [base + i for i in range(args.pool)])
    print(f"{args.task}: {len(pool)} target sequences, {len(pool[0])} frames "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)

    rng = np.random.default_rng(90_000 + args.seed)
    total = 0
    for i in range(args.episodes):
        if (time.perf_counter() - t0) / 60.0 > args.minutes:
            break
        seed = base + i
        mode = str(rng.choice(MODES, p=MODE_WEIGHTS))
        arrays = collect_episode(args.task, seed, pool[i % len(pool)], mode=mode, rng=rng,
                                 probe_cfg=ProbeConfig(), cfg=cfg)
        if not arrays:
            print(f"  episode {i} seed {seed} {mode}: no vision ticks logged", flush=True)
            continue
        np.savez_compressed(out / f"ep_{args.seed:02d}_{i:05d}.npz", **arrays)
        total += int(arrays["k"].shape[0])
        print(f"  episode {i} seed {seed} {mode:7s} ticks={arrays['k'].shape[0]:4d} "
              f"total={total:7d} seated={bool(arrays['seated'])} "
              f"lat={float(arrays['lateral_mm']):6.2f} mm "
              f"({float(arrays['wall_s']):.1f} s, {(time.perf_counter() - t0) / 60.0:.1f} min)",
              flush=True)
    print(f"{args.task}: {total} ticks in {(time.perf_counter() - t0) / 60.0:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
