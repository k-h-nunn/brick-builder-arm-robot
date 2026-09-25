"""Environment throughput [ticks/s], static-arm distance noise and open-loop replay, one process.

    python -m brick_builder.rl.scene_target.bench --out results/scene_target/bench.json

Three measurements per scene-model source:

- **throughput**: episodes with a uniform random action per stage, until `--ticks` steps; wall
  time of `step` alone and including resets and target generation, and the per-tick parts
  (physics, render, scene model, ground-truth pass, codec);
- **static**: the arm held at the start pose of `--static-seeds`; scene-state distance between
  consecutive frames (its mean and standard deviation give `sigma_obs`);
- **replay**: L0a with the feed-forward on and `u = 0` on `--replay-seeds`; final distance and
  ground-truth pose error.

Writes one JSON file. Single process, no worker pools.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import LEVELS, SceneTargetConfig
from .env import SceneTargetEnv

PLANNING_RATES = {"ideal": 5.0, "learned": 2.0, "physics_only": 15.0}
"""Planning estimates per job at 2 threads, nice 19 [vision ticks/s]."""


def _rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20


def _stats(values) -> dict:
    v = np.asarray([x for x in values if x == x], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _source_kwargs(source: str, threads: int, weights: str | None) -> dict:
    if source != "learned":
        return {}
    kw = {"threads": threads}
    if weights:
        kw["weights"] = weights
    return kw


def throughput(level: str, source: str, *, ticks: int, seeds, threads: int,
               weights: str | None, cache_dir: str | None, rng_seed: int = 0) -> dict:
    env = SceneTargetEnv(SceneTargetConfig(level=level, source=source, cache_dir=cache_dir,
                                           source_kwargs=_source_kwargs(source, threads, weights)))
    rng = np.random.default_rng(rng_seed)
    parts = {k: [] for k in ("physics_ms", "render_ms", "source_ms", "gt_ms", "codec_ms")}
    step_s = 0.0
    n = episodes = 0
    failures: dict[str, int] = {}
    t_total = time.perf_counter()
    try:
        seed_list = list(seeds)
        while n < ticks:
            ws = seed_list[episodes % len(seed_list)]
            env.reset(seed=int(ws), options={"world_seed": int(ws)})
            episodes += 1
            done = False
            while not done and n < ticks:
                a = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
                t0 = time.perf_counter()
                _, _, term, trunc, info = env.step(a)
                step_s += time.perf_counter() - t0
                n += 1
                for k in parts:
                    parts[k].append(info["timings"].get(k, float("nan")))
                done = term or trunc
                if done:
                    failures[info["failure"] or "success"] = failures.get(info["failure"] or "success", 0) + 1
        total_s = time.perf_counter() - t_total
        gen = [v for v in env.target_stats.values() if not v.get("cached", True)]
        return {
            "level": level, "level_code": LEVELS[level].code, "source": source,
            "weights_sha256": env.source_info.weights_sha256,
            "weights_path": env.source_info.weights_path, "threads": threads,
            "policy": "uniform random", "ticks": n, "episodes": episodes,
            "episode_ends": failures,
            "step_wall_s": step_s, "total_wall_s": total_s,
            "ticks_per_s": n / step_s if step_s > 0 else float("nan"),
            "ticks_per_s_incl_resets": n / total_s if total_s > 0 else float("nan"),
            "physics_only_ticks_per_s": 1e3 / float(np.nanmean(parts["physics_ms"])),
            "timings_ms": {k: _stats(v) for k, v in parts.items()},
            "targets_generated": len(gen),
            "target_frame_ms": _stats([g["frame_ms"] for g in gen]),
            "rss_mb": _rss_mb(),
        }
    finally:
        env.close()


def static_noise(source: str, *, seeds, ticks_per_seed: int, threads: int,
                 weights: str | None, cache_dir: str | None) -> dict:
    """Consecutive-frame distance with the arm holding its start pose."""
    from .world import observe
    from .codec import SlotFrame

    env = SceneTargetEnv(SceneTargetConfig(level="viewing_pose", source=source, cache_dir=cache_dir,
                                           source_kwargs=_source_kwargs(source, threads, weights)))
    ds = []
    try:
        for s in seeds:
            env.reset(seed=int(s), options={"world_seed": int(s)})
            world = env._world
            cmd = env._start_command
            prev = env._z
            for _ in range(ticks_per_seed):
                for _ in range(5):
                    world.command(cmd[:3], cmd[3], cmd[4])
                z = SlotFrame.from_state(observe(env._source, world.render(), t=world.t))
                ds.append(env.distance(z, prev).d_mm)
                prev = z
    finally:
        env.close()
    d = np.asarray(ds)
    return {"source": source, "seeds": [int(s) for s in seeds], "ticks": int(d.size),
            "d_mean_mm": float(d.mean()), "d_std_mm": float(d.std()),
            "d_p90_mm": float(np.percentile(d, 90)), "d_max_mm": float(d.max())}


def replay(source: str, *, seeds, threads: int, weights: str | None, cache_dir: str | None) -> list:
    """L0a, feed-forward on, `u = 0`: the open-loop control condition in the nominal world."""
    env = SceneTargetEnv(SceneTargetConfig(level="viewing_pose", source=source, feedforward=True,
                                           cache_dir=cache_dir,
                                           source_kwargs=_source_kwargs(source, threads, weights)))
    out = []
    try:
        for s in seeds:
            env.reset(seed=int(s), options={"world_seed": int(s)})
            n, done, info = 0, False, {}
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
                n += 1
                done = term or trunc
            out.append({"source": source, "seed": int(s), "T": info["T"], "ticks": n,
                        "d_final_mm": info["d_mm"], "success_repr": info["success_repr"],
                        "success_gt": info["success_gt"], "failure": info["failure"],
                        "gt_xy_mm": info["gt_final_xy_mm"], "gt_z_mm": info["gt_final_z_mm"],
                        "gt_yaw_deg": info["gt_final_yaw_deg"]})
    finally:
        env.close()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Scene-target environment benchmark")
    p.add_argument("--out", default="results/scene_target/bench.json")
    p.add_argument("--ticks", type=int, default=500)
    p.add_argument("--sources", nargs="+", default=["ideal", "learned"])
    p.add_argument("--levels", nargs="+", default=["viewing_pose", "viewing_pose_displaced"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--static-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--static-ticks", type=int, default=50)
    p.add_argument("--replay-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--weights", default=None, help="checkpoint of the learned source (explicit path)")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    report = {
        "schema": 1, "git_head": _git_head(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"machine": platform.machine(), "cpu_count": os.cpu_count(),
                 "nice": os.nice(0), "loadavg_start": list(os.getloadavg()),
                 "omp_threads": os.environ.get("OMP_NUM_THREADS", ""), "torch_threads": args.threads},
        "planning_rates_ticks_per_s": PLANNING_RATES, "throughput": [], "static": [],
        "replay": [],
    }
    for source in args.sources:
        for level in args.levels:
            r = throughput(level, source, ticks=args.ticks, seeds=args.seeds, threads=args.threads,
                           weights=args.weights, cache_dir=args.cache_dir)
            report["throughput"].append(r)
            print(f"{r['level_code']} | {source} | {r['ticks']} ticks | {r['ticks_per_s']:.2f} ticks/s "
                  f"| incl. resets {r['ticks_per_s_incl_resets']:.2f} ticks/s", flush=True)
        s = static_noise(source, seeds=args.static_seeds, ticks_per_seed=args.static_ticks,
                         threads=args.threads, weights=args.weights, cache_dir=args.cache_dir)
        report["static"].append(s)
        print(f"static | {source} | {s['ticks']} ticks | D mean {s['d_mean_mm']:.3f} mm-eq "
              f"| std {s['d_std_mm']:.3f} mm-eq", flush=True)
        rp = replay(source, seeds=args.replay_seeds, threads=args.threads, weights=args.weights,
                    cache_dir=args.cache_dir)
        report["replay"].extend(rp)
        for x in rp:
            print(f"replay | {source} | seed {x['seed']} | D {x['d_final_mm']:.2f} mm-eq "
                  f"| xy {x['gt_xy_mm']:.2f} mm | ground truth {x['success_gt']}", flush=True)
    report["host"]["loadavg_end"] = list(os.getloadavg())
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
