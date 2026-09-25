"""Scene-target calibration and benchmark: mm scale, static-arm noise, throughput, replay.

    python -m brick_builder.rl.scene_target.bench --out results/scene_target/bench.json

Per scene-model source, in order:

- **calibration** (training seeds >= 1000, written to `calib/<source>_<sha8>_viewing_pose.json`
  under the target cache before anything else runs):
  - `mm_scale`: static poses of the kinematic twin offset 0.5-5 mm in the plane from the target
    pose of each seed; the tool offset [mm] against the scene-state distance [window units],
    least squares through the origin. On the named schema the fit runs on the ground-truth
    source for every source (a centre is a window coordinate; the source's own fit is recorded);
    an opaque schema uses its own fit;
  - `sigma_obs_mm`: the arm held at the start pose in the physics world; standard deviation of
    the distance between consecutive frames, over frame pairs with every target slot matched
    (the all-pairs value and the unmatched fraction are recorded next to it);
- **throughput**: episodes with a uniform random action per stage, until `--ticks` steps; wall
  time of `step` alone and including resets and target generation, and the per-tick parts
  (physics, render, scene model, ground-truth pass, codec);
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

CALIB_SEEDS = tuple(range(1000, 1040))
"""Training seeds of the scale fit (target poses of `viewing_pose_draw`)."""
STATIC_SEEDS = (1000, 1001, 1002, 1003)
"""Training seeds of the static-arm noise measurement."""


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
            "physics_only_ticks_per_s_p50": 1e3 / float(np.nanmedian(parts["physics_ms"])),
            "planning_rate_ticks_per_s": PLANNING_RATES.get(source, float("nan")),
            "measured_over_planning": (n / step_s) / PLANNING_RATES[source]
            if source in PLANNING_RATES and step_s > 0 else float("nan"),
            "tol_mm": env.tol_mm, "mm_scale": float(env.distance.mm_scale),
            "timings_ms": {k: _stats(v) for k, v in parts.items()},
            "targets_generated": len(gen),
            "target_frame_ms": _stats([g["frame_ms"] for g in gen]),
            "rss_mb": _rss_mb(),
        }
    finally:
        env.close()


def static_noise(source: str, *, seeds, ticks_per_seed: int, threads: int,
                 weights: str | None, cache_dir: str | None) -> dict:
    """Consecutive-frame distance with the arm holding its start pose (calibrated scale)."""
    from .world import observe
    from .codec import SlotFrame

    env = SceneTargetEnv(SceneTargetConfig(level="viewing_pose", source=source, cache_dir=cache_dir,
                                           source_kwargs=_source_kwargs(source, threads, weights)))
    ds, missing = [], []
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
                res = env.distance(z, prev)
                ds.append(res.d_mm)
                missing.append(res.n_missing)
                prev = z
        mm_scale = float(env.distance.mm_scale)
    finally:
        env.close()
    d = np.asarray(ds)
    full = d[np.asarray(missing) == 0]
    return {"source": source, "seeds": [int(s) for s in seeds], "ticks": int(d.size),
            "mm_scale": mm_scale,
            "d_mean_mm": float(d.mean()), "d_std_mm": float(d.std()),
            "d_p50_mm": float(np.percentile(d, 50)), "d_p90_mm": float(np.percentile(d, 90)),
            "d_max_mm": float(d.max()),
            "unmatched_fraction": float(np.mean(np.asarray(missing) > 0)),
            "matched_ticks": int(full.size),
            "d_std_matched_mm": float(full.std()) if full.size else float("nan"),
            "d_mean_matched_mm": float(full.mean()) if full.size else float("nan")}


def fit_scale(source: str, *, seeds=CALIB_SEEDS, poses_per_seed: int = 5, threads: int = 2,
              weights: str | None = None, offset_mm=(0.5, 5.0), rng_seed: int = 0) -> dict:
    """mm of tool motion per window unit on static twin poses around viewing-pose targets.

    Each sample: the twin at the target pose of a training seed and at that pose offset in the
    plane by 0.5-5 mm in a random direction, both frames through the source in order; the
    distance at unit scale against the offset. Samples with an unmatched target slot or an
    unreachable pose are skipped. Also records the distance per mm of height and per degree of
    yaw at the fitted scale (one sample each per seed)."""
    from .codec import Calibration, SceneDistance, SlotFrame, fit_mm_scale
    from .targets import viewing_pose_draw
    from .world import NominalTwin, build_source, observe, reset_source

    src, info = build_source(source, _source_kwargs(source, threads, weights))
    twin = NominalTwin()
    rng = np.random.default_rng(rng_seed)
    d_units, truth, z_units, yaw_units = [], [], [], []
    skipped = {"unreachable": 0, "unmatched": 0}
    unit = None

    def pair(t, p):
        twin.reset_seed_pose()
        twin.pose(t[:3], t[3])
        f_t = twin.render()
        twin.pose(p[:3], p[3])
        f_p = twin.render()
        reset_source(src)
        ref = observe(src, f_t)
        obs = observe(src, f_p)
        return ref, obs

    try:
        for seed in seeds:
            t = viewing_pose_draw(int(seed)).target
            kinds = ["xy"] * int(poses_per_seed) + ["z", "yaw"]
            for kind in kinds:
                p = t.copy()
                if kind == "xy":
                    r = 1e-3 * rng.uniform(*offset_mm)
                    a = rng.uniform(0.0, 2.0 * math.pi)
                    p[:2] += r * np.array([math.cos(a), math.sin(a)])
                    mag = 1e3 * r
                elif kind == "z":
                    mag = 2.0
                    p[2] += 1e-3 * mag * rng.choice([-1.0, 1.0])
                else:
                    mag = 2.0
                    p[3] += math.radians(mag) * rng.choice([-1.0, 1.0])
                try:
                    ref, obs = pair(t, p)
                except RuntimeError:
                    skipped["unreachable"] += 1
                    continue
                if unit is None:
                    schema = ref.schema
                    unit = SceneDistance(schema, Calibration(info.name, info.weights_sha256,
                                                             schema.W, mm_scale=1.0))
                res = unit(SlotFrame.from_state(obs), SlotFrame.from_state(ref))
                if res.n_missing:
                    skipped["unmatched"] += 1
                    continue
                d = res.parts["abs"] + res.parts["rel"]
                if kind == "xy":
                    d_units.append(d)
                    truth.append(mag)
                elif kind == "z":
                    z_units.append(d / mag)
                else:
                    yaw_units.append(d / mag)
    finally:
        twin.close()
    x, y = np.asarray(d_units), np.asarray(truth)
    scale = fit_mm_scale(x, y)
    ratio = y / np.maximum(x, 1e-12)
    return {
        "source": info.name, "weights_sha256": info.weights_sha256, "W": int(unit.schema.W),
        "named": bool(unit.named), "mm_scale": scale, "n": int(x.size), "skipped": skipped,
        "seeds": [int(min(seeds)), int(max(seeds))], "poses_per_seed": int(poses_per_seed),
        "offset_mm": [float(v) for v in offset_mm],
        "ratio_mm_per_unit": {"p10": float(np.percentile(ratio, 10)),
                              "p50": float(np.percentile(ratio, 50)),
                              "p90": float(np.percentile(ratio, 90))},
        "residual_rms_mm": float(np.sqrt(np.mean((y - scale * x) ** 2))),
        "d_per_mm_height": float(scale * np.median(z_units)) if z_units else float("nan"),
        "d_per_deg_yaw": float(scale * np.median(yaw_units)) if yaw_units else float("nan"),
        "align_hover_scale": 87.4,
    }


def calibrate(source: str, *, threads: int = 2, weights: str | None = None,
              cache_dir: str | None = None, seeds=CALIB_SEEDS, poses_per_seed: int = 5,
              static_seeds=STATIC_SEEDS, static_ticks: int = 50,
              family: str = "viewing_pose") -> dict:
    """Fit `mm_scale`, then measure `sigma_obs_mm` at that scale; writes the calibration file."""
    from .codec import Calibration, calibration_path
    from .targets import default_cache_dir

    fit = fit_scale(source, seeds=seeds, poses_per_seed=poses_per_seed, threads=threads,
                    weights=weights)
    stats = {"scale_fit": fit}
    mm_scale, scale_from = fit["mm_scale"], fit["source"]
    if fit["named"] and fit["source"] != "ideal":
        # Named schema: a centre is a window coordinate whatever the source, so the mm per
        # window unit is the camera geometry's, fitted on the ground-truth source (exact
        # centres). The source's own fit is kept as a record: its noise floor reads as gain.
        geo = fit_scale("ideal", seeds=seeds, poses_per_seed=poses_per_seed, threads=threads)
        stats["geometry_fit"] = geo
        mm_scale, scale_from = geo["mm_scale"], "ideal"
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = calibration_path(root / "calib", fit["source"], fit["weights_sha256"], family)
    cal = Calibration(source=fit["source"], weights_sha256=fit["weights_sha256"], W=fit["W"],
                      family=family, mm_scale=mm_scale,
                      notes=f"mm_scale: {scale_from} source, twin xy offsets 0.5-5 mm around "
                            "viewing-pose targets of training seeds; sigma_obs_mm: static arm, "
                            "consecutive frames, all target slots matched",
                      stats=stats)
    cal.save(path)            # the static run below measures at the fitted scale
    st = static_noise(source, seeds=static_seeds, ticks_per_seed=static_ticks, threads=threads,
                      weights=weights, cache_dir=cache_dir)
    cal.sigma_obs_mm = st["d_std_matched_mm"]
    cal.n_ticks = st["ticks"]
    cal.stats["static"] = st
    cal.save(path)
    return {"path": str(path), "source": fit["source"], "weights_sha256": fit["weights_sha256"],
            "family": family, "mm_scale": cal.mm_scale, "mm_scale_from": scale_from,
            "sigma_obs_mm": cal.sigma_obs_mm, "tol_mm": cal.tol_mm, "scale_fit": fit,
            "geometry_fit": stats.get("geometry_fit"), "static": st}


def replay(source: str, *, seeds, threads: int, weights: str | None, cache_dir: str | None,
           clock: str = "recording") -> list:
    """L0a, feed-forward on, `u = 0`, nominal world. With the recording clock this is open-loop
    replay (the camera-off control condition); with the progress clock the recorded writes
    pause while the index holds."""
    env = SceneTargetEnv(SceneTargetConfig(level="viewing_pose", source=source, feedforward=True,
                                           feedforward_clock=clock, cache_dir=cache_dir,
                                           source_kwargs=_source_kwargs(source, threads, weights)))
    out = []
    try:
        for s in seeds:
            env.reset(seed=int(s), options={"world_seed": int(s)})
            n, held, done, info = 0, 0, False, {}
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
                n += 1
                held += int(info["held"])
                done = term or trunc
            out.append({"source": source, "clock": clock, "seed": int(s), "T": info["T"], "ticks": n,
                        "held_ticks": held, "tol_mm": info["tol_mm"], "mm_scale": info["mm_scale"],
                        "d_final_mm": info["d_mm"], "success_repr": info["success_repr"],
                        "success_gt": info["success_gt"], "agree": info["agree"],
                        "failure": info["failure"],
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
    p.add_argument("--static-seeds", type=int, nargs="+", default=list(STATIC_SEEDS))
    p.add_argument("--static-ticks", type=int, default=50)
    p.add_argument("--calib-seeds", type=int, nargs=2, default=[CALIB_SEEDS[0], CALIB_SEEDS[-1]],
                   metavar=("FIRST", "LAST"), help="training seeds of the scale fit")
    p.add_argument("--poses-per-seed", type=int, default=5)
    p.add_argument("--no-calibrate", action="store_true",
                   help="keep the existing calibration files; measure the static arm only")
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
        "planning_rates_ticks_per_s": PLANNING_RATES, "calibration": [], "throughput": [],
        "static": [], "replay": [],
    }
    calib_seeds = range(args.calib_seeds[0], args.calib_seeds[1] + 1)
    for source in args.sources:
        if args.no_calibrate:
            s = static_noise(source, seeds=args.static_seeds, ticks_per_seed=args.static_ticks,
                             threads=args.threads, weights=args.weights, cache_dir=args.cache_dir)
        else:
            c = calibrate(source, threads=args.threads, weights=args.weights,
                          cache_dir=args.cache_dir, seeds=calib_seeds,
                          poses_per_seed=args.poses_per_seed, static_seeds=args.static_seeds,
                          static_ticks=args.static_ticks)
            report["calibration"].append(c)
            s = c["static"]
            f = c["scale_fit"]
            print(f"calibration | {source} | mm_scale {c['mm_scale']:.1f} mm per window unit "
                  f"from {c['mm_scale_from']} | own fit {f['mm_scale']:.1f} (n {f['n']}, p10-p90 "
                  f"{f['ratio_mm_per_unit']['p10']:.1f}-{f['ratio_mm_per_unit']['p90']:.1f}) "
                  f"| sigma_obs {c['sigma_obs_mm']:.3f} mm | tol {c['tol_mm']:.3f} mm", flush=True)
        report["static"].append(s)
        print(f"static | {source} | {s['ticks']} ticks | D mean {s['d_mean_mm']:.3f} mm-eq "
              f"| std {s['d_std_mm']:.3f} mm-eq | std, all matched {s['d_std_matched_mm']:.3f} mm-eq "
              f"| unmatched {s['unmatched_fraction']:.3f}", flush=True)
        for level in args.levels:
            r = throughput(level, source, ticks=args.ticks, seeds=args.seeds, threads=args.threads,
                           weights=args.weights, cache_dir=args.cache_dir)
            report["throughput"].append(r)
            print(f"{r['level_code']} | {source} | {r['ticks']} ticks | {r['ticks_per_s']:.2f} ticks/s "
                  f"| incl. resets {r['ticks_per_s_incl_resets']:.2f} ticks/s", flush=True)
        for clock in ("recording", "progress"):
            rp = replay(source, seeds=args.replay_seeds, threads=args.threads,
                        weights=args.weights, cache_dir=args.cache_dir, clock=clock)
            report["replay"].extend(rp)
            for x in rp:
                print(f"replay | {source} | {clock} clock | seed {x['seed']} | {x['ticks']} ticks "
                      f"({x['held_ticks']} held) | D {x['d_final_mm']:.2f} mm "
                      f"| xy {x['gt_xy_mm']:.2f} mm | ground truth {x['success_gt']} "
                      f"| representation {x['success_repr']}", flush=True)
    report["host"]["loadavg_end"] = list(os.getloadavg())
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
