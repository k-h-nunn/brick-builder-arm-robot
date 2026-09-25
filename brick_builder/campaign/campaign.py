"""The loop: perform, train, redeploy, evaluate, once per round, until the budget runs out.

    round n:  perform N episodes per task on fresh seeds with the models deployed now
              -> train F and G on everything logged so far
              -> redeploy as version n
              -> evaluate every task on the held-out seeds, learned inverse and analytic

Training and redeployment go through `action.learn.rounds`, which owns the version ids, the
resumable optimizer state and the deployment record. The campaign points it at its own weights
directory: nothing here writes the checkpoints the repository ships.

The analytic inverse is a fixed baseline. It does not change between rounds, so it is measured
once, in round 0, and carried into every table; `recheck_analytic` re-measures it on one task at
the end as a check that it really did not move.

Everything is resumable. Each phase records itself in `campaign.json` when it finishes, and a
campaign started again skips what is already there. The wall-clock budget is checked before every
phase, so a campaign that runs out of time stops between phases with its state consistent.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..eval.wilson import wilson_interval
from .tasks import CSV_FIELDS, EVAL_SEEDS, BrickRecord, EpisodeRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_WEIGHTS = REPO_ROOT / "brick_builder" / "action" / "weights"


@dataclass(frozen=True)
class CampaignConfig:
    """What to run. Everything here is written into `campaign.json` and read back on a resume."""

    name: str = "closed_loop"
    root: str = "results/campaign"
    rounds: int = 5
    """Training rounds after the baseline. Round 0 measures and trains nothing."""

    episodes: int = 5
    """Episodes per task per performing round."""

    tasks: tuple[str, ...] = ("placement", "pick_and_place", "build_2")
    eval_seeds: tuple[int, ...] = EVAL_SEEDS
    source: str = "ideal"
    """`ideal` for the simulator's own scene model, `cnn` for the convolutional one."""

    processes: int = 2
    """Episode processes at once. Two, because the machine is shared."""

    threads: int = 3
    """`OMP_NUM_THREADS` inside each of them."""

    nice: int = 10
    budget_minutes: float = 240.0
    train_minutes: float = 4.0
    """Budget per model per round. Training is resumable, so a round that hits it continues."""

    train_epochs: int = 60
    seed: int = 0
    predictor_pairs: tuple[str, ...] = ("placement",)
    """Tasks whose every evaluated column is also run with the analytic prediction rule, so the
    two predictors are compared on the same trajectories. The prediction is off the control path,
    so the pair is the same run scored twice; it costs a second run all the same."""

    baseline_learned: bool = True
    """Measure the shipped `F`/`G` pair in round 0, as the point the campaign starts from."""

    recheck_analytic: bool = True
    """Re-measure the analytic baseline on one task in the last round."""

    sparse_tasks: tuple[str, ...] = ()
    """Tasks evaluated in the first and the last round only, and performed in every one. A build
    is two closed-loop cycles per seed, so ten held-out builds are half of what a round of
    evaluation costs; naming it here buys rounds with the budget."""

    round_minutes: float = 38.0
    """What a round is expected to cost. The budget is checked against it before a round starts,
    so the campaign stops between rounds rather than in the middle of one."""

    ideal_pool_seed: int = 900_000
    """Seeds the performing rounds record their target sequences from. One per process, reused
    across every performing episode and every round, so recording is not paid per episode."""

    episode_timeout_s: int = 7200
    train_timeout_s: int = 5400


@dataclass
class RoundRecord:
    """What one round produced."""

    index: int
    version: str = ""
    ticks: int = 0
    episodes_logged: int = 0
    scores: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    wall_s: dict = field(default_factory=dict)
    would_ship: dict = field(default_factory=dict)
    """What `rounds.deploy_if_better` would have decided about the shipped pair this round:
    whether the learned inverse matched the analytic baseline on every task it was measured on,
    and the two fractions per task. The campaign deploys to its own directory whatever this says;
    this is the separate question of whether the round earned the checkpoints the repository
    ships."""


# ----------------------------------------------------------------------------------- the driver ---


class Campaign:
    """One campaign, on disk under `<root>/<name>`."""

    def __init__(self, cfg: CampaignConfig, *, log=print) -> None:
        self.cfg = cfg
        self.log = log
        self.dir = Path(cfg.root) / cfg.name
        self.weights = self.dir / "weights"
        self.ideals = self.dir / "ideals"
        self.data = self.dir / "data"
        self.rounds_dir = self.dir / "rounds"
        for path in (self.dir, self.weights, self.ideals, self.data, self.rounds_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "campaign.json"
        self.state = self._load_state()
        self.t0 = time.perf_counter()
        self.spent_s = float(self.state.get("wall_s", 0.0))

    # -- state ---------------------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"config": asdict(self.cfg), "started": _now(), "phases": {}, "rounds": {},
                "wall_s": 0.0}

    def _save(self) -> None:
        self.state["config"] = asdict(self.cfg)
        self.state["updated"] = _now()
        self.state["wall_s"] = self.spent_s + (time.perf_counter() - self.t0)
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str) + "\n")

    def _done(self, key: str) -> dict | None:
        entry = self.state["phases"].get(key)
        return entry if entry and entry.get("done") else None

    def _finish(self, key: str, **fields) -> dict:
        entry = {"done": True, "at": _now(), **fields}
        self.state["phases"][key] = entry
        self._save()
        return entry

    def _round(self, index: int) -> RoundRecord:
        raw = self.state["rounds"].get(str(index))
        if raw is None:
            return RoundRecord(index=index)
        return RoundRecord(index=index, version=raw.get("version", ""),
                           ticks=int(raw.get("ticks", 0)),
                           episodes_logged=int(raw.get("episodes_logged", 0)),
                           scores=raw.get("scores", {}), metrics=raw.get("metrics", {}),
                           wall_s=raw.get("wall_s", {}),
                           would_ship=raw.get("would_ship", {}))

    def _put_round(self, record: RoundRecord) -> None:
        self.state["rounds"][str(record.index)] = asdict(record)
        self._save()

    # -- budget --------------------------------------------------------------------------------

    @property
    def elapsed_minutes(self) -> float:
        return (self.spent_s + time.perf_counter() - self.t0) / 60.0

    def budget_left(self, need_minutes: float = 0.0) -> bool:
        return self.elapsed_minutes + need_minutes <= self.cfg.budget_minutes

    # -- deployed models -------------------------------------------------------------------------

    def deployed_paths(self) -> tuple[Path | None, Path | None, str]:
        """The campaign's own checkpoints and the version they came from.

        Before the first round there are none, and the baseline round measures the pair the
        repository ships instead, which is where the campaign starts from.
        """
        record = self.weights / "deployed.json"
        if record.exists():
            meta = json.loads(record.read_text())
            return (self.weights / "inverse.npz", self.weights / "predictor.npz",
                    str(meta.get("version", "")))
        return None, None, ""

    def shipped_paths(self) -> tuple[Path, Path]:
        return SHIPPED_WEIGHTS / "inverse.npz", SHIPPED_WEIGHTS / "predictor.npz"

    # -- running episodes ------------------------------------------------------------------------

    def _run_jobs(self, jobs: list[dict], label: str, out_dir: Path) -> list[EpisodeRecord]:
        """Split the jobs across the episode processes by seed and wait for all of them.

        By seed, not round robin: two episodes of the same task and seed want the same recorded
        target sequence, and a process that owns the seed records it once instead of both
        recording it at the same time.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        records_path = out_dir / f"{label}.jsonl"
        if records_path.exists():
            records_path.unlink()

        buckets: list[list[dict]] = [[] for _ in range(max(self.cfg.processes, 1))]
        for job in jobs:
            key = int(job.get("ideal_seed") or job["seed"])
            buckets[key % len(buckets)].append(job)

        procs = []
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "egl",
                    "OMP_NUM_THREADS": str(self.cfg.threads),
                    "MKL_NUM_THREADS": str(self.cfg.threads),
                    "PYTHONPATH": str(REPO_ROOT)})
        for i, bucket in enumerate(buckets):
            if not bucket:
                continue
            job_file = out_dir / f"{label}_jobs_{i}.json"
            job_file.write_text(json.dumps({"jobs": bucket, "out": str(records_path)}, indent=1))
            log_file = out_dir / f"{label}_process_{i}.log"
            cmd = ["nice", "-n", str(self.cfg.nice), sys.executable, "-u",
                   "-m", "brick_builder.campaign.runner", str(job_file)]
            handle = log_file.open("w")
            procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle,
                                           stderr=subprocess.STDOUT), handle, log_file))
            self.log(f"  process {i}: {len(bucket)} episodes -> {log_file}")

        deadline = time.time() + self.cfg.episode_timeout_s
        for proc, handle, log_file in procs:
            try:
                proc.wait(timeout=max(deadline - time.time(), 1.0))
            except subprocess.TimeoutExpired:
                self.log(f"  process timed out after {self.cfg.episode_timeout_s} s; terminating")
                proc.terminate()
                proc.wait(timeout=60)
            handle.close()
            if proc.returncode:
                self.log(f"  process exited {proc.returncode}; see {log_file}")

        records = []
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                if line.strip():
                    records.append(_record_from(json.loads(line)))
        return records

    # -- phases ----------------------------------------------------------------------------------

    def perform(self, index: int) -> list[EpisodeRecord]:
        """Run the tasks with whatever is deployed, logging one tuple file per cycle."""
        from .tasks import perform_seeds

        key = f"r{index}.perform"
        out_dir = self.rounds_dir / f"r{index}"
        done = self._done(key)
        if done:
            self.log(f"round {index}: perform already done "
                     f"({done.get('episodes', 0)} episodes)")
            return _read_csv(out_dir / "perform.csv")

        inverse_path, predictor_path, version = self.deployed_paths()
        inverse = "learned" if inverse_path else "analytic"
        predictor = "learned" if predictor_path else "analytic"
        self.log(f"round {index}: perform {self.cfg.episodes} x {len(self.cfg.tasks)} episodes, "
                 f"{inverse} inverse driving"
                 + (f" (version {version})" if version else ""))

        jobs = []
        for task in self.cfg.tasks:
            for i, seed in enumerate(perform_seeds(task, index, self.cfg.episodes)):
                # A pool of recorded sequences, not one per episode: the recording is nominal, so
                # what a performing episode varies is the world it runs in, not the reference.
                jobs.append(self._job(task, seed, phase="perform", round_index=index,
                                      inverse=inverse, weights=inverse_path,
                                      predictor=predictor, predictor_weights=predictor_path,
                                      inverse_version=version or "analytic",
                                      predictor_version=version or "analytic",
                                      ideal_seed=self.cfg.ideal_pool_seed + i % self.cfg.processes,
                                      tuples=self.data / f"r{index}",
                                      tuple_prefix=f"ep_r{index}"))
        t0 = time.perf_counter()
        records = self._run_jobs(jobs, "perform", out_dir)
        _write_csv(out_dir / "perform.csv", records)
        self._finish(key, episodes=len(records), wall_s=time.perf_counter() - t0,
                     inverse=inverse, version=version)
        return records

    def train(self, index: int):
        """Fit `F` and `G` on every tuple the campaign has logged, through `learn.rounds`."""
        from ..action.learn.rounds import RoundConfig, load_round, train_round

        key = f"r{index}.train"
        done = self._done(key)
        data = sorted(str(p) for p in self.data.glob("r*") if any(p.glob("ep_*.npz")))
        if not data:
            self.log(f"round {index}: nothing logged yet, no training")
            return None
        cfg = RoundConfig(data=tuple(data), seed=self.cfg.seed, minutes=self.cfg.train_minutes,
                          epochs=self.cfg.train_epochs, teacher=1.0,
                          root=str(self.rounds_dir / "checkpoints"),
                          version=f"r{index}")
        if done:
            self.log(f"round {index}: training already done ({done.get('version')})")
            return load_round(done["version"], root=cfg.root)

        self.log(f"round {index}: train on {len(data)} set(s)")
        t0 = time.perf_counter()
        result = train_round(cfg, log=lambda line: self.log(f"    {line}"))
        self._finish(key, version=result.version, ticks=result.ticks,
                     episodes=result.episodes, wall_s=time.perf_counter() - t0)
        return result

    def redeploy(self, index: int, result) -> str:
        """Copy the round's pair onto the campaign's own deployed pair.

        Unconditionally, and never onto the repository's shipped weights. A campaign that only
        deployed rounds that already won would never collect the states a losing round visits,
        which is the data the next round is relabelled on. `rounds.deploy_if_better` is the gate
        for what ships; this is the gate for what the campaign drives with.
        """
        from ..action.learn.rounds import deploy

        key = f"r{index}.deploy"
        done = self._done(key)
        if done:
            return str(done.get("version", ""))
        record = deploy(result, weights=self.weights, note=f"campaign {self.cfg.name} round {index}")
        self.log(f"round {index}: deployed {result.version} -> {self.weights}")
        self._finish(key, version=result.version)
        return record["version"]

    def evaluate(self, index: int, columns: Sequence[dict]) -> list[EpisodeRecord]:
        """Score every task on the held-out seeds, one batch per round."""
        key = f"r{index}.evaluate"
        out_dir = self.rounds_dir / f"r{index}"
        done = self._done(key)
        if done:
            self.log(f"round {index}: evaluation already done ({done.get('episodes', 0)})")
            return _read_csv(out_dir / "evaluate.csv")

        jobs = []
        for column in columns:
            for task in column.get("tasks", self.cfg.tasks):
                for seed in self.cfg.eval_seeds:
                    jobs.append(self._job(task, seed, phase="evaluate", round_index=index,
                                          replan=True, **{k: v for k, v in column.items()
                                                          if k != "tasks"}))
        self.log(f"round {index}: evaluate {len(jobs)} episodes on seeds "
                 f"{self.cfg.eval_seeds[0]}-{self.cfg.eval_seeds[-1]}")
        t0 = time.perf_counter()
        records = self._run_jobs(jobs, "evaluate", out_dir)
        _write_csv(out_dir / "evaluate.csv", records)
        self._finish(key, episodes=len(records), wall_s=time.perf_counter() - t0)
        return records

    # -- job construction ------------------------------------------------------------------------

    def _job(self, task: str, seed: int, **kw) -> dict:
        job = {"task": task, "seed": int(seed), "source_name": self.cfg.source,
               "cache": str(self.ideals), "replan": True}
        job.update({k: (str(v) if isinstance(v, Path) else v) for k, v in kw.items()})
        return job

    def columns_for(self, index: int, version: str, inverse_path: Path | None,
                 predictor_path: Path | None) -> list[dict]:
        """Which columns this round measures.

        The learned inverse every round; the analytic one in round 0, where it is the fixed
        baseline, and once more at the end as a check. Each column is scored with the learned
        prediction rule, and on `predictor_pairs` tasks with the analytic one as well: the pair
        is the same run scored twice, so the two prediction errors are compared on identical
        trajectories rather than across runs.

        A task named in `sparse_tasks` is evaluated in the first and the last round only. It is
        still performed every round; what it costs is the ten held-out episodes, and a build is
        two of them. This is the knob a wall-clock budget turns before it gives up a round.
        """
        columns: list[dict] = []
        wanted: list[tuple[str, Path | None, str]] = []
        if index == 0:
            wanted.append(("analytic", None, "analytic"))
            if self.cfg.baseline_learned:
                wanted.append(("learned", inverse_path, version))
        else:
            wanted.append(("learned", inverse_path, version))
            if self.cfg.recheck_analytic and index == self.cfg.rounds:
                wanted.append(("analytic", None, "analytic"))
        for inverse, weights, tag in wanted:
            base = {"inverse": inverse, "weights": weights, "inverse_version": tag}
            tasks = self.cfg.tasks
            if 0 < index < self.cfg.rounds:
                tasks = tuple(t for t in tasks if t not in self.cfg.sparse_tasks)
            if inverse == "analytic" and index > 0:
                tasks = tuple(t for t in self.cfg.predictor_pairs if t in tasks)
            columns.append({**base, "predictor": "learned" if predictor_path else "analytic",
                            "predictor_weights": predictor_path,
                            "predictor_version": version if predictor_path else "analytic",
                            "tasks": tasks})
            pair = tuple(t for t in self.cfg.predictor_pairs if t in tasks)
            if pair and predictor_path:
                columns.append({**base, "predictor": "analytic", "predictor_weights": None,
                                "predictor_version": "analytic", "tasks": pair})
        return columns

    # -- the loop --------------------------------------------------------------------------------

    def run(self) -> dict:
        """Round 0, then one training round at a time until the budget or `rounds` runs out."""
        self.log(f"campaign {self.cfg.name}: {self.cfg.rounds} rounds, "
                 f"{self.cfg.episodes} episodes per task per round, "
                 f"budget {self.cfg.budget_minutes:.0f} min, source {self.cfg.source}")

        shipped_inverse, shipped_predictor = self.shipped_paths()
        baseline = (shipped_inverse if self.cfg.baseline_learned and shipped_inverse.exists()
                    else None)
        baseline_p = (shipped_predictor if self.cfg.baseline_learned
                      and shipped_predictor.exists() else None)
        records = self.evaluate(0, self.columns_for(0, "shipped", baseline, baseline_p))
        self.score_round(0, "shipped", records)

        for index in range(1, self.cfg.rounds + 1):
            if not self.budget_left(need_minutes=self.cfg.round_minutes):
                self.log(f"budget reached after round {index - 1} "
                         f"({self.elapsed_minutes:.0f} min); stopping")
                break
            self.log(f"--- round {index} ({self.elapsed_minutes:.0f} min elapsed) ---")
            self.perform(index)
            result = self.train(index)
            if result is None:
                break
            self.redeploy(index, result)
            inverse_path, predictor_path, version = self.deployed_paths()
            records = self.evaluate(index, self.columns_for(index, version, inverse_path,
                                                         predictor_path))
            self.score_round(index, version, records, result=result)

        self.summarize()
        self._save()
        return self.state

    # -- scoring ---------------------------------------------------------------------------------

    def score_round(self, index: int, version: str, records: Sequence[EpisodeRecord],
                     result=None) -> RoundRecord:
        record = self._round(index)
        record.version = version or record.version
        record.scores = summarize_records(records)
        if result is not None:
            record.ticks = int(result.ticks)
            record.episodes_logged = int(result.episodes)
            record.metrics = dict(result.metrics)
            record.metrics.update(self._offline_action_error(result))
        for key, entry in self.state["phases"].items():
            if key.startswith(f"r{index}.") and "wall_s" in entry:
                record.wall_s[key.split(".", 1)[1]] = float(entry["wall_s"])
        record.would_ship = self.would_ship(index, record.scores)
        self._put_round(record)
        for name, row in sorted(record.scores.items()):
            self.log(f"  {name:44s} {row['seated_on_target']:>7s}  "
                     f"lateral {row['lateral_mm_median']:.2f} mm  "
                     f"pred {row['median_prediction_mm']:.2f} mm")
        return record

    def _offline_action_error(self, result) -> dict:
        """The learned inverse against the analytic one on held-out ticks of the campaign's log."""
        from ..action.learn import dataset as ds
        from ..action.learn.evaluate import inverse_metrics
        from ..action.models.mlp import Mlp

        data_dirs = sorted(str(p) for p in self.data.glob("r*") if any(p.glob("ep_*.npz")))
        if not data_dirs:
            return {}
        try:
            data = ds.load(data_dirs)
            metrics = inverse_metrics(data, Mlp.load(result.inverse), Mlp.load(result.predictor),
                                      seed=self.cfg.seed)
        except Exception as exc:  # an offline metric must not lose a round
            self.log(f"  offline action error unavailable: {exc}")
            return {}
        return {"action_rmse_mm_x": metrics["action_rmse"][0],
                "action_rmse_mm_y": metrics["action_rmse"][1],
                "action_rmse_deg": metrics["action_rmse"][2],
                "analytic_rms_mm_x": metrics["analytic_rms"][0],
                "analytic_rms_mm_y": metrics["analytic_rms"][1],
                "analytic_rms_deg": metrics["analytic_rms"][2],
                "action_held_out_ticks": metrics["n_held_out"]}

    def would_ship(self, index: int, scores: dict) -> dict:
        """What `rounds.deploy_if_better` would have decided, on the tasks this round measured.

        The gate the shipped pair is behind: the learned inverse has to match the analytic one on
        every task, not win on one and lose on another. The analytic baseline is the fixed one
        from round 0, whatever round the learned column comes from.
        """
        from ..action.learn.rounds import seated_fraction

        baseline = self._round(0).scores
        out: dict = {"tasks": {}, "ship": True, "version": self._round(index).version}
        for key, row in scores.items():
            if row["inverse"] != "learned" or row["predictor"] == "analytic":
                continue
            task = row["task"]
            against = next((r for k, r in baseline.items()
                            if r["task"] == task and r["inverse"] == "analytic"), None)
            if against is None:
                continue
            # `summarize_records` names the count `seated_on_target`; the deployment gate reads
            # `seated_on_target_cell`.
            learned = seated_fraction({"seated_on_target_cell": row["seated_on_target"]})
            analytic = seated_fraction({"seated_on_target_cell": against["seated_on_target"]})
            out["tasks"][task] = {"learned": row["seated_on_target"],
                                  "analytic": against["seated_on_target"],
                                  "meets": learned >= analytic}
            out["ship"] &= learned >= analytic
            del key
        out["ship"] = bool(out["ship"] and out["tasks"])
        return out

    def summarize(self) -> Path:
        """One CSV over the whole campaign: a row per round, task, inverse and prediction rule."""
        rows = []
        for index in sorted(int(k) for k in self.state["rounds"]):
            record = self._round(index)
            # Recomputed from the stored scores, so a round scored before a gate fix reads right.
            ship = self.would_ship(index, record.scores)
            self.state["rounds"][str(index)]["would_ship"] = ship
            for name, row in sorted(record.scores.items()):
                would = ""
                if row["inverse"] == "learned":
                    entry = (ship.get("tasks") or {}).get(row["task"])
                    if entry is not None:
                        would = ("yes" if ship.get("ship") else
                                 ("meets" if entry["meets"] else "no"))
                rows.append({"round": index, "version": record.version,
                             "ticks_trained_on": record.ticks,
                             "episodes_trained_on": record.episodes_logged,
                             "would_ship": would, **row})
        out = self.dir / "summary.csv"
        if rows:
            with out.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        episodes = self.dir / "episodes.csv"
        every: list[EpisodeRecord] = []
        for path in sorted(self.rounds_dir.glob("r*/*.csv")):
            every.extend(_read_csv(path))
        if every:
            _write_csv(episodes, every)
        return out


# ------------------------------------------------------------------------------------ summaries ---


def summarize_records(records: Sequence[EpisodeRecord]) -> dict:
    """Group the episodes by task, inverse and prediction rule, and score each group.

    The success rate is "every brick the task asked for, seated on the cell it was sent to",
    with its Wilson interval. The lateral error is over the bricks that seated on their cell,
    because a brick that missed has no meaningful distance to the cell it missed.
    """
    import statistics

    groups: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        key = f"{record.task} | {record.inverse} | F {record.predictor}"
        groups.setdefault(key, []).append(record)

    out = {}
    for key, rows in groups.items():
        ok = sum(r.success for r in rows)
        interval = wilson_interval(ok, len(rows))
        bricks = [b for r in rows for b in r.bricks]
        seated = sorted(b.lateral_mm for b in bricks if b.on_target)
        tracking = [r.median_tracking_mm for r in rows
                    if r.median_tracking_mm == r.median_tracking_mm]
        prediction = [r.median_prediction_mm for r in rows
                      if r.median_prediction_mm == r.median_prediction_mm]
        dwell = _dwell_ticks(rows)
        failures: dict[str, int] = {}
        for r in rows:
            if r.failure:
                failures[r.failure] = failures.get(r.failure, 0) + 1
        out[key] = {
            "task": rows[0].task, "inverse": rows[0].inverse, "predictor": rows[0].predictor,
            "inverse_version": rows[0].inverse_version,
            "predictor_version": rows[0].predictor_version,
            "episodes": len(rows),
            "seated_on_target": f"{ok}/{len(rows)}",
            "success_rate": interval.p_hat,
            "wilson_low": interval.low, "wilson_high": interval.high,
            "bricks_on_target": f"{sum(b.on_target for b in bricks)}/{len(bricks)}",
            "lateral_mm_median": float(statistics.median(seated)) if seated else float("nan"),
            "lateral_mm_min": seated[0] if seated else float("nan"),
            "lateral_mm_max": seated[-1] if seated else float("nan"),
            "median_tracking_mm": float(statistics.median(tracking)) if tracking
            else float("nan"),
            "median_prediction_mm": float(statistics.median(prediction)) if prediction
            else float("nan"),
            "ticks_align": dwell.get("Align", float("nan")),
            "ticks_approach": dwell.get("Approach", float("nan")),
            "replanned": sum(r.replanned for r in rows),
            "damage": ";".join(sorted({r.damage_state for r in rows if r.damage_state
                                       and r.damage_state != "intact"})) or "intact",
            "failures": ";".join(f"{k}={v}" for k, v in sorted(failures.items())),
            "wall_s_median": float(statistics.median([r.wall_s for r in rows])),
        }
    return out


def _dwell_ticks(rows: Sequence[EpisodeRecord]) -> dict:
    import statistics

    out: dict[str, float] = {}
    for name in ("Align", "Approach"):
        values = [r.ticks_per_step.get(name, 0) for r in rows if r.ticks_per_step.get(name)]
        if values:
            out[name] = float(statistics.median(values))
    return out


# ----------------------------------------------------------------------------------------- csv ---


def _write_csv(path: Path, records: Sequence[EpisodeRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.row())
    return path


def _read_csv(path: Path) -> list[EpisodeRecord]:
    """Read a per-episode CSV back into records, enough of them for the summaries."""
    if not Path(path).exists():
        return []
    out = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            record = EpisodeRecord(
                task=row["task"], seed=int(row["seed"]), phase=row["phase"],
                round_index=int(row["round_index"]), inverse=row["inverse"],
                predictor=row["predictor"], inverse_version=row["inverse_version"],
                predictor_version=row["predictor_version"], source=row["source"],
                failure=row["failure"], damage_state=row["damage_state"],
                median_tracking_mm=_f(row["median_tracking_mm"]),
                median_prediction_mm=_f(row["median_prediction_mm"]),
                peak_wall_stress_MPa=_f(row["peak_wall_stress_MPa"]),
                vision_ticks=int(row["vision_ticks"] or 0), wall_s=_f(row["wall_s"]),
                replanned=row["replanned"] in ("True", "true", "1"))
            for i in range(int(row["n_bricks"] or 0)):
                # Two numbers on the plate, three when the cell names a layer.
                cell = row[f"brick{i}_cell"].split(",")
                record.bricks.append(BrickRecord(
                    index=i, cell=tuple(int(v) for v in cell),
                    seated=row[f"brick{i}_seated"] == "True",
                    at_target_cell=row[f"brick{i}_at_cell"] == "True", connected=False,
                    # A file written before the column existed scored no layer: read as true.
                    at_target_layer=row.get(f"brick{i}_at_layer", "") in ("", "True"),
                    lateral_mm=_f(row[f"brick{i}_lateral_mm"]),
                    depth_error_mm=_f(row[f"brick{i}_depth_error_mm"]), tilt_deg=float("nan"),
                    replanned=row[f"brick{i}_replanned"] == "True"))
            for name in ("Approach", "Grasp", "Lift", "Transport", "Align",
                         "Press the brick down", "Release", "Retract", "Verify"):
                value = int(row.get(f"ticks_{name.replace(' ', '_')}", 0) or 0)
                if value:
                    record.ticks_per_step[name] = value
            out.append(record)
    return out


def _record_from(raw: dict) -> EpisodeRecord:
    bricks = [BrickRecord(**{**b, "cell": tuple(b["cell"])}) for b in raw.pop("bricks", [])]
    record = EpisodeRecord(**{k: v for k, v in raw.items() if k != "bricks"})
    record.bricks = bricks
    return record


def _f(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clear(cfg: CampaignConfig) -> None:
    """Delete a campaign's directory. Only what this package wrote, and never the weights the
    repository ships."""
    path = Path(cfg.root) / cfg.name
    if path.exists():
        shutil.rmtree(path)
