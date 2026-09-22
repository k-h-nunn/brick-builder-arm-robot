"""One round of "perform the task, train `F` and `G` on what happened, redeploy, measure".

This is the stable entry point a campaign driver calls. It owns nothing the rest of the package
does not already do -- `collect.py` logs tuples, `train.py` fits the two models, `evaluate.py`
scores the closed loop -- and adds the three things a repeated loop needs: version ids, a record
of what is deployed, and a gate that refuses to deploy a round that did not earn it.

The round
---------

1. **Perform.** `collect_round` runs task episodes with whichever models are deployed and logs one
   `.npz` per episode. Driving with the learned inverse and keeping the analytic inverse's answer
   as a label is the DAgger relabelling: the states are the learned policy's own, and the teacher
   says what it would have done there.
2. **Train.** `train_round` fits `F` on next-state prediction conditioned on the mechanical state
   and the action, then fits `G` on the difference to the target with three terms -- hindsight
   targets, the distal teacher through the `F` just trained, and regression on the analytic
   inverse's answer at the logged states.
3. **Redeploy.** `deploy` copies a round's two checkpoints onto the pair the controller loads and
   writes `deployed.json` beside them. `deploy_if_better` does it only when the round's own
   closed-loop measurement is at least as good as what is deployed, on both tasks.
4. **Measure.** `measure_round` runs the published closed-loop table for a version.

File layout
-----------

    data_cache/action/<set>/ep_*.npz                  logged episodes, one file per episode
    data_cache/action/rounds/<version>/predictor.npz  the round's own checkpoints
    data_cache/action/rounds/<version>/inverse.npz
    data_cache/action/rounds/<version>/report.json    config, data, held-out losses, wall clock
    data_cache/action/rounds/<version>/state/*.pt     optimizer state, for resuming
    brick_builder/action/weights/predictor.npz        what the loop loads today
    brick_builder/action/weights/inverse.npz
    brick_builder/action/weights/deployed.json        which version those two are, and its scores

Everything under `data_cache/` is git-ignored and rebuildable; the two deployed checkpoints and
the one JSON record are what ships.

Determinism and resuming
------------------------

A version id is a hash of the datasets and the hyperparameters that decide the fit, so the same
call is the same round: it writes to the same directory, resumes from the same optimizer state,
and -- given the same `seed` -- produces the same weights. A round stopped by its own `minutes`
budget continues from the epoch it reached when called again.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from ..models.inverse import WEIGHTS
from ..models.mlp import Mlp
from . import dataset as ds
from .train import TrainConfig, train_inverse, train_predictor

__all__ = ["RoundConfig", "RoundResult", "CollectSpec", "DATA_ROOT", "ROUNDS", "DEPLOYED",
           "collect_round", "train_round", "measure_round", "deploy", "deploy_if_better",
           "deployed", "versions", "load_round"]

DATA_ROOT = Path("data_cache/action")
"""Where logged episodes and round checkpoints live. Git-ignored and rebuildable."""

ROUNDS = DATA_ROOT / "rounds"
DEPLOYED = WEIGHTS / "deployed.json"
"""The record of which round the two shipped checkpoints came from."""

TASKS = ("placement", "pick")


# ------------------------------------------------------------------------------------ perform ---


@dataclass(frozen=True)
class CollectSpec:
    """One batch of task episodes to log."""

    task: str = "placement"
    """`placement` or `pick`."""

    minutes: float = 20.0
    """Wall-clock budget for this batch."""

    inverse: str = "analytic"
    """Which inverse drives the logged runs. `learned` loads the deployed `inverse.npz`, which is
    what makes the round's states the current policy's own -- the DAgger part."""

    seed: int = 0
    """Offsets the collection seeds. Seeds 0-9 are the evaluation table's and are never collected
    from, whatever this is set to (`collect.FIRST_COLLECT_SEED`)."""

    name: str | None = None
    """Directory under `data_cache/action/`. Defaults to `<task>_<inverse>_s<seed>`."""

    @property
    def directory(self) -> Path:
        return DATA_ROOT / (self.name or f"{self.task}_{self.inverse}_s{self.seed}")


def collect_round(specs: Sequence[CollectSpec], *, log=print) -> list[Path]:
    """Run each batch of task episodes and return the directories that were written.

    Appends to a directory that already exists rather than replacing it: a campaign that runs the
    same spec every round accumulates, which is what the hindsight and teacher sets want.
    """
    from .collect import main as collect_main

    out: list[Path] = []
    for spec in specs:
        if spec.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}; got {spec.task!r}")
        spec.directory.mkdir(parents=True, exist_ok=True)
        log(f"collect {spec.task} ({spec.inverse}) -> {spec.directory}")
        collect_main(["--task", spec.task, "--minutes", str(spec.minutes),
                      "--seed", str(spec.seed), "--inverse", spec.inverse,
                      "--out", str(spec.directory)])
        out.append(spec.directory)
    return out


# -------------------------------------------------------------------------------------- train ---


@dataclass(frozen=True)
class RoundConfig:
    """What to train a round on, and how. Everything here goes into the version id."""

    data: tuple[str, ...]
    """Directories of logged episodes, read by `dataset.load`. Order does not matter."""

    seed: int = 0
    minutes: float = 20.0
    """Budget per model. Both are resumable, so a short budget called twice is a long one."""

    epochs: int = 60
    hidden: tuple[int, ...] = (64, 64)
    distal: float = 0.3
    """Weight on the distal-teacher term, which runs `G`'s answer through the round's own `F`."""

    teacher: float = 1.0
    """Weight on regression against the analytic inverse at the logged states -- the relabelling.
    One by default here, because a round is a DAgger round; `train.TrainConfig` defaults it off
    because the first models were trained without it."""

    action_penalty: float = 0.02
    hindsight_repeats: int = 2
    max_ff_mm: float | None = None
    holdout: float = 0.15
    version: str | None = None
    """Override the derived id. Use it to force a fresh directory, never to rename a finished one."""

    train_predictor: bool = True
    train_inverse: bool = True
    """Either model can be skipped; a skipped one is copied from `predictor_from` (or the deployed
    pair) so the round still publishes a complete, self-consistent pair."""

    predictor_from: str | None = None
    """Start from this `F` instead of training one. The distal-teacher term uses whichever `F` the
    round ends up with, so freezing `F` across rounds is a supported and stated choice."""

    root: str = str(ROUNDS)

    def fingerprint(self) -> str:
        """Eight hex digits over everything that decides the fit."""
        payload = json.dumps({
            "data": sorted(str(Path(d)) for d in self.data),
            "seed": self.seed, "epochs": self.epochs, "hidden": list(self.hidden),
            "distal": self.distal, "teacher": self.teacher,
            "action_penalty": self.action_penalty, "repeats": self.hindsight_repeats,
            "max_ff_mm": self.max_ff_mm, "holdout": self.holdout,
            "predictor": self.train_predictor, "inverse": self.train_inverse,
            "predictor_from": self.predictor_from,
        }, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:8]

    def train_config(self) -> TrainConfig:
        return TrainConfig(hidden=tuple(self.hidden), epochs=self.epochs, seed=self.seed,
                           holdout=self.holdout, distal=self.distal,
                           action_penalty=self.action_penalty, teacher=self.teacher,
                           hindsight_repeats=self.hindsight_repeats, max_ff_mm=self.max_ff_mm,
                           minutes=self.minutes)


@dataclass(frozen=True)
class RoundResult:
    """What a round produced. Every path exists by the time this is returned."""

    version: str
    directory: Path
    predictor: Path
    inverse: Path
    report: Path
    ticks: int
    episodes: int
    metrics: dict
    """`predictor_held_out_mse`, `inverse_held_out_mse`, and the two models' shapes and costs."""
    wall_s: float

    def as_dict(self) -> dict:
        return {"version": self.version, "directory": str(self.directory),
                "predictor": str(self.predictor), "inverse": str(self.inverse),
                "report": str(self.report), "ticks": self.ticks, "episodes": self.episodes,
                "metrics": self.metrics, "wall_s": self.wall_s}


def train_round(cfg: RoundConfig, *, log=print) -> RoundResult:
    """Fit `F` then `G` on the logged tuples, and write a versioned pair of checkpoints.

    `F`: `(slate rows, six joint positions, six joint speeds, the tool-target change over the
    tick) -> the change of the same slate rows`, a weighted mean square on the normalized change.

    `G`: `(slate differences, mechanical state) -> the correction increment (dx, dy [m], spin
    [rad])`, three terms at once -- the hindsight target (the increment the controller really
    commanded, against a state drawn one to eight ticks later), the distal teacher (`G`'s own
    answer fed through the frozen `F` and scored against the target state), and the analytic
    inverse's answer at the same tick with weight `teacher`.

    Deterministic given `cfg.seed`; resumable from `<directory>/state/`.
    """
    t0 = time.perf_counter()
    version = cfg.version or f"v{cfg.fingerprint()}"
    directory = Path(cfg.root) / version
    state = directory / "state"
    directory.mkdir(parents=True, exist_ok=True)

    data = ds.load([Path(d) for d in cfg.data])
    episodes = int(data.episode.max() + 1)
    log(f"round {version}: {data.n} ticks from {episodes} episodes, "
        f"{len(cfg.data)} set(s) -> {directory}")

    train_cfg = cfg.train_config()
    p_path, i_path = directory / "predictor.npz", directory / "inverse.npz"

    if cfg.train_predictor:
        predictor = train_predictor(data, train_cfg, p_path, resume=state / "predictor.pt",
                                    log=log)
    else:
        source = Path(cfg.predictor_from) if cfg.predictor_from else WEIGHTS / "predictor.npz"
        shutil.copyfile(source, p_path)
        predictor = Mlp.load(p_path)
        log(f"round {version}: predictor carried over from {source}")

    if cfg.train_inverse:
        inverse = train_inverse(data, predictor, train_cfg, i_path, resume=state / "inverse.pt",
                                log=log)
    else:
        shutil.copyfile(WEIGHTS / "inverse.npz", i_path)
        inverse = Mlp.load(i_path)
        log(f"round {version}: inverse carried over from the deployed checkpoint")

    metrics = {
        "predictor_held_out_mse": float(predictor.meta.get("held_out_mse", float("nan"))),
        "inverse_held_out_mse": float(inverse.meta.get("held_out_mse", float("nan"))),
        "predictor_sizes": list(predictor.sizes), "inverse_sizes": list(inverse.sizes),
        "predictor_macs": predictor.macs, "inverse_macs": inverse.macs,
        "mm_per_centre_unit": ds.mm_per_centre_unit(data),
    }
    result = RoundResult(version=version, directory=directory, predictor=p_path, inverse=i_path,
                         report=directory / "report.json", ticks=int(data.n), episodes=episodes,
                         metrics=metrics, wall_s=float(time.perf_counter() - t0))
    result.report.write_text(json.dumps(
        {"version": version, "config": asdict(cfg), **result.as_dict()},
        indent=2, default=float) + "\n")
    return result


def load_round(version: str, root: Path | str = ROUNDS) -> RoundResult:
    """Read a finished round back by its version id."""
    report = Path(root) / version / "report.json"
    if not report.exists():
        raise FileNotFoundError(f"no round {version!r} under {root}")
    data = json.loads(report.read_text())
    return RoundResult(version=data["version"], directory=Path(data["directory"]),
                       predictor=Path(data["predictor"]), inverse=Path(data["inverse"]),
                       report=report, ticks=int(data["ticks"]), episodes=int(data["episodes"]),
                       metrics=dict(data["metrics"]), wall_s=float(data["wall_s"]))


def versions(root: Path | str = ROUNDS) -> list[str]:
    """Every finished round under `root`, oldest first by write time."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = [p.parent for p in root.glob("*/report.json")]
    return [p.name for p in sorted(found, key=lambda p: (p / "report.json").stat().st_mtime)]


# ------------------------------------------------------------------------------------ measure ---


def measure_round(result: RoundResult | None, seeds: Sequence[int] = range(10), *,
                  tasks: Sequence[str] = TASKS, log=print) -> dict:
    """The published closed-loop table for a round, against the analytic inverse on the same seeds.

    `None` measures whatever is deployed. Returns `{task: {variant: summary}}` with the summaries
    `evaluate.compare` builds -- seated on the target cell, the lateral range, the median tracking
    and prediction errors and the ticks to converge at each dwell.
    """
    from .evaluate import compare

    weights = None if result is None else result.inverse
    variants = [{"name": "analytic", "inverse": "analytic"},
                {"name": "learned", "inverse": "learned", "weights": weights}]
    out = {}
    for task in tasks:
        report = compare(task, list(seeds), variants, log=log)
        out[task] = {name: {k: v for k, v in col.items() if k != "rows"}
                     for name, col in report["columns"].items()}
    return out


# ----------------------------------------------------------------------------------- redeploy ---


def deployed(weights: Path | str = WEIGHTS) -> dict | None:
    """The record of which round the two shipped checkpoints came from, or `None`."""
    path = Path(weights) / DEPLOYED.name
    return json.loads(path.read_text()) if path.exists() else None


def deploy(result: RoundResult, *, weights: Path | str = WEIGHTS, scores: dict | None = None,
           note: str = "") -> dict:
    """Copy a round's checkpoints onto the pair the controller loads, and record which round it is.

    The only place the shipped weights are written. `scores` is whatever `measure_round` returned,
    kept beside them so the deployed pair carries the measurement that justified it.
    """
    weights = Path(weights)
    weights.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.predictor, weights / "predictor.npz")
    shutil.copyfile(result.inverse, weights / "inverse.npz")
    record = {"version": result.version, "ticks": result.ticks, "episodes": result.episodes,
              "metrics": result.metrics, "scores": scores or {}, "note": note}
    (weights / DEPLOYED.name).write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def seated_fraction(summary: dict) -> float:
    """`"7/10"` -> 0.7. The one number the deployment gate reads."""
    text = str(summary.get("seated_on_target_cell", "0/0"))
    top, _, bottom = text.partition("/")
    return float(top) / float(bottom) if float(bottom) else 0.0


def deploy_if_better(result: RoundResult, scores: dict, *, weights: Path | str = WEIGHTS,
                     tasks: Sequence[str] = TASKS, log=print) -> dict | None:
    """Deploy only when the round's learned inverse matches the analytic one on **every** task.

    The bar the project set and the reason the default is still the analytic inverse: a round that
    wins on placement and loses on the pick has not earned the default. Returns the deployment
    record, or `None` with the reason logged.
    """
    for task in tasks:
        column = scores.get(task, {})
        learned = seated_fraction(column.get("learned", {}))
        analytic = seated_fraction(column.get("analytic", {}))
        if learned < analytic:
            log(f"round {result.version} not deployed: {task} seated {learned:.0%} "
                f"against the analytic inverse's {analytic:.0%}")
            return None
    note = "; ".join(
        f"{task}: learned {scores[task]['learned']['seated_on_target_cell']} "
        f"against analytic {scores[task]['analytic']['seated_on_target_cell']}"
        for task in tasks if task in scores)
    log(f"round {result.version} deployed ({note})")
    return deploy(result, weights=weights, scores=scores, note=note)
