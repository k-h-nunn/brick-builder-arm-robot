"""The tasks the campaign performs, and one episode of each, scored.

Three tasks, all closed loop and all scored by `control.pick_place.verify_placement` reading the
simulator after the arm has left:

| Task | Starts | Unknown to the controller |
|---|---|---|
| `placement` | brick already held | baseplate 4-8 mm, 2-3 deg |
| `pick_and_place` | brick loose on the table | the above, and the brick 8-15 mm, 10-20 deg |
| `build_2` | as above, twice | the above, per cycle |

`build_2` is one English command compiled by `brick_builder.language` into two placements and
executed as consecutive pick-and-place cycles. The goal's expected grid is checked after each
cycle and `Goal.remaining_steps` decides what is left to do, which is what lets a failed
placement be re-planned once.

Seeds 0-9 are the held-out evaluation set for every task and are never performed on; the
performing rounds draw from a stream that starts far above them (`FIRST_PERFORM_SEED`).
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..action.worlds import (
    DISTRACTOR, NOMINAL_PLATE_XY, REF_RED, REF_YELLOW, SPARE_BODY, SUPPLY_BODY, SUPPLY_XY,
    TARGET_CELL, WANTED, brick_displacement, displacement, nudge_brick, scene_cfg,
)
from ..bricks.structures import Placement
from ..scene.compose import HeldBrick, build_scene, reset_scene

EVAL_SEEDS = tuple(range(10))
"""The held-out evaluation set, shared by every task. Never performed on, never trained on."""

FIRST_PERFORM_SEED = 300_000
"""Where the performing rounds' fresh-seed stream starts. Far above the evaluation seeds and
above the collection seeds `action.learn.collect` uses, so no world is ever both."""

BUILD_COMMAND = ("put a blue 2x2 at column 5 row 4 "
                 "then put a blue 2x2 next to the blue brick")
"""The whole of `build_2`, in English. It compiles to two 2x2 blue placements on the bottom
layer, at grid cells (4, 3) and (6, 3). The compiler has no pronouns, so the second brick names
the first by colour rather than saying "on top of it"; and both bricks are blue 2x2 because that
is what the supply row offers."""

BASE_PLACED = (REF_YELLOW, REF_RED)
"""What is on the plate before the campaign starts: the two bricks the loop lines itself up on."""

STEP_ORDER = ("Approach", "Grasp", "Lift", "Transport", "Align",
              "Press the brick down", "Release", "Retract", "Verify")


# ------------------------------------------------------------------------------------- the set ---


@dataclass(frozen=True)
class TaskSpec:
    """One task: how many cycles, whether each fetches its brick, and where the bricks go."""

    name: str
    pick: bool
    cells: tuple[tuple[int, int], ...]
    """Target cells, minimum corner in integer studs, one per cycle."""

    command: str | None = None
    """The English command the cells came from, for the tasks that have one."""

    @property
    def cycles(self) -> int:
        return len(self.cells)


def build_goal(command: str = BUILD_COMMAND):
    """Compile the build command against the plate the campaign starts from.

    Raises `ValueError` if the compiler refuses it, because a task whose command does not compile
    is not a task.
    """
    from ..language import compile_command

    result = compile_command(command, current_grid=list(BASE_PLACED))
    if not result.understood or result.goal is None:
        raise ValueError(f"the build command did not compile: {result.problems}")
    return result.goal


def _build_cells() -> tuple[tuple[int, int], ...]:
    return tuple((p.x, p.y) for p in build_goal().placements)


TASKS: dict[str, TaskSpec] = {
    "placement": TaskSpec("placement", pick=False, cells=(TARGET_CELL,)),
    "pick_and_place": TaskSpec("pick_and_place", pick=True, cells=(TARGET_CELL,)),
    "build_2": TaskSpec("build_2", pick=True, cells=_build_cells(), command=BUILD_COMMAND),
}


def perform_seeds(task: str, round_index: int, n: int) -> list[int]:
    """`n` fresh seeds for one task in one round. Deterministic, and disjoint from everything."""
    base = FIRST_PERFORM_SEED + 100_000 * (1 + list(TASKS).index(task)) + 1_000 * round_index
    return [base + i for i in range(n)]


# ------------------------------------------------------------------------------------- records ---


@dataclass
class BrickRecord:
    """One brick of one episode, as the simulator reports it after the arm has left."""

    index: int
    cell: tuple[int, int]
    seated: bool
    at_target_cell: bool
    connected: bool
    lateral_mm: float
    depth_error_mm: float
    tilt_deg: float
    replanned: bool = False
    """True when this brick's cycle was run a second time after `remaining_steps` still listed it."""

    @property
    def on_target(self) -> bool:
        """Seated on the cell it was sent to: the campaign's unit of success."""
        return bool(self.seated and self.at_target_cell)


@dataclass
class EpisodeRecord:
    """One episode of one task, whatever it was run for."""

    task: str
    seed: int
    phase: str
    """`perform` or `evaluate`."""

    round_index: int
    inverse: str
    predictor: str
    inverse_version: str
    predictor_version: str
    source: str
    bricks: list[BrickRecord] = field(default_factory=list)
    ticks_per_step: dict[str, int] = field(default_factory=dict)
    failure: str = ""
    """The first failure any cycle reported, normalized (`failure_class`)."""

    median_tracking_mm: float = float("nan")
    median_prediction_mm: float = float("nan")
    damage_state: str = ""
    peak_wall_stress_MPa: float = float("nan")
    vision_ticks: int = 0
    wall_s: float = 0.0
    replanned: bool = False

    @property
    def success(self) -> bool:
        """Every brick the task asked for, seated on the cell it was sent to."""
        return bool(self.bricks) and all(b.on_target for b in self.bricks)

    def row(self) -> dict:
        """One flat CSV row. Per-brick fields are numbered, so a task with fewer bricks
        leaves the higher columns empty rather than changing the header."""
        row = {k: v for k, v in asdict(self).items()
               if k not in ("bricks", "ticks_per_step")}
        row["success"] = self.success
        row["n_bricks"] = len(self.bricks)
        for i in range(2):
            b = self.bricks[i] if i < len(self.bricks) else None
            row[f"brick{i}_cell"] = "" if b is None else f"{b.cell[0]},{b.cell[1]}"
            row[f"brick{i}_on_target"] = "" if b is None else b.on_target
            row[f"brick{i}_seated"] = "" if b is None else b.seated
            row[f"brick{i}_at_cell"] = "" if b is None else b.at_target_cell
            row[f"brick{i}_lateral_mm"] = "" if b is None else round(b.lateral_mm, 4)
            row[f"brick{i}_depth_error_mm"] = "" if b is None else round(b.depth_error_mm, 4)
            row[f"brick{i}_replanned"] = "" if b is None else b.replanned
        for name in STEP_ORDER:
            row[f"ticks_{name.replace(' ', '_')}"] = self.ticks_per_step.get(name, 0)
        return row


CSV_FIELDS = tuple(EpisodeRecord(
    task="", seed=0, phase="", round_index=0, inverse="", predictor="",
    inverse_version="", predictor_version="", source="",
    bricks=[BrickRecord(0, (0, 0), False, False, False, 0.0, 0.0, 0.0)]).row())
"""The CSV header, derived from the record so the two cannot drift apart."""


def failure_class(episode_failure: str, bricks: list[BrickRecord]) -> str:
    """The run's failure in one of the classes the closed-loop page names.

    The loop reports the ones it can see -- a stall, a grasp that stopped short, a last frame
    outside tolerance. What it cannot see is where the brick ended up, so the scorer's answer is
    folded in here: on the wrong cell, or on the right one but not pressed home.
    """
    text = (episode_failure or "").strip()
    if text:
        if "grasp" in text:
            return "grasp short"
        if "nothing to line up on" in text or "blind" in text:
            return "stalled blind"
        if "held" in text or "stall" in text:
            return "stalled with an error it can see"
        if "tolerance" in text:
            return "finished outside tolerance"
        return text
    for b in bricks:
        if not b.at_target_cell:
            return "seated on the wrong cell"
        if not b.seated:
            return "partial seat"
    return ""


# -------------------------------------------------------------------------------------- worlds ---


def cycle_world(seed: int, cycle: int, placed):
    """The world one cycle of a fetching task really runs in.

    The same construction `action.worlds.build_pick_world` makes -- the plate 4-8 mm and 2-3
    degrees from nominal and the brick to pick 8-15 mm and 10-20 degrees off the supply row --
    with whatever the earlier cycles already put on the plate added to it. The plate's
    displacement is the episode's, because it is one baseplate; the brick's is drawn per cycle,
    because it is a different brick each time.
    """
    d_xy, d_yaw = displacement(seed)
    cfg = scene_cfg(
        plate_xy=np.asarray(NOMINAL_PLATE_XY) + d_xy, plate_yaw=d_yaw,
        placed=list(placed), held=None,
        home_tcp=(NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075), cameras=(),
    )
    cfg.supply = (WANTED, DISTRACTOR)
    cfg.supply_xy = SUPPLY_XY
    scene = build_scene(cfg)
    reset_scene(scene, np.random.default_rng(seed + 10_000 * cycle))
    b_xy, b_yaw = brick_displacement(seed + 10_000 * cycle)
    nudge_brick(scene, SUPPLY_BODY, b_xy, b_yaw)
    return scene, d_xy, d_yaw, b_xy


def held_world(seed: int, placed):
    """The placement task's world: the brick already in the jaws, the plate displaced."""
    d_xy, d_yaw = displacement(seed)
    held = HeldBrick(nx=2, ny=2, colour="blue", offset_jitter=(0.0, 0.002, 0.0))
    cfg = scene_cfg(
        plate_xy=np.asarray(NOMINAL_PLATE_XY) + d_xy, plate_yaw=d_yaw,
        placed=list(placed), held=held,
        home_tcp=(NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075), cameras=(),
    )
    scene = build_scene(cfg)
    reset_scene(scene, np.random.default_rng(seed))
    return scene, d_xy, d_yaw, np.zeros(2)


# -------------------------------------------------------------------- the recorded target sequence ---


def ideal_key(spec: TaskSpec, cycle: int, seed: int, source: str) -> str:
    return f"{spec.name}_c{cycle}_s{seed}_{source}"


def record_for(spec: TaskSpec, cycle: int, seed: int, *, source=None, cache: Path | None = None):
    """The target sequence for one cycle, recorded in the nominal world or read from the cache.

    The recording is feed-forward and nominal: it depends on the task, the target cell, what is
    already on the plate and the seed the adapter's own noise is drawn from, and on nothing the
    models do. That is what makes it safe to record once and reuse it for every round and every
    inverse -- and necessary, because a column measured against a different recording is not
    measuring the inverse.
    """
    import pickle

    from ..action.place import PlaceConfig, record_ideal

    name = ideal_key(spec, cycle, seed, "ideal" if source is None else "cnn")
    path = None if cache is None else Path(cache) / f"{name}.pkl"
    if path is not None and path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)

    cfg = PlaceConfig(pick=spec.pick)
    placed = list(BASE_PLACED) + [Placement(h=2, w=2, x=c[0], y=c[1], z=0, colour="blue")
                                  for c in spec.cells[:cycle]]
    ideal, world = record_ideal(cfg, seed=seed, target=spec.cells[cycle], source=source,
                                placed=placed)
    world.close()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(ideal, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    return ideal


# --------------------------------------------------------------------------------- one episode ---


def _models(inverse: str, weights: Path | None, predictor: str, predictor_weights: Path | None):
    """Load whichever of the two learned models this run drives or reports with."""
    g = p = None
    if inverse == "learned":
        from ..action.models.inverse import LearnedInverse

        g = LearnedInverse.load(weights)
    if predictor == "learned":
        from ..action.models.predictor import LearnedPredictor

        p = LearnedPredictor.load(predictor_weights)
    return g, p


def _episode_kwargs(predictor_model) -> dict:
    """`TrackingEpisode` grew a `predictor` argument; stay callable on a tree that has not."""
    import inspect

    from ..action.place import TrackingEpisode

    if predictor_model is None:
        return {}
    if "predictor" in inspect.signature(TrackingEpisode.__init__).parameters:
        return {"predictor": predictor_model}
    return {}


def run_cycle(spec: TaskSpec, seed: int, cycle: int, placed, *, inverse: str = "analytic",
              weights: Path | None = None, predictor: str = "analytic",
              predictor_weights: Path | None = None, source=None, ideal=None,
              observer=None) -> dict:
    """One closed-loop pick-and-place (or placement), scored.

    Returns the brick's check, the per-step vision-tick counts, the error series and the damage
    the monitor saw, with the scene already closed.
    """
    from ..action.place import PlaceConfig, TrackingEpisode
    from ..control.pick_place import verify_placement

    cfg = PlaceConfig(pick=spec.pick, inverse=inverse, predictor=predictor)
    g, p = _models(inverse, weights, predictor, predictor_weights)
    cell = spec.cells[cycle]

    if spec.pick:
        scene, d_xy, d_yaw, b_xy = cycle_world(seed, cycle, placed)
        brick = SUPPLY_BODY
    else:
        scene, d_xy, d_yaw, b_xy = held_world(seed, placed)
        brick = scene.held_name

    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, source=source, inverse=g,
                              observer=observer, **_episode_kwargs(p))
    t0 = time.perf_counter()
    try:
        for _ in episode.run():
            pass
        centre = (cell[0] + 0.5, cell[1] + 0.5, 0)
        check = verify_placement(scene, brick, centre)
        monitor = getattr(scene, "monitor", None)
        damage = "" if monitor is None else str(monitor.worst_state_seen)
        stress = (float("nan") if monitor is None
                  else 1e-6 * float(monitor.peak_wall_stress_Pa))
        ticks = {}
        for report in episode.reports:
            ticks[report.name] = ticks.get(report.name, 0) + int(report.vision_ticks)
        errs = [e for e in episode.errors if e == e]
        preds = [e for e in episode.pred_errors if e == e]
        out = {
            "check": check, "ticks_per_step": ticks,
            "failure": episode.failure or "",
            "median_tracking_mm": float(statistics.median(errs)) if errs else float("nan"),
            "median_prediction_mm": float(statistics.median(preds)) if preds else float("nan"),
            "vision_ticks": len(episode.errors),
            "damage_state": damage, "peak_wall_stress_MPa": stress,
            "plate_offset_mm": float(1e3 * np.linalg.norm(d_xy)),
            "plate_yaw_deg": float(math.degrees(d_yaw)),
            "brick_offset_mm": float(1e3 * np.linalg.norm(b_xy)),
            "wall_s": float(time.perf_counter() - t0),
        }
    finally:
        episode.close()
    del episode, scene
    gc.collect()
    return out


def run_episode(spec: TaskSpec, seed: int, *, phase: str = "evaluate", round_index: int = 0,
                inverse: str = "analytic", weights: Path | None = None,
                predictor: str = "analytic", predictor_weights: Path | None = None,
                inverse_version: str = "", predictor_version: str = "",
                source=None, source_name: str = "ideal", cache: Path | None = None,
                tuples: Path | None = None, tuple_prefix: str = "ep",
                replan: bool = True, ideal_seed: int | None = None,
                log=None) -> EpisodeRecord:
    """One whole episode of one task: every cycle the task asks for, scored after each.

    After a cycle the goal's remaining steps are recomputed from what is really on the plate. A
    brick that did not land on its cell leaves its own step outstanding, and that cycle is run
    once more before the build moves on; a second failure ends the episode with the brick
    recorded as it landed.

    `ideal_seed` records the target sequence from a different seed than the world is built from.
    The recording is nominal, so a pool of a few sequences serves any number of performing
    episodes -- what varies between them is the world, not the reference. The evaluation leaves
    it at `None`, one recording per seed, which is what the published tables do.
    """
    from ..action.learn.collect import Collector
    from ..language.goal import Goal

    goal = (build_goal(spec.command) if spec.command
            else Goal.from_placements([Placement(h=2, w=2, x=c[0], y=c[1], z=0, colour="blue")
                                       for c in spec.cells]))
    record = EpisodeRecord(
        task=spec.name, seed=seed, phase=phase, round_index=round_index, inverse=inverse,
        predictor=predictor, inverse_version=inverse_version,
        predictor_version=predictor_version, source=source_name)

    placed = list(BASE_PLACED)
    observed: list[Placement] = []
    tracking: list[float] = []
    prediction: list[float] = []
    t0 = time.perf_counter()

    for cycle in range(spec.cycles):
        ideal = record_for(spec, cycle, seed if ideal_seed is None else ideal_seed,
                           source=source, cache=cache)
        attempts = 0
        result = None
        while attempts < (2 if replan else 1):
            collector = None
            if tuples is not None:
                collector = Collector(rng=np.random.default_rng(seed + 977 * cycle + attempts))
            result = run_cycle(spec, seed, cycle, placed, inverse=inverse, weights=weights,
                               predictor=predictor, predictor_weights=predictor_weights,
                               source=source, ideal=ideal, observer=collector)
            if collector is not None and collector.n_ticks():
                _write_tuples(tuples, tuple_prefix, spec, seed, cycle, attempts, collector,
                              result)
            attempts += 1
            check = result["check"]
            cell = spec.cells[cycle]
            landed = Placement(h=2, w=2, x=cell[0], y=cell[1], z=0, colour="blue")
            done = bool(check.seated and check.at_target_cell)
            if done or not replan:
                break
            # `remaining_steps` reads the plate, not the run: a brick that is not where the goal
            # wants it leaves its step outstanding, and that is what asks for the second try.
            left = goal.remaining_steps(observed + ([landed] if done else []))
            if log is not None:
                log(f"    {spec.name} seed {seed} cycle {cycle}: "
                    f"{len(left)} step(s) outstanding, re-planning")
            if not any(s.placement.x == cell[0] and s.placement.y == cell[1] for s in left):
                break

        check = result["check"]
        on_target = bool(check.seated and check.at_target_cell)
        record.bricks.append(BrickRecord(
            index=cycle, cell=spec.cells[cycle], seated=bool(check.seated),
            at_target_cell=bool(check.at_target_cell), connected=bool(check.connected),
            lateral_mm=float(check.lateral_mm), depth_error_mm=float(check.depth_error_mm),
            tilt_deg=float(check.tilt_deg), replanned=attempts > 1))
        record.replanned |= attempts > 1
        for name, n in result["ticks_per_step"].items():
            record.ticks_per_step[name] = record.ticks_per_step.get(name, 0) + n
        record.vision_ticks += int(result["vision_ticks"])
        if result["median_tracking_mm"] == result["median_tracking_mm"]:
            tracking.append(result["median_tracking_mm"])
        if result["median_prediction_mm"] == result["median_prediction_mm"]:
            prediction.append(result["median_prediction_mm"])
        if result["failure"] and not record.failure:
            record.failure = result["failure"]
        if result["damage_state"] and result["damage_state"] != "intact":
            record.damage_state = result["damage_state"]
        elif not record.damage_state:
            record.damage_state = result["damage_state"]
        stress = result["peak_wall_stress_MPa"]
        if stress == stress:
            record.peak_wall_stress_MPa = (
                stress if record.peak_wall_stress_MPa != record.peak_wall_stress_MPa
                else max(record.peak_wall_stress_MPa, stress))

        if on_target:
            cell = spec.cells[cycle]
            observed.append(Placement(h=2, w=2, x=cell[0], y=cell[1], z=0, colour="blue"))
            placed = list(BASE_PLACED) + list(observed)
        elif cycle + 1 < spec.cycles:
            # The next cycle still runs -- the build is scored brick by brick -- but it runs on
            # a plate that does not carry this one, because this one is not on its cell.
            placed = list(BASE_PLACED) + list(observed)

    record.median_tracking_mm = (float(statistics.median(tracking)) if tracking
                                 else float("nan"))
    record.median_prediction_mm = (float(statistics.median(prediction)) if prediction
                                   else float("nan"))
    record.failure = failure_class(record.failure, record.bricks)
    record.wall_s = float(time.perf_counter() - t0)
    return record


def _write_tuples(root: Path, prefix: str, spec: TaskSpec, seed: int, cycle: int, attempt: int,
                  collector, result: dict) -> Path:
    """One `.npz` per cycle, in the layout `action.learn.dataset` reads."""
    check = result["check"]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = collector.arrays(
        seed=np.int64(seed), mode=np.asarray("clean", dtype="<U8"),
        task=np.asarray(spec.name, dtype="<U12"),
        plate_offset_mm=np.float32(result["plate_offset_mm"]),
        plate_yaw_deg=np.float32(result["plate_yaw_deg"]),
        brick_offset_mm=np.float32(result["brick_offset_mm"]),
        seated=np.bool_(check.seated), at_cell=np.bool_(check.at_target_cell),
        lateral_mm=np.float32(check.lateral_mm),
        failure=np.asarray(result["failure"] or "", dtype="<U96"),
        wall_s=np.float32(result["wall_s"]),
    )
    path = root / f"{prefix}_{seed:07d}_{cycle:d}{attempt:d}.npz"
    np.savez_compressed(path, **arrays)
    return path
