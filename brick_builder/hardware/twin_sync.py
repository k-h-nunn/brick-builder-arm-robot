"""Digital-twin synchronisation: keep a simulator scene aligned with whatever is on the bus.

The twin is a second :class:`~brick_builder.scene.compose.Scene` running beside the system under
control.  Every 20 ms tick it is given the same joint targets the real arm was given, then
corrected with what the bus reported, and the difference between the two is recorded.  Nothing
here drives the robot: it consumes the same numbers the controller already has (joint readings off
the bus, slot states out of the scene model) and produces a log.

Three things are synchronised
-----------------------------
**The arm.**  The twin's joint positions are set from ``Present_Position`` each tick -- through the
free-play partner joints, so the *link* angle matches what the encoder reports -- and the twin's
own servo model is then advanced one tick from the same targets.  What comes out is a prediction
of the *next* reading, and ``prediction_rmse`` is how wrong it was.  A second, optional twin is
never corrected: it runs open loop from the same targets, and its divergence is ``joint_rmse`` and
``tool_pos_err``.  The two separate a twin whose one-step dynamics are wrong (prediction error
moves immediately) from one whose steady state is wrong (open-loop error accumulates).

**The bricks and the baseplate.**  Slot states carry a blob centre in normalized window
coordinates.  With the camera's own intrinsics and extrinsics, a centre plus a known plane height
is one world point (:func:`~brick_builder.perception.backproject.backproject_to_plane`); the blob's
fourth moment gives the footprint's yaw.  Each estimate is associated with the nearest twin body,
and a body that no slot claims keeps the pose it had -- held, not reset to nominal.

**The discrepancy record.**  One row per tick: joint RMSE, tool-pose error, scene-state distance
between the twin's rendered view and the observed one, prediction error and bus latency.

The log
-------
One format for every source -- a simulator-only run, a run over the virtual bus, a hybrid and the
real arm -- because a discrepancy is only interesting next to the same number from another run.
``meta.json`` carries the run's description and ``ticks.npz`` one array per column, both written by
:class:`TwinLog`.  No new dependency: ``json`` and ``numpy.savez`` only, with
:meth:`TwinLog.to_csv` for the scalar columns when a spreadsheet is easier.

Status: verified against the virtual bus and the simulator; untested on hardware.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from ..perception.backproject import CameraGeometry, backproject_to_plane
from ..perception.scene_model import SceneState
from ..robot.patch_so101 import ARM_JOINTS
from ..scene.compose import Scene, set_brick_pose
from .sts3215 import JointCalibration, so101_calibration

SOURCES: tuple[str, ...] = ("sim", "virtual_bus", "hybrid", "real")
"""What produced a log.  Written into ``meta.json`` so two runs can be compared knowingly."""

CONTROL_DT = 0.02
N_JOINTS = 6

TWIN_LOG_COLUMNS: dict[str, int] = {
    "tick": 1,
    "t": 1,
    "cmd_tcp": 3,
    "cmd_grip": 1,
    "joint_target": N_JOINTS,
    "joint_measured": N_JOINTS,
    "joint_speed": N_JOINTS,
    "joint_load": N_JOINTS,
    "joint_predicted": N_JOINTS,
    "joint_twin": N_JOINTS,
    "bus_valid": N_JOINTS,
    "tool_measured": 3,
    "tool_twin": 3,
    "joint_rmse": 1,
    "joint_max": 1,
    "tool_pos_err": 1,
    "tool_rot_err": 1,
    "prediction_rmse": 1,
    "prediction_max": 1,
    "scene_distance": 1,
    "bus_rtt_s": 1,
    "overrun_s": 1,
}
"""The log's schema: column name to width.  Units are SI -- radians, metres, seconds -- except
``joint_load``, which is a PWM duty cycle in [-1, 1] because that is what ``Present_Load`` is."""


# ----------------------------------------------------------------------------------- the log ---


@dataclass
class DiscrepancyRecord:
    """One tick's difference between the twin and the system it is tracking."""

    tick: int = 0
    t: float = 0.0
    joint_rmse: float = math.nan
    joint_max: float = math.nan
    tool_pos_err: float = math.nan
    tool_rot_err: float = math.nan
    prediction_rmse: float = math.nan
    prediction_max: float = math.nan
    scene_distance: float = math.nan
    bus_rtt_s: float = math.nan


class TwinLog:
    """``meta.json`` plus a per-tick table, written as ``ticks.npz``.

    Columns are fixed (:data:`TWIN_LOG_COLUMNS`); a row that omits one gets NaN, so a
    simulator-only run and a run over the real bus produce the same columns and can be plotted on
    the same axes.
    """

    def __init__(self, meta: dict | None = None) -> None:
        self.meta = dict(meta or {})
        self.meta.setdefault("source", "sim")
        self.meta.setdefault("columns", {k: v for k, v in TWIN_LOG_COLUMNS.items()})
        self.rows: list[dict[str, np.ndarray]] = []

    def append(self, **values) -> None:
        row: dict[str, np.ndarray] = {}
        for name, width in TWIN_LOG_COLUMNS.items():
            value = values.pop(name, None)
            if value is None:
                row[name] = np.full(width, np.nan)
            else:
                row[name] = np.asarray(value, dtype=float).reshape(width)
        if values:
            raise KeyError(f"not columns of the twin log: {sorted(values)}")
        self.rows.append(row)

    def append_record(self, record: DiscrepancyRecord, **values) -> None:
        fields = {k: v for k, v in asdict(record).items() if not math.isnan(float(v))}
        self.append(**{**fields, **values})

    def __len__(self) -> int:
        return len(self.rows)

    def table(self) -> dict[str, np.ndarray]:
        """Columns as ``(n_ticks, width)`` arrays -- width 1 columns come back one-dimensional."""
        out: dict[str, np.ndarray] = {}
        for name, width in TWIN_LOG_COLUMNS.items():
            stacked = np.stack([row[name] for row in self.rows]) if self.rows else np.zeros((0, width))
            out[name] = stacked[:, 0] if width == 1 else stacked
        return out

    def column(self, name: str) -> np.ndarray:
        return self.table()[name]

    def write(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(out_dir / "ticks.npz", **self.table())
        return out_dir

    @classmethod
    def read(cls, out_dir: str | Path) -> "TwinLog":
        out_dir = Path(out_dir)
        log = cls(json.loads((out_dir / "meta.json").read_text()))
        with np.load(out_dir / "ticks.npz") as data:
            columns = {name: np.asarray(data[name]) for name in TWIN_LOG_COLUMNS if name in data}
        n = len(next(iter(columns.values()))) if columns else 0
        for k in range(n):
            log.rows.append({
                name: np.atleast_1d(columns[name][k]).astype(float)
                if name in columns else np.full(width, np.nan)
                for name, width in TWIN_LOG_COLUMNS.items()
            })
        return log

    def to_csv(self, path: str | Path) -> Path:
        """The scalar columns as plain CSV.  No new dependency, and a spreadsheet opens it."""
        table = self.table()
        names = [n for n, w in TWIN_LOG_COLUMNS.items() if w == 1]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write(",".join(names) + "\n")
            for k in range(len(self)):
                fh.write(",".join(f"{table[n][k]:.9g}" for n in names) + "\n")
        return path

    def summary(self) -> dict[str, float]:
        """Mean and 95th percentile of every scalar discrepancy column, NaNs ignored."""
        table = self.table()
        out: dict[str, float] = {"ticks": float(len(self))}
        for name in ("joint_rmse", "joint_max", "tool_pos_err", "tool_rot_err",
                     "prediction_rmse", "prediction_max", "scene_distance", "bus_rtt_s"):
            values = table[name]
            values = values[np.isfinite(values)]
            if values.size == 0:
                out[f"{name}_mean"] = math.nan
                out[f"{name}_p95"] = math.nan
                continue
            out[f"{name}_mean"] = float(values.mean())
            out[f"{name}_p95"] = float(np.percentile(values, 95))
        return out


# --------------------------------------------------------------------- scene-state distance ---

_GEOMETRY_FIELDS = ("centre", "mass", "moments", "orient4")


def _geometry_rows(state: SceneState) -> tuple[np.ndarray, list[int]]:
    """The geometry part of every present object slot, and which slots they came from."""
    keep: list[int] = []
    rows: list[np.ndarray] = []
    for k in range(state.K):
        if state.is_empty(k):
            continue
        parts = [np.asarray(state.schema.get(state.slots[k], name), dtype=float).ravel()
                 for name in _GEOMETRY_FIELDS if state.schema.has(name)]
        if not parts:
            continue
        keep.append(k)
        rows.append(np.concatenate(parts))
    return (np.stack(rows) if rows else np.zeros((0, 1))), keep


def scene_distance(observed: SceneState, twin: SceneState, *, miss_penalty: float = 1.0) -> float:
    """How far the twin's rendered view is from the observed one, in published-row units.

    Object slots are matched greedily on blob centre, and each matched pair contributes the
    Euclidean distance between its geometry fields (centre, mass, second moments, fourth moment);
    every slot left unmatched on either side contributes ``miss_penalty``.  Colour is left out on
    purpose: it is the one field a lighting change moves without the scene having moved.
    """
    a, ka = _geometry_rows(observed)
    b, kb = _geometry_rows(twin)
    if not ka and not kb:
        return 0.0
    if not ka or not kb:
        return miss_penalty * (len(ka) + len(kb))
    centres_a = np.array([observed.centre(k) for k in ka])
    centres_b = np.array([twin.centre(k) for k in kb])
    cost = np.linalg.norm(centres_a[:, None, :] - centres_b[None, :, :], axis=-1)
    total, used_a, used_b = 0.0, set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if i in used_a or j in used_b:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        total += float(np.linalg.norm(a[i] - b[j]))
    total += miss_penalty * (len(ka) - len(used_a) + len(kb) - len(used_b))
    return total / max(len(used_a), 1)


# ------------------------------------------------------------------------ brick pose estimate ---


@dataclass
class PoseTrack:
    """One twin body's estimated pose, and how long since a slot last supported it."""

    body: str
    xy: np.ndarray
    yaw: float
    plane_z: float
    held_ticks: int = 0
    seen: int = 0


class ScenePoseEstimator:
    """Slot states plus camera geometry to twin body poses, on a known plane.

    The estimator is deliberately the simplest thing that works: a blob centre backprojected to
    the plane the body's top face sits on, a yaw from the blob's fourth moment, and a
    nearest-neighbour association to the twin body it most likely is.  A body no slot claims keeps
    its last pose and its ``held_ticks`` grows, which is the honest answer for something that has
    left the frame.
    """

    def __init__(self, twin: Scene, bodies: Sequence[str], *,
                 max_association_m: float = 0.04) -> None:
        self.twin = twin
        self.max_association_m = float(max_association_m)
        self.tracks: dict[str, PoseTrack] = {}
        for name in bodies:
            pos, quat = twin.body_pose(name)
            self.tracks[name] = PoseTrack(
                body=name, xy=pos[:2].copy(), yaw=_yaw_of(quat), plane_z=_top_of(twin, name))

    def update(self, state: SceneState, geom: CameraGeometry, window) -> dict[str, PoseTrack]:
        """One observation.  Returns the tracks, updated in place."""
        for track in self.tracks.values():
            track.held_ticks += 1
        for k in range(state.K):
            if state.is_empty(k):
                continue
            cx, cy = state.centre(k)
            angle, strength = state.orientation4(k)
            best: PoseTrack | None = None
            best_d = self.max_association_m
            for track in self.tracks.values():
                try:
                    point = self._world_point(cx, cy, geom, window, track.plane_z)
                except ValueError:
                    continue
                d = float(np.linalg.norm(point[:2] - track.xy))
                if d < best_d:
                    best, best_d = track, d
            if best is None:
                continue
            best.xy = self._world_point(cx, cy, geom, window, best.plane_z)[:2]
            if strength > 0.05:
                # `orientation4` is modulo 90 degrees, so it can only correct the yaw inside one
                # quadrant; the quadrant itself comes from the pose the twin already holds.
                quadrant = round((best.yaw - angle) / (0.5 * math.pi))
                best.yaw = angle + quadrant * 0.5 * math.pi
            best.held_ticks = 0
            best.seen += 1
        return self.tracks

    @staticmethod
    def _world_point(cx: float, cy: float, geom: CameraGeometry, window,
                     plane_z: float) -> np.ndarray:
        from ..perception.views import output_to_full

        res = int(getattr(window, "output_size", 256))
        col = 0.5 * (cx + 1.0) * (res - 1)
        row = 0.5 * (cy + 1.0) * (res - 1)
        return backproject_to_plane(geom, output_to_full(window, np.array([col, row])), plane_z)

    def apply(self) -> None:
        """Write every track's pose into the twin."""
        for track in self.tracks.values():
            _place_body(self.twin, track.body, track.xy, track.yaw)


def _yaw_of(quat) -> float:
    return 2.0 * math.atan2(float(quat[3]), float(quat[0]))


def _top_of(scene: Scene, body: str) -> float:
    """World z of a body's highest geom surface -- the face a camera above it sees."""
    model = scene.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise KeyError(f"no body {body!r} in the twin")
    top = float(scene.data.xpos[bid][2])
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        top = max(top, float(scene.data.geom_xpos[g][2] + model.geom_rbound[g]))
    return top


def _place_body(scene: Scene, body: str, xy, yaw: float) -> None:
    """Move a twin body to ``xy`` at ``yaw``, keeping its height.

    A brick hangs off a free joint and moves through ``set_brick_pose``; the baseplate is welded
    into the world and moves by rewriting ``model.body_pos``/``body_quat``, which is what a
    "the plate is 3 mm further out than the twin thinks" correction has to touch.
    """
    quat = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
    try:
        adr = scene.free_joint_adr(body)
    except KeyError:
        bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
        scene.model.body_pos[bid][:2] = np.asarray(xy, dtype=float)
        scene.model.body_quat[bid] = quat
        mujoco.mj_forward(scene.model, scene.data)
        return
    pos = scene.data.qpos[adr:adr + 3].copy()
    pos[:2] = np.asarray(xy, dtype=float)
    set_brick_pose(scene, body, pos, quat)


# ------------------------------------------------------------------------------- the twin ---


class TwinSync:
    """Keeps ``twin`` aligned with a system whose joint readings arrive one tick at a time.

    ``open_loop`` is a second scene, never corrected, whose divergence from the measurement is the
    twin's accumulated modelling error.  Leave it ``None`` to run the predictor alone.
    """

    def __init__(
        self,
        twin: Scene,
        *,
        open_loop: Scene | None = None,
        servo_cfg=None,
        calibration: Sequence[JointCalibration] | None = None,
        bodies: Sequence[str] = (),
        meta: dict | None = None,
        log: TwinLog | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        from ..robot.servo_model import ServoBus, ServoConfig

        rng = np.random.default_rng(0) if rng is None else rng
        cfg = servo_cfg if servo_cfg is not None else ServoConfig()
        self.twin = twin
        self.open_loop = open_loop
        self.calibration = tuple(calibration if calibration is not None else so101_calibration())
        self.servo = ServoBus(twin.model, twin.data, cfg=cfg,
                              rng=np.random.default_rng(int(rng.integers(2**31))))
        self.open_servo = (
            ServoBus(open_loop.model, open_loop.data, cfg=cfg,
                     rng=np.random.default_rng(int(rng.integers(2**31))))
            if open_loop is not None else None
        )
        self.log = log if log is not None else TwinLog(meta)
        self.estimator = ScenePoseEstimator(twin, bodies) if bodies else None
        self.tick = 0
        self.predicted: np.ndarray | None = None
        self._qpos_adr = twin.arm_qpos_adr[:N_JOINTS]
        self._dof_adr = twin.arm_dof_adr[:N_JOINTS]
        play = [mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_backlash")
                for n in ARM_JOINTS]
        self._play_qpos = (twin.model.jnt_qposadr[np.array(play)]
                           if all(j >= 0 for j in play) else None)
        self._play_dof = (twin.model.jnt_dofadr[np.array(play)]
                          if all(j >= 0 for j in play) else None)

    # -- the arm ------------------------------------------------------------------------------

    def seat(self, measured: np.ndarray, speed: np.ndarray | None = None,
             scene: Scene | None = None) -> None:
        """Put a twin's *link* angles on the measured ones, through the free-play partners.

        The encoder reads the output shaft, i.e. the driven joint plus its free-play partner, so
        the driven joint is set to ``measured - play`` and the play keeps whatever state it had.
        The firmware's own internal target inside ``ServoBus`` is untouched: only the plant is
        corrected, which is what a state estimator does.
        """
        scene = self.twin if scene is None else scene
        data = scene.data
        play = data.qpos[self._play_qpos] if self._play_qpos is not None else 0.0
        data.qpos[self._qpos_adr] = np.asarray(measured, dtype=float)[:N_JOINTS] - play
        if speed is not None:
            play_v = data.qvel[self._play_dof] if self._play_dof is not None else 0.0
            data.qvel[self._dof_adr] = np.asarray(speed, dtype=float)[:N_JOINTS] - play_v
        mujoco.mj_forward(scene.model, data)

    def seat_all(self, measured: np.ndarray, speed: np.ndarray | None = None) -> None:
        """Seat both twins.  Run once before a comparison starts, so the open-loop twin is not
        already a couple of ticks behind the system it is being measured against."""
        self.seat(measured, speed)
        if self.open_loop is not None:
            self.seat(measured, speed, self.open_loop)

    def advance(self, targets: np.ndarray, torque_limit: np.ndarray | None = None) -> np.ndarray:
        """Run the twin one 20 ms control tick from ``targets`` and return the predicted reading."""
        return _advance(self.twin, self.servo, targets, torque_limit)

    def link_angles(self, scene: Scene | None = None) -> np.ndarray:
        """A twin's own output-shaft angles -- driven joint plus free-play partner."""
        scene = self.twin if scene is None else scene
        q = scene.data.qpos[scene.arm_qpos_adr[:N_JOINTS]].copy()
        if self._play_qpos is not None:
            q += scene.data.qpos[self._play_qpos]
        return q

    # -- one tick -----------------------------------------------------------------------------

    def step(
        self,
        targets: np.ndarray,
        measured: np.ndarray,
        *,
        speed: np.ndarray | None = None,
        load: np.ndarray | None = None,
        torque_limit: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        cmd_tcp: np.ndarray | None = None,
        cmd_grip: float | None = None,
        bus_rtt_s: float = math.nan,
        overrun_s: float = math.nan,
        observed: SceneState | None = None,
        twin_state: SceneState | None = None,
        geometry: CameraGeometry | None = None,
        window=None,
    ) -> DiscrepancyRecord:
        """Predict, compare, correct, and append one row to the log.

        The order matters and is the causal one.  The measurement this tick is what the real
        system did *in response to* this tick's target, so both twins are advanced with that same
        target before anything is compared: the tracking twin from where the last measurement put
        it (one-step prediction), the open-loop twin from wherever it had got to on its own.  Only
        then is the tracking twin re-seated, ready for the next tick.
        """
        targets = np.asarray(targets, dtype=float).reshape(N_JOINTS)
        measured = np.asarray(measured, dtype=float).reshape(N_JOINTS)
        record = DiscrepancyRecord(tick=self.tick, t=self.tick * CONTROL_DT, bus_rtt_s=bus_rtt_s)

        self.predicted = self.advance(targets, torque_limit)
        err = self.predicted - measured
        record.prediction_rmse = float(np.sqrt(np.mean(err ** 2)))
        record.prediction_max = float(np.max(np.abs(err)))

        tool_real = self.twin.kin.fk(measured[:5])
        twin_q = self.predicted
        if self.open_loop is not None:
            twin_q = _advance(self.open_loop, self.open_servo, targets, torque_limit)
            err = twin_q - measured
            record.joint_rmse = float(np.sqrt(np.mean(err ** 2)))
            record.joint_max = float(np.max(np.abs(err)))
            tool_twin = self.open_loop.kin.fk(twin_q[:5])
            record.tool_pos_err = float(np.linalg.norm(tool_twin[0] - tool_real[0]))
            record.tool_rot_err = _rotation_angle(tool_twin[1], tool_real[1])
        else:
            tool_twin = self.twin.kin.fk(twin_q[:5])

        if observed is not None and twin_state is not None:
            record.scene_distance = scene_distance(observed, twin_state)
        if observed is not None and geometry is not None and self.estimator is not None:
            self.estimator.update(observed, geometry, window)
            self.estimator.apply()

        self.seat(measured, speed)

        self.log.append_record(
            record,
            joint_target=targets,
            joint_measured=measured,
            joint_speed=speed if speed is not None else np.full(N_JOINTS, np.nan),
            joint_load=load if load is not None else np.full(N_JOINTS, np.nan),
            joint_predicted=self.predicted,
            joint_twin=twin_q,
            bus_valid=(np.asarray(valid, dtype=float) if valid is not None
                       else np.full(N_JOINTS, np.nan)),
            tool_measured=tool_real[0],
            tool_twin=tool_twin[0],
            cmd_tcp=cmd_tcp if cmd_tcp is not None else np.full(3, np.nan),
            cmd_grip=cmd_grip if cmd_grip is not None else np.nan,
            overrun_s=overrun_s,
        )
        self.tick += 1
        return record


def _advance(scene: Scene, servo, targets: np.ndarray,
             torque_limit: np.ndarray | None) -> np.ndarray:
    from .virtual_bus import PHYSICS_PER_TICK

    servo.write(np.asarray(targets, dtype=float).reshape(N_JOINTS), torque_limit=torque_limit)
    snap_every = max(int(scene.snap.params.snap_every), 1)
    for k in range(PHYSICS_PER_TICK):
        servo.substep()
        if k % snap_every == 0:
            scene.snap.update(scene.model, scene.data)
        mujoco.mj_step(scene.model, scene.data)
    return servo.read().position.copy()


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking ``a`` to ``b`` [rad]."""
    trace = float(np.trace(a.T @ b))
    return float(math.acos(max(-1.0, min(1.0, 0.5 * (trace - 1.0)))))


# ------------------------------------------------------------------------------- the report ---


REPORT_ROWS: tuple[tuple[str, str, str, float], ...] = (
    ("joint_rmse", "open-loop joint RMSE", "mdeg", 1e3 * 180.0 / math.pi),
    ("joint_max", "open-loop joint max", "mdeg", 1e3 * 180.0 / math.pi),
    ("tool_pos_err", "open-loop tool position", "mm", 1e3),
    ("tool_rot_err", "open-loop tool rotation", "mdeg", 1e3 * 180.0 / math.pi),
    ("prediction_rmse", "one-step prediction RMSE", "mdeg", 1e3 * 180.0 / math.pi),
    ("prediction_max", "one-step prediction max", "mdeg", 1e3 * 180.0 / math.pi),
    ("scene_distance", "scene-state distance", "-", 1.0),
    ("bus_rtt_s", "bus round trip", "ms", 1e3),
)
"""Which columns the report prints, with the unit each is readable in."""


def discrepancy_report(log: TwinLog, figure: str | Path | None = None) -> tuple[str, Path | None]:
    """A markdown table of one run's discrepancies, and (optionally) the figure beside it."""
    summary = log.summary()
    lines = [
        f"Twin discrepancy, {int(summary['ticks'])} ticks, source `{log.meta.get('source', '?')}`",
        "",
        "| metric | unit | mean | p95 |",
        "|---|---|---|---|",
    ]
    for name, label, unit, scale in REPORT_ROWS:
        mean = summary.get(f"{name}_mean", math.nan) * scale
        p95 = summary.get(f"{name}_p95", math.nan) * scale
        if not math.isfinite(mean):
            continue
        lines.append(f"| {label} | {unit} | {mean:.3g} | {p95:.3g} |")
    path = None
    if figure is not None:
        path = figure_discrepancy(log, figure)
        lines.extend(["", f"![Twin discrepancy]({Path(path).name})"])
    return "\n".join(lines) + "\n", path


def figure_discrepancy(log: TwinLog, out_path: str | Path) -> Path:
    """Four panels: joint error, tool error, prediction error and bus round trip against time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = log.table()
    t = table["t"]
    panels = (
        ("joint_rmse", "open-loop joint RMSE [mdeg]", 1e3 * 180.0 / math.pi),
        ("tool_pos_err", "open-loop tool error [mm]", 1e3),
        ("prediction_rmse", "one-step prediction RMSE [mdeg]", 1e3 * 180.0 / math.pi),
        ("bus_rtt_s", "bus round trip [ms]", 1e3),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), dpi=120)
    for ax, (name, label, scale) in zip(axes.ravel(), panels):
        values = table[name] * scale
        ax.plot(t, values, lw=1.4, color="#1f6feb")
        finite = values[np.isfinite(values)]
        if finite.size:
            ax.axhline(float(finite.mean()), color="#b0413e", lw=1.0, ls="--",
                       label=f"mean {finite.mean():.3g}")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Twin discrepancy, source {log.meta.get('source', '?')}")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def comparison_table(runs: dict[str, TwinLog], baseline: str = "nominal") -> str:
    """One markdown table: every run's metrics beside the baseline's, as a ratio.

    This is the perturbation evidence -- one row per perturbation, one column per metric, so
    "which metric moves, and by how much" is read straight off it.
    """
    summaries = {name: log.summary() for name, log in runs.items()}
    base = summaries.get(baseline)
    header = ["| perturbation |"] + [f" {label} [{unit}] |" for _, label, unit, _ in REPORT_ROWS]
    lines = ["".join(header), "|---|" + "---|" * len(REPORT_ROWS)]
    for name, summary in summaries.items():
        cells = [f"| {name} |"]
        for key, _, _, scale in REPORT_ROWS:
            value = summary.get(f"{key}_mean", math.nan)
            if not math.isfinite(value):
                cells.append(" - |")
                continue
            cell = f"{value * scale:.3g}"
            if base is not None and name != baseline:
                ref = base.get(f"{key}_mean", math.nan)
                if math.isfinite(ref) and ref > 0:
                    cell += f" ({value / ref:.1f}x)"
            cells.append(f" {cell} |")
        lines.append("".join(cells))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- the perturbation study ---


@dataclass(frozen=True)
class Perturbation:
    """One way the system on the other end of the bus differs from the twin's idea of it."""

    name: str
    arm: dict = field(default_factory=dict)
    """``ArmConfig`` overrides: servo gain, free play, torque limit, supply voltage."""
    servo: dict = field(default_factory=dict)
    """``ServoConfig`` overrides: dead band, rate limit, read noise."""
    snap: dict = field(default_factory=dict)
    """``SnapParams`` overrides: clutch force per stud, push/pull asymmetry."""
    scene: dict = field(default_factory=dict)
    """``SceneConfig`` overrides: baseplate position, brick friction."""
    joint_friction_scale: float = 1.0
    """Multiplier on every arm joint's dry friction.  ``ArmConfig`` has no field for it -- the
    identified value is compiled in -- so it is applied to ``dof_frictionloss`` after the build,
    which is also where ``arm.joint_friction_Nm`` in the parameter file would land."""


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("nominal"),
    Perturbation("clutch 3x", snap={"f_clutch": 3.0}),
    Perturbation("joint friction 3x", joint_friction_scale=3.0),
    Perturbation("backlash 3x", arm={"backlash_deg": 1.305}),
    Perturbation("servo gain P=32", arm={"p_gain": 32}),
    Perturbation("baseplate +3 mm", scene={"plate_xy": (0.173, 0.0)}),
)
"""The six runs the study makes: a baseline and one perturbation each, all invisible to the twin."""

STUDY_CELL = (3.5, 3.5)
"""Plate cell the study's brick sits on -- the middle of an 8x8 plate."""


def _study_servo_config():
    """The servo configuration both sides of the study run, with every random draw switched off.

    Command-delay jitter and read noise are real, and both are in the model, but a study whose
    baseline already differs from its twin by a few millimetres of wire-delay luck cannot say
    which metric a 3 mm baseplate offset moved.  With the draws off, the baseline is identical by
    construction and every non-zero entry in the table is the perturbation.
    """
    from ..robot.servo_model import ServoConfig

    return ServoConfig(action_delay_jitter_s=0.0, read_noise_ticks=0.0)


def _study_scene(perturbation: Perturbation):
    """The world on the far end of the bus: an 8x8 plate with one 2x4 brick resting on the studs."""
    from ..bricks.snap_api import SnapParams
    from ..bricks.structures import Placement
    from ..robot.patch_so101 import ArmConfig, WRIST_CAM
    from ..scene.compose import SceneConfig, build_scene, reset_scene, set_brick_pose

    cfg = SceneConfig(
        arm=ArmConfig(**perturbation.arm),
        snap=SnapParams(**perturbation.snap),
        placed=[Placement(h=2, w=4, x=2, y=3, z=0, colour="red")],
        held=None,
        tray=False,
        cameras=(WRIST_CAM,),
        damage_monitor=False,
        **perturbation.scene,
    )
    scene = build_scene(cfg)
    if perturbation.joint_friction_scale != 1.0:
        for name in ARM_JOINTS:
            jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            scene.model.dof_frictionloss[scene.model.jnt_dofadr[jid]] *= \
                perturbation.joint_friction_scale
    reset_scene(scene, np.random.default_rng(0))
    # Lift the brick onto the stud tops: the study's press is what pushes it home, and how far it
    # goes is what the clutch force decides.
    name = scene.placed_names[0]
    pos, quat = scene.body_pose(name)
    set_brick_pose(scene, name, pos + np.array([0.0, 0.0, scene.spec.stud_h + 5e-4]), quat)
    mujoco.mj_forward(scene.model, scene.data)
    scene.snap.bind(scene.model, scene.data)
    return scene


def _press_path(scene, *, hover: float = 0.045, depth: float = 0.008) -> list[tuple]:
    """The scripted tool path: hover over the brick, push down on its roof, retreat.

    A press rather than a free-space sweep, because three of the six perturbations -- clutch
    force, brick friction, servo gain -- only show themselves through something the arm leans on.
    The path is computed from the *nominal* twin, so a run whose baseplate has moved is asked to
    press in the wrong place, which is the point of that perturbation.
    """
    from ..control.pick_place import cell_world, tcp_for_foot
    from ..robot.patch_so101 import gripper_angle_for_width

    cell = cell_world(scene, *STUDY_CELL)
    roof = scene.plate_top + scene.spec.height + scene.spec.stud_h
    grip = gripper_angle_for_width(0.002)
    start = scene.kin.fk(scene.home_q)[0]
    above = tcp_for_foot(cell[:2], roof + hover, 0.0)
    touch = tcp_for_foot(cell[:2], roof, 0.0)
    press = tcp_for_foot(cell[:2], roof - depth, 0.0)
    return [
        ("approach", start, above, 50, grip, 1.0),
        ("descend", above, touch, 50, grip, 1.0),
        ("press", touch, press, 40, grip, 1.0),
        ("hold", press, press, 30, grip, 1.0),
        ("retreat", press, above, 30, grip, 1.0),
    ]


def run_against_virtual_bus(
    perturbation: Perturbation = PERTURBATIONS[0],
    *,
    vision_every: int = 5,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> TwinLog:
    """One "real versus twin" run: a perturbed simulator behind the virtual bus, a nominal twin.

    The perturbed scene is driven only through :class:`~brick_builder.hardware.bus.Bus` over a
    pseudo-terminal, exactly as the real arm will be; the twin sees nothing but the joint readings
    and the wrist view.  Returns the run's log.
    """
    from .bus import Bus, BusSettings
    from .real_world import ArmWorldConfig, RealArmWorld, SimulatedCamera
    from .virtual_bus import simulated_bus

    real = _study_scene(perturbation)
    twin = _study_scene(PERTURBATIONS[0])
    open_twin = _study_scene(PERTURBATIONS[0])
    bodies = [twin.placed_names[0], twin.plate.name]

    servo_cfg = _study_servo_config()
    vbus = simulated_bus(real, servo_cfg=servo_cfg, seed=seed).start()
    bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
    try:
        bus.enable_torque()
        # The twin owns the log here: it writes one row per tick with the arm columns *and* the
        # discrepancy columns, so the world must not write a second row of its own.
        world = RealArmWorld(bus, cfg=ArmWorldConfig(
            paced=False, source="virtual_bus", log_every_tick=False))
        meta = {
            "source": "virtual_bus",
            "perturbation": perturbation.name,
            "overrides": {"arm": perturbation.arm, "servo": perturbation.servo,
                          "snap": perturbation.snap,
                          "scene": {k: list(v) if isinstance(v, tuple) else v
                                    for k, v in perturbation.scene.items()}},
            "vision_every": vision_every,
            "seed": seed,
        }
        sync = TwinSync(twin, open_loop=open_twin, bodies=bodies, meta=meta,
                        servo_cfg=servo_cfg, rng=np.random.default_rng(seed))
        world.log = sync.log
        real_cam = SimulatedCamera(real)
        twin_cam = SimulatedCamera(twin)
        # Both twins start from the far end's state at tick 0.  Enabling torque already cost the
        # far end a couple of control ticks -- one of them with the servos limp -- and the brick
        # resting on the stud tops settles during them, so a twin left at its own reset pose would
        # start the comparison already behind and the baseline would not be zero.  Only the
        # initial condition is copied; everything after it reaches the twin over the bus, which is
        # what `seat_all` and `ScenePoseEstimator` do on a run whose far end is not a simulator.
        with vbus.servos.lock:
            for other in (twin, open_twin):
                other.data.qpos[:] = real.data.qpos
                other.data.qvel[:] = real.data.qvel
                mujoco.mj_forward(other.model, other.data)
        try:
            for phase, a, b, ticks, grip, c_vert in _press_path(twin):
                for k in range(ticks):
                    frac = (k + 1) / ticks
                    world.command(np.asarray(a) * (1 - frac) + np.asarray(b) * frac, 0.0, grip,
                                  c_vert=c_vert)
                    observed = twin_state = geometry = None
                    if world.tick % vision_every == 0:
                        with vbus.servos.lock:
                            observed = real_cam.observe(world.t)
                        twin_state = twin_cam.observe(world.t)
                        geometry = twin_cam.geometry()
                    sync.step(
                        world.last_target, world.joint_position, speed=world.joint_speed,
                        load=world.joint_load, valid=world.state.valid.astype(float),
                        torque_limit=world.last_torque,
                        cmd_tcp=world.cmd_tcp, cmd_grip=grip, bus_rtt_s=world.state.rtt_s,
                        observed=observed, twin_state=twin_state, geometry=geometry,
                        window=twin_cam.window,
                    )
            sync.log.meta["phases"] = [p[0] for p in _press_path(twin)]
            sync.log.meta["driver_counters"] = bus.counters.as_dict()
            sync.log.meta["bus_counters"] = vbus.counters.as_dict()
        finally:
            real_cam.close()
            twin_cam.close()
    finally:
        bus.close()
        vbus.stop()
        vbus.raise_if_failed()
    if out_dir is not None:
        sync.log.write(Path(out_dir) / perturbation.name.replace(" ", "_"))
    return sync.log


def perturbation_study(
    perturbations: Sequence[Perturbation] = PERTURBATIONS,
    *,
    out_dir: str | Path | None = None,
    vision_every: int = 5,
    seed: int = 0,
) -> tuple[str, dict[str, TwinLog]]:
    """Every perturbation in turn, and the markdown table saying which metric caught it."""
    runs: dict[str, TwinLog] = {}
    for perturbation in perturbations:
        runs[perturbation.name] = run_against_virtual_bus(
            perturbation, vision_every=vision_every, seed=seed, out_dir=out_dir)
    table = comparison_table(runs)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "perturbations.md").write_text(table)
    return table, runs
