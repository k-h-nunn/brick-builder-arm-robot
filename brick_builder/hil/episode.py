"""The host half of the loop: run the world, hand the controller what it may read, wait.

The controller decides; the host renders, steps physics and keeps the log. Which of the three
transports is behind `controller` makes no difference here, which is the point -- the same host
loop drives software in the loop, the native process and the board.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..action.place import STEP_TITLES, StepReport, TickRecord
from . import bundle as B
from . import plan as planning
from .library import FAILURES


@dataclass
class RunLog:
    """What one run through a C controller produced, beside the tick records."""

    targets: list = field(default_factory=list)
    """Per 50 Hz tick: the six joint targets the controller published."""
    ks: list = field(default_factory=list)
    step_boundaries: list = field(default_factory=list)
    err_mm: list = field(default_factory=list)
    corrections: list = field(default_factory=list)
    ik_overrides: int = 0
    """Ticks where the host's own inverse kinematics did not accept the controller's answer."""
    seconds: float = 0.0


class ControllerEpisode:
    """Drives a `TrackingEpisode`'s world from a C controller instead of its own tick logic.

    Installed as `TrackingEpisode(controller=...)`; `run` is a generator of the same
    `TickRecord`s the Python loop yields, so the video, the demo and `run_placement.py` read it
    without knowing which controller produced it.
    """

    def __init__(self, controller, blob: bytes, *, log: RunLog | None = None) -> None:
        self.controller = controller
        self.blob = blob
        self.log = log if log is not None else RunLog()

    def run(self, episode):
        cfg = episode.cfg
        frames = episode.ideal.frames
        world = episode.world
        started = time.perf_counter()
        self.controller.start(self.blob, world.joint_goal)

        step_errors: dict[int, list[float]] = {}
        state = None
        rgb = None
        pos = np.zeros(3)
        while True:
            targets = self.controller.tick_begin()
            if not targets.commands:
                break
            frame = frames[targets.k]
            for command in targets.commands:
                pos = np.asarray(command.pos, dtype=float)
                world.command(pos, command.spin, command.grip, c_vert=command.c_vert,
                              c_lat=command.c_lat, q_goal=command.q[:5])
                self.log.targets.append(np.asarray(command.q, dtype=float).copy())
                if not np.array_equal(world.joint_goal, command.q[:5]):
                    self.log.ik_overrides += 1
                yield self._record(episode, frame, pos, command, vision=False)

            state, rgb = world.observe_rgb()
            bus = (world.joint_position, world.joint_speed, world.joint_load)
            entry = self.controller.tick_end(state, bus)

            episode.k = entry.k
            episode.correction = np.asarray(entry.correction, dtype=float).copy()
            episode.holds = entry.holds
            episode.matched = entry.matched
            episode.regrasps = entry.regrasps
            episode.absolute_mm = entry.absolute_mm
            episode.yaw_err = entry.yaw_err
            episode.errors.append(entry.err_mm)
            step_errors.setdefault(frame.step_index, []).append(entry.err_mm)
            self.log.ks.append(entry.k)
            self.log.err_mm.append(entry.err_mm)
            self.log.corrections.append(np.asarray(entry.correction, dtype=float).copy())

            record = self._record(episode, frame, pos, targets.commands[-1], vision=True,
                                  err_mm=entry.err_mm, seen=entry.refs_seen)
            record.state = state
            record.rgb = rgb
            nxt = frames[min(entry.k + 1, len(frames) - 1)]
            record.target = nxt.state
            record.target_label = nxt.label
            record.predict_mm = entry.predict_mm
            yield record

            if entry.done:
                episode.failure = FAILURES.get(entry.failure)
                break

        self.log.seconds = time.perf_counter() - started
        self.log.step_boundaries = _boundaries(self.log.ks)
        for index, errs in step_errors.items():
            title = next(f.step for f in frames if f.step_index == index)
            episode.reports.append(
                StepReport(title, len(errs), errs[-1], errs[-1] <= cfg.tol_mm * 4))

    def _record(self, episode, frame, pos, command, *, vision: bool,
                err_mm: float = float("nan"), seen: int = 0) -> TickRecord:
        world = episode.world
        return TickRecord(
            tick=world.tick, t=world.t, step=frame.step, step_index=frame.step_index,
            cmd_pos=np.asarray(pos, dtype=float).copy(), spin=float(command.spin),
            grip=float(command.grip), correction=episode.correction.copy(), k=episode.k,
            vision=vision, err_mm=err_mm if vision else float("nan"),
            tol_mm=episode.cfg.tol_mm, refs_seen=seen,
        )


def _boundaries(ks) -> list[int]:
    """Tick numbers at which the progress index moved, which is the shape being compared."""
    out = []
    last = None
    for i, k in enumerate(ks):
        if last is not None and k != last:
            out.append(i)
        last = k
    return out


def prepare(episode, *, jc_ki: float = 0.0, jc_joints: int = 0) -> tuple[object, bytes]:
    """Measure every step in the twin and write the bundle for this episode's own sequence."""
    cfg = episode.cfg
    made = planning.build_plan(episode.ideal, cfg, seed=episode.seed)
    kin = episode.world.kinematics
    arm = planning.arm_constants(kin)
    grip = planning.grip_range(kin.model)
    blob = B.write_bundle(episode.ideal, made, cfg, arm, grip_range=grip, jc_ki=jc_ki,
                          jc_joints=jc_joints)
    return made, blob


def step_titles() -> tuple[str, ...]:
    return STEP_TITLES
