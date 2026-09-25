"""Everything the digital twin computes for a run, measured once on the host before it starts.

`TrackingEpisode` normally measures a step's Jacobians in the twin while the loop runs, at every
step boundary and every `jacobian_every` frames between them. Which frames those are depends on
the run's own history, so a controller that cannot render cannot be handed "the step for frame
k" unless every frame has one. `build_plan` measures all of them, in frame order, and the result
is what both the Python controller and the C one consume -- the same numbers, so what is compared
afterwards is the control logic and not two different twins.

It costs about twelve renders per reference frame and runs once per episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..action.place import STEP_TITLES, PlaceConfig
from ..action.sequence import IdealSequence
from ..action.twin import BlindStep, DigitalTwin
from ..action.worlds import DISTRACTOR, WANTED, nominal_cfg
from ..scene.compose import HeldBrick


@dataclass
class FramePlan:
    """Per-frame controller plan, beyond the frame's own record."""

    step: object | None
    """The `TwinStep` measured at this frame, or `None` when there is nothing to line up on."""
    trusted: bool
    gain: float
    held: bool
    is_freeze_step: bool
    step_id: int


@dataclass
class Plan:
    frames: list[FramePlan]
    descent_k: int
    grasp_k: int
    grasp_z: float
    freeze_index: int
    n_frames: int = field(default=0)

    def __post_init__(self) -> None:
        self.n_frames = len(self.frames)

    def step_for(self, k: int):
        """Twin step of frame `k`, as `TrackingEpisode._build` returns it.

        Raises `BlindStep` when there is no usable step, as the live twin does."""
        plan = self.frames[k]
        if plan.step is None:
            raise BlindStep(f"frame {k}: the twin has no step to measure an inverse from")
        return plan.step


def _twins(cfg: PlaceConfig, seed: int, twin_prepare) -> dict[bool, DigitalTwin]:
    """The same two twins `TrackingEpisode` builds, for the same reason."""
    held = HeldBrick(nx=WANTED[0], ny=WANTED[1], colour=WANTED[2])
    twins = {
        False: DigitalTwin(nominal_cfg(held=None,
                                       supply=(WANTED, DISTRACTOR) if cfg.pick else ()),
                           seed=seed, prepare=twin_prepare),
    }
    if cfg.pick:
        twins[True] = DigitalTwin(nominal_cfg(held=held, supply=(None, DISTRACTOR)),
                                  seed=seed, prepare=twin_prepare)
    else:
        twins[True] = DigitalTwin(nominal_cfg(held=held), seed=seed, prepare=twin_prepare)
    return twins


def build_plan(ideal: IdealSequence, cfg: PlaceConfig, *, seed: int = 0,
               twin_prepare=None) -> Plan:
    """Measure a step at every reference frame and read off the indices the recording fixes."""
    twins = _twins(cfg, seed, twin_prepare)
    try:
        frames = [_frame_plan(ideal, cfg, twins, frame) for frame in ideal.frames]
    finally:
        for twin in set(twins.values()):
            twin.close()

    descent_k = next((i for i, f in enumerate(ideal.frames)
                      if f.commands and f.commands[0].pos[2] < cfg.pick_freeze_below), -1)
    grasp_k = max((i for i, f in enumerate(ideal.frames) if f.step == "Grasp"), default=-1)
    grasp_z = min((c.pos[2] for f in ideal.frames if f.step in ("Approach", "Grasp")
                   for c in f.commands), default=0.0)
    freeze_index = max((f.step_index for f in ideal.frames if f.step == cfg.freeze_after),
                       default=10 ** 6)
    return Plan(frames=frames, descent_k=descent_k, grasp_k=grasp_k, grasp_z=float(grasp_z),
                freeze_index=int(freeze_index))


def _frame_plan(ideal: IdealSequence, cfg: PlaceConfig, twins, frame) -> FramePlan:
    bodies = dict(frame.bodies or ideal.bodies_by_step.get(frame.step_index, {}))
    held = ideal.held_by_step.get(frame.step_index)
    twin = twins[held is not None]
    if held is not None and twin.scene.held_name is not None:
        held = twin.scene.held_name
    bodies = {k: v for k, v in bodies.items() if v in twin.scene.instance_ids}
    blank = FramePlan(step=None, trusted=False, gain=0.0, held=held is not None,
                      is_freeze_step=frame.step == cfg.freeze_after,
                      step_id=STEP_TITLES.index(frame.step))
    if not bodies:
        return blank

    pose = frame.commands[0]
    try:
        step = twin.build_step(
            frame.step, pose.pos, pose.spin, bodies, brick_body=held, tol_mm=cfg.tol_mm,
            min_refs=1, angle_from_blob=(len(bodies) == 1), square=(WANTED[0] == WANTED[1]),
            grip=pose.grip, brick_dof=2, brick_row=False,
        )
    except (BlindStep, RuntimeError):
        return blank

    sv = np.linalg.svd(step.J[:, :2], compute_uv=False)
    gain = float(sv[-1]) if sv.size else 0.0
    alone = cfg.pick and frame.step in ("Approach", "Grasp")
    trusted = bool(gain >= cfg.min_gain and (len(bodies) >= 2 or alone))

    return FramePlan(step=step, trusted=trusted, gain=gain, held=held is not None,
                     is_freeze_step=frame.step == cfg.freeze_after,
                     step_id=STEP_TITLES.index(frame.step))


def arm_constants(kin) -> dict:
    """The kinematic chain of the five solved joints, read off the model poses are evaluated on.

    Each link is a hinge at its own body origin, so the chain is a parent-relative frame and an
    axis per joint plus the tool site's own offset. Scene setup reading a model, not control.
    """
    import mujoco

    from ..robot.kinematics import KIN_JOINTS

    model = kin._e_model
    site = kin._e_site
    body_of = [int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
               for n in KIN_JOINTS]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in KIN_JOINTS]
    anchored = [float(np.max(np.abs(model.jnt_pos[j]))) for j in jids]
    if max(anchored) > 0.0:
        raise RuntimeError("the C chain assumes every solved joint sits at its own body origin")
    return {
        "body_pos": np.array([model.body_pos[b] for b in body_of], dtype=float),
        "body_quat": np.array([model.body_quat[b] for b in body_of], dtype=float),
        "axis": np.array([model.jnt_axis[j] for j in jids], dtype=float),
        "site_pos": np.asarray(model.site_pos[site], dtype=float).copy(),
        "site_quat": np.asarray(model.site_quat[site], dtype=float).copy(),
        "jnt_lo": np.array([model.jnt_range[j][0] for j in jids], dtype=float),
        "jnt_hi": np.array([model.jnt_range[j][1] for j in jids], dtype=float),
    }


def grip_range(model) -> tuple[float, float]:
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    lo, hi = model.actuator_ctrlrange[gid]
    return float(lo), float(hi)


def radians(deg: float) -> float:
    return math.radians(deg)
