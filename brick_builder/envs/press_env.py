"""``BrickBuilder/PressFromHeld-v0``: seat a held brick on the plate, blind, under misalignment.

This is the heart of Deliverable 1.  A brick is already in the gripper a few millimetres above
its target cell, with lateral, yaw and tilt error in the arm's pose and a further error in how
the brick sits in the hand.  The scripted primitive (:class:`~brick_builder.control.primitives.
PressScripted`) presses straight down and searches when it stalls; the policy is the bounded
residual on top of it plus the two compliance scalars.  Nothing the actor sees comes from a
camera -- the wrist camera cannot see under a held brick, so the press is blind by construction
and has to work from proprioception alone.

What "success" means here is the spec's: every stud of the held brick at least 0.9 stud heights
into a socket **of the correct cell**, held for five consecutive ticks.  Seating one pitch over
is a `wrong_cell` failure, not a success.

The critic, and only the critic, additionally sees the per-stud insertion depths the snap model
computes, the true misalignment and in-hand offset, the force the grip is transmitting, how far
the neighbouring bricks have moved, and this episode's randomization draw.

Levels
------
``env.unwrapped.set_level(k)`` moves along the curriculum of spec section 4:

===== ============ ======== ====================================================================
level lateral      yaw/tilt world
===== ============ ======== ====================================================================
0     +-1 mm       2 / 0.4  empty plate, target in the middle
1     +-2 mm       4 / 0.8  empty plate
2     +-4 mm       8 / 1.4  empty plate
3     +-6 mm       12 / 2.0 empty plate
4     +-6 mm       12 / 2.0 one brick already on the plate; targets next to it and on top of it
5     +-6 mm       12 / 2.0 two bricks on the plate; near-row, far-row and layer-1 targets
===== ============ ======== ====================================================================

Levels 4 and 5 change what the world contains, so moving to or from them rebuilds the MuJoCo
model (about 0.3 s); levels 0-3 differ only in the numbers drawn at reset.  The level also drives
how much of the dynamics/command/in-hand/contact/plate randomization ranges are used
(``dr_level() = level / 5``), so level 0 is the exactly nominal, un-randomized world and is what
the scripted baseline is measured against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from brick_builder.bricks.snap_api import SnapParams, equality_forces
from brick_builder.bricks.structures import Placement
from brick_builder.control.primitives import (
    PHASES,
    Feedback,
    PressConfig,
    PressScripted,
    feedback_from_readings,
)
from brick_builder.envs import obs_spec
from brick_builder.envs.base_env import (
    ANG_CLIP,
    POS_CLIP,
    POS_SCALE,
    SPIN_SCALE,
    TORQUE_HI,
    TORQUE_LO,
    ArmEnv,
    PrimitiveTarget,
)
from brick_builder.eval.results import FailureClass
from brick_builder.robot.kinematics import euler_zyx
from brick_builder.scene.compose import HeldBrick, SceneConfig, tool_reach_below_tcp

# ------------------------------------------------------------------------------ the criterion ---

SEAT_FRACTION = 0.9
"""How deep every stud must be, as a fraction of the stud height, to count as seated."""

SUCCESS_TICKS = 5
"""Consecutive ticks the seat must hold (spec section 4)."""

CELL_TOLERANCE = 0.4
"""How far the brick's centre may be from the target cell's centre, in stud pitches, and still
count as "the correct cell".  0.4 pitch is 6.4 mm: half a pitch would be ambiguous between two
cells, and anything seated is in practice within a few tenths of a millimetre."""

MAX_SOCKETS = 8
"""Widest per-stud-depth vector the critic carries (a 2x4).  Smaller bricks zero-pad, so the
observation keeps one fixed size whatever brick an episode uses."""

START_HOVER = 0.005
"""How far above the seated height the brick starts, metres. "A few mm above its target cell"."""

SLIP_LIMIT = 0.008
"""In-hand movement (metres, TCP frame) past which the brick counts as dropped."""

JOINT_LIMIT_MARGIN = math.radians(0.5)
"""How close an arm joint may get to its own range limit before the episode is called kinematic."""


# ------------------------------------------------------------------------------- the curriculum ---


@dataclass(frozen=True)
class LevelSpec:
    """One rung of the press curriculum: how wrong the start pose may be, and what is on the plate.

    ``targets`` are ``(grid_cx, grid_cy, layer)`` triples giving the *centre* of the brick's
    footprint in plate stud coordinates (stud 0 is the minimum corner, so an 8x8 plate has
    centres from 0 to 7 and a 2-wide footprint sits at a half-integer).
    """

    lateral: float
    yaw: float
    tilt: float
    placed: tuple[Placement, ...] = ()
    targets: tuple[tuple[float, float, int], ...] = ((3.5, 3.5, 0),)


_BASE = Placement(h=4, w=2, x=3, y=0, z=0, colour="blue")
_SIDE = Placement(h=4, w=2, x=5, y=0, z=0, colour="green")

LEVELS: tuple[LevelSpec, ...] = (
    LevelSpec(0.001, math.radians(2.0), math.radians(0.4)),
    LevelSpec(0.002, math.radians(4.0), math.radians(0.8)),
    LevelSpec(0.004, math.radians(8.0), math.radians(1.4)),
    LevelSpec(0.006, math.radians(12.0), math.radians(2.0)),
    LevelSpec(
        0.006, math.radians(12.0), math.radians(2.0),
        placed=(_BASE,),
        targets=((3.5, 5.5, 0), (3.5, 1.5, 1)),
    ),
    LevelSpec(
        0.006, math.radians(12.0), math.radians(2.0),
        placed=(_BASE, _SIDE),
        targets=((1.5, 3.5, 0), (3.5, 5.5, 0), (3.5, 1.5, 1), (5.5, 1.5, 1)),
    ),
)
"""The six rungs.  Index with ``int(round(level))``; ``set_level`` clips into range."""


# ----------------------------------------------------------------------------------- the reward ---


@dataclass(frozen=True)
class PressReward:
    """Reward weights, one field per term of the spec's list (section 4).

    The two potentials are differences of a potential between consecutive ticks, so they add no
    bias to what the optimal policy is -- they only say which way is downhill.  ``progress`` is
    scaled so that driving the brick from untouched to fully seated is worth the same as the
    success bonus; ``align`` is scaled so that closing a full 6 mm of lateral error before the
    studs engage is worth about a tenth of that, enough to point the way without paying more for
    hovering in the right place than for actually pressing.
    """

    progress: float = 10.0        # x (mean stud depth / stud height), as a potential
    align: float = 150.0          # per metre of lateral error, as a potential, before engagement
    success: float = 10.0
    load: float = 2.0             # quadratic on |load| above load_limit, summed over joints
    load_limit: float = 0.8
    force: float = 0.02           # quadratic on press force above force_limit
    force_limit: float = 14.0     # N; the force map's own ceiling at the plate centre
    neighbour: float = 200.0      # per metre a placed brick has been shoved
    neighbour_slack: float = 0.0005
    broken: float = 10.0          # one-off, with termination
    action_rate: float = 0.01
    residual: float = 0.01
    time: float = 0.02


# ------------------------------------------------------------------------- observation entries ---

SEAT_ERROR = obs_spec.register(obs_spec.ObsEntry(
    "press_seat_error", 3, "task", "actor",
    note="where the tool would be if the brick were seated on the target cell, minus where "
    "forward kinematics of the read joint positions says it is, in metres; the taught grid "
    "and the known tool geometry give the first half, the encoders the second",
))
PRESS_PHASE = obs_spec.register(obs_spec.ObsEntry(
    "press_phase", len(PHASES), "fsm", "actor",
    note="one-hot of the scripted primitive's phase, in brick_builder.control.primitives.PHASES order",
))
PRESS_CONTEXT = obs_spec.register(obs_spec.ObsEntry(
    "press_context", 5, "task", "actor",
    note="brick footprint in metres (x, y), target layer height in metres, and the target's "
    "yaw relative to the arm's radial direction as (cos, sin)",
))
PRESS_TIME = obs_spec.register(obs_spec.ObsEntry(
    "press_time_frac", 1, "task", "actor",
    note="fraction of the episode horizon elapsed (tick / horizon)",
))

STUD_DEPTHS = obs_spec.register(obs_spec.ObsEntry(
    "press_stud_depths", MAX_SOCKETS, "sim", "critic",
    note="insertion depth of each of the held brick's sockets in metres, from the snap model, "
    "zero-padded to eight",
))
TRUE_MISALIGN = obs_spec.register(obs_spec.ObsEntry(
    "press_misalign", 4, "sim", "critic",
    note="true error of the held brick against the target cell: dx, dy (metres), dyaw, dtilt "
    "(radians)",
))
TRUE_IN_HAND = obs_spec.register(obs_spec.ObsEntry(
    "press_in_hand", 4, "sim", "critic",
    note="true in-hand offset of the brick against the nominal grasp: dx, dy, dz (metres) in "
    "the TCP frame, and how far it has moved since the episode started (metres)",
))
CONTACT = obs_spec.register(obs_spec.ObsEntry(
    "press_contact", 3, "sim", "critic",
    note="vertical force the grip is transmitting (N, positive pressing down), the largest "
    "distance any placed brick has been shoved (m), and how many of the held brick's studs the "
    "snap model counts as engaged",
))
DR_EXTRA = obs_spec.register(obs_spec.ObsEntry(
    "press_dr_extra", 13, "sim", "critic",
    note="this episode's in_hand (6), contact (4) and plate (3) randomization draw, in "
    "randomize.RANGES order; the dynamics and command halves are already in dr_params",
))

DR_EXTRA_KEYS: tuple[str, ...] = (
    "in_hand.dx", "in_hand.dy", "in_hand.dz", "in_hand.dyaw",
    "in_hand.weld_timeconst", "in_hand.weld_torquescale",
    "contact.brick_friction", "contact.pad_friction", "contact.table_friction",
    "contact.stud_radius_delta",
    "plate.dx", "plate.dy", "plate.dyaw",
)
assert obs_spec.get("press_dr_extra").size == len(DR_EXTRA_KEYS)


# ------------------------------------------------------------------------------------ the env ---


class PressEnv(ArmEnv):
    """``BrickBuilder/PressFromHeld-v0``.  Horizon 200 ticks (4 s)."""

    randomize_groups = ("dynamics", "command", "in_hand", "contact", "plate")

    #: The gripper is frozen: the brick is held by the soft weld that stands for the grip (spec
    #: section 2), so opening or closing the jaws changes nothing except giving the policy a
    #: seventh way to waste a dimension. Everything else -- position, tilt, spin, both compliance
    #: scalars -- is the policy's.
    frozen = frozenset({"grip"})

    def __init__(
        self,
        *,
        horizon: int = 200,
        seed: int | None = None,
        level: int = 0,
        brick: tuple[int, int] = (2, 4),
        misalign: tuple[float, float] | None = None,
        dr: bool = True,
        press: PressConfig = PressConfig(),
        snap: SnapParams = SnapParams(),
        reward: PressReward = PressReward(),
        cameras: tuple[str, ...] = (),
        curriculum: tuple[float, int] | dict | None = None,
        curriculum_seed: int = 0,
    ) -> None:
        """
        ``misalign`` overrides the level's own misalignment with one exact ``(lateral_m,
        yaw_rad)`` pair, drawn at a uniformly random *direction* rather than from a box.  That is
        what the baseline sweep needs: "success at 4 mm" should mean 4 mm, not "somewhere in a
        +-4 mm square whose average error is 2 mm".

        ``dr=False`` pins every randomization group at nominal whatever the level is, which is
        how the scripted baseline is measured (spec section 3: default snap parameters).

        ``curriculum`` is either the original ``(promote_at, window)`` pair -- promote when the
        last ``window`` episodes of this copy cleared ``promote_at``, never look back -- or a
        mapping of :class:`~brick_builder.rl.curriculum.CurriculumConfig` fields, which additionally
        spends ``replay_frac`` of its episodes on levels already unlocked (see
        step 3).  A mapping is what a YAML config can carry, which is why that is the form taken.
        ``curriculum_seed`` seeds the scheduler's own replay draw; a vectorized run gives each
        worker a different one so the workers do not replay in lockstep.
        """
        self._level = float(level)
        self._level_spec = LEVELS[int(np.clip(round(level), 0, len(LEVELS) - 1))]
        self._brick = (int(brick[0]), int(brick[1]))
        if self._brick[0] * self._brick[1] > MAX_SOCKETS:
            raise ValueError(f"brick {self._brick} has more than {MAX_SOCKETS} studs")
        self._misalign = None if misalign is None else (float(misalign[0]), float(misalign[1]))
        self._dr = bool(dr)
        self._press_cfg = press
        self._snap = snap
        self._weights = reward
        self._cameras = tuple(cameras)
        self._curriculum = curriculum if not isinstance(curriculum, dict) else None
        self._scheduler = None
        if isinstance(curriculum, dict):
            from brick_builder.rl.curriculum import CurriculumConfig, LevelScheduler

            self._scheduler = LevelScheduler(
                n_levels=len(LEVELS),
                cfg=CurriculumConfig(**curriculum),
                seed=int(curriculum_seed),
            )
        self._window: list[float] = []
        self.press: PressScripted | None = None
        super().__init__(horizon=horizon, seed=seed)
        # ArmEnv.__init__ sets `_level` to 0 on its way past, so put the requested rung back.
        self._level = float(level)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Promote along the curriculum if this copy of the environment has earned it, then reset.

        The check has to happen *before* :class:`ArmEnv`'s reset, not inside ``reset_arm_q``:
        promoting from level 3 to level 4 rebuilds the MuJoCo model, and by the time
        ``reset_arm_q`` runs this episode's randomization has already been written into the old
        one. Each worker keeps its own window, so a vectorized run naturally ends up with a mix
        of difficulties rather than every process stepping up at the same moment.

        With a :class:`~brick_builder.rl.curriculum.LevelScheduler` (the campaign's own hook) the
        same two things happen, but the scheduler also decides to spend some episodes *below*
        the level it has unlocked, so the level can go down as well as up.
        """
        if self._scheduler is not None:
            if hasattr(self, "_streak"):
                self._scheduler.record(int(round(self._level)), self._streak >= SUCCESS_TICKS)
            self.set_level(self._scheduler.next_level())
        elif self._curriculum is not None and self._window_push():
            self.set_level(min(self._level + 1, len(LEVELS) - 1))
            self._window.clear()
        return super().reset(seed=seed, options=options)

    def _window_push(self) -> bool:
        """Record the episode that just ended and say whether this copy should move up a level."""
        promote_at, window = self._curriculum
        if hasattr(self, "_streak"):
            self._window.append(1.0 if self._streak >= SUCCESS_TICKS else 0.0)
        del self._window[:-window]
        return (
            len(self._window) >= window
            and float(np.mean(self._window)) >= promote_at
            and self._level < len(LEVELS) - 1
        )

    # -- scene -------------------------------------------------------------------------------

    def scene_config(self, rng: np.random.Generator) -> SceneConfig:
        nx, ny = self._brick
        return SceneConfig(
            held=HeldBrick(nx=nx, ny=ny),
            grasp_mode="weld",
            placed=list(self._level_spec.placed),
            snap=self._snap,
            cameras=self._cameras,
            home_hover=START_HOVER,
        )

    def set_level(self, k: float) -> None:
        """Move to curriculum rung ``k``.  Rebuilds the world only if its contents change."""
        spec = LEVELS[int(np.clip(round(k), 0, len(LEVELS) - 1))]
        rebuild = spec.placed != self._level_spec.placed
        self._level = float(k)
        self._level_spec = spec
        if rebuild:
            self._build_scene(0)

    def dr_level(self) -> float:
        """Level 0..5 maps onto the 0..1 randomization fraction; ``dr=False`` pins it at 0."""
        if not self._dr:
            return 0.0
        return float(np.clip(self._level / (len(LEVELS) - 1), 0.0, 1.0))

    # -- episode setup -----------------------------------------------------------------------

    def reset_arm_q(self, rng: np.random.Generator) -> np.ndarray:
        """Draw this episode's target cell and misalignment, and return the start pose.

        Runs before the scene reset because the held brick is put in the hand by that reset, so
        the hand has to already be where this episode wants it.
        """
        spec = self._level_spec
        cx, cy, layer = spec.targets[int(rng.integers(len(spec.targets)))]
        self._target_cell = (float(cx), float(cy), int(layer))

        if self._misalign is None:
            dx, dy = rng.uniform(-spec.lateral, spec.lateral, 2)
            dyaw = float(rng.uniform(-spec.yaw, spec.yaw))
            dtilt = float(rng.uniform(-spec.tilt, spec.tilt))
        else:
            # A sweep point means one axis at a time, at an exact magnitude: the tilt is pinned
            # at zero so "success at 4 mm" is a statement about 4 mm of lateral error and
            # nothing else.
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            dx = self._misalign[0] * math.cos(angle)
            dy = self._misalign[0] * math.sin(angle)
            dyaw = self._misalign[1] * float(rng.choice([-1.0, 1.0]))
            dtilt = 0.0
        self._start_error = np.array([dx, dy, dyaw, dtilt])

        # The arm aims at where it *believes* the cell is, which is the true cell plus this
        # episode's error: the misalignment stands for a mis-taught grid and a brick that is not
        # quite where the hand thinks, neither of which an encoder can see. Every deployable
        # signal is therefore measured against `_aim`, and only the critic is told about `dx, dy`.
        true_cell = self._cell_world(cx, cy, nominal=True)
        self._aim = np.array([true_cell[0] + dx, true_cell[1] + dy, 0.0])
        self._seat_z = float(
            self.scene.plate_top + layer * self.scene.cfg.spec.height
            + tool_reach_below_tcp(self.scene)
        )
        start = np.array([self._aim[0], self._aim[1], self._seat_z + START_HOVER])
        try:
            self._start_q = _tool_ik(self.scene.kin, start, dtilt, dyaw, self.scene.home_q)
        except RuntimeError:
            # An unreachable tilt/yaw combination at the edge of the plate: fall back to the
            # tool-down solve at the same point rather than failing the whole episode.
            self._start_error[2:] = 0.0
            self._start_q = _tool_ik(self.scene.kin, start, 0.0, 0.0, self.scene.home_q)
        return self._start_q

    def task_reset(self, rng: np.random.Generator) -> None:
        cx, cy, _ = self._target_cell
        self._cell_xy = self._cell_world(cx, cy, nominal=False)[:2]
        self._streak = 0
        self._broken = False
        self._prev_action = np.zeros(8)
        self._max_force = 0.0
        self._searches = 0

        held = self.scene.held_name
        self._held_start = self._in_hand_vector()
        self._neighbour_start = {
            name: self.scene.body_pose(name)[0].copy() for name in self.scene.placed_names
        }
        self._initial_links = {
            (c.upper, c.lower) for c in self.scene.snap.connections() if c.upper != held
        }

        self.press = PressScripted(
            self._seat_z, self._press_cfg,
            grip=float(self._home_primitive.grip),
        )
        self.press.reset(self._feedback())
        self._sense()
        self._phi_progress = self._mean_depth / self.scene.cfg.spec.stud_h
        self._phi_align = -self._lateral

    # -- the scripted primitive ---------------------------------------------------------------

    def _feedback(self) -> Feedback:
        return feedback_from_readings(
            self.kin, self._last_reading, grip=float(self._cmd_echo["grip"])
        )

    def primitive_target(self, tick: int) -> PrimitiveTarget:
        target = self.press.update(self._feedback())
        self._searches = self.press.searches
        return PrimitiveTarget(
            pos=target.pos, tilt=target.tilt, spin=target.spin, grip=target.grip,
            c_vert=target.c_vert, c_lat=target.c_lat,
        )

    # -- sensing ------------------------------------------------------------------------------

    def _cell_world(self, cx: float, cy: float, *, nominal: bool) -> np.ndarray:
        """World position of a plate cell centre, from the nominal plate pose or the real one.

        The two differ by whatever the plate-pose randomization drew.  The *nominal* pose is what
        the arm aims at (a real robot only knows its taught grid), the *real* one is what decides
        which studs the brick actually landed on.
        """
        cfg = self.scene.cfg
        pitch = cfg.spec.pitch
        local = np.array([
            (cx - (self.scene.plate.nx - 1) / 2.0) * pitch,
            (cy - (self.scene.plate.ny - 1) / 2.0) * pitch,
            0.0,
        ])
        if nominal:
            origin = np.array([cfg.plate_xy[0], cfg.plate_xy[1], 0.0])
            yaw = cfg.plate_yaw
        else:
            bid = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, self.scene.plate.name)
            origin = self.scene.model.body_pos[bid].copy()
            quat = self.scene.model.body_quat[bid]
            yaw = 2.0 * math.atan2(quat[3], quat[0])
        c, s = math.cos(yaw), math.sin(yaw)
        return origin + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1], 0.0])

    def _in_hand_vector(self) -> np.ndarray:
        """The held brick's origin expressed in the TCP frame: where it sits in the hand."""
        tcp_pos, tcp_rot = self.scene.tcp_pose()
        pos, _ = self.scene.body_pose(self.scene.held_name)
        return tcp_rot.T @ (pos - tcp_pos)

    def _sense(self) -> None:
        """Cache this tick's simulator-truth measurements.

        Called once per tick, from :meth:`success` (which :class:`ArmEnv` evaluates before
        ``reward`` and ``terminated``, so all three see the same snapshot) and once more at the
        end of :meth:`task_reset` so the first observation is not built from stale numbers.
        """
        scene = self.scene
        spec = scene.cfg.spec
        held = scene.held_name

        depths = np.asarray(scene.snap.stud_depths(held), dtype=float)
        self._depths = depths
        self._mean_depth = float(np.mean(np.clip(depths, 0.0, spec.stud_h)))
        self._min_depth = float(np.min(depths))
        self._engaged = int(np.count_nonzero(depths > 0.0))

        pos, quat = scene.body_pose(held)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        yaw, pitch, _ = euler_zyx(rot)
        self._brick_pos = pos
        self._lateral = float(np.linalg.norm(pos[:2] - self._cell_xy))
        # A brick is symmetric under half a turn, and a square one under a quarter turn, so the
        # yaw error is measured modulo whichever of those this brick has.
        period = 0.5 * math.pi if self._brick[0] == self._brick[1] else math.pi
        self._yaw_error = float(math.remainder(yaw, period))
        self._tilt_error = float(abs(pitch))

        self._in_hand = self._in_hand_vector()
        self._slip = float(np.linalg.norm(self._in_hand - self._held_start))

        links = {(c.upper, c.lower) for c in scene.snap.connections() if c.upper != held}
        if not self._initial_links <= links:
            self._broken = True

        self._neighbour_shift = 0.0
        for name, start in self._neighbour_start.items():
            self._neighbour_shift = max(
                self._neighbour_shift, float(np.linalg.norm(scene.body_pose(name)[0] - start))
            )

        self._force = press_force(scene)
        self._max_force = max(self._max_force, self._force)

        q = scene.data.qpos[scene.arm_qpos_adr[:5]]
        lo, hi = self.kin.jnt_range[:, 0], self.kin.jnt_range[:, 1]
        self._at_joint_limit = bool(
            np.any(q <= lo + JOINT_LIMIT_MARGIN) or np.any(q >= hi - JOINT_LIMIT_MARGIN)
        )

    # -- observations ---------------------------------------------------------------------------

    def task_actor_entries(self) -> list[str]:
        return ["press_seat_error", "press_phase", "press_context", "press_time_frac"]

    def task_critic_entries(self) -> list[str]:
        return ["press_stud_depths", "press_misalign", "press_in_hand", "press_contact",
                "press_dr_extra"]

    def task_obs_values(self) -> dict[str, np.ndarray]:
        spec = self.scene.cfg.spec
        read_pos, _ = self.read_tcp_pose()
        believed_seat = np.array([self._aim[0], self._aim[1], self._seat_z])

        phase = np.zeros(len(PHASES))
        phase[PHASES.index(self.press.phase)] = 1.0

        # The target's yaw relative to the arm's own radial direction (design report 02 section 2
        # lists this as task context): the brick is laid out on the plate grid, so its yaw is 0,
        # and the radial direction is where the cell sits from the base.
        rel = -math.atan2(self._aim[1], self._aim[0])
        depths = np.zeros(MAX_SOCKETS)
        depths[: self._depths.size] = self._depths

        return {
            "press_seat_error": believed_seat - read_pos,
            "press_phase": phase,
            "press_context": np.array([
                self._brick[0] * spec.pitch, self._brick[1] * spec.pitch,
                self._target_cell[2] * spec.height, math.cos(rel), math.sin(rel),
            ]),
            "press_time_frac": np.array([self._tick / self.horizon]),
            "press_stud_depths": depths,
            "press_misalign": np.array([
                self._brick_pos[0] - self._cell_xy[0], self._brick_pos[1] - self._cell_xy[1],
                self._yaw_error, self._tilt_error,
            ]),
            "press_in_hand": np.array([*self._in_hand, self._slip]),
            "press_contact": np.array([self._force, self._neighbour_shift, float(self._engaged)]),
            "press_dr_extra": self._dr_extra_vec,
        }

    @property
    def _dr_extra_vec(self) -> np.ndarray:
        from brick_builder.scene import randomize

        sample = getattr(self, "_dr_sample", {})
        return np.array(
            [float(sample.get(k, randomize.NOMINAL[k])) for k in DR_EXTRA_KEYS], dtype=np.float32
        )

    # -- reward and termination -----------------------------------------------------------------

    def success(self) -> bool:
        self._sense()
        spec = self.scene.cfg.spec
        seated = self._min_depth >= SEAT_FRACTION * spec.stud_h
        right_cell = self._lateral <= CELL_TOLERANCE * spec.pitch
        self._streak = self._streak + 1 if (seated and right_cell) else 0
        return self._streak >= SUCCESS_TICKS

    def reward(self) -> float:
        w = self._weights
        spec = self.scene.cfg.spec

        phi_p = self._mean_depth / spec.stud_h
        r = w.progress * (phi_p - self._phi_progress)
        self._phi_progress = phi_p

        phi_a = -self._lateral
        if self._engaged == 0:
            r += w.align * (phi_a - self._phi_align)
        self._phi_align = phi_a

        if self._streak >= SUCCESS_TICKS:
            r += w.success
        if self._broken:
            r -= w.broken

        load = np.abs(self._last_reading.load)
        r -= w.load * float(np.sum(np.square(np.maximum(load - w.load_limit, 0.0))))
        r -= w.force * max(self._force - w.force_limit, 0.0) ** 2
        r -= w.neighbour * max(self._neighbour_shift - w.neighbour_slack, 0.0)

        action = np.array([
            *self._res_pos, self._res_tilt, self._res_spin, self._grip_target,
            self._c_vert, self._c_lat,
        ])
        r -= w.action_rate * float(np.sum(np.square(action - self._prev_action)))
        self._prev_action = action
        r -= w.residual * float(
            np.sum(np.square(self._res_pos / POS_CLIP))
            + (self._res_tilt / ANG_CLIP) ** 2 + (self._res_spin / ANG_CLIP) ** 2
        )
        r -= w.time
        return float(r)

    def terminated(self) -> bool:
        return bool(
            self._broken
            or self._slip > SLIP_LIMIT
            or self._at_joint_limit
            or np.any(self._last_reading.tripped)
        )

    def failure_class(self) -> FailureClass | None:
        return self.diagnose()

    def diagnose(self) -> FailureClass | None:
        """Why this episode is not a success, from simulator state.  ``None`` means it is one.

        Called both as :meth:`failure_class` (which :class:`ArmEnv` only consults on a task
        termination) and from :meth:`episode_result_extra`, so a horizon truncation still records
        the finer diagnosis alongside the base class's ``timeout``.
        """
        spec = self.scene.cfg.spec
        if self._streak >= SUCCESS_TICKS:
            return None
        if self._broken or self._neighbour_shift > 0.002:
            return FailureClass.STRUCTURE_DAMAGE
        if self._slip > SLIP_LIMIT:
            return FailureClass.SLIP
        if np.any(self._last_reading.tripped):
            return FailureClass.OVERLOAD_TRIP
        if self._at_joint_limit:
            return FailureClass.KINEMATIC
        if self._min_depth >= SEAT_FRACTION * spec.stud_h:
            return FailureClass.WRONG_CELL
        if self._engaged > 0:
            return FailureClass.PARTIAL_SEAT
        if self._peak_load >= 0.5:
            return FailureClass.JAM
        return FailureClass.TIMEOUT

    def episode_result_extra(self) -> dict[str, Any]:
        spec = self.scene.cfg.spec
        return {
            "seat_depth_error": float(max(spec.stud_h - self._min_depth, 0.0)),
            "structure_intact": bool(not self._broken and self._neighbour_shift <= 0.002),
            "params": {
                "diagnosis": None if self.diagnose() is None else self.diagnose().value,
                # Which rung this episode actually ran at, so a curriculum run's per-level
                # success rate can be read straight off the episode records instead of being
                # inferred from the misalignment draw.
                "level": int(round(self._level)),
                "brick": list(self._brick),
                "target_cell": list(self._target_cell),
                "start_error": self._start_error.tolist(),
                "lateral_mm": 1e3 * float(np.linalg.norm(self._start_error[:2])),
                "yaw_deg": math.degrees(self._start_error[2]),
                "min_depth_mm": 1e3 * self._min_depth,
                "mean_depth_mm": 1e3 * self._mean_depth,
                "final_lateral_mm": 1e3 * self._lateral,
                "press_force_n": self._max_force,
                "searches": self._searches,
                "press_phase": self.press.phase,
                "press_time_s": self.press.press_time,
                "neighbour_shift_mm": 1e3 * self._neighbour_shift,
            },
        }


# ----------------------------------------------------------------------------------- helpers ---


def press_force(scene) -> float:
    """Vertical force the grip is transmitting, newtons, positive when pressing down.

    In ``weld`` grasp mode the brick hangs off the gripper by one weld equality, so the
    constraint force of that equality *is* the force between tool and brick -- which, once the
    brick is on the plate, is the force reaching the plate.  ``equality_forces`` returns a weld's
    translation rows in world coordinates, so the world -z component is the press.  With nothing
    being pressed this reads about -0.12 N: the brick's own weight, which the grip is holding up.
    """
    if scene.weld_eq < 0 or not scene.data.eq_active[scene.weld_eq]:
        return 0.0
    return float(-equality_forces(scene.model, scene.data, n_rows=6)[scene.weld_eq][2])


def _tool_ik(kin, pos, tilt: float, spin: float, q0: np.ndarray, iters: int = 200) -> np.ndarray:
    """Joint angles for a TCP at ``pos`` with tool pitch ``tilt`` and yaw ``spin``.

    The same fixed-point construction as ``scene.compose.tool_down_ik`` -- ``ArmKinematics.ik``
    has no yaw target, so the yaw is chased by nudging the ``wrist_roll`` seed -- generalized to
    a non-zero tool pitch, which the press needs because a tilt error is part of the episode's
    start pose and a tool-down solve could not express it.
    """
    q = np.asarray(q0, dtype=float).copy()
    result = None
    for _ in range(8):
        result = kin.ik(np.asarray(pos, dtype=float), float(tilt), math.pi, q, iters=iters)
        q = result.q.copy()
        achieved, _, _ = euler_zyx(kin.fk(q)[1])
        err = math.remainder(spin - achieved, 2.0 * math.pi)
        if abs(err) < math.radians(0.05):
            break
        q[4] = float(np.clip(q[4] + err, kin.jnt_range[4, 0], kin.jnt_range[4, 1]))
    if result is None or not result.converged:
        raise RuntimeError(
            f"no pose at {np.asarray(pos)} with tilt {math.degrees(tilt):.1f} deg and yaw "
            f"{math.degrees(spin):.1f} deg "
            f"(residual {0.0 if result is None else result.pos_error * 1e3:.2f} mm)"
        )
    return q


def oracle_action(env: PressEnv, gain: float = 0.25, lift: float = 0.004,
                  align_tol: float = 0.0008, yaw_tol: float = math.radians(0.5)) -> np.ndarray:
    """A hand-written residual that cheats: it reads the true misalignment and cancels it.

    This is not a policy, it is the environment's own shaping test.  If a residual that knows
    exactly which way the brick is off does not score clearly better than doing nothing, the
    reward is not pointing anywhere useful and no learner will find its way either.

    Three things it does that a blind policy would have to learn:

    - **Lift out of contact first.**  The brick starts 5 mm above the seated height and a stud is
      4.5 mm tall, so it is touching the stud tops within a tenth of a second.  Dragging a loaded
      brick sideways scrubs; the oracle raises the residual's z by ``lift`` until it is lined up
      and only then lets the primitive's own ramp take it down.
    - **Correct gently.**  The residual *accumulates*, so an action is a rate, not a target, and
      feeding back the full error every tick is an integrator with unit gain against a plant with
      a hundred milliseconds of servo lag: it oscillates and, measured, it scored *worse* than
      doing nothing.  ``gain`` 0.25 closes a 6 mm error in about a fifth of a second and settles.
    - **Stop once a stud is in.**  Shoving a half-inserted brick sideways levers it back out.
    """
    action = scripted_action(env)      # same compliance as the baseline: only the residual differs
    if env._engaged > 0:
        return action
    error = env._cell_xy - env._brick_pos[:2]
    step_xy = np.clip(gain * error / POS_SCALE[:2], -1.0, 1.0)
    step_spin = float(np.clip(-gain * env._yaw_error / SPIN_SCALE, -1.0, 1.0))
    # Lift for a yaw error too, not only a lateral one: a brick that is only turned has its
    # centre in the right place, so a lift gated on lateral error alone would leave the oracle
    # trying to twist a brick that is pressed onto the stud tops. Measured, that was the whole
    # difference between 0 and 12 out of 12 at 4 degrees.
    off = float(np.linalg.norm(error)) > align_tol or abs(env._yaw_error) > yaw_tol
    step_z = float(np.clip(((lift if off else 0.0) - env._res_pos[2]) / POS_SCALE[2], -1.0, 1.0))
    action[0], action[1], action[2], action[4] = step_xy[0], step_xy[1], step_z, step_spin
    return action


def scripted_action(env: PressEnv) -> np.ndarray:
    """The action that *is* the scripted baseline: no residual, the primitive's own compliance.

    The position/orientation half is zero, which by construction leaves the executed target equal
    to the primitive's own (the residual accumulates from zero and so never moves).  The
    compliance half is not a residual -- spec section 1 has ``c_vert``/``c_lat`` map straight from
    this tick's action onto ``[0.3, 1.0]`` -- so zeros there would command 0.65/0.65, the middle
    of the range, rather than the 1.0/0.45 the press primitive asks for.  A baseline measured
    with zeros would be a measurement of a *different* scripted press than the one
    :class:`~brick_builder.control.primitives.PressScripted` describes, and the difference is exactly
    the sideways softness the whole skill depends on.
    """
    cfg = env.press.cfg
    span = TORQUE_HI - TORQUE_LO
    return np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.0 * (cfg.c_vert - TORQUE_LO) / span - 1.0,
        2.0 * (cfg.c_lat - TORQUE_LO) / span - 1.0,
    ], dtype=np.float32)


def scripted_episode(env: PressEnv, seed: int = 0, action=None) -> dict:
    """Run one episode and return a trace of everything the figures and the sweeps need.

    ``action`` is ``None`` for the scripted baseline (:func:`scripted_action`), a callable
    ``action(env) -> np.ndarray`` for a hand-written residual such as :func:`oracle_action`, or a
    fixed array.
    """
    obs, _ = env.reset(seed=seed)
    zero = scripted_action(env)
    trace: dict[str, Any] = {
        "t": [], "cmd_z": [], "tcp_z": [], "cmd_xy": [], "phase": [], "force": [],
        "depths": [], "reward": [], "seat_z": env._seat_z, "mean_depth": [],
    }
    ret = 0.0
    info: dict[str, Any] = {}
    for tick in range(env.horizon):
        a = zero if action is None else (action(env) if callable(action) else np.asarray(action))
        trace["phase"].append(env.press.phase)
        obs, reward, terminated, truncated, info = env.step(a)
        ret += reward
        trace["t"].append(tick * 0.020)
        trace["cmd_z"].append(float(env._cmd_echo["pos"][2]))
        trace["cmd_xy"].append(np.asarray(env._cmd_echo["pos"][:2]).copy())
        trace["tcp_z"].append(float(env.tcp_pose()[0][2]))
        trace["force"].append(env._force)
        trace["depths"].append(env._depths.copy())
        trace["mean_depth"].append(env._mean_depth)
        trace["reward"].append(reward)
        if terminated or truncated:
            break
    trace["success"] = bool(info.get("success", False))
    trace["return"] = ret
    trace["searches"] = env._searches
    trace["steps"] = len(trace["t"])
    trace["info"] = info
    trace["depths"] = np.asarray(trace["depths"])
    return trace


# ------------------------------------------------------------------------------- the sweeps ---


def evaluate_point(job: dict) -> dict:
    """Run one cell of a misalignment sweep and return its counts.  A top-level function so
    ``multiprocessing`` can pickle it; one process builds one environment and reuses it."""
    env = PressEnv(
        level=job.get("level", 3),
        brick=tuple(job["brick"]),
        misalign=(job["lateral"], job["yaw"]),
        dr=job.get("dr", False),
        seed=0,
    )
    policy = None
    if job.get("checkpoint"):
        policy = _torch_policy(job["checkpoint"])
    elif job.get("policy") == "oracle":
        policy = oracle_action
    successes = 0
    returns: list[float] = []
    steps: list[int] = []
    diagnoses: dict[str, int] = {}
    forces: list[float] = []
    depth_errors: list[float] = []
    try:
        for k in range(job["episodes"]):
            trace = scripted_episode(env, seed=job.get("seed0", 0) + k, action=policy)
            successes += int(trace["success"])
            returns.append(trace["return"])
            steps.append(trace["steps"])
            params = trace["info"].get("params", {})
            key = str(params.get("diagnosis"))
            diagnoses[key] = diagnoses.get(key, 0) + 1
            forces.append(params.get("press_force_n", 0.0))
            depth_errors.append(trace["info"].get("seat_depth_error", 0.0) or 0.0)
    finally:
        env.close()
    return {
        **{k: v for k, v in job.items() if k != "checkpoint"},
        "successes": successes,
        "n": job["episodes"],
        "mean_return": float(np.mean(returns)),
        "median_steps": float(np.median(steps)),
        "diagnoses": diagnoses,
        "peak_force_n": float(np.max(forces)) if forces else 0.0,
        "median_seat_depth_error_m": float(np.median(depth_errors)),
        "learned": bool(job.get("checkpoint")),
    }


def _torch_policy(checkpoint):
    """A deterministic actor loaded from a PPO checkpoint, as an ``action(env)`` callable."""
    import torch

    from brick_builder.rl.ppo_asym import load_actor, select_action

    actor, norm, _ = load_actor(checkpoint)

    def action(env: PressEnv) -> np.ndarray:
        obs = env._assemble_obs()["actor"]
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            act, _ = select_action(actor, norm, tensor, deterministic=True)
        return np.clip(act.squeeze(0).numpy(), -1.0, 1.0)

    return action


def sweep(jobs: list[dict], workers: int = 6) -> list[dict]:
    """Run a list of :func:`evaluate_point` jobs, at most ``workers`` processes at a time.

    ``forkserver`` rather than ``fork``: MuJoCo's EGL context does not survive a fork, and the
    parent of a media build usually has one open already.
    """
    if workers <= 1:
        return [evaluate_point(job) for job in jobs]
    import multiprocessing as mp

    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=min(workers, len(jobs)), maxtasksperchild=4) as pool:
        return pool.map(evaluate_point, jobs)


def baseline_jobs(
    bricks=((2, 4), (2, 2)),
    laterals=(0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006),
    yaws=(0.0, math.radians(2), math.radians(4), math.radians(8), math.radians(12)),
    episodes: int = 100,
    checkpoint=None,
) -> list[dict]:
    """The job list of spec section 3's baseline curve: lateral 0-6 mm and yaw 0-12 degrees,
    one axis at a time, for both brick sizes, at default snap parameters and no randomization."""
    jobs = []
    for brick in bricks:
        for lateral in laterals:
            jobs.append(dict(brick=list(brick), lateral=float(lateral), yaw=0.0, axis="lateral",
                             episodes=episodes, checkpoint=checkpoint))
        for yaw in yaws:
            if yaw == 0.0:
                continue           # the (0, 0) point is already in the lateral list
            jobs.append(dict(brick=list(brick), lateral=0.0, yaw=float(yaw), axis="yaw",
                             episodes=episodes, checkpoint=checkpoint))
    return jobs


# ----------------------------------------------------------------------------------- figures ---


def make_press_figures(out_dir="docs/media/envs", results=None, episodes: int = 100,
                       workers: int = 6, seed: int = 0) -> dict:
    """The press environment's figures: the scripted baseline curve and one press signature.

    ``results`` is the output of ``sweep(baseline_jobs(...))``; pass a previously measured list
    (``runs/press_baseline/curve.json``) to re-draw the figure without re-running 2 200 episodes,
    or leave it ``None`` to measure it here.  This is the ``scripts/make_media.py`` entry point.
    """
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.control.primitives import shade_phases
    from brick_builder.eval.wilson import wilson_interval

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    if results is None:
        results = sweep(baseline_jobs(episodes=episodes), workers=workers)
    elif isinstance(results, (str, Path)):
        results = json.loads(Path(results).read_text())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=150, sharey=True)
    for ax, axis, xlabel, scale in (
        (axes[0], "lateral", "lateral misalignment [mm]", 1e3),
        (axes[1], "yaw", "yaw misalignment [deg]", 180.0 / math.pi),
    ):
        for brick, colour in (((2, 4), "#1f6feb"), ((2, 2), "#bf8700")):
            rows = sorted(
                (r for r in results if tuple(r["brick"]) == brick and r["axis"] == axis),
                key=lambda r: r["lateral"] + r["yaw"],
            )
            if not rows:
                continue
            xs = [scale * (r["lateral"] + r["yaw"]) for r in rows]
            point = [r["successes"] / r["n"] for r in rows]
            lows = [wilson_interval(r["successes"], r["n"]).low for r in rows]
            highs = [wilson_interval(r["successes"], r["n"]).high for r in rows]
            ax.plot(xs, point, "-o", color=colour, markersize=4,
                    label=f"{brick[0]}x{brick[1]} (n={rows[0]['n']} per point)")
            ax.fill_between(xs, lows, highs, color=colour, alpha=0.18)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("scripted press success rate")
    fig.suptitle(
        "Scripted press baseline, default snap parameters, weld grasp, no randomization.\n"
        "Bands are 95 % Wilson intervals.",
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / "press_baseline_curve.png"
    fig.savefig(path)
    plt.close(fig)
    out["press_baseline_curve.png"] = str(path)

    # -- the press signature: force and every stud's depth over one episode ------------------
    # A signature figure should show the signature of a press that works: level 0, and the first
    # seed of a handful that actually seats (a yaw error of more than about 3 degrees leaves one
    # stud in and seven out, which is a different and separately documented story).
    env = PressEnv(level=0, dr=False, seed=seed)
    try:
        trace = scripted_episode(env, seed=seed)
        for candidate in range(seed, seed + 6):
            if trace["success"]:
                break
            trace = scripted_episode(env, seed=candidate)
        stud_h = env.scene.cfg.spec.stud_h
    finally:
        env.close()
    t = np.asarray(trace["t"])
    depths = np.asarray(trace["depths"])
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0), dpi=150, sharex=True)
    axes[0].plot(t, 1e3 * (np.asarray(trace["tcp_z"]) - trace["seat_z"]), color="#1f6feb",
                 label="measured")
    axes[0].plot(t, 1e3 * (np.asarray(trace["cmd_z"]) - trace["seat_z"]), color="#8250df",
                 label="commanded")
    axes[0].axhline(0.0, color="#d1242f", linestyle="--", label="seated height")
    axes[0].set_ylabel("TCP above seat [mm]")
    axes[0].legend(fontsize=7)
    axes[1].plot(t, trace["force"], color="#cf222e")
    axes[1].set_ylabel("press force [N]")
    for k in range(depths.shape[1]):
        axes[2].plot(t, 1e3 * depths[:, k], linewidth=1.0)
    axes[2].axhline(1e3 * SEAT_FRACTION * stud_h, color="#57606a", linestyle="--",
                    label=f"success band ({100 * SEAT_FRACTION:.0f} % of stud height)")
    axes[2].set_ylabel("stud depth [mm]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        shade_phases(ax, t, trace["phase"])
    fig.suptitle(
        f"One scripted press of a 2x4, level 0 ({'seated' if trace['success'] else 'not seated'} "
        f"in {trace['steps']} ticks): every one of the eight studs goes in together",
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / "press_signature.png"
    fig.savefig(path)
    plt.close(fig)
    out["press_signature.png"] = str(path)
    return out
