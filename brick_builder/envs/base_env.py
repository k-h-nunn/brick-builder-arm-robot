"""``ArmEnv``: the shared multi-rate control loop every BrickBuilder task is built from.

Implements the environment contract exactly (timing, action integration, observation assembly)
and decision 6 of section 6b (``BusEmulator`` drives the per-tick read skew, command latency and
packet loss that ``ServoBus`` applies). A task subclasses ``ArmEnv`` and overrides the hooks
listed under "Hooks for subclasses" below; nothing in this file should need editing to add a new
task. ``reach_env.py`` is the first (and, for this package, only) such subclass.

Multi-rate tick (physics 2 ms, control 20 ms -> 10 physics steps per ``env.step``):

    BusEmulator.tick(t) -> timing for this control period
    ServoBus.write(goal, torque_limit)              # once, unless timing.write_lost
    for each of the 10 physics steps:
        ServoBus.substep()
        snap.update()                                # every snap_every physics steps
        mujoco.mj_step()
    ServoBus.read()                                  # once; per-servo stale entries are held
                                                       # over from the previous tick on packet loss

Action integration: the 8 numbers in ``[-1, 1]`` are ``[dx, dy, dz, d_tilt, d_spin, d_grip,
c_vert, c_lat]``. The first five accumulate onto a *residual* that is clipped to a box around
whatever the scripted primitive's own target is this tick (``primitive_target()``; an env with no
primitive, like Reach, holds that centre fixed at the pose recorded at reset). Because the
residual -- not the absolute target -- is what accumulates and clips, a policy that always
outputs zero leaves the residual at zero forever and the executed target exactly equal to the
primitive's own target: "zero residual reproduces the scripted baseline" is true by construction,
not by a separate check. ``c_vert``/``c_lat`` are not residuals; they map straight from this
tick's action to a servo torque-limit fraction, as the spec's wording ("map to ...") says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from brick_builder.envs import obs_spec
from brick_builder.eval.results import EpisodeResult, FailureClass
from brick_builder.robot.bus_emulator import BusEmulator
from brick_builder.robot.kinematics import ArmKinematics, euler_zyx
from brick_builder.robot.patch_so101 import ARM_JOINTS
from brick_builder.robot.servo_model import ServoBus, ServoReadings
from brick_builder.scene import randomize
from brick_builder.scene.compose import Scene, SceneConfig, build_scene, gripper_rest_angle, reset_scene

# ------------------------------------------------------------------------------------- timing ---

PHYSICS_DT = 0.002
CONTROL_DT = 0.020
PHYSICS_PER_TICK = round(CONTROL_DT / PHYSICS_DT)
assert PHYSICS_PER_TICK == 10

# ------------------------------------------------------------------------------------- action ---

ACTION_DIM = 8
POS_SCALE = np.array([0.002, 0.002, 0.002])     # m per unit action, dx dy dz
TILT_SCALE = math.radians(2.0)                  # rad per unit action
SPIN_SCALE = math.radians(2.0)
GRIP_SCALE = 0.05                               # rad per unit action

POS_CLIP = 0.012                                # m, residual box half-width (spec: +-12 mm)
ANG_CLIP = math.radians(20.0)                   # rad, residual box half-width (spec: +-20 deg)

TORQUE_LO, TORQUE_HI = 0.3, 1.0                 # c_vert/c_lat map onto this torque-limit range
_VERT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
_LAT_JOINTS = ("shoulder_pan", "wrist_roll")
_VERT_INDEX = [ARM_JOINTS.index(n) for n in _VERT_JOINTS]
_LAT_INDEX = [ARM_JOINTS.index(n) for n in _LAT_JOINTS]

HISTORY_TICKS = 4
"""How many past ticks of each proprioceptive entry the actor sees."""

# ``_TOOL_DOWN_ROLL`` mirrors the private constant of the same name in scene/compose.py
# (tool-down = pitch 0, roll pi in ArmKinematics.ik's Rz(yaw) @ Ry(pitch) @ Rx(roll) convention).
# Duplicated rather than imported: compose.py is another package's file and this is a one-line
# geometric fact, not a piece of behaviour that the two modules need to share a definition of.
_TOOL_DOWN_ROLL = math.pi

# The 17 randomize.RANGES keys (dynamics.* then command.*) surfaced in the critic's "dr_params"
# entry, in RANGES' own order. Fixed so the entry has a constant size across every level.
DR_PARAM_KEYS: tuple[str, ...] = (
    "dynamics.kp", "dynamics.forcerange", "dynamics.damping", "dynamics.frictionloss",
    "dynamics.armature", "dynamics.backlash_deg", "dynamics.vin", "dynamics.link_mass_scale",
    "command.deadband_ticks", "command.rate_limit", "command.action_delay_s",
    "command.action_delay_jitter_s", "command.read_skew_s", "command.read_noise_ticks",
    "command.obs_delay_ticks", "command.period_jitter_s", "command.packet_loss",
)
assert obs_spec.get("dr_params").size == len(DR_PARAM_KEYS)


@dataclass(frozen=True)
class PrimitiveTarget:
    """One tick's worth of scripted-primitive output: a TCP pose plus grip/compliance.

    ``pos``/``tilt``/``spin`` describe the tool pose the same way the action integrator's
    residual does (world position, and (tilt, spin) = pitch/yaw offsets from tool-down). ``grip``
    is a gripper joint angle [rad]; ``c_vert``/``c_lat`` are torque-limit fractions in
    ``[0.3, 1.0]``. An env with no scripted primitive (``primitive_target`` returns ``None``)
    never produces one; :class:`ArmEnv` then holds the residual's centre at the pose recorded at
    reset.
    """

    pos: np.ndarray
    tilt: float
    spin: float
    grip: float
    c_vert: float = 1.0
    c_lat: float = 1.0


POSE_IK_POS_TOL = 2e-4                       # m, solver tolerance on the TCP position
POSE_IK_ANG_TOL = math.radians(0.2)          # rad, solver tolerance on tool pitch/roll
POSE_IK_YAW_TOL = math.radians(0.1)          # rad, solver tolerance on tool spin
POSE_IK_ANG_ACCEPT = math.radians(2.0)       # rad, how far off the tool axis a usable answer may be
POSE_IK_DIFF_ITERS = 2                       # damped-least-squares steps before falling back

# What "the IK failed" means, and why the spin is not part of it.
#
# The five joints reach a 5-dimensional set of poses, and a tick asks for six numbers: position,
# tool pitch/roll and spin.  When the tool points straight down -- every scripted primitive, and
# anything a trained policy does near a press -- all six are available at once and every residual
# comes out at 1e-5 or below.  Tilt the tool and they stop being independent, and then no solver
# can deliver all six; the only question is where the leftover goes.  `ArmKinematics.ik_diff`
# puts it in the spin (see `YAW_WEIGHT`), so a target counts as reached when the *position* is
# inside 0.2 mm and the tool axis inside 2 degrees.  A failure is then what it says: a target this
# arm cannot get to, from joint limits or from being outside the workspace.
#
# The previous solver gated on the spin as well, but measured it (and the position and pitch/roll)
# at a different joint vector than the one it returned, which made `ik_failed` fire on 76 % of
# random-policy ticks while the pose it handed back was in fact fine.  Freezing the whole joint
# goal -- position included -- because a degree of spin was unavailable is not a behaviour worth
# keeping, so this does not keep it; the change is measured in the WP-PERF report.


@dataclass
class IKStats:
    """Running per-environment tally of what the per-tick IK actually did.

    Public so a throughput or convergence measurement can read it without monkey-patching; the
    environment resets it at every ``reset()``, and ``EpisodeResult.params`` carries the two
    numbers an episode is judged on (``ik_failures``, ``ik_fallbacks``).
    """

    solves: int = 0            # calls to _pose_ik
    iterations: int = 0        # damped-least-squares steps taken by the differential solver
    fallbacks: int = 0         # solves where the differential step did not converge in time
    failures: int = 0          # solves that came back unusable even after the fallback
    pos_error_sum: float = 0.0
    pos_error_max: float = 0.0
    ang_error_sum: float = 0.0
    ang_error_max: float = 0.0
    yaw_error_sum: float = 0.0
    yaw_error_max: float = 0.0

    def record(self, *, iterations: int, fell_back: bool, failed: bool,
               pos_error: float, ang_error: float, yaw_error: float) -> None:
        self.solves += 1
        self.iterations += iterations
        self.fallbacks += int(fell_back)
        self.failures += int(failed)
        self.pos_error_sum += pos_error
        self.pos_error_max = max(self.pos_error_max, pos_error)
        self.ang_error_sum += ang_error
        self.ang_error_max = max(self.ang_error_max, ang_error)
        self.yaw_error_sum += yaw_error
        self.yaw_error_max = max(self.yaw_error_max, yaw_error)

    def summary(self) -> dict[str, float]:
        """Means and maxima, safe to call on an untouched tally."""
        n = max(self.solves, 1)
        return {
            "solves": float(self.solves),
            "iterations_per_solve": self.iterations / n,
            "fallback_rate": self.fallbacks / n,
            "failure_rate": self.failures / n,
            "pos_error_mean": self.pos_error_sum / n,
            "pos_error_max": self.pos_error_max,
            "ang_error_mean": self.ang_error_sum / n,
            "ang_error_max": self.ang_error_max,
            "yaw_error_mean": self.yaw_error_sum / n,
            "yaw_error_max": self.yaw_error_max,
        }


def _pose_ik(
    kin: ArmKinematics, pos: np.ndarray, tilt: float, spin: float, q0: np.ndarray,
    *, outer: int = 2, iters: int = 6, yaw_tol: float = POSE_IK_YAW_TOL,
    stats: IKStats | None = None,
) -> tuple[np.ndarray, bool]:
    """Joint angles for TCP position ``pos``, pitch offset ``tilt``, spin (yaw) ``spin``.

    Between two control ticks the commanded pose moves by a couple of millimetres and a couple of
    degrees, so the normal case is at most two damped-least-squares steps warm-started from the
    previous tick's answer: :meth:`ArmKinematics.ik_diff`, with the analytic ``mj_jacSite``
    Jacobian. Only when that does not land inside tolerance -- a target that has jumped, or one
    that is not reachable at all -- does the full Levenberg-Marquardt solver run, from the same
    seed and with the fixed-point yaw chase on ``wrist_roll`` that
    ``scene.compose.tool_down_ik`` uses.

    The answer is accepted on the same bound either way -- the TCP within 0.2 mm of where it was
    asked to go and the tool axis within 2 degrees, both measured at the joint angles actually
    returned; see the note above on why the spin is reported but not gated on. Returns
    ``(q, converged)``; the caller is responsible for keeping the previous ``q`` on
    ``converged=False``.
    """
    pos = np.asarray(pos, dtype=float)
    diff = kin.ik_diff(
        pos, float(tilt), _TOOL_DOWN_ROLL, q0, target_yaw=float(spin),
        iters=POSE_IK_DIFF_ITERS, pos_tol=POSE_IK_POS_TOL, ang_tol=POSE_IK_ANG_TOL,
        yaw_tol=yaw_tol,
    )
    ok = diff.pos_error < POSE_IK_POS_TOL and diff.ang_error < POSE_IK_ANG_ACCEPT
    if ok:
        if stats is not None:
            stats.record(
                iterations=diff.iterations, fell_back=False, failed=False,
                pos_error=diff.pos_error, ang_error=diff.ang_error,
                yaw_error=abs(diff.yaw_error),
            )
        return diff.q, True

    q = np.asarray(q0, dtype=float).copy()
    err = 0.0
    for _ in range(outer):
        result = kin.ik(
            pos, float(tilt), _TOOL_DOWN_ROLL, q, iters=iters,
            pos_tol=POSE_IK_POS_TOL, ang_tol=POSE_IK_ANG_TOL,
        )
        q = result.q.copy()
        achieved_yaw, _, _ = euler_zyx(kin.fk(q)[1])
        err = math.remainder(spin - achieved_yaw, 2.0 * math.pi)
        if abs(err) < yaw_tol:
            break
        q[4] = float(np.clip(q[4] + err, kin.jnt_range[4, 0], kin.jnt_range[4, 1]))
    # `iters=0` runs no Newton step, it only evaluates: this is the residual at the joint angles
    # that are actually about to be returned, nudged wrist_roll and all.
    final = kin.ik(pos, float(tilt), _TOOL_DOWN_ROLL, q, iters=0)
    ang_error = max(abs(final.pitch_error), abs(final.roll_error))
    ok = final.pos_error < POSE_IK_POS_TOL and ang_error < POSE_IK_ANG_ACCEPT
    if stats is not None:
        stats.record(
            iterations=diff.iterations, fell_back=True, failed=not ok,
            pos_error=final.pos_error, ang_error=ang_error, yaw_error=abs(err),
        )
    return q, ok


class ArmEnv(gym.Env):
    """Base class for every BrickBuilder task.

    Subclasses set ``horizon`` (via ``super().__init__``) and override the hooks below. Nothing
    about the multi-rate loop, action integration, IK or the shared proprioceptive/critic
    observation entries needs to change per task.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    #: Groups drawn from ``brick_builder.scene.randomize`` every reset (see ``dr_level``). Every arm
    #: env randomizes its own dynamics/command path by default -- it is close to free (the model
    #: writes and one ``ServoBus``/``BusEmulator`` rebuild happen once per episode, not per tick)
    #: and it is what makes the critic's ``dr_params`` entry mean anything.
    randomize_groups: tuple[str, ...] = ("dynamics", "command")

    #: Action components this task holds fixed regardless of what the policy outputs. Reach sets
    #: ``{"grip", "compliance"}`` (frozen open gripper, frozen torque limit 1.0 on every joint).
    frozen: frozenset[str] = frozenset()

    def __init__(self, *, horizon: int, seed: int | None = None) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self.actor_names = list(obs_spec.PROPRIO_ENTRIES) + list(self.task_actor_entries())
        obs_spec.check_sources(self.actor_names, scope="actor")
        self.critic_names = (
            list(self.actor_names) + list(obs_spec.CRITIC_ONLY_ENTRIES) + list(self.task_critic_entries())
        )
        self._actor_layout = obs_spec.make_layout(self.actor_names)
        self._critic_layout = obs_spec.make_layout(self.critic_names)
        self.observation_space = spaces.Dict({
            "actor": obs_spec.make_space(self.actor_names),
            "critic": obs_spec.make_space(self.critic_names),
        })

        self._level: float = 0.0
        self._build_scene(seed)

    # =============================================================== hooks for subclasses ===

    def scene_config(self, rng: np.random.Generator) -> SceneConfig:
        """The :class:`SceneConfig` to build the world from. Called once, at construction (and
        again only if a subclass calls :meth:`_build_scene` itself, e.g. because ``set_level``
        changed the scene's topology, not just its difficulty)."""
        raise NotImplementedError

    def primitive_target(self, tick: int) -> PrimitiveTarget | None:
        """This tick's scripted-primitive pose, or ``None`` for "no primitive" (Reach)."""
        return None

    def task_actor_entries(self) -> list[str]:
        """Names of this task's own :mod:`obs_spec` entries, appended to the shared
        proprioceptive set. Register them at module import time with ``obs_spec.register``."""
        return []

    def task_critic_entries(self) -> list[str]:
        """Names of this task's own critic-only entries, appended after the shared ones."""
        return []

    def task_obs_values(self) -> dict[str, np.ndarray]:
        """This tick's values for every name :meth:`task_actor_entries`/`task_critic_entries`
        declared."""
        return {}

    def reset_arm_q(self, rng: np.random.Generator) -> np.ndarray | None:
        """The five arm joint angles this episode starts from, or ``None`` for the home pose.

        Called *before* the scene reset, unlike :meth:`task_reset`, because the held brick is
        placed in the hand by that reset and so has to know where the hand is going to be. A task
        whose episode starts somewhere other than the scene's own home pose (PressFromHeld: over
        a drawn target cell, with a drawn misalignment) draws that here and returns the joint
        angles; everything that does not need it inherits the default and behaves exactly as
        before.
        """
        return None

    def task_reset(self, rng: np.random.Generator) -> None:
        """Draw whatever this task's own episode needs (a goal, a misalignment, ...)."""
        return None

    def reward(self) -> float:
        raise NotImplementedError

    def terminated(self) -> bool:
        raise NotImplementedError

    def success(self) -> bool:
        raise NotImplementedError

    def failure_class(self) -> FailureClass | None:
        """Why this episode ended badly, when ``terminated()`` is true and ``success()`` is not.
        The base class fills in ``TIMEOUT`` on its own for a horizon truncation."""
        return None

    def episode_result_extra(self) -> dict[str, Any]:
        """Extra :class:`EpisodeResult` fields/`params` entries this task wants recorded. Keys
        matching a dataclass field (``seat_depth_error``, ``structure_intact``, ...) override the
        base value; anything else is merged into ``params``."""
        return {}

    def set_level(self, k: float) -> None:
        """Curriculum hook. The base class stores ``k`` as ``self._level`` and, since
        ``dr_level(level)`` reads it by default, also drives how much ``randomize_groups`` are
        randomized. A task with its own notion of ``level`` (e.g. PressFromHeld's 0..5
        misalignment stages) overrides this and decides for itself how much of that maps onto
        ``self._level``."""
        self._level = float(k)

    def dr_level(self) -> float:
        """Domain-randomization fraction in [0, 1] used this reset. Default: ``self._level``
        clipped to [0, 1], so level 0 (a fresh env, or a task that never calls ``set_level``)
        reproduces the nominal, un-randomized dynamics/command path exactly."""
        return float(np.clip(self._level, 0.0, 1.0))

    # ===================================================================== scene lifecycle ===

    def _build_scene(self, seed: int | None) -> None:
        rng0 = np.random.default_rng(seed)
        self.scene: Scene = build_scene(self.scene_config(rng0))
        self.kin: ArmKinematics = self.scene.kin
        gripper_act = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self._grip_range = self.scene.model.actuator_ctrlrange[gripper_act].copy()

    # ============================================================================== gym API ===

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        self._tick = 0
        self._ep_return = 0.0
        self._peak_load = 0.0
        self._ik_failures = 0
        self.ik_stats = IKStats()

        sample = randomize.sample(rng, level=self.dr_level(), groups=self.randomize_groups)
        # `sample` only has the groups this env actually randomizes; `dr_params` always reports
        # all 17 (nominal where a group was left out), so the critic's observation stays fixed
        # size regardless of `randomize_groups`.
        full = {**{k: randomize.NOMINAL[k] for k in DR_PARAM_KEYS}, **sample}
        self._dr_sample = sample
        self._dr_params_vec = np.array([full[k] for k in DR_PARAM_KEYS], dtype=np.float32)
        randomize.apply(self.scene, sample)
        arm_q = self.reset_arm_q(rng)
        arm_q = self.scene.home_q if arm_q is None else np.asarray(arm_q, dtype=float)
        reset_scene(self.scene, rng, arm_q=arm_q)

        servo_cfg = randomize.servo_config(sample)
        bus_cfg = randomize.bus_config(sample)
        self._servo_bus = ServoBus(
            self.scene.model, self.scene.data, cfg=servo_cfg,
            rng=np.random.default_rng(int(rng.integers(2**31))),
        )
        self._bus_emulator = BusEmulator(cfg=bus_cfg, rng=np.random.default_rng(int(rng.integers(2**31))))

        home_pos, home_rot = self.kin.fk(arm_q)
        home_yaw, home_pitch, _ = euler_zyx(home_rot)
        self._home_primitive = PrimitiveTarget(
            pos=home_pos, tilt=float(home_pitch), spin=home_yaw,
            grip=gripper_rest_angle(self.scene), c_vert=1.0, c_lat=1.0,
        )
        self._q_goal = arm_q.copy()
        self._res_pos = np.zeros(3)
        self._res_tilt = 0.0
        self._res_spin = 0.0
        self._grip_target = float(self._home_primitive.grip)
        self._c_vert = 1.0
        self._c_lat = 1.0
        self._cmd_echo = dict(
            pos=home_pos.copy(), tilt=0.0, spin=home_yaw, grip=self._grip_target,
            cvert=1.0, clat=1.0,
        )

        self._last_reading = self._servo_bus.read()
        self._last_stale = np.zeros(6, dtype=bool)
        self._last_write_dropped = False
        self._last_command_latency = 0.0
        self._last_read_skew = 0.0
        self._last_ik_failed = False

        proprio = self._proprio_now()
        # One flat buffer per entry, oldest tick first, rather than a list of arrays glued back
        # together every tick: `_assemble_obs` can then hand the buffer straight to `Layout.pack`.
        self._history = {
            name: np.tile(np.asarray(v, dtype=float), HISTORY_TICKS) for name, v in proprio.items()
        }
        self._torque = np.ones(len(ARM_JOINTS))

        self.task_reset(rng)

        obs = self._assemble_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM), -1.0, 1.0)
        dx_dy_dz, dtilt, dspin, dgrip, cv_raw, cl_raw = (
            action[0:3], float(action[3]), float(action[4]), float(action[5]),
            float(action[6]), float(action[7]),
        )

        prim = self.primitive_target(self._tick)
        if prim is None:
            prim = self._home_primitive

        if "pos" not in self.frozen:
            self._res_pos = np.clip(self._res_pos + dx_dy_dz * POS_SCALE, -POS_CLIP, POS_CLIP)
        if "tilt" not in self.frozen:
            self._res_tilt = float(np.clip(self._res_tilt + dtilt * TILT_SCALE, -ANG_CLIP, ANG_CLIP))
        if "spin" not in self.frozen:
            self._res_spin = float(np.clip(self._res_spin + dspin * SPIN_SCALE, -ANG_CLIP, ANG_CLIP))
        if "grip" not in self.frozen:
            self._grip_target = float(
                np.clip(self._grip_target + dgrip * GRIP_SCALE, self._grip_range[0], self._grip_range[1])
            )
        else:
            self._grip_target = float(prim.grip)
        if "compliance" not in self.frozen:
            self._c_vert = TORQUE_LO + 0.5 * (cv_raw + 1.0) * (TORQUE_HI - TORQUE_LO)
            self._c_lat = TORQUE_LO + 0.5 * (cl_raw + 1.0) * (TORQUE_HI - TORQUE_LO)
        else:
            self._c_vert, self._c_lat = prim.c_vert, prim.c_lat

        target_pos = prim.pos + self._res_pos
        target_tilt = prim.tilt + self._res_tilt
        target_spin = prim.spin + self._res_spin

        q5, ik_ok = _pose_ik(
            self.kin, target_pos, target_tilt, target_spin, self._q_goal, stats=self.ik_stats
        )
        if ik_ok:
            self._q_goal = q5
        else:
            self._ik_failures += 1
        goal6 = np.concatenate([self._gravity_feedforward(self._q_goal), [self._grip_target]])

        torque = self._torque
        torque[_VERT_INDEX] = self._c_vert
        torque[_LAT_INDEX] = self._c_lat

        t_tick = self._tick * CONTROL_DT
        timing = self._bus_emulator.tick(t_tick)
        self._servo_bus.cfg.action_delay_s = float(timing.command_latency_s)
        self._servo_bus.cfg.action_delay_jitter_s = 0.0
        if not timing.write_lost:
            self._servo_bus.write(goal6, torque_limit=torque)

        snap_every = max(int(self.scene.snap.params.snap_every), 1)
        for k in range(PHYSICS_PER_TICK):
            self._servo_bus.substep()
            if k % snap_every == 0:
                self.scene.snap.update(self.scene.model, self.scene.data)
            mujoco.mj_step(self.scene.model, self.scene.data)

        n_servo = max(self._servo_bus.n - 1, 1)
        self._servo_bus.cfg.read_skew_s = float(timing.read_skew_s) / n_servo
        new_reading = self._servo_bus.read()
        stale = np.asarray(timing.reply_lost, dtype=bool)
        if np.any(stale):
            new_reading = ServoReadings(
                position=np.where(stale, self._last_reading.position, new_reading.position),
                speed=np.where(stale, self._last_reading.speed, new_reading.speed),
                load=np.where(stale, self._last_reading.load, new_reading.load),
                current=np.where(stale, self._last_reading.current, new_reading.current),
                tripped=new_reading.tripped,
            )
        self._last_reading = new_reading
        self._last_stale = stale
        self._last_write_dropped = bool(timing.write_lost)
        self._last_command_latency = float(timing.command_latency_s)
        self._last_read_skew = float(timing.read_skew_s)
        self._last_ik_failed = not ik_ok
        self._peak_load = max(self._peak_load, float(np.max(np.abs(new_reading.load))))

        proprio = self._proprio_now()
        for name, value in proprio.items():
            buf = self._history[name]
            width = value.shape[0]
            buf[:-width] = buf[width:]
            buf[-width:] = value
        self._cmd_echo = dict(
            pos=target_pos.copy(), tilt=target_tilt, spin=target_spin,
            grip=self._grip_target, cvert=self._c_vert, clat=self._c_lat,
        )

        # `success()` may update per-tick task bookkeeping (e.g. a within-tolerance streak);
        # `reward()` and `terminated()` are evaluated after it so they can see the result.
        success = bool(self.success())
        reward = float(self.reward())
        self._ep_return += reward
        task_terminated = bool(self.terminated())
        terminated = success or task_terminated
        self._tick += 1
        horizon_hit = self._tick >= self.horizon
        truncated = horizon_hit and not terminated

        info: dict[str, Any] = {}
        if terminated or truncated:
            info.update(self._episode_info(success=success, terminated=terminated, truncated=truncated))

        obs = self._assemble_obs()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return None

    # ================================================================================ misc ===

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """True (simulator) TCP position and rotation matrix, for reward/success/task code."""
        return self.scene.tcp_pose()

    def read_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP position/rotation from forward kinematics of the last *read* joint positions --
        what the actor's own ``tcp_pos``/``tcp_rot6`` entries are computed from. Task code that
        wants "what the policy can see" rather than simulator truth should use this."""
        return self.kin.fk(self._last_reading.position[:5])

    # -------------------------------------------------------------------------- internals ---

    def _gravity_feedforward(self, q5: np.ndarray) -> np.ndarray:
        """The joint command that actually puts the arm at ``q5``, not ``q5`` itself.

        ``ServoBus`` is a plain position servo (``kv=0``, no gravity term) commanding
        ``data.ctrl`` directly: this project's own static-sag measurement
        found the arm settles at ``commanded - gravity_torque(actual) / kp``, tens of
        millimetres off at the TCP for this arm's mass at these gains. Any task whose success
        criterion is a few millimetres (Reach: 3 mm) needs that cancelled, the way a real
        controller always carries a gravity feedforward term. Two fixed-point passes of
        ``commanded = q5 + gravity_torque(commanded) / kp`` bring the *kinematic* IK target and
        the arm's true static rest pose back to within about a millimetre of each other -- the
        residual is well inside what the per-tick position-error feedback (the residual
        accumulating on ``tcp_pos`` error, i.e. integral action) closes the rest of the way.
        """
        kp5 = self._servo_bus.kp[:5]
        comp = q5.copy()
        for _ in range(2):
            comp = q5 + self.kin.gravity_torque(comp) / kp5
        return comp

    def _proprio_now(self) -> dict[str, np.ndarray]:
        r = self._last_reading
        tcp_pos, tcp_rot = self.kin.fk(r.position[:5])
        rot6 = np.concatenate([tcp_rot[:, 0], tcp_rot[:, 1]])
        return {
            "joint_pos": r.position, "joint_vel": r.speed, "joint_load": r.load,
            "tcp_pos": tcp_pos, "tcp_rot6": rot6,
        }

    def _assemble_obs(self) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = dict(self._history)
        values["cmd_pose"] = np.array(
            [*self._cmd_echo["pos"], self._cmd_echo["tilt"], self._cmd_echo["spin"]]
        )
        values["cmd_grip"] = np.array([self._cmd_echo["grip"]])
        values["cmd_compliance"] = np.array([self._cmd_echo["cvert"], self._cmd_echo["clat"]])
        values.update(self.task_obs_values())
        actor = self._actor_layout.pack(values)

        values["read_stale"] = self._last_stale.astype(np.float32)
        values["write_dropped"] = np.array([float(self._last_write_dropped)])
        values["bus_timing"] = np.array([self._last_command_latency, self._last_read_skew])
        values["ik_failed"] = np.array([float(self._last_ik_failed)])
        values["dr_params"] = self._dr_params_vec
        critic = self._critic_layout.pack(values)
        return {"actor": actor, "critic": critic}

    def _episode_info(self, *, success: bool, terminated: bool, truncated: bool) -> dict[str, Any]:
        failure = None
        if not success:
            failure = FailureClass.TIMEOUT if truncated else self.failure_class()
        extra = dict(self.episode_result_extra())
        params = {
            "level": self._level, "ik_failures": self._ik_failures,
            "ik_fallbacks": self.ik_stats.fallbacks,
            "dr": dict(self._dr_sample), **extra.pop("params", {}),
        }
        result_kwargs = dict(
            success=success, steps=self._tick, peak_load=self._peak_load,
            failure=failure, ret=self._ep_return, params=params,
        )
        # The damage monitor rides on the scene, not on the task, so it is read here rather than
        # in every environment's own `episode_result_extra`. A task that wants to override
        # either field still can: `extra` is applied after this.
        monitor = getattr(self.scene, "monitor", None)
        if monitor is not None:
            result_kwargs["damage_state"] = monitor.state
            result_kwargs["peak_wall_stress_Pa"] = monitor.peak_wall_stress_Pa
        result_kwargs.update(extra)
        result = EpisodeResult(**result_kwargs)
        return {
            "success": result.success,
            "failure": result.failure,
            "seat_depth_error": result.seat_depth_error,
            "peak_load": result.peak_load,
            "structure_intact": result.structure_intact,
            "damage_state": result.damage_state,
            "peak_wall_stress_Pa": result.peak_wall_stress_Pa,
            "params": result.params,
            "episode_result": result,
        }


def measure_step_rate(env: ArmEnv, steps: int = 2000, warmup: int = 200) -> float:
    """Single-process ``env.step()`` calls per second, random actions, warmed up first."""
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(warmup):
        obs, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, ACTION_DIM))
        if terminated or truncated:
            env.reset()
    import time

    t0 = time.perf_counter()
    n = 0
    while n < steps:
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, ACTION_DIM))
        n += 1
        if terminated or truncated:
            env.reset()
    dt = time.perf_counter() - t0
    return n / dt
