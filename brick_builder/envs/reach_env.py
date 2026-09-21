"""``BrickBuilder/Reach-v0``: move the TCP to a random reachable target from proprioception alone.

The simplest task the shared multi-rate loop supports, used to prove the loop end to end (no
scripted primitive, no bricks, no vision) before the two heavier tasks (`PressFromHeld`,
`AlignFromTokens`) build on the same base. Grip and compliance are frozen -- the policy only ever
moves ``dx, dy, dz, d_tilt, d_spin`` -- and since :class:`~brick_builder.envs.base_env.ArmEnv` clips
the position/orientation residual to a box around the (here: fixed, since there is no scripted
primitive) primitive pose, the target is drawn from *inside* that same box: that is what "random
reachable target" means for a task with no scripted motion of its own to reach past.
"""

from __future__ import annotations

import numpy as np

from brick_builder.envs import obs_spec
from brick_builder.envs.base_env import POS_CLIP, ArmEnv
from brick_builder.scene.compose import SceneConfig

SUCCESS_TOL = 0.003          # m
SUCCESS_TICKS = 10
GOAL_MARGIN = 0.8            # draw the target from this fraction of the residual box

GOAL_OFFSET = obs_spec.register(obs_spec.ObsEntry(
    "reach_goal", 3, "task", "actor",
    note="target TCP position minus the home TCP position, in metres; drawn once at reset and "
    "handed to the policy as task context, not measured",
))
TIME_FRAC = obs_spec.register(obs_spec.ObsEntry(
    "reach_time_frac", 1, "task", "actor",
    note="fraction of the episode horizon elapsed (tick / horizon)",
))
TRUE_ERROR = obs_spec.register(obs_spec.ObsEntry(
    "reach_true_error", 1, "sim", "critic",
    note="true (simulator) distance from the TCP to the goal, metres",
))


class ReachEnv(ArmEnv):
    """``BrickBuilder/Reach-v0``. Horizon 100 ticks (2 s); success within 3 mm for 10 ticks."""

    frozen = frozenset({"grip", "compliance"})

    def __init__(self, *, horizon: int = 100, seed: int | None = None) -> None:
        super().__init__(horizon=horizon, seed=seed)

    # -- scene ------------------------------------------------------------------------------

    def scene_config(self, rng: np.random.Generator) -> SceneConfig:
        # No brick at all: Reach only ever moves an empty gripper near its home pose.
        #
        # The plate is cut down to 2x2 studs as well. Reach never touches it -- the home pose
        # hovers 20 mm over it and the residual box is 12 mm -- but a build plate is 64 collision
        # cylinders, and MuJoCo pays for them on every one of the 10 physics steps a tick takes
        # (measured: 45.9 us a step with an 8x8 plate, 28.1 us with a 2x2, on an otherwise idle
        # machine). Nothing about the task moves: the plate's top surface, its centre and
        # therefore the home TCP pose and the goal box are all independent of how many studs it
        # has, and a 2x2 plate is still 64 mm square, so a gripper driven to the bottom of the
        # residual box still finds the same surface under it that it did before.
        return SceneConfig(
            held=None, placed=[], cameras=("overview_cam",), plate_studs=(2, 2),
        )

    # -- task -------------------------------------------------------------------------------

    def task_actor_entries(self) -> list[str]:
        return ["reach_goal", "reach_time_frac"]

    def task_critic_entries(self) -> list[str]:
        return ["reach_true_error"]

    def task_obs_values(self) -> dict[str, np.ndarray]:
        return {
            "reach_goal": self._goal_pos - self._home_primitive.pos,
            "reach_time_frac": np.array([self._tick / self.horizon]),
            "reach_true_error": np.array([self._true_error()]),
        }

    def task_reset(self, rng: np.random.Generator) -> None:
        # Orientation (tilt/spin) is left unfrozen but has no target here: only the TCP position
        # is scored, so drawing an orientation goal would be dead weight this task never reads.
        offset = rng.uniform(-GOAL_MARGIN * POS_CLIP, GOAL_MARGIN * POS_CLIP, 3)
        self._goal_pos = self._home_primitive.pos + offset
        self._within_streak = 0

    # -- reward / termination -----------------------------------------------------------------

    def _true_error(self) -> float:
        pos, _ = self.tcp_pose()
        return float(np.linalg.norm(pos - self._goal_pos))

    def success(self) -> bool:
        dist = self._true_error()
        self._within_streak = self._within_streak + 1 if dist < SUCCESS_TOL else 0
        return self._within_streak >= SUCCESS_TICKS

    def reward(self) -> float:
        dist = self._true_error()
        r = -20.0 * dist
        r -= 0.01 * float(np.sum(np.square(self._res_pos / POS_CLIP)))
        if self._within_streak >= SUCCESS_TICKS:
            r += 10.0
        return r

    def terminated(self) -> bool:
        return False  # only success ends this episode early; otherwise it runs to the horizon

    def episode_result_extra(self) -> dict:
        return {"params": {"final_error_m": self._true_error()}}
