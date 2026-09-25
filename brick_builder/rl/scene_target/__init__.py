"""Scene-target reinforcement-learning environment: `BrickBuilder/SceneTarget-v0`.

- `codec` -- slot frames, appearance matching, scene-state distance [mm-eq], calibration.
- `targets` -- target sequences as arrays, `.npz` cache, viewing-pose generator.
- `env` -- `SceneTargetEnv`, one vision tick per step.
- `executor`, `rules` -- action to servo-bus writes; progress index, reward, ground-truth score.
- `world` -- the empty-hand world, the kinematic twin, scene-model sources.
- `wrappers` -- `PredictionScoring`, next-frame prediction error per tick.
- `bench` -- environment throughput [ticks/s].

Importing this package registers the environment id; the environment module itself (MuJoCo,
rendering) is imported only by `gym.make` or by attribute access.
"""

from __future__ import annotations

import gymnasium as gym

from .codec import Calibration, DistanceResult, SceneDistance, SlotFrame
from .config import LEVELS, RewardWeights, SceneTargetConfig
from .targets import TargetSequence

ENV_ID = "BrickBuilder/SceneTarget-v0"

__all__ = [
    "Calibration", "DistanceResult", "ENV_ID", "LEVELS", "RewardWeights", "SceneDistance",
    "SceneTargetConfig", "SceneTargetEnv", "SlotFrame", "TargetSequence",
]

if ENV_ID not in gym.envs.registry:
    gym.register(id=ENV_ID, entry_point="brick_builder.rl.scene_target.env:SceneTargetEnv")


def __getattr__(name: str):
    if name == "SceneTargetEnv":
        from .env import SceneTargetEnv

        return SceneTargetEnv
    raise AttributeError(name)
