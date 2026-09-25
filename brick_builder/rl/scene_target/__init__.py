"""Scene-target reinforcement-learning environment.

- `codec` -- slot frames, appearance matching, scene-state distance [mm-eq], calibration.
- `config` -- stages, action bounds, reward weights.
- `targets` -- target sequences as arrays, `.npz` cache, viewing-pose generator.
- `world` -- the empty-hand world, the kinematic twin, scene-model sources.
"""

from __future__ import annotations

from .codec import Calibration, DistanceResult, SceneDistance, SlotFrame
from .config import LEVELS, RewardWeights, SceneTargetConfig
from .targets import TargetSequence

__all__ = [
    "Calibration", "DistanceResult", "LEVELS", "RewardWeights", "SceneDistance",
    "SceneTargetConfig", "SlotFrame", "TargetSequence",
]
