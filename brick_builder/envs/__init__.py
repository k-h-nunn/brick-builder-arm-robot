"""Gymnasium environments for training and evaluating brick-building skills.

Importing this package registers every BrickBuilder task with gymnasium, so ``gym.make("BrickBuilder/
Reach-v0")`` works after ``import brick_builder.envs`` (or after anything that imports it, such as
``brick_builder.rl``'s own env factories). Registration is guarded against being run twice: pytest's
``--import-mode=importlib`` can import this module more than once in one process, and
``gymnasium.register`` raises if asked to overwrite an existing id.
"""

from __future__ import annotations

import gymnasium as gym

from brick_builder.envs import obs_spec, wrappers
from brick_builder.envs.align_env import AlignFromTokensEnv
from brick_builder.envs.base_env import ArmEnv, PrimitiveTarget
from brick_builder.envs.press_env import PressEnv
from brick_builder.envs.reach_env import ReachEnv

__all__ = [
    "AlignFromTokensEnv", "ArmEnv", "PressEnv", "PrimitiveTarget", "ReachEnv", "obs_spec", "wrappers",
]

_TASKS = {
    "BrickBuilder/Reach-v0": "brick_builder.envs.reach_env:ReachEnv",
    "BrickBuilder/PressFromHeld-v0": "brick_builder.envs.press_env:PressEnv",
    "BrickBuilder/AlignFromTokens-v0": "brick_builder.envs.align_env:AlignFromTokensEnv",
}

for _env_id, _entry_point in _TASKS.items():
    if _env_id not in gym.envs.registry:
        gym.register(id=_env_id, entry_point=_entry_point)
