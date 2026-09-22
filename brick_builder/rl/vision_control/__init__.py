"""Reinforcement learning on the closed-loop placement task, from pixels to a tool correction.

Three pieces:

- `env.VisionControlEnv` -- a Gymnasium environment around `action.place.TrackingEpisode`. The
  action is a residual added to the analytic controller's own correction increment, so a zero
  action reproduces today's controller exactly.
- `policy.VisionControlPolicy` -- the convolutional scene model (shipped weights) followed by the
  differentiable slot head and a `G`-shaped MLP over slot-row differences.
- `train.py` -- asymmetric PPO over that policy, with an auxiliary next-state prediction loss
  through the shipped predictor `F`.

The observation carries the raw wrist frame, so a different scene-model source registered through
`brick_builder.perception.sources` can replace the policy's own front end without the environment
changing.
"""

from .env import VisionControlConfig, VisionControlEnv, make_env

__all__ = ["VisionControlConfig", "VisionControlEnv", "make_env"]
