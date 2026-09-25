"""Scene-target reinforcement-learning environment.

- `codec` -- slot frames, appearance matching, scene-state distance [mm-eq], calibration.
"""

from __future__ import annotations

from .codec import Calibration, DistanceResult, SceneDistance, SlotFrame

__all__ = ["Calibration", "DistanceResult", "SceneDistance", "SlotFrame"]
