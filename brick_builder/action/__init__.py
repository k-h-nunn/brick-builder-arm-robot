"""The closed-loop action side: act on (intended - actual) slot states.

Three pieces, deliberately small:

- `adapter.WorldAdapter` -- the *only* thing here that touches the running simulator, and it
  offers exactly four things: read the servo bus, write joint targets, grab the wrist camera
  frame as a `SceneState`, step time. No body poses, no contact state, no snap state, no
  `MjData`. Everything in `servo.py`/`place.py` talks to the world through it.
- `twin.DigitalTwin` -- a *separate* scene built from the goal and nominal parameters. Posed at
  a step's goal pose it renders the wrist view the controller *expects to see*, and the same
  `IdealSceneModel` turns that into the **intended** `SceneState`. Feed-forward, outside the
  loop: built once per step, never touched by the running world.
- `place.run_episode` -- the step sequence. At every vision tick (10 Hz) it forms the feature
  error (intended - actual) on slot centres and commands a tool change through the feature
  Jacobian the twin measured.

Nothing here imports `readout.py`, `producers.py` or any decode/label path: a controller reads
slot numbers, and that is all.
"""

from .adapter import WRIST_FULL, WorldAdapter
from .servo import FeatureSet, Reference, VisualServo, reference_features
from .twin import DigitalTwin, TwinStep
from .sequence import Command, IdealSequence, ReferenceFrame
from .place import (
    STEP_TITLES, Geometry, PlaceConfig, Script, StepReport, TickRecord, TrackingEpisode,
    record_ideal, reference_bodies,
)
from .worlds import build_pick_world, build_world, displacement, nominal_cfg

__all__ = [
    "WRIST_FULL", "WorldAdapter",
    "FeatureSet", "Reference", "VisualServo", "reference_features",
    "DigitalTwin", "TwinStep",
    "Command", "IdealSequence", "ReferenceFrame",
    "STEP_TITLES", "Geometry", "PlaceConfig", "Script", "StepReport", "TickRecord",
    "TrackingEpisode", "record_ideal", "reference_bodies",
    "build_pick_world", "build_world", "displacement", "nominal_cfg",
]
