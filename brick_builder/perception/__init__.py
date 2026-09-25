"""Perception pipeline: scene-model sources, object-token producers, and the readers on top.

The controller-facing interface is `scene_model.SceneState`: a small set of per-slot states
(`PLACEHOLDER_SLOT_SCHEMA`), produced by `ideal_scene_model.IdealSceneModel` in simulation
(ground truth), `conventional_scene_model.ColourSegmentationSceneModel` (classical, RGB-only) or
`conventional_scene_model.LearnedSceneModel` (the trained convolutional model) from a rendered or
camera frame taken inside a `views.ViewingPose` window. `sources.py` is the registry those three
publish themselves through, and the seam a source from another package registers itself at.
`readout.py` (a per-pixel, decode-based reader) and `slot_ops.py` (slot-space comparison, editing
and logging) sit on top; see each module's own docstring, and `docs/guide/06-perception.md`.
"""

from .scene_model import (
    LOGIT_MAGNITUDE,
    PLACEHOLDER_SLOT_SCHEMA,
    SceneModelSource,
    SceneState,
    SlotSchema,
    TemporalSlotAssigner,
    opaque_schema,
    rank_scale,
    to_window_xy,
)
from .ideal_scene_model import IdealSceneModel
from .conventional_scene_model import (
    ColourSegmentationSceneModel,
    ConventionalGpuSceneModel,
    LearnedSceneModel,
)
from . import sources
from .views import (
    ViewingPose,
    PLATE_VIEW,
    PLATE_VIEW_DENSE,
    TRAY_VIEW,
    VIEWS,
    attention_crop,
    attention_view,
    goto_look,
    render_view,
    roi_box_corners,
    solve_look,
    window_from_slots,
)
from .slot_ops import LogRecord, read_log, select_slot, set_slot_blob, set_slot_offset, slot_delta, slot_distance, write_log

__all__ = [
    "LOGIT_MAGNITUDE",
    "PLACEHOLDER_SLOT_SCHEMA",
    "SceneModelSource",
    "SceneState",
    "SlotSchema",
    "TemporalSlotAssigner",
    "opaque_schema",
    "rank_scale",
    "to_window_xy",
    "IdealSceneModel",
    "ColourSegmentationSceneModel",
    "ConventionalGpuSceneModel",
    "LearnedSceneModel",
    "sources",
    "ViewingPose",
    "PLATE_VIEW",
    "PLATE_VIEW_DENSE",
    "TRAY_VIEW",
    "VIEWS",
    "attention_crop",
    "attention_view",
    "goto_look",
    "render_view",
    "roi_box_corners",
    "solve_look",
    "window_from_slots",
    "LogRecord",
    "read_log",
    "select_slot",
    "set_slot_blob",
    "set_slot_offset",
    "slot_delta",
    "slot_distance",
    "write_log",
]
