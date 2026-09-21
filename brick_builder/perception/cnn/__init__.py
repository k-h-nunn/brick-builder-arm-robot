"""A conventional convolutional scene-model producer, trained on simulator frames.

This package is the laptop-scale alternative to `perception.conventional_scene_model.
ColourSegmentationSceneModel`: an ordinary encoder-decoder convolutional network that reads one
256x256 wrist-camera window and publishes the same `perception.scene_model.SceneState` every other
producer publishes (`PLACEHOLDER_SLOT_SCHEMA` rows plus an ordinal per-slot depth). It is far more
expensive than colour thresholding and cheap enough to run many times a second on a laptop, which
is the point: it is the fallback perception the project can run when the microcontroller path is
unavailable, and a second opinion when it is. Another source can plug into the same interface
without anything downstream changing.

Modules:

- `labels.py` -- the four pixel classes and the instance-id convention shared by everything here.
- `data.py` -- builds the training set from the simulator alone: randomized scenes, the same
  256x256 window `perception.views` gives the ideal source, per-pixel labels from the renderer.
- `model.py` -- the network: a compact U-Net-like encoder-decoder with four heads (per-pixel
  class, brick-centre heat map, offset-to-centre, relative depth).
- `postprocess.py` -- heat-map peaks and offsets to instance masks, then the ideal source's own
  mask-to-blob-statistics code to slot rows.
- `train.py` -- the training loop (CPU by default, time-budgeted, resumable).
- `evaluate.py` -- accuracy against the ideal source, field by field, plus speed.
- `source.py` -- `CnnSceneModel`, the `SceneModelSource` implementation plus the viewer API.
- `demo.py` -- the side-by-side picture for the documentation.

Nothing here imports torch until `model`, `train`, `source` or `evaluate` is imported, so
`data`/`labels` stay usable on a machine that only generates frames.
"""

from .labels import (
    ARM_CLASS,
    ARM_INSTANCE,
    BACKGROUND_CLASS,
    BASEPLATE_CLASS,
    BRICK_CLASS,
    CLASS_NAMES,
    FIRST_BRICK_INSTANCE,
    N_CLASSES,
    PLATE_INSTANCE,
    class_map_from_instance,
)

__all__ = [
    "ARM_CLASS",
    "ARM_INSTANCE",
    "BACKGROUND_CLASS",
    "BASEPLATE_CLASS",
    "BRICK_CLASS",
    "CLASS_NAMES",
    "FIRST_BRICK_INSTANCE",
    "N_CLASSES",
    "PLATE_INSTANCE",
    "class_map_from_instance",
]
