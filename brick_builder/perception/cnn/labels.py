"""The four pixel classes this model predicts, and the instance-id convention behind them.

Two separate label channels come out of the renderer for every training frame and are predicted
back by the network:

- a **class** per pixel, one of four: background (the table, the tray, anything unlabelled), the
  baseplate, the arm (every link plus the fixed and the moving jaw), and a brick;
- an **instance** id per pixel, so that two bricks of the same colour touching each other are two
  objects rather than one. Colour alone cannot separate them, which is exactly why the network
  carries a centre heat map and an offset field as well as a class map.

The instance ids are fixed by convention rather than by whatever the scene happens to compile to:

    0                     background (table, tray, unlabelled)
    ARM_INSTANCE = 1      every arm link, the fixed jaw, the moving jaw
    PLATE_INSTANCE = 2    the baseplate
    3, 4, 5, ...          one id per brick, in the order the scene lists them

This is deliberately *not* `scene.compose.build_scene`'s own `instance_ids` map, which gives the
world body -- and therefore the table plane -- the same id as the arm, so the ideal source sees
table and arm as one object. `data.perception_instance_lut` builds the map above instead, and the
evaluation feeds the ideal source that same map, so the two producers are being asked the same
question about the same objects.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES: tuple[str, ...] = ("background", "baseplate", "arm", "brick")
"""Index i is the name of class i. The order is fixed: a trained checkpoint depends on it."""

BACKGROUND_CLASS, BASEPLATE_CLASS, ARM_CLASS, BRICK_CLASS = 0, 1, 2, 3
N_CLASSES = len(CLASS_NAMES)

ARM_INSTANCE = 1
PLATE_INSTANCE = 2
FIRST_BRICK_INSTANCE = 3
"""Brick instances are numbered from here upwards, one per brick body."""

MAX_INSTANCES = 255
"""Instance maps are stored as `uint8`, which is plenty: a frame never holds anywhere near 250
bricks, and the slot contract keeps only the 7 largest objects anyway."""


def class_map_from_instance(instance: np.ndarray) -> np.ndarray:
    """`[H, W]` instance ids (the convention above) -> `[H, W]` class ids, `uint8`."""
    instance = np.asarray(instance)
    out = np.full(instance.shape, BACKGROUND_CLASS, dtype=np.uint8)
    out[instance == ARM_INSTANCE] = ARM_CLASS
    out[instance == PLATE_INSTANCE] = BASEPLATE_CLASS
    out[instance >= FIRST_BRICK_INSTANCE] = BRICK_CLASS
    return out
