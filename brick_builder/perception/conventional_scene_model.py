"""Scene-model producers other than the ideal ground-truth one.

Behind the same `scene_model.SceneModelSource` interface (`observe(rgb, *, window, ...) ->
SceneState`, `PLACEHOLDER_SLOT_SCHEMA` rows):

- `ColourSegmentationSceneModel`: a real, working, RGB-only producer for a machine with no learned
  model and no simulator ground truth at all -- classical HSV/RGB colour thresholding plus
  connected components (the same colour logic `producers.BlobProducer` already uses), with frame-
  to-frame identity from `scene_model.TemporalSlotAssigner`'s Hungarian matching (it has no
  simulator id to rely on, unlike `ideal_scene_model.IdealSceneModel`). This is what the real
  robot can run *today*, on a laptop CPU, without downloading or training anything.
- `LearnedSceneModel` (alias `ConventionalGpuSceneModel`): the trained convolutional scene
  model, this project's laptop-scale fallback perception -- more accurate than colour
  thresholding and far more expensive, which is the trade it exists to make. The network, the
  simulator-only training-set generator, the training loop and the evaluation live in
  `brick_builder.perception.cnn`; the weights ship in `brick_builder/perception/weights/`. This
  class forwards to `perception.cnn.source.CnnSceneModel` so that every producer can still be
  found in one file. See `docs/guide/06b-conventional-vision-model.md`.

A scene-model source that lives outside this package registers itself with
`brick_builder.perception.sources` and is then reachable by name; nothing here needs to know
about it. A source whose row layout is not a documented set of named fields publishes
`scene_model.opaque_schema(W)` instead of `PLACEHOLDER_SLOT_SCHEMA`, and consumers ask the schema
what is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState, TemporalSlotAssigner, to_window_xy, fourth_moment

DEFAULT_PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.77, 0.09, 0.09),
    "blue": (0.00, 0.32, 0.62),
    "yellow": (0.96, 0.78, 0.13),
    "green": (0.00, 0.46, 0.22),
    "orange": (0.90, 0.42, 0.09),
    "white": (0.95, 0.95, 0.95),
}
"""`brick_builder.bricks.spec.PALETTE`'s values, copied rather than imported so this module (like
`producers.py`) can build and test its colour logic without depending on the bricks package being
present -- see `producers.py`'s own module docstring for the reasoning."""


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.float64)
    if frame.max(initial=0.0) > 1.0 + 1e-6:
        frame = frame / 255.0
    return frame[..., :3]


def _blob_row_from_mask(mask: np.ndarray, frame01: np.ndarray, res: int, age: float) -> np.ndarray:
    """Shared with `ideal_scene_model._blob_row`'s own layout (kept as a separate, small copy
    here rather than a cross-import: this module's colour segmentation and the ideal ground-truth
    source are two independent producers of the same *schema*, and neither should depend on the
    other's implementation)."""
    row = np.zeros(PLACEHOLDER_SLOT_SCHEMA.W, dtype=np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return row
    x, y = to_window_xy(xs, ys, res)
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    sxx, sxy, syy = float(np.mean(dx * dx)), float(np.mean(dx * dy)), float(np.mean(dy * dy))
    rgb01 = frame01[ys, xs]
    colour = 2.0 * rgb01 - 1.0
    mean_colour = colour.mean(axis=0)
    colour_spread = float(np.mean((colour - mean_colour) ** 2))
    on_border = (xs == 0) | (xs == res - 1) | (ys == 0) | (ys == res - 1)
    border_frac = float(np.count_nonzero(on_border)) / len(xs)

    fields = PLACEHOLDER_SLOT_SCHEMA.fields
    row[fields["mass"]] = len(xs) / (res * res)
    row[fields["centre"]] = [mx, my]
    row[fields["moments"]] = [sxx, sxy, syy]
    row[fields["colour"]] = mean_colour
    row[fields["colour_spread"]] = colour_spread
    row[fields["border"]] = border_frac
    row[fields["age"]] = age
    row[fields["orient4"]] = fourth_moment(dx, dy)
    return row


@dataclass
class ColourSegmentationSceneModel:
    """A classical, RGB-only scene-model producer: HSV/RGB colour thresholding against a small
    palette (reusing `producers.hsv_colour_distance`'s embedding), connected components per
    colour, `PLACEHOLDER_SLOT_SCHEMA` statistics per component, and frame-to-frame identity from
    `scene_model.TemporalSlotAssigner`'s Hungarian matching (no simulator id available). No
    learned model, no simulator ground truth -- runs on a CPU, today, on the real robot.
    """

    K: int = 8
    vanish_after: int = 5
    palette: dict[str, tuple[float, float, float]] | None = None
    tol: float = 0.40
    min_area_px: int = 24
    chroma_gain: float = 2.0

    assigner: TemporalSlotAssigner = field(init=False, repr=False)
    last_object_count: int = field(default=0, init=False, repr=False)
    last_kept_count: int = field(default=0, init=False, repr=False)
    last_overflow_count: int = field(default=0, init=False, repr=False)
    _frame_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.palette is None:
            self.palette = DEFAULT_PALETTE
        self.assigner = TemporalSlotAssigner(K=self.K, vanish_after=self.vanish_after)

    def reset(self) -> None:
        self.assigner.reset()

    def observe(
        self, rgb: np.ndarray, *, window, instance=None, depth=None, present_mask=None, t: float | None = None,
    ) -> SceneState:
        del instance  # colour segmentation never sees ground truth
        frame01 = _as_float_rgb(rgb)
        res = frame01.shape[0]
        if frame01.shape[1] != res:
            raise ValueError(f"rgb must be square; got shape {frame01.shape[:2]}")

        valid = np.ones((res, res), dtype=bool) if present_mask is None else np.asarray(present_mask, dtype=bool)

        hsv = cv2.cvtColor(frame01.astype(np.float32), cv2.COLOR_RGB2HSV)
        theta = np.deg2rad(hsv[..., 0])
        s, v = hsv[..., 1], hsv[..., 2]
        cone = np.stack([self.chroma_gain * s * np.cos(theta), self.chroma_gain * s * np.sin(theta), v], axis=-1)

        candidates: list[tuple[np.ndarray, float]] = []  # (mask, mass)
        for name, ref in self.palette.items():
            r, g, b = ref
            rh, rs, rv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0, 0]
            ref_cone = np.array(
                [self.chroma_gain * rs * np.cos(np.deg2rad(rh)), self.chroma_gain * rs * np.sin(np.deg2rad(rh)), rv]
            )
            dist = np.linalg.norm(cone - ref_cone, axis=-1)
            mask = (dist <= self.tol) & valid
            if not np.any(mask):
                continue
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for label in range(1, n_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < self.min_area_px:
                    continue
                candidates.append(((labels == label), float(area)))

        self.last_object_count = len(candidates)
        masses = np.array([m for _, m in candidates], dtype=np.float64)
        rows = np.zeros((len(candidates), PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
        for i, (mask, _) in enumerate(candidates):
            rows[i] = _blob_row_from_mask(mask, frame01, res, age=0.0)

        slot_of = self.assigner.step(masses, rows, external_ids=None) if len(candidates) else np.zeros(0, dtype=np.int64)
        kept_mask = slot_of >= 0
        self.last_kept_count = int(kept_mask.sum())
        self.last_overflow_count = int((~kept_mask).sum())

        background_slot = self.assigner.background_slot
        claimed = np.zeros((res, res), dtype=bool)
        for i, (mask, _) in enumerate(candidates):
            if kept_mask[i]:
                claimed |= mask
        background_pixels = valid & ~claimed

        slots = np.zeros((self.K, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
        slot_ids = np.full(self.K, -1, dtype=np.int64)
        for i, (mask, _) in enumerate(candidates):
            if not kept_mask[i]:
                continue
            k = int(slot_of[i])
            row = rows[i].copy()
            row[PLACEHOLDER_SLOT_SCHEMA.fields["age"]] = float(np.tanh(self.assigner.slot_age(k) / 10.0))
            slots[k] = row
            slot_ids[k] = k  # colour segmentation has no simulator id; the slot index IS the identity token

        bg_age = float(np.tanh(self.assigner.slot_age(background_slot) / 10.0))
        slots[background_slot] = _blob_row_from_mask(background_pixels, frame01, res, age=bg_age)

        # No metric depth available to a colour-only producer -- every present slot ties at
        # "unknown" ordinal depth (present, but 0.5, since there is nothing to rank it against).
        present = np.zeros(self.K, dtype=bool)
        present[background_slot] = True
        for i in range(len(candidates)):
            if kept_mask[i]:
                present[int(slot_of[i])] = True
        slot_depth = np.where(present, 0.5, 0.5).astype(np.float32)

        frame_id = self._frame_counter
        self._frame_counter += 1
        return SceneState(
            slots=slots, depth=slot_depth, schema=PLACEHOLDER_SLOT_SCHEMA,
            t=float(t) if t is not None else float(frame_id), source="colour_segmentation",
            window=window, slot_ids=slot_ids, frame_id=frame_id, _decode_fn=None,
        )


@dataclass
class LearnedSceneModel:
    """The trained convolutional scene model: this project's laptop-scale fallback perception.

    A compact encoder-decoder convolutional network reads one 256x256 wrist-camera window and
    predicts, per pixel, a class (background/table, baseplate, arm, brick), a brick-centre heat
    map, an offset from each brick pixel to its own brick's centre, and relative depth. Peaks of
    the heat map become instance centres and every brick pixel joins the nearest one, which is
    what separates two touching bricks of the same colour -- the case colour thresholding cannot
    handle at all. The masks then go through the same `ideal_scene_model._blob_row` statistics and
    the same `scene_model.TemporalSlotAssigner` the ideal source uses, so the output is an
    ordinary `PLACEHOLDER_SLOT_SCHEMA` `SceneState` like every other producer's.

    Everything real lives in `brick_builder.perception.cnn` (network, training data, training
    loop, evaluation, demo); the trained weights ship in `brick_builder/perception/weights/`. This
    class is a thin forwarder so the whole set of scene-model producers can be read in one file.
    `docs/guide/06b-conventional-vision-model.md` has the accuracy table, the latency, the failure
    modes and the retraining and fine-tuning procedures.

    `device` is `"cpu"` by default and `"cuda"` where a GPU is free. `ConventionalGpuSceneModel`
    remains as an alias; the model is not GPU-only, which is why the clearer name is preferred.
    """

    checkpoint: str | None = None
    """Path to a checkpoint; `None` loads the one that ships with the package."""

    device: str = "cpu"
    K: int = 8
    vanish_after: int = 5
    model_name: str = "conv_net"

    impl: object = field(default=None, repr=False)
    """The real producer, `perception.cnn.source.CnnSceneModel`. Built on construction, so a
    missing checkpoint fails immediately with a message saying how to train one."""

    def __post_init__(self) -> None:
        if self.impl is None:
            from .cnn.source import CnnSceneModel

            self.impl = CnnSceneModel(
                checkpoint=self.checkpoint, device=self.device, K=self.K,
                vanish_after=self.vanish_after,
            )

    @property
    def assigner(self) -> TemporalSlotAssigner:
        return self.impl.assigner

    def reset(self) -> None:
        """Clears slot identity -- a new episode must not remember the last one's objects."""
        self.impl.reset()

    def observe(
        self, rgb: np.ndarray, *, window, instance=None, depth=None, present_mask=None, t=None,
    ) -> SceneState:
        """One vision tick. `instance`/`depth` are accepted and ignored: this producer sees `rgb`
        and nothing else, which is the point of it."""
        return self.impl.observe(rgb, window=window, instance=instance, depth=depth,
                                 present_mask=present_mask, t=t)


ConventionalGpuSceneModel = LearnedSceneModel
"""The name this class carried while it was a stub for "a conventional model on a laptop GPU".
Kept so existing imports keep working; `LearnedSceneModel` is the name to use."""
