"""Two interchangeable sources of `ObjectToken`s, behind one `TokenProducer` protocol.

Control code (skills, the RL policy, the FSM) never talks to a vision model, or to the
simulator's ground truth, directly -- it consumes tokens, and does not know or care which
producer made them:

- `GroundTruthProducer`: for bulk RL training. Reads the simulator's own object state and adds a
  parametric noise model (bias, jitter, dropout, same-colour merging) fitted to a real vision
  model's measured error statistics, so a policy trained against it sees something that behaves
  like that error profile without paying the model's compute on every training step.
- `BlobProducer`: classical HSV/RGB colour thresholding + connected components. No learned model.

A further producer can be added out of tree: implement `TokenProducer` and hand the instance to
whatever selects one. For scene-model sources, `brick_builder.perception.sources` is the
registry that does this by name.

All tokens are in pixel units (see `tokens.py`); metric back-projection is `backproject.py`.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np

from brick_builder.perception.tokens import ObjectToken, grid_queries, slot_statistics

# A small palette shared by the ground-truth surrogate (to label its synthetic objects) and the
# blob fallback (to name what it finds). Deliberately not imported from `brick_builder.bricks.spec`
# (a different package's module): keeping this self-contained is what lets
# `BlobProducer` and `GroundTruthProducer` run with nothing else in the repo built yet.
PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.80, 0.10, 0.10),
    "blue": (0.10, 0.20, 0.80),
    "yellow": (0.90, 0.85, 0.10),
    "green": (0.10, 0.60, 0.15),
    "orange": (0.90, 0.45, 0.05),
    "white": (0.92, 0.92, 0.90),
}

# The default build-plate colour (perception-align-002): a literal copy of
# `brick_builder.bricks.spec.PALETTE["green"]`/`add_baseplate`'s default `colour="green"`, not an
# import of it -- this module's own docstring above explains why it stays decoupled from
# `brick_builder.bricks`. `tests/perception/test_producers.py` cross-checks the two stay in sync, so
# a future change to either palette that lets them drift apart again fails loudly instead of
# silently reintroducing this bug.
DEFAULT_BASEPLATE_RGB: tuple[float, float, float] = (0.00, 0.46, 0.22)

# How much to boost the hue/saturation plane relative to value (V) when telling two colours
# apart -- see `_hsv_cone`. Chosen (with `BlobProducer.tol`'s default) by measuring, across 800
# renders of all six palette colours plus the baseplate through every corner of
# `render.domain_rand.DEFAULT_POSTPROCESS_RANGES` (exposure/gamma/white-balance/noise/blur/
# JPEG), the worst-case distance from a colour's own reference (<=0.41) and the closest any
# colour ever got to a *different* reference or the baseplate (>=0.53); `tol` sits at 0.40,
# just under the self-distance ceiling, so a rare extreme-lighting miss is a dropped detection,
# never a false merge. See `tests/perception/test_producers.py` for the calibration tests.
_DEFAULT_CHROMA_GAIN = 2.0
_DEFAULT_TOL = 0.40


def _hsv_cone(rgb: tuple[float, float, float], chroma_gain: float = _DEFAULT_CHROMA_GAIN) -> np.ndarray:
    """Embeds one RGB colour (each channel in [0, 1]) as `(chroma_gain * S * cos H, chroma_gain
    * S * sin H, V)`.

    Plain RGB Euclidean distance (what `BlobProducer` used before) put its "green" only 0.186
    from the default green baseplate colour -- comfortably inside a tol of 0.25 -- even though
    the two are visibly different (the baseplate is darker and more saturated): raw RGB
    distance conflates "different colour" with "different brightness", and brightness is
    exactly what a scene's lighting and this module's own exposure/gamma/white-balance
    post-process (`render.domain_rand.apply_postprocess`) change from render to render without
    the object actually being repainted.

    Hue and saturation -- not value -- are what tell two paint colours apart under changing
    light, so this embedding boosts their plane's contribution instead of shrinking value's.
    Shrinking value directly was tried first and does fix the baseplate/green collision, but it
    breaks telling an unsaturated colour like "white" apart from a plain grey background,
    because at low saturation hue/saturation carry almost no signal and value is the *only*
    thing left to compare there. Boosting the hue-plane's gain instead widens the gap between
    two saturated-but-similar-hue colours (or a colour and the baseplate) without ever touching
    value, so grey-vs-white stays exactly as separated as plain HSV already had it.
    """
    r, g, b = (float(c) for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    theta = h * 2.0 * math.pi
    return np.array(
        [chroma_gain * s * math.cos(theta), chroma_gain * s * math.sin(theta), v], dtype=np.float64
    )


def hsv_colour_distance(
    a: tuple[float, float, float], b: tuple[float, float, float], chroma_gain: float = _DEFAULT_CHROMA_GAIN
) -> float:
    """Perceptual-ish colour distance used for palette/baseplate separation (see `_hsv_cone`)."""
    return float(np.linalg.norm(_hsv_cone(a, chroma_gain) - _hsv_cone(b, chroma_gain)))


def _hsv_cone_image(img: np.ndarray, chroma_gain: float = _DEFAULT_CHROMA_GAIN) -> np.ndarray:
    """Vectorized `_hsv_cone` for a whole `(H, W, 3)` float32 RGB image in `[0, 1]`, shape
    `(H, W, 3)` out. Uses `cv2.cvtColor` (float32 convention: H in degrees, S/V in `[0, 1]`,
    matching `render.domain_rand.jitter_rgb_hsv`'s own convention) rather than a per-pixel
    Python loop over `colorsys`, for speed; cross-checked against `_hsv_cone` directly in
    `tests/perception/test_producers.py`."""
    hsv = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.float32), cv2.COLOR_RGB2HSV)
    theta = np.deg2rad(hsv[..., 0])
    s = hsv[..., 1]
    v = hsv[..., 2]
    return np.stack(
        [chroma_gain * s * np.cos(theta), chroma_gain * s * np.sin(theta), v], axis=-1
    )


def _nearest_colour(rgb: np.ndarray, palette: dict[str, tuple[float, float, float]]) -> str:
    """The palette name whose RGB (each channel in [0,1]) is closest to `rgb` [3]."""
    names = list(palette)
    table = np.asarray([palette[n] for n in names], dtype=np.float64)
    d = np.sum((table - np.asarray(rgb, dtype=np.float64)) ** 2, axis=-1)
    return names[int(np.argmin(d))]


class TokenProducer(Protocol):
    """Common interface every producer implements; control code depends on this and nothing else."""

    def produce(
        self,
        rng: np.random.Generator,
        *,
        frame: np.ndarray | None = None,
        ground_truth: "Sequence[GroundTruthObject] | None" = None,
    ) -> list[ObjectToken]:
        """This frame's object tokens. Implementations read whichever of `frame` /
        `ground_truth` they need and ignore the other; both are keyword-only so a caller can
        hand every producer the same call site."""
        ...


@dataclass
class GroundTruthObject:
    """One simulator-truth object, already projected to image pixels by the caller.

    `x`, `y` in pixels, `yaw` in radians, `size` in pixels (a characteristic length, e.g. the
    brick's on-screen footprint diagonal); `layer` is the true stack layer (0 = base).
    """

    x: float
    y: float
    yaw: float
    layer: int
    colour: str
    size: float


@dataclass
class GroundTruthProducer:
    """Sim ground truth + a parametric noise model fitted to a vision model's measured errors.

    `bias_px`: a constant offset applied to every position (models a systematic decode bias --
    14-62 px measured on an untrained checkpoint on an unseen scene). `jitter_std_px`: per-axis
    Gaussian position noise (models draw-to-draw centroid jitter, measured sigma 2-20 px).
    `yaw_jitter_rad`: Gaussian yaw noise. `dropout_p`: probability an object is not reported at all
    (models a slot never firing on it). `merge_radius_px`: same-colour objects whose ground-truth
    centres are closer than this are reported as one merged token (models the structural limit that
    same-colour touching objects are separable only by position, and K=8 can run out).
    """

    bias_px: tuple[float, float] = (0.0, 0.0)
    jitter_std_px: float = 4.0
    yaw_jitter_rad: float = 0.05
    dropout_p: float = 0.02
    merge_radius_px: float = 6.0

    def produce(
        self,
        rng: np.random.Generator,
        *,
        frame: np.ndarray | None = None,
        ground_truth: Sequence[GroundTruthObject] | None = None,
    ) -> list[ObjectToken]:
        if ground_truth is None:
            raise ValueError("GroundTruthProducer.produce needs `ground_truth`")

        kept: list[ObjectToken] = []
        for i, obj in enumerate(ground_truth):
            if rng.random() < self.dropout_p:
                continue
            x = obj.x + self.bias_px[0] + rng.normal(0.0, self.jitter_std_px)
            y = obj.y + self.bias_px[1] + rng.normal(0.0, self.jitter_std_px)
            yaw = obj.yaw + rng.normal(0.0, self.yaw_jitter_rad)
            kept.append(
                ObjectToken(
                    x=float(x),
                    y=float(y),
                    yaw=float(yaw),
                    layer=int(obj.layer),
                    colour=obj.colour,
                    size=float(obj.size),
                    confidence=float(1.0 - self.dropout_p),
                    age=0.0,
                    slot=i,
                )
            )

        # Same-colour merge: a K=8 slot budget and single-pixel evidence mean two nearby
        # same-colour objects are not always resolved as separate. Greedy pairwise merge.
        merged: list[ObjectToken] = []
        used = [False] * len(kept)
        for i, a in enumerate(kept):
            if used[i]:
                continue
            group = [a]
            used[i] = True
            for j in range(i + 1, len(kept)):
                if used[j] or kept[j].colour != a.colour:
                    continue
                if np.hypot(kept[j].x - a.x, kept[j].y - a.y) <= self.merge_radius_px:
                    group.append(kept[j])
                    used[j] = True
            if len(group) == 1:
                merged.append(a)
            else:
                merged.append(
                    ObjectToken(
                        x=float(np.mean([t.x for t in group])),
                        y=float(np.mean([t.y for t in group])),
                        yaw=float(np.mean([t.yaw for t in group])),
                        layer=int(round(np.mean([t.layer for t in group]))),
                        colour=a.colour,
                        size=float(np.mean([t.size for t in group])),
                        confidence=float(np.mean([t.confidence for t in group])),
                        age=0.0,
                        slot=a.slot,
                    )
                )
        return merged


@dataclass
class BlobProducer:
    """Classical colour-thresholding fallback: no learned model at all.

    Thresholds the frame against each palette colour (HSV-cone distance <= `tol`, see
    `_hsv_cone`), finds connected components per colour, and keeps those with area >=
    `min_area_px`. Reports one token per component: centroid and orientation from
    `cv2.connectedComponentsWithStats` / `cv2.minAreaRect`, confidence 1.0 (there is no model
    uncertainty to report), layer always 0 (a single visible layer is all a flat colour mask
    can tell apart without a depth register or known geometry).

    perception-align-002: this used to compare in plain RGB space, which put "green" only
    0.186 from the default green baseplate colour -- comfortably inside the old `tol=0.25` --
    so a green brick on the green baseplate merged into one component spanning most of the
    frame. `tol`/`chroma_gain`'s new defaults are calibrated (see `_DEFAULT_CHROMA_GAIN`'s
    comment) to separate the six palette colours from each other and from `baseplate_rgb`
    under `render.domain_rand.apply_postprocess`'s full sampled range. `baseplate_rgb`
    defaults to the scene's actual default plate colour rather than `None` so this protection
    is on by default; `__post_init__` refuses (raises `ValueError`) to construct a producer
    whose configured palette cannot actually be told apart from `baseplate_rgb` at `tol`,
    instead of silently building one that would merge a brick into the plate at runtime. Pass
    `baseplate_rgb=None` to skip the check entirely (e.g. a scene with no baseplate at all).
    """

    palette: dict[str, tuple[float, float, float]] = None  # type: ignore[assignment]
    tol: float = _DEFAULT_TOL
    min_area_px: int = 8
    baseplate_rgb: tuple[float, float, float] | None = DEFAULT_BASEPLATE_RGB
    chroma_gain: float = _DEFAULT_CHROMA_GAIN

    def __post_init__(self) -> None:
        if self.palette is None:
            self.palette = PALETTE
        if self.baseplate_rgb is not None:
            conflicts = [
                f"{name!r} (distance {d:.3f} <= tol {self.tol:.3f})"
                for name, rgb in self.palette.items()
                if (d := hsv_colour_distance(rgb, self.baseplate_rgb, self.chroma_gain)) <= self.tol
            ]
            if conflicts:
                raise ValueError(
                    f"BlobProducer cannot separate the baseplate colour {self.baseplate_rgb} "
                    f"from palette colour(s): {', '.join(conflicts)}. A brick this colour would "
                    "merge into the baseplate instead of being detected. Either change the "
                    "baseplate colour, drop the colour from `palette`, lower `tol`, or pass "
                    "`baseplate_rgb=None` to skip this check (e.g. a scene with no baseplate)."
                )

    def produce(
        self,
        rng: np.random.Generator,
        *,
        frame: np.ndarray | None = None,
        ground_truth: Sequence[GroundTruthObject] | None = None,
    ) -> list[ObjectToken]:
        if frame is None:
            raise ValueError("BlobProducer.produce needs `frame`")
        frame_arr = np.asarray(frame, dtype=np.float32)
        if frame_arr.max() > 1.0 + 1e-6:
            frame_arr = frame_arr / 255.0
        frame_arr = frame_arr[..., :3]
        cones = _hsv_cone_image(frame_arr, self.chroma_gain)

        tokens: list[ObjectToken] = []
        slot = 0
        for name, rgb in self.palette.items():
            dist = np.linalg.norm(cones - _hsv_cone(rgb, self.chroma_gain), axis=-1)
            mask = (dist <= self.tol).astype(np.uint8)
            if mask.sum() == 0:
                continue
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, num_labels):  # label 0 is the background
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < self.min_area_px:
                    continue
                comp_mask = (labels == label).astype(np.uint8)
                ys, xs = np.nonzero(comp_mask)
                pts = np.stack([xs, ys], axis=-1).astype(np.float32)
                rect = cv2.minAreaRect(pts)
                (cx, cy), (rw, rh), angle_deg = rect
                # cv2's own component centroid (image moments, exact for an axis-aligned block).
                cx, cy = centroids[label]
                tokens.append(
                    ObjectToken(
                        x=float(cx),
                        y=float(cy),
                        yaw=float(np.deg2rad(angle_deg)),
                        layer=0,
                        colour=name,
                        size=float(max(rw, rh)),
                        confidence=1.0,
                        age=0.0,
                        slot=slot,
                    )
                )
                slot += 1
        return tokens
