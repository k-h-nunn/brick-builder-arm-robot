"""Video panels and layout helpers for the scene-target environment (WP8).

`scripts/make_scene_target_video.py` is the program that drives a world and writes a file;
this module is the picture and text it draws, plus the pieces small enough to unit-test without
a renderer: the panel bookkeeping, the dense-decode colouriser, the on-screen QA checks, and the
target-sequence decode cache.

**Panels, titled exactly** (`PANEL_TITLES`): "Arm camera POV" (the wrist camera the scene model
reads), "Observed scene state", "Target scene state" and "Predicted next scene state". The three
scene-state panels are the same source's dense decode (`dense_decode_rgb`): each pixel's slot,
by `SceneState.decode`, coloured by that slot's own published colour -- the same idea as
`perception.cnn.source.CnnSceneModel.colourize_instances`, but driven through the decode
function so it works for any producer that builds one (the ideal source's ground-truth label map
and the convolutional model's instance map alike), not just the CNN. A producer built with
`build_decode=False`, or one that never builds a decode at all, gets `slot_row_panel` instead: a
schematic strip of its rows, honestly labelled as "no decode" rather than faked.

**Predicted panel.** No predictor exists yet (WP4/WP5 land later): `Panels` shows copy-last --
the previous tick's *observed* decode -- which is exactly what a copy-last predictor's estimate
of the next state looks like. `scripts/make_scene_target_video.py`'s spec line says so on screen
("Predictor: copy-last").

**Reused, not rebuilt:** `demo.video.Recorder` (the camera, the encode, the frame-rate match)
and `demo.video_panels.Overlay`/`check_layout` (every text box, and the check that none of them
touch or leave the safe area) -- this module draws none of the four fixed elements itself beyond
the pictures that go inside a panel and, in `StripTitleOverlay`, where the title card lands (the
same fix `scripts/make_campaign_video.py` already needed, for the same reason: a shot that keeps
the whole arm in view puts it in the corner `Overlay` would otherwise draw the title over).
`ArmAwareRecorder` adds the one check `Overlay` cannot do on its own: how many of the arm's own
pixels a text box covers.

Nothing here imports `brick_builder.action` (`tests/rl/scene_target/test_guard.py` checks every
module in this package for that), and nothing here decodes on the control path -- decode is
viewer-only everywhere in this project (see `perception/scene_model.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ...perception.scene_model import SlotSchema
from ...demo.video import ReadoutSpec, Recorder
from ...demo.video_panels import (
    DIM_TEXT, LayoutError, Overlay, Rect, STRIP_BOX, TEXT, check_layout,
)

__all__ = [
    "ARM_INSTANCE", "DARK", "DEFAULT_READOUTS", "FALLBACK_PALETTE", "IDENTIFIER", "MAX_MAIN_VIEW_GRADIENT",
    "MIN_FONT_PX", "PANEL_TITLES", "PROJECT_TITLE", "ArmAwareRecorder", "Panels", "StripTitleOverlay",
    "arm_body_names", "check_stills_layout", "count_arm_pixels_under_boxes", "dense_decode_rgb",
    "load_or_render_target_decode", "main_view_gradient", "render_target_decode",
    "screen_identifiers", "slot_row_panel", "spec_line", "target_decode_cache_path",
]

PANEL_TITLES = ("Arm camera POV", "Observed scene state", "Target scene state",
                "Predicted next scene state")
"""The four panel titles, exact (A5, WP8's own acceptance test)."""

PROJECT_TITLE = "Brick Builder Arm Robot"
"""The title card's project name -- the same one `scripts/make_demo_video.py` and
`scripts/make_campaign_video.py` already put on screen; one project, one name."""

DARK = (20, 22, 26)
"""Background of every decode panel: unclaimed pixels, and the fallback panel's own background."""

FALLBACK_PALETTE = (
    (99, 178, 255), (247, 160, 86), (109, 201, 140), (219, 111, 204),
    (233, 196, 57), (140, 140, 231), (224, 110, 110), (116, 201, 201),
)
"""Eight distinct slot colours, used only when a schema has no `colour` field to read (an opaque
producer, WP10) -- the dense decode still colours by slot, it just cannot use the object's own
measured colour."""

DEFAULT_READOUTS = (
    ReadoutSpec(key="d_mm", label="Scene-state distance [mm-eq]", unit="", lo=0.0, hi=None,
               fmt="{:.2f}", plot=True),
    ReadoutSpec(key="pred_err_mm", label="Prediction error [mm-eq]", unit="", lo=0.0, hi=None,
               fmt="{:.2f}"),
    ReadoutSpec(key="tool_err_mm", label="Tool position error [mm]", unit="", lo=0.0, hi=None,
               fmt="{:.2f}"),
)
"""The data strip: scene-state distance (the reward and success signal, plotted), the copy-last
prediction error, and the ground-truth tool position error (A5, section 5.2)."""


def spec_line(*, controller: str, predictor: str, scene_model: str, level: str,
             offset_mm: float, offset_deg: float) -> str:
    """The one line under the title card: titles and values, no sentences (5.2).

    Kept short on purpose: at 1920x1080 the title block has 922 px to the panel column, which
    is three lines of the spec-line face (`video_panels.MAX_SPEC_LINES`) -- a field's own value
    (not just its name) has to fit that, so this stays terse rather than descriptive."""
    return (f"Controller: {controller} | Predictor: {predictor} | Scene model: {scene_model} | "
            f"Level: {level} | Offset: {offset_mm:.1f} mm, {offset_deg:.1f} deg")


# ------------------------------------------------------------------------------- dense decode ---


def _pixel_grid(res: int) -> np.ndarray:
    """`[res*res, 2]` (col, row), row-major so `.reshape(res, res)` undoes it."""
    rr, cc = np.indices((res, res))
    return np.stack([cc.ravel(), rr.ravel()], axis=1).astype(np.int64)


_GRID_CACHE: dict[int, np.ndarray] = {}


def _grid(res: int) -> np.ndarray:
    grid = _GRID_CACHE.get(res)
    if grid is None:
        grid = _pixel_grid(res)
        _GRID_CACHE[res] = grid
    return grid


def _lift(rgb01: np.ndarray) -> np.ndarray:
    """Colour lifted towards white so a dark object stays visible (`colourize_instances`'s own
    recipe): `60 + 195 * c`, clipped to a byte."""
    return np.clip(60.0 + 195.0 * np.clip(rgb01, 0.0, 1.0), 0, 255).astype(np.uint8)


def dense_decode_rgb(state, *, res: int, background=DARK) -> np.ndarray:
    """One scene state's dense decode, coloured by slot: `uint8 [res, res, 3]`.

    Every pixel's slot comes from `state.decode` (argmax over the per-pixel logits), so this
    works for any producer that built one -- the ideal source's ground-truth label map and the
    convolutional model's instance map alike (2.6, 5.2). A producer with no decode function
    (`state.has_decode` false) falls back to `slot_row_panel`.
    """
    if not getattr(state, "has_decode", False):
        return slot_row_panel(state.slots, state.schema, res=res, background=background)
    px = _grid(res)
    logits = np.asarray(state.decode(px))
    k = np.argmax(logits, axis=1).reshape(res, res)
    return colourize_slots(k, state.slots, state.schema, K=int(state.K), background=background)


def colourize_slots(slot_map: np.ndarray, rows: np.ndarray, schema: SlotSchema, *, K: int,
                    background=DARK) -> np.ndarray:
    """`slot_map [H, W]` (one slot index per pixel) -> `uint8 [H, W, 3]`, plus a one-pixel dark
    edge between neighbouring slots, the same convention as `CnnSceneModel.colourize_instances`.
    """
    slot_map = np.asarray(slot_map)
    out = np.empty((*slot_map.shape, 3), dtype=np.uint8)
    out[:] = np.asarray(background, dtype=np.uint8)
    has_colour = schema.has("colour")
    for k in range(K):
        mask = slot_map == k
        if not mask.any():
            continue
        if has_colour:
            rgb01 = (np.asarray(schema.get(rows[k], "colour"), dtype=np.float64) + 1.0) * 0.5
        else:
            r, g, b = FALLBACK_PALETTE[k % len(FALLBACK_PALETTE)]
            rgb01 = np.array([r, g, b], dtype=np.float64) / 255.0
        out[mask] = _lift(rgb01)
    edge = np.zeros(slot_map.shape, dtype=bool)
    edge[:-1] |= slot_map[:-1] != slot_map[1:]
    edge[:, :-1] |= slot_map[:, :-1] != slot_map[:, 1:]
    out[edge] = (10, 11, 13)
    return out


def slot_row_panel(rows: np.ndarray, schema: SlotSchema, *, res: int, background=DARK) -> np.ndarray:
    """A schematic fallback for a producer with no dense decode: one horizontal bar per slot,
    the slot's own colour if the schema has one, sized by its mass if the schema has that too.

    Honest about what it is showing -- rows, not pixels -- rather than fabricating a decode a
    source never built (the owner ruling in 5.2 is against ovals standing in for a real decode;
    this is not a decode at all, and is only reached when one does not exist).
    """
    rows = np.asarray(rows)
    K = rows.shape[0]
    out = np.empty((res, res, 3), dtype=np.uint8)
    out[:] = np.asarray(background, dtype=np.uint8)
    has_colour = schema.has("colour")
    has_mass = schema.has("mass")
    bar_h = max(res // K, 1)
    for k in range(K):
        y0, y1 = k * bar_h, min((k + 1) * bar_h, res)
        if y0 >= res:
            break
        if has_colour:
            rgb01 = (np.asarray(schema.get(rows[k], "colour"), dtype=np.float64) + 1.0) * 0.5
        else:
            r, g, b = FALLBACK_PALETTE[k % len(FALLBACK_PALETTE)]
            rgb01 = np.array([r, g, b], dtype=np.float64) / 255.0
        width = res
        if has_mass:
            mass = float(np.clip(schema.get(rows[k], "mass")[0], 0.0, 1.0))
            width = max(int(round(res * min(mass * 4.0, 1.0))), 2)
        out[y0:y1, :width] = _lift(rgb01)
    return out


# ------------------------------------------------------------------------------------ panels ---


@dataclass
class Panels:
    """The four panel pictures, rebuilt on every vision tick and held between them.

    `update` takes this tick's wrist camera image and the two decode pictures (observed,
    target); the fourth, predicted, is last tick's observed decode -- copy-last (module
    docstring). `entries()` is what `demo.video.Recorder(panel_images=...)` wants: `n_panels`
    pairs of `(title, rgb)`, each exactly `size`.
    """

    size: tuple[int, int]
    background: tuple[int, int, int] = DARK
    images: list = field(init=False, repr=False)
    _last_observed: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        w, h = int(self.size[0]), int(self.size[1])
        self.size = (w, h)
        blank = np.empty((h, w, 3), dtype=np.uint8)
        blank[:] = np.asarray(self.background, dtype=np.uint8)
        self.images = [blank.copy() for _ in PANEL_TITLES]

    def reset(self) -> None:
        """New episode: no earlier observation to copy-last from."""
        self._last_observed = None

    def _fit(self, img: np.ndarray, *, nearest: bool) -> np.ndarray:
        import cv2

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if (arr.shape[1], arr.shape[0]) == self.size:
            return np.ascontiguousarray(arr)
        mode = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(arr, self.size, interpolation=mode)

    def update(self, *, wrist_rgb: np.ndarray, observed_rgb: np.ndarray,
              target_rgb: np.ndarray) -> None:
        """`wrist_rgb` is a photo (smoothed resize); `observed_rgb`/`target_rgb` are decode
        pictures (nearest-neighbour, so slot boundaries stay sharp instead of blurring into a
        colour that names no slot)."""
        self.images[0] = self._fit(wrist_rgb, nearest=False)
        observed = self._fit(observed_rgb, nearest=True)
        self.images[1] = observed
        self.images[2] = self._fit(target_rgb, nearest=True)
        self.images[3] = observed if self._last_observed is None else self._last_observed
        self._last_observed = observed

    def entries(self) -> list[tuple[str, np.ndarray]]:
        return list(zip(PANEL_TITLES, self.images))


# ------------------------------------------------------------------------- arm-aware recorder ---


class StripTitleOverlay(Overlay):
    """`Overlay` with the title card drawn in the data strip's box instead of the top-left
    corner.

    This project's viewing-pose camera keeps the whole arm in shot by framing it broadly, which
    puts the upper arm and wrist in the top-left corner where `Overlay` draws its title -- the
    same problem `scripts/make_campaign_video.py`'s own closed-loop shot has, solved there the
    same way: the strip has nothing measured to show for the opening seconds, so the title
    borrows its box and gives it back once the readouts have something to draw.
    """

    def draw(self, picture, *, title=None, subtitle=None, readouts=(), **kw):
        if not title:
            return super().draw(picture, title=None, subtitle=subtitle, readouts=readouts, **kw)
        from PIL import Image, ImageDraw

        composed, rects = super().draw(picture, title=None, subtitle=None, readouts=(), **kw)
        rect = self.strip_rect()
        max_px = rect.width - 2 * self.pad
        if self.f_title.getlength(title) > max_px:
            raise LayoutError(f"the title does not fit {max_px} px: {title!r}")
        lines = [(title, self.f_title, TEXT)]
        if subtitle:
            lines += [(line, self.f_subtitle, DIM_TEXT)
                      for line in self._spec_lines(subtitle, max_px)]
        height = sum(self._line_height(font) for _, font, _ in lines)
        base = Image.fromarray(np.ascontiguousarray(composed), mode="RGB").convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        self._box(draw, rect, fill=STRIP_BOX)
        y = rect.y0 + max((rect.height - height) // 2, self.pad)
        for text, font, colour in lines:
            draw.text((rect.x0 + self.pad, y), text, font=font, fill=colour)
            y += self._line_height(font)
        rects = dict(rects)
        rects["title"] = rect
        check_layout(rects, self.safe)
        out = Image.alpha_composite(base, layer).convert("RGB")
        return np.asarray(out, dtype=np.uint8), rects


ARM_INSTANCE = 10_000
"""Instance id the main view's segmentation pass gives every arm body, for the QA check that no
text box covers the robot (the same convention `scripts/make_campaign_video.py` uses)."""


def arm_body_names(model, root: str = "base") -> list[str]:
    """Every named body hanging off `root` in the body tree: the arm, jaws included."""
    import mujoco

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    if root_id < 0:
        raise ValueError(f"no body named {root!r} to identify the arm from")
    in_arm = np.zeros(model.nbody, dtype=bool)
    in_arm[root_id] = True
    for b in range(root_id + 1, model.nbody):
        in_arm[b] = in_arm[int(model.body_parentid[b])]
    return [model.body(b).name for b in np.nonzero(in_arm)[0] if model.body(b).name]


def count_arm_pixels_under_boxes(instance_map: np.ndarray, arm_instance: int,
                                 rects: dict[str, Rect]) -> dict[str, int]:
    """How many of `instance_map`'s `arm_instance` pixels fall inside each rectangle.

    Pure and independent of any render, so it is unit-tested directly (`test_viewer.py`) instead
    of only through a real take."""
    instance_map = np.asarray(instance_map)
    out = {}
    for name, r in rects.items():
        y0, y1 = max(r.y0, 0), min(r.y1, instance_map.shape[0])
        x0, x1 = max(r.x0, 0), min(r.x1, instance_map.shape[1])
        out[name] = int(np.count_nonzero(instance_map[y0:y1, x0:x1] == arm_instance))
    return out


class ArmAwareRecorder(Recorder):
    """`Recorder` plus the one QA check `Overlay` cannot do on its own: how many pixels of the
    arm land under a drawn text box, on every frame written."""

    def __init__(self, scene, **kw) -> None:
        super().__init__(scene, **kw)
        self.overlay = StripTitleOverlay(self.quality.width, self.quality.height)
        lut = dict(scene.instance_ids)
        arm = arm_body_names(scene.model)
        lut.update({name: ARM_INSTANCE for name in arm})
        self._main.instance_lut(lut)
        self.arm_bodies = len(arm)
        self.arm_under_text: dict[str, int] = {}
        self._frame = None
        render_free = self._main.render_free

        def keep(*args, **kwargs):
            self._frame = render_free(*args, **kwargs)
            return self._frame

        self._main.render_free = keep

    def compose(self, tick, video_t):
        composed, rects, span, overrun, ray = super().compose(tick, video_t)
        for key, n in count_arm_pixels_under_boxes(self._frame.instance, ARM_INSTANCE, rects).items():
            self.arm_under_text[key] = max(self.arm_under_text.get(key, 0), n)
        return composed, rects, span, overrun, ray


# ------------------------------------------------------------------------------------ QA ---


MAX_MAIN_VIEW_GRADIENT = 5.0
"""Mean absolute difference between horizontally adjacent pixels of the main view [0-255]. A
clean render measures a couple of units; a frame read from a freed framebuffer reads as noise
(the same threshold and the same reasoning as `scripts/make_campaign_video.py`)."""

MIN_FONT_PX = 22
"""Smallest glyph size any on-screen string may use at 1080 lines [px]."""

IDENTIFIER = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b|\w\(|\.py\b|\b[a-z]+[A-Z][A-Za-z]*\b"
                        r"|\b(?=[0-9A-Fa-f]*[0-9])(?=[0-9A-Fa-f]*[A-Fa-f])[0-9A-Fa-f]{6,}\b")
"""snake_case, a call, a file name, camelCase, a hex hash: what a code identifier or a checkpoint
tag looks like on screen. Repeats `scripts/make_campaign_video.py`'s own pattern -- one leak-audit
regex, not two designs of one check."""


def screen_identifiers(strings: Sequence[str], *, allow: frozenset[str] = frozenset()) -> list[str]:
    """Every substring of `strings` that reads as a code identifier or a hash, `allow` excepted."""
    hits = []
    for text in strings:
        for match in IDENTIFIER.findall(text):
            if match not in allow:
                hits.append(match)
    return hits


def main_view_gradient(picture: np.ndarray, clear: Rect) -> float:
    """Mean absolute horizontal-neighbour difference inside `clear`: a sanity check that the
    main view is a render and not a freed framebuffer's noise."""
    view = np.asarray(picture)[clear.y0:clear.y1, clear.x0:clear.x1].astype(np.int16)
    return float(np.mean(np.abs(np.diff(view, axis=1))))


def check_stills_layout(rects: dict[str, Rect], safe: Rect) -> None:
    """No two drawn elements intersect and none leaves the safe area (raises `LayoutError`).

    A thin, named wrapper around `video_panels.check_layout` so a caller (and a test) can name
    the exact requirement WP8's acceptance test asks for, "a programmatic layout check", without
    reaching past this module into `demo.video_panels` for it."""
    check_layout(rects, safe)


# ------------------------------------------------------------------- target-sequence decode ---


def target_decode_cache_path(cache_dir, family: str, seed: int, source: str, weights_sha256: str,
                             variant: str = "") -> Path:
    """`<cache_dir>/decode/<family>/seed<seed>_<source>_<sha8>[_<variant>].npz`.

    A cache of its own, next to `targets.py`'s own `<cache_dir>/targets/...`: the target cache
    stores no per-pixel picture (2.6 -- `TargetSequence` is arrays only, no pickled state, so a
    source's decode closure cannot round-trip through it), and generating one is a video-only
    cost nothing else in this package pays."""
    sha8 = weights_sha256[:8] if weights_sha256 else "none"
    tail = f"_{variant}" if variant else ""
    return Path(cache_dir) / "decode" / family / f"seed{int(seed):05d}_{source}_{sha8}{tail}.npz"


def render_target_decode(seq, *, source_name: str, source_kwargs: dict | None = None,
                         res: int = 256) -> np.ndarray:
    """Replays the exact drawn viewing-pose path recorded in `seq.meta` through a fresh,
    decode-enabled source, and returns every frame's dense decode: `uint8 [T, res, res, 3]`.

    Same replay as `targets.viewing_pose_sequence` (the recorded start pose, target pose, `T`
    and draw number, not a re-drawn one), through a source built with `build_decode=True` --
    the target cache's own source is built without it (2.6, `world.build_source`'s default),
    since nothing else needs the picture.
    """
    from .targets import ViewingPoseDraw
    from .world import NominalTwin, build_source, observe, reset_source

    meta = seq.meta
    draw = ViewingPoseDraw(start=np.asarray(meta["start_pose"], dtype=float),
                           target=np.asarray(meta["target_pose"], dtype=float),
                           T=int(meta.get("T", seq.T)), draw=int(meta.get("draw", 1)))
    if draw.T != seq.T:
        raise ValueError(f"seed {meta.get('seed')}: recorded T={seq.T} but meta T={draw.T}")
    poses = draw.frame_poses()
    source, _ = build_source(source_name, source_kwargs, build_decode=True)
    twin = NominalTwin()
    out = np.empty((seq.T, res, res, 3), dtype=np.uint8)
    try:
        reset_source(source)
        for k, p in enumerate(poses):
            twin.pose(p[:3], p[3])
            frame = twin.render()
            state = observe(source, frame, t=0.1 * k)
            out[k] = dense_decode_rgb(state, res=res)
    finally:
        twin.close()
    return out


def load_or_render_target_decode(seq, *, source_name: str, source_kwargs: dict | None = None,
                                 cache_dir, variant: str = "") -> np.ndarray:
    """`render_target_decode`, cached to `target_decode_cache_path`."""
    meta = seq.meta
    family = str(meta.get("family", "viewing_pose"))
    seed = int(meta.get("seed", -1))
    weights_sha256 = str(meta.get("weights_sha256", ""))
    path = target_decode_cache_path(cache_dir, family, seed, source_name, weights_sha256, variant)
    if path.exists():
        with np.load(path) as z:
            if int(z["T"]) == seq.T:
                return z["images"]
    images = render_target_decode(seq, source_name=source_name, source_kwargs=source_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, images=images, T=np.asarray(seq.T, dtype=np.int64))
    tmp.replace(path)
    return images
