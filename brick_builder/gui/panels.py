"""The four square panels and the composed frame.

Drawn with the video toolkit (`brick_builder.demo.video`, `brick_builder.demo.video_panels`), so
the page and the published video put the same picture in the same place with the same titles:
the free camera behind, four square panels down the right, and the readout strip across the
bottom with one plot on it.
"""

from __future__ import annotations

import numpy as np

from ..demo.video import (
    FreeCameraRenderer, Quality, ReadoutSpec, backdrop, floor_grid, quality_by_name, studio_look,
)
from ..demo.video_panels import Overlay, Readout

PANEL_TITLES = ("Arm camera POV", "Observed scene state",
                "Target scene state", "Ground truth segmentation")

READOUTS = (
    ReadoutSpec(key="err_mm", label="Scene-state error [mm]", unit="", lo=0.0, hi=None,
                fmt="{:.2f}", reference=0.4, plot=True),
    ReadoutSpec(key="miou", label="Segmentation mIoU [%]", unit="", lo=0.0, hi=100.0,
                fmt="{:.0f}"),
    ReadoutSpec(key="predict_mm", label="Prediction error [mm]", unit="", lo=0.0, hi=5.0,
                fmt="{:.2f}"),
    ReadoutSpec(key="press_n", label="Press force [N]", unit="", lo=0.0, hi=20.0, fmt="{:.1f}"),
    ReadoutSpec(key="depth_mm", label="Insertion depth [mm]", unit="", lo=0.0, hi=5.0,
                fmt="{:.2f}"),
)
"""The five the strip holds. The page's own table carries these and the two the strip has no
room for -- the step title and the damage state."""

TRACE_POINTS = 600
"""Points kept behind the plot, about a minute of vision ticks."""

DARK = (20, 22, 26)
ARM_TINT = (176, 158, 106)
PLATE_TINT = (62, 150, 96)
"""One palette for all three label panels, the same one the video uses: background and table
dark, the arm tan, the baseplate green, every brick in its own measured colour."""

VIEW_LOOKAT = (-0.06, 0.14, 0.06)
VIEW_DISTANCE = 1.00
VIEW_AZIMUTH = 268.0
VIEW_ELEVATION = -24.0

DISTANCE_RANGE = (0.35, 2.20)
ELEVATION_RANGE = (-89.0, 5.0)


def paint_labels(class_map, instance_map, rgb) -> np.ndarray:
    """One label image, in the convention all three label panels share.

    `class_map` says background / baseplate / arm / brick per pixel; `instance_map` separates one
    brick from the next. Bricks take the mean colour of their own pixels in `rgb`, lifted towards
    white so a dark brick is still visible.
    """
    from ..perception.cnn.labels import ARM_CLASS, BASEPLATE_CLASS, BRICK_CLASS

    class_map = np.asarray(class_map)
    instance_map = np.asarray(instance_map)
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.empty((*class_map.shape, 3), np.uint8)
    out[:] = DARK
    out[class_map == ARM_CLASS] = ARM_TINT
    out[class_map == BASEPLATE_CLASS] = PLATE_TINT
    bricks = class_map == BRICK_CLASS
    for i in np.unique(instance_map[bricks]):
        mask = bricks & (instance_map == i)
        if not mask.any():
            continue
        mean = rgb[mask].mean(axis=0) / 255.0
        out[mask] = np.clip(60.0 + 195.0 * mean, 0, 255).astype(np.uint8)
    return out


class Labeller:
    """The renderer's own labels for one world, in the convention the accuracy is measured in."""

    def __init__(self, scene):
        from ..action.adapter import WRIST_FULL, WRIST_RENDER
        from ..perception.cnn.data import perception_instance_lut
        from ..render.renderer import SceneRenderer

        self.window = WRIST_FULL
        self.renderer = SceneRenderer(scene.model, width=WRIST_RENDER[0], height=WRIST_RENDER[1])
        self.renderer.instance_lut(perception_instance_lut(scene))

    def __call__(self, scene) -> dict:
        from ..perception.cnn.labels import class_map_from_instance
        from ..perception.views import render_view

        frame, _ = render_view(scene, self.window, self.renderer)
        instance = np.asarray(frame.instance)
        return {"instance": instance, "class": class_map_from_instance(instance),
                "rgb": np.asarray(frame.rgb)}

    def close(self):
        self.renderer.close()


def recording_labeller():
    """A `label_hook` for `record_ideal`, which owns its world and only lends it out per frame."""
    box: dict = {}

    def hook(scene):
        if "labeller" not in box:
            box["labeller"] = Labeller(scene)
        return box["labeller"](scene)

    return hook


class Panels:
    """The four pictures, rebuilt on every vision tick and held between them."""

    def __init__(self, size, source=None):
        self.w, self.h = int(size[0]), int(size[1])
        self.source = source if hasattr(source, "feature_maps") else None
        blank = np.zeros((self.h, self.w, 3), np.uint8)
        blank[:] = DARK
        self.images = [blank.copy() for _ in PANEL_TITLES]
        self.miou = float("nan")

    def _fit(self, img, nearest: bool) -> np.ndarray:
        import cv2

        mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(np.asarray(img), (self.w, self.h), interpolation=mode)

    def update(self, rgb, target_label, truth_label, *, window=None) -> None:
        """`target_label` and `truth_label` are `Labeller` outputs; `rgb` is this frame's camera
        image, which the scene model sees and the brick colours are measured from."""
        if rgb is not None:
            self.images[0] = self._fit(rgb, nearest=False)

        if self.source is not None and rgb is not None:
            maps = self.source.feature_maps(np.asarray(rgb), window=window)
            shown = paint_labels(maps["class_map"], maps["instance_map"], rgb)
            if truth_label is not None:
                try:
                    self.miou = 100.0 * float(self.source.accuracy(
                        maps["instance_map"], truth_label["instance"],
                        truth_label["class"])["miou"])
                except Exception:
                    self.miou = float("nan")
            self.images[1] = self._fit(shown, nearest=True)
        elif truth_label is not None and rgb is not None:
            # A scene model with no label maps of its own publishes slot states only, so the
            # observed panel shows the renderer's labels and the mIoU cell stays empty.
            self.images[1] = self._fit(
                paint_labels(truth_label["class"], truth_label["instance"], rgb), nearest=True)

        if target_label is not None:
            self.images[2] = self._fit(
                paint_labels(target_label["class"], target_label["instance"],
                             target_label["rgb"]), nearest=True)
        if truth_label is not None and rgb is not None:
            self.images[3] = self._fit(
                paint_labels(truth_label["class"], truth_label["instance"], rgb), nearest=True)

    def entries(self):
        return list(zip(PANEL_TITLES, self.images))


class Camera:
    """The free camera, as the page's drag and wheel move it."""

    def __init__(self) -> None:
        self.lookat = list(VIEW_LOOKAT)
        self.distance = VIEW_DISTANCE
        self.azimuth = VIEW_AZIMUTH
        self.elevation = VIEW_ELEVATION

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth = (self.azimuth + float(d_azimuth)) % 360.0
        self.elevation = float(np.clip(self.elevation + float(d_elevation), *ELEVATION_RANGE))

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * float(factor), *DISTANCE_RANGE))

    def reset(self) -> None:
        self.__init__()

    def as_json(self) -> dict:
        return {"distance": round(self.distance, 3), "azimuth": round(self.azimuth, 1),
                "elevation": round(self.elevation, 1)}


class Composer:
    """One renderer plus the overlay: the composed frame the page streams and the video records."""

    def __init__(self, scene, quality: Quality) -> None:
        self.quality = quality
        studio_look(scene.model, quality)
        self.renderer = FreeCameraRenderer(scene.model, width=quality.width,
                                          height=quality.height)
        self.renderer.instance_lut(scene.instance_ids)
        self.overlay = Overlay(quality.width, quality.height)
        self.history: dict[str, list[float]] = {spec.key: [] for spec in READOUTS}
        self.marks: list[float] = []

    @property
    def panel_size(self) -> tuple[int, int]:
        return self.overlay.panel_size(len(PANEL_TITLES))

    def close(self) -> None:
        self.renderer.close()

    def push(self, values: dict) -> None:
        """One vision tick's numbers, for the traces behind the plot."""
        for spec in READOUTS:
            v = values.get(spec.key)
            if v is None or v != v:
                continue
            series = self.history[spec.key]
            series.append(float(v))
            del series[:-TRACE_POINTS]

    def mark_step(self) -> None:
        self.marks.append(float(len(self.history["err_mm"])))
        del self.marks[:-40]

    def readouts(self, values: dict) -> list[Readout]:
        out = []
        n = max(len(self.history["err_mm"]), 1)
        for spec in READOUTS:
            trace = self.history[spec.key]
            value = trace[-1] if trace else None
            hi = spec.hi
            if hi is None:
                hi = max(max(trace, default=0.0) * 1.08, spec.lo + 1e-6)
                if spec.reference is not None:
                    hi = max(hi, spec.reference * 1.15)
            marks = tuple(m / n for m in self.marks) if spec.plot and trace else ()
            out.append(Readout(label=spec.label, value=value, unit=spec.unit, trace=list(trace),
                               lo=spec.lo, hi=hi, fmt=spec.fmt, plot=spec.plot,
                               reference=spec.reference, marks=marks))
        return out

    def render(self, data, camera: Camera) -> np.ndarray:
        frame = self.renderer.render_free(data, camera.lookat, camera.distance,
                                          camera.azimuth, camera.elevation)
        picture = backdrop(frame.rgb, frame.instance)
        return floor_grid(picture, frame.depth, self.renderer.camera_info())

    def compose(self, data, camera: Camera, *, panels, phase: str, values: dict) -> np.ndarray:
        picture = self.render(data, camera)
        composed, _ = self.overlay.draw(
            picture, title=None, subtitle=None, phase=phase or None,
            readouts=self.readouts(values), panels=list(panels), pip=None, pip_label=None)
        return composed


class PlainView:
    """A second main view with no overlay on it -- the twin, beside the first."""

    def __init__(self, scene, width: int, height: int) -> None:
        self.renderer = FreeCameraRenderer(scene.model, width=width, height=height)
        self.renderer.instance_lut(scene.instance_ids)

    def close(self) -> None:
        self.renderer.close()

    def render(self, data, camera: Camera) -> np.ndarray:
        frame = self.renderer.render_free(data, camera.lookat, camera.distance,
                                          camera.azimuth, camera.elevation)
        return backdrop(frame.rgb, frame.instance)


LIVE = Quality("live", 960, 540, 30, msaa=2, shadowsize=1024, crf=20, preset="veryfast",
               export_1080=False, pip_width=240)
"""A smaller main view for a machine that cannot hold the frame rate at 1280x720. Measured on a
16-core machine with no GPU: 1280x720 composes in 76 ms idle and 100 ms with the physics thread
running beside it, 960x540 in 40 ms and 45 ms."""

QUALITIES = ("live", "draft", "hd1080")


def quality(name: str) -> Quality:
    return LIVE if name == "live" else quality_by_name(name)
