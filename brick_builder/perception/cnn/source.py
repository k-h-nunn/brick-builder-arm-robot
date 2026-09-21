"""`CnnSceneModel`: the trained network behind the shared `SceneModelSource` interface.

Same contract as every other producer in `brick_builder.perception`:

    state = source.observe(rgb, window=pose)        # -> SceneState, PLACEHOLDER_SLOT_SCHEMA rows
    source.reset()                                  # new episode: forget slot identity

`rgb` is the only input it uses. `instance` and `depth` are accepted and ignored -- a vision model
that read the simulator's ground truth would not be a vision model -- and `present_mask` excludes
pixels from consideration, exactly as it does for the colour-threshold source.

The checkpoint shipped in `brick_builder/perception/weights/` is loaded by default; `device="cuda"` is
available for a machine with a GPU, and changes nothing about the result beyond arithmetic order.
Nothing on the control path decodes anything: the controller reads slot rows. A per-pixel decode
is built (as the ideal source builds one) purely so a viewer or an accuracy check can draw the
masks, and `build_decode=False` switches even that off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..scene_model import SceneState, TemporalSlotAssigner
from .labels import BRICK_CLASS, CLASS_NAMES, FIRST_BRICK_INSTANCE, N_CLASSES, class_map_from_instance
from .model import SceneModelNet, unpack_weights
from .postprocess import (
    GroupingConfig, Prediction, decode_heads, instances_from_heads, slots_from_masks,
)

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "vision_model.pt"
"""The checkpoint that ships with the package, found relative to this file so it works whatever
the working directory is."""

SOURCE_NAME = "conv_net"


@dataclass
class CnnSceneModel:
    """A trained convolutional network publishing `SceneState`.

    `K`/`vanish_after` are the slot contract's own parameters and mean what they mean everywhere
    else. `grouping` controls how the centre heat map and offsets become instances (see
    `postprocess.GroupingConfig`). `threads` caps the intra-operation thread count for this
    process when set -- useful for a measurement, and for a controller that does not want its
    perception to take every core the machine has.
    """

    checkpoint: str | Path | None = None
    device: str = "cpu"
    K: int = 8
    vanish_after: int = 5
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    build_decode: bool = True
    threads: int | None = None

    net: SceneModelNet = field(init=False, repr=False)
    assigner: TemporalSlotAssigner = field(init=False, repr=False)
    last_object_count: int = field(default=0, init=False, repr=False)
    last_kept_count: int = field(default=0, init=False, repr=False)
    last_overflow_count: int = field(default=0, init=False, repr=False)
    last_prediction: Prediction | None = field(default=None, init=False, repr=False)
    """The decoded heads of the most recent `observe`, for a viewer or an evaluation. Never read
    by the control path."""

    last_masks: list = field(default_factory=list, init=False, repr=False)
    last_kinds: list = field(default_factory=list, init=False, repr=False)
    last_slot_of: np.ndarray = field(default=None, init=False, repr=False)
    """Which slot each of `last_masks` landed in (`-1` for one over the cap). Same use as
    `last_prediction`: viewers and accuracy checks, never control."""

    _frame_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.K < 2:
            raise ValueError(f"K must be at least 2 (1 background + >=1 object slot); got {self.K}")
        if self.threads is not None:
            torch.set_num_threads(int(self.threads))
        path = Path(self.checkpoint) if self.checkpoint is not None else DEFAULT_WEIGHTS
        if not path.exists():
            raise FileNotFoundError(
                f"no trained checkpoint at {path}. Train one with "
                "`python scripts/train_vision_model.py`, or pass `checkpoint=` explicitly."
            )
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net = SceneModelNet(width=float(state.get("width", 1.0)))
        self.net.load_state_dict(unpack_weights(state["model"]))
        self.net.eval()
        self.net.to(self.device)
        self.checkpoint = str(path)
        self.assigner = TemporalSlotAssigner(K=self.K, vanish_after=self.vanish_after)

    # -- the contract --------------------------------------------------------------------------

    def reset(self) -> None:
        """Clears slot identity: a new episode should not remember the last one's objects."""
        self.assigner.reset()

    def observe(
        self,
        rgb: np.ndarray,
        *,
        window,
        instance: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        present_mask: np.ndarray | None = None,
        t: float | None = None,
    ) -> SceneState:
        del instance, depth  # this producer never sees ground truth, by construction
        frame01 = _as_float_rgb(rgb)
        res = int(frame01.shape[0])
        if frame01.shape[1] != res:
            raise ValueError(f"rgb must be square; got shape {frame01.shape[:2]}")
        if res % 32:
            raise ValueError(f"rgb side must be a multiple of 32 for this network; got {res}")

        pred = self.predict(frame01)
        if present_mask is not None:
            pm = np.asarray(present_mask, dtype=bool)
            if pm.shape != (res, res):
                raise ValueError(f"present_mask {pm.shape} must be [{res},{res}]")
            pred.classes = np.where(pm, pred.classes, 0)
            pred.centre = np.where(pm, pred.centre, 0.0)

        masks, kinds = instances_from_heads(pred, self.grouping)
        self.last_object_count = len(masks)

        assembly = slots_from_masks(
            masks, frame01, pred.depth, self.assigner, K=self.K, window=window, source=SOURCE_NAME,
            t=float(t) if t is not None else float(self._frame_counter),
            frame_id=self._frame_counter, build_decode=self.build_decode,
        )
        kept = int((assembly.slot_of >= 0).sum())
        self.last_kept_count = kept
        self.last_overflow_count = len(masks) - kept
        self.last_prediction = pred
        self.last_masks = masks
        self.last_kinds = kinds
        self.last_slot_of = assembly.slot_of
        self._frame_counter += 1
        return assembly.state

    # -- the network on its own ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, frame01: np.ndarray) -> Prediction:
        """One forward pass and its decode, without any slot bookkeeping."""
        x = torch.from_numpy(np.ascontiguousarray(frame01.transpose(2, 0, 1)))[None]
        out = self.net(x.to(self.device))
        if self.device != "cpu":
            out = type(out)(
                class_logits=out.class_logits.cpu(), centre_logits=out.centre_logits.cpu(),
                offset=out.offset.cpu(), depth=out.depth.cpu(),
            )
        return decode_heads(out, self.grouping)

    @property
    def n_parameters(self) -> int:
        return self.net.n_parameters()

    # -- viewer API ------------------------------------------------------------------------------
    #
    # Display and measurement only. The control path reads `observe` and nothing else; none of the
    # three methods below is called from `brick_builder.control`.

    def feature_maps(self, rgb: np.ndarray, *, window=None) -> dict:
        """Every head's output for one frame, at the window resolution.

        Runs one `observe`, so the returned `instance_map` and the returned `state` are the same
        frame and the same slot order: `instance_map == k + 1` marks the pixels of slot `k`, and
        `0` marks a pixel no kept object claimed.

        Keys: `class_prob` `[4, H, W]` (`labels.CLASS_NAMES` order), `centre_heatmap` `[H, W]`,
        `offset` `[2, H, W]` in pixels, `depth` `[H, W]` (relative, per-frame normalized),
        `instance_map` `[H, W]` int32, `state` (the `SceneState`), `class_names`.
        """
        state = self.observe(rgb, window=window)
        pred = self.last_prediction
        instance_map = np.zeros(pred.classes.shape, dtype=np.int32)
        for i, mask in enumerate(self.last_masks):
            k = int(self.last_slot_of[i])
            if k >= 0:
                instance_map[mask] = k + 1
        return {
            "class_prob": np.ascontiguousarray(pred.class_prob, dtype=np.float32),
            "centre_heatmap": np.ascontiguousarray(pred.centre, dtype=np.float32),
            "offset": np.ascontiguousarray(pred.offset, dtype=np.float32),
            "depth": np.ascontiguousarray(pred.depth, dtype=np.float32),
            "instance_map": instance_map,
            "class_map": pred.classes.astype(np.int32),
            "state": state,
            "class_names": tuple(CLASS_NAMES),
        }

    @staticmethod
    def colourize_instances(instance_map: np.ndarray, state: SceneState,
                            background: tuple[int, int, int] = (20, 22, 26)) -> np.ndarray:
        """`instance_map` -> `uint8 [H, W, 3]`, each slot in its own published mean colour.

        The colour comes from the slot row, not from the image, so the picture shows what the
        model reports rather than what the pixels happen to look like. Colours are lifted towards
        white so a dark object is still visible, and a one-pixel dark line separates instances.
        """
        instance_map = np.asarray(instance_map)
        out = np.empty((*instance_map.shape, 3), dtype=np.uint8)
        out[:] = np.asarray(background, dtype=np.uint8)
        for k in range(state.K):
            mask = instance_map == k + 1
            if not mask.any():
                continue
            rgb = np.clip((np.asarray(state.colour(k)) + 1.0) * 0.5, 0.0, 1.0)
            out[mask] = np.clip(60.0 + 195.0 * rgb, 0, 255).astype(np.uint8)
        edge = np.zeros(instance_map.shape, dtype=bool)
        edge[:-1] |= instance_map[:-1] != instance_map[1:]
        edge[:, :-1] |= instance_map[:, :-1] != instance_map[:, 1:]
        out[edge] = (10, 11, 13)
        return out

    @staticmethod
    def colourize_scalar(values: np.ndarray, lo: float | None = None,
                         hi: float | None = None) -> np.ndarray:
        """Any scalar map -> `uint8 [H, W, 3]` through a perceptually uniform colour map.

        `lo`/`hi` default to the 2nd and 98th percentile of the finite values, which keeps a
        single outlying pixel from flattening the rest of the picture.
        """
        import cv2

        v = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(v)
        if not finite.any():
            return np.zeros((*v.shape, 3), dtype=np.uint8)
        if lo is None:
            lo = float(np.percentile(v[finite], 2))
        if hi is None:
            hi = float(np.percentile(v[finite], 98))
        if hi <= lo:
            hi = lo + 1e-6
        u = np.zeros(v.shape, dtype=np.float32)
        u[finite] = np.clip((v[finite] - lo) / (hi - lo), 0.0, 1.0)
        bgr = cv2.applyColorMap((u * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def accuracy(self, instance_map: np.ndarray, gt_instance: np.ndarray,
                 gt_class: np.ndarray | None = None, class_map: np.ndarray | None = None) -> dict:
        """Per-frame accuracy against the renderer's labels, cheap enough for every displayed frame.

        `instance_map` is `feature_maps`'s own output; `gt_instance` is the renderer's instance
        map in `labels.py`'s convention and `gt_class` its class map (derived from `gt_instance`
        when not given). `class_map` defaults to the class map of the most recent frame, which is
        the frame `instance_map` came from.

        Returns `miou` (mean over the four classes, ignoring classes absent from both), `brick_
        recall` and `brick_precision` at an overlap of 0.5, `pixel_accuracy`, and the per-class
        overlaps under `iou`.
        """
        instance_map = np.asarray(instance_map)
        gt_instance = np.asarray(gt_instance)
        gt_class = class_map_from_instance(gt_instance) if gt_class is None else np.asarray(gt_class)
        if class_map is None:
            if self.last_prediction is None:
                raise RuntimeError("no frame has been observed yet; call feature_maps first")
            class_map = self.last_prediction.classes
        class_map = np.asarray(class_map)

        iou = {}
        scores = []
        for c, name in enumerate(CLASS_NAMES):
            p, t = class_map == c, gt_class == c
            union = float(np.logical_or(p, t).sum())
            if union == 0.0:
                iou[name] = float("nan")
                continue
            value = float(np.logical_and(p, t).sum()) / union
            iou[name] = value
            scores.append(value)

        pred_ids = [i for i in np.unique(instance_map) if i != 0]
        pred_bricks = [instance_map == i for i in pred_ids
                       if np.bincount(class_map[instance_map == i].ravel(),
                                      minlength=N_CLASSES).argmax() == BRICK_CLASS]
        true_bricks = [gt_instance == i for i in np.unique(gt_instance)
                       if i >= FIRST_BRICK_INSTANCE]
        hits = 0
        taken = set()
        for t in true_bricks:
            best, best_j = 0.0, -1
            for j, p in enumerate(pred_bricks):
                if j in taken:
                    continue
                union = float(np.logical_or(t, p).sum())
                value = float(np.logical_and(t, p).sum()) / union if union else 0.0
                if value > best:
                    best, best_j = value, j
            if best >= 0.5:
                hits += 1
                taken.add(best_j)
        return {
            "miou": float(np.mean(scores)) if scores else float("nan"),
            "iou": iou,
            "brick_recall": hits / len(true_bricks) if true_bricks else float("nan"),
            "brick_precision": hits / len(pred_bricks) if pred_bricks else float("nan"),
            "pixel_accuracy": float((class_map == gt_class).mean()),
            "bricks_true": len(true_bricks),
            "bricks_predicted": len(pred_bricks),
        }


ConvolutionalSceneModel = CnnSceneModel
"""The name this class had before; kept so existing imports keep working."""


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.float32)
    if frame.max(initial=0.0) > 1.0 + 1e-6:
        frame = frame / 255.0
    return np.ascontiguousarray(frame[..., :3])
