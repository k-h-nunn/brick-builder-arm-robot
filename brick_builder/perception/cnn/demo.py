"""Six-panel figure of the convolutional scene model on one wrist-camera window.

    python -m brick_builder.perception.cnn.demo

Panels: the input frame, the renderer's segmentation, the model's segmentation, and three of the
model's own maps (brick class probability, centre heat map, relative depth). The segmentation
panels are drawn through `source.CnnSceneModel.colourize_instances`, so each object appears in the
mean colour its own slot row publishes; the scalar maps use `colourize_scalar`. No slot ellipses
are drawn -- the picture shows the prediction itself.

Written at least 1600 px wide.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_OUT = Path("docs/media/perception_cnn/demo.png")


def checkpoint_id(path: Path | str) -> str:
    """Eight hex characters of the checkpoint's own hash -- enough to say which weights a figure
    was made with."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]


def build_frame(seed: int = 4):
    """One randomized scene and one wrist-camera window of it, with ground truth."""
    from ...render.renderer import SceneRenderer
    from ...scene.compose import SceneConfig, build_scene, reset_scene, settle
    from .data import (
        RENDER_SIZE, _random_structure, _set_arm, _viewing_pose, jitter_plate,
        perception_instance_lut, randomize_appearance, render_sample, roi_around,
        sample_camera_model, window_for_roi,
    )

    rng = np.random.default_rng(seed)
    placed = _random_structure(rng, touching_same_colour=True)
    scene = build_scene(SceneConfig(held=None, placed=placed, tray=False, cameras=()))
    jitter_plate(scene, rng)
    reset_scene(scene, rng)
    randomize_appearance(scene, rng)
    settle(scene, 0.2)

    import mujoco

    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    plate_xy = scene.model.body_pos[bid][:2].copy()

    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        for _ in range(12):
            q = _viewing_pose(scene, plate_xy, rng)
            if q is None:
                continue
            _set_arm(scene, q)
            diagonal, post = sample_camera_model(rng)
            pose = window_for_roi(
                scene, roi_around([plate_xy[0], plate_xy[1], 0.02], (0.075, 0.075, 0.03)),
                diagonal, None, name="demo")
            if pose is None:
                continue
            sample = render_sample(scene, pose, renderer, post, rng, {"kind": "demo"})
            if (sample.instance >= 3).sum() > 500:
                return sample, pose
        raise RuntimeError("no usable demo view for this seed")
    finally:
        renderer.close()


def ground_truth_instance_image(sample, source) -> np.ndarray:
    """The renderer's instance map, coloured by each object's own mean colour, so the ground-truth
    panel and the predicted panel are drawn by the same rule."""
    from ..ideal_scene_model import IdealSceneModel

    ideal = IdealSceneModel()
    state = ideal.observe(sample.rgb, window=None, instance=sample.instance.astype(np.int32),
                          depth=sample.depth.astype(np.float64))
    slot_of_id = {int(g): k for k, g in enumerate(state.slot_ids) if g != -1}
    instance_map = np.zeros(sample.instance.shape, dtype=np.int32)
    for inst_id, k in slot_of_id.items():
        instance_map[sample.instance == inst_id] = k + 1
    return source.colourize_instances(instance_map, state)


def make_figure(out_path: Path | str = DEFAULT_OUT, seed: int = 4, checkpoint=None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .labels import BRICK_CLASS
    from .source import DEFAULT_WEIGHTS, CnnSceneModel

    sample, pose = build_frame(seed)
    source = CnnSceneModel(checkpoint=checkpoint)
    maps = source.feature_maps(sample.rgb, window=pose)
    scores = source.accuracy(maps["instance_map"], sample.instance, sample.classes)

    panels = [
        ("Arm camera POV", sample.rgb),
        ("Ground truth segmentation", ground_truth_instance_image(sample, source)),
        ("Predicted segmentation", source.colourize_instances(maps["instance_map"], maps["state"])),
        ("Class probability: brick", source.colourize_scalar(maps["class_prob"][BRICK_CLASS], 0.0, 1.0)),
        ("Centre heat-map", source.colourize_scalar(maps["centre_heatmap"], 0.0, 1.0)),
        ("Relative depth", source.colourize_scalar(maps["depth"])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.2), dpi=150)
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
    ident = checkpoint_id(checkpoint or DEFAULT_WEIGHTS)
    fig.suptitle(f"Convolutional scene model - {ident}, mIoU {scores['miou'] * 100:.1f} %",
                 fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.6)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    path = make_figure(args.out, args.seed, args.checkpoint)
    print(path)
    return path


if __name__ == "__main__":
    main()
