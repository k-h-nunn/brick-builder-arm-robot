"""`python -m brick_builder.demo scene`: build the default scene and look at it.

With no `--record`, or `--record out.png`, saves one overview-camera still. With
`--record out.mp4`, instead records a few seconds of the scene settling under gravity with the
arm held at its home pose and nothing commanded -- a demonstration of the *world*
(`brick_builder.scene.compose.build_scene`) rather than of control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from brick_builder.demo.common import write_png, write_video
from brick_builder.render.renderer import SceneRenderer
from brick_builder.scene.compose import OVERVIEW_CAM, SceneConfig, build_scene, reset_scene


def run(record: str | None = None, seed: int = 0, extra=()) -> Path:
    parser = argparse.ArgumentParser(prog="scene", add_help=False)
    parser.add_argument("--seconds", type=float, default=3.0, help="only used for --record *.mp4")
    args = parser.parse_args(list(extra))

    scene = build_scene(SceneConfig(cameras=(OVERVIEW_CAM,)))
    reset_scene(scene, np.random.default_rng(seed))
    renderer = SceneRenderer(scene.model, width=960, height=720)
    try:
        as_video = record is not None and str(record).lower().endswith(".mp4")
        if as_video:
            frames = [renderer.render(scene.data, OVERVIEW_CAM).rgb]
            n_ticks = int(round(args.seconds / 0.020))
            for _ in range(n_ticks):
                scene.step(10)
                frames.append(renderer.render(scene.data, OVERVIEW_CAM).rgb)
            out = write_video(frames, record or "docs/media/demos/scene.mp4", fps=25)
        else:
            frame = renderer.render(scene.data, OVERVIEW_CAM).rgb
            out = write_png(frame, record or "docs/media/demos/scene.png")
    finally:
        renderer.close()
    print(f"wrote {out}")
    return out
