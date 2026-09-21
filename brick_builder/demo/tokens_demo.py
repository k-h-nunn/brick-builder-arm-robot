"""`python -m brick_builder.demo tokens`: the `ObjectToken` readout overlaid on a video.

Runs `BrickBuilder/AlignFromTokens-v0` under the scripted go-to-token controller (zero residual) and
burns the current token's `(x, y, yaw, confidence, age)` -- exactly the `align_token` actor
observation entry -- onto every rendered wrist-camera frame as text, for whichever producer is
asked for (`--producer surrogate|blob`, default the fast ground-truth-with-noise producer used
for training).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import brick_builder.envs  # noqa: F401
from brick_builder.demo.common import write_video
from brick_builder.envs.align_env import AlignFromTokensEnv


def run(record: str | None = None, seed: int = 0, extra=()) -> Path:
    parser = argparse.ArgumentParser(prog="tokens", add_help=False)
    parser.add_argument("--producer", choices=["surrogate", "blob"], default="surrogate")
    args = parser.parse_args(list(extra))

    env = AlignFromTokensEnv(horizon=150, seed=seed, producer=args.producer)
    env.reset(seed=seed)
    zero = np.zeros(8, dtype=np.float32)
    frames = []
    try:
        for _ in range(env.horizon):
            frame = np.ascontiguousarray((env._render_wrist_frame() * 255.0).astype(np.uint8))
            h = frame.shape[0]
            text = (
                f"x={env._token.x:+.3f} y={env._token.y:+.3f} "
                f"yaw={np.degrees(env._token.yaw):+.0f}deg "
                f"conf={env._token.confidence:.2f} age={env._token_age * 1e3:.0f}ms"
            )
            cv2.putText(frame, text, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1, cv2.LINE_AA)
            frames.append(frame)
            _, _, terminated, truncated, _ = env.step(zero)
            if terminated or truncated:
                break
    finally:
        env.close()

    out = write_video(frames, record or "docs/media/demos/tokens.mp4", fps=20)
    print(f"wrote {out} (producer={args.producer})")
    return out
