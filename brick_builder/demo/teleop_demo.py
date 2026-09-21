"""`python -m brick_builder.demo teleop`: jog the TCP through `BrickBuilder/Reach-v0`'s own action space.

Keyboard teleop needs a real display and an OpenCV build with GUI support. This project depends
on `opencv-python-headless` (`pyproject.toml`), which has no GUI backend at all, and every demo
has to run headless with `--record` regardless -- so the path this demo actually exercises,
always, is the scripted jog: a smooth trajectory through the same
`[dx, dy, dz, d_tilt, d_spin, d_grip, c_vert, c_lat]` action every BrickBuilder task takes. Pass
`--interactive` to *attempt* a live keyboard-controlled window instead of recording anything; it
falls back to the scripted jog (printing why) whenever that attempt fails, which it always does on
this project's own dependencies.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

import brick_builder.envs  # noqa: F401 -- registers BrickBuilder/* gymnasium ids
from brick_builder.demo.common import pick_camera, write_video
from brick_builder.render.renderer import SceneRenderer

CONTROL_DT = 0.020


def _scripted_action(t: float) -> np.ndarray:
    """A gentle circle in x/y plus a slow tilt/spin wag -- well inside the residual box `ArmEnv`
    clips to (+-12 mm, +-20 deg): a *jog*, not a target-seeking controller."""
    a = np.zeros(8, dtype=np.float32)
    a[0] = math.sin(2 * math.pi * 0.25 * t)
    a[1] = math.cos(2 * math.pi * 0.25 * t)
    a[2] = 0.4 * math.sin(2 * math.pi * 0.10 * t)
    a[3] = 0.3 * math.sin(2 * math.pi * 0.15 * t)
    a[4] = 0.3 * math.cos(2 * math.pi * 0.15 * t)
    return a


def _gui_available() -> bool:
    try:
        import cv2

        cv2.namedWindow("brick-builder-teleop-probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("brick-builder-teleop-probe")
        return True
    except Exception:
        return False


def run(record: str | None = None, seed: int = 0, extra=()):
    parser = argparse.ArgumentParser(prog="teleop", add_help=False)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(list(extra))

    env = gym.make("BrickBuilder/Reach-v0")
    if args.interactive and record is None:
        if _gui_available():
            return _run_interactive(env, seed)
        print("no GUI-capable display found (opencv-python-headless has no imshow backend); "
              "falling back to the scripted jog")
    return _run_scripted(env, seed, args.seconds, record)


def _run_scripted(env, seed: int, seconds: float, record: str | None) -> Path:
    scene = env.unwrapped.scene
    camera = pick_camera(scene)
    renderer = SceneRenderer(scene.model, width=640, height=480)
    frames = []
    try:
        env.reset(seed=seed)
        frames.append(renderer.render(scene.data, camera).rgb)
        n_ticks = int(round(seconds / CONTROL_DT))
        for k in range(n_ticks):
            t = k * CONTROL_DT
            _, _, terminated, truncated, _ = env.step(_scripted_action(t))
            frames.append(renderer.render(scene.data, camera).rgb)
            if terminated or truncated:
                env.reset(seed=seed + 1)
    finally:
        renderer.close()
        env.close()
    out = write_video(frames, record or "docs/media/demos/teleop.mp4", fps=25)
    print(f"wrote {out} (scripted jog)")
    return out


def _run_interactive(env, seed: int) -> None:
    import cv2

    scene = env.unwrapped.scene
    camera = pick_camera(scene)
    renderer = SceneRenderer(scene.model, width=640, height=480)
    # (dx, dy, dz, dtilt, dspin) per key press.
    key_map = {
        ord("w"): (0, 0, 1, 0, 0), ord("s"): (0, 0, -1, 0, 0),
        ord("a"): (-1, 0, 0, 0, 0), ord("d"): (1, 0, 0, 0, 0),
        ord("q"): (0, 1, 0, 0, 0), ord("e"): (0, -1, 0, 0, 0),
        ord("j"): (0, 0, 0, 0, -1), ord("l"): (0, 0, 0, 0, 1),
    }
    env.reset(seed=seed)
    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # Esc
                break
            dx, dy, dz, dtilt, dspin = key_map.get(key, (0, 0, 0, 0, 0))
            action = np.array([dx, dy, dz, dtilt, dspin, 0, 0, 0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            frame = renderer.render(scene.data, camera).rgb
            cv2.imshow("brick_builder teleop (Esc to quit)", frame[:, :, ::-1])
            if terminated or truncated:
                env.reset()
    finally:
        renderer.close()
        cv2.destroyAllWindows()
        env.close()
