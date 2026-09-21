"""Shared plumbing for `brick_builder/demo/*` and `scripts/eval.py`.

Three things every demo and the evaluator both need: loading a trained checkpoint as a plain
`act_fn(obs) -> action` behind one interface (torch or the exported C actor), rolling out one
episode while rendering a chosen camera every tick, and saving the resulting frames as an mp4 or
a single PNG. Keeping this in one place is what makes a policy demo, a scripted-skill demo and an
evaluation failure-class video all come from the same rendering code rather than three copies of
it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def write_video(frames: list[np.ndarray], path, fps: int = 25) -> Path:
    """Save `frames` (uint8 `[H, W, 3]`, as `SceneRenderer.render(...).rgb` returns) as an mp4."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"no frames to write to {path}")
    # macro_block_size=1: several demos render at sizes (e.g. the 160x120 wrist camera) that are
    # not multiples of ffmpeg's default 16-pixel macroblock, which would otherwise silently resize
    # every frame.
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
    return path


def write_png(frame: np.ndarray, path) -> Path:
    """Save one `[H, W, 3]` uint8 frame as a PNG."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), frame)
    return path


def pick_camera(scene, prefer: tuple[str, ...] = ("overview_cam", "wrist_cam", "grasp_cam")) -> str | None:
    """The first of `prefer` this scene actually has a camera for, else its first camera, else
    `None` (a scene built with `cameras=()`, e.g. `PressFromHeld-v0`'s blind-by-default config)."""
    cams = tuple(scene.cfg.cameras)
    for name in prefer:
        if name in cams:
            return name
    return cams[0] if cams else None


def load_actor_fn(checkpoint, actor_kind: str = "torch", build_dir=None) -> tuple[Callable, dict]:
    """`(act_fn, obs_spec)`: `act_fn(obs_actor)` is the deterministic (mean) action, already
    clamped to the checkpoint's action bounds -- exactly what `brick_builder.rl.ppo_asym.evaluate`
    computes per step -- from either the torch actor (`ppo_asym.load_actor`) or the exported C
    actor (`brick_builder.export.build.build_library` + `brick_builder.export.c_policy.CPolicy`, compiled
    fresh with the system `gcc` into `build_dir`).
    """
    from brick_builder.rl.ppo_asym import load_actor, select_action

    actor, norm, obs_spec = load_actor(checkpoint)
    act_low = np.asarray(obs_spec["act_low"], dtype=np.float32)
    act_high = np.asarray(obs_spec["act_high"], dtype=np.float32)

    if actor_kind == "torch":
        import torch

        def act(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                a, _ = select_action(actor, norm, t, deterministic=True)
            return np.clip(a.squeeze(0).numpy(), act_low, act_high)

        return act, obs_spec

    if actor_kind == "c":
        from brick_builder.export.build import build_library
        from brick_builder.export.c_policy import CPolicy

        build_dir = Path(build_dir) if build_dir is not None else Path(checkpoint).resolve().parent / "_c_build"
        lib_path, _ = build_library(checkpoint, build_dir=build_dir)
        cpolicy = CPolicy(lib_path, obs_spec["actor_dim"], obs_spec["act_dim"])

        def act(obs: np.ndarray) -> np.ndarray:
            return np.clip(cpolicy.act(np.asarray(obs, dtype=np.float32)), act_low, act_high)

        return act, obs_spec

    raise ValueError(f"unknown actor kind {actor_kind!r}; must be 'torch' or 'c'")


def policy_action_fn(act_fn: Callable) -> Callable:
    """Wrap an `act_fn(obs_actor)` (from `load_actor_fn`) as the `action_fn(env)` callable
    `run_episode` wants, reading the actor observation straight off the (unwrapped) env -- the
    same pattern `brick_builder.envs.press_env._torch_policy` already uses internally."""

    def action_fn(env) -> np.ndarray:
        return act_fn(env._assemble_obs()["actor"])

    return action_fn


def run_episode(
    env, action_fn: Callable, seed: int, *, camera: str | None = None, width: int = 480, height: int = 360,
) -> tuple[list[np.ndarray], dict]:
    """Roll out one episode, calling `action_fn(env.unwrapped)` for the action every tick.

    Renders `camera` once before the first action and once after every tick when `camera` is not
    `None`; `camera=None` runs the episode with no rendering at all (the fast path a flat
    evaluation batch wants -- it only needs `info`). Returns `(frames, info)`; `info` is whatever
    the last `env.step` returned, which for a BrickBuilder env carries `episode_result` once the
    episode actually ends.
    """
    from brick_builder.render.renderer import SceneRenderer

    obs, info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    renderer = SceneRenderer(env.unwrapped.scene.model, width=width, height=height) if camera else None
    try:
        if renderer is not None:
            frames.append(renderer.render(env.unwrapped.scene.data, camera).rgb)
        terminated = truncated = False
        while not (terminated or truncated):
            action = action_fn(env.unwrapped)
            obs, reward, terminated, truncated, info = env.step(action)
            if renderer is not None:
                frames.append(renderer.render(env.unwrapped.scene.data, camera).rgb)
        return frames, info
    finally:
        if renderer is not None:
            renderer.close()
