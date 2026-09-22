"""The environment's contract, and the one equivalence the whole design rests on.

A zero action has to reproduce today's controller exactly -- not approximately, and not only in
the placement it ends with, but in the accumulated correction at every vision tick. If that is
true then an untrained policy is the published baseline and every improvement a trained one shows
is its own.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from gymnasium import spaces

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from brick_builder.rl.vision_control.env import (  # noqa: E402
    CRITIC_W, K_SLOTS, MECH_W, PROGRESS_W, RES, W_SLOT, VisionControlConfig, VisionControlEnv,
)


@pytest.mark.slow
def test_env_checker_accepts_the_environment():
    """`gymnasium.utils.env_checker`, with the rendered frame held out of it.

    `check_env` asserts that two resets with the same seed and the same action give observations
    that are equal to the bit. Everything this environment computes is: the mechanical state, the
    target rows, the progress index, the privileged half and the reward are identical. The wrist
    frame is not -- the renderer's own arithmetic moves a few pixels by one grey level between two
    EGL contexts, which is outside this package and is measured rather than asserted away in
    `test_two_resets_with_one_seed_differ_only_in_the_renderers_last_bit`.
    """
    from gymnasium.utils.env_checker import check_env
    from gymnasium.wrappers import TransformObservation

    inner = VisionControlEnv(cfg=VisionControlConfig(max_ticks=4), seed=0)
    space = spaces.Dict({
        "actor": spaces.Dict({k: v for k, v in inner.observation_space["actor"].items()
                              if k != "image"}),
        "critic": inner.observation_space["critic"],
    })
    env = TransformObservation(
        inner,
        lambda obs: {"actor": {k: v for k, v in obs["actor"].items() if k != "image"},
                     "critic": obs["critic"]},
        observation_space=space,
    )
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.slow
def test_two_resets_with_one_seed_differ_only_in_the_renderers_last_bit():
    env = VisionControlEnv(cfg=VisionControlConfig(max_ticks=4), seed=0)
    try:
        def once():
            obs, _ = env.reset(seed=7)
            stepped, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
            return obs, stepped, reward

        a0, a1, r_a = once()
        b0, b1, r_b = once()
    finally:
        env.close()

    assert r_a == r_b
    for x, y in ((a0, b0), (a1, b1)):
        for key in ("mech", "target", "progress"):
            assert np.array_equal(x["actor"][key], y["actor"][key]), key
        assert np.array_equal(x["critic"], y["critic"])
        frames = (x["actor"]["image"].astype(np.int16), y["actor"]["image"].astype(np.int16))
        assert np.max(np.abs(frames[0] - frames[1])) <= 1


@pytest.mark.slow
def test_the_observation_and_action_spaces_are_what_the_guide_says():
    env = VisionControlEnv(cfg=VisionControlConfig(max_ticks=3), seed=0)
    try:
        obs, info = env.reset(seed=0, options={"world_seed": 10})
        actor = obs["actor"]
        assert actor["image"].shape == (RES, RES, 3) and actor["image"].dtype == np.uint8
        assert actor["mech"].shape == (MECH_W,)
        assert actor["target"].shape == (K_SLOTS, W_SLOT)
        assert actor["progress"].shape == (PROGRESS_W,)
        assert obs["critic"].shape == (CRITIC_W,)
        assert env.action_space.shape == (3,)
        assert obs in env.observation_space
        assert "plate_offset_mm" in info
    finally:
        env.close()


@pytest.mark.slow
def test_the_actor_half_carries_nothing_the_arm_could_not_measure():
    """The plate's real displacement is in the critic half and nowhere in the actor's."""
    env = VisionControlEnv(cfg=VisionControlConfig(max_ticks=3), seed=0)
    try:
        obs, info = env.reset(seed=0, options={"world_seed": 12})
        offset = info["plate_offset_mm"]
        assert offset > 0.0
        flat = np.concatenate([np.asarray(obs["actor"][k], dtype=np.float64).reshape(-1)
                               for k in ("mech", "target", "progress")])
        assert not np.any(np.isclose(flat, offset))
        assert np.any(np.isclose(np.asarray(obs["critic"], dtype=np.float64),
                                 env._scene.cfg.plate_xy[0]))
    finally:
        env.close()


@pytest.mark.slow
def test_a_zero_residual_reproduces_the_analytic_controller_to_1e_9():
    """Seed 0, the whole sequence, every tick's accumulated correction, both ways round."""
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import build_world

    cfg = VisionControlConfig(start_step=None, stop_when_frozen=False, max_ticks=10 ** 6)
    env = VisionControlEnv(cfg=cfg, seed=0)
    through_env = []
    try:
        env.reset(seed=0, options={"world_seed": 0})
        through_env.append(env._episode.correction.copy())
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
            done = terminated or truncated
            through_env.append(env._episode.correction.copy())
        scored = dict(env._done_info)
    finally:
        env.close()

    place = PlaceConfig()
    ideal, nominal = record_ideal(place, seed=0)
    nominal.close()
    scene, _, _ = build_world(0)
    episode = TrackingEpisode(scene, ideal, cfg=place, seed=0)
    direct = [episode.correction.copy()]
    try:
        for rec in episode.run():
            if rec.vision:
                direct.append(episode.correction.copy())
    finally:
        episode.close()

    # `reset` has already run the first vision tick's feedback by the time it returns, so the
    # environment's first reading is the loop's second: `direct[0]` is the zero it started from.
    assert len(through_env) == len(direct)
    assert np.max(np.abs(np.asarray(through_env[:-1]) - np.asarray(direct[1:]))) < 1e-9
    assert np.max(np.abs(np.asarray(direct[0]))) == 0.0
    assert scored["seated"] and scored["at_target_cell"]


@pytest.mark.slow
def test_a_residual_moves_the_correction_and_stays_inside_the_loops_own_ceiling():
    cfg = VisionControlConfig(max_ticks=6)
    env = VisionControlEnv(cfg=cfg, seed=0)
    try:
        env.reset(seed=0, options={"world_seed": 11})
        before = env._episode.correction.copy()
        env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        after = env._episode.correction.copy()
        assert not np.allclose(before, after)
        for _ in range(5):
            env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        assert (np.linalg.norm(env._episode.correction[:2])
                <= env._episode.cfg.max_correction + 1e-12)
    finally:
        env.close()
