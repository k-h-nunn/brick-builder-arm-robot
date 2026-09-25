"""`BrickBuilder/SceneTarget-v0`: registration, spaces, the env checker, replay, termination.

Every test uses a temporary target cache, so nothing is written to `data_cache/`.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium as gym  # noqa: E402

import brick_builder.rl.scene_target as st  # noqa: E402
from brick_builder.rl.scene_target.env import (  # noqa: E402
    CRITIC_W, MECH_W, PROGRESS_W, TOOL_W, SceneTargetEnv,
)

ACTOR_KEYS = {"z", "z_depth", "z_valid", "z_matched", "matched_valid", "target", "target_depth",
              "target_valid", "mech", "tool", "feedforward", "progress"}


def make(tmp_path, **kw):
    return gym.make(st.ENV_ID, cache_dir=str(tmp_path), **kw)


def zero(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def run_to_end(env, action_fn=None, max_steps=500):
    info, n = {}, 0
    for n in range(1, max_steps + 1):
        a = zero(env) if action_fn is None else action_fn(env)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            return info, n, term, trunc
    raise AssertionError("episode did not end")


def test_registered_and_unimplemented_parts_refuse_clearly():
    assert st.ENV_ID in gym.envs.registry
    with pytest.raises(NotImplementedError, match="not implemented"):
        SceneTargetEnv(level="align")
    with pytest.raises(NotImplementedError, match="joint"):
        SceneTargetEnv(action_space="joint")
    with pytest.raises(KeyError):
        SceneTargetEnv(level="nonsense")


@pytest.mark.slow
@pytest.mark.parametrize("level,n_act", [("viewing_pose", 4), ("L0b", 3)])
def test_spaces_and_first_observation(tmp_path, level, n_act):
    env = make(tmp_path, level=level)
    try:
        obs, info = env.reset(seed=0, options={"world_seed": 1})
        assert env.action_space.shape == (n_act,)
        assert set(obs["actor"]) == ACTOR_KEYS
        assert set(obs["privileged"]) == {"gt_z", "critic"}
        a = obs["actor"]
        assert a["z"].shape == (8, 14) and a["target"].shape == (5, 8, 14)
        assert a["mech"].shape == (MECH_W,) and a["tool"].shape == (TOOL_W,)
        assert a["feedforward"].shape == (5, 5) and a["progress"].shape == (PROGRESS_W,)
        assert obs["privileged"]["critic"].shape == (CRITIC_W,)
        assert obs in env.observation_space
        for key in ("world_seed", "level", "target_seed", "source", "weights_sha256",
                    "plate_offset_mm", "plate_yaw_deg", "T", "d_mm", "timings"):
            assert key in info
        obs, r, term, trunc, info = env.step(zero(env))
        assert obs in env.observation_space and np.isfinite(r)
        for key in ("k", "T", "step", "holds", "d_mm", "d_abs_mm", "d_rel_mm", "d_missing_mm",
                    "success_repr", "success_gt", "agree", "gt_tool_err_mm", "gt_yaw_err_deg",
                    "exec_action", "transition", "failure"):
            assert key in info
        for key in ("physics_ms", "render_ms", "source_ms", "codec_ms"):
            assert info["timings"][key] >= 0.0
        if level == "viewing_pose":
            assert not np.any(a["feedforward"])
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("level", ["viewing_pose", "viewing_pose_displaced"])
def test_env_checker_with_bit_exact_renders(tmp_path, level):
    """`check_env` with every determinism check on.

    Two settings make a run bit-reproducible: `render_samples=0` (the GPU's 4x multisample
    resolve changes about 200 of 262,144 pixels by one grey level between identical renders)
    and `timings_in_info=False` (wall-clock timings are the one non-deterministic `info`
    entry). `test_default_rendering_differs_only_in_colour_fields` measures what the default
    rendering changes."""
    from gymnasium.utils.env_checker import check_env

    env = make(tmp_path, level=level, render_samples=0, timings_in_info=False)
    try:
        assert env.unwrapped.spec is not None and not env.unwrapped.spec.nondeterministic
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.slow
def test_default_rendering_differs_only_in_colour_fields(tmp_path):
    from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA as S

    env = make(tmp_path, timings_in_info=False)
    try:
        def once():
            obs0, _ = env.reset(seed=4)
            obs1, r, _, _, info = env.step(np.full(env.action_space.shape, 0.5, np.float32))
            return obs0, obs1, r, info["d_mm"]

        a0, a1, ra, da = once()
        b0, b1, rb, db = once()
    finally:
        env.close()
    assert ra == rb and da == db
    colour = np.r_[np.arange(14)[S.fields["colour"]], np.arange(14)[S.fields["colour_spread"]]]
    other = np.setdiff1d(np.arange(14), colour)
    for x, y in ((a0, b0), (a1, b1)):
        for key in ACTOR_KEYS - {"z", "z_matched"}:
            np.testing.assert_array_equal(x["actor"][key], y["actor"][key], err_msg=key)
        for key in ("z", "z_matched"):
            np.testing.assert_array_equal(x["actor"][key][:, other], y["actor"][key][:, other])
            np.testing.assert_allclose(x["actor"][key][:, colour], y["actor"][key][:, colour],
                                       atol=1e-3, err_msg=key)


REPLAY_SEED_3 = (
    "measured: open-loop replay of seed 3 ends 1.19 mm from the target pose in the plane "
    "(tolerance 1.0 mm) when the representation settles; the arm keeps creeping toward the "
    "command at rest (1.02 mm after 0.5 s more, 0.66 mm after 2 s). Seeds 0-2 and 4 end at "
    "0.87-0.91 mm. The servo's open-loop floor sits at the tolerance.")


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, pytest.param(3, marks=pytest.mark.xfail(
    strict=True, reason=REPLAY_SEED_3)), 4])
def test_zero_action_replay_reaches_the_target_on_ground_truth(tmp_path, seed):
    """L0a with the feed-forward on: `u = 0` is open-loop replay of the recorded path in the
    nominal world, and it ends inside the ground-truth tolerance of the target pose."""
    env = make(tmp_path, level="viewing_pose", feedforward=True)
    try:
        env.reset(seed=seed, options={"world_seed": seed})
        info, n, term, trunc = run_to_end(env)
    finally:
        env.close()
    print(f"seed {seed}: T {info['T']}, {n} ticks, xy {info['gt_final_xy_mm']:.2f} mm, "
          f"z {info['gt_final_z_mm']:.2f} mm, yaw {info['gt_final_yaw_deg']:.2f} deg, "
          f"D {info['d_mm']:.2f} mm-eq, representation success {info['success_repr']}, "
          f"failure {info['failure']!r}")
    assert info["success_gt"], info


@pytest.mark.slow
def test_zero_action_without_feedforward_stays_at_the_start_and_stalls(tmp_path):
    env = make(tmp_path, level="viewing_pose")
    try:
        env.reset(seed=0, options={"world_seed": 0})
        start = env.unwrapped.seq.tool_pose[0]
        info, n, term, trunc = run_to_end(env)
        pos, yaw = env.unwrapped._world.tool_pose_truth()
    finally:
        env.close()
    assert term and not trunc
    assert info["failure"] == "stalled with a scene-state error"
    assert not info["success_gt"] and not info["success_repr"]
    assert np.linalg.norm(pos - start[:3]) < 0.002
    assert info["holds"] == 31


@pytest.mark.slow
def test_termination_classes(tmp_path):
    # never settles and never holds: the episode runs to its length bound
    env = make(tmp_path, feedforward=True, tol_mm=0.0, advance_floor_mm=1e6)
    try:
        env.reset(seed=0, options={"world_seed": 0})
        info, n, term, trunc = run_to_end(env)
        assert trunc and not term and info["failure"] == "timeout"
        assert n == env.unwrapped.cfg.max_ticks(info["T"])
    finally:
        env.close()
    # a tube of zero width holds every tick: stalled after max_hold + 1 ticks
    env = make(tmp_path, advance_floor_mm=0.0, tube_scale=0.0, max_hold=3)
    try:
        env.reset(seed=0, options={"world_seed": 0})
        info, n, term, trunc = run_to_end(env)
        assert term and n == 4 and info["failure"] == "stalled with a scene-state error"
    finally:
        env.close()
    # representation success terminates with the success bonus
    env = make(tmp_path, feedforward=True, tol_mm=1e6, advance_floor_mm=1e6)
    try:
        env.reset(seed=0, options={"world_seed": 0})
        info, n, term, trunc = run_to_end(env)
        assert term and info["success_repr"] and info["failure"] == ""
        assert n == info["T"] - 1 + env.unwrapped.cfg.settle_ticks
    finally:
        env.close()


@pytest.mark.slow
def test_zero_action_issues_exactly_the_recorded_commands_on_l0b(tmp_path):
    env = make(tmp_path, level="viewing_pose_displaced")
    inner = env.unwrapped
    try:
        env.reset(seed=0, options={"world_seed": 5})
        issued = []
        world = inner._world
        original = world.command

        def logging_command(tcp, spin, grip, *, c_vert=1.0, c_lat=1.0):
            issued[-1].append(np.array([*np.asarray(tcp), spin, grip, c_vert, c_lat]))
            original(tcp, spin, grip, c_vert=c_vert, c_lat=c_lat)

        world.command = logging_command
        expected = []
        done = False
        while not done:
            seq, k, held = inner.seq, inner.k, inner.holds > 0
            expected.append(np.repeat(seq.commands[k, seq.n_commands[k] - 1][None], 5, axis=0)
                            if held else seq.commands[k])
            issued.append([])
            _, _, term, trunc, info = env.step(zero(env))
            done = term or trunc
        world.command = original
    finally:
        env.close()
    assert len(issued) == len(expected) > 5
    for got, want in zip(issued, expected):
        np.testing.assert_array_equal(np.stack(got), want)
    print(f"L0b seed 5, u = 0: {len(issued)} ticks, failure {info['failure']!r}, "
          f"plate {info['plate_offset_mm']:.2f} mm {info['plate_yaw_deg']:.2f} deg, "
          f"final xy error {info['gt_final_xy_mm']:.2f} mm, D {info['d_mm']:.2f} mm-eq")


@pytest.mark.slow
def test_the_actor_half_carries_no_plate_displacement(tmp_path):
    env = make(tmp_path, level="viewing_pose_displaced")
    try:
        obs, info = env.reset(seed=0, options={"world_seed": 12})
        inner = env.unwrapped
        d_xy, d_yaw = inner.plate_offset, inner.plate_yaw
        secrets = [d_xy[0], d_xy[1], np.linalg.norm(d_xy), 1e3 * np.linalg.norm(d_xy), d_yaw,
                   math.degrees(d_yaw), 1e3 * d_xy[0], 1e3 * d_xy[1]]
        assert info["plate_offset_mm"] > 1.0
        for _ in range(3):
            flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel()
                                   for v in obs["actor"].values()])
            for s in secrets:
                assert not np.any(np.isclose(flat, s, rtol=1e-5, atol=0.0)), s
            critic = obs["privileged"]["critic"].astype(np.float64)
            assert np.isclose(critic[0], d_xy[0], rtol=1e-5) and np.isclose(critic[2], d_yaw, rtol=1e-5)
            obs, *_ = env.step(zero(env))
    finally:
        env.close()


@pytest.mark.slow
def test_memory_stays_flat_over_50_episodes(tmp_path):
    def rss_mb():
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20

    env = make(tmp_path, max_hold=2, settle_control_ticks=5)
    marks = {}
    try:
        for ep in range(50):
            env.reset(seed=ep, options={"world_seed": ep % 5})
            run_to_end(env)
            if ep + 1 in (10, 50):
                marks[ep + 1] = rss_mb()
    finally:
        env.close()
    print(f"RSS after 10 episodes {marks[10]:.0f} MB, after 50 {marks[50]:.0f} MB")
    assert marks[50] - marks[10] < 40.0
