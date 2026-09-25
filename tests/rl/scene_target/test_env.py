"""`BrickBuilder/SceneTarget-v0`: registration, spaces, the env checker, replay, termination.

Every test uses a temporary target cache (and, where one is needed, a calibration fitted into
it), so nothing is written to `data_cache/`.
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
@pytest.mark.parametrize("level,source", [("viewing_pose", "ideal"),
                                          ("viewing_pose_displaced", "ideal"),
                                          ("viewing_pose", "learned")])
def test_env_checker_with_bit_exact_renders(tmp_path, level, source):
    """`check_env` with every determinism check on.

    Two settings make a run bit-reproducible on either source: `render_samples=0` (the GPU's
    4x multisample resolve changes about 50 of 65,536 output pixels between two renders of one
    state) and `timings_in_info=False` (wall-clock timings are then the one non-deterministic
    `info` entry). `test_default_rendering_differs_only_in_colour_fields` measures what the
    default rendering changes on `ideal`; on `learned` it changes every slot field, the reward
    and `d_mm`."""
    from gymnasium.utils.env_checker import check_env

    kw = {"source_kwargs": {"threads": 1}} if source == "learned" else {}
    env = make(tmp_path, level=level, source=source, render_samples=0, timings_in_info=False,
               **kw)
    try:
        assert env.unwrapped.spec is not None and not env.unwrapped.spec.nondeterministic
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.slow
def test_convolutional_source_repeats_exactly_with_bit_exact_renders(tmp_path):
    """With `render_samples=0` two fresh `learned` environments given one seed and one action
    sequence return identical observations, rewards and `info` (timings off)."""
    from gymnasium.utils.env_checker import data_equivalence

    def run():
        env = make(tmp_path, source="learned", source_kwargs={"threads": 1}, render_samples=0,
                   timings_in_info=False)
        rng = np.random.default_rng(99)
        try:
            out = [env.reset(seed=1, options={"world_seed": 2})]
            for _ in range(4):
                out.append(env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)))
        finally:
            env.close()
        return out

    a, b = run(), run()
    for x, y in zip(a, b):
        assert data_equivalence(x, y, exact=True)


@pytest.mark.slow
def test_default_rendering_differs_only_in_colour_fields(tmp_path):
    """Default 4x multisampling on the `ideal` source: only the colour fields change between
    runs, the reward and `d_mm` repeat exactly. Not true on `learned` (see
    `test_env_checker_with_bit_exact_renders`)."""
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


@pytest.fixture(scope="module")
def calibrated_cache(tmp_path_factory):
    """A target cache holding the ground-truth calibration of the viewing-pose family, fitted on
    training seeds 1000-1011 (none of the replay seeds)."""
    from brick_builder.rl.scene_target.bench import calibrate

    root = tmp_path_factory.mktemp("calibrated")
    cal = calibrate("ideal", cache_dir=str(root), seeds=range(1000, 1012), poses_per_seed=5,
                    static_seeds=(1000,), static_ticks=20)
    return root, cal


def record_writes(env):
    """Run to the end with `u = 0`; the five writes of every tick and the last `info`."""
    inner = env.unwrapped
    world = inner._world
    original = world.command
    issued = []

    def logging_command(tcp, spin, grip, *, c_vert=1.0, c_lat=1.0):
        issued[-1].append(np.array([*np.asarray(tcp), spin, grip, c_vert, c_lat]))
        original(tcp, spin, grip, c_vert=c_vert, c_lat=c_lat)

    world.command = logging_command
    try:
        done, info = False, {}
        while not done:
            issued.append([])
            _, _, term, trunc, info = env.step(zero(env))
            done = term or trunc
    finally:
        world.command = original
    return [np.stack(t) for t in issued], info


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_zero_action_replays_the_recording_and_reaches_the_target_on_ground_truth(
        calibrated_cache, seed):
    """L0a, feed-forward on, recording clock, nominal world: `u = 0` issues frame
    `min(tick, T - 1)`'s recorded writes on every tick (open-loop replay), and the arm ends
    inside the ground-truth tolerance of the target pose. Calibrated distance scale."""
    root, cal = calibrated_cache
    env = make(root, level="viewing_pose", feedforward=True, feedforward_clock="recording")
    try:
        _, info0 = env.reset(seed=seed, options={"world_seed": seed})
        assert info0["mm_scale"] == pytest.approx(cal["mm_scale"])
        assert info0["calibration_file"]
        seq = env.unwrapped.seq
        issued, info = record_writes(env)
    finally:
        env.close()
    for tick, got in enumerate(issued):
        np.testing.assert_array_equal(got, seq.commands[min(tick, seq.T - 1)])
    print(f"seed {seed}: T {info['T']}, {len(issued)} ticks, xy {info['gt_final_xy_mm']:.2f} mm, "
          f"z {info['gt_final_z_mm']:.2f} mm, yaw {info['gt_final_yaw_deg']:.2f} deg, "
          f"D {info['d_mm']:.2f} mm, representation success {info['success_repr']}, "
          f"failure {info['failure']!r}, mm_scale {cal['mm_scale']:.1f}")
    assert info["success_gt"], info


@pytest.mark.slow
def test_zero_action_with_the_recording_clock_misses_by_the_displacement_on_l0b(tmp_path):
    """L0b, recording clock: `u = 0` issues exactly the recorded writes and lands where the
    recording goes in the nominal world, off the displaced plate's target by its yaw."""
    env = make(tmp_path, level="viewing_pose_displaced", feedforward_clock="recording")
    try:
        _, info0 = env.reset(seed=0, options={"world_seed": 5})
        seq = env.unwrapped.seq
        issued, info = record_writes(env)
    finally:
        env.close()
    for tick, got in enumerate(issued):
        np.testing.assert_array_equal(got, seq.commands[min(tick, seq.T - 1)])
    print(f"L0b seed 5, recording clock, u = 0: {len(issued)} ticks, failure {info['failure']!r}, "
          f"plate {info0['plate_offset_mm']:.2f} mm {info0['plate_yaw_deg']:.2f} deg, final xy "
          f"{info['gt_final_xy_mm']:.2f} mm, yaw {info['gt_final_yaw_deg']:.2f} deg")
    assert len(issued) >= seq.T and not info["success_gt"]
    assert abs(abs(info["gt_final_yaw_deg"]) - abs(info0["plate_yaw_deg"])) < 0.5
    assert info["gt_final_xy_mm"] > 1.0


def test_the_environment_applies_the_calibration_file(tmp_path):
    from brick_builder.rl.scene_target.codec import Calibration, calibration_path

    cal = Calibration("ideal", "", 14, family="viewing_pose", mm_scale=200.0, sigma_obs_mm=0.4)
    cal.save(calibration_path(tmp_path / "calib", "ideal", "", "viewing_pose"))
    env = SceneTargetEnv(cache_dir=str(tmp_path))
    try:
        assert env.distance.mm_scale == 200.0 and env.sigma_obs_mm == 0.4
        assert env.tol_mm == pytest.approx(0.8) and env.tol_source == "calibration"
    finally:
        env.close()
    bare = SceneTargetEnv(cache_dir=str(tmp_path / "empty"))
    try:
        assert bare.distance.mm_scale == pytest.approx(87.4) and bare.tol_mm == 0.5
        assert bare.tol_source == "default"
    finally:
        bare.close()
    with pytest.raises(ValueError, match="feedforward_clock"):
        SceneTargetEnv(cache_dir=str(tmp_path), feedforward_clock="wall")


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
def test_zero_action_with_the_progress_clock_issues_the_recorded_writes_of_visited_frames(
        tmp_path):
    """L0b, progress clock: advancing ticks issue frame k's writes, held ticks its last write."""
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
@pytest.mark.parametrize("source", ["ideal", "learned"])
def test_rgb_array_render_mode_keeps_the_frame_and_the_dense_decode(tmp_path, source):
    kw = {"source_kwargs": {"threads": 1}} if source == "learned" else {}
    env = make(tmp_path, source=source, render_mode="rgb_array", include_image=True, **kw)
    try:
        obs, _ = env.reset(seed=0, options={"world_seed": 0})
        obs, *_ = env.step(zero(env))
        frame = env.render()
        decode = env.unwrapped.decode
        assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, obs["actor"]["image"])
        assert decode.shape == (256, 256)
        # the decode labels at least the objects the state reports
        assert len(np.unique(decode)) >= int(obs["actor"]["z_valid"].sum()) - 1
    finally:
        env.close()


@pytest.mark.slow
def test_no_ground_truth_pass_leaves_the_ground_truth_fields_empty(tmp_path):
    """`privileged_gt_z=False` on the CNN source skips the ground-truth pass, and the CNN's own
    rows never appear in the ground-truth fields; `PredictionScoring` then reports no error
    against truth. With the default the fields hold the ground-truth pass."""
    from brick_builder.rl.scene_target.wrappers import PredictionScoring

    kw = {"source": "learned", "source_kwargs": {"threads": 1}, "render_samples": 0}
    env = PredictionScoring(make(tmp_path, privileged_gt_z=False, **kw))
    try:
        assert env.unwrapped._gt_source is None and not env.unwrapped.gt_available
        obs, info = env.reset(seed=0, options={"world_seed": 0})
        assert obs in env.observation_space and np.any(obs["actor"]["z_valid"])
        assert not info["gt_z_available"]
        assert not np.any(obs["privileged"]["gt_z"]) and np.all(info["gt_slot_ids"] == -1)
        for _ in range(2):
            obs, _, _, _, info = env.step(zero(env))
            assert obs in env.observation_space
            assert not info["gt_z_available"] and not info["transition"]["gt_z_available"]
            assert not np.any(obs["privileged"]["gt_z"]) and np.all(info["gt_slot_ids"] == -1)
            assert np.all(np.isnan(info["transition"]["gt_z"]))
            assert math.isnan(info["pred_err_vs_truth_mm"])
    finally:
        env.close()

    env = PredictionScoring(make(tmp_path, **kw))
    try:
        assert env.unwrapped._gt_source is not None and env.unwrapped.gt_available
        obs, info = env.reset(seed=0, options={"world_seed": 0})
        assert info["gt_z_available"]
        obs, _, _, _, info = env.step(zero(env))
        assert info["gt_z_available"] and info["transition"]["gt_z_available"]
        gt = obs["privileged"]["gt_z"]
        assert np.any(gt) and np.all(np.isfinite(info["transition"]["gt_z"]))
        assert not np.array_equal(gt, obs["actor"]["z"])
        assert np.any(info["gt_slot_ids"] >= 0)
        assert "pred_err_vs_truth_mm" in info
    finally:
        env.close()


@pytest.mark.slow
def test_ground_truth_source_needs_no_second_pass(tmp_path):
    """On `ideal` the source's rows are the ground truth; `privileged_gt_z` changes nothing."""
    for flag in (True, False):
        env = make(tmp_path, privileged_gt_z=flag, render_samples=0)
        try:
            assert env.unwrapped._gt_source is None and env.unwrapped.gt_available
            obs, info = env.reset(seed=0, options={"world_seed": 0})
            obs, _, _, _, info = env.step(zero(env))
            assert info["gt_z_available"] and info["transition"]["gt_z_available"]
            assert np.array_equal(obs["privileged"]["gt_z"], obs["actor"]["z"])
            assert np.array_equal(info["gt_slot_ids"], info["slot_ids"])
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
