"""Tests for brick_builder/envs/base_env.py: the shared multi-rate tick, action integration, IK
warm-start and the bus-emulator wiring.

Most tests use :class:`ReachEnv` directly rather than inventing a second toy task -- it already
exercises every piece of ``ArmEnv`` (scene, IK, servo bus, bus emulator, domain randomization)
with the smallest possible task logic on top, and a couple of tests use a tiny local subclass
where they need a behaviour Reach's own frozen components hide (an unfrozen grip/compliance).
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from brick_builder.envs import wrappers
from brick_builder.envs.base_env import ANG_CLIP, POS_CLIP, ArmEnv, PrimitiveTarget, measure_step_rate
from brick_builder.envs.reach_env import ReachEnv
from brick_builder.scene.compose import SceneConfig

CUDA_ENV_SET = True  # documents that tests rely on the harness having set CUDA_VISIBLE_DEVICES=""


class _FreeEnv(ArmEnv):
    """A minimal concrete ``ArmEnv`` with nothing frozen, for testing the base machinery
    (action integration, IK, the bus emulator) independent of any task's own reward shaping."""

    frozen = frozenset()

    def __init__(self, *, horizon: int = 20, seed: int | None = None) -> None:
        super().__init__(horizon=horizon, seed=seed)

    def scene_config(self, rng: np.random.Generator) -> SceneConfig:
        return SceneConfig(held=None, placed=[], cameras=())

    def reward(self) -> float:
        return 0.0

    def terminated(self) -> bool:
        return False

    def success(self) -> bool:
        return False


def _run(env, actions):
    obs, _ = env.reset(seed=0)
    trace = [obs]
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        trace.append(obs)
        if terminated or truncated:
            break
    return trace


# --------------------------------------------------------------------------- gymnasium contract ---


def test_check_env_passes_for_reach():
    check_env(ReachEnv(horizon=20), skip_render_check=True)


def test_reset_and_step_match_the_declared_spaces():
    env = ReachEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_gym_make_finds_the_registered_id():
    import gymnasium as gym

    import brick_builder.envs  # noqa: F401 -- registers "BrickBuilder/Reach-v0" as a side effect

    env = gym.make("BrickBuilder/Reach-v0")
    obs, info = env.reset(seed=0)
    assert "actor" in obs and "critic" in obs
    env.close()


# --------------------------------------------------------------------------------- determinism ---


def test_200_step_rollout_is_bit_reproducible_under_a_seed():
    rng = np.random.default_rng(123)
    actions = [rng.uniform(-1, 1, 8).astype(np.float32) for _ in range(200)]

    env_a = ReachEnv(seed=7)
    obs_a, _ = env_a.reset(seed=7)
    trace_a = [obs_a]
    for a in actions:
        obs_a, *_ = env_a.step(a)
        trace_a.append(obs_a)

    env_b = ReachEnv(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    trace_b = [obs_b]
    for a in actions:
        obs_b, *_ = env_b.step(a)
        trace_b.append(obs_b)

    for step, (a, b) in enumerate(zip(trace_a, trace_b)):
        np.testing.assert_array_equal(a["actor"], b["actor"], err_msg=f"actor diverged at step {step}")
        np.testing.assert_array_equal(a["critic"], b["critic"], err_msg=f"critic diverged at step {step}")


def test_different_seeds_give_different_goals():
    e1 = ReachEnv(seed=1)
    e2 = ReachEnv(seed=2)
    e1.reset(seed=1)
    e2.reset(seed=2)
    assert not np.allclose(e1._goal_pos, e2._goal_pos)


# ------------------------------------------------------------------------- action integration ---


def test_zero_action_holds_the_home_pose_when_there_is_no_primitive():
    """Residual structure: with no scripted primitive, zero action forever must leave the
    *executed target* (the IK joint goal handed to the servo bus) exactly at the pose recorded
    at reset -- "zero residual reproduces the scripted baseline" is about the command, not about
    where gravity sag settles the true TCP (a real position servo has steady-state droop under
    load; that is a fact about ``ServoBus``, not something this test is about)."""
    env = _FreeEnv(horizon=15, seed=0)
    env.reset(seed=0)
    home_q = env.scene.home_q.copy()
    zero = np.zeros(8, dtype=np.float32)
    for _ in range(15):
        env.step(zero)
        assert env._res_pos == pytest.approx(np.zeros(3), abs=0.0)
        assert env._res_tilt == 0.0 and env._res_spin == 0.0
    assert np.max(np.abs(env._q_goal - home_q)) < 1e-9


def test_residual_position_stays_inside_the_clip_box():
    env = _FreeEnv(horizon=60, seed=0)
    env.reset(seed=0)
    full_positive = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(60):
        env.step(full_positive)
    assert np.all(np.abs(env._res_pos) <= POS_CLIP + 1e-9)
    # driven fully positive for long enough, the residual must have saturated at the box edge
    assert np.allclose(env._res_pos, POS_CLIP, atol=1e-9)


def test_residual_angle_stays_inside_the_clip_box():
    env = _FreeEnv(horizon=60, seed=0)
    env.reset(seed=0)
    full_positive = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=np.float32)
    for _ in range(60):
        env.step(full_positive)
    assert abs(env._res_tilt) <= ANG_CLIP + 1e-9
    assert abs(env._res_spin) <= ANG_CLIP + 1e-9
    assert env._res_tilt == pytest.approx(ANG_CLIP, abs=1e-9)


def test_reach_freezes_grip_and_compliance():
    env = ReachEnv(horizon=10, seed=0)
    env.reset(seed=0)
    grip0, cv0, cl0 = env._grip_target, env._c_vert, env._c_lat
    extreme = np.array([0, 0, 0, 0, 0, 1, 1, -1], dtype=np.float32)
    for _ in range(10):
        env.step(extreme)
        assert env._grip_target == grip0
        assert env._c_vert == cv0
        assert env._c_lat == cl0


def test_unfrozen_grip_and_compliance_do_move():
    env = _FreeEnv(horizon=5, seed=0)
    env.reset(seed=0)
    grip0 = env._grip_target
    env.step(np.array([0, 0, 0, 0, 0, 1, 1, -1], dtype=np.float32))
    assert env._grip_target != grip0
    assert env._c_vert > 0.9  # action=+1 -> near TORQUE_HI
    assert env._c_lat < 0.4   # action=-1 -> near TORQUE_LO


# ---------------------------------------------------------------------------------------- IK ---


def test_ik_failure_counter_starts_at_zero_and_is_reported_at_episode_end():
    env = ReachEnv(horizon=5, seed=0)
    env.reset(seed=0)
    assert env._ik_failures == 0
    info = {}
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    assert "ik_failures" in info["params"]
    assert info["params"]["ik_failures"] == env._ik_failures


# --------------------------------------------------------------------------------- bus wiring ---


def _with_packet_loss(env, loss: float):
    import dataclasses

    env._bus_emulator.cfg = dataclasses.replace(env._bus_emulator.cfg, packet_loss=loss)


def test_packet_loss_produces_a_flagged_stale_read():
    env = ReachEnv(horizon=5, seed=0)
    env.reset(seed=0)
    _with_packet_loss(env, 1.0)  # force every write/reply to be "lost"
    obs, *_ = env.step(np.zeros(8, dtype=np.float32))
    stale_slice = env._critic_layout.slices["read_stale"]
    write_dropped_slice = env._critic_layout.slices["write_dropped"]
    assert np.all(obs["critic"][stale_slice] == 1.0)
    assert obs["critic"][write_dropped_slice][0] == 1.0


def test_no_packet_loss_means_no_stale_reads():
    env = ReachEnv(horizon=5, seed=0)
    env.reset(seed=0)
    _with_packet_loss(env, 0.0)
    obs, *_ = env.step(np.zeros(8, dtype=np.float32))
    stale_slice = env._critic_layout.slices["read_stale"]
    assert np.all(obs["critic"][stale_slice] == 0.0)


# --------------------------------------------------------------------- domain randomization ---


def test_dr_params_are_nominal_at_level_zero():
    from brick_builder.scene import randomize
    from brick_builder.envs.base_env import DR_PARAM_KEYS

    env = ReachEnv(seed=0)
    env.set_level(0.0)
    obs, _ = env.reset(seed=0)
    dr_slice = env._critic_layout.slices["dr_params"]
    nominal = np.array([randomize.NOMINAL[k] for k in DR_PARAM_KEYS], dtype=np.float32)
    np.testing.assert_allclose(obs["critic"][dr_slice], nominal, rtol=1e-5)


def test_dr_params_stay_within_the_published_ranges_at_full_level():
    from brick_builder.scene import randomize
    from brick_builder.envs.base_env import DR_PARAM_KEYS

    env = ReachEnv(seed=0)
    env.set_level(1.0)
    for seed in range(15):
        obs, _ = env.reset(seed=seed)
        dr_slice = env._critic_layout.slices["dr_params"]
        values = obs["critic"][dr_slice]
        for key, value in zip(DR_PARAM_KEYS, values):
            lo, hi = randomize.RANGES[key]
            assert lo - 1e-9 <= value <= hi + 1e-9, f"{key}={value} outside [{lo}, {hi}]"


# --------------------------------------------------------------------------------- throughput ---


@pytest.mark.slow
def test_measured_step_rate_is_reported():
    env = ReachEnv(seed=0)
    rate = measure_step_rate(env, steps=300, warmup=50)
    assert rate > 20.0  # sanity floor; the real number is reported, not asserted tightly


# -------------------------------------------------------------------------------- wrappers ---


def test_history_stack_wrapper_stacks_k_frames():
    env = wrappers.HistoryStack(ReachEnv(seed=0), k=3)
    obs, _ = env.reset(seed=0)
    inner_dim = ReachEnv().observation_space["actor"].shape[0]
    assert obs["actor"].shape == (3 * inner_dim,)
    # right after reset every one of the k slots is a copy of the same first observation
    first_slot = obs["actor"][:inner_dim]
    last_slot = obs["actor"][-inner_dim:]
    np.testing.assert_array_equal(first_slot, last_slot)


def test_observation_delay_wrapper_lags_by_n_ticks():
    base = ReachEnv(seed=0)
    delayed = wrappers.ObservationDelay(ReachEnv(seed=0), delay_ticks=3)
    base.reset(seed=0)
    delayed.reset(seed=0)
    rng = np.random.default_rng(0)
    undelayed_trace = []
    delayed_trace = []
    for _ in range(6):
        a = rng.uniform(-1, 1, 8).astype(np.float32)
        obs_b, *_ = base.step(a)
        obs_d, *_ = delayed.step(a)
        undelayed_trace.append(obs_b["actor"])
        delayed_trace.append(obs_d["actor"])
    # tick index 5 (0-based) of the delayed stream must equal tick index 2 of the plain stream
    np.testing.assert_array_equal(delayed_trace[5], undelayed_trace[2])


def test_action_repeat_sums_reward_over_the_repeat_window():
    plain = ReachEnv(seed=0)
    repeated = wrappers.ActionRepeatLatency(ReachEnv(seed=0), repeat=4)
    plain.reset(seed=0)
    repeated.reset(seed=0)
    action = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    total = 0.0
    for _ in range(4):
        _, r, terminated, truncated, _ = plain.step(action)
        total += r
        if terminated or truncated:
            break
    _, r_rep, _, _, _ = repeated.step(action)
    assert r_rep == pytest.approx(total, rel=1e-5)


def test_episode_stats_wrapper_reports_on_episode_end():
    env = wrappers.EpisodeStats(ReachEnv(horizon=5, seed=0))
    env.reset(seed=0)
    info = {}
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
        if terminated or truncated:
            break
    assert "episode" in info
    assert info["episode"]["l"] >= 1
    assert isinstance(info["episode"]["success"], bool)


# ------------------------------------------------------------------------------------ misc ---


def test_close_does_not_raise():
    env = ReachEnv(seed=0)
    env.reset(seed=0)
    env.close()


def test_primitive_target_dataclass_holds_the_fields_it_promises():
    pt = PrimitiveTarget(pos=np.zeros(3), tilt=0.0, spin=0.0, grip=0.0)
    assert pt.c_vert == 1.0 and pt.c_lat == 1.0
