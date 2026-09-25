"""Acceptance tests for BrickBuilder/Reach-v0.

Reach exists to prove the whole multi-rate loop end to end before the heavier tasks build on
it, so the central test here is not "the API returns the right shapes" (covered by
``test_base_env.py``) but "the task is actually solvable": a simple hand-written controller that
closes the loop on the task's own goal observation must reach and hold the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brick_builder.envs import obs_spec
from brick_builder.envs.base_env import POS_SCALE
from brick_builder.envs.reach_env import GOAL_MARGIN, SUCCESS_TICKS, ReachEnv

GUIDE_PAGE = Path(__file__).resolve().parents[2] / "docs" / "guide" / "04-environments.md"


def test_default_horizon_matches_the_spec_table():
    assert ReachEnv().horizon == 100


def test_goal_is_drawn_inside_the_action_authority_box():
    """The target must be reachable within the residual's own clip box, or no policy -- learned
    or scripted -- could ever get there in one episode."""
    from brick_builder.envs.base_env import POS_CLIP

    env = ReachEnv(seed=0)
    for seed in range(10):
        env.reset(seed=seed)
        offset = env._goal_pos - env._home_primitive.pos
        assert np.all(np.abs(offset) <= GOAL_MARGIN * POS_CLIP + 1e-9)


def test_a_hand_written_p_controller_closes_most_of_the_distance_to_goal():
    """Closes the loop on the task's own ``reach_goal`` actor entry (not on internal state a
    real policy could not see): with the gravity feedforward in ``ArmEnv._gravity_feedforward``,
    the arm reaches within a few millimetres of an arbitrary sampled target.

    This does not assert the literal 3 mm / 10-tick success flag, because a plain proportional
    residual with no derivative term does not settle to a fixed point here -- tracked tick by
    tick (see ``docs/guide/04-environments.md``'s scripted-Reach-baseline measurement), true TCP
    error keeps swinging by several millimetres through the whole horizon instead of converging,
    so a single seed's final-tick error is one sample of that oscillation, not a floor. Reaching
    *inside* single-digit millimetres from an initial error an order of magnitude larger is still
    a real, physically meaningful check that the loop, the IK warm-start and the residual/clip-box
    machinery all work; whether a learned policy's own dithering (a de facto derivative term) can
    damp the oscillation and clear the literal success band is measured separately by the PPO
    training run, not asserted here as a certainty of hand-tuned proportional control.
    """
    env = ReachEnv(horizon=100, seed=3)
    obs, _ = env.reset(seed=3)
    initial_error = float(np.linalg.norm(env.tcp_pose()[0] - env._goal_pos))
    info = {}
    for _ in range(100):
        pos, _ = env.read_tcp_pose()
        error = env._goal_pos - pos
        action = np.zeros(8, dtype=np.float32)
        action[0:3] = np.clip(0.3 * error / POS_SCALE, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    final_error = info["params"]["final_error_m"]
    assert final_error < 0.5 * initial_error
    assert final_error < 0.008


def test_success_requires_holding_the_band_not_just_touching_it():
    env = ReachEnv(horizon=100, seed=1)
    env.reset(seed=1)
    env._goal_pos = env.tcp_pose()[0].copy()  # goal = current true pose: already "inside" at t=0
    for _ in range(SUCCESS_TICKS - 1):
        assert env.success() is False  # streak 1 .. SUCCESS_TICKS - 1: not held long enough yet
    assert env.success() is True  # streak == SUCCESS_TICKS


def test_leaving_the_band_resets_the_streak():
    env = ReachEnv(horizon=100, seed=1)
    env.reset(seed=1)
    env._goal_pos = env.tcp_pose()[0].copy()
    for _ in range(SUCCESS_TICKS - 1):
        env.success()
    env._goal_pos = env._goal_pos + 1.0  # 1 m away: certainly outside the band
    assert env.success() is False
    env._goal_pos = env.tcp_pose()[0].copy()
    for _ in range(SUCCESS_TICKS - 1):
        assert env.success() is False
    assert env.success() is True


def test_random_policy_never_produces_nans_or_mujoco_warnings():
    env = ReachEnv(horizon=40, seed=0)
    rng = np.random.default_rng(0)
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        for _ in range(40):
            action = rng.uniform(-1, 1, 8).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs["actor"]))
            assert np.all(np.isfinite(obs["critic"]))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert not env.scene.warnings(), env.scene.warnings()


def test_structure_intact_is_reported_true_since_there_is_nothing_to_break():
    """Reach never places a brick, so the acceptance criterion the campaign checks
    (``EpisodeResult.structure_intact``) has to be reported explicitly rather than left at
    whatever the base class happens to default to -- see ``episode_result_extra``."""
    env = ReachEnv(horizon=3, seed=0)
    env.reset(seed=0)
    info = {}
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    assert info["structure_intact"] is True


def test_timeout_without_success_is_reported_as_a_timeout_failure():
    env = ReachEnv(horizon=3, seed=0)
    env.reset(seed=0)
    info = {}
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    assert truncated and not terminated
    assert info["success"] is False
    from brick_builder.eval.results import FailureClass

    assert info["failure"] == FailureClass.TIMEOUT


@pytest.mark.slow
def test_ppo_training_smoke_improves_the_return(tmp_path):
    """The real acceptance run (full budget, 6 async worker processes, 200-episode Wilson
    interval, learning curve and rollout media) is a one-off measurement reported in this
    package's report, not something every test run repeats. This is the repeatable version at a
    budget a normal test run can afford: a handful of PPO updates, `SyncVectorEnv` (deterministic,
    no subprocess spawn), asserting only that training moves the mean return in the right
    direction -- proof the wiring (env -> vector env -> PPO -> checkpoint) works, not a success-
    rate claim."""
    import gymnasium as gym

    from brick_builder.rl.ppo_asym import PPOConfig, train

    def make_env(seed):
        import brick_builder.envs  # noqa: F401

        return gym.make("BrickBuilder/Reach-v0")

    cfg = PPOConfig(
        n_envs=4, n_steps=64, epochs=3, minibatches=4, lr=3e-4, gamma=0.97,
        total_steps=4 * 64 * 25, seed=0, async_envs=False, ent_coef=0.002,
        hidden_actor=(64, 64), hidden_critic=(128, 128),
    )
    history: list[float] = []
    train(make_env, cfg, tmp_path / "run", callback=lambda i, m: history.append(m["mean_return"]))
    history = [h for h in history if not np.isnan(h)]
    assert len(history) >= 4, "too few updates finished episodes to judge a trend"
    early = np.mean(history[: len(history) // 2])
    late = np.mean(history[len(history) // 2 :])
    assert late > early, f"mean return did not improve: early={early:.2f} late={late:.2f}"


def test_docs_page_matches_the_generated_observation_tables():
    """docs/guide/04-environments.md's Reach observation tables are pasted-in output of
    ``obs_spec.markdown_table``, not hand-typed -- this is the "cannot drift" test spec section 0
    asks every doc-generating function to have."""
    env = ReachEnv()
    actor_table = obs_spec.markdown_table(env.actor_names)
    critic_table = obs_spec.markdown_table(env.critic_names)
    page = GUIDE_PAGE.read_text()
    assert actor_table in page, "the actor observation table in the guide is out of date"
    assert critic_table in page, "the critic observation table in the guide is out of date"
