"""Acceptance tests for BrickBuilder/PressFromHeld-v0.

The heavy measurements -- the scripted baseline curve over misalignment, the 100-episode
perfect-alignment rate, the PPO run -- are marked ``slow``. What runs in a normal test pass is
the contract: the gymnasium interface, that the actor cannot see the misalignment it is supposed
to be blind to, that a zero residual is exactly the scripted primitive, that the criterion really
does require the correct cell, and that a residual which *does* know the misalignment scores
clearly better (the reward-shaping check spec section 4 asks for).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from brick_builder.envs import obs_spec
from brick_builder.envs.press_env import (
    CELL_TOLERANCE,
    LEVELS,
    SEAT_FRACTION,
    SUCCESS_TICKS,
    PressEnv,
    oracle_action,
    press_force,
    scripted_action,
    scripted_episode,
)
from brick_builder.eval.results import FailureClass
from brick_builder.eval.wilson import wilson_interval

GUIDE_PAGE = Path(__file__).resolve().parents[2] / "docs" / "guide" / "04-environments.md"


def rate(env, episodes, seeds=range(0, 100), action=None):
    """Successes and mean return over a list of seeds."""
    ok, returns = 0, []
    for seed in list(seeds)[:episodes]:
        trace = scripted_episode(env, seed=seed, action=action)
        ok += int(trace["success"])
        returns.append(trace["return"])
    return ok, float(np.mean(returns))


# -------------------------------------------------------------------------- gymnasium contract ---


def test_check_env_passes():
    check_env(PressEnv(horizon=20, dr=False), skip_render_check=True)


def test_gym_make_finds_the_registered_id():
    import gymnasium as gym

    import brick_builder.envs  # noqa: F401 -- registration happens on import

    env = gym.make("BrickBuilder/PressFromHeld-v0")
    assert env.unwrapped.horizon == 200
    env.close()


def test_default_horizon_matches_the_spec_table():
    assert PressEnv(dr=False).horizon == 200


def test_two_hundred_steps_are_bit_for_bit_reproducible_under_a_seed():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (200, 8)).astype(np.float32)
    traces = []
    for _ in range(2):
        env = PressEnv(level=2, seed=7)
        obs, _ = env.reset(seed=7)
        run = [obs["actor"].copy(), obs["critic"].copy()]
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            run.append(obs["actor"].copy())
            run.append(np.array([reward]))
            if terminated or truncated:
                break
        traces.append(run)
        env.close()
    assert len(traces[0]) == len(traces[1])
    for a, b in zip(*traces):
        assert np.array_equal(a, b)


def test_random_policy_never_produces_nans_or_mujoco_warnings():
    env = PressEnv(level=3, horizon=40, seed=0)
    rng = np.random.default_rng(0)
    for ep in range(6):
        obs, _ = env.reset(seed=ep)
        for _ in range(40):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 8).astype(np.float32))
            assert np.all(np.isfinite(obs["actor"])) and np.all(np.isfinite(obs["critic"]))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert not env.scene.warnings(), env.scene.warnings()
    env.close()


@pytest.mark.slow
def test_fifty_random_episodes_stay_clean():
    """The spec's own number: 50 episodes of a random policy with no NaN and no MuJoCo warning."""
    env = PressEnv(level=5, horizon=60, seed=0)
    rng = np.random.default_rng(1)
    for ep in range(50):
        obs, _ = env.reset(seed=1000 + ep)
        for _ in range(60):
            obs, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 8).astype(np.float32))
            assert np.all(np.isfinite(obs["actor"])) and np.all(np.isfinite(obs["critic"]))
            if terminated or truncated:
                break
        assert not env.scene.warnings(), (ep, env.scene.warnings())
    env.close()


# ------------------------------------------------------------------------------- blindness ---


def test_no_actor_entry_has_a_privileged_source():
    env = PressEnv(dr=False)
    obs_spec.check_sources(env.actor_names, scope="actor")
    for name in env.actor_names:
        assert obs_spec.get(name).source != "sim"
    env.close()


def test_the_actor_cannot_see_how_misaligned_it_is():
    """The press is blind by construction: the arm aims at the cell it *believes* is the target,
    and every deployable signal is measured against that belief. If ``press_seat_error`` were
    measured against the true cell instead, the misalignment would be handed straight to the
    policy and the whole task would collapse into a reach."""
    env = PressEnv(level=3, dr=False, misalign=(0.006, 0.0), seed=0)
    layout = env._actor_layout
    errors = []
    for seed in range(6):
        env.reset(seed=seed)
        obs = env._assemble_obs()
        errors.append(obs["actor"][layout.slices["press_seat_error"]])
        assert np.linalg.norm(env._start_error[:2]) == pytest.approx(0.006, abs=1e-9)
    errors = np.asarray(errors)
    # Every episode is 6 mm off in a different direction, yet the lateral part of what the actor
    # sees is the same (sub-millimetre) number every time: the servo tracking error, not the
    # misalignment.
    assert np.max(np.abs(errors[:, :2])) < 0.0015, errors[:, :2]
    assert np.ptp(errors[:, 2]) < 0.0015
    env.close()


def test_the_critic_does_see_the_misalignment_and_the_stud_depths():
    env = PressEnv(level=3, dr=False, misalign=(0.005, 0.0), seed=0)
    env.reset(seed=3)
    obs = env._assemble_obs()
    mis = obs["critic"][env._critic_layout.slices["press_misalign"]]
    assert np.linalg.norm(mis[:2]) == pytest.approx(0.005, abs=5e-4)

    for _ in range(40):
        env.step(np.zeros(8, dtype=np.float32))
    obs = env._assemble_obs()
    depths = obs["critic"][env._critic_layout.slices["press_stud_depths"]]
    truth = env.scene.snap.stud_depths(env.scene.held_name)
    assert np.allclose(depths[: truth.size], truth, atol=1e-6)
    assert np.all(depths[truth.size:] == 0.0)
    env.close()


# --------------------------------------------------------------------- the residual contract ---


def test_zero_residual_reproduces_the_scripted_primitive_exactly():
    """Spec section 1: "setting the residual scale to 0 gives the scripted baseline". With the
    residual accumulating from zero, a zero action leaves the executed target exactly equal to
    the primitive's own -- this checks the commanded pose tick by tick, not just the outcome."""
    env = PressEnv(level=3, dr=False, seed=0)
    env.reset(seed=4)
    zero = scripted_action(env)

    wanted = []
    original = env.primitive_target

    def spy(tick):
        target = original(tick)
        wanted.append(target)
        return target

    env.primitive_target = spy
    for _ in range(60):
        obs, reward, terminated, truncated, _ = env.step(zero)
        expected = wanted[-1]
        assert np.array_equal(env._cmd_echo["pos"], expected.pos)
        assert env._cmd_echo["tilt"] == expected.tilt
        assert env._cmd_echo["spin"] == expected.spin
        assert env._cmd_echo["cvert"] == pytest.approx(expected.c_vert)
        assert env._cmd_echo["clat"] == pytest.approx(expected.c_lat)
        if terminated or truncated:
            break
    assert len(wanted) > 10
    env.close()


def test_a_non_zero_residual_moves_the_executed_target_off_the_primitive():
    env = PressEnv(level=1, dr=False, seed=0)
    env.reset(seed=0)
    action = np.zeros(8, dtype=np.float32)
    action[0] = 1.0
    for _ in range(5):
        env.step(action)
    assert env._res_pos[0] == pytest.approx(5 * 0.002)
    env.close()


# ------------------------------------------------------------------------------ the criterion ---


def test_success_needs_every_stud_deep_enough_and_held_for_five_ticks():
    """What the criterion is made of, exercised on a press that does seat.

    ``misalign=(0.0, 0.0)`` rather than level 0's own +-1 mm: the criterion is what is under
    test here, not whether a blind press survives a millimetre of error it cannot see. It does
    not, most of the time -- that rate is its own test below.
    """
    env = PressEnv(level=0, dr=False, misalign=(0.0, 0.0), seed=0)
    trace = scripted_episode(env, seed=0)
    assert trace["success"]
    stud_h = env.scene.cfg.spec.stud_h
    assert env._min_depth >= SEAT_FRACTION * stud_h
    assert env._streak >= SUCCESS_TICKS

    # One tick inside the band is not enough: the streak has to be rebuilt from scratch.
    env._streak = 0
    for _ in range(SUCCESS_TICKS - 1):
        assert env.success() is False
    assert env.success() is True
    env.close()


def test_seating_one_pitch_over_is_a_wrong_cell_failure_not_a_success():
    """The criterion is "the correct cell", so the same seated brick has to fail once the target
    moves a whole stud pitch away. Perfectly aligned, for the reason the test above gives."""
    env = PressEnv(level=0, dr=False, misalign=(0.0, 0.0), seed=0)
    trace = scripted_episode(env, seed=0)
    assert trace["success"]
    pitch = env.scene.cfg.spec.pitch
    env._cell_xy = env._cell_xy + np.array([pitch, 0.0])
    env._streak = 0
    for _ in range(SUCCESS_TICKS + 2):
        assert env.success() is False
    assert env.diagnose() == FailureClass.WRONG_CELL
    assert CELL_TOLERANCE * pitch < pitch
    env.close()


def test_a_broken_neighbour_connection_terminates_the_episode():
    env = PressEnv(level=4, dr=False, seed=0)
    env.reset(seed=0)
    env.step(np.zeros(8, dtype=np.float32))
    assert env._initial_links, "level 4 puts a brick on the plate, so there is a link to break"
    env._broken = True
    assert env.terminated() is True
    assert env.diagnose() == FailureClass.STRUCTURE_DAMAGE
    assert env.reward() < -5.0, "breaking a connection has to cost the -10 the spec names"
    env.close()


def test_press_force_reads_the_brick_weight_at_rest_and_newtons_while_pressing():
    env = PressEnv(level=0, dr=False, seed=0)
    env.reset(seed=0)
    assert press_force(env.scene) < 0.5, "nothing is being pressed yet"
    for _ in range(60):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(8, dtype=np.float32))
        if terminated or truncated:
            break
    assert env._max_force > 4.0, env._max_force
    assert env._max_force < 25.0, "10 mm of commanded overshoot must not ask for 25 N"
    env.close()


# --------------------------------------------------------------------------- reward shaping ---


def test_an_oracle_residual_scores_clearly_better_than_a_zero_residual():
    """The reward-shaping check. A residual that reads the true misalignment and cancels it must
    score clearly higher than doing nothing at a misalignment the scripted press cannot handle,
    or the reward is not pointing anywhere a learner could follow."""
    env = PressEnv(level=3, dr=False, misalign=(0.004, 0.0), seed=0)
    ok_zero, ret_zero = rate(env, 12, action=None)
    ok_oracle, ret_oracle = rate(env, 12, action=oracle_action)
    env.close()
    # Measured at 4 mm while this was written: scripted 3/8 at +4.87 mean return against the
    # oracle's 5/8 at +11.63, and 10 episodes later 5.25 against 9.17. The margin is real and
    # large but it is a dozen episodes of a bimodal outcome, so the threshold is set where a
    # normal run of luck cannot cross it and a broken reward still would.
    assert ok_oracle > ok_zero, (ok_oracle, ok_zero)
    assert ret_oracle > ret_zero + 2.5, (ret_oracle, ret_zero)


def test_the_progress_potential_pays_for_pushing_studs_in():
    """A seat has to be worth far more than the ticks it costs, and a press that seats nothing
    has to come out behind. Both halves matter: the first says the potential is worth following,
    the second says the time penalty is not being drowned out by shaping noise."""
    env = PressEnv(level=0, dr=False, misalign=(0.0, 0.0), seed=0)
    seated = scripted_episode(env, seed=0)
    env.close()
    assert seated["success"]
    assert seated["return"] > 10.0, seated["return"]

    # Seed 0 at level 0's own +-1 mm does not seat (see the blind-rate test below).
    env = PressEnv(level=0, dr=False, seed=0)
    missed = scripted_episode(env, seed=0)
    env.close()
    assert not missed["success"]
    assert missed["return"] < 0.0, missed["return"]


@pytest.mark.slow
def test_the_blind_press_seats_a_millimetre_of_error_only_sometimes():
    """A documented limitation, not a target.

    Level 0 offsets the brick by +-1 mm in a direction the actor cannot see. The tool now
    settles within about 0.4 mm of its commanded pose (the encoder sits on the output shaft --
    see the servo-model guide), so a blind press has no lateral motion left to find the sockets
    with, and the outcome is bimodal: a clean seat when the error happens to point somewhere the
    chamfers can still catch, and nothing at all otherwise. Measured 3/12 while this was written.
    A learned residual is what is supposed to close this gap; that is the point of the task.
    """
    env = PressEnv(level=0, dr=False, seed=0)
    ok, mean_return = rate(env, 12, seeds=range(12))
    env.close()
    assert 1 <= ok <= 7, f"{ok}/12 seated, mean return {mean_return:+.2f}"


# --------------------------------------------------------------------------- the curriculum ---


def test_the_levels_grow_monotonically_and_end_where_the_spec_says():
    assert len(LEVELS) == 6
    assert LEVELS[0].lateral == pytest.approx(0.001)
    assert LEVELS[0].yaw == pytest.approx(math.radians(2.0))
    assert LEVELS[3].lateral == pytest.approx(0.006)
    assert LEVELS[3].yaw == pytest.approx(math.radians(12.0))
    laterals = [level.lateral for level in LEVELS]
    assert laterals == sorted(laterals)
    assert LEVELS[4].placed and LEVELS[5].placed
    assert any(target[2] == 1 for target in LEVELS[4].targets), "level 4 adds a layer-1 target"
    assert len(LEVELS[5].targets) > len(LEVELS[4].targets), "level 5 adds far-row targets"


def test_set_level_changes_the_drawn_error_and_rebuilds_only_when_the_world_changes():
    env = PressEnv(level=0, dr=False, seed=0)
    model_before = env.scene.model
    env.set_level(3)
    assert env.scene.model is model_before, "levels 0-3 differ only in numbers, not in the world"
    env.reset(seed=0)
    assert np.max(np.abs(env._start_error[:2])) <= LEVELS[3].lateral + 1e-12

    env.set_level(5)
    assert env.scene.model is not model_before, "level 5 puts bricks on the plate"
    env.reset(seed=0)
    assert env.scene.placed_names, env.scene.placed_names
    env.close()


def test_a_self_scheduling_copy_promotes_itself_once_it_clears_the_window():
    """``curriculum=(rate, window)`` is what a vectorized training run uses instead of reaching
    into worker processes: each copy watches its own last ``window`` episodes and steps up on its
    own. Driven here by hand rather than by running episodes, so it stays a fast test."""
    env = PressEnv(level=0, dr=False, curriculum=(0.6, 4), seed=0)
    env.reset(seed=0)
    assert env._level == 0.0
    for _ in range(3):
        env._streak = SUCCESS_TICKS       # pretend the episode that just ran succeeded
        env.reset(seed=0)
        assert env._level == 0.0, "three successes out of a window of four is not enough yet"
    env._streak = SUCCESS_TICKS
    env.reset(seed=0)
    assert env._level == 1.0
    assert env._window == [], "the window restarts after a promotion"

    for _ in range(6):                    # and a copy that keeps failing stays where it is
        env._streak = 0
        env.reset(seed=0)
    assert env._level == 1.0
    env.close()


def test_the_randomization_level_is_zero_at_level_zero_and_one_at_level_five():
    assert PressEnv(level=0).dr_level() == pytest.approx(0.0)
    assert PressEnv(level=5).dr_level() == pytest.approx(1.0)
    assert PressEnv(level=5, dr=False).dr_level() == pytest.approx(0.0)


@pytest.mark.slow
def test_every_level_resets_and_runs_without_warnings():
    for level in range(len(LEVELS)):
        env = PressEnv(level=level, horizon=30, seed=0)
        for seed in range(4):
            env.reset(seed=seed)
            for _ in range(30):
                obs, reward, terminated, truncated, _ = env.step(np.zeros(8, dtype=np.float32))
                if terminated or truncated:
                    break
            assert not env.scene.warnings(), (level, seed, env.scene.warnings())
        env.close()


# ------------------------------------------------------------------------------- the baseline ---


@pytest.mark.slow
@pytest.mark.parametrize("brick", [(2, 4), (2, 2)])
def test_the_scripted_press_seats_a_perfectly_aligned_brick(brick):
    """Spec section 3's acceptance: at least 95 of 100 randomized episodes with perfect
    alignment, for a 2x2 and a 2x4."""
    env = PressEnv(level=0, brick=brick, dr=False, misalign=(0.0, 0.0), seed=0)
    ok, _ = rate(env, 100, seeds=range(100))
    env.close()
    interval = wilson_interval(ok, 100)
    assert ok >= 95, f"{brick}: {ok}/100, Wilson {interval.low:.3f}-{interval.high:.3f}"


@pytest.mark.slow
def test_success_decays_monotonically_with_lateral_misalignment():
    """The baseline curve the learned policy has to beat. Monotone is checked with a tolerance
    of one episode in 30, because two neighbouring points of a 30-episode estimate can cross
    each other by that much on noise alone."""
    rates = []
    for mm in (0.0, 2.0, 4.0, 6.0):
        env = PressEnv(level=3, dr=False, misalign=(mm * 1e-3, 0.0), seed=0)
        ok, _ = rate(env, 30, seeds=range(30))
        env.close()
        rates.append(ok / 30)
    for a, b in zip(rates, rates[1:]):
        assert b <= a + 1 / 30, rates
    assert rates[0] > rates[-1] + 0.5, rates


# ---------------------------------------------------------------------------------- the docs ---


def test_docs_page_matches_the_generated_observation_tables():
    env = PressEnv(dr=False)
    actor_table = obs_spec.markdown_table(env.actor_names)
    critic_table = obs_spec.markdown_table(env.critic_names)
    page = GUIDE_PAGE.read_text()
    assert actor_table in page, "the press actor table in the guide is out of date"
    assert critic_table in page, "the press critic table in the guide is out of date"
    env.close()
