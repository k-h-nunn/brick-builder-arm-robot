"""`brick_builder.rl.curriculum`: the success-gated level scheduler the press campaign trains with.

Everything here is arithmetic on the scheduler itself -- no MuJoCo, no environment -- so the
whole module is fast. The one test that does build an environment checks the hook actually
reaches `PressEnv`.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.rl.curriculum import CurriculumConfig, LevelScheduler


def test_rejects_configurations_that_could_never_promote():
    with pytest.raises(ValueError):
        CurriculumConfig(promote_at=0.0)
    with pytest.raises(ValueError):
        CurriculumConfig(promote_at=1.5)
    with pytest.raises(ValueError):
        CurriculumConfig(window=0)
    with pytest.raises(ValueError):
        CurriculumConfig(replay_frac=1.0)
    with pytest.raises(ValueError):
        LevelScheduler(n_levels=0)


def test_promotes_exactly_when_the_window_is_full_and_clears_the_bar():
    sched = LevelScheduler(n_levels=6, cfg=CurriculumConfig(promote_at=0.8, window=10))
    # Nine successes is a rate of 1.0, but the window is not full yet.
    for _ in range(9):
        assert sched.record(0, True) is False
    assert sched.unlocked == 0
    assert sched.record(0, True) is True
    assert sched.unlocked == 1
    assert sched.window_len == 0, "the window restarts after a promotion"

    # 7/10 is below 0.8 and must not promote, however many times it is repeated.
    for _ in range(50):
        for k in range(10):
            sched.record(1, k < 7)
    assert sched.unlocked == 1
    assert sched.window_rate == pytest.approx(0.7)


def test_the_bar_is_a_rate_over_the_window_not_a_streak():
    """Two failures at the end of the window do not reset anything: what counts is the rate
    over the whole window, which is exactly 0.8 here and so promotes on the tenth episode."""
    sched = LevelScheduler(n_levels=3, cfg=CurriculumConfig(promote_at=0.8, window=10))
    pattern = [True] * 8 + [False] * 2
    promoted = [sched.record(0, ok) for ok in pattern]
    assert promoted == [False] * 9 + [True], promoted

    # And a copy that only ever manages a run of nine in ten never gets there, however long
    # it keeps going -- a streak is not what the gate measures.
    slow = LevelScheduler(n_levels=3, cfg=CurriculumConfig(promote_at=0.95, window=10))
    for _ in range(100):
        for k in range(10):
            slow.record(0, k < 9)
    assert slow.unlocked == 0
    assert slow.window_rate == pytest.approx(0.9)


def test_it_stops_at_the_top_rung():
    sched = LevelScheduler(n_levels=3, cfg=CurriculumConfig(promote_at=0.5, window=2))
    for _ in range(100):
        sched.record(sched.next_level(), True)
    assert sched.unlocked == 2
    assert sched.promotions == 2


def test_replay_spends_about_the_configured_share_below_the_unlocked_level():
    sched = LevelScheduler(n_levels=6, cfg=CurriculumConfig(replay_frac=0.2), seed=3)
    sched.unlocked = 4
    levels = np.array([sched.next_level() for _ in range(20_000)])
    replay = float(np.mean(levels < 4))
    assert replay == pytest.approx(0.2, abs=0.02), replay
    assert set(np.unique(levels)) == {0, 1, 2, 3, 4}
    # the replayed levels are drawn uniformly, so each of the four is about a twentieth
    for level in range(4):
        assert float(np.mean(levels == level)) == pytest.approx(0.05, abs=0.01)


def test_replay_episodes_do_not_count_toward_promotion():
    """A copy sitting at level 3 that keeps succeeding on replayed level-0 episodes must not
    be promoted by them -- otherwise the curriculum climbs on work it has already done."""
    sched = LevelScheduler(n_levels=6, cfg=CurriculumConfig(promote_at=0.8, window=10))
    sched.unlocked = 3
    for _ in range(500):
        assert sched.record(0, True) is False
    assert sched.unlocked == 3
    assert sched.window_len == 0
    # but they are still tallied, because that is how we see whether level 0 is still solved
    assert sched.summary()["episodes_per_level"][0] == 500
    assert sched.summary()["successes_per_level"][0] == 500


def test_no_replay_when_the_share_is_zero_or_nothing_has_been_unlocked():
    sched = LevelScheduler(n_levels=6, cfg=CurriculumConfig(replay_frac=0.0), seed=1)
    sched.unlocked = 5
    assert {sched.next_level() for _ in range(500)} == {5}

    fresh = LevelScheduler(n_levels=6, seed=1)
    assert {fresh.next_level() for _ in range(500)} == {0}


def test_two_copies_with_different_seeds_replay_differently_but_each_is_reproducible():
    a = LevelScheduler(n_levels=6, seed=0)
    b = LevelScheduler(n_levels=6, seed=1)
    a.unlocked = b.unlocked = 3
    seq_a = [a.next_level() for _ in range(200)]
    seq_b = [b.next_level() for _ in range(200)]
    assert seq_a != seq_b

    again = LevelScheduler(n_levels=6, seed=0)
    again.unlocked = 3
    assert [again.next_level() for _ in range(200)] == seq_a


def test_summary_is_json_safe_and_counts_what_happened():
    import json

    sched = LevelScheduler(n_levels=4, cfg=CurriculumConfig(promote_at=0.5, window=4))
    for k in range(8):
        sched.record(sched.next_level(), k % 2 == 0)
    summary = sched.summary()
    assert json.loads(json.dumps(summary)) == summary
    assert summary["episodes"] == 8
    assert sum(summary["episodes_per_level"]) == 8
    assert sum(summary["successes_per_level"]) == 4


@pytest.mark.slow
def test_press_env_takes_the_scheduler_and_moves_both_ways():
    """The hook actually reaches the environment: a mapping in `curriculum=` builds a
    scheduler, promotion changes the level the next episode runs at, and replay takes it back
    down again. Slow only because it builds the press scene."""
    from brick_builder.envs.press_env import LEVELS, SUCCESS_TICKS, PressEnv

    env = PressEnv(
        level=0, dr=False, seed=0,
        curriculum={"promote_at": 0.5, "window": 2, "replay_frac": 0.5},
        curriculum_seed=0,
    )
    try:
        assert env._scheduler is not None
        assert env._scheduler.n_levels == len(LEVELS)
        env.reset(seed=0)
        assert env._level == 0.0
        seen = []
        for _ in range(40):
            env._streak = SUCCESS_TICKS         # pretend the episode that just ran succeeded
            env.reset(seed=0)
            seen.append(int(env._level))
        assert env._scheduler.unlocked >= 2, env._scheduler.summary()
        assert any(
            seen[i] < max(seen[:i + 1]) for i in range(1, len(seen))
        ), f"no episode was replayed below the unlocked level: {seen}"
        assert env._start_error is not None
    finally:
        env.close()
