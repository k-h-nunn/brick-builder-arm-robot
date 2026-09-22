"""Reward sign checks, on synthetic sequences.

`env.step_reward` is a function of numbers alone -- the tracked error before and after, the action,
whether the progress index is held and whether the damage state rose -- so every sign can be
checked without a simulator. What it deliberately does not cover is the terminal bonus, which
needs a scored placement and is exercised in `test_env.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.rl.vision_control.env import VisionControlConfig, step_reward

CFG = VisionControlConfig()
ZERO = np.zeros(3)


def _r(**kw) -> float:
    args = dict(prev_err_mm=5.0, err_mm=5.0, action=ZERO, holding=False, damage_rise=0,
                scored=False)
    args.update(kw)
    return step_reward(CFG, **args)


def test_closing_the_tracked_error_pays_and_opening_it_costs():
    assert _r(prev_err_mm=5.0, err_mm=3.0) == pytest.approx(2.0 * CFG.w_progress)
    assert _r(prev_err_mm=3.0, err_mm=5.0) == pytest.approx(-2.0 * CFG.w_progress)
    assert _r(prev_err_mm=4.0, err_mm=4.0) == pytest.approx(0.0)


def test_a_sequence_that_converges_earns_the_whole_distance_it_closed():
    errs = [8.0, 6.0, 4.5, 3.0, 1.0]
    total = sum(step_reward(CFG, prev_err_mm=a, err_mm=b, action=ZERO, holding=False,
                            damage_rise=0, scored=False)
                for a, b in zip(errs, errs[1:]))
    assert total == pytest.approx(CFG.w_progress * (errs[0] - errs[-1]))


def test_a_larger_action_always_costs_more():
    small = _r(action=np.array([0.1, 0.0, 0.0]))
    large = _r(action=np.array([1.0, 1.0, 1.0]))
    assert large < small < 0.0
    assert large == pytest.approx(-CFG.w_action)


def test_holding_the_progress_index_costs_a_fixed_amount_per_tick():
    assert _r(holding=True) - _r(holding=False) == pytest.approx(-CFG.w_stall)


def test_damage_costs_once_per_step_of_the_ladder():
    assert _r(damage_rise=1) == pytest.approx(-CFG.w_damage)
    assert _r(damage_rise=2) == pytest.approx(-2.0 * CFG.w_damage)
    assert _r(damage_rise=0) == pytest.approx(0.0)


def test_an_unreadable_tick_pays_no_progress_rather_than_a_guess():
    """A blind tick has no tracked error at all. It must not read as a large improvement."""
    assert _r(prev_err_mm=float("nan"), err_mm=2.0) == pytest.approx(0.0)
    assert _r(prev_err_mm=2.0, err_mm=float("nan")) == pytest.approx(0.0)


def test_the_last_tick_pays_no_progress_because_there_is_no_state_to_compare():
    assert _r(prev_err_mm=9.0, err_mm=1.0, scored=True) == pytest.approx(0.0)
