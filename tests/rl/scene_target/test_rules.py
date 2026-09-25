"""Progress index, reward signs and ground-truth pose error, on synthetic numbers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brick_builder.rl.scene_target.config import RewardWeights
from brick_builder.rl.scene_target.rules import (
    GT_TOL_XY_MM, ProgressState, advance, pose_error, progress_update, step_reward,
    target_in_plate_frame, terminal_reward, tube_width,
)

W = RewardWeights()


# ---------------------------------------------------------------------------- progress ---


def place_rule(errs, travel_mm, *, T, advance_mm=1.0, tube_scale=1.6, tol_mm=0.4, settle=3):
    """The index logic of `action.place.TrackingEpisode.run` for finite errors, written out:

        tube = max(cfg.advance_mm, floor, cfg.tube_scale * 1e3 * frame.travel)
        if err_mm <= tube:
            if last: settled += 1 if err_mm <= cfg.tol_mm else reset; break at settle_ticks
            else: k += 1
            holds = 0
        else: settled = 0; holds += 1
    """
    k = holds = settled = 0
    trace = []
    for err in errs:
        tube = max(advance_mm, advance_mm, tube_scale * travel_mm[k])
        last = k == T - 1
        if err <= tube:
            if last:
                settled = settled + 1 if err <= tol_mm else 0
            else:
                k += 1
            holds = 0
        else:
            settled = 0
            holds += 1
        trace.append((k, holds, settled))
        if last and settled >= settle:
            break
    return trace


@pytest.mark.parametrize("seed", range(20))
def test_progress_update_matches_the_tracking_loop_rule(seed):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(5, 30))
    travel = rng.uniform(0.0, 3.0, size=T)
    travel[-1] = 0.0
    errs = rng.gamma(2.0, 0.6, size=400)
    expected = place_rule(errs, travel, T=T)
    state = ProgressState(0)
    got = []
    for err in errs[: len(expected)]:
        tube = tube_width(travel[state.k], floor_mm=1.0, sigma_obs_mm=0.0, tube_scale=1.6)
        state, _, _ = progress_update(state, T=T, d_mm=float(err), tube_mm=tube, tol_mm=0.4)
        got.append((state.k, state.holds, state.settled))
    assert got == expected


def test_index_never_runs_ahead_by_more_than_one_frame_per_tick():
    state = ProgressState(0)
    for _ in range(50):
        new, advanced, held = progress_update(state, T=10, d_mm=0.0, tube_mm=1.0, tol_mm=0.5)
        assert new.k - state.k in (0, 1)
        assert not (advanced and held)
        state = new
    assert state.k == 9


def test_non_finite_distance_holds():
    assert not advance(float("nan"), ref_step_mm=5.0)
    s, adv, held = progress_update(ProgressState(2), T=10, d_mm=float("nan"), tube_mm=1.0, tol_mm=0.5)
    assert s.k == 2 and held and not adv


def test_tube_width_takes_the_largest_bound():
    assert tube_width(0.1) == 1.0
    assert tube_width(0.1, sigma_obs_mm=0.5) == pytest.approx(1.5)
    assert tube_width(2.0) == pytest.approx(3.2)
    assert advance(3.2, ref_step_mm=2.0) and not advance(3.21, ref_step_mm=2.0)


def test_last_frame_settles_only_inside_tolerance():
    s = ProgressState(4)
    for _ in range(2):
        s, _, _ = progress_update(s, T=5, d_mm=0.3, tube_mm=1.0, tol_mm=0.5)
    assert s.settled == 2
    s, _, held = progress_update(s, T=5, d_mm=0.8, tube_mm=1.0, tol_mm=0.5)
    assert s.settled == 0 and not held and s.holds == 0
    s, _, held = progress_update(s, T=5, d_mm=1.5, tube_mm=1.0, tol_mm=0.5)
    assert held and s.holds == 1


# ------------------------------------------------------------------------------ reward ---


def r(**kw):
    args = dict(d_prev=3.0, d_next=3.0, advanced=False, u=np.zeros(3), u_prev=np.zeros(3),
                held=False, damage_rise=0, tol_mm=0.5)
    args.update(kw)
    return step_reward(W, **args)


def test_closing_the_distance_pays_and_opening_it_costs():
    assert r(d_prev=3.0, d_next=2.0) - r(d_prev=3.0, d_next=3.0) > 0
    assert r(d_prev=2.0, d_next=3.0) < r(d_prev=3.0, d_next=3.0)
    assert r(d_prev=3.0, d_next=2.0) - r(d_prev=2.0, d_next=2.0) == pytest.approx(W.w_p)


def test_tracking_term_is_capped():
    assert r(d_prev=100.0, d_next=100.0) == pytest.approx(-5 * W.w_d)
    assert r(d_prev=0.0, d_next=0.0) == pytest.approx(0.0)


def test_advance_pays_hold_costs():
    base = r()
    assert r(advanced=True) - base == pytest.approx(W.w_adv)
    assert r(held=True) - base == pytest.approx(-W.w_h)


def test_effort_smoothness_and_damage_cost():
    base = r()
    assert r(u=np.ones(3), u_prev=np.ones(3)) - base == pytest.approx(-W.w_u)
    assert r(u=np.ones(3)) - base == pytest.approx(-W.w_u - W.w_s)
    assert r(damage_rise=2) - base == pytest.approx(-2 * W.w_dmg)
    assert r(damage_rise=-1) == base


def test_terminal_reward_classes():
    assert terminal_reward(W, success=True, failure="") == pytest.approx(W.success)
    assert terminal_reward(W, success=True, failure="", lateral_mm=0.25) == pytest.approx(
        W.success + W.lateral * 0.75)
    for failure in ("stalled with a scene-state error", "stalled blind", "lost view", "damage"):
        assert terminal_reward(W, success=False, failure=failure) == pytest.approx(-W.failure)
    assert terminal_reward(W, success=False, failure="timeout") == 0.0


# -------------------------------------------------------------------------- ground truth ---


def test_pose_error_in_the_nominal_frame():
    e = pose_error((0.1712, 0.0, 0.0605), 0.0, (0.170, 0.0, 0.060, 0.0))
    assert e.xy_mm == pytest.approx(1.2) and e.z_mm == pytest.approx(0.5)
    assert e.xy_mm > GT_TOL_XY_MM and not e.success()
    assert pose_error((0.1705, 0.0, 0.060), math.radians(0.5), (0.170, 0.0, 0.060, 0.0)).success()


def test_target_follows_the_displaced_plate():
    c = (0.17, 0.0)
    target = (0.20, 0.0, 0.06, 0.1)
    yaw = math.radians(90.0)
    t = target_in_plate_frame(target, c, (0.005, 0.0), yaw)
    np.testing.assert_allclose(t, [0.175, 0.03, 0.06, 0.1 + yaw], atol=1e-12)
    e = pose_error(t[:3], t[3], target, plate_nominal_xy=c, plate_offset_xy=(0.005, 0.0),
                   plate_yaw=yaw)
    assert e.xy_mm == pytest.approx(0.0, abs=1e-9) and e.success()
    assert pose_error(target[:3], target[3], target, plate_nominal_xy=c,
                      plate_offset_xy=(0.005, 0.0), plate_yaw=0.0).xy_mm == pytest.approx(5.0)
