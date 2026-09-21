"""What the per-tick differential IK actually converges to.

The speed of :class:`~brick_builder.envs.base_env.ArmEnv` rests on the per-tick solve landing inside
tolerance in two Gauss-Newton steps, so that the Levenberg-Marquardt solver only runs for targets
that have genuinely jumped. These tests measure that rather than assume it: how often the
fallback fires, what residual the answer is actually left at, and -- the one that matters for the
physics -- that a scripted press still reaches the same joint goals it always did.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from brick_builder.envs.base_env import (
    ACTION_DIM,
    POSE_IK_ANG_ACCEPT,
    POSE_IK_POS_TOL,
    _TOOL_DOWN_ROLL,
)
from brick_builder.robot.kinematics import ArmKinematics, euler_zyx


def _rollout_ik_stats(env, ticks: int, policy: str = "random", seed: int = 0) -> dict:
    """Sum the env's own ``IKStats`` over enough episodes to cover ``ticks`` control ticks."""
    rng = np.random.default_rng(seed)
    fields = ("solves", "iterations", "fallbacks", "failures",
              "pos_error_sum", "ang_error_sum", "yaw_error_sum")
    total = dict.fromkeys(fields, 0.0)
    peak = {"pos_error_max": 0.0, "ang_error_max": 0.0, "yaw_error_max": 0.0}

    def absorb():
        for key in fields:
            total[key] += getattr(env.ik_stats, key)
        for key in peak:
            peak[key] = max(peak[key], getattr(env.ik_stats, key))

    env.reset(seed=seed)
    done = 0
    while done < ticks:
        action = rng.uniform(-1, 1, ACTION_DIM) if policy == "random" else np.zeros(ACTION_DIM)
        _, _, terminated, truncated, _ = env.step(action)
        done += 1
        if terminated or truncated:
            absorb()
            env.reset()
    absorb()
    n = max(total["solves"], 1.0)
    return {
        "ticks": done,
        "iterations_per_solve": total["iterations"] / n,
        "fallback_rate": total["fallbacks"] / n,
        "failure_rate": total["failures"] / n,
        "pos_error_mean": total["pos_error_sum"] / n,
        "ang_error_mean": total["ang_error_sum"] / n,
        "yaw_error_mean": total["yaw_error_sum"] / n,
        **peak,
    }


@pytest.mark.slow
def test_random_policy_rollouts_hardly_ever_need_the_global_solver():
    """20 000 random-policy ticks across Reach and the press.

    A random policy drives the tilt and spin residuals to the edges of their +-20 degree box
    every few ticks, which is far harder than any trained policy or scripted primitive asks for.
    Even there, two damped-least-squares steps put the TCP inside 0.2 mm and the tool axis inside
    2 degrees: measured, Reach never once needs the Levenberg-Marquardt solver, and the press
    needs it on 0.55 % of ticks. Every one of those is a target the global solver cannot reach
    either -- the arm has been walked into a joint limit or off the edge of its workspace -- which
    is what the ``fallbacks == failures`` check says.
    """
    from brick_builder.envs.press_env import PressEnv
    from brick_builder.envs.reach_env import ReachEnv

    reach = _rollout_ik_stats(ReachEnv(seed=0), 12_000)
    press = _rollout_ik_stats(PressEnv(level=2, seed=0), 8_000)
    assert reach["ticks"] + press["ticks"] >= 20_000

    assert reach["fallback_rate"] == 0.0, reach
    assert reach["failure_rate"] == 0.0, reach
    assert reach["pos_error_max"] < POSE_IK_POS_TOL, reach
    assert reach["ang_error_max"] < POSE_IK_ANG_ACCEPT, reach

    assert press["fallback_rate"] <= 0.03, press
    assert press["failure_rate"] == press["fallback_rate"], (
        f"a fallback that then succeeded would mean the differential step gave up too early: {press}"
    )
    for name, stats in (("Reach", reach), ("PressFromHeld", press)):
        assert stats["iterations_per_solve"] <= 2.0, f"{name}: {stats}"
        # Mean over every solve, the handful of genuinely unreachable targets included.
        assert stats["pos_error_mean"] < POSE_IK_POS_TOL, f"{name}: {stats}"


@pytest.mark.slow
def test_the_scripted_press_needs_no_more_than_one_step_per_tick():
    """Zero residual: the target moves so little that most ticks are already inside tolerance."""
    from brick_builder.envs.press_env import PressEnv

    stats = _rollout_ik_stats(PressEnv(level=0, dr=False, seed=0), 1500, policy="zero")
    assert stats["fallback_rate"] == 0.0, stats
    assert stats["failure_rate"] == 0.0, stats
    assert stats["iterations_per_solve"] <= 1.0, stats
    # Nothing conflicts when the tool points straight down, so all six pose numbers are met.
    assert stats["ang_error_mean"] < math.radians(0.01), stats
    assert stats["yaw_error_mean"] < math.radians(0.01), stats


def test_a_tool_down_target_is_solved_in_every_one_of_its_six_numbers():
    """The case that matters: position, tool axis and spin all together.

    Asked for micrometres rather than the per-tick tolerances, so this says what the solver can
    reach, not where it is told to stop.
    """
    from brick_builder.scene.compose import SceneConfig, build_scene

    scene = build_scene(SceneConfig(held=None, placed=[], cameras=()))
    kin = scene.kin
    q0 = scene.home_q.copy()
    home_pos, home_rot = kin.fk(q0)
    home_yaw, _, _ = euler_zyx(home_rot)

    rng = np.random.default_rng(3)
    q = q0.copy()
    worst_pos = worst_ang = worst_yaw = 0.0
    for _ in range(200):
        target = home_pos + rng.uniform(-0.010, 0.010, 3)
        spin = home_yaw + rng.uniform(-0.2, 0.2)
        result = kin.ik_diff(
            target, 0.0, _TOOL_DOWN_ROLL, q, target_yaw=spin, iters=6,
            pos_tol=1e-8, ang_tol=1e-8, yaw_tol=1e-8,
        )
        q = result.q
        pos, rot = kin.fk(q)
        yaw, pitch, roll = euler_zyx(rot)
        worst_pos = max(worst_pos, float(np.linalg.norm(pos - target)))
        worst_ang = max(worst_ang, abs(pitch), abs(abs(roll) - math.pi))
        worst_yaw = max(worst_yaw, abs(math.remainder(yaw - spin, 2.0 * math.pi)))

    assert worst_pos < 1e-6, f"position off by {worst_pos * 1e3:.5f} mm"
    assert math.degrees(worst_ang) < 0.001, f"tool axis off by {math.degrees(worst_ang):.5f} deg"
    assert math.degrees(worst_yaw) < 0.001, f"spin off by {math.degrees(worst_yaw):.5f} deg"


def test_a_tilted_target_gives_up_the_spin_and_not_the_position():
    """When the six requested numbers conflict, the position is the one that must survive.

    Five joints reach a five-dimensional set of poses, so a tilted tool cannot hold its position,
    its axis *and* its spin at once. ``YAW_WEIGHT`` decides where the leftover goes, and this is
    the assertion that it goes into the spin rather than being spread over the other five:
    measured at a 15 degree tilt with a 15 degree spin asked for on top of it, the TCP lands
    0.11 mm from where it was asked to be, inside the per-tick position tolerance, and the tool
    axis inside the bound a usable answer has to meet.
    """
    from brick_builder.scene.compose import SceneConfig, build_scene

    scene = build_scene(SceneConfig(held=None, placed=[], cameras=()))
    kin = scene.kin
    q0 = scene.home_q.copy()
    home_pos, home_rot = kin.fk(q0)
    home_yaw, _, _ = euler_zyx(home_rot)

    tilt = math.radians(15.0)
    spin = home_yaw + math.radians(15.0)
    result = kin.ik_diff(home_pos, tilt, _TOOL_DOWN_ROLL, q0, target_yaw=spin, iters=6)
    pos, _ = kin.fk(result.q)

    assert float(np.linalg.norm(pos - home_pos)) < POSE_IK_POS_TOL
    assert result.ang_error < POSE_IK_ANG_ACCEPT


def test_the_analytic_jacobian_rows_match_finite_differences_of_the_task_error():
    """The claim the whole differential solve rests on.

    The world x and y components of the yaw-pinned angular error have no chain-rule term through
    the target frame's own yaw, so near the solution -- which is where a warm-started per-tick
    solve always is -- their derivative is minus two rows of ``mj_jacSite``, with no finite
    differences and nothing to get wrong. An earlier revision found the local pitch/roll pair
    hard to differentiate correctly for exactly the term this basis does not have, so the claim
    is checked against finite differences of the real residual function rather than trusted.

    ``mj_jacSite`` is an exact derivative of the pose, so what is left over is the usual
    small-angle step in the orientation error itself: at the solution the two agree to 1e-8, and
    away from it the mismatch is bounded by the angular residual that causes it rather than
    growing on its own.
    """
    import mujoco

    from brick_builder.scene.compose import SceneConfig, build_scene

    scene = build_scene(SceneConfig(held=None, placed=[], cameras=()))
    kin = scene.kin
    target, _ = kin.fk(scene.home_q)        # a pose this arm provably reaches, tool straight down

    rng = np.random.default_rng(11)
    eps = 1e-7
    residual = np.empty(5)

    def jacobians(q):
        kin._diff_residual(q, target, 0.0, _TOOL_DOWN_ROLL, residual)
        base = residual.copy()
        mujoco.mj_comPos(kin._e_model, kin._e_data)
        mujoco.mj_jacSite(kin._e_model, kin._e_data, kin._jacp, kin._jacr, kin._e_site)
        analytic = np.vstack([
            -kin._jacp[:, kin._e_dofadr], -kin._jacr[0:2, kin._e_dofadr]
        ])
        numeric = np.zeros((5, 5))
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = eps
            kin._diff_residual(q + dq, target, 0.0, _TOOL_DOWN_ROLL, residual)
            numeric[:, j] = (residual - base) / eps
        return base, analytic, numeric

    solved = kin.ik_diff(
        target, 0.0, _TOOL_DOWN_ROLL, scene.home_q, iters=8, pos_tol=1e-10, ang_tol=1e-10
    ).q

    _, analytic, numeric = jacobians(solved)
    at_solution = float(np.max(np.abs(analytic - numeric)))
    assert at_solution < 1e-6, f"at the solution the analytic Jacobian is off by {at_solution:.2e}"

    for scale in (1e-4, 1e-3, 1e-2, 3e-2):
        for _ in range(5):
            base, analytic, numeric = jacobians(solved + rng.uniform(-scale, scale, 5))
            mismatch = float(np.max(np.abs(analytic - numeric)))
            angular_residual = float(np.max(np.abs(base[3:])))
            assert mismatch < 1e-6 + 1.2 * angular_residual, (angular_residual, mismatch)
