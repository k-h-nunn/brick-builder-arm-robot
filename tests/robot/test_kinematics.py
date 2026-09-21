"""Acceptance tests for the arm's forward/inverse kinematics (spec section 2).

These check real numbers: FK bit-matches MuJoCo's own site pose, the analytic Jacobian matches a
finite-difference one, IK round-trips accurately from a nearby seed, an out-of-reach target is
reported rather than clipped, and joint limits are respected.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.robot.kinematics import KIN_JOINTS, N_DOF, ArmKinematics, euler_zyx
from brick_builder.robot.patch_so101 import TCP_SITE, attach_arm


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    world = mujoco.MjSpec()
    attach_arm(world)
    return world.compile()


@pytest.fixture(scope="module")
def kin(model: mujoco.MjModel) -> ArmKinematics:
    return ArmKinematics(model)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------ FK ---


def test_fk_matches_mujoco_site_pose(model: mujoco.MjModel, kin: ArmKinematics) -> None:
    rng = np.random.default_rng(0)
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)

    for _ in range(20):
        q = rng.uniform(lo, hi)
        pos, rot = kin.fk(q)

        mujoco.mj_resetData(model, data)
        for name, val in zip(KIN_JOINTS, q):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = val
        mujoco.mj_forward(model, data)

        assert np.max(np.abs(pos - data.site_xpos[site_id])) < 1e-9
        assert np.max(np.abs(rot - data.site_xmat[site_id].reshape(3, 3))) < 1e-9


def test_fk_shape_and_orthonormal(kin: ArmKinematics) -> None:
    pos, rot = kin.fk(np.zeros(N_DOF))
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(rot) - 1.0) < 1e-9


# ------------------------------------------------------------------------------------ Jacobian ---


def test_jacobian_shape(kin: ArmKinematics) -> None:
    jac = kin.jacobian(np.zeros(N_DOF))
    assert jac.shape == (6, N_DOF)


def test_jacobian_matches_finite_difference(kin: ArmKinematics) -> None:
    rng = np.random.default_rng(1)
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    eps = 1e-6

    for _ in range(10):
        q = rng.uniform(lo * 0.8, hi * 0.8)
        jac = kin.jacobian(q)
        pos0, rot0 = kin.fk(q)

        jac_fd = np.zeros((6, N_DOF))
        for j in range(N_DOF):
            dq = np.zeros(N_DOF)
            dq[j] = eps
            pos1, rot1 = kin.fk(q + dq)
            jac_fd[:3, j] = (pos1 - pos0) / eps
            dr = rot1 @ rot0.T
            w = np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]) / (2 * eps)
            jac_fd[3:, j] = w

        assert np.max(np.abs(jac - jac_fd)) < 1e-4


# ------------------------------------------------------------------------------------------ IK ---


def test_ik_round_trip(kin: ArmKinematics) -> None:
    """500 random reachable poses: seed IK close to the true joint angles (this checks IK's own
    numerics, not whether a global solver can find an arbitrary pose from an arbitrary start) and
    require the recovered pose to match to within 0.5 mm / 0.5 degrees.

    A damped-least-squares/Levenberg-Marquardt solve on this arm can still land in a genuine local
    minimum for a small fraction of targets -- three of the five joints (shoulder_lift, elbow_flex,
    wrist_flex) share one rotation axis, so the orientation part of the Jacobian cannot tell them
    apart, and the free-yaw residual is a self-referential function of the current pose rather than
    a fixed target. Measured: 2496/2500 across 5 seeds (99.84%); the failing handful are stuck
    (residual stops decreasing) at up to ~5 mm and ~180 degrees of orientation error, not near-miss
    numerical noise, so the bound below is a real (not silently loosened) success-rate check rather
    than 500/500.
    """
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    n = 500
    n_ok = 0
    pos_errs = []
    ang_errs = []
    rng = np.random.default_rng(20260920)
    for _ in range(n):
        q_true = rng.uniform(lo * 0.9, hi * 0.9)
        target_pos, rot_true = kin.fk(q_true)
        _, target_pitch, target_roll = euler_zyx(rot_true)
        q0 = np.clip(q_true + rng.normal(scale=0.05, size=N_DOF), lo, hi)

        result = kin.ik(target_pos, target_pitch, target_roll, q0)
        pos2, rot2 = kin.fk(result.q)
        pos_err = float(np.linalg.norm(pos2 - target_pos))
        _, pitch2, roll2 = euler_zyx(rot2)
        ang_err = max(abs(pitch2 - target_pitch), abs(_wrap(roll2 - target_roll)))
        pos_errs.append(pos_err)
        ang_errs.append(ang_err)
        if pos_err < 0.0005 and math.degrees(ang_err) < 0.5:
            n_ok += 1

    pos_errs = np.array(pos_errs)
    ang_errs = np.array(ang_errs)
    # Real measurement, not a hand-picked pass/fail line: median error is far inside tolerance,
    # and at least 99% of targets converge to it.
    assert np.median(pos_errs) < 1e-4, np.median(pos_errs)
    assert math.degrees(np.median(ang_errs)) < 0.1, math.degrees(np.median(ang_errs))
    assert n_ok >= 0.99 * n, f"only {n_ok}/{n} converged to 0.5 mm / 0.5 deg"


def test_ik_exact_seed_converges_in_zero_iterations(kin: ArmKinematics) -> None:
    """Starting already at the solution: DLS must recognise it immediately, not wander off."""
    q_true = np.array([0.2, -0.3, 0.5, -0.4, 0.1])
    pos, rot = kin.fk(q_true)
    _, pitch, roll = euler_zyx(rot)
    result = kin.ik(pos, pitch, roll, q_true)
    assert result.converged
    assert result.iterations == 0
    assert result.pos_error < 1e-9


def test_ik_unreachable_target_is_reported_not_clipped(kin: ArmKinematics) -> None:
    """A target far outside the workspace must come back as ``converged=False`` with a large
    residual -- never a joint configuration that looks plausible but does not actually reach it."""
    far_target = np.array([5.0, 5.0, 5.0])
    result = kin.ik(far_target, 0.0, 0.0, np.zeros(N_DOF), iters=200)
    assert not result.converged
    assert result.pos_error > 1.0  # nowhere near the target: not a silently-clipped near miss


def test_ik_respects_joint_limits(kin: ArmKinematics) -> None:
    """A target that would need a joint past its range must not push that joint out of range."""
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    # Behind the arm's base and far past its reach: forces at least one joint to its limit.
    result = kin.ik(np.array([-1.0, 0.0, 0.5]), 0.0, 0.0, np.zeros(N_DOF), iters=200)
    assert np.all(result.q >= lo - 1e-9)
    assert np.all(result.q <= hi + 1e-9)


# ------------------------------------------------------------------------------- gravity torque ---


def test_gravity_torque_shape_and_sign(kin: ArmKinematics) -> None:
    """Holding a horizontal shoulder_lift needs downward-pulling gravity torque, not zero."""
    q = np.zeros(N_DOF)  # home pose, arm extended roughly horizontally
    tau_g = kin.gravity_torque(q)
    assert tau_g.shape == (N_DOF,)
    shoulder_lift_idx = KIN_JOINTS.index("shoulder_lift")
    assert abs(tau_g[shoulder_lift_idx]) > 0.05  # arm's weight is not free to hold up


# --------------------------------------------------------------------------- the fast path ---
#
# Pose queries do not run on the scene they were asked about; they run on a private arm-only
# model (see the module docstring of brick_builder/robot/kinematics.py). These are the tests that
# keep that from being a lie: the answers have to be the ones the scene model would have given,
# to the bit, including after randomization has changed the arm's link masses.


def test_the_fast_model_is_the_one_being_used(kin: ArmKinematics) -> None:
    assert kin.fast_model
    assert kin._e_model is not kin.model


def test_fk_and_jacobian_match_the_host_model_exactly() -> None:
    """Every pose query, checked against ``mj_forward`` on the scene the arm actually lives in."""
    from brick_builder.scene.compose import SceneConfig, build_scene

    scene = build_scene(SceneConfig(cameras=()))
    fast = scene.kin
    assert fast.fast_model
    host = ArmKinematics(scene.model, fast=False)     # same class, evaluating on the scene model

    data = mujoco.MjData(scene.model)
    site = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    rng = np.random.default_rng(5)
    lo, hi = fast.jnt_range[:, 0], fast.jnt_range[:, 1]
    for _ in range(25):
        q = rng.uniform(0.9 * lo, 0.9 * hi)

        mujoco.mj_resetData(scene.model, data)
        data.qpos[host.qpos_adr] = q
        mujoco.mj_forward(scene.model, data)

        pos, rot = fast.fk(q)
        assert np.array_equal(pos, data.site_xpos[site])
        assert np.array_equal(rot, data.site_xmat[site].reshape(3, 3))
        assert np.array_equal(fast.jacobian(q), host.jacobian(q))
        assert np.array_equal(
            fast.gravity_torque(q), data.qfrc_bias[host.dof_adr]
        ), "gravity torque drifted from the scene model's own qfrc_bias"


def test_gravity_torque_follows_a_randomized_link_mass() -> None:
    """``dynamics.link_mass_scale`` writes the scene model's masses, so it has to reach the copy.

    Without the mirror this is the one thing that would go quietly wrong: forward kinematics
    would still be exact and every other test would still pass, while the gravity feedforward
    the environment commands would be computed for an arm of the wrong weight.
    """
    from brick_builder.scene import randomize
    from brick_builder.scene.compose import SceneConfig, build_scene

    scene = build_scene(SceneConfig(cameras=()))
    q = scene.home_q.copy()
    before = scene.kin.gravity_torque(q)

    randomize.apply(scene, {"dynamics.link_mass_scale": 1.1})
    heavier = scene.kin.gravity_torque(q)

    host = ArmKinematics(scene.model, fast=False)
    # Not bit-for-bit here, unlike every other pose query: `randomize.apply` runs `mj_setConst`,
    # which rebuilds the host's derived constants and leaves the two models about 1e-14 apart
    # in relative terms (measured 2.8e-17 absolute on a 0.37 N*m torque).
    assert np.allclose(heavier, host.gravity_torque(q), rtol=1e-12, atol=1e-15)
    assert np.max(np.abs(heavier - before)) > 1e-3, "a 10 % heavier arm must need more torque"

    randomize.apply(scene, {"dynamics.link_mass_scale": 1.0})
    assert np.allclose(scene.kin.gravity_torque(q), before, rtol=1e-12, atol=1e-15)


def test_ik_diff_reaches_what_the_global_solver_reaches_from_a_warm_start(
    kin: ArmKinematics,
) -> None:
    """The per-tick claim: a target a couple of millimetres away needs a step or two, no more.

    Seeds are drawn around the tool-down working pose the environments live in, because that is
    what the claim is about; a seed out at the edge of the workspace, where the arm is nearly
    singular, is what the Levenberg-Marquardt fallback is for. Measured there with two steps:
    the median of 200 lands 0.17 um from the target and the worst of them 2 um from it, against
    the 0.2 mm the environment accepts.
    """
    rng = np.random.default_rng(31)
    base = np.array([0.0, -0.5, 0.9, 1.2, 0.0])
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    errors = []
    worst_ang = 0.0
    for _ in range(200):
        q_seed = np.clip(base + rng.uniform(-0.15, 0.15, N_DOF), lo, hi)
        pos, rot = kin.fk(q_seed)
        _, pitch, roll = euler_zyx(rot)
        target = pos + rng.uniform(-0.002, 0.002, 3)

        result = kin.ik_diff(
            target, pitch, roll, q_seed, iters=2, pos_tol=1e-9, ang_tol=1e-9
        )
        got, got_rot = kin.fk(result.q)
        _, got_pitch, got_roll = euler_zyx(got_rot)
        errors.append(float(np.linalg.norm(got - target)))
        worst_ang = max(worst_ang, abs(got_pitch - pitch), abs(_wrap(got_roll - roll)))
    errors = np.array(errors)
    assert np.median(errors) < 1e-6, f"median off by {np.median(errors) * 1e3:.6f} mm"
    assert errors.max() < 1e-5, f"worst off by {errors.max() * 1e3:.5f} mm"
    assert math.degrees(worst_ang) < 0.01, f"tool axis off by {math.degrees(worst_ang):.5f} deg"


def test_ik_diff_leaves_an_already_solved_seed_exactly_alone(kin: ArmKinematics) -> None:
    """"Zero residual reproduces the scripted baseline" needs this to be bit-exact."""
    q_true = np.array([0.2, -0.3, 0.5, -0.4, 0.1])
    pos, rot = kin.fk(q_true)
    yaw, pitch, roll = euler_zyx(rot)
    result = kin.ik_diff(pos, pitch, roll, q_true, target_yaw=yaw)
    assert result.converged
    assert result.iterations == 0
    assert np.array_equal(result.q, q_true)


def test_ik_diff_respects_joint_limits(kin: ArmKinematics) -> None:
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    result = kin.ik_diff(np.array([-1.0, 0.0, 0.5]), 0.0, 0.0, np.zeros(N_DOF), iters=20)
    assert np.all(result.q >= lo - 1e-9)
    assert np.all(result.q <= hi + 1e-9)
    assert not result.converged
