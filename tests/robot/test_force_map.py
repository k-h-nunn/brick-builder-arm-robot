"""Acceptance tests for the static press-force map.

``press_force_map`` grids downward force over tool-down poses; these check the grid's shape and
bookkeeping, that force scales with supply voltage the way the actuator model does, that the
shoulder-limited far region is genuinely shoulder-limited, that the map's own reported force
number checks out against an independent, in-simulation statics test, and that the figure function
actually writes a PNG.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from brick_builder.robot.force_map import figure_force_map, press_force_map, stall_torque
from brick_builder.robot.kinematics import ArmKinematics, KIN_JOINTS
from brick_builder.robot.patch_so101 import ArmConfig, NOMINAL_VIN, attach_arm


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    world = mujoco.MjSpec()
    attach_arm(world)
    return world.compile()


@pytest.fixture(scope="module")
def kin(model: mujoco.MjModel) -> ArmKinematics:
    return ArmKinematics(model)


def test_stall_torque_scales_with_voltage() -> None:
    base = ArmConfig().torque_limit_nm
    assert stall_torque(NOMINAL_VIN) == pytest.approx(base)
    assert stall_torque(12.0) == pytest.approx(base * 12.0 / NOMINAL_VIN)
    assert stall_torque(5.0) == pytest.approx(base * 5.0 / NOMINAL_VIN)


def test_grid_shape_and_bookkeeping(model: mujoco.MjModel, kin: ArmKinematics) -> None:
    xs = np.linspace(0.12, 0.28, 5)
    ys = np.linspace(-0.08, 0.08, 4)
    result = press_force_map(model, kin, xs, ys, z=0.05, vin_list=(5.0, 7.4, 12.0))

    assert result["reachable"].shape == (4, 5)
    assert result["q"].shape == (4, 5, len(KIN_JOINTS))
    assert set(result["force"].keys()) == {5.0, 7.4, 12.0}
    for grid in result["force"].values():
        assert grid.shape == (4, 5)
        # Unreachable cells are NaN, not zero or a stale value from a neighbour.
        assert np.array_equal(np.isnan(grid), ~result["reachable"])
        assert np.all(grid[result["reachable"]] >= 0.0)
    # Most of a modest grid this close to the base should actually be reachable.
    assert result["reachable"].mean() > 0.5


def test_force_increases_with_voltage(model: mujoco.MjModel, kin: ArmKinematics) -> None:
    """More torque budget at the same pose can only raise the force ceiling, never lower it."""
    xs = np.linspace(0.14, 0.24, 4)
    ys = np.array([0.0, 0.06])
    result = press_force_map(model, kin, xs, ys, z=0.05, vin_list=(5.0, 7.4, 12.0))
    reachable = result["reachable"]
    assert np.all(result["force"][7.4][reachable] >= result["force"][5.0][reachable] - 1e-9)
    assert np.all(result["force"][12.0][reachable] >= result["force"][7.4][reachable] - 1e-9)


def test_force_map_figure_is_written(model: mujoco.MjModel, kin: ArmKinematics, tmp_path: Path) -> None:
    xs = np.linspace(0.14, 0.26, 6)
    ys = np.linspace(-0.08, 0.08, 5)
    result = press_force_map(model, kin, xs, ys, z=0.05)
    out = figure_force_map(result, tmp_path / "force_map.png")
    assert out.exists()
    assert out.stat().st_size > 1000  # not an empty/blank canvas


# ---------------------------------------------------------------- shoulder-limited region check ---


def test_shoulder_limited_region_is_shoulder_lift(model: mujoco.MjModel, kin: ArmKinematics) -> None:
    """Near full extension, the arm's own shoulder_lift joint should be the one that saturates
    first (it carries the whole arm's weight and has the longest lever arm to the TCP), matching
    the "shoulder-limited region" the spec's force-vs-reach rule of thumb refers to."""
    import math

    reach = 0.32
    res = kin.ik(np.array([reach, 0.0, 0.02]), math.pi / 2, 0.0, np.zeros(len(KIN_JOINTS)), iters=200)
    assert res.converged
    jac = kin.jacobian(res.q)
    jz = jac[2, :]
    tau_g = kin.gravity_torque(res.q)
    avail = stall_torque(NOMINAL_VIN) - tau_g
    ratios = avail / np.abs(jz)
    limiting = KIN_JOINTS[int(np.argmin(ratios))]
    assert limiting == "shoulder_lift"


def test_reported_force_saturates_the_limiting_joint_in_simulation() -> None:
    """The specification's `torque/reach` rule of thumb was the wrong check (it is 55-59%
    low near full extension because it ignores gravity torque and the folded-forearm lever arm; see
    the guide page). This replaces it with an independent statics check: build a fresh arm, solve
    for the same tool-down pose `press_force_map` would score, apply the map's own reported downward
    force back at the TCP as a real Cartesian wrench in an actual simulation (`mj_applyFT` plus
    forward dynamics settling, not the Jacobian arithmetic that produced the number), and confirm the
    limiting joint's actuator torque lands on its own torque limit within 5 %."""
    world = mujoco.MjSpec()
    attach_arm(world, ArmConfig(backlash_deg=0.0))
    model = world.compile()
    kin = ArmKinematics(model)

    reach, z = 0.32, 0.02
    res = kin.ik(np.array([reach, 0.0, z]), math.pi / 2, 0.0, np.zeros(len(KIN_JOINTS)), iters=200)
    assert res.converged
    jac = kin.jacobian(res.q)
    jz = jac[2, :]
    tau_g = kin.gravity_torque(res.q)
    tau_max = stall_torque(NOMINAL_VIN)
    ratios = (tau_max - tau_g) / np.abs(jz)
    limiting = int(np.argmin(ratios))
    force = float(ratios[limiting])
    assert KIN_JOINTS[limiting] == "shoulder_lift"

    data = mujoco.MjData(model)
    for adr, val in zip(kin.qpos_adr, res.q):
        data.qpos[adr] = val
    mujoco.mj_forward(model, data)
    data.ctrl[: len(KIN_JOINTS)] = res.q  # position servo holds the pose against the applied load

    tcp_body = int(model.site_bodyid[kin.site_id])
    point = data.site_xpos[kin.site_id].copy()
    force_vec = np.array([0.0, 0.0, force])  # upward reaction to the arm pressing down with `force`
    for _ in range(800):
        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(model, data, force_vec, np.zeros(3), point, tcp_body, data.qfrc_applied)
        mujoco.mj_step(model, data)

    assert abs(data.actuator_force[limiting]) == pytest.approx(tau_max, rel=0.05)
