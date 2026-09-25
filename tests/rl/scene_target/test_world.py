"""The environment's world, twin and constants against `brick_builder.action`'s.

The environment restates these rather than importing them (importing `brick_builder.action`
loads the analytic servo). These tests import both and check they agree.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from brick_builder.action import adapter, worlds  # noqa: E402
from brick_builder.rl.scene_target import world as W  # noqa: E402


def test_constants_equal_the_action_packages():
    assert W.NOMINAL_PLATE_XY == worlds.NOMINAL_PLATE_XY
    assert W.REF_YELLOW == worlds.REF_YELLOW and W.REF_RED == worlds.REF_RED
    assert W.WRIST_FULL == adapter.WRIST_FULL
    assert W.WRIST_RENDER == adapter.WRIST_RENDER
    assert W.VISION_EVERY == adapter.VISION_EVERY


def test_empty_hand_config_is_the_worlds_configuration_without_a_held_brick():
    ref = worlds.scene_cfg(plate_xy=worlds.NOMINAL_PLATE_XY, plate_yaw=0.0,
                           placed=[worlds.REF_YELLOW, worlds.REF_RED], held=None,
                           home_tcp=(worlds.NOMINAL_PLATE_XY[0], worlds.NOMINAL_PLATE_XY[1], 0.075),
                           cameras=())
    assert W.empty_hand_config() == ref


@pytest.mark.parametrize("seed", list(range(10)) + [20, 49, 1000])
def test_plate_displacement_is_the_worlds_draw(seed):
    d_xy, d_yaw = worlds.displacement(seed)
    e_xy, e_yaw = W.plate_displacement(seed)
    np.testing.assert_array_equal(e_xy, d_xy)
    assert e_yaw == d_yaw
    h_xy, h_yaw = W.plate_displacement(seed, 0.25)
    np.testing.assert_allclose(h_xy, 0.25 * d_xy)
    assert h_yaw == pytest.approx(0.25 * d_yaw)


@pytest.mark.slow
def test_replaced_plate_equals_a_world_built_displaced():
    """Moving the plate in the compiled model then resetting equals compiling it displaced."""
    from brick_builder.render.renderer import SceneRenderer
    from brick_builder.scene.compose import build_scene, reset_scene, tool_down_ik

    seed = 3
    d_xy, d_yaw = worlds.displacement(seed)
    tcp, yaw = np.array([0.18, 0.01, 0.065]), 0.1
    ours = W.ViewingPoseWorld()
    fresh = build_scene(W.empty_hand_config(np.asarray(W.NOMINAL_PLATE_XY) + d_xy, d_yaw))
    renderer = SceneRenderer(fresh.model, width=W.WRIST_RENDER[0], height=W.WRIST_RENDER[1])
    try:
        ours.reset(plate_offset_xy=d_xy, plate_yaw=d_yaw, arm_tcp=tcp, arm_yaw=yaw, seed=seed)
        q = tool_down_ik(fresh.kin, tcp, yaw, q0=fresh.home_q)
        reset_scene(fresh, np.random.default_rng(seed), arm_q=q)
        assert ours.scene.model.nbody == fresh.model.nbody
        for b in range(fresh.model.nbody):
            name = mujoco.mj_id2name(fresh.model, mujoco.mjtObj.mjOBJ_BODY, b)
            assert name == mujoco.mj_id2name(ours.scene.model, mujoco.mjtObj.mjOBJ_BODY, b)
        np.testing.assert_allclose(ours.scene.data.xpos, fresh.data.xpos, atol=1e-12)
        np.testing.assert_allclose(ours.scene.data.xquat, fresh.data.xquat, atol=1e-12)
        renderer.instance_lut(fresh.instance_ids)
        from brick_builder.perception.views import render_view

        a = ours.render()
        b, _ = render_view(fresh, W.WRIST_FULL, renderer)
        np.testing.assert_array_equal(a.instance, b.instance)
        np.testing.assert_array_equal(a.depth, b.depth)
    finally:
        renderer.close()
        ours.close()


@pytest.mark.slow
def test_nominal_twin_poses_and_sees_what_the_digital_twin_does():
    from brick_builder.action.twin import DigitalTwin
    from brick_builder.perception.ideal_scene_model import IdealSceneModel

    ref = DigitalTwin(worlds.nominal_cfg(held=None))
    ours = W.NominalTwin()
    try:
        src = IdealSceneModel(build_decode=False)
        for tcp, yaw in (((0.19, 0.02, 0.06), 0.2), ((0.17, 0.0, 0.07), 0.0),
                         ((0.15, -0.02, 0.05), -0.2)):
            state_ref, _ = ref.observe_rgb(tcp, yaw)
            q = ours.pose(tcp, yaw)
            np.testing.assert_array_equal(q, ref.scene.data.qpos[ref.scene.arm_qpos_adr[:5]])
            src.reset()
            frame = ours.render()
            state = W.observe(src, frame)
            s = state.schema
            for field in ("mass", "centre", "moments", "border", "orient4"):
                np.testing.assert_array_equal(s.get(state.slots, field), s.get(state_ref.slots, field))
            np.testing.assert_allclose(s.get(state.slots, "colour"),
                                       s.get(state_ref.slots, "colour"), atol=2e-3)
            np.testing.assert_array_equal(state.slot_ids, state_ref.slot_ids)
    finally:
        ref.close()
        ours.close()
