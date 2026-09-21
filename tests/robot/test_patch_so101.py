"""Acceptance tests for the patched SO-101 model (spec section 1).

These assert physics, not "it ran": the servo law against BAM's measured numbers, static sag against
``tau / kp``, force saturation against the 7.4 V stall torque, and that the gripper can actually hold
a 32 mm brick -- which it cannot do with the upstream mesh collision geometry.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.robot import patch_so101 as P
from brick_builder.robot.patch_so101 import (
    ARM_JOINTS,
    FIXED_PAD_GEOM,
    MOVING_PAD_GEOM,
    TCP_SITE,
    WRIST_CAM,
    ArmConfig,
    attach_arm,
    build_arm_spec,
    servo_kp,
)
from brick_builder.robot.servo_model import ServoBus, ServoConfig, measure_sag


def _compile(cfg: ArmConfig = ArmConfig()):
    model = build_arm_spec(cfg).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


# ---------------------------------------------------------------------------------- the law ---


def test_servo_kp_reproduces_the_bam_measurements():
    assert servo_kp(16, 7.4) == pytest.approx(9.4, abs=0.05)
    assert servo_kp(32, 7.4) == pytest.approx(18.8, abs=0.1)


def test_servo_kp_is_linear_in_both_p_and_voltage():
    assert servo_kp(32, 7.4) == pytest.approx(2.0 * servo_kp(16, 7.4))
    assert servo_kp(16, 12.0) == pytest.approx(servo_kp(16, 7.4) * 12.0 / 7.4)
    assert servo_kp(0, 7.4) == 0.0


def test_duty_from_error_is_the_firmware_law_clipped_at_max_pwm():
    # Present_Load is this number, which is why it means nothing in newton-metres.
    assert P.duty_from_error(0.0, 16) == pytest.approx(0.0)
    assert P.duty_from_error(0.1, 16) == pytest.approx(0.1 * 16 * 0.166 * P.BAM_ERROR_GAIN_RATIO)
    assert P.duty_from_error(1.0, 32) == pytest.approx(P.BAM_MAX_DUTY)
    assert P.duty_from_error(-1.0, 32) == pytest.approx(-P.BAM_MAX_DUTY)
    # duty times the full-duty torque is the same spring servo_kp describes
    torque = P.duty_from_error(0.05, 16) * P.BAM_KT * 7.4 / P.BAM_R
    assert torque == pytest.approx(0.05 * servo_kp(16, 7.4))


def test_damping_is_back_emf_plus_viscous():
    # kt^2/R is exactly the -kt^2 dq / R term of BAM's torque law; upstream's 0.60 is the same
    # number arrived at by a different route, which is why it happens to be nearly right.
    assert P.servo_damping() == pytest.approx(0.62, abs=0.01)
    assert P.servo_damping() > P.BAM_KT**2 / P.BAM_R


def test_the_upstream_file_really_does_have_the_wrong_numbers():
    """Guards the premise of this whole module: if upstream ever fixes it, this test says so."""
    upstream = mujoco.MjModel.from_xml_path(str(P.VENDOR_XML))
    assert upstream.actuator_gainprm[:, 0] == pytest.approx(998.22)
    assert upstream.actuator_forcerange[:, 1] == pytest.approx(3.35)
    assert servo_kp(16, 7.4) < upstream.actuator_gainprm[0, 0] / 50


# ------------------------------------------------------------------------------ the model ---


def test_actuators_use_the_corrected_gain_and_torque_limit():
    cfg = ArmConfig()
    model, _ = _compile(cfg)
    assert model.nu == len(ARM_JOINTS)
    for name in ARM_JOINTS:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert model.actuator_gainprm[i, 0] == pytest.approx(servo_kp(cfg.p_gain, cfg.vin))
        assert model.actuator_biasprm[i, 1] == pytest.approx(-servo_kp(cfg.p_gain, cfg.vin))
        assert model.actuator_biasprm[i, 2] == 0.0  # kv = 0; damping lives in the joint
        assert model.actuator_forcerange[i] == pytest.approx([-1.9, 1.9])


def test_torque_limit_scales_with_supply_voltage():
    model, _ = _compile(ArmConfig(vin=12.0, torque_limit_nm=1.9))
    assert model.actuator_forcerange[0, 1] == pytest.approx(1.9 * 12.0 / 7.4)


def test_joint_dynamics_come_from_bam():
    model, _ = _compile(ArmConfig(backlash_deg=0.0))
    for name in ARM_JOINTS:
        dof = model.jnt_dofadr[_jid(model, name)]
        assert model.dof_damping[dof] == pytest.approx(0.62, abs=0.01)
        assert model.dof_frictionloss[dof] == pytest.approx(0.052, abs=0.001)
        assert model.dof_armature[dof] == pytest.approx(0.026, abs=0.001)


def test_solver_options_follow_the_design_report():
    model, _ = _compile(ArmConfig(timestep=0.002))
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(10.0)
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.noslip_iterations == 0


def test_model_steps_five_seconds_under_gravity_without_warnings():
    cfg = ArmConfig()
    world = mujoco.MjSpec()
    world.worldbody.add_light(pos=[0, 0, 1])
    world.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], pos=[0, 0, -0.001]
    )
    attach_arm(world, cfg)
    model = world.compile()
    data = mujoco.MjData(model)
    bus = ServoBus(model, data, cfg=ServoConfig())

    n = int(5.0 / model.opt.timestep)
    for k in range(n):
        if k % 10 == 0:  # 50 Hz control on a 500 Hz physics step
            phase = k * model.opt.timestep
            bus.write(
                np.array([0.3, 0.4, -0.6, 0.3, 0.0, 0.6]) * math.sin(2.0 * phase)
            )
        bus.substep()
        mujoco.mj_step(model, data)

    assert list(data.warning.number) == [0] * len(data.warning.number)
    assert np.all(np.isfinite(data.qpos))


# -------------------------------------------------------------------------------- sag / step ---


def test_static_sag_equals_tau_over_kp():
    """Hang a known torque on shoulder_lift; the joint should give way by tau/kp within 10 %."""
    cfg = ArmConfig(backlash_deg=0.0)
    torques = np.array([0.4, 0.8, 1.2, 1.6])
    measured = measure_sag(cfg, "shoulder_lift", torques)
    predicted = torques / servo_kp(cfg.p_gain, cfg.vin)
    rel = np.abs(measured - predicted) / predicted
    assert rel.max() < 0.10, f"measured {measured}, predicted {predicted}, rel err {rel}"


def test_static_sag_halves_when_p_gain_doubles():
    torques = np.array([1.0])
    soft = measure_sag(ArmConfig(p_gain=16, backlash_deg=0.0), "shoulder_lift", torques)[0]
    stiff = measure_sag(ArmConfig(p_gain=32, backlash_deg=0.0), "shoulder_lift", torques)[0]
    assert stiff == pytest.approx(soft / 2.0, rel=0.1)


def test_actuator_force_saturates_at_the_torque_limit():
    cfg = ArmConfig(backlash_deg=0.0)
    model, data = _compile(cfg)
    data.ctrl[1] = 1.5  # far past anything the spring can reach
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert abs(data.actuator_force[1]) == pytest.approx(1.9, rel=1e-6)
    assert abs(data.actuator_force[1]) <= model.actuator_forcerange[1, 1] + 1e-9


# --------------------------------------------------------------------------------- backlash ---


def test_backlash_joints_exist_and_can_be_switched_off():
    with_play = build_arm_spec(ArmConfig(backlash_deg=0.5)).compile()
    without = build_arm_spec(ArmConfig(backlash_deg=0.0)).compile()
    assert with_play.njnt == without.njnt + len(ARM_JOINTS)
    assert without.njnt == len(ARM_JOINTS)
    for name in ARM_JOINTS:
        j = _jid(with_play, f"{name}_backlash")
        assert j >= 0
        assert with_play.jnt_range[j] == pytest.approx([-math.radians(0.5), math.radians(0.5)])
        # the actuator still drives the real joint, not the free-play one
        assert with_play.actuator_trnid[
            mujoco.mj_name2id(with_play, mujoco.mjtObj.mjOBJ_ACTUATOR, name), 0
        ] == _jid(with_play, name)


def test_backlash_play_at_the_tcp_exceeds_the_brick_capture_tolerance():
    """0.5 deg of gear play is worth millimetres at the tool, far more than a stud's 0.3 mm."""
    model, data = _compile(ArmConfig(backlash_deg=0.5))
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    play = {}
    for name in ARM_JOINTS:
        adr = model.jnt_qposadr[_jid(model, f"{name}_backlash")]
        rng = model.jnt_range[_jid(model, f"{name}_backlash")]
        data.qpos[:] = 0.0
        data.qpos[adr] = rng[0]
        mujoco.mj_forward(model, data)
        lo = data.site_xpos[site].copy()
        data.qpos[adr] = rng[1]
        mujoco.mj_forward(model, data)
        play[name] = float(np.linalg.norm(data.site_xpos[site] - lo))
    assert max(play.values()) > 0.0015, play
    assert play["shoulder_pan"] > 0.003
    assert play["gripper"] < 0.001  # the jaw hinge barely moves the TCP


# ----------------------------------------------------------------------- sites, camera, geoms ---


def test_tcp_site_sits_between_the_pads_and_points_along_the_approach():
    model, data = _compile(ArmConfig(backlash_deg=0.0))
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    assert site >= 0
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, P.GRIPPER_BODY)
    assert model.site_bodyid[site] == gripper

    # z axis of the site == -z of the gripper body, i.e. it points out past the fingertips
    site_z = data.site_xmat[site].reshape(3, 3)[:, 2]
    body_z = data.xmat[gripper].reshape(3, 3)[:, 2]
    assert np.dot(site_z, body_z) == pytest.approx(-1.0, abs=1e-6)

    # and it really is between the two pads, at the nominal opening
    fixed = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FIXED_PAD_GEOM)
    moving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, MOVING_PAD_GEOM)
    data.qpos[model.jnt_qposadr[_jid(model, "gripper")]] = P.gripper_angle_for_width(
        P.NOMINAL_GRIP_WIDTH
    )
    mujoco.mj_forward(model, data)
    tcp = data.site_xpos[site]
    mid = 0.5 * (data.geom_xpos[fixed] + data.geom_xpos[moving])
    lateral = np.dot(tcp - mid, data.xmat[gripper].reshape(3, 3)[:, 0])
    # The jaw swings on a hinge, so the pad faces are not parallel and the midpoint of the two pad
    # *centres* is a few mm off the plane where a brick actually sits.  A few mm is the right
    # tolerance here; the sign is what matters, and it must be between the two pads.
    assert abs(lateral) < 0.004, f"TCP is {lateral * 1000:.1f} mm off the pad midplane"
    gx = data.xmat[gripper].reshape(3, 3)[:, 0]
    assert np.dot(data.geom_xpos[fixed], gx) < np.dot(tcp, gx) < np.dot(data.geom_xpos[moving], gx)


def test_wrist_camera_is_on_the_gripper_body_and_sees_the_tcp():
    model, data = _compile(ArmConfig(backlash_deg=0.0))
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    assert cam >= 0
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, P.GRIPPER_BODY)
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, P.MOVING_JAW_BODY)
    assert model.cam_bodyid[cam] == gripper
    assert model.cam_bodyid[cam] != jaw  # the local xacro parents it to the jaw; do not copy that

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    to_tcp = data.site_xpos[site] - data.cam_xpos[cam]
    distance = np.linalg.norm(to_tcp)
    forward = -data.cam_xmat[cam].reshape(3, 3)[:, 2]  # MuJoCo cameras look down -z
    # The real mount is bolted on with a fixed geometric relationship to the jaw -- it is not
    # independently aimed at the grasp point the way the old guessed pose was, so this is a
    # generous "roughly toward the TCP" bound (about 20 degrees), not an exact-aim assertion.
    # `test_wrist_camera_pose_matches_the_derived_mount_constants` below pins the exact axis.
    cos_angle = np.dot(to_tcp / distance, forward)
    assert cos_angle > math.cos(math.radians(20.0)), (
        f"wrist cam is {math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))):.1f} deg off the TCP"
    )
    assert 0.03 < distance < 0.20, f"working distance {distance * 1000:.0f} mm"


def test_wrist_camera_pose_matches_the_derived_mount_constants():
    """Pins the camera to the exact pose derived from TheRobotStudio's own camera-equipped
    assembly (``Simulation/SO101/so101_new_calib_camera.xml`` -- see the constants' docstrings in
    ``patch_so101.py``), both as a regression guard and as the "camera looks along the intended
    axis" check: the axis under test is the mount's own documented forward direction, not merely
    "somewhere near the TCP" (the previous, looser check)."""
    model, data = _compile(ArmConfig(backlash_deg=0.0))
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, P.GRIPPER_BODY)

    gripper_rot = data.xmat[gripper].reshape(3, 3)
    cam_pos_in_gripper = gripper_rot.T @ (data.cam_xpos[cam] - data.xpos[gripper])
    cam_rot_in_gripper = gripper_rot.T @ data.cam_xmat[cam].reshape(3, 3)
    forward_in_gripper = -cam_rot_in_gripper[:, 2]

    assert cam_pos_in_gripper == pytest.approx(P.WRIST_CAM_POS_IN_GRIPPER, abs=1e-6)
    assert forward_in_gripper == pytest.approx(P.WRIST_CAM_FORWARD_IN_GRIPPER, abs=1e-3)
    # the compiled quat, independently converted back to a rotation matrix, must agree too (not
    # just its forward column) -- pins the image's "up"/roll direction as well as the view axis.
    quat_rot = np.zeros(9)
    mujoco.mju_quat2Mat(quat_rot, P.WRIST_CAM_QUAT_IN_GRIPPER)
    assert cam_rot_in_gripper == pytest.approx(quat_rot.reshape(3, 3), abs=1e-6)


# --------------------------------------------------------------------------- lens field of view ---


def test_fovy_from_diagonal_fov_round_trips_to_the_same_diagonal():
    """Re-deriving the diagonal half-angle from the returned fovx/fovy must reproduce the input,
    for both a 4:3 and a 16:9 sensor and across the stated 100-130 degree range."""
    for diagonal in (100.0, 115.0, 130.0):
        for width, height in [(640, 480), (1920, 1080), (256, 256)]:
            fovy = P.fovy_from_diagonal_fov(diagonal, width, height)
            fovx = P.fovx_from_diagonal_fov(diagonal, width, height)
            half_diag = math.degrees(math.atan(math.sqrt(
                math.tan(math.radians(fovx) / 2.0) ** 2 + math.tan(math.radians(fovy) / 2.0) ** 2
            )))
            assert 2.0 * half_diag == pytest.approx(diagonal, abs=1e-6)


def test_fovy_from_diagonal_fov_shrinks_as_the_image_gets_wider():
    """At a fixed diagonal fov, a wider aspect ratio spends more of that diagonal horizontally,
    leaving a smaller vertical angle: square > 4:3 > 16:9, for the same diagonal."""
    diag = 120.0
    fovy_square = P.fovy_from_diagonal_fov(diag, 256, 256)
    fovy_4_3 = P.fovy_from_diagonal_fov(diag, 640, 480)
    fovy_16_9 = P.fovy_from_diagonal_fov(diag, 1920, 1080)
    assert fovy_square > fovy_4_3 > fovy_16_9
    # and the fovy this module actually compiles in is the 4:3 number, not some other aspect
    assert fovy_4_3 == pytest.approx(92.19, abs=0.05)


def test_wrist_cam_fovy_deg_matches_the_documented_diagonal_and_resolution():
    lo, hi = P.WRIST_CAM_DIAGONAL_FOV_RANGE_DEG
    assert lo < P.WRIST_CAM_DIAGONAL_FOV_DEG < hi
    assert lo == pytest.approx(100.0) and hi == pytest.approx(130.0)
    expected = P.fovy_from_diagonal_fov(P.WRIST_CAM_DIAGONAL_FOV_DEG, *P.WRIST_CAM_NOMINAL_RESOLUTION)
    assert P.WRIST_CAM_FOVY_DEG == pytest.approx(expected)
    # replaces the previous guess's OV5647-derived 41.4 deg, which has nothing to do with this lens
    assert P.WRIST_CAM_FOVY_DEG != pytest.approx(41.4, abs=1.0)


def test_compiled_camera_uses_the_documented_fovy():
    model, _ = _compile(ArmConfig())
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    assert model.cam_fovy[cam] == pytest.approx(P.WRIST_CAM_FOVY_DEG)


# ------------------------------------------------------------------- fixed-jaw share of the image ---


def _fixed_jaw_pixel_share(model: mujoco.MjModel, data: mujoco.MjData, width=320, height=240) -> float:
    """Fraction of the wrist camera's image covered by the fixed jaw's own visual mesh.

    Uses MuJoCo's native per-geom segmentation renderer (not ``brick_builder.render.renderer
    .SceneRenderer``, which only resolves to body granularity -- the fixed jaw's mesh geom shares
    ``GRIPPER_BODY`` with the wrist-roll servo and the camera itself, so a body-level LUT could not
    isolate it) to find exactly the geom rendering ``_FIXED_JAW_MESH`` and count its pixels.
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=WRIST_CAM)
        seg = renderer.render()
    finally:
        renderer.close()

    gripper_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, P.GRIPPER_BODY)
    jaw_gid = next(
        i for i in range(model.ngeom)
        if model.geom_dataid[i] >= 0
        and model.geom_bodyid[i] == gripper_bid
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[i]) == P._FIXED_JAW_MESH
    )
    mask = (seg[..., 0] == jaw_gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    return float(mask.mean())


def test_fixed_jaw_share_of_the_image_is_small_at_the_home_pose():
    """The bug this whole module fixes: the previous guessed pose put the fixed jaw at 80-85% of
    the frame (measured, see docs/guide/03-robot-and-servo-model.md and
    docs/media/robot/wrist_camera_home.png). At the home pose (qpos = 0, the arm's rest
    configuration) the corrected mount+lens keeps it under a fifth of the frame."""
    model, data = _compile(ArmConfig(backlash_deg=0.0))
    share = _fixed_jaw_pixel_share(model, data)
    assert share < 0.20, f"fixed jaw covers {share * 100:.1f}% of the wrist-cam image"


def test_primitive_collision_replaces_every_collision_mesh():
    model, _ = _compile(ArmConfig(primitive_collision=True))
    collidable = [
        i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]
    ]
    assert collidable, "the arm has to collide with something"
    for i in collidable:
        assert model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_group[i] == 3

    meshy, _ = _compile(ArmConfig(primitive_collision=False))
    assert any(
        meshy.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and (meshy.geom_contype[i] or meshy.geom_conaffinity[i])
        for i in range(meshy.ngeom)
    )


def test_fingertip_pads_use_condim_four_and_the_configured_friction():
    model, _ = _compile(ArmConfig(pad_friction=0.7))
    for name in (FIXED_PAD_GEOM, MOVING_PAD_GEOM):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0
        assert model.geom_condim[g] == 4
        assert model.geom_friction[g, 0] == pytest.approx(0.7)
        assert model.geom_priority[g] == 1  # so the pad's friction wins over whatever it touches


# ---------------------------------------------------------------------------- the STL reader ---


def test_stl_reader_agrees_with_the_compiled_mesh():
    """Our numpy reader has to land the vertices exactly where MuJoCo's loader does."""
    spec = mujoco.MjSpec.from_file(str(P.VENDOR_XML))
    model = mujoco.MjSpec.from_file(str(P.VENDOR_XML)).compile()
    geom = next(g for g in spec.geoms if g.meshname == "moving_jaw_so101_v1")
    ours = P._mesh_points_in_body(geom)

    gid = next(
        i
        for i in range(model.ngeom)
        if model.geom_dataid[i] >= 0
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[i])
        == "moving_jaw_so101_v1"
    )
    mid = model.geom_dataid[gid]
    verts = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    quat = np.zeros(9)
    mujoco.mju_quat2Mat(quat, model.geom_quat[gid])
    theirs = verts.astype(float) @ quat.reshape(3, 3).T + model.geom_pos[gid]

    assert ours.min(axis=0) == pytest.approx(theirs.min(axis=0), abs=1e-6)
    assert ours.max(axis=0) == pytest.approx(theirs.max(axis=0), abs=1e-6)


def test_stl_reader_rejects_a_file_that_is_not_a_binary_stl(tmp_path):
    bad = tmp_path / "ascii.stl"
    bad.write_text("solid foo\nfacet normal 0 0 1\nendsolid foo\n")
    with pytest.raises(ValueError):
        P.read_binary_stl(bad)


# -------------------------------------------------------------------------------- the grasp ---


def _grasp_world(primitive: bool, width: float = 0.032, clearance: float = 0.002):
    cfg = ArmConfig(primitive_collision=primitive, backlash_deg=0.0)
    world = mujoco.MjSpec()
    world.worldbody.add_light(pos=[0, 0, 1])
    attach_arm(world, cfg)
    body = world.worldbody.add_body(name="brick", pos=[0, 0, 0.5])
    body.add_freejoint()
    body.add_geom(
        name="brick_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[width / 2] * 3,
        mass=0.0122,          # a DUPLO 2x4 weighs 12.2 g
        condim=4,
        friction=[0.6, 0.005, 0.0001],
        rgba=[0.9, 0.2, 0.2, 1],
    )
    model = world.compile()
    data = mujoco.MjData(model)

    gripper_q = model.jnt_qposadr[_jid(model, "gripper")]
    data.qpos[gripper_q] = P.gripper_angle_for_width(width + clearance)
    mujoco.mj_forward(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, P.GRIPPER_BODY)
    brick = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brick")
    adr = model.jnt_qposadr[model.body_jntadr[brick]]
    data.qpos[adr : adr + 3] = data.site_xpos[site]
    data.qpos[adr + 3 : adr + 7] = data.xquat[gid]
    mujoco.mj_forward(model, data)
    return cfg, model, data, site, adr, width


def _close_and_hold(model, data, site, adr, width, seconds=2.0, squeeze=0.004):
    bus = ServoBus(model, data, cfg=ServoConfig(action_delay_jitter_s=0.0))
    goal = np.array(
        [data.qpos[model.jnt_qposadr[_jid(model, n)]] for n in ARM_JOINTS]
    )
    goal[-1] = P.gripper_angle_for_width(width - squeeze)
    bus.write(goal)
    start = data.qpos[adr : adr + 3] - data.site_xpos[site]
    for _ in range(int(seconds / model.opt.timestep)):
        bus.substep()
        mujoco.mj_step(model, data)
    end = data.qpos[adr : adr + 3] - data.site_xpos[site]
    return float(np.linalg.norm(end - start))


def _brick_contacts(model, data):
    brick = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick_geom")
    names = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if brick in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == brick else c.geom1
            names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
    return names


def test_gripper_holds_a_32_mm_brick_against_gravity():
    _, model, data, site, adr, width = _grasp_world(primitive=True)
    assert data.ncon == 0, "the brick should fit at the TCP without any initial penetration"
    slip = _close_and_hold(model, data, site, adr, width)
    assert slip < 0.003, f"brick slipped {slip * 1000:.1f} mm relative to the TCP"
    touching = _brick_contacts(model, data)
    assert {FIXED_PAD_GEOM, MOVING_PAD_GEOM} <= touching, touching
    assert list(data.warning.number) == [0] * len(data.warning.number)


def test_the_grasp_is_friction_and_not_a_weld():
    """The brick is held by pad friction with a large margin, and a big enough pull tears it out.

    The pads squeeze with about 17 N, which against mu = 1.0 is well over a hundred times the
    brick's 0.12 N weight -- so "holds it against gravity" is not a close call.  Pulling with 2 N
    still barely moves it; 20 N rips it out, which is the check that nothing here is glued.
    """
    _, model, data, site, adr, width = _grasp_world(primitive=True)
    _close_and_hold(model, data, site, adr, width)

    pads = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in (FIXED_PAD_GEOM, MOVING_PAD_GEOM)
    ]
    brick_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick_geom")
    normal = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if brick_geom in (c.geom1, c.geom2) and (c.geom1 in pads or c.geom2 in pads):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            normal += abs(force[0])
    weight = 0.0122 * 9.81
    assert normal > 10.0 * weight, f"only {normal:.2f} N of pinch against {weight:.3f} N of brick"

    def slip_under(pull_n: float) -> float:
        _, m, d, s, a, w = _grasp_world(primitive=True)
        _close_and_hold(m, d, s, a, w)
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "brick")
        before = d.qpos[a : a + 3] - d.site_xpos[s]
        for _ in range(int(1.0 / m.opt.timestep)):
            d.xfrc_applied[body, 2] = -pull_n
            mujoco.mj_step(m, d)
        return float(np.linalg.norm((d.qpos[a : a + 3] - d.site_xpos[s]) - before))

    assert slip_under(2.0) < 0.003
    assert slip_under(20.0) > 0.02


def test_mesh_collision_cannot_grasp_because_the_hull_fills_the_jaw():
    """Documents why ``primitive_collision`` exists.

    MuJoCo collides against a mesh's convex hull.  The hull of ``wrist_roll_follower`` spans the jaw
    opening, so the pinch point is *inside* solid geometry: the brick starts 8 mm interpenetrated
    and is squirted out instead of held, and it never touches the fingertips.
    """
    _, model, data, site, adr, width = _grasp_world(primitive=False)
    penetration = min(
        (data.contact[i].dist for i in range(data.ncon)), default=0.0
    )
    assert penetration < -0.005, (
        f"expected the hull to swallow the pinch point, deepest contact was {penetration * 1000:.1f} mm"
    )
    slip = _close_and_hold(model, data, site, adr, width)
    assert slip > 0.005, f"mesh collision unexpectedly held the brick ({slip * 1000:.1f} mm slip)"


# ------------------------------------------------------------------------------ attach_arm ---


def test_attach_arm_keeps_names_and_carries_the_solver_options():
    world = mujoco.MjSpec()
    world.worldbody.add_light(pos=[0, 0, 1])
    attach_arm(world, ArmConfig(), pos=(0.1, -0.2, 0.05))
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name in ARM_JOINTS:
        assert _jid(model, name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE) >= 0
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(10.0)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert data.xpos[base] == pytest.approx([0.1, -0.2, 0.05])


def test_gripper_angle_for_width_is_monotone_and_clamped():
    widths = np.linspace(0.005, 0.09, 12)
    angles = np.array([P.gripper_angle_for_width(w) for w in widths])
    assert np.all(np.diff(angles) >= 0)
    assert P.gripper_angle_for_width(10.0) == pytest.approx(1.74, abs=0.01)
    assert P.gripper_angle_for_width(0.0) == pytest.approx(-0.18, abs=0.01)
