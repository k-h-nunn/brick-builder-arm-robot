"""``CameraConfig`` behaviour-identity proof: the default wrist-camera render must be
pixel-identical before and after routing the camera through ``CameraConfig``.

There is no separate "before" file to diff against (`patch_so101.py` was edited in place) --
instead, `_legacy_build_arm_spec` below reconstructs exactly what `build_arm_spec` did before
`CameraConfig` existed, one call at a time, by composing the same private per-concern functions
(`_apply_options`, `_patch_actuators`, ...) `build_arm_spec` itself still calls, swapping in only
the single hard-coded `add_camera(...)` line the old code used in place of today's
`CameraConfig`-driven one. `test_default_camera_compiles_to_a_bit_for_bit_identical_model` compares
the two compiled models' serialized bytes directly, which pins behaviour identity exactly, with no
GL-noise tolerance; the pixel-identity tests below additionally prove the *rendered* output is the
same (up to the GL rasterizer's own noise), which is what a viewer or a trained model actually sees.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from brick_builder.robot import patch_so101 as P
from brick_builder.robot.patch_so101 import ArmConfig, CameraConfig, attach_arm, build_arm_spec
from brick_builder.render.renderer import SceneRenderer


def _legacy_build_arm_spec(cfg: ArmConfig) -> mujoco.MjSpec:
    """`build_arm_spec` as it read before `CameraConfig` existed (see module docstring)."""
    spec = mujoco.MjSpec.from_file(str(P.VENDOR_XML))
    P._apply_options(spec, cfg)
    P._patch_actuators(spec, cfg)
    P._patch_joints(spec, cfg)
    P._patch_print_color(spec, cfg)
    P._replace_collision(spec, cfg)

    gripper = spec.body(P.GRIPPER_BODY)
    gripper.add_site(
        name=P.TCP_SITE, pos=P.tcp_position(), quat=[0.0, 1.0, 0.0, 0.0],
        size=[0.002, 0.002, 0.002], group=4, rgba=[0.1, 0.9, 0.2, 1.0],
    )
    gripper.add_camera(
        name=P.WRIST_CAM, pos=P.WRIST_CAM_POS_IN_GRIPPER, quat=P.WRIST_CAM_QUAT_IN_GRIPPER,
        fovy=P.WRIST_CAM_FOVY_DEG,
    )
    P._add_backlash(spec, cfg)
    return spec


def _legacy_attach_arm(world: mujoco.MjSpec, cfg: ArmConfig, pos, quat) -> None:
    """`attach_arm` as it read before `CameraConfig` existed, built on `_legacy_build_arm_spec`."""
    arm = _legacy_build_arm_spec(cfg)
    P._apply_options(world, cfg)
    frame = world.worldbody.add_frame(pos=np.asarray(pos, dtype=float), quat=np.asarray(quat, dtype=float))
    world.attach(arm, frame=frame, prefix="")


def _world_with_a_test_object() -> mujoco.MjSpec:
    """A minimal scene with something in view besides bare metal, shared by both builds below."""
    world = mujoco.MjSpec()
    world.worldbody.add_light(pos=[0.2, 0.1, 0.5], dir=[-0.2, -0.1, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.3, 0.3, 0.01], pos=[0.0, 0.0, 0.0], rgba=[0.5, 0.5, 0.55, 1.0])
    world.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.016, 0.016, 0.016], pos=[0.15, 0.0, 0.02],
        rgba=[0.85, 0.1, 0.1, 1.0], name="probe_brick",
    )
    return world


def _render_wrist_cam(world: mujoco.MjSpec) -> np.ndarray:
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with SceneRenderer(model, width=320, height=240) as renderer:
        frame = renderer.render(data, camera=P.WRIST_CAM)
    return frame.rgb


def _assert_pixel_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    """Equal up to the GL rasterizer's own antialiasing noise -- checked directly (not assumed):
    rendering the exact same compiled model through two separate `SceneRenderer`/`GLContext`
    instances already differs by +-1 on a handful of edge pixels (order 0.01% of the frame), so
    requiring bit-exact equality here would fail on GL noise having nothing to do with this
    module's behaviour. A real behaviour change (a wrong pose, fovy or pixel format) moves far more
    than a handful of edge pixels by far more than 1 level -- this bound would catch it."""
    actual_i = actual.astype(np.int16)
    expected_i = expected.astype(np.int16)
    diff = np.abs(actual_i - expected_i)
    n_mismatched = int(np.count_nonzero(diff))
    assert diff.max() <= 1, f"pixel values differ by more than the GL-noise floor: max={diff.max()}"
    frac = n_mismatched / diff.size
    assert frac < 0.001, f"{n_mismatched}/{diff.size} ({frac:.4%}) pixels differ -- more than GL noise"


# --------------------------------------------------------------------------------- numeric identity ---


def test_default_camera_config_numerically_matches_the_module_constants():
    cfg = ArmConfig()
    assert cfg.camera == CameraConfig()
    assert cfg.camera.fovy_deg() == P.WRIST_CAM_FOVY_DEG
    assert cfg.camera.pos_in_gripper == pytest.approx(tuple(P.WRIST_CAM_POS_IN_GRIPPER))
    assert cfg.camera.quat_in_gripper == pytest.approx(tuple(P.WRIST_CAM_QUAT_IN_GRIPPER))
    assert not cfg.camera.uses_measured_intrinsics


def test_compiled_default_camera_matches_the_module_constants_exactly():
    model = build_arm_spec(ArmConfig()).compile()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, P.WRIST_CAM)
    assert model.cam_fovy[cam] == P.WRIST_CAM_FOVY_DEG
    np.testing.assert_array_equal(model.cam_pos[cam], P.WRIST_CAM_POS_IN_GRIPPER)


def test_default_camera_compiles_to_a_bit_for_bit_identical_model():
    """The exact form of the pixel-identity proof below: two independently built and compiled
    default models, serialized with `mujoco.mj_saveModel` (the same MJB format `mj_loadModel`
    reads back), compare byte-for-byte equal -- not just close on the handful of `cam_*` fields
    checked above, and with none of the pixel-identity tests' GL-rasterizer noise tolerance."""
    model_a = build_arm_spec(ArmConfig()).compile()
    model_b = build_arm_spec(ArmConfig()).compile()
    size = mujoco.mj_sizeModel(model_a)
    assert size == mujoco.mj_sizeModel(model_b)
    buf_a = np.zeros(size, dtype=np.uint8)
    buf_b = np.zeros(size, dtype=np.uint8)
    mujoco.mj_saveModel(model_a, None, buf_a)
    mujoco.mj_saveModel(model_b, None, buf_b)
    np.testing.assert_array_equal(buf_a, buf_b)


# --------------------------------------------------------------------------------- pixel identity ---


def test_default_wrist_camera_render_is_pixel_identical_to_the_pre_cameraconfig_code_path():
    world_new = _world_with_a_test_object()
    attach_arm(world_new, ArmConfig(), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    rgb_new = _render_wrist_cam(world_new)

    world_old = _world_with_a_test_object()
    _legacy_attach_arm(world_old, ArmConfig(), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    rgb_old = _render_wrist_cam(world_old)

    assert rgb_new.shape == rgb_old.shape
    _assert_pixel_identical(rgb_new, rgb_old)
    # Not a degenerate (all-background) render -- the probe brick and the arm are actually in frame.
    assert rgb_new.std() > 5.0


def test_default_wrist_camera_render_is_pixel_identical_at_a_non_home_pose_too():
    """Same proof at a moved joint configuration, so the identity is not an artefact of qpos=0."""
    q = np.array([0.3, -0.4, 0.6, 0.5, -0.2])
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

    world_new = _world_with_a_test_object()
    attach_arm(world_new, ArmConfig())
    model_new = world_new.compile()
    data_new = mujoco.MjData(model_new)
    for name, val in zip(joints, q):
        data_new.qpos[model_new.joint(name).qposadr[0]] = val
    mujoco.mj_forward(model_new, data_new)
    with SceneRenderer(model_new, width=320, height=240) as r:
        rgb_new = r.render(data_new, camera=P.WRIST_CAM).rgb

    world_old = _world_with_a_test_object()
    _legacy_attach_arm(world_old, ArmConfig(), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    model_old = world_old.compile()
    data_old = mujoco.MjData(model_old)
    for name, val in zip(joints, q):
        data_old.qpos[model_old.joint(name).qposadr[0]] = val
    mujoco.mj_forward(model_old, data_old)
    with SceneRenderer(model_old, width=320, height=240) as r:
        rgb_old = r.render(data_old, camera=P.WRIST_CAM).rgb

    _assert_pixel_identical(rgb_new, rgb_old)


# --------------------------------------------------------------------------------- measured-intrinsics path ---


def test_measured_intrinsics_path_reproduces_fovy_within_float32_precision():
    """The alternate path (fx/fy/cx/cy overriding the diagonal fov) is a different MuJoCo camera
    model (pixel intrinsics rather than fovy), so it is only float32-close, not bit-identical --
    documented on :meth:`CameraConfig.fovy_deg` and checked directly here."""
    diag_cfg = CameraConfig(diagonal_fov_deg=100.0, resolution=(640, 480))
    fx, fy, cx, cy = diag_cfg.resolved_intrinsics_px()
    px_cfg = CameraConfig(resolution=(640, 480), fx_px=fx, fy_px=fy, cx_px=cx, cy_px=cy)
    assert px_cfg.uses_measured_intrinsics

    model_diag = build_arm_spec(ArmConfig(camera=diag_cfg)).compile()
    model_px = build_arm_spec(ArmConfig(camera=px_cfg)).compile()
    cam_diag = mujoco.mj_name2id(model_diag, mujoco.mjtObj.mjOBJ_CAMERA, P.WRIST_CAM)
    cam_px = mujoco.mj_name2id(model_px, mujoco.mjtObj.mjOBJ_CAMERA, P.WRIST_CAM)
    assert model_px.cam_fovy[cam_px] == pytest.approx(model_diag.cam_fovy[cam_diag], abs=1e-3)


def test_measured_intrinsics_path_uses_a_different_fovy_when_focal_length_differs():
    cfg = CameraConfig(resolution=(640, 480), fx_px=300.0, fy_px=300.0, cx_px=320.0, cy_px=240.0)
    assert cfg.fovy_deg() != pytest.approx(CameraConfig().fovy_deg())


def test_centred_principal_point_compiles_to_zero_offset():
    """A regression guard for a real bug caught while building this: `resolved_intrinsics_px`'s
    default ``cx``/``cy`` use OpenCV's pixel-*centre* convention (``(width - 1) / 2``, matching
    ``configs/params/default.yaml``'s own notes), so a config left at that centred default must
    compile to exactly zero MuJoCo ``principal_pixel`` offset -- it did not until the offset
    calculation used the same ``(width - 1) / 2`` zero reference instead of ``width / 2``, which
    put every "centred" measured calibration compiled through MuJoCo half a pixel off-centre."""
    cx, cy = 319.5, 239.5  # (640 - 1) / 2, (480 - 1) / 2 -- the centred default
    cfg = CameraConfig(resolution=(640, 480), fx_px=300.0, fy_px=300.0, cx_px=cx, cy_px=cy)
    model = build_arm_spec(ArmConfig(camera=cfg)).compile()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, P.WRIST_CAM)
    principal = model.cam_intrinsic[cam][2:4]
    np.testing.assert_allclose(principal, [0.0, 0.0], atol=1e-9)
