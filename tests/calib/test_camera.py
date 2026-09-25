"""Fast, MuJoCo-free tests for `brick_builder.calib.camera`'s pure algorithms.

Everything here runs on synthetic geometry (`cv2.projectPoints`, hand-built rotation matrices) --
no rendering, so this file is fast enough to run outside `-m "not slow"`. The simulated,
rendering-based validation (15+ known arm poses, a perturbed pose, a perturbed field of view) is
`test_camera_sim.py`, marked `@pytest.mark.slow`.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from brick_builder.calib import camera as C


def _rand_unit_quat(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def _homog(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = t
    return out


def _inv(t: np.ndarray) -> np.ndarray:
    rot, pos = t[:3, :3], t[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ pos
    return out


# --------------------------------------------------------------------------------- conventions ---


def test_quat_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(30):
        q = _rand_unit_quat(rng)
        rot = C.quat_wxyz_to_matrix(q)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)
        q2 = C.matrix_to_quat_wxyz(rot)
        rot2 = C.quat_wxyz_to_matrix(q2)
        assert np.allclose(rot, rot2, atol=1e-8)


def test_angle_between_rotations_matches_known_angles():
    identity = np.eye(3)
    assert C.angle_between_rotations_deg(identity, identity) == pytest.approx(0.0, abs=1e-9)
    rot_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert C.angle_between_rotations_deg(identity, rot_z90) == pytest.approx(90.0, abs=1e-6)
    rot_z180 = rot_z90 @ rot_z90
    assert C.angle_between_rotations_deg(identity, rot_z180) == pytest.approx(180.0, abs=1e-6)


def test_cv_rotation_to_mujoco_is_an_involution_and_changes_the_rotation():
    rng = np.random.default_rng(1)
    rot_cv = C.quat_wxyz_to_matrix(_rand_unit_quat(rng))
    rot_mj = C.cv_rotation_to_mujoco(rot_cv)
    assert not np.allclose(rot_cv, rot_mj)  # the 180 degree x-flip is not the identity
    assert np.allclose(C.cv_rotation_to_mujoco(rot_mj), rot_cv, atol=1e-9)


def test_cv_to_mujoco_reproduces_the_documented_wrist_cam_relationship():
    """`docs/guide/03`'s own worked example: a camera looking straight down its OpenCV +z is the
    same physical direction as a MuJoCo camera looking down its own -z, once you also flip +y."""
    rot_cv = np.eye(3)  # OpenCV: +x right, +y down, +z forward (into the scene)
    rot_mj = C.cv_rotation_to_mujoco(rot_cv)
    np.testing.assert_allclose(rot_mj, np.diag([1.0, -1.0, -1.0]))


# --------------------------------------------------------------------------------- checkerboard ---


def test_object_points_are_centred_and_evenly_spaced():
    target = C.CheckerboardTarget(pattern=(9, 6), square_m=0.015)
    pts = target.object_points()
    assert pts.shape == (54, 3)
    assert np.allclose(pts[:, 2], 0.0)
    assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-12)
    xs = np.unique(np.round(pts[:, 0], 9))
    assert len(xs) == 9
    assert np.allclose(np.diff(xs), 0.015)


def test_rotated_180_is_its_own_inverse_and_not_a_mirror():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, size=(11, 9, 3), dtype=np.uint8)
    once = C.rotated_180(img)
    twice = C.rotated_180(once)
    assert np.array_equal(twice, img)
    assert not np.array_equal(once, img)
    # a genuine 180 degree rotation, not a left-right mirror: the top-left pixel lands opposite
    assert np.array_equal(once[0, 0], img[-1, -1])


# --------------------------------------------------------------------------------- intrinsics ---


def _synthetic_camera_matrix(width=640, height=480, fx=255.0):
    return np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def _project(objp, rvec, tvec, camera_matrix, dist=None):
    dist = np.zeros(5) if dist is None else dist
    pts, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist)
    return pts.reshape(-1, 2)


def _checkerboard_template(target: C.CheckerboardTarget, px_per_square: int = 80) -> np.ndarray:
    """A flat, head-on checkerboard image (grayscale, black/white squares): the "physical" board
    texture :func:`_synthetic_checkerboard_image` warps into a camera view below."""
    cols, rows = target.pattern
    squares_x, squares_y = cols + 1, rows + 1
    img = np.full((squares_y * px_per_square, squares_x * px_per_square), 255, dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 1:
                img[r * px_per_square:(r + 1) * px_per_square, c * px_per_square:(c + 1) * px_per_square] = 0
    return img


def _synthetic_checkerboard_image(
    target: C.CheckerboardTarget, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray,
    width: int = 640, height: int = 480, px_per_square: int = 80,
) -> np.ndarray:
    """A synthetic, undistorted photograph of ``target`` at pose ``(rvec, tvec)`` -- a real image
    `cv2.findChessboardCorners` has to detect a pattern in, not pre-computed corner coordinates it
    cannot tell apart from real ones. Perspective-warps :func:`_checkerboard_template` with the
    same board-plane homography (``H = K [r1 r2 t]``) `cv2.projectPoints` uses for the four square
    corners; every interior pixel in between, not just the internal corners, is correctly warped.
    """
    template = _checkerboard_template(target, px_per_square)
    squares_x, squares_y = target.pattern[0] + 1, target.pattern[1] + 1
    scale = target.square_m / px_per_square
    template_to_board = np.array([
        [scale, 0.0, -0.5 * squares_x * target.square_m],
        [0.0, scale, -0.5 * squares_y * target.square_m],
        [0.0, 0.0, 1.0],
    ])
    rot, _ = cv2.Rodrigues(rvec)
    board_to_image = camera_matrix @ np.column_stack([rot[:, 0], rot[:, 1], np.asarray(tvec, dtype=float)])
    return cv2.warpPerspective(
        template, board_to_image @ template_to_board, (width, height),
        flags=cv2.INTER_LINEAR, borderValue=180,
    )


def test_calibrate_intrinsics_recovers_a_known_camera_matrix_from_synthetic_corners():
    target = C.CheckerboardTarget()
    true_k = _synthetic_camera_matrix(fx=260.0)
    rng = np.random.default_rng(3)

    images = []
    for _ in range(14):
        rvec = rng.normal(scale=0.15, size=3)
        rvec[2] = 0.0  # keep the board roughly facing the camera
        tvec = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(0.15, 0.3)])
        images.append(_synthetic_checkerboard_image(target, rvec, tvec, true_k))

    result = C.calibrate_intrinsics(images, target=target)
    assert len(result.used_image_indices) >= 10
    assert result.rms_reprojection_px < 2.0
    assert result.fx == pytest.approx(true_k[0, 0], rel=0.03)
    assert result.fy == pytest.approx(true_k[1, 1], rel=0.03)


def test_calibrate_intrinsics_raises_below_three_usable_images():
    blank = [np.full((480, 640, 3), 200, dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="need at least 3"):
        C.calibrate_intrinsics(blank)


def test_solve_checkerboard_pose_recovers_a_known_pose():
    target = C.CheckerboardTarget()
    objp = target.object_points().astype(np.float64)
    k = _synthetic_camera_matrix()
    true_rvec = np.array([0.05, -0.03, 0.02])
    true_tvec = np.array([0.01, -0.01, 0.22])
    corners = _project(objp, true_rvec, true_tvec, k)

    rot_true, _ = cv2.Rodrigues(true_rvec)
    rot, t = C.solve_checkerboard_pose(corners, k, target=target)
    assert C.angle_between_rotations_deg(rot_true, rot) < 0.5
    assert np.linalg.norm(t - true_tvec) < 1e-3


def test_solve_checkerboard_pose_reversed_corner_order_gives_a_different_self_consistent_pose():
    """Documents the limit `solve_checkerboard_pose`'s own docstring states: reversing which
    detected corner is "first" is a 180-degree relabelling *about the board's own normal*, which a
    single image's geometry alone cannot tell apart from the true pose -- both fits are internally
    consistent (near-zero residual), just physically different by construction. Resolving it needs
    more than one image (`check_sensor_flip`, cross-image hand-eye consistency), not a per-image
    heuristic."""
    target = C.CheckerboardTarget()
    objp = target.object_points().astype(np.float64)
    k = _synthetic_camera_matrix()
    true_rvec = np.array([0.05, -0.03, 0.02])
    true_tvec = np.array([0.01, -0.01, 0.22])
    corners = _project(objp, true_rvec, true_tvec, k)
    rot_true, _ = cv2.Rodrigues(true_rvec)

    rot_rev, t_rev = C.solve_checkerboard_pose(corners[::-1].copy(), k, target=target)
    # Reprojecting the *reversed* corners through the reversed-order fit is itself accurate --
    # this is a correct fit to a relabelled problem, not a numerical failure.
    reproj = _project(objp, cv2.Rodrigues(rot_rev)[0], t_rev, k)
    assert np.max(np.linalg.norm(reproj - corners[::-1], axis=1)) < 1e-6
    # ... yet it disagrees with the true pose by (up to numerical slack) a rotation about the
    # board's own normal -- not the same pose, and not a random one either.
    assert C.angle_between_rotations_deg(rot_true, rot_rev) > 90.0


# --------------------------------------------------------------------------------- hand-eye ---


def _synthetic_hand_eye_case(rng, n=15):
    r_c2g_true = C.quat_wxyz_to_matrix(_rand_unit_quat(rng))
    t_c2g_true = np.array([0.0025, 0.07344, 0.00659])
    t_c2g = _homog(r_c2g_true, t_c2g_true)

    r_tgt2base = C.quat_wxyz_to_matrix(_rand_unit_quat(rng))
    t_tgt2base = _homog(r_tgt2base, np.array([0.2, 0.0, 0.0]))

    gripper_to_base, target_to_cam = [], []
    for _ in range(n):
        r_g2b = C.quat_wxyz_to_matrix(_rand_unit_quat(rng))
        t_g2b_vec = rng.uniform(-0.05, 0.05, size=3) + np.array([0.0, 0.0, 0.3])
        t_g2b = _homog(r_g2b, t_g2b_vec)
        t_c2b = t_g2b @ t_c2g
        t_tgt2cam = _inv(t_c2b) @ t_tgt2base
        gripper_to_base.append((r_g2b, t_g2b_vec))
        target_to_cam.append((t_tgt2cam[:3, :3], t_tgt2cam[:3, 3]))
    return r_c2g_true, t_c2g_true, gripper_to_base, target_to_cam


def test_calibrate_hand_eye_recovers_a_known_camera_to_gripper_transform_noiseless():
    rng = np.random.default_rng(4)
    r_true, t_true, g2b, t2c = _synthetic_hand_eye_case(rng)
    result = C.calibrate_hand_eye(g2b, t2c, method="tsai")
    assert C.angle_between_rotations_deg(result.rotation_cv, r_true) < 0.1
    assert np.linalg.norm(result.translation_m_cv - t_true) < 1e-4


def test_calibrate_hand_eye_is_robust_to_small_pose_noise():
    rng = np.random.default_rng(5)
    r_true, t_true, g2b, t2c = _synthetic_hand_eye_case(rng, n=20)
    noisy = []
    for rot, t in t2c:
        wiggle = C.quat_wxyz_to_matrix(np.array([1.0, *rng.normal(scale=0.002, size=3)]))
        noisy.append((rot @ wiggle, t + rng.normal(scale=0.0008, size=3)))
    result = C.calibrate_hand_eye(g2b, noisy, method="tsai")
    assert C.angle_between_rotations_deg(result.rotation_cv, r_true) < 1.0
    assert np.linalg.norm(result.translation_m_cv - t_true) * 1000.0 < 2.0  # mm


def test_calibrate_hand_eye_rejects_too_few_poses():
    rng = np.random.default_rng(6)
    _, _, g2b, t2c = _synthetic_hand_eye_case(rng, n=2)
    with pytest.raises(ValueError, match="at least 3"):
        C.calibrate_hand_eye(g2b, t2c)


def test_calibrate_hand_eye_rejects_mismatched_lengths():
    rng = np.random.default_rng(7)
    _, _, g2b, t2c = _synthetic_hand_eye_case(rng, n=5)
    with pytest.raises(ValueError, match="same length"):
        C.calibrate_hand_eye(g2b, t2c[:-1])


# --------------------------------------------------------------------------------- flip check ---


def test_check_sensor_flip_detects_a_180_degree_optical_axis_rotation():
    rng = np.random.default_rng(8)
    expected_quat = _rand_unit_quat(rng)
    expected_rot = C.quat_wxyz_to_matrix(expected_quat)

    good = C.check_sensor_flip(expected_rot, expected_quat)
    assert not good.likely_flipped
    assert good.angle_to_expected_deg < 1e-6

    flipped_rot = expected_rot @ np.diag([-1.0, -1.0, 1.0])
    bad = C.check_sensor_flip(flipped_rot, expected_quat)
    assert bad.likely_flipped
    assert bad.angle_to_expected_flipped_deg < 1e-6


def test_check_sensor_flip_not_fooled_by_an_unrelated_small_error():
    rng = np.random.default_rng(9)
    expected_quat = _rand_unit_quat(rng)
    expected_rot = C.quat_wxyz_to_matrix(expected_quat)
    small_error = C.quat_wxyz_to_matrix(np.array([1.0, *rng.normal(scale=0.01, size=3)]))
    result = C.check_sensor_flip(expected_rot @ small_error, expected_quat)
    assert not result.likely_flipped
    assert result.angle_to_expected_deg < 5.0


# --------------------------------------------------------------------------------- fragment ---


def test_parameter_fragment_round_trips_through_yaml_with_the_right_keys():
    yaml = pytest.importorskip("yaml")
    intrinsics = C.IntrinsicsResult(
        camera_matrix=_synthetic_camera_matrix(fx=254.8),
        dist_coeffs=np.array([-0.2, 0.03, 0.001, -0.0005, 0.0]),
        rms_reprojection_px=0.31,
        image_size=(640, 480),
        per_view_corners=[], rvecs=[], tvecs=[],
        used_image_indices=list(range(18)),
    )
    hand_eye = C.HandEyeResult(
        rotation_cv=C.quat_wxyz_to_matrix(np.array([0.9, 0.1, 0.2, 0.3])),
        translation_m_cv=np.array([0.0025, 0.07344, 0.00659]),
        n_poses=18, method="tsai", used_cv2_binding=False,
    )
    text = C.parameter_fragment(intrinsics, hand_eye, measured_by="unit-test", measured_on="2026-09-25")
    data = yaml.safe_load(text)
    cam = data["camera"]
    for key in (
        "fx_px", "fy_px", "cx_px", "cy_px", "resolution_width_px", "resolution_height_px",
        "opencv_k1", "opencv_k2", "opencv_p1", "opencv_p2", "opencv_k3",
        "mount_x_mm", "mount_y_mm", "mount_z_mm", "mount_roll_deg", "mount_pitch_deg", "mount_yaw_deg",
    ):
        assert key in cam, key
        assert cam[key]["status"] == "measured"
    assert cam["fx_px"]["value"] == pytest.approx(254.8)
    assert cam["resolution_width_px"]["value"] == 640
    # The provenance note names what actually solved AX = XB -- not just echoing the requested
    # method as if cv2.calibrateHandEye always ran it (see HandEyeResult.solver_description).
    assert hand_eye.solver_description in cam["mount_x_mm"]["notes"]


def test_parameter_fragment_without_hand_eye_omits_mount_keys():
    yaml = pytest.importorskip("yaml")
    intrinsics = C.IntrinsicsResult(
        camera_matrix=_synthetic_camera_matrix(),
        dist_coeffs=np.zeros(5),
        rms_reprojection_px=0.2,
        image_size=(640, 480),
        per_view_corners=[], rvecs=[], tvecs=[], used_image_indices=[0, 1, 2],
    )
    text = C.parameter_fragment(intrinsics)
    data = yaml.safe_load(text)
    assert "mount_x_mm" not in data["camera"]


# --------------------------------------------------------------------------------- params bridge ---


def test_euler_xyz_round_trips_through_matrix():
    rng = np.random.default_rng(10)
    for _ in range(20):
        roll, pitch, yaw = rng.uniform(-45, 45, size=3)
        rot = C._euler_xyz_deg_to_matrix(roll, pitch, yaw)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        roll2, pitch2, yaw2 = C._euler_xyz_deg_from_matrix(rot)
        assert roll2 == pytest.approx(roll, abs=1e-6)
        assert pitch2 == pytest.approx(pitch, abs=1e-6)
        assert yaw2 == pytest.approx(yaw, abs=1e-6)


def test_euler_xyz_pure_roll_matches_the_documented_wrist_cam_quat_convention():
    """`configs/params/default.yaml`'s own `mount_roll_deg: -25.073` (pitch=yaw=0) claims this
    reproduces `WRIST_CAM_QUAT_IN_GRIPPER` -- checked directly, not just asserted in a comment."""
    from brick_builder.robot.patch_so101 import WRIST_CAM_QUAT_IN_GRIPPER

    rot = C._euler_xyz_deg_to_matrix(-25.073, 0.0, 0.0)
    quat = C.matrix_to_quat_wxyz(rot)
    assert quat == pytest.approx(WRIST_CAM_QUAT_IN_GRIPPER, abs=1e-4)


def test_camera_config_from_params_reproduces_the_shipped_default():
    """The shipped `configs/params/default.yaml` camera block was written to match
    `CameraConfig()`'s own defaults exactly (`fx_px`/etc. at `status: assumed`, derived from
    `diagonal_fov_deg`, not a real calibration -- see the file's own notes fields); loading it must
    reproduce the default `CameraConfig` exactly, not merely something with the same numbers."""
    from brick_builder import params as P
    from brick_builder.robot.patch_so101 import CameraConfig

    pf = P.load_params(P.DEFAULT_PARAMS_PATH)
    cfg = C.camera_config_from_params(pf)
    default = CameraConfig()
    assert not cfg.uses_measured_intrinsics
    # Field by field with a tolerance, not `cfg == default`: the file's numbers went through a
    # YAML round trip and an Euler-angle-to-quaternion conversion, `default`'s did not, and a
    # `.225`-class literal is not bit-identical to the same value computed as `0.5 * (-0.35 - 0.10)`
    # even though both are "the same number" to any precision anyone would ever type or read.
    assert cfg.resolution == default.resolution
    assert cfg.diagonal_fov_deg == pytest.approx(default.diagonal_fov_deg)
    assert cfg.distortion_k1 == pytest.approx(default.distortion_k1)
    assert cfg.distortion_k2 == pytest.approx(default.distortion_k2)
    assert cfg.pos_in_gripper == pytest.approx(default.pos_in_gripper, abs=1e-9)
    assert cfg.quat_in_gripper == pytest.approx(default.quat_in_gripper, abs=1e-6)
    assert cfg.fovy_deg() == pytest.approx(default.fovy_deg())


def test_camera_config_from_params_switches_to_measured_intrinsics_only_when_measured(tmp_path):
    """A member's terse override (`camera: {fx_px: 300.0, ...}`, a bare scalar -- `status:
    measured` by `brick_builder.params`'s own shorthand convention) must switch the built config
    onto the measured-intrinsics path; the shipped default file's own `assumed` values (previous
    test) must not."""
    from brick_builder import params as P

    override = tmp_path / "override.yaml"
    override.write_text(
        "camera:\n  fx_px: 300.0\n  fy_px: 300.0\n  cx_px: 330.0\n  cy_px: 250.0\n"
    )
    pf = P.load_params(override)
    assert pf.get("camera.fx_px").status == "measured"
    cfg = C.camera_config_from_params(pf)
    assert cfg.uses_measured_intrinsics
    assert cfg.resolved_intrinsics_px() == pytest.approx((300.0, 300.0, 330.0, 250.0))


def test_to_arm_config_measured_intrinsics_file_reaches_the_compiled_model(tmp_path):
    """`brick_builder.params.to_arm_config`'s `camera` field is not just a Python-side dataclass:
    a measured-intrinsics override must switch the *compiled* wrist camera onto MuJoCo's
    pixel-intrinsics model too, with a checkably different frustum from the default."""
    import mujoco

    from brick_builder import params as P
    from brick_builder.robot.patch_so101 import ArmConfig, WRIST_CAM, build_arm_spec

    override = tmp_path / "override.yaml"
    override.write_text(
        "camera:\n  fx_px: 300.0\n  fy_px: 300.0\n  cx_px: 330.0\n  cy_px: 250.0\n"
    )
    pf = P.load_params(override)
    arm_cfg = P.to_arm_config(pf)
    assert arm_cfg.camera.uses_measured_intrinsics

    model = build_arm_spec(arm_cfg).compile()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    assert tuple(model.cam_sensorsize[cam]) != (0.0, 0.0)
    # 300 px focal is longer than the 254.828 px default -> a narrower, checkably different fovy,
    # not just a different Python-side CameraConfig that never reached the compiled model.
    default_model = build_arm_spec(ArmConfig()).compile()
    default_cam = mujoco.mj_name2id(default_model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    assert model.cam_fovy[cam] < default_model.cam_fovy[default_cam]
