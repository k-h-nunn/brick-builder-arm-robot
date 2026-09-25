"""Simulated calibration validation: a test-only checkerboard scene, 15+ known arm poses rendered
with a known `CameraConfig`, a perturbed pose and a perturbed field of view, and the recovery
errors `brick_builder.calib.camera` achieves against that known truth.

Every test here renders with MuJoCo and is therefore `@pytest.mark.slow`; the pure-algorithm tests
(`test_camera.py`) and the pixel-identity proof (`test_camera_config.py`) are the fast subset.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from brick_builder.calib import camera as calib
from brick_builder.calib import camera_sim as sim
from brick_builder.robot.patch_so101 import CameraConfig

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------------- the scene ---


def test_checkerboard_scene_compiles_and_is_never_reachable_from_the_default_scene_builder():
    spec = sim.build_checkerboard_spec()
    model = spec.compile()
    assert model.nbody > 1
    import brick_builder.scene.compose as compose_module
    # The test-only body name must not appear anywhere in the real scene builder's source --
    # a cheap, direct check that this fixture was never wired in there.
    import inspect
    assert sim.CHECKER_BODY not in inspect.getsource(compose_module)


def test_generate_tool_down_poses_returns_the_requested_count_and_valid_ranges():
    spec = sim.build_checkerboard_spec()
    model = spec.compile()
    rng = np.random.default_rng(0)
    qs = sim.generate_tool_down_poses(model, 5, rng)
    assert len(qs) == 5
    for q in qs:
        assert q.shape == (5,)
        assert np.all(np.isfinite(q))


# --------------------------------------------------------------------------------- rendering ---


def test_render_calibration_set_produces_rgb_frames_and_ground_truth_poses():
    calib_set = sim.render_calibration_set(n_poses=6, seed=0)
    assert len(calib_set.shots) == 6
    for shot in calib_set.shots:
        assert shot.rgb.shape == (480, 640, 3)
        assert shot.rgb.dtype == np.uint8
        r_grip, t_grip = shot.gripper_to_base
        assert r_grip.shape == (3, 3)
        assert np.allclose(r_grip @ r_grip.T, np.eye(3), atol=1e-5)
        assert t_grip.shape == (3,)
        r_cam, t_cam = shot.camera_to_base_truth
        assert r_cam.shape == (3, 3)
        # The camera is offset from (not coincident with) the gripper body origin.
        assert np.linalg.norm(t_cam - t_grip) > 1e-4


def test_render_calibration_set_finds_the_checkerboard_in_a_meaningful_fraction_of_shots():
    """A close hover *and* a tilted (not just straight-down) tool pose -- needed for hand-eye
    conditioning, see `generate_tool_down_poses`'s docstring -- means many poses lose the board's
    outermost row/column to the gripper's own silhouette (see `render_calibration_set`'s
    docstring). This is a floor on the yield rate, not a claim of near-100%."""
    calib_set = sim.render_calibration_set(n_poses=40, seed=1)
    found = sum(calib.find_corners(shot.rgb, calib_set.target) is not None for shot in calib_set.shots)
    assert found >= 8, f"only {found}/40 renders had a detectable checkerboard"


# --------------------------------------------------------------------------------- recovery: default ---


@pytest.fixture(scope="module")
def default_report() -> sim.RecoveryReport:
    return sim.validate_recovery(CameraConfig(), n_poses=65, seed=0, label="default")


def test_default_camera_recovery_finds_enough_poses(default_report: sim.RecoveryReport):
    assert default_report.n_corners_found >= 15


def test_default_camera_recovery_reprojection_error_is_sub_pixel(default_report: sim.RecoveryReport):
    assert default_report.reprojection_rms_px < 1.0


def test_default_camera_recovery_focal_length_error_is_small(default_report: sim.RecoveryReport):
    assert default_report.focal_error_pct < 3.0


def test_default_camera_recovery_position_error_is_a_few_millimetres(default_report: sim.RecoveryReport):
    assert default_report.position_error_mm < 8.0


def test_default_camera_recovery_rotation_error_is_a_couple_degrees(default_report: sim.RecoveryReport):
    # Measured 4.05 deg at seed 0 (16 usable poses -- just above the 15-pose floor); the bound
    # carries margin for run-to-run pose-set variance, not because 4.05 itself is marginal.
    assert default_report.rotation_error_deg < 6.0


def test_default_camera_recovery_is_not_flagged_as_a_180_degree_flip(default_report: sim.RecoveryReport):
    assert not default_report.flip.likely_flipped


# --------------------------------------------------------------------------------- recovery: perturbed pose ---


def _perturbed_pose_camera_cfg() -> CameraConfig:
    base = CameraConfig()
    pos = np.array(base.pos_in_gripper) + np.array([0.004, -0.003, 0.002])  # a few mm
    wiggle = calib.quat_wxyz_to_matrix(np.array([1.0, np.radians(1.0), np.radians(-0.7), np.radians(0.5)]))
    rot = calib.quat_wxyz_to_matrix(np.array(base.quat_in_gripper)) @ wiggle  # a couple of degrees
    quat = calib.matrix_to_quat_wxyz(rot)
    return dataclasses.replace(base, pos_in_gripper=tuple(pos), quat_in_gripper=tuple(quat))


@pytest.fixture(scope="module")
def perturbed_pose_report() -> sim.RecoveryReport:
    return sim.validate_recovery(
        _perturbed_pose_camera_cfg(), n_poses=65, seed=1, label="perturbed pose (mm/deg)"
    )


@pytest.fixture(scope="module")
def perturbed_pose_calibration_set() -> sim.CalibrationSet:
    return sim.render_calibration_set(camera_cfg=_perturbed_pose_camera_cfg(), n_poses=65, seed=1)


def test_perturbed_pose_recovery_still_finds_enough_poses(perturbed_pose_report: sim.RecoveryReport):
    assert perturbed_pose_report.n_corners_found >= 15


def test_perturbed_pose_recovery_reprojection_error(perturbed_pose_report: sim.RecoveryReport):
    """Measured 0.528 px at seed 1, in the same range as the default/perturbed-fov cases' own
    0.594/1.003 px, not the outlier a fixed 11x11 ``cornerSubPix`` window once made it (1.021 px,
    against 0.668/0.742 px for the other two): ``find_corners`` now sizes that window from each
    image's own detected corner spacing (``brick_builder.calib.camera._corner_subpix_window``),
    which removed the excess error a fixed window left on this pose set's foreshortened far rows."""
    assert perturbed_pose_report.reprojection_rms_px < 1.2


def test_perturbed_pose_recovery_focal_length_error(perturbed_pose_report: sim.RecoveryReport):
    """Measured 0.59% at seed 1, down from 6.15% before ``find_corners``' corner-window fix -- see
    ``test_perturbed_pose_recovery_reprojection_error``."""
    assert perturbed_pose_report.focal_error_pct < 3.0


def test_perturbed_pose_recovery_position_error_is_bounded(perturbed_pose_report: sim.RecoveryReport):
    """Measured 5.15 mm at seed 1, in the same range as the default/perturbed-fov cases' own
    7.46/6.46 mm, not the outlier it was before ``find_corners``' corner-window fix (31.85 mm,
    against 5.85/5.42 mm for the other two). ``solve_checkerboard_pose`` uses the fitted (not the
    true) camera matrix to get each view's checkerboard pose, so corner-detection error reaches
    both the intrinsics fit and, through it, the hand-eye solve; a fixed 11x11 ``cornerSubPix``
    window overran neighbouring corners on this pose set's foreshortened far rows, only ~10-15 px
    apart, degrading exactly that chain (``test_perturbed_pose_hand_eye_solve_is_accurate_given_the_true_intrinsics``
    below isolates the hand-eye solve itself from the intrinsics fit and confirms the solve was
    never the weak step). This is not a claim that a genuine few-mm mount offset is resolved (the
    module-level caveat below): the perturbation being tested here is itself only 5.4 mm.
    """
    assert perturbed_pose_report.position_error_mm < 12.0


def test_perturbed_pose_recovery_rotation_error_is_a_few_degrees(perturbed_pose_report: sim.RecoveryReport):
    """Measured 0.451 deg at seed 1, down from 1.515 deg before ``find_corners``' corner-window
    fix."""
    assert perturbed_pose_report.rotation_error_deg < 2.0


def test_perturbed_pose_hand_eye_solve_is_accurate_given_the_true_intrinsics(
    perturbed_pose_calibration_set: sim.CalibrationSet,
):
    """Isolates the hand-eye ``AX = XB`` solve from the intrinsics fit that feeds it.

    Re-detects the same rendered corners and re-solves each used view's checkerboard pose with
    the known-true camera matrix (instead of this pose set's own fitted one), then re-runs
    hand-eye, and checks the result against the injected perturbation itself
    (``_perturbed_pose_camera_cfg`` vs the un-perturbed ``CameraConfig()``, roughly 5.4 mm / 2.6
    deg) rather than a fixed constant: a bound with no reference to the perturbation being tested
    (a former ``< 10.0`` mm / ``< 3.0`` deg here was looser than the perturbation itself) would
    still pass if this solve returned the *unperturbed* pose -- the fixed-recovery outcome this
    test exists to rule out. Measured 0.762 mm / 0.320 deg -- about 14%/12% of the injected
    perturbation left, comfortably inside the 30% floor below, and confirming the closed-form
    solve itself was never the weak step once given accurate corners (see
    ``test_perturbed_pose_recovery_position_error_is_bounded``, whose 31.85 mm pre-fix number came
    from corner-detection error reaching the *fitted* intrinsics this test bypasses). Not a claim
    that this pipeline resolves a genuine few-mm mount offset (the module-level caveat below): the
    perturbation itself is only 5.4 mm, and CAD-nominal prior uncertainty is of the same order.
    """
    calib_set = perturbed_pose_calibration_set
    cfg = calib_set.camera_cfg
    fx, fy, cx, cy = cfg.resolved_intrinsics_px()
    true_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    gripper_to_base, target_to_cam = [], []
    for shot in calib_set.shots:
        corners = calib.find_corners(shot.rgb, calib_set.target)
        if corners is None:
            continue
        rot, tvec = calib.solve_checkerboard_pose(corners, true_k, target=calib_set.target)
        gripper_to_base.append(shot.gripper_to_base)
        target_to_cam.append((rot, tvec))
    assert len(gripper_to_base) >= 15

    hand_eye = calib.calibrate_hand_eye(gripper_to_base, target_to_cam, method="tsai")
    r_true, t_true = sim._camera_to_gripper_truth(cfg)
    position_error_mm = float(np.linalg.norm(hand_eye.translation_m_mujoco() - t_true) * 1000.0)
    rotation_error_deg = calib.angle_between_rotations_deg(r_true, hand_eye.rotation_mujoco())

    r_nominal, t_nominal = sim._camera_to_gripper_truth(CameraConfig())
    position_perturbation_mm = float(np.linalg.norm(t_true - t_nominal) * 1000.0)
    rotation_perturbation_deg = calib.angle_between_rotations_deg(r_nominal, r_true)

    # 30% of the injected perturbation, not 90%: measured 14%/12% left at seed 1 (docstring
    # above), so 30% keeps real margin for run-to-run pose-set variance while staying far below
    # the loophole a looser-than-the-perturbation bound left open (returning the unperturbed pose
    # scores 100%).
    assert position_error_mm < 0.3 * position_perturbation_mm, (
        f"{position_error_mm:.2f} mm recovered vs. a {position_perturbation_mm:.2f} mm "
        "perturbation -- no better than returning the unperturbed pose"
    )
    assert rotation_error_deg < 0.3 * rotation_perturbation_deg, (
        f"{rotation_error_deg:.3f} deg recovered vs. a {rotation_perturbation_deg:.3f} deg "
        "perturbation -- no better than returning the unperturbed pose"
    )


# --------------------------------------------------------------------------------- recovery: perturbed fov ---


@pytest.fixture(scope="module")
def perturbed_fov_report() -> sim.RecoveryReport:
    cfg = dataclasses.replace(CameraConfig(), diagonal_fov_deg=100.0)  # inside WRIST_CAM_DIAGONAL_FOV_RANGE_DEG
    return sim.validate_recovery(cfg, n_poses=65, seed=2, label="perturbed fov (100 deg)")


def test_perturbed_fov_recovery_finds_enough_poses(perturbed_fov_report: sim.RecoveryReport):
    assert perturbed_fov_report.n_corners_found >= 15


def test_perturbed_fov_recovery_focal_length_error_is_small(perturbed_fov_report: sim.RecoveryReport):
    assert perturbed_fov_report.focal_error_pct < 3.0


def test_perturbed_fov_recovery_is_distinguishable_from_the_default_fov(
    default_report: sim.RecoveryReport, perturbed_fov_report: sim.RecoveryReport,
):
    """The whole point of this case: a wrong assumed fov, not the default one, must be what
    actually gets *recovered* -- otherwise a bug that silently ignores `diagonal_fov_deg` (e.g.
    always calibrating against the default fov regardless of `camera_cfg`) would still pass the
    error-threshold tests above (both fovs are within a few percent of *something* plausible)."""
    default_fx, default_fy, _, _ = CameraConfig().resolved_intrinsics_px()
    perturbed_fx, perturbed_fy, _, _ = CameraConfig(diagonal_fov_deg=100.0).resolved_intrinsics_px()
    assert abs(perturbed_fx - default_fx) / default_fx > 0.05

    # The *recovered* focal length actually tracks its own case's ground truth, not the other
    # case's -- not just "within a few percent of something plausible" for both.
    assert default_report.recovered_fx_px == pytest.approx(default_fx, rel=0.03)
    assert perturbed_fov_report.recovered_fx_px == pytest.approx(perturbed_fx, rel=0.03)
    assert abs(perturbed_fov_report.recovered_fx_px - default_fx) / default_fx > 0.05


# --------------------------------------------------------------------------------- recovery: off-centre principal point ---


def test_off_centre_principal_point_lands_on_the_requested_side():
    """Regression guard for a real sign bug caught while building this: MuJoCo's
    ``principal_pixel`` offsets the sensor, not the image, so a positive offset shifts the rendered
    image the *opposite* way on both axes (see ``patch_so101._add_sites_and_camera``'s own
    comment) -- passing the offset straight through unnegated compiled a camera whose rendered
    principal point was at ``(width - 1) - cx``/``(height - 1) - cy``, the mirror image of the
    requested ``cx``/``cy`` on each axis, for any off-centre measured calibration.

    Renders a checkerboard with both ``cx`` and ``cy`` moved off centre (``fx``/``fy`` at the
    default focal length so this is otherwise an ordinary measured-intrinsics config) and checks
    the detected corners land close to the OpenCV pinhole projection of their known 3D positions
    using the *requested* ``cx``/``cy`` on both axes -- not close to the projection using the
    mirrored ``(width - 1) - cx``/``(height - 1) - cy``, which is what the pre-fix sign produced.
    """
    import cv2

    cfg = CameraConfig(fx_px=254.828, fy_px=254.828, cx_px=340.0, cy_px=260.0)
    calib_set = sim.render_calibration_set(camera_cfg=cfg, n_poses=8, seed=0)
    shot = next(s for s in calib_set.shots if calib.find_corners(s.rgb, calib_set.target) is not None)
    corners = calib.find_corners(shot.rgb, calib_set.target)

    r_board, t_board = sim.board_pose_world(board_xy=calib_set.board_xy)
    r_cam_mujoco, t_cam_mujoco = shot.camera_to_base_truth
    r_cam_cv = r_cam_mujoco @ calib.FLIP_CV_TO_MUJOCO  # MuJoCo -> OpenCV camera convention
    objp_world = (r_board @ calib_set.target.object_points().T).T + t_board
    objp_cam = (r_cam_cv.T @ (objp_world - t_cam_mujoco).T).T

    fx, fy, cx, cy = cfg.resolved_intrinsics_px()
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    projected, _ = cv2.projectPoints(objp_cam, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(5))
    mean_offset_px = (corners - projected.reshape(-1, 2)).mean(axis=0)

    mirrored_cx = (cfg.resolution[0] - 1) - cx
    mirrored_cy = (cfg.resolution[1] - 1) - cy
    mirror_offset_x_px = mirrored_cx - cx  # ~41 px: what the pre-fix sign error produced
    mirror_offset_y_px = mirrored_cy - cy  # ~41 px, same bug on the y axis

    assert abs(mean_offset_px[0]) < 2.0, (
        f"principal point landed {mean_offset_px[0]:+.1f} px from the requested cx "
        f"(the pre-fix mirror-image bug would show ~{-mirror_offset_x_px:+.1f} px here)"
    )
    assert abs(mean_offset_px[1]) < 2.0, (
        f"principal point landed {mean_offset_px[1]:+.1f} px from the requested cy "
        f"(the pre-fix mirror-image bug would show ~{-mirror_offset_y_px:+.1f} px here)"
    )


# --------------------------------------------------------------------------------- 180-degree flip ---


def test_flipped_images_are_detected_by_the_180_degree_flip_check():
    """The flip check, exercised end to end on real rendered images (not just on hand-built
    rotation matrices, see ``test_camera.py``): rotate every image in a rendered set 180 degrees
    -- what a sensor mounted rotated 180 degrees in its socket would produce -- and calibration
    against the flipped set must be flagged by `check_sensor_flip`."""
    cfg = CameraConfig()
    calib_set = sim.render_calibration_set(camera_cfg=cfg, n_poses=90, seed=2)
    flipped_shots = [
        dataclasses.replace(shot, rgb=calib.rotated_180(shot.rgb)) for shot in calib_set.shots
    ]
    flipped_set = dataclasses.replace(calib_set, shots=flipped_shots)

    intrinsics, hand_eye = sim.calibrate_from_calibration_set(flipped_set, hand_eye_method="tsai")
    flip = calib.check_sensor_flip(hand_eye.rotation_mujoco(), np.asarray(cfg.quat_in_gripper))
    assert flip.likely_flipped
    assert flip.angle_to_expected_flipped_deg < flip.angle_to_expected_deg


# --------------------------------------------------------------------------------- report table ---


def test_format_report_table_includes_every_case(
    default_report: sim.RecoveryReport,
    perturbed_pose_report: sim.RecoveryReport,
    perturbed_fov_report: sim.RecoveryReport,
):
    table = sim.format_report_table([default_report, perturbed_pose_report, perturbed_fov_report])
    assert "default" in table
    assert "perturbed pose" in table
    assert "perturbed fov" in table
    assert table.count("\n") >= 4  # header + separator + 3 rows
