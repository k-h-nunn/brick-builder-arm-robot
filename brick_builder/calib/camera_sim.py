"""Simulated wrist-camera calibration validation.

Builds a test-only scene -- a flat printed checkerboard on the table, never wired into
`brick_builder.scene.compose`'s default scene builder -- renders it from the wrist camera at many
known arm poses, and runs it back through `brick_builder.calib.camera`'s calibration routines to
report how well they recover a `CameraConfig` this module already knows exactly. This is the
"validated in simulation" half of the camera-model package; the real-hardware half is
`scripts/calibrate_camera.py` run against photographs, with no MuJoCo import required for that.

The checkerboard is a grid of individual flat box geoms with alternating black/white colour, not
a texture: the physical position of every printed corner is then exactly the shared edge between
two adjacent boxes, computed directly from this module's own square size and grid -- there is no
texture-UV mapping to get subtly wrong between "the board cv2 sees" and "the board whose corners
this module claims ground truth for".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import mujoco
import numpy as np

from . import camera as calib
from ..render.renderer import SceneRenderer
from ..robot.kinematics import ArmKinematics
from ..robot.patch_so101 import ArmConfig, CameraConfig, GRIPPER_BODY, WRIST_CAM, attach_arm

CHECKER_BODY = "calib_checkerboard"
"""Body name of the test-only checkerboard target this module adds to its own scene."""

DEFAULT_BOARD_XY = (0.17, 0.0)
"""Matches `brick_builder.scene.compose.SceneConfig.plate_xy`'s default -- a position every
default scene already proves is inside the SO-101's reach at this base position, not a fact this
module re-derives."""

_TABLE_Z = 0.0
_BOARD_GEOM_HALF_THICKNESS = 0.0005
_BOARD_BODY_CLEARANCE = 0.0006  # board body origin sits this far above the table plane


# --------------------------------------------------------------------------------- the scene ---


def board_top_z(table_z: float = _TABLE_Z) -> float:
    """World z of the checkerboard's printed (top) face -- where every corner's z coordinate is."""
    return table_z + _BOARD_BODY_CLEARANCE + _BOARD_GEOM_HALF_THICKNESS


def board_pose_world(
    board_xy: tuple[float, float] = DEFAULT_BOARD_XY, table_z: float = _TABLE_Z
) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` of the checkerboard's own object frame (see
    ``calib.CheckerboardTarget.object_points``) in the world frame: no rotation, centred at
    ``board_xy`` at :func:`board_top_z`. Ground truth this module controls by construction --
    the board body is added with exactly this pose and never moves."""
    return np.eye(3), np.array([board_xy[0], board_xy[1], board_top_z(table_z)])


def build_checkerboard_spec(
    camera_cfg: CameraConfig = CameraConfig(),
    arm_cfg: ArmConfig | None = None,
    target: calib.CheckerboardTarget = calib.CheckerboardTarget(),
    board_xy: tuple[float, float] = DEFAULT_BOARD_XY,
    table_z: float = _TABLE_Z,
) -> mujoco.MjSpec:
    """A standalone world: a table plane, the SO-101 arm (origin, identity pose -- the same base
    placement every default scene uses), and a flat checkerboard target.

    Test-only: this scene exists for calibration validation alone and is never reachable from
    `brick_builder.scene.compose`. ``camera_cfg`` is threaded onto ``arm_cfg.camera`` (see
    `patch_so101.CameraConfig`/`ArmConfig`); the rest of ``arm_cfg`` defaults to
    `patch_so101.ArmConfig`'s own defaults.
    """
    arm_cfg = ArmConfig(camera=camera_cfg) if arm_cfg is None else replace(arm_cfg, camera=camera_cfg)

    world = mujoco.MjSpec()
    # `castshadow=False`: a diffusely lit workspace (no hard arm shadow crossing the board), the
    # same thing a real calibration setup avoids by not lighting the checkerboard from one harsh
    # point source. Some close-hover poses still lose the outermost corner row to the gripper's
    # own silhouette rather than a shadow (checked directly by pattern-size probing -- see
    # `render_calibration_set`'s docstring); this only removes the shadow contribution.
    world.worldbody.add_light(
        pos=[0.5, 0.4, 0.5], dir=[-0.5, -0.4, -0.6], castshadow=False,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.45, 0.45, 0.45], ambient=[0.35, 0.35, 0.35],
    )
    world.worldbody.add_light(
        pos=[-0.4, -0.3, 0.5], dir=[0.4, 0.3, -0.6], castshadow=False,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.3, 0.3, 0.3], ambient=[0.15, 0.15, 0.15],
    )
    world.worldbody.add_geom(
        name="calib_table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.4, 0.4, 0.01],
        pos=[0.0, 0.0, table_z],
        rgba=[0.5, 0.52, 0.55, 1.0],
    )

    cols, rows = target.pattern
    n_x, n_y = cols + 1, rows + 1
    square = target.square_m
    board = world.worldbody.add_body(
        name=CHECKER_BODY, pos=[board_xy[0], board_xy[1], table_z + _BOARD_GEOM_HALF_THICKNESS]
    )
    for i in range(n_x):
        for j in range(n_y):
            x = (i - (n_x - 1) / 2.0) * square
            y = (j - (n_y - 1) / 2.0) * square
            white = (i + j) % 2 == 0
            board.add_geom(
                name=f"checker_{i}_{j}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, 0.0],
                size=[square / 2.0, square / 2.0, _BOARD_GEOM_HALF_THICKNESS],
                rgba=[0.92, 0.92, 0.90, 1.0] if white else [0.04, 0.04, 0.04, 1.0],
            )
    # A visible border strip a shade darker than the white squares so `findChessboardCorners`
    # (which relies on a quiet margin around the pattern, per its own documentation) has one.
    # Generously sized (a full square): a thin margin leaves the outermost corner row/column
    # right at the board edge, where render antialiasing and the gripper's own silhouette most
    # often cost that row enough contrast/visibility to drop the detection (see
    # `render_calibration_set`'s docstring).
    border = square * 1.3
    board.add_geom(
        name="checker_border",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, -_BOARD_GEOM_HALF_THICKNESS * 0.5],
        size=[n_x * square / 2.0 + border, n_y * square / 2.0 + border, _BOARD_GEOM_HALF_THICKNESS * 0.5],
        rgba=[0.75, 0.75, 0.72, 1.0],
    )

    attach_arm(world, arm_cfg, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    return world


# --------------------------------------------------------------------------------- arm poses ---


_NOMINAL_PITCH = 0.0
_NOMINAL_ROLL = math.pi
"""Tool pointing straight down (`brick_builder.scene.compose`'s own `_TOOL_DOWN_PITCH`/`_ROLL`);
the centre this module's pitch/roll jitter is sampled around."""

_IK_SEED_Q = np.array([0.0, -0.5, 0.9, 1.2, 0.0])
"""Default kinematic seed (matches `scene.compose.tool_down_ik`'s own default `q0`)."""


def generate_tool_down_poses(
    model: mujoco.MjModel,
    n_poses: int,
    rng: np.random.Generator,
    board_xy: tuple[float, float] = DEFAULT_BOARD_XY,
    hover_range_m: tuple[float, float] = (0.078, 0.095),
    xy_jitter_m: float = 0.015,
    pitch_jitter_rad: float = 0.22,
    roll_jitter_rad: float = 0.22,
    max_attempts: int = 800,
) -> list[np.ndarray]:
    """``n_poses`` random joint configurations (5-vectors) hovering near ``board_xy``, tool
    pointing generally down but with pitch and roll both perturbed around straight-down -- not
    held exactly straight down.

    That perturbation is required, not cosmetic: a first version of this function held the tool
    exactly straight down and only varied yaw (via `brick_builder.scene.compose.tool_down_ik`),
    which puts every sampled gripper rotation's axis in the same one- or two-dimensional family
    (all "mostly a rotation about world z"). Hand-eye calibration's `AX = XB` solve needs rotation
    axes that actually span 3D (`brick_builder.calib.camera.calibrate_hand_eye`'s own docstring);
    with only yaw varying, the camera-to-gripper rotation's component along the shared axis is
    unobservable, and the translation equation divides by ``(R_A - I)``, which is then singular
    along that same direction -- both the rotation and position fits came out wrong by orders of
    magnitude (checked directly: >10 m position error) before this function varied pitch and roll
    too. Uses `brick_builder.robot.kinematics.ArmKinematics.ik` directly (position + pitch + roll)
    rather than ``tool_down_ik``'s always-straight-down convenience wrapper. IK failures (a sampled
    pose just outside reach) are silently skipped and re-sampled; raises if ``max_attempts`` is
    exhausted first.
    """
    kin = ArmKinematics(model)
    poses: list[np.ndarray] = []
    q_seed = _IK_SEED_Q
    attempts = 0
    while len(poses) < n_poses and attempts < max_attempts:
        attempts += 1
        x = board_xy[0] + rng.uniform(-xy_jitter_m, xy_jitter_m)
        y = board_xy[1] + rng.uniform(-xy_jitter_m, xy_jitter_m)
        z = rng.uniform(*hover_range_m)
        pitch = _NOMINAL_PITCH + rng.uniform(-pitch_jitter_rad, pitch_jitter_rad)
        roll = _NOMINAL_ROLL + rng.uniform(-roll_jitter_rad, roll_jitter_rad)
        result = kin.ik(np.array([x, y, z]), pitch, roll, q_seed, iters=150)
        if not result.converged:
            continue
        poses.append(result.q)
        q_seed = result.q
    if len(poses) < n_poses:
        raise RuntimeError(
            f"only found {len(poses)}/{n_poses} reachable poses near {board_xy} in "
            f"{max_attempts} attempts"
        )
    return poses


# --------------------------------------------------------------------------------- rendering ---


@dataclass
class CalibrationShot:
    rgb: np.ndarray
    gripper_to_base: tuple[np.ndarray, np.ndarray]     # (R, t), MuJoCo convention, metres
    camera_to_base_truth: tuple[np.ndarray, np.ndarray]  # (R, t), MuJoCo convention, ground truth


@dataclass
class CalibrationSet:
    shots: list[CalibrationShot]
    camera_cfg: CameraConfig
    target: calib.CheckerboardTarget
    board_xy: tuple[float, float]


ARM_JOINTS_5 = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def render_calibration_set(
    camera_cfg: CameraConfig = CameraConfig(),
    arm_cfg: ArmConfig | None = None,
    target: calib.CheckerboardTarget = calib.CheckerboardTarget(),
    n_poses: int = 90,
    seed: int = 0,
    board_xy: tuple[float, float] = DEFAULT_BOARD_XY,
    extra_qpos: list[np.ndarray] | None = None,
) -> CalibrationSet:
    """Renders ``n_poses`` (plus any ``extra_qpos``) wrist-camera views of the checkerboard.

    ``extra_qpos`` lets a caller add specific, deliberately chosen poses (e.g. one far outlier)
    on top of the random tool-down spread; each is a 5-vector for
    ``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll`` (the same joints
    `generate_tool_down_poses` returns).

    ``n_poses`` defaults to 90, not the 15+ the validation itself needs: at this close a hover (a
    few cm, the same distance the accepted `docs/media/robot/wrist_camera_baseplate_view.png`
    reference was shot at) with the tool tilted, not held straight down (`generate_tool_down_poses`
    varies pitch and roll -- required for hand-eye conditioning, not cosmetic), somewhere around a
    quarter to a third of poses have the gripper's own silhouette crossing the board's outermost
    corner row or column tightly enough that `cv2.findChessboardCorners` drops the whole shot --
    checked directly by probing smaller pattern sizes on a "clean-looking" rejected render, which
    detects at ``(9, 5)``/``(9, 4)`` but not ``(9, 6)``. This is not a rendering bug to chase out; a
    real calibration session drops unusable shots the same way, and a session that also needs
    rotational spread (not just straight-down views) faces the same framing trade-off. Oversampling
    here is what keeps `brick_builder.calib.camera.calibrate_intrinsics` supplied with 15+ usable
    images without the caller having to retry.
    """
    rng = np.random.default_rng(seed)
    spec = build_checkerboard_spec(camera_cfg=camera_cfg, arm_cfg=arm_cfg, target=target, board_xy=board_xy)
    model = spec.compile()
    data = mujoco.MjData(model)

    qs = generate_tool_down_poses(model, n_poses, rng, board_xy=board_xy)
    if extra_qpos:
        qs = qs + list(extra_qpos)

    qpos_adr = [model.joint(name).qposadr[0] for name in ARM_JOINTS_5]
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    width, height = camera_cfg.resolution

    shots = []
    with SceneRenderer(model, width=width, height=height) as renderer:
        for q in qs:
            mujoco.mj_resetData(model, data)
            for adr, val in zip(qpos_adr, q):
                data.qpos[adr] = val
            mujoco.mj_forward(model, data)

            frame = renderer.render(data, camera=WRIST_CAM)

            r_grip = data.xmat[gripper_id].reshape(3, 3).copy()
            t_grip = data.xpos[gripper_id].copy()
            r_cam = data.cam_xmat[cam_id].reshape(3, 3).copy()
            t_cam = data.cam_xpos[cam_id].copy()

            shots.append(CalibrationShot(
                rgb=frame.rgb,
                gripper_to_base=(r_grip, t_grip),
                camera_to_base_truth=(r_cam, t_cam),
            ))
    return CalibrationSet(shots=shots, camera_cfg=camera_cfg, target=target, board_xy=board_xy)


# --------------------------------------------------------------------------------- recovery report ---


@dataclass(frozen=True)
class RecoveryReport:
    """One row of the simulated-validation table: what calibration recovered vs. this module's
    own ground truth ``CameraConfig``."""

    label: str
    n_poses_requested: int
    n_corners_found: int
    reprojection_rms_px: float
    focal_error_pct: float
    position_error_mm: float
    rotation_error_deg: float
    flip: calib.FlipCheckResult


def _camera_to_gripper_truth(camera_cfg: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    r_true = calib.quat_wxyz_to_matrix(np.asarray(camera_cfg.quat_in_gripper, dtype=float))
    t_true = np.asarray(camera_cfg.pos_in_gripper, dtype=float)
    return r_true, t_true


def calibrate_from_calibration_set(
    calib_set: CalibrationSet,
    target: calib.CheckerboardTarget | None = None,
    hand_eye_method: str = "tsai",
    min_corners_found: int = 15,
) -> tuple[calib.IntrinsicsResult, calib.HandEyeResult]:
    """Runs `brick_builder.calib.camera`'s intrinsics + hand-eye calibration against a rendered
    `CalibrationSet`'s images and known gripper poses -- the same two calls a real calibration run
    against photographs makes (see `scripts/calibrate_camera.py`), factored out so both that script's
    ``--simulate`` mode and :func:`validate_recovery` share one rendering pass.

    Raises if fewer than ``min_corners_found`` images had a detectable checkerboard (see
    `render_calibration_set`'s docstring for why some renders are expected to fail).
    """
    target = calib_set.target if target is None else target
    images = [shot.rgb for shot in calib_set.shots]
    intrinsics = calib.calibrate_intrinsics(images, target=target)
    if len(intrinsics.used_image_indices) < min_corners_found:
        raise RuntimeError(
            f"only {len(intrinsics.used_image_indices)}/{len(images)} renders had a detectable "
            f"checkerboard, need >= {min_corners_found}; raise n_poses"
        )

    gripper_to_base, target_to_cam = [], []
    for idx, r_vec, t_vec in zip(intrinsics.used_image_indices, intrinsics.rvecs, intrinsics.tvecs):
        r_t2c, _ = cv2.Rodrigues(r_vec)
        gripper_to_base.append(calib_set.shots[idx].gripper_to_base)
        target_to_cam.append((r_t2c, t_vec))
    hand_eye = calib.calibrate_hand_eye(gripper_to_base, target_to_cam, method=hand_eye_method)
    return intrinsics, hand_eye


def report_from_calibration(
    camera_cfg: CameraConfig,
    intrinsics: calib.IntrinsicsResult,
    hand_eye: calib.HandEyeResult,
    n_poses_requested: int,
    label: str = "default",
) -> RecoveryReport:
    """Compares a calibration result against the ``CameraConfig`` ground truth that produced it --
    the "recovery error" half of the simulated validation; needs no rendering itself."""
    fx_true, fy_true, _, _ = camera_cfg.resolved_intrinsics_px()
    focal_error_pct = 100.0 * (
        abs(intrinsics.fx - fx_true) / fx_true + abs(intrinsics.fy - fy_true) / fy_true
    ) / 2.0

    r_true, t_true = _camera_to_gripper_truth(camera_cfg)
    r_rec = hand_eye.rotation_mujoco()
    t_rec = hand_eye.translation_m_mujoco()
    position_error_mm = float(np.linalg.norm(t_rec - t_true) * 1000.0)
    rotation_error_deg = calib.angle_between_rotations_deg(r_true, r_rec)

    flip = calib.check_sensor_flip(r_rec, np.asarray(camera_cfg.quat_in_gripper, dtype=float))

    return RecoveryReport(
        label=label,
        n_poses_requested=n_poses_requested,
        n_corners_found=len(intrinsics.used_image_indices),
        reprojection_rms_px=intrinsics.rms_reprojection_px,
        focal_error_pct=focal_error_pct,
        position_error_mm=position_error_mm,
        rotation_error_deg=rotation_error_deg,
        flip=flip,
    )


def validate_recovery(
    camera_cfg: CameraConfig = CameraConfig(),
    arm_cfg: ArmConfig | None = None,
    target: calib.CheckerboardTarget = calib.CheckerboardTarget(),
    n_poses: int = 90,
    seed: int = 0,
    board_xy: tuple[float, float] = DEFAULT_BOARD_XY,
    label: str = "default",
    hand_eye_method: str = "tsai",
    min_corners_found: int = 15,
) -> RecoveryReport:
    """Renders ``n_poses`` checkerboard shots at ``camera_cfg`` and reports how well
    `brick_builder.calib.camera` recovers it: reprojection error, focal-length error, and
    hand-eye position/rotation error against ground truth this module knows exactly.

    Raises if fewer than ``min_corners_found`` of the ``n_poses`` renders had a detectable
    checkerboard (see `render_calibration_set`'s docstring for why some are expected to fail) --
    the task this package was set is "15+ arm poses", meaning 15+ *usable* ones.
    """
    calib_set = render_calibration_set(
        camera_cfg=camera_cfg, arm_cfg=arm_cfg, target=target, n_poses=n_poses, seed=seed, board_xy=board_xy,
    )
    intrinsics, hand_eye = calibrate_from_calibration_set(
        calib_set, target=target, hand_eye_method=hand_eye_method, min_corners_found=min_corners_found,
    )
    return report_from_calibration(camera_cfg, intrinsics, hand_eye, n_poses_requested=n_poses, label=label)


def format_report_table(reports: list[RecoveryReport]) -> str:
    """A Markdown table -- ``docs/guide/03``'s camera section pastes this directly."""
    header = (
        "| Case | poses (found/requested) | reprojection RMS [px] | focal error [%] | "
        "position error [mm] | rotation error [deg] |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {r.label} | {r.n_corners_found}/{r.n_poses_requested} | {r.reprojection_rms_px:.3f} | "
        f"{r.focal_error_pct:.2f} | {r.position_error_mm:.2f} | {r.rotation_error_deg:.3f} |\n"
        for r in reports
    )
    return header + rows
