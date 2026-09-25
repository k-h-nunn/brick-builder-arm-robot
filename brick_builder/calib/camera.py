"""Wrist camera calibration: intrinsics + distortion from checkerboard images, hand-eye
camera-to-gripper calibration from arm poses, and a check for a sensor mounted rotated 180
degrees in its socket.

Pure OpenCV/numpy -- no MuJoCo import here, so this module (and a real calibration run against
photographs) never needs a GL context. The simulated validation that renders its own checkerboard
images with MuJoCo lives in :mod:`brick_builder.calib.camera_sim`, which imports this module, not
the other way around.

Conventions, matched to ``docs/guide/03-robot-and-servo-model.md`` and
``docs/guide/08-hardware-wiring-calibration.md``:

- **OpenCV camera convention**: +z forward (into the scene), +y down, +x right -- what
  ``cv2.calibrateCamera``/``cv2.solvePnP``/``cv2.calibrateHandEye`` all use.
- **MuJoCo camera convention**: -z forward, +y up, +x right (a camera looks down its own -z) --
  what :class:`brick_builder.robot.patch_so101.CameraConfig` and every compiled wrist camera use.
- The two are related by a fixed 180 degree rotation about the camera's own x axis
  (``FLIP_CV_TO_MUJOCO`` below): rotations convert through it, translations do not change (the
  camera's origin in the gripper frame does not depend on which way its own axes point).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------------- conventions ---

FLIP_CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])
"""Applied on the right of an OpenCV-convention rotation matrix to get the same physical
rotation expressed for a MuJoCo-convention camera frame (see the module docstring); its own
inverse, so the same matrix converts MuJoCo -> OpenCV too."""


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a ``(w, x, y, z)`` quaternion (MuJoCo's own ordering)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quaternion from a proper rotation matrix (Shepperd's method)."""
    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def angle_between_rotations_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    """Smallest rotation angle [deg] taking ``r1`` to ``r2``."""
    rel = np.asarray(r1, dtype=float).T @ np.asarray(r2, dtype=float)
    cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def cv_rotation_to_mujoco(r_cv: np.ndarray) -> np.ndarray:
    """An OpenCV-convention rotation matrix, expressed for a MuJoCo-convention camera frame."""
    return np.asarray(r_cv, dtype=float) @ FLIP_CV_TO_MUJOCO


# --------------------------------------------------------------------------- checkerboard corners ---


@dataclass(frozen=True)
class CheckerboardTarget:
    """A checkerboard's internal-corner grid and physical square size.

    ``pattern`` is ``(cols, rows)`` internal corners -- OpenCV's own ``(points_per_row,
    points_per_col)`` convention. Defaults match the target this project prints (docs guide 03/08):
    9x6 internal corners (10x7 squares), 15 mm squares.
    """

    pattern: tuple[int, int] = (9, 6)
    square_m: float = 0.015

    def object_points(self) -> np.ndarray:
        """``(cols*rows, 3)`` object points, board frame, z=0, centred on the board's own middle.

        Raster order matches what :func:`find_corners` returns: row by row (y), left to right
        (x) -- the standard OpenCV tutorial convention (``np.mgrid[0:cols, 0:rows].T``).
        """
        cols, rows = self.pattern
        pts = np.zeros((cols * rows, 3), dtype=np.float64)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
        pts[:, 0] = (grid[:, 0] - (cols - 1) / 2.0) * self.square_m
        pts[:, 1] = (grid[:, 1] - (rows - 1) / 2.0) * self.square_m
        return pts


def find_corners(image: np.ndarray, target: CheckerboardTarget = CheckerboardTarget()) -> np.ndarray | None:
    """Sub-pixel checkerboard corners, or ``None`` if the pattern was not found.

    ``image`` is grayscale or RGB uint8 -- this module's convention throughout (a MuJoCo render's
    ``Frame.rgb`` is already RGB; :func:`scripts.calibrate_camera._load_images_and_poses` converts
    ``cv2.imread``'s BGR to RGB on load). A checkerboard's near-neutral pixels (R ≈ G ≈ B)
    make the RGB/BGR distinction numerically negligible for corner detection either way, but the
    rest of this module is not colour-neutral (:func:`solve_checkerboard_pose` via the camera
    matrix does not care, but a caller doing its own colour work on the same array would) -- stay
    consistent and hand this RGB, not BGR.
    """
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, target.pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def rotated_180(image: np.ndarray) -> np.ndarray:
    """An image rotated 180 degrees -- what a sensor mounted rotated 180 degrees in its socket
    produces from the same scene. ``image[::-1, ::-1]`` flips both axes, which is exactly a
    180-degree rotation about the image centre (not a mirror)."""
    return image[::-1, ::-1, ...]


# --------------------------------------------------------------------------------- intrinsics ---


@dataclass(frozen=True)
class IntrinsicsResult:
    """What ``cv2.calibrateCamera`` returns, plus the corners it was fit against."""

    camera_matrix: np.ndarray          # 3x3
    dist_coeffs: np.ndarray            # (k1, k2, p1, p2, k3)
    rms_reprojection_px: float
    image_size: tuple[int, int]        # (width, height)
    per_view_corners: list[np.ndarray] = field(repr=False)
    rvecs: list[np.ndarray] = field(repr=False)     # target-to-camera, OpenCV convention
    tvecs: list[np.ndarray] = field(repr=False)
    used_image_indices: list[int] = field(default_factory=list)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def k1(self) -> float:
        return float(self.dist_coeffs.reshape(-1)[0])

    @property
    def k2(self) -> float:
        return float(self.dist_coeffs.reshape(-1)[1])


def calibrate_intrinsics(
    images: list[np.ndarray],
    target: CheckerboardTarget = CheckerboardTarget(),
) -> IntrinsicsResult:
    """Checkerboard intrinsics + distortion from a set of images (``cv2.calibrateCamera``).

    Images where the pattern is not found are skipped, not errored on -- ``used_image_indices``
    says which of ``images`` were actually used. Raises if fewer than 3 images have the pattern
    (``cv2.calibrateCamera`` is under-determined below that).
    """
    objp = target.object_points()
    obj_points, img_points, used = [], [], []
    image_size = None
    for i, image in enumerate(images):
        corners = find_corners(image, target)
        if corners is None:
            continue
        obj_points.append(objp.astype(np.float32))
        img_points.append(corners.astype(np.float32))
        used.append(i)
        h, w = image.shape[:2]
        image_size = (w, h)
    if len(used) < 3:
        raise ValueError(
            f"checkerboard found in only {len(used)}/{len(images)} image(s); need at least 3"
        )
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return IntrinsicsResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.reshape(-1),
        rms_reprojection_px=float(rms),
        image_size=image_size,
        per_view_corners=img_points,
        rvecs=[r.reshape(3) for r in rvecs],
        tvecs=[t.reshape(3) for t in tvecs],
        used_image_indices=used,
    )


def solve_checkerboard_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    target: CheckerboardTarget = CheckerboardTarget(),
) -> tuple[np.ndarray, np.ndarray]:
    """Target-to-camera ``(R, t)`` for one image's corners (OpenCV convention, metres), matched
    against :meth:`CheckerboardTarget.object_points`'s own raster order (the same order
    ``cv2.findChessboardCorners`` returns).

    Note what this does *not* handle: a planar checkerboard's corner order can occasionally come
    back rotated 180 degrees about the board's own normal (``cv2.findChessboardCorners`` walks the
    grid from whichever corner its own heuristic finds first). That ambiguity is a 180-degree
    *in-plane* relabelling -- it does not change which way the board's normal faces, so nothing in
    a single image's own geometry (this function's only input) can detect it; a wrong-order image
    just fits its own self-consistent, wrong-by-180-degrees pose with no larger residual than a
    correctly-ordered one would. What *does* detect it is cross-image consistency once several
    poses are combined -- ``calibrate_hand_eye``'s fit degrades sharply if some input images are
    labelled 180 degrees apart, and :func:`check_sensor_flip` catches the specific, systematic
    version of this (every image flipped the same way, e.g. an upside-down sensor) by comparing the
    resulting camera-to-gripper orientation against the CAD-nominal mount pose.
    """
    objp = target.object_points().astype(np.float64)
    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(objp, corners.astype(np.float64), camera_matrix, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rot, _ = cv2.Rodrigues(rvec)
    return rot, tvec.reshape(3)


# --------------------------------------------------------------------------------- hand-eye ---


@dataclass(frozen=True)
class HandEyeResult:
    """Camera-to-gripper transform (``cv2.calibrateHandEye``'s own direction), OpenCV convention."""

    rotation_cv: np.ndarray    # 3x3
    translation_m_cv: np.ndarray  # (3,), metres
    n_poses: int
    method: str
    used_cv2_binding: bool
    """Whether ``cv2.calibrateHandEye`` itself ran ``method``. ``False`` means
    :func:`_solve_axb_axis_correspondence` ran instead (see :data:`HAS_CV2_HAND_EYE_BINDING`),
    which is not a Tsai/Park/Horaud/Andreff/Daniilidis implementation and does not read ``method``
    at all -- :attr:`method` is then only the caller's request, not what actually solved it."""

    def rotation_mujoco(self) -> np.ndarray:
        """The same rotation, expressed for a MuJoCo-convention camera frame (see module docstring)."""
        return cv_rotation_to_mujoco(self.rotation_cv)

    def translation_m_mujoco(self) -> np.ndarray:
        """Camera position in the gripper frame -- unaffected by the OpenCV/MuJoCo axis
        convention (see module docstring)."""
        return self.translation_m_cv

    @property
    def solver_description(self) -> str:
        """One line naming what actually solved ``AX = XB`` -- for provenance notes and printouts,
        so a fallback run is never reported as ``cv2.calibrateHandEye``."""
        if self.used_cv2_binding:
            return f"cv2.calibrateHandEye ({self.method})"
        return (
            "closed-form axis-correspondence AX=XB solve "
            f"(cv2.calibrateHandEye unavailable; requested method {self.method!r} had no effect)"
        )


_HAND_EYE_METHODS = {
    "tsai": getattr(cv2, "CALIB_HAND_EYE_TSAI", None),
    "park": getattr(cv2, "CALIB_HAND_EYE_PARK", None),
    "horaud": getattr(cv2, "CALIB_HAND_EYE_HORAUD", None),
    "andreff": getattr(cv2, "CALIB_HAND_EYE_ANDREFF", None),
    "daniilidis": getattr(cv2, "CALIB_HAND_EYE_DANIILIDIS", None),
}

HAS_CV2_HAND_EYE_BINDING = hasattr(cv2, "calibrateHandEye")
"""False on this project's OpenCV 5.0.0 build: the ``CALIB_HAND_EYE_*`` enum constants are
exposed but the ``cv2.calibrateHandEye`` Python function itself is not bound (checked directly:
absent from ``dir(cv2)`` and from the installed ``cv2/__init__.pyi`` stub). :func:`calibrate_hand_eye`
calls it when present and otherwise falls back to :func:`_solve_axb_axis_correspondence`, a from-scratch
closed-form ``AX = XB`` solve so this module works on the venv actually installed. Re-check this
flag after any OpenCV upgrade -- the fallback can be dropped once the real binding exists."""


def _pairwise_axb(
    gripper_to_base: list[tuple[np.ndarray, np.ndarray]],
    target_to_cam: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """All ``n*(n-1)/2`` pairwise ``A_ij = g2b_j^-1 @ g2b_i``, ``B_ij = t2c_j @ t2c_i^-1``.

    This is the standard ``AX = XB`` pair construction: with the target fixed in the base frame,
    ``g2b_i @ X @ t2c_i == g2b_j @ X @ t2c_j`` for every ``i, j`` (both sides are the same
    target-to-base transform), which rearranges to ``A_ij @ X == X @ B_ij``. Using every pair
    rather than only consecutive shots gives the closed-form solve below far more, and
    better-conditioned, equations from the same handful of poses. ``A`` and ``B`` are built with
    *different* orderings of the same ``(i, j)`` (``A`` is ``j^-1 @ i``, ``B`` is ``j @ i^-1``) --
    that asymmetry is required by the derivation above, not a typo.
    """
    n = len(gripper_to_base)
    r_a, t_a, r_b, t_b = [], [], [], []
    for i in range(n):
        rg_i, tg_i = gripper_to_base[i]
        rt_i, tt_i = target_to_cam[i]
        for j in range(i + 1, n):
            rg_j, tg_j = gripper_to_base[j]
            rt_j, tt_j = target_to_cam[j]
            r_a.append(rg_j.T @ rg_i)
            t_a.append(rg_j.T @ (tg_i - tg_j))
            rb = rt_j @ rt_i.T
            r_b.append(rb)
            t_b.append(tt_j - rb @ tt_i)
    return r_a, t_a, r_b, t_b


def _solve_axb_axis_correspondence(
    r_a: list[np.ndarray], t_a: list[np.ndarray], r_b: list[np.ndarray], t_b: list[np.ndarray],
    min_angle_deg: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``AX = XB`` solve (rotation via an axis-correspondence Procrustes fit, then a
    linear least-squares solve for translation) -- see :data:`HAS_CV2_HAND_EYE_BINDING`.

    Conjugating a rotation by ``X`` (``R_a = R_x R_b R_x^-1``, the rotation part of ``AX = XB``
    rearranged) rotates its axis by ``R_x`` and leaves its angle unchanged, so every pair with a
    large-enough rotation angle contributes one ``axis_a = R_x @ axis_b`` correspondence; pairs
    below ``min_angle_deg`` are dropped (their axis is numerically undefined). ``R_x`` minimizing
    ``sum |axis_a - R_x @ axis_b|^2`` is the standard orthogonal-Procrustes SVD solution. With
    ``R_x`` fixed, ``(R_a - I) t_x = R_x t_b - t_a`` is linear in ``t_x``; stacked over every pair
    and solved by least squares.
    """
    axes_a, axes_b, rows_r, rows_t = [], [], [], []
    min_angle = math.radians(min_angle_deg)
    for ra, ta, rb, tb in zip(r_a, t_a, r_b, t_b):
        rvec_a, _ = cv2.Rodrigues(ra)
        rvec_b, _ = cv2.Rodrigues(rb)
        angle_a, angle_b = np.linalg.norm(rvec_a), np.linalg.norm(rvec_b)
        if angle_a < min_angle or angle_b < min_angle:
            continue
        axes_a.append((rvec_a / angle_a).reshape(3))
        axes_b.append((rvec_b / angle_b).reshape(3))
        rows_r.append(ra)
        rows_t.append((ta, tb))
    if len(axes_a) < 3:
        raise ValueError(
            f"only {len(axes_a)} pose pair(s) had enough rotation (>= {min_angle_deg} deg) to "
            "solve hand-eye calibration from; use more, more rotationally varied poses"
        )
    axes_a = np.array(axes_a)
    axes_b = np.array(axes_b)
    h = axes_b.T @ axes_a
    u, _, vt = np.linalg.svd(h)
    d = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    r_x = vt.T @ d @ u.T

    lhs, rhs = [], []
    for ra, (ta, tb) in zip(rows_r, rows_t):
        lhs.append(ra - np.eye(3))
        rhs.append(r_x @ tb - ta)
    lhs = np.concatenate(lhs, axis=0)
    rhs = np.concatenate(rhs, axis=0)
    t_x, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return r_x, t_x


def calibrate_hand_eye(
    gripper_to_base: list[tuple[np.ndarray, np.ndarray]],
    target_to_cam: list[tuple[np.ndarray, np.ndarray]],
    method: str = "tsai",
) -> HandEyeResult:
    """Camera-to-gripper ``(R, t)`` from paired gripper and checkerboard poses.

    ``gripper_to_base``/``target_to_cam`` are same-length lists of ``(R, t)`` -- rotation matrix
    plus translation vector (metres) -- one pair per shot: the arm's own gripper pose (from
    forward kinematics, MuJoCo or real) and the checkerboard's pose relative to the camera (from
    :func:`solve_checkerboard_pose`, OpenCV convention). Solves the ``AX = XB`` hand-eye problem,
    through ``cv2.calibrateHandEye`` when the installed OpenCV build actually exposes it
    (:data:`HAS_CV2_HAND_EYE_BINDING`) and :func:`_solve_axb_axis_correspondence` otherwise. Needs at least
    3 poses with enough rotational spread between them to be well conditioned, 8-10+ recommended.
    """
    if len(gripper_to_base) != len(target_to_cam):
        raise ValueError("gripper_to_base and target_to_cam must have the same length")
    if len(gripper_to_base) < 3:
        raise ValueError(f"need at least 3 poses for hand-eye calibration, got {len(gripper_to_base)}")
    if method not in _HAND_EYE_METHODS:
        raise ValueError(f"unknown hand-eye method {method!r}; choose one of {sorted(_HAND_EYE_METHODS)}")

    r_g2b = [np.asarray(r, dtype=np.float64) for r, _ in gripper_to_base]
    t_g2b = [np.asarray(t, dtype=np.float64).reshape(3) for _, t in gripper_to_base]
    r_t2c = [np.asarray(r, dtype=np.float64) for r, _ in target_to_cam]
    t_t2c = [np.asarray(t, dtype=np.float64).reshape(3) for _, t in target_to_cam]

    if HAS_CV2_HAND_EYE_BINDING:
        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_g2b, t_g2b, r_t2c, t_t2c, method=_HAND_EYE_METHODS[method]
        )
        t_cam2gripper = t_cam2gripper.reshape(3)
    else:
        r_a, t_a, r_b, t_b = _pairwise_axb(list(zip(r_g2b, t_g2b)), list(zip(r_t2c, t_t2c)))
        r_cam2gripper, t_cam2gripper = _solve_axb_axis_correspondence(r_a, t_a, r_b, t_b)

    return HandEyeResult(
        rotation_cv=r_cam2gripper,
        translation_m_cv=t_cam2gripper.reshape(3),
        n_poses=len(gripper_to_base),
        method=method,
        used_cv2_binding=HAS_CV2_HAND_EYE_BINDING,
    )


# --------------------------------------------------------------------------- 180-degree flip check ---


@dataclass(frozen=True)
class FlipCheckResult:
    """Whether a calibrated camera-to-gripper orientation looks like a sensor mounted rotated
    180 degrees in its socket, rather than a pose error.

    See the module and function docstrings: a sensor die soldered 180 degrees rotated on its
    board (a real, documented risk for this class of purchased UVC module -- see
    ``patch_so101.WRIST_CAM_QUAT_IN_GRIPPER``'s docstring) flips every image upside down, which
    hand-eye calibration cannot tell apart from "the mount is rotated 180 degrees about the
    camera's own optical axis" -- both explain the same images equally well. It shows up as the
    calibrated orientation matching the CAD-nominal mount pose rotated 180 degrees about that
    axis far better than it matches the CAD pose directly.
    """

    angle_to_expected_deg: float
    angle_to_expected_flipped_deg: float
    likely_flipped: bool

    @property
    def better_match_deg(self) -> float:
        return min(self.angle_to_expected_deg, self.angle_to_expected_flipped_deg)


def check_sensor_flip(
    calibrated_rotation_mujoco: np.ndarray,
    expected_quat_wxyz: np.ndarray,
) -> FlipCheckResult:
    """Compare a calibrated camera-to-gripper rotation against the CAD-nominal mount orientation.

    ``calibrated_rotation_mujoco`` is normally
    :meth:`HandEyeResult.rotation_mujoco`; ``expected_quat_wxyz`` is normally
    :data:`brick_builder.robot.patch_so101.WRIST_CAM_QUAT_IN_GRIPPER` (or a
    ``CameraConfig.quat_in_gripper``). ``likely_flipped`` is set when the calibrated orientation
    is closer to the expected pose rotated 180 degrees about the camera's own z axis than to the
    expected pose itself -- the fix is :func:`rotated_180` on every input image and a recalibrate,
    not a pose correction.
    """
    expected = quat_wxyz_to_matrix(np.asarray(expected_quat_wxyz, dtype=float))
    rot180_z = np.diag([-1.0, -1.0, 1.0])
    expected_flipped = expected @ rot180_z

    angle = angle_between_rotations_deg(expected, calibrated_rotation_mujoco)
    angle_flipped = angle_between_rotations_deg(expected_flipped, calibrated_rotation_mujoco)
    return FlipCheckResult(
        angle_to_expected_deg=angle,
        angle_to_expected_flipped_deg=angle_flipped,
        likely_flipped=angle_flipped < angle,
    )


# --------------------------------------------------------------------------- parameter fragment ---


_FRAGMENT_TEMPLATE = """\
camera:

  fx_px:
    value: {fx:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "{intrinsics_notes}"

  fy_px:
    value: {fy:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See fx_px."

  cx_px:
    value: {cx:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See fx_px."

  cy_px:
    value: {cy:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See fx_px."

  resolution_width_px:
    value: {width}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "Resolution the calibration images were captured at; fx_px/fy_px/cx_px/cy_px are only valid at this resolution."

  resolution_height_px:
    value: {height}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See resolution_width_px."

  opencv_k1:
    value: {k1:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "cv2.calibrateCamera, {n_images} image(s), RMS reprojection {rms:.4g} px. Brown-Conrady convention -- do not copy into distortion_k1 (a different, image-space parametrization; see camera.distortion_k1's own notes)."

  opencv_k2:
    value: {k2:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See opencv_k1."

  opencv_p1:
    value: {p1:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See opencv_k1."

  opencv_p2:
    value: {p2:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See opencv_k1."

  opencv_k3:
    value: {k3:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See opencv_k1."
"""

_HAND_EYE_FRAGMENT = """
  mount_x_mm:
    value: {x_mm:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "{solver_description}, {n_poses} pose(s), converted from OpenCV's camera convention to MuJoCo's (180 degree flip about the camera's own x axis)."

  mount_y_mm:
    value: {y_mm:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See mount_x_mm."

  mount_z_mm:
    value: {z_mm:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See mount_x_mm."

  mount_roll_deg:
    value: {roll_deg:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See mount_x_mm. Extracted from the calibrated rotation matrix as R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- brick_builder.robot.kinematics.euler_zyx's own convention, yaw/pitch/roll about the fixed base-frame axes, not the body's own rotated axes."

  mount_pitch_deg:
    value: {pitch_deg:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See mount_roll_deg."

  mount_yaw_deg:
    value: {yaw_deg:.6g}
    status: measured
    measured_by: {measured_by}
    measured_on: {measured_on}
    notes: "See mount_roll_deg."
"""


def _euler_xyz_deg_from_matrix(rot: np.ndarray) -> tuple[float, float, float]:
    """Roll/pitch/yaw [deg] from a rotation matrix, inverting :func:`_euler_xyz_deg_to_matrix`'s
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` -- roll/pitch/yaw about the fixed base-frame axes, not
    the body's own rotated axes (the convention ``camera.mount_roll_deg``/``_pitch_deg``/
    ``_yaw_deg`` documents; a pure roll reproduces ``WRIST_CAM_QUAT_IN_GRIPPER`` when
    pitch=yaw=0). Same decomposition as :func:`brick_builder.robot.kinematics.euler_zyx`, in
    ``(roll, pitch, yaw)`` order and degrees rather than ``(yaw, pitch, roll)`` in radians."""
    r = np.asarray(rot, dtype=float)
    pitch = math.asin(np.clip(-r[2, 0], -1.0, 1.0))
    if abs(r[2, 0]) < 0.9999999:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _euler_xyz_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Inverse of :func:`_euler_xyz_deg_from_matrix`: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    rot_y = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    rot_z = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def camera_config_from_params(params) -> "CameraConfig":  # noqa: F821 -- see docstring
    """Build a :class:`brick_builder.robot.patch_so101.CameraConfig` from a loaded
    :class:`brick_builder.params.ParamFile`'s ``camera.*`` fields.

    ``fx_px``/``fy_px``/``cx_px``/``cy_px`` only switch the built config onto
    :attr:`CameraConfig.uses_measured_intrinsics` when at least one of them has
    ``status: measured`` in the loaded file -- the shipped default file carries values for all
    four (derived from ``diagonal_fov_deg`` and a centred principal point, ``status: assumed``) so
    that a member reading the file sees a real number rather than a blank, not because that
    number is an actual calibration; passing them through unconditionally would build a
    measured-intrinsics config from the un-measured default file, which compiles a different
    MuJoCo camera model than the ``diagonal_fov_deg`` path even though the numbers agree (see
    :meth:`CameraConfig.fovy_deg`).

    Imported lazily (``patch_so101`` pulls in ``mujoco`` at module level) so nothing else in this
    module needs it -- the rest of ``brick_builder.calib.camera`` stays MuJoCo-free (module
    docstring).
    """
    from ..robot.patch_so101 import CameraConfig

    width = int(params.value("camera.resolution_width_px"))
    height = int(params.value("camera.resolution_height_px"))
    rot = _euler_xyz_deg_to_matrix(
        float(params.value("camera.mount_roll_deg")),
        float(params.value("camera.mount_pitch_deg")),
        float(params.value("camera.mount_yaw_deg")),
    )
    intrinsics_measured = any(
        params.get(f"camera.{name}").status == "measured"
        for name in ("fx_px", "fy_px", "cx_px", "cy_px")
    )
    fx = float(params.value("camera.fx_px")) if intrinsics_measured else None
    fy = float(params.value("camera.fy_px")) if intrinsics_measured else None
    cx = float(params.value("camera.cx_px")) if intrinsics_measured else None
    cy = float(params.value("camera.cy_px")) if intrinsics_measured else None
    return CameraConfig(
        resolution=(width, height),
        diagonal_fov_deg=float(params.value("camera.diagonal_fov_deg")),
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        distortion_k1=float(params.value("camera.distortion_k1")),
        distortion_k2=float(params.value("camera.distortion_k2")),
        pos_in_gripper=(
            float(params.value("camera.mount_x_mm")) / 1000.0,
            float(params.value("camera.mount_y_mm")) / 1000.0,
            float(params.value("camera.mount_z_mm")) / 1000.0,
        ),
        quat_in_gripper=tuple(float(v) for v in matrix_to_quat_wxyz(rot)),
    )


def parameter_fragment(
    intrinsics: IntrinsicsResult,
    hand_eye: HandEyeResult | None = None,
    measured_by: str = "unknown",
    measured_on: str = "unknown",
) -> str:
    """A ``configs/params/`` YAML fragment for the ``camera.*`` keys this calibration measured.

    Matches ``configs/params/default.yaml``'s block format exactly (``value``/``status``/
    ``measured_by``/``measured_on``/``notes`` per leaf) so it can be pasted straight into a
    member's copy of the parameter file; ``brick_builder.params`` is the only thing that loads
    that file (this module does not import it -- see the module docstring).
    """
    k1, k2, p1, p2, k3 = (list(intrinsics.dist_coeffs) + [0.0] * 5)[:5]
    text = _FRAGMENT_TEMPLATE.format(
        fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.cx, cy=intrinsics.cy,
        width=intrinsics.image_size[0], height=intrinsics.image_size[1],
        k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
        n_images=len(intrinsics.used_image_indices), rms=intrinsics.rms_reprojection_px,
        measured_by=measured_by, measured_on=measured_on,
        intrinsics_notes=(
            f"cv2.calibrateCamera, {len(intrinsics.used_image_indices)} image(s), "
            f"RMS reprojection {intrinsics.rms_reprojection_px:.4g} px."
        ),
    )
    if hand_eye is not None:
        t_mm = hand_eye.translation_m_mujoco() * 1000.0
        roll, pitch, yaw = _euler_xyz_deg_from_matrix(hand_eye.rotation_mujoco())
        text += _HAND_EYE_FRAGMENT.format(
            x_mm=t_mm[0], y_mm=t_mm[1], z_mm=t_mm[2],
            roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw,
            solver_description=hand_eye.solver_description, n_poses=hand_eye.n_poses,
            measured_by=measured_by, measured_on=measured_on,
        )
    return text
