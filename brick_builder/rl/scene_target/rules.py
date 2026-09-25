"""Progress index, reward and ground-truth scoring as pure functions of numbers.

Kept free of the simulator so every sign and every branch is tested on synthetic sequences
(`tests/rl/scene_target/test_rules.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RewardWeights

__all__ = [
    "GT_TOL_XY_MM", "GT_TOL_YAW_DEG", "GT_TOL_Z_MM", "PoseError", "ProgressState", "advance",
    "pose_error", "progress_update", "step_reward", "terminal_reward", "tube_width",
]

GT_TOL_XY_MM = 1.0
GT_TOL_Z_MM = 1.5
GT_TOL_YAW_DEG = 1.0
"""Ground-truth success on the viewing-pose stages: tool position within 1.0 mm in the plane
and 1.5 mm in height, tool yaw within 1.0 deg, of the target pose (in the displaced plate's
frame on L0b)."""


# --------------------------------------------------------------------------- progress index ---


def tube_width(ref_step_mm: float, *, floor_mm: float = 1.0, sigma_obs_mm: float = 0.0,
               tube_scale: float = 1.6) -> float:
    """Advance tolerance [mm-eq]: `max(floor, 3 sigma_obs, tube_scale * D(S*_k, S*_{k+1}))`."""
    return float(max(floor_mm, 3.0 * sigma_obs_mm, tube_scale * ref_step_mm))


def advance(d_mm: float, *, ref_step_mm: float, floor_mm: float = 1.0,
            sigma_obs_mm: float = 0.0, tube_scale: float = 1.6) -> bool:
    """Whether the index may advance: `d <= tube`. A non-finite distance never advances."""
    if not math.isfinite(d_mm):
        return False
    return bool(d_mm <= tube_width(ref_step_mm, floor_mm=floor_mm, sigma_obs_mm=sigma_obs_mm,
                                   tube_scale=tube_scale))


@dataclass(frozen=True)
class ProgressState:
    k: int
    holds: int = 0
    settled: int = 0


def progress_update(state: ProgressState, *, T: int, d_mm: float, tube_mm: float,
                    tol_mm: float) -> tuple[ProgressState, bool, bool]:
    """One tick of the progress index. Returns `(state, advanced, held)`.

    Before the last frame the index advances when `d <= tube` and holds otherwise. At the last
    frame a `d` inside the tube counts toward settling when also `d <= tol` (and resets the
    settle count when not); a `d` outside the tube holds. The rule of
    `action.place.TrackingEpisode.run`, with the tube in mm-eq of scene-state distance.
    """
    k, holds, settled = state.k, state.holds, state.settled
    inside = math.isfinite(d_mm) and d_mm <= tube_mm
    if k >= T - 1:
        if inside:
            settled = settled + 1 if d_mm <= tol_mm else 0
            return ProgressState(T - 1, 0, settled), False, False
        return ProgressState(T - 1, holds + 1, 0), False, True
    if inside:
        return ProgressState(k + 1, 0, 0), True, False
    return ProgressState(k, holds + 1, 0), False, True


# ------------------------------------------------------------------------------------ reward ---


def step_reward(w: RewardWeights, *, d_prev: float, d_next: float, advanced: bool,
                u: np.ndarray, u_prev: np.ndarray, held: bool, damage_rise: int,
                tol_mm: float) -> float:
    """Per-tick reward; both distances are against the pre-step target `S*_{k+1}`.

    `w_p (d_prev - d_next) - w_d min(d_next / tol, 5) + w_adv [advanced] - w_h [held]
    - w_u mean(u^2) - w_s mean((u - u_prev)^2) - w_dmg damage_rise`.
    """
    u = np.asarray(u, dtype=np.float64)
    u_prev = np.asarray(u_prev, dtype=np.float64)
    r = w.w_p * (float(d_prev) - float(d_next))
    r -= w.w_d * min(float(d_next) / max(float(tol_mm), 1e-9), 5.0)
    if advanced:
        r += w.w_adv
    if held:
        r -= w.w_h
    r -= w.w_u * float(np.mean(u ** 2)) if u.size else 0.0
    r -= w.w_s * float(np.mean((u - u_prev) ** 2)) if u.size else 0.0
    r -= w.w_dmg * max(int(damage_rise), 0)
    return float(r)


def terminal_reward(w: RewardWeights, *, success: bool, failure: str,
                    lateral_mm: float | None = None) -> float:
    """`+success` on success, `+lateral * max(0, 1 - lateral_mm)` on placement stages when
    seated, `-failure` on a stall, lost view, grasp failure or damage. A timeout pays nothing."""
    r = 0.0
    if success:
        r += w.success
        if lateral_mm is not None and math.isfinite(lateral_mm):
            r += w.lateral * max(0.0, 1.0 - float(lateral_mm))
    if failure and failure != "timeout":
        r -= w.failure
    return float(r)


# ------------------------------------------------------------------------ ground-truth score ---


@dataclass(frozen=True)
class PoseError:
    xy_mm: float
    z_mm: float
    yaw_deg: float
    xyz_mm: float

    def success(self, *, tol_xy_mm: float = GT_TOL_XY_MM, tol_z_mm: float = GT_TOL_Z_MM,
                tol_yaw_deg: float = GT_TOL_YAW_DEG) -> bool:
        return (self.xy_mm <= tol_xy_mm and abs(self.z_mm) <= tol_z_mm
                and abs(self.yaw_deg) <= tol_yaw_deg)


def target_in_plate_frame(target_pose, plate_nominal_xy, plate_offset_xy, plate_yaw: float):
    """Target pose (nominal plate frame) in the displaced plate's frame: (x, y, z, yaw)."""
    t = np.asarray(target_pose, dtype=np.float64)
    c = np.asarray(plate_nominal_xy, dtype=np.float64)
    rel = t[:2] - c
    cs, sn = math.cos(plate_yaw), math.sin(plate_yaw)
    xy = c + np.asarray(plate_offset_xy, dtype=np.float64) + np.array(
        [cs * rel[0] - sn * rel[1], sn * rel[0] + cs * rel[1]])
    return np.array([xy[0], xy[1], t[2], t[3] + plate_yaw])


def pose_error(tool_pos, tool_yaw: float, target_pose, *, plate_nominal_xy=(0.17, 0.0),
               plate_offset_xy=(0.0, 0.0), plate_yaw: float = 0.0) -> PoseError:
    """Tool pose error against a target pose defined in the nominal plate frame."""
    t = target_in_plate_frame(target_pose, plate_nominal_xy, plate_offset_xy, plate_yaw)
    p = np.asarray(tool_pos, dtype=np.float64)
    dxy = 1e3 * float(np.linalg.norm(p[:2] - t[:2]))
    dz = 1e3 * float(p[2] - t[2])
    dyaw = math.degrees(math.remainder(float(tool_yaw) - float(t[3]), 2.0 * math.pi))
    return PoseError(xy_mm=dxy, z_mm=dz, yaw_deg=dyaw, xyz_mm=math.hypot(dxy, dz))
