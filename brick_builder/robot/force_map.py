"""Static downward force map for the SO-101 at a press pose.

For a tool-down TCP pose at each ``(x, y)`` grid point, the arm's own weight already spends part of
each joint's torque budget; what is left drives the tool through the transpose Jacobian.  By the
principle of virtual work, a joint torque ``tau_i`` alone would produce a vertical TCP force
``tau_i / J_z,i`` (``J_z,i`` = row-z of the position Jacobian's column ``i``); the joint that
saturates first sets the ceiling for the whole arm, so the available downward force is

    F = min_i (tau_max_i - tau_gravity_i) / |J_z,i|

which is a per-joint torque-budget bound, not a full multi-joint optimisation -- it is the
simplified estimate the specification asks for.

Dependency-free maths on top of ``ArmKinematics`` (itself built only on ``mj_forward`` and
``mj_jacSite``): the gravity torque comes from ``gravity_torque`` (zero-velocity ``qfrc_bias``),
the Jacobian from ``jacobian``, and reachability from ``ik``.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from brick_builder.robot.kinematics import KIN_JOINTS, ArmKinematics
from brick_builder.robot.patch_so101 import ArmConfig, NOMINAL_VIN, attach_arm

_TORQUE_LIMIT_NM = ArmConfig().torque_limit_nm
"""Stall torque at ``NOMINAL_VIN`` (1.9 N*m, the datasheet number): the same number
``build_arm_spec`` scales by voltage to set each actuator's ``forcerange`` by default, so the force
map uses exactly the torque budget the patched model actually enforces unless told otherwise."""


def stall_torque(vin: float, torque_limit_nm: float | None = None) -> float:
    """Per-joint torque limit at ``vin`` volts: the same linear scaling ``patch_so101`` applies.

    ``torque_limit_nm`` is the stall torque *at* ``NOMINAL_VIN`` to scale from; it defaults to the
    1.9 N*m datasheet number (``_TORQUE_LIMIT_NM``). The datasheet number and the
    BAM servo law's own 3.41 N*m ceiling at full duty (``patch_so101.servo_stall_torque``) disagree
    by a factor of 1.8 and neither has been measured on the real arm, so every force number derived
    from this function must say which one it used -- pass
    ``torque_limit_nm=servo_stall_torque(NOMINAL_VIN)`` to report force under the identified-law
    ceiling instead of the datasheet one.
    """
    limit = _TORQUE_LIMIT_NM if torque_limit_nm is None else torque_limit_nm
    return limit * vin / NOMINAL_VIN


def press_force_map(
    model: mujoco.MjModel,
    kin: ArmKinematics,
    xs: np.ndarray,
    ys: np.ndarray,
    z: float,
    vin_list: tuple[float, ...] = (5.0, 7.4, 12.0),
    *,
    roll: float = 0.0,
    q_seed: np.ndarray | None = None,
    torque_limit_nm: float | None = None,
) -> dict:
    """Grid the workspace at height ``z`` (arm base frame, metres) and score downward press force.

    For every ``(x, y)`` on the grid, IK is solved for a tool-down pose (TCP z-axis pointing at
    world -z) at that point; unreachable points are left out of ``force`` (``NaN``) but still
    appear in ``reachable`` (``False``), never silently filled in with a nearby answer.

    Each cell is solved from up to three seeds -- the already-solved cell to its left, the
    already-solved cell above it, and the caller's ``q_seed`` as a fresh-start fallback -- and,
    when more than one seed converges, the solution closest in joint space to whichever neighbour(s)
    already exist is kept. Neighbouring press poses are close in joint space and share an elbow
    branch, so this multi-start-plus-continuity choice is what keeps a whole
    row or column on one branch instead of a lone cell jumping to a different elbow configuration
    and showing up as a spurious unreachable patch in the figure.

    ``torque_limit_nm`` picks which per-joint stall-torque budget (at ``NOMINAL_VIN``) the force
    numbers are computed under; see ``stall_torque`` for why this matters (decision 7).

    Returns a dict: ``xs``, ``ys``, ``z``, ``reachable`` (``(ny, nx)`` bool), ``q``
    (``(ny, nx, 5)``, ``NaN`` where unreachable) and ``force`` (``{vin: (ny, nx) array, newtons}``,
    ``NaN`` where unreachable).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nx, ny = len(xs), len(ys)
    reachable = np.zeros((ny, nx), dtype=bool)
    q_grid = np.full((ny, nx, len(KIN_JOINTS)), np.nan)
    force = {vin: np.full((ny, nx), np.nan) for vin in vin_list}
    tau_max = {vin: stall_torque(vin, torque_limit_nm) for vin in vin_list}

    # Tool-down: the TCP z-axis (approach direction) points at world -z.  In this module's
    # Rz(yaw) Ry(pitch) Rx(roll) convention that is pitch = +pi/2 for any yaw, so yaw (free, set by
    # whichever way shoulder_pan needs to turn to reach x, y) does not have to be chosen here.
    target_pitch = math.pi / 2
    q_seed = np.zeros(len(KIN_JOINTS)) if q_seed is None else np.asarray(q_seed, dtype=float)

    def neighbour_qs(iy: int, ix: int) -> list[np.ndarray]:
        """Already-solved cells sharing an edge with ``(iy, ix)``, in whatever order they list."""
        out = []
        for ny_, nx_ in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
            if 0 <= ny_ < ny and 0 <= nx_ < nx and reachable[ny_, nx_]:
                out.append(q_grid[ny_, nx_])
        return out

    def solve_cell(iy: int, ix: int, x: float, y: float, neighbours: list[np.ndarray]) -> bool:
        """Try IK from every given neighbour plus the fresh-start seed; keep whichever converged
        solution sits closest (in joint space) to the neighbours, for elbow-branch continuity."""
        target = np.array([x, y, z])
        candidates = []
        for seed in (*neighbours, q_seed):
            res = kin.ik(target, target_pitch, roll, seed)
            if res.converged:
                candidates.append(res.q)
        if not candidates:
            return False
        q = (
            min(candidates, key=lambda c: float(np.linalg.norm(c - np.mean(neighbours, axis=0))))
            if neighbours
            else candidates[0]
        )
        reachable[iy, ix] = True
        q_grid[iy, ix] = q

        jac = kin.jacobian(q)
        jz = jac[2, :]  # row 2 of jacp: how a joint's velocity moves the TCP vertically
        tau_g = kin.gravity_torque(q)
        for vin in vin_list:
            available = tau_max[vin] - tau_g
            with np.errstate(divide="ignore", invalid="ignore"):
                per_joint = np.where(np.abs(jz) > 1e-9, available / np.abs(jz), np.inf)
            force[vin][iy, ix] = max(0.0, float(np.min(per_joint)))
        return True

    # First pass: raster scan, each cell seeded from its already-solved left and above neighbours.
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            solve_cell(iy, ix, x, y, neighbour_qs(iy, ix))

    # Repair pass: a cell whose only converging branch was to its right or below (not yet solved
    # when the raster scan reached it) is retried now that every neighbour direction is available.
    # This is what "multi-start" means here beyond the raster order alone -- it is exactly the
    # spurious-unreachable-patch failure mode decision 3 calls out.
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if not reachable[iy, ix]:
                solve_cell(iy, ix, x, y, neighbour_qs(iy, ix))

    return {"xs": xs, "ys": ys, "z": z, "reachable": reachable, "q": q_grid, "force": force}


def figure_force_map(result: dict, out_path: str | Path = "docs/media/robot/force_map.png") -> Path:
    """One heatmap per supply voltage: axes in metres (base frame), 5 N / 10 N contours, N colour bar."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs, ys, reachable = result["xs"], result["ys"], result["reachable"]
    vins = sorted(result["force"].keys())
    reachable_vals = [result["force"][v][reachable] for v in vins if reachable.any()]
    vmax = float(np.nanmax(np.concatenate(reachable_vals))) if reachable_vals else 1.0

    fig, axes = plt.subplots(1, len(vins), figsize=(4.4 * len(vins), 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    im = None
    for ax, vin in zip(axes, vins):
        grid = np.where(reachable, result["force"][vin], np.nan)
        im = ax.pcolormesh(xs, ys, grid, shading="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
        filled = np.nan_to_num(grid, nan=-1.0)
        if np.any(filled >= 5.0):
            cs = ax.contour(xs, ys, filled, levels=[5.0, 10.0], colors="white", linewidths=1.0)
            ax.clabel(cs, fmt="%d N", fontsize=8)
        ax.set_title(f"{vin:g} V")
        ax.set_xlabel("x (m, base frame)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (m, base frame)")
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="downward force at TCP (N)")
    fig.suptitle(f"Static press force map, z = {result['z']:.3f} m")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_figures(out_dir: str | Path = "docs/media/robot") -> dict:
    """Build a default arm, grid a plausible build-plate area, and save the force-map figure."""
    world = mujoco.MjSpec()
    attach_arm(world)
    model = world.compile()
    kin = ArmKinematics(model)

    xs = np.linspace(0.08, 0.30, 23)
    ys = np.linspace(-0.16, 0.16, 21)
    result = press_force_map(model, kin, xs, ys, z=0.02)
    path = figure_force_map(result, Path(out_dir) / "force_map.png")
    return {"force_map": path, "result": result}
