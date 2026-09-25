"""Settling-error map: commanded vs settled tool position over a workspace grid, from bus
readings and forward kinematics alone.

Every grid point is a tool-down pose (:func:`~brick_builder.scene.compose.tool_down_ik`); the arm
is driven there with :meth:`~brick_builder.hardware.bus.Bus.cycle` exactly like any other bring-up
measurement, held long enough to settle, and the settled tool position comes from forward
kinematics on the *measured* joint angles.  Unlike ``servo_sysid.py``'s per-joint step and ramp
tests, which isolate one servo law parameter at a time, this map reports what a caller of
``Bus.cycle`` actually gets: the P-gain droop, the dry friction, the gear play and the firmware
dead band of all five arm joints, together, at wherever in the workspace the task needs the tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..hardware.bus import Bus
from ..robot.kinematics import ArmKinematics

N_ARM = 5


@dataclass(frozen=True)
class WorkspaceGridConfig:
    """A polar grid, tool pointing straight down, one fixed height."""

    radii_m: tuple[float, ...] = (0.10, 0.15, 0.20)
    angles_deg: tuple[float, ...] = (-50.0, -25.0, 0.0, 25.0, 50.0)
    z_m: float = 0.03
    """Tool height above the arm's own base plane -- a plausible near-table height; there is no
    table in the bare-arm scene this runs against, so nothing stops a lower value, but this is
    what the grid ships with."""
    settle_ticks: int = 100
    """5 s at 50 Hz -- past the worst settling time this project has measured (guide 03's 0.5 s
    nominal, times a wide margin for a joint deep in friction plus gear play)."""
    settle_avg_ticks: int = 20
    """Ticks at the end of the hold the settled reading is averaged over, same idea as
    ``servo_sysid.py``'s ``SETTLE_TICKS``."""
    yaw_rad: float = 0.0


@dataclass
class GridPoint:
    radius_m: float
    angle_deg: float
    target_xyz_m: np.ndarray
    reachable: bool
    settled_xyz_m: np.ndarray | None = None
    error_mm: float | None = None
    detail: str = ""


@dataclass
class WorkspaceMap:
    points: list[GridPoint]
    config: WorkspaceGridConfig

    @property
    def reached(self) -> list[GridPoint]:
        return [p for p in self.points if p.reachable]

    def summary(self) -> dict[str, float]:
        errs = np.array([p.error_mm for p in self.reached if p.error_mm is not None])
        if errs.size == 0:
            return {"n": 0, "n_unreachable": len(self.points),
                    "mean_mm": math.nan, "p95_mm": math.nan, "max_mm": math.nan}
        return {
            "n": int(errs.size), "n_unreachable": len(self.points) - int(errs.size),
            "mean_mm": float(errs.mean()), "p95_mm": float(np.percentile(errs, 95)),
            "max_mm": float(errs.max()),
        }

    def table(self) -> str:
        lines = [
            "settling-error map, tool down", "",
            "| radius [mm] | angle [deg] | target x,y,z [mm] | settled x,y,z [mm] | error [mm] |",
            "|---|---|---|---|---|",
        ]
        for p in self.points:
            if not p.reachable:
                lines.append(f"| {p.radius_m * 1e3:.0f} | {p.angle_deg:.0f} | - | - | "
                             f"unreachable ({p.detail}) |")
                continue
            t = p.target_xyz_m * 1e3
            s = p.settled_xyz_m * 1e3
            lines.append(
                f"| {p.radius_m * 1e3:.0f} | {p.angle_deg:.0f} | "
                f"{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f} | {s[0]:.1f}, {s[1]:.1f}, {s[2]:.1f} | "
                f"{p.error_mm:.2f} |"
            )
        summary = self.summary()
        lines.append("")
        if summary["n"]:
            lines.append(f"n = {summary['n']} reached ({summary['n_unreachable']} unreachable), "
                         f"mean {summary['mean_mm']:.2f} mm, p95 {summary['p95_mm']:.2f} mm, "
                         f"max {summary['max_mm']:.2f} mm")
        else:
            lines.append("no grid point was reachable")
        return "\n".join(lines)


def grid_targets(cfg: WorkspaceGridConfig) -> list[tuple[float, float, np.ndarray]]:
    """Every ``(radius, angle, target_xyz)`` the grid asks for, angle measured from +x."""
    out = []
    for r in cfg.radii_m:
        for a in cfg.angles_deg:
            rad = math.radians(a)
            out.append((r, a, np.array([r * math.cos(rad), r * math.sin(rad), cfg.z_m])))
    return out


def measure_workspace_map(
    bus: Bus, kin: ArmKinematics, cfg: WorkspaceGridConfig = WorkspaceGridConfig()
) -> WorkspaceMap:
    """Drive every grid point in turn, tool down, and report commanded vs settled.

    Points are visited in the grid's own order and each pose is seeded from the previous one's
    solution (nearer poses converge faster and stay on the same IK branch), so an unreachable
    point in the middle of a run does not perturb the ones after it beyond losing a good seed.
    """
    from ..scene.compose import tool_down_ik

    points: list[GridPoint] = []
    q_seed = None
    for r, a, target in grid_targets(cfg):
        point = GridPoint(radius_m=r, angle_deg=a, target_xyz_m=target, reachable=False)
        try:
            q = tool_down_ik(kin, target, cfg.yaw_rad, q0=q_seed)
        except RuntimeError as exc:
            point.detail = str(exc)
            points.append(point)
            continue
        q_seed = q
        goal = np.zeros(6)
        goal[:N_ARM] = q
        history = []
        for k in range(cfg.settle_ticks):
            state = bus.cycle(goal)
            if k >= cfg.settle_ticks - cfg.settle_avg_ticks:
                history.append(state.position.copy())
        measured = np.mean(np.stack(history), axis=0)
        tool_pos, _ = kin.fk(measured[:N_ARM])
        point.reachable = True
        point.settled_xyz_m = tool_pos
        point.error_mm = float(np.linalg.norm(tool_pos - target) * 1e3)
        points.append(point)
    return WorkspaceMap(points, cfg)


def figure_workspace_map(wmap: WorkspaceMap, out_path) -> None:
    """A top-down scatter of the grid, each point coloured by its settling error [mm]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reached = wmap.reached
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=120)
    if reached:
        xy = np.array([p.target_xyz_m[:2] for p in reached]) * 1e3
        errs = np.array([p.error_mm for p in reached])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=errs, cmap="viridis", s=90, edgecolors="#222")
        fig.colorbar(sc, ax=ax, label="settling error [mm]")
    missed = [p for p in wmap.points if not p.reachable]
    if missed:
        xy = np.array([p.target_xyz_m[:2] for p in missed]) * 1e3
        ax.scatter(xy[:, 0], xy[:, 1], marker="x", c="#b0413e", s=60, label="unreachable")
        # Every grid target has x > 0 (the heading sweep is +-50 deg either side of +x, so
        # cos(heading) never reaches 0), which leaves the left half of the axes clear of data for
        # the legend -- "upper right" sat on top of a real point at one radius/heading.
        ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Settling-error map, tool down")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
