"""Action to servo-bus writes: accumulated offset, recorded feed-forward, one tick of commands.

The action `u` in `[-1, 1]^n` is the change of the tool-target offset in units of the stage's
per-tick scale; the offset accumulates and is held inside the stage's ceiling. A tick is always
five 20 ms writes, so one step is always 100 ms of physics:

- feed-forward on, advancing tick: frame k's five recorded writes, each plus the offset;
- feed-forward on, held tick: frame k's last recorded write, five times (idempotent);
- feed-forward off (L0a): the start pose plus the offset, five times.

Offsets are in the nominal-plate world frame. `u = 0` with the feed-forward on issues exactly the
recorded writes: open-loop replay.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DIM_INDEX", "apply_action", "tick_commands"]

DIM_INDEX = {"dx": 0, "dy": 1, "dz": 2, "dyaw": 3}
"""Offset component of each action dimension: (x, y, z [m], yaw [rad])."""


def apply_action(offset, u, *, dims, scale, ceiling) -> np.ndarray:
    """`offset + scale * u`, then held inside `ceiling = (xy norm, |z|, |yaw|)`."""
    u = np.clip(np.asarray(u, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if u.shape != (len(dims),):
        raise ValueError(f"action must have {len(dims)} components; got {u.shape}")
    out = np.asarray(offset, dtype=np.float64).copy()
    for i, name in enumerate(dims):
        out[DIM_INDEX[name]] += float(scale[i]) * float(u[i])
    xy_max, z_max, yaw_max = (float(c) for c in ceiling)
    n = float(np.hypot(out[0], out[1]))
    if n > xy_max:
        out[:2] *= xy_max / n if n > 0 else 0.0
    out[2] = float(np.clip(out[2], -z_max, z_max))
    out[3] = float(np.clip(out[3], -yaw_max, yaw_max))
    return out


def tick_commands(commands: np.ndarray, n_commands: np.ndarray, k: int, *, held: bool,
                  offset, feedforward: bool, start_command=None) -> np.ndarray:
    """`[5, 7]` writes for one tick: (x, y, z, spin, grip, c_vert, c_lat).

    `commands`/`n_commands` are a `TargetSequence`'s arrays; `start_command` (`[7]`, the start
    pose with its grip and compliance) is what the feed-forward-off case holds."""
    T = commands.shape[0]
    k = int(min(max(k, 0), T - 1))
    if feedforward:
        if held:
            last = commands[k, int(n_commands[k]) - 1]
            out = np.repeat(last[None, :], 5, axis=0).copy()
        else:
            out = commands[k].copy()
    else:
        if start_command is None:
            raise ValueError("feed-forward off needs the start command")
        base = np.asarray(start_command, dtype=np.float64).reshape(7)
        out = np.repeat(base[None, :], 5, axis=0).copy()
    off = np.asarray(offset, dtype=np.float64)
    out[:, :3] += off[:3]
    out[:, 3] += off[3]
    return out
