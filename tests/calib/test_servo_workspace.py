"""The settling-error map against the virtual bus: reachable points get a finite mm error,
unreachable ones are reported rather than raising, and the table/summary render."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brick_builder.calib.servo_bench import bare_virtual_arm
from brick_builder.calib.servo_workspace import (
    WorkspaceGridConfig,
    grid_targets,
    measure_workspace_map,
)
from brick_builder.hardware.bus import Bus, BusSettings


@pytest.fixture
def virtual_arm():
    scene, vbus = bare_virtual_arm(seed=0)
    vbus.start()
    bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
    try:
        bus.enable_torque(check=False)
        yield bus, scene
    finally:
        bus.close()
        vbus.stop()
        vbus.raise_if_failed()


def test_grid_targets_cover_every_radius_and_angle():
    cfg = WorkspaceGridConfig(radii_m=(0.10, 0.20), angles_deg=(-30.0, 0.0, 30.0))
    targets = grid_targets(cfg)
    assert len(targets) == 6
    for r, a, xyz in targets:
        assert xyz[2] == pytest.approx(cfg.z_m)
        assert math.hypot(xyz[0], xyz[1]) == pytest.approx(r, abs=1e-9)


def test_reachable_points_settle_close_to_their_command(virtual_arm):
    bus, scene = virtual_arm
    cfg = WorkspaceGridConfig(radii_m=(0.15,), angles_deg=(-20.0, 0.0, 20.0),
                              settle_ticks=60, settle_avg_ticks=15)
    wmap = measure_workspace_map(bus, scene.kin, cfg)
    assert len(wmap.points) == 3
    assert all(p.reachable for p in wmap.points)
    summary = wmap.summary()
    assert summary["n"] == 3
    # No gravity feed-forward here (this is a raw bus command, not `RealArmWorld`), so the droop
    # is real; the map exists to measure it, not to assume it away. A working IK/FK round trip
    # still has to land within a couple of centimetres, not tens of centimetres.
    assert summary["max_mm"] < 30.0
    for p in wmap.points:
        assert np.all(np.isfinite(p.settled_xyz_m))


def test_an_unreachable_point_is_reported_not_raised(virtual_arm):
    bus, scene = virtual_arm
    cfg = WorkspaceGridConfig(radii_m=(2.0,), angles_deg=(0.0,), settle_ticks=10)
    wmap = measure_workspace_map(bus, scene.kin, cfg)
    assert len(wmap.points) == 1
    assert wmap.points[0].reachable is False
    assert wmap.points[0].detail
    assert wmap.summary()["n"] == 0
    assert "unreachable" in wmap.table()
