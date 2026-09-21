"""The loose supply bricks: bricks on a table, and nothing else holding them.

The hardware is a stock arm on a bare table with no tray or fixture, so the closed-loop
pick-and-place scene puts its spare bricks straight on the table. These checks are the ones that would catch that quietly changing: that the bricks exist
with their own bodies, that no tray gets built around them, that they rest on the table rather
than hovering or sinking, and that the tray option still works for the scenes that use it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from brick_builder.scene.compose import (
    SUPPLY_PREFIX, SceneConfig, build_scene, reset_scene, settle,
)


def _geom_names(scene):
    return {
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for g in range(scene.model.ngeom)
    }


@pytest.fixture(scope="module")
def supply_scene():
    cfg = SceneConfig(held=None, placed=[], tray=False, cameras=(),
                      supply=((2, 2, "blue"), (2, 4, "orange")))
    scene = build_scene(cfg)
    reset_scene(scene, np.random.default_rng(0))
    return scene


def test_the_supply_bricks_exist_and_nothing_else_does(supply_scene) -> None:
    names = sorted(n for n in supply_scene.bricks if n.startswith(SUPPLY_PREFIX))
    assert names == ["supply_brick_0", "supply_brick_1"]
    assert not any(g.startswith("tray_") for g in _geom_names(supply_scene)), "a tray was built"


def test_they_lie_on_the_table_and_stay_there(supply_scene) -> None:
    settle(supply_scene, 1.0)
    for name in (n for n in supply_scene.bricks if n.startswith(SUPPLY_PREFIX)):
        z = float(supply_scene.body_pose(name)[0][2])
        assert -0.001 <= z <= 0.001, f"{name} settled at z={z * 1e3:.2f} mm, not on the table"


def test_they_are_far_enough_apart_for_the_open_jaws(supply_scene) -> None:
    """Wide enough that the tool can come down around one without touching its neighbour."""
    xy = [supply_scene.body_pose(n)[0][:2]
          for n in sorted(supply_scene.bricks) if n.startswith(SUPPLY_PREFIX)]
    gap = float(np.linalg.norm(xy[1] - xy[0]))
    assert gap > 0.060, f"only {gap * 1e3:.0f} mm between supply bricks"


def test_the_tray_still_works_and_is_off_by_default() -> None:
    assert SceneConfig().tray is False
    assert SceneConfig().supply == ()
    scene = build_scene(SceneConfig(held=None, placed=[], tray=True, tray_bricks=2, cameras=()))
    reset_scene(scene, np.random.default_rng(0))
    assert any(g.startswith("tray_") for g in _geom_names(scene))
    assert sorted(n for n in scene.bricks if n.startswith("tray_brick_")) == [
        "tray_brick_0", "tray_brick_1"
    ]
