"""Tests for brick_builder.bricks.generator.

Physics tests build a fresh `mujoco.MjSpec` per test (never share a compiled model across
tests) and check real simulated behaviour, not just "it compiles".
"""

from __future__ import annotations

import numpy as np
import pytest

import mujoco

from brick_builder.bricks.generator import (
    COLLISION_GROUP,
    MARKER_SITE_GROUP,
    VISUAL_GROUP,
    add_baseplate,
    add_brick,
)
from brick_builder.bricks.spec import DUPLO, PALETTE


# ---------------------------------------------------------------------------
# Site / tube counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nx, ny, n_studs, n_tubes", [(2, 4, 8, 3), (2, 2, 4, 1)])
def test_stud_socket_and_tube_counts(nx, ny, n_studs, n_tubes):
    world = mujoco.MjSpec()
    handle = add_brick(world, "b", nx, ny, "red", pos=(0.0, 0.0, 0.05))
    assert len(handle.stud_sites) == n_studs
    assert len(handle.socket_sites) == n_studs

    model = world.compile()
    tube_geoms = [
        i
        for i in range(model.ngeom)
        if model.geom(i).name.startswith("b_col_tube_")
    ]
    assert len(tube_geoms) == n_tubes


def test_collision_geoms_are_all_convex_primitives():
    """No mesh geom may be used for the whole-brick hull; a mesh is only ever a small stud
    chamfer tip, never the roof/wall/tube shapes."""

    world = mujoco.MjSpec()
    add_brick(world, "b", 2, 4, "red", pos=(0, 0, 0.05), chamfer=True)
    model = world.compile()
    for i in range(model.ngeom):
        geom = model.geom(i)
        if geom.group[0] != COLLISION_GROUP:
            continue
        if int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            assert "chamfer" in geom.name or "tip" in geom.name, geom.name
        else:
            assert int(geom.type[0]) in (
                int(mujoco.mjtGeom.mjGEOM_BOX),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            )


def test_collision_and_visual_groups_are_separate_and_visual_does_not_collide():
    world = mujoco.MjSpec()
    add_brick(world, "b", 2, 2, "green", pos=(0, 0, 0.05))
    model = world.compile()
    groups = set()
    for i in range(model.ngeom):
        g = model.geom(i)
        groups.add(int(g.group[0]))
        if g.group[0] == VISUAL_GROUP:
            assert g.contype[0] == 0
            assert g.conaffinity[0] == 0
    assert groups == {COLLISION_GROUP, VISUAL_GROUP}


@pytest.mark.parametrize("colour", ["red", "blue", "yellow"])
def test_every_visible_part_of_a_brick_has_the_brick_colour_studs_included(colour):
    # A stud is moulded with its brick. Studs left at MuJoCo's default grey put a grey disc on
    # about a quarter of every top face, in every camera image and every training frame.
    world = mujoco.MjSpec()
    add_brick(world, "b", 2, 4, colour, pos=(0, 0, 0.05))
    add_baseplate(world, "plate", 4, 4, pos=(0.2, 0, 0), colour="green")
    model = world.compile()
    seen = {"b": 0, "plate": 0}
    for i in range(model.ngeom):
        g = model.geom(i)
        if g.group[0] != VISUAL_GROUP:
            continue
        owner = "b" if g.name.startswith("b_") else "plate"
        want = PALETTE[colour] if owner == "b" else PALETTE["green"]
        np.testing.assert_allclose(g.rgba[:3], want, atol=1e-6, err_msg=g.name)
        seen[owner] += 1
    assert seen["b"] >= 1 + 4 + 3 + 8  # roof, walls, tubes, studs
    assert seen["plate"] == 1 + 16  # slab, studs


def test_stud_and_socket_marker_sites_are_in_a_group_no_camera_draws():
    world = mujoco.MjSpec()
    add_brick(world, "b", 2, 2, "red", pos=(0, 0, 0.05))
    add_baseplate(world, "plate", 2, 2, pos=(0.2, 0, 0))
    model = world.compile()
    drawn_by_default = np.flatnonzero(mujoco.MjvOption().sitegroup)
    assert model.nsite == 4 + 4 + 4
    for i in range(model.nsite):
        assert int(model.site(i).group[0]) == MARKER_SITE_GROUP
        assert int(model.site(i).group[0]) not in drawn_by_default


# ---------------------------------------------------------------------------
# Mass and inertia
# ---------------------------------------------------------------------------


def test_2x4_mass_within_10_percent_of_measured():
    world = mujoco.MjSpec()
    handle = add_brick(world, "b", 2, 4, "red", pos=(0, 0, 0.05))
    model = world.compile()
    body_id = model.body(handle.name).id
    mass = model.body_mass[body_id]
    assert mass == pytest.approx(0.0122, rel=0.10)


@pytest.mark.parametrize("nx, ny", [(2, 4), (2, 2), (1, 4), (2, 6)])
def test_inertia_is_physical(nx, ny):
    """Positive definite (diag > 0) and satisfies the triangle inequality every physical
    rigid body's principal moments must obey: the sum of any two >= the third."""

    world = mujoco.MjSpec()
    handle = add_brick(world, "b", nx, ny, "blue", pos=(0, 0, 0.05))
    model = world.compile()
    body_id = model.body(handle.name).id
    ixx, iyy, izz = model.body_inertia[body_id]
    assert ixx > 0 and iyy > 0 and izz > 0
    assert ixx + iyy >= izz - 1e-12
    assert iyy + izz >= ixx - 1e-12
    assert izz + ixx >= iyy - 1e-12


# ---------------------------------------------------------------------------
# Drop test: rests flat at z=0
# ---------------------------------------------------------------------------


def test_brick_dropped_on_plane_rests_flat_at_z0():
    world = mujoco.MjSpec()
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.2, 0.2, 0.1])
    add_brick(world, "b", 2, 4, "red", pos=(0.0, 0.0, 0.05))
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    n_steps = int(2.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[2]) < 1e-4  # 0.1 mm
    # settled flat: roll/pitch negligible (quaternion close to identity up to sign)
    quat = data.qpos[3:7]
    assert abs(abs(quat[0]) - 1.0) < 1e-3
    assert data.warning.number.sum() == 0


# ---------------------------------------------------------------------------
# Kinematic insertion test
# ---------------------------------------------------------------------------


def _lower_onto_baseplate(offset_xy: tuple[float, float], start_z: float = 0.03, target_end: float = -0.02, n: int = 4000):
    """Kinematically lower a 2x4 brick onto a 4x8 baseplate along z with a soft position
    actuator on its own slide joint (x, y and rotation are held fixed, matching the
    acceptance test's "perfect alignment" / "4 mm lateral offset" wording)."""

    world = mujoco.MjSpec()
    plate = add_baseplate(world, "plate", 4, 8, pos=(0.0, 0.0, 0.0))
    handle = add_brick(
        world, "brick", 2, 4, "blue", pos=(offset_xy[0], offset_xy[1], start_z), free=False
    )
    joint = handle.body.add_joint(
        name="slide_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], damping=2.0
    )
    actuator = world.add_actuator(name="act_z", target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    kp = 50.0
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    gainprm = np.zeros(10)
    gainprm[0] = kp
    actuator.gainprm = gainprm
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    biasprm = np.zeros(10)
    biasprm[1] = -kp
    actuator.biasprm = biasprm
    actuator.ctrlrange = [-1.0, 1.0]

    model = world.compile()
    data = mujoco.MjData(model)
    for t in range(n):
        frac = t / n
        data.ctrl[0] = (target_end - start_z) * frac
        mujoco.mj_step(model, data)
    final_z = start_z + data.qpos[0]
    return final_z, int(data.warning.number.sum()), plate


def test_aligned_insertion_reaches_baseplate_surface():
    from brick_builder.bricks.generator import _BASEPLATE_THICKNESS_FACTOR

    final_z, n_warn, _ = _lower_onto_baseplate((0.0, 0.0))
    plate_top = DUPLO.roof * _BASEPLATE_THICKNESS_FACTOR
    assert final_z == pytest.approx(plate_top, abs=3e-4)  # within 0.3 mm of flush
    assert n_warn == 0


def test_misaligned_insertion_rests_on_stud_tops_not_flush():
    from brick_builder.bricks.generator import _BASEPLATE_THICKNESS_FACTOR

    final_z, n_warn, _ = _lower_onto_baseplate((0.004, 0.0))  # 4 mm lateral offset
    plate_top = DUPLO.roof * _BASEPLATE_THICKNESS_FACTOR
    expected_rest = plate_top + DUPLO.stud_h
    assert final_z == pytest.approx(expected_rest, abs=3e-4)
    assert n_warn == 0


def test_aligned_vs_misaligned_travel_is_one_stud_height():
    z_aligned, _, _ = _lower_onto_baseplate((0.0, 0.0))
    z_offset, _, _ = _lower_onto_baseplate((0.004, 0.0))
    assert (z_offset - z_aligned) == pytest.approx(DUPLO.stud_h, abs=2e-4)  # +-0.2 mm


# ---------------------------------------------------------------------------
# Two-brick stack stability
# ---------------------------------------------------------------------------


def test_two_stacked_bricks_stay_still_for_3s():
    from brick_builder.bricks.generator import _BASEPLATE_THICKNESS_FACTOR

    world = mujoco.MjSpec()
    plate = add_baseplate(world, "plate", 4, 8, pos=(0.0, 0.0, 0.0))
    plate_top = DUPLO.roof * _BASEPLATE_THICKNESS_FACTOR
    h1 = add_brick(world, "b1", 2, 4, "blue", pos=(0.0, 0.0, plate_top), free=True)
    h2 = add_brick(world, "b2", 2, 4, "red", pos=(0.0, 0.0, plate_top + DUPLO.height), free=True)
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    b1_adr = model.joint(f"{h1.name}_free").qposadr[0]
    b2_adr = model.joint(f"{h2.name}_free").qposadr[0]
    start = data.qpos[[b1_adr, b1_adr + 1, b1_adr + 2, b2_adr, b2_adr + 1, b2_adr + 2]].copy()

    n_steps = int(3.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    end = data.qpos[[b1_adr, b1_adr + 1, b1_adr + 2, b2_adr, b2_adr + 1, b2_adr + 2]].copy()

    drift = np.abs(end - start)
    assert np.all(drift < 2e-4)  # 0.2 mm: "keep still"
    assert data.warning.number.sum() == 0


# ---------------------------------------------------------------------------
# Baseplate basics
# ---------------------------------------------------------------------------


def test_baseplate_has_no_joint_and_no_socket_sites():
    world = mujoco.MjSpec()
    handle = add_baseplate(world, "plate", 4, 8, pos=(0.0, 0.0, 0.0))
    assert handle.socket_sites == []
    assert len(handle.stud_sites) == 4 * 8
    model = world.compile()
    body_id = model.body(handle.name).id
    # a body with no joint has jnt_num 0
    assert model.body_jntnum[body_id] == 0


def test_add_brick_rejects_degenerate_footprint():
    world = mujoco.MjSpec()
    with pytest.raises(ValueError):
        add_brick(world, "b", 0, 4, "red", pos=(0, 0, 0.05))
