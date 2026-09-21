"""Tests for brick_builder.bricks.breakable.

The breakable brick exists to check that the cheap estimate's thresholds sit where pieces
actually come apart, so these tests are about *where* it breaks and where it does not: a crush
three orders of magnitude beyond the arm breaks it, the arm's own press does not, and an
ordinary drop does not either.
"""

import mujoco
import numpy as np
import pytest

from brick_builder.bricks import materials as mat
from brick_builder.bricks import breakable as brk
from brick_builder.bricks.damage import DamageModel
from brick_builder.bricks.generator import add_baseplate
from brick_builder.bricks.snap import ColletSnap
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import DUPLO


# ---------------------------------------------------------------------------
# It is the same brick
# ---------------------------------------------------------------------------


def test_a_breakable_brick_has_the_same_pieces_and_names_as_a_generated_one():
    world = mujoco.MjSpec()
    handle = brk.add_breakable_brick(world, "brick", 2, 4, "red", (0.0, 0.0, 0.0))
    model = world.compile()
    # 15 separable pieces (4 walls, 3 tubes, 8 studs), each welded to the roof they hang off.
    assert len(handle.pieces) == 15
    assert model.nbody == 1 + 1 + 15         # world, roof, fifteen pieces
    assert model.neq == 15
    assert len(handle.handle.stud_sites) == 8
    assert len(handle.handle.socket_sites) == 8
    for name in ("brick_col_roof", "brick_col_wall_front", "brick_col_tube_0",
                 "brick_col_stud_1_3"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


def test_the_pieces_add_up_to_the_right_mass():
    """A brick built from sixteen pieces must still mass what a DUPLO 2x4 masses."""

    world = mujoco.MjSpec()
    brk.add_breakable_brick(world, "brick", 2, 4, "red", (0.0, 0.0, 0.0))
    model = world.compile()
    total = sum(
        model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
        for name in (
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
        )
    )
    assert total * 1000.0 == pytest.approx(12.2, rel=0.05)


def test_an_intact_breakable_brick_holds_itself_together():
    """The welds must be quiet under self weight: 0.12 N against limits of a kilonewton."""

    model, data, handle, manager = brk.build_crush_scene(ram=False)
    for _ in range(400):
        mujoco.mj_step(model, data)
        manager.update()
    roof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brick")
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brick_piece_wall_front")
    assert abs(data.xpos[roof][2] - data.xpos[wall][2]) < 1e-4
    assert manager.n_released == 0
    assert manager.intact()
    force, moment = manager.weld_loads()
    assert force.max() < 1.0
    assert manager.peak_utilisation.max() < 0.01


def test_the_press_fit_model_still_works_on_an_intact_breakable_brick():
    """The sites are where the generator puts them, so ColletSnap binds unchanged."""

    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    world.default.geom.solref = [0.004, 1.0]
    world.default.geom.solimp = [0.95, 0.999, 0.0005, 0.5, 2.0]
    plate = add_baseplate(world, "plate", 4, 6, pos=(0.0, 0.0, 0.0))
    top = DUPLO.roof * 2.0
    handle = brk.add_breakable_brick(world, "brick", 2, 4, "red", (0.0, 0.0, top))
    snap = ColletSnap(SnapParams())
    snap.declare(world, [handle.handle], [plate])
    model = world.compile()
    data = mujoco.MjData(model)
    snap.bind(model, data)
    manager = brk.BreakableBricks([handle])
    manager.bind(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
        snap.update(model, data)
        manager.update()
    connections = snap.connections()
    assert connections, "the collet model captured nothing on a breakable brick"
    assert connections[0].n_engaged == 8
    assert connections[0].seated
    assert manager.n_released == 0


# ---------------------------------------------------------------------------
# What breaks it, and what does not
# ---------------------------------------------------------------------------


def test_twenty_kilonewtons_breaks_a_brick_and_twenty_newtons_does_not():
    """The guide's crush number, either side. The arm produces 10 to 14 N."""

    model, data, handle, manager = brk.build_crush_scene()
    for _ in range(200):
        mujoco.mj_step(model, data)
        manager.update()
    gentle = brk.crush(model, data, manager, 20.0, steps=400, ramp_steps=200)
    assert gentle["n_released"] == 0
    assert gentle["peak_utilisation"] < 0.1

    model, data, handle, manager = brk.build_crush_scene()
    for _ in range(200):
        mujoco.mj_step(model, data)
        manager.update()
    crushed = brk.crush(model, data, manager, 20_000.0, steps=600, ramp_steps=300)
    assert crushed["n_released"] >= 1
    assert crushed["peak_utilisation"] >= 1.0
    kinds = {e["kind"] for e in crushed["events"]}
    assert "wall" in kinds, "a crush should fail the walls, which buckle before anything yields"


def test_the_crush_threshold_sits_near_the_derived_capacity():
    """It must not break at 2 kN and must break at 20 kN: the capacity is 13.2 kN."""

    for force, should_break in ((2_000.0, False), (20_000.0, True)):
        model, data, handle, manager = brk.build_crush_scene()
        for _ in range(200):
            mujoco.mj_step(model, data)
            manager.update()
        result = brk.crush(model, data, manager, force, steps=600, ramp_steps=300)
        assert (result["n_released"] > 0) is should_break, (
            f"{force:.0f} N released {result['n_released']} welds "
            f"(peak utilisation {result['peak_utilisation']:.2f})"
        )
    capacity = mat.crush_load(DUPLO, mat.ABS_BRICK_23C, 2, 4)["total_N"]
    assert 2_000.0 < capacity < 20_000.0


@pytest.mark.parametrize("height", [0.3, 1.0])
def test_a_dropped_brick_does_not_fracture(height):
    """The guide's negative control, and the one it says any damage model must pass.

    ABS has a notched Izod impact strength of 300 J/m and DUPLO bricks survive being dropped by
    toddlers all day. A 1 m drop reaches 4.4 m/s, which the guide's ``F = v sqrt(m k)`` puts at
    about 160 N against weld limits of 1.4 kN and up. Nothing should come apart.
    """

    model, data, handle, manager = brk.build_crush_scene(drop_height=height, ram=False)
    result = brk.drop(model, data, manager, height, steps=900)
    assert result["impact_speed_m_s"] == pytest.approx(np.sqrt(2 * 9.81 * height), rel=0.01)
    assert result["n_released"] == 0
    assert result["peak_utilisation"] < 1.0

    predicted = mat.impact_force(0.0125, 1.0e5, result["impact_speed_m_s"])
    weakest = min(p.force_N for p in handle.pieces)
    assert predicted < weakest, (
        f"the guide's own estimate ({predicted:.0f} N) already exceeds the weakest weld "
        f"({weakest:.0f} N); the negative control would not be meaningful"
    )


def test_a_released_weld_changes_the_appearance_and_records_the_event():
    model, data, handle, manager = brk.build_crush_scene()
    piece = handle.pieces[0]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, piece.geom_names[-1])
    before = model.geom_rgba[gid].copy()
    for _ in range(200):
        mujoco.mj_step(model, data)
        manager.update()
    brk.crush(model, data, manager, 20_000.0, steps=600, ramp_steps=300)
    assert manager.events
    event = manager.events[0]
    assert event.utilisation >= 1.0
    assert event.brick == "brick"
    assert event.time > 0.0
    released = {e.piece for e in manager.events}
    for name in released:
        p = next(q for q in handle.pieces if q.name == name)
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p.geom_names[-1])
        assert np.allclose(model.geom_rgba[g], brk.FRACTURED_RGBA)
    assert not np.allclose(model.geom_rgba[gid], before) or piece.name not in released
    summary = manager.summary()
    assert summary["n_released"] == len(manager.events)
    assert summary["n_welds"] == 15


def test_a_fracture_latches_the_damage_model_to_broken():
    model, data, handle, manager = brk.build_crush_scene()
    damage = DamageModel(model, data, [handle.handle], track_connections=False)
    manager.damage = damage
    for _ in range(200):
        mujoco.mj_step(model, data)
        manager.update()
    assert damage.worst_state() == "intact"
    brk.crush(model, data, manager, 20_000.0, steps=600, ramp_steps=300)
    assert damage.worst_state() == "broken"
    assert any(e.mode == "weld_release" for e in damage.events)


# ---------------------------------------------------------------------------
# Limits and cost
# ---------------------------------------------------------------------------


def test_weld_limits_come_from_the_material_and_scale():
    world = mujoco.MjSpec()
    nominal = brk.add_breakable_brick(world, "a", 2, 4, "red", (0.0, 0.0, 0.0))
    weak = brk.add_breakable_brick(world, "b", 2, 4, "blue", (0.1, 0.0, 0.0), limit_scale=0.5)
    limits = mat.weld_limits(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    tube = next(p for p in nominal.pieces if p.kind == "tube")
    assert tube.force_N == pytest.approx(limits["tube"].force_N)
    assert tube.moment_Nm == pytest.approx(limits["tube"].moment_Nm)
    weak_tube = next(p for p in weak.pieces if p.kind == "tube")
    assert weak_tube.force_N == pytest.approx(0.5 * tube.force_N)


def test_folding_the_studs_into_the_roof_halves_the_piece_count():
    world = mujoco.MjSpec()
    handle = brk.add_breakable_brick(
        world, "brick", 2, 4, "red", (0.0, 0.0, 0.0), include_studs=False
    )
    model = world.compile()
    assert len(handle.pieces) == 7           # 4 walls and 3 tubes; the studs join the roof
    assert model.neq == 7
    # The mass must not go missing when the studs stop being their own bodies.
    total = sum(model.body_mass[i] for i in range(1, model.nbody))
    assert total * 1000.0 == pytest.approx(12.2, rel=0.05)


@pytest.mark.slow
def test_the_breakable_brick_is_much_more_expensive_than_a_monolithic_one():
    """Which is why it is for validation, not for training."""

    result = brk.cost_comparison(2, 4, steps=150)
    assert result["breakable"]["nbody"] > result["monolithic"]["nbody"]
    assert result["breakable"]["nv"] == 96
    assert result["breakable"]["neq"] == 15
    assert result["slowdown"] > 1.5
