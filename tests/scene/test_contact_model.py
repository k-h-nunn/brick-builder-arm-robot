"""The contact model a composed scene is built with, and what it costs.

The claim this file keeps honest is that the simulator's contacts come from the material table
rather than from MuJoCo's shipped defaults. Three things have to be true for that to mean
anything: the derived numbers reach the compiled model, the fall-back switch really falls back,
and the penetration the derived setting produces is both small and the same at 2, 1 and 0.5 ms.

The second clip is the part worth reading. The derivation lands on a 4 ms time constant, which
is the solver's own floor at a 2 ms step, and 4 ms jams the press: a brick lowered onto the
plate stops on the stud tips, 4.50 mm (one stud height) short of seated. Nothing between 4 and
20 ms rescues it either, so the pair that carries the press fit keeps MuJoCo's own settings and
the derived one stays selectable. The measurements are in `docs/guide/02d`.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.bricks import materials as mat
from brick_builder.bricks.generator import (
    CONTACT_TIMESTEP,
    MUJOCO_CONTACT,
    add_baseplate,
    add_brick,
    material_contact,
)
from brick_builder.scene.compose import (
    TABLE_GEOM,
    SceneConfig,
    build_scene,
    scene_contact,
)


def _geom_solref(model, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, name
    return tuple(model.geom_solref[gid]), tuple(model.geom_solimp[gid])


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------


def test_the_derived_stiffness_comes_from_the_contact_modulus_and_a_stated_patch():
    """`k = 2 a E*`, with `E*` from the material and `a` written down rather than tuned."""

    e_star = mat.ABS_BRICK_23C.contact_modulus
    assert e_star == pytest.approx(1.311e9, rel=0.01)
    stud_r = mat.DUPLO.stud_d / 2.0 - mat.DUPLO.clearance
    assert mat.flat_punch_stiffness(stud_r, e_star) == pytest.approx(2.0 * stud_r * e_star)

    s = mat.contact_settings(CONTACT_TIMESTEP)["brick_on_brick"]
    assert s.stiffness_N_m == pytest.approx(2.0 * stud_r * e_star, rel=1e-9)
    # The time constant is the mass-spring period against a stated mass, and nothing else.
    assert s.solref_derived[0] == pytest.approx(
        math.sqrt(mat.CONTACT_REFERENCE_MASS_KG / s.stiffness_N_m)
    )


def test_a_dissimilar_pair_uses_the_series_contact_modulus():
    """PLA on ABS is stiffer than ABS on ABS, and a rigid table stiffer still."""

    abs_abs = mat.pair_contact_modulus(mat.ABS_BRICK_23C, mat.ABS_BRICK_23C)
    pla_abs = mat.pair_contact_modulus(mat.ABS_BRICK_23C, mat.PLA_PRINTED)
    rigid = mat.pair_contact_modulus(mat.ABS_BRICK_23C, None)
    assert abs_abs < pla_abs < rigid
    assert rigid == pytest.approx(2.0 * abs_abs)


# ---------------------------------------------------------------------------
# what reaches the compiled model
# ---------------------------------------------------------------------------


def test_the_derived_model_writes_the_material_contact_onto_the_table():
    """The derived model applies the material value to the table, with priority 1 so the table
    decides a brick-on-table contact outright."""

    scene = build_scene(SceneConfig(contact_model="derived"))
    table = material_contact(pair="brick_on_table", press_limited=False)
    ref, imp = _geom_solref(scene.model, TABLE_GEOM)
    assert ref == pytest.approx(table.solref)
    assert imp == pytest.approx(table.solimp)
    gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    assert int(scene.model.geom_priority[gid]) == 1


def test_the_scene_default_keeps_mujocos_contact_on_the_table():
    """The default keeps MuJoCo's own settings on the table as it does on the press pair: the
    closed-loop pick-and-place's grasp and seating depth were measured against them (derived
    table contact: 0.39 mm short of full depth; MuJoCo's: 0.00 mm)."""

    scene = build_scene(SceneConfig())
    ref, imp = _geom_solref(scene.model, TABLE_GEOM)
    assert tuple(ref) == pytest.approx((0.02, 1.0))
    assert tuple(imp)[:3] == pytest.approx((0.9, 0.95, 0.001))
    gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    assert int(scene.model.geom_priority[gid]) == 0


def test_the_press_pair_keeps_mujocos_own_settings_and_says_so():
    scene = build_scene(SceneConfig())
    press = material_contact(pair="brick_on_brick")
    assert press.solref == pytest.approx(mat.MUJOCO_DEFAULT_SOLREF)
    for name in ("held_col_wall_front", "held_col_tube_0", "plate_col_slab"):
        ref, imp = _geom_solref(scene.model, name)
        assert ref == pytest.approx(mat.MUJOCO_DEFAULT_SOLREF), name
        assert imp == pytest.approx(mat.MUJOCO_DEFAULT_SOLIMP), name


def test_the_visual_geoms_are_left_alone():
    """Only collision geoms carry contact settings; a visual geom never collides."""

    scene = build_scene(SceneConfig(contact_model="derived"))
    ref, imp = _geom_solref(scene.model, "held_vis_wall_front")
    assert ref == pytest.approx(mat.MUJOCO_DEFAULT_SOLREF)
    assert imp == pytest.approx(mat.MUJOCO_DEFAULT_SOLIMP)
    ref, _ = _geom_solref(scene.model, "held_col_wall_front")
    assert ref[0] == pytest.approx(2.0 * CONTACT_TIMESTEP)


def test_the_switch_falls_back_to_mujocos_own_defaults():
    scene = build_scene(SceneConfig(contact_model="mujoco"))
    for name in ("held_col_wall_front", "plate_col_slab", TABLE_GEOM):
        ref, imp = _geom_solref(scene.model, name)
        assert ref == pytest.approx(mat.MUJOCO_DEFAULT_SOLREF), name
        assert imp == pytest.approx(mat.MUJOCO_DEFAULT_SOLIMP), name
    assert scene_contact(SceneConfig(contact_model="mujoco")) is MUJOCO_CONTACT


def test_the_derived_setting_is_still_selectable_even_though_it_jams_the_press():
    """The 4 ms the derivation asks for is reachable, and it is what the fall-back is against.

    It is not the default because a brick lowered onto the plate with it stops on the stud tips
    (measured: 4.50 mm short of seated, one stud height, on every placement seed). See
    `docs/guide/02d`.
    """

    scene = build_scene(SceneConfig(contact_model="derived"))
    for name in ("held_col_wall_front", "plate_col_slab", TABLE_GEOM):
        ref, imp = _geom_solref(scene.model, name)
        assert ref[0] == pytest.approx(2.0 * CONTACT_TIMESTEP), name
        assert imp[:3] == pytest.approx((0.95, 0.999, 0.0005)), name
    assert 2.0 * CONTACT_TIMESTEP < material_contact(pair="brick_on_brick").solref[0]


def test_the_jaw_pads_keep_condim_four_and_priority_one():
    """The pads already carried the derived setting; nothing here may take it away."""

    scene = build_scene(SceneConfig())
    for name in ("fixed_pad", "moving_pad"):
        gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, name
        assert int(scene.model.geom_condim[gid]) == 4
        assert int(scene.model.geom_priority[gid]) == 1
        assert tuple(scene.model.geom_solref[gid]) == pytest.approx((0.004, 1.0))


# ---------------------------------------------------------------------------
# penetration, and that it does not move with the step
# ---------------------------------------------------------------------------


def _resting_penetration(timestep: float, contact) -> float:
    """How far a 2x4 sinks into a baseplate under its own weight [m], once settled."""

    world = mujoco.MjSpec()
    world.option.timestep = timestep
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.impratio = 10.0
    world.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    world.option.iterations = 100
    plate = add_baseplate(world, "plate", 4, 6, pos=(0.0, 0.0, 0.0), contact=contact)
    top = 2.0 * plate.spec.roof + plate.spec.stud_h
    add_brick(world, "brick", 2, 4, "red", pos=(0.0, 0.0, top + 0.0002), contact=contact)
    model = world.compile()
    data = mujoco.MjData(model)
    for _ in range(int(round(1.5 / timestep))):
        mujoco.mj_step(model, data)
    worst = 0.0
    for c in range(data.ncon):
        worst = max(worst, -float(data.contact.dist[c]))
    return worst


def test_resting_penetration_is_far_under_the_press_fit_interference():
    """A 12.5 g brick resting on the plate must not sink anywhere near the 86 um fit."""

    for contact in (material_contact(), material_contact(press_limited=False)):
        depth = _resting_penetration(0.002, contact)
        assert depth < 20e-6, f"{contact.name}: resting penetration {depth * 1e6:.1f} um"


def test_resting_penetration_is_converged_in_the_time_step():
    """2, 1 and 0.5 ms have to agree: the time constant sets the stiffness, not the step.

    The contact model is fixed at `CONTACT_TIMESTEP`, so a run stepped finer keeps the same
    settings and must therefore read the same penetration.
    """

    contact = material_contact()
    depths = [_resting_penetration(dt, contact) for dt in (0.002, 0.001, 0.0005)]
    ref = depths[0]
    for dt, d in zip((0.002, 0.001, 0.0005), depths):
        assert abs(d - ref) <= 0.05 * max(ref, 1e-9) + 1e-7, (
            f"{dt * 1e3:g} ms read {d * 1e6:.2f} um against {ref * 1e6:.2f} um at 2 ms"
        )


def test_the_derived_setting_is_stiffer_than_the_one_that_ships():
    """Which is the whole reason it cannot ship: stiffer is what jams the press.

    Measured: 0.32 um against 15.80 um at a 2 ms step, a factor of fifty.
    """

    shipped = material_contact(pair="brick_on_brick")
    derived = material_contact(pair="brick_on_brick", press_limited=False)
    assert derived.solref[0] < shipped.solref[0]
    assert _resting_penetration(0.002, derived) < 0.25 * _resting_penetration(0.002, shipped)


def test_the_resting_penetration_of_each_model_is_what_the_guide_quotes():
    assert _resting_penetration(0.002, material_contact()) == pytest.approx(15.8e-6, rel=0.1)
    assert _resting_penetration(
        0.002, material_contact(press_limited=False)
    ) == pytest.approx(0.32e-6, rel=0.2)
