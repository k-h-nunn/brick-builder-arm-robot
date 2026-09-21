"""Tests for brick_builder.bricks.damage.

These run the simulator and check what the damage estimate says about real contact forces.
The ones that matter are the pair at the top of the guide's section 5 - a correct press must
accumulate no structural damage, and a wall pushed hard enough must crack - plus the
convergence check of section 6.4, which the first version of the misalignment rig failed.
"""

import mujoco
import numpy as np
import pytest

from brick_builder.bricks import materials as mat
from brick_builder.bricks.bench import BenchConfig, PressBench, build_stack_scene
from brick_builder.bricks.damage import (
    MODES,
    STATES,
    DamageModel,
    DamageThresholds,
    build_press_rig,
    build_wall_rig,
    misalignment_sweep,
    run_press,
    run_wall_push,
    wall_push_onset,
)
from brick_builder.bricks.snap import ColletSnap
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import DUPLO


def _settle(bench: PressBench, snap, steps: int, speed: float = 0.002, damage=None):
    start = float(bench.data.qpos[bench._ram_qadr])
    t0 = float(bench.data.time)
    peak = 0.0
    for _ in range(steps):
        bench.data.ctrl[0] = start - speed * (bench.data.time - t0)
        mujoco.mj_step(bench.model, bench.data)
        snap.update(bench.model, bench.data)
        if damage is not None:
            damage.update()
        peak = max(peak, float(-bench.data.actuator_force[0]))
    return peak


# ---------------------------------------------------------------------------
# The reading itself
# ---------------------------------------------------------------------------


def test_contact_forces_match_mj_contact_force_exactly():
    """The vectorised gather is the same numbers ``mj_contactForce`` gives, not an estimate."""

    snap = ColletSnap(SnapParams())
    model, data, bricks = build_stack_scene(snap, [(0, 0), (0, 1), (0, 2)])
    for _ in range(200):
        mujoco.mj_step(model, data)
        snap.update(model, data)
    damage = DamageModel(model, data, bricks, snap=snap)
    _, _, world = damage.contact_forces()
    assert data.ncon > 0
    buf = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, buf)
        frame = data.contact.frame[i].reshape(3, 3)
        assert np.allclose(frame.T @ buf[:3], world[i], atol=1e-12)


def test_every_collision_geom_of_a_brick_is_classified():
    """A 2x4 has 16 collision primitives: 4 walls, 3 tubes, 8 studs and a roof."""

    snap = ColletSnap(SnapParams())
    bench = PressBench(snap, BenchConfig())
    damage = DamageModel(bench.model, bench.data, bench.bricks, snap=snap)
    tagged = damage.g_brick >= 0
    assert int(tagged.sum()) == 16
    parts = damage.g_part[tagged]
    counts = {int(p): int((parts == p).sum()) for p in np.unique(parts)}
    from brick_builder.bricks.damage import PART_ROOF, PART_STUD, PART_TUBE, PART_WALL

    assert counts[PART_WALL] == 4
    assert counts[PART_TUBE] == 3
    assert counts[PART_STUD] == 8
    assert counts[PART_ROOF] == 1
    # Visual geoms never collide, so they must not be tagged as a load path.
    for gi in np.nonzero(tagged)[0]:
        name = mujoco.mj_id2name(bench.model, mujoco.mjtObj.mjOBJ_GEOM, int(gi))
        assert "_col_" in name


def test_section_properties_come_from_the_materials_module():
    snap = ColletSnap(SnapParams())
    bench = PressBench(snap, BenchConfig())
    damage = DamageModel(bench.model, bench.data, bench.bricks, snap=snap)
    sections = mat.sections(DUPLO, 2, 4)
    from brick_builder.bricks.damage import PART_STUD

    studs = np.nonzero((damage.g_brick >= 0) & (damage.g_part == PART_STUD))[0]
    assert studs.size == 8
    assert damage.g_area[studs[0]] == pytest.approx(sections["stud"].area)
    assert damage.g_modulus[studs[0]] == pytest.approx(sections["stud"].modulus)


# ---------------------------------------------------------------------------
# A correct press does no harm
# ---------------------------------------------------------------------------


def test_a_correct_press_at_15_newtons_does_no_structural_damage():
    """The central negative control. A brick pressed on straight stays intact.

    The press is driven past seating so the ram reads well over 15 N, which is above the arm's
    own 10 to 14 N ceiling, and still nothing bends: the load runs down the walls and tubes as
    compression, and the wall-bending mode reads exactly zero because it uses the force
    component perpendicular to the wall's own plane.
    """

    snap = ColletSnap(SnapParams())
    bench = PressBench(snap, BenchConfig())
    damage = DamageModel(bench.model, bench.data, bench.bricks, snap=snap)
    peak = _settle(bench, snap, 2200, damage=damage)
    assert peak >= 15.0, f"the rig only reached {peak:.1f} N; the test needs at least 15 N"

    record = damage.brick("brick")
    assert record.state == "intact"
    allowable = mat.ABS_BRICK_23C.flexural_strength.nominal
    for mode in ("wall_bending", "stud_bending", "roof_bending", "stud_shear"):
        # Not exactly zero: the friction tangents of a vertical contact put a few pascals into
        # the bending modes. A few pascals against a 70 MPa allowable is seven orders of
        # magnitude of margin, which is what "no structural damage" has to mean in a solver
        # that resolves friction.
        assert record.peak_by_mode[mode] < allowable * 1e-4, f"{mode} loaded by a straight press"
    assert record.peak_ratio < 1.0
    assert damage.summary()["n_cracked"] == 0
    assert not damage.any_damage()


def test_a_force_controlled_aligned_press_seats_and_stays_intact():
    result = run_press(build_press_rig(offset_x=0.0), 15.0)
    assert result["seated"]
    assert result["state"] == "intact"
    assert result["peak_wall_stress_Pa"] == 0.0
    # Quasi-static means quasi-static: the reading is taken with the ram at rest, not on its
    # way through. A millimetre per second on a 0.5 kg ram is 0.25 uJ of kinetic energy.
    assert result["settled"] and abs(result["ram_v_m_s"]) < 1e-3
    # The approach ends where the brick meets the plate, not three seconds later in mid-air.
    # An aligned brick makes no geometric contact on the way in (the snap model owns the fit),
    # so the gap closing is what ends it.
    assert result["t_contact_s"] < 1.0
    assert not result["contacted"]


def test_a_vertical_load_on_a_wall_is_compression_not_bending():
    """Mode discrimination: the wall-bending stress must use the perpendicular component only.

    A brick resting flat puts its whole weight down the walls. If the model used the force
    magnitude instead of the component normal to the wall, a brick sitting on a table would
    read a bending stress, and every episode would accumulate damage for doing nothing.
    """

    snap = ColletSnap(SnapParams())
    model, data, bricks = build_stack_scene(snap, [(0, 0)])
    damage = DamageModel(model, data, bricks, snap=snap)
    for _ in range(400):
        mujoco.mj_step(model, data)
        snap.update(model, data)
        damage.update()
    record = damage.brick("brick0")
    assert record.peak_normal_N > 0.0, "the brick is not actually resting on anything"
    # A few pascals of residual from the friction tangents, against a 70 MPa allowable. If the
    # model used the force magnitude instead of the perpendicular component this would be
    # megapascals: the brick's own weight on a 15 mm lever over a 1.55 mm wall.
    assert record.peak_by_mode["wall_bending"] < 1e4
    assert record.state == "intact"


# ---------------------------------------------------------------------------
# A misaligned press cannot reach a wall, and the sweep that says so converges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset_mm", [-0.18, -0.2, -0.5, -1.9, -2.1])
def test_a_misaligned_press_never_loads_a_wall(offset_mm):
    """The geometry forbids it, and this is the assertion that keeps the claim honest.

    Stud and tube overlap by 86 um on the radius, so about 0.18 mm of misalignment already
    stops a stud entering its cell; the brick then stands on the stud tips. The nearest wall is
    1.65 mm away - nineteen times further - so no misalignment lets a plate stud reach a wall's
    inner face. Every newton of lateral wall load the first version of this rig reported came
    from an under-damped ram plunging into an overlap a quasi-static press cannot create.
    """

    result = run_press(build_press_rig(offset_x=offset_mm * 1e-3), 20.0)
    assert result["peak_wall_stress_Pa"] == 0.0
    assert result["state"] == "intact"
    assert result["min_depth_m"] == 0.0, "a stud got in at a misalignment that should jam it"
    assert abs(result["ram_v_m_s"]) < 2e-3, "the rig had not settled when it was read"


def test_the_misalignment_sweep_is_converged_in_the_time_step():
    """The acceptance check the review asked for, and the one the old rig failed.

    The old rig read "broken" at a 2 ms step and "undamaged" at 1 ms for the same physical
    scenario, because its ram overshot during the ramp and the damage model latched the
    overshoot. This runs the identical experiment - identical in seconds, not in steps - at
    2 ms, 1 ms and 0.5 ms, and requires the peak lateral force and the peak stress ratio to
    agree within 10 % and the damage state to be identical at every offset.
    """

    offsets = np.array([0.0, -0.10, -0.15, -0.18, -0.20, -0.30, -1.00]) * 1e-3
    runs = {dt: misalignment_sweep(offsets, 20.0, timestep=dt)
            for dt in (0.002, 0.001, 0.0005)}

    for j, offset in enumerate(offsets):
        states = {dt: runs[dt][j]["state"] for dt in runs}
        assert len(set(states.values())) == 1, (
            f"at {offset * 1e3:.2f} mm the damage state depends on the time step: {states}"
        )
        for key, floor in (("peak_ratio", 1e-3), ("peak_lateral_N", 0.05)):
            values = np.array([runs[dt][j][key] for dt in runs])
            if values.max() <= floor:      # nothing to compare: all three read zero
                continue
            spread = (values.max() - values.min()) / values.max()
            assert spread <= 0.10, (
                f"{key} at {offset * 1e3:.2f} mm spans {spread * 100:.1f} % over "
                f"2/1/0.5 ms: {values}"
            )
        depths = np.array([runs[dt][j]["min_depth_m"] for dt in runs])
        assert depths.max() - depths.min() < 5e-5, (
            f"seating depth at {offset * 1e3:.2f} mm depends on the time step: {depths}"
        )

    # A convergence check on a sweep where nothing happens is not a check. The sweep has to
    # cross the seating cliff and has to load something somewhere, or agreement is vacuous.
    at_2ms = runs[0.002]
    assert at_2ms[0]["seated"] and not at_2ms[-1]["seated"]
    assert max(r["peak_ratio"] for r in at_2ms) > 0.01
    assert all(r["settled"] for r in at_2ms)


def test_press_force_decides_seating_and_never_produces_a_wall_load():
    """The press force decides how deep the brick goes. It never decides what gets damaged.

    Aligned, the rig reproduces the push-on force the clutch implies: with the simulator's
    1.0 N per stud and beta = 0.15, a 2x4 needs 8 x 1.0 x 1.15 = 9.2 N to go all the way in,
    and the rig seats at 10 N and stalls part-way at 8 N. Misaligned past the jam, the same
    range of force changes nothing at all, and no wall is loaded either way.
    """

    params = SnapParams()
    needed = 8 * params.f_clutch * (1.0 + params.beta)
    below = run_press(build_press_rig(offset_x=0.0), 0.85 * needed)
    above = run_press(build_press_rig(offset_x=0.0), 1.1 * needed)
    assert not below["seated"] and below["min_depth_m"] < 0.9 * DUPLO.stud_h
    assert above["seated"]
    assert below["peak_wall_stress_Pa"] == 0.0 and above["peak_wall_stress_Pa"] == 0.0

    jammed = [run_press(build_press_rig(offset_x=-0.5e-3), force) for force in (10.0, 30.0)]
    assert all(r["min_depth_m"] == 0.0 for r in jammed)
    assert all(r["peak_wall_stress_Pa"] == 0.0 for r in jammed)
    assert all(r["state"] == "intact" for r in jammed)


# ---------------------------------------------------------------------------
# A wall loaded the way section 7.7 loads one does crack
# ---------------------------------------------------------------------------


def test_the_wall_rig_reproduces_the_beam_formula():
    """A rod on the inside of a wall must read exactly 6 F L / (b_eff t^2), at any time step.

    This is the whole point of a well-posed experiment: the contact normal does not depend on
    which way the solver resolves a corner, so the answer is arithmetic and the simulator has
    to agree with it.
    """

    for timestep in (0.002, 0.001, 0.0005):
        rig = build_wall_rig(timestep=timestep)
        result = run_wall_push(rig, 40.0)
        assert result["analytic_wall_stress_Pa"] == pytest.approx(
            mat.wall_bending_stress(40.0, rig.lever_m, rig.b_eff_m, DUPLO.wall)
        )
        assert result["peak_wall_stress_Pa"] == pytest.approx(
            result["analytic_wall_stress_Pa"], rel=0.02
        )
        assert result["peak_mode"] == "wall_bending"
        assert abs(result["rod_v_m_s"]) < 1e-4


def test_wall_stress_rises_with_force_and_crosses_cracked_then_broken():
    """The guide's failure mode, reproduced the way a bench would reproduce it.

    Section 7.7 pushes a rounded rod against the inside of a wall near its bottom edge and
    reads the force at a visible set and at a crack. This is that test, and the stress has to
    rise in proportion to the force and cross both allowables in the right order.
    """

    forces = np.array([10.0, 30.0, 50.0, 64.0, 80.0, 96.0])
    rows = [run_wall_push(build_wall_rig(), float(f)) for f in forces]
    stress = np.array([r["peak_wall_stress_Pa"] for r in rows])
    states = [r["state"] for r in rows]

    # Proportional, not merely monotone: sigma / F must be the same number every time.
    per_newton = stress / forces
    assert per_newton.std() / per_newton.mean() < 0.02

    flexural = mat.ABS_BRICK_23C.flexural_strength.nominal
    assert stress.min() < flexural < stress.max()
    assert states[0] == "intact"
    assert states[-1] == "broken"
    first_crack = min(i for i, s in enumerate(states) if s in ("cracked", "broken"))
    assert stress[first_crack] >= flexural
    assert stress[first_crack - 1] < flexural

    # Where it lands, in newtons. The guide's band is 19 to 56 N for a load spread of 10 to
    # 30 mm; this rod spreads over b_eff = 2L + 4 mm = 34 mm, so it must sit just above the
    # top of that band and nowhere near an order of magnitude from it.
    crack_lo, crack_hi = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)["crack_band_N"]
    assert crack_hi <= forces[first_crack] <= 1.5 * crack_hi


def test_where_damage_starts_is_far_beyond_what_the_arm_can_apply():
    """The headline of section 6.4: 64 N to crack a wall against 10 to 14 N of press."""

    onset = wall_push_onset()["onset_N"]
    assert onset["cracked"] is not None and onset["broken"] is not None
    assert onset["broken"] > onset["cracked"]
    # Four and a half times the arm's press ceiling at the nominal fillet ...
    assert onset["cracked"] >= 4.0 * max(mat.ARM_PRESS_FORCE_N)
    # ... and still twice it at the pessimistic K_t of 2.0, which halves every threshold.
    pessimistic = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C, k_t=2.0)["crack_band_N"]
    nominal = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)["crack_band_N"]
    assert pessimistic[0] == pytest.approx(nominal[0] / 2.0)
    assert 2.0 * max(mat.ARM_PRESS_FORCE_N) <= onset["cracked"] / 2.0 < 4.0 * max(
        mat.ARM_PRESS_FORCE_N
    )


def test_the_implied_load_spread_is_physically_bounded():
    """Back out b_eff from the reported stress and check it lies between a rod and a wall.

    The 45 degree spread rule is the weakest assumption in the whole estimate, so it is worth
    checking that it never produces a width narrower than the contact or wider than the member.
    """

    rig = build_wall_rig()
    result = run_wall_push(rig, 40.0)
    stress = result["peak_wall_stress_Pa"]
    assert stress > 0.0
    # sigma = 6 F L / (b t^2), with L the lever the rig places the rod at.
    b_eff = 6.0 * 40.0 * rig.lever_m / (stress * DUPLO.wall**2)
    assert 2.0 * rig.rod_radius_m <= b_eff
    assert b_eff <= mat.sections(DUPLO, 2, 4)["wall_left_right"].length * 1.05
    assert b_eff == pytest.approx(rig.b_eff_m, rel=0.02)


# ---------------------------------------------------------------------------
# The guard against latching a solver transient
# ---------------------------------------------------------------------------


def _push_with_pulse(rig, base_N: float, pulse_N: float, pulse_s: float) -> dict:
    """Load the wall steadily at ``base_N``, then step to ``pulse_N`` for ``pulse_s``."""

    m, d = rig.model, rig.data
    dt = m.opt.timestep
    body = rig.rod_body

    def step(force):
        d.xfrc_applied[body] = 0.0
        d.xfrc_applied[body, 0] = force
        mujoco.mj_step(m, d)
        rig.damage.update()

    for _ in range(int(3.0 / dt)):
        step(0.5)
        if d.ncon > 0:
            break
    n_ramp = int(0.3 / dt)
    for i in range(n_ramp):
        step(base_N * (i + 1) / n_ramp)
    for _ in range(int(0.1 / dt)):
        step(base_N)
    for _ in range(max(int(round(pulse_s / dt)), 1)):
        step(pulse_N)
    for _ in range(int(0.2 / dt)):
        step(base_N)
    d.xfrc_applied[body] = 0.0
    record = rig.damage.brick("brick")
    return {"state": record.state, "peak_ratio": record.peak_ratio,
            "longest_overload_s": record.longest_overload_s}


def test_a_sustained_overload_registers():
    """The guard must not make the model blind: a real overload still latches, and says so."""

    result = run_wall_push(build_wall_rig(), 96.0)
    assert result["state"] == "broken"
    assert result["peak_ratio"] > 1.5
    # It held for most of a second, which is what makes it an overload and not a spike.
    assert result["longest_overload_s"] > 0.3
    events = build_wall_rig().damage.events
    assert events == []


def test_a_brief_spike_does_not_latch_but_the_same_spike_held_does():
    """The guard is a filter on duration, and this checks it filters where it says it does.

    Same rod, same force, same contact: only the length of the excursion differs. A 6 ms spike
    past the flexural strength leaves the brick intact while still recording the peak it
    reached; the identical spike held for 200 ms cracks it. The physical justification for
    10 ms is in DamageThresholds.persist_s - a collision that can crack a wall lasts 25 ms and
    carries 150 mN*s, a one-step solver artefact lasts 2 ms and carries a few mN*s.
    """

    brief = _push_with_pulse(build_wall_rig(), 50.0, 96.0, 0.006)
    assert brief["peak_ratio"] > 1.0, "the spike did not actually pass an allowable"
    # The pulse is 6 ms past the flexural strength, so the crack rung cannot latch. It is
    # necessarily longer than that past the tensile yield - the ramp up and the ramp down both
    # pass through it - so the permanent-set rung does latch, which is the right answer: a wall
    # held past 44 MPa for more than 10 ms keeps the bend.
    assert brief["state"] == "yielded", "a 6 ms spike cracked the brick"

    held = _push_with_pulse(build_wall_rig(), 50.0, 96.0, 0.2)
    assert held["state"] in ("cracked", "broken")
    assert held["longest_overload_s"] >= 0.010
    assert held["peak_ratio"] == pytest.approx(brief["peak_ratio"], rel=0.05)


def test_turning_the_guard_off_lets_the_same_spike_through():
    """Which proves the guard is what made the difference, not some other change.

    With persist_s below the pulse length the same 6 ms spike latches, so the model has not
    simply stopped noticing: the duration threshold is doing exactly the filtering it claims.
    """

    thresholds = DamageThresholds.from_material(mat.ABS_BRICK_23C)
    fast = DamageThresholds(**{**vars(thresholds), "persist_s": 0.002})
    result = _push_with_pulse(build_wall_rig(thresholds=fast), 50.0, 96.0, 0.006)
    assert result["state"] in ("cracked", "broken")

    # Turning it off entirely must mean "latch on the first step past an allowable", not
    # "latch on any step at all": a brick under a load it can carry stays intact.
    off = DamageThresholds(**{**vars(thresholds), "persist_s": 0.0})
    safe = run_wall_push(build_wall_rig(thresholds=off), 30.0)
    assert safe["peak_ratio"] < 1.0 and safe["state"] == "intact"
    over = run_wall_push(build_wall_rig(thresholds=off), 96.0)
    assert over["state"] == "broken"


def test_instantaneous_peaks_are_recorded_whether_or_not_they_persist():
    """Peaks and states are computed separately, and the peak is never filtered.

    With the guard set longer than the run, a genuine overload leaves the brick intact but the
    recorded peak stress is identical to the run where it latches. A reader who wants to know
    what force appeared, rather than what the brick ended up as, still gets an honest answer.
    """

    thresholds = DamageThresholds.from_material(mat.ABS_BRICK_23C)
    deaf = DamageThresholds(**{**vars(thresholds), "persist_s": 1e6})
    guarded = run_wall_push(build_wall_rig(thresholds=deaf), 96.0)
    normal = run_wall_push(build_wall_rig(), 96.0)
    assert guarded["state"] == "intact"
    assert normal["state"] == "broken"
    assert guarded["peak_wall_stress_Pa"] == pytest.approx(
        normal["peak_wall_stress_Pa"], rel=1e-6
    )
    assert guarded["peak_ratio"] == pytest.approx(normal["peak_ratio"], rel=1e-6)


# ---------------------------------------------------------------------------
# Accumulation, consequences, state
# ---------------------------------------------------------------------------


def test_the_wall_fillet_concentration_moves_the_estimate_the_way_it_should():
    """K_t multiplies the wall-bending stress and nothing else, and it only ever weakens.

    The stress the rod produces is a fact about the contact; K_t is a statement about the
    corner it is bending. So the same push has to read K_t times the stress and crack at K_t
    times less force, with every other mode untouched.
    """

    base = DamageThresholds.from_material(mat.ABS_BRICK_23C)
    sharp = DamageThresholds(**{**vars(base), "wall_kt": 2.0})
    nominal = run_wall_push(build_wall_rig(), 40.0)
    concentrated = run_wall_push(build_wall_rig(thresholds=sharp), 40.0)

    assert concentrated["peak_wall_stress_Pa"] == pytest.approx(
        2.0 * nominal["peak_wall_stress_Pa"], rel=0.01
    )
    # 40 N bends the wall past its 44 MPa yield but not past its 70 MPa flexural strength, so
    # the nominal corner takes a permanent set; the sharp one cracks at the same load.
    assert nominal["state"] == "yielded"
    assert concentrated["state"] in ("cracked", "broken")
    assert mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C, k_t=2.0)["crack_band_N"][0] == (
        pytest.approx(mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)["crack_band_N"][0] / 2.0)
    )


def test_thresholds_follow_the_material_and_scale_together():
    thresholds = DamageThresholds.from_material(mat.ABS_BRICK_23C, scale=0.7)
    assert thresholds.allowable("wall_bending") == pytest.approx(0.7 * 70e6)
    assert thresholds.allowable("stud_shear") == pytest.approx(0.7 * 26e6)
    assert thresholds.allowable("marking") == pytest.approx(0.7 * 44e6)
    assert set(MODES) == {
        "wall_bending", "stud_shear", "stud_bending",
        "tube_ovalisation", "roof_bending", "marking",
    }


def test_a_weaker_brick_cracks_where_a_nominal_one_does_not():
    """Randomizing the allowables must actually change where the boundary sits."""

    weak = DamageThresholds.from_material(mat.ABS_BRICK_23C, scale=0.7)
    nominal = run_wall_push(build_wall_rig(), 50.0)
    fragile = run_wall_push(build_wall_rig(thresholds=weak), 50.0)
    assert nominal["state"] == "yielded"      # past yield, short of the flexural strength
    assert fragile["state"] in ("cracked", "broken")
    # The stress itself is unchanged: it is the allowable that moved, not the physics.
    assert fragile["peak_wall_stress_Pa"] == pytest.approx(
        nominal["peak_wall_stress_Pa"], rel=1e-6
    )


def test_states_latch_and_never_improve():
    damage = build_press_rig(offset_x=-2.1e-3).damage
    damage.mark_broken("brick", mode="test")
    assert damage.worst_state() == "broken"
    damage.update()
    assert damage.worst_state() == "broken"
    assert damage.states() == {"brick": "broken"}
    assert damage.summary()["n_cracked"] == 1


def test_reward_penalty_is_paid_once_per_transition():
    rig = build_press_rig(offset_x=0.0)
    assert rig.damage.reward_penalty() == 0.0
    rig.damage.mark_broken("brick")
    first = rig.damage.reward_penalty()
    assert first < 0.0
    assert rig.damage.reward_penalty() == 0.0     # the increment, not the state


def test_apply_consequences_scales_the_clutch_through_a_public_field():
    rig = build_press_rig(offset_x=0.0)
    model, damage = rig.model, rig.damage
    dofs = damage.collet_dofs[0]
    assert dofs.size == 24, "8 sockets x (1 axial + 2 lateral) collet dofs"
    run_press(rig, 10.0, ramp_s=0.2, hold_s=0.1)
    before = model.dof_frictionloss[dofs].copy()
    damage.mark_broken("brick")
    damage.apply_consequences()
    after = model.dof_frictionloss[dofs]
    assert np.all(after <= before + 1e-15)
    assert np.any(after < before), "a broken brick's clutch was not reduced"
    assert np.all(after >= 0.3 * before - 1e-15)


def test_cycle_counting_and_fatigue_need_a_load_excursion():
    rig = build_press_rig(offset_x=0.0)
    run_press(rig, 10.0, ramp_s=0.3, hold_s=0.2)
    record = rig.damage.brick("brick")
    assert record.cycles >= 0
    assert record.fatigue >= 0.0
    assert 0.2 <= record.clutch_multiplier <= 1.0


def test_the_fatigue_constants_describe_the_event_they_are_anchored_on():
    """The cited rig halved the clutch of a System brick in 37,112 cycles. So must the model.

    Cantin's own write-up says "both pieces failed at 50 %" after 37,112 assemble-separate
    cycles, so the number the constants have to reproduce is a clutch multiplier of 0.5 at
    cycles_ref, not the `loosened` boundary of 0.8 - which arrives 2.5x sooner and which no
    source speaks to.
    """

    f = mat.ABS_BRICK_23C.fatigue
    cycles_to = lambda multiplier: (1.0 - multiplier) / f.fatigue_weight * f.cycles_ref
    assert cycles_to(0.5) == pytest.approx(f.cycles_ref)
    assert cycles_to(0.5) == pytest.approx(37112.0, rel=1e-6)
    loosened = DamageThresholds().loosened_clutch
    assert cycles_to(loosened) == pytest.approx(14845.0, rel=0.01)
    assert cycles_to(f.clutch_floor) == pytest.approx(59379.0, rel=0.01)
    # One cycle at the reference stress is one cycles_ref-th of the sum, by construction.
    assert (f.s_ref / f.s_ref) ** f.exponent / f.cycles_ref == pytest.approx(1.0 / 37112.0)


# ---------------------------------------------------------------------------
# Reproducibility and save/restore
# ---------------------------------------------------------------------------


def test_damage_is_reproducible_under_a_seed():
    results = [
        run_press(build_press_rig(offset_x=-0.18e-3, seed=11), 20.0) for _ in range(2)
    ]
    assert results[0]["peak_ratio"] > 0.0, "the scenario has to load something to be a check"
    assert results[0]["peak_ratio"] == results[1]["peak_ratio"]
    assert results[0]["peak_normal_N"] == results[1]["peak_normal_N"]
    assert results[0]["state"] == results[1]["state"]


def test_randomized_thresholds_are_reproducible_under_a_seed():
    snap = ColletSnap(SnapParams())
    bench = PressBench(snap, BenchConfig())
    a = DamageModel(bench.model, bench.data, bench.bricks, snap=snap, seed=5, randomize=True)
    b = DamageModel(bench.model, bench.data, bench.bricks, snap=snap, seed=5, randomize=True)
    c = DamageModel(bench.model, bench.data, bench.bricks, snap=snap, seed=6, randomize=True)
    assert a.thresholds == b.thresholds
    assert a.thresholds != c.thresholds
    lo, _, hi = mat.randomization_ranges()["allowable_scale"]
    assert lo <= a.thresholds.allowable_scale <= hi


def test_damage_state_survives_save_and_restore():
    """A restored model must continue a run identically, which is what checkpointing needs.

    The persistence timers are part of that state: a brick that is nine milliseconds into an
    overload when the checkpoint is taken has to be nine milliseconds into it when the
    checkpoint is restored, or a restored run would need a fresh 10 ms to latch anything.
    """

    rig = build_wall_rig(seed=3)
    run_wall_push(rig, 60.0, ramp_s=0.3, hold_s=0.1)
    saved = rig.damage.get_state()
    sim_state = np.zeros(mujoco.mj_stateSize(rig.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(rig.model, rig.data, sim_state, mujoco.mjtState.mjSTATE_INTEGRATION)

    continued = run_wall_push(rig, 96.0, ramp_s=0.05, hold_s=0.15)

    mujoco.mj_setState(rig.model, rig.data, sim_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    rig.damage.set_state(saved)
    replayed = run_wall_push(rig, 96.0, ramp_s=0.05, hold_s=0.15)

    assert continued["state"] == "broken", "the continuation has to do something to be a check"
    assert replayed["peak_wall_stress_Pa"] == pytest.approx(
        continued["peak_wall_stress_Pa"], rel=1e-9
    )
    assert replayed["state"] == continued["state"]
    assert replayed["longest_overload_s"] == pytest.approx(
        continued["longest_overload_s"], rel=1e-9
    )


def test_get_state_is_a_deep_copy():
    rig = build_press_rig(offset_x=0.0)
    saved = rig.damage.get_state()
    copy = saved.copy()
    saved.fatigue[0] = 123.0
    assert copy.fatigue[0] == 0.0


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_the_estimate_is_cheap_next_to_a_physics_step():
    """It has to be affordable at training scale: one pass over the contact list per step."""

    from brick_builder.bricks.damage import benchmark

    snap = ColletSnap(SnapParams())
    model, data, bricks = build_stack_scene(snap, [(0, 0), (0, 1), (0, 2)])
    for _ in range(100):
        mujoco.mj_step(model, data)
        snap.update(model, data)
    result = benchmark(model, data, bricks, snap, steps=200)
    assert result["damage_us_per_call"] > 0.0
    # Generous, because the machine this runs on may be busy; the point is that it is a
    # fraction of a step and not a multiple of one.
    assert result["fraction_of_step"] < 1.0
