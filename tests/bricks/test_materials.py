"""Tests for brick_builder.bricks.materials.

These check physics, not execution. Every expected value is a number from
`docs/guide/02c-materials-and-damage.md`, which is the specification this module implements,
and the tolerances are set by how precisely that page states the number - tight where the
arithmetic is exact, loose where the page itself gives a band.
"""

import math

import mujoco
import pytest

from brick_builder.bricks import materials as mat
from brick_builder.bricks.generator import add_brick
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import DUPLO, LEGO_SYSTEM


# ---------------------------------------------------------------------------
# Geometry of the press fit (guide section 2.1)
# ---------------------------------------------------------------------------


def test_duplo_interference_is_86_microns():
    """d = pitch sqrt(2)/2 = 11.314 mm against r_stud + r_tube = 11.400 mm."""

    geom = mat.press_fit_geometry(DUPLO, 2, 4)
    assert geom.half_diagonal * 1e3 == pytest.approx(11.314, abs=0.001)
    assert geom.radius_sum * 1e3 == pytest.approx(11.400, abs=0.001)
    assert geom.interference * 1e6 == pytest.approx(86.3, abs=0.5)
    assert geom.has_geometric_clutch


def test_the_walls_are_not_in_contact():
    """The wall inner face is 1.65 mm clear of the nearest stud surface.

    A real DUPLO brick bridges that gap with internal ribs; the generator models none. Any
    analysis that puts the clutch in the walls is describing a brick this project does not have.
    """

    assert mat.press_fit_geometry(DUPLO, 2, 4).wall_gap * 1e3 == pytest.approx(1.65, abs=0.01)


@pytest.mark.parametrize(
    "nx, ny, n_contacts", [(2, 2, 4), (2, 4, 12), (2, 6, 20), (4, 4, 36), (1, 4, 0)]
)
def test_stud_on_tube_contact_count(nx, ny, n_contacts):
    """12 contacts on a 2x4, 1.5 per stud - counted from the grid, not assumed."""

    geom = mat.press_fit_geometry(DUPLO, nx, ny)
    assert geom.n_contacts == n_contacts
    if nx > 1 and ny > 1:
        assert geom.contacts_per_stud == pytest.approx(n_contacts / (nx * ny))


def test_lego_system_variant_has_no_geometric_clutch():
    """A finding, not a passing grade: LEGO_SYSTEM as configured has a 1.9 um *clearance*.

    The LEGO System design rule is pitch sqrt(2) - stud diameter = tube diameter, which makes
    the nominal interference exactly zero, and `spec.py`'s tube_od of 6.51 mm lands just the
    wrong side of it. The clutch of a real System brick comes from mould draft, tolerance and
    elastic recovery after ejection, not from a designed interference.
    """

    geom = mat.press_fit_geometry(LEGO_SYSTEM, 2, 2)
    assert geom.interference * 1e6 == pytest.approx(-1.9, abs=0.2)
    assert not geom.has_geometric_clutch
    design_rule = LEGO_SYSTEM.pitch * math.sqrt(2.0) - LEGO_SYSTEM.stud_d
    assert design_rule * 1e3 == pytest.approx(6.514, abs=0.001)


# ---------------------------------------------------------------------------
# Member stiffnesses (guide section 2.2)
# ---------------------------------------------------------------------------


def test_member_stiffnesses_match_the_guide():
    s = mat.press_fit_stiffness(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert s.k_tube / 1e3 == pytest.approx(130.0, rel=0.03)
    assert s.k_stud / 1e3 == pytest.approx(562.0, rel=0.03)
    assert s.k_series / 1e3 == pytest.approx(106.0, rel=0.03)
    assert s.k_wall_soft / 1e3 == pytest.approx(7.2, rel=0.10)
    # The stiff wall case is a factor of fourteen above the compliant one: a wall has no single
    # stiffness, it has one per load case.
    assert s.k_wall_stiff / s.k_wall_soft == pytest.approx(14.0, rel=0.20)
    assert s.k_series / s.k_wall_soft == pytest.approx(15.0, rel=0.15)


def test_hertzian_flattening_is_negligible():
    """Line contact accounts for under 3 % of the take-up, so the press fit is flexure."""

    half_width, p_max, approach = mat.hertz_line_contact(
        5.0, DUPLO.stud_d / 2.0, DUPLO.tube_od / 2.0, DUPLO.stud_h
    )
    assert half_width * 1e6 == pytest.approx(55.0, abs=3.0)
    assert p_max / 1e6 == pytest.approx(13.0, abs=1.0)
    assert approach * 1e6 == pytest.approx(2.6, abs=0.5)
    interference = mat.press_fit_geometry(DUPLO, 2, 4).interference
    # 2.60 um of approach against 86.3 um of interference: 3.0 %, which is what the guide
    # rounds to "under 3 %". Either way it is small enough that ring and beam formulas are the
    # right tools and a contact-mechanics treatment of the press fit is not needed.
    assert approach / interference < 0.05


def test_seating_contact_pressure_is_under_yield():
    """A correctly seated stud does not mark its tube: 13 MPa at 5 N, 17 MPa at 9 N."""

    for force, expected in ((5.0, 13.0), (9.0, 17.0)):
        _, p_max, _ = mat.hertz_line_contact(
            force, DUPLO.stud_d / 2.0, DUPLO.tube_od / 2.0, DUPLO.stud_h
        )
        assert p_max / 1e6 == pytest.approx(expected, abs=1.5)
        assert p_max < mat.ABS_BRICK_23C.tensile_yield.nominal


# ---------------------------------------------------------------------------
# Clutch force (guide section 2.3) - the headline number
# ---------------------------------------------------------------------------


def test_derived_clutch_is_inside_the_guides_band():
    """The guide's first-principles answer spans 0.3 to 6 N per stud, a factor of twenty."""

    est = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert 0.3 <= est.per_stud <= 6.0
    assert est.per_stud_lo <= est.per_stud <= est.per_stud_hi
    assert est.per_brick == pytest.approx(est.per_stud * 8)


def test_stud_on_tube_path_reproduces_the_guides_table():
    """mu = 0.3, nominal interference: 4.1 N per stud, 33 N per 2x4."""

    est = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4, mu=0.3)
    assert est.per_stud == pytest.approx(4.1, abs=0.2)
    assert est.per_brick == pytest.approx(33.0, abs=1.5)
    assert est.contacts_per_stud == pytest.approx(1.5)


def test_wall_path_reproduces_the_first_pass_estimate_and_is_the_wrong_path():
    """The estimate that put the clutch in the walls lands near the 1.0 N placeholder.

    It is inside the guide's 0.3 to 0.4 N band, it is close to the simulator's default, and it
    is wrong, because the walls are 1.65 mm clear of the studs.
    """

    wall = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4, mu=0.3, path="wall")
    tube = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4, mu=0.3)
    assert 0.25 <= wall.per_stud <= 0.45
    assert tube.per_stud / wall.per_stud == pytest.approx(15.0, rel=0.15)


def test_clutch_scales_linearly_with_friction_and_interference():
    """Both are linear, which is why the friction coefficient is the load-bearing unknown."""

    base = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4, mu=0.2)
    doubled = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4, mu=0.4)
    assert doubled.per_stud == pytest.approx(2.0 * base.per_stud, rel=1e-9)

    delta = mat.press_fit_geometry(DUPLO, 2, 4).interference
    bigger = mat.clutch_force(
        DUPLO, mat.ABS_BRICK_23C, 2, 4, mu=0.2, interference=2.0 * delta
    )
    assert bigger.per_stud == pytest.approx(2.0 * base.per_stud, rel=1e-9)


def test_simulator_default_is_below_the_derived_nominal():
    """SnapParams.f_clutch = 1.0 N is not wrong, but it is several times low.

    This reads the dataclass default; it does not modify snap_api in any way.
    """

    default = SnapParams().f_clutch
    est = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert default == 1.0
    assert default < est.per_stud
    assert est.per_stud / default > 3.0


# ---------------------------------------------------------------------------
# Force against depth (guide section 2.4)
# ---------------------------------------------------------------------------


def test_one_degree_of_draft_is_the_size_of_the_whole_interference():
    """79 um of radius change over a 4.5 mm stud, against an 86 um interference."""

    change = mat.draft_radius_change(DUPLO, 1.0)
    assert change * 1e6 == pytest.approx(78.5, abs=1.0)
    interference = mat.press_fit_geometry(DUPLO, 2, 4).interference
    assert 0.8 < change / interference < 1.1


def test_draft_moves_the_exponent_from_one_toward_two():
    """Zero draft gives the simulator's ramp with exponent 1; 1 degree gives close to 2."""

    assert mat.draft_exponent(DUPLO, 0.0) == pytest.approx(1.0)
    assert mat.draft_exponent(DUPLO, 1.0) == pytest.approx(1.91, abs=0.05)
    assert 1.0 < mat.draft_exponent(DUPLO, 0.5) < mat.draft_exponent(DUPLO, 1.0)
    lo, nominal, hi = mat.randomization_ranges()["exponent"]
    assert lo <= nominal <= hi


def test_force_vs_depth_is_monotone_and_ends_at_the_push_on_force():
    import numpy as np

    depth = np.linspace(0.0, DUPLO.stud_h, 60)
    force = mat.force_vs_depth(depth, DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert force[0] == 0.0
    assert np.all(np.diff(force) >= -1e-12)
    est = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert force[-1] == pytest.approx(est.push_on_per_brick, rel=1e-9)


def test_breakaway_travel_is_tens_of_microns_on_the_loaded_path():
    """Stick-slip on a scale far below anything the arm can command or observe."""

    travel = mat.breakaway_travel(DUPLO, mat.ABS_BRICK_23C)
    assert 20e-6 < travel["stud_on_tube_m"] < 100e-6
    assert travel["wall_path_m"] / travel["stud_on_tube_m"] > 10.0


# ---------------------------------------------------------------------------
# Temperature (guide section 2.7)
# ---------------------------------------------------------------------------


def test_thermal_expansion_barely_moves_the_interference():
    """The interference is a difference of same-material dimensions, so it scales with them.

    A 10 K change alters an 86 um interference by 0.09 %. Thermal expansion would matter only
    if the two parts were at different temperatures or made of different plastics, and in this
    task they are neither.
    """

    alpha = mat.ABS_BRICK_23C.thermal_expansion.nominal
    geom = mat.press_fit_geometry(DUPLO, 2, 4)
    for delta_t in (-5.0, 5.0, 10.0):
        scaled = geom.interference * (1.0 + alpha * delta_t)
        relative = abs(scaled - geom.interference) / geom.interference
        assert relative < 0.002


def test_modulus_over_a_room_temperature_range_is_a_few_percent():
    """18 to 28 C changes the modulus by about +-3.5 %, and the clutch with it."""

    material = mat.ABS_BRICK_23C
    cold = material.modulus_at(18.0) / material.E
    hot = material.modulus_at(28.0) / material.E
    assert cold == pytest.approx(1.035, abs=0.005)
    assert hot == pytest.approx(0.965, abs=0.005)
    est_cold = mat.clutch_force(
        DUPLO, mat.replace_modulus(material, material.modulus_at(18.0)), 2, 4
    )
    est_hot = mat.clutch_force(
        DUPLO, mat.replace_modulus(material, material.modulus_at(28.0)), 2, 4
    )
    assert est_cold.per_stud / est_hot.per_stud == pytest.approx(1.035 / 0.965, rel=0.01)


def test_room_temperature_matters_to_the_arm_not_to_the_bricks():
    """PLA's heat deflection temperature is below the servo's own thermal cutout.

    A motor bracket clamped around a working servo can sit above the temperature at which its
    own material creeps, while the bricks two centimetres away are unaffected at 23 C.
    """

    assert mat.PLA_PRINTED.heat_deflection.nominal < mat.STS3215_GEAR_TRAIN.thermal_cutout.nominal
    assert mat.PLA_PRINTED.glass_transition.nominal < mat.ABS_BRICK_23C.glass_transition.nominal
    assert mat.ABS_BRICK_23C.glass_transition.nominal > 100.0


def test_printed_pla_is_an_order_of_magnitude_weaker_through_the_layers():
    """The interlayer bond is what fails on a printed part, and it is about a tenth as strong."""

    in_plane = mat.PLA_PRINTED.tensile_yield.nominal
    through = mat.PLA_PRINTED.tensile_strength_through.nominal
    assert through / in_plane < 0.15


# ---------------------------------------------------------------------------
# Failure modes (guide section 4)
# ---------------------------------------------------------------------------


def test_wall_bending_thresholds():
    """Permanent set at 12 to 35 N, crack at 19 to 56 N, and the set arrives first."""

    walls = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)
    set_lo, set_hi = walls["set_band_N"]
    crack_lo, crack_hi = walls["crack_band_N"]
    assert set_lo == pytest.approx(11.7, abs=0.5)
    assert set_hi == pytest.approx(35.1, abs=1.0)
    assert crack_lo == pytest.approx(18.6, abs=0.5)
    assert crack_hi == pytest.approx(55.9, abs=1.5)
    assert set_lo < crack_lo and set_hi < crack_hi


def test_wall_bending_stress_and_force_are_inverses():
    force = 20.0
    lever, b_eff, t = 0.015, 0.020, DUPLO.wall
    sigma = mat.wall_bending_stress(force, lever, b_eff, t)
    assert mat.wall_bending_force(sigma, lever, b_eff, t) == pytest.approx(force)
    for k_t in (1.0, 1.3, 2.0):
        sigma = mat.wall_bending_stress(force, lever, b_eff, t, k_t=k_t)
        assert mat.wall_bending_force(sigma, lever, b_eff, t, k_t=k_t) == pytest.approx(force)


def test_the_fillet_stress_concentration_only_ever_weakens_the_wall():
    """K_t is disclosed, defaults to 1.0, and its error is one-sided.

    The beam formula gives the nominal stress at the wall root; a moulded internal corner
    concentrates it by 1.3 to 2.0 for the fillet radii a 1.55 mm wall would carry. Nobody has
    measured it on a brick, so the default stays at nominal - but a real wall can only be
    weaker than this model says, never stronger, and at the pessimistic end the permanent-set
    floor drops inside what the arm can apply.
    """

    nominal = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)
    assert nominal["k_t"] == 1.0                       # the default is the nominal stress
    assert mat.WALL_KT_RANGE == (1.0, 2.0)
    lo, mid, hi = mat.randomization_ranges()["wall_kt"]
    assert (lo, mid, hi) == (1.0, 1.0, 2.0), "the range must be one-sided upward"

    for k_t, expected_set, expected_crack in (
        (1.3, (9.0, 27.0), (14.3, 43.0)),
        (2.0, (5.9, 17.6), (9.3, 27.9)),
    ):
        band = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C, k_t=k_t)
        assert band["set_band_N"] == pytest.approx(expected_set, abs=0.15)
        assert band["crack_band_N"] == pytest.approx(expected_crack, abs=0.15)
        # Every threshold divides by K_t exactly: it is a stress multiplier, nothing else.
        for key in ("set_band_N", "crack_band_N"):
            assert band[key][0] == pytest.approx(nominal[key][0] / k_t)

    # The consequence worth naming: at K_t = 2 the permanent-set floor is inside the arm's
    # reach (10 to 14 N of press, 17 N of grip) rather than just outside it.
    pessimistic_set = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C, k_t=2.0)["set_band_N"][0]
    assert pessimistic_set < min(mat.ARM_PRESS_FORCE_N)
    assert nominal["set_band_N"][0] > min(mat.ARM_PRESS_FORCE_N)


def test_thin_walls_buckle_before_they_yield():
    """The correction that turns a 19 kN crush estimate into 13.2 kN."""

    sigma_cr = mat.plate_buckling_stress(
        mat.ABS_BRICK_23C.E, mat.ABS_BRICK_23C.nu, DUPLO.wall, 0.0177, 0.0638
    )
    assert sigma_cr / 1e6 == pytest.approx(19.2, abs=1.0)
    assert sigma_cr < mat.ABS_BRICK_23C.tensile_yield.nominal


def test_crush_is_three_orders_of_magnitude_beyond_the_arm():
    crush = mat.crush_load(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert 11e3 < crush["total_N"] < 15e3
    assert crush["walls"]["wall_left_right"]["mode"] == "buckling"
    assert crush["tube_mode"] == "yield"
    assert crush["total_N"] / max(mat.ARM_PRESS_FORCE_N) > 800.0


def test_the_crush_load_is_13_2_kilonewtons_per_wall_pair():
    """The number the guide quotes, and the derivation behind it, term by term.

    Each pair of walls buckles at its own critical stress because buckling depends on the
    wall's length, and the four walls of a 2x4 are not the same length. Taking the softest
    wall's 19.2 MPa for all of them - which an earlier draft of the guide did - gives 12.2 kN
    and is 8 % low. This checks the per-pair terms, not just the total, so the two derivations
    cannot be confused again.
    """

    crush = mat.crush_load(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    front = crush["walls"]["wall_front_back"]
    side = crush["walls"]["wall_left_right"]
    assert front["sigma_used_Pa"] / 1e6 == pytest.approx(28.4, abs=0.3)
    assert side["sigma_used_Pa"] / 1e6 == pytest.approx(19.5, abs=0.3)
    assert front["sigma_used_Pa"] > side["sigma_used_Pa"], "the shorter pair must be stiffer"
    assert front["force_N"] / 1e3 == pytest.approx(2.80, abs=0.05)
    assert side["force_N"] / 1e3 == pytest.approx(3.66, abs=0.05)
    assert crush["tube_force_N"] / 1e3 == pytest.approx(6.75, abs=0.05)
    assert crush["total_N"] / 1e3 == pytest.approx(13.2, abs=0.05)
    assert crush["total_area_m2"] * 1e6 == pytest.approx(440.1, abs=0.5)

    # And the same derivation on a 2x2, which the guide also quotes.
    small = mat.crush_load(DUPLO, mat.ABS_BRICK_23C, 2, 2)
    assert small["total_N"] / 1e3 == pytest.approx(7.85, abs=0.05)

    # The superseded simplification, reproduced here so the 8 % is visible rather than folded
    # away: one stress for every wall gives 12.2 kN.
    sections = mat.sections(DUPLO, 2, 4)
    softest = mat.plate_buckling_stress(
        mat.ABS_BRICK_23C.E, mat.ABS_BRICK_23C.nu, DUPLO.wall,
        DUPLO.height - DUPLO.roof, 0.0638,
    )
    uniform = crush["wall_area_m2"] * softest + crush["tube_force_N"]
    assert uniform / 1e3 == pytest.approx(12.2, abs=0.15)
    assert crush["total_N"] > uniform
    assert sections["wall_left_right"].length * 1e3 == pytest.approx(60.7, abs=0.1)


def test_crush_cross_checks_against_the_open_university_measurement():
    """A System 2x2 measured 4240 N with a slow plastic collapse; its walls should yield.

    The model gives 82.9 mm^2 of section and a governing stress at the ABS yield, so the
    predicted load is within about 15 % of the measurement, and it agrees on the mechanism as
    well as on the number.
    """

    crush = mat.crush_load(LEGO_SYSTEM, mat.ABS_BRICK_23C, 2, 2)
    assert crush["total_area_m2"] * 1e6 == pytest.approx(82.9, abs=2.0)
    assert crush["total_N"] == pytest.approx(4240.0, rel=0.25)
    assert crush["walls"]["wall_front_back"]["mode"] == "yield"


def test_stud_shear_is_about_a_kilonewton():
    shear = mat.stud_shear_load(DUPLO, mat.ABS_BRICK_23C)
    assert shear["area_m2"] * 1e6 == pytest.approx(36.2, abs=0.5)
    assert shear["shear_N"] == pytest.approx(942.0, abs=20.0)
    assert shear["shear_N"] / max(mat.ARM_PRESS_FORCE_N) > 60.0


def test_wedge_ratio_matches_the_guides_examples():
    assert mat.wedge_ratio(45.0, 0.3) == pytest.approx(0.54, abs=0.01)
    assert mat.wedge_ratio(60.0, 0.3) == pytest.approx(0.94, abs=0.01)
    assert mat.wedge_ratio(30.0, 0.3) == pytest.approx(0.24, abs=0.01)
    # A 14 N press on a badly misaligned stud delivers 3 to 13 N sideways.
    lateral = [14.0 * mat.wedge_ratio(theta, 0.3) for theta in (30.0, 60.0)]
    assert 3.0 <= min(lateral) and max(lateral) <= 14.0


def test_impact_is_the_one_route_to_breakage():
    """A held brick at 0.3 m/s reaches 19 N; a free brick at 1 m/s absorbs 10 N."""

    held_slow = mat.impact_force(0.5, 8000.0, 0.3)
    held_fast = mat.impact_force(0.5, 8000.0, 1.0)
    free = mat.impact_force(0.0125, 8000.0, 1.0)
    assert held_slow == pytest.approx(19.0, abs=1.0)
    assert held_fast == pytest.approx(63.0, abs=2.0)
    assert free == pytest.approx(10.0, abs=1.0)
    crack_lo, crack_hi = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)["crack_band_N"]
    assert held_slow >= crack_lo          # 0.3 m/s already reaches the crack band
    assert free < crack_lo                # a free brick is just knocked away
    assert min(mat.ARM_EFFECTIVE_MASS_KG) < 0.5 < max(mat.ARM_EFFECTIVE_MASS_KG) + 0.1


def test_policy_limits_keep_a_factor_of_two_against_the_thresholds():
    walls = mat.wall_thresholds(DUPLO, mat.ABS_BRICK_23C)
    assert mat.POLICY_LIMITS["lateral_wall_force_N"][0] <= walls["set_band_N"][0] / 2.0 + 0.5
    assert mat.POLICY_LIMITS["press_force_N"][0] <= max(mat.ARM_PRESS_FORCE_N) / 2.0


# ---------------------------------------------------------------------------
# Weld limits for the breakable brick (guide section 6.2)
# ---------------------------------------------------------------------------


def test_weld_limits_match_the_guides_table():
    limits = mat.weld_limits(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert limits["tube"].force_N == pytest.approx(2249.0, rel=0.02)
    assert limits["tube"].moment_Nm == pytest.approx(9.81, rel=0.02)
    assert limits["stud"].force_N == pytest.approx(942.0, rel=0.02)
    assert limits["stud"].moment_Nm == pytest.approx(4.40, rel=0.02)
    # The guide quotes 4351 N for a long wall using the full 63.8 mm footprint edge; the
    # generator shortens the left and right walls at the corners, so the tension limit derived
    # from the geometry that is actually built is a little lower.
    assert limits["wall_left_right"].force_tension_N == pytest.approx(4140.0, rel=0.05)
    assert limits["wall_left_right"].moment_Nm == pytest.approx(1.70, rel=0.05)
    # In compression a wall buckles first, so the governing limit is well under the tension one.
    assert limits["wall_left_right"].force_N < limits["wall_left_right"].force_tension_N


def test_total_weld_capacity_is_the_same_order_as_the_crush_load():
    limits = mat.weld_limits(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    total = (
        2 * limits["wall_front_back"].force_N
        + 2 * limits["wall_left_right"].force_N
        + 3 * limits["tube"].force_N
    )
    crush = mat.crush_load(DUPLO, mat.ABS_BRICK_23C, 2, 4)["total_N"]
    assert total == pytest.approx(crush, rel=0.20)


# ---------------------------------------------------------------------------
# The database itself
# ---------------------------------------------------------------------------


def test_density_reproduces_the_catalogue_mass():
    """The generated 2x4 masses 12.5 g against 12.2 g catalogue, within 3 %."""

    world = mujoco.MjSpec()
    add_brick(world, "b", 2, 4, "red", (0.0, 0.0, 0.0))
    model = world.compile()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    assert model.body_mass[body] * 1000.0 == pytest.approx(12.2, rel=0.04)
    assert mat.ABS_BRICK_23C.density.nominal == pytest.approx(DUPLO.density)


def test_every_property_has_a_source_string_and_a_usable_range():
    for material in (mat.ABS_BRICK_23C, mat.PLA_PRINTED):
        for name, value in vars(material).items():
            if not isinstance(value, mat.Prop):
                continue
            assert value.source, f"{material.name}.{name} has no source"
            assert value.unit, f"{material.name}.{name} has no unit"
            lo, hi = value.range
            assert lo <= value.nominal <= hi, f"{material.name}.{name} range excludes nominal"


def test_the_friction_coefficients_are_flagged_as_unsourced():
    """The most load-bearing unknown in the model, and it must say so out loud.

    The Engineering ToolBox friction table does not list ABS at all, and no ABS-on-ABS or
    ABS-on-PLA measurement was found anywhere. Every clutch force scales linearly with it.
    """

    flagged = {name for name, _, _ in mat.unsourced_properties()}
    assert "friction pair abs_on_abs" in flagged
    assert "friction pair pla_on_abs" in flagged
    assert mat.FRICTION_PAIRS["abs_on_abs"].mu.confidence == "unsourced"


def test_contact_settings_respect_the_time_step_floor():
    """The solref time constant must be at least twice the step; 2 ms forces a 4 ms floor.

    The floor is now the solver's own ``2 * dt`` rather than a hard 4 ms, because the applied
    value is the derived stiffness clipped at that floor: at a 0.5 ms step the solver can run
    1 ms, which is four times stiffer and four times closer to the material. The project's own
    contact model is fixed at :data:`brick_builder.bricks.generator.CONTACT_TIMESTEP` (2 ms), so
    what a scene is built with does not move when a run is stepped finer.
    """

    for dt in (0.0005, 0.002, 0.004):
        for setting in mat.contact_settings(dt).values():
            assert setting.solref[0] >= 2.0 * dt - 1e-12
            assert setting.solref[0] == pytest.approx(2.0 * dt)
    for setting in mat.contact_settings(0.002).values():
        assert setting.solref[0] == pytest.approx(0.004)
    pads = mat.contact_settings(0.002)["pad_on_brick"]
    assert pads.condim == 4 and pads.priority == 1


def test_every_contact_pair_is_derived_from_the_material_and_then_clipped():
    """Each pair gets a stiffness from a contact modulus and a stated patch, and every one of
    them is far past what the solver runs at a 2 ms step.

    The clip is the headline number of the whole contact model: a flat ABS stud tip is 11.8 MN/m
    and the stiffest setting the solver is stable at is four orders of magnitude softer, so the
    simulated contact is a statement about the solver, not about the plastic.
    """

    settings = mat.contact_settings(0.002)
    assert set(settings) == {
        "brick_on_brick", "pad_on_brick", "brick_on_table", "arm_on_anything"
    }
    for name, s in settings.items():
        assert s.stiffness_N_m > 1e6, name
        assert s.contact_geometry, name
        assert s.solref_derived[0] < 1e-4, name          # tens of microseconds
        assert s.clip_ratio > 1e4, name
        # The clip ratio is exactly the square of the time-constant ratio, which is the whole
        # arithmetic of the statement "this is N times softer than the material".
        assert s.clip_ratio == pytest.approx((s.solref[0] / s.solref_derived[0]) ** 2)

    # Brick on brick is the pair every seated stud uses; its numbers are quoted in the guide.
    bb = settings["brick_on_brick"]
    assert bb.stiffness_N_m == pytest.approx(11.79e6, rel=0.02)
    assert bb.solref_derived[0] == pytest.approx(32.6e-6, rel=0.02)
    assert bb.clip_ratio == pytest.approx(15073.0, rel=0.02)

    # A rigid table is stiffer than ABS on ABS, so it clips harder, not less.
    assert settings["brick_on_table"].clip_ratio > bb.clip_ratio


def test_restitution_is_recorded_as_a_gap_not_as_a_setting():
    """MuJoCo has no restitution parameter, and the simulator under-bounces at about 0.2."""

    e = mat.ABS_BRICK_23C.restitution
    assert e.confidence == "unsourced"
    assert e.range == (0.40, 0.70)
    assert e.nominal > 0.2      # the simulator's own value, which this says is too low


def test_randomization_ranges_bracket_their_nominals():
    for name, (lo, nominal, hi) in mat.randomization_ranges().items():
        assert lo <= nominal <= hi, name
        assert lo < hi, name


def test_sampling_is_reproducible_and_stays_inside_the_ranges():
    import numpy as np

    a = mat.ABS_BRICK_23C.sample(np.random.default_rng(7))
    b = mat.ABS_BRICK_23C.sample(np.random.default_rng(7))
    assert a.E == b.E and a.tensile_yield.nominal == b.tensile_yield.nominal
    lo, hi = mat.ABS_BRICK_23C.youngs_modulus.range
    assert lo <= a.E <= hi


def test_derived_snap_params_do_not_touch_the_defaults():
    """Everything here is opt-in: asking for derived parameters changes nothing."""

    before = SnapParams()
    derived = mat.derived_snap_params(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert SnapParams().f_clutch == before.f_clutch == 1.0
    assert SnapParams().exponent == before.exponent == 1.0
    assert derived.f_clutch > before.f_clutch
    assert derived.exponent > before.exponent
    assert derived.curve == "ramp"


# ---------------------------------------------------------------------------
# Can the stock arm seat a brick? (the section of that name)
# ---------------------------------------------------------------------------


def test_push_on_force_scales_with_studs_and_with_the_direction_of_travel():
    """Push-on = clutch force x studs x (1+beta), and a tilted press only fights row_studs."""

    beta = 0.15
    ratio = 1.0 + beta
    assert mat.push_on_force(1.0, 8) == pytest.approx(8.0 * ratio)
    assert mat.push_on_force(1.0, 4) == pytest.approx(4.0 * ratio)
    # A tilted press engages one row of two, whatever the brick, so both bricks land on the
    # same line.
    assert mat.push_on_force(1.0, 8, tilted=True) == pytest.approx(2.0 * ratio)
    assert mat.push_on_force(1.0, 4, tilted=True) == pytest.approx(2.0 * ratio)
    assert mat.push_on_force(1.0, 8) / mat.push_on_force(1.0, 8, tilted=True) == pytest.approx(
        4.0
    )
    # The detent's entry bump is a peak above the seated force, so it multiplies every one.
    gain = mat.detent_peak_gain(DUPLO)
    assert 1.3 < gain < 1.6
    assert mat.push_on_force(1.0, 8, shape="detent") == pytest.approx(
        gain * mat.push_on_force(1.0, 8)
    )
    assert mat.push_on_force(1.0, 8, shape="plateau") == mat.push_on_force(1.0, 8)


def test_push_on_force_matches_pressbench_for_a_straight_press():
    """Pins the fix directly: push_on_force(clutch, n) must agree with what the real simulator
    mechanism (ColletSnap/engagement, run through PressBench) actually measures, for 2x2 and
    2x4. This is the agreement the module docstring and seating_feasibility.py both claim.

    PressBench stops the instant every socket is within `seat_tolerance` (0.3 mm) of full
    depth, not at exactly full depth, so with SnapParams' default exponent-1 ramp the measured
    peak sits at about 93% of the asymptotic `push_on_force` value (matching
    quasi_static_band's own "stops at 93% of full depth" note in bench.py) rather than dead
    equal -- 9% comfortably covers that gap and is nowhere near the old (1-beta)-inflated
    figure (about 18% high).
    """
    from brick_builder.bricks.bench import BenchConfig, PressBench
    from brick_builder.bricks.snap import ColletSnap

    clutch = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4).per_stud
    params = SnapParams(f_clutch=clutch, beta=0.15, curve="ramp")
    for nx, ny in ((2, 2), (2, 4)):
        bench = PressBench(ColletSnap(params), BenchConfig(nx=nx, ny=ny))
        trace = bench.press()
        assert bench.seated()
        predicted = mat.push_on_force(clutch, nx * ny, beta=params.beta, shape="ramp")
        assert trace.peak_force == pytest.approx(predicted, rel=0.09), (nx, ny)
        # And the old, buggy convention (dividing by (1-beta) too) is well outside that margin.
        wrong = predicted / (1.0 - params.beta)
        assert trace.peak_force != pytest.approx(wrong, rel=0.09), (nx, ny)


def test_tilted_press_needs_fewer_studs_worth_of_force_by_the_simplified_model():
    """push_on_force's ``tilted`` reduction is a per-row simplification, not a full accounting
    of what a tilted press's peak does over an entire run.

    brick_builder.bricks.bench.measure_tilted_press_ratio (tests/bricks/test_snap_bench.py) shows that,
    for the simulator's default "ramp" curve, a tilted press's OVERALL peak stays close to a
    flat press's (ratio 0.9 to 1.1 there): every stud still has to reach full depth by the end
    of the roll, and a rigid brick's ram reads the sum of every stud's force whatever the tool
    is doing. push_on_force(tilted=True) instead models only the force to get row_studs moving,
    which is genuinely lower and is what the guide's "tilted ceiling" numbers report -- this
    test just pins that the simplification is internally consistent (fewer studs, same per-stud
    law) without claiming it is the run's overall PressBench peak, which it is not.
    """
    clutch = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4).per_stud
    straight = mat.push_on_force(clutch, 8, shape="ramp")
    tilted = mat.push_on_force(clutch, 8, shape="ramp", tilted=True)
    assert tilted == pytest.approx(straight / 4.0)


def test_the_derived_clutch_puts_a_flat_2x4_press_out_of_the_arms_reach():
    """The project's central feasibility question, as an assertion.

    At the first-principles 4.79 N per stud a 2x4 needs 44 N to press on flat (n_studs x
    clutch x (1+beta): test_push_on_force_matches_pressbench_for_a_straight_press pins this
    against the real simulator mechanism). The arm's whole force map - both torque assumptions,
    all three supply voltages, best cell of the build plate - tops out at about 37 N, so no
    straight press of a 2x4 works at that clutch. A tilted press's per-row estimate is 11 N,
    which is inside the arm's range at 7.4 V (see the caveat on the tilted estimate above).
    """

    est = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    arm = mat.arm_press_force_band()
    best = max(hi for law in ("datasheet", "identified") for _, hi in arm[law].values())

    flat = mat.push_on_force(est.per_stud, 8)
    tilted = mat.push_on_force(est.per_stud, 8, tilted=True)
    assert flat == pytest.approx(44.07, abs=1.0)
    assert tilted == pytest.approx(11.02, abs=0.5)
    assert flat > best, "a flat 2x4 press is supposed to be out of reach at the derived clutch"
    assert tilted < max(arm["datasheet"][7.4]), "a tilted press is supposed to be within reach"
    # The 2x2 is out of reach flat at 7.4 V under the datasheet torque, and clears (at least
    # partly) once the identified law is used, even without waiting for 12 V.
    flat_2x2 = mat.push_on_force(est.per_stud, 4)
    assert flat_2x2 > max(arm["datasheet"][7.4])
    assert flat_2x2 < max(arm["identified"][12.0])


def test_the_force_map_band_is_read_not_invented():
    """The arm's forces come from brick_builder.robot.force_map, and nothing is written back.

    Voltage scales the torque budget linearly and the identified law is 1.8x the datasheet, so
    both orderings have to hold everywhere; the guide's own 10 to 14 N at 7.4 V has to fall out.
    """

    from brick_builder.robot.force_map import _TORQUE_LIMIT_NM

    arm = mat.arm_press_force_band()
    assert arm["torque_limit_Nm"]["datasheet"] == pytest.approx(1.9)
    assert arm["torque_limit_Nm"]["identified"] == pytest.approx(3.41, abs=0.01)
    assert _TORQUE_LIMIT_NM == 1.9, "reading force_map must not have changed its defaults"

    for law in ("datasheet", "identified"):
        lo5, hi5 = arm[law][5.0]
        lo74, hi74 = arm[law][7.4]
        lo12, hi12 = arm[law][12.0]
        assert lo5 < lo74 < lo12 and hi5 < hi74 < hi12
    for vin in (5.0, 7.4, 12.0):
        assert arm["identified"][vin][1] > arm["datasheet"][vin][1]
    # The number the rest of the guide quotes for the design's own supply.
    assert arm["datasheet"][7.4] == pytest.approx(mat.ARM_PRESS_FORCE_N, abs=1.0)


def test_seating_feasibility_says_where_each_supply_runs_out():
    result = mat.seating_feasibility()
    ceilings = result["hold_max_N"]["datasheet_7.4V"]
    assert ceilings["2x4_ramp"][1] == pytest.approx(1.52, abs=0.1)
    assert ceilings["2x4_tilted_ramp"][1] == pytest.approx(6.10, abs=0.2)
    assert ceilings["2x2_ramp"][1] == pytest.approx(3.05, abs=0.2)
    # A detent curve lowers every ceiling, because the bump is what has to be beaten.
    assert ceilings["2x4_tilted_detent"][1] < ceilings["2x4_tilted_ramp"][1]
    # More voltage buys more clutch, in the same order as the force map.
    assert (result["hold_max_N"]["datasheet_12V"]["2x4_ramp"][1]
            > ceilings["2x4_ramp"][1]
            > result["hold_max_N"]["datasheet_5V"]["2x4_ramp"][1])
    # And the derived clutch sits above the straight-press ceiling at every supply: that is
    # the finding, not an accident of the numbers.
    derived = mat.clutch_force(DUPLO, mat.ABS_BRICK_23C, 2, 4).per_stud
    assert all(v["2x4_ramp"][1] < derived for v in result["hold_max_N"].values())


def test_comparison_table_reads_the_simulator_and_renders():
    text = mat.comparison_table(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert "f_clutch per stud" in text
    assert "1 N" in text                      # the simulator's default, read from SnapParams
    assert "radial interference" in text
    assert SnapParams().f_clutch == 1.0       # ... and still not modified
    rows = mat.comparison_rows(DUPLO, mat.ABS_BRICK_23C, 2, 4)
    assert all(len(r) == 4 for r in rows)
