"""Material properties at room temperature, and the simulator parameters derived from them.

This is the code form of `docs/guide/02c-materials-and-damage.md`. That page is the
specification; this module is the arithmetic, so that every number the simulator uses can be
traced back to a material property and a piece of geometry instead of being a placeholder.

Three things live here:

1. **A small sourced database.** :data:`ABS_BRICK_23C` (what a DUPLO brick is moulded from),
   :data:`PLA_PRINTED` (what the SO-101's structural parts are printed from) and
   :data:`STS3215_GEAR_TRAIN` (the servo's own mechanics). Every property carries its nominal
   value, the range it should be randomized over, its units, its conditions and the source
   string the guide page cites. Where no source exists the property says so in its
   ``source`` and its ``confidence`` is ``"assumed"`` or ``"unsourced"`` - that is not a
   formality, it is the single most important fact about the friction coefficient.

2. **Derivations.** Functions that turn a :class:`~brick_builder.bricks.spec.BrickSpec` plus a
   :class:`Material` into the numbers the simulator needs: the press-fit interference, the
   member stiffnesses that take it up, the holding and push-on force per stud, the
   force-against-depth curve, per-pair contact settings as MuJoCo ``solref``/``solimp``, and
   every failure threshold in the guide's section 4.

3. **A comparison table.** :func:`comparison_table` reads the simulator's *current* defaults
   out of :mod:`brick_builder.bricks.snap_api` and :mod:`brick_builder.bricks.spec` (read only - nothing
   here writes to either) and prints them beside what the material model derives, so the gap
   between the two is visible rather than implied.

Nothing in this module changes any default. It is opt-in: a caller asks for a derived
:class:`~brick_builder.bricks.snap_api.SnapParams` and decides for itself whether to use it.

Units are SI throughout: metres, kilograms, seconds, newtons, pascals, radians, kelvin. Every
public function's docstring states the unit of what it returns. Temperatures are in degrees
Celsius because that is how every material datasheet quotes them, and the conversion to kelvin
is only ever a difference.

The honest summary of the uncertainty, from the guide's section 2.3, repeated here because a
caller reading this file should not have to go and find it: **the press fit is uncertain by a
factor of about twenty**, because it depends on the difference between two dimensions near
13 mm, one of which (the DUPLO tube diameter) is a hobbyist caliper measurement rather than a
LEGO drawing, multiplied by a friction coefficient for which no measurement exists anywhere.
Every clutch number this module produces is a band, and the band is wide on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

import numpy as np

from brick_builder.bricks.spec import DUPLO, LEGO_SYSTEM, BrickSpec

Confidence = Literal["measured", "datasheet", "literature", "assumed", "unsourced"]


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prop:
    """One material property: a nominal value, a randomization range, and where it came from.

    - ``nominal`` is what a simulation should use when it is not randomizing.
    - ``lo``/``hi`` bracket the range the guide page recommends randomizing over. They are the
      spread of the published values, not a confidence interval, and for the unsourced
      properties they are a judgement about what is physically plausible.
    - ``confidence`` is deliberately blunt. ``"unsourced"`` means no source was found at all
      and the number is there so the code runs, not because anyone measured it.
    """

    nominal: float
    unit: str
    lo: float | None = None
    hi: float | None = None
    source: str = ""
    confidence: Confidence = "literature"

    @property
    def range(self) -> tuple[float, float]:
        """``(lo, hi)``, falling back to the nominal value when no range is published."""

        return (
            self.nominal if self.lo is None else self.lo,
            self.nominal if self.hi is None else self.hi,
        )

    def sample(self, rng: np.random.Generator) -> float:
        """One value drawn uniformly from the randomization range."""

        lo, hi = self.range
        return float(rng.uniform(lo, hi))

    def __float__(self) -> float:
        return float(self.nominal)


@dataclass(frozen=True)
class FatigueModel:
    """How a connection loses its grip with cycles and with time under load.

    The form is the guide's section 6.1, which is a Palmgren-Miner sum on a power law plus a
    linear creep term:

        fatigue += (s / s_ref) ** exponent / cycles_ref      once per load cycle
        relax   += max(s - relax_threshold, 0) * dt / tau    every step
        clutch   = clip(1 - fatigue_weight*fatigue - relax_weight*relax, floor, 1)

    with ``s`` the worst stress ratio (stress over allowable) the connection saw.

    ``cycles_ref = 37112`` is the *only* cycle-life datum that exists for any LEGO part: a
    hobbyist rig assembled and separated System bricks 37,112 times over about ten days before
    they lost their clutch. There is no DUPLO number, no S-N curve for injection-moulded ABS at
    23 C, and no published relaxation modulus. The exponent of 8 is a guess at the *shape* as
    well as the value. Randomize both over at least a factor of three.
    """

    cycles_ref: float = 37112.0
    s_ref: float = 0.5
    exponent: float = 8.0
    tau_relax_s: float = 3600.0
    relax_threshold: float = 0.2
    fatigue_weight: float = 0.5
    relax_weight: float = 0.3
    clutch_floor: float = 0.2
    source: str = (
        "cycles_ref: HOTH Bricks / Philippe Cantin 'Brickbusters' rig, 37,112 System "
        "assemble-separate cycles over ~10 days. Exponent, tau and weights: assumed shape, "
        "guide 02c section 6.1; no S-N curve for moulded ABS at 23 C was found."
    )


@dataclass(frozen=True)
class Material:
    """A bulk (or, for a printed part, an effective) engineering material at stated conditions.

    ``youngs_modulus_through`` and ``tensile_strength_through`` are only populated for printed
    parts, where the through-layer direction is a different material in every way that matters:
    a printed part loaded so that a layer line is in tension is roughly a tenth as strong as the
    same part loaded along its layers. For an injection-moulded brick they are ``None``.
    """

    name: str
    conditions: str
    density: Prop
    youngs_modulus: Prop
    poisson: Prop
    tensile_yield: Prop
    flexural_strength: Prop
    shear_strength: Prop
    elongation_at_break: Prop
    izod_notched: Prop
    glass_transition: Prop
    heat_deflection: Prop
    thermal_expansion: Prop
    restitution: Prop
    d_modulus_dt: Prop
    fatigue: FatigueModel = field(default_factory=FatigueModel)
    youngs_modulus_through: Prop | None = None
    tensile_strength_through: Prop | None = None
    sources: tuple[str, ...] = ()

    # -- convenience accessors, all returning the nominal value in SI ------

    @property
    def E(self) -> float:
        """Young's modulus [Pa]."""

        return self.youngs_modulus.nominal

    @property
    def nu(self) -> float:
        """Poisson's ratio [-]."""

        return self.poisson.nominal

    @property
    def plane_strain_modulus(self) -> float:
        """``E / (1 - nu**2)`` [Pa], the stiffness of a wide plate strip in bending."""

        return self.E / (1.0 - self.nu**2)

    @property
    def contact_modulus(self) -> float:
        """``E / (2 (1 - nu**2))`` [Pa], the ``E*`` of two identical bodies in Hertz contact."""

        return self.E / (2.0 * (1.0 - self.nu**2))

    def modulus_at(self, temp_c: float) -> float:
        """Young's modulus [Pa] at ``temp_c``, linearised about the material's own conditions.

        Amorphous thermoplastics soften continuously toward their glass transition. No source
        found gives dE/dT for ABS near 23 C, so :attr:`d_modulus_dt` carries the commonly used
        -0.7 %/K for amorphous thermoplastics well below Tg, as an assumption. Over an 18 to
        28 C room that is about +-3.5 %, which is smaller than the tolerance spread on the
        geometry and much smaller than the friction uncertainty.
        """

        return self.E * (1.0 + self.d_modulus_dt.nominal * (temp_c - 23.0))

    def sample(self, rng: np.random.Generator) -> "Material":
        """A copy with every ranged property drawn once from its randomization range."""

        def draw(p: Prop | None) -> Prop | None:
            return None if p is None else replace(p, nominal=p.sample(rng))

        return replace(
            self,
            name=f"{self.name} (sampled)",
            density=draw(self.density),
            youngs_modulus=draw(self.youngs_modulus),
            poisson=draw(self.poisson),
            tensile_yield=draw(self.tensile_yield),
            flexural_strength=draw(self.flexural_strength),
            shear_strength=draw(self.shear_strength),
            izod_notched=draw(self.izod_notched),
            restitution=draw(self.restitution),
            youngs_modulus_through=draw(self.youngs_modulus_through),
            tensile_strength_through=draw(self.tensile_strength_through),
        )


def replace_modulus(material: Material, modulus_Pa: float) -> Material:
    """A copy of ``material`` with its Young's modulus replaced, everything else kept.

    Used to ask what a temperature (or a randomization draw) does to a derived quantity without
    mutating the database entry, which is frozen on purpose.
    """

    return replace(
        material,
        youngs_modulus=replace(material.youngs_modulus, nominal=modulus_Pa),
    )


@dataclass(frozen=True)
class FrictionPair:
    """Sliding friction between two named materials, plus the MuJoCo contact settings to use.

    ``condim`` and ``priority`` are the settings the guide's section 3.2 table recommends for
    this pair: ``condim 4`` with ``priority 1`` on the jaw pads gives the pad torsional friction
    (which is what stops a gripped brick spinning about the pinch axis) and makes the pad's own
    friction win over the brick's.
    """

    a: str
    b: str
    mu: Prop
    condim: int = 3
    priority: int = 0
    torsional: float = 0.005
    rolling: float = 0.0001

    @property
    def friction(self) -> tuple[float, float, float]:
        """The three numbers MuJoCo's ``geom friction`` attribute wants."""

        return (self.mu.nominal, self.torsional, self.rolling)


@dataclass(frozen=True)
class GearTrain:
    """The servo's mechanics. Not a material, but it belongs in the same table.

    The gear train is steel in a glass-filled nylon case and is not a candidate for damage: the
    overload trip is a mechanical fuse that already exists, and the servo gives up long before
    it does anything violent. It is here because the *force the arm can apply* is the other half
    of every damage question, and that comes from this table, not from the plastics.
    """

    name: str
    stall_torque_datasheet: Prop
    stall_torque_identified: Prop
    dry_friction: Prop
    viscous_friction: Prop
    rotor_inertia: Prop
    backlash: Prop
    overload_trip_fraction: Prop
    overload_trip_time: Prop
    thermal_cutout: Prop
    sources: tuple[str, ...] = ()


_ABS_SOURCES = (
    "Grade: unknown. LEGO/DUPLO bricks are moulded in ABS (about 80 % of LEGO elements); "
    "the grade usually named, Novodur P2H-AT, could not be confirmed against a manufacturer "
    "datasheet and secondary sources disagree about the supplier. Treat this as an unfilled "
    "injection-moulding ABS. (BrickNerd; LEGO materials page; guide 02c section 1.1)",
    "Tg is the only property here measured on a real brick: DSC on an unaged LEGO brick, "
    "PMC10422395.",
    "Aged or dark-coloured bricks are a different material: 1390 h under a xenon lamp drops "
    "the Tg from 104 C to 85 C by chain scission. Calibrate on new, light-coloured bricks.",
)

ABS_BRICK_23C = Material(
    name="ABS, injection-moulded brick",
    conditions="23 C, unaged, new light-coloured brick",
    density=Prop(1050.0, "kg/m^3", 1030.0, 1070.0,
                 "Novodur P2H-AT listed at 1040 (plastics-database.org); generic unfilled ABS "
                 "1030 to 1070", "datasheet"),
    youngs_modulus=Prop(2.3e9, "Pa", 1.8e9, 2.7e9,
                        "Novodur P2H-AT 2500 MPa (Albis listing); generic ABS 1.58 to 2.63 GPa "
                        "(ASTM D638), 1.79 to 2.89 GPa (ISO 527-2)", "datasheet"),
    poisson=Prop(0.35, "-", 0.34, 0.38,
                 "aggregated literature; ABS datasheets rarely publish it", "literature"),
    tensile_yield=Prop(44e6, "Pa", 36e6, 52e6,
                       "Novodur P2H-AT 44 MPa; generic 35.1 to 51.0 MPa (ASTM D638)",
                       "datasheet"),
    flexural_strength=Prop(70e6, "Pa", 60e6, 95e6,
                           "generic ABS 72 to 97 MPa (MakeItFrom)", "literature"),
    shear_strength=Prop(26e6, "Pa", 22e6, 31e6,
                        "ASSUMPTION: 0.6 x tensile yield. No measured ABS shear strength was "
                        "found in any source", "assumed"),
    elongation_at_break=Prop(0.20, "-", 0.03, 0.50,
                             "generic ABS 3.4 to 26 % (ISO 527-2), 1.0 to 57 % (ASTM D638)",
                             "literature"),
    izod_notched=Prop(300.0, "J/m", 70.0, 370.0,
                      "Novodur P2H-AT 300 J/m at 23 C; generic range from MakeItFrom",
                      "datasheet"),
    glass_transition=Prop(104.0, "degC", 100.0, 106.0,
                          "DSC on an actual unaged LEGO brick, PMC10422395", "measured"),
    heat_deflection=Prop(94.0, "degC", 75.0, 94.0,
                         "94 C at 0.45 MPa for Novodur P2H-AT (plastics-database.org); "
                         "generic unfilled ABS 75 to 88 C at 1.8 MPa", "datasheet"),
    thermal_expansion=Prop(90e-6, "1/K", 70e-6, 110e-6, "generic ABS", "literature"),
    restitution=Prop(0.55, "-", 0.40, 0.70,
                     "NO SOURCE EXISTS for ABS or for a LEGO brick. 0.4 to 0.7 is what a hard "
                     "plastic on a hard table usually gives. MuJoCo has no restitution "
                     "parameter; this is here to say what the simulator is missing, not to be "
                     "set anywhere. See guide 02c section 3.3", "unsourced"),
    d_modulus_dt=Prop(-0.007, "1/K", -0.012, -0.004,
                      "ASSUMPTION: the commonly used -0.7 %/K for amorphous thermoplastics "
                      "well below Tg. No dE/dT for ABS near 23 C was found", "assumed"),
    sources=_ABS_SOURCES,
)

PLA_PRINTED = Material(
    name="PLA+, FDM printed, 15 % infill, 0.2 mm layers",
    conditions="23 C, as printed, orientation unspecified",
    density=Prop(1240.0, "kg/m^3", 1210.0, 1260.0, "bulk PLA", "literature"),
    youngs_modulus=Prop(3.5e9, "Pa", 2.5e9, 3.9e9,
                        "FDM tensile study, room temperature, layers parallel to the load",
                        "literature"),
    poisson=Prop(0.36, "-", 0.33, 0.39, "aggregated literature", "literature"),
    tensile_yield=Prop(55e6, "Pa", 45e6, 65e6, "FDM tensile study, in-plane", "literature"),
    flexural_strength=Prop(80e6, "Pa", 60e6, 100e6, "generic printed PLA", "literature"),
    shear_strength=Prop(33e6, "Pa", 27e6, 39e6,
                        "ASSUMPTION: 0.6 x in-plane tensile yield", "assumed"),
    elongation_at_break=Prop(0.05, "-", 0.02, 0.10, "generic printed PLA", "literature"),
    izod_notched=Prop(25.0, "J/m", 15.0, 45.0, "generic printed PLA", "literature"),
    glass_transition=Prop(57.0, "degC", 55.0, 60.0, "PLA", "literature"),
    heat_deflection=Prop(55.0, "degC", 45.0, 60.0, "PLA", "literature"),
    thermal_expansion=Prop(70e-6, "1/K", 60e-6, 85e-6, "generic PLA", "literature"),
    restitution=Prop(0.5, "-", 0.3, 0.7, "NO SOURCE. Not used anywhere", "unsourced"),
    d_modulus_dt=Prop(-0.012, "1/K", -0.020, -0.008,
                      "ASSUMPTION. PLA's Tg is only 34 K above room temperature, so it softens "
                      "faster with temperature than ABS does", "assumed"),
    youngs_modulus_through=Prop(2.1e9, "Pa", 1.5e9, 2.5e9,
                                "about 40 % below in-plane (FDM tensile study)", "literature"),
    tensile_strength_through=Prop(5e6, "Pa", 3e6, 15e6,
                                  "interlayer bond, about 91 % below in-plane. THIS is what "
                                  "fails on a printed part", "literature"),
    sources=(
        "Upstream recommends PLA+ at 15 % infill, 0.2 mm layers, no wall count specified "
        "(TheRobotStudio/SO-ARM100 README).",
        "Print orientation is not specified upstream, so which of the two strengths applies to "
        "any given bracket depends on whoever printed it.",
        "Room temperature means something different on the arm: PLA's heat deflection "
        "temperature is 45 to 60 C and the servo cuts out at about 70 C internal, with a bench "
        "test recording a 15 C rise after ten minutes under load. A bracket clamped around a "
        "working servo can sit above the temperature at which its own material creeps, while "
        "the bricks two centimetres away are unaffected at 23 C.",
        "The one failure actually reported on this kit is the gripper servo horn screw "
        "stripping the threads cut into the printed boss (Waveshare assembly wiki), not the "
        "servo's steel gears - which is the through-layer number failing, not the in-plane one.",
    ),
)

STS3215_GEAR_TRAIN = GearTrain(
    name="Feetech STS3215, steel gears in a glass-filled nylon case",
    stall_torque_datasheet=Prop(1.9, "N*m", 1.7, 2.1,
                                "Feetech datasheet at 7.4 V; matches ArmConfig.torque_limit_nm",
                                "datasheet"),
    stall_torque_identified=Prop(3.41, "N*m", 3.0, 3.8,
                                 "BAM identified servo law at full duty, 7.4 V "
                                 "(patch_so101.servo_stall_torque)", "measured"),
    dry_friction=Prop(0.0515, "N*m", 0.04, 0.07,
                      "BAM identification, BAM_FRICTION_BASE", "measured"),
    viscous_friction=Prop(0.0599, "N*m*s/rad", 0.045, 0.075,
                          "BAM identification, BAM_FRICTION_VISCOUS", "measured"),
    rotor_inertia=Prop(0.0261, "kg*m^2", 0.020, 0.032,
                       "apparent rotor inertia at the output, BAM_ARMATURE", "measured"),
    backlash=Prop(0.87, "deg", 0.5, 1.2,
                  "0.5 deg specified (Feetech), 0.87 deg measured (Robonine bench test)",
                  "measured"),
    overload_trip_fraction=Prop(0.80, "-", 0.7, 0.9,
                                "Feetech datasheet, translated", "datasheet"),
    overload_trip_time=Prop(0.025, "s", 0.015, 0.040,
                            "Feetech datasheet, translated", "datasheet"),
    thermal_cutout=Prop(70.0, "degC", 65.0, 75.0,
                        "Feetech datasheet, translated", "datasheet"),
    sources=(
        "The overload trip is a mechanical fuse that already exists: the servo gives up before "
        "it does anything violent, which is one more reason the arm is not a brick-breaking "
        "machine.",
    ),
)


#: Friction pairs, per the guide's section 3.2 table. Not one of these has a measurement behind
#: it: the Engineering ToolBox friction table does not list ABS at all, and no ABS-on-ABS or
#: ABS-on-PLA figure was found anywhere. Every clutch force in this module scales linearly with
#: ``abs_on_abs``, which makes it the single most load-bearing unsourced number in the model.
FRICTION_PAIRS: dict[str, FrictionPair] = {
    "abs_on_abs": FrictionPair(
        "ABS brick", "ABS brick",
        Prop(0.35, "-", 0.20, 0.50,
             "NO SOURCE EXISTS. 0.35 is the value already in BenchConfig.friction; the 0.3 "
             "carried through this project's design work traces to nothing. Guide 02c section "
             "7.3 measures it with a tilt test in ten minutes", "unsourced"),
        condim=3,
    ),
    "pla_on_abs": FrictionPair(
        "PLA jaw face", "ABS brick",
        Prop(0.40, "-", 0.30, 1.00,
             "NO SOURCE. ArmConfig.pad_friction is currently 1.0, which is a modelling choice "
             "to make the grasp hold, not a measurement; PLA is reported to have lower friction "
             "than ABS, so the physical value is more likely 0.3 to 0.5. The upper end of the "
             "range is kept at 1.0 so a randomized episode can still reproduce today's "
             "behaviour", "unsourced"),
        condim=4, priority=1,
    ),
    "abs_on_table": FrictionPair(
        "ABS brick", "table",
        Prop(0.40, "-", 0.20, 0.80,
             "depends entirely on the table; no measurement", "unsourced"),
        condim=3,
    ),
}


# ---------------------------------------------------------------------------
# Geometry of the press fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PressFitGeometry:
    """Where the interference is, how big it is, and how many places carry it.

    All lengths metres. ``interference`` is on the *radius* and is positive when the stud and
    the tube overlap, which is the only case that produces a clutch.
    """

    half_diagonal: float
    radius_sum: float
    interference: float
    wall_gap: float
    n_contacts: int
    n_studs: int
    contacts_per_stud: float

    @property
    def has_geometric_clutch(self) -> bool:
        return self.interference > 0.0


def press_fit_geometry(spec: BrickSpec = DUPLO, nx: int = 2, ny: int = 4) -> PressFitGeometry:
    """Work the press-fit geometry out of a :class:`BrickSpec`, rather than quoting it.

    A stud from the brick below rises into the square cell bounded by four tube positions. Its
    centre sits a half-diagonal ``pitch * sqrt(2) / 2`` from each neighbouring tube centre, and
    the two radii that have to fit into that distance are the stud's and the tube's. The
    difference is the interference on the radius.

    For DUPLO that is 11.314 mm against 11.400 mm, an overlap of **86 um**. For
    :data:`~brick_builder.bricks.spec.LEGO_SYSTEM` as it is currently configured it comes out
    *negative* (a 1.9 um clearance), which means that variant has no geometric clutch at all -
    see :func:`comparison_table`.

    ``wall_gap`` is the clear distance from a stud's outer surface to the inner face of the
    nearest wall. It is 1.65 mm on a DUPLO 2x4, which is why the walls are **not** in contact:
    a real DUPLO brick bridges that gap with internal ribs, and the generator models none. Any
    analysis that puts the clutch in the walls is describing a brick this project does not have.

    ``n_contacts`` counts stud-on-tube contacts for an ``nx`` by ``ny`` brick pressed onto an
    identical one: each interior stud is a half-diagonal from two tube positions and each
    perimeter stud from one, giving 12 on a 2x4, an average of 1.5 per stud.
    """

    d = spec.pitch * math.sqrt(2.0) / 2.0
    r_sum = spec.stud_d / 2.0 + spec.tube_od / 2.0
    fx = nx * spec.pitch - spec.clearance
    # Stud centres sit at +-(nx-1)/2 * pitch; the outermost reaches stud_d/2 beyond that.
    outer_stud = (nx - 1) / 2.0 * spec.pitch + spec.stud_d / 2.0
    wall_inner = fx / 2.0 - spec.wall
    n_tubes_x, n_tubes_y = max(nx - 1, 0), max(ny - 1, 0)
    # A stud at grid (i, j) neighbours the tube positions at (i-1..i, j-1..j) that exist.
    n_contacts = 0
    for i in range(nx):
        for j in range(ny):
            for ti in (i - 1, i):
                for tj in (j - 1, j):
                    if 0 <= ti < n_tubes_x and 0 <= tj < n_tubes_y:
                        n_contacts += 1
    n_studs = nx * ny
    return PressFitGeometry(
        half_diagonal=d,
        radius_sum=r_sum,
        interference=r_sum - d,
        wall_gap=wall_inner - outer_stud,
        n_contacts=n_contacts,
        n_studs=n_studs,
        contacts_per_stud=n_contacts / n_studs if n_studs else 0.0,
    )


# ---------------------------------------------------------------------------
# Member stiffnesses
# ---------------------------------------------------------------------------


def ring_radial_stiffness(r_mean: float, t: float, band: float, E: float) -> float:
    """Radial stiffness [N/m] of a thin ring squeezed between two opposite point loads.

    A ring loaded diametrically changes diameter by ``dd = 0.149 F R**3 / (E I)`` (the standard
    Roark case), so the stiffness at one of the two load points, measured against the ring's own
    centre, is ``2 F / dd``. ``band`` is the axial length of ring that is actually loaded, which
    for a stud in a tube is the engaged stud height.

    This is what both the tube and the hollow stud do when a press fit is taken up.
    """

    inertia = band * t**3 / 12.0
    return 2.0 * E * inertia / (0.149 * r_mean**3)


def wall_cantilever_stiffness(E: float, nu: float, t: float, L: float, b: float) -> float:
    """Stiffness [N/m] of a wall strip built into the roof and free at the bottom.

    ``k = 3 E' I / L**3`` with ``I = b t**3 / 12`` and ``E' = E / (1 - nu**2)`` for plane
    strain. ``L`` is the distance from the load to the roof and ``b`` the width the load spreads
    over.

    A wall has no single stiffness. ``k`` goes as ``b / L**3``, so the same DUPLO wall is
    7.2 kN/m against a 10 mm-wide load near its bottom edge (L = 15 mm) and 100 kN/m against a
    30 mm-wide load at mid-height (L = 9 mm) - a factor of fourteen across load cases this arm
    can actually produce.
    """

    inertia = b * t**3 / 12.0
    return 3.0 * (E / (1.0 - nu**2)) * inertia / L**3


@dataclass(frozen=True)
class PressFitStiffness:
    """The elastic members that take up the interference, and what they add up to.

    All stiffnesses N/m, all forces N. ``k_series`` is the tube and the hollow stud in series,
    which is the load path that actually carries a press fit in this geometry.
    """

    k_tube: float
    k_stud: float
    k_series: float
    k_wall_soft: float
    k_wall_stiff: float
    k_hertz: float
    interference: float
    normal_force_per_contact: float


def press_fit_stiffness(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
    interference: float | None = None,
) -> PressFitStiffness:
    """Every member stiffness of the press fit, for one brick family.

    Three members can take up the interference and they are wildly unequal. For DUPLO in ABS at
    23 C: the tube ring is 130 kN/m, the hollow stud ring 562 kN/m, the two in series 106 kN/m,
    and the most compliant wall case 7.2 kN/m - fifteen times softer than the path that is
    actually loaded.

    Local Hertzian flattening is included as ``k_hertz`` and is deliberately reported so it can
    be seen to be negligible: two parallel cylinders approach each other by about 2.6 um at 5 N
    over a 4.5 mm band, under 3 % of an 86 um take-up. **The press fit is structural flexure,
    not contact deformation**, which is why simple ring and beam formulas are the right tools
    here and a contact-mechanics treatment is not needed.
    """

    geom = press_fit_geometry(spec, nx, ny)
    delta = geom.interference if interference is None else interference
    band = spec.stud_h
    E = material.E

    r_tube_mean = (spec.tube_od + spec.tube_id) / 4.0
    t_tube = (spec.tube_od - spec.tube_id) / 2.0
    k_tube = ring_radial_stiffness(r_tube_mean, t_tube, band, E)

    if spec.stud_inner_d > 0.0:
        r_stud_mean = (spec.stud_d + spec.stud_inner_d) / 4.0
        t_stud = (spec.stud_d - spec.stud_inner_d) / 2.0
        k_stud = ring_radial_stiffness(r_stud_mean, t_stud, band, E)
    else:
        # A solid stud does not flex as a ring; it is stiff enough that the tube takes
        # essentially all of the interference.
        k_stud = math.inf

    k_series = 1.0 / (1.0 / k_tube + 1.0 / k_stud) if math.isfinite(k_stud) else k_tube

    wall_h = spec.height - spec.roof
    k_wall_soft = wall_cantilever_stiffness(E, material.nu, spec.wall, 0.85 * wall_h, 0.010)
    k_wall_stiff = wall_cantilever_stiffness(E, material.nu, spec.wall, 0.51 * wall_h, 0.030)

    f_probe = 5.0
    _, _, approach = hertz_line_contact(
        f_probe, spec.stud_d / 2.0, spec.tube_od / 2.0, band, material
    )
    k_hertz = f_probe / approach if approach > 0.0 else math.inf

    return PressFitStiffness(
        k_tube=k_tube,
        k_stud=k_stud,
        k_series=k_series,
        k_wall_soft=k_wall_soft,
        k_wall_stiff=k_wall_stiff,
        k_hertz=k_hertz,
        interference=delta,
        normal_force_per_contact=max(k_series * delta, 0.0),
    )


def hertz_line_contact(
    force: float, r1: float, r2: float, length: float, material: Material = ABS_BRICK_23C
) -> tuple[float, float, float]:
    """Half-width [m], peak pressure [Pa] and approach [m] for two parallel cylinders.

    Johnson's line-contact result with ``1/R = 1/r1 + 1/r2`` and
    ``E* = E / (2 (1 - nu**2))`` for two bodies of the same material. The approach is the
    logarithmic form, evaluated with the contact half-width, which is enough for the "is this
    negligible?" question it is asked here.

    At a designed clutch load the peak pressure is 13 MPa at 5 N and 17 MPa at 9 N, both well
    under the 44 MPa yield: **a correctly seated stud does not mark its tube**.
    """

    if force <= 0.0 or length <= 0.0:
        return (0.0, 0.0, 0.0)
    r_eff = 1.0 / (1.0 / r1 + 1.0 / r2)
    e_star = material.contact_modulus
    p_line = force / length
    half_width = math.sqrt(4.0 * p_line * r_eff / (math.pi * e_star))
    p_max = 2.0 * p_line / (math.pi * half_width)
    approach = (p_line / (math.pi * e_star)) * (2.0 * math.log(4.0 * r_eff / half_width) - 1.0)
    return (half_width, p_max, max(approach, 0.0))


# ---------------------------------------------------------------------------
# Clutch force
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClutchEstimate:
    """Holding and push-on force derived from material and geometry. Forces in newtons.

    ``per_stud`` is the **clutch force per stud**: the same quantity as ``SnapParams.f_clutch``
    and ``configs/params/default.yaml``'s ``bricks.clutch_force_per_stud_N`` -- the average of
    push-on and pull-off force per stud, not either one alone. ``push_on_per_stud``/
    ``push_on_per_brick`` are what a press actually has to reach: ``push_on = per_stud * (1 +
    beta)``, matching ``SnapParams``/``snap_api.engagement()`` (and checked directly against
    ``brick_builder.bricks.bench.PressBench``) -- pull-off would be ``per_stud * (1 - beta)``, which
    this estimate does not compute because nothing here needs it.

    ``lo``/``hi`` are the honest band, not an error bar: they come from sweeping the
    interference over the nominal-to-caliper spread and the friction over its randomization
    range. On DUPLO that band is **0.3 to 6 N per stud**, a factor of twenty, and the
    uncertainty is not in the mechanics - it is in a single geometric difference of two numbers
    near 13 mm and in a friction coefficient nobody has measured.
    """

    per_stud: float
    per_brick: float
    per_stud_lo: float
    per_stud_hi: float
    push_on_per_stud: float
    push_on_per_brick: float
    normal_force_per_contact: float
    contacts_per_stud: float
    n_studs: int
    mu: float
    interference: float
    path: str


def clutch_force(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
    mu: float | None = None,
    beta: float = 0.15,
    interference: float | None = None,
    path: Literal["stud_on_tube", "wall"] = "stud_on_tube",
) -> ClutchEstimate:
    """Clutch force per stud and per brick, from first principles.

    ``clutch force per stud = mu * k * delta * (contacts per stud)``. With the stud-on-tube
    path, DUPLO nominal geometry and mu = 0.35 that is about 4.8 N per stud and 38 N for a 2x4.
    This is the average of push-on and pull-off force per stud (the same quantity as
    ``SnapParams.f_clutch``), not either one alone; see :class:`ClutchEstimate`.

    ``path="wall"`` reproduces the first-pass estimate that treated the wall as the member that
    deflects, which gives 0.3 to 0.4 N per stud - close to the simulator's 1.0 N placeholder,
    and wrong, because :func:`press_fit_geometry` shows the wall is 1.65 mm clear of the studs
    in the geometry the generator actually builds. It is kept so the two can be compared.

    The band is produced by sweeping the interference from the nominal value to the value the
    caliper figure for the tube gives (0.086 to 0.126 mm on DUPLO), and the friction over
    :data:`FRICTION_PAIRS`\\ ``["abs_on_abs"]``. Where the geometry gives no interference at all
    - which is what the LEGO System design rule produces, and what this project's
    :data:`~brick_builder.bricks.spec.LEGO_SYSTEM` variant currently has - the lower end of the band
    is zero and only a measurement can settle it.
    """

    pair = FRICTION_PAIRS["abs_on_abs"]
    mu_nom = pair.mu.nominal if mu is None else mu
    mu_lo, mu_hi = pair.mu.range
    geom = press_fit_geometry(spec, nx, ny)
    delta = geom.interference if interference is None else interference
    # The caliper figure for the DUPLO tube (13.48 mm against the nominal 13.4) is 0.04 mm on
    # the diameter, so 0.04 mm more on the radius sum; the same fraction is used for any spec.
    delta_hi = delta + 0.04e-3 * (spec.tube_od / DUPLO.tube_od)

    def one(d: float, m: float) -> tuple[float, float]:
        stiff = press_fit_stiffness(spec, material, nx, ny, interference=d)
        if path == "wall":
            wall_h = spec.height - spec.roof
            k = wall_cantilever_stiffness(
                material.E, material.nu, spec.wall, 0.85 * wall_h, 0.010
            )
            fn = max(k * d, 0.0)
        else:
            fn = stiff.normal_force_per_contact
        return fn, m * fn * geom.contacts_per_stud

    fn_nom, per_stud = one(delta, mu_nom)
    _, lo = one(delta, mu_lo)
    _, hi = one(delta_hi, mu_hi)
    return ClutchEstimate(
        per_stud=per_stud,
        per_brick=per_stud * geom.n_studs,
        per_stud_lo=min(lo, hi),
        per_stud_hi=max(lo, hi),
        push_on_per_stud=per_stud * (1.0 + beta),
        push_on_per_brick=per_stud * geom.n_studs * (1.0 + beta),
        normal_force_per_contact=fn_nom,
        contacts_per_stud=geom.contacts_per_stud,
        n_studs=geom.n_studs,
        mu=mu_nom,
        interference=delta,
        path=path,
    )


def draft_exponent(spec: BrickSpec = DUPLO, draft_deg: float = 1.0) -> float:
    """The force-against-depth exponent implied by a given mould draft angle.

    For a straight-sided interference between two parallel cylinders the normal force per unit
    of engaged length is constant, friction is proportional to engaged length, and the press
    force rises **linearly** with depth - exactly ``engagement()``'s ``"ramp"`` family with
    ``exponent = 1``.

    Mould draft changes that. Injection-moulded parts need 0.5 to 1.5 degrees per side to eject,
    and over a 4.5 mm DUPLO stud **1 degree of draft changes the radius by 79 um**, which is the
    size of the whole nominal interference. If the parts are drafted so the fit tightens going
    in, the interference grows with depth, friction goes as the product, and the force rises
    roughly as **depth squared**.

    This function interpolates between the two: exponent 1 at zero draft, rising toward 2 as the
    draft-induced radius change approaches the nominal interference, and beyond 2 when it
    exceeds it. The right default is therefore not 1 but an exponent randomized over 1.0 to 2.5
    until the curve is measured - the research found no published force-against-displacement
    curve for seating a LEGO brick, from any source.
    """

    geom = press_fit_geometry(spec)
    d_radius = spec.stud_h * math.tan(math.radians(draft_deg))
    if geom.interference <= 0.0:
        return 2.0 if d_radius > 0.0 else 1.0
    return 1.0 + min(d_radius / geom.interference, 1.5)


def draft_radius_change(spec: BrickSpec = DUPLO, draft_deg: float = 1.0) -> float:
    """Radius change [m] over the engaged stud height for a given draft angle per side."""

    return spec.stud_h * math.tan(math.radians(draft_deg))


def force_vs_depth(
    depth: np.ndarray | float,
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
    mu: float | None = None,
    beta: float = 0.15,
    draft_deg: float = 1.0,
    engage_depth: float = 0.0005,
    n_studs: int | None = None,
) -> np.ndarray:
    """Derived push-on force [N] against insertion depth [m], for a whole brick.

    The shape is the physics, not a fitted family: friction is ``mu`` times the normal force
    integrated over the engaged length, the normal force per unit length is the member
    stiffness times the local interference, and the local interference grows with depth when
    there is mould draft. That gives

        F(d) = n_studs * f_clutch * (1 + beta) * ((d - engage) / (h - engage)) ** p

    with ``p`` from :func:`draft_exponent`. At zero draft ``p = 1`` and this is exactly the
    simulator's ``"ramp"`` default; at 1 degree of draft on DUPLO it is close to 2.

    What this curve does **not** have is the entry peak of a chamfered stud tip, where the tip
    wedges a larger local deflection than the steady state before it slides in - the "click".
    That is the ``"detent"`` family's job and its bump parameters are what a real measurement
    would fit.
    """

    d = np.asarray(depth, dtype=np.float64)
    est = clutch_force(spec, material, nx, ny, mu=mu, beta=beta)
    n = est.n_studs if n_studs is None else n_studs
    p = draft_exponent(spec, draft_deg)
    span = max(spec.stud_h - engage_depth, 1e-9)
    ramp = np.clip((d - engage_depth) / span, 0.0, 1.0)
    return n * est.per_stud * (1.0 + beta) * ramp**p


def breakaway_travel(spec: BrickSpec = DUPLO, material: Material = ABS_BRICK_23C) -> dict:
    """Elastic travel [m] before a stuck stud breaks away, for the stiff and soft load paths.

    Dry friction against an elastic member is a stick-slip system, and the length scale is
    ``delta_break = f_clutch / k``. On the tube path at 4 N per stud that is 38 um; on the wall
    path it is 550 um. A real press should therefore judder on a scale of tens of microns, well
    below anything this arm can either command or observe, so the simulator does not need to
    resolve it. MuJoCo solves dry friction properly, so the per-stud friction joint model gets
    the behaviour for free anyway.
    """

    est = clutch_force(spec, material)
    stiff = press_fit_stiffness(spec, material)
    return {
        "f_clutch_per_stud_N": est.per_stud,
        "stud_on_tube_m": est.per_stud / stiff.k_series,
        "wall_path_m": est.per_stud / stiff.k_wall_soft,
    }


# ---------------------------------------------------------------------------
# Contact settings for the simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactSettings:
    """MuJoCo contact parameters for one pair, with the reasoning attached."""

    pair: str
    condim: int
    priority: int
    solref: tuple[float, float]
    solimp: tuple[float, float, float]
    friction: tuple[float, float, float]
    friction_range: tuple[float, float]
    note: str


def contact_settings(timestep: float = 0.002) -> dict[str, ContactSettings]:
    """Per-pair ``solref``/``solimp``/``friction``, with the time-step floor respected.

    MuJoCo's soft constraint takes a ``solref`` of (time constant, damping ratio) and the time
    constant must be at least twice the step, so **a 2 ms step forces a 4 ms floor** and this
    project sits exactly on it. Three measured facts from the guide's section 3 drive these
    numbers:

    - At the arm's own 14 N press ceiling, a brick sinks **180.7 um** into the plate - deeper
      than the entire 86 um interference the clutch is made of. That is the quantitative reason
      the clutch lives on a dry-friction joint in ``snap.py`` instead of being built out of
      geometric interference: ordinary contact at this step cannot hold a fit that is smaller
      than its own penetration error.
    - Reducing the step does not help. At 0.5, 1 and 2 ms with the time constant held at 4 ms
      the penetration is identical, because the time constant sets the stiffness, not the step.
    - A damping ratio of 0.4 **creates energy** (restitution above 1) and is not usable at any
      step size. At 1.0 the simulator's restitution is about 0.2, which under-bounces a real
      brick (0.4 to 0.7 would be typical). Since the only thing that depends on it is how far a
      dropped brick skitters, that is acceptable - but it should be known, not discovered.
    """

    ref = (max(0.004, 2.0 * timestep), 1.0)
    imp = (0.95, 0.999, 0.0005)
    abs_abs = FRICTION_PAIRS["abs_on_abs"]
    pla_abs = FRICTION_PAIRS["pla_on_abs"]
    table = FRICTION_PAIRS["abs_on_table"]
    return {
        "brick_on_brick": ContactSettings(
            "brick on brick / brick on baseplate", 3, 0, ref, imp,
            abs_abs.friction, abs_abs.mu.range,
            "friction has no source; randomize 0.2 to 0.5",
        ),
        "pad_on_brick": ContactSettings(
            "jaw pad on brick (PLA on ABS)", 4, 1, ref, imp,
            pla_abs.friction, pla_abs.mu.range,
            "condim 4 with priority 1 is already correct and should stay: it gives the pad "
            "torsional friction, which stops a gripped brick spinning about the pinch axis, "
            "and the priority makes the pad's friction win over the brick's. The value of 1.0 "
            "currently in ArmConfig is a modelling choice, not a measurement",
        ),
        "brick_on_table": ContactSettings(
            "brick on table", 3, 0, ref, imp,
            table.friction, table.mu.range,
            "depends entirely on the table",
        ),
        "arm_on_anything": ContactSettings(
            "arm link on anything", 3, 0, ref, imp,
            (0.35, 0.005, 0.0001), (0.2, 0.5),
            "collisions, not manipulation; nothing rests on these",
        ),
    }


def randomization_ranges(spec: BrickSpec = DUPLO, material: Material = ABS_BRICK_23C) -> dict:
    """Everything worth randomizing per episode, as ``name -> (lo, nominal, hi)``.

    The sensitivity behind the clutch range, from the guide's section 2.4:

    ===========================================  ==========================
    Perturbation                                 Effect on holding force
    ===========================================  ==========================
    +-0.01 mm on the tube diameter (mould tol.)  +-6 to +-12 %
    +-0.05 mm (caliper uncertainty)              +-30 to +-60 %
    friction 0.2 to 0.5                          -33 % to +67 %
    E over 1.8 to 2.7 GPa                        +-20 %
    1 or 2 engaged contacts per stud             +-33 %
    ===========================================  ==========================

    Randomizing ``f_clutch`` over a factor of three either side of nominal is not excessive
    caution. It is an honest reflection of what is known.

    Temperature is in here as a range of 18 to 28 C, and it is worth being precise about how
    little it does. **Thermal expansion does almost nothing**: the interference is a difference
    between dimensions of two parts of the same material at the same temperature, so both scale
    by ``(1 + alpha dT)`` and so does the difference - a 10 K change alters an 86 um
    interference by 0.09 %. The modulus does the work, and over an 18 to 28 C room that is about
    +-3.5 %. The temperature effect that *does* matter in this project is on the arm, where a
    printed bracket beside a working servo can reach its own heat deflection temperature of 45
    to 60 C while the bricks two centimetres away are still at 23 C.
    """

    est = clutch_force(spec, material)
    return {
        "f_clutch_N": (est.per_stud_lo, est.per_stud, est.per_stud_hi),
        "f_clutch_factor": (1.0 / 3.0, 1.0, 3.0),
        "beta": (0.10, 0.15, 0.28),
        "exponent": (1.0, draft_exponent(spec, 1.0), 2.5),
        "mu_abs_on_abs": (0.20, 0.35, 0.50),
        "mu_pla_on_abs": (0.30, 0.40, 1.00),
        "mu_brick_on_table": (0.20, 0.40, 0.80),
        # The fillet stress concentration at the wall-to-roof junction. Nominal 1.0 keeps the
        # default a nominal-stress model; the range is one-sided upward because a real moulded
        # corner can only make the wall weaker than the beam formula says. See WALL_KT_RANGE.
        "wall_kt": (WALL_KT_RANGE[0], 1.0, WALL_KT_RANGE[1]),
        "youngs_modulus_Pa": (1.8e9, 2.3e9, 2.7e9),
        "density_kg_m3": (1030.0, 1050.0, 1070.0),
        "allowable_scale": (0.70, 1.00, 1.30),
        "fatigue_cycles_ref": (12371.0, 37112.0, 111336.0),
        "fatigue_exponent": (4.0, 8.0, 12.0),
        "draft_deg": (0.5, 1.0, 1.5),
        "lead_in_deg": (30.0, 45.0, 60.0),
        "temperature_C": (18.0, 23.0, 28.0),
    }


def derived_snap_params(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
    draft_deg: float = 1.0,
    mu: float | None = None,
):
    """A :class:`~brick_builder.bricks.snap_api.SnapParams` built from the material model.

    Opt-in and non-destructive: this constructs a *new* params object and returns it. Nothing
    in the simulator changes unless a caller passes it to a snap model.

    Be aware of what this implies. If the clutch really is around 4 to 5 N per stud, a 2x4's
    whole-brick clutch (8 studs) is 33 to 48 N, and seating it flat takes about 1.15x that in
    push -- more than this arm has; its measured static press at the build plate is 10 to 14 N.
    **A flat press of a 2x4 may simply not be within this arm's capability**, which turns the
    tilted press from an optimisation into a requirement. That is a planning consequence worth
    knowing before training runs rather than after, and it is why the guide's section 7.2
    measurement matters more than any of this arithmetic.
    """

    from brick_builder.bricks.snap_api import SnapParams  # local: keep this module import-light

    est = clutch_force(spec, material, nx, ny, mu=mu)
    return SnapParams(
        f_clutch=est.per_stud,
        exponent=draft_exponent(spec, draft_deg),
        curve="ramp",
    )


# ---------------------------------------------------------------------------
# Sections and failure thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """One structural cross-section of a generated brick, as the generator actually builds it.

    ``area`` m^2, ``modulus`` m^3 (the elastic section modulus Z = I / c), ``length`` and
    ``thickness`` m. Bending about the weak axis, which is the one that matters for a wall.
    """

    name: str
    area: float
    modulus: float
    length: float
    thickness: float


def sections(spec: BrickSpec = DUPLO, nx: int = 2, ny: int = 4) -> dict[str, Section]:
    """Cross-sections of every structural member of an ``nx`` by ``ny`` brick.

    Taken from the same shell :mod:`brick_builder.bricks.generator` builds, including its
    ``clearance``-based footprint shrink and its corner-overlap convention (front and back walls
    span the full footprint, left and right are shortened so the four boxes do not overlap).

    For a DUPLO 2x4 that is four walls totalling 286.8 mm^2 and three tubes totalling
    153.3 mm^2, 440.1 mm^2 in all. The generated 2x4 masses 12.52 g against the 12.2 g catalogue
    figure, and the generated System 2x2 masses 1.115 g against the 1.152 g weighed in the Open
    University test, so the shell model and the 1050 kg/m^3 density are both within 3 %.
    """

    fx = nx * spec.pitch - spec.clearance
    fy = ny * spec.pitch - spec.clearance
    t = spec.wall
    out: dict[str, Section] = {}
    for name, length in (("wall_front_back", fx), ("wall_left_right", fy - 2.0 * t)):
        out[name] = Section(name, length * t, length * t**2 / 6.0, length, t)
    r_o, r_i = spec.tube_od / 2.0, spec.tube_id / 2.0
    tube_area = math.pi * (r_o**2 - r_i**2)
    tube_z = math.pi * (spec.tube_od**4 - spec.tube_id**4) / (32.0 * spec.tube_od)
    out["tube"] = Section("tube", tube_area, tube_z, spec.height - spec.roof,
                          (spec.tube_od - spec.tube_id) / 2.0)
    s_o, s_i = spec.stud_d / 2.0, spec.stud_inner_d / 2.0
    stud_area = math.pi * (s_o**2 - s_i**2)
    stud_z = math.pi * (spec.stud_d**4 - spec.stud_inner_d**4) / (32.0 * spec.stud_d)
    out["stud"] = Section("stud", stud_area, stud_z, spec.stud_h,
                          (spec.stud_d - spec.stud_inner_d) / 2.0)
    out["roof"] = Section("roof", fx * spec.roof, fx * spec.roof**2 / 6.0, fx, spec.roof)
    return out


def plate_buckling_stress(
    E: float, nu: float, t: float, a: float, b: float
) -> float:
    """Critical compressive stress [Pa] of a flat plate loaded along its height ``a``.

    ``sigma_cr = k pi**2 E / (12 (1 - nu**2)) * (t / b)**2`` with ``k = (b/a + a/b)**2`` for a
    short wide plate simply supported on its edges.

    **Thin walls buckle before they yield**, which is the correction that turns a first-pass
    crush estimate of 19 kN into 13.2 kN. A DUPLO 2x4 long wall at the guide's idealised
    63.8 mm footprint edge (a = 17.7 mm, t = 1.55 mm) gives k = 15.1 and sigma_cr = 19.2 MPa,
    well below the 44 MPa yield; the generator's own left/right wall is a little shorter
    (60.7 mm, after the corner-overlap shrink) and goes critical at 19.5 MPa. Tubes do not
    buckle: a cylinder's critical stress is ``0.6 E t / R`` = 309 MPa, so they yield first.
    """

    k = (b / a + a / b) ** 2
    return k * math.pi**2 * E / (12.0 * (1.0 - nu**2)) * (t / b) ** 2


def crush_load(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
) -> dict:
    """Compressive load [N] to collapse a brick, and which mode gets there first.

    For a DUPLO 2x4 this is **13.2 kN, by wall buckling, not by yielding**, and that number is
    worth deriving once because an earlier draft of the guide said 12.2 kN:

    - Each *pair* of walls gets its own critical stress, because buckling depends on the wall's
      length. The front and back walls are 31.8 mm long and go critical at 28.4 MPa; the left
      and right walls are 60.7 mm long (the generator shortens them so the four boxes do not
      overlap at the corners) and go critical at 19.5 MPa. That is 2.80 kN plus 3.66 kN,
      6.46 kN of wall.
    - The three tubes do not buckle (309 MPa critical against a 44 MPa yield), so they yield:
      153.3 mm^2 at 44 MPa, 6.75 kN.
    - Total 13.2 kN over 440.1 mm^2.

    The 12.2 kN came from applying the *softest* wall's buckling stress (19.2 MPa, worked at
    the idealised 63.8 mm footprint edge) to all 286.8 mm^2 of wall, which is a fair
    conservative simplification but is not what this function does. Per-wall-pair is the more
    accurate of the two, so 13.2 kN is what the guide now quotes everywhere. A DUPLO 2x2 comes
    out at **7.85 kN** by the same route (2.80 + 2.80 kN of wall, one tube at 2.25 kN).

    The cross-check: the Open University measured a LEGO System 2x2 failing at 4240 N with a
    slow plastic collapse rather than a fracture, and its 82.9 mm^2 section makes that a mean
    stress of 51.1 MPa; that brick's walls are short and stubby (k = 5.6, sigma_cr = 69 MPa) so
    it really should have yielded rather than buckled, and it did, at close to the ABS yield
    stress. The two cases agree on the number and on the mechanism.

    Whichever number is used, this is **three orders of magnitude beyond the arm**, whose
    measured static press at the plate is 10 to 14 N. It is in the model as a ceiling, not as a
    risk.
    """

    sec = sections(spec, nx, ny)
    wall_h = spec.height - spec.roof
    E, nu = material.E, material.nu
    walls = {}
    total_wall_force = 0.0
    total_wall_area = 0.0
    for key in ("wall_front_back", "wall_left_right"):
        s = sec[key]
        sigma_cr = plate_buckling_stress(E, nu, s.thickness, wall_h, s.length)
        sigma = min(sigma_cr, material.tensile_yield.nominal)
        n = 2  # two of each
        walls[key] = {
            "sigma_buckling_Pa": sigma_cr,
            "sigma_used_Pa": sigma,
            "mode": "buckling" if sigma_cr < material.tensile_yield.nominal else "yield",
            "force_N": n * s.area * sigma,
        }
        total_wall_force += n * s.area * sigma
        total_wall_area += n * s.area

    n_tubes = max(nx - 1, 0) * max(ny - 1, 0)
    tube = sec["tube"]
    r_mean = (spec.tube_od + spec.tube_id) / 4.0
    sigma_cyl = 0.6 * E * tube.thickness / r_mean
    tube_sigma = min(sigma_cyl, material.tensile_yield.nominal)
    tube_force = n_tubes * tube.area * tube_sigma
    return {
        "walls": walls,
        "wall_force_N": total_wall_force,
        "wall_area_m2": total_wall_area,
        "tube_force_N": tube_force,
        "tube_area_m2": n_tubes * tube.area,
        "tube_sigma_cr_Pa": sigma_cyl,
        "tube_mode": "buckling" if sigma_cyl < material.tensile_yield.nominal else "yield",
        "total_N": total_wall_force + tube_force,
        "total_area_m2": total_wall_area + n_tubes * tube.area,
    }


def stud_shear_load(spec: BrickSpec = DUPLO, material: Material = ABS_BRICK_23C) -> dict:
    """Force [N] to shear one stud off the roof.

    The stud root is an annulus; at the assumed shear strength (0.6 x yield = 26 MPa) a DUPLO
    stud takes **942 N**, and 1412 N if the flexural figure is used instead. Two orders of
    magnitude beyond the arm.
    """

    area = sections(spec)["stud"].area
    return {
        "area_m2": area,
        "shear_N": area * material.shear_strength.nominal,
        "flexural_bound_N": area * material.flexural_strength.nominal * 0.6,
    }


#: Stress concentration factor at the wall-to-roof junction, as a named approximation rather
#: than a silent assumption. The beam formula ``sigma = 6 F L / (b t**2)`` gives the *nominal*
#: stress at the root; the real junction is a moulded internal corner with a fillet, and the
#: local stress there is higher by ``K_t``. Published charts for a filleted shoulder in bending
#: put ``K_t`` at roughly 1.3 to 2.0 for fillet radii of the order a 1.55 mm wall would carry
#: (r/t around 0.1 to 0.5), falling toward 1.0 only with a generous radius.
#:
#: **Nobody has measured it on a DUPLO brick and this model does not apply it by default.** The
#: default is 1.0, so every number this module prints is a nominal-stress number, exactly as
#: before. What matters is the *direction* of the error, which is one-sided: a real wall is
#: weaker than this model says, by up to a factor of two, never stronger. Randomize over
#: :data:`WALL_KT_RANGE` to see what that does to a policy, and settle it with the guide's
#: section 7.7 destructive test, which measures the force at a visible set directly and so
#: contains ``K_t`` whether or not anyone names it.
WALL_KT_RANGE: tuple[float, float] = (1.0, 2.0)


def wall_bending_stress(
    force: float, lever: float, b_eff: float, thickness: float, k_t: float = 1.0
) -> float:
    """Bending stress [Pa] in a wall strip treated as a cantilever built into the roof.

    ``sigma = K_t 6 F L / (b t**2)``, with ``F`` the component of the contact force
    perpendicular to the wall's own plane, ``L`` the distance from the contact up to the roof,
    ``b`` the width the load spreads over, and ``K_t`` the stress concentration at the
    wall-to-roof fillet (:data:`WALL_KT_RANGE`; 1.0, the nominal-stress case, by default).

    **This is the failure mode that matters.** A misaligned stud whose chamfer rides up the
    inside of a wall pushes that wall outward, and on a DUPLO wall at L = 15 mm the thresholds
    are a permanent set at 12 to 35 N and a crack at 19 to 56 N depending on how far the load
    spreads. The permanent-set threshold arrives first, and a wall that has taken a set never
    grips properly again - a functional failure even though the brick is intact.
    """

    if b_eff <= 0.0 or thickness <= 0.0:
        return 0.0
    return k_t * 6.0 * force * lever / (b_eff * thickness**2)


def wall_bending_force(
    sigma: float, lever: float, b_eff: float, thickness: float, k_t: float = 1.0
) -> float:
    """The inverse of :func:`wall_bending_stress`: force [N] that produces ``sigma``."""

    return sigma * b_eff * thickness**2 / (6.0 * lever * max(k_t, 1e-9))


def wall_thresholds(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    lever: float | None = None,
    k_t: float = 1.0,
) -> dict:
    """Permanent-set and crack forces [N] on a wall, over the range of load spreads.

    The spread is the weakest assumption in the whole damage model, which is why this returns a
    band for three widths rather than a single number. Section 7.7 of the guide is the
    destructive test that would narrow it, at a cost of three bricks.

    ``k_t`` is the fillet stress concentration of :data:`WALL_KT_RANGE`. It divides every
    threshold, so on a DUPLO 2x4 at L = 15 mm:

    ======  ===============  ================
    K_t     permanent set    crack
    ======  ===============  ================
    1.0     11.7 to 35.1 N   18.6 to 55.9 N
    1.3     9.0 to 27.0 N    14.3 to 43.0 N
    2.0     5.9 to 17.6 N    9.3 to 27.9 N
    ======  ===============  ================

    At the pessimistic end that puts the permanent-set floor at 5.9 N, which is *inside* what
    the arm can apply (10 to 14 N of press, 17 N of grip) rather than at the edge of it. The
    default stays 1.0 so nothing moves silently, but a reader deciding how much margin to leave
    should be looking at the bottom row, not the top one.
    """

    wall_h = spec.height - spec.roof
    L = 0.85 * wall_h if lever is None else lever
    out = {"lever_m": L, "thickness_m": spec.wall, "k_t": k_t}
    for b in (0.010, 0.020, 0.030):
        out[f"b_{int(b * 1000)}mm"] = {
            "set_N": wall_bending_force(material.tensile_yield.nominal, L, b, spec.wall, k_t),
            "crack_N": wall_bending_force(
                material.flexural_strength.nominal, L, b, spec.wall, k_t
            ),
        }
    out["set_band_N"] = (out["b_10mm"]["set_N"], out["b_30mm"]["set_N"])
    out["crack_band_N"] = (out["b_10mm"]["crack_N"], out["b_30mm"]["crack_N"])
    return out


def wedge_ratio(theta_deg: float, mu: float = 0.35) -> float:
    """Sideways force per unit of vertical press, for a press on an inclined lead-in.

    ``F_lateral / F_vertical = (tan(theta) - mu) / (1 + mu tan(theta))`` with theta measured
    from the press axis. At 45 degrees and mu = 0.3 the ratio is 0.54, at 60 degrees 0.94, at
    30 degrees 0.24. So a 14 N press on a badly misaligned stud delivers 3 to 13 N sideways -
    under the permanent-set threshold, but within a factor of two of it, and the margin
    disappears entirely if the contact is a sharp edge rather than a spread one.
    """

    t = math.tan(math.radians(theta_deg))
    return (t - mu) / (1.0 + mu * t)


def impact_force(m_eff: float, k: float, v: float) -> float:
    """Peak force [N] of a striker of effective mass ``m_eff`` at speed ``v`` on a spring ``k``.

    ``F = v sqrt(m k)``, the energy balance for a linear spring.

    **This is the only way this arm can break a brick, and it is a speed limit rather than a
    force limit.** The measured operational-space inertia of the patched arm at press poses,
    ``1 / (J M^-1 J^T)_zz``, is 0.32 to 0.62 kg over the working region. A brick held in the jaw
    is backed by that whole mass, so driving it into something solid at 0.3 m/s gives 19 N and
    at 1.0 m/s gives 63 N - across the 19 to 56 N wall-cracking band. A *free* brick is backed
    only by its own 12 g, absorbs about 10 N and is simply knocked away.

    A brick dropped 0.3 m onto a table computes to 83 N, and that is in the guide as a negative
    control: ABS has a notched Izod impact strength of 300 J/m, DUPLO bricks survive being
    dropped by toddlers all day, and any damage model that flags an ordinary drop as breakage is
    wrong.
    """

    return v * math.sqrt(max(m_eff, 0.0) * max(k, 0.0))


#: Operational-space inertia of the patched SO-101 at press poses, measured on this repository:
#: 0.62 kg at (0.17, 0, 0.03) m, 0.50 kg at (0.17, 0, 0.08) m, 0.32 kg at (0.25, 0, 0.05) m.
ARM_EFFECTIVE_MASS_KG: tuple[float, float] = (0.32, 0.62)

#: The arm's measured static press at the build plate, from this repository's own force map at
#: 7.4 V with the datasheet torque limit. The identified servo law's higher ceiling gives 16 to
#: 23 N instead; both are in the guide's section 5.
ARM_PRESS_FORCE_N: tuple[float, float] = (10.0, 14.0)

#: Grip force the stock gripper can apply to a brick face, same source.
ARM_GRIP_FORCE_N: float = 17.0

#: The press-policy limits of the guide's section 5, with a safety factor of 2 against the
#: bottom of each band. These are the numbers a reward function should be shaped around.
POLICY_LIMITS: dict[str, tuple[float, str]] = {
    "press_force_N": (7.0, "half of 14 N, and half the 12 N permanent-set threshold once the "
                           "wedge factor is applied"),
    "lateral_wall_force_N": (6.0, "half of the 12 N permanent-set threshold"),
    "contact_speed_m_s": (0.15, "half the 0.3 m/s that reaches 19 N with 0.5 kg of tool"),
    "grip_force_N": (8.0, "half of 17 N; also keeps a 3 mm edge contact under 60 MPa"),
    "wrist_torque_Nm": (0.5, "a quarter of the 1.9 N*m servo limit, leaving room for gravity"),
}


@dataclass(frozen=True)
class WeldLimit:
    """Force [N] and moment [N*m] one piece-to-roof joint can carry before it lets go.

    ``force_N`` is the **governing** axial capacity, which for a wall or a tube is the lower of
    yielding and buckling, and for a stud is shear at its root. ``force_tension_N`` is the
    yield-only figure, which is what the guide's section 6.2 table quotes; the two differ
    wherever the member is thin enough to go critical first, and the DUPLO 2x4 long wall does
    exactly that at 19.2 MPa against a 44 MPa yield.
    """

    name: str
    area: float
    modulus: float
    force_N: float
    moment_Nm: float
    mode: str
    force_tension_N: float = 0.0
    critical_stress_Pa: float = 0.0


def weld_limits(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
) -> dict[str, WeldLimit]:
    """Limits for the breakable brick's weld equalities, from the real cross-sections.

    Walls and tubes are limited by the tensile yield in axial load and the flexural strength in
    bending; a stud is limited by shear at its root, which is the mode that actually detaches
    one. For a DUPLO 2x4 in ABS this gives roughly:

    =================  ===========  ===========
    Joint              Axial limit  Moment
    =================  ===========  ===========
    Long wall to roof  ~4.1 kN      ~1.7 N*m
    Short wall         ~2.2 kN      ~0.9 N*m
    Tube to roof       2249 N       9.81 N*m
    Stud to roof       942 N        4.40 N*m
    =================  ===========  ===========

    The guide's table quotes 4351 N and 1.79 N*m for the long wall because it uses the full
    63.8 mm footprint edge; the generator shortens the left and right walls so the four boxes do
    not overlap at the corners, so the number derived from the geometry that is actually built
    is a little lower. The totals match, since either convention describes the same perimeter.
    """

    sec = sections(spec, nx, ny)
    yield_s = material.tensile_yield.nominal
    flex_s = material.flexural_strength.nominal
    shear_s = material.shear_strength.nominal
    wall_h = spec.height - spec.roof
    out: dict[str, WeldLimit] = {}
    for key in ("wall_front_back", "wall_left_right"):
        s = sec[key]
        sigma_cr = plate_buckling_stress(
            material.E, material.nu, s.thickness, wall_h, s.length
        )
        governing = min(sigma_cr, yield_s)
        out[key] = WeldLimit(
            key, s.area, s.modulus, s.area * governing, s.modulus * flex_s,
            "wall root, " + ("buckling" if sigma_cr < yield_s else "yield")
            + " / flexural rupture",
            force_tension_N=s.area * yield_s, critical_stress_Pa=sigma_cr,
        )
    s = sec["tube"]
    r_mean = (spec.tube_od + spec.tube_id) / 4.0
    sigma_cyl = 0.6 * material.E * s.thickness / r_mean
    governing = min(sigma_cyl, yield_s)
    out["tube"] = WeldLimit(
        "tube", s.area, s.modulus, s.area * governing, s.modulus * flex_s,
        "tube root, " + ("buckling" if sigma_cyl < yield_s else "yield")
        + " / flexural rupture",
        force_tension_N=s.area * yield_s, critical_stress_Pa=sigma_cyl,
    )
    s = sec["stud"]
    out["stud"] = WeldLimit(
        "stud", s.area, s.modulus, s.area * shear_s, s.modulus * flex_s,
        "stud root, shear / flexural rupture",
        force_tension_N=s.area * yield_s, critical_stress_Pa=math.inf,
    )
    return out


# ---------------------------------------------------------------------------
# Comparison with the simulator's current defaults
# ---------------------------------------------------------------------------


def _fmt(x: float, unit: str = "", digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "inf"
    return f"{x:.{digits}g}{(' ' + unit) if unit else ''}"


def comparison_rows(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
) -> list[tuple[str, str, str, str]]:
    """``(quantity, derived, simulator default, comment)`` rows for :func:`comparison_table`.

    The simulator's defaults are read out of :class:`~brick_builder.bricks.snap_api.SnapParams` and
    :mod:`brick_builder.bricks.spec`. Neither module is modified, imported for side effects, or
    monkey-patched: this only reads their dataclass defaults.
    """

    from brick_builder.bricks.snap_api import SnapParams

    d = SnapParams()
    est = clutch_force(spec, material, nx, ny)
    wall_est = clutch_force(spec, material, nx, ny, path="wall")
    geom = press_fit_geometry(spec, nx, ny)
    stiff = press_fit_stiffness(spec, material, nx, ny)
    crush = crush_load(spec, material, nx, ny)
    walls = wall_thresholds(spec, material)
    rows: list[tuple[str, str, str, str]] = [
        ("radial interference", _fmt(geom.interference * 1e6, "um"), "-",
         "positive = stud and tube overlap; this is the whole clutch"),
        ("wall-to-stud gap", _fmt(geom.wall_gap * 1e3, "mm"), "-",
         "the walls are NOT in contact; a real brick bridges this with ribs the generator "
         "does not model"),
        ("stud-on-tube contacts", f"{geom.n_contacts} ({geom.contacts_per_stud:.2f}/stud)", "-",
         "counted from the grid, not assumed"),
        ("series stiffness", _fmt(stiff.k_series / 1e3, "kN/m"), "-",
         f"tube {stiff.k_tube / 1e3:.0f} + stud {stiff.k_stud / 1e3:.0f} in series"),
        ("most compliant wall", _fmt(stiff.k_wall_soft / 1e3, "kN/m"), "-",
         "fifteen times softer than the loaded path; the wrong member to put the clutch in"),
        ("f_clutch per stud", _fmt(est.per_stud, "N"), _fmt(d.f_clutch, "N"),
         f"derived band {est.per_stud_lo:.2f} to {est.per_stud_hi:.2f} N; the default sits "
         "near the bottom and is more likely several times low than high"),
        ("f_clutch, wall path", _fmt(wall_est.per_stud, "N"), _fmt(d.f_clutch, "N"),
         "the first-pass estimate; close to the default, and the wrong load path"),
        ("clutch, whole brick", _fmt(est.per_brick, "N"),
         _fmt(d.f_clutch * geom.n_studs, "N"), "n_studs x per-stud (average of push-on and pull-off)"),
        ("push-on, whole brick", _fmt(est.push_on_per_brick, "N"),
         _fmt(d.f_clutch * geom.n_studs * (1 + d.beta), "N"),
         "compare with the arm's 10 to 14 N: a flat press of a 2x4 may be out of reach"),
        ("beta (push/pull)", "0.10 to 0.28", _fmt(d.beta),
         "draft of 0.5 to 1.5 deg gives a push-to-pull ratio of 1.2 to 1.8; nothing measures it"),
        ("exponent", _fmt(draft_exponent(spec, 1.0)), _fmt(d.exponent),
         "1 is right only with zero mould draft; 1 deg of draft moves it toward 2"),
        ("curve", "ramp (draft-corrected)", d.curve,
         "'plateau' has no mechanism: no straight-sided interference gives full force on entry"),
        ("break-away travel", _fmt(est.per_stud / stiff.k_series * 1e6, "um"), "-",
         "stick-slip length scale; far below anything the arm can command or observe"),
        ("wall set threshold", f"{walls['set_band_N'][0]:.1f} to {walls['set_band_N'][1]:.1f} N",
         "-", "permanent set arrives before cracking and is a functional failure"),
        ("wall crack threshold",
         f"{walls['crack_band_N'][0]:.1f} to {walls['crack_band_N'][1]:.1f} N", "-",
         "reachable only dynamically, from about 0.3 m/s with a held brick"),
        ("stud shear", _fmt(stud_shear_load(spec, material)["shear_N"], "N"), "-",
         "two orders of magnitude beyond the arm"),
        ("crush", _fmt(crush["total_N"] / 1e3, "kN"), "-",
         f"mode: {crush['walls']['wall_left_right']['mode']}; three orders beyond the arm"),
        ("density", _fmt(material.density.nominal, "kg/m^3"), _fmt(spec.density, "kg/m^3"),
         "the generated 2x4 masses 12.5 g against 12.2 g catalogue"),
        ("mu, brick on brick", _fmt(FRICTION_PAIRS['abs_on_abs'].mu.nominal),
         "0.35 (BenchConfig)", "NO SOURCE EXISTS for either number"),
        ("mu, pad on brick", _fmt(FRICTION_PAIRS['pla_on_abs'].mu.nominal),
         "1.0 (ArmConfig.pad_friction)",
         "1.0 is a modelling choice; PLA on ABS is physically more like 0.3 to 0.5"),
    ]
    return rows


def comparison_table(
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    nx: int = 2,
    ny: int = 4,
    width: int = 100,
) -> str:
    """The derived values beside the simulator's current defaults, as plain text."""

    rows = comparison_rows(spec, material, nx, ny)
    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(len(r[1]) for r in rows) + 2
    w2 = max(len(r[2]) for r in rows) + 2
    head = f"{'quantity':<{w0}}{'derived':<{w1}}{'simulator':<{w2}}comment"
    lines = [
        f"Material model vs simulator defaults - {nx}x{ny} brick, {material.name}",
        f"  conditions: {material.conditions}",
        "",
        head,
        "-" * min(width, len(head) + 30),
    ]
    for name, derived, default, comment in rows:
        pad = " " * (w0 + w1 + w2)
        wrapped = _wrap(comment, width - len(pad))
        lines.append(f"{name:<{w0}}{derived:<{w1}}{default:<{w2}}{wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"{pad}{extra}")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    if width < 20:
        width = 20
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    out.append(cur)
    return out or [""]


def print_comparison(*args, **kwargs) -> None:
    """Print :func:`comparison_table` to stdout."""

    print(comparison_table(*args, **kwargs))


def unsourced_properties(materials: Iterable[Material] = (ABS_BRICK_23C, PLA_PRINTED)) -> list:
    """Every property in the database with no source behind it, as ``(material, field, Prop)``.

    Worth calling in a report. The list is short and it is the honest health check on the whole
    model: if one of these moves, so does everything derived from it.
    """

    out = []
    for m in materials:
        for key, value in vars(m).items():
            if isinstance(value, Prop) and value.confidence in ("assumed", "unsourced"):
                out.append((m.name, key, value))
    for key, pair in FRICTION_PAIRS.items():
        if pair.mu.confidence in ("assumed", "unsourced"):
            out.append((f"friction pair {key}", "mu", pair.mu))
    return out


if __name__ == "__main__":  # pragma: no cover - a convenience, not an entry point
    print_comparison()
    print()
    print("Unsourced or assumed properties:")
    for owner, field_name, prop in unsourced_properties():
        print(f"  {owner:<40} {field_name:<24} {prop.nominal!r} {prop.unit}")
    print()
    print(f"LEGO System check: interference = "
          f"{press_fit_geometry(LEGO_SYSTEM, 2, 2).interference * 1e6:.1f} um")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_force_vs_depth(out_path: str, spec: BrickSpec = DUPLO, nx: int = 2, ny: int = 4,
                          material: Material = ABS_BRICK_23C) -> dict:
    """The derived force-against-depth curve against the simulator's three curve families.

    The derived curve is the mechanics: friction is the coefficient times the normal force
    integrated over the engaged length, and with mould draft the local interference grows with
    depth, so the force rises as roughly depth squared rather than linearly. The three families
    from ``snap_api.engagement`` are drawn on the same axes at the simulator's own ``f_clutch``
    of 1.0 N per stud, which is what makes the gap between the two visible: it is a factor in
    the height of the curve and a change in its shape.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.bricks.snap_api import SnapParams, engagement

    depth = np.linspace(0.0, spec.stud_h, 400)
    est = clutch_force(spec, material, nx, ny)
    derived_1deg = force_vs_depth(depth, spec, material, nx, ny, draft_deg=1.0)
    derived_0deg = force_vs_depth(depth, spec, material, nx, ny, draft_deg=0.0)
    n = est.n_studs

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    colours = {"ramp": "#1f6fb2", "plateau": "#7a5195", "detent": "#c1272d"}

    for ax, (title, scale) in zip(
        axes,
        (("Derived curve against the simulator's shapes, as configured", 1.0),
         ("Same, with the simulator scaled to the derived clutch", est.per_stud)),
    ):
        for curve in ("ramp", "plateau", "detent"):
            params = SnapParams(curve=curve)
            g = engagement(depth, params, spec.stud_h)
            force = n * scale * params.f_clutch * (1.0 + params.beta) * g
            ax.plot(depth * 1e3, force, color=colours[curve], lw=1.6,
                    label=f"simulator, {curve}")
        ax.plot(depth * 1e3, derived_1deg, color="#111111", lw=2.6,
                label=f"derived, 1 deg draft (exponent {draft_exponent(spec, 1.0):.2f})")
        ax.plot(depth * 1e3, derived_0deg, color="#111111", lw=1.4, ls="--",
                label="derived, no draft (exponent 1)")
        ax.axhspan(*sorted((min(ARM_PRESS_FORCE_N), max(ARM_PRESS_FORCE_N))),
                   color="#d94801", alpha=0.12, zorder=0)
        ax.text(0.05, max(ARM_PRESS_FORCE_N), "  what the arm can press (10 to 14 N)",
                color="#8c3d00", va="bottom", fontsize=8)
        ax.set_xlabel("insertion depth [mm]")
        ax.set_ylabel("push-on force, whole brick [N]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlim(0.0, spec.stud_h * 1e3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"{nx}x{ny} DUPLO press-on force: derived from ABS at 23 C vs the simulator's families",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return {
        "derived_peak_N": float(derived_1deg[-1]),
        "simulator_peak_N": float(n * 1.0 * 1.15),
        "exponent": draft_exponent(spec, 1.0),
        "f_clutch_derived_N": est.per_stud,
    }


# ---------------------------------------------------------------------------
# Can the stock arm seat a brick?
# ---------------------------------------------------------------------------


def detent_peak_gain(spec: BrickSpec = DUPLO) -> float:
    """Peak of the ``"detent"`` engagement curve, as a multiple of the seated holding force.

    The other two families peak at 1.0 - a stud is hardest to push at full seat ("ramp") or
    from the moment it grips ("plateau"). The detent family has an entry bump, so its peak is
    higher than the force the same stud ends up holding with, and it is the bump, not the
    seated force, that a press has to get past.
    """

    from brick_builder.bricks.snap_api import SnapParams, engagement

    depth = np.linspace(0.0, spec.stud_h, 2001)
    return float(engagement(depth, SnapParams(curve="detent"), spec.stud_h).max())


def push_on_force(
    clutch_per_stud: float,
    n_studs: int,
    beta: float = 0.15,
    shape: Literal["ramp", "plateau", "detent"] = "ramp",
    tilted: bool = False,
    row_studs: int = 2,
    spec: BrickSpec = DUPLO,
) -> float:
    """Force [N] a press has to reach to seat a brick, from the clutch force per stud.

    ``clutch_per_stud`` is the **clutch force per stud** -- the average of push-on and pull-off
    force per stud, the same quantity as ``SnapParams.f_clutch`` and this file's own
    :attr:`ClutchEstimate.per_stud` -- not either one alone. Pushing on is harder than pulling
    off, because mould draft tightens the fit going in and because static friction is higher
    than kinetic: the model's ``beta`` makes ``push = clutch * (1+b)`` and
    ``pull = clutch * (1-b)`` per stud, a push-to-pull ratio of ``(1+b)/(1-b)``, 1.35 at the
    default beta of 0.15. This function returns the push-on side only, ``n * clutch_per_stud *
    (1+b)`` (times the shape's peak gain), which is exactly what
    ``SnapParams``/``snap_api.engagement()`` produce at their peak and what
    ``brick_builder.bricks.bench.PressBench`` measures directly -- see
    ``tests/bricks/test_materials.py`` for the pinned agreement, straight and tilted, 2x2 and
    2x4.

    ``tilted`` is the tilted press: the tool tips the brick so one row of ``row_studs`` studs
    enters at a time instead of all of them, and only the studs that are actually sliding
    resist. On a 2x4 that is two studs instead of eight, so the force needed drops by four; on
    a 2x2 it drops by two. What it does **not** do is change the force *per stud*, so it buys a
    factor and not a reprieve.

    The shape matters only through where its peak is. ``"ramp"`` and ``"plateau"`` both peak at
    the seated value; ``"detent"`` peaks at its entry bump, about 1.5x higher, and that peak is
    what has to be got past whether the press is straight or tilted.
    """

    gain = detent_peak_gain(spec) if shape == "detent" else 1.0
    n = min(row_studs, n_studs) if tilted else n_studs
    return n * clutch_per_stud * gain * (1.0 + beta)


def arm_press_force_band(
    x_range: tuple[float, float] = (0.12, 0.24),
    y_range: tuple[float, float] = (-0.06, 0.06),
    z: float = 0.02,
    nx: int = 7,
    ny: int = 5,
    vins: tuple[float, ...] = (5.0, 7.4, 12.0),
) -> dict:
    """Static downward force [N] the arm has over a build-plate patch, per supply and torque law.

    Read-only: this imports :mod:`brick_builder.robot.force_map` and calls it, and changes nothing.
    The two torque limits are the guide's section 1.3 disagreement - the Feetech datasheet's
    1.9 N*m at 7.4 V against the 3.41 N*m the identified servo law reaches at full duty - and
    neither has been measured on the real arm, so every force here comes in a pair.

    Returns ``{"datasheet": {vin: (lo, hi)}, "identified": {...}, ...}`` with ``lo``/``hi`` the
    range over the reachable cells of the patch.
    """

    import mujoco

    from brick_builder.robot.force_map import press_force_map
    from brick_builder.robot.kinematics import ArmKinematics
    from brick_builder.robot.patch_so101 import NOMINAL_VIN, ArmConfig, attach_arm, servo_stall_torque

    world = mujoco.MjSpec()
    attach_arm(world)
    model = world.compile()
    kin = ArmKinematics(model)
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)

    limits = {"datasheet": ArmConfig().torque_limit_nm,
              "identified": servo_stall_torque(NOMINAL_VIN)}
    out: dict = {"x_range": x_range, "y_range": y_range, "z": z, "torque_limit_Nm": limits}
    for name, limit in limits.items():
        result = press_force_map(model, kin, xs, ys, z=z, vin_list=vins, torque_limit_nm=limit)
        reach = result["reachable"]
        out[name] = {
            vin: (float(np.nanmin(result["force"][vin][reach])),
                  float(np.nanmax(result["force"][vin][reach])))
            for vin in vins
        }
    return out


def seating_feasibility(
    clutch_per_stud: np.ndarray | None = None,
    arm: dict | None = None,
    spec: BrickSpec = DUPLO,
) -> dict:
    """Where the push-on curves cross what the arm has: the clutch at which seating fails.

    ``hold_max_N`` in the result is the largest clutch force per stud that each supply and
    torque assumption can still seat, for a straight press and for a tilted one, on a 2x4 and
    on a 2x2. It is the single number the guide's section 7.2 measurement decides.
    """

    arm = arm_press_force_band() if arm is None else arm
    hold = np.linspace(0.3, 5.0, 48) if clutch_per_stud is None else np.asarray(clutch_per_stud)
    ratio = {}
    for nx, ny in ((2, 4), (2, 2)):
        n = nx * ny
        for tilted in (False, True):
            for shape in ("ramp", "detent"):
                key = f"{nx}x{ny}{'_tilted' if tilted else ''}_{shape}"
                ratio[key] = push_on_force(1.0, n, shape=shape, tilted=tilted, spec=spec)
    out = {"clutch_per_stud_N": hold.tolist(), "force_per_unit_hold": ratio, "arm": arm,
           "hold_max_N": {}}
    for law in ("datasheet", "identified"):
        for vin, (lo, hi) in arm[law].items():
            for key, per_unit in ratio.items():
                out["hold_max_N"].setdefault(f"{law}_{vin:g}V", {})[key] = (
                    lo / per_unit, hi / per_unit
                )
    return out


def figure_seating_feasibility(
    out_path: str,
    spec: BrickSpec = DUPLO,
    material: Material = ABS_BRICK_23C,
    arm: dict | None = None,
) -> dict:
    """Can the stock arm seat a brick? The one figure the whole project turns on.

    Three panels. The first two put the push-on force for a 2x4 and a 2x2 against the clutch
    force per stud (the average of push-on and pull-off), over the whole plausible range of
    clutch (0.3 to 5 N per stud), and draw across them every downward force the arm has: three
    supply voltages times the two stall-torque assumptions that nobody has settled. Where a
    press-on line leaves the arm's band is the clutch at which that press stops being possible.
    The third panel is the shape of the insertion curve, which decides where the peak sits and
    therefore what a tilted press is worth.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.bricks.snap_api import SnapParams, engagement

    ink, muted = "#1a1a1a", "#5a5a5a"
    series = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    arm = arm_press_force_band() if arm is None else arm
    hold = np.linspace(0.3, 5.0, 200)
    est = clutch_force(spec, material, 2, 4)
    detent = detent_peak_gain(spec)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.8))
    for ax in axes:
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.18, color=muted, linewidth=0.7)
        ax.set_axisbelow(True)

    crossings: dict = {}
    for panel, (nx, ny) in enumerate(((2, 4), (2, 2))):
        ax = axes[panel]
        n = nx * ny
        straight = push_on_force(hold, n, shape="ramp", spec=spec)
        straight_d = push_on_force(hold, n, shape="detent", spec=spec)
        tilted = push_on_force(hold, n, shape="ramp", tilted=True, spec=spec)
        tilted_d = push_on_force(hold, n, shape="detent", tilted=True, spec=spec)
        ax.fill_between(hold, straight, straight_d, color=series[1], alpha=0.16)
        ax.plot(hold, straight, lw=2.6, color=series[1],
                label=f"straight press, all {n} studs at once")
        ax.plot(hold, straight_d, lw=1.4, ls="--", color=series[1],
                label=f"  ... with a detent curve (x{detent:.2f} at the entry bump)")
        ax.fill_between(hold, tilted, tilted_d, color=series[2], alpha=0.16)
        ax.plot(hold, tilted, lw=2.6, color=series[2], label="tilted press, one row of 2")
        ax.plot(hold, tilted_d, lw=1.4, ls="--", color=series[2],
                label="  ... with a detent curve")

        styles = {5.0: ":", 7.4: "-", 12.0: "-."}
        label_x = {"datasheet": (1.35, "left"), "identified": (4.92, "right")}
        for law, colour in (("datasheet", series[0]), ("identified", series[3])):
            x_at, align = label_x[law]
            for vin, (lo, hi) in arm[law].items():
                ax.axhspan(lo, hi, color=colour, alpha=0.10)
                ax.plot(hold, np.full_like(hold, hi), styles[vin], lw=1.8, color=colour)
                torque = arm["torque_limit_Nm"][law]
                ax.text(x_at, hi, f"{vin:g} V, {torque:.2f} N*m ", fontsize=9.5, color=colour,
                        va="center", ha=align,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                                  alpha=0.75))
        ax.set_yscale("log")
        ax.set_ylim(1.0, 200.0)
        ax.set_xlim(0.3, 5.0)
        ax.set_xlabel("clutch force per stud, the average of push-on and pull-off  [N]", fontsize=12)
        if panel == 0:
            ax.set_ylabel("force the tool has to reach to seat the brick  [N]", fontsize=12)
        # Where the derived estimate and the simulator's placeholder sit on this axis.
        for value, label, colour in (
            (SnapParams().f_clutch, "shipped default\n(1.0, assumed)", muted),
            (est.per_stud, "first-principles\nupper scenario (mu 0.35, 86 um)", ink),
        ):
            ax.axvline(value, color=colour, ls=":", lw=1.6)
            ax.text(value + 0.06, 1.25, label, fontsize=10, color=colour)
        lo74, hi74 = arm["datasheet"][7.4]
        per_unit = push_on_force(1.0, n, shape="ramp", spec=spec)
        crossings[f"{nx}x{ny}"] = {
            "straight_7.4V_datasheet_N_per_stud": (lo74 / per_unit, hi74 / per_unit),
            "tilted_7.4V_datasheet_N_per_stud": (
                lo74 / push_on_force(1.0, n, shape="ramp", tilted=True, spec=spec),
                hi74 / push_on_force(1.0, n, shape="ramp", tilted=True, spec=spec),
            ),
        }
        ax.set_title(
            f"{nx}x{ny}: a straight press runs out at "
            f"{hi74 / per_unit:.1f} N per stud\n(7.4 V, datasheet torque, best cell of the "
            f"build plate)", fontsize=13, color=ink)
        ax.legend(fontsize=9.5, loc="upper left")

    ax = axes[2]
    depth = np.linspace(0.0, spec.stud_h, 500)
    for k, curve in enumerate(("ramp", "plateau", "detent")):
        g = engagement(depth, SnapParams(curve=curve), spec.stud_h)
        ax.plot(depth * 1e3, g, lw=2.4, color=series[k], label=f'"{curve}"')
    ax.set_xlabel("how far one stud has gone in  [mm]", fontsize=12)
    ax.set_ylabel("force per stud, as a multiple of its seated holding force", fontsize=12)
    ax.set_title("The three shapes nobody has measured\n"
                 "the peak is what a press has to beat", fontsize=13, color=ink)
    ax.legend(fontsize=11, loc="lower right")
    ax.text(0.15, 1.92,
            "A tilted press beats a straight press by the number of\n"
            "studs it keeps out of the fight - four, on a 2x4 - whichever\n"
            "of these three shapes is right. The detent bump raises\n"
            f"every peak by {detent:.2f}, so it is the shape that makes even\n"
            "a tilted press harder than the clutch force suggests.",
            fontsize=10.5, color=ink, va="top")
    ax.set_ylim(0.0, 1.95)

    stiff = press_fit_stiffness(spec, material, 2, 4)
    geom = press_fit_geometry(spec, 2, 4)
    per_mu = stiff.k_series * geom.contacts_per_stud
    fig.text(
        0.5, 0.035,
        "What the left two panels' x axis is made of: clutch force per stud = "
        "mu x k x interference x contacts per stud, with "
        f"k = {stiff.k_series / 1e3:.0f} kN/m (tube and stud rings in series) and "
        f"{geom.contacts_per_stud:.1f} contacts per stud.\n"
        f"That is {per_mu * geom.interference:.1f} x mu at the nominal "
        f"{geom.interference * 1e6:.0f} um of interference and "
        f"{per_mu * 126e-6:.1f} x mu at the caliper figure's 126 um, so friction anywhere in "
        f"0.2 to 0.5 covers {per_mu * geom.interference * 0.2:.1f} to "
        f"{per_mu * 126e-6 * 0.5:.1f} N per stud - the whole width of those panels.",
        fontsize=11.0, color=ink, ha="center", va="bottom")

    fig.suptitle("Can the stock SO-101 seat a DUPLO brick? Push-on force against clutch, "
                 "with every force the arm has", fontsize=16, color=ink)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)

    return {
        "arm": arm,
        "detent_peak_gain": detent,
        "derived_per_stud_N": est.per_stud,
        "derived_band_N": (est.per_stud_lo, est.per_stud_hi),
        "push_on_2x4_at_derived_N": push_on_force(est.per_stud, 8, spec=spec),
        "push_on_2x2_at_derived_N": push_on_force(est.per_stud, 4, spec=spec),
        "tilted_2x4_at_derived_N": push_on_force(est.per_stud, 8, tilted=True, spec=spec),
        "crossings": crossings,
    }
