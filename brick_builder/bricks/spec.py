"""Physical parameters of a brick family (DUPLO, LEGO System, ...).

All lengths are metres, density is kg/m^3, following the SI convention in
`docs/guide/02-simulator-and-bricks.md`.

The DUPLO numbers come from the dimension table in
`docs/guide/02-simulator-and-bricks.md`, which cross-checks three
independent sources (a brick-generator project's parameters, a caliper measurement, and a
parametric CAD tool). See that table for the measured ranges each nominal value sits inside.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrickSpec:
    """One brick family's geometry and material, in metres and kg/m^3.

    Fields describe a single stud cell and the shell built from it:

    - pitch: centre-to-centre spacing between adjacent studs.
    - height: outer height of the brick body, bottom face to top (roof) face, NOT
      including the studs, which project an additional `stud_h` above that.
    - stud_d, stud_h, stud_inner_d: outer diameter, height, and hollow inner diameter of
      one stud (a real stud is itself a short hollow tube, open at the top).
    - wall: thickness of the four outer side walls.
    - tube_od, tube_id: outer/inner diameter of the hollow support tubes hanging from the
      underside of the roof at interior grid intersections (a 2x2 footprint has 1, a 2x4
      has 3 - one per interior intersection of the stud grid).
    - roof: thickness of the top plate the studs stand on and the tubes hang from.
    - clearance: radius removed from every stud so it can physically enter a socket or
      slide past a tube without the parts interfering geometrically. The actual snap-fit
      holding force is a separate model (WP-SNAP); this field only makes room.
    - chamfer: height of an optional tapered lead-in at the tip of each stud, which helps
      self-centre an insertion. A brick generated with `chamfer=0` (or the generator's
      `taper=False`) gets a plain cylindrical stud instead.
    - density: bulk material density used to compute mass and inertia from the shell
      volume (PLAN section 6.3: a 2x4 DUPLO brick built this way should mass about 12.2 g).
    """

    pitch: float
    height: float
    stud_d: float
    stud_h: float
    stud_inner_d: float
    wall: float
    tube_od: float
    tube_id: float
    roof: float
    clearance: float
    chamfer: float
    density: float


# DUPLO: the dimension table in docs/guide/02-simulator-and-bricks.md and
# its defaults. Every field here is one of that table's rows.
DUPLO = BrickSpec(
    pitch=0.016,
    height=0.0192,
    stud_d=0.0094,
    stud_h=0.0045,
    stud_inner_d=0.0065,
    wall=0.00155,
    tube_od=0.0134,
    tube_id=0.0107,
    roof=0.0015,
    clearance=0.0002,
    chamfer=0.0005,
    density=1050.0,
)

# LEGO System ("regular" LEGO): a configuration variant, not a new code path. The generator
# treats it exactly like DUPLO. pitch/height/stud_d/stud_h/wall/tube_od come from the design
# report's inline example (section 2, "LEGO System as a config change"); the remaining fields
# are not in that report and are estimated here by halving the DUPLO shell proportions (LEGO
# System pitch is exactly half of DUPLO's) and by the well-known fact that most System studs
# are solid, not hollow. These are placeholders for calibration, not measured values, and no
# acceptance test in WP-BRICKS checks them numerically.
LEGO_SYSTEM = BrickSpec(
    pitch=0.008,
    height=0.0096,
    stud_d=0.0048,
    stud_h=0.0017,
    stud_inner_d=0.0,  # most System studs are solid; 0 disables the stud's hollow bore
    wall=0.0012,
    tube_od=0.00651,
    tube_id=0.00516,
    roof=0.0009,
    clearance=0.0001,
    chamfer=0.0003,
    density=1050.0,
)

# Approximate DUPLO palette colours, sRGB in [0, 1], alpha omitted (the generator appends 1.0).
PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.77, 0.09, 0.09),
    "blue": (0.00, 0.32, 0.62),
    "yellow": (0.96, 0.78, 0.13),
    "green": (0.00, 0.46, 0.22),
    "orange": (0.90, 0.42, 0.09),
    "white": (0.95, 0.95, 0.95),
}
