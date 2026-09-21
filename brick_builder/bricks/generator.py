"""Parametric DUPLO/LEGO brick generator.

Builds a brick or a baseplate as a MuJoCo body inside an in-memory `mujoco.MjSpec`, with two
separate geometry sets:

- **Collision** (geom group `COLLISION_GROUP`): a handful of convex primitives only - a roof
  box, four wall boxes, solid tube cylinders, and stud cylinders shrunk by `spec.clearance` -
  never a single mesh hull, because a hull of the whole brick would fill in the hollow
  underside and make it geometrically impossible for a stud to ever enter (see
  `docs/guide/02-simulator-and-bricks.md`).
- **Visual** (geom group `VISUAL_GROUP`, `contype=0 conaffinity=0` so it never collides): the
  same layout at nominal (non-shrunk) size, for a normal-looking render.

Brick frame convention: the body origin is the centre of the brick's bottom face, +z up. Stud
`(i, j)` (i in [0, nx), j in [0, ny)) sits at local
`x = (i - (nx - 1) / 2) * pitch`, `y = (j - (ny - 1) / 2) * pitch`, on top of the roof.
Every grid position has both a stud site (top of the stud) and a socket site (on the bottom
face, directly below that stud) - the "socket" is not a separate solid shape, it is the open
column of air a stud from the layer below rises through; docstrings below explain why that is
enough to reproduce a real insertion.

Why the geometry permits insertion (worked out from the numbers, not asserted):
walls and tubes both run the full interior height, from the bottom face (z=0) up to the roof's
underside (z = height - roof), and are positioned away from every stud/socket column, so a
correctly-aligned stud from below rises freely through open space with no solid brick material
in its path. A brick lowered with **no** stud grid underneath (a bare table) therefore rests
with its bottom face flush at z=0, by construction. A brick lowered onto a baseplate:
- **aligned**: nothing blocks the descent (shrunk studs pass the tube gaps with clearance to
  spare - see the interference numbers in the design report), so it can travel all the way down
  until the walls and tube bottoms (both at local z=0) meet the plate's flat top surface, i.e.
  fully flush - which happens exactly one `stud_h` below where it would otherwise stop, because
  a stud's top sits `stud_h` above that same flat surface.
- **misaligned enough that a tube's footprint now overlaps a stud** (verified for a 4 mm
  offset on a 2x4 brick, since the wall bottoms remain clear of any stud column at that offset):
  the tube's flat bottom face lands on the stud's flat top face, stopping the descent exactly
  `stud_h` higher than the flush position - "resting on top of the studs."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from brick_builder.bricks.spec import DUPLO, BrickSpec

VISUAL_GROUP = 2
COLLISION_GROUP = 3
# Stud and socket sites only mark positions for the connection model. MuJoCo draws site groups
# 0-2 by default, so they sit in a group no camera shows; otherwise each stud carries a dot.
MARKER_SITE_GROUP = 4

# Nominal baseplate slab thickness, from the bottom face (z=0) up to the plane the studs stand
# on. Not part of BrickSpec (a baseplate is not a brick shape), and not fixed by any acceptance
# test; the roof thickness is a reasonable stand-in for "thin structural plate."
_BASEPLATE_THICKNESS_FACTOR = 2.0  # slab thickness = spec.roof * this factor


@dataclass
class BrickHandle:
    """Everything a caller needs to refer back to a generated brick or baseplate."""

    name: str
    body: "mujoco.MjsBody"
    nx: int
    ny: int
    spec: BrickSpec
    stud_sites: list[str] = field(default_factory=list)
    socket_sites: list[str] = field(default_factory=list)
    colour: str = "white"


def _stud_xy(nx: int, ny: int, pitch: float) -> list[tuple[int, int, float, float]]:
    """Grid positions `(i, j, x, y)` for every stud/socket in an `nx` x `ny` footprint."""

    out = []
    for i in range(nx):
        x = (i - (nx - 1) / 2.0) * pitch
        for j in range(ny):
            y = (j - (ny - 1) / 2.0) * pitch
            out.append((i, j, x, y))
    return out


def _tube_xy(nx: int, ny: int, pitch: float) -> list[tuple[float, float]]:
    """Tube positions: one at every interior intersection of the stud grid.

    A 2x2 footprint has 1 interior intersection, a 2x4 has 3, a 1xN or Nx1 has 0 (there is no
    interior intersection to hang a tube from without a second row of studs).
    """

    if nx < 2 or ny < 2:
        return []
    xs = [(i - (nx - 1) / 2.0) * pitch for i in range(nx)]
    ys = [(j - (ny - 1) / 2.0) * pitch for j in range(ny)]
    interior_x = [(xs[i] + xs[i + 1]) / 2.0 for i in range(nx - 1)]
    interior_y = [(ys[j] + ys[j + 1]) / 2.0 for j in range(ny - 1)]
    return [(x, y) for x in interior_x for y in interior_y]


def _footprint(nx: int, ny: int, spec: BrickSpec) -> tuple[float, float]:
    """Outer footprint size `(fx, fy)`, slightly under `nx*pitch` x `ny*pitch`.

    The design table gives the footprint as `n*pitch - 0.2 mm` for an n-stud run: a small gap
    so neighbouring bricks are not modelled as touching exactly. `BrickSpec` has no separate
    field for that 0.2 mm gap, so this reuses `clearance` (also 0.2 mm for DUPLO) for it.
    """

    return nx * spec.pitch - spec.clearance, ny * spec.pitch - spec.clearance


def _add_stud_geom(
    body: "mujoco.MjsBody",
    world: mujoco.MjSpec,
    name: str,
    x: float,
    y: float,
    z0: float,
    radius: float,
    stud_h: float,
    chamfer: float,
    group: int,
    contype: int,
    rgba: tuple[float, float, float, float] | None = None,
) -> None:
    """One stud as collision/visual geometry: a cylinder, or a cylinder plus a tapered tip.

    `z0` is the local z of the stud's base (where it meets the roof top). `chamfer` is the
    height of the tapered tip; 0 gives a plain cylinder (a config option, per the design
    report). The taper is a small convex frustum mesh (a 16-gon), never used for the whole
    stud, so a single stud is still a union of convex primitives.

    `rgba` colours the stud; a real stud is moulded with its brick, so the visual stud must
    carry the brick's colour (left at `None`, MuJoCo paints it mid-grey).
    """

    kwargs = {}
    if rgba is not None:
        kwargs["rgba"] = rgba

    if chamfer <= 0.0 or chamfer >= stud_h:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, stud_h / 2.0, 0.0],
            pos=[x, y, z0 + stud_h / 2.0],
            group=group,
            contype=contype,
            conaffinity=contype,
            **kwargs,
        )
        return

    body_h = stud_h - chamfer
    body.add_geom(
        name=name + "_body",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, body_h / 2.0, 0.0],
        pos=[x, y, z0 + body_h / 2.0],
        group=group,
        contype=contype,
        conaffinity=contype,
        **kwargs,
    )
    n_sides = 16
    tip_radius = radius * 0.6
    verts = np.empty((2 * n_sides, 3), dtype=np.float64)
    for k in range(n_sides):
        ang = 2.0 * math.pi * k / n_sides
        verts[k] = [radius * math.cos(ang), radius * math.sin(ang), 0.0]
        verts[n_sides + k] = [tip_radius * math.cos(ang), tip_radius * math.sin(ang), chamfer]
    mesh_name = name + "_chamfer_mesh"
    world.add_mesh(name=mesh_name, uservert=verts.reshape(-1))
    body.add_geom(
        name=name + "_tip",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        pos=[x, y, z0 + body_h],
        group=group,
        contype=contype,
        conaffinity=contype,
        **kwargs,
    )


def _add_shell_geoms(
    body: "mujoco.MjsBody",
    nx: int,
    ny: int,
    spec: BrickSpec,
    group: int,
    contype: int,
    rgba: tuple[float, float, float, float] | None,
    stud_radius: float,
    chamfer: float,
    world: mujoco.MjSpec,
    name_prefix: str,
) -> None:
    """Roof + 4 walls + tubes + studs, all convex primitives, for one geom group."""

    fx, fy = _footprint(nx, ny, spec)
    wall_h = spec.height - spec.roof
    kwargs = {}
    if rgba is not None:
        kwargs["rgba"] = rgba

    # Roof: a thin box spanning the whole footprint, from the wall top to the brick top.
    body.add_geom(
        name=f"{name_prefix}_roof",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[fx / 2.0, fy / 2.0, spec.roof / 2.0],
        pos=[0.0, 0.0, spec.height - spec.roof / 2.0],
        group=group,
        contype=contype,
        conaffinity=contype,
        **kwargs,
    )

    # Four walls at the perimeter, front/back spanning the full width, left/right shortened so
    # the four boxes do not overlap at the corners.
    wz = wall_h / 2.0
    walls = [
        ("front", (0.0, (fy - spec.wall) / 2.0, wz), (fx / 2.0, spec.wall / 2.0, wz)),
        ("back", (0.0, -(fy - spec.wall) / 2.0, wz), (fx / 2.0, spec.wall / 2.0, wz)),
        ("left", (-(fx - spec.wall) / 2.0, 0.0, wz), (spec.wall / 2.0, (fy - 2 * spec.wall) / 2.0, wz)),
        ("right", ((fx - spec.wall) / 2.0, 0.0, wz), (spec.wall / 2.0, (fy - 2 * spec.wall) / 2.0, wz)),
    ]
    for side, pos, size in walls:
        body.add_geom(
            name=f"{name_prefix}_wall_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=list(pos),
            group=group,
            contype=contype,
            conaffinity=contype,
            **kwargs,
        )

    # Tubes: solid cylinders (the design report calls for solid tubes in collision; nothing
    # needs to enter a tube's own bore), spanning the same interior height as the walls.
    for k, (tx, ty) in enumerate(_tube_xy(nx, ny, spec.pitch)):
        body.add_geom(
            name=f"{name_prefix}_tube_{k}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[spec.tube_od / 2.0, wall_h / 2.0, 0.0],
            pos=[tx, ty, wz],
            group=group,
            contype=contype,
            conaffinity=contype,
            **kwargs,
        )

    # Studs, on top of the roof.
    for i, j, x, y in _stud_xy(nx, ny, spec.pitch):
        _add_stud_geom(
            body,
            world,
            name=f"{name_prefix}_stud_{i}_{j}",
            x=x,
            y=y,
            z0=spec.height,
            radius=stud_radius,
            stud_h=spec.stud_h,
            chamfer=chamfer,
            group=group,
            contype=contype,
            rgba=rgba,
        )


def _mass_properties(nx: int, ny: int, spec: BrickSpec) -> tuple[float, float, tuple[float, float, float]]:
    """Mass, centre-of-mass z (measured from the bottom face), and diagonal inertia (kg*m^2).

    Computed by summing the analytic mass and inertia of every collision primitive (the same
    shapes `_add_shell_geoms` places for collision: 4 walls, 1 roof, N tubes, N studs) about
    the brick's own axes, then shifting each to the brick's overall centre of mass with the
    parallel-axis theorem. The brick is symmetric enough (grid of studs, ring of walls, ring of
    tubes) that the off-diagonal terms cancel and the x/y centre of mass sits on the brick's
    central axis, so a diagonal inertia tensor in body axes is exact, not an approximation of
    a general tensor.
    """

    fx, fy = _footprint(nx, ny, spec)
    wall_h = spec.height - spec.roof
    stud_radius = spec.stud_d / 2.0 - spec.clearance

    parts: list[tuple[float, tuple[float, float, float], tuple[float, float, float]]] = []
    # Each entry: (mass, (x, y, z) centre, (Ixx, Iyy, Izz) about its own centre, own axes).

    def box(mass: float, c: tuple[float, float, float], lx: float, ly: float, lz: float) -> None:
        ixx = mass / 12.0 * (ly**2 + lz**2)
        iyy = mass / 12.0 * (lx**2 + lz**2)
        izz = mass / 12.0 * (lx**2 + ly**2)
        parts.append((mass, c, (ixx, iyy, izz)))

    def tube(mass: float, c: tuple[float, float, float], r_out: float, r_in: float, length: float) -> None:
        izz = mass / 2.0 * (r_out**2 + r_in**2)
        ixx = mass / 12.0 * (3.0 * (r_out**2 + r_in**2) + length**2)
        parts.append((mass, c, (ixx, ixx, izz)))

    # Roof.
    roof_vol = fx * fy * spec.roof
    box(roof_vol * spec.density, (0.0, 0.0, spec.height - spec.roof / 2.0), fx, fy, spec.roof)

    # Walls.
    wz = wall_h / 2.0
    wall_defs = [
        ((0.0, (fy - spec.wall) / 2.0, wz), (fx, spec.wall, wall_h)),
        ((0.0, -(fy - spec.wall) / 2.0, wz), (fx, spec.wall, wall_h)),
        ((-(fx - spec.wall) / 2.0, 0.0, wz), (spec.wall, fy - 2 * spec.wall, wall_h)),
        (((fx - spec.wall) / 2.0, 0.0, wz), (spec.wall, fy - 2 * spec.wall, wall_h)),
    ]
    for c, (lx, ly, lz) in wall_defs:
        box(lx * ly * lz * spec.density, c, lx, ly, lz)

    # Tubes: hollow (real tubes are hollow; only the collision geom is modelled solid).
    for tx, ty in _tube_xy(nx, ny, spec.pitch):
        r_out, r_in = spec.tube_od / 2.0, spec.tube_id / 2.0
        vol = math.pi * (r_out**2 - r_in**2) * wall_h
        tube(vol * spec.density, (tx, ty, wz), r_out, r_in, wall_h)

    # Studs: hollow.
    for _, _, x, y in _stud_xy(nx, ny, spec.pitch):
        r_out, r_in = stud_radius, spec.stud_inner_d / 2.0
        r_in = min(r_in, r_out * 0.99)  # guard degenerate specs (e.g. LEGO_SYSTEM's solid stud)
        vol = math.pi * (r_out**2 - r_in**2) * spec.stud_h
        tube(vol * spec.density, (x, y, spec.height + spec.stud_h / 2.0), r_out, r_in, spec.stud_h)

    total_mass = sum(m for m, _, _ in parts)
    com_z = sum(m * c[2] for m, c, _ in parts) / total_mass

    ixx = iyy = izz = 0.0
    for m, (x, y, z), (pixx, piyy, pizz) in parts:
        dz = z - com_z
        ixx += pixx + m * (y**2 + dz**2)
        iyy += piyy + m * (x**2 + dz**2)
        izz += pizz + m * (x**2 + y**2)
    return total_mass, com_z, (ixx, iyy, izz)


def add_brick(
    world: mujoco.MjSpec,
    name: str,
    nx: int,
    ny: int,
    colour: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    spec: BrickSpec = DUPLO,
    free: bool = True,
    chamfer: bool = False,
    parent: "mujoco.MjsBody | None" = None,
) -> BrickHandle:
    """Add an `nx` x `ny` brick body to `world` and return a handle to it.

    `free=True` (the default) gives the brick a 6-DoF free joint. `free=False` adds no joint
    at all, leaving the body fixed at `pos`/`quat` unless the caller adds its own joint to
    `handle.body` before compiling - e.g. a single slide joint plus a position actuator, for a
    kinematic lowering test.

    `parent` puts the brick body under another body instead of directly under the worldbody,
    so a test rig can carry the brick on its own joint chain (`bench.py` does exactly that).
    `pos`/`quat` are then relative to that parent. Meshes and materials still go into `world`,
    which is why both are passed around separately.

    `chamfer` selects a tapered stud tip (from `spec.chamfer`) over a plain cylinder; it
    defaults to off because a plain cylinder is the simpler, more robust choice for collision
    and the tapered tip only helps a later capture/self-centring stage (WP-SNAP), not the
    insertion geometry itself.
    """

    from brick_builder.bricks.spec import PALETTE  # local import: keep spec.py free of generator deps

    if nx < 1 or ny < 1:
        raise ValueError(f"brick footprint must be at least 1x1, got {nx}x{ny}")

    attach_to = world.worldbody if parent is None else parent
    body = attach_to.add_body(name=name, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint(name=f"{name}_free")

    stud_radius = spec.stud_d / 2.0 - spec.clearance
    chamfer_h = spec.chamfer if chamfer else 0.0

    _add_shell_geoms(
        body, nx, ny, spec,
        group=COLLISION_GROUP, contype=1, rgba=None,
        stud_radius=stud_radius, chamfer=chamfer_h, world=world, name_prefix=f"{name}_col",
    )
    rgba = tuple(PALETTE.get(colour, PALETTE["white"])) + (1.0,)
    _add_shell_geoms(
        body, nx, ny, spec,
        group=VISUAL_GROUP, contype=0, rgba=rgba,
        stud_radius=spec.stud_d / 2.0, chamfer=0.0, world=world, name_prefix=f"{name}_vis",
    )

    mass, com_z, (ixx, iyy, izz) = _mass_properties(nx, ny, spec)
    body.mass = mass
    body.inertia = [ixx, iyy, izz]
    body.ipos = [0.0, 0.0, com_z]
    body.explicitinertial = True

    stud_sites, socket_sites = [], []
    for i, j, x, y in _stud_xy(nx, ny, spec.pitch):
        stud_name = f"{name}_stud_{i}_{j}"
        socket_name = f"{name}_socket_{i}_{j}"
        body.add_site(
            name=stud_name, pos=[x, y, spec.height + spec.stud_h],
            size=[0.0005, 0.0005, 0.0005], group=MARKER_SITE_GROUP,
        )
        body.add_site(
            name=socket_name, pos=[x, y, 0.0],
            size=[0.0005, 0.0005, 0.0005], group=MARKER_SITE_GROUP,
        )
        stud_sites.append(stud_name)
        socket_sites.append(socket_name)

    return BrickHandle(
        name=name, body=body, nx=nx, ny=ny, spec=spec,
        stud_sites=stud_sites, socket_sites=socket_sites, colour=colour,
    )


def add_baseplate(
    world: mujoco.MjSpec,
    name: str,
    nx: int,
    ny: int,
    pos: tuple[float, float, float],
    spec: BrickSpec = DUPLO,
    colour: str = "green",
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> BrickHandle:
    """Add a baseplate: a flat slab with studs on top, welded to the world (no joint).

    `pos` is the bottom face of the slab, matching `add_brick`'s convention. There is no
    hollow underside and no tubes - a baseplate is never inserted onto anything from below in
    this model - so `socket_sites` is empty.

    `quat` orients the whole slab; the bench uses it to turn a plate upside down so a brick
    can hang underneath it, which is the only way to test that a connection holds against
    gravity rather than resting on it.
    """

    from brick_builder.bricks.spec import PALETTE

    if nx < 1 or ny < 1:
        raise ValueError(f"baseplate footprint must be at least 1x1, got {nx}x{ny}")

    body = world.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    fx, fy = _footprint(nx, ny, spec)
    slab_h = spec.roof * _BASEPLATE_THICKNESS_FACTOR
    stud_radius = spec.stud_d / 2.0 - spec.clearance

    body.add_geom(
        name=f"{name}_col_slab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[fx / 2.0, fy / 2.0, slab_h / 2.0],
        pos=[0.0, 0.0, slab_h / 2.0],
        group=COLLISION_GROUP, contype=1, conaffinity=1,
    )
    rgba = tuple(PALETTE.get(colour, PALETTE["green"])) + (1.0,)
    body.add_geom(
        name=f"{name}_vis_slab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[fx / 2.0, fy / 2.0, slab_h / 2.0],
        pos=[0.0, 0.0, slab_h / 2.0],
        group=VISUAL_GROUP, contype=0, conaffinity=0, rgba=rgba,
    )

    stud_sites = []
    for i, j, x, y in _stud_xy(nx, ny, spec.pitch):
        _add_stud_geom(
            body, world, name=f"{name}_col_stud_{i}_{j}", x=x, y=y, z0=slab_h,
            radius=stud_radius, stud_h=spec.stud_h, chamfer=0.0,
            group=COLLISION_GROUP, contype=1,
        )
        _add_stud_geom(
            body, world, name=f"{name}_vis_stud_{i}_{j}", x=x, y=y, z0=slab_h,
            radius=spec.stud_d / 2.0, stud_h=spec.stud_h, chamfer=0.0,
            group=VISUAL_GROUP, contype=0, rgba=rgba,
        )
        stud_name = f"{name}_stud_{i}_{j}"
        body.add_site(
            name=stud_name, pos=[x, y, slab_h + spec.stud_h],
            size=[0.0005, 0.0005, 0.0005], group=MARKER_SITE_GROUP,
        )
        stud_sites.append(stud_name)

    # A slab is dense enough, and always welded, that its mass/inertia is not load-bearing
    # physics; give it a plausible explicit mass anyway so the compiled model has no
    # unspecified-density warnings if the compiler is ever asked to check it as free.
    vol = fx * fy * slab_h
    body.mass = vol * spec.density
    box_i = body.mass / 12.0
    body.inertia = [
        box_i * (fy**2 + slab_h**2),
        box_i * (fx**2 + slab_h**2),
        box_i * (fx**2 + fy**2),
    ]
    body.ipos = [0.0, 0.0, slab_h / 2.0]
    body.explicitinertial = True

    return BrickHandle(
        name=name, body=body, nx=nx, ny=ny, spec=spec,
        stud_sites=stud_sites, socket_sites=[], colour=colour,
    )
