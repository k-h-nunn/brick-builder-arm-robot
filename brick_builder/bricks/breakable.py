"""The breakable brick: a DUPLO brick built from pieces that can actually come apart.

This is the validation half of the guide's section 6, and it is deliberately *not* for
training. A 2x4 rebuilt as sixteen welded pieces steps in 716 us against 105 us for the
monolithic brick in the same scene - a **6.8x slowdown for one brick** - so a four-brick build
would be unusable. It exists for three jobs:

1. validating that the cheap estimate in :mod:`brick_builder.bricks.damage` puts its thresholds
   where pieces actually come apart,
2. producing the demonstration of a brick failing, which is what makes the damage model
   credible to anyone who has not read the arithmetic,
3. answering specific "what if the arm did X at full speed" questions.

How it is built
---------------

The brick becomes a roof body plus one body per wall, per tube and per stud, each with its own
free joint, joined to the roof by a ``weld`` equality. Every piece body has its **origin at the
brick's own origin** (the centre of the bottom face), with the geometry placed at its
brick-local position inside the piece. That is not an accident of convenience: it makes every
weld's relative pose the identity, so the only thing that has to be got right is the anchor,
and the anchor is put at the joint's own root - the point where the wall meets the roof, or
where the stud meets the roof. The weld then reports its constraint moment about that root,
which is exactly the root bending moment the allowable is written against.

Pieces of the same brick are excluded from colliding with each other, because they touch by
construction: a stud's base sits exactly on the roof's top face and a wall's top sits exactly
under the roof.

The stud and socket sites stay on the roof body with the names and positions the generator
gives them, so :class:`~brick_builder.bricks.snap.ColletSnap` binds to a breakable brick unchanged
and the press-fit model still works on an intact one. The limitation that comes with that is
stated plainly: once a stud piece has been released, its *site* is still on the roof, so the
snap model will go on treating that stud as attached. Making a broken-off stud lose its
connection needs the sites to move onto the piece bodies, which is a change in
``generator.py``/``snap_api.py`` and is in the integration plan rather than done here.

Limits
------

Every limit comes from :func:`brick_builder.bricks.materials.weld_limits`, which derives it from the
real cross-section of the member and the ABS allowables. Walls and tubes in compression are
limited by buckling rather than by yield - the DUPLO 2x4 left/right wall goes critical at
19.5 MPa against a 44 MPa yield, and the shorter front/back pair at 28.4 MPa - which is the
correction that turns a first-pass crush estimate of 19 kN into the **13.2 kN**
:func:`~brick_builder.bricks.materials.crush_load` computes. (An earlier draft of the guide said
12.2 kN, from applying the softest wall's critical stress to every wall; the per-wall-pair
figure is the one the code uses and the one the guide now quotes.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from brick_builder.bricks import materials as mat
from brick_builder.bricks.generator import (
    COLLISION_GROUP,
    MARKER_SITE_GROUP,
    VISUAL_GROUP,
    BrickHandle,
    _stud_xy,
    _tube_xy,
)
from brick_builder.bricks.snap_api import equality_forces
from brick_builder.bricks.spec import DUPLO, PALETTE, BrickSpec

#: Colour a piece is repainted when its weld lets go, so a fracture is visible in a render.
FRACTURED_RGBA: tuple[float, float, float, float] = (0.25, 0.25, 0.28, 1.0)

_BASEPLATE_THICKNESS_FACTOR = 2.0


@dataclass
class Piece:
    """One separable part of a breakable brick."""

    name: str          # body name
    kind: str          # "wall" | "tube" | "stud"
    section: str       # key into materials.weld_limits()
    anchor: tuple[float, float, float]   # joint root, in brick-local coordinates
    force_N: float
    moment_Nm: float
    geom_names: list[str] = field(default_factory=list)


@dataclass
class FractureEvent:
    """One weld letting go: when, which piece, and under what load."""

    time: float
    brick: str
    piece: str
    kind: str
    force_N: float
    moment_Nm: float
    force_limit_N: float
    moment_limit_Nm: float
    utilisation: float


@dataclass
class BreakableHandle:
    """What :func:`add_breakable_brick` hands back.

    ``handle`` is an ordinary :class:`~brick_builder.bricks.generator.BrickHandle` pointing at the
    roof body and carrying the stud and socket sites, so it can be passed straight to a snap
    model or to :class:`~brick_builder.bricks.damage.DamageModel` alongside monolithic bricks.
    """

    name: str
    handle: BrickHandle
    pieces: list[Piece]
    weld_names: list[str]
    spec: BrickSpec
    nx: int
    ny: int


def _box_inertia(mass: float, lx: float, ly: float, lz: float) -> tuple[float, float, float]:
    return (
        mass / 12.0 * (ly**2 + lz**2),
        mass / 12.0 * (lx**2 + lz**2),
        mass / 12.0 * (lx**2 + ly**2),
    )


def _tube_inertia(
    mass: float, r_out: float, r_in: float, length: float
) -> tuple[float, float, float]:
    izz = mass / 2.0 * (r_out**2 + r_in**2)
    ixx = mass / 12.0 * (3.0 * (r_out**2 + r_in**2) + length**2)
    return (ixx, ixx, izz)


def _set_inertial(body, mass: float, ipos, inertia) -> None:
    body.mass = max(mass, 1e-7)
    body.inertia = [max(i, 1e-11) for i in inertia]
    body.ipos = list(ipos)
    body.explicitinertial = True


def add_breakable_brick(
    world: mujoco.MjSpec,
    name: str,
    nx: int = 2,
    ny: int = 4,
    colour: str = "red",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    spec: BrickSpec = DUPLO,
    material: mat.Material = mat.ABS_BRICK_23C,
    include_studs: bool = True,
    limit_scale: float = 1.0,
    solref: tuple[float, float] = (0.004, 1.0),
) -> BreakableHandle:
    """Add a brick built from separable pieces welded to its roof.

    The result is geometrically identical to
    :func:`brick_builder.bricks.generator.add_brick` - the same roof box, the same four wall boxes,
    the same tube cylinders, the same stud cylinders, the same names, the same sites - so a
    scene can swap one for the other and every downstream consumer that works by name keeps
    working. What changes is that each of those parts is now its own body.

    ``include_studs=False`` welds the studs into the roof body instead of giving them their own
    bodies. That halves the piece count on a 2x4 (16 pieces down to 8) and roughly halves the
    cost, and it is the right choice whenever the question being asked is about walls, which is
    every question the arm can actually pose: a stud needs 942 N to shear off.

    ``limit_scale`` multiplies every weld limit at once, for randomization or for making a
    demonstration fail at a convenient load.
    """

    if nx < 1 or ny < 1:
        raise ValueError(f"brick footprint must be at least 1x1, got {nx}x{ny}")

    fx, fy = nx * spec.pitch - spec.clearance, ny * spec.pitch - spec.clearance
    wall_h = spec.height - spec.roof
    stud_radius = spec.stud_d / 2.0 - spec.clearance
    rgba = tuple(PALETTE.get(colour, PALETTE["white"])) + (1.0,)
    limits = mat.weld_limits(spec, material, nx, ny)

    def new_piece_body(piece_name: str):
        b = world.worldbody.add_body(name=piece_name, pos=list(pos), quat=list(quat))
        b.add_freejoint(name=f"{piece_name}_free")
        return b

    # -- the roof, which everything else hangs off ------------------------
    roof = world.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    roof.add_freejoint(name=f"{name}_free")
    roof_z = spec.height - spec.roof / 2.0
    for group, contype, kw in ((COLLISION_GROUP, 1, {}), (VISUAL_GROUP, 0, {"rgba": rgba})):
        tag = "col" if group == COLLISION_GROUP else "vis"
        roof.add_geom(
            name=f"{name}_{tag}_roof",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[fx / 2.0, fy / 2.0, spec.roof / 2.0],
            pos=[0.0, 0.0, roof_z],
            group=group, contype=contype, conaffinity=contype, **kw,
        )
    roof_mass = fx * fy * spec.roof * spec.density
    roof_inertia = list(_box_inertia(roof_mass, fx, fy, spec.roof))
    roof_com = [0.0, 0.0, roof_z]

    pieces: list[Piece] = []
    piece_bodies: list[str] = []

    # -- walls ------------------------------------------------------------
    wz = wall_h / 2.0
    wall_defs = [
        ("front", (0.0, (fy - spec.wall) / 2.0, wz), (fx, spec.wall, wall_h),
         "wall_front_back"),
        ("back", (0.0, -(fy - spec.wall) / 2.0, wz), (fx, spec.wall, wall_h),
         "wall_front_back"),
        ("left", (-(fx - spec.wall) / 2.0, 0.0, wz), (spec.wall, fy - 2 * spec.wall, wall_h),
         "wall_left_right"),
        ("right", ((fx - spec.wall) / 2.0, 0.0, wz), (spec.wall, fy - 2 * spec.wall, wall_h),
         "wall_left_right"),
    ]
    for side, centre, (lx, ly, lz), section in wall_defs:
        pname = f"{name}_piece_wall_{side}"
        body = new_piece_body(pname)
        geoms = []
        for group, contype, kw in ((COLLISION_GROUP, 1, {}), (VISUAL_GROUP, 0, {"rgba": rgba})):
            tag = "col" if group == COLLISION_GROUP else "vis"
            gname = f"{name}_{tag}_wall_{side}"
            body.add_geom(
                name=gname, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[lx / 2.0, ly / 2.0, lz / 2.0], pos=list(centre),
                group=group, contype=contype, conaffinity=contype, **kw,
            )
            geoms.append(gname)
        m = lx * ly * lz * spec.density
        _set_inertial(body, m, centre, _box_inertia(m, lx, ly, lz))
        lim = limits[section]
        pieces.append(Piece(
            pname, "wall", section,
            (centre[0], centre[1], wall_h),       # root: where the wall meets the roof
            lim.force_N * limit_scale, lim.moment_Nm * limit_scale, geoms,
        ))
        piece_bodies.append(pname)

    # -- tubes -------------------------------------------------------------
    r_o, r_i = spec.tube_od / 2.0, spec.tube_id / 2.0
    for k, (tx, ty) in enumerate(_tube_xy(nx, ny, spec.pitch)):
        pname = f"{name}_piece_tube_{k}"
        body = new_piece_body(pname)
        geoms = []
        for group, contype, kw in ((COLLISION_GROUP, 1, {}), (VISUAL_GROUP, 0, {"rgba": rgba})):
            tag = "col" if group == COLLISION_GROUP else "vis"
            gname = f"{name}_{tag}_tube_{k}"
            body.add_geom(
                name=gname, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r_o, wall_h / 2.0, 0.0], pos=[tx, ty, wz],
                group=group, contype=contype, conaffinity=contype, **kw,
            )
            geoms.append(gname)
        m = math.pi * (r_o**2 - r_i**2) * wall_h * spec.density
        _set_inertial(body, m, (tx, ty, wz), _tube_inertia(m, r_o, r_i, wall_h))
        lim = limits["tube"]
        pieces.append(Piece(
            pname, "tube", "tube", (tx, ty, wall_h),
            lim.force_N * limit_scale, lim.moment_Nm * limit_scale, geoms,
        ))
        piece_bodies.append(pname)

    # -- studs -------------------------------------------------------------
    stud_grid = _stud_xy(nx, ny, spec.pitch)
    s_i = min(spec.stud_inner_d / 2.0, stud_radius * 0.99)
    for i, j, x, y in stud_grid:
        col = f"{name}_col_stud_{i}_{j}"
        vis = f"{name}_vis_stud_{i}_{j}"
        host = roof
        pname = None
        if include_studs:
            pname = f"{name}_piece_stud_{i}_{j}"
            host = new_piece_body(pname)
        for group, contype, kw, gname, radius in (
            (COLLISION_GROUP, 1, {}, col, stud_radius),
            (VISUAL_GROUP, 0, {"rgba": rgba}, vis, spec.stud_d / 2.0),
        ):
            host.add_geom(
                name=gname, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[radius, spec.stud_h / 2.0, 0.0],
                pos=[x, y, spec.height + spec.stud_h / 2.0],
                group=group, contype=contype, conaffinity=contype, **kw,
            )
        m = math.pi * (stud_radius**2 - s_i**2) * spec.stud_h * spec.density
        centre = (x, y, spec.height + spec.stud_h / 2.0)
        if include_studs:
            _set_inertial(host, m, centre, _tube_inertia(m, stud_radius, s_i, spec.stud_h))
            lim = limits["stud"]
            pieces.append(Piece(
                pname, "stud", "stud", (x, y, spec.height),
                lim.force_N * limit_scale, lim.moment_Nm * limit_scale, [col, vis],
            ))
            piece_bodies.append(pname)
        else:
            # Fold the stud into the roof's own inertial properties.
            total = roof_mass + m
            new_com = [
                (roof_mass * roof_com[k] + m * centre[k]) / total for k in range(3)
            ]
            stud_i = _tube_inertia(m, stud_radius, s_i, spec.stud_h)
            for k in range(3):
                a = [q for q in range(3) if q != k]
                roof_inertia[k] += roof_mass * sum(
                    (roof_com[q] - new_com[q]) ** 2 for q in a
                )
                roof_inertia[k] += stud_i[k] + m * sum(
                    (centre[q] - new_com[q]) ** 2 for q in a
                )
            roof_mass, roof_com = total, new_com
    _set_inertial(roof, roof_mass, roof_com, roof_inertia)

    # -- sites, on the roof, exactly where the generator puts them ---------
    stud_sites, socket_sites = [], []
    for i, j, x, y in stud_grid:
        sname, kname = f"{name}_stud_{i}_{j}", f"{name}_socket_{i}_{j}"
        roof.add_site(name=sname, pos=[x, y, spec.height + spec.stud_h],
                      size=[0.0005, 0.0005, 0.0005], group=MARKER_SITE_GROUP)
        roof.add_site(name=kname, pos=[x, y, 0.0], size=[0.0005, 0.0005, 0.0005],
                      group=MARKER_SITE_GROUP)
        stud_sites.append(sname)
        socket_sites.append(kname)

    # -- welds -------------------------------------------------------------
    weld_names = []
    for piece in pieces:
        eq = world.add_equality()
        eq.name = f"weld_{piece.name}"
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = piece.name
        eq.name2 = name
        eq.active = True
        data = np.zeros(mujoco.mjNEQDATA)
        # MuJoCo's weld pins the point `data[0:3]` of body2 to the point `data[3:6]` of body1,
        # each expressed in its own body's frame, and `data[6:10]` is the relative orientation.
        # Both pieces' frames coincide with the brick frame at rest, so both points are the
        # same brick-local coordinates: the joint's own root, which is what makes the weld's
        # reported moment the root bending moment the allowable is written against.
        data[0:3] = piece.anchor      # the weld point, in the roof's frame
        data[3:6] = piece.anchor      # the same point, in the piece's frame
        data[6] = 1.0                 # relative orientation: identity quaternion
        data[10] = 1.0                # torquescale, so the moment rows read in N*m
        eq.data = data
        # A soft weld would let a piece drift before the constraint force registers, which
        # would decouple the measured load from the fracture criterion.
        eq.solimp = [0.9999, 0.9999, 0.001, 0.5, 2.0]
        eq.solref = list(solref)
        weld_names.append(eq.name)

    # -- exclusions ---------------------------------------------------------
    # Pieces of the same brick touch by construction (a stud's base sits exactly on the roof's
    # top face, a wall's top exactly under the roof), so without these every brick would
    # generate dozens of spurious internal contacts.
    all_bodies = [name] + piece_bodies
    for a in range(len(all_bodies)):
        for b in range(a + 1, len(all_bodies)):
            ex = world.add_exclude()
            ex.name = f"nocol_{all_bodies[a]}__{all_bodies[b]}"
            ex.bodyname1 = all_bodies[a]
            ex.bodyname2 = all_bodies[b]

    handle = BrickHandle(
        name=name, body=roof, nx=nx, ny=ny, spec=spec,
        stud_sites=stud_sites, socket_sites=socket_sites, colour=colour,
    )
    return BreakableHandle(
        name=name, handle=handle, pieces=pieces, weld_names=weld_names,
        spec=spec, nx=nx, ny=ny,
    )


class BreakableBricks:
    """Watches every weld and releases the ones whose transmitted load is past the allowable.

    Call :meth:`bind` once on the compiled model, then :meth:`update` once per physics step.
    The check is the standard beam-column interaction,

        F / F_limit + M / M_limit >= 1

    with ``F`` the magnitude of the weld's translational constraint force and ``M`` the
    magnitude of its moment about the joint root. Using the sum rather than the larger of the
    two is the conservative and conventional choice for a member carrying axial load and
    bending at once, which is exactly what a wall does in a jam.

    On release the weld is deactivated, the piece's geoms are repainted
    :data:`FRACTURED_RGBA`, and a :class:`FractureEvent` is recorded. If a
    :class:`~brick_builder.bricks.damage.DamageModel` was handed in, the brick is latched to
    ``broken`` there too, so the cheap estimate and the explicit model agree about what
    happened.
    """

    def __init__(
        self,
        bricks: list[BreakableHandle],
        damage=None,
        repaint: bool = True,
    ) -> None:
        self.bricks = list(bricks)
        self.damage = damage
        self.repaint = repaint
        self.events: list[FractureEvent] = []
        self._bound = False
        self._pristine_rgba: list[np.ndarray] | None = None

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Resolve weld and geom ids, and reactivate every weld."""

        self.model, self.data = model, data
        eq_ids, force_lim, moment_lim, owner, piece = [], [], [], [], []
        self._geom_ids: list[np.ndarray] = []
        pristine: list[np.ndarray] = []
        for handle in self.bricks:
            for p, wname in zip(handle.pieces, handle.weld_names):
                eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, wname)
                if eid < 0:
                    raise RuntimeError(f"weld {wname!r} is not in the compiled model")
                eq_ids.append(eid)
                force_lim.append(p.force_N)
                moment_lim.append(p.moment_Nm)
                owner.append(handle.name)
                piece.append(p)
                gids = np.asarray(
                    [
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                        for g in p.geom_names
                    ],
                    dtype=np.int32,
                )
                gids = gids[gids >= 0]
                self._geom_ids.append(gids)
                # Remember the colours from the FIRST bind only, so re-binding after a run that
                # already repainted a fractured piece restores the brick rather than freezing
                # the fractured grey in as the original.
                if self._pristine_rgba is None:
                    pristine.append(model.geom_rgba[gids].copy())
        self.eq_ids = np.asarray(eq_ids, dtype=np.int32)
        self.force_limit = np.asarray(force_lim, dtype=np.float64)
        self.moment_limit = np.asarray(moment_lim, dtype=np.float64)
        self.owner = owner
        self.pieces = piece
        self.released = np.zeros(self.eq_ids.size, dtype=bool)
        self.peak_utilisation = np.zeros(self.eq_ids.size)
        if self._pristine_rgba is None:
            self._pristine_rgba = pristine
        data.eq_active[self.eq_ids] = 1
        for gids, rgba in zip(self._geom_ids, self._pristine_rgba):
            if gids.size:
                model.geom_rgba[gids] = rgba
        self.events = []
        self._bound = True

    def weld_loads(self) -> tuple[np.ndarray, np.ndarray]:
        """``(force magnitude [N], moment magnitude [N*m])`` for every weld this step."""

        f = equality_forces(self.model, self.data, n_rows=6)[self.eq_ids]
        return np.linalg.norm(f[:, 0:3], axis=1), np.linalg.norm(f[:, 3:6], axis=1)

    def utilisation(self) -> np.ndarray:
        """``F/F_limit + M/M_limit`` for every weld; 1 is the fracture criterion."""

        force, moment = self.weld_loads()
        return force / np.maximum(self.force_limit, 1e-9) + moment / np.maximum(
            self.moment_limit, 1e-9
        )

    def update(self) -> list[FractureEvent]:
        """One pass. Returns the fractures that happened on this call (usually none)."""

        if not self._bound:
            raise RuntimeError("update() called before bind()")
        force, moment = self.weld_loads()
        util = force / np.maximum(self.force_limit, 1e-9) + moment / np.maximum(
            self.moment_limit, 1e-9
        )
        live = ~self.released
        np.maximum(self.peak_utilisation, np.where(live, util, 0.0),
                   out=self.peak_utilisation)
        hit = np.nonzero(live & (util >= 1.0))[0]
        out: list[FractureEvent] = []
        for k in hit:
            self.released[k] = True
            self.data.eq_active[self.eq_ids[k]] = 0
            gids = self._geom_ids[k]
            if self.repaint and gids.size:
                self.model.geom_rgba[gids] = FRACTURED_RGBA
            ev = FractureEvent(
                time=float(self.data.time),
                brick=self.owner[k],
                piece=self.pieces[k].name,
                kind=self.pieces[k].kind,
                force_N=float(force[k]),
                moment_Nm=float(moment[k]),
                force_limit_N=float(self.force_limit[k]),
                moment_limit_Nm=float(self.moment_limit[k]),
                utilisation=float(util[k]),
            )
            out.append(ev)
            self.events.append(ev)
            if self.damage is not None:
                self.damage.mark_broken(self.owner[k], mode="weld_release",
                                        force_N=ev.force_N)
        return out

    # -- queries -----------------------------------------------------------

    @property
    def n_released(self) -> int:
        return int(np.count_nonzero(self.released))

    def intact(self, brick: str | None = None) -> bool:
        """True while no weld of ``brick`` (or of any brick) has let go."""

        if brick is None:
            return self.n_released == 0
        return not any(
            self.released[k] for k, o in enumerate(self.owner) if o == brick
        )

    def summary(self) -> dict:
        return {
            "n_welds": int(self.eq_ids.size),
            "n_released": self.n_released,
            "peak_utilisation": float(self.peak_utilisation.max())
            if self.eq_ids.size else 0.0,
            "events": [vars(e) for e in self.events],
        }


# ---------------------------------------------------------------------------
# A rig for the validation runs
# ---------------------------------------------------------------------------


def build_crush_scene(
    nx: int = 2,
    ny: int = 4,
    spec: BrickSpec = DUPLO,
    material: mat.Material = mat.ABS_BRICK_23C,
    timestep: float = 0.002,
    gravity: bool = True,
    include_studs: bool = True,
    limit_scale: float = 1.0,
    drop_height: float = 0.0,
    ram: bool = True,
    friction: float = 0.35,
    offwidth: int = 0,
    offheight: int = 0,
    light_background: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData, BreakableHandle, BreakableBricks]:
    """A breakable brick on a floor, optionally under a virtual ram.

    The ram is a flat plate above the brick on a vertical slide with no actuator: load is
    applied by writing ``data.xfrc_applied`` on the ram body, which is the cleanest way to
    command a *force* rather than a position and makes "press with exactly 20 kN" mean what it
    says. :func:`crush` drives it.

    ``drop_height`` puts the brick that far above the floor instead, for the impact case, and
    is what the negative-control drop test uses.
    """

    world = mujoco.MjSpec()
    opt = world.option
    opt.timestep = timestep
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.iterations = 100
    opt.ls_iterations = 50
    if not gravity:
        opt.gravity = [0.0, 0.0, 0.0]
    world.default.geom.solref = [0.004, 1.0]
    world.default.geom.solimp = [0.95, 0.999, 0.0005, 0.5, 2.0]
    world.default.geom.friction = [friction, 0.005, 0.0001]
    if light_background:
        # A light background and a matte floor, so the figure prints and the fractured pieces
        # read as dark grey against something rather than against black.
        world.add_texture(
            name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.97, 0.97, 0.98], rgb2=[0.86, 0.88, 0.92], width=256, height=256,
        )
        world.add_texture(
            name="floortex", type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.92, 0.92, 0.94], rgb2=[0.84, 0.84, 0.87], width=300, height=300,
        )
        floor_material = world.add_material(name="floormat", texrepeat=[6, 6], reflectance=0.0)
        floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floortex"
        world.visual.headlight.ambient = [0.45, 0.45, 0.45]
        world.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        world.visual.headlight.specular = [0.1, 0.1, 0.1]
    if offwidth or offheight:
        # The offscreen framebuffer is fixed at compile time, so a figure that wants a
        # full-resolution still has to ask for it here rather than at render time.
        world.visual.global_.offwidth = max(offwidth, 640)
        world.visual.global_.offheight = max(offheight, 480)
    world.worldbody.add_light(name="key", pos=[0.10, -0.12, 0.18], dir=[-0.4, 0.5, -1.0])
    world.worldbody.add_light(name="fill", pos=[-0.12, 0.10, 0.14], dir=[0.5, -0.4, -1.0],
                              diffuse=[0.35, 0.35, 0.35])

    floor_kwargs = {"material": "floormat"} if light_background else {
        "rgba": [0.92, 0.92, 0.94, 1.0]
    }
    world.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.4, 0.4, 0.01],
        pos=[0.0, 0.0, 0.0], group=COLLISION_GROUP, contype=1, conaffinity=1,
        **floor_kwargs,
    )

    brick = add_breakable_brick(
        world, "brick", nx, ny, "red", pos=(0.0, 0.0, drop_height), spec=spec,
        material=material, include_studs=include_studs, limit_scale=limit_scale,
    )

    if ram:
        fx, fy = nx * spec.pitch - spec.clearance, ny * spec.pitch - spec.clearance
        top = spec.height + spec.stud_h
        body = world.worldbody.add_body(name="ram", pos=[0.0, 0.0, top + 0.0005])
        body.add_joint(name="ram_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        body.add_geom(
            name="ram_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[fx / 2.0 + 0.004, fy / 2.0 + 0.004, 0.004], pos=[0.0, 0.0, 0.004],
            group=COLLISION_GROUP, contype=1, conaffinity=1, rgba=[0.55, 0.57, 0.62, 1.0],
        )
        body.mass = 0.05
        body.inertia = [1e-5, 1e-5, 1e-5]
        body.ipos = [0.0, 0.0, 0.004]
        body.explicitinertial = True

    model = world.compile()
    data = mujoco.MjData(model)
    manager = BreakableBricks([brick])
    manager.bind(model, data)
    mujoco.mj_forward(model, data)
    return model, data, brick, manager


def crush(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    manager: BreakableBricks,
    force_N: float,
    steps: int = 400,
    ramp_steps: int = 200,
    on_step=None,
) -> dict:
    """Press the ram down with up to ``force_N`` and report what broke.

    The force is ramped over ``ramp_steps`` so the brick is loaded quasi-statically rather than
    struck; an instantaneous 20 kN at a 2 ms step is an impact, not a crush, and would tell a
    different story. ``on_step(i, data)`` is called after every step, which is how the figure
    script grabs its stills.
    """

    ram_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ram")
    if ram_body < 0:
        raise RuntimeError("this scene has no ram; build it with ram=True")
    peak = 0.0
    for i in range(steps):
        frac = min(1.0, (i + 1) / max(ramp_steps, 1))
        data.xfrc_applied[ram_body] = 0.0
        data.xfrc_applied[ram_body, 2] = -force_N * frac
        mujoco.mj_step(model, data)
        manager.update()
        peak = max(peak, float(manager.utilisation().max()))
        if on_step is not None:
            on_step(i, data)
    data.xfrc_applied[ram_body] = 0.0
    return {
        "commanded_force_N": force_N,
        "peak_utilisation": peak,
        "n_released": manager.n_released,
        "events": [vars(e) for e in manager.events],
    }


def drop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    manager: BreakableBricks,
    height: float,
    steps: int = 600,
) -> dict:
    """Drop the brick from ``height`` onto the floor and report what broke.

    This is the guide's negative control. ABS has a notched Izod impact strength of 300 J/m and
    DUPLO bricks survive being dropped by toddlers all day: **any damage model that flags an
    ordinary drop as breakage is wrong**, and that is exactly what this checks.
    """

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brick")
    mujoco.mj_resetData(model, data)
    # Every piece body was declared at the same origin, so lifting the whole assembly is one
    # offset applied to each free joint's z - the welds stay satisfied and nothing is stressed
    # at t = 0.
    for handle in manager.bricks:
        for pname in [handle.name] + [p.name for p in handle.pieces]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pname)
            adr = model.jnt_qposadr[model.body_jntadr[bid]]
            data.qpos[adr + 2] += height
    manager.bind(model, data)
    mujoco.mj_forward(model, data)
    peak_speed = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        manager.update()
        peak_speed = max(peak_speed, abs(float(data.cvel[body, 5])))
    return {
        "height_m": height,
        "impact_speed_m_s": math.sqrt(2.0 * 9.81 * height),
        "peak_speed_m_s": peak_speed,
        "peak_utilisation": float(manager.peak_utilisation.max()),
        "n_released": manager.n_released,
        "events": [vars(e) for e in manager.events],
    }


def cost_comparison(nx: int = 2, ny: int = 4, steps: int = 200) -> dict:
    """Microseconds per step for a monolithic brick against a breakable one, same scene.

    The guide measured 105.5 us against 716.2 us on this machine, a 6.8x slowdown for one
    brick, which is the reason the breakable model is for validation only.
    """

    from time import perf_counter

    from brick_builder.bricks.generator import add_baseplate, add_brick

    def timed(model, data) -> float:
        for _ in range(20):
            mujoco.mj_step(model, data)
        t0 = perf_counter()
        for _ in range(steps):
            mujoco.mj_step(model, data)
        return (perf_counter() - t0) * 1e6 / steps

    out = {}
    for label in ("monolithic", "breakable"):
        world = mujoco.MjSpec()
        world.option.timestep = 0.002
        world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        world.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        world.default.geom.solref = [0.004, 1.0]
        world.default.geom.solimp = [0.95, 0.999, 0.0005, 0.5, 2.0]
        add_baseplate(world, "plate", 4, 6, pos=(0.0, 0.0, 0.0))
        z = DUPLO.roof * _BASEPLATE_THICKNESS_FACTOR
        if label == "monolithic":
            add_brick(world, "brick", nx, ny, "red", pos=(0.0, 0.0, z))
        else:
            add_breakable_brick(world, "brick", nx, ny, "red", pos=(0.0, 0.0, z))
        model = world.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        out[label] = {
            "us_per_step": timed(model, data),
            "nbody": int(model.nbody),
            "nv": int(model.nv),
            "neq": int(model.neq),
        }
    out["slowdown"] = out["breakable"]["us_per_step"] / out["monolithic"]["us_per_step"]
    return out


def figure_crush_sequence(
    out_path: str,
    force_N: float = 20_000.0,
    steps: int = 700,
    ramp_steps: int = 350,
    n_stills: int = 5,
    width: int = 900,
    height: int = 700,
) -> dict:
    """Full-resolution stills of a breakable brick being crushed by the virtual ram.

    The pieces that let go are repainted as they fail, so the sequence reads without a caption:
    the walls buckle out from under the roof one after another while the tubes, which are far
    stiffer in compression, hold on longest. The load is ramped rather than applied at once,
    because 20 kN delivered in a single 2 ms step is an impact and would tell a different
    story.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fresh():
        model, data, handle, manager = build_crush_scene(
            gravity=True, friction=0.6, offwidth=width, offheight=height,
            light_background=True,
        )
        for _ in range(200):
            mujoco.mj_step(model, data)
            manager.update()
        return model, data, handle, manager

    # The brick is intact for the whole ramp and then comes apart in a few tens of
    # milliseconds, so a first pass finds when that is and the second pass puts the stills
    # around it. Both passes are identical and deterministic, so the timestamps are honest.
    model, data, handle, manager = fresh()
    crush(model, data, manager, force_N, steps=steps, ramp_steps=ramp_steps)
    if manager.events:
        first = int(round(manager.events[0].time / model.opt.timestep)) - 200
    else:
        first = steps // 2
    model, data, handle, manager = fresh()

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.010]
    camera.distance = 0.21
    camera.azimuth = 135.0
    camera.elevation = -14.0
    option = mujoco.MjvOption()

    offsets = (-int(0.25 * first), -6, 8, 40, steps - 1 - first)[:n_stills]
    grab_at = sorted({int(np.clip(first + o, 0, steps - 1)) for o in offsets})
    shots: list[tuple[int, np.ndarray, int]] = []
    renderer = mujoco.Renderer(model, height=height, width=width)

    def on_step(i: int, d: mujoco.MjData) -> None:
        if i in grab_at:
            renderer.update_scene(d, camera=camera, scene_option=option)
            shots.append((i, renderer.render().copy(), manager.n_released))

    result = crush(model, data, manager, force_N, steps=steps, ramp_steps=ramp_steps,
                   on_step=on_step)
    renderer.close()

    fig, axes = plt.subplots(1, len(shots), figsize=(3.1 * len(shots), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (step, image, released) in zip(axes, shots):
        ax.imshow(image)
        ax.axis("off")
        load = force_N * min(1.0, (step + 1) / max(ramp_steps, 1)) / 1000.0
        broken = "intact" if released == 0 else f"{released} welds released"
        ax.set_title(f"t = {step * model.opt.timestep * 1000:.0f} ms\n"
                     f"{load:.1f} kN, {broken}", fontsize=9)
    fig.suptitle(
        f"A DUPLO 2x4 built from {len(handle.pieces) + 1} welded pieces, crushed by a "
        f"{force_N / 1000:.0f} kN virtual ram\n"
        "(the arm's own press ceiling is 10 to 14 N, three orders of magnitude below this)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return {
        "n_released": result["n_released"],
        "peak_utilisation": result["peak_utilisation"],
        "events": result["events"],
        "stills": [s[0] for s in shots],
    }
