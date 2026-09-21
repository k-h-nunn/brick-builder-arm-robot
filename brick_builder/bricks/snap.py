"""The per-socket collet snap-fit model.

This is the model design report 01 section 3 recommends, and it is the reason the project can
train a *press* skill at all: it gives a force that depends on how deep each individual stud
has gone in, rather than a single connected/disconnected flag.

How one socket works
--------------------

Every socket on a brick's underside gets a tiny extra body, the *collet*, hanging off the
brick on a slide joint along the brick's own z axis. The collet has no geometry: it never
collides with anything and is not a part of a real brick. It exists so that the relative
motion between a brick and the stud it is being pressed onto becomes one scalar joint
coordinate that MuJoCo's constraint solver can put dry friction on.

When a stud lines up under a socket and has gone `engage_depth` past the mouth, the model
switches on a `connect` equality that was declared inactive at compile time, pinning the collet
to that stud's tip. From then on the collet's joint coordinate *is* the insertion depth: the
brick can only move down relative to the lower brick by sliding the collet, and the slide
joint's `frictionloss` is what resists. Scheduling that friction with depth,

    frictionloss(d) = f_clutch * g(d),   g as in `snap_api.engagement`

gives a genuine force-against-depth curve with real stick-slip behaviour, because MuJoCo
already solves dry friction properly. What shape `g` has is a parameter (`SnapParams.curve`:
"ramp", "plateau" or "detent") rather than a fact, because no one has put a real DUPLO brick on
a force gauge yet. Nothing in this file cares which shape is in use: the clutch is scheduled
from whatever `engagement` returns, and all three shapes end at `g(stud_h) = 1`, so a seated
stud holds `f_clutch * (1 - beta)` whichever one is chosen. The shape does decide whether a
rocking press is worth anything - see `docs/guide/02-simulator-and-bricks.md`.

Scaling that same bound by `1 + beta` while the stud is
going in and `1 - beta` while it is coming out or standing still makes pushing cost
`f_clutch * (1 + beta)` per stud and pulling out release at `f_clutch * (1 - beta)`, which is
the asymmetry a real brick has. (The design report gets the asymmetry from a bias force
instead; `_asymmetry` below says why that had to change.)

The collet also carries two short sideways slides, limited to the socket's clearance. Without
them a `connect` pins a stud to a point fixed in the brick, and a brick with two studs in
cannot tilt at all - which makes rocking and peeling impossible. With them a stud has the play
a real socket gives it, and the tilt-binding limit `atan(2c/d)` comes out for free.

Nothing in here special-cases rocking or peeling. A brick pressed at an angle simply has its
sockets at different depths, so the leading row is already at full `g` while the trailing row
is barely gripping; the leading row then acts as the hinge. Peeling is the same thing in
reverse. Both come out of the per-stud bookkeeping for free.

Two MuJoCo facts this model leans on, both checked against mujoco 3.13.0
-----------------------------------------------------------------------

1. `model.dof_frictionloss` and `model.eq_data` are re-read by `mj_step` every step, so the
   clutch schedule and the capture anchors can be written at run time. `data.eq_active`
   switches a pre-declared equality on and off. Geom collision bits are never touched at run
   time, because that does not port to MJX or MuJoCo-Warp.
2. A dof's friction constraint is regularized by `R = dof_invweight0 * (1 - imp) / imp`, and
   `imp` is clamped at 0.9999. For a 0.2 g collet `dof_invweight0` is 5000 1/kg, which leaves
   `R = 0.5` - so a joint told to hold 1.0 N actually creeps about 5 mm per second at 0.9 N,
   and breaks away at roughly 0.15 N. That is measured, not assumed (see
   `tests/bricks/test_snap_bench.py::test_collet_friction_regularization`). The model therefore
   overwrites `dof_invweight0` for its own collet dofs with `params.collet_invweight` (10 by
   default), which is a pure regularization choice: it changes no mass, no inertia and no other
   constraint, and it brings the creep at 0.9 N down to about 0.009 mm/s while the break-away
   force lands on `f_clutch` itself.

The MJX trap from the design report is respected: collet friction is initialised to
`params.friction_floor` (1e-4 N), never to 0, because MJX freezes `dof_frictionloss > 0` when
the model is put on the device.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from brick_builder.bricks.generator import BrickHandle
from brick_builder.bricks.snap_api import (
    Connection,
    SnapGeometry,
    SnapParams,
    SnapState,
    engagement,
    equality_forces,
)


class ColletSnap:
    """Per-socket collet snap model. See the module docstring for the mechanism."""

    def __init__(self, params: SnapParams = SnapParams()) -> None:
        self.params = params
        self._bricks: list[BrickHandle] = []
        self._plates: list[BrickHandle] = []
        self._collet_joints: list[str] = []          # axial slide, one per socket
        self._lateral_joints: list[str] = []         # two sideways slides per socket
        self._eq_names: dict[tuple[int, str], str] = {}   # (socket index, target body) -> eq name
        self._bound = False
        self.observers: list = []

    # -- before compile ----------------------------------------------------

    def declare(
        self,
        world: mujoco.MjSpec,
        bricks: list[BrickHandle],
        plates: list[BrickHandle],
    ) -> None:
        """Add one collet body per socket and one inactive `connect` per (collet, other body).

        The collet bodies carry real mass, so the same mass is taken off the brick shell: a
        2x4 brick with eight 0.2 g collets keeps its 12.5 g total instead of gaining 13 % of
        extra mass out of nowhere.
        """

        if self._bound:
            raise RuntimeError("declare() must be called before bind(), on a fresh MjSpec")
        self._bricks = list(bricks)
        self._plates = list(plates)
        self._collet_joints = []
        self._lateral_joints = []
        self._eq_names = {}

        p = self.params
        # A brick is usually one body, but a breakable one is a roof plus a piece per stud, and
        # a stud site lives on its own piece. `BrickHandle.targets` is the list of bodies a
        # connection may be made to, so a released stud carries its connection with it.
        target_names = [b for h in list(bricks) + list(plates) for b in h.targets]

        socket_index = 0
        for handle in bricks:
            spec = handle.spec
            n_sockets = len(handle.socket_sites)
            handle.body.mass = max(handle.body.mass - n_sockets * p.collet_mass, 1e-6)
            for site_name in handle.socket_sites:
                suffix = site_name[len(handle.name) + len("_socket_"):]
                local = _site_local_pos(handle, site_name)
                collet = handle.body.add_body(
                    name=f"{handle.name}_collet_{suffix}", pos=list(local)
                )
                # A collet is a modelling device, not a part: no geometry, and gravity is
                # cancelled so a free (uncaptured) collet does not sag to its joint limit and
                # add a spurious impulse the moment it is captured.
                collet.mass = p.collet_mass
                collet.inertia = [1e-9, 1e-9, 1e-9]
                collet.ipos = [0.0, 0.0, 0.0]
                collet.explicitinertial = True
                collet.gravcomp = 0.0
                # Two sideways slides limited to the socket clearance, then the axial slide the
                # clutch acts on. Without the sideways pair a `connect` would pin the stud to a
                # point fixed in the brick, so a brick with two or more studs in could not tilt
                # at all and rocking became impossible (measured: 22 N of resistance at 2 mm of
                # insertion, with zero contacts). With them, a stud has exactly the play a real
                # socket gives it, which also reproduces the tilt-binding limit atan(2c/d).
                play = p.lateral_play * spec.clearance
                for axis, tag in (([1.0, 0.0, 0.0], "x"), ([0.0, 1.0, 0.0], "y")):
                    lat = collet.add_joint(
                        name=f"{handle.name}_collet{tag}_{suffix}",
                        type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=axis,
                        range=[-play, play],
                    )
                    lat.armature = p.collet_armature
                    lat.damping = [p.collet_damping, 0.0, 0.0]
                    lat.frictionloss = p.friction_floor
                    lat.stiffness = [p.lateral_stiffness, 0.0, 0.0]
                    lat.solimp_friction = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                    lat.solimp_limit = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                    lat.solref_limit = [0.004, 1.0]
                    self._lateral_joints.append(lat.name)
                joint = collet.add_joint(
                    name=f"{handle.name}_colletj_{suffix}",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[0.0, 0.0, 1.0],
                    range=[-p.collet_margin, spec.stud_h + p.collet_margin],
                )
                joint.armature = p.collet_armature
                # A little viscosity, as the design report calls for: without it the collet
                # joints are an almost undamped stiff mode and a settled three-layer stack
                # rings at about 1 mm instead of standing still.
                joint.damping = [p.collet_damping, 0.0, 0.0]
                joint.frictionloss = p.friction_floor
                # Hardest impedance MuJoCo allows on a friction row; see the module docstring.
                joint.solimp_friction = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                self._collet_joints.append(joint.name)

                own = set(handle.targets)
                for target in target_names:
                    if target in own:
                        continue
                    eq = world.add_equality()
                    eq.name = f"snapeq_{socket_index}_{target}"
                    eq.type = mujoco.mjtEq.mjEQ_CONNECT
                    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                    eq.name1 = collet.name
                    eq.name2 = target
                    eq.active = False
                    eq.data = np.zeros(mujoco.mjNEQDATA)
                    # A soft connect would stretch under the clutch force and let the brick
                    # move without moving the collet, which would decouple the measured depth
                    # from the joint the friction acts on. Measured on this rig: with MuJoCo's
                    # default equality impedance the brick slides 1.4 mm out of a seated socket
                    # before the clutch registers anything.
                    eq.solimp = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                    eq.solref = [0.004, 1.0]
                    self._eq_names[(socket_index, target)] = eq.name
                socket_index += 1

    # -- after compile -----------------------------------------------------

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Resolve ids, apply the regularization override, and clear all connections."""

        self._model, self._data = model, data
        self.geom = SnapGeometry.build(model, self._bricks, self._plates)
        n = self.geom.n_socket
        if n != len(self._collet_joints):
            raise RuntimeError("bind() called with a model that does not match declare()")

        self._dof = np.empty(n, dtype=np.int32)
        self._qadr = np.empty(n, dtype=np.int32)
        for k, jname in enumerate(self._collet_joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._dof[k] = model.jnt_dofadr[jid]
            self._qadr[k] = model.jnt_qposadr[jid]
        self._lat_dof = np.empty((n, 2), dtype=np.int32)
        self._lat_qadr = np.empty((n, 2), dtype=np.int32)
        for k, jname in enumerate(self._lateral_joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._lat_dof[k // 2, k % 2] = model.jnt_dofadr[jid]
            self._lat_qadr[k // 2, k % 2] = model.jnt_qposadr[jid]

        # eq id for (socket, target body id)
        self._eq_of = np.full((n, model.nbody), -1, dtype=np.int32)
        eq_ids = []
        for (socket, target), eq_name in self._eq_names.items():
            eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
            self._eq_of[socket, bid] = eid
            eq_ids.append(eid)
        self._eq_ids = np.asarray(sorted(eq_ids), dtype=np.int32)

        model.dof_invweight0[self._dof] = self.params.collet_invweight
        model.dof_invweight0[self._lat_dof.reshape(-1)] = self.params.collet_invweight
        model.dof_frictionloss[self._dof] = self.params.friction_floor
        model.dof_frictionloss[self._lat_dof.reshape(-1)] = self.params.friction_floor

        self._partner = np.full(n, -1, dtype=np.int32)
        self._anchor = np.zeros((n, 3), dtype=np.float64)
        self._blend_from = np.zeros(n, dtype=np.float64)
        self._depth = np.zeros(n, dtype=np.float64)
        self._step_count = 0
        data.eq_active[self._eq_ids] = 0
        data.qfrc_applied[self._dof] = 0.0
        self._bound = True

    # -- per step ----------------------------------------------------------

    @property
    def _capture_reach(self) -> float:
        """How far a stud tip can be from a socket mouth and still be capturable [m].

        Exactly the gate ``_capture`` applies, turned into one distance: depth is measured along
        the socket axis and lateral error across it, so a pair that passes both bounds is within
        ``hypot(max depth, max lateral)`` of the mouth. Read off ``self.params`` every call
        because randomization replaces them between episodes.
        """

        p = self.params
        return math.hypot(self.geom.stud_h + p.collet_margin, p.capture_lateral) * (1.0 + 1e-9)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """One bookkeeping pass, then every observer.

        Observers are how anything that is not a controller gets a per-step look at the scene -
        `brick_builder.bricks.damage.DamageMonitor` is the one this project ships. They run after
        the pass, so a monitor that writes to `model.dof_frictionloss` is not overwritten by this
        model's own scheduling, and they see the contact forces the last `mj_step` produced.
        """

        self._update_pass(model, data)
        for observer in self.observers:
            observer.notify(model, data)

    def _update_pass(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """One bookkeeping pass: release, re-schedule the clutch, then capture.

        Everything an engaged collet needs happens on every call, because that is what the
        force-against-depth curve is made of. The *capture* half is what gets skipped: a stud can
        only be picked up while its tip is inside a socket column, so the depth/lateral/tilt
        tables are only built against the studs whose bounding box is within a stud height of a
        socket mouth (:meth:`SnapGeometry.capture_candidates`), which on an 8x8 plate under a 2x4
        brick is four studs out of seventy-two while it is being pressed and none at all while it
        hovers. The test is conservative -- it can only ever say "look properly", never "there is
        nothing there" about a pair that could connect -- so what gets captured is unchanged.
        """

        if not self._bound:
            raise RuntimeError("update() called before bind()")
        self._step_count += 1
        if self._step_count % self.params.snap_every:
            return

        p = self.params
        n = self.geom.n_socket
        if n == 0:
            return      # a scene with no brick in it (Reach): nothing can ever snap to anything
        frames = self.geom.frames(data)
        rows = np.arange(n)
        engaged = self._partner >= 0

        self._depth[:] = 0.0
        if engaged.any():
            live = rows[engaged]
            idx = self._partner[live]
            d, lat = self.geom.pair_depth(frames, live, idx)
            self._depth[live] = d

            # Release: the stud has come back out of the mouth, or the pair has been torn
            # sideways far enough that the stud is no longer inside the socket column.
            drop = (d <= 0.0) | (lat > p.capture_lateral + p.blend_distance)

            # Shallow cam-out: while the connection is still only chamfer-deep, a large
            # sideways constraint force levers the stud straight back out. Reading the
            # constraint rows is not cheap, so it only happens when something is that shallow.
            g = engagement(d, p, self.geom.stud_h)
            shallow = (~drop) & (d < p.camout_depth)
            if shallow.any():
                f_eq = equality_forces(model, data, n_rows=3)
                eq_ids = self._eq_of[live[shallow], self.geom.stud_body[idx[shallow]]]
                f = f_eq[eq_ids]
                axis = frames[1][live[shallow]]
                f_axial = (f * axis).sum(axis=1)
                f_lat = np.sqrt(((f - f_axial[:, None] * axis) ** 2).sum(axis=1))
                limit = p.camout_ratio * np.maximum(p.f_clutch * g[shallow], p.friction_floor)
                drop[shallow] |= f_lat > limit

            if drop.any():
                self._release(model, data, live[drop])
                engaged = self._partner >= 0

            # Re-schedule the clutch for whatever is still engaged.
            still = np.nonzero(engaged)[0]
            if still.size:
                d = self._depth[still]
                g = engagement(d, p, self.geom.stud_h)
                clutch = p.f_clutch * g * _asymmetry(data.qvel[self._dof[still]], p)
                model.dof_frictionloss[self._dof[still]] = np.maximum(clutch, p.friction_floor)
                # The sideways play grips too, or a seated brick could be slid across the
                # studs for nothing; at `lateral_ratio` of the axial clutch it still lets a
                # brick be rocked or peeled, which needs only tens of microns of side travel.
                model.dof_frictionloss[self._lat_dof[still].reshape(-1)] = np.maximum(
                    np.repeat(p.lateral_ratio * clutch, 2), p.friction_floor
                )

        free = np.nonzero(self._partner < 0)[0]
        if free.size:
            cols = self.geom.capture_candidates(frames, free, self._capture_reach)
            if cols.size:
                depth, lateral, tilt, valid = self.geom.pair_metrics(data, free, frames, cols)
                self._capture(model, data, frames, free, cols, depth, lateral, tilt, valid)

    def _release(self, model: mujoco.MjModel, data: mujoco.MjData, sockets: np.ndarray) -> None:
        if sockets.size == 0:
            return
        eq_ids = self._eq_of[sockets, self.geom.stud_body[self._partner[sockets]]]
        data.eq_active[eq_ids] = 0
        model.dof_frictionloss[self._dof[sockets]] = self.params.friction_floor
        model.dof_frictionloss[self._lat_dof[sockets].reshape(-1)] = self.params.friction_floor
        self._partner[sockets] = -1
        self._anchor[sockets] = 0.0
        self._depth[sockets] = 0.0

    def _capture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        frames: tuple,
        free: np.ndarray,
        cols: np.ndarray,
        depth: np.ndarray,
        lateral: np.ndarray,
        tilt: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        """Connect every free socket that is lined up over a free stud and deep enough.

        ``free`` and ``cols`` say which sockets and which studs the four tables describe, so
        ``depth[r, j]`` is socket ``free[r]`` against stud ``cols[j]``.
        """

        p = self.params
        ok = (
            valid
            & (lateral <= p.capture_lateral)
            & (depth >= p.engage_depth)
            & (depth <= self.geom.stud_h + p.collet_margin)
            & (tilt <= np.deg2rad(p.capture_tilt_deg))
        )
        taken = self._partner[self._partner >= 0]
        if taken.size:
            ok[:, np.isin(cols, taken)] = False
        candidates = np.nonzero(ok.any(axis=1))[0]
        if candidates.size == 0:
            return

        s, z, _, _ = frames
        xmat = data.xmat.reshape(-1, 3, 3)
        for r in candidates:
            choices = np.nonzero(ok[r])[0]
            if choices.size == 0:
                continue
            j = int(choices[np.argmin(lateral[r, choices])])
            ok[:, j] = False
            t = int(cols[j])
            k = int(free[r])
            d = float(depth[r, j])
            body = int(self.geom.stud_body[t])
            eq = int(self._eq_of[k, body])
            # Take up the residual offset with the collet's own play instead of writing it
            # into the equality: the anchor then stays at the collet origin, the connect is
            # satisfied exactly at the moment of capture, and the centring spring on the two
            # sideways joints pulls the stud to the middle of the socket from there.
            lat_vec = (
                data.xpos[body]
                + xmat[body] @ self.geom.stud_local[t]
                - (s[k] + d * z[k])
            )
            rb = xmat[int(self.geom.socket_body[k])]
            local = rb.T @ lat_vec
            model.eq_data[eq, 0:3] = 0.0
            model.eq_data[eq, 3:6] = self.geom.stud_local[t]
            data.qpos[self._qadr[k]] = d
            data.qvel[self._dof[k]] = 0.0
            data.qpos[self._lat_qadr[k]] = local[:2]
            data.qvel[self._lat_dof[k]] = 0.0
            data.eq_active[eq] = 1
            self._partner[k] = t
            self._anchor[k] = local
            self._blend_from[k] = d
            self._depth[k] = d
            g = engagement(np.array([d]), p, self.geom.stud_h)[0]
            model.dof_frictionloss[self._dof[k]] = max(p.f_clutch * g, p.friction_floor)
            model.dof_frictionloss[self._lat_dof[k]] = max(
                p.lateral_ratio * p.f_clutch * g, p.friction_floor
            )

    # -- queries -----------------------------------------------------------

    def stud_depths(self, brick: str) -> np.ndarray:
        bi = self.geom.brick_names.index(brick)
        return self._depth[self.geom.socket_brick == bi].copy()

    def connections(self) -> list[Connection]:
        out: dict[tuple[str, str], list[float]] = {}
        for k in np.nonzero(self._partner >= 0)[0]:
            upper = self.geom.brick_names[int(self.geom.socket_brick[k])]
            lower = self.geom.body_name(int(self.geom.stud_body[self._partner[k]]))
            out.setdefault((upper, lower), []).append(float(self._depth[k]))
        seat = self.geom.stud_h - self.params.seat_tolerance
        return [
            Connection(u, l, len(ds), bool(min(ds) >= seat))
            for (u, l), ds in sorted(out.items())
        ]

    # -- save / restore ----------------------------------------------------

    def get_state(self) -> SnapState:
        """Snapshot the model-side writes plus this object's own bookkeeping.

        `mjSTATE_INTEGRATION` covers `qpos`, `qvel`, `qfrc_applied` and `eq_active`, but there
        is no state spec for anything on the `mjModel` side, so `eq_data` and
        `dof_frictionloss` have to be captured here for a restore to be exact.
        """

        model, data = self._model, self._data
        dofs = np.concatenate([self._dof, self._lat_dof.reshape(-1)])
        return SnapState(
            eq_ids=self._eq_ids.copy(),
            eq_active=np.asarray(data.eq_active[self._eq_ids], dtype=np.uint8).copy(),
            eq_data=model.eq_data[self._eq_ids].copy(),
            dof_ids=dofs,
            dof_frictionloss=model.dof_frictionloss[dofs].copy(),
            qfrc_applied=data.qfrc_applied[dofs].copy(),
            partner=self._partner.copy(),
            anchor=self._anchor.copy(),
            blend_from=self._blend_from.copy(),
            step_count=self._step_count,
        )

    def set_state(self, s: SnapState) -> None:
        s.apply_to(self._model, self._data)
        self._partner = s.partner.copy()
        self._anchor = s.anchor.copy()
        self._blend_from = s.blend_from.copy()
        self._step_count = s.step_count
        self._depth[:] = 0.0


def _asymmetry(qvel: np.ndarray, params: SnapParams) -> np.ndarray:
    """`1 + beta` while a stud is being pushed in, `1 - beta` while it is coming out or still.

    Design report 01 section 3 gets the push/pull asymmetry from a constant bias force
    `-beta * f_c * g` on the collet joint. That reproduces both break-away thresholds, but it
    also leaves a permanent force pushing a seated brick back off its studs: 0.15 N per stud,
    1.2 N for a 2x4 brick, ten times the brick's own 0.12 N weight. Measured consequence on
    this rig: a three-layer stack climbed 5.6 mm off the baseplate in 10 s of standing still.

    Scaling the friction *bound* by direction instead costs nothing statically and gives the
    same two thresholds, so that is what this model does. A stationary joint is treated as
    about to come out, which is the case the holding force has to cover.
    """

    return 1.0 + params.beta * np.where(qvel > params.stick_velocity, 1.0, -1.0)


def _site_local_pos(handle: BrickHandle, site_name: str) -> np.ndarray:
    """Position of one of the generator's sites in its own body frame, before compiling."""

    for site in handle.body.sites:
        if site.name == site_name:
            return np.asarray(site.pos, dtype=np.float64).copy()
    raise KeyError(f"{handle.name} has no site {site_name!r}")
