"""The simple fallback snap model: a funnel, a weld, and a force threshold.

`DetentSnap` implements the same `SnapModel` protocol as the collet model, with none of its
machinery. A brick that comes down close enough to a stud grid gets nudged into line by a
funnel force; once it is at full depth it is welded to whatever it landed on; the weld breaks
when the force across it passes `n * f_clutch * (1 - beta)`.

What it is for
--------------

It is the control against which the collet model earns its keep, and the escape hatch if the
collet model ever turns out to be too slow or too fragile for bulk training. It answers
"connected or not" correctly and cheaply. It cannot answer "how hard do I have to push right
now", because a weld is either on or off - there is no force against depth, so a policy
trained against it would never learn to press. That is the whole argument for `snap.py`, and
the two models being interchangeable here is what makes the argument checkable: the same bench
tests run against both, and this one is only expected to pass the four that do not depend on
the shape of the insertion force curve (PLAN section 6.3 tests 2, 3, 5 and 8).

The weld data layout
--------------------

MuJoCo stores a weld's parameters in `eq_data` as `[anchor(3), relpose_pos(3), relpose_quat(4),
torquescale(1)]`, and `mj_instantiateEquality` reads them as: `data[0:3]` is the weld point in
body2's frame, `data[3:6]` is the same point in body1's frame, and `data[6:10]` is the
orientation of body2 relative to body1. Writing the current relative pose into those ten
numbers and then setting `eq_active` therefore freezes the pair exactly where it is, with no
jump. `tests/bricks/test_snap_detent.py::test_weld_activation_does_not_move_the_brick` checks
that against the installed MuJoCo rather than trusting the layout.
"""

from __future__ import annotations

import mujoco
import numpy as np

from brick_builder.bricks.generator import BrickHandle
from brick_builder.bricks.snap_api import (
    Connection,
    SnapGeometry,
    SnapParams,
    SnapState,
    equality_forces,
)


class DetentSnap:
    """Funnel + weld + break threshold, behind the `SnapModel` protocol."""

    def __init__(self, params: SnapParams = SnapParams()) -> None:
        self.params = params
        self._bricks: list[BrickHandle] = []
        self._plates: list[BrickHandle] = []
        self._eq_names: dict[tuple[int, str], str] = {}
        self._bound = False

    # -- before compile ----------------------------------------------------

    def declare(
        self,
        world: mujoco.MjSpec,
        bricks: list[BrickHandle],
        plates: list[BrickHandle],
    ) -> None:
        """One inactive weld per (brick, other stud-bearing body). No extra bodies or joints."""

        self._bricks = list(bricks)
        self._plates = list(plates)
        self._eq_names = {}
        target_names = [h.name for h in list(bricks) + list(plates)]
        for bi, handle in enumerate(bricks):
            for target in target_names:
                if target == handle.name:
                    continue
                eq = world.add_equality()
                eq.name = f"detenteq_{bi}_{target}"
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                eq.name1 = handle.name
                eq.name2 = target
                eq.active = False
                eq.data = np.zeros(mujoco.mjNEQDATA)
                eq.data[6] = 1.0     # identity relative orientation until a real one is written
                eq.data[10] = 1.0    # torquescale
                eq.solimp = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                eq.solref = [0.004, 1.0]
                self._eq_names[(bi, target)] = eq.name

    # -- after compile -----------------------------------------------------

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model, self._data = model, data
        self.geom = SnapGeometry.build(model, self._bricks, self._plates)
        n_brick = len(self._bricks)
        self._eq_of = np.full((n_brick, model.nbody), -1, dtype=np.int32)
        eq_ids = []
        for (bi, target), eq_name in self._eq_names.items():
            eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
            self._eq_of[bi, bid] = eid
            eq_ids.append(eid)
        self._eq_ids = np.asarray(sorted(eq_ids), dtype=np.int32)

        # Translational dofs of each brick's free joint, for the funnel force. A brick carried
        # by a rig instead of floating free has no free joint and simply gets no funnel.
        self._free_dof = np.full((n_brick, 3), -1, dtype=np.int32)
        for bi, handle in enumerate(self._bricks):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{handle.name}_free")
            if jid >= 0:
                adr = model.jnt_dofadr[jid]
                self._free_dof[bi] = [adr, adr + 1, adr + 2]

        self._welded = np.full(n_brick, -1, dtype=np.int32)   # body id of the partner, or -1
        self._n_engaged = np.zeros(n_brick, dtype=np.int32)
        # A brick whose weld has just broken must come back out before it may re-weld,
        # otherwise the same update that breaks it seats it again and it never comes off.
        self._blocked = np.zeros(n_brick, dtype=bool)
        self._depth = np.zeros(self.geom.n_socket, dtype=np.float64)
        self._step_count = 0
        data.eq_active[self._eq_ids] = 0
        self._bound = True

    # -- per step ----------------------------------------------------------

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self._bound:
            raise RuntimeError("update() called before bind()")
        self._step_count += 1
        if self._step_count % self.params.snap_every:
            return

        p = self.params
        depth, lateral, tilt, valid = self.geom.pair_metrics(data)
        big = np.where(valid, lateral, np.inf)
        best = np.argmin(big, axis=1)
        rows = np.arange(self.geom.n_socket)
        d_best = depth[rows, best]
        lat_best = big[rows, best]
        tilt_best = tilt[rows, best]
        aligned = (
            (lat_best <= p.capture_lateral)
            & (d_best > 0.0)
            & (tilt_best <= np.deg2rad(p.capture_tilt_deg))
        )
        self._depth[:] = np.where(aligned, np.clip(d_best, 0.0, None), 0.0)

        self._break_welds(model, data)
        self._funnel(data, lateral, depth, best, lat_best, d_best, tilt_best)
        self._seat(model, data, d_best, lat_best, tilt_best, best)

    def _break_welds(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Switch a weld off once the force across it passes the pull-off threshold."""

        p = self.params
        live = np.nonzero(self._welded >= 0)[0]
        if live.size == 0:
            return
        forces = equality_forces(model, data, n_rows=3)
        for bi in live:
            eq = int(self._eq_of[bi, self._welded[bi]])
            limit = max(self._n_engaged[bi], 1) * p.f_clutch * (1.0 - p.beta)
            if np.linalg.norm(forces[eq]) > limit:
                data.eq_active[eq] = 0
                self._welded[bi] = -1
                self._n_engaged[bi] = 0
                self._blocked[bi] = True

    def _funnel(
        self,
        data: mujoco.MjData,
        lateral: np.ndarray,
        depth: np.ndarray,
        best: np.ndarray,
        lat_best: np.ndarray,
        d_best: np.ndarray,
        tilt_best: np.ndarray,
    ) -> None:
        """Pull a brick that is nearly over the grid the rest of the way into line.

        The real thing is done by the chamfer on the stud and the taper of the socket mouth,
        which push a slightly-off brick sideways as it comes down. This is the same effect as
        a spring: a force toward the stud axis, strongest at the mouth and gone by full depth,
        and only while the brick is close enough that a real chamfer would have caught it.
        """

        p = self.params
        s, z, pt, _ = self.geom.frames(data)
        for bi, handle in enumerate(self._bricks):
            if self._welded[bi] >= 0 or self._free_dof[bi, 0] < 0:
                continue
            mine = np.nonzero(self.geom.socket_brick == bi)[0]
            active = mine[
                (lat_best[mine] <= p.funnel_lateral)
                & (d_best[mine] > 0.0)
                & (d_best[mine] < self.geom.stud_h)
                & (tilt_best[mine] <= np.deg2rad(p.capture_tilt_deg))
            ]
            if active.size == 0:
                continue
            t = best[active]
            offs = pt[t] - (s[active] + d_best[active, None] * z[active])
            taper = 1.0 - d_best[active] / self.geom.stud_h
            force = p.funnel_gain * np.sum(offs * taper[:, None], axis=0)
            data.qfrc_applied[self._free_dof[bi]] += force

    def _seat(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        d_best: np.ndarray,
        lat_best: np.ndarray,
        tilt_best: np.ndarray,
        best: np.ndarray,
    ) -> None:
        """Weld a brick to what it is sitting on, once its studs are all the way in."""

        p = self.params
        seat = self.geom.stud_h - p.seat_tolerance
        clear = seat - p.release_depth
        for bi in range(len(self._bricks)):
            if self._welded[bi] >= 0:
                continue
            mine = np.nonzero(self.geom.socket_brick == bi)[0]
            if self._blocked[bi]:
                if float(np.max(d_best[mine], initial=-1.0)) < clear:
                    self._blocked[bi] = False
                continue
            home = mine[
                (d_best[mine] >= seat)
                & (lat_best[mine] <= p.capture_lateral)
                & (tilt_best[mine] <= np.deg2rad(p.capture_tilt_deg))
            ]
            if home.size == 0:
                continue
            partner_bodies = self.geom.stud_body[best[home]]
            body = int(np.bincount(partner_bodies).argmax())
            n = int(np.count_nonzero(partner_bodies == body))
            eq = int(self._eq_of[bi, body])
            if eq < 0:
                continue
            _write_weld_pose(model, data, eq, int(self.geom.socket_body[home[0]]), body)
            data.eq_active[eq] = 1
            self._welded[bi] = body
            self._n_engaged[bi] = n

    # -- queries -----------------------------------------------------------

    def stud_depths(self, brick: str) -> np.ndarray:
        bi = self.geom.brick_names.index(brick)
        return self._depth[self.geom.socket_brick == bi].copy()

    def connections(self) -> list[Connection]:
        out = []
        seat = self.geom.stud_h - self.params.seat_tolerance
        for bi, name in enumerate(self.geom.brick_names):
            if self._welded[bi] < 0:
                continue
            depths = self._depth[self.geom.socket_brick == bi]
            engaged = depths[depths > 0.0]
            out.append(
                Connection(
                    upper=name,
                    lower=self.geom.body_name(int(self._welded[bi])),
                    n_engaged=int(self._n_engaged[bi]),
                    seated=bool(engaged.size and np.min(engaged) >= seat),
                )
            )
        return out

    # -- save / restore ----------------------------------------------------

    def get_state(self) -> SnapState:
        model, data = self._model, self._data
        empty = np.zeros(0, dtype=np.int32)
        return SnapState(
            eq_ids=self._eq_ids.copy(),
            eq_active=np.asarray(data.eq_active[self._eq_ids], dtype=np.uint8).copy(),
            eq_data=model.eq_data[self._eq_ids].copy(),
            dof_ids=empty,
            dof_frictionloss=np.zeros(0),
            qfrc_applied=np.zeros(0),
            partner=self._welded.copy(),
            anchor=np.repeat(self._blocked.astype(np.float64)[:, None], 3, axis=1),
            blend_from=self._n_engaged.astype(np.float64),
            step_count=self._step_count,
        )

    def set_state(self, s: SnapState) -> None:
        s.apply_to(self._model, self._data)
        self._welded = s.partner.copy()
        self._blocked = s.anchor[:, 0] > 0.5
        self._n_engaged = s.blend_from.astype(np.int32)
        self._step_count = s.step_count
        self._depth[:] = 0.0


def _write_weld_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eq: int,
    body1: int,
    body2: int,
) -> None:
    """Fill `eq_data` so that activating weld `eq` holds the two bodies exactly where they are."""

    point = data.xpos[body1]                       # any point works; body1's origin is simplest
    r1 = data.xmat[body1].reshape(3, 3)
    r2 = data.xmat[body2].reshape(3, 3)
    model.eq_data[eq, 0:3] = r2.T @ (point - data.xpos[body2])
    model.eq_data[eq, 3:6] = r1.T @ (point - data.xpos[body1])
    conj1 = np.empty(4)
    rel = np.empty(4)
    mujoco.mju_negQuat(conj1, data.xquat[body1])
    mujoco.mju_mulQuat(rel, conj1, data.xquat[body2])
    model.eq_data[eq, 6:10] = rel
    model.eq_data[eq, 10] = 1.0
