"""The interface every snap-fit model implements, plus the geometry both models share.

A "snap model" is the part of the simulator that says what happens when a stud goes into a
socket: how hard you have to push, how deep it is, when the two bricks count as connected, and
how hard you have to pull to get them apart again. The geometry in `generator.py` deliberately
leaves a clearance gap everywhere, so the raw collision shapes hold nothing together; all the
holding force comes from the model behind this interface.

Two implementations live next door:

- `snap.py` (`ColletSnap`) - the per-socket collet of
  `docs/guide/02-simulator-and-bricks.md`. It models force against
  depth for every stud separately, so partial insertion, the tilted press and peeling all fall
  out of the same mechanism.
- `snap_detent.py` (`DetentSnap`) - a much simpler fallback: a funnel that pulls a nearly
  aligned brick into place, a rigid weld once it is seated, and a force threshold that breaks
  the weld. It has no force-against-depth curve at all, which is exactly why the collet model
  exists.

Conventions used throughout:

- SI units: metres, radians, seconds, newtons. Quaternions are `(w, x, y, z)`.
- A *socket* is a grid position on a brick's underside; the brick with the socket is the
  *upper* body. A *stud* is a grid position on top of a brick or baseplate: the *lower* body.
- *Depth* `d` is how far a stud has gone into a socket, measured along the upper brick's local
  z axis from the socket mouth plane (the brick's bottom face) to the stud's tip. `d <= 0`
  means the stud has not entered; `d = spec.stud_h` means the brick is fully seated, because
  that is exactly where the brick's walls and tubes go flush with the surface below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import mujoco
import numpy as np

from brick_builder.bricks.generator import BrickHandle


@dataclass(frozen=True)
class Connection:
    """One live connection between two bodies, as the snap model currently sees it."""

    upper: str          # body name of the brick whose sockets are engaged
    lower: str          # body name of the brick or baseplate providing the studs
    n_engaged: int      # how many studs of that pair are currently in
    seated: bool        # every engaged stud is at (nearly) full depth


@dataclass(frozen=True)
class SnapParams:
    """Everything a snap model can be tuned with.

    Defaults are the "until we can measure real bricks" values of design report 01 section 4
    and PLAN section 6.3. Nothing here has been calibrated against hardware yet; the
    randomization ranges quoted in the design report belong to the training config, not here.

    - f_clutch: holding force of one fully inserted stud, newtons.
    - beta: push/pull asymmetry. Pushing a stud in costs `f_clutch * (1 + beta)`, pulling it
      out releases at `f_clutch * (1 - beta)`, which is why DUPLO feels stiffer going on.
    - exponent: shape of the engagement ramp `g(d)`; 1 is a straight line.
    - curve: which force-against-depth family `g(d)` comes from - "ramp", "plateau" or
      "detent". Nobody has measured a real brick yet, so the shape is a parameter rather than a
      fact; see `engagement` below for the three laws and `docs/guide/02-simulator-and-bricks.md`
      for what each one does to a press. All three end at `g(stud_h) = 1`, so the holding
      (pull-off) force of a seated stud is `f_clutch * (1 - beta)` whichever shape is chosen.
    - plateau_gain: the height of the flat part of the "detent" curve, as a fraction of
      `f_clutch`. Unused by the other two shapes.
    - detent_gain, detent_depth, detent_width: the entry bump of the "detent" curve - how far
      above the plateau it rises (fraction of `f_clutch`), the depth it is centred on, and its
      full width at about a tenth of its height.
    - engage_depth: how far past the socket mouth a stud must be before it grips at all, and
      the depth at which `g` starts rising.
    - capture_lateral, capture_tilt_deg: the gates a socket/stud pair must pass to connect.
    - seat_tolerance: how close to `stud_h` counts as "seated".
    - snap_every: run the update once every N physics steps.
    - stick_velocity: below this joint speed the clutch counts as stationary, which is what
      decides whether the push or the pull bound applies.
    - blend_distance: extra sideways slack a live connection is allowed before it is dropped.
    - release_depth: how far a broken connection must back out before it may form again.
    - camout_depth, camout_ratio: a shallow connection is torn sideways out of the socket when
      the sideways constraint force exceeds `camout_ratio` times the current clutch force.
    - collet_mass, collet_armature, collet_damping, collet_margin, collet_invweight,
      friction_floor, lateral_play, lateral_stiffness, lateral_ratio: collet model internals,
      documented in `snap.py`.
    - funnel_lateral, funnel_gain: DetentSnap internals, documented in `snap_detent.py`.
    """

    f_clutch: float = 1.0
    beta: float = 0.15
    exponent: float = 1.0
    curve: Literal["ramp", "plateau", "detent"] = "ramp"
    plateau_gain: float = 0.35     # height of the detent curve's flat part, x f_clutch
    detent_gain: float = 1.0       # height of the entry bump above that plateau, x f_clutch
    detent_depth: float = 0.001    # depth the entry bump is centred on, metres
    detent_width: float = 0.0006   # width of the entry bump, metres
    capture_lateral: float = 0.0003
    capture_tilt_deg: float = 6.0
    engage_depth: float = 0.0005
    seat_tolerance: float = 0.0003
    stick_velocity: float = 5e-5   # below this the clutch counts as stationary, m/s
    snap_every: int = 1
    blend_distance: float = 0.0005
    release_depth: float = 0.001   # how far a broken connection must back out before it may re-form
    camout_depth: float = 0.001
    camout_ratio: float = 1.5
    collet_mass: float = 2e-4
    collet_armature: float = 2e-3
    collet_damping: float = 2.0    # viscous term on a collet joint, N.s/m
    collet_margin: float = 0.0003
    collet_invweight: float = 10.0
    friction_floor: float = 1e-4
    lateral_play: float = 1.0          # sideways play as a fraction of spec.clearance
    lateral_stiffness: float = 2000.0  # N/m centring spring on a collet's sideways play
    lateral_ratio: float = 1.0         # sideways clutch friction, as a fraction of the axial one
    funnel_lateral: float = 0.002
    funnel_gain: float = 30.0


@dataclass
class SnapState:
    """Everything a snap model writes outside `mjSTATE_INTEGRATION`, so a run can be restored.

    `mjSTATE_INTEGRATION` already carries `qpos`, `qvel`, `act`, `ctrl`, `qfrc_applied`,
    `xfrc_applied`, `eq_active`, `mocap`, `userdata`, `time` and the warm-start cache. It does
    *not* carry anything on the model side, and both snap models write `model.eq_data` (capture
    anchors) and `model.dof_frictionloss` (the clutch schedule) as they run. Those two arrays,
    plus the model's own bookkeeping, live here.

    `eq_ids` and `dof_ids` are stored alongside the values so `apply_to` needs nothing else.
    """

    eq_ids: np.ndarray            # int32 (n_eq,) model equality indices this snap model owns
    eq_active: np.ndarray         # uint8 (n_eq,)
    eq_data: np.ndarray           # float64 (n_eq, mjNEQDATA)
    dof_ids: np.ndarray           # int32 (n_dof,) dofs whose frictionloss the model schedules
    dof_frictionloss: np.ndarray  # float64 (n_dof,)
    qfrc_applied: np.ndarray      # float64 (n_dof,)
    partner: np.ndarray           # int32 (n_socket,) stud index per socket, -1 when free
    anchor: np.ndarray            # float64 (n_socket, 3) capture-time lateral residual
    blend_from: np.ndarray        # float64 (n_socket,) depth at which the blend started
    step_count: int = 0

    def apply_to(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Write the model-side and data-side parts of this state back into a live sim."""

        model.eq_data[self.eq_ids] = self.eq_data
        data.eq_active[self.eq_ids] = self.eq_active
        model.dof_frictionloss[self.dof_ids] = self.dof_frictionloss
        data.qfrc_applied[self.dof_ids] = self.qfrc_applied

    def copy(self) -> "SnapState":
        return SnapState(
            eq_ids=self.eq_ids.copy(),
            eq_active=self.eq_active.copy(),
            eq_data=self.eq_data.copy(),
            dof_ids=self.dof_ids.copy(),
            dof_frictionloss=self.dof_frictionloss.copy(),
            qfrc_applied=self.qfrc_applied.copy(),
            partner=self.partner.copy(),
            anchor=self.anchor.copy(),
            blend_from=self.blend_from.copy(),
            step_count=self.step_count,
        )


@runtime_checkable
class SnapModel(Protocol):
    """What the rest of the project is allowed to assume about a snap model."""

    def declare(
        self,
        world: mujoco.MjSpec,
        bricks: list[BrickHandle],
        plates: list[BrickHandle],
    ) -> None:
        """Add whatever bodies, joints and equalities the model needs, before `world.compile()`."""

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Resolve ids against a compiled model and reset the model's own bookkeeping."""

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Advance the snap bookkeeping. Call between `mj_step` calls, every `snap_every` steps."""

    def stud_depths(self, brick: str) -> np.ndarray:
        """Insertion depth in metres for each socket of `brick`, in that brick's socket order."""

    def connections(self) -> list[Connection]:
        """Every live connection, one entry per (upper body, lower body) pair."""

    def get_state(self) -> SnapState: ...

    def set_state(self, s: SnapState) -> None: ...


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------


@dataclass
class SnapGeometry:
    """Flat, vectorised tables of every socket and every stud in a compiled scene.

    Both snap models need the same three numbers for every (socket, stud) pair - insertion
    depth, lateral error and relative tilt - and both need them cheaply, every physics step.
    This class stores the constant part (which body each socket and stud belongs to, and where
    they sit in that body's own frame) once, and turns the per-step body poses into those three
    tables with a handful of numpy calls and no Python loop over pairs.

    Layout:
    - sockets are numbered 0..n_socket-1, grouped by brick in the order `bricks` was given;
      `socket_body[k]` is the body id, `socket_local[k]` the socket mouth centre in that frame.
    - studs are numbered 0..n_stud-1 over every stud-bearing body (bricks and plates);
      `stud_local[t]` is the stud *tip* centre (top of the stud) in its body frame.
    """

    socket_body: np.ndarray    # int32 (n_socket,)
    socket_local: np.ndarray   # float64 (n_socket, 3)
    socket_brick: np.ndarray   # int32 (n_socket,) index into `brick_names`
    stud_body: np.ndarray      # int32 (n_stud,)
    stud_local: np.ndarray     # float64 (n_stud, 3)
    brick_names: list[str] = field(default_factory=list)
    body_names: list[str] = field(default_factory=list)
    body_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    brick_body_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    stud_h: float = 0.0

    def body_name(self, body_id: int) -> str:
        """Name of a stud-bearing body this geometry knows about."""

        hit = np.nonzero(self.body_ids == body_id)[0]
        return self.body_names[int(hit[0])] if hit.size else str(body_id)

    @staticmethod
    def build(
        model: mujoco.MjModel,
        bricks: list[BrickHandle],
        plates: list[BrickHandle],
    ) -> "SnapGeometry":
        """Read socket and stud positions out of the sites the generator already placed."""

        socket_body, socket_local, socket_brick = [], [], []
        for bi, handle in enumerate(bricks):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, handle.name)
            for site in handle.socket_sites:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
                socket_body.append(body_id)
                socket_local.append(model.site_pos[sid].copy())
                socket_brick.append(bi)

        stud_body, stud_local, body_ids = [], [], []
        all_names, all_ids = [], []
        for handle in list(bricks) + list(plates):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, handle.name)
            body_ids.append(body_id)
            for target in handle.targets:
                all_names.append(target)
                all_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target))
            for site in handle.stud_sites:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
                # The site's own body, not the handle's: a breakable brick puts each stud site
                # on the stud's own piece, so that a stud that breaks off takes its connection
                # with it rather than leaving it behind on the roof.
                stud_body.append(int(model.site_bodyid[sid]))
                stud_local.append(model.site_pos[sid].copy())

        spec = bricks[0].spec if bricks else plates[0].spec
        return SnapGeometry(
            socket_body=np.asarray(socket_body, dtype=np.int32),
            socket_local=np.asarray(socket_local, dtype=np.float64).reshape(-1, 3),
            socket_brick=np.asarray(socket_brick, dtype=np.int32),
            stud_body=np.asarray(stud_body, dtype=np.int32),
            stud_local=np.asarray(stud_local, dtype=np.float64).reshape(-1, 3),
            brick_names=[h.name for h in bricks],
            body_names=all_names,
            body_ids=np.asarray(all_ids, dtype=np.int32),
            brick_body_ids=np.asarray(body_ids[: len(bricks)], dtype=np.int32),
            stud_h=spec.stud_h,
        )

    @property
    def n_socket(self) -> int:
        return int(self.socket_body.shape[0])

    @property
    def n_stud(self) -> int:
        return int(self.stud_body.shape[0])

    def frames(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """World socket mouths `S`, socket axes `Z`, stud tips `P`, stud axes `Zt`."""

        xpos = data.xpos
        xmat = data.xmat.reshape(-1, 3, 3)
        rb = xmat[self.socket_body]
        s = xpos[self.socket_body] + np.einsum("kij,kj->ki", rb, self.socket_local)
        z = rb[:, :, 2]
        rt = xmat[self.stud_body]
        p = xpos[self.stud_body] + np.einsum("tij,tj->ti", rt, self.stud_local)
        zt = rt[:, :, 2]
        return s, z, p, zt

    def capture_candidates(self, frames: tuple, rows: np.ndarray, reach: float) -> np.ndarray:
        """Which studs could possibly be within ``reach`` of one of these sockets' mouths.

        A stud can only be captured while its tip is inside the socket column, so it has to be
        within a stud height or so of a mouth in every direction. This is the cheap gate in front
        of :meth:`pair_metrics`: the axis-aligned box around the socket mouths, grown by
        ``reach``, is tested against each stud tip -- a handful of reductions and one comparison,
        instead of the depth/lateral/tilt tables over every (socket, stud) pair.

        The answer is conservative in the direction that matters. A stud within ``reach`` of a
        socket mouth is inside that grown box on every axis, so nothing capturable is ever
        dropped; a stud that survives only has to be looked at properly. With a brick hovering a
        few millimetres above the plate, or parked anywhere else, nothing survives at all, and
        while it is being pressed onto an 8x8 plate the four studs under it survive out of 72.
        """

        s, _, p, _ = frames
        if rows.size == 0 or p.shape[0] == 0:
            return np.zeros(0, dtype=np.intp)
        sr = s[rows]
        lo = sr.min(axis=0) - reach
        hi = sr.max(axis=0) + reach
        return np.nonzero(((p >= lo) & (p <= hi)).all(axis=1))[0]

    def pair_metrics(
        self,
        data: mujoco.MjData,
        rows: np.ndarray | None = None,
        frames: tuple | None = None,
        cols: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Depth, lateral error, tilt and validity for every (socket, stud) pair.

        Returns `(depth, lateral, tilt, valid)`, each `(len(rows), n_stud)`:

        - `depth[k, t]` is the stud tip's position along socket k's axis, measured from the
          socket mouth. Negative means the stud is still below the mouth plane.
        - `lateral[k, t]` is the distance from the stud tip to socket k's axis line.
        - `tilt[k, t]` is the angle in radians between the two bodies' z axes.
        - `valid[k, t]` is False when the stud belongs to the socket's own body (a brick can
          never plug into itself).

        `rows` restricts the work to a subset of sockets. The full matrix is only needed to
        look for new connections, so a scene whose bricks are all already seated pays for the
        few rows it actually has to re-check instead of all of them.

        `cols` restricts the studs the same way, and then the tables are `(len(rows), len(cols))`
        with column `j` describing stud `cols[j]`. :meth:`capture_candidates` is what produces
        one: on an 8x8 plate under a 2x4 brick it is four studs out of seventy-two, and the
        tables shrink with it.

        `frames` hands in a `frames(data)` the caller has already computed for the same `data`;
        leaving it out recomputes it, which is what every caller outside the per-step loop wants.
        """

        s, z, p, zt = self.frames(data) if frames is None else frames
        if rows is not None:
            s, z = s[rows], z[rows]
            own = self.socket_body[rows]
        else:
            own = self.socket_body
        if cols is not None:
            p, zt = p[cols], zt[cols]
            other = self.stud_body[cols]
        else:
            other = self.stud_body
        v = p[None, :, :] - s[:, None, :]
        depth = (v * z[:, None, :]).sum(axis=2)
        perp = v - depth[:, :, None] * z[:, None, :]
        lateral = np.sqrt((perp * perp).sum(axis=2))
        cos = np.clip(z @ zt.T, -1.0, 1.0)
        tilt = np.arccos(np.abs(cos))
        valid = own[:, None] != other[None, :]
        return depth, lateral, tilt, valid

    def pair_depth(
        self, frames: tuple, sockets: np.ndarray, studs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Depth and lateral error for a list of already-matched (socket, stud) pairs."""

        s, z, p, _ = frames
        v = p[studs] - s[sockets]
        zk = z[sockets]
        depth = (v * zk).sum(axis=1)
        perp = v - depth[:, None] * zk
        return depth, np.sqrt((perp * perp).sum(axis=1))


def equality_forces(model: mujoco.MjModel, data: mujoco.MjData, n_rows: int = 3) -> np.ndarray:
    """Constraint force of every equality, as `(model.neq, n_rows)` in world-frame newtons.

    The efc rows are rebuilt from scratch on every step and `nefc` changes as constraints come
    and go, so the rows belonging to one equality have to be found again each time. Rows are
    selected by `efc_type == mjCNSTR_EQUALITY` and identified by `efc_id`, which holds the
    equality index. A connect equality contributes 3 rows and they are the world-frame x, y, z
    components of the force holding `pos1` onto `pos2`; a weld contributes 6, translation
    first. Inactive equalities contribute no rows and come back as zeros.
    """

    out = np.zeros((model.neq, n_rows), dtype=np.float64)
    if data.nefc == 0:
        return out
    eq_rows = np.nonzero(data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
    if eq_rows.size == 0:
        return out
    ids = data.efc_id[eq_rows]
    # efc rows are emitted in equality order, so the rows of one equality are contiguous and
    # `ids` is non-decreasing; the offset of a row inside its own block is then just its
    # position minus where that block starts.
    starts = np.searchsorted(ids, np.arange(model.neq))
    slot = np.arange(eq_rows.size) - starts[ids]
    keep = slot < n_rows
    out[ids[keep], slot[keep]] = data.efc_force[eq_rows[keep]]
    return out


def engagement(depth: np.ndarray, params: SnapParams, stud_h: float) -> np.ndarray:
    """The force-against-depth shape `g(d)`, in one of three families.

    This is the only place the shape of the insertion curve is defined. Nobody has put a real
    DUPLO brick on a force gauge yet, so which family is right is an open question (design
    report 01, open question 2) and `params.curve` picks one:

    - "ramp": `g = clip((d - engage) / (h - engage), 0, 1) ** exponent`. Resistance grows all
      the way in and peaks at full seat. This is the original law and stays the default.
    - "plateau": `g = 1` as soon as the stud is engaged. Resistance is whatever one stud can
      hold, from the moment it grips until it is home.
    - "detent": a plateau at `plateau_gain` that ramps to 1 at full seat, with a Gaussian bump
      of height `detent_gain` centred at `detent_depth`. This is the interference snap a real
      brick is supposed to have: a stud is hardest to push past the entry bump and slides more
      easily afterwards.

    All three are 0 until the stud is `engage_depth` past the socket mouth (a real stud's
    chamfered tip is not gripping anything yet) and all three give `g(stud_h) = 1`, so the
    holding force of a seated stud is the same whichever shape is used - only the work of
    getting it in there differs. The bump of the "detent" curve is narrow enough (0.6 mm at a
    tenth of its height, against a 4.5 mm stud) that it has died away long before full seat.

    Calibrating the curve therefore means fitting `curve` first, then `f_clutch`,
    `engage_depth` and whichever of `exponent` / `plateau_gain` / `detent_*` that shape uses.
    """

    span = max(stud_h - params.engage_depth, 1e-9)
    ramp = np.clip((depth - params.engage_depth) / span, 0.0, 1.0)
    live = np.asarray(depth, dtype=np.float64) >= params.engage_depth
    if params.curve == "ramp":
        return ramp ** params.exponent
    if params.curve == "plateau":
        return np.where(live, 1.0, 0.0)
    if params.curve != "detent":
        raise ValueError(f"unknown SnapParams.curve {params.curve!r}")
    # Half-width at 1/e of a Gaussian that is `detent_width` wide at about a tenth of its
    # height: exp(-(w/2 / sigma)^2) = 0.1 gives sigma = w / (2 * sqrt(ln 10)).
    sigma = max(params.detent_width, 1e-9) / (2.0 * math.sqrt(math.log(10.0)))
    base = params.plateau_gain + (1.0 - params.plateau_gain) * ramp ** params.exponent
    bump = params.detent_gain * np.exp(-(((depth - params.detent_depth) / sigma) ** 2))
    return np.where(live, base + bump, 0.0)
