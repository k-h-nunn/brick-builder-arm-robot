"""The cheap damage estimate: read the contacts, turn them into stresses, accumulate.

This is the training-time half of the guide's section 6. It adds no bodies, no constraints and
no geometry to the scene. It costs one pass over the contact list per step, everything it needs
is already in `mjData`, and it is entirely opt-in: a scene that never constructs a
:class:`DamageModel` behaves exactly as it does today.

What it is for
--------------

Stated plainly, from the guide's section 5: **this arm cannot crush, shear or punch a brick.**
Crushing a DUPLO 2x4 needs 13.2 kN, shearing a stud off needs about 1 kN, and the arm's
measured static press at the build plate is 10 to 14 N. What the arm *can* do is

1. scuff a brick wherever a sharp edge concentrates a few newtons,
2. permanently bend a wall, at 12 to 35 N of lateral load, in a jam where a misaligned stud
   wedges against a wall or the jaw pinches an edge rather than a face,
3. crack a wall in a collision, at 19 to 56 N, which it reaches only dynamically - about
   0.3 m/s while holding a brick against something solid,
4. wear the clutch out, over tens of thousands of cycles.

So "brick destruction" here means a wall going out of shape and a connection losing its grip,
not a brick exploding. That is worth simulating precisely because the loads that do it are the
same order as the loads a press policy will be tempted to use.

How it works
------------

At construction the model reads every geom name once and builds flat per-geom tables: which
brick it belongs to, which part it is (wall, tube, stud, roof), and the section properties of
that part. The generator's naming makes this possible without any lookup table, because it is
already self-describing: ``{brick}_col_wall_{side}``, ``{brick}_col_tube_{k}``,
``{brick}_col_stud_{i}_{j}``, ``{brick}_col_roof``.

Per call, :meth:`DamageModel.update` gathers every contact force in one vectorised pass,
rotates it into each involved brick's own frame, evaluates the six stress formulas of the
guide's section 6.1, and reduces to one worst stress ratio per brick. There is no Python loop
over contacts.

The weakest assumption in the whole estimate is the load-spread rule ``b_eff = 2L + w_contact``
-- a 45 degree spread from the contact up to the roof. It is an approximation, it is the reason
the wall-bending threshold is a band (12 to 35 N) rather than a number, and the guide's section
7.7 is the destructive test on three real bricks that would narrow it. It is said here rather
than buried, because everything downstream inherits it.

Two more simplifications belong in the same paragraph rather than in a footnote:

- **Simultaneous contacts on one structural member are combined by maximum, not superposed.**
  Every reduction from per-contact rows to a per-brick number is ``np.maximum``, so an interior
  stud touching two tubes at once (which the geometry says happens: 1.5 contacts per stud on
  average) is scored by the worse of the two contacts, not by their sum. Superposition would be
  the right answer only if the two forces were colinear, and in the diagonal grid they are
  roughly 90 degrees apart, so summing their magnitudes would over-state the load about as
  often as the maximum under-states it. The maximum is also what keeps a brick resting on a
  plate from accumulating damage for doing nothing.
- **A load has to last to count.** Anything above an allowable must persist for
  :attr:`DamageThresholds.persist_s` (10 ms by default) before it can move a brick's permanent
  state. Instantaneous peaks are still recorded, unfiltered, every step. See
  :attr:`DamageThresholds.persist_s` for why 10 ms and not some other number.

Damage and training
-------------------

The policy must **not** observe the damage state: a real robot cannot see it. Damage belongs in
the reward (as a penalty on the *increment*, so a policy is punished at the moment it does harm
rather than for the rest of the episode), in the failure classes, and in termination on
``cracked`` or worse. What a real robot *can* observe, and what should be in the observation if
anything is, is the servo's own ``Present_Load`` register.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from brick_builder.bricks import materials as mat
from brick_builder.bricks.generator import BrickHandle
from brick_builder.bricks.snap_api import equality_forces
from brick_builder.bricks.spec import BrickSpec

# Part codes, used as an index into the per-geom tables.
PART_OTHER = 0
PART_WALL = 1
PART_TUBE = 2
PART_STUD = 3
PART_ROOF = 4
PART_NAMES = ("other", "wall", "tube", "stud", "roof")

#: States, in order. Each one latches: a brick never gets better.
STATES = ("intact", "marked", "loosened", "cracked", "broken")
_STATE_RANK = {s: i for i, s in enumerate(STATES)}

#: The failure modes the estimate evaluates, in the order of the guide's section 6.1 table.
MODES = (
    "wall_bending",
    "stud_shear",
    "stud_bending",
    "tube_ovalisation",
    "roof_bending",
    "marking",
)

#: Which modes count as structural bending, i.e. the ones that can crack or break a brick.
#: ``marking`` and ``stud_shear`` are deliberately not in here: marking is cosmetic and stud
#: shear needs 942 N, two orders of magnitude beyond anything this arm can produce.
BENDING_MODES = ("wall_bending", "stud_bending", "roof_bending")

# Column indices into the per-row stress table, resolved once so the hot loop never calls
# MODES.index().
_M_WALL = MODES.index("wall_bending")
_M_STUD_SHEAR = MODES.index("stud_shear")
_M_STUD_BEND = MODES.index("stud_bending")
_M_TUBE = MODES.index("tube_ovalisation")
_M_ROOF = MODES.index("roof_bending")
_M_MARK = MODES.index("marking")
_BENDING_COLS = np.array([MODES.index(m) for m in BENDING_MODES])
_N_MODES = len(MODES)


@dataclass(frozen=True)
class DamageThresholds:
    """Allowable stresses and the state boundaries, all in SI.

    ``allowable_scale`` multiplies every stress allowable at once, which is the randomization
    the guide's section 6.3 asks for: +-30 %, covering the material table's own spread. A policy
    that only works with one particular wall strength has learned the wrong thing.

    The state boundaries:

    - ``marked`` when a contact pressure passes the yield stress anywhere. Visual only.
    - ``loosened`` when the clutch multiplier falls below ``loosened_clutch``.
    - ``cracked`` when any bending mode passes the flexural strength.
    - ``broken`` when any bending mode passes ``broken_ratio`` times the flexural strength, or
      when a breakable brick releases a weld.

    ``persist_s`` is the guard against latching a solver transient, and it is the one number
    here that is about the *simulator* rather than about the plastic. ``marked``, ``cracked``
    and ``broken`` all latch permanently, by design, so that a policy is punished at the moment
    it does harm. That makes them defenceless against a one- or two-step force spike: a contact
    solver resolving an over-deep interpenetration can report tens of newtons for a single step,
    and without a guard the brick would carry "broken" for the rest of the episode on the
    strength of a numerical artefact. Two physical readings set the 10 ms:

    - **Duration.** The gentlest collision the guide's section 4.8 says can crack a wall is
      0.5 kg of tool at 0.3 m/s against an 8 kN/m wall. That is a half-sine contact lasting
      ``pi sqrt(m/k)`` = 25 ms, of which about 14 ms is spent above the permanent-set load. A
      real event that can hurt a brick is therefore tens of milliseconds long, while a solver
      transient at a 2 ms step is one or two steps.
    - **Impulse.** That same collision carries ``m v`` = 150 mN*s. A one-step transient of a
      ram that is moving at millimetres per second carries a few mN*s - about 2 % of it. ABS
      does not yield or crack on 2 % of the impulse that is only just enough to crack it.

    ``loosened`` is deliberately *not* guarded: it comes from the fatigue and relaxation
    accumulators, which are already integrals over time and cannot be moved by one step.
    :meth:`DamageModel.mark_broken` is not guarded either, because it is an explicit statement
    from outside that a piece has come apart.
    """

    yield_Pa: float = 44e6
    flexural_Pa: float = 70e6
    shear_Pa: float = 26e6
    allowable_scale: float = 1.0
    loosened_clutch: float = 0.8
    broken_ratio: float = 1.5
    edge_radius: float = 0.0002
    cycle_on: float = 0.05
    cycle_off: float = 0.02
    persist_s: float = 0.010
    #: Stress concentration at the wall-to-roof fillet, applied to the wall-bending mode only.
    #: 1.0 is the nominal-stress default; :data:`~brick_builder.bricks.materials.WALL_KT_RANGE` is
    #: the 1.0 to 2.0 a moulded internal corner plausibly has, and the error is one-sided - a
    #: real wall is weaker than this model says, never stronger. Guide section 4.3.
    wall_kt: float = 1.0

    @staticmethod
    def from_material(
        material: mat.Material = mat.ABS_BRICK_23C, scale: float = 1.0
    ) -> "DamageThresholds":
        """Allowables taken from a :class:`~brick_builder.bricks.materials.Material`."""

        return DamageThresholds(
            yield_Pa=material.tensile_yield.nominal,
            flexural_Pa=material.flexural_strength.nominal,
            shear_Pa=material.shear_strength.nominal,
            allowable_scale=scale,
        )

    def allowable(self, mode: str) -> float:
        """The allowable stress [Pa] for one mode of :data:`MODES`."""

        if mode in ("wall_bending", "stud_bending", "roof_bending"):
            base = self.flexural_Pa
        elif mode == "stud_shear":
            base = self.shear_Pa
        else:  # tube_ovalisation, marking
            base = self.yield_Pa
        return base * self.allowable_scale


@dataclass
class BrickDamage:
    """Everything the estimate knows about one brick. All stresses Pa, forces N."""

    name: str
    state: str = "intact"
    peak_ratio: float = 0.0
    peak_mode: str = ""
    peak_wall_stress_Pa: float = 0.0
    peak_lateral_N: float = 0.0
    peak_normal_N: float = 0.0
    peak_connection_N: float = 0.0
    peak_by_mode: dict = field(default_factory=lambda: {m: 0.0 for m in MODES})
    fatigue: float = 0.0
    relax: float = 0.0
    cycles: int = 0
    clutch_multiplier: float = 1.0
    #: Longest unbroken time [s] this brick has spent past an allowable. It is the number that
    #: separates a real overload from a solver transient, and a reader of ``peak_ratio`` should
    #: look at it before believing that peak meant anything.
    longest_overload_s: float = 0.0
    events: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "peak_ratio": self.peak_ratio,
            "peak_mode": self.peak_mode,
            "peak_wall_stress_MPa": self.peak_wall_stress_Pa / 1e6,
            "peak_lateral_N": self.peak_lateral_N,
            "peak_normal_N": self.peak_normal_N,
            "peak_connection_N": self.peak_connection_N,
            "peak_by_mode_MPa": {k: v / 1e6 for k, v in self.peak_by_mode.items()},
            "fatigue": self.fatigue,
            "relax": self.relax,
            "cycles": self.cycles,
            "clutch_multiplier": self.clutch_multiplier,
            "longest_overload_s": self.longest_overload_s,
            "n_events": len(self.events),
        }


@dataclass(frozen=True)
class DamageEvent:
    """One state transition, recorded so an evaluation run can say what happened and when.

    ``held_s`` is how long the overload had already lasted when the state changed, so a reader
    can see that the transition was not one step of solver noise. It is
    :attr:`DamageThresholds.persist_s` for anything the contact pass triggered, and zero for a
    state latched from outside by :meth:`DamageModel.mark_broken`.
    """

    time: float
    brick: str
    state: str
    mode: str
    stress_Pa: float
    ratio: float
    force_N: float
    held_s: float = 0.0


@dataclass
class DamageState:
    """Everything :class:`DamageModel` carries between steps, for save and restore.

    The arrays are the whole of it: the model holds no other mutable state that affects a
    future step, so restoring this plus ``mjSTATE_INTEGRATION`` reproduces a run exactly.
    """

    peak_ratio: np.ndarray
    peak_mode: np.ndarray
    peak_wall: np.ndarray
    peak_lateral: np.ndarray
    peak_normal: np.ndarray
    peak_connection: np.ndarray
    peak_by_mode: np.ndarray
    fatigue: np.ndarray
    relax: np.ndarray
    cycles: np.ndarray
    state_rank: np.ndarray
    cycle_open: np.ndarray
    cycle_peak: np.ndarray
    hold: np.ndarray
    peak_hold: np.ndarray
    n_calls: int
    events: list

    def copy(self) -> "DamageState":
        return DamageState(
            peak_ratio=self.peak_ratio.copy(),
            peak_mode=self.peak_mode.copy(),
            peak_wall=self.peak_wall.copy(),
            peak_lateral=self.peak_lateral.copy(),
            peak_normal=self.peak_normal.copy(),
            peak_connection=self.peak_connection.copy(),
            peak_by_mode=self.peak_by_mode.copy(),
            fatigue=self.fatigue.copy(),
            relax=self.relax.copy(),
            cycles=self.cycles.copy(),
            state_rank=self.state_rank.copy(),
            cycle_open=self.cycle_open.copy(),
            cycle_peak=self.cycle_peak.copy(),
            hold=self.hold.copy(),
            peak_hold=self.peak_hold.copy(),
            n_calls=self.n_calls,
            events=list(self.events),
        )


#: Columns of the persistence-guard timer, in state order. ``loosened`` is absent on purpose:
#: it comes from time integrals that a single step cannot move.
GUARDED_STATES = ("marked", "cracked", "broken")
_GUARD_RANK = np.array([_STATE_RANK[s] for s in GUARDED_STATES])


class DamageModel:
    """Per-brick damage accumulated from contact forces and connection forces.

    Attach it to a compiled scene and call :meth:`update` once per physics step (or once per
    control tick, with ``dt`` set to the tick length - the accumulation is in stress ratio and
    time, so either works as long as ``dt`` is honest).

    ``snap`` is optional. With it, the model also reads the snap model's connection forces and
    can scale a damaged brick's clutch through :meth:`apply_consequences`. Without it, the
    estimate is contact-only, which is everything except the connection force and the
    consequence.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bricks: list[BrickHandle],
        snap=None,
        material: mat.Material = mat.ABS_BRICK_23C,
        thresholds: DamageThresholds | None = None,
        spec: BrickSpec | None = None,
        seed: int | None = None,
        randomize: bool = False,
        track_connections: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.bricks = list(bricks)
        self.snap = snap
        # Reading the connection forces costs one pass over the constraint rows, which is the
        # single most expensive thing this model does on a lightly loaded scene. Turn it off
        # when the per-brick connection force is not wanted; everything else is unaffected.
        self.track_connections = track_connections
        self.rng = np.random.default_rng(seed)
        self.material = material.sample(self.rng) if randomize else material
        if thresholds is None:
            scale, kt = 1.0, 1.0
            if randomize:
                ranges = mat.randomization_ranges()
                lo, _, hi = ranges["allowable_scale"]
                scale = float(self.rng.uniform(lo, hi))
                lo, _, hi = ranges["wall_kt"]
                kt = float(self.rng.uniform(lo, hi))
            thresholds = replace(
                DamageThresholds.from_material(self.material, scale), wall_kt=kt
            )
        self.thresholds = thresholds
        self._allow = np.array([thresholds.allowable(m) for m in MODES])
        self.fatigue_model = self.material.fatigue
        self.spec = spec if spec is not None else (bricks[0].spec if bricks else None)
        self.names = [h.name for h in self.bricks]
        self._n = len(self.bricks)
        self._build_geom_tables()
        self._build_equality_tables()
        self._build_collet_dofs()
        self.reset()

    # -- construction ------------------------------------------------------

    def _build_geom_tables(self) -> None:
        """One pass over the geom names, turned into flat arrays indexed by geom id.

        Section properties are attached per geom here rather than looked up per contact, so the
        per-step work is pure gathering. A geom that belongs to no brick gets ``brick = -1`` and
        is skipped by every mode.
        """

        m = self.model
        n = m.ngeom
        self.g_brick = np.full(n, -1, dtype=np.int32)
        self.g_part = np.zeros(n, dtype=np.int8)
        self.g_wall_axis = np.full(n, -1, dtype=np.int8)
        self.g_length = np.zeros(n)        # wall length / member length [m]
        self.g_thickness = np.zeros(n)     # wall or ring thickness [m]
        self.g_area = np.zeros(n)          # root cross-section area [m^2]
        self.g_modulus = np.zeros(n)       # root section modulus [m^3]
        self.g_radius = np.zeros(n)        # mean radius, for the ring formula [m]
        self.g_width = np.zeros(n)         # contact-width proxy [m]
        self.brick_body = np.zeros(self._n, dtype=np.int32)
        self.brick_spec: list[BrickSpec] = []

        # `int()` on both sides is not decoration. `m.geom_type[gi]` is a numpy integer and a
        # pybind enum does not compare equal to one through `in`, so writing the round-geom
        # test as `gtype in (mjGEOM_SPHERE, ...)` silently takes the fallback branch for every
        # stud, tube and rod in the scene and gives them a bounding-sphere radius instead of a
        # diameter - 5.2 mm instead of 9.4 mm on a DUPLO stud, which lands directly in b_eff.
        round_types = {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
        }
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        for gi in range(n):
            size = m.geom_size[gi]
            gtype = int(m.geom_type[gi])
            if gtype == box_type:
                # The median half-extent is the best single "how wide is this face" number: the
                # smallest is the plate thickness and the largest is its span.
                self.g_width[gi] = 2.0 * float(np.sort(size)[1])
            elif gtype in round_types:
                self.g_width[gi] = 2.0 * float(size[0])
            else:
                self.g_width[gi] = float(m.geom_rbound[gi])

        for bi, handle in enumerate(self.bricks):
            spec = handle.spec
            self.brick_spec.append(spec)
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, handle.name)
            self.brick_body[bi] = body
            sec = mat.sections(spec, handle.nx, handle.ny)
            prefix = f"{handle.name}_col_"
            for gi in range(n):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
                if not name.startswith(prefix):
                    continue
                tail = name[len(prefix):]
                self.g_brick[gi] = bi
                if tail.startswith("wall_"):
                    side = tail[len("wall_"):]
                    # front/back are the walls whose own normal is the brick's y axis.
                    is_y = side in ("front", "back")
                    key = "wall_front_back" if is_y else "wall_left_right"
                    s = sec[key]
                    self.g_part[gi] = PART_WALL
                    self.g_wall_axis[gi] = 1 if is_y else 0
                    self.g_length[gi] = s.length
                    self.g_thickness[gi] = s.thickness
                    self.g_area[gi] = s.area
                    self.g_modulus[gi] = s.modulus
                elif tail.startswith("tube_"):
                    s = sec["tube"]
                    self.g_part[gi] = PART_TUBE
                    self.g_length[gi] = s.length
                    self.g_thickness[gi] = s.thickness
                    self.g_area[gi] = s.area
                    self.g_modulus[gi] = s.modulus
                    self.g_radius[gi] = (spec.tube_od + spec.tube_id) / 4.0
                elif tail.startswith("stud_"):
                    s = sec["stud"]
                    self.g_part[gi] = PART_STUD
                    self.g_length[gi] = spec.stud_h
                    self.g_thickness[gi] = s.thickness
                    self.g_area[gi] = s.area
                    self.g_modulus[gi] = s.modulus
                    self.g_radius[gi] = spec.stud_d / 2.0
                elif tail.startswith("roof"):
                    s = sec["roof"]
                    self.g_part[gi] = PART_ROOF
                    self.g_length[gi] = s.length
                    self.g_thickness[gi] = spec.roof
                    self.g_area[gi] = s.area
                    self.g_modulus[gi] = s.modulus
                else:
                    self.g_part[gi] = PART_OTHER

        # Per-brick scalars, gathered by brick index in the hot loop instead of looked up.
        self._wall_h = np.zeros(self._n)
        self._roof_t = np.zeros(self._n)
        self._height = np.zeros(self._n)
        self._stud_h = np.zeros(self._n)
        for bi, spec in enumerate(self.brick_spec):
            self._wall_h[bi] = spec.height - spec.roof
            self._roof_t[bi] = spec.roof
            self._height[bi] = spec.height
            self._stud_h[bi] = spec.stud_h

    def _build_equality_tables(self) -> None:
        """Map every equality whose first object is a collet body onto the brick that owns it.

        The snap model names its collet bodies ``{brick}_collet_{suffix}``, so this needs no
        access to the snap model's internals - only the compiled model and the brick names.
        """

        m = self.model
        self.eq_brick = np.full(m.neq, -1, dtype=np.int32)
        for eq in range(m.neq):
            obj = int(m.eq_obj1id[eq])
            if obj < 0 or obj >= m.nbody:
                continue
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, obj) or ""
            for bi, name in enumerate(self.names):
                if bname.startswith(f"{name}_collet"):
                    self.eq_brick[eq] = bi
                    break
        self._has_eq = bool(np.any(self.eq_brick >= 0))

    def _build_collet_dofs(self) -> None:
        """Collet dofs per brick, found by the snap model's public joint-naming convention.

        :meth:`apply_consequences` scales ``model.dof_frictionloss`` on these, which is the only
        public field a damaged brick's clutch can be reduced through today. The names are
        derived from the brick handles, not read out of the snap model, so this works with any
        model that follows the same convention and does nothing at all when there is none.
        """

        m = self.model
        self.collet_dofs: list[np.ndarray] = []
        for handle in self.bricks:
            dofs: list[int] = []
            for site in handle.socket_sites:
                suffix = site[len(handle.name) + len("_socket_"):]
                for jname in (
                    f"{handle.name}_colletj_{suffix}",
                    f"{handle.name}_colletx_{suffix}",
                    f"{handle.name}_collety_{suffix}",
                ):
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        dofs.append(int(m.jnt_dofadr[jid]))
            self.collet_dofs.append(np.asarray(dofs, dtype=np.int32))

    # -- state -------------------------------------------------------------

    def reset(self) -> None:
        """Clear every accumulator. The material and the thresholds are kept."""

        n, nm = self._n, len(MODES)
        self._peak_ratio = np.zeros(n)
        self._peak_mode = np.full(n, -1, dtype=np.int8)
        self._peak_wall = np.zeros(n)
        self._peak_lateral = np.zeros(n)
        self._peak_normal = np.zeros(n)
        self._peak_connection = np.zeros(n)
        self._peak_by_mode = np.zeros((n, nm))
        self._fatigue = np.zeros(n)
        self._relax = np.zeros(n)
        self._cycles = np.zeros(n, dtype=np.int64)
        self._rank = np.zeros(n, dtype=np.int8)
        self._cycle_open = np.zeros(n, dtype=bool)
        self._cycle_peak = np.zeros(n)
        # How long each guarded condition has held without a break [s], and the longest any of
        # them has ever held, which is the evidence that an overload was not a transient.
        self._hold = np.zeros((n, len(GUARDED_STATES)))
        self._peak_hold = np.zeros(n)
        self._n_calls = 0
        self._multiplier = np.ones(n)
        self._reward_rank = np.zeros(n, dtype=np.int8)
        self.events: list[DamageEvent] = []

    def get_state(self) -> DamageState:
        return DamageState(
            peak_ratio=self._peak_ratio.copy(),
            peak_mode=self._peak_mode.copy(),
            peak_wall=self._peak_wall.copy(),
            peak_lateral=self._peak_lateral.copy(),
            peak_normal=self._peak_normal.copy(),
            peak_connection=self._peak_connection.copy(),
            peak_by_mode=self._peak_by_mode.copy(),
            fatigue=self._fatigue.copy(),
            relax=self._relax.copy(),
            cycles=self._cycles.copy(),
            state_rank=self._rank.copy(),
            cycle_open=self._cycle_open.copy(),
            cycle_peak=self._cycle_peak.copy(),
            hold=self._hold.copy(),
            peak_hold=self._peak_hold.copy(),
            n_calls=self._n_calls,
            events=list(self.events),
        )

    def set_state(self, s: DamageState) -> None:
        self._peak_ratio = s.peak_ratio.copy()
        self._peak_mode = s.peak_mode.copy()
        self._peak_wall = s.peak_wall.copy()
        self._peak_lateral = s.peak_lateral.copy()
        self._peak_normal = s.peak_normal.copy()
        self._peak_connection = s.peak_connection.copy()
        self._peak_by_mode = s.peak_by_mode.copy()
        self._fatigue = s.fatigue.copy()
        self._relax = s.relax.copy()
        self._cycles = s.cycles.copy()
        self._rank = s.state_rank.copy()
        self._cycle_open = s.cycle_open.copy()
        self._cycle_peak = s.cycle_peak.copy()
        self._hold = s.hold.copy()
        self._peak_hold = s.peak_hold.copy()
        self._n_calls = s.n_calls
        self.events = list(s.events)

    # -- the per-step pass -------------------------------------------------

    def contact_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(geoms, positions, world forces)`` for every contact, vectorised.

        With MuJoCo's elliptic friction cone - which is what every scene in this project uses -
        the contact-frame force is exactly ``efc_force`` at the contact's ``efc_address``, so
        all of them can be gathered in one fancy-index instead of ``ncon`` calls into
        ``mj_contactForce``. With a pyramidal cone the components are a different basis, so the
        fallback loop is taken and this is slower; nothing in this repository configures that,
        but a caller might.
        """

        d, m = self.data, self.model
        ncon = d.ncon
        if ncon == 0:
            return (np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3)), np.zeros((0, 3)))
        geoms = np.asarray(d.contact.geom[:ncon], dtype=np.int32).reshape(ncon, 2)
        pos = np.asarray(d.contact.pos[:ncon], dtype=np.float64).reshape(ncon, 3)
        frame = np.asarray(d.contact.frame[:ncon], dtype=np.float64).reshape(ncon, 3, 3)
        if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
            adr = np.asarray(d.contact.efc_address[:ncon], dtype=np.int64)
            dim = np.asarray(d.contact.dim[:ncon], dtype=np.int64)
            efc = np.asarray(d.efc_force[: d.nefc], dtype=np.float64)
            if efc.size == 0:
                return (geoms, pos, np.zeros((ncon, 3)))
            # Gather the (up to) three rows of every contact in one indexed read. Rows a
            # contact does not have - a frictionless condim 1 contact, or one that is not in
            # the constraint list at all - are zeroed by the mask rather than skipped, which
            # keeps this branch-free.
            cols = np.arange(3)
            take = np.clip(adr[:, None] + cols[None, :], 0, efc.size - 1)
            local = efc[take]
            local *= ((cols[None, :] < dim[:, None]) & (adr[:, None] >= 0))
        else:  # pragma: no cover - no scene in this repository uses a pyramidal cone
            local = np.zeros((ncon, 3))
            buf = np.zeros(6)
            for i in range(ncon):
                mujoco.mj_contactForce(m, d, i, buf)
                local[i] = buf[:3]
        world = np.einsum("nij,ni->nj", frame, local)
        return geoms, pos, world

    def update(self, dt: float | None = None) -> None:
        """One damage pass. Call after ``mj_step`` (and after the snap model's own update).

        ``dt`` defaults to the model's timestep; pass the control period instead when calling
        once per control tick rather than once per physics step.
        """

        d, m = self.data, self.model
        step = m.opt.timestep if dt is None else dt
        self._n_calls += 1
        n = self._n
        if n == 0:
            return

        stress = np.zeros((n, len(MODES)))
        lateral = np.zeros(n)
        normal = np.zeros(n)

        geoms, pos, world = self.contact_forces()
        if geoms.shape[0]:
            bi = self.g_brick[geoms]                       # (ncon, 2)
            keep = np.nonzero((bi >= 0).any(axis=1))[0]
            if keep.size:
                self._accumulate_contacts(
                    geoms[keep], pos[keep], world[keep], bi[keep], stress, lateral, normal
                )

        conn = np.zeros(n)
        if self._has_eq and self.track_connections and d.nefc:
            f_eq = equality_forces(m, d, n_rows=3)
            mag = np.sqrt((f_eq * f_eq).sum(axis=1))
            for b in range(n):
                sel = self.eq_brick == b
                if sel.any():
                    conn[b] = float(mag[sel].max())

        self._accumulate(stress, lateral, normal, conn, step)

    def _accumulate_contacts(
        self,
        geoms: np.ndarray,
        pos: np.ndarray,
        world: np.ndarray,
        bi: np.ndarray,
        stress: np.ndarray,
        lateral: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        """Evaluate the six stress modes for every (contact, brick side) pair, vectorised.

        Each contact contributes up to two rows, one per side, because both bricks in a
        brick-on-brick contact are loaded by it. The six modes are written into one
        ``(rows, modes)`` table and reduced to per-brick maxima at the end. The reduction is a
        short Python loop over bricks rather than ``np.maximum.at`` on purpose: a scene has a
        handful of bricks and dozens of contacts, and ``ufunc.at`` is the slowest way in numpy
        to scatter a maximum.
        """

        d = self.data
        xmat = d.xmat.reshape(-1, 3, 3)
        k0 = geoms.shape[0]
        idx = np.arange(k0)
        side_c = np.concatenate([idx, idx])
        side_g = np.concatenate([geoms[:, 0], geoms[:, 1]])
        side_b = np.concatenate([bi[:, 0], bi[:, 1]])
        ok = side_b >= 0
        side_c, side_g, side_b = side_c[ok], side_g[ok], side_b[ok]
        if side_c.size == 0:
            return

        body = self.brick_body[side_b]
        rot = xmat[body]                                   # (k, 3, 3), rows are body axes
        rel = pos[side_c] - d.xpos[body]
        local_pos = np.einsum("kij,ki->kj", rot, rel)      # R^T (p - x)
        local_f = np.einsum("kij,ki->kj", rot, world[side_c])
        f_abs = np.abs(local_f)
        f_lat = np.hypot(f_abs[:, 0], f_abs[:, 1])
        f_mag = np.sqrt((local_f * local_f).sum(axis=1))

        part = self.g_part[side_g]
        width = np.minimum(self.g_width[geoms[side_c, 0]], self.g_width[geoms[side_c, 1]])
        rows = np.zeros((side_c.size, _N_MODES + 2))

        # -- wall bending ---------------------------------------------------
        # sigma = 6 F L / (b_eff t^2), F the force perpendicular to the wall's own plane, L the
        # distance from the contact up to the roof, b_eff = min(2L + w_contact, wall length).
        # The perpendicular component is the point: a purely vertical load on a wall's bottom
        # face is compression, not bending, and must not raise this stress at all.
        w = np.nonzero(part == PART_WALL)[0]
        if w.size:
            axis = self.g_wall_axis[side_g[w]].astype(np.intp)
            f_perp = f_abs[w, axis]
            lever = np.clip(self._wall_h[side_b[w]] - local_pos[w, 2], 1e-4, None)
            b_eff = np.minimum(2.0 * lever + width[w], self.g_length[side_g[w]])
            t = self.g_thickness[side_g[w]]
            rows[w, _M_WALL] = (self.thresholds.wall_kt * 6.0 * f_perp * lever
                                / np.maximum(b_eff * t**2, 1e-18))

        # -- stud root shear and stud root bending --------------------------
        s_idx = np.nonzero(part == PART_STUD)[0]
        if s_idx.size:
            area = np.maximum(self.g_area[side_g[s_idx]], 1e-12)
            rows[s_idx, _M_STUD_SHEAR] = f_lat[s_idx] / area
            # Lever: how far up the stud the load acts, measured from the stud's own root.
            h = np.clip(
                local_pos[s_idx, 2] - self._height[side_b[s_idx]],
                0.0, self.g_length[side_g[s_idx]],
            )
            z_mod = np.maximum(self.g_modulus[side_g[s_idx]], 1e-18)
            rows[s_idx, _M_STUD_BEND] = f_lat[s_idx] * h / z_mod

        # -- tube ovalisation -----------------------------------------------
        # sigma = 0.318 F R (t/2) / I with I = h t^3 / 12 over the engaged band.
        t_idx = np.nonzero(part == PART_TUBE)[0]
        if t_idx.size:
            r = self.g_radius[side_g[t_idx]]
            th = self.g_thickness[side_g[t_idx]]
            inertia = np.maximum(self._stud_h[side_b[t_idx]] * th**3 / 12.0, 1e-20)
            rows[t_idx, _M_TUBE] = 0.318 * f_lat[t_idx] * r * (th / 2.0) / inertia

        # -- roof bending near an edge --------------------------------------
        r_idx = np.nonzero(part == PART_ROOF)[0]
        if r_idx.size:
            half = self.g_length[side_g[r_idx]] / 2.0
            edge = np.maximum(half - np.abs(local_pos[r_idx, 0]), 1e-4)
            b_eff = np.maximum(width[r_idx], 1e-4)
            t = np.maximum(self._roof_t[side_b[r_idx]], 1e-6)
            rows[r_idx, _M_ROOF] = 6.0 * f_abs[r_idx, 2] * edge / (b_eff * t**2)

        # -- marking ---------------------------------------------------------
        # Marking and cracking are both questions of contact WIDTH, not of force: the same 17 N
        # grip spread over 3 mm instead of 30 mm raises a wall's bending stress from 13 MPa to
        # past rupture. The projected patch of an edge contact is the contact width times twice
        # the assumed edge radius.
        rows[:, _M_MARK] = f_mag / np.maximum(width * 2.0 * self.thresholds.edge_radius, 1e-12)
        rows[:, _N_MODES] = f_lat
        rows[:, _N_MODES + 1] = f_mag

        # -- reduce to one worst value per brick ------------------------------
        # One sort plus one segmented maximum, rather than a mask and three reductions per
        # brick: with six bricks that was three dozen numpy calls on arrays of a few dozen
        # rows, which is all call overhead and no arithmetic.
        if self._n == 1:
            worst = rows.max(axis=0)
        else:
            order = np.argsort(side_b, kind="stable")
            sorted_b = side_b[order]
            uniq, first = np.unique(sorted_b, return_index=True)
            red = np.maximum.reduceat(rows[order], first, axis=0)
            worst = np.zeros((self._n, _N_MODES + 2))
            worst[uniq] = red
        np.maximum(stress, worst[..., :_N_MODES].reshape(stress.shape), out=stress)
        np.maximum(lateral, worst[..., _N_MODES].reshape(lateral.shape), out=lateral)
        np.maximum(normal, worst[..., _N_MODES + 1].reshape(normal.shape), out=normal)

    def _accumulate(
        self,
        stress: np.ndarray,
        lateral: np.ndarray,
        normal: np.ndarray,
        conn: np.ndarray,
        dt: float,
    ) -> None:
        """Turn this step's stresses into ratios, then into wear, cycles and states.

        Most steps of a training episode carry no meaningful load at all - a brick sitting on a
        plate, or held in the air - so the expensive half of this (fatigue, relaxation, cycle
        counting and the state ladder) is behind an early exit. Skipping it is exact, not an
        approximation: with every ratio below the cycle threshold no accumulator can move, no
        state can rise (the states latch and only ever go up), and every persistence timer is
        reset on the way out, which is what an early exit has to do to stay honest - a load
        that stopped has not persisted.
        """

        ratio = stress / self._allow[None, :]
        worst = ratio.max(axis=1)

        np.maximum(self._peak_by_mode, stress, out=self._peak_by_mode)
        # Peak wall stress is tracked on its own because it is the number the guide's whole
        # failure analysis turns on, and an episode record wants it without a mode lookup.
        np.maximum(self._peak_wall, stress[:, _M_WALL], out=self._peak_wall)
        np.maximum(self._peak_lateral, lateral, out=self._peak_lateral)
        np.maximum(self._peak_normal, normal, out=self._peak_normal)
        np.maximum(self._peak_connection, conn, out=self._peak_connection)
        if (worst > self._peak_ratio).any():
            better = worst > self._peak_ratio
            self._peak_mode[better] = ratio.argmax(axis=1)[better].astype(np.int8)
            np.maximum(self._peak_ratio, worst, out=self._peak_ratio)

        f = self.fatigue_model
        th = self.thresholds
        peak_worst = float(worst.max())
        if peak_worst <= th.cycle_off and not self._cycle_open.any():
            if self._hold.any():
                self._hold[:] = 0.0
            return

        # Stress relaxation: sustained load above a threshold, integrated in time.
        self._relax += np.maximum(worst - f.relax_threshold, 0.0) * dt / f.tau_relax_s

        # Cycle counting with hysteresis. A cycle opens when the worst ratio rises past
        # `cycle_on` and closes when it falls back below `cycle_off`; the Palmgren-Miner
        # increment uses the peak the excursion reached, which is what an S-N curve wants.
        self._cycle_open |= worst > th.cycle_on
        np.maximum(self._cycle_peak, np.where(self._cycle_open, worst, 0.0),
                   out=self._cycle_peak)
        closing = self._cycle_open & (worst < th.cycle_off)
        if closing.any():
            peak = self._cycle_peak[closing]
            self._fatigue[closing] += (peak / f.s_ref) ** f.exponent / f.cycles_ref
            self._cycles[closing] += 1
            self._cycle_open[closing] = False
            self._cycle_peak[closing] = 0.0

        multiplier = np.clip(
            1.0 - f.fatigue_weight * self._fatigue - f.relax_weight * self._relax,
            f.clutch_floor, 1.0,
        )
        self._multiplier = multiplier

        if peak_worst < 1.0 and (multiplier >= th.loosened_clutch).all():
            if self._hold.any():
                self._hold[:] = 0.0
            return

        # The persistence guard. A condition has to hold without a break for `persist_s` before
        # it is allowed to latch a permanent state; the timer is reset the moment it lapses, so
        # a one-step spike followed by nothing counts for nothing. The instantaneous peaks above
        # are untouched by this: they are recorded whether or not they lasted.
        bend = ratio[:, _BENDING_COLS].max(axis=1)
        cond = np.stack(
            [ratio[:, _M_MARK] >= 1.0, bend >= 1.0, bend >= th.broken_ratio], axis=1
        )
        self._hold = np.where(cond, self._hold + dt, 0.0)
        np.maximum(self._peak_hold, self._hold.max(axis=1), out=self._peak_hold)
        # `cond` has to be in the test as well as the timer: with persist_s = 0 the timer of a
        # brick under no load at all is 0.0, which would otherwise clear the threshold.
        held = cond & (self._hold >= th.persist_s - 1e-12)

        # States, each latching. Evaluated worst-first so one step can jump more than one rank.
        new_rank = np.zeros(self._n, dtype=np.int64)
        for col in range(len(GUARDED_STATES)):
            np.maximum(new_rank, np.where(held[:, col], _GUARD_RANK[col], 0), out=new_rank)
        np.maximum(new_rank,
                   np.where(multiplier < th.loosened_clutch, _STATE_RANK["loosened"], 0),
                   out=new_rank)
        risen = np.nonzero(new_rank > self._rank)[0]
        for i in risen:
            mode = MODES[int(np.argmax(ratio[i]))]
            self.events.append(
                DamageEvent(
                    time=float(self.data.time),
                    brick=self.names[i],
                    state=STATES[int(new_rank[i])],
                    mode=mode,
                    stress_Pa=float(stress[i, MODES.index(mode)]),
                    ratio=float(ratio[i].max()),
                    force_N=float(lateral[i]),
                    held_s=float(self._hold[i].max()),
                )
            )
        self._rank = np.maximum(self._rank, new_rank).astype(np.int8)

    # -- consequences ------------------------------------------------------

    def apply_consequences(self) -> None:
        """Scale each damaged brick's clutch, through public simulator fields only.

        The only public handle on a brick's clutch today is ``model.dof_frictionloss`` on that
        brick's collet dofs, which :class:`~brick_builder.bricks.snap.ColletSnap` rewrites from
        ``params.f_clutch`` on every one of its own updates. So this must be called **after**
        the snap model's ``update`` in the same tick, and it multiplies rather than assigns.

        That is a workaround, not an interface. What the snap model should grow is a public
        per-brick clutch scale that its own scheduling multiplies in - see the integration plan
        in the report. Until then this is correct but order-dependent, and a caller that forgets
        the ordering silently gets no consequence at all rather than a wrong one.

        A ``cracked`` brick's multiplier is additionally capped at 0.3, per the guide's state
        table.
        """

        mult = self._multiplier
        cracked = self._rank >= _STATE_RANK["cracked"]
        mult = np.where(cracked, np.minimum(mult, 0.3), mult)
        for bi, dofs in enumerate(self.collet_dofs):
            if dofs.size and mult[bi] < 1.0:
                self.model.dof_frictionloss[dofs] *= float(mult[bi])

    def mark_broken(self, brick: str, mode: str = "weld_release", force_N: float = 0.0) -> None:
        """Latch a brick to ``broken`` from outside, e.g. when a breakable weld lets go.

        This bypasses the persistence guard on purpose: the caller is not reporting a stress
        that might be numerical noise, it is reporting that a piece has come apart.
        """

        i = self.names.index(brick)
        if self._rank[i] < _STATE_RANK["broken"]:
            self._rank[i] = _STATE_RANK["broken"]
            self.events.append(
                DamageEvent(float(self.data.time), brick, "broken", mode, 0.0, 0.0, force_N)
            )

    # -- queries -----------------------------------------------------------

    def brick(self, name: str) -> BrickDamage:
        """The full damage record for one brick."""

        i = self.names.index(name)
        mult = self._multiplier
        return BrickDamage(
            name=name,
            state=STATES[int(self._rank[i])],
            peak_ratio=float(self._peak_ratio[i]),
            peak_mode=MODES[int(self._peak_mode[i])] if self._peak_mode[i] >= 0 else "",
            peak_wall_stress_Pa=float(self._peak_by_mode[i, MODES.index("wall_bending")]),
            peak_lateral_N=float(self._peak_lateral[i]),
            peak_normal_N=float(self._peak_normal[i]),
            peak_connection_N=float(self._peak_connection[i]),
            peak_by_mode={m: float(self._peak_by_mode[i, k]) for k, m in enumerate(MODES)},
            fatigue=float(self._fatigue[i]),
            relax=float(self._relax[i]),
            cycles=int(self._cycles[i]),
            clutch_multiplier=float(mult[i]),
            longest_overload_s=float(self._peak_hold[i]),
            events=[e for e in self.events if e.brick == name],
        )

    def states(self) -> dict[str, str]:
        """``brick name -> state``."""

        return {n: STATES[int(r)] for n, r in zip(self.names, self._rank)}

    def worst_state(self) -> str:
        """The worst state any brick is in, which is what a termination check wants."""

        return STATES[int(self._rank.max())] if self._n else "intact"

    def any_damage(self) -> bool:
        """True once any brick has left ``intact``."""

        return bool(self._n and self._rank.max() > 0)

    def summary(self) -> dict:
        """One dict for the whole episode, shaped for an evaluation record."""

        return {
            "worst_state": self.worst_state(),
            "n_damaged": int(np.count_nonzero(self._rank > 0)),
            "n_cracked": int(np.count_nonzero(self._rank >= _STATE_RANK["cracked"])),
            "peak_wall_stress_MPa": float(
                self._peak_by_mode[:, MODES.index("wall_bending")].max() / 1e6
            ) if self._n else 0.0,
            "peak_lateral_N": float(self._peak_lateral.max()) if self._n else 0.0,
            "peak_stress_ratio": float(self._peak_ratio.max()) if self._n else 0.0,
            "longest_overload_s": float(self._peak_hold.max()) if self._n else 0.0,
            "total_cycles": int(self._cycles.sum()),
            "calls": self._n_calls,
            "allowable_scale": self.thresholds.allowable_scale,
            "bricks": {n: self.brick(n).as_dict() for n in self.names},
            "events": [vars(e) for e in self.events],
        }

    def reward_penalty(self, weights: dict[str, float] | None = None) -> float:
        """A penalty for the damage done since the last call, for use in a reward.

        The guide's section 6.3 asks for a penalty on the damage *increment*, not on the state,
        so a policy is punished at the moment it does harm rather than for the rest of the
        episode. Weight it so that marking a brick costs about as much as a small positional
        error and cracking one costs more than failing the task: a policy should prefer not to
        place the brick at all over breaking it.
        """

        w = {"marked": 0.05, "loosened": 0.2, "cracked": 2.0, "broken": 5.0}
        if weights:
            w.update(weights)
        prev = self._reward_rank
        total = 0.0
        for i in range(self._n):
            for r in range(int(prev[i]) + 1, int(self._rank[i]) + 1):
                total += w.get(STATES[r], 0.0)
        self._reward_rank = self._rank.copy()
        return -total


def benchmark(model, data, bricks, snap=None, steps: int = 200) -> dict:
    """Microseconds per :meth:`DamageModel.update` call, measured on a live scene."""

    from time import perf_counter

    dm = DamageModel(model, data, bricks, snap=snap)
    for _ in range(20):
        mujoco.mj_step(model, data)
        if snap is not None:
            snap.update(model, data)
        dm.update()
    t0 = perf_counter()
    for _ in range(steps):
        dm.update()
    only_damage = (perf_counter() - t0) * 1e6 / steps
    t0 = perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if snap is not None:
            snap.update(model, data)
    step_cost = (perf_counter() - t0) * 1e6 / steps
    return {
        "n_bricks": len(bricks),
        "damage_us_per_call": only_damage,
        "step_us": step_cost,
        "ncon": int(data.ncon),
        "fraction_of_step": only_damage / max(step_cost, 1e-9),
    }


# ---------------------------------------------------------------------------
# A validation rig: a force-controlled ram pressing a (possibly misaligned) brick
# ---------------------------------------------------------------------------


#: Ram mass [kg] for :func:`build_press_rig`. The arm's own operational-space inertia at press
#: poses, ``1/(J M^-1 J^T)_zz``, measures 0.32 to 0.62 kg over the working region (guide 02c
#: section 4.8), so a brick on this ram is backed by the same inertia a brick in the jaw is.
PRESS_RAM_MASS_KG = 0.5

#: Dashpot on the ram's slide [N*s/m]. It is what makes the rig a press rather than a dropped
#: weight: the terminal speed under a commanded force is ``F / PRESS_RAM_DAMPING``, so even the
#: 45 N top of the sweep can only descend at 30 mm/s and the 2 N approach at 1.3 mm/s. It is
#: also about three times critical against the softest contact the rig can meet, so the ram
#: settles onto a contact without overshooting it. A real quasi-static press is a screw, which
#: is likewise not backdrivable; this is the same idea with a number attached.
PRESS_RAM_DAMPING = 1500.0

#: Constraint damping ratio for this rig's contacts, against the project default of 1.0. This
#: is the one setting that decides whether the rig converges. At 1.0 and a 2 ms step a brick
#: resting on the plate's stud tips does not settle: it runs a limit cycle of about +-35 um at
#: 20 Hz, with the contact force swinging between 10 and 26 N under a steady 20 N press, and
#: that wander is what used to push a misaligned stud deep enough into a wall for the solver to
#: report tens of newtons sideways. At 2.0 the same scene settles to within 0.4 um and reads
#: the same at 2 ms, 1 ms and 0.5 ms. The cost is stated rather than hidden: MuJoCo's stiffness
#: goes as ``1/dampratio**2``, so the rig's contacts are four times softer than the simulator's,
#: which at a 20 N press is 2.4 um of penetration instead of 0.6 um - both far under the 86 um
#: interference, so neither changes what the press fit does.
PRESS_SOLREF_DAMPRATIO = 2.0


@dataclass
class PressRig:
    """A baseplate, a brick carried rigidly on a vertical slide, and a damage model.

    The ram is **force** controlled, not position controlled, and that distinction is the whole
    point. A stiff position-controlled ram driven into a geometric jam produces whatever force
    the solver needs to satisfy the command, which on a misaligned brick is kilonewtons - an
    artefact of the command, not a measurement of anything. Writing a force on the ram body
    instead means "press with exactly 20 N" means what it says, which is the only way to ask
    what a 10 to 14 N arm can do to a brick.

    The brick is held rigidly in the lateral directions, as it is in a real gripper, so a
    lateral misalignment becomes a *displacement*-controlled load: the contact force is set by
    how far the geometry overlaps, not by how hard the ram pushes. That is worth knowing before
    reading any sweep produced with this rig.

    **This is a quasi-static rig and it has to be run like one.** Everything about it - the ram
    mass, the dashpot, the velocity-limited approach, the ramp that starts only once the brick
    is touching, the hold that follows it - exists so that what it reports is a force balance
    and not a transient. :func:`run_press` takes its phase lengths in *seconds*, so the same
    physical experiment runs at any time step, and the rig is checked at 2, 1 and 0.5 ms by
    ``test_the_misalignment_sweep_is_converged_in_the_time_step``.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    bricks: list
    snap: object
    damage: "DamageModel"
    ram_body: int
    offset_x: float
    ram_qadr: int = 0
    ram_dadr: int = 0
    damping: float = PRESS_RAM_DAMPING
    start_gap: float = 0.0005

    @property
    def ram_z(self) -> float:
        """Travel of the ram [m] from where it started, negative downward."""

        return float(self.data.qpos[self.ram_qadr])

    @property
    def ram_v(self) -> float:
        """Ram speed [m/s], which a quasi-static reading needs to be near zero."""

        return float(self.data.qvel[self.ram_dadr])


def build_press_rig(
    offset_x: float = 0.0,
    spec: BrickSpec | None = None,
    nx: int = 2,
    ny: int = 4,
    f_clutch: float = 1.0,
    material: mat.Material = mat.ABS_BRICK_23C,
    thresholds: DamageThresholds | None = None,
    start_gap: float = 0.0005,
    timestep: float = 0.002,
    friction: float | None = None,
    seed: int | None = None,
    ram_mass: float = PRESS_RAM_MASS_KG,
    ram_damping: float = PRESS_RAM_DAMPING,
) -> PressRig:
    """Build the rig described by :class:`PressRig`, with the damage model already attached.

    ``offset_x`` shifts the whole ram sideways in metres, which is the misalignment. On a DUPLO
    2x4 pressed onto a baseplate the interesting range is far smaller than it looks: the studs
    and tubes overlap by 86 um on the radius, so a misalignment of about 0.18 mm is already
    enough to stop a stud entering its cell at all, and the brick then stands on the stud tips.
    The 1.65 mm of clear air between a stud and the nearest wall (the ribs that would bridge it
    on a real brick are not in the generator's geometry) is nineteen times further away than
    that, so **no misalignment lets a plate stud reach a wall's inner face**: the brick is held
    off by the tube rims long before. See the guide's section 6.4.

    The ram's dynamics are the fix for the convergence failure the review found: a 1 g ram on a
    2 N*s/m dashpot was light enough to plunge 0.8 mm past its own resting place during the
    first half second of the ramp, and at a 2 ms step the damage model latched that plunge as a
    cracked wall. :data:`PRESS_RAM_MASS_KG`, :data:`PRESS_RAM_DAMPING` and
    :data:`PRESS_SOLREF_DAMPRATIO` say what replaced it and why.
    """

    from brick_builder.bricks.generator import add_baseplate, add_brick
    from brick_builder.bricks.snap import ColletSnap
    from brick_builder.bricks.snap_api import SnapParams
    from brick_builder.bricks.spec import DUPLO

    spec = DUPLO if spec is None else spec
    settings = mat.contact_settings(timestep)["brick_on_brick"]
    mu = settings.friction[0] if friction is None else friction

    world = mujoco.MjSpec()
    opt = world.option
    opt.timestep = timestep
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.iterations = 100
    opt.ls_iterations = 50
    opt.gravity = [0.0, 0.0, 0.0]        # the gauge is tared, as a bench gauge is
    world.default.geom.solref = [settings.solref[0], PRESS_SOLREF_DAMPRATIO]
    world.default.geom.solimp = list(settings.solimp) + [0.5, 2.0]
    world.default.geom.friction = [mu, settings.friction[1], settings.friction[2]]

    plate = add_baseplate(world, "plate", 4, 6, pos=(0.0, 0.0, 0.0), spec=spec)
    stud_tip = spec.roof * 2.0 + spec.stud_h
    ram = world.worldbody.add_body(name="ram", pos=[offset_x, 0.0, stud_tip + start_gap])
    joint = ram.add_joint(name="ram_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    # No armature: the ram now carries a real mass, so there is nothing to pad the mass matrix
    # with. The old 0.02 kg of armature was twenty times the ram body's own mass and more than
    # the brick it carried - most of what moved on this dof, and none of it physical.
    joint.armature = 0.0
    joint.damping = [ram_damping, 0.0, 0.0]
    ram.mass = ram_mass
    ram.inertia = [1e-6, 1e-6, 1e-6]
    ram.ipos = [0.0, 0.0, 0.0]
    ram.explicitinertial = True
    brick = add_brick(world, "brick", nx, ny, "red", pos=(0.0, 0.0, 0.0), spec=spec,
                      free=False, parent=ram)

    snap = ColletSnap(SnapParams(f_clutch=f_clutch))
    snap.declare(world, [brick], [plate])
    model = world.compile()
    data = mujoco.MjData(model)
    snap.bind(model, data)
    mujoco.mj_forward(model, data)
    damage = DamageModel(model, data, [brick], snap=snap, material=material,
                         thresholds=thresholds, seed=seed, track_connections=False)
    ram_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ram")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ram_z")
    return PressRig(model, data, [brick], snap, damage, ram_body, offset_x,
                    ram_qadr=int(model.jnt_qposadr[jid]), ram_dadr=int(model.jnt_dofadr[jid]),
                    damping=ram_damping, start_gap=start_gap)


def run_press(
    rig: PressRig,
    force_N: float,
    approach_force_N: float = 2.0,
    ramp_s: float = 0.5,
    hold_s: float = 0.4,
    max_approach_s: float = 3.0,
    settle_s: float = 2.0,
    settle_speed: float = 2e-4,
) -> dict:
    """Press with ``force_N``, quasi-statically, and report the settled reading.

    Three phases, all measured in seconds so that the same experiment runs at any time step:

    1. **Approach.** The ram descends under ``approach_force_N`` alone, which on the rig's
       dashpot is a terminal speed of about 1.3 mm/s, until the first contact appears. Nothing
       is ramped while the brick is still in the air, so the press force never has a chance to
       accelerate the ram into the plate.
    2. **Ramp.** The commanded force rises linearly to ``force_N`` over ``ramp_s``. At 20 N over
       half a second against a contact of a few hundred kN/m the ram creeps at under a
       millimetre per second, and the dashpot's share of the commanded force is under 1 %.
    3. **Hold, until it stops moving.** The force is held for at least ``hold_s`` and then until
       the ram is slower than ``settle_speed`` (0.2 mm/s), giving up after a further
       ``settle_s``. Holding for a fixed time would not do: seating a brick against the clutch
       takes about a second of creep at 15 N, and a reading taken while the brick is still
       going in is not a force balance. ``ram_v_m_s`` and ``settled`` in the result say how
       close to rest it actually got, and a reader who does not check them is not entitled to
       call the number quasi-static.

    The returned ``peak_*`` fields are instantaneous peaks over the whole run, unfiltered.
    ``longest_overload_s`` is how long the worst stress stayed past an allowable, which is what
    separates a real overload from a transient; see :attr:`DamageThresholds.persist_s`.
    """

    m, d = rig.model, rig.data
    dt = m.opt.timestep
    body = rig.ram_body

    def step(force: float) -> None:
        d.xfrc_applied[body] = 0.0
        d.xfrc_applied[body, 2] = -force
        mujoco.mj_step(m, d)
        rig.snap.update(m, d)
        rig.damage.update()

    # The approach ends at the first contact, or when the ram has closed the gap it started
    # with - an *aligned* brick makes no contact at all on the way in, because the snap model
    # owns the stud-in-tube fit and those geoms do not collide, so "wait for ncon" alone would
    # spend the whole approach budget in mid-air.
    for _ in range(int(round(max_approach_s / dt))):
        step(approach_force_N)
        if d.ncon > 0 or -rig.ram_z >= rig.start_gap:
            break
    touched = bool(d.ncon > 0)
    t_contact = float(d.time)

    n_ramp = max(int(round(ramp_s / dt)), 1)
    for i in range(n_ramp):
        step(force_N * (i + 1) / n_ramp)
    n_hold = int(round(hold_s / dt))
    for i in range(n_hold + int(round(settle_s / dt))):
        step(force_N)
        if i >= n_hold and abs(rig.ram_v) <= settle_speed:
            break
    d.xfrc_applied[body] = 0.0

    rec = rig.damage.brick("brick")
    depths = rig.snap.stud_depths("brick")
    return {
        "offset_x_m": rig.offset_x,
        "force_N": force_N,
        "state": rec.state,
        "peak_wall_stress_Pa": rec.peak_by_mode["wall_bending"],
        "peak_by_mode_Pa": dict(rec.peak_by_mode),
        "peak_lateral_N": rec.peak_lateral_N,
        "peak_normal_N": rec.peak_normal_N,
        "peak_ratio": rec.peak_ratio,
        "peak_mode": rec.peak_mode,
        "longest_overload_s": rec.longest_overload_s,
        "seated": bool(depths.size and depths.min() >= rig.bricks[0].spec.stud_h - 3e-4),
        "min_depth_m": float(depths.min()) if depths.size else 0.0,
        "contacted": touched,
        "t_contact_s": t_contact,
        "ram_z_m": rig.ram_z,
        "ram_v_m_s": rig.ram_v,
        "settled": bool(abs(rig.ram_v) <= settle_speed),
        "duration_s": float(d.time),
    }


def misalignment_sweep(
    offsets_m,
    force_N: float = 20.0,
    timestep: float = 0.002,
    **rig_kwargs,
) -> list[dict]:
    """One quasi-static press per offset, at one time step. The rig's headline experiment."""

    return [
        run_press(build_press_rig(offset_x=float(o), timestep=timestep, **rig_kwargs), force_N)
        for o in offsets_m
    ]


@dataclass
class WallRig:
    """One brick fixed to the bench and a rounded rod pushed against the inside of a wall.

    This is the guide's section 7.7 destructive test, done in the simulator: "stand a brick on
    a kitchen scale and push a rounded rod against the inside of a wall near its bottom edge,
    increasing until the wall takes a visible set, then until it cracks." It exists because the
    misalignment press cannot load a wall at all in the geometry the generator builds (see
    :func:`build_press_rig`), and a failure mode that nothing exercises is a failure mode
    nobody has checked.

    It is also the well-posed version of the experiment. The brick is fixed, the rod is force
    controlled, the contact is a sphere on a flat face with a normal that does not depend on
    which way the solver resolves a corner, and the force balance at rest is the applied force.
    So the wall bending stress it reports is ``6 F L / (b_eff t^2)`` with every term known in
    advance, and the same number comes out at any time step.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    bricks: list
    damage: "DamageModel"
    rod_body: int
    rod_qadr: int
    rod_dadr: int
    lever_m: float
    b_eff_m: float
    rod_radius_m: float

    def stress_per_newton(self) -> float:
        """The analytic ``6 L / (b_eff t^2)`` [Pa/N] this rig should reproduce."""

        return mat.wall_bending_stress(1.0, self.lever_m, self.b_eff_m,
                                       self.bricks[0].spec.wall)


def build_wall_rig(
    spec: BrickSpec | None = None,
    nx: int = 2,
    ny: int = 4,
    rod_radius: float = 0.002,
    lever: float | None = None,
    timestep: float = 0.002,
    material: mat.Material = mat.ABS_BRICK_23C,
    thresholds: DamageThresholds | None = None,
    start_gap: float = 0.0002,
    rod_mass: float = 0.1,
    rod_damping: float = 500.0,
    seed: int | None = None,
) -> WallRig:
    """Build the section 7.7 rig: a fixed brick and a rod aimed at one wall's inner face.

    ``lever`` is the distance from the contact up to the roof, and defaults to the 15.0 mm the
    guide's section 4.3 works its thresholds at (0.85 of the wall height, "a contact near the
    bottom edge"). The rod sits between two tubes so nothing but the wall is in its way.
    """

    from brick_builder.bricks.generator import add_brick
    from brick_builder.bricks.spec import DUPLO

    spec = DUPLO if spec is None else spec
    settings = mat.contact_settings(timestep)["brick_on_brick"]
    wall_h = spec.height - spec.roof
    lever = 0.85 * wall_h if lever is None else lever
    contact_z = wall_h - lever
    wall_inner = (nx * spec.pitch - spec.clearance) / 2.0 - spec.wall

    world = mujoco.MjSpec()
    opt = world.option
    opt.timestep = timestep
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.iterations = 100
    opt.ls_iterations = 50
    opt.gravity = [0.0, 0.0, 0.0]
    world.default.geom.solref = [settings.solref[0], PRESS_SOLREF_DAMPRATIO]
    world.default.geom.solimp = list(settings.solimp) + [0.5, 2.0]
    world.default.geom.friction = [settings.friction[0], settings.friction[1],
                                   settings.friction[2]]

    brick = add_brick(world, "brick", nx, ny, "red", pos=(0.0, 0.0, 0.0), spec=spec, free=False)
    rod = world.worldbody.add_body(
        name="rod",
        pos=[wall_inner - rod_radius - start_gap, spec.pitch / 2.0, contact_z],
    )
    joint = rod.add_joint(name="rod_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    joint.armature = 0.0
    joint.damping = [rod_damping, 0.0, 0.0]
    rod.mass = rod_mass
    rod.inertia = [1e-7, 1e-7, 1e-7]
    rod.ipos = [0.0, 0.0, 0.0]
    rod.explicitinertial = True
    rod.add_geom(name="rod_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[rod_radius, 0, 0],
                 rgba=[0.2, 0.2, 0.25, 1.0])

    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    damage = DamageModel(model, data, [brick], material=material, thresholds=thresholds,
                         seed=seed, track_connections=False)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_x")
    # The same b_eff the damage model will use: a 45 degree spread from the contact up to the
    # roof, plus the contact width, capped at the wall's own length.
    b_eff = min(2.0 * lever + 2.0 * rod_radius,
                mat.sections(spec, nx, ny)["wall_left_right"].length)
    return WallRig(model, data, [brick], damage,
                   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rod"),
                   int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]),
                   lever, b_eff, rod_radius)


def run_wall_push(
    rig: WallRig,
    force_N: float,
    ramp_s: float = 0.5,
    hold_s: float = 0.4,
    approach_force_N: float = 0.5,
    max_approach_s: float = 3.0,
) -> dict:
    """Push the wall with ``force_N``, quasi-statically, and report the settled reading.

    Same three phases as :func:`run_press` and for the same reason.
    """

    m, d = rig.model, rig.data
    dt = m.opt.timestep
    body = rig.rod_body

    def step(force: float) -> None:
        d.xfrc_applied[body] = 0.0
        d.xfrc_applied[body, 0] = force
        mujoco.mj_step(m, d)
        rig.damage.update()

    for _ in range(int(round(max_approach_s / dt))):
        step(approach_force_N)
        if d.ncon > 0:
            break
    n_ramp = max(int(round(ramp_s / dt)), 1)
    for i in range(n_ramp + int(round(hold_s / dt))):
        step(force_N * min(1.0, (i + 1) / n_ramp))
    d.xfrc_applied[body] = 0.0

    rec = rig.damage.brick("brick")
    return {
        "force_N": force_N,
        "state": rec.state,
        "peak_wall_stress_Pa": rec.peak_by_mode["wall_bending"],
        "peak_lateral_N": rec.peak_lateral_N,
        "peak_ratio": rec.peak_ratio,
        "peak_mode": rec.peak_mode,
        "longest_overload_s": rec.longest_overload_s,
        "analytic_wall_stress_Pa": rig.stress_per_newton() * force_N,
        "rod_v_m_s": float(d.qvel[rig.rod_dadr]),
    }


def wall_push_onset(
    forces_N=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0),
    timestep: float = 0.002,
    **rig_kwargs,
) -> dict:
    """Sweep the rod force and report where each state first appears. Where damage starts."""

    rows = [run_wall_push(build_wall_rig(timestep=timestep, **rig_kwargs), float(f))
            for f in forces_N]
    onset = {}
    for state in ("marked", "cracked", "broken"):
        hit = [r["force_N"] for r in rows
               if _STATE_RANK[r["state"]] >= _STATE_RANK[state]]
        onset[state] = min(hit) if hit else None
    return {"forces_N": list(forces_N), "rows": rows, "onset_N": onset}




# Okabe-Ito, in fixed order. It is the published colour-blind-safe qualitative set, which
# matters here because three time steps and five press forces are told apart by colour.
_SERIES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
_INK = "#1a1a1a"
_MUTED = "#5a5a5a"


def figure_wall_stress_map(
    out_path: str,
    offsets_mm: tuple = (0.0, -0.05, -0.10, -0.15, -0.18, -0.20, -0.25, -0.50, -1.00, -2.00),
    forces_N: tuple = (5.0, 10.0, 20.0, 30.0, 45.0),
    timesteps: tuple = (0.002, 0.001, 0.0005),
    wall_forces_N: tuple = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 64.0, 70.0, 80.0, 96.0, 110.0),
) -> dict:
    """What a misaligned press does to a brick, what it does not, and the check that says so.

    Four panels, and the first two are the answer to the question the old version of this
    figure got wrong:

    - **Where the load goes.** Every cell is one quasi-static press of a misaligned 2x4 onto a
      baseplate. Nothing in the sweep reaches any allowable: the worst stress ratio anywhere is
      about 0.05 except in one cell, at 45 N (three times what the arm has) and 0.18 mm of
      misalignment, where the press forces a stud past a tube rim and loads that tube to 62 %
      of its allowable. **Wall bending is exactly zero in every cell**, because a plate stud
      cannot reach a wall: the studs and tubes overlap by 86 um on the radius and jam at about
      0.18 mm of misalignment, while the nearest wall is 1.65 mm away.
    - **Why.** The seating curve shows the cliff. Under 0.15 mm the brick seats; past 0.18 mm
      it stands on the stud tips and goes no further, whatever the press force.
    - **The convergence check.** The same sweep at 2, 1 and 0.5 ms. This is the acceptance test
      the review asked for, and it is drawn rather than asserted because the old rig failed it
      qualitatively: the same offset read "broken" at 2 ms and "undamaged" at 1 ms.
    - **Where damage does start.** The wall cannot be loaded by a press, so it is loaded the
      way the guide's section 7.7 loads a real one - a rounded rod against the inside of the
      wall near its bottom edge - and that is where the numbers for "the arm would have to do
      X" come from.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.bricks.spec import DUPLO

    th = DamageThresholds.from_material(mat.ABS_BRICK_23C)
    offsets = np.asarray(offsets_mm, dtype=float)
    forces = np.asarray(forces_N, dtype=float)

    ratio = np.zeros((forces.size, offsets.size))
    wall = np.zeros_like(ratio)
    depth = np.zeros_like(ratio)
    mode = np.empty(ratio.shape, dtype=object)
    for i, f in enumerate(forces):
        for j, o in enumerate(offsets):
            out = run_press(build_press_rig(offset_x=float(o) * 1e-3), float(f))
            ratio[i, j] = out["peak_ratio"]
            wall[i, j] = out["peak_wall_stress_Pa"] / 1e6
            depth[i, j] = out["min_depth_m"] * 1e3
            mode[i, j] = out["peak_mode"]

    conv = {}
    for dt in timesteps:
        conv[dt] = misalignment_sweep(offsets * 1e-3, 20.0, timestep=dt)

    wall_rows = [run_wall_push(build_wall_rig(), float(f)) for f in wall_forces_N]

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0))
    for ax in axes.ravel():
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.18, color=_MUTED, linewidth=0.7)
        ax.set_axisbelow(True)

    # -- A: where the load goes ------------------------------------------------
    ax = axes[0, 0]
    mesh = ax.pcolormesh(np.arange(offsets.size + 1), np.arange(forces.size + 1), ratio,
                         cmap="Oranges", vmin=0.0, vmax=1.0, edgecolors="white", linewidth=2)
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label("peak stress ratio  (worst stress / its allowable)", fontsize=12)
    bar.ax.tick_params(labelsize=11)
    for i in range(forces.size):
        for j in range(offsets.size):
            ax.text(j + 0.5, i + 0.5, f"{ratio[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color=_INK if ratio[i, j] < 0.55 else "white")
    ax.set_xticks(np.arange(offsets.size) + 0.5)
    ax.set_xticklabels([f"{o:.2f}" for o in offsets], fontsize=11)
    ax.set_yticks(np.arange(forces.size) + 0.5)
    ax.set_yticklabels([f"{f:.0f}" for f in forces], fontsize=11)
    ax.set_xlabel("lateral misalignment of the tool  [mm]", fontsize=12)
    ax.set_ylabel("commanded press force  [N]", fontsize=12)
    ax.set_title("A.  Nothing in a misaligned press reaches an allowable\n"
                 f"wall bending is 0.00 MPa in all {ratio.size} cells;\n"
                 "the one hot cell is a tube, at 45 N", fontsize=13, color=_INK)
    ax.grid(False)

    # -- B: the seating cliff --------------------------------------------------
    ax = axes[0, 1]
    stud_h = DUPLO.stud_h * 1e3
    for k, f in enumerate(forces):
        ax.plot(-offsets, depth[k], "o-", lw=2.0, ms=8, color=_SERIES[k % len(_SERIES)],
                label=f"{f:.0f} N press")
    ax.axhline(stud_h, color=_MUTED, ls="--", lw=1.5)
    ax.text(0.02, stud_h, " fully seated (4.5 mm)", va="bottom", fontsize=11, color=_MUTED)
    ax.axvline(0.086, color=_INK, ls=":", lw=1.5)
    ax.text(0.092, 3.55, "86 um: the whole\nstud-tube interference", fontsize=11, color=_INK)
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.4)
    ax.set_xlim(0.0, float(-offsets.min()) * 1.15)
    ax.set_xlabel("lateral misalignment  [mm, log scale above 0.05 mm]", fontsize=12)
    ax.set_ylabel("depth the studs reach  [mm]", fontsize=12)
    ax.set_title("B.  Why: the brick stops entering at 0.18 mm of misalignment\n"
                 "past 0.2 mm it stands on the stud tips at any press force;\n"
                 "at 0.18 mm only 45 N, three times the arm's, forces it through",
                 fontsize=13, color=_INK)
    ax.legend(fontsize=11, loc="upper right")

    # -- C: the convergence check ---------------------------------------------
    ax = axes[1, 0]
    spread = 0.0
    for k, dt in enumerate(timesteps):
        y = np.array([r["peak_ratio"] for r in conv[dt]])
        ax.plot(-offsets, y, ["o-", "s--", "^:"][k % 3], lw=2.0, ms=9 + 2 * k,
                color=_SERIES[k % len(_SERIES)], mfc="none" if k else None,
                label=f"{dt * 1e3:g} ms step")
    stack = np.array([[r["peak_ratio"] for r in conv[dt]] for dt in timesteps])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (stack.max(axis=0) - stack.min(axis=0)) / np.maximum(stack.max(axis=0), 1e-9)
    spread = float(np.nanmax(rel))
    states = {r["state"] for dt in timesteps for r in conv[dt]}
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.4)
    ax.set_xlim(0.0, float(-offsets.min()) * 1.15)
    ax.set_ylim(0.0, float(stack.max()) * 1.45)
    ax.text(float(-offsets.min()), float(stack.max()) * 1.36,
            "note the axis: an allowable is a ratio of 1.0,\nabout fifty times the top of this "
            "panel", fontsize=11, color=_MUTED, ha="right", va="top")
    ax.set_xlabel("lateral misalignment  [mm, log scale above 0.05 mm]", fontsize=12)
    ax.set_ylabel("peak stress ratio at a 20 N press", fontsize=12)
    ax.set_title("C.  The same experiment at three time steps\n"
                 f"worst disagreement {spread * 100:.1f} %; damage state "
                 f"{'/'.join(sorted(states))} at every offset and every step",
                 fontsize=13, color=_INK)
    ax.legend(fontsize=11, loc="upper left")

    # -- D: where damage does start -------------------------------------------
    ax = axes[1, 1]
    f_rod = np.array([r["force_N"] for r in wall_rows])
    s_rod = np.array([r["peak_wall_stress_Pa"] for r in wall_rows]) / 1e6
    analytic = np.array([r["analytic_wall_stress_Pa"] for r in wall_rows]) / 1e6
    ax.plot(f_rod, analytic, "-", lw=2.0, color=_MUTED,
            label=r"$\sigma = 6FL/(b_{eff} t^2)$, by hand")
    ax.plot(f_rod, s_rod, "o", ms=9, color=_SERIES[0], label="what the damage model reads")
    ax.set_ylim(0.0, 145.0)
    ax.set_xlim(0.0, float(f_rod.max()) * 1.08)
    for level, text, colour in ((44.0, "44 MPa: permanent set", _SERIES[4]),
                                (70.0, "70 MPa: crack", _SERIES[1])):
        ax.axhline(level, color=colour, ls="--", lw=1.6)
        ax.text(2.0, level + 2.5, text, fontsize=11, color=colour)
    cracked = [r["force_N"] for r in wall_rows if r["state"] in ("cracked", "broken")]
    broken = [r["force_N"] for r in wall_rows if r["state"] == "broken"]
    for force, label, colour, y in ((min(cracked) if cracked else None, "cracked",
                                     _SERIES[1], 126.0),
                                    (min(broken) if broken else None, "broken",
                                     "#7a1010", 108.0)):
        if force is not None:
            ax.axvline(force, color=colour, ls=":", lw=1.6)
            ax.text(force - 1.5, y, f"first {label} at {force:.0f} N ", fontsize=11,
                    color=colour, ha="right")
    ax.axvspan(min(mat.ARM_PRESS_FORCE_N), mat.ARM_GRIP_FORCE_N, color=_SERIES[0], alpha=0.12)
    ax.text(mat.ARM_GRIP_FORCE_N + 1.5, 138.0,
            "all this arm has: 10-14 N press, 17 N grip", fontsize=11, color=_SERIES[0],
            va="top")
    ax.set_xlabel("force of a 4 mm rounded rod on the inside of the wall  [N]", fontsize=12)
    ax.set_ylabel("wall bending stress  [MPa]", fontsize=12)
    ax.set_title("D.  Where damage does start: the guide's 7.7 test, simulated\n"
                 "the mode a press cannot reach, loaded the way a bench would",
                 fontsize=13, color=_INK)
    ax.legend(fontsize=11, loc="lower right")

    fig.suptitle("A misaligned press on a DUPLO 2x4: what it loads, and what it takes to hurt "
                 "a wall", fontsize=16, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)

    return {
        "offsets_mm": list(offsets),
        "forces_N": list(forces),
        "stress_ratio": ratio.tolist(),
        "wall_stress_MPa": wall.tolist(),
        "stud_depth_mm": depth.tolist(),
        "mode": mode.tolist(),
        "worst_ratio": float(ratio.max()),
        "max_wall_stress_MPa": float(wall.max()),
        "timestep_spread": spread,
        "states": sorted(states),
        "wall_push": {
            "force_N": f_rod.tolist(),
            "stress_MPa": s_rod.tolist(),
            "analytic_MPa": analytic.tolist(),
            "first_cracked_N": min(cracked) if cracked else None,
            "first_broken_N": min(broken) if broken else None,
        },
        "thresholds_MPa": (th.yield_Pa / 1e6, th.flexural_Pa / 1e6),
    }
