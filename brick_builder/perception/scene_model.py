"""The scene-model interface: a small set of per-slot states, and nothing else, is what the
controller consumes.

Earlier revisions of this module treated the per-slot rows as opaque and routed everything through
a per-pixel decode. That is superseded: the action model predicts and compares **slot states**
directly (`brick_builder.action` reasons about "this slot's blob should move here"), so the slot
rows carry named, readable fields -- not an opaque embedding -- and a decode/label image is
optional, useful only for a human viewer or an accuracy evaluation, never for control.

`SceneState` is the contract every producer publishes:

- `slots` `[K, W]` float32 -- one row per slot, `K = 8`. What each column means is declared by
  `schema` (a `SlotSchema`), not fixed in code: different producers (the ideal ground-truth
  source, a classical colour-segmentation source, a future learned model with its own native
  width) can publish different schemas, and a consumer asks the schema what is available instead
  of indexing by position.
- `depth` `[K]` float32, ordinal (relative front-to-back), `[0, 1]`, nearer = smaller, `0.5` for
  an empty slot.
- `t`: a timestamp (seconds, or any monotonic counter a producer chooses).
- `source`: a short name for which producer/backend built this state.
- `window`: the view (`views.ViewingPose`, generally an attention window -- see `views.py`) the
  256x256 frame this state was computed from was cropped to. Needed to turn a slot's own
  normalized image coordinates back into a metric position later.
- `slot_ids`: `[K]`, a persistent identity per slot this frame (`-1` for an unclaimed/empty slot)
  -- see `TemporalSlotAssigner`.

`PLACEHOLDER_SLOT_SCHEMA` is this project's own representation: a conventional colour-and-position
blob per object (mass, centre, second moments, mean colour, colour spread, border fraction, age,
fourth-order orientation moment) -- see `ideal_scene_model.py`'s module docstring for the exact
field layout and how each one is computed. It is a generic, textbook blob descriptor. A source
with a different native layout declares its own schema; `brick_builder.perception.sources` is the
registry a source outside this package registers itself with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from .views import ViewingPose


# --------------------------------------------------------------------------------- the schema ---


@dataclass(frozen=True)
class SlotSchema:
    """Declares what each column of one slot's `[W]` row means.

    `fields` maps a name to the column slice that holds it. `get`/`has` are how every named
    accessor on `SceneState` reads a field -- no code anywhere in this project indexes a slot row
    by a bare integer column number. A producer with no declared layout at all (an opaque learned
    embedding, before its own schema is documented) uses `opaque_schema(w)`: `has` is `False` for
    everything, which is the correct, honest answer.
    """

    W: int
    fields: dict[str, slice]

    def has(self, name: str) -> bool:
        return name in self.fields

    def get(self, row: np.ndarray, name: str) -> np.ndarray:
        if not self.has(name):
            raise KeyError(f"schema has no field {name!r}; available: {sorted(self.fields)}")
        return row[..., self.fields[name]]

    def names(self) -> tuple[str, ...]:
        return tuple(self.fields)


def opaque_schema(w: int) -> SlotSchema:
    """A schema declaring nothing -- for a producer whose row is a learned, unlabelled embedding."""
    return SlotSchema(W=w, fields={})


PLACEHOLDER_SLOT_SCHEMA = SlotSchema(
    W=14,
    fields={
        "mass": slice(0, 1),
        "centre": slice(1, 3),
        "moments": slice(3, 6),
        "colour": slice(6, 9),
        "colour_spread": slice(9, 10),
        "border": slice(10, 11),
        "age": slice(11, 12),
        "orient4": slice(12, 14),
    },
)
"""This project's placeholder slot layout (see the module docstring). Column meanings, all
computed from the object's own *visible* pixels inside the current 256x256 attention window,
normalized the same way the window's own evidence coordinates are (`x` = column, `y` = row, both
in `[-1, 1]`; colour channels as `2*rgb - 1`):

    [0]    mass           fraction of the window's pixels this object claims
    [1:3]  mean x, y       centroid, normalized window coordinates
    [3:6]  sxx, sxy, syy   second central moments of (x, y) over the object's pixels
    [6:9]  mean r, g, b    mean colour, normalized (2*rgb - 1)
    [9]    colour spread   mean squared deviation of pixel colour from the mean (one scalar)
    [10]   border fraction share of the object's pixels lying on the window's own border --
                           tells a consumer the object is cut off by the crop
    [11]   age             tanh(frames_active / 10); 0 for a just-appeared object, -> 1 as it
                           stays visible
    [12:14] orient4        the fourth-order complex moment, real and imaginary parts (see
                           `fourth_moment`). Second moments give an orientation only for an
                           elongated blob; a square footprint's are isotropic and its angle is
                           noise. This field carries the angle of a four-fold symmetric shape,
                           which is what a 2x2 brick seen from above is.

An empty slot is all zeros, with `SceneState.depth` exactly `0.5` at that slot.
"""

LOGIT_MAGNITUDE = 10.0
"""The +/-L an optional, evaluation-only decode/label function uses for a hard, unambiguous
per-pixel assignment (see `SceneState.decode`, `ideal_scene_model.py`'s ground-truth label map).
Not read by anything on the control path -- decode is viewer/evaluation-only."""


# ---------------------------------------------------------------------------------- the state ---


@dataclass
class SceneState:
    """One vision tick's worth of slot states -- what the controller (eventually) reads."""

    slots: np.ndarray
    depth: np.ndarray
    schema: SlotSchema
    t: float
    source: str
    window: "ViewingPose"
    slot_ids: np.ndarray
    frame_id: int = 0
    _decode_fn: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.slots = np.asarray(self.slots, dtype=np.float32)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.slot_ids = np.asarray(self.slot_ids, dtype=np.int64)
        if self.slots.ndim != 2:
            raise ValueError(f"slots must be [K,W]; got shape {self.slots.shape}")
        if self.depth.shape != (self.K,):
            raise ValueError(f"depth must be [K]={self.K}; got shape {self.depth.shape}")
        if self.slot_ids.shape != (self.K,):
            raise ValueError(f"slot_ids must be [K]={self.K}; got shape {self.slot_ids.shape}")
        if self.schema.W != self.W:
            raise ValueError(f"schema declares W={self.schema.W} but slots has W={self.W}")

    @property
    def K(self) -> int:
        return int(self.slots.shape[0])

    @property
    def W(self) -> int:
        return int(self.slots.shape[1])

    def is_empty(self, k: int) -> bool:
        return bool(self.depth[k] == 0.5 and not np.any(self.slots[k]))

    @property
    def res(self) -> int:
        """The square frame resolution this state's `window` was cropped/resized to (256 for every
        `ViewingPose` this project defines) -- a convenience for viewer/
        evaluation code (`readout.py`) that otherwise has to reach into `self.window.output_size`
        itself. Raises if this state has no `window` (a bare/synthetic state built without one,
        e.g. some tests)."""
        if self.window is None:
            raise AttributeError("this SceneState has no window; res is undefined")
        return int(self.window.output_size)

    # -- named accessors: no consumer indexes a slot row by a magic column number -------------

    def _field(self, k: int, name: str) -> np.ndarray:
        return self.schema.get(self.slots[k], name)

    def mass(self, k: int) -> float:
        return float(self._field(k, "mass")[0])

    def centre(self, k: int) -> tuple[float, float]:
        c = self._field(k, "centre")
        return float(c[0]), float(c[1])

    def moments(self, k: int) -> tuple[float, float, float]:
        m = self._field(k, "moments")
        return float(m[0]), float(m[1]), float(m[2])

    def colour(self, k: int) -> tuple[float, float, float]:
        c = self._field(k, "colour")
        return float(c[0]), float(c[1]), float(c[2])

    def colour_spread(self, k: int) -> float:
        return float(self._field(k, "colour_spread")[0])

    def border_fraction(self, k: int) -> float:
        return float(self._field(k, "border")[0])

    def age(self, k: int) -> float:
        return float(self._field(k, "age")[0])

    def orientation4(self, k: int) -> tuple[float, float]:
        """Orientation of a four-fold symmetric blob [rad, modulo pi/2] and its strength.

        The strength is `abs(c4)` (see `fourth_moment`): near 0.2 for a square footprint seen
        from above, near 0 for a round or featureless blob, whose angle means nothing. A reader
        that wants "which way round is this brick" uses this for a square footprint and
        `orientation_and_axes` for an elongated one.
        """
        re, im = self._field(k, "orient4")
        return 0.25 * math.atan2(float(im), float(re)), math.hypot(float(re), float(im))

    def orientation_and_axes(self, k: int) -> tuple[float, tuple[float, float]]:
        """Orientation [rad] and (major, minor) axis lengths (normalized window units, the "2
        sigma" convention: `2 * sqrt(eigenvalue)`) of the second-moment ellipse -- derived from
        `moments`, never a separate stored field."""
        sxx, sxy, syy = self.moments(k)
        cov = np.array([[sxx, sxy], [sxy, syy]], dtype=float)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]
        angle = float(math.atan2(evecs[1, 0], evecs[0, 0]))
        axes = (float(2.0 * math.sqrt(evals[0])), float(2.0 * math.sqrt(evals[1])))
        return angle, axes

    # -- optional, evaluation/viewer-only ------------------------------------------------------

    def decode(self, px: np.ndarray) -> np.ndarray:
        """`px [n,2]` (col, row) -> logits `[n, K]`, if this state's producer built one (the ideal
        ground-truth source always does; a real vision model would not, in production, since the
        controller never needs it). **Viewer/evaluation only** -- see `readout.py`'s own banner.
        Raises `RuntimeError` if this state has no decode function.
        """
        if self._decode_fn is None:
            raise RuntimeError(
                f"this SceneState (source={self.source!r}) has no decode/label function -- "
                "decode is viewer/evaluation-only and not every producer provides one"
            )
        px = np.asarray(px)
        if px.ndim != 2 or px.shape[-1] != 2:
            raise ValueError(f"px must be [n,2] (col,row); got shape {px.shape}")
        logits = np.asarray(self._decode_fn(px), dtype=np.float32)
        if logits.shape != (px.shape[0], self.K):
            raise ValueError(f"decode fn returned {logits.shape}; expected {(px.shape[0], self.K)}")
        return logits

    @property
    def has_decode(self) -> bool:
        return self._decode_fn is not None

    # -- pickling ---------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop `_decode_fn` before pickling; restore every other field unchanged.

        A source that builds a decode closure locally (`perception/cnn/postprocess.slots_from_
        masks`'s `decode_fn`, for example) makes it unpicklable, and `record_for`
        (`brick_builder/campaign/tasks.py`) caches the target sequence's `SceneState` rows to
        disk under `--source cnn`. `decode` is viewer/evaluation-only and is never read on the
        control path (see the class docstring and `decode`'s own), so a round-tripped state
        answers every control-path query exactly as before; it just has no decode function any
        more (`has_decode` is `False` and `decode` raises the same "no decode function"
        `RuntimeError` it already raises for any producer that never built one).
        """
        state = self.__dict__.copy()
        state["_decode_fn"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


@runtime_checkable
class SceneModelSource(Protocol):
    """What every scene-state producer implements."""

    def observe(
        self,
        rgb: np.ndarray,
        *,
        window: "ViewingPose",
        instance: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        present_mask: np.ndarray | None = None,
        t: float | None = None,
    ) -> SceneState:
        """One vision tick's `SceneState`, computed inside `window`. `rgb` is the only input
        every source needs; `instance` (ground-truth segmentation) and `depth` (metric, metres,
        NaN = invalid) are consumed by sources that have them (the ideal source requires both)
        and ignored otherwise. `present_mask` excludes pixels from consideration (e.g. a region
        already known to be the gripper). `t` is a timestamp; omit it to let the source count its
        own frames. Identity across calls (`SceneState.slot_ids`) is this source's own
        `TemporalSlotAssigner`'s job -- call `observe` once per tick, in order; there is no
        separate "reset" in this protocol (a source that wants one exposes it as its own method).
        """
        ...


# ------------------------------------------------------------------------------ shared helpers ---


def rank_scale(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """`[K]` raw values (any units; NaN/inf ignored) -> `[K]` ordinal rank in `[0, 1]`.

    `present[k]` says whether slot `k` has any content; a present slot's value is its rank among
    the other present slots by ascending `values` (nearest/smallest = rank 0 = smallest output),
    linearly spread across `[0, 1]` (a lone present slot gets `0.0`). An absent slot always gets
    exactly `0.5` -- every producer's `depth` field uses this so they mean the same thing.
    """
    values = np.asarray(values, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    K = values.shape[0]
    out = np.full(K, 0.5, dtype=np.float32)
    idx = np.nonzero(present)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(values[idx], kind="stable")]
    m = order.size
    for rank, slot in enumerate(order):
        out[slot] = 0.0 if m <= 1 else rank / (m - 1)
    return out


def fourth_moment(dx: np.ndarray, dy: np.ndarray) -> tuple[float, float]:
    """The normalized fourth-order complex moment of a blob's centred pixel coordinates.

    `c4 = mean((x + iy)**4) / mean(x**2 + y**2)**2`, returned as `(real, imaginary)`.

    Second central moments describe an ellipse, so they give an orientation only when the blob is
    elongated; for anything with four-fold symmetry -- a square brick's footprint seen from above
    -- they are isotropic and the angle they imply is whatever the noise and the perspective
    happen to say. `arg(c4) / 4` is defined for exactly that case, modulo a quarter turn, which
    is also the symmetry of the object, so it is the most that can be known and all that is
    needed. `abs(c4)` says how four-fold the shape really is: about 0.2 for a square seen from
    above, near 0 for a disc, so a reader can tell a usable angle from a meaningless one.
    """
    r2 = float(np.mean(dx * dx + dy * dy))
    if r2 <= 0.0:
        return 0.0, 0.0
    z = (dx + 1j * dy) ** 4
    c4 = complex(np.mean(z)) / (r2 * r2)
    return float(c4.real), float(c4.imag)


def to_window_xy(col: np.ndarray, row: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel `(col, row)` in a `res x res` window -> normalized `(x, y)` in `[-1, 1]`, the same
    `linspace(-1, 1, res)[index]` convention every slot row is expressed in."""
    lin = np.linspace(-1.0, 1.0, res)
    return lin[np.asarray(col, dtype=np.int64)], lin[np.asarray(row, dtype=np.int64)]


# ---------------------------------------------------------------------- temporal slot identity ---


@dataclass
class TemporalSlotAssigner:
    """Gives each observed object a stable slot across frames.

    First come, first served: a newly seen object claims any free slot and keeps it for as long
    as it stays visible. When an object vanishes, its slot is held (remembered, but reported
    empty) for `vanish_after` frames in case it reappears (a brief occlusion), then decays to
    fully free and can be claimed by something else. When more than `K - 1` objects are visible
    at once, the largest (by `mass`) keep their slots and the rest are treated as unseen this
    frame (folded into the background slot upstream, in the producer). One slot is never claimed
    by an object at all -- whichever slot nothing claims this frame is the background slot
    (`background_slot`), which is where "everything else" (table, unlabelled pixels) belongs.

    Identity comes from one of two places: `external_ids` (the simulator's own stable object id,
    for the ground-truth producer -- exact, no matching needed), or, when a producer has no such
    id (colour segmentation, a real vision model), a minimum-cost one-to-one match between this
    frame's candidate rows and the previous frame's slot rows (`scipy.optimize.linear_sum_
    assignment` on Euclidean distance in slot-row space) -- a proper assignment, not a greedy
    nearest-neighbour search that can double-book a slot.
    """

    K: int = 8
    vanish_after: int = 5
    match_max_distance: float = 1.5

    _identity: list[object | None] = field(init=False, repr=False)
    _missing: list[int] = field(init=False, repr=False)
    _age: list[int] = field(init=False, repr=False)
    _prev_row: list[np.ndarray | None] = field(init=False, repr=False)
    _background_slot: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._identity = [None] * self.K
        self._missing = [0] * self.K
        self._age = [0] * self.K
        self._prev_row = [None] * self.K
        self._background_slot = 0

    @property
    def background_slot(self) -> int:
        return self._background_slot

    def slot_age(self, k: int) -> int:
        return self._age[k]

    def step(
        self, masses: np.ndarray, rows: np.ndarray, external_ids: list | None = None,
    ) -> np.ndarray:
        """`masses [N]`, `rows [N, W]`: this frame's candidate objects (NOT including "everything
        else"). Returns `slot_of [N]` int64, each either a slot in `[0, K)` (one of the up to
        `K - 1` largest kept) or `-1` (over the cap -- caller folds it into the background slot).
        """
        masses = np.asarray(masses, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        n = masses.shape[0]
        slot_of = np.full(n, -1, dtype=np.int64)
        if n == 0:
            self._advance(claimed=set())
            return slot_of

        order = np.argsort(-masses)
        kept = [int(i) for i in order[: self.K - 1]]
        claimed: set[int] = set()

        if external_ids is not None:
            for i in kept:
                eid = external_ids[i]
                for k in range(self.K):
                    if self._identity[k] == eid:
                        slot_of[i] = k
                        claimed.add(k)
                        break
        else:
            prev_slots = [k for k in range(self.K) if self._prev_row[k] is not None]
            unmatched = [i for i in kept if slot_of[i] == -1]
            if prev_slots and unmatched:
                cost = np.zeros((len(unmatched), len(prev_slots)))
                for a, i in enumerate(unmatched):
                    for b, k in enumerate(prev_slots):
                        cost[a, b] = np.linalg.norm(rows[i] - self._prev_row[k])
                r_idx, c_idx = linear_sum_assignment(cost)
                for a, b in zip(r_idx, c_idx):
                    if cost[a, b] <= self.match_max_distance:
                        i, k = unmatched[a], prev_slots[b]
                        slot_of[i] = k
                        claimed.add(k)

        for i in kept:
            if slot_of[i] != -1:
                continue
            free = [
                k for k in range(self.K)
                if k not in claimed and (self._identity[k] is None or self._missing[k] > self.vanish_after)
            ]
            if not free:
                continue
            k = free[0]
            slot_of[i] = k
            claimed.add(k)
            self._identity[k] = external_ids[i] if external_ids is not None else object()
            self._age[k] = 0

        for i in kept:
            k = int(slot_of[i])
            if k != -1:
                self._prev_row[k] = rows[i].copy()

        self._advance(claimed)
        return slot_of

    def _advance(self, claimed: set[int]) -> None:
        for k in range(self.K):
            if k in claimed:
                self._missing[k] = 0
                self._age[k] += 1
            elif self._identity[k] is not None:
                self._missing[k] += 1
                if self._missing[k] > self.vanish_after:
                    self._identity[k] = None
                    self._prev_row[k] = None
                    self._age[k] = 0
        free_now = [k for k in range(self.K) if k not in claimed]
        self._background_slot = free_now[0] if free_now else (self.K - 1)
