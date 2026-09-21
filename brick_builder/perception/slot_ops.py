"""Slot-space tools for the controller: comparing, editing, selecting, and logging `SceneState`s.

Everything here operates purely on slot rows (`SceneState.slots`, read and written through the
schema's named fields) -- never on a decode, an `instance` array, or any other ground-truth
channel. This is the toolbox the (future) action model uses to express "what I expect to happen
next" as an edited, *intended* `SceneState`, compare it against what a producer then actually
observes, and log `(slot state, joint state, action)` tuples at the vision rate for later
training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .scene_model import SceneState

# ------------------------------------------------------------------------------- comparing ---


def slot_delta(intended: SceneState, actual: SceneState) -> np.ndarray:
    """`[K, W]`: `intended.slots - actual.slots`.

    Empty-slot handling is deliberately simple and symmetric: an all-zero row compares against an
    all-zero row as exactly zero (a correctly predicted "nothing here"), and an all-zero row
    compared against a real object's row is a large, real delta in every field that object's row
    has non-zero -- "predicted an object here and none appeared" (or the reverse) should look
    like a big surprise, not get masked out. Raises if the two states do not share a schema width
    or slot count (comparing a placeholder-schema state against an opaque one is not meaningful).
    """
    if intended.schema.W != actual.schema.W:
        raise ValueError(f"schema width mismatch: intended W={intended.schema.W}, actual W={actual.schema.W}")
    if intended.K != actual.K:
        raise ValueError(f"K mismatch: intended K={intended.K}, actual K={actual.K}")
    return intended.slots.astype(np.float64) - actual.slots.astype(np.float64)


def slot_distance(intended: SceneState, actual: SceneState, *, weights: np.ndarray | None = None) -> float:
    """A single scalar summarizing `slot_delta`: its root-mean-square, optionally weighted per
    column (e.g. to count a colour mismatch for less than a position one) -- `0.0` for an exact
    match, growing with how surprised the comparison should be."""
    delta = slot_delta(intended, actual)
    if weights is not None:
        delta = delta * np.asarray(weights, dtype=np.float64)[None, :]
    return float(np.sqrt(np.mean(delta ** 2)))


# -------------------------------------------------------------------------------- selecting ---


def select_slot(
    state: SceneState, *,
    colour: tuple[float, float, float] | None = None, colour_tol: float = 0.7,
    size: tuple[float, float] | None = None, size_tol: float = 0.6,
    min_mass: float | None = None, max_mass: float | None = None,
    exclude: Sequence[int] = (),
) -> int | None:
    """Which slot best matches the given constraints, read ONLY from that slot's own published
    fields (mean colour, mass, the second-moment ellipse's axis lengths) -- never a decode. This
    is how a caller finds "the red 2x4" (`select_slot(state, colour=(0.77, 0.09, 0.09), size=
    (studs-converted axis lengths))`) using slot numbers alone.

    `colour` is a plain `[0, 1]` RGB triple (converted internally to the state's own `2*rgb - 1`
    storage convention); `size` is a `(major, minor)` pair of second-moment axis lengths in the
    same normalized window units `SceneState.orientation_and_axes` returns, unordered (compared
    to the slot's own axis lengths sorted the same way, so orientation does not matter).

    Returns `None` if nothing satisfies every given constraint within its tolerance (including:
    the state's schema does not have a field a constraint needs).
    """
    best, best_score = None, math.inf
    for k in range(state.K):
        if k in exclude or state.is_empty(k):
            continue
        if min_mass is not None and (not state.schema.has("mass") or state.mass(k) < min_mass):
            continue
        if max_mass is not None and (not state.schema.has("mass") or state.mass(k) > max_mass):
            continue

        score = 0.0
        if colour is not None:
            if not state.schema.has("colour"):
                return None
            ref = np.array([2.0 * c - 1.0 for c in colour])
            d = float(np.linalg.norm(np.array(state.colour(k)) - ref))
            if d > colour_tol:
                continue
            score += d
        if size is not None:
            if not state.schema.has("moments"):
                return None
            _, axes = state.orientation_and_axes(k)
            d = math.hypot(*(a - b for a, b in zip(sorted(size), sorted(axes))))
            if d > size_tol:
                continue
            score += d

        if best is None or score < best_score:
            best, best_score = k, score
    return best


# ------------------------------------------------------------------------------------ editing ---


def set_slot_blob(
    state: SceneState, k: int, *,
    centre: tuple[float, float] | None = None, mass: float | None = None,
    colour: tuple[float, float, float] | None = None, moments: tuple[float, float, float] | None = None,
) -> SceneState:
    """An edited copy of `state` (same `SceneState`, a new `slots` array) with slot `k`'s row's
    given fields overwritten and everything else -- including every other slot -- unchanged. This
    is the basic move: "this slot's blob centred at image position `p` with mass `m`"."""
    fields = state.schema.fields
    for name, value in (("centre", centre), ("mass", mass), ("colour", colour), ("moments", moments)):
        if value is not None and name not in fields:
            raise KeyError(f"schema has no {name!r} field")

    slots = state.slots.copy()
    row = slots[k].copy()
    if centre is not None:
        row[fields["centre"]] = centre
    if mass is not None:
        row[fields["mass"]] = mass
    if colour is not None:
        row[fields["colour"]] = [2.0 * c - 1.0 for c in colour]
    if moments is not None:
        row[fields["moments"]] = moments
    slots[k] = row
    return replace(state, slots=slots)


def set_slot_offset(
    state: SceneState, k: int, *, reference_slot: int, offset: tuple[float, float],
    scale: tuple[float, float] | None = None,
) -> SceneState:
    """An edited copy of `state` with slot `k`'s centre placed at `offset` from `reference_slot`'s
    own centre, expressed in `reference_slot`'s own orientation and (unless `scale` overrides it)
    its own second-moment axis lengths -- exactly what "the held brick belongs over stud cell
    `(x, y)` of the plate" needs to express, using only the plate blob's own centre, orientation
    and extent (`reference_slot`), with no metric camera geometry anywhere in this function.
    `offset` is in units of `scale` (or of `reference_slot`'s own axis lengths): `(1, 0)` means
    "one of that slot's own major-axis lengths further along its own major axis".
    """
    angle, axes = state.orientation_and_axes(reference_slot)
    sx, sy = scale if scale is not None else axes
    ox, oy = offset
    c, s = math.cos(angle), math.sin(angle)
    dx = c * (ox * sx) - s * (oy * sy)
    dy = s * (ox * sx) + c * (oy * sy)
    rx, ry = state.centre(reference_slot)
    return set_slot_blob(state, k, centre=(rx + dx, ry + dy))


# ------------------------------------------------------------------------------------ logging ---


@dataclass
class LogRecord:
    """One vision tick's worth of logged data, for training a future action model: the slot
    state (as raw arrays, not the full `SceneState`, since `window`/`schema` are not easily
    stacked into one archive -- see `write_log`), the arm's joint state, and the action taken."""

    t: float
    slots: np.ndarray       # [K, W]
    depth: np.ndarray       # [K]
    slot_ids: np.ndarray    # [K]
    joint_state: np.ndarray  # [n_joints] (or [n_joints, 2] for pos+vel -- caller's convention)
    action: np.ndarray       # [action_dim]

    @classmethod
    def from_state(cls, state: SceneState, *, joint_state: np.ndarray, action: np.ndarray) -> "LogRecord":
        return cls(
            t=state.t, slots=np.asarray(state.slots), depth=np.asarray(state.depth),
            slot_ids=np.asarray(state.slot_ids), joint_state=np.asarray(joint_state),
            action=np.asarray(action),
        )


def write_log(records: Sequence[LogRecord], path: str | Path) -> None:
    """Writes `records` (in order) to one `.npz` archive at `path` -- one array per field, each
    with a leading `n_records` axis. `joint_state`/`action` must have the same shape across every
    record (a fixed joint/action dimension for the whole log)."""
    if not records:
        raise ValueError("write_log needs at least one record")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array([r.t for r in records], dtype=np.float64),
        slots=np.stack([r.slots for r in records]).astype(np.float32),
        depth=np.stack([r.depth for r in records]).astype(np.float32),
        slot_ids=np.stack([r.slot_ids for r in records]).astype(np.int64),
        joint_state=np.stack([r.joint_state for r in records]).astype(np.float32),
        action=np.stack([r.action for r in records]).astype(np.float32),
    )


def read_log(path: str | Path) -> list[LogRecord]:
    """Inverse of `write_log`."""
    with np.load(Path(path)) as data:
        n = data["t"].shape[0]
        return [
            LogRecord(
                t=float(data["t"][i]), slots=data["slots"][i], depth=data["depth"][i],
                slot_ids=data["slot_ids"][i], joint_state=data["joint_state"][i], action=data["action"][i],
            )
            for i in range(n)
        ]
