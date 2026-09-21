"""The fixed-width vectors the predictor and the learned inverse read.

Both models consume slot rows and nothing else. A vision tick publishes up to `K = 8` slots, of
which only a handful are the objects the step lines itself up on, so the two models read a fixed
**slate**: three reference slots, ordered by their own titles so the order is the same every tick
and in every episode, plus the brick in the hand. A slate entry that is not matched this tick is
marked invalid and zeroed rather than dropped, which keeps the width fixed.

Per slate entry, eight numbers are read from the slot's own row through the schema -- mass, the
two centre coordinates, the three second moments and the two components of `orient4` -- plus a
validity flag and the blob's weight (`1 - border fraction`). Colour and age are left out: colour
is what the slot was *selected* by, so its difference against the target is near zero by
construction, and age counts frames rather than describing the scene.

The inverse reads differences (observed minus target); the predictor reads the observed rows
themselves together with the action, and predicts the change of those same eight fields. Nothing
here renders, reads the simulator or decodes: it is arithmetic on published slot rows.
"""

from __future__ import annotations

import numpy as np

ROW_FIELDS: tuple[str, ...] = ("mass", "centre", "moments", "orient4")
"""Slot-row fields both models read, in order. Widths 1, 2, 3, 2 -- eight numbers per slot."""

ROW_W = 8
"""Numbers read from one slot row."""

REF_SLOTS = 3
"""Reference slots in the slate. Three is what the placement task has in view (the baseplate and
the two bricks already on it); a step with fewer fills the rest with invalid entries."""

N_SLATE = REF_SLOTS + 1
"""Reference slots plus the brick in the hand, which is always the last entry."""

PER_ROW = 2 + ROW_W
"""Validity flag, weight, and the eight fields."""

MECH_W = 12
"""Mechanical state the models read: six joint positions and six joint speeds, as the bus reports
them. The six loads are logged as well but not read -- see `docs/guide/12-closed-loop-control.md`."""

ACTION_W = 4
"""Tool-target change over one vision tick: dx, dy, dz [m] and spin [rad]."""

INVERSE_OUT = 3
"""What the inverse commands: the correction increment in dx, dy [m] and spin [rad]. The
controller does not correct z, so the analytic inverse has no z column either."""

INVERSE_IN = N_SLATE * PER_ROW + MECH_W + 2
"""Slate differences, mechanical state, and two context flags (brick in hand, matched references)."""

PREDICTOR_IN = N_SLATE * PER_ROW + MECH_W + ACTION_W + 1
PREDICTOR_OUT = N_SLATE * ROW_W

HELD_NAME = "the brick in the hand"
"""The title `twin.build_step` gives the held brick's reference. Never a thing to line up on; its
row is the last slate entry because it says how the brick sits in the jaws."""


def row_of(state, k: int) -> np.ndarray:
    """One slot's eight read fields, through the schema -- never by column number."""
    row = state.slots[int(k)]
    return np.concatenate(
        [np.asarray(state.schema.get(row, name), dtype=np.float64).reshape(-1)
         for name in ROW_FIELDS]
    )


def slate_names(order) -> list[str]:
    """The reference titles of a slate, sorted, at most `REF_SLOTS` of them.

    Sorted so a step that shows the same objects builds the same slate whatever order the slots
    came back in; the held brick is appended separately by `read_slate`.
    """
    return sorted(n for n in order if n != HELD_NAME)[:REF_SLOTS]


def read_slate(state, slots: dict, weights: dict, names: list[str],
               held_slot: int | None, held_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(rows [N_SLATE, ROW_W], valid [N_SLATE], weight [N_SLATE])` for one scene state.

    `slots` maps a title to the slot index it was matched to this tick (`None` when it was not).
    """
    rows = np.zeros((N_SLATE, ROW_W), dtype=np.float64)
    valid = np.zeros(N_SLATE, dtype=np.float64)
    weight = np.zeros(N_SLATE, dtype=np.float64)
    for i, name in enumerate(names[:REF_SLOTS]):
        k = slots.get(name)
        if k is None:
            continue
        rows[i] = row_of(state, k)
        valid[i] = 1.0
        weight[i] = float(weights.get(name, 1.0))
    if held_slot is not None:
        rows[REF_SLOTS] = row_of(state, int(held_slot))
        valid[REF_SLOTS] = 1.0
        weight[REF_SLOTS] = float(held_weight)
    return rows, valid, weight


def encode_inverse(obs_rows: np.ndarray, obs_valid: np.ndarray, obs_weight: np.ndarray,
                   tgt_rows: np.ndarray, tgt_valid: np.ndarray, mech: np.ndarray,
                   in_hand: float, n_refs: float) -> np.ndarray:
    """`[INVERSE_IN]`: the slate differences, the mechanical state and two context flags.

    An entry is differenced only where both states matched it; anywhere else the difference is
    zero and the flag says the entry carries nothing. Scaling and clipping happen inside the
    network's own normalizer (`models.mlp.Mlp`), from statistics saved with the checkpoint, so a
    target far outside the range the loop ever sees reads as a direction rather than a magnitude.
    """
    both = (np.asarray(obs_valid) > 0) & (np.asarray(tgt_valid) > 0)
    diff = (np.asarray(obs_rows, dtype=np.float64) - np.asarray(tgt_rows, dtype=np.float64))
    diff = diff * both[:, None]
    per_row = np.concatenate(
        [both.astype(np.float64)[:, None], (np.asarray(obs_weight) * both)[:, None], diff], axis=1)
    return np.concatenate([per_row.reshape(-1), np.asarray(mech, dtype=np.float64).reshape(-1)[:MECH_W],
                           [float(in_hand), float(n_refs)]])


def encode_predictor(rows: np.ndarray, valid: np.ndarray, weight: np.ndarray,
                     mech: np.ndarray, action: np.ndarray, in_hand: float) -> np.ndarray:
    """`[PREDICTOR_IN]`: the observed slate, the mechanical state, the action, and one flag."""
    per_row = np.concatenate([np.asarray(valid, dtype=np.float64)[:, None],
                              np.asarray(weight, dtype=np.float64)[:, None],
                              np.asarray(rows, dtype=np.float64) * np.asarray(valid)[:, None]],
                             axis=1)
    return np.concatenate([per_row.reshape(-1),
                           np.asarray(mech, dtype=np.float64).reshape(-1)[:MECH_W],
                           np.asarray(action, dtype=np.float64).reshape(-1)[:ACTION_W],
                           [float(in_hand)]])
