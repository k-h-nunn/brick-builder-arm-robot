"""The learned one-step inverse: "make the scene look like that" -> a tool correction.

`G(observed scene state, target scene state, mechanical state)` returns the correction increment
for one vision tick, in the units the analytic inverse already commands: metres of tool motion in
x and y and radians of spin. It is dropped in where `-lambda * J^+ (s - s*)` was, behind the same
gates, the same saturation and the same integrator, so the two can be compared on the same ten
seeds with nothing else changed.

Inputs are slot-row differences only (`models.features`). Nothing here renders, reads the
simulator, decodes a frame or imports torch: the forward pass is `models.mlp.Mlp`, a few thousand
multiply-accumulates of plain NumPy, and the same arithmetic the C export performs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import HELD_NAME, encode_inverse, read_slate, slate_names
from .mlp import Mlp

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
"""Where the shipped checkpoints live."""


def tick_slate(state, target, found, want):
    """The slate both models read, for one tick, from the two `FeatureSet`s the servo built.

    Returns `(names, obs_rows, obs_valid, obs_weight, tgt_rows, tgt_valid)`.
    """
    names = slate_names(found.order)
    obs_rows, obs_valid, obs_weight = read_slate(
        state, found.slots, found.weights, names,
        found.slots.get(HELD_NAME), found.weights.get(HELD_NAME, 0.0))
    tgt_rows, tgt_valid, _ = read_slate(
        target, want.slots, want.weights, names,
        want.slots.get(HELD_NAME), want.weights.get(HELD_NAME, 0.0))
    return names, obs_rows, obs_valid, obs_weight, tgt_rows, tgt_valid


class LearnedInverse:
    """`G`, ready to be called once per vision tick."""

    def __init__(self, net: Mlp, *, residual: bool = False) -> None:
        self.net = net
        self.residual = bool(residual)
        """Add the network's output to the analytic correction instead of replacing it. The
        remedy tried when the replacement measured worse; reported either way."""
        self.calls = 0

    @classmethod
    def load(cls, path: str | Path | None = None, *, residual: bool = False) -> "LearnedInverse":
        path = Path(path) if path is not None else WEIGHTS / "inverse.npz"
        return cls(Mlp.load(path), residual=residual)

    def features(self, *, state, target, found, want, mech, in_hand: bool) -> np.ndarray:
        names, obs_rows, obs_valid, obs_weight, tgt_rows, tgt_valid = tick_slate(
            state, target, found, want)
        return encode_inverse(obs_rows, obs_valid, obs_weight, tgt_rows, tgt_valid, mech,
                              float(bool(in_hand)), float(int(np.sum(obs_valid[:-1]))))

    def delta(self, *, state, target, found, want, mech, in_hand: bool,
              analytic: np.ndarray | None = None) -> np.ndarray:
        """The correction increment for this tick, `[3]` = (dx [m], dy [m], spin [rad]).

        The caller applies the same saturation and the same integrator it applies to the analytic
        term; nothing is clipped here.
        """
        x = self.features(state=state, target=target, found=found, want=want, mech=mech,
                          in_hand=in_hand)
        out = np.asarray(self.net.forward(x), dtype=np.float64)
        self.calls += 1
        if self.residual and analytic is not None:
            return np.asarray(analytic, dtype=np.float64) + out
        return out
