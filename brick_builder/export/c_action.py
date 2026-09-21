"""`CActionNet`: a ctypes wrapper around the compiled `action_inverse` and `action_predict`.

Presents the same call as `brick_builder.action.models.mlp.Mlp.forward`, so a closed-loop episode
can be run with the C inverse in the loop (`CInverse` below) and the two compared on the same
seeds.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_FLOAT_P = ctypes.POINTER(ctypes.c_float)


class CActionNet:
    """Loads `action_float.so` and calls its two entry points."""

    def __init__(self, lib_path: Path, inv_in: int, inv_out: int, pred_in: int, pred_out: int):
        self.inv_in, self.inv_out = int(inv_in), int(inv_out)
        self.pred_in, self.pred_out = int(pred_in), int(pred_out)
        self._lib = ctypes.CDLL(str(Path(lib_path).resolve()))
        for name in ("action_inverse", "action_predict"):
            fn = getattr(self._lib, name)
            fn.argtypes = [_FLOAT_P, _FLOAT_P]
            fn.restype = None
        self._inv_buf = (ctypes.c_float * self.inv_out)()
        self._pred_buf = (ctypes.c_float * self.pred_out)()

    def _call(self, fn, buf, x: np.ndarray, n_in: int) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (n_in,):
            raise ValueError(f"expected shape ({n_in},), got {x.shape}")
        fn(x.ctypes.data_as(_FLOAT_P), ctypes.cast(buf, _FLOAT_P))
        return np.ctypeslib.as_array(buf).copy()

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._lib.action_inverse, self._inv_buf, x, self.inv_in)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._call(self._lib.action_predict, self._pred_buf, x, self.pred_in)

    def inverse_batch(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.inverse(row) for row in np.asarray(x, dtype=np.float32)])

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.predict(row) for row in np.asarray(x, dtype=np.float32)])


class CInverse:
    """`LearnedInverse` with the forward pass in C.

    Same interface as `brick_builder.action.models.inverse.LearnedInverse`, so
    `TrackingEpisode(inverse=...)` takes either and a whole episode can be run through the
    exported network.
    """

    def __init__(self, net: CActionNet, features, *, residual: bool = False):
        self.net = net
        self._features = features
        self.residual = bool(residual)
        self.calls = 0

    @classmethod
    def load(cls, lib_path: Path, weights: Path | None = None, *, residual: bool = False):
        from brick_builder.action.models.inverse import LearnedInverse
        from brick_builder.action.models.mlp import Mlp

        reference = LearnedInverse.load(weights)
        g = reference.net
        predictor = Mlp.load(Path(weights).parent / "predictor.npz" if weights is not None
                             else Path(__file__).resolve().parents[1] / "action" / "weights"
                             / "predictor.npz")
        net = CActionNet(lib_path, g.sizes[0], g.sizes[-1], predictor.sizes[0],
                         predictor.sizes[-1])
        return cls(net, reference.features, residual=residual)

    def delta(self, *, state, target, found, want, mech, in_hand: bool,
              analytic: np.ndarray | None = None) -> np.ndarray:
        x = self._features(state=state, target=target, found=found, want=want, mech=mech,
                           in_hand=in_hand)
        out = np.asarray(self.net.inverse(x), dtype=np.float64)
        self.calls += 1
        if self.residual and analytic is not None:
            return np.asarray(analytic, dtype=np.float64) + out
        return out
