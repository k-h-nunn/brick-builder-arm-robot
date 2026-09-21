"""`CPolicy`: a ctypes wrapper around the compiled `policy_act` that presents the same
`act(obs)` call as the torch actor's deterministic path
(`brick_builder.rl.ppo_asym.evaluate`: normalize, take the Gaussian mean, clamp to the action
bounds).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_FLOAT_P = ctypes.POINTER(ctypes.c_float)


class CPolicy:
    """Loads a `policy_act(const float *obs, float *act)` shared library and calls it.

    `obs_dim` and `act_dim` must match the checkpoint the library was built from
    (`brick_builder.export.build.build_library` returns the header alongside the library, whose
    `POLICY_OBS_DIM`/`POLICY_ACT_DIM` macros this class does not re-parse -- the caller
    already has `obs_spec` from the same checkpoint via `brick_builder.rl.ppo_asym.load_actor`).
    """

    def __init__(self, lib_path: Path, obs_dim: int, act_dim: int):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self._lib = ctypes.CDLL(str(Path(lib_path).resolve()))
        self._lib.policy_act.argtypes = [_FLOAT_P, _FLOAT_P]
        self._lib.policy_act.restype = None
        self._act_buf = (ctypes.c_float * self.act_dim)()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Mean action for one observation, shape `(obs_dim,)` in, `(act_dim,)` out."""
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected obs shape ({self.obs_dim},), got {obs.shape}")
        obs_buf = obs.ctypes.data_as(_FLOAT_P)
        self._lib.policy_act(obs_buf, ctypes.cast(self._act_buf, _FLOAT_P))
        return np.ctypeslib.as_array(self._act_buf).copy()

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """`act` applied row-wise; a convenience for parity checks and benchmarks, not a
        real vectorized C call (the C side has no batch entry point, matching the firmware,
        which only ever runs one observation per control tick)."""
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        return np.stack([self.act(row) for row in obs])
