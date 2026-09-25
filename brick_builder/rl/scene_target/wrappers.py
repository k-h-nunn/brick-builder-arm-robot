"""`PredictionScoring`: next-frame prediction error per tick, for any controller.

Before each step the wrapper asks a predictor for `z_{t+1}` (rows in the order of the current
observation `z_t`); after the step it writes to `info`:

| Key | Contents |
|---|---|
| `pred_err_mm` | predicted vs observed centres, slots paired by slot id [mm-eq] |
| `copy_last_err_mm` | `z_t` vs observed (the copy-last baseline) [mm-eq] |
| `linear_err_mm` | `z_t + (z_t - z_{t-1})` vs observed (linear extrapolation) [mm-eq] |
| `pred_err_vs_truth_mm` | predicted vs the ground-truth rows of the same next frame, paired by appearance (sources other than `ideal`) [mm-eq] |

Errors are the RMS centre difference of the slots present in both states, weighted by
`1 - border`, times 87.4 mm per window unit; NaN when no slot pairs up.
"""

from __future__ import annotations

import math
from typing import Callable

import gymnasium as gym
import numpy as np

from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SlotSchema
from .codec import MM_PER_WINDOW_UNIT, SceneDistance, SlotFrame

__all__ = ["PredictionScoring", "centre_error_mm"]

Predictor = Callable[[dict, np.ndarray], np.ndarray]
"""`predictor(obs, u) -> rows [K, W]`: the predicted next rows, in the slot order of `obs`."""


def centre_error_mm(pred_rows, pred_ids, obs_rows, obs_ids,
                    schema: SlotSchema = PLACEHOLDER_SLOT_SCHEMA) -> float:
    """RMS centre error [mm-eq] over slots with the same non-negative slot id in both states."""
    c = schema.fields["centre"]
    b = schema.fields.get("border")
    num = den = 0.0
    for i, sid in enumerate(np.asarray(pred_ids)):
        if sid < 0:
            continue
        j = np.nonzero(np.asarray(obs_ids) == sid)[0]
        if not j.size:
            continue
        j = int(j[0])
        w = 1.0
        if b is not None:
            w = max(0.0, 1.0 - max(float(pred_rows[i][b][0]), float(obs_rows[j][b][0])))
        d = np.asarray(pred_rows[i][c], dtype=np.float64) - np.asarray(obs_rows[j][c], dtype=np.float64)
        num += w * float(np.dot(d, d))
        den += w
    if den <= 0.0:
        return float("nan")
    return MM_PER_WINDOW_UNIT * math.sqrt(num / den)


class PredictionScoring(gym.Wrapper):
    """Scores a next-frame predictor online; `predictor=None` scores copy-last."""

    def __init__(self, env: gym.Env, predictor: Predictor | None = None) -> None:
        super().__init__(env)
        self.predictor = predictor
        self._obs = None
        self._ids = None
        self._prev_rows = None
        self._prev_ids = None
        self._distance = SceneDistance(PLACEHOLDER_SLOT_SCHEMA)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs, self._ids = obs, np.asarray(info["slot_ids"]).copy()
        self._prev_rows = self._prev_ids = None
        return obs, info

    def step(self, action):
        z_t = np.asarray(self._obs["actor"]["z"], dtype=np.float64)
        ids_t = self._ids
        pred = z_t if self.predictor is None else np.asarray(
            self.predictor(self._obs, np.asarray(action)), dtype=np.float64)
        linear = None
        if self._prev_rows is not None:
            linear = z_t.copy()
            c = PLACEHOLDER_SLOT_SCHEMA.fields["centre"]
            for i, sid in enumerate(ids_t):
                j = np.nonzero(self._prev_ids == sid)[0] if sid >= 0 else []
                if len(j):
                    linear[i, c] = 2.0 * z_t[i, c] - self._prev_rows[int(j[0]), c]
        obs, reward, terminated, truncated, info = self.env.step(action)
        z_next = np.asarray(obs["actor"]["z"], dtype=np.float64)
        ids_next = np.asarray(info["slot_ids"])
        info["pred_err_mm"] = centre_error_mm(pred, ids_t, z_next, ids_next)
        info["copy_last_err_mm"] = centre_error_mm(z_t, ids_t, z_next, ids_next)
        info["linear_err_mm"] = (float("nan") if linear is None
                                 else centre_error_mm(linear, ids_t, z_next, ids_next))
        if info.get("source", "ideal") != "ideal":
            gt = np.asarray(obs["privileged"]["gt_z"], dtype=np.float64)
            gt_ids = np.asarray(info["gt_slot_ids"])
            depth = np.asarray(self._obs["actor"]["z_depth"])
            pf = SlotFrame.from_arrays(pred.astype(np.float32), depth, None, ids_t)
            gf = SlotFrame.from_arrays(gt.astype(np.float32),
                                       np.full(gt.shape[0], 0.5, dtype=np.float32),
                                       np.any(gt != 0.0, axis=1), gt_ids)
            perm = self._distance.match(pf, gf)
            paired_ids = np.full(gt.shape[0], -1)
            for j, i in enumerate(perm):
                if i >= 0:
                    paired_ids[j] = ids_t[i]
            info["pred_err_vs_truth_mm"] = centre_error_mm(pred, ids_t, gt, paired_ids)
        self._prev_rows, self._prev_ids = z_t, ids_t
        self._obs, self._ids = obs, ids_next.copy()
        return obs, reward, terminated, truncated, info
