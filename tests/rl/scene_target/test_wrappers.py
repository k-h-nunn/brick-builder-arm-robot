"""`PredictionScoring`: online next-frame prediction error."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA as S  # noqa: E402
from brick_builder.rl.scene_target.codec import MM_PER_WINDOW_UNIT  # noqa: E402
from brick_builder.rl.scene_target.wrappers import PredictionScoring, centre_error_mm  # noqa: E402


def test_centre_error_pairs_slots_by_id():
    a = np.zeros((8, S.W))
    b = np.zeros((8, S.W))
    a[1, S.fields["centre"]] = (0.1, 0.0)
    b[4, S.fields["centre"]] = (0.12, 0.0)
    ids_a = np.array([-1, 7, -1, -1, -1, -1, -1, -1])
    ids_b = np.array([-1, -1, -1, -1, 7, -1, -1, -1])
    assert centre_error_mm(a, ids_a, b, ids_b) == pytest.approx(MM_PER_WINDOW_UNIT * 0.02)
    assert math.isnan(centre_error_mm(a, ids_a, b, np.full(8, -1)))
    assert centre_error_mm(a, ids_a, b, ids_b, mm_scale=210.0) == pytest.approx(210.0 * 0.02)


@pytest.mark.slow
def test_copy_last_predictor_scores_as_the_copy_last_baseline(tmp_path):
    import gymnasium as gym

    import brick_builder.rl.scene_target as st

    env = PredictionScoring(gym.make(st.ENV_ID, cache_dir=str(tmp_path), feedforward=True))
    try:
        env.reset(seed=0, options={"world_seed": 0})
        errs = []
        for _ in range(6):
            _, _, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
            errs.append((info["pred_err_mm"], info["copy_last_err_mm"], info["linear_err_mm"]))
            assert "pred_err_vs_truth_mm" not in info
            if term or trunc:
                break
    finally:
        env.close()
    for pred, copy, _ in errs:
        assert pred == copy and pred >= 0.0
    assert math.isnan(errs[0][2]) and all(np.isfinite(e[2]) for e in errs[1:])
    # the replayed path moves, so copy-last has a non-zero error while the arm moves
    assert max(e[1] for e in errs) > 0.0
