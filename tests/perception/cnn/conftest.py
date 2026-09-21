"""Fixtures and the shared scene-state contract checks for the convolutional scene model.

`assert_scene_state_contract` is the set of invariants *every* producer in `brick_builder.perception`
has to satisfy, written once here and applied in `test_source.py` to the ideal ground-truth
source, the colour-threshold source and the convolutional one in the same test -- so "it fills the
same contract" is checked rather than asserted in a docstring.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / "data_cache" / "vision_model"
WEIGHTS = REPO_ROOT / "brick_builder" / "perception" / "weights" / "vision_model.pt"


def assert_scene_state_contract(state, K: int = 8) -> None:
    """Every invariant `scene_model.SceneState` promises its consumers."""
    from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA

    assert state.slots.shape == (K, PLACEHOLDER_SLOT_SCHEMA.W)
    assert state.slots.dtype == np.float32
    assert state.depth.shape == (K,)
    assert state.slot_ids.shape == (K,)
    assert state.K == K
    assert state.schema.W == state.W
    assert np.all(np.isfinite(state.slots))
    assert np.all((state.depth >= 0.0) & (state.depth <= 1.0))
    for k in range(K):
        if state.is_empty(k):
            assert state.depth[k] == pytest.approx(0.5)
            assert not np.any(state.slots[k])
        else:
            assert 0.0 <= state.mass(k) <= 1.0
            cx, cy = state.centre(k)
            assert -1.001 <= cx <= 1.001 and -1.001 <= cy <= 1.001
            sxx, sxy, syy = state.moments(k)
            assert sxx >= 0.0 and syy >= 0.0
            for c in state.colour(k):
                assert -1.001 <= c <= 1.001
            assert 0.0 <= state.border_fraction(k) <= 1.0
            assert 0.0 <= state.age(k) <= 1.0
    assert isinstance(state.source, str) and state.source


def cached_samples(split: str = "val", limit: int = 20):
    """A few cached frames, or `None` when the (git-ignored, rebuildable) cache is absent."""
    from brick_builder.perception.cnn.data import load_samples, shard_paths

    if not shard_paths(CACHE, split):
        return None
    return load_samples(CACHE, split, limit=limit)


@pytest.fixture(scope="session")
def val_samples():
    samples = cached_samples("val", 20)
    if samples is None:
        pytest.skip("no rendered dataset under data_cache/ -- run scripts/train_vision_model.py "
                    "--generate to build one")
    return samples


@pytest.fixture(scope="session")
def trained_source():
    if not WEIGHTS.exists():
        pytest.skip(f"no trained checkpoint at {WEIGHTS}")
    import torch

    from brick_builder.perception.cnn.source import CnnSceneModel

    torch.set_num_threads(2)
    return CnnSceneModel(checkpoint=WEIGHTS)
