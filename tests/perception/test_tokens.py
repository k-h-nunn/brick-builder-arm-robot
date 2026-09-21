"""Unit tests for `brick_builder.perception.tokens`: query builders, slot statistics, the overlay figure.

None of this needs a vision model itself -- these are pure functions of arrays a decode would
produce, so they run unconditionally (no skip).
"""

from __future__ import annotations

import numpy as np
import torch

from brick_builder.perception.tokens import (
    ObjectToken,
    grid_queries,
    plot_decode_overlay,
    region_queries,
    slot_statistics,
)


def test_grid_queries_shape_and_spacing() -> None:
    px = grid_queries(step=128, res=256)
    assert px.dtype == np.int64
    assert px.shape == (4, 2)
    expected = {(64, 64), (192, 64), (64, 192), (192, 192)}
    assert set(map(tuple, px.tolist())) == expected


def test_grid_queries_rejects_bad_args() -> None:
    import pytest

    with pytest.raises(ValueError):
        grid_queries(step=0)
    with pytest.raises(ValueError):
        grid_queries(step=8, res=0)


def test_region_queries_matches_mask() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 3] = True  # row=1, col=3
    mask[4, 0] = True  # row=4, col=0
    px = region_queries(mask)
    assert px.shape == (2, 2)
    assert set(map(tuple, px.tolist())) == {(3, 1), (0, 4)}


def test_region_queries_empty_mask() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    px = region_queries(mask)
    assert px.shape == (0, 2)


def test_slot_statistics_centroid_and_colour() -> None:
    # Four query pixels, two strongly assigned to slot 0 and two to slot 1; the frame is red
    # where slot-0 pixels sit and blue where slot-1 pixels sit, so we know the right answer.
    px = torch.tensor([[[0, 0], [10, 0], [0, 10], [10, 10]]], dtype=torch.float32)  # [1,4,2]
    big = 50.0
    logits = torch.tensor([[[big, 0.0], [big, 0.0], [0.0, big], [0.0, big]]])  # [1,4,2]

    frame = np.zeros((16, 16, 3), dtype=np.float32)
    frame[0, 0] = frame[0, 10] = (1.0, 0.0, 0.0)  # red, note frame is [row,col,:]
    frame[10, 0] = frame[10, 10] = (0.0, 0.0, 1.0)  # blue
    frames = torch.as_tensor(frame)[None]  # [1,16,16,3]

    stats = slot_statistics(logits, px, frames)
    assert stats.mass.shape == (1, 2)
    np.testing.assert_allclose(stats.mass[0], [2.0, 2.0], atol=1e-3)
    np.testing.assert_allclose(stats.centroid[0, 0], [5.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(stats.centroid[0, 1], [5.0, 10.0], atol=1e-2)
    np.testing.assert_allclose(stats.mean_colour[0, 0], [1.0, 0.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(stats.mean_colour[0, 1], [0.0, 0.0, 1.0], atol=1e-2)


def test_slot_statistics_empty_slot_is_nan() -> None:
    px = torch.tensor([[[0, 0], [10, 0]]], dtype=torch.float32)
    big = 50.0
    # both queries land entirely on slot 0; slot 1 and slot 2 get essentially nothing
    logits = torch.tensor([[[big, 0.0, 0.0], [big, 0.0, 0.0]]])
    frames = torch.zeros(1, 16, 16, 3)
    stats = slot_statistics(logits, px, frames)
    assert stats.mass[0, 1] < 1e-3
    assert np.isnan(stats.centroid[0, 1]).all()


def test_object_token_fields() -> None:
    tok = ObjectToken(x=1.0, y=2.0, yaw=0.3, layer=1, colour="red", size=4.0, confidence=0.9, age=0.0, slot=2)
    assert (tok.x, tok.y, tok.yaw, tok.layer, tok.colour) == (1.0, 2.0, 0.3, 1, "red")
    assert (tok.size, tok.confidence, tok.age, tok.slot) == (4.0, 0.9, 0.0, 2)


def test_plot_decode_overlay_saves_png(tmp_path) -> None:
    rng = np.random.default_rng(0)
    frame = rng.random((32, 32, 3)).astype(np.float32)
    px = np.stack([rng.integers(0, 32, 20), rng.integers(0, 32, 20)], axis=-1)
    logits = torch.as_tensor(rng.normal(size=(20, 3)).astype(np.float32))

    out_path = tmp_path / "overlay.png"
    fig = plot_decode_overlay(frame, px, logits, save_path=str(out_path))
    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(fig)
