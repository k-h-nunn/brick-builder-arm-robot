"""Shapes, normalization, determinism and cost of the two action models."""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.action.models import features as ft
from brick_builder.action.models.mlp import Mlp, std_floor


def _net(sizes, seed: int = 0) -> Mlp:
    rng = np.random.default_rng(seed)
    layers = [(rng.normal(0.0, 0.3, size=(sizes[i + 1], sizes[i])),
               rng.normal(0.0, 0.1, size=sizes[i + 1])) for i in range(len(sizes) - 1)]
    return Mlp(layers=layers, in_mean=rng.normal(size=sizes[0]),
               in_std=std_floor(np.abs(rng.normal(size=sizes[0])) + 0.2),
               out_mean=rng.normal(size=sizes[-1]),
               out_std=std_floor(np.abs(rng.normal(size=sizes[-1])) + 0.2))


def test_encoded_widths_match_the_declared_dimensions() -> None:
    rows = np.arange(ft.N_SLATE * ft.ROW_W, dtype=float).reshape(ft.N_SLATE, ft.ROW_W)
    valid = np.array([1.0, 1.0, 0.0, 1.0])
    weight = np.full(ft.N_SLATE, 0.8)
    mech = np.arange(ft.MECH_W, dtype=float)

    x = ft.encode_inverse(rows, valid, weight, rows * 0.5, valid, mech, 1.0, 2.0)
    assert x.shape == (ft.INVERSE_IN,)
    y = ft.encode_predictor(rows, valid, weight, mech, np.ones(ft.ACTION_W), 1.0)
    assert y.shape == (ft.PREDICTOR_IN,)


def test_an_unmatched_slate_entry_contributes_nothing() -> None:
    rows = np.full((ft.N_SLATE, ft.ROW_W), 3.0)
    valid = np.array([1.0, 0.0, 1.0, 1.0])
    both = ft.encode_inverse(rows, valid, np.ones(ft.N_SLATE), np.zeros_like(rows), valid,
                             np.zeros(ft.MECH_W), 0.0, 3.0)
    per_row = both[: ft.N_SLATE * ft.PER_ROW].reshape(ft.N_SLATE, ft.PER_ROW)
    assert np.all(per_row[1] == 0.0), "an unmatched entry must be flagged and zeroed"
    assert per_row[0, 0] == 1.0 and np.all(per_row[0, 2:] == 3.0)


def test_a_target_the_slate_did_not_match_is_not_differenced() -> None:
    rows = np.full((ft.N_SLATE, ft.ROW_W), 2.0)
    x = ft.encode_inverse(rows, np.ones(ft.N_SLATE), np.ones(ft.N_SLATE), rows * 0.0,
                          np.array([1.0, 1.0, 1.0, 0.0]), np.zeros(ft.MECH_W), 1.0, 3.0)
    per_row = x[: ft.N_SLATE * ft.PER_ROW].reshape(ft.N_SLATE, ft.PER_ROW)
    assert np.all(per_row[3] == 0.0)


def test_the_slate_order_does_not_depend_on_the_order_the_slots_came_back_in() -> None:
    a = ft.slate_names(("placed brick 1", "baseplate", ft.HELD_NAME, "placed brick 0"))
    b = ft.slate_names(("baseplate", "placed brick 0", "placed brick 1", ft.HELD_NAME))
    assert a == b == ["baseplate", "placed brick 0", "placed brick 1"]
    assert ft.HELD_NAME not in a


def test_normalization_maps_the_mean_to_zero_and_saturates_a_far_input() -> None:
    net = _net((6, 8, 3))
    assert np.allclose(net.normalize(net.in_mean), 0.0, atol=1e-6)
    far = net.in_mean + 1000.0 * net.in_std
    assert np.allclose(net.normalize(far), net.clip)
    assert np.allclose(net.normalize(-far + 2.0 * net.in_mean), -net.clip)


def test_a_saved_checkpoint_round_trips_exactly(tmp_path) -> None:
    net = _net((5, 7, 2), seed=3)
    path = net.save(tmp_path / "net.npz")
    back = Mlp.load(path)
    x = np.random.default_rng(1).normal(size=(16, 5))
    assert np.array_equal(net.forward(x), back.forward(x))
    assert back.sizes == net.sizes and back.n_params == net.n_params


def test_the_forward_pass_is_deterministic_and_batches_like_it_singles() -> None:
    net = _net((9, 12, 4), seed=7)
    x = np.random.default_rng(2).normal(size=(5, 9))
    assert np.array_equal(net.forward(x), net.forward(x)), "the same call twice, bit for bit"
    # A batch goes through BLAS and a single row does not, so the two agree to float32 rounding
    # rather than exactly -- which is also the bound the C export is checked against.
    assert np.allclose(net.forward(x)[2], net.forward(x[2]), rtol=0, atol=1e-6)


def test_the_reported_cost_is_the_arithmetic_the_forward_pass_does() -> None:
    net = _net((10, 16, 3))
    assert net.macs == 10 * 16 + 16 * 3
    assert net.n_params == net.macs + 16 + 3
    assert net.weight_bytes == 4 * (net.n_params + 10 + 10 + 3 + 3)


@pytest.mark.parametrize("kind", ("inverse", "predictor"))
def test_the_shipped_checkpoints_have_the_declared_shape(kind: str) -> None:
    from brick_builder.action.models.inverse import WEIGHTS

    path = WEIGHTS / f"{kind}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} has not been trained yet")
    net = Mlp.load(path)
    expect_in = ft.INVERSE_IN if kind == "inverse" else ft.PREDICTOR_IN
    expect_out = ft.INVERSE_OUT if kind == "inverse" else ft.PREDICTOR_OUT
    assert net.sizes[0] == expect_in and net.sizes[-1] == expect_out
    assert net.macs < 50_000, "the budget is a few tens of thousands of multiply-accumulates"
