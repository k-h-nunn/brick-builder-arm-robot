"""The C export of the two action models: it compiles, and it computes what NumPy computes."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from brick_builder.action.models.features import (
    INVERSE_IN, INVERSE_OUT, PREDICTOR_IN, PREDICTOR_OUT,
)
from brick_builder.action.models.mlp import Mlp, std_floor

PARITY_BOUND = 1e-6


def _net(sizes, seed: int) -> Mlp:
    rng = np.random.default_rng(seed)
    layers = [(rng.normal(0.0, 0.6 / np.sqrt(sizes[i]), size=(sizes[i + 1], sizes[i])),
               rng.normal(0.0, 0.05, size=sizes[i + 1])) for i in range(len(sizes) - 1)]
    return Mlp(layers=layers, in_mean=rng.normal(0.0, 0.3, size=sizes[0]),
               in_std=std_floor(np.abs(rng.normal(0.0, 1.0, size=sizes[0])) + 0.3),
               out_mean=rng.normal(0.0, 1e-3, size=sizes[-1]),
               out_std=std_floor(np.full(sizes[-1], 1e-3)))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("no gcc on this machine")
    from brick_builder.export.action_to_c import build_library

    out = tmp_path_factory.mktemp("action_export")
    g = _net((INVERSE_IN, 64, 64, INVERSE_OUT), seed=1)
    f = _net((PREDICTOR_IN, 64, 64, PREDICTOR_OUT), seed=2)
    g_path, f_path = g.save(out / "inverse.npz"), f.save(out / "predictor.npz")
    lib, header = build_library(g_path, f_path, out)
    return {"lib": lib, "header": header, "g": g, "f": f, "g_path": g_path, "f_path": f_path}


def test_the_generated_header_declares_both_networks(built) -> None:
    text = built["header"].read_text()
    for macro in ("ACTION_INV_IN_DIM", "ACTION_INV_OUT_DIM", "ACTION_PRED_IN_DIM",
                  "ACTION_PRED_OUT_DIM", "ACTION_MAX_WIDTH"):
        assert macro in text
    assert "DO NOT EDIT BY HAND" in text


def test_parity_with_numpy_on_a_thousand_inputs(built) -> None:
    from brick_builder.export.action_to_c import check_parity

    rng = np.random.default_rng(0)
    g, f = built["g"], built["f"]
    x_inv = rng.normal(g.in_mean, g.in_std, size=(1000, INVERSE_IN)).astype(np.float32)
    x_pred = rng.normal(f.in_mean, f.in_std, size=(1000, PREDICTOR_IN)).astype(np.float32)
    report = check_parity(built["lib"], built["g_path"], built["f_path"], x_inv, x_pred)
    assert report.n == 1000
    assert report.inverse_max_abs_error <= PARITY_BOUND, report
    assert report.predictor_max_abs_error <= PARITY_BOUND, report


def test_parity_holds_where_the_normalizer_saturates(built) -> None:
    """A far target is what the clipping is for, so the C has to clip the same way."""
    from brick_builder.export.c_action import CActionNet

    g = built["g"]
    net = CActionNet(built["lib"], INVERSE_IN, INVERSE_OUT, PREDICTOR_IN, PREDICTOR_OUT)
    rng = np.random.default_rng(3)
    x = (g.in_mean + 50.0 * g.in_std * rng.normal(size=(200, INVERSE_IN))).astype(np.float32)
    assert np.max(np.abs(net.inverse_batch(x) - g.forward(x))) <= PARITY_BOUND


@pytest.mark.slow
def test_one_closed_loop_episode_with_the_c_inverse_in_the_loop(tmp_path) -> None:
    from brick_builder.action.models.inverse import WEIGHTS

    if shutil.which("gcc") is None or not (WEIGHTS / "inverse.npz").exists():
        pytest.skip("needs gcc and a trained inverse")
    from brick_builder.action.learn.evaluate import run_episode
    from brick_builder.export.action_to_c import build_library
    from brick_builder.export.c_action import CInverse

    lib, _ = build_library(WEIGHTS / "inverse.npz", WEIGHTS / "predictor.npz", tmp_path)
    c_inverse = CInverse.load(lib)
    from brick_builder.action.place import PlaceConfig, TrackingEpisode, record_ideal
    from brick_builder.action.worlds import build_world
    from brick_builder.control.pick_place import verify_placement

    cfg = PlaceConfig(inverse="learned")
    ideal, nominal = record_ideal(cfg, seed=1)
    nominal.close()
    scene, _, _ = build_world(1)
    episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=1, inverse=c_inverse)
    try:
        for _ in episode.run():
            pass
        check = verify_placement(scene, scene.held_name, (4.5, 3.5, 0))
    finally:
        episode.close()
    assert c_inverse.calls > 0
    assert check.lateral_mm == check.lateral_mm
    del run_episode
