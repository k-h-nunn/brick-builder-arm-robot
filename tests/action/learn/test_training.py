"""Training on the toy linear world: the pairs are built correctly, the predictor learns the
world's own matrix, and the distal-teacher loss comes down."""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.action.learn import dataset as ds
from brick_builder.action.learn.train import TrainConfig, train_inverse, train_predictor
from brick_builder.action.models.features import INVERSE_IN, INVERSE_OUT, PREDICTOR_IN, PREDICTOR_OUT


def test_pairs_stay_inside_one_episode_and_one_step(toy) -> None:
    pairs = ds.predictor_pairs(toy)
    assert pairs["x"].shape[1] == PREDICTOR_IN
    assert pairs["y"].shape[1] == PREDICTOR_OUT
    assert np.all(toy.episode[pairs["i"]] == toy.episode[pairs["j"]])

    hind = ds.hindsight_pairs(toy, np.random.default_rng(0), repeats=1)
    assert hind["x"].shape[1] == INVERSE_IN
    assert hind["y"].shape[1] == INVERSE_OUT
    assert np.all(toy.episode[hind["i"]] == toy.episode[hind["j"]])
    assert np.all(hind["j"] > hind["i"]) and np.all(hind["j"] - hind["i"] <= 8)


def test_the_correction_read_back_from_the_log_is_the_one_that_was_commanded(toy) -> None:
    """`corr = cmd - ff` by construction, so its increment is the logged delta."""
    corr = toy.corrections()
    inc = corr[1:] - corr[:-1]
    same = toy.episode[1:] == toy.episode[:-1]
    assert np.allclose(inc[same], toy["delta"][:-1][same], atol=1e-6)


def test_the_predictor_beats_predicting_no_change(toy, tmp_path) -> None:
    cfg = TrainConfig(hidden=(24, 24), epochs=12, batch=128, minutes=2.0)
    model = train_predictor(toy, cfg, tmp_path / "p.npz", log=lambda *a: None)
    pairs = ds.predictor_pairs(toy)
    err = ((model.forward(pairs["x"]) - pairs["y"]) ** 2 * pairs["w"]).sum()
    naive = ((0.0 - pairs["y"]) ** 2 * pairs["w"]).sum()
    assert err < 0.5 * naive, f"model {err:.3e} vs no change {naive:.3e}"


def test_the_distal_teacher_loss_decreases(toy, tmp_path) -> None:
    """The inverse is trained through the frozen predictor; the slot distance it is asked to
    close has to come down over the run."""
    cfg = TrainConfig(hidden=(24, 24), epochs=10, batch=128, minutes=2.0, distal=1.0,
                      action_penalty=0.0)
    predictor = train_predictor(toy, cfg, tmp_path / "p.npz", log=lambda *a: None)

    lines: list[str] = []
    train_inverse(toy, predictor, cfg, tmp_path / "g.npz", log=lines.append)
    distal = [float(line.split("distal")[1].split()[0]) for line in lines if "distal" in line]
    assert len(distal) >= 4
    assert np.mean(distal[-2:]) < np.mean(distal[:2]), distal


def test_a_trained_inverse_has_the_deployed_shape(toy, tmp_path) -> None:
    cfg = TrainConfig(hidden=(16, 16), epochs=3, batch=128, minutes=1.0)
    predictor = train_predictor(toy, cfg, tmp_path / "p.npz", log=lambda *a: None)
    inverse = train_inverse(toy, predictor, cfg, tmp_path / "g.npz", log=lambda *a: None)
    assert inverse.sizes[0] == INVERSE_IN and inverse.sizes[-1] == INVERSE_OUT
    assert inverse.meta["kind"] == "inverse"


def test_training_is_reproducible_from_a_seed(toy, tmp_path) -> None:
    cfg = TrainConfig(hidden=(16, 16), epochs=3, batch=128, minutes=1.0, seed=5)
    a = train_predictor(toy, cfg, tmp_path / "a.npz", log=lambda *m: None)
    b = train_predictor(toy, cfg, tmp_path / "b.npz", log=lambda *m: None)
    x = np.random.default_rng(0).normal(size=(4, PREDICTOR_IN))
    assert np.array_equal(a.forward(x), b.forward(x))


def test_resuming_continues_rather_than_restarting(toy, tmp_path) -> None:
    state = tmp_path / "state.pt"
    cfg = TrainConfig(hidden=(16, 16), epochs=2, batch=128, minutes=1.0)
    train_predictor(toy, cfg, tmp_path / "p.npz", resume=state, log=lambda *m: None)
    assert state.exists()
    lines: list[str] = []
    train_predictor(toy, TrainConfig(hidden=(16, 16), epochs=4, batch=128, minutes=1.0),
                    tmp_path / "p.npz", resume=state, log=lines.append)
    assert any("resumed at epoch 2" in line for line in lines), lines
