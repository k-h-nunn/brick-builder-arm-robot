"""The differentiable slot head against the hard post-processing it mirrors.

Two kinds of check. The cheap ones feed a 0/1 mask through the soft statistics and ask for the
ideal source's own `_blob_row` back, field for field -- with a hard mask the two are the same
arithmetic and the agreement is exact to float32. The expensive one runs the trained network over
cached frames and compares the whole head against `CnnSceneModel.observe`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brick_builder.perception.cnn.slot_head import (
    SlotHead, SlotHeadConfig, _soft_rank, _weighted_median, parity, soft_blob_rows,
)
from brick_builder.perception.ideal_scene_model import _blob_row
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, rank_scale

from .conftest import CACHE, WEIGHTS, cached_samples

RES = 64


def _frame(rng: np.random.Generator) -> np.ndarray:
    return rng.random((RES, RES, 3)).astype(np.float32)


def _mask(rng: np.random.Generator) -> np.ndarray:
    m = np.zeros((RES, RES), dtype=bool)
    r0, c0 = rng.integers(2, 30, size=2)
    m[r0:r0 + rng.integers(8, 25), c0:c0 + rng.integers(8, 25)] = True
    return m


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_soft_statistics_match_the_ideal_blob_row_on_a_hard_mask(seed):
    rng = np.random.default_rng(seed)
    frame, mask = _frame(rng), _mask(rng)
    want = _blob_row(mask, frame.astype(np.float64), RES, amodal=False, age=0.0)

    soft = soft_blob_rows(torch.from_numpy(mask.astype(np.float32))[None],
                          torch.from_numpy(frame.transpose(2, 0, 1)))[0].numpy()

    for name in ("mass", "centre", "moments", "colour", "colour_spread", "border", "orient4"):
        sl = PLACEHOLDER_SLOT_SCHEMA.fields[name]
        assert soft[sl] == pytest.approx(want[sl], abs=2e-5, rel=2e-4), name
    assert soft[PLACEHOLDER_SLOT_SCHEMA.fields["age"]] == pytest.approx(0.0)


def test_a_soft_mask_moves_the_centre_towards_the_pixels_it_weights():
    """The whole point of the soft form: a pixel's weight is what its contribution is."""
    mask = np.zeros((RES, RES), dtype=np.float32)
    mask[10:20, 10:20] = 1.0
    heavier = mask.copy()
    heavier[10:20, 20:30] = 0.5
    frame = torch.zeros(3, RES, RES)
    a = soft_blob_rows(torch.from_numpy(mask)[None], frame)[0]
    b = soft_blob_rows(torch.from_numpy(heavier)[None], frame)[0]
    cx = PLACEHOLDER_SLOT_SCHEMA.fields["centre"].start
    assert float(b[cx]) > float(a[cx])


def test_the_statistics_carry_a_gradient_to_the_mask():
    mask = torch.full((1, RES, RES), 0.0, requires_grad=True)
    with torch.no_grad():
        mask[0, 8:24, 8:24] = 1.0
    rows = soft_blob_rows(mask, torch.zeros(3, RES, RES))
    rows[0, PLACEHOLDER_SLOT_SCHEMA.fields["centre"].start].backward()
    assert mask.grad is not None and float(mask.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_median_matches_the_plain_median_on_a_hard_mask(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=400)
    weights = (rng.random(400) > 0.4).astype(np.float64)
    want = float(np.median(values[weights > 0]))
    got = float(_weighted_median(torch.as_tensor(values), torch.as_tensor(weights)))
    # A median of an even count interpolates in numpy and picks a side here, so the answer is one
    # of the two middle samples rather than their mean.
    kept = np.sort(values[weights > 0])
    assert got in kept
    assert abs(got - want) <= abs(kept[len(kept) // 2] - kept[len(kept) // 2 - 1]) + 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_soft_rank_reproduces_rank_scale_at_a_small_temperature(seed):
    rng = np.random.default_rng(seed)
    depths = rng.normal(0.0, 1.0, size=8)
    present = rng.random(8) > 0.4
    want = rank_scale(depths, present)
    got = _soft_rank(torch.as_tensor(depths, dtype=torch.float32),
                     torch.as_tensor(present), tau=1e-4).numpy()
    assert got == pytest.approx(want, abs=1e-5)


def test_absent_slots_keep_the_contract_value():
    got = _soft_rank(torch.zeros(4), torch.zeros(4, dtype=torch.bool), tau=0.01).numpy()
    assert got == pytest.approx(np.full(4, 0.5))


# ------------------------------------------------------------------- against the shipped model ---


@pytest.mark.slow
def test_parity_with_the_hard_post_processing_on_held_out_frames():
    """The whole head, on cached frames, against `CnnSceneModel.observe` slot for slot."""
    if not WEIGHTS.exists():
        pytest.skip(f"no trained checkpoint at {WEIGHTS}")
    samples = cached_samples("val", 8)
    if samples is None:
        pytest.skip(f"no rendered dataset under {CACHE}")
    torch.set_num_threads(2)

    report = parity(samples, checkpoint=WEIGHTS)
    per_object = report["per_object"]
    assert per_object["scored"] > 0
    assert per_object["frames_with_a_different_candidate_count"] == 0
    # With the straight-through class gate the only softness left is the grouping between two
    # bricks, which differs from the argmin in the thin band where their votes are equidistant.
    assert per_object["mean_abs"]["centre"] < 1e-4
    assert per_object["mean_abs"]["mass"] < 1e-4
    assert per_object["max_abs"]["centre"] < 2e-3
    assert report["per_slot"]["one_sided"] == 0
    assert report["per_slot"]["depth_mean_abs"] < 0.05


@pytest.mark.slow
def test_the_plain_soft_gate_is_the_worse_one_and_is_not_the_default():
    """A per-pixel class probability over sixty thousand pixels is not a small number."""
    if not WEIGHTS.exists():
        pytest.skip(f"no trained checkpoint at {WEIGHTS}")
    samples = cached_samples("val", 6)
    if samples is None:
        pytest.skip(f"no rendered dataset under {CACHE}")
    torch.set_num_threads(2)

    hard_gate = parity(samples, checkpoint=WEIGHTS)
    soft_gate = parity(samples, checkpoint=WEIGHTS, cfg=SlotHeadConfig(class_gate="soft"))
    assert (soft_gate["per_object"]["mean_abs"]["centre"]
            > 50.0 * hard_gate["per_object"]["mean_abs"]["centre"])
    assert SlotHeadConfig().class_gate == "straight_through"


@pytest.mark.slow
def test_the_head_publishes_the_scene_state_contract():
    if not WEIGHTS.exists():
        pytest.skip(f"no trained checkpoint at {WEIGHTS}")
    samples = cached_samples("val", 2)
    if samples is None:
        pytest.skip(f"no rendered dataset under {CACHE}")
    from brick_builder.perception.cnn.model import SceneModelNet, unpack_weights

    from .conftest import assert_scene_state_contract

    torch.set_num_threads(2)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    net = SceneModelNet(width=float(state.get("width", 1.0)))
    net.load_state_dict(unpack_weights(state["model"]))
    net.eval()
    head = SlotHead(cfg=SlotHeadConfig())
    for sample in samples:
        frame = np.asarray(sample.rgb, dtype=np.float32)
        if frame.max(initial=0.0) > 1.0 + 1e-6:
            frame = frame / 255.0
        x = torch.from_numpy(np.ascontiguousarray(frame[..., :3].transpose(2, 0, 1)))[None]
        with torch.no_grad():
            out = net(x)
        soft = head(out, x[0])
        assert_scene_state_contract(soft.state(window=getattr(sample, "window", None)))
