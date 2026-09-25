"""Scene-state codec: matching, distance properties, opaque schema, calibration files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA as S
from brick_builder.rl.scene_target.codec import (
    MISSING_MM, MM_PER_WINDOW_UNIT, Calibration, SceneDistance, SlotFrame, calibration_path,
    fit_feature_std, fit_mm_scale, reference_mask, valid_mask,
)

K, W = 8, S.W


def row(mass, cx, cy, colour, *, moments=(0.004, 0.0, 0.002), border=0.0, spread=0.01,
        age=0.5, orient4=(0.1, 0.05)):
    r = np.zeros(W, dtype=np.float32)
    r[S.fields["mass"]] = mass
    r[S.fields["centre"]] = (cx, cy)
    r[S.fields["moments"]] = moments
    r[S.fields["colour"]] = colour
    r[S.fields["colour_spread"]] = spread
    r[S.fields["border"]] = border
    r[S.fields["age"]] = age
    r[S.fields["orient4"]] = orient4
    return r


def scene(shift=(0.0, 0.0), *, held_shift=(0.0, 0.0), drop=(), order=None):
    """Background (slot 0, id -1), table-and-arm blob, plate, yellow 2x4, red 2x2, held blue."""
    objs = [
        (-1, row(0.02, 0.0, 0.0, (-0.05, -0.08, -0.2), moments=(0.3, 0.0, 0.3))),
        (11, row(0.80, 0.01, 0.0, (0.0, -0.05, -0.1), moments=(0.25, 0.01, 0.3), border=0.05)),
        (12, row(0.07, 0.06, -0.08, (-0.93, -0.08, -0.52), moments=(0.02, 0.001, 0.02))),
        (13, row(0.019, -0.15, -0.17, (0.15, -0.06, -0.83), moments=(0.004, 0.001, 0.001))),
        (14, row(0.007, 0.25, 0.20, (0.10, -0.84, -0.84), moments=(0.001, 0.0, 0.001))),
        (15, row(0.05, 0.0, 0.45, (-0.7, -0.6, 0.6), moments=(0.006, 0.0, 0.005))),
    ]
    rows = np.zeros((K, W), dtype=np.float32)
    depth = np.full(K, 0.5, dtype=np.float32)
    ids = np.full(K, -1, dtype=np.int64)
    c = S.fields["centre"]
    for k, (sid, r) in enumerate(objs):
        if sid in drop:
            continue
        r = r.copy()
        if sid >= 0:
            r[c] += np.asarray(shift, dtype=np.float32)
        if sid == 15:
            r[c] += np.asarray(held_shift, dtype=np.float32)
        rows[k] = r
        depth[k] = 0.1 * k
        ids[k] = sid
    f = SlotFrame.from_arrays(rows, depth, None, ids)
    return f if order is None else f.permuted(order)


D = SceneDistance()
HELD = 5  # the held brick's slot in `scene()`'s order


def test_valid_mask_follows_the_empty_slot_convention():
    rows = np.zeros((3, W), dtype=np.float32)
    rows[1, 0] = 0.1
    depth = np.array([0.5, 0.5, 0.2], dtype=np.float32)
    assert valid_mask(rows, depth).tolist() == [False, True, True]


def test_zero_on_identical_states():
    a = scene()
    res = D(a, a)
    assert res.d_mm == 0.0
    assert res.n_missing == 0
    assert all(v == 0.0 for v in res.parts.values())
    assert D(a, a, held=HELD).d_mm == 0.0


def test_background_slot_is_never_matched():
    a = scene()
    perm = D.match(a, a)
    assert perm[0] == -1
    assert list(perm[1:6]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("seed", range(5))
def test_invariant_to_slot_order_on_either_side(seed):
    rng = np.random.default_rng(seed)
    target = scene()
    obs = scene(shift=(0.02, -0.01), held_shift=(0.01, 0.0))
    ref = D(obs, target, held=HELD).d_mm
    o1 = rng.permutation(K)
    assert D(obs.permuted(o1), target, held=HELD).d_mm == pytest.approx(ref, rel=1e-9, abs=1e-12)
    o2 = rng.permutation(K)
    held2 = int(np.nonzero(o2 == HELD)[0][0])
    assert D(obs, target.permuted(o2), held=held2).d_mm == pytest.approx(ref, rel=1e-9, abs=1e-12)
    assert D(obs.permuted(o1), target.permuted(o2), held=held2).d_mm == pytest.approx(
        ref, rel=1e-9, abs=1e-12)


def test_monotone_in_a_common_centre_shift():
    target = scene()
    ds = [D(scene(shift=(s, 0.5 * s)), target).d_mm for s in np.linspace(0.0, 0.08, 9)]
    assert ds[0] == 0.0
    assert all(b > a for a, b in zip(ds, ds[1:]))


def test_a_common_centre_shift_reads_in_mm_eq():
    """Every fully visible slot moved by 0.01 window units reads about 0.87 mm-eq."""
    target = scene(drop=(11,))          # leave out the one slot with a border fraction
    d = D(scene(shift=(0.01, 0.0), drop=(11,)), target)
    assert d.parts["abs"] == pytest.approx(MM_PER_WINDOW_UNIT * 0.01, rel=1e-4)


def test_a_calibrated_scale_replaces_87_4_on_the_named_schema():
    target = scene(drop=(11,))
    obs = scene(shift=(0.01, 0.0), drop=(11,))
    cal = Calibration("ideal", "", W, family="viewing_pose", mm_scale=210.0)
    d = SceneDistance(S, cal)(obs, target)
    assert d.parts["abs"] == pytest.approx(210.0 * 0.01, rel=1e-4)
    assert d.parts["missing"] == 0.0


def test_shifting_only_the_held_brick_moves_the_relative_term():
    target = scene()
    held_only = D(scene(held_shift=(0.02, 0.0)), target, held=HELD)
    everything = D(scene(shift=(0.02, 0.0)), target, held=HELD)
    assert held_only.parts["rel"] == pytest.approx(MM_PER_WINDOW_UNIT * 0.02, rel=1e-4)
    assert everything.parts["rel"] == pytest.approx(0.0, abs=1e-5)   # float32 rows
    assert everything.parts["abs"] > 0.0
    no_held = D(scene(held_shift=(0.02, 0.0)), target)
    assert no_held.parts["rel"] == 0.0


def test_missing_target_slot_costs_two_mm_eq_each():
    target = scene()
    one = D(scene(drop=(14,)), target)
    two = D(scene(drop=(13, 14)), target)
    assert one.n_missing == 1 and two.n_missing == 2
    assert one.parts["missing"] == pytest.approx(MISSING_MM)
    assert two.parts["missing"] == pytest.approx(2 * MISSING_MM)
    assert one.parts["abs"] == pytest.approx(0.0, abs=1e-9)
    # an extra observed object that the target does not have costs nothing
    assert D(target, scene(drop=(14,))).d_mm == pytest.approx(0.0, abs=1e-9)


def test_objects_of_different_colour_are_not_matched():
    target = scene(drop=(12, 13, 15, 11))           # the red brick only
    obs = scene(drop=(12, 14, 15, 11))              # the yellow brick only
    res = D(obs, target)
    assert res.n_missing == 1 and not res.matched.any()


def test_align_puts_observed_rows_in_target_order():
    target = scene()
    obs = scene(shift=(0.01, 0.0)).permuted([3, 0, 5, 1, 7, 2, 6, 4])
    rows, depth, ok = D.align(obs, target)
    assert ok.tolist() == [False, True, True, True, True, True, False, False]
    c = S.fields["centre"]
    np.testing.assert_allclose(rows[ok][:, c], target.rows[ok][:, c] + [0.01, 0.0], atol=1e-6)
    assert np.all(rows[~ok] == 0) and np.all(depth[~ok] == 0.5)


def test_border_weight_discounts_cut_off_slots():
    target = scene(drop=(11,))
    moved = scene(drop=(11,)).rows.copy()
    moved[2, S.fields["centre"]] += (0.05, 0.0)                       # the plate's slot moves
    obs_full = SlotFrame.from_arrays(moved, target.depth, target.valid, target.slot_ids)
    moved_b = moved.copy()
    moved_b[2, S.fields["border"]] = 0.9
    obs_cut = SlotFrame.from_arrays(moved_b, target.depth, target.valid, target.slot_ids)
    assert D(obs_cut, target).d_mm < D(obs_full, target).d_mm


def test_reference_mask_excludes_background_cut_off_and_oversized_slots():
    f = scene()
    ref = reference_mask(f)
    assert ref.tolist() == [False, False, True, True, True, True, False, False]


def test_from_ideal_scene_model_states():
    """Distance on real producer output: a rendered-looking frame and a shifted copy."""
    from brick_builder.perception.ideal_scene_model import IdealSceneModel

    res = 256
    inst = np.zeros((res, res), dtype=np.int32)
    inst[40:80, 50:130] = 3
    inst[150:190, 160:200] = 4
    inst[100:140, 20:60] = 5
    depth = np.full((res, res), 0.3, dtype=np.float32)
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    rgb[inst == 3] = (230, 200, 30)
    rgb[inst == 4] = (200, 30, 30)
    rgb[inst == 5] = (30, 60, 200)
    src = IdealSceneModel(build_decode=False)
    a = src.observe(rgb, window=None, instance=inst, depth=depth)
    shifts = []
    for px in (0, 2, 4, 8):
        src.reset()
        b = src.observe(np.roll(rgb, px, axis=1), window=None,
                        instance=np.roll(inst, px, axis=1), depth=depth)
        shifts.append(D(b, a).d_mm)
    assert shifts[0] == 0.0
    assert all(y > x for x, y in zip(shifts, shifts[1:]))
    # 2 px of 256 in window units is 2 * 2/255; in mm-eq about 87.4 * 0.0157
    assert shifts[1] == pytest.approx(MM_PER_WINDOW_UNIT * 2 * 2 / 255, rel=0.1)


# ------------------------------------------------------------------------------ opaque ---


def opaque_frames(seed=0, W32=32):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(K, W32)).astype(np.float32)
    rows[6:] = 0.0
    depth = np.linspace(0.0, 1.0, K).astype(np.float32)
    depth[6:] = 0.5
    ids = np.array([-1, 1, 2, 3, 4, 5, -1, -1])
    return SlotFrame.from_arrays(rows, depth, None, ids)


def test_opaque_schema_zero_invariance_and_monotone():
    d = SceneDistance.opaque(32)
    a = opaque_frames()
    assert d(a, a).d_mm == 0.0
    rng = np.random.default_rng(3)
    b_rows = a.rows.copy()
    b_rows[1:6] += 0.01 * rng.normal(size=(5, 32)).astype(np.float32)
    b = SlotFrame.from_arrays(b_rows, a.depth, a.valid, a.slot_ids)
    ref = d(b, a).d_mm
    assert ref > 0.0
    for s in range(3):
        o = np.random.default_rng(s).permutation(K)
        assert d(b.permuted(o), a).d_mm == pytest.approx(ref, rel=1e-9)
    grow = []
    for eps in (0.0, 0.05, 0.1, 0.2):
        r = a.rows.copy()
        r[1:6] += eps
        grow.append(d(SlotFrame.from_arrays(r, a.depth, a.valid, a.slot_ids), a).parts["abs"])
    assert all(y > x for x, y in zip(grow, grow[1:]))
    # a common offset moves every slot the same way: the relative term stays zero
    r = a.rows.copy()
    r[1:6] += 0.1
    assert d(SlotFrame.from_arrays(r, a.depth, a.valid, a.slot_ids), a).parts["rel"] == pytest.approx(
        0.0, abs=1e-6)


def test_opaque_schema_uses_calibrated_scales():
    a = opaque_frames()
    r = a.rows.copy()
    r[1:6, 0] += 0.01
    b = SlotFrame.from_arrays(r, a.depth, a.valid, a.slot_ids)
    std = np.ones(32)
    loose = SceneDistance.opaque(32, Calibration("x", "", 32, feature_std=list(std)))(b, a)
    std[0] = 0.01
    tight = SceneDistance.opaque(32, Calibration("x", "", 32, feature_std=list(std)))(b, a)
    assert loose.n_missing == 0 and tight.n_missing == 0
    assert tight.parts["abs"] == pytest.approx(100.0 * loose.parts["abs"], rel=1e-4)


def test_calibration_json_round_trip_and_refusal(tmp_path):
    frames = []
    base = opaque_frames()
    rng = np.random.default_rng(1)
    for t in range(5):
        r = base.rows.copy()
        r[1:6] += 0.02 * t + 0.001 * rng.normal(size=(5, 32)).astype(np.float32)
        frames.append(SlotFrame.from_arrays(r, base.depth, base.valid, base.slot_ids))
    std, dstd = fit_feature_std(frames)
    assert std.shape == (32,) and np.all(std > 0)
    cal = Calibration(source="opaque_test", weights_sha256="ab" * 32, W=32,
                      feature_std=std.tolist(), depth_std=dstd,
                      mm_scale=fit_mm_scale([1.0, 2.0], [3.0, 6.0]), sigma_obs_mm=0.4, n_ticks=5)
    path = calibration_path(tmp_path, cal.source, cal.weights_sha256)
    assert path.name == "opaque_test_abababab.json"
    cal.save(path)
    back = Calibration.load(path, source="opaque_test", weights_sha256="ab" * 32, W=32)
    assert back == cal
    assert json.loads(path.read_text())["mm_scale"] == pytest.approx(3.0)
    assert back.tol_mm == pytest.approx(0.8)
    assert Calibration("x", "", 14).tol_mm == 0.5
    with pytest.raises(ValueError):
        Calibration.load(path, weights_sha256="cd" * 32)
    with pytest.raises(ValueError):
        Calibration.load(path, W=14)
    fam = Calibration(source="learned", weights_sha256="ef" * 32, W=14, family="viewing_pose",
                      mm_scale=210.0, sigma_obs_mm=1.2, stats={"scale_fit": {"n": 3}})
    fpath = calibration_path(tmp_path, fam.source, fam.weights_sha256, fam.family)
    assert fpath.name == "learned_efefefef_viewing_pose.json"
    fam.save(fpath)
    assert Calibration.load(fpath, family="viewing_pose") == fam
    assert Calibration.load(fpath).tol_mm == pytest.approx(2.4)
    with pytest.raises(ValueError):
        Calibration.load(fpath, family="placement")
