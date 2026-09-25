"""Predictive scene model: predict from the motion, update with the observation, by slot id."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from brick_builder.perception import sources  # noqa: E402
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA as S  # noqa: E402
from brick_builder.perception.scene_model import SceneModelSource, SceneState  # noqa: E402
from brick_builder.perception.views import ViewingPose  # noqa: E402
from brick_builder.rl.scene_target.codec import MM_PER_WINDOW_UNIT, SlotFrame  # noqa: E402
from brick_builder.rl.scene_target.model import (  # noqa: E402
    ScenePredictor, SlotLayout, context_vector,
)
from brick_builder.rl.scene_target.predictive import (  # noqa: E402
    SOURCE_NAME, PredictiveSceneModel, ScoringPredictor, gains, sigma_f_from_validation,
    sigma_obs_static,
)

K, W = 8, S.W
L = SlotLayout.for_schema(S)
CTX = context_vector(np.zeros(18), np.r_[0.2, 0.0, 0.07, 0.0, 0.0, np.zeros(8)])
A = np.array([1e-3, 0.0, 0.0, 0.0])
C = S.fields["centre"]


def frame(objs, background_slot=None) -> SlotFrame:
    """`objs`: {slot: (id, cx, cy, colour_r)}."""
    rows = np.zeros((K, W), dtype=np.float32)
    depth = np.full(K, 0.5, dtype=np.float32)
    ids = np.full(K, -1)
    valid = np.zeros(K, dtype=bool)
    for slot, (sid, cx, cy, r) in objs.items():
        rows[slot, S.fields["mass"]] = 0.01
        rows[slot, C] = (cx, cy)
        rows[slot, S.fields["colour"]] = (r, 0.0, 0.0)
        depth[slot], ids[slot], valid[slot] = 0.3, sid, True
    if background_slot is not None:
        rows[background_slot, S.fields["mass"]] = 0.5
        valid[background_slot] = True
    return SlotFrame(rows, depth, valid, ids)


class ShiftF:
    """Stub `F`: every object's centre moves +0.01 in x."""

    layout = L

    def __init__(self):
        self.calls = []

    def predict(self, fr, ctx, a):
        self.calls.append((fr, np.asarray(ctx), np.asarray(a)))
        rows = fr.rows.copy()
        rows[fr.objects, 1] += 0.01
        return SlotFrame(rows, fr.depth.copy(), fr.valid.copy(), fr.slot_ids.copy())


class ListSource:
    """Stub source: publishes the given frames in order."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.i = 0

    def observe(self, rgb, *, window, instance=None, depth=None, present_mask=None, t=None):
        fr = self.frames[self.i]
        self.i += 1
        return SceneState(slots=fr.rows, depth=fr.depth, schema=S, t=float(self.i),
                          source="stub", window=window, slot_ids=fr.slot_ids, frame_id=self.i,
                          _decode_fn=lambda px: np.zeros((len(px), K), np.float32))


# ---------------------------------------------------------------------------------- gains ---


def test_gain_formula():
    k = gains([1.0, 0.0, 2.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert k == pytest.approx([0.5, 0.0, 1.0, 1.0])
    assert gains(3.0, 4.0) == pytest.approx(9.0 / 25.0)
    assert sigma_f_from_validation([0.5, 0.2], [0.3, 0.3]) == pytest.approx([0.4, 0.0])


def test_static_observation_noise_per_component():
    rng = np.random.default_rng(0)
    base = frame({0: (7, 0.1, 0.2, 0.5), 2: (9, -0.3, 0.1, -0.5)}, background_slot=5)
    frames = []
    for _ in range(600):
        rows = base.rows.copy()
        rows[base.objects, 1] += rng.normal(0.0, 0.002, 2).astype(np.float32)
        frames.append(SlotFrame(rows, base.depth, base.valid, base.slot_ids))
    s = sigma_obs_static(frames, L)
    assert s.shape == (L.D,)
    assert s[0] == pytest.approx(0.002, rel=0.1)
    assert np.all(s[1:] < 1e-6)


# ------------------------------------------------------------------------------ the filter ---


def test_without_motion_the_observation_passes_through():
    m = PredictiveSceneModel(None, ShiftF(), gain=0.0)
    a = frame({0: (7, 0.1, 0.2, 0.5)})
    b = frame({0: (7, 0.15, 0.2, 0.5)})
    assert m.update(a) is a
    assert m.update(b) is b  # no set_motion before this tick
    assert m.last_prediction is None and np.all(np.isnan(m.innovation))
    assert math.isnan(m.innovation_mm) and not m.F.calls


@pytest.mark.parametrize("kappa", [1.0, 0.0, 0.5])
def test_fixed_gain_update(kappa):
    F = ShiftF()
    m = PredictiveSceneModel(None, F, gain=kappa)
    a = frame({0: (7, 0.10, 0.20, 0.5), 1: (9, -0.2, 0.0, -0.5)}, background_slot=4)
    m.update(a)
    b = frame({0: (7, 0.14, 0.20, 0.6), 1: (9, -0.2, 0.0, -0.4)}, background_slot=4)
    m.set_motion(CTX, A)
    out = m.update(b)
    assert F.calls[-1][0] is a and np.array_equal(F.calls[-1][2], A)
    pred_x = {7: 0.11, 9: -0.19}
    for slot, sid in ((0, 7), (1, 9)):
        want = pred_x[sid] + kappa * (b.rows[slot, 1] - pred_x[sid])
        assert out.rows[slot, 1] == pytest.approx(want, abs=1e-6)
        assert out.rows[slot, 2] == pytest.approx(b.rows[slot, 2])
        # appearance and flags come from the observation
        assert np.array_equal(out.rows[slot, S.fields["colour"]], b.rows[slot, S.fields["colour"]])
    assert np.array_equal(out.rows[4], b.rows[4])  # background passes through
    assert np.array_equal(out.slot_ids, b.slot_ids) and np.array_equal(out.valid, b.valid)
    assert m.innovation[0, 0] == pytest.approx(0.14 - 0.11, abs=1e-6)
    assert m.innovation[1, 0] == pytest.approx(-0.2 + 0.19, abs=1e-6)
    assert np.all(np.isnan(m.innovation[4]))
    want_mm = MM_PER_WINDOW_UNIT * math.sqrt((0.03 ** 2 + 0.01 ** 2) / 2.0)
    assert m.innovation_mm == pytest.approx(want_mm, rel=1e-4)


def test_slots_pair_by_id_across_a_new_slot_order():
    F = ShiftF()
    m = PredictiveSceneModel(None, F, gain=0.0)
    m.update(frame({0: (7, 0.1, 0.2, 0.5), 1: (9, -0.2, 0.0, -0.5)}))
    m.set_motion(CTX, A)
    b = frame({3: (9, -0.25, 0.0, -0.5), 5: (7, 0.3, 0.2, 0.5), 6: (11, 0.5, 0.5, 0.9)})
    out = m.update(b)
    assert out.rows[5, 1] == pytest.approx(0.11) and out.rows[3, 1] == pytest.approx(-0.19)
    assert out.rows[6, 1] == pytest.approx(0.5)  # new slot: no prediction, observation kept
    assert np.all(np.isnan(m.innovation[6])) and np.isfinite(m.innovation[5]).all()
    # the next prediction starts from the filtered state in its new slot order
    m.set_motion(CTX, A)
    m.update(frame({5: (7, 0.2, 0.2, 0.5)}))
    assert F.calls[-1][0] is out
    # an id the source no longer reports is dropped
    assert list(m.last_filtered.slot_ids[m.last_filtered.objects]) == [7]


def test_reset_clears_the_filter_and_the_source_identity():
    src = ListSource([frame({0: (7, 0.1, 0.2, 0.5)})] * 3)
    m = PredictiveSceneModel(src, ShiftF(), gain=0.0)
    m.update(frame({0: (7, 0.1, 0.2, 0.5)}))
    m.set_motion(CTX, A)
    m.reset()
    assert src.resets == 1 and m.ticks == 0 and m.last_filtered is None
    out = m.update(frame({0: (7, 0.3, 0.2, 0.5)}))
    assert out.rows[0, 1] == pytest.approx(0.3)  # motion set before the reset is dropped


def test_motion_shapes_are_checked():
    m = PredictiveSceneModel(None, ShiftF())
    with pytest.raises(ValueError):
        m.set_motion(np.zeros(42), A)
    with pytest.raises(ValueError):
        m.set_motion(CTX, np.zeros(3))
    with pytest.raises(ValueError):
        m.update(SlotFrame.from_arrays(np.zeros((K, 32)), np.full(K, 0.5)))


def test_observe_implements_the_scene_model_source():
    frames = [frame({0: (7, 0.1, 0.2, 0.5)}), frame({2: (7, 0.13, 0.2, 0.5)})]
    m = PredictiveSceneModel(ListSource(frames), ShiftF(), gain=0.5)
    assert isinstance(m, SceneModelSource)
    rgb = np.zeros((8, 8, 3), np.uint8)
    s0 = m.observe(rgb, window=None)
    m.set_motion(CTX, A)
    s1 = m.observe(rgb, window=None)
    assert isinstance(s1, SceneState) and s1.source == f"{SOURCE_NAME}/stub"
    assert s1.slots[2, 1] == pytest.approx(0.5 * (0.11 + 0.13))
    assert np.array_equal(s1.slot_ids, frames[1].slot_ids) and s1.frame_id == 2
    assert s1.has_decode and s0.t == 1.0
    assert m.last_observation.rows[2, 1] == pytest.approx(0.13)


def test_zero_gain_follows_the_learned_predictor():
    F = ScenePredictor.create(L, seed=3)
    m = PredictiveSceneModel(None, F, gain=0.0)
    a = frame({0: (7, 0.1, 0.2, 0.5), 1: (9, -0.2, 0.0, -0.5)}, background_slot=2)
    m.update(a)
    m.set_motion(CTX, A)
    b = frame({1: (7, 0.1, 0.2, 0.5), 0: (9, -0.2, 0.0, -0.5)}, background_slot=2)
    out = m.update(b)
    pred = F.predict(a, CTX, A)
    dp = L.dyn(pred.rows, pred.depth)
    do = L.dyn(out.rows, out.depth)
    assert np.allclose(do[1], dp[0], atol=1e-6) and np.allclose(do[0], dp[1], atol=1e-6)


# --------------------------------------------------------------------------- registration ---


def _ideal_inputs(shift=0):
    res = 64
    inst = np.zeros((res, res), np.int32)
    inst[10 + shift:20 + shift, 10:22] = 5
    inst[40:50, 30 + shift:44 + shift] = 6
    rgb = np.zeros((res, res, 3), np.uint8)
    rgb[inst == 5] = (220, 30, 30)
    rgb[inst == 6] = (30, 30, 220)
    depth = np.where(inst > 0, 0.25, 0.4)
    window = ViewingPose(name="test", camera="wrist", render_size=(res, res),
                         crop=(0, 0, res, res), fovy_deg=60.0, output_size=res)
    return rgb, inst, depth, window


def test_registered_by_name_over_the_ideal_source(tmp_path):
    F = ScenePredictor.create(L, seed=0, meta={"source": "ideal", "weights_sha256": "", "W": 14})
    path = F.save(tmp_path / "F.pt")
    m = sources.get(SOURCE_NAME)(base="ideal", predictor=str(path), gain=0.5)
    assert isinstance(m, PredictiveSceneModel) and type(m.source).__name__ == "IdealSceneModel"
    rgb, inst, depth, window = _ideal_inputs()
    s0 = m.observe(rgb, window=window, instance=inst, depth=depth)
    assert s0.source.startswith(SOURCE_NAME) and int(np.sum(s0.slot_ids >= 0)) == 2
    m.set_motion(CTX, A)
    rgb, inst, depth, window = _ideal_inputs(shift=2)
    s1 = m.observe(rgb, window=window, instance=inst, depth=depth)
    assert m.last_prediction is not None and np.sum(np.isfinite(m.innovation[:, 0])) == 2
    assert np.isfinite(m.innovation_mm)
    assert not np.array_equal(s1.slots, m.last_observation.rows)

    wrong = ScenePredictor.create(L, seed=0, meta={"source": "learned", "weights_sha256": "x",
                                                   "W": 14})
    wpath = wrong.save(tmp_path / "wrong.pt")
    with pytest.raises(ValueError):
        sources.get(SOURCE_NAME)(base="ideal", predictor=str(wpath))
    with pytest.raises(ValueError):
        sources.get(SOURCE_NAME)(base="ideal")


# -------------------------------------------------------------------------------- scoring ---


def test_scoring_predictor_uses_the_feedforward_and_the_stage_scale():
    F = ScenePredictor.create(L, seed=1)
    sp = ScoringPredictor(F, dims=("dx", "dy", "dyaw"), scale=(0.002, 0.002, math.radians(1.5)))
    ff = np.zeros((5, 5))
    ff[0, :4] = (0.001, -0.002, 0.0005, 0.01)
    fr = frame({0: (7, 0.1, 0.2, 0.5)}, background_slot=3)
    obs = {"actor": {"z": fr.rows, "z_depth": fr.depth, "z_valid": fr.valid.astype(np.float32),
                     "mech": np.zeros(18), "tool": np.r_[0.2, 0.0, 0.07, 0.0, 0.0, np.zeros(8)],
                     "progress": np.zeros(12), "feedforward": ff}}
    a = sp.action(obs, [0.5, -1.0, 2.0])
    assert a == pytest.approx([0.002, -0.004, 0.0005, 0.01 + math.radians(1.5)])
    rows = sp(obs, [0.5, -1.0, 2.0])
    assert rows.shape == (K, W)
    want = F.predict(SlotFrame(fr.rows, fr.depth, fr.valid, np.where(fr.valid, np.arange(K), -1)),
                     context_vector(obs["actor"]["mech"], obs["actor"]["tool"]), a).rows
    assert np.array_equal(rows, want)


@pytest.mark.slow
def test_prediction_scoring_runs_with_the_learned_predictor(tmp_path):
    import gymnasium as gym

    import brick_builder.rl.scene_target as st
    from brick_builder.rl.scene_target.wrappers import PredictionScoring

    F = ScenePredictor.create(L, seed=0)
    env = gym.make(st.ENV_ID, cache_dir=str(tmp_path))
    stage = env.unwrapped.cfg.stage
    sp = ScoringPredictor(F, dims=stage.action_dims, scale=env.unwrapped.cfg.resolved_scale())
    env = PredictionScoring(env, sp)
    try:
        obs, _ = env.reset(seed=0, options={"world_seed": 0})
        rng = np.random.default_rng(0)
        for _ in range(4):
            u = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            expected = sp.action(obs, u)
            obs, _, term, trunc, info = env.step(u)
            assert np.isfinite(info["pred_err_mm"]) and np.isfinite(info["copy_last_err_mm"])
            # feed-forward off on L0a, offset inside the ceiling: the executed change is scale * u
            assert np.allclose(info["exec_action"], expected, atol=1e-12)
            if term or trunc:
                break
    finally:
        env.close()


@pytest.mark.slow
def test_environment_runs_on_the_predictive_source_by_name(tmp_path):
    import gymnasium as gym

    import brick_builder.rl.scene_target as st

    F = ScenePredictor.create(L, seed=0, meta={"source": "ideal", "weights_sha256": "", "W": 14})
    path = F.save(tmp_path / "F.pt")
    env = gym.make(st.ENV_ID, source=SOURCE_NAME, cache_dir=str(tmp_path / "cache"),
                   source_kwargs={"base": "ideal", "predictor": str(path), "gain": 0.5})
    try:
        _, info = env.reset(seed=0, options={"world_seed": 0})
        assert info["source"] == SOURCE_NAME
        src = env.unwrapped._source
        assert isinstance(src, PredictiveSceneModel)
        for _ in range(3):
            _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
            assert np.isfinite(info["d_mm"])
        # the environment does not call set_motion: every tick passes the observation through
        assert src.last_prediction is None and src.ticks == 4
    finally:
        env.close()
