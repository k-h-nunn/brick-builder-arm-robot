"""Next-scene-state predictor `F`: control-affine structure, synthetic toy, determinism, resume."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA as S
from brick_builder.perception.scene_model import opaque_schema
from brick_builder.rl.scene_target.codec import NAMED_SCALES, OTHER_WEIGHT, SlotFrame
from brick_builder.rl.scene_target.dataset import (
    EpisodeLog, ToyConfig, Windows, load_episodes, split_logs, synthetic_logs,
)
from brick_builder.rl.scene_target.model import (
    CONTEXT_W, MECH_W, STEP_W, TOOL_W, ModelConfig, ScenePredictor, SlotLayout,
    context_from_obs, context_vector, fit_input_norm,
)
from brick_builder.rl.scene_target.train_model import (
    TrainConfig, action_response, b_cosine, one_step_report, prediction_report, train,
)

PKG = Path(__file__).resolve().parents[3] / "brick_builder" / "rl" / "scene_target"
TOY = ToyConfig()
META = {"source": "synthetic", "weights_sha256": "", "W": S.W}


def _no_seconds(history):
    return [{k: v for k, v in h.items() if k != "seconds"} for h in history]


def _state_equal(a: ScenePredictor, b: ScenePredictor) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _random_frame(rng, K=8, n_obj=5):
    rows = np.zeros((K, S.W), dtype=np.float32)
    depth = np.full(K, 0.5, dtype=np.float32)
    ids = np.full(K, -1)
    valid = np.zeros(K, dtype=bool)
    for i in range(n_obj):
        rows[i] = rng.uniform(-0.5, 0.5, S.W)
        rows[i, S.fields["border"]] = rng.uniform(0.0, 0.2)
        depth[i] = rng.uniform(0.0, 1.0)
        ids[i], valid[i] = 10 + i, True
    valid[n_obj] = True  # background: valid, id -1
    rows[n_obj, S.fields["mass"]] = 0.5
    return SlotFrame(rows, depth, valid, ids)


def _random_context(rng):
    return context_vector(rng.normal(size=MECH_W), rng.normal(size=TOOL_W) * 0.1)


# ------------------------------------------------------------------------------ structure ---


def test_layout_uses_the_codec_fields_and_scales():
    L = SlotLayout.for_schema(S)
    assert L.named and L.W == 14 and L.D == 9
    assert L.dyn_names == ("centre_0", "centre_1", "mass_0", "moments_0", "moments_1",
                           "moments_2", "orient4_0", "orient4_1", "depth")
    assert L.dyn_cols == (1, 2, 0, 3, 4, 5, 12, 13)
    assert L.app_cols == (6, 7, 8, 9)
    assert L.border_col == 10
    want = [NAMED_SCALES[n] for n in ("centre", "centre", "mass", "moments", "moments",
                                      "moments", "orient4", "orient4", "depth")]
    assert L.dyn_scale == pytest.approx(want)
    assert L.dyn_weight == pytest.approx([1.0, 1.0] + [OTHER_WEIGHT] * 7)
    assert L.centre == (0, 1)
    assert SlotLayout.from_dict(json.loads(json.dumps(L.to_dict()))) == L

    std = np.linspace(0.1, 3.2, 32)
    O = SlotLayout.for_schema(opaque_schema(32), feature_std=std, depth_std=0.3)
    assert not O.named and O.D == 33 and O.A == 0 and O.border_col == -1
    assert O.dyn_scale == pytest.approx(list(std) + [0.3])
    assert O.centre == ()


def test_context_layout_matches_the_environment():
    from brick_builder.rl.scene_target import env

    assert (MECH_W, TOOL_W, STEP_W) == (env.MECH_W, env.TOOL_W, env.PROGRESS_W - 3)
    assert CONTEXT_W == 43
    mech, tool = np.arange(18.0), 100.0 + np.arange(13.0)
    step = np.eye(9)[4]
    c = context_vector(mech, tool, step, (0.3, 0.7), 0.9)
    assert c.shape == (43,)
    assert np.array_equal(c[:18], mech) and np.array_equal(c[18:31], tool)
    assert np.array_equal(c[31:40], step) and np.allclose(c[40:], [0.3, 0.7, 0.9])
    assert context_vector(mech, tool)[42] == tool[4]
    actor = {"mech": mech, "tool": tool, "progress": np.concatenate([[0.1, 0.2, 0.3], step])}
    assert np.array_equal(context_from_obs(actor, (0.3, 0.7), 0.9), c)


def test_slots_are_permutation_equivariant():
    rng = np.random.default_rng(3)
    model = ScenePredictor.create(SlotLayout.for_schema(S), seed=0)
    fr = _random_frame(rng)
    ctx = torch.as_tensor(_random_context(rng))[None]
    perm = rng.permutation(8)
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x))[None].to(dt)  # noqa: E731
    f1, B1, v1, q1 = model.heads(t(fr.rows), t(fr.depth), t(fr.objects, torch.bool), ctx)
    f2, B2, v2, q2 = model.heads(t(fr.rows[perm]), t(fr.depth[perm]),
                                 t(fr.objects[perm], torch.bool), ctx)
    assert torch.allclose(f1[0, perm], f2[0], atol=1e-6)
    assert torch.allclose(B1[0, perm], B2[0], atol=1e-6)
    assert torch.allclose(v1[0, perm], v2[0], atol=1e-6)
    assert torch.allclose(q1, q2, atol=1e-6)


def test_prediction_is_affine_in_the_action_at_any_size():
    rng = np.random.default_rng(4)
    model = ScenePredictor.create(SlotLayout.for_schema(S), seed=1)
    fr, ctx = _random_frame(rng), _random_context(rng)
    L = model.layout
    _, B = model.affine(fr, ctx)
    obj = fr.objects
    assert np.linalg.norm(B[obj]) > 0.0
    assert np.all(B[~obj] == 0.0)
    base = np.asarray(TOY.action_range)
    for factor in (0.0, 1.0, 10.0, 1000.0):
        J = model.action_jacobian(fr, ctx, factor * base)
        assert np.allclose(J[obj], B[obj], rtol=1e-5, atol=1e-6 * np.abs(B).max())
        assert np.all(J[~obj] == 0.0)

    def dyn(a):
        p = model.predict(fr, ctx, a)
        return L.dyn(p.rows.astype(np.float64), p.depth.astype(np.float64))

    z0, z1 = dyn(np.zeros(4)), dyn(base)
    for factor in (10.0, 100.0):
        dev = (dyn(factor * base) - z0) - factor * (z1 - z0)
        assert np.max(np.abs(dev[obj]) / np.asarray(L.dyn_scale)) < 1e-2 * factor
    # appearance, flags, background and empty slots are copied
    p = model.predict(fr, ctx, 10.0 * base)
    assert np.array_equal(p.rows[:, list(L.app_cols)], fr.rows[:, list(L.app_cols)])
    assert np.array_equal(p.rows[~obj], fr.rows[~obj]) and np.array_equal(p.depth[~obj], fr.depth[~obj])


def _fitted_model(seed=0):
    """An untrained `F` with the input normalization fitted on toy data (non-trivial scales)."""
    logs, toy = synthetic_logs(300, seed=seed)
    w = Windows.from_logs(logs, horizon=1)
    norm = fit_input_norm(toy.layout, w.rows[:, 0], w.depth[:, 0], w.present[:, 0], w.ctx[:, 0])
    return ScenePredictor.create(toy.layout, seed=seed, norm=norm), logs


def test_background_and_empty_rows_are_not_inputs():
    model, logs = _fitted_model(seed=11)
    log = logs[0]
    ctx, a = log.contexts()[4], log.exec_action[5]
    fr = log.frame(4)
    bg, obj = fr.background, fr.objects
    assert bg.sum() == 1 and obj.sum() >= 2
    p0 = model.predict(fr, ctx, a)
    f0, B0 = model.affine(fr, ctx)
    # any background row, NaN included, or no background at all: objects unchanged
    for fill in (1e3, -7.0, np.nan):
        rows = fr.rows.copy()
        rows[bg] = fill
        depth = fr.depth.copy()
        depth[bg] = fill
        empty = ~fr.valid
        rows[empty], depth[empty] = fill, fill
        for other in (SlotFrame(rows, depth, fr.valid, fr.slot_ids),
                      SlotFrame(rows, depth, fr.valid & ~bg, fr.slot_ids)):
            p = model.predict(other, ctx, a)
            f, B = model.affine(other, ctx)
            assert np.array_equal(p.rows[obj], p0.rows[obj]), fill
            assert np.array_equal(p.depth[obj], p0.depth[obj]), fill
            assert np.array_equal(f, f0) and np.array_equal(B, B0), fill
            assert np.array_equal(model.action_jacobian(other, ctx, a)[obj],
                                  model.action_jacobian(fr, ctx, a)[obj])


def test_window_path_equals_the_public_api_on_logged_frames():
    """The training path (window tensors) and `rollout` / `predict` on the logged frames agree,
    background slot included."""
    model, logs = _fitted_model(seed=12)
    w = Windows.from_logs(logs, horizon=3)
    T = w.tensors()
    idx = np.arange(0, len(w), max(len(w) // 40, 1))
    ti = torch.from_numpy(idx)
    with torch.no_grad():
        steps = model.rollout_t(T["rows"][ti, 0], T["depth"][ti, 0], T["present"][ti, 0],
                                T["ctx"][ti], T["action"][ti])
    by_id = {log.episode_id: log for log in logs}
    n_bg = 0
    for n, i in enumerate(idx):
        log, t = by_id[int(w.episode[i])], int(w.start[i])
        fr = log.frame(t)
        wf = w.frame(i)
        for k in ("rows", "depth", "valid", "slot_ids"):
            assert np.array_equal(getattr(wf, k), getattr(fr, k)), k
        n_bg += int(fr.background.sum())
        ctxs = log.contexts()[t:t + 3]
        acts = log.exec_action[t + 1:t + 4]
        pub = model.rollout(fr, ctxs, acts)
        for h in range(3):
            assert np.allclose(pub[h].rows, steps[h].rows[n].numpy(), atol=1e-6)
        one = model.predict(fr, ctxs[0], acts[0])
        assert np.allclose(one.rows, steps[0].rows[n].numpy(), atol=1e-6)
    assert n_bg == len(idx)  # every toy frame has its background slot


def test_no_clamp_or_clip_in_the_predictor():
    banned = {"clamp", "clamp_", "clip", "clip_", "clamp_min", "clamp_max", "hardtanh",
              "nan_to_num", "Hardtanh"}
    tree = ast.parse((PKG / "model.py").read_text())
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
             if isinstance(n, (ast.Attribute, ast.Name))}
    assert not names & banned, names & banned


def test_opaque_width_32_trains_and_predicts():
    rng = np.random.default_rng(5)
    logs, _ = synthetic_logs(300, seed=5)
    proj = np.linalg.qr(rng.normal(size=(32, 32)))[0][:14]
    for log in logs:
        log.z = (log.z @ proj).astype(np.float32) * log.valid[..., None]
    w = Windows.from_logs(logs, horizon=3)
    layout = SlotLayout.for_schema(opaque_schema(32), feature_std=np.full(32, 0.01))
    res = train(w, TrainConfig(epochs=1, batch=64), layout=layout, log=None,
                meta={"source": "opaque", "weights_sha256": "", "W": 32})
    p = res.model.predict(w.frame(0), w.ctx[0, 0], w.action[0, 0])
    assert p.rows.shape == (8, 32) and np.all(np.isfinite(p.rows))


# -------------------------------------------------------------------------------- dataset ---


def _tiny_log():
    """Three slots over 5 ticks: ids 7 and 9 swap slots at tick 2; id 9 is missed at tick 3."""
    N, K, W = 5, 3, S.W
    z = np.zeros((N, K, W), dtype=np.float32)
    depth = np.full((N, K), 0.5, dtype=np.float32)
    ids = np.full((N, K), -1)
    valid = np.zeros((N, K), dtype=bool)
    for t in range(N):
        a, b = (0, 1) if t < 2 else (1, 0)
        z[t, a, 1] = 0.1 * t          # id 7 centre x
        z[t, b, 1] = -0.2 * t - 0.01  # id 9 centre x
        ids[t, a], ids[t, b] = 7, 9
        valid[t, a] = valid[t, b] = True
        depth[t, a], depth[t, b] = 0.2, 0.8
        z[t, 2, 0] = 0.5              # background
        valid[t, 2] = True
    z[3, 0] = 0.0
    depth[3, 0], ids[3, 0], valid[3, 0] = 0.5, -1, False
    mech = np.arange(N)[:, None] * 1.0 + np.zeros((N, 18))
    tool = np.arange(N)[:, None] * 10.0 + np.zeros((N, 13))
    act = np.arange(N)[:, None] * 0.001 + np.zeros((N, 4))
    act[0] = np.nan
    return EpisodeLog(z=z, depth=depth, valid=valid, slot_ids=ids, mech=mech, tool=tool,
                      exec_action=act, grip=np.arange(N) * 0.5,
                      compliance=np.stack([np.arange(N), -np.arange(N)], 1).astype(float),
                      step_onehot=np.zeros((N, 9)), meta={"episode_id": 3})


def test_windows_pair_slots_by_id_and_align_contexts():
    w = Windows.from_logs([_tiny_log()], horizon=2)
    assert len(w) == 3 and w.rows.shape == (3, 3, 3, S.W)
    # window t=1: id 7 in slot 0, id 9 in slot 1
    assert w.rows[1, 0, 0, 1] == pytest.approx(0.1) and w.rows[1, 0, 1, 1] == pytest.approx(-0.21)
    assert w.rows[1, 1, 0, 1] == pytest.approx(0.2)    # id 7 at t=2 (slot 1 there)
    assert w.rows[1, 1, 1, 1] == pytest.approx(-0.41)  # id 9 at t=2 (slot 0 there)
    assert w.rows[1, 2, 0, 1] == pytest.approx(0.3)    # id 7 at t=3
    assert not w.present[1, 2, 1] and np.all(w.rows[1, 2, 1] == 0.0)  # id 9 missed at t=3
    assert w.depth[1, 2, 1] == 0.5
    assert list(w.present[1, 0]) == [True, True, False]  # background never an object
    assert list(w.valid0[1]) == [True, True, True]
    # the first tick is the logged frame, background row and slot ids included
    assert w.rows[1, 0, 2, 0] == pytest.approx(0.5) and list(w.ids0[1]) == [7, 9, -1]
    assert np.all(w.rows[1, 1:, 2] == 0.0) and not w.present[1, 1:, 2].any()
    fr = w.frame(1)
    assert np.array_equal(fr.rows, _tiny_log().z[1]) and list(fr.background) == [False, False, True]
    # context of tick t -> t+1: mech_t, tool_t, then compliance and grip of row t+1
    assert np.allclose(w.ctx[1, 0, :18], 1.0) and np.allclose(w.ctx[1, 0, 18:31], 10.0)
    assert np.allclose(w.ctx[1, 0, 40:], [2.0, -2.0, 1.0])
    assert np.allclose(w.ctx[1, 1, 40:], [3.0, -3.0, 1.5])
    assert np.allclose(w.action[1], [[0.002] * 4, [0.003] * 4])
    assert np.allclose(w.q_next[1], [[2.0] * 6, [3.0] * 6])
    assert np.all(w.episode == 3) and list(w.start) == [0, 1, 2]
    assert w.fingerprint() == Windows.from_logs([_tiny_log()], horizon=2).fingerprint()


def test_five_step_windows_train_open_loop():
    logs, toy = synthetic_logs(400, seed=6)
    w5 = Windows.from_logs(logs, horizon=5)
    w3 = Windows.from_logs(logs, horizon=3)
    assert w5.rows.shape[1] == 6 and w5.ctx.shape[1] == 5 and len(w5) == len(w3) - 2 * len(logs)
    assert np.array_equal(w5.rows[:, :4], w3.subset(np.nonzero(w3.start < 46)[0]).rows)
    res = train(w5, TrainConfig(epochs=2, batch=64, teacher_frac=0.5), layout=toy.layout,
                log=None, meta=META)
    assert [h["teacher"] for h in res.history] == [True, False]
    assert all(np.isfinite(h["loss"]) for h in res.history)
    rep = prediction_report(res.model, w5)
    assert len(rep["centre_rmse_mm"]) == 5


def test_every_seventh_episode_is_held_out():
    logs, _ = synthetic_logs(15 * TOY.episode_ticks, seed=1)
    assert len(logs) == 15
    w = Windows.from_logs(logs, horizon=3)
    tr, va = w.split()
    assert set(np.unique(va.episode)) == {6, 13}
    assert not set(np.unique(tr.episode)) & {6, 13}
    assert len(tr) + len(va) == len(w)


def test_episode_log_round_trips_without_pickle(tmp_path):
    log = _tiny_log()
    log.extras["d_mm"] = np.arange(5.0)
    p = log.save(tmp_path / "ep_000003.npz")
    back = EpisodeLog.load(p)
    for k in ("z", "depth", "valid", "slot_ids", "mech", "tool", "grip", "compliance",
              "step_onehot"):
        assert np.array_equal(getattr(back, k), getattr(log, k)), k
    assert np.array_equal(back.exec_action, log.exec_action, equal_nan=True)
    assert back.meta["episode_id"] == 3 and np.array_equal(back.extras["d_mm"], np.arange(5.0))
    with np.load(p, allow_pickle=False) as f:
        assert "meta" in f.files
    # a log without step one-hot takes it from `progress`; the id from the file name
    arr = {k: v for k, v in log.arrays().items() if k not in ("step_onehot", "meta")}
    arr["progress"] = np.concatenate([np.zeros((5, 3)), np.tile(np.eye(9)[2], (5, 1))], 1)
    np.savez(tmp_path / "ep_000011.npz", **arr)
    other = load_episodes(tmp_path)[1]
    assert other.episode_id == 11 and np.all(other.step_onehot[:, 2] == 1.0)


def test_episode_log_from_environment_records():
    rng = np.random.default_rng(0)
    K, W, H = 8, S.W, 5

    def obs():
        return {"actor": {"z": rng.normal(size=(K, W)).astype(np.float32),
                          "z_depth": rng.uniform(size=K).astype(np.float32),
                          "z_valid": np.ones(K, np.float32), "mech": rng.normal(size=18),
                          "tool": rng.normal(size=13), "progress": np.zeros(12),
                          "feedforward": np.zeros((H, 5))}}

    o0 = obs()
    steps = []
    for _ in range(3):
        o = obs()
        tr = {"z": o["actor"]["z"], "depth": o["actor"]["z_depth"], "valid": np.ones(K, bool),
              "slot_ids": np.arange(K), "mech": o["actor"]["mech"], "tool": o["actor"]["tool"],
              "exec_action": rng.normal(size=4), "u": rng.normal(size=4), "grip": 0.25,
              "compliance": np.array([0.5, 0.6]), "d_mm": 1.0, "k": 1}
        steps.append((o, {"transition": tr}))
    log = EpisodeLog.from_steps(o0, {"slot_ids": np.arange(K), "world_seed": 1000}, steps)
    assert log.n_ticks == 4 and log.n_transitions == 3
    assert np.all(np.isnan(log.exec_action[0]))
    assert np.array_equal(log.z[2], steps[1][0]["actor"]["z"])
    assert np.array_equal(log.exec_action[3], steps[2][1]["transition"]["exec_action"])
    assert log.grip[0] == o0["actor"]["tool"][4] and log.grip[1] == 0.25
    assert log.meta["world_seed"] == 1000 and log.extras["u"].shape == (3, 4)
    assert len(Windows.from_logs([log], horizon=3)) == 1


# ---------------------------------------------------------------------------- acceptance ---


@pytest.fixture(scope="module")
def toy_run():
    logs, toy = synthetic_logs(8000, seed=0)
    w = Windows.from_logs(logs, horizon=3)
    tr, va = w.split()
    res = train(tr, TrainConfig(seed=0, epochs=30), layout=toy.layout, log=None, meta=META)
    _, held_logs = split_logs(logs)
    return res, tr, va, toy, held_logs


def test_synthetic_toy_recovers_B(toy_run):
    """`B` through `ScenePredictor.affine` on the held-out logged frames, background included."""
    res, _, _, toy, held_logs = toy_run
    assert res.completed
    assert held_logs and all(log.episode_id % 7 == 6 for log in held_logs)
    assert all(log.frame(t).background.any() for log in held_logs for t in (0, 10))
    stats = b_cosine(res.model, held_logs, toy.affine)
    assert stats["all"]["n"] > 1000
    assert stats["all"]["cos_median"] >= 0.99, stats
    assert stats["all"]["cos_mean"] >= 0.99, stats
    assert stats["centre"]["cos_median"] >= 0.99, stats
    assert 0.95 <= stats["all"]["gain_median"] <= 1.05, stats


def test_action_response_is_B_at_ten_times_the_training_range(toy_run):
    res, _, _, toy, held_logs = toy_run
    model = res.model
    rep = action_response(model, held_logs, TOY.action_range, factor=10.0, n=32,
                          truth=toy.affine)
    assert rep["n"] == 32
    assert rep["jacobian_over_B_min"] == pytest.approx(1.0, abs=1e-5)
    assert rep["nonlinearity_max_scaled"] < 1e-3
    assert 0.9 <= rep["jacobian_over_true_B_min"] and rep["jacobian_over_true_B_max"] <= 1.1, rep


def test_one_step_prediction_on_logged_frames_beats_copy_last(toy_run):
    """`ScenePredictor.predict` on the held-out logged frames, background included."""
    res, _, _, _, held_logs = toy_run
    rep = one_step_report(res.model, held_logs)
    assert rep["n_ticks"] > 1000
    assert rep["centre_rmse_mm"][0] < 0.2 * rep["centre_rmse_copy_last_mm"][0], rep


def test_prediction_beats_copy_last_on_held_out_episodes(toy_run):
    res, _, va, _, _ = toy_run
    rep = prediction_report(res.model, va)
    for h in range(3):
        assert rep["centre_rmse_mm"][h] < 0.5 * rep["centre_rmse_copy_last_mm"][h], rep


def test_training_is_deterministic_by_seed():
    logs, toy = synthetic_logs(1500, seed=2)
    w = Windows.from_logs(logs, horizon=3)
    cfg = TrainConfig(seed=7, epochs=3, batch=128)
    a = train(w, cfg, layout=toy.layout, log=None, meta=META)
    b = train(w, cfg, layout=toy.layout, log=None, meta=META)
    c = train(w, TrainConfig(seed=8, epochs=3, batch=128), layout=toy.layout, log=None, meta=META)
    assert _state_equal(a.model, b.model)
    assert _no_seconds(a.history) == _no_seconds(b.history)
    assert a.version == b.version
    assert not _state_equal(a.model, c.model) and a.version != c.version


def test_resume_after_an_interrupted_epoch_is_exact(tmp_path):
    logs, toy = synthetic_logs(1500, seed=3)
    w = Windows.from_logs(logs, horizon=3)
    tr, va = w.split()
    cfg = TrainConfig(seed=5, epochs=4, batch=128)
    full = train(tr, cfg, layout=toy.layout, val_w=va, log=None, meta=META)

    state = tmp_path / "train_state.pt"

    class Kill(Exception):
        pass

    def kill(epoch, step):
        if epoch == 2 and step == 3:
            raise Kill

    with pytest.raises(Kill):
        train(tr, cfg, layout=toy.layout, val_w=va, state_path=state, log=None, meta=META,
              on_step=kill)
    part = torch.load(state, weights_only=True)
    assert part["epoch"] == 2
    res = train(tr, cfg, layout=toy.layout, val_w=va, state_path=state,
                out_path=tmp_path / "model.pt", log=None, meta=META)
    assert res.resumed_from == 2 and res.completed
    assert _state_equal(full.model, res.model)
    assert _no_seconds(full.history) == _no_seconds(res.history)
    loaded = ScenePredictor.load(tmp_path / "model.pt", source="synthetic", W=S.W,
                                 weights_sha256="")
    assert _state_equal(loaded, res.model)
    # a finished run resumed again runs no epoch and returns the same weights
    again = train(tr, cfg, layout=toy.layout, val_w=va, state_path=state, log=None, meta=META)
    assert again.resumed_from == 4 and _state_equal(again.model, res.model)
    # the state file refuses a different configuration or data set
    with pytest.raises(ValueError):
        train(tr, TrainConfig(seed=5, epochs=4, batch=128, lr=2e-3), layout=toy.layout,
              val_w=va, state_path=state, log=None, meta=META)
    with pytest.raises(ValueError):
        train(va, cfg, layout=toy.layout, val_w=va, state_path=state, log=None, meta=META)


def test_time_budget_stops_at_an_epoch_boundary_and_resumes(tmp_path):
    logs, toy = synthetic_logs(600, seed=4)
    w = Windows.from_logs(logs, horizon=3)
    cfg = TrainConfig(seed=0, epochs=2, batch=128)
    state = tmp_path / "s.pt"
    stopped = train(w, cfg, layout=toy.layout, state_path=state, minutes=0.0, log=None, meta=META)
    assert not stopped.completed and stopped.history == []
    done = train(w, cfg, layout=toy.layout, state_path=state, log=None, meta=META)
    full = train(w, cfg, layout=toy.layout, log=None, meta=META)
    assert done.completed and _state_equal(done.model, full.model)


def test_checkpoint_refuses_another_source_width_or_weights(tmp_path):
    rng = np.random.default_rng(6)
    model = ScenePredictor.create(SlotLayout.for_schema(S), seed=2,
                                  meta={"source": "learned", "weights_sha256": "ab" * 32, "W": 14})
    p = model.save(tmp_path / "F.pt")
    back = ScenePredictor.load(p, source="learned", W=14, weights_sha256="ab" * 32)
    fr, ctx = _random_frame(rng), _random_context(rng)
    a = np.array([1e-3, -1e-3, 0.0, 0.01])
    assert np.array_equal(back.predict(fr, ctx, a).rows, model.predict(fr, ctx, a).rows)
    for bad in ({"source": "ideal"}, {"W": 32}, {"weights_sha256": "cd" * 32}):
        with pytest.raises(ValueError):
            ScenePredictor.load(p, **bad)
    with pytest.raises(ValueError):
        ScenePredictor.create(SlotLayout.for_schema(S), seed=0).check_compatible(source="ideal")


def test_input_normalization_is_affine_and_never_clipped():
    L = SlotLayout.for_schema(S)
    rng = np.random.default_rng(7)
    rows = rng.normal(size=(50, 8, 14))
    depth = rng.uniform(size=(50, 8))
    valid = rng.random((50, 8)) < 0.7
    ctx = rng.normal(size=(50, CONTEXT_W))
    ctx[:, 5] = 3.0  # constant component keeps unit scale
    norm = fit_input_norm(L, rows, depth, valid, ctx)
    assert norm.ctx_std[5] == 1.0 and norm.ctx_mean[5] == 3.0
    model = ScenePredictor.create(L, ModelConfig(), seed=0, norm=norm)
    fr = _random_frame(rng)
    big = 1000.0 * _random_context(rng)
    x, c = model._inputs(torch.as_tensor(fr.rows)[None], torch.as_tensor(fr.depth)[None],
                         torch.as_tensor(fr.objects)[None], torch.as_tensor(big)[None])
    want = (big - norm.ctx_mean) / norm.ctx_std
    assert np.allclose(c[0].numpy(), want, rtol=1e-5)
    assert float(torch.max(torch.abs(c))) > 100.0
    dyn = L.dyn(fr.rows.astype(np.float64), fr.depth.astype(np.float64))
    want_x = (dyn - norm.slot_mean[:L.D]) / norm.slot_std[:L.D]
    obj = fr.objects
    assert fr.background.any()
    assert np.allclose(x[0, obj, :L.D].numpy(), want_x[obj], rtol=1e-4, atol=1e-5)
    assert np.all(x[0, ~obj].numpy() == 0.0)  # background and empty slots
