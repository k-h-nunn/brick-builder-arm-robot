"""Target sequences: path draws, `.npz` round trips, source encoding, rejection, conversion."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA as S  # noqa: E402
from brick_builder.rl.scene_target import targets as TG  # noqa: E402
from brick_builder.rl.scene_target.targets import (  # noqa: E402
    MIN_REFERENCES, TargetSequence, load_or_generate, min_jerk, reference_counts,
    viewing_pose_draw, viewing_pose_sequence,
)
from brick_builder.rl.scene_target.world import build_source  # noqa: E402


def test_min_jerk_profile():
    t = np.linspace(0.0, 1.0, 101)
    s = min_jerk(t)
    assert s[0] == 0.0 and s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(s) >= 0.0)
    v = np.gradient(s, t)
    assert abs(v[0]) < 1e-2 and abs(v[-1]) < 1e-2
    assert s[50] == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(30))
def test_path_draws_stay_inside_their_ranges(seed):
    d = viewing_pose_draw(seed)
    assert abs(d.target[0] - 0.17) <= 0.030 and abs(d.target[1]) <= 0.030
    assert 0.045 <= d.target[2] <= 0.095 and 0.045 <= d.start[2] <= 0.095
    assert abs(d.target[3]) <= math.radians(15.0) + 1e-12
    dist = np.linalg.norm(d.start[:3] - d.target[:3])
    assert 0.010 - 1e-12 <= dist <= 0.040 + 1e-12
    assert abs(d.start[3] - d.target[3]) <= math.radians(10.0) + 1e-12
    assert 20 <= d.T <= 40
    again = viewing_pose_draw(seed)
    np.testing.assert_array_equal(again.start, d.start)
    np.testing.assert_array_equal(again.target, d.target)


def test_frame_commands_follow_the_path():
    d = viewing_pose_draw(4)
    poses = d.frame_poses()
    cmds = d.frame_commands(0.6)
    np.testing.assert_allclose(poses[0], d.start)
    np.testing.assert_allclose(poses[-1], d.target)
    for k in range(d.T - 1):
        np.testing.assert_allclose(cmds[k, -1, :4], poses[k + 1], atol=1e-12)
    np.testing.assert_allclose(cmds[-1, :, :4], np.repeat(d.target[None], 5, axis=0))
    assert np.all(cmds[..., 4] == 0.6) and np.all(cmds[..., 5:] == 1.0)


def test_reference_counts_on_synthetic_frames():
    T, K = 4, 8
    rows = np.zeros((T, K, S.W), dtype=np.float32)
    depth = np.full((T, K), 0.5, dtype=np.float32)
    ids = np.full((T, K), -1, dtype=np.int64)
    for t in range(T):
        for k, (sid, mass, colour) in enumerate(((1, 0.07, (-0.9, 0.0, -0.5)),
                                                 (2, 0.02, (0.2, 0.0, -0.8)),
                                                 (3, 0.01, (0.1, -0.8, -0.8)))):
            if t == 2 and sid != 1:
                continue
            rows[t, k, S.fields["mass"]] = mass
            rows[t, k, S.fields["centre"]] = (0.1 * k, 0.01 * t)
            rows[t, k, S.fields["moments"]] = (0.003, 0.0, 0.002)
            rows[t, k, S.fields["colour"]] = colour
            depth[t, k] = 0.2 * k
            ids[t, k] = sid
    valid = ~((depth == 0.5) & np.all(rows == 0, axis=-1))
    counts = reference_counts(rows, depth, valid, ids)
    assert counts.tolist() == [3, 1, 1, 1]
    assert counts.min() < MIN_REFERENCES


def _no_object_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return all(z[k].dtype != object for k in z.files)


@pytest.mark.slow
def test_ground_truth_sequence_of_40_frames_round_trips_bitwise(tmp_path):
    seq, stats = load_or_generate(7, source_name="ideal", cache_dir=tmp_path)
    assert stats["cached"] is False
    # 40 frames regardless of the drawn length: generate once more with T forced
    src, info = build_source("ideal")
    seq40, st40 = viewing_pose_sequence(7, source=src, source_info=info, T=40)
    assert seq40.T == 40 and seq40.rows.shape == (40, 8, S.W)
    path = seq40.save(tmp_path / "seq40.npz")
    back = TargetSequence.load(path)
    assert back.bitwise_equal(seq40) and _no_object_arrays(path)
    np.testing.assert_allclose(seq40.tool_pose[0], seq40.meta["start_pose"])
    np.testing.assert_allclose(seq40.tool_pose[-1], seq40.meta["target_pose"])
    assert seq40.meta["min_references"] >= MIN_REFERENCES
    assert np.all(seq40.n_commands == 5) and seq40.freeze_k == 40
    print(f"twin frame cost, ground truth: {st40['frame_ms']:.1f} ms per frame "
          f"(render {1e3 * st40['render_s'] / 40:.1f}, encode {1e3 * st40['encode_s'] / 40:.1f})")
    again, st2 = load_or_generate(7, source_name="ideal", cache_dir=tmp_path)
    assert st2["cached"] is True and again.bitwise_equal(seq)


class _Sabotaged:
    """The ideal source, with all but one reference object removed from frame 3 of its first
    pass: that draw must be rejected."""

    def __init__(self):
        self.inner, _ = build_source("ideal")
        self.passes = 0
        self.i = 0

    def reset(self):
        self.inner.reset()
        self.passes += 1
        self.i = 0

    def observe(self, rgb, **kw):
        state = self.inner.observe(rgb, **kw)
        if self.passes == 1 and self.i == 3:
            refs = [k for k in range(state.K) if state.slot_ids[k] >= 0
                    and 0.0012 <= state.mass(k) <= 0.5]
            for k in refs[1:]:
                state.slots[k] = 0.0
                state.depth[k] = 0.5
                state.slot_ids[k] = -1
        self.i += 1
        return state


@pytest.mark.slow
def test_a_frame_with_fewer_than_two_references_rejects_the_draw():
    _, info = build_source("ideal")
    seq, stats = viewing_pose_sequence(11, source=_Sabotaged(), source_info=info)
    assert any("has 1 matched reference slot" in r for r in seq.meta["rejected"])
    assert seq.meta["min_references"] >= MIN_REFERENCES
    assert stats["draws"] >= 2


@pytest.mark.slow
def test_convolutional_sequence_round_trips_without_pickling_and_reencodes_bitwise(tmp_path):
    from brick_builder.perception.cnn.source import DEFAULT_WEIGHTS
    from brick_builder.rl.scene_target.targets import encode_frames, _stack_states
    from brick_builder.rl.scene_target.world import sha256_file

    src, info = build_source("learned", {"weights": str(DEFAULT_WEIGHTS), "threads": 1})
    assert info.weights_sha256 == sha256_file(DEFAULT_WEIGHTS)
    seq, stats = viewing_pose_sequence(2, source=src, source_info=info, T=20, keep_frames=True)
    path = seq.save(tmp_path / "cnn.npz")
    assert _no_object_arrays(path)
    back = TargetSequence.load(path)
    assert back.bitwise_equal(seq)
    assert back.meta["weights_sha256"] == info.weights_sha256 and back.meta["source"] == "learned"
    fresh, _ = build_source("learned", {"weights": str(DEFAULT_WEIGHTS), "threads": 1})
    states = encode_frames(stats["frames_rendered"], fresh)
    rows, depth, valid, ids = _stack_states(states, 8, S.W)
    np.testing.assert_array_equal(rows, seq.rows)
    np.testing.assert_array_equal(depth, seq.depth)
    np.testing.assert_array_equal(ids, seq.slot_ids)
    print(f"twin frame cost, convolutional (1 thread): {stats['frame_ms']:.1f} ms per frame")


def _synthetic_ideal(T=6):
    from brick_builder.action.sequence import Command, IdealSequence, ReferenceFrame
    from brick_builder.perception.scene_model import SceneState

    rng = np.random.default_rng(0)
    steps = ["Transport", "Transport", "Align", "Align", "Press the brick down", "Release"]
    frames = []
    for k in range(T):
        slots = np.zeros((8, S.W), dtype=np.float32)
        slots[1:4] = rng.normal(size=(3, S.W))
        depth = np.full(8, 0.5, dtype=np.float32)
        depth[1:4] = (0.0, 0.5, 1.0)
        ids = np.array([-1, 3, 4, 5, -1, -1, -1, -1])
        state = SceneState(slots=slots, depth=depth, schema=S, t=0.1 * k, source="ideal",
                           window=None, slot_ids=ids)
        n = 5 if k < T - 1 else 2
        cmds = [Command(pos=rng.normal(size=3), spin=float(rng.normal()), grip=0.5,
                        c_vert=0.7 if steps[k] == "Press the brick down" else 1.0, c_lat=1.0)
                for _ in range(n)]
        f = ReferenceFrame(k=k, t=0.1 * k, step=steps[k],
                           step_index=[0, 0, 1, 1, 2, 3][k], state=state, commands=cmds,
                           action=rng.normal(size=4))
        frames.append(f)
    return IdealSequence(frames=frames, bodies_by_step={})


def test_arrays_equal_the_ideal_sequence_fields():
    ideal = _synthetic_ideal()
    seq = TG.from_ideal_sequence(ideal, meta={"family": "placement"})
    assert seq.T == 6 and seq.titles == tuple(ideal.titles())
    for k, f in enumerate(ideal.frames):
        np.testing.assert_array_equal(seq.rows[k], f.state.slots)
        np.testing.assert_array_equal(seq.depth[k], f.state.depth)
        np.testing.assert_array_equal(seq.slot_ids[k], f.state.slot_ids)
        np.testing.assert_array_equal(seq.action[k], f.action)
        assert seq.step_index[k] == f.step_index
        assert seq.n_commands[k] == len(f.commands)
        for i in range(5):
            c = f.commands[min(i, len(f.commands) - 1)]
            np.testing.assert_array_equal(seq.commands[k, i, :3], c.pos)
            assert tuple(seq.commands[k, i, 3:]) == (c.spin, c.grip, c.c_vert, c.c_lat)
    assert seq.freeze_k == 4          # first frame after the last Align frame
    np.testing.assert_array_equal(seq.tool_pose[3], seq.commands[2, 4, :4])


@pytest.mark.slow
def test_arrays_equal_a_real_recording(tmp_path):
    """`record_ideal` (placement, ground truth) converted, stored and reloaded."""
    from brick_builder.action.place import PlaceConfig, record_ideal

    ideal, world = record_ideal(PlaceConfig())
    world.close()
    seq = TG.from_ideal_sequence(ideal, meta={"family": "placement"})
    assert seq.T == len(ideal.frames)
    for k, f in enumerate(ideal.frames):
        np.testing.assert_array_equal(seq.rows[k], f.state.slots)
        np.testing.assert_array_equal(seq.slot_ids[k], f.state.slot_ids)
        np.testing.assert_array_equal(seq.action[k], f.action)
        n = len(f.commands)
        assert seq.n_commands[k] == min(n, 5)
        np.testing.assert_array_equal(seq.commands[k, :n, :3], np.stack([c.pos for c in f.commands]))
    back = TargetSequence.load(seq.save(tmp_path / "placement.npz"))
    assert back.bitwise_equal(seq)
    align = [k for k, f in enumerate(ideal.frames) if f.step == "Align"]
    assert seq.freeze_k == max(align) + 1
