"""Action to servo-bus writes: offset accumulation, ceilings, replay of the recorded commands."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brick_builder.rl.scene_target.config import LEVELS
from brick_builder.rl.scene_target.executor import apply_action, tick_commands

L0A, L0B = LEVELS["viewing_pose"], LEVELS["viewing_pose_displaced"]


def ceiling(spec):
    return spec.ceiling_xy, spec.ceiling_z, spec.ceiling_yaw


def test_offset_accumulates_in_units_of_the_stage_scale():
    off = apply_action(np.zeros(4), [1.0, -0.5, 0.25, 1.0], dims=L0A.action_dims,
                       scale=L0A.action_scale, ceiling=ceiling(L0A))
    np.testing.assert_allclose(off, [0.003, -0.0015, 0.0005, math.radians(2.0)])
    off = apply_action(off, [1.0, 0.0, 0.0, 0.0], dims=L0A.action_dims,
                       scale=L0A.action_scale, ceiling=ceiling(L0A))
    assert off[0] == pytest.approx(0.006)


def test_actions_are_clipped_to_the_box():
    off = apply_action(np.zeros(4), [5.0, -5.0, 0.0], dims=L0B.action_dims,
                       scale=L0B.action_scale, ceiling=ceiling(L0B))
    np.testing.assert_allclose(off, [0.002, -0.002, 0.0, 0.0])


def test_l0b_has_no_height_dimension():
    off = apply_action(np.zeros(4), [0.0, 0.0, 1.0], dims=L0B.action_dims,
                       scale=L0B.action_scale, ceiling=ceiling(L0B))
    assert off[2] == 0.0 and off[3] == pytest.approx(math.radians(1.5))
    with pytest.raises(ValueError):
        apply_action(np.zeros(4), [0.0] * 4, dims=L0B.action_dims, scale=L0B.action_scale,
                     ceiling=ceiling(L0B))


def test_ceiling_holds_the_plane_norm_height_and_yaw():
    off = np.zeros(4)
    for _ in range(200):
        off = apply_action(off, [1.0, 1.0, 1.0, 1.0], dims=L0A.action_dims,
                           scale=L0A.action_scale, ceiling=ceiling(L0A))
    assert math.hypot(off[0], off[1]) == pytest.approx(0.050)
    assert off[0] == pytest.approx(off[1])
    assert off[2] == pytest.approx(0.030) and off[3] == pytest.approx(math.radians(20.0))


def recorded(T=6, seed=0):
    rng = np.random.default_rng(seed)
    commands = rng.normal(size=(T, 5, 7))
    commands[:, :, 5:] = 1.0
    n = np.full(T, 5)
    n[-1] = 3
    return commands, n


def test_feedforward_advancing_tick_is_the_recorded_frame_plus_offset():
    commands, n = recorded()
    off = np.array([0.001, -0.002, 0.0005, 0.01])
    out = tick_commands(commands, n, 2, held=False, offset=off, feedforward=True)
    np.testing.assert_allclose(out[:, :3], commands[2, :, :3] + off[:3])
    np.testing.assert_allclose(out[:, 3], commands[2, :, 3] + off[3])
    np.testing.assert_array_equal(out[:, 4:], commands[2, :, 4:])


def test_held_tick_repeats_the_frames_last_recorded_write():
    commands, n = recorded()
    out = tick_commands(commands, n, 5, held=True, offset=np.zeros(4), feedforward=True)
    assert out.shape == (5, 7)
    np.testing.assert_array_equal(out, np.repeat(commands[5, 2][None], 5, axis=0))


def test_feedforward_off_holds_the_start_pose_plus_offset():
    commands, n = recorded()
    start = np.array([0.18, 0.01, 0.06, 0.1, 0.63, 1.0, 1.0])
    off = np.array([0.003, 0.0, -0.002, 0.02])
    for k, held in ((0, False), (3, True), (5, False)):
        out = tick_commands(commands, n, k, held=held, offset=off, feedforward=False,
                            start_command=start)
        np.testing.assert_allclose(out[:, :4], np.repeat((start[:4] + off)[None], 5, axis=0))
        np.testing.assert_array_equal(out[:, 4:], np.repeat(start[None, 4:], 5, axis=0))
    with pytest.raises(ValueError):
        tick_commands(commands, n, 0, held=False, offset=off, feedforward=False)


def test_zero_action_replays_every_recorded_write_exactly():
    """`u = 0` with the feed-forward on and an index that advances every tick: the issued writes
    are the recording, bit for bit."""
    commands, n = recorded(T=12, seed=3)
    off = np.zeros(4)
    issued = []
    for k in range(12):
        off = apply_action(off, np.zeros(3), dims=L0B.action_dims, scale=L0B.action_scale,
                           ceiling=ceiling(L0B))
        issued.append(tick_commands(commands, n, k, held=False, offset=off, feedforward=True))
    np.testing.assert_array_equal(np.stack(issued), commands)
