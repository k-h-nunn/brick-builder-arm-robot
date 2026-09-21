"""Tests for the half-duplex bus timing model.

The interesting assertions are the protocol arithmetic: if the frame sizes are right, the durations
fall out at the 0.26 ms and 1.4 ms the plan quotes, and the read skew is a consequence rather than a
tuned constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.robot.bus_emulator import BusConfig, BusEmulator


def test_frame_sizes_match_the_feetech_protocol():
    bus = BusEmulator()
    assert bus.sync_write_bytes() == 26          # 8 overhead + 6 * (id + 2 data)
    assert bus.sync_read_request_bytes() == 14   # 8 overhead + 6 ids
    assert bus.sync_read_reply_bytes() == 21     # 6 overhead + 15 payload
    assert bus.sync_read_request_bytes() + 6 * bus.sync_read_reply_bytes() == 140


def test_wire_times_are_the_numbers_the_plan_quotes():
    bus = BusEmulator()
    assert bus.sync_write_duration() == pytest.approx(0.26e-3, rel=1e-9)
    assert bus.sync_read_duration() == pytest.approx(1.4e-3, rel=1e-9)


def test_halving_the_baud_rate_doubles_every_duration():
    fast = BusEmulator()
    slow = BusEmulator(BusConfig(baud=500_000))
    assert slow.sync_write_duration() == pytest.approx(2 * fast.sync_write_duration())
    assert slow.sync_read_duration() == pytest.approx(2 * fast.sync_read_duration())


def test_read_time_grows_with_the_number_of_servos():
    one = BusEmulator(BusConfig(n_servos=1)).sync_read_duration()
    six = BusEmulator(BusConfig(n_servos=6)).sync_read_duration()
    assert six > one
    # only the replies scale; the request grows by one byte per servo
    assert six - one == pytest.approx(5 * (21 + 1) * 10 / 1_000_000)


def test_one_tick_fits_comfortably_inside_a_50_hz_period():
    bus = BusEmulator()
    assert bus.fits_in(0.02)
    assert bus.tick_duration() < 0.02 / 4
    assert not bus.fits_in(0.002)


def test_timestamps_are_ordered_and_the_write_comes_first():
    timing = BusEmulator(BusConfig(packet_loss=0.0)).tick(1.0)
    assert timing.t_tick <= timing.t_write_start < timing.t_write_end
    assert timing.t_write_end <= timing.t_read_start < timing.t_read_end
    assert np.all(np.diff(timing.reply_times) > 0)
    assert np.all(timing.sample_times < timing.reply_times)
    assert timing.reply_times[-1] == pytest.approx(timing.t_read_end)
    assert timing.command_latency_s == pytest.approx(0.26e-3)


def test_every_servo_is_sampled_at_a_different_time():
    cfg = BusConfig(packet_loss=0.0)
    timing = BusEmulator(cfg).tick(0.0)
    gaps = np.diff(timing.sample_times)
    reply = BusEmulator(cfg).sync_read_reply_bytes() * 10 / cfg.baud
    assert np.allclose(gaps, reply + cfg.turnaround_s)
    assert timing.read_skew_s == pytest.approx(5 * (reply + cfg.turnaround_s))
    assert 0.5e-3 < timing.read_skew_s < 5e-3


def test_no_turnaround_means_the_wire_time_is_the_whole_story():
    cfg = BusConfig(turnaround_s=0.0, packet_loss=0.0)
    bus = BusEmulator(cfg)
    timing = bus.tick(0.0)
    assert timing.t_read_end - timing.t_read_start == pytest.approx(bus.sync_read_duration())
    assert timing.busy_s == pytest.approx(bus.sync_write_duration() + bus.sync_read_duration())


def test_packet_loss_happens_at_roughly_the_configured_rate():
    cfg = BusConfig(packet_loss=0.005)
    bus = BusEmulator(cfg, rng=np.random.default_rng(3))
    ticks = bus.schedule(20_000, 0.02)
    write_loss = np.mean([t.write_lost for t in ticks])
    assert write_loss == pytest.approx(cfg.packet_loss, abs=0.002)
    # a reply is lost either on its own or because the whole request was, so the per-reply rate is
    # about twice the per-packet rate
    reply_loss = np.mean([t.reply_lost.mean() for t in ticks])
    assert reply_loss == pytest.approx(2 * cfg.packet_loss, abs=0.003)


def test_a_lost_request_loses_every_reply_in_that_tick():
    bus = BusEmulator(BusConfig(packet_loss=0.5), rng=np.random.default_rng(0))
    ticks = bus.schedule(400, 0.02)
    mixed = [t for t in ticks if t.reply_lost.any() and not t.reply_lost.all()]
    allgone = [t for t in ticks if t.reply_lost.all()]
    assert allgone, "a lost request should wipe out a whole read"
    assert mixed, "individual replies should also be able to drop on their own"


def test_zero_loss_means_zero_loss():
    bus = BusEmulator(BusConfig(packet_loss=0.0), rng=np.random.default_rng(0))
    ticks = bus.schedule(200, 0.02)
    assert not any(t.write_lost for t in ticks)
    assert not any(t.reply_lost.any() for t in ticks)


def test_the_same_seed_gives_the_same_schedule():
    a = BusEmulator(BusConfig(period_jitter_s=0.002), rng=np.random.default_rng(42)).schedule(50, 0.02)
    b = BusEmulator(BusConfig(period_jitter_s=0.002), rng=np.random.default_rng(42)).schedule(50, 0.02)
    c = BusEmulator(BusConfig(period_jitter_s=0.002), rng=np.random.default_rng(43)).schedule(50, 0.02)
    assert [t.t_write_start for t in a] == [t.t_write_start for t in b]
    assert [list(t.reply_lost) for t in a] == [list(t.reply_lost) for t in b]
    assert [t.t_write_start for t in a] != [t.t_write_start for t in c]


def test_schedule_lays_ticks_out_at_the_requested_period():
    bus = BusEmulator(BusConfig(packet_loss=0.0))
    ticks = bus.schedule(10, 0.02, t0=0.5)
    assert [t.t_tick for t in ticks] == pytest.approx(0.5 + np.arange(10) * 0.02)
    assert np.all(np.diff([t.t_read_end for t in ticks]) > 0)
    # the bus is idle most of the time at 50 Hz
    assert sum(t.busy_s for t in ticks) < 0.25 * 10 * 0.02


def test_period_jitter_moves_the_start_but_not_the_nominal_tick():
    cfg = BusConfig(period_jitter_s=0.002, packet_loss=0.0)
    bus = BusEmulator(cfg, rng=np.random.default_rng(1))
    ticks = bus.schedule(200, 0.02)
    offsets = np.array([t.t_write_start - t.t_tick for t in ticks])
    assert np.all(np.abs(offsets) <= cfg.period_jitter_s + 1e-12)
    assert offsets.std() > 0.0005
