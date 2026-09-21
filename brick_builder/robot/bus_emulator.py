"""Timing model of the half-duplex Feetech servo bus.  Pure Python, no I/O, no hardware.

The real arm hangs six STS3215 servos off one 1 Mbps half-duplex line.  Every control tick the host
sends one sync-write and one sync-read, and the six replies come back one after another because only
one device may drive the line at a time.  That is where the multi-rate env loop's latencies come
from, and it is why an observation is not a snapshot: servo 5 is sampled about 1 ms after servo 0.

Packet sizes follow the Feetech protocol (the same frame layout as Dynamixel 1.0):

===================  ===========================================  =====
Frame                Bytes                                        n=6
===================  ===========================================  =====
sync-write           ``8 + n * (1 + payload)``                     26
sync-read request    ``8 + n``                                     14
sync-read reply      ``6 + payload`` per servo                     21
===================  ===========================================  =====

At 1 Mbps with 8N1 framing a byte costs 10 us, so a 2-byte Goal_Position sync-write is 260 us and a
15-byte sync-read of addresses 56-70 is 140 + 6 * 210 = 1400 us on the wire.  Those are exactly the
0.26 ms and 1.4 ms the plan quotes, which is the check that the frame arithmetic here is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BITS_PER_BYTE = 10  # 8N1: start bit, eight data bits, stop bit


@dataclass(frozen=True)
class BusConfig:
    """Wire and protocol parameters.  Every field is a domain-randomization handle."""

    baud: int = 1_000_000
    n_servos: int = 6
    write_payload: int = 2      # Goal_Position, addresses 42-43
    read_payload: int = 15      # addresses 56-70: position, speed, load, voltage, temp, ..., current
    turnaround_s: float = 0.0002
    """Direction-switch dead time on a half-duplex line, once before each reply."""
    servo_latency_s: float = 0.0
    """Extra time a servo takes to start answering, on top of the turnaround."""
    packet_loss: float = 0.005
    """Probability that any single packet (the write, the request, or one reply) is lost."""
    period_jitter_s: float = 0.0
    """Uniform +- jitter on when the host actually starts a tick."""


@dataclass
class BusTiming:
    """When everything happened during one control tick.  All times are absolute seconds."""

    t_tick: float                 # when the host meant to start
    t_write_start: float
    t_write_end: float
    t_read_start: float
    t_read_end: float
    reply_times: np.ndarray       # when each servo's reply finished arriving
    sample_times: np.ndarray      # when each servo actually latched its registers
    write_lost: bool
    reply_lost: np.ndarray        # per servo
    busy_s: float                 # how much of the tick the bus was occupied

    @property
    def read_skew_s(self) -> float:
        """Spread between the first and last servo sample in this read."""
        return float(self.sample_times.max() - self.sample_times.min())

    @property
    def command_latency_s(self) -> float:
        """Host tick start to the moment the servos have the new goal."""
        return self.t_write_end - self.t_tick


class BusEmulator:
    """Hands out per-tick timestamps for the sync-write / sync-read pair.

    Nothing here touches MuJoCo: an env asks for a tick's timing and uses the numbers to decide when
    a command becomes visible and how stale each observation is.  Keeping it separate means the
    timing model can be unit-tested against the protocol arithmetic instead of against physics.
    """

    def __init__(self, cfg: BusConfig = BusConfig(), rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)

    # ------------------------------------------------------------------------------ wire maths ---

    @property
    def byte_time_s(self) -> float:
        return BITS_PER_BYTE / self.cfg.baud

    def sync_write_bytes(self) -> int:
        """Header (2) + id (1) + length (1) + instruction (1) + address (1) + data length (1)
        + n * (id + payload) + checksum (1)."""
        return 8 + self.cfg.n_servos * (1 + self.cfg.write_payload)

    def sync_read_request_bytes(self) -> int:
        return 8 + self.cfg.n_servos

    def sync_read_reply_bytes(self) -> int:
        """Header (2) + id (1) + length (1) + error (1) + payload + checksum (1)."""
        return 6 + self.cfg.read_payload

    def sync_write_duration(self) -> float:
        """Time on the wire for one sync-write [s] -- 260 us for six servos at 1 Mbps."""
        return self.sync_write_bytes() * self.byte_time_s

    def sync_read_duration(self) -> float:
        """Wire time for a sync-read, request plus all replies [s] -- 1.4 ms for six servos."""
        return (
            self.sync_read_request_bytes()
            + self.cfg.n_servos * self.sync_read_reply_bytes()
        ) * self.byte_time_s

    def tick_duration(self) -> float:
        """Worst-case bus occupancy of one control tick, turnarounds included [s]."""
        return (
            self.sync_write_duration()
            + self.sync_read_duration()
            + self.cfg.n_servos * (self.cfg.turnaround_s + self.cfg.servo_latency_s)
            + self.cfg.turnaround_s
        )

    # ----------------------------------------------------------------------------------- ticks ---

    def tick(self, t_tick: float) -> BusTiming:
        """Run one sync-write + sync-read starting at ``t_tick`` and report the timestamps.

        A servo latches its registers when it starts composing its reply, so ``sample_times`` sits
        one reply-length before ``reply_times``.  A lost reply leaves that servo's entry in place
        but flags it, so the caller can hold the previous reading -- which is what the real driver
        does rather than stalling the control loop.
        """
        cfg = self.cfg
        start = t_tick
        if cfg.period_jitter_s > 0.0:
            start += float(self.rng.uniform(-cfg.period_jitter_s, cfg.period_jitter_s))

        write_lost = bool(self.rng.random() < cfg.packet_loss)
        t_write_start = start
        t_write_end = t_write_start + self.sync_write_duration()

        t_read_start = t_write_end + cfg.turnaround_s
        t_cursor = t_read_start + self.sync_read_request_bytes() * self.byte_time_s

        reply_time = self.sync_read_reply_bytes() * self.byte_time_s
        reply_times = np.zeros(cfg.n_servos)
        sample_times = np.zeros(cfg.n_servos)
        reply_lost = np.zeros(cfg.n_servos, dtype=bool)
        request_lost = bool(self.rng.random() < cfg.packet_loss)
        for i in range(cfg.n_servos):
            t_cursor += cfg.turnaround_s + cfg.servo_latency_s
            sample_times[i] = t_cursor
            t_cursor += reply_time
            reply_times[i] = t_cursor
            reply_lost[i] = request_lost or bool(self.rng.random() < cfg.packet_loss)

        return BusTiming(
            t_tick=t_tick,
            t_write_start=t_write_start,
            t_write_end=t_write_end,
            t_read_start=t_read_start,
            t_read_end=t_cursor,
            reply_times=reply_times,
            sample_times=sample_times,
            write_lost=write_lost,
            reply_lost=reply_lost,
            busy_s=(t_write_end - t_write_start) + (t_cursor - t_read_start),
        )

    def schedule(self, n_ticks: int, period: float, t0: float = 0.0) -> list[BusTiming]:
        """``n_ticks`` consecutive control ticks at ``period`` seconds apart."""
        return [self.tick(t0 + k * period) for k in range(n_ticks)]

    def fits_in(self, period: float) -> bool:
        """Does one tick's traffic fit in a control period?  At 50 Hz there is plenty of room."""
        return self.tick_duration() <= period
