"""Scripted skills: small state machines that turn feedback into one tool target per tick.

Every primitive here obeys two rules that come straight from the control specification:

1. **Deployable signals only.**  A primitive may look at :class:`Feedback`, which is forward
   kinematics of the joint positions the servo bus *read*, the servos' own load registers, and
   the gripper angle we ourselves commanded.  Nothing in this file can see simulator truth: no
   stud depths, no brick pose, no contact forces.  That is what makes the scripted press a fair
   baseline for the learned one, and what lets the same logic run on the real arm.
2. **Portable arithmetic.**  Plain floats and small fixed-size arrays, no dictionaries and no
   allocation in the per-tick path, no numpy broadcasting tricks.  The press state machine below
   is meant to be transcribable into C99 more or less line by line.

A primitive produces a :class:`ToolTarget` -- the same six numbers the action space is defined
against (``pos``, ``tilt``, ``spin``, ``grip``, ``c_vert``, ``c_lat``).  In an environment this is
the *primitive* half of "executed target = primitive target + bounded residual", so a policy that
outputs zero reproduces the scripted skill exactly.

What the press actually does
----------------------------
:class:`PressScripted` descends straight down from wherever it was handed the tool, at 10 mm/s,
with the sideways torque limit turned down so the studs can centre themselves.  It is blind: it
never learns which way it is misaligned, only that it has stopped going down.  When the measured
descent stalls it lets go, reads the tool's height back, and -- if the brick is not in -- lifts
2 mm, skims outward along a spiral, and presses again the moment the tool drops into the
sockets.  "Let go before you judge" is the part that is easy to get wrong, and
:meth:`PressScripted._check` explains what it costs to skip.

The capture basin this search has to cover was measured, not assumed: with the default snap
parameters a straight press seats a 2x4 from up to 3 mm of lateral error and a 2x2 from 2 mm,
which is why the default spiral pitch is 2.5 mm rather than the 0.5 mm the spec sketch names --
see :class:`PressConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from brick_builder.robot.kinematics import ArmKinematics, euler_zyx

CONTROL_DT = 0.020
"""Seconds per control tick (50 Hz), the rate every primitive advances its clock at.

The same number as ``brick_builder.envs.base_env.CONTROL_DT``; it is repeated rather than imported so
that this package depends only on numpy and :mod:`brick_builder.robot`, which is what keeps it
portable to the firmware.  Every primitive takes ``dt`` as a constructor argument anyway.
"""

PHASES: tuple[str, ...] = (
    "idle", "descend", "rock", "check", "lift", "search", "seated", "stuck",
)
"""Every value :attr:`PressScripted.phase` can take, in the order they normally occur."""


@dataclass(frozen=True)
class ToolTarget:
    """One tick of scripted output: where the tool should be and how hard it may push.

    ``pos`` is the TCP position in the world frame; ``tilt`` and ``spin`` are the tool's pitch
    and yaw offsets from straight down (the convention ``ArmKinematics.ik`` is driven in);
    ``grip`` is a gripper joint angle; ``c_vert``/``c_lat`` are torque-limit fractions for the
    lifting joints and the sideways ones.
    """

    pos: np.ndarray
    tilt: float = 0.0
    spin: float = 0.0
    grip: float = 0.0
    c_vert: float = 1.0
    c_lat: float = 1.0


@dataclass(frozen=True)
class Feedback:
    """Everything a scripted skill is allowed to look at this tick.

    ``tcp_pos``/``tcp_tilt``/``tcp_spin`` come from forward kinematics of the *read* joint
    positions, so they lag and quantize exactly as much as the real encoders do; ``load`` is the
    servos' ``Present_Load`` register (commanded PWM duty in [-1, 1], not a torque); ``grip`` is
    our own last gripper command; ``tripped`` is the overload flag the servos latch.
    """

    tcp_pos: np.ndarray
    tcp_tilt: float = 0.0
    tcp_spin: float = 0.0
    load: np.ndarray | None = None
    grip: float = 0.0
    tripped: bool = False

    @property
    def peak_load(self) -> float:
        """Largest |load| over the joints, which is what "is the arm straining" means here."""
        if self.load is None:
            return 0.0
        return float(np.max(np.abs(self.load)))


def feedback_from_readings(kin: ArmKinematics, readings, grip: float = 0.0) -> Feedback:
    """Build a :class:`Feedback` from one :class:`~brick_builder.robot.servo_model.ServoReadings`.

    The forward kinematics uses the five arm joints only (the gripper does not move the TCP), and
    the tool's pitch/yaw are read off the resulting rotation with the same Euler convention the
    IK is driven in.
    """
    pos, rot = kin.fk(np.asarray(readings.position)[:5])
    spin, tilt, _ = euler_zyx(rot)
    return Feedback(
        tcp_pos=pos,
        tcp_tilt=float(tilt),
        tcp_spin=float(spin),
        load=np.asarray(readings.load, dtype=float),
        grip=float(grip),
        tripped=bool(np.any(readings.tripped)),
    )


# ------------------------------------------------------------------------------------ MoveTo ---


class MoveTo:
    """Straight-line Cartesian move to ``target`` at ``speed``, then a settle.

    The commanded point walks along the straight line from wherever the tool was when
    :meth:`reset` was called to ``target``, at a constant speed.  Once it arrives it stays there
    for up to ``settle_s`` seconds, and the move counts as done when the *measured* tool is within
    ``tol`` of the target or that settle time runs out -- a soft position servo never lands
    exactly, so waiting for zero error would never finish.
    """

    def __init__(
        self,
        target,
        speed: float = 0.05,
        *,
        tilt: float | None = None,
        spin: float | None = None,
        grip: float = 0.0,
        c_vert: float = 1.0,
        c_lat: float = 1.0,
        settle_s: float = 0.15,
        tol: float = 0.0015,
        dt: float = CONTROL_DT,
    ) -> None:
        self.target = np.asarray(target, dtype=float).copy()
        self.speed = float(speed)
        self.tilt = tilt
        self.spin = spin
        self.grip = float(grip)
        self.c_vert = float(c_vert)
        self.c_lat = float(c_lat)
        self.settle_s = float(settle_s)
        self.tol = float(tol)
        self.dt = float(dt)
        self.phase = "idle"
        self.done = False
        self._start = np.zeros(3)
        self._t = 0.0
        self._arrived_at: float | None = None

    def reset(self, fb: Feedback) -> None:
        self._start = np.asarray(fb.tcp_pos, dtype=float).copy()
        if self.tilt is None:
            self.tilt = float(fb.tcp_tilt)
        if self.spin is None:
            self.spin = float(fb.tcp_spin)
        self._t = 0.0
        self._arrived_at = None
        self.done = False
        self.phase = "move"

    def update(self, fb: Feedback) -> ToolTarget:
        delta = self.target - self._start
        distance = float(np.linalg.norm(delta))
        travelled = self.speed * self._t
        if distance <= 1e-12 or travelled >= distance:
            pos = self.target.copy()
            if self._arrived_at is None:
                self._arrived_at = self._t
                self.phase = "settle"
        else:
            pos = self._start + delta * (travelled / distance)

        if self._arrived_at is not None:
            error = float(np.linalg.norm(np.asarray(fb.tcp_pos) - self.target))
            if error <= self.tol or (self._t - self._arrived_at) >= self.settle_s:
                self.done = True
                self.phase = "done"

        self._t += self.dt
        return ToolTarget(
            pos=pos, tilt=float(self.tilt), spin=float(self.spin), grip=self.grip,
            c_vert=self.c_vert, c_lat=self.c_lat,
        )

    @property
    def elapsed(self) -> float:
        return self._t


# ----------------------------------------------------------------------------- PressScripted ---


@dataclass(frozen=True)
class PressConfig:
    """Tuning for :class:`PressScripted`.  Every default below was measured on this rig.

    - ``descend_speed`` 10 mm/s is the spec's number.
    - ``press_depth`` is how far past the seated height the *commanded* tool may be driven.  The
      arm is a position servo with no force loop, so commanded overshoot is the only throttle on
      press force: 10 mm of overshoot measures 14 N at the plate centre, which is inside the
      12-16 N the force map reports and comfortably seats a 2x4 (8 studs x 1.15 N = 9.2 N
      at the default clutch).  6 mm gives 9 N and takes twice as long to seat; 20 mm gives 25 N,
      which the real arm cannot deliver.
    - ``c_lat`` 0.45 is the sideways torque-limit fraction during the press: soft enough that a
      stud landing on a chamfer can shove the tool aside and centre itself, stiff enough that the
      arm still tracks the spiral.
    - ``spiral_pitch`` **deviates from the spec's 0.5 mm.**  The measured capture basin of a
      straight press is about 3 mm for a 2x4 and 2 mm for a 2x2, so a 0.5 mm pitch covers the
      same disc six times over: 100 mm of spiral path, 6.7 s (334 ticks) at ``spiral_speed``,
      which does not fit inside the 200-tick (4 s) horizon the spec gives the press environment,
      let alone leave time for the press that follows it.  2.5 mm still passes within 1.25 mm of
      every point of the disc -- well inside the basin -- in 20 mm of path and 1.4 s (69 ticks).
    - ``stall_time``/``stall_speed``: the tool must have descended less than 0.7 mm/s averaged
      over the last 0.4 s before the press believes it is stuck.  Averaging over a window rather
      than differencing two ticks matters: the encoder quantizes to about 0.1 mm at the TCP, so a
      single-tick difference is 5 mm/s of pure noise.
    """

    descend_speed: float = 0.010      # m/s
    press_depth: float = 0.010        # m of commanded overshoot past the seated height
    c_vert: float = 1.0
    c_lat: float = 0.45

    stall_speed: float = 0.0007       # m/s, averaged over stall_time
    stall_time: float = 0.40          # s
    stall_load: float = 0.10          # |load| above which a stall counts as "under load"

    check_time: float = 0.25          # s of holding the seated height before reading it back
    check_height: float = 0.0025      # m above the seated height that still counts as "in"

    lift: float = 0.002               # m, the unload hop before a search
    lift_time: float = 0.10           # s
    spiral_pitch: float = 0.0025      # m of radius gained per turn
    spiral_radius: float = 0.004      # m, largest search radius
    spiral_speed: float = 0.015       # m/s along the spiral
    skim_press: float = 0.003         # m of commanded overshoot while skimming the stud tops
    drop_detect: float = 0.0015       # m the tool must drop for the search to call it a find
    max_searches: int = 3

    saturated_load: float = 0.80
    saturated_time: float = 0.30      # s of saturated load before the press gives up
    max_time: float = 5.0             # s before the press gives up whatever else is happening

    rocking: bool = False
    rock_tilt: float = math.radians(4.0)
    rock_engage: float = 0.0015       # m of descent that counts as "one stud row is in"
    rock_level_time: float = 0.30     # s to bring the tool back to level


class PressScripted:
    """Press a held brick onto the plate, blind, from wherever the tool currently is.

    ``seat_z`` is the TCP height a seated brick puts the tool at: plate top, plus the target
    layer's height, plus how far the brick hangs below the TCP.  On the real arm that number
    comes from the taught grid and the known tool geometry, which is why a deployable primitive
    is allowed to have it.

    Phases, readable from :attr:`phase`:

    ``descend``  the 10 mm/s ramp, sideways-soft.  Ends at ``check`` when the descent stalls
                 under load.
    ``rock``     only when ``PressConfig.rocking``: the same ramp held at ``rock_tilt`` until one
                 stud row engages, then the tool is levelled out over ``rock_level_time``.
    ``check``    stop pushing, settle at the seated height, and read the height back.  This is
                 the only place the press decides whether the brick went in; see :meth:`_check`.
    ``lift``     a 2 mm hop to unload the brick before it is skimmed sideways.
    ``search``   an outward spiral at the skim height; ends the moment the tool drops into the
                 sockets, which puts it back in ``descend``.
    ``seated``   done, holding the pressed pose.
    ``stuck``    gave up: the load stayed saturated, or the searches ran out.  The tool holds
                 where it is rather than pushing harder.
    """

    def __init__(self, seat_z: float, cfg: PressConfig = PressConfig(), *,
                 grip: float = 0.0, dt: float = CONTROL_DT) -> None:
        self.seat_z = float(seat_z)
        self.cfg = cfg
        self.grip = float(grip)
        self.dt = float(dt)
        self.phase = "idle"
        self.done = False
        self.seated = False
        self.searches = 0
        self.press_time = 0.0

        self._window = max(int(round(cfg.stall_time / self.dt)), 2)
        self._zbuf = np.zeros(self._window)
        self._zfill = 0
        self._ref = np.zeros(3)
        self._spin = 0.0
        self._tilt = 0.0
        self._t = 0.0
        self._sat_for = 0.0
        self._skim_z = 0.0
        self._lift_xy = np.zeros(2)
        self._spiral_theta = 0.0
        self._spiral_centre = np.zeros(2)
        self._level_at: float | None = None
        self.descent_rate = 0.0

    # -- lifecycle ---------------------------------------------------------------------------

    def reset(self, fb: Feedback) -> None:
        """Latch the pose the press works from.  This is the whole of its alignment knowledge."""
        self._ref = np.asarray(fb.tcp_pos, dtype=float).copy()
        self._spiral_centre = self._ref[:2].copy()
        self._spin = float(fb.tcp_spin)
        self._tilt = self.cfg.rock_tilt if self.cfg.rocking else 0.0
        self._zbuf[:] = float(fb.tcp_pos[2])
        self._zfill = 0
        self._t = 0.0
        self._sat_for = 0.0
        self._spiral_theta = 0.0
        self._level_at = None
        self.descent_rate = 0.0
        self.searches = 0
        self.done = False
        self.seated = False
        self.press_time = 0.0
        self.phase = "rock" if self.cfg.rocking else "descend"

    def _enter(self, phase: str, ref: np.ndarray | None = None) -> None:
        """Switch phase, restarting the phase clock and (optionally) re-latching the ramp origin.

        Re-latching matters because the descent ramp is written as ``ref_z - speed * t``: coming
        back from a search with ``t`` reset but ``ref_z`` still at the pose the press started
        from would jump the command back up by however far the tool has already gone down.
        """
        self.phase = phase
        self._t = 0.0
        if ref is not None:
            self._ref = np.asarray(ref, dtype=float).copy()

    # -- per tick ----------------------------------------------------------------------------

    def update(self, fb: Feedback) -> ToolTarget:
        cfg = self.cfg
        z = float(fb.tcp_pos[2])

        # Descent rate, averaged over the stall window (see PressConfig on why not two ticks).
        oldest = self._zbuf[0]
        self._zbuf[:-1] = self._zbuf[1:]
        self._zbuf[-1] = z
        self._zfill = min(self._zfill + 1, self._window)
        self.descent_rate = (oldest - z) / (self._window * self.dt)

        peak = fb.peak_load
        self._sat_for = self._sat_for + self.dt if peak >= cfg.saturated_load else 0.0
        if not self.done:
            self.press_time += self.dt

        if self.phase in ("descend", "rock"):
            pos, tilt = self._descend(z, peak >= cfg.stall_load)
        elif self.phase == "check":
            pos, tilt = self._check(z)
        elif self.phase == "lift":
            pos, tilt = self._lift()
        elif self.phase == "search":
            pos, tilt = self._search(z)
        else:  # "seated" or "stuck": hold the last commanded pose
            pos, tilt = self._ref.copy(), self._tilt

        self._t += self.dt
        return ToolTarget(
            pos=pos, tilt=float(tilt), spin=self._spin, grip=self.grip,
            c_vert=cfg.c_vert, c_lat=cfg.c_lat,
        )

    # -- phases ------------------------------------------------------------------------------

    def _descend(self, z: float, loaded: bool) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        floor = self.seat_z - cfg.press_depth
        cmd_z = max(self._ref[2] - cfg.descend_speed * self._t, floor)
        pos = np.array([self._ref[0], self._ref[1], cmd_z])
        tilt = self._tilt

        if self.phase == "rock":
            # Seat one stud row first: hold the tilt until the tool has actually gone down, then
            # level out over `rock_level_time` while the ramp keeps pressing.
            if self._level_at is None and (self._ref[2] - z >= cfg.rock_engage or self._t >= 1.5):
                self._level_at = self._t
            if self._level_at is not None:
                frac = min((self._t - self._level_at) / cfg.rock_level_time, 1.0)
                tilt = cfg.rock_tilt * (1.0 - frac)
                self._tilt = tilt
                if frac >= 1.0:
                    self._enter("descend", pos)
                    return pos, tilt

        if self._sat_for >= cfg.saturated_time or self.press_time >= cfg.max_time:
            # The saturated-load exit is the spec's; `max_time` is the backstop it needs. A brick
            # jammed a hair off centre keeps creeping down a few tenths of a millimetre a second
            # under 10 N -- never fast enough to call progress, never slow enough to call a stall
            # -- and without a clock the press would lean on it for the whole episode.
            self.done = True
            self._enter("stuck", pos)
            return pos, tilt

        # A stall only means anything once the ramp has had a full window to move the tool. What
        # it means, though, cannot be decided here: see `_check`.
        stalled = (
            self._zfill >= self._window
            and self._t >= cfg.stall_time + 0.1
            and self.descent_rate < cfg.stall_speed
            and loaded
        )
        if stalled:
            self._enter("check", pos)
        return pos, tilt

    def _check(self, z: float) -> tuple[np.ndarray, float]:
        """Stop pushing, let the tool settle at the seated height, and read it back.

        This phase exists because of a measurement.  While the press is pushing, the tool's own
        height says nothing about whether the brick went in: at 10 N this arm deflects about
        2 mm through its gearboxes and backlash joints, and forward kinematics of the *read*
        encoder positions -- all a real robot has -- cannot see any of that.  Measured over 24
        presses, a jammed brick and a seated one both left the read height somewhere between
        -1.5 and +3.2 mm of the seated height; the two populations overlap completely, so no
        threshold on the pressing tool can separate them.

        Let go first and they separate cleanly.  Commanding the seated height with no overshoot
        unloads the arm in a couple of hundred milliseconds, and what is left is the brick: a
        seated 2x4 holds the tool 2.0-2.3 mm high (almost all of it the soft weld's own stretch),
        one resting on the stud tops holds it 4.6-6.5 mm high.  ``check_height`` sits in that gap.

        So the press ends here, always: a stall is not a failure, it is the question, and this is
        where it gets answered.
        """
        cfg = self.cfg
        pos = np.array([self._ref[0], self._ref[1], self.seat_z])
        if self._t < cfg.check_time:
            return pos, self._tilt
        if z - self.seat_z <= cfg.check_height:
            self.seated = True
            self.done = True
            self._enter("seated", pos)
        elif self.searches >= cfg.max_searches:
            self.done = True
            self._enter("stuck", pos)
        else:
            self._skim_z = z
            self._lift_xy = self._ref[:2].copy()
            self._enter("lift")
        return pos, self._tilt

    def _lift(self) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        pos = np.array([self._lift_xy[0], self._lift_xy[1], self._skim_z + cfg.lift])
        if self._t >= cfg.lift_time:
            self.searches += 1
            self._spiral_theta = 0.0
            self._enter("search")
        return pos, self._tilt

    def _search(self, z: float) -> tuple[np.ndarray, float]:
        """Skim outward along a spiral until the tool drops into the sockets.

        The spiral is walked at a constant speed along its own arc, not at a constant angular
        rate, so the outer turns are not swept faster than the inner ones: with
        ``r = b * theta`` and ``b = pitch / 2 pi``, ``ds = sqrt(r^2 + b^2) d(theta)``.

        While skimming, the tool is commanded ``skim_press`` below the height the brick rests at,
        so it stays pressed against the stud tops; the brick cannot follow, so the measured tool
        barely moves.  When the sockets line up, the brick drops and the tool goes with it, which
        is the ``drop_detect`` test.  ``skim_press`` is deliberately larger than ``drop_detect``,
        or a perfectly free descent could never cross the threshold.
        """
        cfg = self.cfg
        b = cfg.spiral_pitch / (2.0 * math.pi)
        r = b * self._spiral_theta
        self._spiral_theta += cfg.spiral_speed * self.dt / math.sqrt(r * r + b * b)
        r = b * self._spiral_theta
        x = self._spiral_centre[0] + r * math.cos(self._spiral_theta)
        y = self._spiral_centre[1] + r * math.sin(self._spiral_theta)
        pos = np.array([x, y, self._skim_z - cfg.skim_press])

        if z <= self._skim_z - cfg.drop_detect:
            self._enter("descend", np.array([x, y, z]))      # found a socket: press from here
        elif r >= cfg.spiral_radius:
            self.done = True                                  # swept the whole disc, nothing there
            self._enter("stuck", pos)
        return pos, self._tilt


# ----------------------------------------------------------------------------------- figures ---


def make_control_figures(out_dir="docs/media/control", seed: int = 0) -> dict:
    """The two figures this module owes: the pattern the search walks, and one real
    episode that stalls and recovers through it.  The ``scripts/make_media.py`` entry point.

    matplotlib and the simulator are imported inside the function so that importing this module
    stays free of both.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.envs.press_env import PressEnv, scripted_episode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    # -- 1. the search pattern itself, straight out of the state machine -------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), dpi=150)
    for ax, pitch in zip(axes, (PressConfig().spiral_pitch, 0.0005)):
        xs, ys = _spiral_path(PressConfig(spiral_pitch=pitch))
        ax.plot(xs, ys, linewidth=0.9, color="#1f6feb")
        ax.add_patch(plt.Circle((0, 0), 3.0, fill=False, color="#d1242f", linestyle="--"))
        ax.set_aspect("equal")
        ax.set_xlabel("x offset [mm]")
        ax.set_ylabel("y offset [mm]")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"pitch {pitch * 1e3:.1f} mm: {len(xs)} ticks = {len(xs) * CONTROL_DT:.1f} s",
                     fontsize=9)
    fig.suptitle(
        "Spiral search, against the measured 3 mm capture basin of a 2x4 (dashed).\n"
        "The spec's 0.5 mm pitch covers the same disc six times over and cannot finish "
        "inside a 4 s episode.",
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / "spiral_search.png"
    fig.savefig(path)
    plt.close(fig)
    out["spiral_search.png"] = str(path)

    # -- 2. a real episode that stalls and recovers ----------------------------------------
    # Look for an episode that actually uses the search, and prefer one where it works: a figure
    # of the recovery mechanism should show the mechanism. Falls back to whatever the first seed
    # gave if no seed in range searches at all.
    env = PressEnv(level=2, misalign=(0.003, 0.0), dr=False, seed=seed)
    try:
        trace = scripted_episode(env, seed=seed)
        for candidate in range(seed, seed + 8):
            other = scripted_episode(env, seed=candidate)
            if other["searches"] >= 1 and other["success"]:
                trace = other
                break
            if other["searches"] >= 1 and trace["searches"] == 0:
                trace = other
    finally:
        env.close()
    t = np.asarray(trace["t"])
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), dpi=150, sharex=True)
    axes[0].plot(t, 1e3 * np.asarray(trace["cmd_z"]), label="commanded z", color="#8250df")
    axes[0].plot(t, 1e3 * np.asarray(trace["tcp_z"]), label="measured z", color="#1f6feb")
    axes[0].axhline(1e3 * trace["seat_z"], color="#d1242f", linestyle="--", label="seated height")
    axes[0].set_ylabel("TCP height [mm]")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    xy = 1e3 * (np.asarray(trace["cmd_xy"]) - np.asarray(trace["cmd_xy"])[0])
    axes[1].plot(t, xy[:, 0], label="commanded dx", color="#1f6feb")
    axes[1].plot(t, xy[:, 1], label="commanded dy", color="#bf8700")
    axes[1].set_ylabel("lateral offset [mm]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, alpha=0.3)
    for ax in axes:
        shade_phases(ax, t, trace["phase"])
    fig.suptitle(
        f"Scripted press, 2x4, 3 mm lateral error: "
        f"{'seated' if trace['success'] else 'did not seat'}, {trace['searches']} search(es)\n"
        "shading: orange = check, yellow = lift, blue = search, green = seated, red = gave up",
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / "press_recovery.png"
    fig.savefig(path)
    plt.close(fig)
    out["press_recovery.png"] = str(path)
    return out


def _spiral_path(cfg: PressConfig) -> tuple[list[float], list[float]]:
    """The xy path (in mm) the search phase walks for one config, with no simulator involved."""
    press = PressScripted(0.0, cfg)
    press.reset(Feedback(tcp_pos=np.zeros(3), load=np.zeros(6)))
    press._skim_z = 0.0
    press._enter("search")
    xs: list[float] = []
    ys: list[float] = []
    # A feedback that never drops: the search then runs until it has swept the whole disc.
    never = Feedback(tcp_pos=np.zeros(3), load=np.zeros(6))
    while press.phase == "search" and len(xs) < 100000:
        target = press.update(never)
        xs.append(float(target.pos[0]) * 1e3)
        ys.append(float(target.pos[1]) * 1e3)
    return xs, ys


PHASE_COLOURS: dict[str, str] = {
    "rock": "#fff1e5", "check": "#ffe8cc", "lift": "#fff8c5", "search": "#ddf4ff",
    "seated": "#dafbe1", "stuck": "#ffebe9",
}
"""Background colour per press phase, shared by every figure that draws a press over time."""


def shade_phases(ax, t, phases) -> None:
    """Colour a time axis' background by the press's phase, one span per run of equal phases."""
    start = 0
    n = len(phases)
    for k in range(1, n + 1):
        if k == n or phases[k] != phases[start]:
            colour = PHASE_COLOURS.get(phases[start])
            if colour is not None:
                ax.axvspan(t[start], t[min(k, n - 1)], color=colour, zorder=0)
            start = k
