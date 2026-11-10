"""What the real STS3215 bus gives you and takes from you, emulated at the control rate.

A position command does not arrive as a number the physics engine reads that instant.  It is
quantized to encoder ticks, ignored if it moved less than the dead zone, put on a wire that takes a
few milliseconds, then chased by a firmware target that can only slew at about 5.2 rad/s.  What
comes back is quantized too, sampled a few hundred microseconds apart per servo, and ``Present_Load``
is a PWM duty cycle, not a torque.  Every one of those is a sim-to-real gap, and every field of
:class:`ServoConfig` is a domain-randomization handle for it.

Where the encoder sits
----------------------
On the output shaft, after the gearbox.  The STS3215's 12-bit magnetic encoder is mounted past the
1:345 reduction, which is the only way one absolute 0-4095 reading can cover one *output* turn, and
which independent bench tests state directly.  So ``Present_Position`` is the angle of the link, not
of the motor, and the firmware's proportional law closes on the link angle too.  In this model the
link angle is the driven joint plus its free-play partner (``<joint>_backlash``, see
:func:`~brick_builder.robot.patch_so101._add_backlash`), while the motor torque is still applied at the
driven joint: torque reaches the link only through the play, but the measurement does not.

Two consequences the rest of the project depends on.  Gear play is *not* a steady-state placement
error the robot cannot see -- the loop drives the output shaft, so it takes the slack up itself.
What play costs is dead time (the motor crosses the slack before any torque arrives at the link) and
freedom inside the firmware's own dead band, where no correction is commanded at all
(:attr:`ServoConfig.error_deadband_ticks`, the CW/CCW dead-band registers 26/27).

Register semantics come from design report 02 section 0: P/I/D at addresses 21-23 (EPROM, so the
gain only changes at skill boundaries), CW/CCW dead band at 26-27 (EPROM), ``Torque_Limit`` at 48
(RAM, safe to write every tick), ``Present_Position`` 56, ``Present_Speed`` 58, ``Present_Load`` 60,
``Present_Current`` 69.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from brick_builder.robot.patch_so101 import (
    ARM_JOINTS,
    BAM_ERROR_GAIN,
    BAM_ERROR_GAIN_RATIO,
    BAM_KT,
    BAM_MAX_DUTY,
    BAM_MAX_VELOCITY,
    BAM_R,
    NOMINAL_VIN,
    ArmConfig,
    build_arm_spec,
    servo_kp,
)


STS3215_ERROR_DEADBAND_TICKS = 2.0
"""Firmware position dead band supported by the bench measurements [encoder counts].

Registers 26 (CW) and 27 (CCW) hold it and nobody in this project has read them off a real servo
yet, so the number is bracketed rather than known.  The bench report that measured this part says
"10 counts", but its own unloaded repeatability of about +-2 counts rules that out: a servo that
stopped correcting anywhere inside +-10 counts could not return to the same count twice.  Two counts
is the largest value consistent with both, so it is the one to randomize around and the one a
teammate should compare against when they finally read registers 26/27 (guide 03 says how).
"""

ERROR_DEADBAND_TICKS_RANGE = (0.0, 10.0)
"""Randomization range for :attr:`ServoConfig.error_deadband_ticks`: zero to the bench report's
own figure, which is also the range the command dead zone already uses."""


@dataclass
class ServoConfig:
    """Servo-side behaviour.  Defaults are the measured/centre values; comments give the DR range."""

    ticks_per_rev: int = 4096
    deadband_ticks: int = 0          # command dead zone; randomize 0-10 (0-0.9 deg)
    error_deadband_ticks: float = 0.0
    """Firmware position dead band (registers 26/27) in encoder counts: no correction at all while
    the position error is inside it.  Randomize 0-10; :data:`STS3215_ERROR_DEADBAND_TICKS` is the
    value the bench measurements support, and the default is 0 for the same reason
    ``deadband_ticks`` is -- nominal means the identified law with nothing added to it."""
    rate_limit: float = BAM_MAX_VELOCITY   # firmware target slew [rad/s]; randomize 4-6
    action_delay_s: float = 0.008    # mean command delay; randomize with the jitter below
    action_delay_jitter_s: float = 0.004   # +-, so the delay spans the measured 4-12 ms
    read_skew_s: float = 0.0005      # per-servo sampling skew inside one sync-read
    load_lsb: float = 0.001          # Present_Load is duty in 0.1 % steps, NOT torque
    current_lsb: float = 0.0065      # Present_Current, 6.5 mA per LSB
    overload_duty: float = 0.95
    overload_time_s: float = 1.5
    overload_derate: float = 0.2
    read_noise_ticks: float = 0.0    # extra position read noise, in encoder ticks
    vin: float = 7.4                 # supply the duty estimate assumes; randomize 6.8-7.6
    encoder_on_output: bool = True
    """Where the encoder is.  True is the hardware: after the gearbox, on the output shaft.

    False puts it on the motor side, which is what this project modelled until the correction and
    is wrong about the STS3215.  It is kept because the difference between the two is the whole
    point of the correction and the only way to measure it is to run both, and because with no
    free-play joints in the model the two are the same thing, which is the regression guard.  Do
    not run an experiment with it False.
    """


@dataclass
class ServoReadings:
    """One sync-read.  ``load`` is duty cycle in [-1, 1]; ``current`` is amps; ``tripped`` is bool.

    ``position`` and ``speed`` are output-shaft quantities -- driven joint plus free-play partner --
    because that is where the encoder is.
    """

    position: np.ndarray
    speed: np.ndarray
    load: np.ndarray
    current: np.ndarray
    tripped: np.ndarray


class ServoBus:
    """Sits between a policy and ``data.ctrl`` and makes the arm behave like six serial servos.

    Call :meth:`write` once per control tick, :meth:`substep` once per physics step, and
    :meth:`read` whenever the policy wants observations.  The bus owns ``data.ctrl`` and
    ``model.actuator_forcerange`` for its joints; nothing else should write them.

    ``data.ctrl`` is a *motor-side* number, not the firmware's target.  MuJoCo's position actuator
    computes ``kp * (ctrl - qpos_driven)``, and the error the real firmware works on is
    ``target - (qpos_driven + qpos_play)``, so the bus writes ``target - qpos_play`` and the two
    agree.  With no free-play partner the two are the same number and the write is unchanged.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joint_names=ARM_JOINTS,
        cfg: ServoConfig = ServoConfig(),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.names = tuple(arm_joint_names)
        self.n = len(self.names)

        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        )
        if np.any(self.joint_ids < 0):
            missing = [n for n, i in zip(self.names, self.joint_ids) if i < 0]
            raise ValueError(f"joints not in the model: {missing}")
        self.act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.names]
        )
        if np.any(self.act_ids < 0):
            missing = [n for n, i in zip(self.names, self.act_ids) if i < 0]
            raise ValueError(f"actuators not in the model: {missing}")
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.dof_adr = model.jnt_dofadr[self.joint_ids]

        # The free-play partners, when the model was built with them.  Either every driven joint
        # has one or none does (`_add_backlash` adds them as a set), so one flag covers the lot.
        play_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_backlash") for n in self.names
        ]
        # One flag drives every path that needs the output-shaft angle -- the reading, the
        # firmware's error, the duty estimate and the power-on seed -- so `encoder_on_output=False`
        # falls all the way back to the motor-side model in one place.
        self.has_play = cfg.encoder_on_output and all(j >= 0 for j in play_ids)
        if self.has_play:
            self.play_qpos_adr = model.jnt_qposadr[np.array(play_ids)]
            self.play_dof_adr = model.jnt_dofadr[np.array(play_ids)]
        else:
            self.play_qpos_adr = np.zeros(0, dtype=np.intp)
            self.play_dof_adr = np.zeros(0, dtype=np.intp)

        self.tick = 2.0 * np.pi / cfg.ticks_per_rev
        self.base_forcerange = model.actuator_forcerange[self.act_ids].copy()
        self.kp = model.actuator_gainprm[self.act_ids, 0].copy()
        # Torque the motor would make at duty 1.0 -- the denominator that turns a commanded torque
        # back into the duty cycle the firmware reports as Present_Load.
        self.torque_per_duty = BAM_KT * cfg.vin / BAM_R

        self._queue: deque = deque()
        self._t = 0.0
        self._skew_index = np.arange(self.n, dtype=float)
        self._forcerange_dirty = True
        self._overload_running = False
        # Plain Python copies of the things the per-physics-step path reads, and the model's
        # timestep, which never changes after a compile.  `substep` runs ten times per control
        # tick over six numbers; see `_update_overload` for why that is worth doing by hand.
        self._timestep = float(model.opt.timestep)
        # Goal positions are clipped to what the joint can actually reach.  That is the control
        # range, except that `_add_backlash` widens the control range to leave room for the
        # encoder-side offset `_apply` writes, so the joint's own range is the tighter of the two
        # and the one a Goal_Position is worth clipping to.  Both are fixed at compile time.
        limited = model.jnt_limited[self.joint_ids].astype(bool)
        self._ctrl_lo = np.where(
            limited,
            np.maximum(model.actuator_ctrlrange[self.act_ids, 0], model.jnt_range[self.joint_ids, 0]),
            model.actuator_ctrlrange[self.act_ids, 0],
        )
        self._ctrl_hi = np.where(
            limited,
            np.minimum(model.actuator_ctrlrange[self.act_ids, 1], model.jnt_range[self.joint_ids, 1]),
            model.actuator_ctrlrange[self.act_ids, 1],
        )
        self._act_list = [int(a) for a in self.act_ids]
        self._qpos_list = [int(a) for a in self.qpos_adr]
        self._play_list = [int(a) for a in self.play_qpos_adr]
        self._kp_list = [float(k) for k in self.kp]
        self.reset()

    # ------------------------------------------------------------------------------- lifecycle ---

    def reset(self) -> None:
        """Drop queued commands and re-seed the firmware target from where the arm actually is.

        This is what a servo does at power-on, and it is what an episode reset needs so the first
        step does not chase a target left over from the previous episode.  "Where the arm actually
        is" means what the encoder reads, i.e. the output shaft.
        """
        q = self._encoder_angles()
        self._queue.clear()
        self._t = 0.0
        self._goal = q.copy()          # latest command that has arrived over the wire
        self._accepted = q.copy()      # latest command that got past the dead band
        self._target = q.copy()        # firmware's internal, rate-limited target
        self._duty = np.zeros(self.n)
        self._overload_timer = np.zeros(self.n)
        self._overload_running = False
        self.tripped = np.zeros(self.n, dtype=bool)
        self._torque_scale = np.ones(self.n)
        self._forcerange_dirty = True
        self._apply()

    def reset_overload(self) -> None:
        """Clear a latched overload trip, the way power-cycling torque on the real servo does."""
        self.tripped[:] = False
        self._overload_timer[:] = 0.0
        self._overload_running = False
        self._forcerange_dirty = True
        self._apply_forcerange()

    # --------------------------------------------------------------------------------- writing ---

    def write(self, goal_rad: np.ndarray, torque_limit: np.ndarray | None = None) -> None:
        """Send one sync-write: goal positions, and optionally a per-joint torque limit in [0, 1].

        The goal is clipped to the joint's control range, quantized to encoder ticks, dropped if it
        did not move further than the dead band, and queued behind the wire delay.  ``torque_limit``
        scales the actuator's forcerange and takes effect immediately, which is how the real
        ``Torque_Limit`` register (address 48, RAM) behaves.
        """
        goal = np.asarray(goal_rad, dtype=float).reshape(self.n)
        goal = np.clip(goal, self._ctrl_lo, self._ctrl_hi)
        goal = np.round(goal / self.tick) * self.tick

        if self.cfg.deadband_ticks > 0:
            moved = np.abs(goal - self._accepted) >= self.cfg.deadband_ticks * self.tick
            goal = np.where(moved, goal, self._accepted)
        self._accepted = goal.copy()

        jitter = self.cfg.action_delay_jitter_s
        delay = self.cfg.action_delay_s
        if jitter > 0.0:
            delay += float(self.rng.uniform(-jitter, jitter))
        self._queue.append((self._t + max(delay, 0.0), goal))

        if torque_limit is None:
            self._torque_scale = np.ones(self.n)
        else:
            self._torque_scale = np.clip(
                np.asarray(torque_limit, dtype=float).reshape(self.n), 0.0, 1.0
            )
        self._forcerange_dirty = True
        self._apply()

    def substep(self) -> None:
        """Advance the servo one physics step: deliver the wire, slew the target, time the overload.

        Call this immediately before ``mj_step``.
        """
        dt = self._timestep
        self._t += dt
        self._apply()
        self._update_overload(dt)

    # -------------------------------------------------------------------------------- reading ---

    def read(self) -> ServoReadings:
        """One sync-read of addresses 56-70, quantized, skewed and (optionally) noisy.

        Servo *i* is sampled ``i * read_skew_s`` after servo 0, which we model by walking its
        position back along its current velocity: the simulator cannot be rewound, and to first
        order that is the same thing.
        """
        cfg = self.cfg
        q = self._encoder_angles()
        dq = self.data.qvel[self.dof_adr].copy()
        if self.has_play:
            dq += self.data.qvel[self.play_dof_adr]

        skew = self._skew_index * cfg.read_skew_s
        pos = q - dq * skew
        if cfg.read_noise_ticks > 0.0:
            pos = pos + self.rng.normal(0.0, cfg.read_noise_ticks * self.tick, self.n)
        pos = np.round(pos / self.tick) * self.tick
        speed = np.round(dq / self.tick) * self.tick
        load = np.round(self._duty / cfg.load_lsb) * cfg.load_lsb
        amps = (cfg.vin * self._duty - BAM_KT * dq) / BAM_R
        current = np.round(amps / cfg.current_lsb) * cfg.current_lsb
        return ServoReadings(
            position=pos,
            speed=speed,
            load=load,
            current=current,
            tripped=self.tripped.copy(),
        )

    def _encoder_angles(self) -> np.ndarray:
        """Output-shaft angle of each servo [rad]: driven joint plus its free-play partner.

        This is ``Present_Position`` before quantization, and it is the angle the firmware's
        proportional law is closed on.  With no partner joints it is the driven angle itself, bit
        for bit.
        """
        q = self.data.qpos[self.qpos_adr].copy()
        if self.has_play:
            q += self.data.qpos[self.play_qpos_adr]
        return q

    @property
    def target(self) -> np.ndarray:
        """The firmware's internal rate-limited target, in output-shaft (encoder) angle.

        ``data.ctrl`` holds this minus the free-play partner's angle, which is the motor-side
        command that makes MuJoCo's position actuator produce the firmware's own error term.
        """
        return self._target.copy()

    @property
    def duty(self) -> np.ndarray:
        """Commanded duty cycle, clipped at +-0.97.  This is what ``Present_Load`` reports."""
        return self._duty.copy()

    # ---------------------------------------------------------------------------------- guts ---

    def _apply(self) -> None:
        while self._queue and self._queue[0][0] <= self._t + 1e-12:
            _, goal = self._queue.popleft()
            self._goal = goal
        step = self.cfg.rate_limit * self._timestep
        deadband = self.cfg.error_deadband_ticks * self.tick
        # `np.clip(goal, target - step, target + step)`, one joint at a time and in place. Same
        # comparisons, same numbers; see `_update_overload` for why the loop is the fast version
        # at this size.
        target, goal, ctrl = self._target, self._goal, self.data.ctrl
        qpos, play_list = self.data.qpos, self._play_list
        for i, act in enumerate(self._act_list):
            lo = target[i] - step
            hi = target[i] + step
            value = goal[i]
            if value > hi:
                value = hi
            elif value < lo:
                value = lo
            target[i] = value
            if play_list:
                # The encoder is on the output shaft, so the firmware's error is against the link
                # angle.  Subtracting the play here leaves the actuator's own `ctrl - qpos_driven`
                # equal to that error while the torque still enters at the driven joint.
                value -= qpos[play_list[i]]
            if deadband > 0.0:
                driven = qpos[self._qpos_list[i]]
                error = value - driven
                if -deadband <= error <= deadband:
                    # Inside registers 26/27 the firmware commands nothing at all, which is what
                    # lets the output shaft drift about inside the gear slack.
                    value = driven
            ctrl[act] = value
        self._apply_forcerange()

    def _apply_forcerange(self) -> None:
        """Write the actuator force limits, but only when they have actually moved.

        The limits are a function of the commanded ``Torque_Limit`` and the latched overload
        state, neither of which can change inside a control tick's ten physics steps -- only
        :meth:`write`, :meth:`reset`, :meth:`reset_overload` and a fresh trip move them, and each
        of those raises the flag. Writing them anyway ten times a tick was pure cost.
        """
        if not self._forcerange_dirty:
            return
        scale = self._torque_scale * np.where(self.tripped, self.cfg.overload_derate, 1.0)
        self.model.actuator_forcerange[self.act_ids] = (
            self.base_forcerange * scale[:, None]
        )
        self._forcerange_dirty = False

    def _update_overload(self, dt: float) -> None:
        """Duty over 0.95 for 1.5 s latches the servo into its derated state (design report 01).

        Written as a loop over six joints rather than six-element numpy expressions, which is the
        one place in this project where that is the faster shape: a numpy call costs about a
        microsecond of its own before it touches any data, and this runs ten times per control
        tick.  Measured on this machine, the duty pass costs 10.7 us vectorised and 3.0 us as
        this loop.  The arithmetic is the same operations in the same order, so the duty comes
        out bit for bit identical; the loop only decides *whether* the per-joint overload timers
        need touching, which is the part that costs the other arrays.
        """
        max_duty = 0.0
        duty, target, qpos = self._duty, self._target, self.data.qpos
        kp_list, tpd = self._kp_list, self.torque_per_duty
        play_list = self._play_list
        deadband = self.cfg.error_deadband_ticks * self.tick
        for i, adr in enumerate(self._qpos_list):
            error = target[i] - qpos[adr]
            if play_list:
                error -= qpos[play_list[i]]
            if deadband > 0.0 and -deadband <= error <= deadband:
                duty[i] = 0.0
                continue
            value = error * kp_list[i] / tpd
            if value > BAM_MAX_DUTY:
                value = BAM_MAX_DUTY
            elif value < -BAM_MAX_DUTY:
                value = -BAM_MAX_DUTY
            duty[i] = value
            magnitude = -value if value < 0.0 else value
            if magnitude > max_duty:
                max_duty = magnitude

        if max_duty < self.cfg.overload_duty:
            # Nothing is over the line, so every timer resets -- but only write them once, on the
            # step the run ends, instead of zeroing six numbers a thousand times a second.
            if self._overload_running:
                self._overload_timer[:] = 0.0
                self._overload_running = False
            return

        self._overload_running = True
        hot = np.abs(self._duty) >= self.cfg.overload_duty
        # `timer += dt` then `timer *= hot` is `where(hot, timer + dt, 0)` with one fewer array.
        self._overload_timer += dt
        self._overload_timer *= hot
        if self._overload_timer.max() < self.cfg.overload_time_s:
            return
        newly = hot & (self._overload_timer >= self.cfg.overload_time_s)
        if np.any(newly & ~self.tripped):
            self.tripped |= newly
            self._forcerange_dirty = True
            self._apply_forcerange()


# ------------------------------------------------------------------------------------ figures ---


def _step_response(
    cfg: ArmConfig,
    joint: str,
    amplitude: float,
    seconds: float,
    through_bus: bool,
    servo_cfg: ServoConfig | None = None,
):
    """Drive one joint from 0 to ``amplitude`` and log the angle.  Gravity off, so it is the law.

    The angle logged is the *link* angle -- driven joint plus free-play partner -- which is both
    what the encoder reports and what the load actually does.  With no play the two coincide.
    """
    spec = build_arm_spec(cfg)
    spec.option.gravity = [0.0, 0.0, 0.0]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    adr = model.jnt_qposadr[jid]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{joint}_backlash")
    play_adr = int(model.jnt_qposadr[bid]) if bid >= 0 else -1
    bus = ServoBus(
        model, data, cfg=servo_cfg or ServoConfig(action_delay_jitter_s=0.0)
    )
    goal = np.zeros(len(ARM_JOINTS))
    goal[list(ARM_JOINTS).index(joint)] = amplitude

    n = int(seconds / model.opt.timestep)
    ts = np.arange(n) * model.opt.timestep
    qs = np.zeros(n)
    if through_bus:
        bus.write(goal)
    else:
        data.ctrl[:] = goal
    for k in range(n):
        if through_bus:
            bus.substep()
        else:
            data.ctrl[:] = goal
        mujoco.mj_step(model, data)
        qs[k] = data.qpos[adr] if play_adr < 0 else data.qpos[adr] + data.qpos[play_adr]
    return ts, qs


def settling_time(ts: np.ndarray, qs: np.ndarray, target: float, band: float = 0.02) -> float:
    """Last time the trace is outside ``band`` (fraction of the step) of its target."""
    outside = np.abs(qs - target) > band * abs(target)
    if not outside.any():
        return 0.0
    return float(ts[np.max(np.nonzero(outside))])


def overshoot(qs: np.ndarray, target: float) -> float:
    """Peak excursion past the target, as a fraction of the step."""
    return float(max(0.0, (np.max(qs) - target) / abs(target)))


def sustained_chatter(ts: np.ndarray, qs: np.ndarray, tail_s: float = 0.5) -> float:
    """Peak-to-peak motion over the last ``tail_s`` seconds of a trace [rad].

    What a step response leaves behind once the transient is over.  A real arm buzzes, so the
    honest question is not whether this is zero but whether it is small and whether it is *growing*
    -- see :func:`chatter_growth`.
    """
    tail = qs[ts >= ts[-1] - tail_s]
    return float(tail.max() - tail.min())


def chatter_growth(ts: np.ndarray, qs: np.ndarray, window_s: float = 1.0) -> float:
    """Ratio of the last window's peak-to-peak motion to the window before it.

    Above 1 the oscillation is growing, which is the thing free play can do to a proportional
    loop and the thing this model must not do.  Returns 0 when the earlier window is already flat.

    The window is a second rather than the half second :func:`sustained_chatter` reports, because
    a hunt across the gap has a period of roughly one second at these gains: half-second windows
    catch a peak against a trough and read as growth where there is none.
    """
    late = sustained_chatter(ts, qs, window_s)
    mid = qs[(ts >= ts[-1] - 2.0 * window_s) & (ts < ts[-1] - window_s)]
    if mid.size == 0:
        return 0.0
    early = float(mid.max() - mid.min())
    if early <= 0.0:
        return 0.0 if late <= 0.0 else float("inf")
    return late / early


def step_response_sweep(
    joints=ARM_JOINTS,
    gains=(7.0, 9.384, 14.0, 20.0),
    amplitude: float = 0.4,
    seconds: float = 3.0,
    play_deg: float = ArmConfig().backlash_deg,
    deadband_ticks: float = 0.0,
) -> list[dict]:
    """Step every arm joint at every gain in the randomized range and report what it left behind.

    ``gains`` is in N*m/rad, the units ``dynamics.kp`` is randomized in (7-20); each is turned back
    into the P value the law needs.  One row per (joint, gain) with the overshoot, the 2 % settling
    time, the peak-to-peak motion still present in the last half second, and whether that motion is
    growing.  This is the stability evidence for the free-play joints: a proportional loop closed
    around a gap can hunt across it, and a loop that hunts *and grows* is a model nobody can train
    against.
    """
    rows = []
    for kp in gains:
        p_gain = kp / (
            (BAM_KT / BAM_R) * NOMINAL_VIN * BAM_ERROR_GAIN * BAM_ERROR_GAIN_RATIO
        )
        cfg = ArmConfig(p_gain=p_gain, backlash_deg=play_deg)
        servo_cfg = ServoConfig(
            action_delay_jitter_s=0.0, error_deadband_ticks=deadband_ticks
        )
        for joint in joints:
            ts, qs = _step_response(
                cfg, joint, amplitude, seconds, through_bus=True, servo_cfg=servo_cfg
            )
            rows.append({
                "joint": joint,
                "kp": float(kp),
                "overshoot": overshoot(qs, amplitude),
                "settle_s": settling_time(ts, qs, amplitude),
                "chatter_deg": math.degrees(sustained_chatter(ts, qs)),
                "growth": chatter_growth(ts, qs),
                "final_err_deg": math.degrees(abs(qs[-1] - amplitude)),
            })
    return rows


def figure_step_response(out_path: str | Path = "docs/media/robot/step_response.png"):
    """Two panels: what the firmware's rate limit does to a step, and what the gear play does.

    Left, the original point: a bare position actuator at the BAM gain rings, and the real servo
    does not, because the firmware slews its own target at about 5.2 rad/s before the P loop ever
    sees it.  Anything tuned against the raw actuator is tuned against a robot that does not exist.

    Right, the point of the encoder correction: with the loop closed on the output shaft the free
    play does not leave a standing offset -- the servo takes the slack up itself -- but inside the
    firmware's dead band it stops correcting and the link is free to sit anywhere in the gap.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amp = 0.5
    tail_s = 4.0
    rigid = ArmConfig(backlash_deg=0.0)
    play = ArmConfig()
    ts_bus, q_bus = _step_response(rigid, "shoulder_lift", amp, 1.2, through_bus=True)
    ts_raw, q_raw = _step_response(rigid, "shoulder_lift", amp, 1.2, through_bus=False)
    ts_rigid, q_rigid = _step_response(rigid, "shoulder_lift", amp, tail_s, through_bus=True)
    ts_play, q_play = _step_response(play, "shoulder_lift", amp, tail_s, through_bus=True)
    ts_db, q_db = _step_response(
        play, "shoulder_lift", amp, tail_s, through_bus=True,
        servo_cfg=ServoConfig(
            action_delay_jitter_s=0.0, error_deadband_ticks=STS3215_ERROR_DEADBAND_TICKS
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.2), dpi=110)
    ax = axes[0]
    ax.axhline(amp, color="0.55", lw=1.0)
    ax.axhspan(0.98 * amp, 1.02 * amp, color="0.88", zorder=0, label="2 % band")
    ax.plot(ts_raw, q_raw, lw=1.8, color="#b0413e", label="bare position actuator")
    ax.plot(ts_bus, q_bus, lw=2.2, color="#1f6feb", label="through ServoBus (rate limit + delay)")
    ax.set_xlabel("time [s]", fontsize=13)
    ax.set_ylabel("shoulder_lift, link angle [rad]", fontsize=13)
    ax.set_title("Joint Step Response", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

    ax = axes[1]
    one_count = 2.0 * math.pi / ServoConfig().ticks_per_rev
    ax.axhline(0.0, color="0.55", lw=1.0)
    for ts, qs, colour, label in (
        (ts_rigid, q_rigid, "#1f6feb", "no free play"),
        (ts_play, q_play, "#2f9e70", f"play ±{play.backlash_deg}°, no dead band"),
        (ts_db, q_db, "#e0a458",
         f"play ±{play.backlash_deg}°, dead band {STS3215_ERROR_DEADBAND_TICKS:g} counts"),
    ):
        ax.plot(ts, (qs - amp) / one_count, lw=2.0, color=colour, label=label)
    ax.set_xlim(0.4, tail_s)
    ax.set_ylim(-11.0, 11.0)
    ax.set_xlabel("time [s]", fontsize=13)
    ax.set_ylabel("link angle − target [encoder counts]", fontsize=13)
    ax.set_title("Gear Free Play", fontsize=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

    fig.suptitle(
        f"shoulder_lift, 0.5 rad step, kp = {servo_kp(rigid.p_gain, rigid.vin):.1f} N·m/rad "
        f"(P={rigid.p_gain}, {rigid.vin} V), gravity off",
        fontsize=15,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def measure_sag(
    cfg: ArmConfig,
    joint: str,
    torques: np.ndarray,
    seconds: float = 2.5,
    servo_cfg: ServoConfig | None = None,
    signed: bool = False,
):
    """Hold a joint at zero, hang a known torque on it, and return the steady-state error.

    Gravity is off so the only thing bending the joint is the torque we applied.  This is the sim
    version of hanging masses off the real arm and logging ``Present_Position``, and the number it
    returns is read the same way: through :meth:`ServoBus.read`, quantized to encoder counts.

    The torque goes on the *link*.  The driven joint and its free-play partner turn about the same
    axis, so a torque on the link is one generalized force on each of them; loading only the driven
    joint would be a torque on the motor instead, and would hold the play off its stop.

    ``signed`` keeps the sign, which is what a clockwise-against-counter-clockwise comparison
    needs; the default returns the magnitude, as the sag figure wants.
    """
    spec = build_arm_spec(cfg)
    spec.option.gravity = [0.0, 0.0, 0.0]
    model = spec.compile()
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    dof = model.jnt_dofadr[jid]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{joint}_backlash")
    play_dof = int(model.jnt_dofadr[bid]) if bid >= 0 else -1
    index = list(ARM_JOINTS).index(joint)

    errors = np.zeros(len(torques))
    n = int(seconds / model.opt.timestep)
    for i, tau in enumerate(torques):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        bus = ServoBus(
            model, data,
            cfg=servo_cfg or ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0),
        )
        bus.write(np.zeros(len(ARM_JOINTS)))
        for _ in range(n):
            bus.substep()
            data.qfrc_applied[dof] = tau
            if play_dof >= 0:
                data.qfrc_applied[play_dof] = tau
            mujoco.mj_step(model, data)
        value = float(bus.read().position[index])
        errors[i] = value if signed else abs(value)
    return errors


def figure_sag(out_path: str | Path = "docs/media/robot/sag.png"):
    """Plot measured static sag against the ``tau / kp`` the BAM law predicts, for P=16 and P=32.

    The upstream gain is on the same axes so the scale of the error is visible: at 1 N*m upstream
    sags by 0.06 deg and the real servo by about 6 deg.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torques = np.linspace(0.1, 1.6, 10)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    for p_gain, colour in ((16, "C0"), (32, "C1")):
        cfg = ArmConfig(p_gain=p_gain, backlash_deg=0.0)
        err = measure_sag(cfg, "shoulder_lift", torques)
        kp = servo_kp(p_gain, cfg.vin)
        ax.plot(torques, np.degrees(err), "o", color=colour, ms=4,
                label=f"measured, P={p_gain} (kp={kp:.1f})")
        ax.plot(torques, np.degrees(torques / kp), "-", color=colour, lw=1.0, alpha=0.7,
                label=f"predicted tau/kp, P={p_gain}")
    ax.plot(torques, np.degrees(torques / 998.22), "k--", lw=1.0,
            label="upstream kp=998.22")
    ax.set_xlabel("applied joint torque [N·m]")
    ax.set_ylabel("steady-state joint error [deg]")
    ax.set_title("Static Joint Deflection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def make_figures(out_dir: str | Path = "docs/media/robot") -> dict:
    """The WP-ROBOT figures.  Entry point for ``scripts/make_media.py``.

    The settling-error panel drives the arm to 49 poses four times over, which takes a few minutes;
    the other two are seconds.  It is imported lazily so that the servo model does not depend on
    the correction module at import time.
    """
    from brick_builder.robot.joint_correction import figure_settling_error

    out_dir = Path(out_dir)
    return {
        "step_response": figure_step_response(out_dir / "step_response.png"),
        "sag": figure_sag(out_dir / "sag.png"),
        "settling_error": figure_settling_error(out_dir / "settling_error.png"),
    }
