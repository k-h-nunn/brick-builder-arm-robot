"""Unit tests for the servo emulation: quantization, dead band, delay, rate limit, overload.

Each one isolates a single register-level behaviour, because these are the parts a policy will learn
to exploit if they are wrong, and they are invisible in an aggregate "the arm moved" test.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.robot import servo_model as S
from brick_builder.robot.patch_so101 import (
    ARM_JOINTS,
    ArmConfig,
    build_arm_spec,
    servo_kp,
    servo_stall_torque,
)
from brick_builder.robot.servo_model import (
    STS3215_ERROR_DEADBAND_TICKS,
    ServoBus,
    ServoConfig,
)

TICK = 2.0 * math.pi / 4096
"""One encoder count [rad].  0.088 degrees, and the floor under every number in this file."""

PLAY = ArmConfig().backlash_deg
"""The default half-range of the free-play joints [deg]."""


def _arm(cfg: ArmConfig = ArmConfig(backlash_deg=0.0)):
    model = build_arm_spec(cfg).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _bus(cfg: ServoConfig, arm_cfg: ArmConfig = ArmConfig(backlash_deg=0.0), seed: int = 0):
    model, data = _arm(arm_cfg)
    return model, data, ServoBus(model, data, cfg=cfg, rng=np.random.default_rng(seed))


# --------------------------------------------------------------------------------- quantizing ---


def test_goal_is_quantized_to_encoder_ticks():
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    goal = np.zeros(6)
    goal[0] = 0.4 * tick          # less than half a tick: rounds to zero
    bus.write(goal)
    assert data.ctrl[0] == pytest.approx(0.0, abs=1e-12)
    goal[0] = 0.6 * tick          # more than half a tick: rounds up to one
    bus.write(goal)
    assert data.ctrl[0] == pytest.approx(tick, abs=1e-12)
    assert np.all(np.abs(np.round(data.ctrl / tick) * tick - data.ctrl) < 1e-12)


def test_readings_are_quantized_to_ticks():
    cfg = ServoConfig()
    model, data, bus = _bus(cfg)
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    data.qpos[model.jnt_qposadr[0]] = 0.123456
    mujoco.mj_forward(model, data)
    pos = bus.read().position
    assert pos[0] != pytest.approx(0.123456, abs=1e-6)
    assert pos[0] == pytest.approx(round(0.123456 / tick) * tick, abs=1e-12)


def test_dead_band_swallows_small_command_changes():
    cfg = ServoConfig(deadband_ticks=10, action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    big = np.zeros(6)
    big[0] = 40 * tick
    bus.write(big)
    for _ in range(400):
        bus.substep()
        mujoco.mj_step(model, data)
    settled = data.ctrl[0]
    assert settled == pytest.approx(40 * tick)

    nudge = big.copy()
    nudge[0] = 45 * tick          # 5 ticks: inside the dead band, ignored
    bus.write(nudge)
    bus.substep()
    assert data.ctrl[0] == pytest.approx(settled)

    step = big.copy()
    step[0] = 55 * tick           # 15 ticks: outside it, accepted
    bus.write(step)
    for _ in range(200):
        bus.substep()
        mujoco.mj_step(model, data)
    assert data.ctrl[0] == pytest.approx(55 * tick)


def test_zero_dead_band_passes_single_ticks():
    cfg = ServoConfig(deadband_ticks=0, action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    goal = np.zeros(6)
    goal[0] = tick
    bus.write(goal)
    assert data.ctrl[0] == pytest.approx(tick)


# -------------------------------------------------------------------------- delay, rate limit ---


def test_a_command_only_arrives_after_the_wire_delay():
    cfg = ServoConfig(action_delay_s=0.008, action_delay_jitter_s=0.0, rate_limit=1e6)
    model, data, bus = _bus(cfg)
    goal = np.zeros(6)
    goal[0] = 0.5
    bus.write(goal)
    dt = model.opt.timestep
    n_late = int(round(cfg.action_delay_s / dt))
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    for k in range(n_late):
        assert data.ctrl[0] == pytest.approx(0.0), f"command arrived early, at step {k}"
        bus.substep()
    assert data.ctrl[0] == pytest.approx(round(0.5 / tick) * tick)


def test_delay_jitter_stays_inside_its_band_and_is_seeded():
    cfg = ServoConfig(action_delay_s=0.008, action_delay_jitter_s=0.004)
    arrivals = []
    for seed in (0, 1, 2, 3, 4, 5):
        model, data, bus = _bus(cfg, seed=seed)
        goal = np.zeros(6)
        goal[0] = 0.5
        bus.write(goal)
        k = 0
        while data.ctrl[0] == 0.0 and k < 200:
            bus.substep()
            k += 1
        arrivals.append(k * model.opt.timestep)
    assert min(arrivals) >= 0.004 - 1e-9
    assert max(arrivals) <= 0.012 + model.opt.timestep
    assert len(set(arrivals)) > 1, "jitter should actually vary between seeds"

    _, _, a = _bus(cfg, seed=7)
    _, _, b = _bus(cfg, seed=7)
    a.write(np.zeros(6))
    b.write(np.zeros(6))
    assert a._queue[0][0] == b._queue[0][0]


def test_the_firmware_target_slews_at_the_rate_limit():
    cfg = ServoConfig(rate_limit=5.0, action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    goal = np.zeros(6)
    goal[0] = 1.5
    bus.write(goal)
    dt = model.opt.timestep
    previous = data.ctrl[0]
    for _ in range(100):
        bus.substep()
        assert data.ctrl[0] - previous <= cfg.rate_limit * dt + 1e-12
        previous = data.ctrl[0]
    # and it does get there: 1.5 rad at 5 rad/s is 0.3 s
    for _ in range(300):
        bus.substep()
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    assert data.ctrl[0] == pytest.approx(round(1.5 / tick) * tick, abs=1e-9)


# ----------------------------------------------------------------------- torque limit, overload ---


def test_torque_limit_scales_the_forcerange_for_that_tick():
    model, data, bus = _bus(ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0))
    base = model.actuator_forcerange.copy()
    bus.write(np.zeros(6), torque_limit=np.full(6, 0.3))
    assert model.actuator_forcerange[:, 1] == pytest.approx(0.3 * base[:, 1])
    bus.write(np.zeros(6), torque_limit=None)
    assert model.actuator_forcerange[:, 1] == pytest.approx(base[:, 1])
    bus.write(np.zeros(6), torque_limit=np.full(6, 5.0))  # clipped to 1.0
    assert model.actuator_forcerange[:, 1] == pytest.approx(base[:, 1])


def test_a_torque_limit_actually_weakens_the_joint():
    """With the joint stalled against a load it cannot beat, the force sits on the limit."""
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0)
    forces = {}
    for limit in (1.0, 0.3):
        model, data, bus = _bus(cfg)
        dof = model.jnt_dofadr[1]
        goal = np.zeros(6)
        goal[1] = 1.0
        for _ in range(400):
            bus.write(goal, torque_limit=np.full(6, limit))
            bus.substep()
            data.qfrc_applied[dof] = -3.0     # a load the 1.9 N*m servo cannot lift
            mujoco.mj_step(model, data)
        forces[limit] = abs(data.actuator_force[1])
    assert forces[1.0] == pytest.approx(1.9, rel=1e-3)
    assert forces[0.3] == pytest.approx(0.3 * 1.9, rel=1e-3)


def test_load_is_a_duty_cycle_not_a_torque():
    """``Present_Load`` saturates at 0.97 while the joint torque saturates at forcerange."""
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    dof = model.jnt_dofadr[1]
    goal = np.zeros(6)
    goal[1] = 1.0
    for _ in range(600):
        bus.write(goal)
        bus.substep()
        data.qfrc_applied[dof] = -3.0
        mujoco.mj_step(model, data)
    readings = bus.read()
    assert abs(readings.load[1]) == pytest.approx(0.97, abs=cfg.load_lsb)
    assert abs(data.actuator_force[1]) == pytest.approx(1.9, rel=1e-3)
    # the two disagree on purpose: duty 0.97 would be 3.4 N*m if the datasheet limit let it
    assert servo_stall_torque(cfg.vin) > 1.9
    assert np.all(np.abs(np.round(readings.load / cfg.load_lsb) * cfg.load_lsb - readings.load)
                  < 1e-12)


def test_overload_trips_after_sustained_saturation_and_derates_the_servo():
    cfg = ServoConfig(
        action_delay_s=0.0, action_delay_jitter_s=0.0, overload_time_s=0.5, overload_derate=0.2
    )
    model, data, bus = _bus(cfg)
    dof = model.jnt_dofadr[1]
    base = model.actuator_forcerange[1, 1]
    goal = np.zeros(6)
    goal[1] = 1.0
    tripped_at = None
    for k in range(1500):
        bus.write(goal)
        bus.substep()
        data.qfrc_applied[dof] = -3.0     # stalled: the duty pins at 0.97 and stays there
        mujoco.mj_step(model, data)
        if bus.tripped[1] and tripped_at is None:
            tripped_at = k * model.opt.timestep
    assert tripped_at is not None, "never tripped while pinned at full duty"
    assert tripped_at == pytest.approx(cfg.overload_time_s, abs=0.05)
    assert model.actuator_forcerange[1, 1] == pytest.approx(0.2 * base)
    assert abs(data.actuator_force[1]) == pytest.approx(0.2 * base, rel=1e-3)

    bus.reset_overload()
    assert not bus.tripped.any()
    assert model.actuator_forcerange[1, 1] == pytest.approx(base)


def test_a_servo_inside_its_duty_limit_never_trips():
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0, overload_time_s=0.2)
    model, data, bus = _bus(cfg)
    goal = np.zeros(6)
    goal[1] = 0.05                     # small error, small duty
    for _ in range(1000):
        bus.write(goal)
        bus.substep()
        mujoco.mj_step(model, data)
    assert not bus.tripped.any()
    assert abs(bus.duty[1]) < cfg.overload_duty


# --------------------------------------------------------------------------------- reading ---


def test_read_skew_walks_each_servo_back_along_its_own_velocity():
    cfg = ServoConfig(read_skew_s=0.001)
    model, data, bus = _bus(cfg)
    for i, name in enumerate(ARM_JOINTS):
        data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = 1.0
    mujoco.mj_forward(model, data)
    pos = bus.read().position
    # servo i was sampled i * skew earlier, so at 1 rad/s it reads i * skew radians lower
    tick = 2.0 * math.pi / cfg.ticks_per_rev
    expected = np.round(-np.arange(6) * cfg.read_skew_s / tick) * tick
    assert pos == pytest.approx(expected, abs=1e-12)

    zero_skew = ServoBus(model, data, cfg=ServoConfig(read_skew_s=0.0))
    assert np.all(zero_skew.read().position == 0.0)


def test_read_noise_is_seeded_and_off_by_default():
    model, data, bus = _bus(ServoConfig())
    assert np.all(bus.read().position == bus.read().position)

    noisy_cfg = ServoConfig(read_noise_ticks=3.0, read_skew_s=0.0)
    _, _, a = _bus(noisy_cfg, seed=11)
    _, _, b = _bus(noisy_cfg, seed=11)
    _, _, c = _bus(noisy_cfg, seed=12)
    assert np.all(a.read().position == b.read().position)
    assert np.any(a.read().position != c.read().position)


def test_current_follows_the_motor_equation():
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    goal = np.zeros(6)
    goal[1] = 0.3
    for _ in range(400):
        bus.write(goal)
        bus.substep()
        mujoco.mj_step(model, data)
    readings = bus.read()
    dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift")]
    from brick_builder.robot.patch_so101 import BAM_KT, BAM_R

    expected = (cfg.vin * bus.duty[1] - BAM_KT * data.qvel[dof]) / BAM_R
    assert readings.current[1] == pytest.approx(expected, abs=cfg.current_lsb)


def test_reset_reseeds_the_target_from_where_the_arm_is():
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0)
    model, data, bus = _bus(cfg)
    goal = np.zeros(6)
    goal[0] = 0.8
    bus.write(goal)
    for _ in range(600):
        bus.substep()
        mujoco.mj_step(model, data)
    assert data.ctrl[0] == pytest.approx(0.8, abs=1e-3)

    mujoco.mj_resetData(model, data)
    data.qpos[model.jnt_qposadr[0]] = -0.4
    mujoco.mj_forward(model, data)
    bus.reset()
    assert data.ctrl[0] == pytest.approx(-0.4, abs=1e-3)
    assert not bus.tripped.any()


def test_bus_rejects_joints_the_model_does_not_have():
    model, data = _arm()
    with pytest.raises(ValueError, match="joints not in the model"):
        ServoBus(model, data, arm_joint_names=("no_such_joint",))


# --------------------------------------------------------------------------- step response ---


@pytest.mark.parametrize("joint", ["shoulder_lift", "wrist_flex"])
def test_step_response_settles_within_600_ms(joint):
    cfg = ArmConfig(backlash_deg=0.0)
    ts, qs = S._step_response(cfg, joint, 0.5, 1.5, through_bus=True)
    settle = S.settling_time(ts, qs, 0.5)
    assert settle < 0.6, f"{joint} took {settle:.3f} s to settle to 2 %"
    assert qs[-1] == pytest.approx(0.5, abs=0.01)


def test_step_response_matches_the_measured_underdamped_character():
    """The specification's "over-damped" was an unverified assumption. At the identified
    BAM constants (joint damping 0.62 N*m*s/rad, well under the ~0.99 an over-damped response would
    need at this kp and armature) the arm is under-damped instead, so this asserts the character that
    was actually measured -- overshoot <= 15 %, 2 % settling < 0.65 s for a 0.5 rad step -- instead of
    an over-damped bound the servo constants cannot meet. System identification on the real arm will
    calibrate these numbers further; see the guide page for the measured values this bounds."""
    cfg = ArmConfig(backlash_deg=0.0)
    ts, qs = S._step_response(cfg, "shoulder_lift", 0.5, 1.5, through_bus=True)
    over = S.overshoot(qs, 0.5)
    settle = S.settling_time(ts, qs, 0.5)
    assert over <= 0.15, f"overshoot {over:.3f} exceeds the 15% bound"
    assert settle < 0.65, f"settling time {settle:.3f} s exceeds the 0.65 s bound"


def test_the_rate_limit_is_what_slows_a_large_step():
    """A big step is ramp-dominated: it cannot move faster than the firmware's own slew."""
    cfg = ArmConfig(backlash_deg=0.0)
    ts, qs = S._step_response(cfg, "shoulder_lift", 1.0, 1.5, through_bus=True)
    reached = ts[np.argmax(qs > 0.9)]
    assert reached >= 0.9 / S.ServoConfig().rate_limit


# -------------------------------------------------------------- the encoder is on the output ---


def _play_adr(model, joint: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{joint}_backlash")
    return int(model.jnt_qposadr[jid])


def test_the_encoder_reads_the_output_shaft_not_the_motor():
    """``Present_Position`` is driven + free play, because the encoder is after the gearbox."""
    model, data, bus = _bus(ServoConfig(read_skew_s=0.0), ArmConfig(backlash_deg=PLAY))
    driven = int(model.jnt_qposadr[bus.joint_ids[1]])
    data.qpos[driven] = 0.1
    data.qpos[_play_adr(model, "shoulder_lift")] = math.radians(0.4)
    mujoco.mj_forward(model, data)
    link = 0.1 + math.radians(0.4)
    assert bus.read().position[1] == pytest.approx(round(link / TICK) * TICK, abs=1e-12)

    motor_side = ServoBus(model, data, cfg=ServoConfig(read_skew_s=0.0, encoder_on_output=False))
    assert motor_side.read().position[1] == pytest.approx(round(0.1 / TICK) * TICK, abs=1e-12)


def test_the_firmware_law_closes_on_the_output_shaft():
    """The torque is ``kp * (target - link)``, and it still enters at the driven joint."""
    model, data, bus = _bus(
        ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0), ArmConfig(backlash_deg=PLAY)
    )
    driven = int(model.jnt_qposadr[bus.joint_ids[1]])
    play = _play_adr(model, "shoulder_lift")
    data.qpos[driven] = 0.05
    data.qpos[play] = math.radians(0.3)
    mujoco.mj_forward(model, data)

    goal = np.zeros(6)
    goal[1] = 0.2
    bus.write(goal)
    mujoco.mj_forward(model, data)
    error = bus.target[1] - (data.qpos[driven] + data.qpos[play])
    assert data.actuator_force[1] == pytest.approx(bus.kp[1] * error, rel=1e-9)
    # and the transmission is still the driven joint, so the torque has to cross the play
    assert model.actuator_trnid[bus.act_ids[1], 0] == bus.joint_ids[1]


def test_with_no_free_play_the_two_encoder_positions_are_the_same_model():
    """Regression guard: at ``backlash_deg = 0`` the correction is a no-op, bit for bit.

    Everything the bus produces -- ``data.ctrl``, the readings, the duty -- is compared step by
    step against the motor-side model this project ran before, over a move that uses the rate
    limit, the wire delay and a torque limit.
    """
    goal = np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.4])
    traces = []
    for on_output in (True, False):
        model, data = _arm(ArmConfig(backlash_deg=0.0))
        bus = ServoBus(
            model, data,
            cfg=ServoConfig(action_delay_jitter_s=0.0, encoder_on_output=on_output),
            rng=np.random.default_rng(3),
        )
        rows = []
        for k in range(300):
            if k % 10 == 0:
                bus.write(goal, torque_limit=np.full(6, 0.6))
            bus.substep()
            mujoco.mj_step(model, data)
            reading = bus.read()
            rows.append(np.concatenate([
                data.ctrl.copy(), reading.position, reading.speed, reading.load, bus.duty
            ]))
        traces.append(np.array(rows))
    assert np.array_equal(traces[0], traces[1])
    # and `data.ctrl` is still literally the firmware target, as it was before the correction
    assert np.array_equal(traces[0][-1][:6], bus.target)


def test_the_firmware_dead_band_commands_nothing_inside_itself():
    """Registers 26/27: no correction at all while the error is inside the dead band."""
    cfg = ServoConfig(action_delay_s=0.0, action_delay_jitter_s=0.0, error_deadband_ticks=6.0)
    model, data, bus = _bus(cfg, ArmConfig(backlash_deg=PLAY))
    driven = int(model.jnt_qposadr[bus.joint_ids[1]])
    goal = np.zeros(6)
    goal[1] = 4.0 * TICK          # four counts: inside the six-count dead band
    bus.write(goal)
    mujoco.mj_forward(model, data)
    assert data.ctrl[bus.act_ids[1]] == pytest.approx(data.qpos[driven], abs=1e-15)
    assert data.actuator_force[1] == pytest.approx(0.0, abs=1e-12)
    assert bus.duty[1] == 0.0

    goal[1] = 9.0 * TICK          # nine counts: outside it, and the law is back
    bus.write(goal)
    mujoco.mj_forward(model, data)
    error = bus.target[1] - data.qpos[driven] - data.qpos[_play_adr(model, "shoulder_lift")]
    assert data.actuator_force[1] == pytest.approx(bus.kp[1] * error, rel=1e-9)


# ------------------------------------------------------ against the published bench numbers ---
#
# Robonine's STS3215 bench report is the only independent
# measurement of a real one of these servos this project has.  Their rig is a 12 V part at what
# the numbers imply is the factory default P = 32, so that is what these reproduce.

BENCH_P_GAIN = 32
BENCH_VIN = 12.0
KG_CM = 9.80665e-2   # N*m per kg*cm


def _bench_arm(**kwargs):
    return ArmConfig(p_gain=BENCH_P_GAIN, vin=BENCH_VIN, **kwargs)


def _bench_servo(deadband: float = 0.0):
    return ServoConfig(
        action_delay_s=0.0, action_delay_jitter_s=0.0, error_deadband_ticks=deadband
    )


@pytest.mark.parametrize("load_kg_cm,measured_counts", [(10.0, 22.5), (15.0, 30.0)])
def test_droop_under_static_load_is_tau_over_kp(load_kg_cm, measured_counts):
    """Sag read through the encoder is the law's own ``tau / kp``, and matches the bench.

    Robonine measured about 22.5 counts at 10 kg*cm and about 30 at 15 kg*cm.  The BAM law at
    P = 32 and 12 V predicts 1.85 and 2.77 degrees, i.e. 21 and 31.5 counts.
    """
    tau = load_kg_cm * KG_CM
    kp = servo_kp(BENCH_P_GAIN, BENCH_VIN)
    error = S.measure_sag(
        _bench_arm(), "shoulder_lift", np.array([tau]), seconds=4.0, servo_cfg=_bench_servo()
    )[0]
    assert error == pytest.approx(tau / kp, abs=1.5 * TICK)
    assert error / TICK == pytest.approx(measured_counts, abs=2.0)


def test_clockwise_versus_counterclockwise_loaded_deviation_matches_the_bench():
    """The loaded CW/CCW deviation is ``2 tau / kp``, not the gear play.

    Robonine read 14.78 counts off the encoder under "about 3 kg*cm".  With the encoder on the
    output shaft the servo takes the slack up itself, so what that measurement sees is twice the
    elastic droop: 3.0 kg*cm gives 12 counts here and 14.78 counts implies 3.5 kg*cm, inside
    "about 3".  Sweeping the play changes nothing, which is the point.
    """
    tau = 3.0 * KG_CM
    kp = servo_kp(BENCH_P_GAIN, BENCH_VIN)
    deviations = {}
    for play in (0.0, PLAY, 1.0):
        pair = S.measure_sag(
            _bench_arm(backlash_deg=play), "shoulder_lift", np.array([+tau, -tau]),
            seconds=4.0, servo_cfg=_bench_servo(), signed=True,
        )
        deviations[play] = abs(pair[0] - pair[1])
    assert deviations[PLAY] == pytest.approx(2.0 * tau / kp, abs=1.5 * TICK)
    assert deviations[PLAY] / TICK == pytest.approx(12.0, abs=1.0)
    assert max(deviations.values()) - min(deviations.values()) <= 1.5 * TICK
    # the bench's own 14.78 counts is this law at a slightly heavier load
    implied = kp * 14.78 * TICK / 2.0
    assert implied / KG_CM == pytest.approx(3.5, abs=0.1)


def test_free_play_is_visible_in_the_encoder_inside_the_dead_band():
    """The dial-gauge measurement: a gentle push moves the output shaft by the whole slack.

    With the firmware's dead band wide enough that it does not fight back, and a torque under the
    gear train's own breakaway friction, the link swings across its free play and the encoder
    reports every bit of it -- which is exactly how the bench got 0.87 degrees, and exactly what
    the motor-side model could not have produced.
    """
    gentle = 0.02      # N*m, well under the 0.0515 frictionloss at the driven joint
    servo_cfg = _bench_servo(deadband=10.0)
    swings = {}
    for play in (0.0, PLAY):
        pair = S.measure_sag(
            _bench_arm(backlash_deg=play), "shoulder_lift", np.array([+gentle, -gentle]),
            seconds=4.0, servo_cfg=servo_cfg, signed=True,
        )
        swings[play] = abs(pair[0] - pair[1])
    # the free part of the swing is the configured total play, plus a couple of counts of the
    # solver's own soft friction and limit compliance (which is what the play=0 case measures)
    free = swings[PLAY] - swings[0.0]
    assert free / TICK == pytest.approx(2.0 * PLAY / math.degrees(TICK), abs=3.0)

    # and with no dead band the servo corrects all of it away
    tight = S.measure_sag(
        _bench_arm(backlash_deg=PLAY), "shoulder_lift", np.array([+gentle, -gentle]),
        seconds=4.0, servo_cfg=_bench_servo(), signed=True,
    )
    assert abs(tight[0] - tight[1]) <= 1.0 * TICK


def test_one_count_is_still_0_088_degrees():
    """Quantization did not move: 4096 counts per output turn, both ways."""
    cfg = ServoConfig(read_skew_s=0.0)
    model, data, bus = _bus(cfg, ArmConfig(backlash_deg=PLAY))
    assert math.degrees(bus.tick) == pytest.approx(0.087890625, abs=1e-12)
    data.qpos[int(model.jnt_qposadr[bus.joint_ids[0]])] = 0.123456
    data.qpos[_play_adr(model, "shoulder_pan")] = 0.001
    mujoco.mj_forward(model, data)
    pos = bus.read().position
    assert np.all(np.abs(np.round(pos / bus.tick) * bus.tick - pos) < 1e-12)
    goal = np.zeros(6)
    goal[0] = 0.123456
    bus.write(goal)
    assert np.all(np.abs(np.round(bus.target / bus.tick) * bus.tick - bus.target) < 1e-12)


# ------------------------------------------------------------------------------- stability ---


QUANTIZATION_FLOOR_DEG = 2.0 * math.degrees(TICK)
"""Below two counts of motion a peak-to-peak ratio is comparing rounding, not oscillation."""

CHATTER_BOUND_DEG = 1.2
"""Largest sustained hunt any joint is allowed at any gain in the randomized range.

The measured worst is 1.02 degrees -- ``shoulder_lift`` at the top of the range (20 N*m/rad),
gravity off, where nothing preloads the play against a stop and the link is free to rattle across
it.  It is a real limit cycle, it is flat to within 0.2 counts over twenty seconds, and with
gravity on (every scene in this project) the same joint buzzes about half as far.
"""


@pytest.mark.parametrize("deadband", [0.0, STS3215_ERROR_DEADBAND_TICKS])
@pytest.mark.slow
def test_every_joint_is_bounded_across_the_randomized_gain_range(deadband):
    """A proportional loop around a gap can hunt.  It must not hunt harder as time goes on.

    Steps every arm joint at every corner of the randomized gain range (7-20 N*m/rad) with the
    play on, and checks that what is left at the end is small and not growing.  The arm does buzz
    -- that is honest, and the numbers are in the report -- but the buzz is bounded.
    """
    rows = S.step_response_sweep(seconds=6.0, deadband_ticks=deadband)
    assert len(rows) == len(ARM_JOINTS) * 4
    for row in rows:
        assert row["chatter_deg"] <= CHATTER_BOUND_DEG, row
        assert row["final_err_deg"] <= 0.6, row
        if row["chatter_deg"] > QUANTIZATION_FLOOR_DEG:
            assert row["growth"] <= 1.05, row


@pytest.mark.slow
def test_the_worst_hunt_is_flat_rather_than_growing():
    """Twenty seconds at the top of the gain range: the limit cycle neither grows nor decays."""
    from brick_builder.robot.patch_so101 import (
        BAM_ERROR_GAIN,
        BAM_ERROR_GAIN_RATIO,
        BAM_KT,
        BAM_R,
        NOMINAL_VIN,
    )

    p_gain = 20.0 / ((BAM_KT / BAM_R) * NOMINAL_VIN * BAM_ERROR_GAIN * BAM_ERROR_GAIN_RATIO)
    ts, qs = S._step_response(
        ArmConfig(p_gain=p_gain), "shoulder_lift", 0.4, 20.0, through_bus=True,
        servo_cfg=ServoConfig(action_delay_jitter_s=0.0),
    )
    windows = [
        (qs[(ts >= a) & (ts < a + 2.0)].max() - qs[(ts >= a) & (ts < a + 2.0)].min()) / TICK
        for a in range(2, 20, 2)
    ]
    assert max(windows) - min(windows) < 1.0, windows      # counts, over eighteen seconds
    assert max(windows) < CHATTER_BOUND_DEG / math.degrees(TICK)


# ------------------------------------------------------------------------------- the figures ---


def test_the_fast_figures_are_written(tmp_path):
    for path in (S.figure_step_response(tmp_path / "step.png"), S.figure_sag(tmp_path / "sag.png")):
        assert path.exists()
        assert path.stat().st_size > 10_000  # a real plot, not a blank canvas


@pytest.mark.slow
def test_make_figures_writes_all_three(tmp_path):
    paths = S.make_figures(tmp_path)
    assert set(paths) == {"step_response", "sag", "settling_error"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 10_000


def test_measure_sag_is_monotone_in_torque():
    errors = S.measure_sag(ArmConfig(backlash_deg=0.0), "shoulder_lift", np.array([0.4, 0.9, 1.4]))
    assert np.all(np.diff(errors) > 0)
    assert errors[0] < errors[-1] < 0.25
    assert errors == pytest.approx(
        np.array([0.4, 0.9, 1.4]) / servo_kp(16, 7.4), rel=0.10
    )
