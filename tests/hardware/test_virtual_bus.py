"""The virtual bus with a MuJoCo scene behind it, and the adapter verbs on top of the driver."""

import numpy as np
import pytest

from brick_builder.hardware.bus import Bus, BusSettings
from brick_builder.hardware.real_world import ArmWorldConfig, RealArmWorld
from brick_builder.hardware.virtual_bus import simulated_bus
from brick_builder.robot.patch_so101 import ArmConfig
from brick_builder.scene.compose import SceneConfig, build_scene, reset_scene

SETTINGS = BusSettings(timeout_s=1.0, retries=1)


def bare_arm(**kwargs):
    scene = build_scene(SceneConfig(arm=ArmConfig(**kwargs), held=None, placed=[], tray=False,
                                    table=False, cameras=(), damage_monitor=False))
    reset_scene(scene, np.random.default_rng(0))
    return scene


@pytest.fixture
def driven_arm():
    scene = bare_arm()
    vbus = simulated_bus(scene).start()
    driver = Bus(vbus.port, SETTINGS)
    try:
        yield scene, driver, vbus
    finally:
        driver.close()
        vbus.stop()
        vbus.raise_if_failed()


def test_one_sync_write_is_one_control_tick_of_physics(driven_arm):
    scene, driver, vbus = driven_arm
    before = vbus.servos.ticks
    for _ in range(5):
        driver.cycle(np.zeros(6))
    assert vbus.servos.ticks - before == 5


def test_the_driver_moves_the_simulated_arm_to_where_it_asked(driven_arm):
    scene, driver, vbus = driven_arm
    driver.enable_torque(check=False)
    goal = np.zeros(6)
    goal[1] = -0.4
    for _ in range(120):
        state = driver.cycle(goal)
    # A proportional servo sits tau/kp off its target under gravity, which at this pose is a
    # couple of degrees; what matters is that it went there at all and nothing was lost.
    assert state.position[1] == pytest.approx(-0.4, abs=0.12)
    assert driver.counters.errors == 0
    assert state.valid.all()


def test_a_limp_servo_does_not_hold_its_target(driven_arm):
    scene, driver, vbus = driven_arm
    goal = np.zeros(6)
    for _ in range(40):
        state = driver.cycle(goal)
    sag = abs(state.position[1])
    driver.enable_torque(check=False)
    for _ in range(80):
        state = driver.cycle(goal)
    assert abs(state.position[1]) < sag


def test_present_load_is_a_duty_cycle_that_grows_with_the_tracking_error(driven_arm):
    scene, driver, vbus = driven_arm
    driver.enable_torque(check=False)
    small = np.zeros(6)
    for _ in range(30):
        near = driver.cycle(small)
    big = np.zeros(6)
    big[1] = -1.2
    far = driver.cycle(big)
    assert abs(far.load[1]) > abs(near.load[1])
    assert np.all(np.abs(far.load) <= 0.97 + 1e-9)


def test_the_readings_come_back_as_whole_encoder_counts(driven_arm):
    from brick_builder.hardware import sts3215 as proto

    scene, driver, vbus = driven_arm
    state = driver.cycle(np.zeros(6))
    counts = state.position / proto.RAD_PER_TICK
    assert np.allclose(counts, np.round(counts), atol=1e-9)


def test_the_arm_world_answers_the_adapter_verbs_over_the_bus(driven_arm):
    scene, driver, vbus = driven_arm
    driver.enable_torque(check=False)
    world = RealArmWorld(driver, cfg=ArmWorldConfig(paced=False, source="virtual_bus"))
    start = world.kin.fk(world.joint_position[:5])[0]
    target = start + np.array([0.0, 0.0, -0.015])
    for k in range(80):
        frac = min((k + 1) / 40.0, 1.0)
        world.command(start * (1 - frac) + target * frac, 0.0, 0.4)
    feedback = world.feedback(0.4)
    assert world.tick == 80
    assert world.ik_failures == 0
    assert np.linalg.norm(feedback.tcp_pos - target) < 0.006
    assert world.joint_position.shape == (6,)
    assert world.joint_speed.shape == (6,)
    assert world.joint_load.shape == (6,)
    assert len(world.log) == 80
    assert np.isfinite(world.log.column("bus_rtt_s")).all()


def test_the_pacer_keeps_a_wall_clock_period():
    import time

    from brick_builder.hardware.real_world import Pacer

    pacer = Pacer(0.01)
    t0 = time.monotonic()
    for _ in range(10):
        pacer.wait()
    elapsed = time.monotonic() - t0
    assert 0.09 < elapsed < 0.30
    assert pacer.tick == 10


def test_the_pacer_reports_an_overrun_instead_of_bursting():
    import time

    from brick_builder.hardware.real_world import Pacer

    pacer = Pacer(0.002)
    time.sleep(0.02)
    late = pacer.wait()
    assert late > 0.0
    assert pacer.overruns == 1
