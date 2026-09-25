"""Acceptance tests for the scripted pick and place with the stock gripper.

Each of these asserts something physical about a run of the real simulator - what the jaws can
reach, whether a brick actually ended up connected to the plate at the cell it was aimed at, and
whether it was still there once the arm had gone home - rather than that a function returned.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.control.pick_place import (
    PHASES,
    ArmDriver,
    PickPlace,
    PickPlaceConfig,
    bench,
    cell_world,
    gripper_press_force,
    jaw_tip_offset,
    make_scene,
    run_once,
    tcp_for_brick,
    tcp_for_foot,
    verify_placement,
)
from brick_builder.robot.patch_so101 import jaw_geometry
from brick_builder.scene.compose import SUPPLY_SPACING, grasp_pose


@pytest.fixture(scope="module")
def world():
    """One world, reused: building it costs about a second."""
    return make_scene(pad_friction=0.5)


@pytest.fixture(scope="module")
def world_2x2():
    """The same world with 2x2 bricks in the supply.

    Four studs need half the force of eight and a 32 mm footprint has nowhere near the leverage
    of a 64 mm one to stand itself up on one row of studs, so a 2x2 is the size the blind
    sequence can still seat. See `docs/guide/02-simulator-and-bricks.md`, "Brick seating with
    the stock gripper".
    """
    return make_scene(pad_friction=0.5, brick=(2, 2))


@pytest.fixture
def scene_2x2(world_2x2):
    from brick_builder.scene.compose import reset_scene

    reset_scene(world_2x2, np.random.default_rng(1))
    return world_2x2


def taught_slot(scene, brick: str) -> np.ndarray:
    """Where the supply slot is, which is what the arm aims at.

    `reset_scene` puts a supply brick exactly on its taught slot; `bench` is what adds the +-2 mm
    of jitter the arm is not told about. Reading it here and passing it as `pick_at` is the
    zero-unknown-offset case.
    """
    adr = scene.free_joint_adr(brick)
    return scene.data.qpos[adr:adr + 3].copy()


@pytest.fixture
def scene(world):
    """That world, put back to its start, so no test inherits another's brick."""
    from brick_builder.scene.compose import reset_scene

    reset_scene(world, np.random.default_rng(0))
    return world


# ------------------------------------------------------------------------- the tool geometry ---


def test_the_fixed_jaw_is_what_reaches_lowest_when_the_jaws_are_together():
    """The whole closed-jaw press rests on this: one foot touches, and it is the fixed one."""
    geo = jaw_geometry()
    fixed_centre, fixed_half = geo.fixed_pad
    moving_centre, moving_half = geo.moving_boxes[0]
    fixed_tip = fixed_centre[2] - fixed_half[2]
    moving_tip = moving_centre[2] - moving_half[2]
    assert moving_tip > fixed_tip + 0.005, (
        f"the moving jaw reaches to {moving_tip * 1e3:.1f} mm and the fixed one to "
        f"{fixed_tip * 1e3:.1f} mm in the gripper frame; they are too close to the same height "
        "for the fixed jaw to be the only thing that touches"
    )
    offset = jaw_tip_offset()
    assert offset[2] == pytest.approx(0.008, abs=1e-6), "the tip should reach 8 mm below the TCP"
    assert abs(offset[0]) > 0.02, "the foot is off to one side of the tool axis, and by a lot"


def test_the_two_tool_transforms_invert_the_grasp_and_the_foot():
    grasp = grasp_pose(2, 4)
    want = np.array([0.17, -0.02, 0.0035])
    tcp = tcp_for_brick(want, 0.0, grasp)
    # Putting the brick's origin back through the same rotation has to land on `want`.
    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert tcp + rot @ grasp.pos_in_tcp == pytest.approx(want, abs=1e-12)

    foot = np.array([0.17, -0.02, 0.028])
    tcp = tcp_for_foot(foot[:2], foot[2], 0.0)
    assert tcp[2] - jaw_tip_offset()[2] == pytest.approx(foot[2], abs=1e-12)


def test_the_supply_spacing_leaves_the_stock_jaws_room_to_get_around_one_brick(scene):
    """The bricks are spaced by what the jaws are wide, not by what looks tidy."""
    geo = jaw_geometry()
    fixed_outer = geo.fixed_boxes[0][0][0] - geo.fixed_boxes[0][1][0]
    tool_reach = abs(fixed_outer - geo.fixed_inner_x)
    spacing = SUPPLY_SPACING * scene.cfg.spec.pitch
    brick_width = 2 * scene.cfg.spec.pitch - scene.cfg.spec.clearance
    assert spacing - brick_width > 2.0 * tool_reach, (
        f"{1e3 * spacing:.1f} mm spacing only leaves "
        f"{1e3 * (spacing - brick_width) / 2:.1f} mm either side of a brick, and the jaw needs "
        f"{1e3 * tool_reach:.1f} mm"
    )
    first = scene.body_pose("supply_brick_0")[0]
    second = scene.body_pose("supply_brick_1")[0]
    assert np.linalg.norm(first[:2] - second[:2]) == pytest.approx(spacing, abs=1e-6)


# ---------------------------------------------------------------------------- the whole run ---


def test_the_scripted_sequence_runs_end_to_end_and_its_record_is_consistent(scene):
    """The whole sequence, on the real simulator, with a 2x4 and an unknown supply offset.

    What is asserted is that it ran and that its own record is true, not that the brick seated:
    a blind press with a correct servo model seats a 2x4 in 0-17 % of runs (the numbers are in
    `docs/guide/02-simulator-and-bricks.md`, "Brick seating with the stock gripper", and the low
    rate has its own test below). Closed-loop seating is `brick_builder.action`, and
    `tests/action/`.
    """
    phases = []
    result = run_once(
        scene, brick="supply_brick_0", target=(3.5, 3.5, 0),
        on_step=lambda step: phases.append(step.phase),
    )
    assert result.failure is None, result.failure
    assert result.warnings == {}, result.warnings
    assert not result.tripped
    assert phases and set(phases) <= set(PHASES)

    # The brick was carried, not shoved along the table, and the arm really did push on it
    # without ever asking for force it cannot deliver.
    assert result.lifted_by_mm > 30.0, result.lifted_by_mm
    assert result.peak_press_n > 6.0, result.peak_press_n
    assert result.held_press_n < 25.0, result.held_press_n
    assert result.peak_torque_nm <= 1.9 + 1e-9, "the servo limit was exceeded"

    # The record has to agree with the simulator read back afterwards, whatever the outcome was.
    placement = result.placement
    again = verify_placement(scene, "supply_brick_0", (3.5, 3.5, 0))
    assert again.connected == placement.connected
    assert again.at_target_cell == placement.at_target_cell
    assert again.n_engaged == placement.n_engaged
    assert again.depth_error_mm == pytest.approx(placement.depth_error_mm, abs=0.05)
    assert placement.success == (
        placement.connected and placement.at_target_cell and placement.depth_error_mm <= 0.3
    )


def test_a_2x2_with_no_unknown_offset_seats_and_stays_seated(scene_2x2):
    """The seating case the blind sequence does handle: a 2x2, aimed at the taught supply slot,
    with no error the arm was not told about. Fixed seed, and it is the run this asserts."""
    result = run_once(
        scene_2x2, brick="supply_brick_0", target=(3.5, 3.5, 0),
        pick_at=taught_slot(scene_2x2, "supply_brick_0"), rng=np.random.default_rng(1),
    )
    assert result.failure is None, result.failure
    placement = result.placement
    assert placement.connected, "the snap model never connected the brick to the plate"
    assert placement.at_target_cell, f"{placement.lateral_mm:.2f} mm from the target cell"
    assert placement.depth_error_mm <= 0.3, f"{placement.depth_error_mm:.3f} mm short of full depth"
    assert placement.n_engaged == 4, placement.n_engaged

    # Still there once nothing is holding it is the other half of the criterion.
    assert verify_placement(scene_2x2, "supply_brick_0", (3.5, 3.5, 0)).success


def test_the_run_is_a_generator_that_hands_a_recorder_the_scene_and_the_phase(scene):
    """What a video script needs: one step per control tick, in order, with the live scene."""
    job = PickPlace(scene, brick="supply_brick_0", cfg=PickPlaceConfig(press_strategy="gripped"))
    seen = []
    last_tick = 0
    for step in job.run():
        assert step.phase in PHASES, step.phase
        assert step.scene is scene
        assert step.tick == last_tick + 1
        last_tick = step.tick
        assert np.all(np.isfinite(step.tcp_pos))
        if not seen or seen[-1] != step.phase:
            seen.append(step.phase)
    assert seen[0] == "home" and seen[-1] == "done"
    for name in ("to_tray", "descend", "close", "lift", "carry", "press", "go_home"):
        assert name in seen, f"{name} never happened: {seen}"
    assert last_tick * 0.02 < 30.0, f"a clean run should not take {last_tick * 0.02:.1f} s"


def test_verify_placement_tells_a_seated_brick_from_one_cell_over():
    """The cell check has to be a check, not a formality.

    Built rather than driven to: the scene composer seats the brick exactly where it is asked
    to, so the two verdicts differ only in the target handed to `verify_placement` -- no run of
    the arm, lucky or otherwise, is involved.
    """
    from brick_builder.bricks.structures import Placement
    from brick_builder.scene.compose import SceneConfig, build_scene, reset_scene

    # A 2x4 whose footprint covers cells x=3..4, y=2..5, so its centre is cell (3.5, 3.5).
    built = build_scene(SceneConfig(
        held=None, tray=False, cameras=(),
        placed=[Placement(h=4, w=2, x=3, y=2, z=0, colour="red")],
    ))
    reset_scene(built, np.random.default_rng(0))
    brick = built.placed_names[0]

    good = verify_placement(built, brick, (3.5, 3.5, 0))
    one_over = verify_placement(built, brick, (4.5, 3.5, 0))

    assert good.success, good
    assert good.at_target_cell
    assert not one_over.at_target_cell
    # Same brick, same depths: only the cell verdict may differ.
    assert one_over.connected == good.connected
    assert one_over.depth_error_mm == pytest.approx(good.depth_error_mm, abs=1e-9)
    pitch = built.cfg.spec.pitch
    assert one_over.lateral_mm == pytest.approx(1e3 * pitch, abs=1.0)


def test_the_press_force_is_zero_when_nothing_is_touching_the_brick(scene):
    """A force measurement that reads non-zero at rest would make every other number a guess."""
    assert gripper_press_force(scene, "supply_brick_0") == 0.0


def test_the_driver_runs_the_same_multi_rate_loop_the_environments_do(scene):
    """Fifty writes, five hundred physics steps, and a readable arm at the end of it."""
    driver = ArmDriver(scene, rng=np.random.default_rng(0))
    target = scene.kin.fk(scene.home_q)[0] + np.array([0.0, 0.0, 0.01])
    before = float(scene.data.time)
    for _ in range(50):
        driver.step(target, 0.0, 0.3)
    assert driver.tick == 50
    assert float(scene.data.time) - before == pytest.approx(50 * 0.020, abs=1e-9)
    assert driver.ik_failures == 0
    assert np.all(np.isfinite(driver.reading.position))
    assert np.linalg.norm(scene.tcp_pose()[0] - target) < 0.005


def test_cell_world_lands_on_the_plate_grid(scene):
    centre = cell_world(scene, 3.5, 3.5)
    assert centre[:2] == pytest.approx(scene.cfg.plate_xy, abs=1e-9)
    assert centre[2] == pytest.approx(scene.plate_top, abs=1e-12)
    one_over = cell_world(scene, 4.5, 3.5)
    assert one_over[0] - centre[0] == pytest.approx(scene.cfg.spec.pitch, abs=1e-12)


def test_a_second_brick_goes_on_top_of_the_first_and_neither_comes_off(scene_2x2):
    """Two bricks, one on the plate and one on top of it, both seated, in one world. 2x2s at the
    taught slot, for the reason `test_a_2x2_with_no_unknown_offset_seats_and_stays_seated` gives."""
    first = run_once(scene_2x2, brick="supply_brick_0", target=(3.5, 3.5, 0),
                     pick_at=taught_slot(scene_2x2, "supply_brick_0"),
                     rng=np.random.default_rng(1))
    assert first.placement.success, first.placement

    second = run_once(scene_2x2, brick="supply_brick_1", target=(3.5, 3.5, 1),
                      pick_at=taught_slot(scene_2x2, "supply_brick_1"),
                      rng=np.random.default_rng(1))
    assert second.failure is None, second.failure
    assert second.placement.success, second.placement
    assert verify_placement(scene_2x2, "supply_brick_0", (3.5, 3.5, 0)).success, (
        "putting the second brick on shook the first one loose"
    )
    links = {(c.upper, c.lower) for c in scene_2x2.snap.connections()}
    assert ("supply_brick_0", scene_2x2.plate.name) in links, links
    assert ("supply_brick_1", "supply_brick_0") in links, links


# ------------------------------------------------------------------------------ the numbers ---


@pytest.mark.slow
def test_the_blind_sequence_seats_a_2x4_only_rarely(capsys):
    """A documented limitation, measured rather than asserted from memory.

    The servo bus used to report the motor-side encoder position, which hid the gear backlash
    from the robot; the tool arrived several millimetres off target and crept sideways into the
    cell as it pressed, dragging the brick with it, and that is what used to seat a 2x4. The
    encoder is on the output shaft (see
    `docs/guide/03-robot-and-servo-model.md`, "Gear free play and encoder position"), the tool
    now settles within about 0.4 mm of its commanded pose, and a blind press with a 0.25 mm
    capture window has no lateral motion left to find the sockets with.

    Measured rates are in `docs/guide/02-simulator-and-bricks.md`, "Brick seating with the stock
    gripper": 0-17 % for a 2x4 across the three press strategies and 0-2 mm of placement error.
    The band here is that measurement with room for thirty runs' worth of sampling noise. It is
    not a target: closed-loop visual alignment is what seats a 2x4, and it has its own tests in
    `tests/action/`.
    """
    report = bench(pad_friction=0.5, place_error=0.001, trials=30, seed0=7000)
    with capsys.disabled():
        print(
            f"\n  30 runs, jaw friction 0.5, 1 mm of placement error: "
            f"{report['successes']}/30, median depth error "
            f"{report['median_depth_error_mm']:.3f} mm, "
            f"median {report['median_seconds']:.1f} s, "
            f"peak press {report['max_peak_press_n']:.1f} N"
        )
    assert report["success_rate"] <= 0.25, (
        f"{report['successes']}/30 seated -- higher than the measured 0-17 % band; if the blind "
        "sequence really has improved, re-measure the table in "
        "docs/guide/02-simulator-and-bricks.md and move this bound with it"
    )
