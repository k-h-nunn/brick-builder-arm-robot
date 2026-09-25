"""Acceptance tests for WP-SCENE: the composed world, the grasp, and the reset.

Everything here asserts a physical fact about the compiled scene - what touches what, how far
something moves under a known load, whether two runs are the same bytes - rather than that a
function returned without raising.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.bricks.snap_api import Connection
from brick_builder.bricks.spec import DUPLO
from brick_builder.bricks.structures import Placement
from brick_builder.robot.patch_so101 import (
    ArmConfig,
    FIXED_PAD_GEOM,
    GRIPPER_BODY,
    MOVING_PAD_GEOM,
    jaw_geometry,
    tcp_position,
)
from brick_builder.scene.compose import (
    HELD_BODY,
    OVERVIEW_CAM,
    HeldBrick,
    Scene,
    SceneConfig,
    _connected_from_baseplate,
    build_scene,
    grasp_pose,
    home_tcp_target,
    jaw_angle_for_clearance,
    jaw_clearance,
    make_scene_figures,
    measure_step_rate,
    push_held_brick,
    reset_scene,
    set_brick_pose,
    settle,
    tool_down_ik,
    weld_slip_ceiling,
)

TWO_BRICKS = [Placement(2, 4, 1, 2, 0, "blue"), Placement(2, 2, 5, 3, 0, "yellow")]


@pytest.fixture(scope="module")
def built() -> Scene:
    """One structured scene, reused: building it costs about a third of a second."""
    return build_scene(SceneConfig(placed=list(TWO_BRICKS)))


def _geom(model, name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"no geom {name!r}"
    return gid


def _contacts_with(model, data, gid: int) -> list[tuple[str, float]]:
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            out.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other), float(c.dist)))
    return out


# ------------------------------------------------------------------------- the world itself ---


def test_scene_settles_for_two_seconds_with_no_warnings_and_every_brick_connected(built):
    report = settle(built, 2.0)
    assert report.warnings == {}, report.warnings
    assert report.connected_placed == len(TWO_BRICKS), report.connections
    for c in report.connections:
        if c.lower == built.plate.name:
            assert c.seated, c
    # "Settled" has to mean standing still, not merely un-exploded.
    assert report.max_speed < 5e-3, f"still moving at {report.max_speed * 1e3:.3f} mm/s"
    for name in built.placed_names:
        assert report.drift[name] < 1e-3, f"{name} drifted {report.drift[name] * 1e3:.3f} mm"


# ------------------------------------------------------------------------ multi-layer counting ---


def test_connected_from_baseplate_walks_more_than_one_layer():
    """The pure graph walk `settle()` counts with: connections chain lower -> upper, so a brick
    two hops from the plate (its own connection's `lower` is another brick, not the plate) still
    has to come out reachable, and a sibling with no connection at all must not."""

    two_layer = [Connection("B", "plate", 8, True), Connection("C", "B", 8, True)]
    assert _connected_from_baseplate("plate", two_layer) == {"plate", "B", "C"}

    three_layer = [
        Connection("B", "plate", 8, True),
        Connection("C", "B", 8, True),
        Connection("D", "C", 8, True),
    ]
    assert _connected_from_baseplate("plate", three_layer) == {"plate", "B", "C", "D"}

    # "E" is not connected to anything: resting on "B" (say) is not the same as being in this
    # list at all, since `connections()` only reports captured stud/socket pairs.
    with_a_stranger = two_layer + [Connection("F", "G", 4, False)]
    assert _connected_from_baseplate("plate", with_a_stranger) == {"plate", "B", "C"}

    # A breakable scene's socket connects to a stud's own piece body, not the brick's roof
    # (`bricks.breakable.add_breakable_brick`), so `lower` can be `'B_piece_stud_0_0'` instead of
    # `'B'`. The walk has to recognise that as the same brick.
    breakable_two_layer = [
        Connection("B", "plate", 8, True),
        Connection("C", "B_piece_stud_0_0", 1, True),
    ]
    assert _connected_from_baseplate("plate", breakable_two_layer) == {"plate", "B", "C"}

    # Once that piece's own weld has released, it is no longer structurally part of "B" (see
    # `_owning_brick`), even though `connections()` can still list it -- the snap model has no way
    # to know the stud that used to be there broke off. The walk has to drop the edge, not credit
    # "B" with carrying "C" through a piece that came away.
    assert _connected_from_baseplate(
        "plate", breakable_two_layer, released_pieces=frozenset({"B_piece_stud_0_0"}),
    ) == {"plate", "B"}


def test_settle_reports_a_two_layer_stack_as_fully_connected():
    """A brick stacked on a brick stacked on the plate: reachable in two hops, so this has to
    report 2 of 2, the case the direct-connections-only count used to answer 1 of 2 on."""

    stack = [Placement(2, 4, 2, 2, 0, "blue"), Placement(2, 4, 2, 2, 1, "yellow")]
    scene = build_scene(SceneConfig(placed=stack, held=None))
    report = settle(scene, 1.0)
    assert report.warnings == {}, report.warnings
    assert report.connected_placed == len(stack), report.connections
    assert any(c.upper == "struct_1" and c.lower == "struct_0" for c in report.connections), (
        report.connections
    )


def test_settle_reports_a_two_layer_breakable_stack_as_fully_connected():
    """The same two-layer stack, built breakable: its sockets connect to stud pieces
    (`'struct_0_piece_stud_i_j'`), not `struct_0` itself, so this is the case the graph walk has
    to map back to the brick rather than the one the plain monolithic test already covers."""

    stack = [Placement(2, 4, 2, 2, 0, "blue"), Placement(2, 4, 2, 2, 1, "yellow")]
    scene = build_scene(SceneConfig(placed=stack, held=None, breakable=True))
    report = settle(scene, 0.5)
    assert report.warnings == {}, report.warnings
    assert any("_piece_stud_" in c.lower for c in report.connections), (
        "this scene is not exercising the stud-piece naming the test is about", report.connections
    )
    assert report.connected_placed == len(stack), report.connections


def test_settle_stops_crediting_a_layer_carried_only_by_broken_off_studs():
    """The case breakable mode exists for: once the lower brick's studs have actually broken off
    (`BreakableBricks.released`), the brick they used to carry is no longer connected, even though
    the snap model can still report the loose pieces resting near their old sockets (nothing tells
    it a weld let go underneath). This is the scenario `_owning_brick` maps a piece to `None`
    for -- if it instead kept crediting the brick above with every stud-piece connection the way
    the plain graph walk used to, this would still read 2 of 2."""

    stack = [Placement(2, 4, 2, 2, 0, "blue"), Placement(2, 4, 2, 2, 1, "yellow")]
    scene = build_scene(SceneConfig(placed=stack, held=None, breakable=True))
    report = settle(scene, 0.5)
    assert report.connected_placed == len(stack), report.connections

    manager = scene.breakable
    released_any = False
    for k, (owner, piece) in enumerate(zip(manager.owner, manager.pieces)):
        if owner == "struct_0" and piece.kind == "stud":
            manager.released[k] = True
            scene.data.eq_active[manager.eq_ids[k]] = 0
            released_any = True
    assert released_any, "no stud pieces found on struct_0 -- the scene stopped matching this test"

    report2 = settle(scene, 0.5)
    assert any("_piece_stud_" in c.lower for c in report2.connections), (
        "the loose studs should still show up in connections() -- that is what this test is "
        "checking settle() sees through", report2.connections
    )
    assert report2.connected_placed == 1, report2.connections  # struct_0 only now


def test_settle_reports_a_three_layer_stack_as_fully_connected():
    """Three hops from the plate: the same walk has to keep going past the first one."""

    stack = [
        Placement(2, 4, 2, 2, 0, "blue"),
        Placement(2, 4, 2, 2, 1, "yellow"),
        Placement(2, 4, 2, 2, 2, "red"),
    ]
    # A hover the arm can actually reach over a three-tall stack at this plate radius; the
    # scene's own home-pose search puts the tool above the arm's reach here (see
    # `home_tcp_target`), which is a kinematics question, not what this test is about.
    scene = build_scene(SceneConfig(placed=stack, held=None, home_tcp=(0.17, 0.0, 0.07)))
    report = settle(scene, 1.0)
    assert report.warnings == {}, report.warnings
    assert report.connected_placed == len(stack), report.connections
    assert report.max_speed < 5e-3, f"still moving at {report.max_speed * 1e3:.3f} mm/s"


def test_a_brick_resting_on_stud_tops_without_capture_is_not_counted_connected():
    """A second *placed* brick moved half a stud pitch off-grid rests on top of the first
    brick's stud tops instead of seating (`tests/bricks/test_generator.py`'s misaligned-insertion
    case) and never engages a socket. It has to be one of `scene.placed_names` for this to be a
    real test of `settle()`'s count -- a rider outside `placed` (a supply or tray brick) is never
    counted connected either way, so `connected_placed == 1` could not fail on its account."""

    # The rider (2 studs by 4) crosses the base (4 by 2): its footprint sits entirely within the
    # base's footprint in x, but overhangs it by one full pitch (16 mm) at each end in y. That
    # overhang does not topple it -- the base's stud grid still sits under its centre of mass --
    # so lining the footprints up exactly is not needed to make the point.
    base = [Placement(2, 4, 1, 2, 0, "blue"), Placement(4, 2, 1, 4, 0, "yellow")]
    scene = build_scene(SceneConfig(placed=base, held=None))
    base_name, rider_name = scene.placed_names
    assert len(base) == 2

    base_pos, base_quat = scene.body_pose(base_name)
    spec = scene.cfg.spec
    # 1 mm above the base's roof -- 3.5 mm *below* the stud tops (`stud_h` is 4.5 mm) -- so the
    # rider starts penetrating the studs rather than being lowered onto them; contact correction
    # is what pushes it back up to rest on the tops over the following steps. A real lowering,
    # started above the stud tops instead, does not produce the numbers checked below (measured
    # separately: it slides further and settles close to flush rather than clearly on the studs),
    # so this penetrating start is deliberate, not an oversight.
    rider_start = np.array([base_pos[0] + 0.5 * spec.pitch, base_pos[1], base_pos[2] + spec.height + 0.001])
    set_brick_pose(scene, rider_name, rider_start, base_quat)
    # `set_brick_pose` only writes `qpos`; it does not recompute forward kinematics, so
    # `data.xpos`/`data.site_xpos` stay at the rider's *old*, seated place until something calls
    # `mj_forward`. Skip this and the first `snap.update()` inside `settle()`'s `scene.step()`
    # reads the stale site position, mistakes the move for the rider's stud pulling out of its old
    # socket, and yanks it back towards that socket instead of letting it simply rest where it was
    # put (measured: dx 15.2 mm, dz -0.7 mm, quat_w 0.981 -- tipped, not resting). `snap.bind()`
    # clears captured anchors but does not touch any body's kinematics and changes nothing here
    # (measured with and without it, `mj_forward` present both times: same dx/dz to within 0.05
    # mm), so it is not called.
    mujoco.mj_forward(scene.model, scene.data)

    report = settle(scene, 1.0)
    assert report.warnings == {}, report.warnings
    assert report.connected_placed == 1, report.connections  # the base only, not the rider
    assert all(c.upper != rider_name for c in report.connections), report.connections

    rider_pos, _ = scene.body_pose(rider_name)
    # Resting on the stud tops sits it one stud height above the flush-seated position, not
    # falling through and not sliding off.
    seated_z = base_pos[2] + spec.height
    assert seated_z + 0.6 * spec.stud_h < rider_pos[2] < seated_z + 1.4 * spec.stud_h, rider_pos
    assert abs(rider_pos[0] - rider_start[0]) < 0.001, "the rider drifted sideways off its offset"


# -------------------------------------------------------------------------- the weld's ceiling ---


def test_the_weld_grip_slips_above_its_force_ceiling():
    """The soft weld holds at any force by construction; `weld_slip_force` gives it the same
    ceiling the stock jaws' friction grip has, and a load past it must let go."""

    ceiling = weld_slip_ceiling(SceneConfig())
    scene = build_scene(SceneConfig(weld_slip_force=ceiling))
    scene.step(200)
    assert scene.data.eq_active[scene.weld_eq] == 1

    report = push_held_brick(scene, force=2.0 * ceiling, seconds=1.0)
    assert report["weld_released"], report
    assert scene.data.eq_active[scene.weld_eq] == 0


def test_the_weld_grip_holds_below_its_force_ceiling():
    """The same ceiling must not trip on the ordinary 5 N acceptance load."""

    ceiling = weld_slip_ceiling(SceneConfig())
    scene = build_scene(SceneConfig(weld_slip_force=ceiling))
    scene.step(200)

    report = push_held_brick(scene, force=5.0, seconds=1.0)
    assert not report["weld_released"], report
    assert scene.data.eq_active[scene.weld_eq] == 1


def test_the_weld_grip_never_slips_when_no_ceiling_is_set():
    """`weld_slip_force` defaults to `None`: the weld has to keep holding at a load far past any
    physically grounded ceiling, which is the behaviour every default weld-mode scene still gets."""

    scene = build_scene(SceneConfig())
    assert scene.cfg.weld_slip_force is None
    scene.step(200)

    report = push_held_brick(scene, force=150.0, seconds=1.0)
    assert not report["weld_released"], report
    assert scene.data.eq_active[scene.weld_eq] == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "weld_slip_force only takes effect through Scene.step or push_held_brick, the only two "
        "callers of _release_overloaded_weld (see its docstring for the full list of loops that "
        "call mujoco.mj_step directly and so never reach it). This asserts the behaviour that "
        "would actually be wanted -- the ceiling holding even on a loop that steps the model "
        "directly -- so it starts failing, rather than staying silently green, the day one of "
        "those loops (or a scene-side stepping hook) closes that gap; at that point this test "
        "should be rewritten as a real assertion instead of an expected failure."
    ),
)
def test_the_ceiling_still_applies_to_a_loop_that_steps_mujoco_directly():
    ceiling = weld_slip_ceiling(SceneConfig())
    scene = build_scene(SceneConfig(weld_slip_force=ceiling))
    scene.step(200)

    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.held_name)
    scene.data.xfrc_applied[bid, :3] = [0.0, 0.0, -2.0 * ceiling]
    for _ in range(500):
        scene.snap.update(scene.model, scene.data)
        mujoco.mj_step(scene.model, scene.data)  # note: no `_release_overloaded_weld` call
    scene.data.xfrc_applied[bid, :3] = 0.0
    assert scene.data.eq_active[scene.weld_eq] == 0, "a raw mj_step loop should still see the ceiling"


def test_push_held_brick_never_reports_a_weld_release_in_friction_mode():
    """`weld_released` means *this call* turned an active weld off. In friction mode the weld was
    never active (`eq_active` is 0 from the reset on), so a friction-mode pull must not claim a
    release that never happened."""

    scene = build_scene(SceneConfig(grasp_mode="friction"))
    scene.step(200)
    assert scene.data.eq_active[scene.weld_eq] == 0

    report = push_held_brick(scene, force=1.0, seconds=0.2)
    assert not report["weld_released"], report


def test_the_world_has_the_frame_the_spec_fixes(built):
    """Arm base at the origin, table top at z = 0, plate centre inside 0.10-0.24 m of the base."""
    model = built.model
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert np.allclose(model.body_pos[base], 0.0, atol=1e-9)

    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, built.plate.name)
    plate_xy = model.body_pos[plate_id][:2]
    assert np.allclose(plate_xy, built.cfg.plate_xy)
    half = 0.5 * built.plate.nx * DUPLO.pitch
    assert 0.10 <= plate_xy[0] - half and plate_xy[0] + half <= 0.24, plate_xy

    assert abs(built.plate_top - 2.0 * DUPLO.roof) < 1e-12
    assert model.body_pos[plate_id][2] == 0.0


def test_the_named_cameras_exist_and_render_the_instance_map(built):
    from brick_builder.render.renderer import SceneRenderer

    with SceneRenderer(built.model, width=160, height=120) as renderer:
        renderer.instance_lut(built.instance_ids)
        frame = renderer.render(built.data, OVERVIEW_CAM)
    held_id = built.instance_ids[HELD_BODY]
    plate_id = built.instance_ids[built.plate.name]
    assert (frame.instance == held_id).sum() > 20, "the held brick is not visible from overview_cam"
    assert (frame.instance == plate_id).sum() > 100
    assert (frame.instance == -1).any(), "nothing is background, so the camera is inside something"


def test_the_tool_points_straight_down_at_the_home_pose(built):
    # Read the commanded pose off the kinematics, not off `data`: a settled arm has drooped a few
    # millimetres under its own weight, which is the servo behaving correctly, not a bad home pose.
    _, rot = built.kin.fk(built.home_q)
    # The TCP site's z axis is the approach direction.
    assert rot[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=2e-3)
    # and its x axis is the brick's width axis, lined up with the plate grid at home_yaw = 0.
    assert rot[:, 0] == pytest.approx([1.0, 0.0, 0.0], abs=2e-3)

    # The droop is worth pinning down too: it is what the scripted primitive has to anticipate.
    reset_scene(built, np.random.default_rng(0))
    commanded, _ = built.tcp_pose()
    settle(built, 2.0)
    settled, _ = built.tcp_pose()
    assert np.linalg.norm(settled - commanded) < 0.015, "the arm sagged more than 15 mm"


def test_tool_down_ik_hits_the_position_and_the_yaw_it_is_asked_for(built):
    from brick_builder.robot.kinematics import euler_zyx

    target = home_tcp_target(built) + np.array([0.01, 0.015, 0.02])
    for yaw in (0.0, math.radians(20.0), math.radians(-35.0)):
        q = tool_down_ik(built.kin, target, yaw, q0=built.home_q)
        pos, rot = built.kin.fk(q)
        achieved_yaw, pitch, _ = euler_zyx(rot)
        assert np.linalg.norm(pos - target) < 1e-4, f"{yaw}: {np.linalg.norm(pos - target) * 1e3} mm"
        assert abs(math.remainder(achieved_yaw - yaw, 2 * math.pi)) < math.radians(0.1)
        assert abs(pitch) < math.radians(0.1)


def test_an_unreachable_pose_is_reported_not_silently_clipped(built):
    with pytest.raises(RuntimeError, match="no tool-down pose"):
        tool_down_ik(built.kin, np.array([0.05, 0.0, 0.35]), 0.0)


def test_invalid_input_is_rejected_before_anything_is_built():
    with pytest.raises(ValueError, match="unsupported|unknown camera"):
        build_scene(SceneConfig(cameras=("wrist_cam", "nope"), held=None))
    with pytest.raises(ValueError, match="invalid structure"):
        build_scene(SceneConfig(placed=[Placement(2, 2, 0, 0, 3, "red")], held=None))
    with pytest.raises(ValueError, match="invalid structure"):
        build_scene(SceneConfig(placed=[Placement(2, 2, 40, 0, 0, "red")], held=None))


# ------------------------------------------------------------------------------- the grasp ---


def test_the_grasp_pose_is_the_one_the_jaw_geometry_implies():
    grasp = grasp_pose(2, 4, DUPLO)
    geo = jaw_geometry()

    # Across the width, flush against the fixed pad's working face.
    assert grasp.width == pytest.approx(2 * DUPLO.pitch - DUPLO.clearance)
    assert grasp.pos_in_gripper[0] == pytest.approx(geo.fixed_inner_x + 0.5 * grasp.width)

    # The jaws really do just touch the brick's top outer edge at `gripper_angle`.
    assert jaw_clearance(grasp.contact_point, grasp.gripper_angle) == pytest.approx(0.0, abs=1e-9)
    # ... and the brick's bottom outer edge is clear, because the gap opens towards the tips.
    bottom = grasp.contact_point - np.array([0.0, 0.0, DUPLO.height])
    assert jaw_clearance(bottom, grasp.gripper_angle) > 1e-3

    # The TCP-to-brick transform round-trips through the TCP frame.
    from_tcp = np.diag([1.0, -1.0, -1.0]) @ grasp.pos_in_tcp + tcp_position()
    assert from_tcp == pytest.approx(grasp.pos_in_gripper, abs=1e-12)
    # With the tool down the brick's bottom face is below the TCP, so the offset points along +z
    # of the site frame (which is the approach direction).
    assert grasp.pos_in_tcp[2] > 0.0

    # The brick hangs past the fingertips: otherwise they would hit the plate before it seated.
    pad_centre, pad_half = geo.fixed_pad
    tip_z = pad_centre[2] - pad_half[2]
    assert grasp.pos_in_gripper[2] == pytest.approx(tip_z - grasp.tip_clearance)
    assert grasp.tip_clearance > 0.005


def test_the_jaw_gap_really_does_open_towards_the_fingertips():
    """The fact the whole grasp construction turns on, measured rather than assumed."""
    geo = jaw_geometry()
    grasp = grasp_pose(2, 4, DUPLO)
    pad_centre, pad_half = geo.fixed_pad
    narrow = np.array([geo.fixed_inner_x + grasp.width, pad_centre[1],
                       geo.moving_inner_point[1] + geo.jaw_pos[2]])
    tip = np.array([geo.fixed_inner_x + grasp.width, pad_centre[1],
                    pad_centre[2] - pad_half[2]])
    angle = jaw_angle_for_clearance(narrow, 0.0)
    assert jaw_clearance(tip, angle) > 0.006, "the tips should be several mm wider than the top"


def test_the_weld_grasp_holds_a_five_newton_downward_load():
    scene = build_scene(SceneConfig(grasp_mode="weld"))
    scene.step(500)
    report = push_held_brick(scene, force=5.0, seconds=5.0)
    assert report["warnings"] == {}
    assert report["peak_slip_m"] < 0.001, f"{report['peak_slip_m'] * 1e3:.3f} mm"


def test_the_weld_grasp_is_compliant_and_not_a_rigid_bolt():
    """The spec asks for a *soft* weld: it has to give measurably, or it is not standing in for
    a grip at all."""
    scene = build_scene(SceneConfig(grasp_mode="weld"))
    scene.step(500)
    report = push_held_brick(scene, force=5.0, seconds=1.0)
    assert report["peak_slip_m"] > 2e-5, "a 5 N load should move the brick at least 20 microns"


def test_the_friction_grasp_holds_the_press_direction_by_wedging_not_by_friction():
    """Upward is the direction a press loads the grip in, and the wedge closes on it.

    The number this used to assert (0.48 mm) was measured with a part on the fixed jaw that does
    not exist: a ledge over the brick's roof took most of a 5 N upward pull.  With the stock jaws
    the same pull moves the brick 1.04 mm, which misses the project's 1 mm acceptance target -
    but *only just*, and it misses it by the same amount whether the plastic-on-plastic
    coefficient is 0.3 or 1.0.  That is the finding worth pinning down: upwards the brick is held
    by the gap narrowing on it, not by friction, so the grip does not fall apart at the low
    coefficient a printed jaw on an ABS brick really has.  Downwards it does - see the strict
    xfail below.
    """

    slips = {}
    for mu in (0.3, 1.0):
        scene = build_scene(SceneConfig(grasp_mode="friction", arm=ArmConfig(pad_friction=mu)))
        scene.step(500)
        report = push_held_brick(scene, force=5.0, seconds=5.0, direction=(0.0, 0.0, 1.0))
        assert report["warnings"] == {}, (mu, report["warnings"])
        slips[mu] = report["slip_m"]
    assert max(slips.values()) < 0.0015, slips
    assert abs(slips[0.3] - slips[1.0]) < 0.0003, (
        f"upward slip went from {slips[0.3] * 1e3:.2f} mm at a coefficient of 0.3 to "
        f"{slips[1.0] * 1e3:.2f} mm at 1.0, so friction is carrying it after all"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured 2.08 mm against a 1.0 mm target, and 28.4 mm once the plastic-on-plastic "
        "coefficient is the 0.3 a printed jaw on an ABS brick plausibly has. The stock SO-101 "
        "jaw is a downward-opening wedge: at the angle that leaves 31.8 mm at the narrow end of "
        "the gap it leaves 39.9 mm at the tips, and the brick has to be held near the tips "
        "because otherwise they reach 6.7 mm below its bottom face and would hit the baseplate "
        "before the brick seated. Pinching harder pushes the brick out faster, not less: with no "
        "load at all the brick creeps 0.85 mm in 5 s at 18.8 N of pinch and 1.22 mm at 33 N. "
        "The hardware is a stock kit and stays one, so this is a fact to build around rather "
        "than a fault to fix: the same grasp holds 5 N *upwards* in about 1 mm at every "
        "coefficient (the wedge closes on the brick), which is the direction a press loads it, "
        "and brick_builder.control.pick_place seats bricks with it. Do not hang anything heavy off a "
        "held brick, and let go of it before the arm leans on anything."
    ),
)
def test_the_friction_grasp_holds_a_five_newton_downward_load():
    scene = build_scene(SceneConfig(grasp_mode="friction"))
    scene.step(500)
    report = push_held_brick(scene, force=5.0, seconds=5.0)
    assert report["warnings"] == {}
    assert report["peak_slip_m"] < 0.001, f"{report['peak_slip_m'] * 1e3:.3f} mm"


def test_the_friction_grasp_squeezes_the_brick_between_both_pads():
    scene = build_scene(SceneConfig(grasp_mode="friction"))
    scene.step(500)
    model, data = scene.model, scene.data
    pads = {_geom(model, FIXED_PAD_GEOM), _geom(model, MOVING_PAD_GEOM)}
    held_geoms = set()
    normal = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in pads or c.geom2 in pads:
            pad = c.geom1 if c.geom1 in pads else c.geom2
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            normal += abs(f[0])
            held_geoms.add(pad)
    assert held_geoms == pads, "only one pad is touching the brick"
    weight = 0.0125 * 9.81
    assert normal > 10.0 * weight, f"only {normal:.2f} N of pinch"


# -------------------------------------------------------------------------------- the reset ---


def _state(scene: Scene) -> np.ndarray:
    return np.concatenate([scene.data.qpos.copy(), scene.data.qvel.copy()])


def test_reset_is_bit_reproducible_under_a_seed():
    cfg = SceneConfig(
        placed=list(TWO_BRICKS),
        held=HeldBrick(offset_jitter=(0.002, 0.002, 0.001), yaw_jitter=math.radians(4.0)),
    )
    scene = build_scene(cfg)

    reset_scene(scene, np.random.default_rng(11))
    first_offset = scene.in_hand_offset.copy()
    scene.step(100)
    first = _state(scene)

    reset_scene(scene, np.random.default_rng(11))
    assert np.array_equal(scene.in_hand_offset, first_offset)
    scene.step(100)
    assert np.array_equal(_state(scene), first), "same seed gave a different trajectory"

    reset_scene(scene, np.random.default_rng(12))
    assert not np.allclose(scene.in_hand_offset, first_offset), "the jitter never fired"


def test_the_in_hand_offset_actually_moves_the_brick():
    scene = build_scene(SceneConfig(held=HeldBrick()))
    reset_scene(scene, np.random.default_rng(0))
    nominal, _ = scene.body_pose(HELD_BODY)

    shifted = build_scene(SceneConfig(held=HeldBrick(offset=(0.003, -0.002, 0.0),
                                                     yaw=math.radians(5.0))))
    reset_scene(shifted, np.random.default_rng(0))
    moved, quat = shifted.body_pose(HELD_BODY)
    _, tcp_rot = shifted.tcp_pose()
    delta = tcp_rot.T @ (moved - nominal)
    assert delta[:2] == pytest.approx([0.003, -0.002], abs=2e-4)

    ref_quat = scene.body_pose(HELD_BODY)[1]
    rel = np.zeros(4)
    conj = np.zeros(4)
    mujoco.mju_negQuat(conj, ref_quat)
    mujoco.mju_mulQuat(rel, conj, quat)
    angle = 2.0 * math.acos(min(1.0, abs(rel[0])))
    assert angle == pytest.approx(math.radians(5.0), abs=math.radians(0.5))


def test_reset_puts_everything_back_after_a_long_run(built):
    reset_scene(built, np.random.default_rng(0))
    start = _state(built)
    built.step(600)
    assert not np.allclose(_state(built), start), "600 steps changed nothing, so nothing moved"
    reset_scene(built, np.random.default_rng(0))
    assert np.array_equal(_state(built), start)
    assert built.snap.connections() != [], "the snap state was not rebuilt by the reset"


def test_reset_clears_the_snap_state_when_a_brick_has_been_pulled_off():
    scene = build_scene(SceneConfig(placed=[Placement(2, 4, 3, 2, 0, "blue")], held=None))
    settle(scene, 0.3)
    assert scene.snap.connections() != []

    body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "struct_0")
    for _ in range(300):
        scene.data.xfrc_applied[body, 2] = 60.0
        scene.step(1)
    scene.data.xfrc_applied[body, 2] = 0.0
    assert scene.snap.connections() == [], "60 N should have torn a 2x4 off the plate"

    reset_scene(scene, np.random.default_rng(0))
    settle(scene, 0.3)
    assert scene.snap.connections() != []


# ----------------------------------------------------------------------------- the variants ---


def test_a_rotated_plate_carries_its_bricks_round_with_it():
    yaw = math.radians(12.0)
    scene = build_scene(SceneConfig(placed=list(TWO_BRICKS), plate_yaw=yaw, held=None))
    report = settle(scene, 1.0)
    assert report.warnings == {}
    assert report.connected_placed == len(TWO_BRICKS), report.connections

    pos, quat = scene.body_pose("struct_0")
    assert 2.0 * math.atan2(quat[3], quat[0]) == pytest.approx(yaw, abs=math.radians(0.5))
    centre = np.array(scene.cfg.plate_xy)
    # The brick is off the plate's centre line, so a 12 degree turn has to have moved it in y.
    assert abs(pos[1] - centre[1]) > 0.002


def test_the_detent_snap_model_connects_the_same_bricks():
    scene = build_scene(SceneConfig(placed=list(TWO_BRICKS), snap_model="detent", held=None))
    report = settle(scene, 1.0)
    assert report.warnings == {}
    assert report.connected_placed == len(TWO_BRICKS), report.connections


def test_a_scene_with_no_held_brick_has_no_weld_and_no_grasp():
    scene = build_scene(SceneConfig(held=None))
    assert scene.weld_eq == -1
    assert scene.grasp is None
    report = settle(scene, 0.5)
    assert report.warnings == {}


def test_the_tray_bricks_stay_in_the_tray():
    scene = build_scene(SceneConfig(tray=True, held=None))
    report = settle(scene, 1.0)
    assert report.warnings == {}
    tray_x, tray_y = scene.cfg.tray_xy
    for name in scene.bricks:
        if not name.startswith("tray_brick_"):
            continue
        pos, _ = scene.body_pose(name)
        assert abs(pos[1] - tray_y) < 0.03, f"{name} left the tray sideways"
        assert 0.0 < pos[2] < 0.01, f"{name} is at z = {pos[2]:.4f}"
        assert report.drift[name] < 0.002


def test_a_three_stud_wide_brick_builds_and_is_held_across_its_width():
    scene = build_scene(SceneConfig(held=HeldBrick(nx=3, ny=4)))
    assert settle(scene, 0.3).warnings == {}
    assert scene.grasp.width == pytest.approx(3 * DUPLO.pitch - DUPLO.clearance)


# ------------------------------------------------------------------------------- the figures ---


def test_the_media_figures_render(tmp_path):
    out = make_scene_figures(tmp_path)
    assert set(out) == {"overview.png", "wrist_view.png", "grasp_closeup.png"}
    for path in out.values():
        from pathlib import Path

        assert Path(path).stat().st_size > 10_000


@pytest.mark.slow
def test_physics_step_rate_is_reported(built, capsys):
    """Not a pass/fail on speed - the number is what the report needs."""
    rate = measure_step_rate(built, steps=3000)
    with capsys.disabled():
        print(f"\n  default scene (arm + 8x8 plate + held 2x4 + 2 placed): {rate:.0f} physics steps/s")
    assert rate > 100.0


def test_the_held_bricks_own_grasp_mode_overrides_the_scene_default():
    """`HeldBrick.grasp_mode` is what an env sets per episode; it has to win over the config."""
    scene = build_scene(
        SceneConfig(grasp_mode="weld", held=HeldBrick(grasp_mode="friction"))
    )
    assert scene.data.eq_active[scene.weld_eq] == 0, "the weld should be off in friction mode"
    scene.step(500)
    model, data = scene.model, scene.data
    pads = {_geom(model, FIXED_PAD_GEOM), _geom(model, MOVING_PAD_GEOM)}
    touching = {
        c.geom1 if c.geom1 in pads else c.geom2
        for c in (data.contact[i] for i in range(data.ncon))
        if c.geom1 in pads or c.geom2 in pads
    }
    assert touching == pads

    welded = build_scene(SceneConfig(grasp_mode="friction", held=HeldBrick(grasp_mode="weld")))
    assert welded.data.eq_active[welded.weld_eq] == 1
