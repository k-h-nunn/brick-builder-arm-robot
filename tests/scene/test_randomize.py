"""Acceptance tests for WP-SCENE's domain randomization.

The spec's bar: every group has to measurably change the model field it names, and 1 000 samples
have to stay inside the published range.  Both are tested against the compiled model, not against
the dictionary the sampler returned.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.bricks.structures import Placement
from brick_builder.robot.patch_so101 import ARM_JOINTS, FIXED_PAD_GEOM, MOVING_PAD_GEOM
from brick_builder.scene import randomize as R
from brick_builder.scene.compose import HELD_BODY, SceneConfig, build_scene, reset_scene, settle

PLACED = [Placement(2, 4, 1, 2, 0, "blue"), Placement(2, 2, 5, 3, 0, "yellow")]


@pytest.fixture
def scene():
    return build_scene(SceneConfig(placed=list(PLACED)))


def _dofs(model) -> np.ndarray:
    return np.asarray(
        [
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ARM_JOINTS
        ]
    )


def _snapshot(model) -> dict:
    return {
        "gain": model.actuator_gainprm.copy(),
        "bias": model.actuator_biasprm.copy(),
        "forcerange": model.actuator_forcerange.copy(),
        "damping": model.dof_damping.copy(),
        "frictionloss": model.dof_frictionloss.copy(),
        "armature": model.dof_armature.copy(),
        "jnt_range": model.jnt_range.copy(),
        "body_mass": model.body_mass.copy(),
        "geom_friction": model.geom_friction.copy(),
        "geom_size": model.geom_size.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "body_pos": model.body_pos.copy(),
        "body_quat": model.body_quat.copy(),
        "light_pos": model.light_pos.copy(),
        "cam_pos": model.cam_pos.copy(),
        "cam_fovy": model.cam_fovy.copy(),
        "eq_solref": model.eq_solref.copy(),
    }


# ----------------------------------------------------------------------------- the sampling ---


def test_every_sampled_value_stays_inside_its_published_range():
    rng = np.random.default_rng(0)
    seen = {k: [] for k in R.RANGES}
    for _ in range(1000):
        s = R.sample(rng, 1.0)
        for key, value in s.items():
            if key == "snap.curve":
                assert value in R.SNAP_CURVES
                continue
            lo, hi = R.RANGES[key]
            assert lo - 1e-12 <= value <= hi + 1e-12, f"{key} = {value} outside [{lo}, {hi}]"
            seen[key].append(value)
    # ... and actually uses the range, rather than sitting on the nominal value.
    for key, values in seen.items():
        lo, hi = R.RANGES[key]
        span = max(values) - min(values)
        assert span > 0.5 * (hi - lo), f"{key} only spanned {span} of {hi - lo}"


def test_every_range_has_a_nominal_value_inside_it():
    for key, (lo, hi) in R.RANGES.items():
        assert key in R.NOMINAL, f"{key} has a range but no nominal value"
        assert lo <= R.NOMINAL[key] <= hi, f"{key}: nominal {R.NOMINAL[key]} outside [{lo}, {hi}]"
    assert set(R.NOMINAL) == set(R.RANGES)


def test_level_zero_returns_the_nominal_value_of_everything():
    s = R.sample(np.random.default_rng(5), 0.0)
    for key, value in s.items():
        assert value == pytest.approx(R.NOMINAL[key], rel=1e-12, abs=1e-15), key
    assert "snap.curve" not in s, "level 0 must not change the force-curve shape either"


def test_level_interpolates_between_the_nominal_and_the_full_range():
    half = np.array([R.sample(np.random.default_rng(i), 0.5)["plate.dx"] for i in range(200)])
    full = np.array([R.sample(np.random.default_rng(i), 1.0)["plate.dx"] for i in range(200)])
    assert np.allclose(half, 0.5 * full)
    assert np.abs(half).max() < np.abs(full).max()


def test_the_same_seed_gives_the_same_sample():
    a = R.sample(np.random.default_rng(42), 1.0)
    b = R.sample(np.random.default_rng(42), 1.0)
    assert a == b
    c = R.sample(np.random.default_rng(43), 1.0)
    assert a != c


def test_the_clutch_force_is_drawn_log_uniformly():
    rng = np.random.default_rng(1)
    draws = np.array([R.sample_snap(rng, 1.0)["snap.f_clutch"] for _ in range(4000)])
    lo, hi = R.RANGES["snap.f_clutch"]
    # A log-uniform draw puts half its mass below the geometric mean, a uniform one would put
    # only ~35% there for this range.
    below = float(np.mean(draws < math.sqrt(lo * hi)))
    assert 0.45 < below < 0.55, below


def test_an_unknown_group_is_refused():
    with pytest.raises(ValueError, match="unknown randomization group"):
        R.sample(np.random.default_rng(0), 1.0, groups=("dynamics", "gremlins"))


# --------------------------------------------------------- each group changes what it names ---


def test_the_dynamics_group_changes_every_field_it_names(scene):
    before = _snapshot(scene.model)
    s = R.sample_dynamics(np.random.default_rng(2), 1.0)
    R.apply(scene, s)
    model = scene.model
    dofs = _dofs(model)
    n = len(ARM_JOINTS)

    assert model.actuator_gainprm[0, 0] == pytest.approx(s["dynamics.kp"])
    assert model.actuator_biasprm[0, 1] == pytest.approx(-s["dynamics.kp"])
    assert model.actuator_forcerange[:n, 1] == pytest.approx(s["dynamics.forcerange"])
    assert model.dof_damping[dofs] == pytest.approx(s["dynamics.damping"])
    assert model.dof_frictionloss[dofs] == pytest.approx(s["dynamics.frictionloss"])
    assert model.dof_armature[dofs] == pytest.approx(s["dynamics.armature"])

    backlash = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_backlash") for j in ARM_JOINTS
    ]
    assert model.jnt_range[backlash, 1] == pytest.approx(
        math.radians(s["dynamics.backlash_deg"])
    )

    links = np.unique(model.jnt_bodyid[[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS
    ]])
    assert model.body_mass[links] == pytest.approx(
        before["body_mass"][links] * s["dynamics.link_mass_scale"]
    )
    # and nothing that is not a link changed mass
    other = np.setdiff1d(np.arange(model.nbody), links)
    assert np.array_equal(model.body_mass[other], before["body_mass"][other])


def test_the_command_group_reaches_the_servo_and_bus_configs():
    s = R.sample_command(np.random.default_rng(3), 1.0)
    s["dynamics.vin"] = 7.0
    servo = R.servo_config(s)
    assert servo.deadband_ticks == round(s["command.deadband_ticks"])
    assert servo.rate_limit == pytest.approx(s["command.rate_limit"])
    assert servo.action_delay_s == pytest.approx(s["command.action_delay_s"])
    assert servo.read_skew_s == pytest.approx(s["command.read_skew_s"])
    assert servo.vin == pytest.approx(7.0)

    bus = R.bus_config(s)
    assert bus.packet_loss == pytest.approx(s["command.packet_loss"])
    assert bus.period_jitter_s == pytest.approx(s["command.period_jitter_s"])


def test_a_randomized_command_path_actually_changes_the_bus_timing():
    slow = R.servo_config({"command.action_delay_s": 0.03, "command.rate_limit": 4.0})
    fast = R.servo_config({"command.action_delay_s": 0.0, "command.rate_limit": 6.0})
    assert slow.action_delay_s > fast.action_delay_s
    assert slow.rate_limit < fast.rate_limit


def test_the_snap_group_changes_the_snap_models_parameters(scene):
    before = scene.snap.params
    s = R.sample_snap(np.random.default_rng(4), 1.0)
    R.apply(scene, s)
    after = scene.snap.params
    assert after.f_clutch == pytest.approx(s["snap.f_clutch"])
    assert after.beta == pytest.approx(s["snap.beta"])
    assert after.exponent == pytest.approx(s["snap.exponent"])
    assert after.capture_tilt_deg == pytest.approx(s["snap.capture_tilt_deg"])
    assert after.f_clutch != before.f_clutch
    if "snap.curve" in s:
        assert after.curve == s["snap.curve"]
    # Nothing outside the sampled keys moved.
    assert after.collet_mass == before.collet_mass
    assert after.snap_every == before.snap_every


def test_the_contact_group_changes_the_friction_and_the_stud_size(scene):
    before = _snapshot(scene.model)
    s = R.sample_contact(np.random.default_rng(6), 1.0)
    R.apply(scene, s)
    model = scene.model

    for pad in (FIXED_PAD_GEOM, MOVING_PAD_GEOM):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad)
        assert model.geom_friction[gid, 0] == pytest.approx(s["contact.pad_friction"])

    brick_geoms = R.brick_geoms(scene)
    assert brick_geoms.size > 50
    assert model.geom_friction[brick_geoms, 0] == pytest.approx(s["contact.brick_friction"])

    studs = R.stud_geoms(scene)
    assert studs.size > 60, "the plate alone has 64 studs"
    assert model.geom_size[studs, 0] == pytest.approx(
        before["geom_size"][studs, 0] + s["contact.stud_radius_delta"]
    )


def test_the_plate_group_moves_the_plate_and_the_bricks_on_it(scene):
    before_pos, _ = scene.body_pose("struct_0")
    s = R.sample_plate(np.random.default_rng(8), 1.0)
    R.apply(scene, s)
    reset_scene(scene, np.random.default_rng(0))

    plate_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    assert scene.model.body_pos[plate_id][0] == pytest.approx(
        scene.cfg.plate_xy[0] + s["plate.dx"]
    )
    assert scene.model.body_pos[plate_id][1] == pytest.approx(
        scene.cfg.plate_xy[1] + s["plate.dy"]
    )
    yaw = 2.0 * math.atan2(scene.model.body_quat[plate_id][3], scene.model.body_quat[plate_id][0])
    assert yaw == pytest.approx(s["plate.dyaw"], abs=1e-9)

    # The brick came with it, and is still connected after settling.
    after_pos, _ = scene.body_pose("struct_0")
    assert not np.allclose(after_pos, before_pos)
    report = settle(scene, 1.0)
    assert report.warnings == {}
    assert report.connected_placed == len(PLACED), report.connections


def test_the_in_hand_group_moves_the_brick_in_the_hand_and_softens_the_weld(scene):
    reset_scene(scene, np.random.default_rng(0))
    nominal, _ = scene.body_pose(HELD_BODY)

    s = R.sample_in_hand(np.random.default_rng(9), 1.0)
    R.apply(scene, s)
    reset_scene(scene, np.random.default_rng(0))

    assert scene.in_hand_offset == pytest.approx(
        [s["in_hand.dx"], s["in_hand.dy"], s["in_hand.dz"], s["in_hand.dyaw"]]
    )
    moved, _ = scene.body_pose(HELD_BODY)
    _, tcp_rot = scene.tcp_pose()
    delta = tcp_rot.T @ (moved - nominal)
    assert delta[:2] == pytest.approx([s["in_hand.dx"], s["in_hand.dy"]], abs=3e-4)

    assert scene.model.eq_solref[scene.weld_eq, 0] == pytest.approx(s["in_hand.weld_timeconst"])
    assert scene.weld_torquescale == pytest.approx(s["in_hand.weld_torquescale"])
    assert scene.model.eq_data[scene.weld_eq, 10] == pytest.approx(scene.weld_torquescale)


def test_the_visual_group_changes_lights_cameras_and_brick_colour(scene):
    before = _snapshot(scene.model)
    s = R.sample_visual(np.random.default_rng(10), 1.0)
    R.apply(scene, s)
    model = scene.model

    assert model.light_pos[0] == pytest.approx(
        before["light_pos"][0] + [s["visual.light_dx"], s["visual.light_dy"], s["visual.light_dz"]]
    )
    assert model.cam_pos[0] == pytest.approx(
        before["cam_pos"][0] + [s["visual.cam_dx"], s["visual.cam_dy"], s["visual.cam_dz"]]
    )
    assert model.cam_fovy[0] == pytest.approx(
        before["cam_fovy"][0] + s["visual.cam_fovy_deg"]
    )
    assert not np.allclose(model.geom_rgba, before["geom_rgba"]), "no brick changed colour"
    # Only visual geoms were recoloured; collision geoms keep whatever they had.
    changed = np.nonzero(~np.isclose(model.geom_rgba, before["geom_rgba"]).all(axis=1))[0]
    for gid in changed:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
        assert "_vis_" in name, name

    post = R.postprocess_config(s)
    assert post.exposure == pytest.approx(s["visual.post_exposure"])
    assert post.k1 == pytest.approx(s["visual.post_k1"])


# ---------------------------------------------------------------------- applying it for real ---


def test_applying_a_sample_twice_does_not_drift(scene):
    s = R.sample(np.random.default_rng(13), 1.0)
    R.apply(scene, s)
    once = _snapshot(scene.model)
    R.apply(scene, s)
    twice = _snapshot(scene.model)
    for key in once:
        assert np.allclose(once[key], twice[key]), f"{key} drifted on a second apply"


def test_a_level_zero_sample_leaves_the_model_exactly_as_built(scene):
    before = _snapshot(scene.model)
    R.apply(scene, R.sample(np.random.default_rng(14), 0.0))
    after = _snapshot(scene.model)
    for key in before:
        assert np.allclose(before[key], after[key]), key


def test_a_randomized_scene_still_settles_cleanly(scene):
    for seed in (0, 1, 2):
        R.apply(scene, R.sample(np.random.default_rng(seed), 1.0))
        reset_scene(scene, np.random.default_rng(seed))
        report = settle(scene, 1.0)
        assert report.warnings == {}, (seed, report.warnings)
        assert report.connected_placed == len(PLACED), (seed, report.connections)
        assert report.max_speed < 5e-3, (seed, report.max_speed)


def test_a_randomized_reset_is_still_reproducible(scene):
    s = R.sample(np.random.default_rng(21), 1.0)
    R.apply(scene, s)
    reset_scene(scene, np.random.default_rng(21))
    scene.step(200)
    first = np.concatenate([scene.data.qpos.copy(), scene.data.qvel.copy()])

    R.apply(scene, s)
    reset_scene(scene, np.random.default_rng(21))
    scene.step(200)
    assert np.array_equal(
        np.concatenate([scene.data.qpos, scene.data.qvel]), first
    ), "the same sample and the same seed gave a different trajectory"


def test_the_sample_is_recorded_on_the_scene_for_the_episode_log(scene):
    s = R.sample(np.random.default_rng(15), 1.0)
    R.apply(scene, s)
    assert scene.randomization == s
    assert scene.randomization is not s, "the scene should keep its own copy"
    assert all(isinstance(v, (int, float, str)) for v in scene.randomization.values())
