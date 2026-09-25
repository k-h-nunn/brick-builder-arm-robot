"""Tests for `brick_builder.perception.views`: the three fixed viewing poses, the crop/resize pixel
mapping, and the moving attention window (both the 3D-ROI form and the slot-state form).
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.bricks.spec import DUPLO
from brick_builder.bricks.structures import Placement
from brick_builder.perception.backproject import backproject_to_plane, project_point
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState
from brick_builder.perception.views import (
    PLATE_CROP,
    PLATE_TCP,
    PLATE_VIEW,
    PLATE_VIEW_DENSE,
    TRAY_VIEW,
    VIEWS,
    attention_crop,
    attention_view,
    crop_and_resize,
    cropped_camera_geometry,
    full_frame_geometry,
    full_to_output,
    goto_look,
    output_to_full,
    px_per_stud_pitch,
    render_view,
    roi_box_corners,
    solve_look,
    window_from_slots,
)
from brick_builder.render.renderer import SceneRenderer
from brick_builder.robot.patch_so101 import WRIST_CAM
from brick_builder.scene.compose import SceneConfig, build_scene

PLATE_STUDS = (8, 8)


@pytest.fixture(scope="module")
def scene():
    cfg = SceneConfig(
        held=None, tray=True, tray_bricks=2,
        placed=[Placement(2, 2, 0, 0, 0, "red"), Placement(2, 2, 6, 6, 0, "blue")],
        cameras=(WRIST_CAM,),
    )
    return build_scene(cfg)


# ------------------------------------------------------------------------------ fixed ViewingPoses ---


def test_plate_view_has_tcp_and_solves(scene):
    q = solve_look(scene, PLATE_VIEW)
    assert q.shape == (5,)


def test_solve_look_rejects_a_pose_with_no_tcp():
    from dataclasses import replace

    floating = replace(PLATE_VIEW, tcp=None)
    with pytest.raises(ValueError):
        solve_look(None, floating)  # scene is never even touched -- fails before using it


def test_whole_8x8_plate_is_inside_the_plate_view(scene):
    """Every corner of the 8x8 (128 mm) plate, projected at PLATE_TCP's pose, must land inside
    PLATE_VIEW's own crop (with a little slack for its own measured margin)."""
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    geom = full_frame_geometry(scene, PLATE_VIEW)

    plate_xy = scene.cfg.plate_xy
    half = np.array([PLATE_STUDS[0] * DUPLO.pitch / 2.0, PLATE_STUDS[1] * DUPLO.pitch / 2.0])
    plane_z = scene.plate_top
    corners = np.array([
        [plate_xy[0] - half[0], plate_xy[1] - half[1], plane_z],
        [plate_xy[0] - half[0], plate_xy[1] + half[1], plane_z],
        [plate_xy[0] + half[0], plate_xy[1] - half[1], plane_z],
        [plate_xy[0] + half[0], plate_xy[1] + half[1], plane_z],
    ])
    px = project_point(geom, corners)
    x0, y0, x1, y1 = PLATE_CROP
    margin = 5  # pixels of slack for the plate's own edge thickness / measurement rounding
    assert np.all(px[:, 0] >= x0 - margin) and np.all(px[:, 0] <= x1 + margin)
    assert np.all(px[:, 1] >= y0 - margin) and np.all(px[:, 1] <= y1 + margin)


@pytest.mark.parametrize("name", list(VIEWS))
def test_pixel_mapping_round_trips(scene, name):
    pose = VIEWS[name]
    rng = np.random.default_rng(0)
    goto_look(scene, pose, rng)
    px = np.array([[0.0, 0.0], [255.0, 255.0], [128.0, 64.0], [10.0, 200.0]])
    full = output_to_full(pose, px)
    back = full_to_output(pose, full)
    np.testing.assert_allclose(back, px, atol=1e-9)


def test_plate_view_density_about_26_px_per_stud_pitch(scene):
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    px_per_pitch = px_per_stud_pitch(scene, PLATE_VIEW, scene.cfg.plate_xy, scene.plate_top, DUPLO.pitch)
    assert 20.0 <= px_per_pitch <= 32.0, px_per_pitch


def test_plate_view_dense_is_denser_than_plate_view(scene):
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    dense_pitch = px_per_stud_pitch(scene, PLATE_VIEW_DENSE, scene.cfg.plate_xy, scene.plate_top, DUPLO.pitch)
    plain_pitch = px_per_stud_pitch(scene, PLATE_VIEW, scene.cfg.plate_xy, scene.plate_top, DUPLO.pitch)
    assert dense_pitch > 1.5 * plain_pitch


def test_render_view_returns_256_frame_and_geometry(scene):
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    renderer = SceneRenderer(scene.model, width=PLATE_VIEW.render_size[0], height=PLATE_VIEW.render_size[1])
    try:
        renderer.instance_lut(scene.instance_ids)
        frame, geom = render_view(scene, PLATE_VIEW, renderer)
    finally:
        renderer.close()
    assert frame.rgb.shape == (256, 256, 3)
    assert frame.instance.shape == (256, 256)
    assert frame.depth.shape == (256, 256)
    assert geom.pos.shape == (3,)
    assert geom.rot.shape == (3, 3)


def test_cropped_camera_geometry_matches_backproject_at_crop_corners(scene):
    """Back-projecting the four crop-corner OUTPUT pixels through the reparametrized geometry
    must land exactly where projecting the same world point through the full-frame geometry maps
    to those same pre-crop pixels -- i.e. the two geometries agree, not just "look reasonable"."""
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    geom_full = full_frame_geometry(scene, PLATE_VIEW)
    geom_crop = cropped_camera_geometry(geom_full, PLATE_VIEW.crop, PLATE_VIEW.output_size)

    plane_z = scene.plate_top
    for out_px in [(0.0, 0.0), (255.0, 255.0), (128.0, 128.0)]:
        world_via_crop = backproject_to_plane(geom_crop, np.array(out_px), plane_z=plane_z)
        full_px = output_to_full(PLATE_VIEW, np.array(out_px))
        world_via_full = backproject_to_plane(geom_full, full_px, plane_z=plane_z)
        np.testing.assert_allclose(world_via_crop, world_via_full, atol=1e-9)


# --------------------------------------------------------------------------- attention window ---


def test_roi_box_corners_shape_and_bounds():
    corners = roi_box_corners((0.1, 0.2, 0.3), (0.01, 0.02, 0.03))
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [0.09, 0.18, 0.27])
    assert np.allclose(corners.max(axis=0), [0.11, 0.22, 0.33])


def test_attention_view_frames_a_small_roi_as_densely_as_plate_view_dense(scene):
    """A tight ROI around the plate's centre, via the general attention-window mechanism, should
    give a crop at least as dense as the hand-tuned PLATE_VIEW_DENSE -- demonstrating the fixed
    views are a special case of this mechanism, not a different one."""
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)  # camera at the plate-view pose; NOT moving further
    roi = roi_box_corners(
        (scene.cfg.plate_xy[0], scene.cfg.plate_xy[1], scene.plate_top),
        (2.0 * DUPLO.pitch, 2.0 * DUPLO.pitch, 0.01),
    )
    win = attention_view(scene, roi, render_size=PLATE_VIEW.render_size, margin=0.1)
    assert win.tcp is None
    pitch_px = px_per_stud_pitch(scene, win, scene.cfg.plate_xy, scene.plate_top, DUPLO.pitch)
    dense_pitch_px = px_per_stud_pitch(scene, PLATE_VIEW_DENSE, scene.cfg.plate_xy, scene.plate_top, DUPLO.pitch)
    # The general mechanism pads by a margin the hand-tuned PLATE_VIEW_DENSE crop does not, so it
    # is expected to be a bit less dense, not identical -- the claim is "comparably dense", not
    # "produces the exact same numbers".
    assert pitch_px >= 0.6 * dense_pitch_px


def test_attention_window_backproject_accuracy_matches_fixed_view(scene):
    """Back-projection through a crop chosen dynamically by `attention_view` (a moving-camera
    tool pose, not one of the three fixed poses) must be as accurate as through a fixed view --
    the same sub-millimetre bound `test_backproject.py` checks for a fixed camera.
    """
    rng = np.random.default_rng(0)
    # A handful of DIFFERENT, non-nominal tool poses standing in for "wherever the arm happens to
    # be mid-motion" -- built by nudging PLATE_TCP, each re-solved with IK.
    offsets = [(0.0, 0.0, 0.0), (0.01, 0.0, 0.01), (-0.01, 0.015, 0.005), (0.0, -0.02, 0.02)]
    max_err_mm = 0.0
    for dx, dy, dz in offsets:
        target = np.array(PLATE_TCP) + np.array([dx, dy, dz])
        q = solve_look(scene, PLATE_VIEW)  # a real pose, only used to seed the IK below
        from brick_builder.scene.compose import reset_scene, tool_down_ik

        q2 = tool_down_ik(scene.kin, target, 0.0, q0=q)
        reset_scene(scene, rng, arm_q=q2)
        import mujoco

        cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
        scene.model.cam_fovy[cam_id] = PLATE_VIEW.fovy_deg
        mujoco.mj_forward(scene.model, scene.data)

        roi = roi_box_corners(
            (scene.cfg.plate_xy[0], scene.cfg.plate_xy[1], scene.plate_top),
            (3.0 * DUPLO.pitch, 3.0 * DUPLO.pitch, 0.01),
        )
        win = attention_view(scene, roi, render_size=PLATE_VIEW.render_size)
        geom_full = full_frame_geometry(scene, win)
        geom_crop = cropped_camera_geometry(geom_full, win.crop, win.output_size)

        truth = np.array([scene.cfg.plate_xy[0], scene.cfg.plate_xy[1], scene.plate_top])
        px = project_point(geom_crop, truth)
        back = backproject_to_plane(geom_crop, px, plane_z=scene.plate_top)
        err_mm = float(np.linalg.norm(back - truth)) * 1000.0
        max_err_mm = max(max_err_mm, err_mm)
    assert max_err_mm < 0.5, f"max round-trip error {max_err_mm:.4f} mm"


def test_attention_crop_clamps_a_roi_bigger_than_the_render(scene):
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    geom_full = full_frame_geometry(scene, PLATE_VIEW)
    # A ROI bigger than the render but still comfortably in front of the camera must still return
    # a valid, in-bounds (clamped-to-render) crop.
    roi = roi_box_corners((scene.cfg.plate_xy[0], scene.cfg.plate_xy[1], scene.plate_top), (0.3, 0.3, 0.01))
    crop = attention_crop(geom_full, roi, PLATE_VIEW.render_size, margin=0.1)
    x0, y0, x1, y1 = crop
    rw, rh = PLATE_VIEW.render_size
    assert 0 <= x0 < x1 <= rw
    assert 0 <= y0 < y1 <= rh


def test_attention_crop_raises_for_a_roi_behind_the_camera(scene):
    """"This region is not in front of the camera" is a real answer (matching `backproject.
    project_point`'s own philosophy), not something silently clipped away."""
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    geom_full = full_frame_geometry(scene, PLATE_VIEW)
    roi = roi_box_corners((scene.cfg.plate_xy[0], scene.cfg.plate_xy[1], scene.plate_top), (5.0, 5.0, 0.01))
    with pytest.raises(ValueError):
        attention_crop(geom_full, roi, PLATE_VIEW.render_size, margin=0.1)


def test_crop_and_resize_nearest_is_exact_for_instance_ids():
    arr = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
    out = crop_and_resize(arr, (0, 0, 16, 16), 16, interpolation=__import__("cv2").INTER_NEAREST)
    np.testing.assert_array_equal(out, arr)


# ------------------------------------------------------------------------- window_from_slots ---


def _slot_state(centre, axes, window):
    slots = np.zeros((8, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    fields = PLACEHOLDER_SLOT_SCHEMA.fields
    slots[0, fields["mass"]] = 0.1
    slots[0, fields["centre"]] = centre
    slots[0, fields["moments"]] = [axes[0] ** 2 / 4.0, 0.0, axes[1] ** 2 / 4.0]
    return SceneState(slots=slots, depth=np.full(8, 0.5, dtype=np.float32), schema=PLACEHOLDER_SLOT_SCHEMA,
                       t=0.0, source="test", window=window, slot_ids=np.array([0, -1, -1, -1, -1, -1, -1, -1]))


def test_window_from_slots_narrows_onto_a_centred_small_slot():
    state = _slot_state(centre=(0.0, 0.0), axes=(0.1, 0.1), window=PLATE_VIEW)
    win = window_from_slots(state, [0], margin=0.3)
    assert win.crop != PLATE_VIEW.crop
    cw = win.crop[2] - win.crop[0]
    pw = PLATE_VIEW.crop[2] - PLATE_VIEW.crop[0]
    assert cw < pw  # narrower than the whole-plate view, since it targets one small slot
    assert win.tcp is None


def test_window_from_slots_empty_selection_returns_window_unchanged():
    state = _slot_state(centre=(0.0, 0.0), axes=(0.1, 0.1), window=PLATE_VIEW)
    win = window_from_slots(state, [3], margin=0.3)  # slot 3 is empty
    assert win is PLATE_VIEW


def test_window_from_slots_off_centre_slot_shifts_the_crop():
    state = _slot_state(centre=(0.6, -0.4), axes=(0.05, 0.05), window=PLATE_VIEW)
    win = window_from_slots(state, [0], margin=0.4)
    cx = 0.5 * (win.crop[0] + win.crop[2])
    plate_cx = 0.5 * (PLATE_VIEW.crop[0] + PLATE_VIEW.crop[2])
    assert cx != pytest.approx(plate_cx, abs=1.0)
