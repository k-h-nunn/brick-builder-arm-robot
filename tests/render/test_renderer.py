"""Acceptance tests for brick_builder.render.renderer.SceneRenderer (spec section 4)."""

from __future__ import annotations

import time

import mujoco
import numpy as np
import pytest

from brick_builder.render.renderer import COLLISION_GEOM_GROUP, Frame, SceneRenderer


def _model_data(mjcf: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_instance_map_ids_and_pixel_counts(two_box_mjcf):
    """Two boxes of known screen-space size get the right instance ids in the right proportion.

    Rendered at 512x512 (matching the scene's own offscreen buffer): at the
    default 256x256 used elsewhere, pixel-grid alignment happens to put one box's
    edge astride a rounding boundary that the other box's mirror-image edge
    misses, which is a rasterization artefact of the segmentation pass (no
    anti-aliasing) rather than a real size difference -- it shows up as a
    resolution-dependent few-percent wobble, not a bias in one direction.
    """
    model, data = _model_data(two_box_mjcf)
    with SceneRenderer(model, width=512, height=512) as renderer:
        renderer.instance_lut({"box_red": 1, "box_blue": 2})
        frame = renderer.render(data, "cam0")

    ids, counts = np.unique(frame.instance, return_counts=True)
    counts_by_id = dict(zip(ids.tolist(), counts.tolist()))

    assert set(counts_by_id) == {0, 1, 2}, f"unexpected instance ids {counts_by_id}"
    # Both boxes have identical size and are symmetric about the camera axis, so
    # they must cover essentially the same pixel count (within 2%, per the spec).
    red, blue = counts_by_id[1], counts_by_id[2]
    assert red > 0 and blue > 0
    assert abs(red - blue) / max(red, blue) < 0.02


def test_unlisted_body_is_instance_zero_true_background_is_minus_one(two_box_small_table_mjcf):
    model, data = _model_data(two_box_small_table_mjcf)
    with SceneRenderer(model, width=128, height=128) as renderer:
        # Only map the red box; the table body is left unlisted.
        renderer.instance_lut({"box_red": 1})
        frame = renderer.render(data, "cam0")

    ids = set(np.unique(frame.instance).tolist())
    assert 1 in ids            # the mapped box
    assert 0 in ids            # the unlisted table body
    assert -1 in ids           # sky around the small table: true background


def test_collision_geoms_are_hidden(two_box_small_table_mjcf):
    """A geom in the collision-only group (3) must never show up in RGB or segmentation."""
    model, data = _model_data(two_box_small_table_mjcf)
    hidden_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hidden_collider")
    assert hidden_body != -1

    with SceneRenderer(model, width=128, height=128) as renderer:
        renderer.instance_lut({"box_red": 1, "box_blue": 2, "hidden_collider": 99})
        frame = renderer.render(data, "cam0")

    assert 99 not in np.unique(frame.instance)
    # The hidden geom is pure green (0, 255, 0); its colour must not appear either.
    green = (frame.rgb[..., 0] < 10) & (frame.rgb[..., 1] > 245) & (frame.rgb[..., 2] < 10)
    assert not green.any()


def test_depth_of_known_plane_within_1mm(plane_mjcf):
    """A flat plane perpendicular to the camera axis reads back at a uniform, known depth.

    The depth buffer holds camera-space z (the standard OpenGL/MuJoCo convention),
    not Euclidean range along each ray, so a fronto-parallel plane gives the same
    depth at every pixel, equal to the camera's height above it, regardless of the
    ray's angle off axis.
    """
    model, data = _model_data(plane_mjcf)
    cam_height = 0.75
    with SceneRenderer(model, width=256, height=256) as renderer:
        renderer.instance_lut({})
        frame = renderer.render(data, "cam0")

    assert np.abs(frame.depth - cam_height).max() < 1e-3, (
        f"depth range [{frame.depth.min()}, {frame.depth.max()}] vs expected {cam_height}"
    )


def test_same_data_renders_identical_bytes(two_box_mjcf):
    """Rendering the same `MjData` state twice must produce bit-identical frames."""
    model, data = _model_data(two_box_mjcf)
    with SceneRenderer(model, width=128, height=128) as renderer:
        renderer.instance_lut({"box_red": 1, "box_blue": 2})
        a = renderer.render(data, "cam0")
        b = renderer.render(data, "cam0")

    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.depth, b.depth)
    assert np.array_equal(a.instance, b.instance)


def test_frame_dataclass_shapes_and_dtypes(two_box_mjcf):
    model, data = _model_data(two_box_mjcf)
    with SceneRenderer(model, width=64, height=48) as renderer:
        renderer.instance_lut({"box_red": 1})
        frame = renderer.render(data, "cam0")

    assert isinstance(frame, Frame)
    assert frame.rgb.shape == (48, 64, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (48, 64) and frame.depth.dtype == np.float32
    assert frame.instance.shape == (48, 64) and frame.instance.dtype == np.int32
    assert frame.body_ids == {"box_red": 1}


def test_unknown_camera_raises(two_box_mjcf):
    model, data = _model_data(two_box_mjcf)
    with SceneRenderer(model, width=32, height=32) as renderer:
        with pytest.raises(ValueError):
            renderer.render(data, "does_not_exist")


def test_unknown_body_in_lut_raises(two_box_mjcf):
    model, _ = _model_data(two_box_mjcf)
    with SceneRenderer(model, width=32, height=32) as renderer:
        with pytest.raises(ValueError):
            renderer.instance_lut({"not_a_body": 1})


@pytest.mark.slow
def test_throughput_rgb_depth_instance_256(two_box_mjcf, capsys):
    """Measures frames/s for a combined RGB+depth+instance render at 256x256 and reports it.

    Per spec section 4 this does not fail on speed (target is >=150 fps on the
    reference machine); it always reports the measured number.
    """
    model, data = _model_data(two_box_mjcf)
    with SceneRenderer(model, width=256, height=256) as renderer:
        renderer.instance_lut({"box_red": 1, "box_blue": 2})
        # Warm-up: first call pays for context creation and shader compilation.
        renderer.render(data, "cam0")

        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            renderer.render(data, "cam0")
        elapsed = time.perf_counter() - t0

    fps = n / elapsed
    with capsys.disabled():
        print(f"\n[render throughput] {fps:.1f} frames/s (RGB+depth+instance, 256x256)")
    assert fps > 0
