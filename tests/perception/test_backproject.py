"""Tests for `brick_builder.perception.backproject`: pixel -> metric ray/plane intersection.

The accuracy test builds a real, compiled MuJoCo camera (fovy from the model, pose from
`mj_forward`'s own kinematics -- the "simulator ground truth" this check needs) and checks
`backproject_to_plane` against pixel coordinates computed by an *independent* pinhole-projection
formula written directly in this file (not by calling `project_point`), for a camera 0.25 m above a
horizontal plane, against the specified bound: under 0.5 mm at the image centre, under 1.5 mm at
the corners.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from brick_builder.perception.backproject import (
    CameraGeometry,
    backproject,
    backproject_to_plane,
    camera_geometry,
    pixel_rays,
    project_point,
)

WIDTH, HEIGHT = 241, 241  # odd, so the principal point falls exactly on an integer pixel
FOVY_DEG = 55.0
CAM_HEIGHT = 0.25


def _make_camera_model(pos, quat, fovy=FOVY_DEG) -> mujoco.MjModel:
    xml = f"""
    <mujoco>
      <visual><global offwidth="320" offheight="320"/></visual>
      <worldbody>
        <geom type="plane" size="1 1 0.01" pos="0 0 0"/>
        <camera name="cam" pos="{pos[0]} {pos[1]} {pos[2]}"
                quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" fovy="{fovy}"/>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _forward(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def _tilt_x_project(cam_pos: np.ndarray, alpha: float, geom: CameraGeometry, point: np.ndarray) -> np.ndarray:
    """Independent forward-projection formula for a camera at `cam_pos`, rotated `alpha` radians
    about world x from straight-down (see the test docstring below for the derivation): local
    axes are `x=(1,0,0)`, `y=(0,cos a,sin a)`, `z=(0,-sin a,cos a)` in world coordinates, i.e. the
    camera's own rotation matrix is `Rx(alpha)`. This mirrors the same pinhole model
    `backproject.py` uses, but is coded independently (no call into that module) precisely so the
    two can be checked against each other.
    """
    rel = np.asarray(point, dtype=float) - cam_pos
    c, s = math.cos(alpha), math.sin(alpha)
    local_x = rel[0]
    local_y = c * rel[1] + s * rel[2]
    local_z = -s * rel[1] + c * rel[2]
    assert local_z < 0.0, "point must be in front of the camera"
    depth = -local_z
    col = geom.fx * (local_x / depth) + geom.cx
    row = geom.cy - geom.fy * (local_y / depth)
    return np.array([col, row])


# --------------------------------------------------------------------------------------- basics ---


def test_camera_geometry_reads_fovy_pos_and_rot_off_the_model() -> None:
    pos = np.array([0.1, -0.2, CAM_HEIGHT])
    model = _make_camera_model(pos, (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    np.testing.assert_allclose(geom.pos, pos)
    np.testing.assert_allclose(geom.rot, np.eye(3), atol=1e-9)
    expected_fy = (0.5 * HEIGHT) / math.tan(0.5 * math.radians(FOVY_DEG))
    assert geom.fx == pytest.approx(expected_fy)
    assert geom.fy == pytest.approx(expected_fy)
    assert geom.cx == pytest.approx((WIDTH - 1) / 2.0)
    assert geom.cy == pytest.approx((HEIGHT - 1) / 2.0)


def test_camera_geometry_rejects_unknown_camera() -> None:
    model = _make_camera_model((0, 0, 0.25), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    with pytest.raises(ValueError):
        camera_geometry(model, data, "nope", WIDTH, HEIGHT)


def test_camera_geometry_rejects_bad_render_size() -> None:
    model = _make_camera_model((0, 0, 0.25), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    with pytest.raises(ValueError):
        camera_geometry(model, data, "cam", 0, HEIGHT)


def test_pixel_rays_straight_down_camera_points_at_minus_z() -> None:
    model = _make_camera_model((0, 0, CAM_HEIGHT), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    origin, direction = pixel_rays(geom, np.array([geom.cx, geom.cy]))
    np.testing.assert_allclose(origin, geom.pos)
    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0], atol=1e-9)


# ---------------------------------------------------------------- forward/backward are inverses ---


def test_project_point_inverts_pixel_rays_round_trip() -> None:
    model = _make_camera_model((0.05, -0.03, 0.3), (0.92388, 0.38268, 0.0, 0.0))  # ~40 deg about x
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", 200, 150)
    rng = np.random.default_rng(0)
    for _ in range(20):
        px = rng.uniform([0, 0], [199, 149])
        origin, direction = pixel_rays(geom, px)
        point = origin + 0.4 * direction  # some point along the ray, in front of the camera
        back = project_point(geom, point)
        np.testing.assert_allclose(back, px, atol=1e-6)


def test_project_point_rejects_points_behind_the_camera() -> None:
    model = _make_camera_model((0, 0, 0.25), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        project_point(geom, np.array([0.0, 0.0, 0.30]))  # above the camera, i.e. behind it


# ------------------------------------------------------------------------- ground-truth accuracy ---


def test_backproject_matches_ground_truth_at_image_centre() -> None:
    """A camera 0.25 m above the table, straight down: the pixel exactly at the principal point
    must back-project to the point on the plane directly below the camera, to well under the
    spec's 0.5 mm bound."""
    cam_pos = np.array([0.12, -0.07, CAM_HEIGHT])
    model = _make_camera_model(cam_pos, (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)

    point = backproject_to_plane(geom, np.array([geom.cx, geom.cy]), plane_z=0.0)
    truth = np.array([cam_pos[0], cam_pos[1], 0.0])
    error_mm = float(np.linalg.norm(point - truth)) * 1000.0
    assert error_mm < 0.5, f"centre error {error_mm:.4f} mm"


def test_backproject_matches_ground_truth_at_the_corners() -> None:
    """A camera 0.25 m above the table, tilted 25 degrees off straight-down about world x (a
    stand-in for a wrist camera that is never perfectly nadir): pixels at the four image corners
    must back-project, to under the spec's 1.5 mm bound, to the world points an independently
    coded pinhole projection (`_tilt_x_project`) says those corners actually see.
    """
    alpha = math.radians(25.0)
    cam_pos = np.array([0.15, -0.05, CAM_HEIGHT])
    quat = (math.cos(alpha / 2.0), math.sin(alpha / 2.0), 0.0, 0.0)  # rotation by alpha about x
    model = _make_camera_model(cam_pos, quat)
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)

    # Ground truth: where each corner pixel's ray (per backproject.py) actually meets the plane,
    # computed with the independent formula above from the *same* ray direction backproject.py
    # would use -- so what is being checked is that backproject's own intersection-with-the-plane
    # arithmetic agrees with an independently written pinhole model, not merely with itself.
    corners = np.array([[0, 0], [WIDTH - 1, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1]], dtype=float)
    max_error_mm = 0.0
    for px in corners:
        origin, direction = pixel_rays(geom, px)
        # independent closed-form ray direction for this tilt, to cross-check pixel_rays itself
        x = (px[0] - geom.cx) / geom.fx
        y = (geom.cy - px[1]) / geom.fy
        local = np.array([x, y, -1.0])
        local /= np.linalg.norm(local)
        c, s = math.cos(alpha), math.sin(alpha)
        expected_dir = np.array([
            local[0],
            c * local[1] - s * local[2],
            s * local[1] + c * local[2],
        ])
        np.testing.assert_allclose(direction, expected_dir, atol=1e-9)

        t = (0.0 - origin[2]) / expected_dir[2]
        assert t > 0.0
        truth = origin + t * expected_dir

        # `_tilt_x_project` is the same pinhole model in the *forward* direction (world point ->
        # pixel); projecting `truth` back through it must recover the original corner pixel,
        # an independent round-trip check that does not go anywhere near `backproject.py`.
        np.testing.assert_allclose(_tilt_x_project(cam_pos, alpha, geom, truth), px, atol=1e-6)

        point = backproject_to_plane(geom, px, plane_z=0.0)
        error_mm = float(np.linalg.norm(point - truth)) * 1000.0
        max_error_mm = max(max_error_mm, error_mm)
    assert max_error_mm < 1.5, f"max corner error {max_error_mm:.4f} mm"


def test_backproject_one_shot_matches_the_two_step_form() -> None:
    model = _make_camera_model((0.1, 0.0, 0.25), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    px = np.array([70.0, 130.0])
    one_shot = backproject(model, data, "cam", px, WIDTH, HEIGHT, plane_z=0.0)
    two_step = backproject_to_plane(geom, px, plane_z=0.0)
    np.testing.assert_allclose(one_shot, two_step)


def test_backproject_accepts_a_batch_of_pixels() -> None:
    model = _make_camera_model((0.0, 0.0, 0.25), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    px = np.array([[geom.cx, geom.cy], [10.0, 10.0], [230.0, 200.0]])
    points = backproject_to_plane(geom, px, plane_z=0.0)
    assert points.shape == (3, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0], atol=1e-9)


def test_backproject_rejects_a_plane_behind_the_camera() -> None:
    model = _make_camera_model((0.0, 0.0, 0.25), (1.0, 0.0, 0.0, 0.0))
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        # the plane at z=0.30 is *above* a camera at z=0.25 looking down -- behind it
        backproject_to_plane(geom, np.array([geom.cx, geom.cy]), plane_z=0.30)


def test_backproject_rejects_a_ray_parallel_to_the_plane() -> None:
    """A camera aimed horizontally (its own -z axis lies flat in the plane's own horizontal
    plane): the principal-point ray never reaches ``z = plane_z`` at all."""
    quat = (math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0, 0.0)  # 90 deg about x: looks -y
    model = _make_camera_model((0.0, 0.0, 0.25), quat)
    data = _forward(model)
    geom = camera_geometry(model, data, "cam", WIDTH, HEIGHT)
    origin, direction = pixel_rays(geom, np.array([geom.cx, geom.cy]))
    assert abs(direction[2]) < 1e-9  # confirms the ray is exactly horizontal, as intended
    with pytest.raises(ValueError):
        backproject_to_plane(geom, np.array([geom.cx, geom.cy]), plane_z=0.0)