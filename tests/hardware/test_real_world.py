"""The camera path, with a rendered frame and an image file standing in for the device.

Nothing here opens a camera.  ``ArraySource`` and ``ImageFileSource`` are the interfaces that
exist so the undistort-crop-resize-scene-model chain can be run end to end with no hardware, and
the geometry the twin needs is checked by projecting a known world point back to its own pixel.
"""

import numpy as np
import pytest

from brick_builder.hardware.real_world import (
    ArraySource,
    CameraIntrinsics,
    ImageFileSource,
    RealCamera,
    UVC_RESOLUTION,
    camera_window,
    kinematic_arm,
)
from brick_builder.perception.backproject import backproject_to_plane, project_point


class CountingModel:
    """A scene model that records what it was handed.  The chain is under test, not the network."""

    def __init__(self):
        self.calls = []

    def observe(self, rgb, *, window, **kwargs):
        self.calls.append((np.asarray(rgb).shape, window.output_size))
        return "state"


def checker_frame(size=UVC_RESOLUTION):
    width, height = size
    grid = (((np.arange(height)[:, None] // 32) + (np.arange(width)[None, :] // 32)) % 2)
    return np.repeat((grid * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def test_the_nominal_intrinsics_are_the_lens_s_own_field_of_view():
    intr = CameraIntrinsics.nominal(115.0)
    assert intr.diagonal_fov_deg == pytest.approx(115.0, abs=1e-3)
    assert intr.cx == pytest.approx(319.5) and intr.cy == pytest.approx(239.5)


def test_intrinsics_come_out_of_the_parameter_file():
    from brick_builder.params import load_params

    intr = CameraIntrinsics.from_params(load_params())
    assert intr.width == 640 and intr.height == 480
    assert intr.diagonal_fov_deg == pytest.approx(115.0, abs=0.1)
    assert intr.matrix.shape == (3, 3)
    assert intr.distortion.shape == (5,)


def test_the_window_is_the_largest_centred_square_resized_to_the_model_s_input():
    window = camera_window(CameraIntrinsics.nominal())
    assert window.crop == (80, 0, 560, 480)
    assert window.output_size == 256
    assert window.tcp is None


def test_the_camera_path_hands_the_scene_model_a_square_frame():
    model = CountingModel()
    camera = RealCamera(ArraySource(checker_frame()), model=model)
    state, rgb = camera.observe_rgb(t=0.0)
    assert state == "state"
    assert rgb.shape == (256, 256, 3)
    assert model.calls == [((256, 256, 3), 256)]
    assert camera.frames == 1


def test_a_frame_of_the_wrong_size_is_refused_rather_than_silently_rescaled():
    camera = RealCamera(ArraySource(np.zeros((240, 320, 3), np.uint8)), model=CountingModel())
    with pytest.raises(ValueError):
        camera.window_frame()


def test_undistortion_leaves_an_undistorted_lens_alone():
    frame = checker_frame()
    camera = RealCamera(ArraySource(frame), CameraIntrinsics.nominal(), model=CountingModel())
    assert np.array_equal(camera.rectify(frame), frame)


def test_undistortion_changes_the_frame_when_the_lens_has_distortion():
    frame = checker_frame()
    intr = CameraIntrinsics.nominal()
    barrel = CameraIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy, k1=-0.3, k2=0.08)
    camera = RealCamera(ArraySource(frame), barrel, model=CountingModel())
    assert not np.array_equal(camera.rectify(frame), frame)


def test_a_recorded_image_file_stands_in_for_the_device(tmp_path):
    import cv2

    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), cv2.cvtColor(checker_frame(), cv2.COLOR_RGB2BGR))
    camera = RealCamera(ImageFileSource(path), model=CountingModel())
    assert camera.window_frame().shape == (256, 256, 3)


def test_the_window_geometry_backprojects_its_own_pixels_to_the_right_world_point():
    _, _, kin = kinematic_arm()
    camera = RealCamera(ArraySource(checker_frame()), model=CountingModel())
    # A camera two metres up, looking straight down: MuJoCo's convention is -z forward, +y up.
    rot = np.diag([1.0, -1.0, 1.0])
    geom = camera.geometry(np.array([0.1, 0.2, 2.0]), rot)
    for pixel in ([128.0, 128.0], [40.0, 200.0]):
        point = backproject_to_plane(geom, np.array(pixel), plane_z=0.0)
        assert point[2] == pytest.approx(0.0)
        assert project_point(geom, point) == pytest.approx(np.array(pixel), abs=1e-6)


def test_the_kinematic_arm_is_geometry_only_and_knows_where_the_wrist_camera_is():
    import mujoco

    from brick_builder.robot.patch_so101 import WRIST_CAM

    model, data, kin = kinematic_arm()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM) >= 0
    pos, _ = kin.fk(np.zeros(5))
    assert np.isfinite(pos).all()
