"""Pixel <-> metric position: a ray from a known MuJoCo camera, met with a horizontal plane.

This is the last step of the perception chain (``docs/guide/06-perception.md``): every
token producer (``producers.py``) reports positions in **pixel** units, because that is all a
single rendered frame gives you.  Turning a pixel into a metric world position needs the camera's
actual geometry -- intrinsics from ``model.cam_fovy`` and the render resolution, extrinsics from
``data.cam_xpos``/``data.cam_xmat`` -- which is exactly what a real robot has too (a calibrated
camera pose from forward kinematics, not a special simulator privilege), so this module is legal
for an *actor* observation to depend on, unlike anything in ``brick_builder.scene``.

Camera convention (MuJoCo, and this project's own patch_so101/compose cameras): the camera frame's
x axis points right, y points up, z points backward, so the viewing direction is the camera's own
``-z``.  ``data.cam_xmat[cam_id].reshape(3, 3)`` is the rotation whose *columns* are those three
axes expressed in the world frame (the same convention ``ArmKinematics`` and ``compose.py`` use for
every other body/site frame in this project), so a local direction ``d`` becomes ``R @ d`` in world
coordinates.

Pixel convention: ``(col, row)``, `col` in ``[0, width)`` growing right, `row` in ``[0, height)``
growing down (image-array convention, matching ``tokens.py`` and ``producers.py``), with pixel
*centres* at integer coordinates -- pixel ``(0, 0)`` is the centre of the top-left pixel, so the
principal point sits at ``((width - 1) / 2, (height - 1) / 2)``, the geometric centre of the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraGeometry:
    """A pinhole camera's intrinsics and its extrinsics at one instant.

    ``fx``/``fy`` are in pixels; square pixels are assumed (MuJoCo's own convention: the vertical
    field of view is ``cam_fovy`` and the horizontal one stretches to match the requested render
    aspect ratio, which makes ``fx == fy`` in pixels whenever ``width``/``height`` match the aspect
    the frame was actually rendered at). ``pos`` is the camera's world position (metres); ``rot``
    is the 3x3 rotation whose columns are the camera's (x=right, y=up, z=back) axes in world
    coordinates.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    pos: np.ndarray
    rot: np.ndarray


def _camera_id(model: mujoco.MjModel, camera: str | int) -> int:
    if isinstance(camera, (int, np.integer)):
        cam_id = int(camera)
    else:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"unknown camera {camera!r}")
    return cam_id


def camera_geometry(
    model: mujoco.MjModel, data: mujoco.MjData, camera: str | int, width: int, height: int
) -> CameraGeometry:
    """Read one camera's current intrinsics + extrinsics off a compiled model/data pair.

    ``width``/``height`` must be the pixel size of the frame the caller's pixel coordinates refer
    to (whatever ``SceneRenderer.render``, or another renderer, actually produced) -- the
    intrinsics are meaningless without knowing what resolution they belong to.  ``data`` must
    already reflect the pose to backproject at (after ``mujoco.mj_forward``/``mj_step``): a camera
    fixed to a moving body, like the wrist camera, has a different ``cam_xpos``/``cam_xmat`` at
    every arm configuration, and this function always reads the current one, never a cached value.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive; got {width}x{height}")
    cam_id = _camera_id(model, camera)
    fovy_rad = math.radians(float(model.cam_fovy[cam_id]))
    fy = (0.5 * height) / math.tan(0.5 * fovy_rad)
    fx = fy  # square pixels at the render aspect ratio -- see the module/class docstring
    return CameraGeometry(
        fx=fx,
        fy=fy,
        cx=0.5 * (width - 1),
        cy=0.5 * (height - 1),
        pos=data.cam_xpos[cam_id].copy(),
        rot=data.cam_xmat[cam_id].reshape(3, 3).copy(),
    )


def pixel_rays(geom: CameraGeometry, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``px`` ``[..., 2]`` (col, row) -> one shared world-frame origin and a unit direction each.

    Directions point away from the camera, into the scene (whatever they eventually hit).
    """
    px = np.asarray(px, dtype=float)
    if px.shape[-1] != 2:
        raise ValueError(f"px must be [...,2] (col,row); got shape {px.shape}")
    x = (px[..., 0] - geom.cx) / geom.fx
    y = (geom.cy - px[..., 1]) / geom.fy
    d_local = np.stack([x, y, -np.ones_like(x)], axis=-1)
    d_local /= np.linalg.norm(d_local, axis=-1, keepdims=True)
    d_world = d_local @ geom.rot.T  # (R @ d) written for a batch of row-vectors d
    return geom.pos.copy(), d_world


def project_point(geom: CameraGeometry, point_world: np.ndarray) -> np.ndarray:
    """World point(s) ``[..., 3]`` -> pixel ``(col, row)`` ``[..., 2]``. Inverse of the ray built
    by :func:`pixel_rays` for a point actually on that ray.

    Raises ``ValueError`` if any point is behind the camera (on the far side of its own image
    plane) -- there is no pixel for that, and returning a nonsense finite value would be worse
    than failing loudly.
    """
    point_world = np.asarray(point_world, dtype=float)
    rel = point_world - geom.pos
    local = rel @ geom.rot  # R^T @ rel, written for a batch of row-vectors rel
    z = local[..., 2]
    if np.any(z >= 0.0):
        raise ValueError("point(s) behind the camera (local z >= 0)")
    depth = -z
    col = geom.fx * (local[..., 0] / depth) + geom.cx
    row = geom.cy - geom.fy * (local[..., 1] / depth)
    return np.stack([col, row], axis=-1)


def backproject_to_plane(
    geom: CameraGeometry, px: np.ndarray, plane_z: float = 0.0
) -> np.ndarray:
    """``px`` ``[..., 2]`` (col, row) -> world point(s) ``[..., 3]`` on the horizontal plane
    ``z = plane_z``.

    Raises ``ValueError`` if a ray is (near-)parallel to the plane, or if the plane lies behind
    the camera along that ray -- both mean "this pixel cannot see that plane," not a silently
    wrong answer.
    """
    origin, direction = pixel_rays(geom, px)
    dz = direction[..., 2]
    if np.any(np.abs(dz) < 1e-9):
        raise ValueError("ray parallel to the plane; no intersection")
    t = (plane_z - origin[2]) / dz
    if np.any(t <= 0.0):
        raise ValueError("the plane is behind the camera along at least one ray")
    return origin + t[..., np.newaxis] * direction


def backproject(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | int,
    px: np.ndarray,
    width: int,
    height: int,
    plane_z: float = 0.0,
) -> np.ndarray:
    """One-shot convenience: camera name/id + pixel(s) -> world point(s) on a horizontal plane.

    Equivalent to ``backproject_to_plane(camera_geometry(model, data, camera, width, height), px,
    plane_z)``; use the two-step form directly when backprojecting many pixels against the same
    camera pose (the align environment does), so the camera's
    intrinsics/extrinsics are only read off ``data`` once.
    """
    geom = camera_geometry(model, data, camera, width, height)
    return backproject_to_plane(geom, px, plane_z)
