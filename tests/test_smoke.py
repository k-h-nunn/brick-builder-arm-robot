"""Smoke tests: the package imports, MuJoCo physics runs, and offscreen rendering works."""

import numpy as np
import pytest

import brick_builder

_BOX_MJCF = """
<mujoco model="smoke_box">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="top" pos="0 0 1.5" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" rgba="0.8 0.2 0.2 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_import():
    assert brick_builder.__doc__


def test_box_falls_and_rests_on_plane():
    import mujoco

    model = mujoco.MjModel.from_xml_string(_BOX_MJCF)
    data = mujoco.MjData(model)

    for _ in range(100):
        mujoco.mj_step(model, data)

    box_z = data.qpos[2]  # first 3 entries of the freejoint qpos are the box's xyz position
    # The box's rest height is its half-size (0.05) above the plane at z=0.
    assert box_z > 0.0, f"box should be above the plane, got z={box_z}"
    assert abs(box_z - 0.05) < 0.02, f"box should have settled near z=0.05, got z={box_z}"


def test_offscreen_render_rgb():
    import mujoco

    try:
        model = mujoco.MjModel.from_xml_string(_BOX_MJCF)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=256, width=256)
    except Exception as exc:  # pragma: no cover - depends on the machine's GL setup
        pytest.skip(f"EGL offscreen rendering unavailable: {exc}")

    renderer.update_scene(data)
    image = renderer.render()

    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8
    assert image.std() > 0, "rendered image is constant; rendering did not produce real pixels"
