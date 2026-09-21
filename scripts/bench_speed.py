"""Benchmark MuJoCo physics and rendering speed, and report the torch/CUDA environment.

Prints a single JSON dict to stdout with:
  - mujoco_version
  - physics_steps_per_s: single-thread physics step rate on a tiny model, timed over 2 wall-clock seconds
  - render_rgb_fps: offscreen RGB render rate at 256x256
  - render_rgbd_seg_fps: offscreen RGB+depth+segmentation render rate at 256x256 (three passes per frame)
  - torch_version
  - cuda_available
  - cpu_count
"""

import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

_BOX_MJCF = """
<mujoco model="bench_box">
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


def bench_physics_steps_per_s(duration_s: float = 2.0) -> float:
    model = mujoco.MjModel.from_xml_string(_BOX_MJCF)
    data = mujoco.MjData(model)

    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        mujoco.mj_step(model, data)
        steps += 1
    elapsed = time.perf_counter() - start
    return steps / elapsed


def bench_render_rgb_fps(n_frames: int = 200) -> float:
    model = mujoco.MjModel.from_xml_string(_BOX_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=256, width=256)

    renderer.update_scene(data)
    renderer.render()  # warm up

    start = time.perf_counter()
    for _ in range(n_frames):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        renderer.render()
    elapsed = time.perf_counter() - start
    return n_frames / elapsed


def bench_render_rgbd_seg_fps(n_frames: int = 200) -> float:
    model = mujoco.MjModel.from_xml_string(_BOX_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=256, width=256)

    renderer.update_scene(data)
    renderer.render()  # warm up rgb pass

    renderer.enable_depth_rendering()
    renderer.update_scene(data)
    renderer.render()  # warm up depth pass

    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data)
    renderer.render()  # warm up segmentation pass
    renderer.disable_segmentation_rendering()

    start = time.perf_counter()
    for _ in range(n_frames):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)

        renderer.render()  # rgb pass

        renderer.enable_depth_rendering()
        renderer.render()  # depth pass
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.render()  # segmentation pass
        renderer.disable_segmentation_rendering()
    elapsed = time.perf_counter() - start
    return n_frames / elapsed


def main() -> None:
    result = {
        "mujoco_version": mujoco.__version__,
        "physics_steps_per_s": bench_physics_steps_per_s(),
    }

    try:
        result["render_rgb_fps"] = bench_render_rgb_fps()
        result["render_rgbd_seg_fps"] = bench_render_rgbd_seg_fps()
    except Exception as exc:  # pragma: no cover - depends on the machine's GL setup
        result["render_rgb_fps"] = None
        result["render_rgbd_seg_fps"] = None
        result["render_error"] = str(exc)

    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover
        result["torch_version"] = None
        result["cuda_available"] = None
        result["torch_error"] = str(exc)

    result["cpu_count"] = os.cpu_count()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
