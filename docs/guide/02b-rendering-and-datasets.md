# Rendering and datasets

`brick_builder/render/` (`renderer.py`, `domain_rand.py`) converts an `MjModel`/`MjData` pair into
RGB, metric depth, and a per-object instance map, and randomizes the scene and image for training.
Scene construction is `brick_builder/bricks/`, `brick_builder/robot/`, and `brick_builder/scene/`;
this module only renders the `MjModel` it receives.

## The problem `mujoco.Renderer` doesn't solve

`mujoco.Renderer`'s depth and segmentation modes are mutually exclusive per instance. RGB, depth,
and an instance map for one time step require a thin renderer built directly on `GLContext`,
`MjvScene`, and `MjrContext`, rendering the scene twice:

1. **Pass 1 (RGB + depth).** Render normally; read both from one framebuffer with
   `mjr_readPixels(rgb, depth, rect, ctx)`.
2. **Pass 2 (segmentation).** Enable `scene.flags[mjRND_SEGMENT]` and `scene.flags[mjRND_IDCOLOR]`,
   render again, and decode the RGB triple at each pixel as `segid + 1` (0 = background).
   `scene.geoms[k].objid` / `.objtype` give the geom; `model.geom_bodyid[objid]` gives its body.

The depth conversion (reverse-Z NDC to metres via `<visual><map znear/zfar>` scaled by
`model.stat.extent`) matches `mujoco` 3.13.0's own `mujoco.Renderer` source exactly. Fallback on an
unreliable driver: run `mujoco.Renderer` twice, with `enable_depth_rendering()` then
`enable_segmentation_rendering()` — slower, public API only.

## `SceneRenderer`

```python
from brick_builder.render.renderer import SceneRenderer

renderer = SceneRenderer(model, width=256, height=256)
renderer.instance_lut({"box_red": 1, "box_blue": 2})   # body name -> instance id
frame = renderer.render(data, camera="cam0")
renderer.close()   # or: with SceneRenderer(...) as renderer: ...
```

- The GL context and `MjrContext` build lazily, on first use, never in `__init__` or at import:
  `gymnasium.vector.AsyncVectorEnv` forks a subprocess per environment, and a forked process must
  not inherit its parent's open GL context.
- `instance_lut` maps body name to instance id. An unmapped body renders as instance `0`; true
  background (no geom hit) renders as `-1`.
- Collision-only geoms are masked via `MjvOption.geomgroup`: this codebase's MJCF convention
  (shared with the vendored SO-101 model) puts visual meshes in geom group 2 and collision
  primitives in group 3, so group 3 is hidden (`COLLISION_GEOM_GROUP` in `renderer.py`).
- `render()` returns `Frame(rgb, depth, instance, body_ids)`: `rgb` is `uint8 [H, W, 3]`, `depth`
  is `float32 [H, W]` metres, `instance` is `int32 [H, W]`, `body_ids` is the active instance map.

### Camera-space depth, not ray length

The depth buffer follows the OpenGL/MuJoCo convention: each pixel holds distance along the camera's
optical axis, not Euclidean ray distance. A flat plane perpendicular to the camera reads the same
depth — its height above the plane — at every pixel, including off-axis edge pixels
(`tests/render/test_renderer.py::test_depth_of_known_plane_within_1mm`, accurate to better than
1 mm).

### Contact sheet

![Contact sheet](../media/render/contact_sheet.png)

RGB, instance and depth for one frame of a self-contained two-box test scene (see [Reproducing the
example figures](#reproducing-the-example-figures)). `plot_contact_sheet(frame, out_path)` in
`renderer.py` produces this figure for `scripts/make_media.py`.

## Domain randomization: `domain_rand.py`

Two kinds of randomization. **Scene-side**, applied to an `mujoco.MjSpec` before compilation —
where `brick_builder` randomizes other physical parameters (`kp`, `forcerange`) — so rendering
conditions are fixed once per episode, not per frame:

```python
from brick_builder.render.domain_rand import (
    ColourJitterConfig, randomize_material_colour,
    LightRigConfig, randomize_lights,
    CameraJitterConfig, randomize_camera,
)

rng = np.random.default_rng(seed)
randomize_material_colour(spec.material("mat_brick_red"), (0.8, 0.1, 0.1), ColourJitterConfig(), rng)
randomize_lights(spec, LightRigConfig(n_lights=(1, 4)), rng)
randomize_camera(spec, "wrist_cam", CameraJitterConfig(pos_jitter_m=0.005, rot_jitter_deg=3.0), rng)
model = spec.compile()
```

- `randomize_material_colour` jitters HSV (hue, saturation, value, each ranged independently) and
  writes `material.rgba`; used for brick- and table-colour alike.
- `randomize_lights` deletes every worldbody light and adds 1-4 point lights at randomized
  positions with randomized diffuse/specular gain.
- `randomize_camera` perturbs a named camera's position (uniform box), orientation (random small
  rotation about a random axis), and vertical fov, from one `rng`.

**Image-side**, applied per rendered `Frame`: camera and sensor artefacts with no MJCF equivalent.

```python
from brick_builder.render.domain_rand import sample_postprocess_config, apply_postprocess

cfg = sample_postprocess_config(rng)     # or build a PostProcessConfig by hand
noisy_frame = apply_postprocess(frame, cfg, rng)
```

`PostProcessConfig` covers radial lens distortion (`k1`, `k2`), exposure, per-channel white
balance, gamma, shot noise (signal-dependent), read noise (constant), Gaussian blur, and optional
JPEG re-encoding. The lens-distortion remap uses bilinear sampling for RGB but nearest-neighbour for
depth and instance (a distorted pixel is always a copy of one input pixel, never a blend of two
objects or distances —
`tests/render/test_domain_rand.py::test_distortion_remap_uses_nearest_neighbour_for_depth_and_instance`).
At neutral parameters (`PostProcessConfig()` default), `apply_postprocess` is the identity.

### Before / after

![Post-process before and after](../media/render/postprocess_before_after.png)

`plot_postprocess_before_after(frame, cfg, rng, out_path)` in `domain_rand.py` produces this panel
for `scripts/make_media.py`.

## Measured numbers

CPU-only, `MUJOCO_GL=egl`: a combined RGB + depth + instance render of the two-box scene at
256x256, after a warm-up call, runs at **about 700 frames/s** against a spec target of >=150 fps.
`tests/render/test_renderer.py::test_throughput_rgb_depth_instance_256` measures and prints this on
every run (marked `slow`, 200 frames); per spec section 4 it reports the number without failing the
suite on speed.

## Reproducing the example figures

Both figures use a self-contained two-box MJCF scene (grey table, red and blue box, one point
light, one downward-looking camera), defined as `two_box_mjcf` in `tests/render/conftest.py`;
render's own tests carry no dependency on the brick or robot modules. The contact sheet applies
`randomize_material_colour`, `randomize_lights`, and `randomize_camera` with seed 0. The
before/after panel uses a hand-picked (not sampled) `PostProcessConfig` with visible barrel
distortion, warm white balance, gamma lift, noise, and blur; a dataset instead draws
`PostProcessConfig` from `sample_postprocess_config`.

## GPU-side non-determinism

`SceneRenderer` is deterministic for a given `MjModel`/`MjData` state and instance map: same input,
same bytes out (`tests/render/test_renderer.py::test_same_data_renders_identical_bytes`). That
test, and the two example-scene MJCF fragments in `tests/render/conftest.py`, set `<visual><quality
offsamples="0"/></visual>`, disabling multisample anti-aliasing on the offscreen framebuffer. At the
default `offsamples` (4x MSAA), repeated renders of one scene state can differ by a handful of
shadow/edge pixels, off by one greyscale value each: GPU-driver blending non-determinism, not a
module defect. Frame-identity does not hold with anti-aliasing on.
