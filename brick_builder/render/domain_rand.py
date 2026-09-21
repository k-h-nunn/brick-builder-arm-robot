"""Domain randomization for rendering: scene-side (lights, colour, camera) and
image-side (post-process) randomization, per
`docs/guide/02b-rendering-and-datasets.md`.

Scene-side randomizers mutate an `mujoco.MjSpec` before it is compiled, the same
place the rest of brick_builder randomizes physical parameters (kp, forcerange, ...),
so a whole episode's rendering conditions are fixed once per episode rather than
drifting frame to frame. Image-side post-processing runs per rendered `Frame` and
approximates camera-and-sensor artefacts that have no MJCF equivalent: lens
distortion, exposure/white-balance/gamma, sensor noise, blur and JPEG.

Every randomizer takes an explicit `numpy.random.Generator`; there is no global
random state, so the same generator state reproduces the same scene or image byte
for byte (see tests/render/test_domain_rand.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from brick_builder.render.renderer import Frame

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

DEPTH_INVALID: float = float("nan")
"""The `Frame.depth` value at a pixel with no real depth reading (perception-align-003).

Used for exactly one case today: a pixel the lens-distortion remap in `apply_postprocess`
pulls from outside the pristine render's bounds. The old code filled that case with `0.0`,
which reads as "touching the camera lens" -- a physically impossible depth for this scene (its
true range is a few centimetres to tens of centimetres) that a downstream consumer could easily
mistake for a real near-field reading instead of "no data here". `Frame.instance` already has
exactly this problem solved for its own (integer, categorical) channel with an out-of-range
sentinel, `-1` ("background/no instance") -- see `brick_builder/render/renderer.py`. Depth is a
continuous physical measurement rather than a category, so its equivalent "clearly not a real
value" sentinel is NaN, not another magic number that still parses as a plausible depth:

- It cannot collide with any real reading (a real depth is always a positive, finite float).
- Ordinary arithmetic propagates it (mean, `+`, `*`, and `np.clip` all keep a NaN a NaN)
  instead of silently blending it into a plausible-looking finite number the way `0.0` did --
  so a future post-process step that touches depth without an explicit `np.isnan` check fails
  loudly (visibly NaN output) rather than quietly wrong.
- It matches the convention already used for "no valid depth" point clouds and depth maps
  elsewhere (e.g. Open3D/PCL treat an invalid point as NaN), so a consumer written against that
  convention needs no special-casing for this codebase.

Everything in `apply_postprocess` after the distortion remap (exposure, white balance, gamma,
noise, blur, JPEG) operates on the `rgb` channel only and never touches `depth`/`instance` --
that is what keeps this sentinel from ever being smeared into a neighbouring valid pixel by one
of those steps; if a future change ever needs to run any of them over depth too, it must
preserve NaN through that step explicitly (e.g. `cv2.GaussianBlur` does NOT propagate NaN the
way plain arithmetic does -- it would smear a NaN border into nearby valid pixels -- so such a
step would need to blur around/mask out NaNs rather than blur across them).
"""

# ---------------------------------------------------------------------------
# Colour (HSV) jitter, used for both bricks and the table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourJitterConfig:
    hue_jitter_deg: float = 10.0
    sat_jitter: float = 0.15
    val_jitter: float = 0.15


def jitter_rgb_hsv(
    rgb: tuple[float, float, float], cfg: ColourJitterConfig, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Jitters one RGB colour (each channel in [0, 1]) in HSV space and returns RGB."""
    arr = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hsv[0, 0, 0] = (hsv[0, 0, 0] + rng.uniform(-cfg.hue_jitter_deg, cfg.hue_jitter_deg)) % 360.0
    hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * (1.0 + rng.uniform(-cfg.sat_jitter, cfg.sat_jitter)), 0.0, 1.0)
    hsv[0, 0, 2] = np.clip(hsv[0, 0, 2] * (1.0 + rng.uniform(-cfg.val_jitter, cfg.val_jitter)), 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(3)
    return tuple(float(v) for v in np.clip(out, 0.0, 1.0))


def randomize_material_colour(
    material: mujoco.MjsMaterial,
    base_rgb: tuple[float, float, float],
    cfg: ColourJitterConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Sets `material.rgba` to a jittered version of `base_rgb`, keeping alpha. Returns the new RGB."""
    r, g, b = jitter_rgb_hsv(base_rgb, cfg, rng)
    alpha = float(material.rgba[3]) if material.rgba is not None else 1.0
    material.rgba = [r, g, b, alpha]
    return (r, g, b)


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LightRigConfig:
    n_lights: tuple[int, int] = (1, 4)
    pos_lo: tuple[float, float, float] = (-0.3, -0.3, 0.3)
    pos_hi: tuple[float, float, float] = (0.3, 0.3, 0.8)
    diffuse_base: tuple[float, float, float] = (0.7, 0.7, 0.7)
    colour_jitter: float = 0.3   # fractional per-channel gain jitter
    ambient: tuple[float, float, float] = (0.1, 0.1, 0.1)


def randomize_lights(spec: mujoco.MjSpec, cfg: LightRigConfig, rng: np.random.Generator) -> int:
    """Replaces every light on the worldbody with a freshly sampled rig. Returns the light count."""
    world = spec.worldbody
    for light in list(world.lights):
        spec.delete(light)

    n = int(rng.integers(cfg.n_lights[0], cfg.n_lights[1] + 1))
    lo = np.asarray(cfg.pos_lo, dtype=np.float64)
    hi = np.asarray(cfg.pos_hi, dtype=np.float64)
    base = np.asarray(cfg.diffuse_base, dtype=np.float64)
    for _ in range(n):
        pos = rng.uniform(lo, hi)
        gain = 1.0 + rng.uniform(-cfg.colour_jitter, cfg.colour_jitter, size=3)
        diffuse = np.clip(base * gain, 0.0, 1.0)
        world.add_light(
            pos=pos.tolist(),
            diffuse=diffuse.tolist(),
            specular=(diffuse * 0.3).tolist(),
            ambient=list(cfg.ambient),
            type=mujoco.mjtLightType.mjLIGHT_POINT,
        )
    return n


# ---------------------------------------------------------------------------
# Camera extrinsics / fov
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraJitterConfig:
    pos_jitter_m: float = 0.005
    rot_jitter_deg: float = 3.0
    fovy_jitter_deg: float = 3.0


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation of `angle` radians about `axis`."""
    half = angle * 0.5
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b, both (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def randomize_camera(
    spec: mujoco.MjSpec, camera_name: str, cfg: CameraJitterConfig, rng: np.random.Generator
) -> None:
    """Perturbs a named camera's position, orientation and vertical fov in place."""
    cam = spec.camera(camera_name)
    if cam is None:
        raise ValueError(f"unknown camera {camera_name!r}")

    cam.pos = np.asarray(cam.pos, dtype=np.float64) + rng.uniform(
        -cfg.pos_jitter_m, cfg.pos_jitter_m, size=3
    )

    axis = rng.normal(size=3)
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    angle = np.deg2rad(rng.uniform(-cfg.rot_jitter_deg, cfg.rot_jitter_deg))
    dq = _axis_angle_quat(axis, angle)
    cam.quat = _quat_mul(np.asarray(cam.quat, dtype=np.float64), dq)

    cam.fovy = float(cam.fovy) + rng.uniform(-cfg.fovy_jitter_deg, cfg.fovy_jitter_deg)


# ---------------------------------------------------------------------------
# Image post-processing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostProcessConfig:
    k1: float = 0.0                                        # radial distortion, 2nd order
    k2: float = 0.0                                        # radial distortion, 4th order
    exposure: float = 1.0                                  # multiplicative gain
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gamma: float = 1.0
    shot_noise: float = 0.0                                # signal-dependent noise scale
    read_noise: float = 0.0                                # constant gaussian sigma, in [0, 1] units
    blur_sigma: float = 0.0
    jpeg_quality: int | None = None                        # None disables the JPEG round-trip


# Sampling ranges for `sample_postprocess_config`, chosen to be visible but not
# scene-destroying; a caller doing its own sweep passes its own ranges instead.
DEFAULT_POSTPROCESS_RANGES: dict[str, tuple[float, float]] = {
    "k1": (-0.15, 0.15),
    "k2": (-0.05, 0.05),
    "exposure": (0.8, 1.25),
    "gamma": (0.85, 1.2),
    "shot_noise": (0.0, 0.02),
    "read_noise": (0.0, 0.01),
    "blur_sigma": (0.0, 1.0),
}


def sample_postprocess_config(
    rng: np.random.Generator, ranges: dict[str, tuple[float, float]] | None = None
) -> PostProcessConfig:
    """Draws one randomized `PostProcessConfig`."""
    r = dict(DEFAULT_POSTPROCESS_RANGES)
    if ranges:
        r.update(ranges)
    wb = 1.0 + rng.uniform(-0.1, 0.1, size=3)
    return PostProcessConfig(
        k1=float(rng.uniform(*r["k1"])),
        k2=float(rng.uniform(*r["k2"])),
        exposure=float(rng.uniform(*r["exposure"])),
        white_balance=tuple(float(v) for v in wb),
        gamma=float(rng.uniform(*r["gamma"])),
        shot_noise=float(rng.uniform(*r["shot_noise"])),
        read_noise=float(rng.uniform(*r["read_noise"])),
        blur_sigma=float(rng.uniform(*r["blur_sigma"])),
    )


def _distortion_maps(width: int, height: int, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray]:
    """Builds the `cv2.remap` source-coordinate maps for a simple radial distortion.

    Coordinates are normalized so the frame's half-diagonal-ish scale is 1, centred
    on the image; `factor = 1 + k1*r^2 + k2*r^4` maps distorted image points back to
    where they came from in the pristine render (this remap direction is what
    `cv2.remap` wants: for each output pixel, where in the source to sample).
    """
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    scale = max(width, height) / 2.0
    x = (xs - cx) / scale
    y = (ys - cy) / scale
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    map_x = (x * factor * scale + cx).astype(np.float32)
    map_y = (y * factor * scale + cy).astype(np.float32)
    return map_x, map_y


def apply_postprocess(frame: Frame, cfg: PostProcessConfig, rng: np.random.Generator) -> Frame:
    """Applies lens distortion, exposure/colour, noise, blur and JPEG to a rendered `Frame`.

    Depth and instance channels only ever go through the distortion remap, and with
    nearest-neighbour sampling, so a depth or instance value in the output was always copied
    whole from one input pixel, never blended -- except at a pixel the remap pulls from outside
    the pristine render's bounds, which has no source pixel to copy at all. Those get
    `DEPTH_INVALID` (NaN) for depth and `-1` for instance (already the case), never a value that
    could pass for a real reading. Nothing below this point (exposure, white balance, gamma,
    noise, blur, JPEG) touches `depth`/`instance` at all -- they operate on `rgb` only -- so
    there is nothing after the remap that could smear either sentinel into a neighbouring valid
    pixel; see `DEPTH_INVALID`'s own docstring for what a future change that did touch depth
    would need to preserve.
    """
    height, width = frame.rgb.shape[:2]
    rgb = frame.rgb.astype(np.float32) / 255.0
    depth = frame.depth
    instance = frame.instance

    if cfg.k1 != 0.0 or cfg.k2 != 0.0:
        map_x, map_y = _distortion_maps(width, height, cfg.k1, cfg.k2)
        rgb = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0.0)
        depth = cv2.remap(
            depth, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderValue=DEPTH_INVALID
        )
        instance = cv2.remap(
            instance.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderValue=-1.0
        )

    wb = np.asarray(cfg.white_balance, dtype=np.float32)
    rgb = rgb * cfg.exposure * wb

    if cfg.gamma != 1.0:
        rgb = np.clip(rgb, 0.0, None) ** (1.0 / cfg.gamma)
    rgb = np.clip(rgb, 0.0, 1.0)

    if cfg.shot_noise > 0.0:
        rgb = rgb + rng.normal(0.0, 1.0, size=rgb.shape).astype(np.float32) * np.sqrt(rgb) * cfg.shot_noise
    if cfg.read_noise > 0.0:
        rgb = rgb + rng.normal(0.0, cfg.read_noise, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    if cfg.blur_sigma > 0.0:
        k = max(3, int(2 * round(3 * cfg.blur_sigma) + 1))
        rgb = cv2.GaussianBlur(rgb, (k, k), cfg.blur_sigma)

    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if cfg.jpeg_quality is not None:
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            rgb_u8 = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return Frame(
        rgb=rgb_u8,
        depth=depth.astype(np.float32),
        instance=np.rint(instance).astype(np.int32),
        body_ids=frame.body_ids,
    )


def plot_postprocess_before_after(frame: Frame, cfg: PostProcessConfig, rng: np.random.Generator, out_path, title: str = "Camera post-process"):
    """Saves a PNG with the raw render next to the post-processed one, `cfg` printed below.

    The figure-producing function `scripts/make_media.py` calls for this work
    package's before/after panel (spec section 4).
    """
    import matplotlib

    matplotlib.use("Agg")  # no GUI windows in tests or batch media builds
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = apply_postprocess(frame, cfg, rng)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=150)
    axes[0].imshow(frame.rgb)
    axes[0].set_title("Before")
    axes[1].imshow(processed.rgb)
    axes[1].set_title("After")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    caption = (
        f"k1={cfg.k1:.3f} k2={cfg.k2:.3f} exposure={cfg.exposure:.2f} gamma={cfg.gamma:.2f} "
        f"shot={cfg.shot_noise:.3f} read={cfg.read_noise:.3f} blur_sigma={cfg.blur_sigma:.2f} "
        f"jpeg_q={cfg.jpeg_quality}"
    )
    fig.suptitle(title)
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
