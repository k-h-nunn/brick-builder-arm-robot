"""Tests for brick_builder.render.domain_rand."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from brick_builder.render.domain_rand import (
    DEPTH_INVALID,
    CameraJitterConfig,
    ColourJitterConfig,
    LightRigConfig,
    PostProcessConfig,
    apply_postprocess,
    jitter_rgb_hsv,
    randomize_camera,
    randomize_lights,
    randomize_material_colour,
    sample_postprocess_config,
)
from brick_builder.render.renderer import SceneRenderer


def _model_data(mjcf: str):
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# -- colour jitter -----------------------------------------------------------


def test_hsv_jitter_stays_in_gamut_and_is_seeded():
    base = (0.8, 0.1, 0.1)
    cfg = ColourJitterConfig(hue_jitter_deg=20.0, sat_jitter=0.3, val_jitter=0.3)

    a = jitter_rgb_hsv(base, cfg, np.random.default_rng(0))
    b = jitter_rgb_hsv(base, cfg, np.random.default_rng(0))
    c = jitter_rgb_hsv(base, cfg, np.random.default_rng(1))

    assert a == b                      # same seed -> identical result
    assert a != c                      # different seed -> (almost certainly) different result
    assert all(0.0 <= v <= 1.0 for v in a)
    assert a != base                   # jitter actually did something


def test_zero_jitter_is_identity():
    base = (0.3, 0.6, 0.9)
    cfg = ColourJitterConfig(hue_jitter_deg=0.0, sat_jitter=0.0, val_jitter=0.0)
    out = jitter_rgb_hsv(base, cfg, np.random.default_rng(0))
    assert np.allclose(out, base, atol=1e-5)


def test_randomize_material_colour_mutates_spec_material(two_box_mjcf):
    spec = mujoco.MjSpec.from_string(two_box_mjcf)
    mat = spec.material("mat_red")
    before = tuple(mat.rgba)
    new_rgb = randomize_material_colour(mat, (0.8, 0.1, 0.1), ColourJitterConfig(), np.random.default_rng(0))

    assert tuple(mat.rgba[:3]) == pytest.approx(new_rgb)
    assert mat.rgba[3] == before[3]     # alpha untouched
    assert tuple(mat.rgba) != before

    # The mutated spec still compiles and renders.
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with SceneRenderer(model, 32, 32) as renderer:
        renderer.instance_lut({"box_red": 1})
        frame = renderer.render(data, "cam0")
    assert frame.rgb.shape == (32, 32, 3)


# -- lights --------------------------------------------------------------


def test_randomize_lights_count_and_seed(two_box_mjcf):
    cfg = LightRigConfig(n_lights=(2, 5))

    spec_a = mujoco.MjSpec.from_string(two_box_mjcf)
    n_a = randomize_lights(spec_a, cfg, np.random.default_rng(7))
    lights_a = [(l.pos.tolist(), l.diffuse.tolist()) for l in spec_a.worldbody.lights]

    spec_b = mujoco.MjSpec.from_string(two_box_mjcf)
    n_b = randomize_lights(spec_b, cfg, np.random.default_rng(7))
    lights_b = [(l.pos.tolist(), l.diffuse.tolist()) for l in spec_b.worldbody.lights]

    assert 2 <= n_a <= 5
    assert n_a == n_b
    assert lights_a == lights_b          # same seed -> identical rig

    spec_a.compile()                     # still compiles


def test_randomize_lights_replaces_existing(two_box_mjcf):
    spec = mujoco.MjSpec.from_string(two_box_mjcf)
    assert len(list(spec.worldbody.lights)) == 1   # the original scene light
    randomize_lights(spec, LightRigConfig(n_lights=(3, 3)), np.random.default_rng(0))
    assert len(list(spec.worldbody.lights)) == 3


# -- camera ----------------------------------------------------------------


def test_randomize_camera_perturbs_pose_and_fovy_and_is_seeded(two_box_mjcf):
    cfg = CameraJitterConfig(pos_jitter_m=0.01, rot_jitter_deg=5.0, fovy_jitter_deg=4.0)

    spec_a = mujoco.MjSpec.from_string(two_box_mjcf)
    cam_a = spec_a.camera("cam0")
    base_pos, base_fovy = np.array(cam_a.pos), cam_a.fovy
    randomize_camera(spec_a, "cam0", cfg, np.random.default_rng(3))

    spec_b = mujoco.MjSpec.from_string(two_box_mjcf)
    randomize_camera(spec_b, "cam0", cfg, np.random.default_rng(3))
    cam_b = spec_b.camera("cam0")

    assert not np.allclose(cam_a.pos, base_pos)
    assert cam_a.fovy != base_fovy
    assert np.allclose(cam_a.pos, cam_b.pos)             # same seed -> same pose
    assert cam_a.fovy == pytest.approx(cam_b.fovy)
    assert np.allclose(cam_a.quat, cam_b.quat)
    assert np.isclose(np.linalg.norm(cam_a.quat), 1.0)   # still a unit quaternion


def test_randomize_camera_unknown_name_raises(two_box_mjcf):
    spec = mujoco.MjSpec.from_string(two_box_mjcf)
    with pytest.raises(ValueError):
        randomize_camera(spec, "no_such_camera", CameraJitterConfig(), np.random.default_rng(0))


# -- post-processing ---------------------------------------------------------


def _render_frame(mjcf: str, width=64, height=64):
    model, data = _model_data(mjcf)
    with SceneRenderer(model, width, height) as renderer:
        renderer.instance_lut({"box_red": 1, "box_blue": 2})
        return renderer.render(data, "cam0")


def test_sample_postprocess_config_is_seeded(two_box_mjcf):
    a = sample_postprocess_config(np.random.default_rng(5))
    b = sample_postprocess_config(np.random.default_rng(5))
    c = sample_postprocess_config(np.random.default_rng(6))
    assert a == b
    assert a != c


def test_postprocess_changes_rgb_but_preserves_shape_and_dtype(two_box_mjcf):
    frame = _render_frame(two_box_mjcf)
    cfg = sample_postprocess_config(np.random.default_rng(1))
    out = apply_postprocess(frame, cfg, np.random.default_rng(2))

    assert out.rgb.shape == frame.rgb.shape and out.rgb.dtype == np.uint8
    assert out.depth.shape == frame.depth.shape and out.depth.dtype == np.float32
    assert out.instance.shape == frame.instance.shape and out.instance.dtype == np.int32
    assert not np.array_equal(out.rgb, frame.rgb)


def test_postprocess_is_seeded(two_box_mjcf):
    frame = _render_frame(two_box_mjcf)
    cfg = PostProcessConfig(read_noise=0.05, shot_noise=0.02, blur_sigma=0.5, gamma=1.2, exposure=1.1)

    a = apply_postprocess(frame, cfg, np.random.default_rng(42))
    b = apply_postprocess(frame, cfg, np.random.default_rng(42))
    c = apply_postprocess(frame, cfg, np.random.default_rng(43))

    assert np.array_equal(a.rgb, b.rgb)
    assert not np.array_equal(a.rgb, c.rgb)


def test_postprocess_identity_config_only_rounds(two_box_mjcf):
    """With every effect at its neutral value, post-process must not change the image."""
    frame = _render_frame(two_box_mjcf)
    cfg = PostProcessConfig()   # k=0, exposure=1, wb=1,1,1, gamma=1, no noise, no blur, no jpeg
    out = apply_postprocess(frame, cfg, np.random.default_rng(0))
    assert np.array_equal(out.rgb, frame.rgb)
    assert np.array_equal(out.depth, frame.depth)
    assert np.array_equal(out.instance, frame.instance)


def test_distortion_remap_uses_nearest_neighbour_for_depth_and_instance(two_box_mjcf):
    """Every depth/instance value after distortion must have come from some input pixel, or --
    for a pixel the remap pulls from outside the source frame -- be that channel's documented
    invalid-value sentinel (perception-align-003): NaN for depth, -1 for instance, never a
    value that could pass for a real reading (the old bug: depth used 0.0, "touching the
    camera lens", a physically impossible reading for this scene)."""
    frame = _render_frame(two_box_mjcf, width=64, height=64)
    cfg = PostProcessConfig(k1=0.2, k2=-0.05)
    out = apply_postprocess(frame, cfg, np.random.default_rng(0))

    border = np.isnan(out.depth)
    assert border.any()  # this k1/k2 does push some pixels out of the source frame

    in_depth_values = set(np.unique(frame.depth).tolist())
    out_depth_in_bounds = set(np.unique(out.depth[~border]).tolist())
    assert out_depth_in_bounds <= in_depth_values
    assert 0.0 not in out_depth_in_bounds  # the old sentinel must never appear, in bounds or out

    in_instance_values = set(np.unique(frame.instance).tolist())
    out_instance_values = set(np.unique(out.instance).tolist())
    assert out_instance_values <= (in_instance_values | {-1})

    # This fixture has no true background (the table fills the whole frame), so every -1 in the
    # output, like every NaN, can only have come from the same border-fill case -- both remaps
    # share the same source-coordinate maps, so the two sentinels must land on exactly the same
    # pixels.
    assert np.array_equal(border, out.instance == -1)

    # And the RGB channel actually moved (bilinear-sampled distortion, not a no-op).
    assert not np.array_equal(out.rgb, frame.rgb)


def test_distortion_remap_is_area_preserving_ish(two_box_mjcf):
    """A pincushion/barrel remap must not collapse the whole frame to one colour."""
    frame = _render_frame(two_box_mjcf, width=64, height=64)
    cfg = PostProcessConfig(k1=0.3)
    out = apply_postprocess(frame, cfg, np.random.default_rng(0))
    assert out.rgb.std() > 0


# --- perception-align-003: out-of-bounds depth must be NaN, never 0.0 -------------------------


def test_lens_distortion_at_k1_0_15_fills_out_of_bounds_depth_with_nan(two_box_mjcf):
    """The review's own reproduction parameter: at k1=0.15 (the top of
    `DEFAULT_POSTPROCESS_RANGES['k1']`), a real fraction of the frame -- including all four
    corners -- falls outside the source image once remapped (the review measured 20.5% of
    pixels reading exactly depth=0.0m against a true scene range of a few centimetres). Every
    one of those pixels must now read NaN, and no in-bounds pixel may read the old sentinel
    either."""
    frame = _render_frame(two_box_mjcf, width=64, height=64)
    cfg = PostProcessConfig(k1=0.15)
    out = apply_postprocess(frame, cfg, np.random.default_rng(0))

    border = np.isnan(out.depth)
    assert border.any()
    assert border[0, 0] and border[0, -1] and border[-1, 0] and border[-1, -1]  # all 4 corners

    in_bounds_depth = out.depth[~border]
    assert in_bounds_depth.size > 0
    assert not np.isnan(in_bounds_depth).any()
    assert (in_bounds_depth > 0.0).all()  # no in-bounds pixel got the old 0.0 "at the lens" value

    # Same border positions as the instance channel's own -1 sentinel (see the nearest-neighbour
    # test above for why that must hold on this no-true-background fixture).
    assert np.array_equal(border, out.instance == -1)

    # A quantization/rounding pass over depth (there isn't one today, but if one is ever added)
    # must not turn NaN into a finite number -- pin down that plain float32 rounding does not.
    assert np.isnan(np.round(out.depth[border]).astype(np.float32)).all()


def test_depth_invalid_sentinel_is_nan_and_documented():
    """`DEPTH_INVALID` is the one place this sentinel is spelled out; a consumer (or another
    test) checks for it with `np.isnan`, so it must actually be NaN, not e.g. a large negative
    number that merely reads that way in a docstring."""
    assert np.isnan(DEPTH_INVALID)


def test_zero_distortion_never_introduces_the_sentinel(two_box_mjcf):
    """No distortion (k1=k2=0, the identity-config test's own scenario) skips the remap
    entirely -- depth must come through byte-for-byte with no NaN introduced."""
    frame = _render_frame(two_box_mjcf, width=32, height=32)
    cfg = PostProcessConfig(k1=0.0, k2=0.0)
    out = apply_postprocess(frame, cfg, np.random.default_rng(0))
    assert not np.isnan(out.depth).any()
    assert np.array_equal(out.depth, frame.depth)
