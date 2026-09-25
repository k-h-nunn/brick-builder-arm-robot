"""The training-frame generator: the instance convention, the targets, and shard storage."""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.perception.cnn import data as D
from brick_builder.perception.cnn.labels import ARM_INSTANCE, FIRST_BRICK_INSTANCE, PLATE_INSTANCE
from brick_builder.perception.cnn.model import OFFSET_SCALE
from brick_builder.perception.cnn.train import make_targets


@pytest.fixture(scope="module")
def scene():
    from brick_builder.bricks.structures import Placement
    from brick_builder.scene.compose import SceneConfig, build_scene

    return build_scene(SceneConfig(
        held=None, tray=True, tray_bricks=2, cameras=(),
        placed=[Placement(h=2, w=4, x=1, y=2, z=0, colour="blue")],
    ))


def test_instance_map_separates_the_table_from_the_arm(scene):
    """`scene.compose` gives the world body -- and so the table plane -- the arm's instance id.
    This package's own map must not: the table is background."""
    lut = D.perception_instance_lut(scene)
    assert "world" not in lut
    assert "tray" not in lut
    for name in D.ARM_BODIES:
        assert lut[name] == ARM_INSTANCE
    assert lut[scene.plate.name] == PLATE_INSTANCE
    bricks = [v for k, v in lut.items() if v >= FIRST_BRICK_INSTANCE]
    assert sorted(bricks) == list(range(FIRST_BRICK_INSTANCE, FIRST_BRICK_INSTANCE + len(bricks)))
    assert scene.instance_ids["world"] == ARM_INSTANCE  # the behaviour this map exists to avoid


def test_repainting_a_brick_changes_its_studs_too(scene):
    """A stud is moulded with its brick and must carry its colour; a grey stud would be a
    rendering bug, and the model would learn the wrong thing from it."""
    import mujoco

    name = next(n for n in scene.bricks if n.startswith("tray_brick_"))
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
    start = int(scene.model.body_geomadr[bid])
    n_geom = int(scene.model.body_geomnum[bid])
    stud_geoms = [
        g for g in range(start, start + n_geom)
        if "_vis_stud_" in (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
    ]
    assert stud_geoms, "the brick generator should build visual stud geoms"
    D.paint_brick(scene, name, (0.91, 0.12, 0.13))
    for g in stud_geoms:
        np.testing.assert_allclose(scene.model.geom_rgba[g, :3], (0.91, 0.12, 0.13), atol=1e-6)


def test_arm_colour_is_randomized_over_printed_part_colours(scene):
    seen = set()
    for seed in range(30):
        seen.add(D.paint_arm(scene, np.random.default_rng(seed)))
    assert len(seen) >= 4, seen
    assert seen <= set(D.PRINTED_PART_COLOURS)


def test_one_rendered_sample_has_aligned_labels(scene):
    from brick_builder.render.renderer import SceneRenderer
    from brick_builder.scene.compose import reset_scene

    rng = np.random.default_rng(3)
    reset_scene(scene, rng)
    D.randomize_appearance(scene, rng)
    q = D._viewing_pose(scene, scene.cfg.plate_xy, rng)
    assert q is not None
    D._set_arm(scene, q)
    diagonal, post = D.sample_camera_model(rng)
    pose = D.window_for_roi(scene, D.roi_around([*scene.cfg.plate_xy, 0.02]), diagonal, None)
    assert pose is not None
    renderer = SceneRenderer(scene.model, width=D.RENDER_SIZE[0], height=D.RENDER_SIZE[1])
    try:
        renderer.instance_lut(D.perception_instance_lut(scene))
        sample = D.render_sample(scene, pose, renderer, post, rng, {"kind": "test"})
    finally:
        renderer.close()
    assert sample.rgb.shape == (256, 256, 3) and sample.rgb.dtype == np.uint8
    assert sample.instance.shape == (256, 256) and sample.instance.dtype == np.uint8
    assert sample.depth.shape == (256, 256)
    assert set(np.unique(sample.instance)) <= set(range(0, 16))
    assert sample.classes.max() <= 3


def test_shards_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    samples = [
        D.Sample(
            rgb=rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            instance=rng.integers(0, 5, (256, 256)).astype(np.uint8),
            depth=rng.random((256, 256)).astype(np.float16),
            meta={"kind": "synthetic", "i": i},
        )
        for i in range(3)
    ]
    path = tmp_path / "train" / "shard_0000.npz"
    D.write_shard(path, samples)
    back = D.read_shard(path)
    assert len(back) == 3
    np.testing.assert_array_equal(back[1].rgb, samples[1].rgb)
    np.testing.assert_array_equal(back[1].instance, samples[1].instance)
    assert back[2].meta["i"] == 2
    assert D.dataset_size(tmp_path, "train") == 3


def test_merge_split_renumbers_without_losing_a_shard(tmp_path):
    def block(split, n):
        for i in range(n):
            D.write_shard(tmp_path / split / f"shard_{i:04d}.npz", [
                D.Sample(rgb=np.zeros((4, 4, 3), np.uint8), instance=np.zeros((4, 4), np.uint8),
                         depth=np.zeros((4, 4), np.float16), meta={"split": split, "i": i})
            ])

    block("train", 2)
    block("train_b", 3)
    assert D.merge_split(tmp_path, "train_b", "train") == 3
    assert len(D.shard_paths(tmp_path, "train")) == 5
    assert not D.shard_paths(tmp_path, "train_b")


# ------------------------------------------------------------------------------- the targets ---


def test_targets_point_every_brick_pixel_at_its_own_centre():
    instance = np.zeros((64, 64), dtype=np.uint8)
    instance[10:30, 10:20] = FIRST_BRICK_INSTANCE
    instance[40:60, 40:50] = FIRST_BRICK_INSTANCE + 1
    depth = np.full((64, 64), 0.2, dtype=np.float16)
    t = make_targets(instance, depth)

    for inst_id, (r0, r1, c0, c1) in ((FIRST_BRICK_INSTANCE, (10, 30, 10, 20)),
                                      (FIRST_BRICK_INSTANCE + 1, (40, 60, 40, 50))):
        cy, cx = (r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0
        row, col = r0 + 2, c0 + 3
        assert t["offset"][0, row, col] * OFFSET_SCALE == pytest.approx(cx - col, abs=1e-3)
        assert t["offset"][1, row, col] * OFFSET_SCALE == pytest.approx(cy - row, abs=1e-3)
        # The bump peaks at the true (fractional) centroid, so the nearest whole pixel is up to
        # half a pixel off its top in each axis -- close to 1, not exactly 1.
        assert t["heat"][int(round(cy)), int(round(cx))] > 0.9
        far = t["heat"][r0 - 8:r0 - 4, c0 - 8:c0 - 4]      # well outside this brick
        assert far.size and far.max() < 0.05
    assert np.all(t["offset"][:, instance == 0] == 0.0)
    assert t["offset_mask"].sum() == float((instance >= FIRST_BRICK_INSTANCE).sum())


def test_no_centre_bump_for_the_arm_or_the_baseplate():
    instance = np.zeros((64, 64), dtype=np.uint8)
    instance[:20] = ARM_INSTANCE
    instance[30:50] = PLATE_INSTANCE
    t = make_targets(instance, np.full((64, 64), 0.2, dtype=np.float16))
    assert t["heat"].max() == 0.0


def test_depth_target_is_scale_invariant():
    instance = np.zeros((32, 32), dtype=np.uint8)
    depth = np.linspace(0.1, 0.3, 32 * 32).reshape(32, 32).astype(np.float16)
    a = make_targets(instance, depth)["depth"]
    b = make_targets(instance, (depth.astype(np.float32) * 2.0 + 1.0).astype(np.float16))["depth"]
    np.testing.assert_allclose(a, b, atol=0.02)


def test_depth_target_marks_invalid_pixels_rather_than_inventing_a_value():
    depth = np.full((32, 32), 0.2, dtype=np.float32)
    depth[:8] = np.nan                       # `domain_rand.DEPTH_INVALID`
    t = make_targets(np.zeros((32, 32), dtype=np.uint8), depth.astype(np.float16))
    assert t["depth_mask"][:8].sum() == 0.0
    assert np.all(np.isfinite(t["depth"]))


def test_a_structure_is_always_a_valid_one():
    from brick_builder.bricks.structures import validate_structure

    for seed in range(40):
        placed = D._random_structure(np.random.default_rng(seed), touching_same_colour=bool(seed % 2))
        assert validate_structure(placed, grid_shape=(8, 8)) == []
        assert len(placed) <= 6


def test_bright_plate_scene_forces_a_light_brick_on_a_pale_table():
    """The targeted scene kind (white bricks on a bright table, see the guide's failure
    analysis): every frame carries a forced white or yellow brick, and the table colour is drawn
    from the pale half of `TABLE_RGB_RANGE` only."""
    lo, hi = D.TABLE_RGB_RANGE
    mid = 0.5 * (np.asarray(lo) + np.asarray(hi))
    samples = list(D._bright_light_plate_scene(np.random.default_rng(0), frames=3))
    assert samples, "the scene kind should yield at least one frame"
    for s in samples:
        assert s.meta["kind"] == "bright_plate"
        assert set(s.meta["forced_light"]) <= {"white", "yellow"}
        table_rgb = np.asarray(s.meta["table_rgb"])
        assert np.all(table_rgb >= mid - 1e-9)
        assert np.all(table_rgb <= np.asarray(hi) + 1e-9)
