"""`CnnSceneModel`: the shared producer contract, the viewer API, determinism, and a loose
check on cached frames.

The contract checks are the same function applied to the ideal ground-truth source, the
colour-threshold source and this one (`conftest.assert_scene_state_contract`), because the whole
point of the interface is that a consumer cannot tell which producer it is reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.perception.cnn.postprocess import GroupingConfig
from brick_builder.perception.conventional_scene_model import ColourSegmentationSceneModel
from brick_builder.perception.ideal_scene_model import IdealSceneModel
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneModelSource

from .conftest import WEIGHTS, assert_scene_state_contract


def test_every_producer_fills_the_same_contract(val_samples, trained_source):
    """One set of assertions, three producers."""
    sample = val_samples[0]
    states = {
        "ideal": IdealSceneModel().observe(
            sample.rgb, window=None, instance=sample.instance.astype(np.int32),
            depth=sample.depth.astype(np.float64)),
        "colour": ColourSegmentationSceneModel().observe(sample.rgb, window=None),
        "conv_net": trained_source.observe(sample.rgb, window=None),
    }
    for name, state in states.items():
        assert_scene_state_contract(state)
        assert state.schema is PLACEHOLDER_SLOT_SCHEMA, name
    assert states["conv_net"].source == "conv_net"


def test_satisfies_the_runtime_protocol(trained_source):
    assert isinstance(trained_source, SceneModelSource)
    assert hasattr(trained_source, "reset")


def test_missing_checkpoint_says_how_to_make_one(tmp_path):
    from brick_builder.perception.cnn.source import CnnSceneModel

    with pytest.raises(FileNotFoundError, match="train_vision_model"):
        CnnSceneModel(checkpoint=tmp_path / "nothing.pt")


def test_same_frame_gives_the_same_slot_rows(val_samples):
    """Two freshly built sources see the same frame and must publish identical rows -- there is
    no sampling anywhere in this producer, and the slot bookkeeping starts from the same state."""
    from brick_builder.perception.cnn.source import CnnSceneModel

    if not WEIGHTS.exists():
        pytest.skip("no trained checkpoint")
    rgb = val_samples[1].rgb
    a = CnnSceneModel(checkpoint=WEIGHTS).observe(rgb, window=None)
    b = CnnSceneModel(checkpoint=WEIGHTS).observe(rgb, window=None)
    np.testing.assert_array_equal(a.slots, b.slots)
    np.testing.assert_array_equal(a.depth, b.depth)
    np.testing.assert_array_equal(a.slot_ids, b.slot_ids)


def test_ground_truth_arguments_are_ignored(val_samples, trained_source):
    """A vision model that quietly read the simulator's instance map would not be one."""
    sample = val_samples[2]
    trained_source.reset()
    plain = trained_source.observe(sample.rgb, window=None)
    trained_source.reset()
    nonsense = np.full_like(sample.instance, 9)
    fed = trained_source.observe(sample.rgb, window=None, instance=nonsense.astype(np.int32),
                                 depth=np.zeros_like(sample.depth, dtype=np.float64))
    np.testing.assert_array_equal(plain.slots, fed.slots)


def test_present_mask_excludes_pixels(val_samples, trained_source):
    sample = val_samples[0]
    mask = np.zeros(sample.rgb.shape[:2], dtype=bool)
    mask[:64] = True
    trained_source.reset()
    state = trained_source.observe(sample.rgb, window=None, present_mask=mask)
    background = trained_source.assigner.background_slot
    for k in range(state.K):
        # The background slot covers every unclaimed pixel of the frame, masked or not -- the same
        # convention the ideal and colour sources use -- so it is not an object and is skipped.
        if state.is_empty(k) or k == background:
            continue
        _, cy = state.centre(k)
        assert cy < 0.0, "no object may be reported outside the region left visible"


def test_reset_clears_identity(val_samples, trained_source):
    sample = val_samples[3]
    trained_source.reset()
    first = trained_source.observe(sample.rgb, window=None)
    second = trained_source.observe(sample.rgb, window=None)
    ages_grow = any(second.age(k) > first.age(k) for k in range(first.K) if not first.is_empty(k))
    assert ages_grow
    trained_source.reset()
    after = trained_source.observe(sample.rgb, window=None)
    np.testing.assert_allclose(after.slots, first.slots, atol=1e-6)


def test_rejects_a_frame_it_cannot_process(trained_source):
    with pytest.raises(ValueError, match="square"):
        trained_source.observe(np.zeros((256, 128, 3), dtype=np.uint8), window=None)
    with pytest.raises(ValueError, match="multiple of 32"):
        trained_source.observe(np.zeros((100, 100, 3), dtype=np.uint8), window=None)


def test_accepts_float_and_byte_frames_alike(val_samples, trained_source):
    sample = val_samples[4]
    trained_source.reset()
    a = trained_source.observe(sample.rgb, window=None)
    trained_source.reset()
    b = trained_source.observe(sample.rgb.astype(np.float32) / 255.0, window=None)
    np.testing.assert_allclose(a.slots, b.slots, atol=1e-5)


def test_grouping_thresholds_are_honoured(val_samples):
    from brick_builder.perception.cnn.source import CnnSceneModel

    if not WEIGHTS.exists():
        pytest.skip("no trained checkpoint")
    sample = val_samples[0]
    strict = CnnSceneModel(checkpoint=WEIGHTS,
                                     grouping=GroupingConfig(centre_threshold=0.99))
    loose = CnnSceneModel(checkpoint=WEIGHTS,
                                    grouping=GroupingConfig(centre_threshold=0.05))
    strict.observe(sample.rgb, window=None)
    loose.observe(sample.rgb, window=None)
    assert strict.last_object_count <= loose.last_object_count


# ------------------------------------------------------------------------- accuracy, loosely ---


def test_accuracy_smoke_on_cached_frames(val_samples, trained_source):
    """Twenty held-out frames, thresholds set well below what the trained model reaches -- this
    is a "the checkpoint in the repository is not broken" test, not the evaluation. The real
    numbers live in `brick_builder.perception.cnn.evaluate` and in the guide's table."""
    from brick_builder.perception.cnn.evaluate import evaluate_frames

    report = evaluate_frames(val_samples[:20], trained_source)
    assert report["iou_brick"] > 0.45, report["iou_brick"]
    assert report["iou_baseplate"] > 0.50, report["iou_baseplate"]
    assert report["iou_arm"] > 0.50, report["iou_arm"]
    assert report["miou"] > 0.55, report["miou"]
    assert report["brick_recall"] > 0.50, report["brick_recall"]
    assert report["centroid_px"] < 12.0, report["centroid_px"]
    assert report["cnn_slot_match_rate"] > 0.55, report["cnn_slot_match_rate"]


def test_checkpoint_stays_small_enough_to_ship():
    if not WEIGHTS.exists():
        pytest.skip("no trained checkpoint")
    assert WEIGHTS.stat().st_size < 10 * 1024 * 1024, "the shipped checkpoint must stay under 10 MB"


# ------------------------------------------------------------------------------- viewer API ---


def test_feature_maps_shapes_and_slot_consistency(val_samples, trained_source):
    """`instance_map` must index the very slots of the `SceneState` returned with it."""
    from brick_builder.perception.cnn.labels import CLASS_NAMES

    trained_source.reset()
    maps = trained_source.feature_maps(val_samples[0].rgb, window=None)
    res = val_samples[0].rgb.shape[0]
    assert maps["class_prob"].shape == (len(CLASS_NAMES), res, res)
    assert maps["class_prob"].dtype == np.float32
    assert maps["centre_heatmap"].shape == (res, res)
    assert maps["offset"].shape == (2, res, res)
    assert maps["depth"].shape == (res, res)
    assert maps["instance_map"].shape == (res, res)
    assert maps["class_names"] == CLASS_NAMES
    np.testing.assert_allclose(maps["class_prob"].sum(axis=0), 1.0, atol=1e-4)
    assert 0.0 <= maps["centre_heatmap"].min() and maps["centre_heatmap"].max() <= 1.0

    state = maps["state"]
    for value in np.unique(maps["instance_map"]):
        if value == 0:
            continue
        assert not state.is_empty(int(value) - 1), "instance_map indexes an empty slot"


def test_feature_maps_agrees_with_observe_on_the_same_frame(val_samples, trained_source):
    rgb = val_samples[1].rgb
    trained_source.reset()
    a = trained_source.observe(rgb, window=None)
    trained_source.reset()
    b = trained_source.feature_maps(rgb, window=None)["state"]
    np.testing.assert_array_equal(a.slots, b.slots)


def test_colourize_instances_paints_each_slot_its_published_colour(val_samples, trained_source):
    trained_source.reset()
    maps = trained_source.feature_maps(val_samples[0].rgb, window=None)
    image = trained_source.colourize_instances(maps["instance_map"], maps["state"])
    assert image.shape == (*maps["instance_map"].shape, 3)
    assert image.dtype == np.uint8
    for k in range(maps["state"].K):
        mask = maps["instance_map"] == k + 1
        if mask.sum() < 50:
            continue
        published = np.clip((np.asarray(maps["state"].colour(k)) + 1.0) * 0.5, 0.0, 1.0)
        painted = image[mask].astype(np.float64).mean(axis=0) / 255.0
        # The painted colour is the published one lifted towards white; the ordering of the
        # channels must survive that, which is what makes the picture readable.
        assert np.argmax(painted) == np.argmax(published) or published.ptp() < 0.05


def test_colourize_scalar_is_a_perceptual_ramp():
    from brick_builder.perception.cnn.source import CnnSceneModel

    ramp = np.linspace(0.0, 1.0, 64)[None].repeat(8, axis=0).astype(np.float32)
    image = CnnSceneModel.colourize_scalar(ramp, 0.0, 1.0)
    assert image.shape == (8, 64, 3) and image.dtype == np.uint8
    luma = image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    assert luma[0, -1] > luma[0, 0]                     # bright end is the high end
    assert np.all(np.diff(luma[0]) > -12.0)             # monotone enough to read as an ordering


def test_accuracy_matches_the_evaluation_on_a_perfect_prediction(val_samples, trained_source):
    """Fed the renderer's own labels as if they were the prediction, every score must be 1."""
    from brick_builder.perception.cnn.labels import FIRST_BRICK_INSTANCE

    sample = val_samples[0]
    trained_source.reset()
    trained_source.feature_maps(sample.rgb, window=None)       # so a class map exists
    perfect = np.zeros(sample.instance.shape, dtype=np.int32)
    for n, inst_id in enumerate(i for i in np.unique(sample.instance) if i != 0):
        perfect[sample.instance == inst_id] = n + 1
    scores = trained_source.accuracy(perfect, sample.instance, sample.classes,
                                     class_map=sample.classes.astype(np.int32))
    assert scores["miou"] == pytest.approx(1.0)
    assert scores["pixel_accuracy"] == pytest.approx(1.0)
    assert scores["brick_recall"] == pytest.approx(1.0)
    assert scores["brick_precision"] == pytest.approx(1.0)
    assert scores["bricks_true"] == int((np.unique(sample.instance) >= FIRST_BRICK_INSTANCE).sum())


def test_accuracy_on_the_real_prediction_is_within_range(val_samples, trained_source):
    sample = val_samples[0]
    trained_source.reset()
    maps = trained_source.feature_maps(sample.rgb, window=None)
    scores = trained_source.accuracy(maps["instance_map"], sample.instance, sample.classes)
    for key in ("miou", "pixel_accuracy", "brick_recall", "brick_precision"):
        value = scores[key]
        assert np.isnan(value) or 0.0 <= value <= 1.0, (key, value)
    assert set(scores["iou"]) == set(maps["class_names"])
