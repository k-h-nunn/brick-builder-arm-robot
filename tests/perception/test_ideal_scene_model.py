"""Tests for `brick_builder.perception.ideal_scene_model.IdealSceneModel`.

Two kinds of test: pure array tests against synthetic ground truth (fast, exact), and a handful
against an actual rendered scene (`brick_builder.scene.compose`/`brick_builder.perception.views`), so the
whole render -> ground truth -> slot-state path is exercised at least once end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.bricks.structures import Placement
from brick_builder.perception.ideal_scene_model import IdealSceneModel
from brick_builder.perception.views import PLATE_VIEW, goto_look, render_view
from brick_builder.render.renderer import SceneRenderer
from brick_builder.robot.patch_so101 import WRIST_CAM
from brick_builder.scene.compose import SceneConfig, build_scene


def _synthetic(res=64):
    instance = np.full((res, res), -1, dtype=np.int32)
    instance[10:20, 10:20] = 2   # baseplate-like
    instance[30:34, 30:34] = 4   # brick
    instance[40:44, 40:44] = 5   # brick, nearer
    instance[0:5, 0:5] = 1       # gripper-like
    depth = np.full((res, res), np.nan, dtype=np.float32)
    depth[10:20, 10:20] = 0.20
    depth[30:34, 30:34] = 0.15
    depth[40:44, 40:44] = 0.10
    depth[0:5, 0:5] = 0.05
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    rgb[10:20, 10:20] = (0, 120, 60)
    rgb[30:34, 30:34] = (200, 20, 20)
    rgb[40:44, 40:44] = (20, 20, 200)
    rgb[0:5, 0:5] = (255, 210, 30)
    return rgb, instance, depth


# ------------------------------------------------------------------------------- synthetic ---


def test_observe_requires_instance_and_depth():
    rgb, instance, depth = _synthetic()
    src = IdealSceneModel()
    with pytest.raises(ValueError):
        src.observe(rgb, window=None, instance=instance)
    with pytest.raises(ValueError):
        src.observe(rgb, window=None, depth=depth)


def test_background_ids_minus_one_and_zero_are_unlabelled():
    res = 32
    instance = np.zeros((res, res), dtype=np.int32)
    instance[:16, :] = -1
    depth = np.full((res, res), 0.3, dtype=np.float32)
    rgb = np.full((res, res, 3), 128, dtype=np.uint8)
    src = IdealSceneModel()
    st = src.observe(rgb, window=None, instance=instance, depth=depth)
    assert src.last_object_count == 0  # nothing but 0/-1 present
    non_empty = [k for k in range(st.K) if not st.is_empty(k)]
    assert len(non_empty) == 1
    assert st.mass(non_empty[0]) == pytest.approx(1.0, abs=1e-6)  # the WHOLE frame is background


def test_slot_depth_ranks_nearest_smallest_and_absent_is_half():
    rgb, instance, depth = _synthetic()
    src = IdealSceneModel()
    st = src.observe(rgb, window=None, instance=instance, depth=depth)
    present = {int(g): k for k, g in enumerate(st.slot_ids) if g != -1}
    # id5 (depth 0.10) nearer than id4 (0.15) nearer than id2 (0.20)
    assert st.depth[present[5]] < st.depth[present[4]] < st.depth[present[2]]
    empty_slots = [k for k in range(st.K) if st.is_empty(k)]
    for k in empty_slots:
        assert st.depth[k] == pytest.approx(0.5)


def test_overflow_keeps_largest_and_reports_the_rest():
    res = 64
    instance = np.full((res, res), -1, dtype=np.int32)
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    depth = np.full((res, res), np.nan, dtype=np.float32)
    sizes = list(range(9, 0, -1))  # 9 objects, decreasing size -> ids 10..18 in size order
    for k, sz in enumerate(sizes):
        r0 = 2 + k * 6
        instance[r0:r0 + sz, 2:2 + sz] = 10 + k
        depth[r0:r0 + sz, 2:2 + sz] = 0.1 + 0.01 * k
    src = IdealSceneModel(K=8)
    st = src.observe(rgb, window=None, instance=instance, depth=depth)
    assert src.last_object_count == 9
    assert src.last_kept_count == 7
    assert src.last_overflow_count == 2
    kept_ids = set(int(g) for g in st.slot_ids if g != -1)
    assert kept_ids == {10 + k for k in range(7)}  # the 7 largest (ids 10..16, sizes 9..3)


def test_k_must_be_at_least_two():
    with pytest.raises(ValueError):
        IdealSceneModel(K=1)


def test_amodal_option_off_by_default():
    src = IdealSceneModel()
    assert src.amodal is False
    assert src.boundary_noise_px == 0
    assert src.merge_same_colour_touching is False
    assert src.min_object_px == 0
    assert src.label_noise_p == 0.0


def test_amodal_fills_a_partially_occluded_l_shape():
    """A ring-shaped (partially occluded) object: the convex hull under `amodal=True` should
    report a larger mass than the raw, occluded pixel count -- the hull "fills in" the notch."""
    res = 40
    instance = np.full((res, res), -1, dtype=np.int32)
    instance[5:35, 5:35] = 4          # a big square...
    instance[15:25, 15:25] = -1       # ...with a notch cut out of the middle (occluded)
    depth = np.full((res, res), np.nan, dtype=np.float32)
    depth[instance == 4] = 0.15
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    rgb[instance == 4] = (200, 20, 20)

    modal = IdealSceneModel(amodal=False).observe(rgb, window=None, instance=instance, depth=depth)
    amodal = IdealSceneModel(amodal=True).observe(rgb, window=None, instance=instance, depth=depth)
    slot_modal = int(np.nonzero(modal.slot_ids == 4)[0][0])
    slot_amodal = int(np.nonzero(amodal.slot_ids == 4)[0][0])
    assert amodal.mass(slot_amodal) > modal.mass(slot_modal)


def test_min_object_px_drops_tiny_objects():
    res = 32
    instance = np.full((res, res), -1, dtype=np.int32)
    instance[0:2, 0:2] = 4  # 4 pixels
    instance[10:20, 10:20] = 5  # 100 pixels
    depth = np.full((res, res), np.nan, dtype=np.float32)
    depth[instance == 4] = 0.1
    depth[instance == 5] = 0.2
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    src = IdealSceneModel(min_object_px=10)
    st = src.observe(rgb, window=None, instance=instance, depth=depth)
    assert src.last_object_count == 1
    assert 4 not in st.slot_ids
    assert 5 in st.slot_ids


# ---------------------------------------------------------------------- rendered scene tests ---


@pytest.fixture(scope="module")
def rendered_scene_frame():
    cfg = SceneConfig(
        held=None, tray=False,
        placed=[
            Placement(2, 2, 1, 1, 0, "red"),
            Placement(2, 2, 1, 1, 1, "blue"),  # stacked directly on the red one
            Placement(2, 4, 4, 2, 0, "orange"),
        ],
        cameras=(WRIST_CAM,),
    )
    scene = build_scene(cfg)
    rng = np.random.default_rng(0)
    goto_look(scene, PLATE_VIEW, rng)
    renderer = SceneRenderer(scene.model, width=PLATE_VIEW.render_size[0], height=PLATE_VIEW.render_size[1])
    try:
        renderer.instance_lut(scene.instance_ids)
        frame, geom = render_view(scene, PLATE_VIEW, renderer)
    finally:
        renderer.close()
    return scene, frame, geom


def test_decode_reproduces_ground_truth_partition_exactly(rendered_scene_frame):
    scene, frame, geom = rendered_scene_frame
    src = IdealSceneModel()
    st = src.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth)
    assert st.has_decode

    step = 4
    coords = np.arange(0, frame.instance.shape[0], step)
    cols, rows = np.meshgrid(coords, coords)
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1)
    logits = st.decode(px)
    decoded_slot = np.argmax(logits, axis=-1)

    true_id = frame.instance[rows.ravel(), cols.ravel()]
    id_to_slot = {int(g): k for k, g in enumerate(st.slot_ids) if g != -1}
    # `src.assigner.background_slot` is the authoritative source -- a scene where everything in
    # frame happens to carry an explicit instance id (this project's own `scene.instance_ids`
    # gives the table itself an id, not the renderer's "0 = unlabelled" default) can leave the
    # background slot with literally zero pixels, i.e. `is_empty`, so it cannot be inferred from
    # "nonempty and unclaimed" alone.
    background_slot = src.assigner.background_slot

    expected_slot = np.array([
        id_to_slot.get(int(i), background_slot) if int(i) not in (-1, 0) else background_slot
        for i in true_id
    ])
    assert np.array_equal(decoded_slot, expected_slot)


def test_decode_query_independence(rendered_scene_frame):
    scene, frame, geom = rendered_scene_frame
    src = IdealSceneModel()
    st = src.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth)
    px_all = np.array([[10, 10], [200, 200], [128, 128], [50, 220]])
    logits_all = st.decode(px_all)
    for i in range(len(px_all)):
        logits_one = st.decode(px_all[i : i + 1])
        np.testing.assert_allclose(logits_all[i], logits_one[0])


def test_identity_persists_across_frames_on_a_rendered_scene(rendered_scene_frame):
    scene, frame, geom = rendered_scene_frame
    src = IdealSceneModel()
    st1 = src.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth, t=0.0)
    st2 = src.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth, t=1.0)
    np.testing.assert_array_equal(st1.slot_ids, st2.slot_ids)
    present = [k for k in range(st1.K) if st1.slot_ids[k] != -1]
    for k in present:
        assert st2.age(k) > st1.age(k)


def test_depth_order_matches_true_order_for_stacked_bricks(rendered_scene_frame):
    """The blue brick (layer 1) sits directly on top of the red one (layer 0): its slot depth
    must rank strictly nearer (smaller) than every other brick's."""
    scene, frame, geom = rendered_scene_frame
    src = IdealSceneModel()
    st = src.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth)

    blue_id = scene.instance_ids[scene.placed_names[1]]
    orange_id = scene.instance_ids[scene.placed_names[2]]
    id_to_slot = {int(g): k for k, g in enumerate(st.slot_ids) if g != -1}
    assert blue_id in id_to_slot and orange_id in id_to_slot
    assert st.depth[id_to_slot[blue_id]] < st.depth[id_to_slot[orange_id]]


def test_read_objects_tokens_are_unaffected_by_which_slot_an_object_lands_in(rendered_scene_frame):
    """Slot order carries no meaning: forcing two `IdealSceneModel`s to hand the same objects
    different slot indices (by observing them in a different order/assigner history) must not
    change what `readout.read_objects` reports -- position, colour, size, confidence -- only the
    diagnostic `.slot` field may differ."""
    from brick_builder.perception import readout as R

    scene, frame, geom = rendered_scene_frame
    src_a = IdealSceneModel()
    st_a = src_a.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth)

    # A second source, warmed up on unrelated frames first so its assigner hands out slots in a
    # different order for the very same objects.
    src_b = IdealSceneModel()
    warm_res = frame.instance.shape[0]
    warm_instance = np.full((warm_res, warm_res), -1, dtype=np.int32)
    warm_instance[0:5, 0:5] = 999
    warm_depth = np.full((warm_res, warm_res), np.nan, dtype=np.float32)
    warm_depth[0:5, 0:5] = 0.3
    warm_rgb = np.zeros_like(frame.rgb)
    src_b.observe(warm_rgb, window=PLATE_VIEW, instance=warm_instance, depth=warm_depth)
    st_b = src_b.observe(frame.rgb, window=PLATE_VIEW, instance=frame.instance, depth=frame.depth)

    assert not np.array_equal(st_a.slot_ids, st_b.slot_ids), "test setup should have produced different slot orders"

    tokens_a = R.read_objects(st_a, frame.rgb, geom)
    tokens_b = R.read_objects(st_b, frame.rgb, geom)

    def by_colour_and_pos(tokens):
        return sorted((t.colour, round(t.x, 6), round(t.y, 6), round(t.yaw, 6), t.nx, t.ny, t.layer) for t in tokens)

    assert by_colour_and_pos(tokens_a) == by_colour_and_pos(tokens_b)
