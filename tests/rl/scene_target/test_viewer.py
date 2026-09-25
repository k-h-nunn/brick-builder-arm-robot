"""WP8's own pieces, on synthetic data: no renderer, no MuJoCo, no CNN weights.

The one thing this file cannot check without a real render -- whether a drawn text box actually
falls on the arm's own pixels in a live take -- is exercised on a synthetic instance map instead
(`test_count_arm_pixels_under_boxes_*`); `scripts/make_scene_target_video.py`'s own QA pass runs
the same function on the real thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.demo.video_panels import LayoutError, Overlay, Rect
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState, opaque_schema
from brick_builder.rl.scene_target import viewer as V

RES = 32


def _state(rows, *, decode_fn=None, schema=PLACEHOLDER_SLOT_SCHEMA, K=None):
    K = rows.shape[0] if K is None else K
    return SceneState(slots=rows, depth=np.full(K, 0.5, dtype=np.float32), schema=schema,
                      t=0.0, source="test", window=None,
                      slot_ids=np.arange(K, dtype=np.int64), _decode_fn=decode_fn)


def _named_rows(colours) -> np.ndarray:
    """`[K, 14]` rows with only the `colour` field set, everything else zero."""
    rows = np.zeros((len(colours), PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    for k, c in enumerate(colours):
        rows[k, PLACEHOLDER_SLOT_SCHEMA.fields["colour"]] = c
    return rows


# --------------------------------------------------------------------------------- panels ---


def test_panel_titles_are_the_four_exact_strings():
    assert V.PANEL_TITLES == ("Arm camera POV", "Observed scene state", "Target scene state",
                              "Predicted next scene state")


def test_panels_predicted_is_copy_last_of_the_previous_observed():
    panels = V.Panels(size=(16, 16))
    wrist = np.full((16, 16, 3), 40, dtype=np.uint8)
    first_obs = np.full((16, 16, 3), 10, dtype=np.uint8)
    target = np.full((16, 16, 3), 200, dtype=np.uint8)
    panels.update(wrist_rgb=wrist, observed_rgb=first_obs, target_rgb=target)
    entries = dict(panels.entries())
    # No earlier observation yet: predicted falls back to this tick's own observed picture.
    np.testing.assert_array_equal(entries["Predicted next scene state"], first_obs)

    second_obs = np.full((16, 16, 3), 90, dtype=np.uint8)
    panels.update(wrist_rgb=wrist, observed_rgb=second_obs, target_rgb=target)
    entries = dict(panels.entries())
    # Now predicted is the *previous* tick's observed picture, not this tick's.
    np.testing.assert_array_equal(entries["Predicted next scene state"], first_obs)
    np.testing.assert_array_equal(entries["Observed scene state"], second_obs)


def test_panels_reset_clears_the_copy_last_history():
    panels = V.Panels(size=(8, 8))
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    obs = np.full((8, 8, 3), 55, dtype=np.uint8)
    panels.update(wrist_rgb=blank, observed_rgb=obs, target_rgb=blank)
    panels.reset()
    obs2 = np.full((8, 8, 3), 77, dtype=np.uint8)
    panels.update(wrist_rgb=blank, observed_rgb=obs2, target_rgb=blank)
    entries = dict(panels.entries())
    np.testing.assert_array_equal(entries["Predicted next scene state"], obs2)


def test_panels_resizes_every_picture_to_the_panel_size():
    panels = V.Panels(size=(20, 12))
    wrist = np.zeros((256, 256, 3), dtype=np.uint8)
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    target = np.zeros((64, 64, 3), dtype=np.uint8)
    panels.update(wrist_rgb=wrist, observed_rgb=obs, target_rgb=target)
    for _, img in panels.entries():
        assert img.shape == (12, 20, 3)
        assert img.dtype == np.uint8


# --------------------------------------------------------------------------- dense decode ---


def test_dense_decode_rgb_colours_each_slot_by_its_own_published_colour():
    # Two slots, red and blue (2*rgb - 1 convention: 1.0 -> pure channel, -1.0 -> none).
    rows = _named_rows([(1.0, -1.0, -1.0), (-1.0, -1.0, 1.0)])

    def decode_fn(px):
        px = np.asarray(px)
        logits = np.full((px.shape[0], 2), -10.0, dtype=np.float32)
        left = px[:, 0] < RES // 2
        logits[left, 0] = 10.0
        logits[~left, 1] = 10.0
        return logits

    state = _state(rows, decode_fn=decode_fn, K=2)
    img = V.dense_decode_rgb(state, res=RES)
    assert img.shape == (RES, RES, 3)
    assert img.dtype == np.uint8
    left_px = img[RES // 2, 2]
    right_px = img[RES // 2, RES - 2]
    assert left_px[0] > left_px[2]     # left half reads red-ish
    assert right_px[2] > right_px[0]   # right half reads blue-ish
    # A pixel of unclaimed background never appears here (every column falls on one side).
    assert not np.array_equal(left_px, np.asarray(V.DARK, dtype=np.uint8))


def test_dense_decode_rgb_falls_back_to_slot_rows_without_a_decode():
    rows = _named_rows([(1.0, -1.0, -1.0)])
    state = _state(rows, decode_fn=None, K=1)
    assert not state.has_decode
    img = V.dense_decode_rgb(state, res=RES)
    assert img.shape == (RES, RES, 3)
    # The fallback still shows the slot's own colour somewhere, not a blank background frame.
    assert np.any(img != np.asarray(V.DARK, dtype=np.uint8))


def test_colourize_slots_draws_a_dark_edge_between_neighbouring_slots():
    slot_map = np.zeros((8, 8), dtype=np.int64)
    slot_map[:, 4:] = 1
    rows = _named_rows([(1.0, -1.0, -1.0), (-1.0, 1.0, -1.0)])
    img = V.colourize_slots(slot_map, rows, PLACEHOLDER_SLOT_SCHEMA, K=2)
    np.testing.assert_array_equal(img[0, 3], (10, 11, 13))


def test_colourize_slots_uses_the_fallback_palette_on_an_opaque_schema():
    schema = opaque_schema(4)
    slot_map = np.zeros((6, 6), dtype=np.int64)
    rows = np.zeros((1, 4), dtype=np.float32)
    img = V.colourize_slots(slot_map, rows, schema, K=1)
    expected = V._lift(np.asarray(V.FALLBACK_PALETTE[0], dtype=np.float64) / 255.0)
    np.testing.assert_array_equal(img[0, 0], expected)


# --------------------------------------------------------------------------------- QA ---


def test_check_stills_layout_passes_on_non_overlapping_rects_and_fails_on_overlap():
    safe = Rect(0, 0, 100, 100)
    ok = {"a": Rect(0, 0, 10, 10), "b": Rect(20, 20, 30, 30)}
    V.check_stills_layout(ok, safe)  # does not raise
    bad = {"a": Rect(0, 0, 10, 10), "b": Rect(5, 5, 15, 15)}
    with pytest.raises(LayoutError):
        V.check_stills_layout(bad, safe)


def test_strip_title_overlay_puts_the_title_in_the_strip_box_not_the_corner():
    overlay = V.StripTitleOverlay(1920, 1080)
    picture = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, rects = overlay.draw(picture, title=V.PROJECT_TITLE, subtitle="Level: Viewing pose")
    strip = overlay.strip_rect()
    assert rects["title"] == strip
    assert rects["title"].y0 > 100  # nowhere near the top-left corner a plain Overlay would use


def test_strip_title_overlay_draws_the_phase_label_normally_once_the_title_is_gone():
    overlay = V.StripTitleOverlay(1920, 1080)
    picture = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, rects = overlay.draw(picture, phase="Viewing pose")
    assert rects["phase"].y0 < 100  # the ordinary top-left phase label, unaffected


def test_overlay_draws_the_four_panels_without_raising():
    quality_w, quality_h = 1920, 1080
    overlay = Overlay(quality_w, quality_h)
    side, _ = overlay.panel_size(4)
    picture = np.zeros((quality_h, quality_w, 3), dtype=np.uint8)
    panels = [(title, np.full((side, side, 3), 30 * i, dtype=np.uint8))
              for i, title in enumerate(V.PANEL_TITLES)]
    from brick_builder.demo.video_panels import Readout

    readouts = [Readout(label=r.label, value=1.2, unit=r.unit, lo=r.lo, hi=5.0, fmt=r.fmt,
                        plot=r.plot) for r in V.DEFAULT_READOUTS]
    composed, rects = overlay.draw(picture, title=V.PROJECT_TITLE,
                                   subtitle=V.spec_line(controller="open-loop replay (u = 0)",
                                                        predictor="copy-last (placeholder)",
                                                        scene_model="ground truth",
                                                        level="Viewing pose", offset_mm=0.0,
                                                        offset_deg=0.0),
                                   readouts=readouts, panels=panels)
    assert composed.shape == (quality_h, quality_w, 3)
    assert {k for k in rects if k.startswith("panel_")} == {f"panel_{i}" for i in range(4)}
    V.check_stills_layout(rects, overlay.safe)


def test_count_arm_pixels_under_boxes_counts_only_inside_each_rect():
    instance_map = np.zeros((100, 100), dtype=np.int32)
    instance_map[40:60, 40:60] = V.ARM_INSTANCE
    rects = {"over_arm": Rect(30, 30, 50, 50), "clear": Rect(0, 0, 10, 10)}
    counts = V.count_arm_pixels_under_boxes(instance_map, V.ARM_INSTANCE, rects)
    assert counts["clear"] == 0
    assert counts["over_arm"] == 10 * 10  # the 40-50, 40-50 quadrant of the arm square


def test_main_view_gradient_is_near_zero_on_a_flat_render_and_large_on_noise():
    clear = Rect(0, 0, 64, 64)
    flat = np.full((64, 64, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8)
    assert V.main_view_gradient(flat, clear) < 0.5
    assert V.main_view_gradient(noisy, clear) > V.MAX_MAIN_VIEW_GRADIENT


@pytest.mark.parametrize("text,bad", [
    ("Scene-state distance [mm-eq]", False),
    ("Predicted next scene state", False),
    ("Baseplate offset: 6.2 mm, 2.4 deg", False),
    ("weights_sha256 3e07a382", True),
    ("vision_model.pt", True),
    ("camelCaseName", True),
    ("3e07a3821234", True),
])
def test_screen_identifiers_flags_code_looking_strings_only(text, bad):
    hits = V.screen_identifiers([text])
    assert bool(hits) == bad, (text, hits)


# --------------------------------------------------------------------- target decode cache ---


def test_target_decode_cache_path_follows_the_targets_cache_convention(tmp_path):
    path = V.target_decode_cache_path(tmp_path, "viewing_pose", 7, "ideal", "abcdef0123456789")
    assert path == tmp_path / "decode" / "viewing_pose" / "seed00007_ideal_abcdef01.npz"
    none_hash = V.target_decode_cache_path(tmp_path, "viewing_pose", 7, "ideal", "")
    assert none_hash.name == "seed00007_ideal_none.npz"


def test_load_or_render_target_decode_caches_after_the_first_call(tmp_path, monkeypatch):
    class FakeSeq:
        T = 3
        meta = {"family": "viewing_pose", "seed": 5, "weights_sha256": "", "source": "ideal"}

    calls = {"n": 0}

    def fake_render(seq, *, source_name, source_kwargs=None, res=256):
        calls["n"] += 1
        return np.full((seq.T, 4, 4, 3), calls["n"], dtype=np.uint8)

    monkeypatch.setattr(V, "render_target_decode", fake_render)
    seq = FakeSeq()
    first = V.load_or_render_target_decode(seq, source_name="ideal", cache_dir=tmp_path)
    assert calls["n"] == 1
    np.testing.assert_array_equal(first, np.full((3, 4, 4, 3), 1, dtype=np.uint8))

    # A second call must read the cache, not render again -- even though a re-render would now
    # return a different (and wrong) fill value.
    second = V.load_or_render_target_decode(seq, source_name="ideal", cache_dir=tmp_path)
    assert calls["n"] == 1
    np.testing.assert_array_equal(second, first)


def test_spec_line_is_titles_and_values_with_no_sentence_punctuation():
    line = V.spec_line(controller="open-loop replay (u = 0)", predictor="copy-last (placeholder)",
                       scene_model="ground truth", level="Viewing pose", offset_mm=0.0,
                       offset_deg=0.0)
    assert line == ("Controller: open-loop replay (u = 0) | Predictor: copy-last (placeholder) | "
                    "Scene model: ground truth | Level: Viewing pose | "
                    "Offset: 0.0 mm, 0.0 deg")
    assert not line.endswith(".")
