"""Tests for the `TokenProducer`s.

`GroundTruthProducer` and `BlobProducer` both run unconditionally: neither needs a checkpoint,
a GPU or anything outside this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from brick_builder.perception.producers import (
    DEFAULT_BASEPLATE_RGB,
    PALETTE,
    BlobProducer,
    GroundTruthObject,
    GroundTruthProducer,
    hsv_colour_distance,
)
from brick_builder.perception.tokens import ObjectToken

REPO_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_scene():
    """256x256 grey background with four flat, non-overlapping coloured rectangles."""
    img = np.full((256, 256, 3), 0.5, dtype=np.float32)
    boxes = {
        "red": (20, 20, 60, 60),
        "blue": (150, 20, 190, 60),
        "yellow": (20, 150, 60, 190),
        "green": (150, 150, 190, 190),
    }
    for colour, (r0, c0, r1, c1) in boxes.items():
        img[r0:r1, c0:c1] = PALETTE[colour]
    return img, boxes


# --------------------------------------------------------------------------------------------
# GroundTruthProducer
# --------------------------------------------------------------------------------------------


def test_ground_truth_producer_passes_through_noise_free() -> None:
    producer = GroundTruthProducer(
        bias_px=(0.0, 0.0), jitter_std_px=0.0, yaw_jitter_rad=0.0, dropout_p=0.0, merge_radius_px=1.0
    )
    gt = [GroundTruthObject(x=10.0, y=20.0, yaw=0.1, layer=0, colour="red", size=5.0)]
    tokens = producer.produce(np.random.default_rng(0), ground_truth=gt)
    assert len(tokens) == 1
    tok = tokens[0]
    assert (tok.x, tok.y, tok.yaw, tok.layer, tok.colour, tok.size) == (10.0, 20.0, 0.1, 0, "red", 5.0)
    assert tok.confidence == 1.0
    assert tok.age == 0.0


def test_ground_truth_producer_dropout_all() -> None:
    producer = GroundTruthProducer(dropout_p=1.0)
    gt = [GroundTruthObject(x=0.0, y=0.0, yaw=0.0, layer=0, colour="red", size=1.0)]
    tokens = producer.produce(np.random.default_rng(0), ground_truth=gt)
    assert tokens == []


def test_ground_truth_producer_merges_close_same_colour() -> None:
    producer = GroundTruthProducer(
        jitter_std_px=0.0, yaw_jitter_rad=0.0, dropout_p=0.0, merge_radius_px=10.0
    )
    gt = [
        GroundTruthObject(x=0.0, y=0.0, yaw=0.0, layer=0, colour="red", size=5.0),
        GroundTruthObject(x=3.0, y=4.0, yaw=0.0, layer=0, colour="red", size=5.0),  # distance 5 < 10
        GroundTruthObject(x=100.0, y=100.0, yaw=0.0, layer=0, colour="blue", size=5.0),
    ]
    tokens = producer.produce(np.random.default_rng(0), ground_truth=gt)
    assert len(tokens) == 2
    reds = [t for t in tokens if t.colour == "red"]
    assert len(reds) == 1
    assert reds[0].x == pytest.approx(1.5)
    assert reds[0].y == pytest.approx(2.0)
    blues = [t for t in tokens if t.colour == "blue"]
    assert len(blues) == 1


def test_ground_truth_producer_requires_ground_truth() -> None:
    with pytest.raises(ValueError):
        GroundTruthProducer().produce(np.random.default_rng(0))


# --------------------------------------------------------------------------------------------
# BlobProducer
# --------------------------------------------------------------------------------------------


def test_blob_producer_finds_four_rectangles_within_one_pixel() -> None:
    img, boxes = _synthetic_scene()
    producer = BlobProducer()
    tokens = producer.produce(np.random.default_rng(0), frame=img)

    assert len(tokens) == 4
    by_colour = {t.colour: t for t in tokens}
    assert set(by_colour) == set(boxes)
    for colour, (r0, c0, r1, c1) in boxes.items():
        expected_x = (c0 + c1 - 1) / 2.0
        expected_y = (r0 + r1 - 1) / 2.0
        tok = by_colour[colour]
        assert abs(tok.x - expected_x) < 1.0, f"{colour} centroid x error {tok.x - expected_x}"
        assert abs(tok.y - expected_y) < 1.0, f"{colour} centroid y error {tok.y - expected_y}"
        assert tok.confidence == 1.0
        assert tok.layer == 0


def test_blob_producer_ignores_background() -> None:
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)  # no palette colour present
    producer = BlobProducer()
    tokens = producer.produce(np.random.default_rng(0), frame=img)
    assert tokens == []


def test_blob_producer_requires_frame() -> None:
    with pytest.raises(ValueError):
        BlobProducer().produce(np.random.default_rng(0))


# --------------------------------------------------------------------------------------------
# perception-align-002: BlobProducer's colour match vs. the default green baseplate
# --------------------------------------------------------------------------------------------


def test_default_baseplate_rgb_matches_bricks_spec_green() -> None:
    """`DEFAULT_BASEPLATE_RGB` is a literal copy (this module deliberately does not import
    `brick_builder.bricks`), not a fresh guess -- this is the drift guard: if `brick_builder.bricks.spec`'s
    green or `add_baseplate`'s default colour ever changes without this copy being updated, this
    must fail loudly instead of quietly reintroducing a colour-matching bug like this one."""
    from brick_builder.bricks.spec import PALETTE as BRICKS_PALETTE

    assert DEFAULT_BASEPLATE_RGB == BRICKS_PALETTE["green"]


def test_old_rgb_distance_would_have_missed_the_green_baseplate_collision() -> None:
    """Documents the actual bug being fixed: plain RGB Euclidean distance put producers.py's
    "green" only ~0.186 from the real default baseplate colour, comfortably inside the old
    tol=0.25 default -- so a green brick on the green baseplate merged with it."""
    rgb_dist = float(np.linalg.norm(np.asarray(PALETTE["green"]) - np.asarray(DEFAULT_BASEPLATE_RGB)))
    assert rgb_dist == pytest.approx(0.1857, abs=1e-3)
    assert rgb_dist <= 0.25  # inside the OLD default tol: this is why it merged


def test_default_blob_producer_separates_every_colour_from_the_baseplate() -> None:
    """The fixed metric/tolerance must keep every palette colour, not just green, comfortably
    outside `tol` of the baseplate colour."""
    producer = BlobProducer()
    for name, rgb in PALETTE.items():
        d = hsv_colour_distance(rgb, DEFAULT_BASEPLATE_RGB, producer.chroma_gain)
        assert d > producer.tol, f"{name} ({d:.3f}) is not separated from the baseplate at tol={producer.tol}"


def test_blob_producer_refuses_a_palette_that_cannot_be_separated_from_the_baseplate() -> None:
    """Constructing a producer whose baseplate colour collides with one of its own palette
    colours must refuse loudly (a clear error) rather than silently building a producer that
    would merge a brick into the plate at runtime -- here the baseplate is deliberately set to
    exactly the palette's own "green"."""
    with pytest.raises(ValueError, match="green"):
        BlobProducer(baseplate_rgb=PALETTE["green"])


def test_blob_producer_baseplate_check_can_be_disabled() -> None:
    """`baseplate_rgb=None` is the documented escape hatch (e.g. a scene with no baseplate at
    all) -- it must never raise, even for the exact colliding configuration that raises in
    `test_blob_producer_refuses_a_palette_that_cannot_be_separated_from_the_baseplate` above."""
    BlobProducer(baseplate_rgb=None)
    BlobProducer(palette={"green": PALETTE["green"]}, baseplate_rgb=None)  # colliding, but opted out


def _scene_on_baseplate(colour: str, baseplate_rgb=DEFAULT_BASEPLATE_RGB, size: int = 400):
    """A `size`x`size` image: the whole frame filled with `baseplate_rgb`, one `colour`-filled
    square brick-sized patch in the middle third, well clear of the border."""
    img = np.full((size, size, 3), baseplate_rgb, dtype=np.float32)
    lo, hi = size // 3, 2 * size // 3
    img[lo:hi, lo:hi] = PALETTE[colour]
    return img, (lo, lo, hi, hi)


@pytest.mark.parametrize("colour", list(PALETTE))
def test_blob_producer_finds_each_colour_on_the_default_baseplate(colour: str) -> None:
    """The direct reproduction of the finding, generalized to all six colours: a `colour` brick
    on the default green baseplate must be found as its own, correctly-coloured, brick-sized
    component -- not merged into a frame-spanning baseplate blob (the old bug's exact symptom:
    "one 1145px blob spanning nearly half the frame")."""
    img, (r0, c0, r1, c1) = _scene_on_baseplate(colour)
    producer = BlobProducer()
    tokens = producer.produce(np.random.default_rng(0), frame=img)

    matching = [t for t in tokens if t.colour == colour]
    assert len(matching) == 1, f"expected exactly one {colour} token, got {[t.colour for t in tokens]}"
    tok = matching[0]
    brick_area = (r1 - r0) * (c1 - c0)
    assert tok.size < (r1 - r0) * 1.5, f"{colour} blob size {tok.size} looks baseplate-sized, not brick-sized"
    expected_x, expected_y = (c0 + c1 - 1) / 2.0, (r0 + r1 - 1) / 2.0
    assert abs(tok.x - expected_x) < 2.0
    assert abs(tok.y - expected_y) < 2.0
    # And no OTHER colour (in particular not the baseplate's own near-miss, if any) produced a
    # huge, frame-spanning component.
    for other in tokens:
        assert other.size < brick_area, f"{other.colour} blob ({other.size}px) looks like the baseplate, not a brick"


@pytest.mark.parametrize("colour", list(PALETTE))
def test_blob_producer_finds_each_colour_on_the_default_baseplate_after_postprocess(colour: str) -> None:
    """Same reproduction, through a real, moderately strong `apply_postprocess` pass (exposure,
    gamma, white balance, noise, blur -- no lens distortion, which is orthogonal geometry, not
    colour, and is covered by tests/render/test_domain_rand.py): the colour match must survive
    the post-process, not just the pristine render."""
    from brick_builder.render.domain_rand import PostProcessConfig, apply_postprocess
    from brick_builder.render.renderer import Frame

    img, (r0, c0, r1, c1) = _scene_on_baseplate(colour)
    rgb_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    frame = Frame(
        rgb=rgb_u8,
        depth=np.zeros(img.shape[:2], dtype=np.float32),
        instance=np.zeros(img.shape[:2], dtype=np.int32),
        body_ids={},
    )
    cfg = PostProcessConfig(
        exposure=1.15, gamma=0.9, white_balance=(1.08, 0.94, 1.03),
        shot_noise=0.01, read_noise=0.005, blur_sigma=0.4,
    )
    out = apply_postprocess(frame, cfg, np.random.default_rng(0))

    producer = BlobProducer()
    tokens = producer.produce(np.random.default_rng(0), frame=out.rgb)

    matching = [t for t in tokens if t.colour == colour]
    assert len(matching) == 1, f"expected exactly one {colour} token post-process, got {[t.colour for t in tokens]}"
    brick_area = (r1 - r0) * (c1 - c0)
    for other in tokens:
        assert other.size < brick_area, (
            f"{other.colour} blob ({other.size}px) looks like the baseplate, not a brick, after post-process"
        )
