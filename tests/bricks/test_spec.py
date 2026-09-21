"""Tests for brick_builder.bricks.spec."""

import pytest

from brick_builder.bricks.spec import DUPLO, LEGO_SYSTEM, PALETTE, BrickSpec

# Millimetre table from docs/guide/02-simulator-and-bricks.md, converted to the metres BrickSpec stores.
_DUPLO_TABLE_MM = {
    "pitch": 16.0,
    "height": 19.2,
    "stud_d": 9.40,
    "stud_h": 4.5,
    "stud_inner_d": 6.5,
    "wall": 1.55,
    "tube_od": 13.4,
    "tube_id": 10.7,
}


@pytest.mark.parametrize("field, mm", sorted(_DUPLO_TABLE_MM.items()))
def test_duplo_matches_dimension_table(field, mm):
    got_mm = getattr(DUPLO, field) * 1000.0
    assert got_mm == pytest.approx(mm, abs=1e-9)


def test_duplo_density_and_defaults():
    # PLAN section 6.3 defaults: clearance 0.20 mm, chamfer 0.5 mm.
    assert DUPLO.clearance * 1000.0 == pytest.approx(0.20)
    assert DUPLO.chamfer * 1000.0 == pytest.approx(0.5)
    assert DUPLO.density == pytest.approx(1050.0)


def test_diagonal_stud_gap_vs_tube_interference():
    """The table's headline physical fact: the tube is slightly larger than the gap it must
    squeeze into, by 0.1-0.2 mm - the entire basis of the real click (see design report
    section 2). This checks the *raw*, un-shrunk numbers (the generator separately shrinks
    stud radius by `clearance` so the collision geometry can actually be assembled)."""

    import math

    diagonal_gap = DUPLO.pitch * math.sqrt(2) - DUPLO.stud_d
    interference = DUPLO.tube_od - diagonal_gap
    assert 0.05e-3 < interference < 0.3e-3


def test_brickspec_is_frozen():
    with pytest.raises(Exception):
        DUPLO.pitch = 0.02  # type: ignore[misc]


def test_lego_system_is_smaller_and_valid():
    assert isinstance(LEGO_SYSTEM, BrickSpec)
    assert LEGO_SYSTEM.pitch < DUPLO.pitch
    assert LEGO_SYSTEM.height < DUPLO.height
    assert LEGO_SYSTEM.stud_d < DUPLO.stud_d
    # every length field must be positive (density and stud_inner_d may be 0 - a solid stud)
    for field in ("pitch", "height", "stud_d", "stud_h", "wall", "tube_od", "tube_id", "roof", "density"):
        assert getattr(LEGO_SYSTEM, field) > 0.0


def test_palette_has_the_required_colours():
    for name in ("red", "blue", "yellow", "green", "orange", "white"):
        assert name in PALETTE
        rgb = PALETTE[name]
        assert len(rgb) == 3
        assert all(0.0 <= c <= 1.0 for c in rgb)
