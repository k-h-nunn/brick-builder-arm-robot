"""Flush-neighbour geometry: align hover and aim offset.

Two bricks on adjacent cells have 0.2 mm of air between them, which is under the loop's own
lateral floor, and a neighbour's studs stand 3.7 mm above the height the align dwell hovers at.
Both are geometry, so none of this steps any physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.action.place import Geometry, PlaceConfig
from brick_builder.action.worlds import REF_RED, REF_YELLOW, build_nominal_world
from brick_builder.bricks.structures import Placement
from brick_builder.scene.compose import HeldBrick

FIRST = Placement(h=2, w=2, x=4, y=3, z=0, colour="blue")
"""First brick, layer 0: the neighbour of cells (2, 3), (6, 3) and (4, 1)."""

HELD = HeldBrick(nx=2, ny=2, colour="blue")


@pytest.fixture(scope="module")
def world():
    scene = build_nominal_world(False, held=HELD, placed=[REF_YELLOW, REF_RED, FIRST])
    yield scene


def _geometry(scene, target, **kw):
    return Geometry(scene, target, HELD, PlaceConfig(pick=True, **kw))


def test_flush_neighbours_have_two_tenths_of_a_millimetre_between_them(world) -> None:
    """Gap between flush neighbours [mm]: one clearance.

    The footprint is `n * pitch` minus one clearance.
    """
    spec = world.cfg.spec
    footprint = 2 * spec.pitch - spec.clearance
    gap = 2 * spec.pitch - footprint
    assert gap == pytest.approx(spec.clearance)
    assert 1e3 * gap == pytest.approx(0.2, abs=0.01)
    assert gap < spec.chamfer, "the studs' chamfer is wider than the gap, which is the point"


@pytest.mark.parametrize("target, away", [
    ((2, 3), (-1, 0)),   # the first brick is at +x
    ((6, 3), (+1, 0)),   # and here at -x
    ((4, 1), (0, -1)),   # and here at +y
])
def test_the_aim_leans_away_from_what_is_standing_beside_the_cell(world, target, away) -> None:
    g = _geometry(world, target)
    assert g.has_neighbour, f"no neighbour found beside {target}"
    # (2, 3) also has the yellow 2x4 flush above it, so only the sign is asserted here.
    for got, want in zip(g.away, away):
        if want:
            assert np.sign(got) == np.sign(want), f"leans {g.away} for a neighbour at {away}"
    assert 1e3 * float(np.linalg.norm(g.bias)) == pytest.approx(0.3, abs=1e-6)


def test_a_cell_with_nothing_beside_it_is_untouched(world) -> None:
    """Cell without a neighbour: no aim offset, default hover."""
    g = _geometry(world, (4, 3, 1))
    assert not g.has_neighbour
    assert np.allclose(g.bias, 0.0)
    assert g.hover == PlaceConfig().hover
    plain = _geometry(world, (4, 3, 1), neighbour_bias=0.0, neighbour_clear=0.0)
    assert np.allclose(g.place_tcp(g.hover), plain.place_tcp(plain.hover))


def test_a_brick_a_layer_down_is_not_a_neighbour(world) -> None:
    """Brick one layer down: the seat, not a neighbour."""
    assert not _geometry(world, (4, 3, 1)).has_neighbour
    assert _geometry(world, (2, 3, 0)).has_neighbour


def test_the_align_dwell_stands_above_a_neighbour_s_studs(world) -> None:
    """Align hover above a neighbour's stud tops.

    The default hover, 20 mm over the seat, is 3.7 mm under them.
    """
    spec = world.cfg.spec
    beside = _geometry(world, (2, 3))
    alone = _geometry(world, (4, 3, 1))
    studs = spec.height + spec.stud_h
    assert alone.hover < studs, "the plain hover is below a neighbour's studs, which is the fault"
    assert beside.hover > studs, (
        f"the align dwell hovers {1e3 * beside.hover:.1f} mm over the seat, "
        f"under the {1e3 * studs:.1f} mm the neighbour's studs reach")
    assert beside.carry_z >= beside.place_tcp(beside.hover)[2] - 1e-9


def test_the_push_on_the_roof_stays_over_the_cell(world) -> None:
    """Aim offset on the descent only; the roof push at `cell_nominal`."""
    for target, want in (((2, 3), 0.3), ((4, 3, 1), 0.0)):
        leaning = _geometry(world, target)
        square = _geometry(world, target, neighbour_bias=0.0)
        moved = 1e3 * float(np.linalg.norm(
            np.asarray(leaning.place_tcp(0.0)[:2]) - np.asarray(square.place_tcp(0.0)[:2])))
        assert moved == pytest.approx(want, abs=1e-6), (
            f"the descent aims {moved:.2f} mm off the cell, expected {want:.2f} mm")
        # `Script` pushes on the roof at `cell_nominal`, which the lean never moves.
        assert np.allclose(leaning.cell_nominal, square.cell_nominal)
