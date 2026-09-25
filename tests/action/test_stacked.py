"""A brick that goes on top of another one, rather than on the baseplate.

The heights are a geometry question and are checked without stepping any physics: the script
seats on the brick underneath, so every height it derives has to move up by one brick height per
layer while everything measured from the plate itself stays where it was. The scoring is checked
the same way, by putting a brick where a layer says and asking what the verifier makes of it.

Only the last test runs a build, and it is slow-marked.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.action.place import Geometry, PlaceConfig
from brick_builder.action.worlds import REF_RED, REF_YELLOW, build_nominal_world
from brick_builder.bricks.structures import Placement
from brick_builder.control.pick_place import tcp_for_foot, verify_placement
from brick_builder.scene.compose import HeldBrick

FIRST = Placement(h=2, w=2, x=4, y=3, z=0, colour="blue")
"""The brick the second one goes on top of."""


@pytest.fixture(scope="module")
def nominal():
    """The nominal world of a build's second cycle: the first brick already on the plate."""
    held = HeldBrick(nx=2, ny=2, colour="blue")
    scene = build_nominal_world(False, held=held, placed=[REF_YELLOW, REF_RED, FIRST])
    yield scene, held


def _heights(scene, held, target) -> dict:
    """Every height the script derives from one target, in metres."""
    cfg = PlaceConfig(pick=True)
    g = Geometry(scene, target, held, cfg)
    spec = scene.cfg.spec
    foot_seated = g.seat_bottom + spec.height + spec.stud_h
    return {
        "seat": g.seat_bottom,
        "hover": float(g.place_tcp(cfg.hover)[2]),
        "low": float(g.place_tcp(cfg.low)[2]),
        "press": float(g.place_tcp(0.0)[2]),
        "clear": g.seat_bottom + 2.0 * spec.height + 0.012,
        "foot": float(tcp_for_foot(g.cell_nominal[:2], foot_seated, 0.0)[2]),
        "carry": g.carry_z,
        "view": float(g.view_tcp[2]),
        "cell_xy": np.asarray(g.cell_nominal[:2]),
    }


def test_a_target_with_no_layer_is_the_baseplate(nominal) -> None:
    """The two-element target every single-brick task uses must not move a millimetre."""
    scene, held = nominal
    plain = _heights(scene, held, (4, 3))
    zero = _heights(scene, held, (4, 3, 0))
    for name, value in plain.items():
        assert np.allclose(value, zero[name]), f"{name} moved between (4, 3) and (4, 3, 0)"
    assert plain["seat"] == pytest.approx(scene.plate_top)


def test_the_descent_moves_up_one_brick_height_per_layer(nominal) -> None:
    """Hover, low dwell, press, release clearance and the push on the roof, all of them."""
    scene, held = nominal
    spec = scene.cfg.spec
    ground = _heights(scene, held, (4, 3, 0))
    upper = _heights(scene, held, (4, 3, 1))
    for name in ("seat", "hover", "low", "press", "clear", "foot"):
        assert upper[name] - ground[name] == pytest.approx(spec.height), (
            f"{name} moved {1e3 * (upper[name] - ground[name]):.2f} mm, "
            f"not the {1e3 * spec.height:.2f} mm a layer is"
        )
    assert np.allclose(upper["cell_xy"], ground["cell_xy"]), "the cell moved sideways"


def test_the_press_aims_at_the_brick_underneath_not_the_plate(nominal) -> None:
    """The seat height is the roof of whatever stands on the target cell."""
    scene, held = nominal
    spec = scene.cfg.spec
    roof = scene.plate_top + (FIRST.z + 1) * spec.height
    assert _heights(scene, held, (4, 3, 1))["seat"] == pytest.approx(roof)


def test_the_carry_clears_what_is_already_standing_on_the_plate(nominal) -> None:
    """The brick is carried above the tallest stack, whichever layer it is going to."""
    scene, held = nominal
    spec = scene.cfg.spec
    cfg = PlaceConfig(pick=True)
    studs = scene.plate_top + (FIRST.z + 1) * spec.height + spec.stud_h
    for layer in (0, 1, 2):
        g = Geometry(scene, (4, 3, layer), held, cfg)
        bottom = g.carry_z - float(g.grasp.pos_in_tcp[2])
        assert bottom > studs, f"layer {layer} carries the brick through the stack"
        assert g.carry_z >= g.place_tcp(cfg.hover)[2] - 1e-9, (
            f"layer {layer} carries below the hover it descends from")


def test_the_verifier_scores_against_the_layer_it_was_sent_to(nominal) -> None:
    """A brick on the plate is not a brick on the first one, whatever its studs say."""
    scene, _ = nominal
    spec = scene.cfg.spec
    name = scene.placed_names[-1]
    on_plate = verify_placement(scene, name, (4.5, 3.5, 0))
    assert on_plate.at_target_layer and on_plate.at_target_cell
    assert on_plate.height_error_mm == pytest.approx(0.0, abs=0.2)

    a_layer_up = verify_placement(scene, name, (4.5, 3.5, 1))
    assert not a_layer_up.at_target_layer, "a brick on the plate scored as one layer up"
    assert a_layer_up.height_error_mm == pytest.approx(-1e3 * spec.height, abs=0.5)
    assert not a_layer_up.success
    assert a_layer_up.at_target_cell, "the cell is the same one either way"


def test_the_verifier_names_the_body_underneath(nominal) -> None:
    scene, _ = nominal
    check = verify_placement(scene, scene.placed_names[-1], (4.5, 3.5, 0))
    assert check.lower_bodies == (scene.plate.name,), check.lower_bodies


@pytest.mark.slow
def test_a_nominal_two_brick_tower_lands_both_bricks_on_their_cells() -> None:
    """One brick on the plate, one on top of it, in a world with nothing wrong with it.

    The world is exactly the one the sequences were recorded in, so anything the loop has to
    correct for here is its own.
    """
    from brick_builder.action.place import TrackingEpisode, record_ideal
    from brick_builder.action.worlds import (
        DISTRACTOR, SUPPLY_BODY, SUPPLY_XY, WANTED, nominal_cfg,
    )
    from brick_builder.campaign.tasks import TaskSpec
    from brick_builder.scene.compose import build_scene, reset_scene

    spec = TaskSpec("nominal_tower", pick=True, cells=((4, 3, 0), (4, 3, 1)))
    for cycle in range(spec.cycles):
        cfg = PlaceConfig(pick=True)
        placed = spec.placed_before(cycle)
        ideal, nominal_world = record_ideal(cfg, seed=0, target=spec.cells[cycle],
                                            placed=placed)
        nominal_world.close()
        # The nominal world, plus whatever the earlier cycle was sent to place.
        cfg_scene = nominal_cfg(held=None, placed=placed, supply=(WANTED, DISTRACTOR))
        cfg_scene.supply_xy = SUPPLY_XY
        scene = build_scene(cfg_scene)
        reset_scene(scene, np.random.default_rng(0))
        episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=0,
                                  target=spec.cells[cycle], placed=list(placed))
        try:
            for _ in episode.run():
                pass
            check = verify_placement(scene, SUPPLY_BODY, (4.5, 3.5, cycle))
        finally:
            episode.close()
        assert episode.failure is None, episode.failure
        assert check.at_target_layer, (
            f"cycle {cycle} landed {check.height_error_mm:+.2f} mm off layer {cycle}")
        assert check.at_target_cell and check.seated, (
            f"cycle {cycle}: lateral {check.lateral_mm:.2f} mm, "
            f"depth error {check.depth_error_mm:.2f} mm")
        assert check.lateral_mm < 0.5, f"cycle {cycle}: lateral {check.lateral_mm:.2f} mm"
