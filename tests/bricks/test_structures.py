"""Tests for brick_builder.bricks.structures."""

from __future__ import annotations

import mujoco
import pytest

from brick_builder.bricks.generator import add_baseplate
from brick_builder.bricks.spec import DUPLO
from brick_builder.bricks.structures import (
    Placement,
    build_structure,
    format_placement,
    format_structure,
    occupancy_grid,
    parse_placement,
    parse_structure,
    validate_structure,
)


def test_parse_and_format_single_placement_round_trip():
    text = "2x4 (0,0,0) red"
    p = parse_placement(text)
    assert p == Placement(h=2, w=4, x=0, y=0, z=0, colour="red")
    assert format_placement(p) == text


def test_parse_negative_grid_coordinates():
    p = parse_placement("2x2 (-3,4,1) blue")
    assert (p.x, p.y, p.z) == (-3, 4, 1)


def test_parse_placement_rejects_garbage():
    with pytest.raises(ValueError):
        parse_placement("not a placement")


def test_structure_round_trip():
    text = "2x4 (0,0,0) red\n2x2 (0,4,0) blue\n2x4 (0,0,1) yellow"
    placements = parse_structure(text)
    assert len(placements) == 3
    assert format_structure(placements) == text


def test_occupancy_grid_matches_footprint_cells():
    placements = parse_structure("2x4 (0,0,0) red")
    grid = occupancy_grid(placements)
    assert len(grid) == 2 * 4
    assert grid[(0, 0, 0)] == 0
    assert grid[(3, 1, 0)] == 0
    assert (4, 0, 0) not in grid


def test_validate_structure_accepts_a_valid_stack():
    text = "2x4 (0,0,0) red\n2x4 (0,0,1) blue"
    problems = validate_structure(parse_structure(text), grid_shape=(8, 8))
    assert problems == []


def test_validate_structure_flags_overlap():
    text = "2x4 (0,0,0) red\n2x2 (1,0,0) blue"
    problems = validate_structure(parse_structure(text))
    assert any("overlap" in p for p in problems)


def test_validate_structure_flags_out_of_grid():
    text = "2x4 (7,0,0) red"
    problems = validate_structure(parse_structure(text), grid_shape=(8, 8))
    assert any("out of the" in p for p in problems)


def test_validate_structure_flags_unsupported_brick():
    text = "2x4 (0,0,1) red"  # layer 1, nothing at layer 0 underneath
    problems = validate_structure(parse_structure(text))
    assert any("unsupported" in p for p in problems)


def test_validate_structure_flags_unknown_colour():
    problems = validate_structure([Placement(h=2, w=4, x=0, y=0, z=0, colour="chartreuse")])
    assert any("unknown colour" in p for p in problems)


def test_build_structure_places_bricks_and_compiles():
    world = mujoco.MjSpec()
    plate = add_baseplate(world, "plate", 8, 8, pos=(0.0, 0.0, 0.0))
    placements = parse_structure("2x4 (0,0,0) red\n2x4 (0,0,1) blue")
    handles = build_structure(world, placements, plate)
    assert len(handles) == 2
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.warning.number.sum() == 0

    # the second (upper) brick must sit exactly one brick height above the first
    body0 = model.body(handles[0].name).id
    body1 = model.body(handles[1].name).id
    assert data.xpos[body1, 2] - data.xpos[body0, 2] == pytest.approx(DUPLO.height, abs=1e-9)
    # both share the same x, y (same footprint position, stacked directly)
    assert data.xpos[body1, 0] == pytest.approx(data.xpos[body0, 0], abs=1e-9)
    assert data.xpos[body1, 1] == pytest.approx(data.xpos[body0, 1], abs=1e-9)


def test_built_structure_is_stable_at_rest():
    """A valid, supported structure built directly at its resting positions should not need
    to fall or settle - it is already resting flush, so it should barely move."""

    world = mujoco.MjSpec()
    plate = add_baseplate(world, "plate", 8, 8, pos=(0.0, 0.0, 0.0))
    placements = parse_structure("2x4 (0,0,0) red\n2x4 (0,0,1) blue")
    handles = build_structure(world, placements, plate)
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    z0 = [data.qpos[model.joint(f"{h.name}_free").qposadr[0] + 2] for h in handles]
    for _ in range(int(1.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    z1 = [data.qpos[model.joint(f"{h.name}_free").qposadr[0] + 2] for h in handles]
    for a, b in zip(z0, z1):
        assert abs(a - b) < 2e-4
