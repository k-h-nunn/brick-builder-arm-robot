"""Fast checks on the closed-loop action package.

Nothing here renders a film or runs a whole placement: those are minutes, and these are seconds.
What they do check is the three things that would quietly rot -- the twin being repeatable, the
twin's expectation matching the camera when the world really is nominal, and the controller not
growing a dependency on viewer or decode code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ACTION = Path(__file__).resolve().parents[2] / "brick_builder" / "action"

FORBIDDEN_MODULES = ("readout", "producers", "conventional_scene_model", "cinema", "demo")
"""A controller reads slot numbers. A dense label image is a viewer's business."""


def test_nothing_in_action_imports_viewer_or_decode_code() -> None:
    offences = []
    for path in sorted(ACTION.glob("*.py")):
        source = path.read_text()
        if ".decode(" in source:
            offences.append(f"{path.name} calls .decode(")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            for name in names:
                for bad in FORBIDDEN_MODULES:
                    if bad in name.split("."):
                        offences.append(f"{path.name} imports {name}")
    assert not offences, "; ".join(offences)


def test_the_controller_never_reads_the_simulator_directly() -> None:
    """`place.py` talks to the world only through the adapter's four verbs."""
    source = (ACTION / "place.py").read_text()
    for banned in ("scene_for_scoring", "body_pose(", "mj_step", "MjData", ".snap.",
                   "data.qpos"):
        assert banned not in source, f"place.py reaches into the simulator: {banned}"


@pytest.fixture(scope="module")
def recorded():
    """One ideal sequence, recorded once and shared by the tests that read it."""
    from brick_builder.action.place import PlaceConfig, record_ideal

    ideal, world = record_ideal(PlaceConfig(), seed=0)
    yield ideal
    world.close()


def test_the_ideal_sequence_is_a_sequence_not_a_goal(recorded) -> None:
    """Every tick has a state and an action, the steps are in order, and the titles are titles."""
    from brick_builder.action.place import STEP_TITLES

    assert len(recorded) > 40
    for frame in recorded.frames:
        assert frame.state is not None
        assert frame.commands, f"frame {frame.k} has no action"
        assert frame.action.shape == (4,)
        assert frame.step in STEP_TITLES, frame.step
    assert [f.k for f in recorded.frames] == list(range(len(recorded)))
    indices = [f.step_index for f in recorded.frames]
    assert indices == sorted(indices), "steps are out of order"


def test_every_step_knows_what_it_lines_up_on(recorded) -> None:
    """A step with nothing identifiable in view cannot be tracked, and must say so rather than
    quietly lining up on whatever happens to be there."""
    for index, bodies in recorded.bodies_by_step.items():
        assert isinstance(bodies, dict)
        held = recorded.held_by_step.get(index)
        assert held not in bodies.values(), "the brick in the hand is not a thing to line up on"


def test_the_twin_is_deterministic() -> None:
    """The same pose, visited twice, has to describe the scene the same way."""
    import numpy as np

    from brick_builder.action.place import TARGET_CELL
    from brick_builder.action.twin import DigitalTwin
    from brick_builder.action.worlds import nominal_cfg
    from brick_builder.control.pick_place import cell_world, grasp_pose, tcp_for_brick
    from brick_builder.scene.compose import HeldBrick

    twin = DigitalTwin(nominal_cfg(held=HeldBrick(nx=2, ny=2, colour="blue")))
    try:
        grasp = grasp_pose(2, 2, twin.scene.cfg.spec)
        cell = cell_world(twin.scene, TARGET_CELL[0] + 0.5, TARGET_CELL[1] + 0.5)
        tcp = tcp_for_brick([cell[0], cell[1], cell[2] + 0.020], 0.0, grasp)

        first = twin.observe(tcp, 0.0)
        twin.observe(np.array(tcp) + np.array([0.01, 0.01, 0.0]), 0.0)
        second = twin.observe(tcp, 0.0)

        def by_id(state):
            return {int(state.slot_ids[k]): {f: state.schema.get(state.slots[k], f)
                                             for f in ("mass", "centre", "moments", "colour")}
                    for k in range(state.K) if not state.is_empty(k)}

        a, b = by_id(first), by_id(second)
        common = set(a) & set(b)
        assert len(common) >= 4, (sorted(a), sorted(b))
        tolerance = {"centre": 1e-3, "moments": 1e-3, "mass": 5e-3, "colour": 5e-3}
        for key in common:
            for field, tol in tolerance.items():
                assert np.allclose(a[key][field], b[key][field], atol=tol), (key, field)
    finally:
        twin.close()


def test_tracking_a_displaced_world_brings_the_error_down(recorded) -> None:
    """The point of the whole thing: the same recording, run against a world nobody described,
    ends with the brick on the cell rather than where the recording said it would be."""
    import numpy as np

    from brick_builder.action.place import PlaceConfig, TrackingEpisode
    from brick_builder.action.worlds import build_world

    scene, d_xy, _ = build_world(1)
    assert np.linalg.norm(d_xy) > 0.003, "this world is supposed to be wrong"
    episode = TrackingEpisode(scene, recorded, cfg=PlaceConfig(), seed=1)
    try:
        for _ in episode.run():
            pass
        errs = [e for e in episode.errors if e == e]
        assert errs, "the loop never measured anything"
        assert min(errs) < 1.0, f"best tracking error was {min(errs):.2f} mm"
        assert episode.failure is None, episode.failure
    finally:
        episode.close()
