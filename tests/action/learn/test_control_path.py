"""The learned inverse is control code, so it answers to the same rules `place.py` does: slot
numbers only, no viewer or decode code, no simulator, and no torch at run time."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[3] / "brick_builder" / "action" / "models"
FORBIDDEN_MODULES = ("readout", "producers", "conventional_scene_model", "video", "demo",
                     "torch", "mujoco")


def test_the_models_package_imports_nothing_a_controller_may_not_have() -> None:
    offences = []
    for path in sorted(MODELS.glob("*.py")):
        source = path.read_text()
        if ".decode(" in source:
            offences.append(f"{path.name} calls .decode(")
        for banned in ("scene_for_scoring", "body_pose(", "mj_step", "MjData", "data.qpos"):
            if banned in source:
                offences.append(f"{path.name} reaches into the simulator: {banned}")
        for node in ast.walk(ast.parse(source)):
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


def test_place_does_not_import_the_data_or_evaluation_code() -> None:
    source = (MODELS.parent / "place.py").read_text()
    for banned in ("learn.collect", "learn.evaluate", "from .learn", "verify_placement"):
        assert banned not in source, f"place.py pulls in off-control-path code: {banned}"


def test_the_default_inverse_is_the_analytic_one() -> None:
    from brick_builder.action.place import PlaceConfig

    assert PlaceConfig().inverse == "analytic"
    assert PlaceConfig(pick=True).inverse == "analytic"


# ------------------------------------------------------------------------------ the tick call ---


class _Features:
    """A `FeatureSet` as far as the inverse is concerned."""

    def __init__(self, slots, weights, order):
        self.slots, self.weights, self.order = slots, weights, order


def _state(rng, k: int = 8):
    from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState

    return SceneState(slots=rng.normal(0.0, 0.1, size=(k, 14)), depth=np.full(k, 0.3),
                      schema=PLACEHOLDER_SLOT_SCHEMA, t=0.0, source="test", window=None,
                      slot_ids=np.arange(k))


def test_the_inverse_answers_one_tick_with_three_numbers(tmp_path) -> None:
    from brick_builder.action.models.features import HELD_NAME, INVERSE_IN, INVERSE_OUT
    from brick_builder.action.models.inverse import LearnedInverse
    from brick_builder.action.models.mlp import Mlp, std_floor

    rng = np.random.default_rng(0)
    net = Mlp(layers=[(rng.normal(0, 0.2, (16, INVERSE_IN)), np.zeros(16)),
                      (rng.normal(0, 0.2, (INVERSE_OUT, 16)), np.zeros(INVERSE_OUT))],
              in_mean=np.zeros(INVERSE_IN), in_std=std_floor(np.ones(INVERSE_IN)),
              out_mean=np.zeros(INVERSE_OUT), out_std=std_floor(np.full(INVERSE_OUT, 1e-3)))
    g = LearnedInverse(net)

    order = ("baseplate", "placed brick 0", HELD_NAME)
    found = _Features({"baseplate": 1, "placed brick 0": 2, HELD_NAME: 3},
                      {"baseplate": 0.9, "placed brick 0": 0.8, HELD_NAME: 0.7}, order)
    want = _Features({"baseplate": 4, "placed brick 0": 5, HELD_NAME: 6},
                     {"baseplate": 1.0, "placed brick 0": 1.0, HELD_NAME: 1.0}, order)
    state, target = _state(rng), _state(rng)
    mech = rng.normal(size=12)

    x = g.features(state=state, target=target, found=found, want=want, mech=mech, in_hand=True)
    assert x.shape == (INVERSE_IN,)
    delta = g.delta(state=state, target=target, found=found, want=want, mech=mech, in_hand=True)
    assert delta.shape == (INVERSE_OUT,) and np.all(np.isfinite(delta))
    assert np.array_equal(delta, g.delta(state=state, target=target, found=found, want=want,
                                         mech=mech, in_hand=True))


def test_the_residual_form_adds_to_the_analytic_answer() -> None:
    from brick_builder.action.models.features import INVERSE_IN, INVERSE_OUT
    from brick_builder.action.models.inverse import LearnedInverse
    from brick_builder.action.models.mlp import Mlp, std_floor

    net = Mlp(layers=[(np.zeros((INVERSE_OUT, INVERSE_IN)), np.zeros(INVERSE_OUT))],
              in_mean=np.zeros(INVERSE_IN), in_std=std_floor(np.ones(INVERSE_IN)),
              out_mean=np.zeros(INVERSE_OUT), out_std=std_floor(np.ones(INVERSE_OUT)))
    g = LearnedInverse(net, residual=True)
    rng = np.random.default_rng(1)
    order = ("baseplate",)
    found = _Features({"baseplate": 1}, {"baseplate": 1.0}, order)
    analytic = np.array([1e-3, -2e-3, 4e-3])
    out = g.delta(state=_state(rng), target=_state(rng), found=found, want=found,
                  mech=np.zeros(12), in_hand=False, analytic=analytic)
    assert np.allclose(out, analytic), "a zero network must leave the analytic answer alone"


@pytest.mark.slow
def test_one_closed_loop_episode_with_the_learned_inverse() -> None:
    """A whole placement, learned inverse in the loop. Minutes, not seconds."""
    from brick_builder.action.models.inverse import WEIGHTS

    if not (WEIGHTS / "inverse.npz").exists():
        pytest.skip("inverse.npz has not been trained yet")
    from brick_builder.action.learn.evaluate import run_episode

    row = run_episode("placement", 1, inverse="learned")
    assert row["vision_ticks"] > 40
    assert row["lateral_mm"] == row["lateral_mm"]
