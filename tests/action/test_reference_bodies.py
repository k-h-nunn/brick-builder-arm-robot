"""Reference bodies of a recording made through a scene model other than ground truth

Only the ground-truth source's `slot_ids` are renderer instance ids; the convolutional scene
model's are slot indices. `reference_bodies` read them as instance ids for every source, so a
recording made through the scene model named the wrong reference bodies, or none (see
`docs/guide/16b-campaign-on-the-scene-model.md`, Mislabeled reference bodies). It now names a
non-ground-truth slot after the ground-truth blob of the same frame it matches by centre, mass
and colour (`place.match_slots`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brick_builder.action.place import match_slots, reference_bodies
from brick_builder.perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SceneState

PLACE = Path(__file__).resolve().parents[2] / "brick_builder" / "action" / "place.py"

# Instance ids as `scene.compose.build_scene` hands them out: 1 the arm, 2 the plate, 3 the
# held brick, 4 onwards the bricks on the plate.
SCENE = SimpleNamespace(
    plate=SimpleNamespace(name="plate"),
    placed_names=["placed_0", "placed_1"],
    bricks={"held": None, "placed_0": None, "placed_1": None},
    instance_ids={"arm": 1, "plate": 2, "held": 3, "placed_0": 4, "placed_1": 5},
)

YELLOW = (0.9, 0.7, -0.8)
RED = (0.5, -0.8, -0.8)
BLUE = (-0.8, -0.4, 0.6)
GREY = (0.1, 0.1, 0.1)


@dataclass(frozen=True)
class Blob:
    centre: tuple[float, float]
    mass: float
    colour: tuple[float, float, float]
    border: float = 0.0


# One frame over the plate at Align, as ground truth sees it: instance id -> blob.
TRUTH = {
    1: Blob((0.0, 0.85), 0.20, GREY, border=0.4),     # arm, jaws in view
    2: Blob((0.0, 0.0), 0.45, GREY, border=0.5),      # baseplate, running off the frame
    3: Blob((0.0, 0.55), 0.08, BLUE),                 # held brick
    4: Blob((-0.45, 0.2), 0.020, YELLOW),             # placed_0
    5: Blob((0.35, -0.3), 0.006, RED),                # placed_1
}


def state_of(blobs: dict[int, Blob], *, K: int, source: str,
             slot_of: dict[int, int] | None = None) -> SceneState:
    """`blobs` as slot rows. `slot_of` puts each blob in a slot; `None` is slot = key."""
    slots = np.zeros((K, PLACEHOLDER_SLOT_SCHEMA.W), dtype=np.float32)
    ids = np.full(K, -1, dtype=np.int64)
    f = PLACEHOLDER_SLOT_SCHEMA.fields
    for key, b in blobs.items():
        k = key if slot_of is None else slot_of[key]
        slots[k, f["mass"]] = b.mass
        slots[k, f["centre"]] = b.centre
        slots[k, f["moments"]] = (0.01, 0.0, 0.01)
        slots[k, f["colour"]] = b.colour
        slots[k, f["border"]] = b.border
        ids[k] = key if source == "ideal" else k
    return SceneState(slots=slots, depth=np.full(K, 0.5, dtype=np.float32),
                      schema=PLACEHOLDER_SLOT_SCHEMA, t=0.0, source=source, window=None,
                      slot_ids=ids)


def truth_state() -> SceneState:
    return state_of(TRUTH, K=16, source="ideal")


def model_state(blobs: dict[int, Blob], slot_of: dict[int, int]) -> SceneState:
    """A trained model's state: slot index as the identity token, source not `ideal`."""
    return state_of(blobs, K=8, source="conv_net", slot_of=slot_of)


# The model's own slot order, chosen so every blob sits in a slot whose index is ANOTHER
# body's instance id: slot 2 (the plate's id) holds the yellow brick, slot 4 (placed_0's id)
# holds the red one, slot 5 (placed_1's id) holds the arm.
MISLEADING = {4: 2, 5: 4, 1: 5, 2: 1, 3: 3}
EXPECTED = {"placed brick 0": "placed_0", "placed brick 1": "placed_1"}


def test_ground_truth_names_by_instance_id() -> None:
    assert reference_bodies(SCENE, truth_state(), exclude="held") == EXPECTED
    truth = truth_state()
    assert reference_bodies(SCENE, truth, exclude="held", truth=truth) == EXPECTED


def test_model_slots_are_named_by_appearance_not_by_index() -> None:
    state = model_state(TRUTH, MISLEADING)
    assert reference_bodies(SCENE, state, exclude="held", truth=truth_state()) == EXPECTED


def test_reading_model_slot_indices_as_instance_ids_names_the_wrong_body() -> None:
    """The defect, reproduced: the same state read as if its tokens were instance ids."""
    state = model_state(TRUTH, MISLEADING)
    as_if_ideal = replace(state, source="ideal")
    wrong = reference_bodies(SCENE, as_if_ideal, exclude="held")
    assert wrong != EXPECTED
    assert wrong.get("placed brick 1") != "placed_1"


def test_model_state_without_truth_is_refused() -> None:
    state = model_state(TRUTH, MISLEADING)
    with pytest.raises(ValueError, match="not instance ids"):
        reference_bodies(SCENE, state, exclude="held")


def test_match_is_independent_of_slot_order() -> None:
    truth = truth_state()
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(8)[:len(TRUTH)]
        slot_of = {key: int(k) for key, k in zip(TRUTH, perm)}
        state = model_state(TRUTH, slot_of)
        matched = match_slots(state, truth)
        assert matched == {slot_of[key]: key for key in TRUTH}
        assert reference_bodies(SCENE, state, exclude="held", truth=truth) == EXPECTED


def test_small_segmentation_error_still_matches() -> None:
    """A blob a few pixels off with a slightly different outline is still the same body."""
    noisy = dict(TRUTH)
    noisy[4] = Blob((-0.42, 0.23), 0.024, (0.85, 0.65, -0.75))
    noisy[5] = Blob((0.33, -0.28), 0.0045, (0.45, -0.75, -0.8))
    state = model_state(noisy, MISLEADING)
    assert reference_bodies(SCENE, state, exclude="held", truth=truth_state()) == EXPECTED


def test_two_bricks_reported_as_one_blob_name_neither() -> None:
    """Two equal bricks side by side, reported as one blob at their joint centre with their
    joint mass: that blob is neither brick, and neither is named."""
    side = dict(TRUTH)
    side[4] = Blob((-0.1, 0.0), 0.010, RED)
    side[5] = Blob((0.1, 0.0), 0.010, RED)
    truth = state_of(side, K=16, source="ideal")
    merged = {k: b for k, b in side.items() if k != 5}
    merged[4] = Blob((0.0, 0.0), 0.020, RED)
    state = model_state(merged, {1: 5, 2: 1, 3: 3, 4: 2})
    assert reference_bodies(SCENE, state, exclude="held", truth=truth) == {}


def test_a_brick_merged_into_the_held_one_is_not_named() -> None:
    """A second layer: the brick under the held one, same colour, reported as part of the held
    brick's blob. The blob is the held brick (excluded); the brick under it is not named."""
    stack = dict(TRUTH)
    stack[5] = Blob((0.0, 0.40), 0.03, BLUE)
    truth = state_of(stack, K=16, source="ideal")
    merged = {k: b for k, b in stack.items() if k != 5}
    merged[3] = Blob((0.0, 0.509), 0.11, BLUE)
    state = model_state(merged, {1: 5, 2: 1, 3: 3, 4: 2})
    named = reference_bodies(SCENE, state, exclude="held", truth=truth)
    assert named == {"placed brick 0": "placed_0"}


def test_a_missed_body_is_not_named() -> None:
    missed = {k: b for k, b in TRUTH.items() if k != 5}
    state = model_state(missed, {1: 5, 2: 1, 3: 3, 4: 2})
    assert reference_bodies(SCENE, state, exclude="held",
                            truth=truth_state()) == {"placed brick 0": "placed_0"}


def test_a_blob_in_the_right_place_with_the_wrong_colour_is_not_named() -> None:
    wrong = dict(TRUTH)
    wrong[5] = Blob(TRUTH[5].centre, TRUTH[5].mass, BLUE)
    state = model_state(wrong, MISLEADING)
    assert reference_bodies(SCENE, state, exclude="held",
                            truth=truth_state()) == {"placed brick 0": "placed_0"}


def test_the_model_background_slot_is_never_named() -> None:
    """A model's background slot carries a row (everything unclaimed) and token -1."""
    state = model_state(TRUTH, MISLEADING)
    f = PLACEHOLDER_SLOT_SCHEMA.fields
    k = 7
    state.slots[k, f["mass"]] = TRUTH[4].mass
    state.slots[k, f["centre"]] = TRUTH[4].centre
    state.slots[k, f["colour"]] = TRUTH[4].colour
    assert int(state.slot_ids[k]) == -1
    assert k not in match_slots(state, truth_state())


def test_the_held_brick_is_matched_and_then_excluded() -> None:
    state = model_state(TRUTH, MISLEADING)
    truth = truth_state()
    assert match_slots(state, truth)[MISLEADING[3]] == 3
    named = reference_bodies(SCENE, state, exclude=None, truth=truth)
    assert named["held"] == "held"
    assert "held" not in reference_bodies(SCENE, state, exclude="held", truth=truth).values()


def test_only_the_recording_reads_ground_truth_beside_the_source() -> None:
    """`observe_labelled` hands ground truth to its caller; `record_ideal` is the only one."""
    tree = ast.parse(PLACE.read_text())
    calls = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.ClassDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr == "observe_labelled":
                    calls.append(fn.name)
    assert calls, "record_ideal no longer reads ground truth beside the source"
    outer = {name for name in calls if name != "snapshot"}
    assert outer == {"record_ideal"}, calls


@dataclass
class SlotIndexTokens:
    """Ground-truth blobs published under slot-index tokens, as a trained model publishes them.

    The rows are exact, so every difference from a ground-truth recording is the naming alone.
    """

    inner: object

    def observe(self, rgb, *, window, instance=None, depth=None, present_mask=None, t=None):
        state = self.inner.observe(rgb, window=window, instance=instance, depth=depth, t=t)
        tokens = np.where(state.slot_ids > 0, np.arange(state.K), -1)
        return replace(state, slot_ids=tokens, source="slot_index_tokens")


@pytest.fixture(scope="module")
def recordings():
    """The placement task recorded twice in the same nominal world: ground truth, then the
    same rows under slot-index tokens."""
    from brick_builder.action.place import PlaceConfig, record_ideal
    from brick_builder.perception.ideal_scene_model import IdealSceneModel

    truth, world = record_ideal(PlaceConfig(), seed=0)
    world.close()
    tokens, world = record_ideal(PlaceConfig(), seed=0,
                                 source=SlotIndexTokens(IdealSceneModel(build_decode=False)))
    scene = world.scene_for_scoring
    yield truth, tokens, scene
    world.close()


def test_recording_through_another_source_names_the_ground_truth_bodies(recordings) -> None:
    truth, tokens, _ = recordings
    assert len(truth) == len(tokens)
    for a, b in zip(truth.frames, tokens.frames):
        assert (a.step, a.step_index) == (b.step, b.step_index)
        assert b.state.source == "slot_index_tokens"
        assert a.bodies == b.bodies, (a.k, a.step, a.bodies, b.bodies)
    assert truth.bodies_by_step == tokens.bodies_by_step
    align = [f for f in tokens.frames if f.step == "Align"]
    assert align and all(len(f.bodies) >= 2 for f in align)


def test_the_same_recording_read_by_slot_index_is_wrong(recordings) -> None:
    """The old reading of the same states disagrees with ground truth: the defect is real here."""
    truth, tokens, scene = recordings
    wrong = 0
    for a, b in zip(truth.frames, tokens.frames):
        held = tokens.held_body if a.step in ("Transport", "Align", "Press the brick down") \
            else None
        old = reference_bodies(scene, replace(b.state, source="ideal"), exclude=held)
        wrong += old != a.bodies
    assert wrong > len(truth) // 2, f"{wrong} of {len(truth)} frames"
