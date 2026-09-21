"""Tests for scene-bricks-1: `randomize.apply()` must merge into `scene.randomization`, not
replace it wholesale, and `randomize.reset_to_nominal()` must restore every randomized model
field to nominal and clear the record.

Self-contained (a local `scene` fixture, not shared with tests/scene/test_randomize.py) per this
work's instruction not to touch any other file in tests/scene/.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.bricks.structures import Placement
from brick_builder.robot.patch_so101 import ARM_JOINTS
from brick_builder.scene import randomize as R
from brick_builder.scene.compose import SceneConfig, build_scene

PLACED = [Placement(2, 4, 1, 2, 0, "blue"), Placement(2, 2, 5, 3, 0, "yellow")]


@pytest.fixture
def scene():
    return build_scene(SceneConfig(placed=list(PLACED)))


def _snapshot(model) -> dict:
    """Every model array a `RANGES` group can write. Same fields as
    tests/scene/test_randomize.py's own `_snapshot`, duplicated locally rather than shared."""
    return {
        "gain": model.actuator_gainprm.copy(),
        "bias": model.actuator_biasprm.copy(),
        "forcerange": model.actuator_forcerange.copy(),
        "damping": model.dof_damping.copy(),
        "frictionloss": model.dof_frictionloss.copy(),
        "armature": model.dof_armature.copy(),
        "jnt_range": model.jnt_range.copy(),
        "body_mass": model.body_mass.copy(),
        "geom_friction": model.geom_friction.copy(),
        "geom_size": model.geom_size.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "body_pos": model.body_pos.copy(),
        "body_quat": model.body_quat.copy(),
        "light_pos": model.light_pos.copy(),
        "cam_pos": model.cam_pos.copy(),
        "cam_fovy": model.cam_fovy.copy(),
        "eq_solref": model.eq_solref.copy(),
    }


def _assert_snapshots_equal(a: dict, b: dict) -> None:
    for key in a:
        assert np.allclose(a[key], b[key]), key


# --------------------------------------------------------------------------------- merging ---


def test_two_successive_partial_applies_merge_into_one_record(scene):
    """The exact scenario the finding names: `apply()` with `groups=("dynamics",)` then
    `groups=("contact",)` must leave BOTH groups' fields in `scene.randomization`, not just the
    most recent call's -- `apply()`'s own docstring/contract is that this record always
    reproduces the current model state, and both groups are still live in it."""
    dynamics_sample = R.sample(np.random.default_rng(1), 1.0, groups=("dynamics",))
    R.apply(scene, dynamics_sample)
    assert scene.randomization == dynamics_sample

    contact_sample = R.sample(np.random.default_rng(2), 1.0, groups=("contact",))
    R.apply(scene, contact_sample)

    # scene-bricks-1's exact failure mode: dynamics.* must not have vanished from the record.
    assert set(scene.randomization) == set(dynamics_sample) | set(contact_sample)
    for key, value in dynamics_sample.items():
        assert scene.randomization[key] == pytest.approx(value), f"{key} was lost on the second apply"
    for key, value in contact_sample.items():
        assert scene.randomization[key] == pytest.approx(value)

    # And the model itself still reflects the first call's dynamics values -- the record isn't
    # merely "not wrong", it still matches what is physically live in the model.
    kp = float(dynamics_sample["dynamics.kp"])
    assert np.allclose(scene.model.actuator_gainprm[: len(ARM_JOINTS), 0], kp)
    friction = float(contact_sample["contact.brick_friction"])
    assert np.allclose(scene.model.geom_friction[R.brick_geoms(scene), 0], friction)


def test_reapplying_a_group_updates_only_that_groups_keys_in_the_record(scene):
    """Merging must still let a key be *updated*: a second `dynamics` sample must replace the
    first one's values in the record (and the model), while an unrelated `contact` sample
    applied in between stays untouched."""
    first_dynamics = R.sample(np.random.default_rng(3), 1.0, groups=("dynamics",))
    R.apply(scene, first_dynamics)

    contact_sample = R.sample(np.random.default_rng(4), 1.0, groups=("contact",))
    R.apply(scene, contact_sample)

    second_dynamics = R.sample(np.random.default_rng(5), 1.0, groups=("dynamics",))
    assert second_dynamics["dynamics.kp"] != first_dynamics["dynamics.kp"]  # sanity: actually different
    R.apply(scene, second_dynamics)

    assert set(scene.randomization) == set(second_dynamics) | set(contact_sample)
    for key, value in second_dynamics.items():
        assert scene.randomization[key] == pytest.approx(value)
    for key, value in contact_sample.items():
        assert scene.randomization[key] == pytest.approx(value), f"{key} was disturbed by the unrelated re-apply"

    kp = float(second_dynamics["dynamics.kp"])
    assert np.allclose(scene.model.actuator_gainprm[: len(ARM_JOINTS), 0], kp)


def test_apply_still_makes_its_own_detached_copy_of_the_record(scene):
    """Regression guard on a property the old wholesale-replace code already had: mutating the
    dict the caller passed to `apply()` afterwards must not reach back into `scene.randomization`."""
    sample = R.sample(np.random.default_rng(6), 1.0, groups=("plate",))
    R.apply(scene, sample)
    sample["plate.dx"] = 999.0
    assert scene.randomization["plate.dx"] != 999.0


# ------------------------------------------------------------------------- reset_to_nominal ---


def test_reset_to_nominal_restores_the_model_and_clears_the_record(scene):
    before = _snapshot(scene.model)

    R.apply(scene, R.sample(np.random.default_rng(7), 1.0))  # every group, far from nominal
    assert scene.randomization  # sanity: something was actually recorded
    after_randomize = _snapshot(scene.model)
    assert any(not np.allclose(before[k], after_randomize[k]) for k in before), (
        "the randomized sample did not actually change the model; test fixture is not exercising anything"
    )

    R.reset_to_nominal(scene)

    assert scene.randomization == {}
    _assert_snapshots_equal(before, _snapshot(scene.model))


def test_reset_to_nominal_after_two_partial_applies_clears_both_groups(scene):
    """The record-merging fix and `reset_to_nominal` composed together: two successive partial
    applies (scene-bricks-1's own scenario), then a full reset, must leave nothing randomized
    from either group -- not just the most recently-applied one."""
    before = _snapshot(scene.model)

    R.apply(scene, R.sample(np.random.default_rng(8), 1.0, groups=("dynamics",)))
    R.apply(scene, R.sample(np.random.default_rng(9), 1.0, groups=("contact",)))
    assert {k.split(".")[0] for k in scene.randomization} == {"dynamics", "contact"}

    R.reset_to_nominal(scene)

    assert scene.randomization == {}
    _assert_snapshots_equal(before, _snapshot(scene.model))


def test_reset_to_nominal_is_idempotent(scene):
    """Calling it twice in a row (e.g. between two resets with nothing randomized in between)
    must not raise and must leave the same nominal state and empty record."""
    R.apply(scene, R.sample(np.random.default_rng(10), 1.0))
    R.reset_to_nominal(scene)
    once = _snapshot(scene.model)

    R.reset_to_nominal(scene)

    assert scene.randomization == {}
    _assert_snapshots_equal(once, _snapshot(scene.model))


def test_reset_to_nominal_on_a_freshly_built_scene_is_a_no_op_on_the_model(scene):
    """A scene that has never been randomized is already at nominal; resetting it must not
    perturb the model (it is written through the same values it already has) and must leave the
    (already-empty) record empty."""
    before = _snapshot(scene.model)
    assert scene.randomization == {}

    R.reset_to_nominal(scene)

    assert scene.randomization == {}
    _assert_snapshots_equal(before, _snapshot(scene.model))
