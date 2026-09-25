"""DigitalTwin.build_step with too few reference blobs

Under `--source cnn` the recorded reference set for a step can point at bodies the twin's own
render never shows (the scene model's slot ids are not the renderer's instance ids -- see
`docs/guide/16b-campaign-on-the-scene-model.md`). Before this test, that raised a bare
`RuntimeError` that `place.py`'s `_prepare_step` does not catch, which loses the whole episode.
`build_step` now raises `BlindStep`, the same exception `_prepare_step` already catches for the
Jacobian-probing check just below this one, so the step is marked untrusted instead.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def twin():
    from brick_builder.action.twin import DigitalTwin
    from brick_builder.action.worlds import nominal_cfg

    twin = DigitalTwin(nominal_cfg(held=None))
    yield twin
    twin.close()


def test_zero_named_bodies_raises_blind_step_not_runtime_error(twin) -> None:
    from brick_builder.action.twin import BlindStep

    goal_tcp = np.array([0.20, 0.0, 0.05])
    with pytest.raises(BlindStep) as excinfo:
        twin.build_step("Approach", goal_tcp, 0.0, {}, min_refs=1)
    assert "shows 0 of the 0 reference blobs" in str(excinfo.value)
    # `BlindStep` is a `RuntimeError`, so the old bare-`RuntimeError` call sites (and callers that
    # still only catch `RuntimeError`) keep working.
    assert isinstance(excinfo.value, RuntimeError)


def test_zero_named_bodies_logs_the_count(twin, capsys) -> None:
    from brick_builder.action.twin import BlindStep

    goal_tcp = np.array([0.20, 0.0, 0.05])
    with pytest.raises(BlindStep):
        twin.build_step("Approach", goal_tcp, 0.0, {}, min_refs=1)
    err = capsys.readouterr().err
    assert "0 of 0 reference" in err
    assert "Approach" in err


def test_one_named_body_out_of_frame_raises_blind_step(twin, monkeypatch, capsys) -> None:
    """A body that `calibrate` resolves to a real slot (so it counts as a reference) but that
    `reference_features` then finds no usable blob for (the "0 of the 1" case seen at `Lift` in
    `results/campaign/closed_loop_cnn/rounds/r0/evaluate_process_0.log:59`) is the same branch as
    the zero-bodies case above, with a non-zero denominator. `reference_features` is patched to
    drop every centre it would otherwise report -- the truncation or tolerance mismatch that does
    that in a real render is not reproduced here, only its effect on `build_step`.
    """
    from brick_builder.action import servo as servo_mod
    from brick_builder.action.twin import BlindStep

    real_reference_features = servo_mod.reference_features

    def no_centres(state, references):
        found = real_reference_features(state, references)
        for name in found.centres:
            found.centres[name] = None
        return found

    monkeypatch.setattr(servo_mod, "reference_features", no_centres)

    goal_tcp = np.array([0.20, 0.0, 0.05])
    bodies = {"the plate": twin.scene.plate.name}
    with pytest.raises(BlindStep) as excinfo:
        twin.build_step("Approach", goal_tcp, 0.0, bodies, min_refs=1)
    assert "shows 0 of the 1 reference blobs" in str(excinfo.value)
    err = capsys.readouterr().err
    assert "0 of 1 reference" in err
