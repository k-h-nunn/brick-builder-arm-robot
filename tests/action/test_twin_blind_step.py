"""`DigitalTwin.build_step` with nothing to line up on: a scored failure, not a batch-killing
crash.

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


def test_zero_reference_blobs_raises_blind_step_not_runtime_error(twin, capsys) -> None:
    from brick_builder.action.twin import BlindStep

    goal_tcp = np.array([0.20, 0.0, 0.05])
    with pytest.raises(BlindStep) as excinfo:
        twin.build_step("Approach", goal_tcp, 0.0, {}, min_refs=1)
    assert "shows 0 of the 0 reference blobs" in str(excinfo.value)
    # `BlindStep` is a `RuntimeError`, so the old bare-`RuntimeError` call sites (and callers that
    # still only catch `RuntimeError`) keep working.
    assert isinstance(excinfo.value, RuntimeError)


def test_zero_reference_blobs_logs_the_count(twin, capsys) -> None:
    goal_tcp = np.array([0.20, 0.0, 0.05])
    with pytest.raises(Exception):
        twin.build_step("Approach", goal_tcp, 0.0, {}, min_refs=1)
    err = capsys.readouterr().err
    assert "0 of 0 reference" in err
    assert "Approach" in err
