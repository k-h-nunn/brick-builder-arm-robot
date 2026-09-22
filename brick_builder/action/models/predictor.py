"""`F` wired as the controller's one-step prediction of the tracked residual.

The loop already predicts, every tick, what the next tick's residual will look like, and reports
the error of that prediction. Its own rule is one line and carries no model: *the residual will
change by exactly the correction I just commanded*, `s_hat = measured + delta`. It ignores the
feed-forward, the target frame's own motion and everything the scene does that the tool did not
ask for.

This module is the other rule. `F` predicts the change of the observed slot rows over one tick,
given the action the world is about to see; those predicted rows are written back into a copy of
the scene state and handed to the **same** `VisualServo.update` the real reading goes through, so
what comes out is a predicted residual in the same metres of tool motion, through the same
feature matching, the same weights and the same pseudo-inverse. The two predictions are then
comparable tick for tick, which is the whole point of the switch.

Nothing here steers the arm. The prediction is reported (`TickRecord.predict_mm`,
`TrackingEpisode.pred_errors`) and nothing on the control path reads it, so selecting the learned
predictor changes what is measured and not what is done.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import (
    HELD_NAME, MECH_W, N_SLATE, REF_SLOTS, ROW_FIELDS, ROW_W, encode_predictor, read_slate,
    slate_names,
)
from .inverse import WEIGHTS
from .mlp import Mlp


class LearnedPredictor:
    """`F`, called once per vision tick to predict the next tick's residual."""

    def __init__(self, net: Mlp) -> None:
        self.net = net
        self.calls = 0
        self.fallbacks = 0
        """Ticks where the predicted state could not be read back through the servo (the
        references stopped matching on it), and the controller's own rule stood in. Reported
        rather than hidden: a prediction that cannot be measured is not a prediction of zero."""

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LearnedPredictor":
        return cls(Mlp.load(Path(path) if path is not None else WEIGHTS / "predictor.npz"))

    # -- the tick ---------------------------------------------------------------------------------

    def predict(self, episode, frame, out: dict, measured: np.ndarray,
                delta: np.ndarray) -> np.ndarray:
        """`[3]` = the predicted next residual, (dx [m], dy [m], spin [rad]).

        `episode` is the running `TrackingEpisode`; everything read from it is what the loop has
        already read this tick -- the observed state, the matched target frame, the step's
        references and the mechanical state -- plus the recorded tool targets, which are nominal
        task knowledge.
        """
        state = episode._state
        target = episode.ideal.frames[episode.matched].state
        found, want = out["found"], out.get("want")
        if state is None or want is None or episode._servo is None or episode._step is None:
            self.fallbacks += 1
            return np.asarray(measured, dtype=float) + np.asarray(delta, dtype=float)

        names = slate_names(found.order)
        obs_rows, obs_valid, obs_weight = read_slate(
            state, found.slots, found.weights, names,
            found.slots.get(HELD_NAME), found.weights.get(HELD_NAME, 0.0))
        mech = np.concatenate([episode.world.joint_position, episode.world.joint_speed])[:MECH_W]
        in_hand = episode.ideal.held_by_step.get(frame.step_index) is not None
        action = _action(episode, frame, delta)

        x = encode_predictor(obs_rows, obs_valid, obs_weight, mech, action, float(in_hand))
        change = np.asarray(self.net.forward(x), dtype=np.float64).reshape(N_SLATE, ROW_W)
        self.calls += 1

        nxt = _advanced(state, found, names, obs_valid, change)
        rel = episode.frozen and episode.ideal.held_by_step.get(frame.step_index) is None
        trial = episode._servo.update(nxt, episode._step.refs, reference=target, dry_run=True,
                                      relative=rel)
        if trial["blind"]:
            self.fallbacks += 1
            return np.asarray(measured, dtype=float) + np.asarray(delta, dtype=float)
        return np.array([trial["d_r"][0] + trial["d_b"][0],
                         trial["d_r"][1] + trial["d_b"][1], trial["d_r"][2]])


def _action(episode, frame, delta: np.ndarray) -> np.ndarray:
    """The tool-target change the world is about to see: the recording's own step, plus the
    correction increment that was just added to it. The same quantity the log calls `action`."""
    frames = episode.ideal.frames
    k = min(episode.k + 1, len(frames) - 1)
    here, nxt = frame.commands[-1], frames[k].commands[-1]
    return np.array([nxt.pos[0] - here.pos[0] + delta[0],
                     nxt.pos[1] - here.pos[1] + delta[1],
                     nxt.pos[2] - here.pos[2],
                     nxt.spin - here.spin + delta[2]])


def _advanced(state, found, names, obs_valid: np.ndarray, change: np.ndarray):
    """A copy of `state` with the slate's predicted rows one tick ahead.

    Only the eight fields `F` predicts are moved. Colour, colour spread, border fraction and age
    are carried over unchanged: `F` does not predict them, and inventing a value for them would
    change which slot the references match on for a reason that has nothing to do with the
    prediction.
    """
    from dataclasses import replace

    slots = np.array(state.slots, dtype=np.float32, copy=True)
    order = list(names[:REF_SLOTS]) + [HELD_NAME]
    for i, name in enumerate(order):
        k = found.slots.get(name)
        if k is None or obs_valid[i] <= 0.0:
            continue
        row = slots[int(k)]
        offset = 0
        for field in ROW_FIELDS:
            sl = state.schema.fields[field]
            width = sl.stop - sl.start
            row[sl] = row[sl] + change[i, offset:offset + width]
            offset += width
    return replace(state, slots=slots)
