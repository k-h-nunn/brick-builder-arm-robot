"""Predictive scene model: `F` inside the scene-model interface (predict, then update).

Per tick:

    set_motion(context, a)        context [43] (`model.context_vector`), executed action [4]
    observe(rgb, window=...)      the wrapped source's `z_obs`; returns the filtered state

    z^  = F(z_f(t-1), context, a)        slot order and ids of the previous filtered state
    z_f = z^ + kappa (z_obs - z^)        dynamic components, slots paired by slot id
    z_f = z_obs                          appearance and flags; slots without a prediction

`kappa = sigma_F^2 / (sigma_F^2 + sigma_obs^2)` per dynamic component (`gains`); the default
1 passes the observation through. Without `set_motion` before an `observe` there is no
prediction and the observation passes through. A slot the source no longer reports is dropped.

Logged every tick: `last_prediction`, `last_observation`, `innovation` (`z_obs - z^`,
`[K, D]`, NaN where unpaired) and `innovation_mm` (centre RMS [mm-eq]).

`register_source()` (run at import) registers the factory `build` in `perception.sources` as
`predictive`, with keyword arguments `base` (`ideal`, `learned`, ...), `predictor` (checkpoint
path), `gain`, `weights` and `threads` (for `learned`), `base_kwargs`, `K`.

`ScoringPredictor` adapts `F` to `wrappers.PredictionScoring` (`predictor(obs, u) -> rows`);
`ScoringPredictor.wrap(env)` builds the scoring wrapper with the slot ids tracked from
`info["slot_ids"]`, so the background slot stays out of `F`'s slot set.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np

from ...perception.scene_model import SceneState
from .codec import MM_PER_WINDOW_UNIT, SlotFrame
from .executor import DIM_INDEX
from .model import ACTION_W, CONTEXT_W, ScenePredictor, context_from_obs

__all__ = [
    "PredictiveSceneModel", "SOURCE_NAME", "ScoringPredictor", "SlotIdTracker", "build", "gains",
    "register_source", "sigma_f_from_validation", "sigma_obs_static",
]

SOURCE_NAME = "predictive"


# --------------------------------------------------------------------------------- gains ---


def gains(sigma_f, sigma_obs) -> np.ndarray:
    """`kappa = sigma_F^2 / (sigma_F^2 + sigma_obs^2)` per component; 1 where both are 0."""
    sf = np.asarray(sigma_f, dtype=np.float64) ** 2
    so = np.asarray(sigma_obs, dtype=np.float64) ** 2
    den = sf + so
    return np.where(den > 0.0, sf / np.where(den > 0.0, den, 1.0), 1.0)


def sigma_f_from_validation(rmse_one_step, sigma_obs) -> np.ndarray:
    """`sigma_F` per component from the validation one-step RMSE against observed states:
    `sqrt(max(rmse^2 - sigma_obs^2, 0))` (the observed target carries `sigma_obs`)."""
    r = np.asarray(rmse_one_step, dtype=np.float64)
    so = np.asarray(sigma_obs, dtype=np.float64)
    return np.sqrt(np.maximum(r * r - so * so, 0.0))


def sigma_obs_static(frames: Sequence[SlotFrame], layout) -> np.ndarray:
    """`[D]` observation noise per dynamic component from consecutive frames of a static scene:
    std of the one-tick change over objects paired by slot id, divided by sqrt(2)."""
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        da = layout.dyn(a.rows.astype(np.float64), a.depth.astype(np.float64))
        db = layout.dyn(b.rows.astype(np.float64), b.depth.astype(np.float64))
        for i in np.nonzero(a.objects)[0]:
            j = np.nonzero(b.objects & (b.slot_ids == a.slot_ids[i]))[0]
            if j.size:
                diffs.append(db[j[0]] - da[i])
    if not diffs:
        raise ValueError("no object persists across two consecutive frames")
    return np.std(np.asarray(diffs), axis=0) / math.sqrt(2.0)


# ------------------------------------------------------------------------------- the model ---


class PredictiveSceneModel:
    """Scene-model source with `F` inside: predict from the motion, update with the observation."""

    def __init__(self, source, predictor: ScenePredictor, *, gain=1.0,
                 mm_scale: float = MM_PER_WINDOW_UNIT) -> None:
        self.source = source
        self.F = predictor
        D = predictor.layout.D
        g = np.asarray(gain, dtype=np.float64)
        self.gain = np.broadcast_to(g, (D,)).copy()
        """`[D]` kappa per dynamic component."""
        self.mm_scale = float(mm_scale)
        schema = getattr(source, "schema", None)
        if schema is not None:
            self.schema = schema
        self.ticks = 0
        self._filtered: SlotFrame | None = None
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self.last_prediction: SlotFrame | None = None
        """`z^` of the last tick, in the previous filtered state's slot order and ids."""
        self.last_observation: SlotFrame | None = None
        self.last_filtered: SlotFrame | None = None
        self.innovation: np.ndarray | None = None
        """`[K, D]` `z_obs - z^` in the observation's slot order; NaN where unpaired."""
        self.innovation_mm = float("nan")
        """RMS over paired objects of the centre innovation, times `mm_scale` [mm-eq]."""

    def reset(self) -> None:
        """Clears the filter and the wrapped source's slot identity."""
        reset = getattr(self.source, "reset", None)
        if callable(reset):
            reset()
        self.ticks = 0
        self._filtered = self._pending = None
        self.last_prediction = self.last_observation = self.last_filtered = None
        self.innovation = None
        self.innovation_mm = float("nan")

    def set_motion(self, context, action) -> None:
        """Motion of the coming tick: context `[43]` and executed tool-target change `[4]`."""
        c = np.asarray(context, dtype=np.float32).reshape(-1)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if c.shape != (CONTEXT_W,) or a.shape != (ACTION_W,):
            raise ValueError(f"context must be [{CONTEXT_W}] and action [{ACTION_W}]; "
                             f"got {c.shape}, {a.shape}")
        self._pending = (c, a)

    def observe(self, rgb, *, window, instance=None, depth=None, present_mask=None, t=None,
                **kwargs) -> SceneState:
        """The wrapped source's observation, filtered (`SceneModelSource.observe`)."""
        state = self.source.observe(rgb, window=window, instance=instance, depth=depth,
                                    present_mask=present_mask, t=t, **kwargs)
        filt = self.update(SlotFrame.from_state(state))
        return SceneState(slots=filt.rows, depth=filt.depth, schema=state.schema, t=state.t,
                          source=f"{SOURCE_NAME}/{state.source}", window=state.window,
                          slot_ids=state.slot_ids, frame_id=state.frame_id,
                          _decode_fn=state._decode_fn)

    def update(self, obs: SlotFrame) -> SlotFrame:
        """One filter step on slot frames: predict from the pending motion, then correct."""
        L = self.F.layout
        if obs.W != L.W:
            raise ValueError(f"observation is W={obs.W}; the predictor is for W={L.W}")
        self.last_observation = obs
        K, D = obs.K, L.D
        innov = np.full((K, D), np.nan)
        if self._pending is None or self._filtered is None:
            self.last_prediction = None
            filt = obs
        else:
            ctx, a = self._pending
            pred = self.F.predict(self._filtered, ctx, a)
            self.last_prediction = pred
            d_obs = L.dyn(obs.rows.astype(np.float64), obs.depth.astype(np.float64))
            d_pred = L.dyn(pred.rows.astype(np.float64), pred.depth.astype(np.float64))
            rows = obs.rows.astype(np.float64).copy()
            depth = obs.depth.astype(np.float64).copy()
            for j in np.nonzero(obs.objects)[0]:
                i = np.nonzero(pred.objects & (pred.slot_ids == obs.slot_ids[j]))[0]
                if not i.size:
                    continue
                innov[j] = d_obs[j] - d_pred[i[0]]
                r, d = L.with_dyn(rows[j], d_pred[i[0]] + self.gain * innov[j])
                rows[j], depth[j] = r, float(d)
            filt = SlotFrame(rows.astype(np.float32), depth.astype(np.float32),
                             obs.valid.copy(), obs.slot_ids.copy())
        self.innovation = innov
        ci = list(L.centre)
        paired = np.all(np.isfinite(innov), axis=1)
        if ci and paired.any():
            e = np.sum(innov[paired][:, ci] ** 2, axis=1)
            self.innovation_mm = self.mm_scale * float(math.sqrt(np.mean(e)))
        else:
            self.innovation_mm = float("nan")
        self._filtered = filt
        self.last_filtered = filt
        self._pending = None
        self.ticks += 1
        return filt


# ---------------------------------------------------------------------------- registration ---


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(base: str = "ideal", *, predictor=None, gain=1.0, base_kwargs=None, weights=None,
          threads=None, K: int = 8, mm_scale: float = MM_PER_WINDOW_UNIT) -> PredictiveSceneModel:
    """`PredictiveSceneModel` over the registered source `base` with the checkpoint `predictor`.

    `learned`: `weights` is the checkpoint path (explicit); the predictor must have been trained
    on the same source and weights (sha256), otherwise loading is refused."""
    from ...perception import sources

    if predictor is None:
        raise ValueError("predictor: the path of a trained F checkpoint is required")
    kw = dict(base_kwargs or {})
    sha = ""
    if base == "learned":
        if weights is None:
            from ...perception.cnn.source import DEFAULT_WEIGHTS

            weights = DEFAULT_WEIGHTS
        weights = str(Path(weights).resolve())
        sha = _sha256(weights)
        if threads is not None:
            import torch

            torch.set_num_threads(int(threads))
        kw.setdefault("checkpoint", weights)
        kw.setdefault("device", "cpu")
        kw.setdefault("K", K)
    elif base == "ideal":
        kw.setdefault("K", K)
    elif weights is not None:
        sha = _sha256(weights)
    F = ScenePredictor.load(predictor, source=base, weights_sha256=sha)
    src = sources.get(base)(**kw)
    return PredictiveSceneModel(src, F, gain=gain, mm_scale=mm_scale)


def register_source(name: str = SOURCE_NAME) -> None:
    """Registers `build` in `perception.sources` under `name`."""
    from ...perception import sources

    sources.register(name, lambda **kw: build(**kw), replace=True)


register_source()


# ------------------------------------------------------------------------------ scoring ---


class ScoringPredictor:
    """`F` as `wrappers.PredictionScoring`'s `predictor(obs, u) -> rows [K, W]`.

    Action: the first feed-forward row (an advancing tick) plus `scale * u` on the stage's
    dimensions; the offset ceiling is not applied.

    Slot ids are not in the actor observation. They come from `slot_ids` (the observation's
    `info["slot_ids"]`), kept current by the wrapper `wrap(env)` builds; without them the call
    is refused, because treating the background slot as an object would put it in `F`'s slot
    set."""

    def __init__(self, model: ScenePredictor, *, dims: Sequence[str], scale: Sequence[float],
                 compliance=(1.0, 1.0)) -> None:
        if len(dims) != len(scale):
            raise ValueError("dims and scale differ in length")
        self.model = model
        self.dims = tuple(dims)
        self.scale = tuple(float(s) for s in scale)
        self.compliance = tuple(float(c) for c in compliance)
        self.slot_ids: np.ndarray | None = None
        """`[K]` slot ids of the observation the next call predicts from."""

    def wrap(self, env):
        """`PredictionScoring(env, self)`, with `slot_ids` set from every `reset` and `step`."""
        from .wrappers import PredictionScoring

        return PredictionScoring(SlotIdTracker(env, self), self)

    def action(self, obs: dict, u) -> np.ndarray:
        """`[4]` expected executed tool-target change [m, rad]."""
        u = np.clip(np.asarray(u, dtype=np.float64).reshape(-1), -1.0, 1.0)
        a = np.asarray(obs["actor"]["feedforward"], dtype=np.float64)[0, :ACTION_W].copy()
        for i, name in enumerate(self.dims):
            a[DIM_INDEX[name]] += self.scale[i] * float(u[i])
        return a

    def frame(self, obs: dict, slot_ids=None) -> SlotFrame:
        """The observation's slot frame with its slot ids (`slot_ids`, else the tracked ones)."""
        actor = obs["actor"]
        ids = self.slot_ids if slot_ids is None else slot_ids
        if ids is None:
            raise ValueError("ScoringPredictor needs the observation's slot ids "
                             "(info['slot_ids']): use ScoringPredictor.wrap(env) or pass slot_ids")
        valid = np.asarray(actor["z_valid"]) > 0.5
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.shape != valid.shape:
            raise ValueError(f"slot ids must be [{valid.shape[0]}]; got {ids.shape}")
        return SlotFrame(np.asarray(actor["z"], dtype=np.float32),
                         np.asarray(actor["z_depth"], dtype=np.float32), valid, ids)

    def __call__(self, obs: dict, u, slot_ids=None) -> np.ndarray:
        ctx = context_from_obs(obs["actor"], self.compliance)
        return self.model.predict(self.frame(obs, slot_ids), ctx, self.action(obs, u)).rows


class SlotIdTracker(gym.Wrapper):
    """Copies `info["slot_ids"]` of every `reset` and `step` to `receiver.slot_ids`."""

    def __init__(self, env: gym.Env, receiver) -> None:
        super().__init__(env)
        self.receiver = receiver

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.receiver.slot_ids = np.asarray(info["slot_ids"]).copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.receiver.slot_ids = np.asarray(info["slot_ids"]).copy()
        return obs, reward, terminated, truncated, info
