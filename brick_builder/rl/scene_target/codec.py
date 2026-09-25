"""Scene-state codec: slot frames, appearance matching, scene-state distance [mm-eq].

One definition serves the reward, the success test, the progress tube, the predictor's loss
weights, the model-predictive cost and the evaluation.

**Slot frame.** `SlotFrame` is one tick of slot state as plain arrays: `rows [K, W]`,
`depth [K]`, `valid [K]`, `slot_ids [K]`. The background slot is the valid slot with a negative
slot id (every source in `brick_builder.perception` marks it that way) and is never matched.

**Fields.** Under the named schema (`PLACEHOLDER_SLOT_SCHEMA`, W = 14) the dynamic fields are
`centre` (2), `mass` (1), `moments` (3), `orient4` (2) and the per-slot depth (1); the appearance
fields are `colour` (3) and `colour_spread` (1); `border`, `age` and the valid flag are flags.
Under an opaque schema (`opaque_schema(W)`) every feature is both dynamic and appearance, unless
the schema declares fields named `dynamic` or `match`.

**Scales.** Named schema: fixed physical scales, never data statistics -- centre 0.01 window
units (about 0.87 mm), mass 0.002, moments 0.002, orient4 0.2, depth 0.25. Opaque schema: the
per-feature standard deviation of one-tick changes from a calibration run
(`Calibration.feature_std`).

**Matching.** Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on an appearance
cost: colour, log mass and second-moment shape, plus a small centre term that breaks ties
between objects of identical appearance. Pairs costing more than 1.5 stay unmatched.

**Distance, named schema.**

    D_abs = sqrt( sum_i w_i e_i / sum_i w_i ),  e_i = |d centre_i|^2 + 0.25 sum_f (s_c d f_i / s_f)^2,
            w_i = 1 - border_i
    D_rel = RMS over matched reference slots j of |(c_held - c_j) - (c*_held - c*_j)|,
            only when a held brick is named and matched
    D_mm  = 87.4 (D_abs + D_rel) + 2.0 n_missing

87.4 mm per window unit is the scale at the align hover, so the unit is mm-eq. `n_missing` counts
the valid, non-background target slots with no match.

**Distance, opaque schema.** Whitened matched RMS plus the RMS of whitened pairwise differences
(the relative term), times `Calibration.mm_scale`, plus the same missing-slot term.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SlotSchema, opaque_schema

__all__ = [
    "BORDER_REF_MAX", "CENTRE_SCALE", "Calibration", "DistanceResult", "MATCH_MAX_COST",
    "MISSING_MM", "MM_PER_WINDOW_UNIT", "NAMED_SCALES", "SceneDistance", "SlotFrame",
    "appearance_cost", "calibration_path", "fit_feature_std", "fit_mm_scale", "reference_mask",
    "valid_mask",
]

MM_PER_WINDOW_UNIT = 87.4
"""Millimetres of tool motion per window unit of slot-centre motion at the align hover."""

MISSING_MM = 2.0
"""Penalty per unmatched target slot [mm-eq]."""

MATCH_MAX_COST = 1.5
"""Appearance cost above which a pair is left unmatched."""

CENTRE_SCALE = 0.01
"""s_c: the centre scale [window units], about 0.87 mm-eq."""

NAMED_SCALES = {"centre": CENTRE_SCALE, "mass": 0.002, "moments": 0.002, "orient4": 0.2,
                "depth": 0.25}
"""Fixed physical scales of the named schema's dynamic fields."""

OTHER_WEIGHT = 0.25
"""Weight of the non-centre dynamic fields in `e_i`."""

BORDER_REF_MAX = 0.25
"""Largest border fraction of a slot that counts as a reference (fully in view)."""

REF_MASS_RANGE = (0.0012, 0.5)
"""Mass range of a reference slot: large enough to measure, smaller than the table-and-arm blob."""

COLOUR_SCALE = 0.3
LOG_MASS_SCALE = 1.0
SHAPE_SCALE = 0.5
TIE_BREAK_CENTRE = 0.05
"""Appearance-cost scales (colour in `2 rgb - 1` units, natural log of mass, normalized
second-moment shape) and the weight of the centre tie-break term [per window unit]."""


# ----------------------------------------------------------------------------- slot frames ---


def valid_mask(rows: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """`[K]` non-empty flag: a slot is empty when its row is all zeros and its depth is 0.5."""
    rows = np.asarray(rows)
    depth = np.asarray(depth)
    return ~((depth == 0.5) & np.all(rows == 0, axis=-1))


@dataclass(frozen=True)
class SlotFrame:
    """One tick of slot state as arrays."""

    rows: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    slot_ids: np.ndarray

    @classmethod
    def from_arrays(cls, rows, depth, valid=None, slot_ids=None) -> "SlotFrame":
        rows = np.asarray(rows, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if rows.ndim != 2 or depth.shape != (rows.shape[0],):
            raise ValueError(f"rows must be [K, W] and depth [K]; got {rows.shape}, {depth.shape}")
        valid = valid_mask(rows, depth) if valid is None else np.asarray(valid, dtype=bool)
        if slot_ids is None:
            slot_ids = np.where(valid, np.arange(rows.shape[0]), -1)
        return cls(rows=rows, depth=depth, valid=valid,
                   slot_ids=np.asarray(slot_ids, dtype=np.int64))

    @classmethod
    def from_state(cls, state) -> "SlotFrame":
        """From a `perception.scene_model.SceneState`."""
        return cls.from_arrays(state.slots, state.depth, None, state.slot_ids)

    @property
    def K(self) -> int:
        return int(self.rows.shape[0])

    @property
    def W(self) -> int:
        return int(self.rows.shape[1])

    @property
    def background(self) -> np.ndarray:
        """`[K]`: the valid slot(s) with a negative slot id."""
        return self.valid & (self.slot_ids < 0)

    @property
    def objects(self) -> np.ndarray:
        """`[K]`: valid, non-background slots -- the matchable ones."""
        return self.valid & (self.slot_ids >= 0)

    def permuted(self, order: Sequence[int]) -> "SlotFrame":
        order = np.asarray(order, dtype=np.int64)
        return SlotFrame(self.rows[order], self.depth[order], self.valid[order],
                         self.slot_ids[order])


def _as_frame(x) -> SlotFrame:
    if isinstance(x, SlotFrame):
        return x
    if hasattr(x, "slots") and hasattr(x, "schema"):
        return SlotFrame.from_state(x)
    raise TypeError(f"expected a SlotFrame or a SceneState; got {type(x).__name__}")


# ---------------------------------------------------------------------------- calibration ---


@dataclass
class Calibration:
    """Per-source calibration: opaque-schema scales, the mm scale and the observation noise."""

    source: str
    weights_sha256: str
    W: int
    feature_std: list[float] = field(default_factory=list)
    """`[W]` std of one-tick changes per feature (opaque schema); empty = unit scales."""
    depth_std: float = 0.25
    mm_scale: float = MM_PER_WINDOW_UNIT
    """mm-eq per whitened unit (opaque) or per window unit (named)."""
    sigma_obs_mm: float | None = None
    """Std of D between consecutive frames of a static arm [mm-eq]."""
    d_floor_mm: float | None = None
    """D at the end of open-loop replay in the nominal world [mm-eq]."""
    n_ticks: int = 0
    notes: str = ""

    @property
    def tol_mm(self) -> float:
        """`max(0.5, 2 sigma_obs)` [mm-eq]; 0.5 when `sigma_obs_mm` is unknown."""
        if self.sigma_obs_mm is None or not math.isfinite(self.sigma_obs_mm):
            return 0.5
        return max(0.5, 2.0 * float(self.sigma_obs_mm))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Calibration":
        data = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path, *, source: str | None = None, weights_sha256: str | None = None,
             W: int | None = None) -> "Calibration":
        """Loads and refuses a file recorded for a different source, weights or width."""
        cal = cls.from_json(Path(path).read_text())
        for name, want, got in (("source", source, cal.source),
                                ("weights_sha256", weights_sha256, cal.weights_sha256),
                                ("W", W, cal.W)):
            if want is not None and want != got:
                raise ValueError(f"calibration {path} is for {name}={got!r}, not {want!r}")
        return cal


def calibration_path(root, source: str, weights_sha256: str) -> Path:
    """`<root>/<source>_<sha8>.json`; `sha8` is `none` for a source without weights."""
    sha8 = weights_sha256[:8] if weights_sha256 else "none"
    return Path(root) / f"{source}_{sha8}.json"


def fit_feature_std(frames: Sequence[SlotFrame], *, floor: float = 1e-6) -> tuple[np.ndarray, float]:
    """Per-feature std of one-tick changes over consecutive frames, slots paired by slot id.

    Returns `(feature_std [W], depth_std)`. The background slot is excluded.
    """
    diffs, ddiffs = [], []
    for a, b in zip(frames[:-1], frames[1:]):
        a, b = _as_frame(a), _as_frame(b)
        for i in np.nonzero(a.objects)[0]:
            j = np.nonzero(b.objects & (b.slot_ids == a.slot_ids[i]))[0]
            if j.size:
                diffs.append(b.rows[j[0]].astype(np.float64) - a.rows[i])
                ddiffs.append(float(b.depth[j[0]]) - float(a.depth[i]))
    if not diffs:
        raise ValueError("no slot persists across two consecutive frames")
    std = np.maximum(np.std(np.asarray(diffs), axis=0), floor)
    dstd = max(float(np.std(ddiffs)), floor)
    return std, dstd


def fit_mm_scale(d_units: Sequence[float], truth_mm: Sequence[float]) -> float:
    """Least-squares scale through the origin: `truth_mm ~ scale * d_units`."""
    x = np.asarray(d_units, dtype=np.float64)
    y = np.asarray(truth_mm, dtype=np.float64)
    den = float(np.dot(x, x))
    if den <= 0.0:
        raise ValueError("all distances are zero")
    return float(np.dot(x, y) / den)


# ------------------------------------------------------------------------------- matching ---


def _named_shape(rows: np.ndarray, schema: SlotSchema) -> np.ndarray:
    """`[K, 2]` normalized second-moment shape `((sxx - syy), 2 sxy) / (sxx + syy)`."""
    m = schema.get(rows, "moments").astype(np.float64)
    tr = m[:, 0] + m[:, 2]
    safe = np.where(tr > 1e-12, tr, 1.0)
    shape = np.stack([(m[:, 0] - m[:, 2]) / safe, 2.0 * m[:, 1] / safe], axis=1)
    shape[tr <= 1e-12] = 0.0
    return shape


def appearance_cost(obs: SlotFrame, target: SlotFrame, schema: SlotSchema,
                    feature_std: np.ndarray | None = None) -> np.ndarray:
    """`[K_obs, K_target]` appearance cost. Background and invalid slots cost `inf`."""
    obs, target = _as_frame(obs), _as_frame(target)
    if schema.has("colour") and schema.has("mass") and schema.has("moments"):
        ca = schema.get(obs.rows, "colour").astype(np.float64)
        cb = schema.get(target.rows, "colour").astype(np.float64)
        la = np.log(np.maximum(schema.get(obs.rows, "mass")[:, 0].astype(np.float64), 1e-6))
        lb = np.log(np.maximum(schema.get(target.rows, "mass")[:, 0].astype(np.float64), 1e-6))
        sa, sb = _named_shape(obs.rows, schema), _named_shape(target.rows, schema)
        dc = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1) / COLOUR_SCALE
        dl = np.abs(la[:, None] - lb[None, :]) / LOG_MASS_SCALE
        ds = np.linalg.norm(sa[:, None, :] - sb[None, :, :], axis=-1) / SHAPE_SCALE
        cost = np.sqrt((dc ** 2 + dl ** 2 + ds ** 2) / 3.0)
        if schema.has("centre"):
            pa = schema.get(obs.rows, "centre").astype(np.float64)
            pb = schema.get(target.rows, "centre").astype(np.float64)
            cost = cost + TIE_BREAK_CENTRE * np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    else:
        cols = _columns(schema, "match")
        std = np.ones(obs.W) if feature_std is None or len(feature_std) == 0 else np.asarray(feature_std)
        a = obs.rows[:, cols].astype(np.float64) / std[cols]
        b = target.rows[:, cols].astype(np.float64) / std[cols]
        cost = np.sqrt(np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))
    allowed = obs.objects[:, None] & target.objects[None, :]
    return np.where(allowed, cost, np.inf)


def _columns(schema: SlotSchema, name: str) -> np.ndarray:
    """Column indices of a declared field, or every column when the schema does not declare it."""
    if schema.has(name):
        return np.arange(schema.W)[schema.fields[name]]
    return np.arange(schema.W)


def reference_mask(frame, schema: SlotSchema = PLACEHOLDER_SLOT_SCHEMA) -> np.ndarray:
    """`[K]` reference slots: matchable, fully in view, and of a measurable size.

    Named schema: border fraction at most `BORDER_REF_MAX` and mass inside `REF_MASS_RANGE`.
    Opaque schema: every matchable slot.
    """
    frame = _as_frame(frame)
    out = frame.objects.copy()
    if schema.has("border"):
        out &= schema.get(frame.rows, "border")[:, 0] <= BORDER_REF_MAX
    if schema.has("mass"):
        m = schema.get(frame.rows, "mass")[:, 0]
        out &= (m >= REF_MASS_RANGE[0]) & (m <= REF_MASS_RANGE[1])
    return out


# ------------------------------------------------------------------------------- distance ---


@dataclass(frozen=True)
class DistanceResult:
    """One scene-state distance and the assignment it was computed on."""

    d_mm: float
    parts: dict
    """`abs`, `rel`, `missing` [mm-eq]; they sum to `d_mm`."""
    perm: np.ndarray
    """`[K]`: for each target slot, the matched observed slot, or -1."""
    matched: np.ndarray
    """`[K]` bool over target slots."""
    n_missing: int


class SceneDistance:
    """Matching and distance for one slot schema (the `SceneDistance` protocol)."""

    def __init__(self, schema: SlotSchema = PLACEHOLDER_SLOT_SCHEMA,
                 calibration: Calibration | None = None, *,
                 max_cost: float = MATCH_MAX_COST, missing_mm: float = MISSING_MM) -> None:
        self.schema = schema
        self.calibration = calibration
        self.max_cost = float(max_cost)
        self.missing_mm = float(missing_mm)
        self.named = schema.has("centre")
        if self.named:
            self.mm_scale = (MM_PER_WINDOW_UNIT if calibration is None
                             else float(calibration.mm_scale))
            other = [n for n in ("mass", "moments", "orient4") if schema.has(n)]
            cols, scales = [], []
            for n in other:
                idx = np.arange(schema.W)[schema.fields[n]]
                cols.extend(idx.tolist())
                scales.extend([NAMED_SCALES[n]] * idx.size)
            self._other_cols = np.asarray(cols, dtype=np.int64)
            self._other_scale = np.asarray(scales, dtype=np.float64)
            self._centre_cols = np.arange(schema.W)[schema.fields["centre"]]
            self._feature_std = None
        else:
            if calibration is not None and calibration.W != schema.W:
                raise ValueError(f"calibration W={calibration.W} does not match schema W={schema.W}")
            std = (np.ones(schema.W) if calibration is None or not calibration.feature_std
                   else np.asarray(calibration.feature_std, dtype=np.float64))
            if std.shape != (schema.W,):
                raise ValueError(f"feature_std must be [{schema.W}]; got {std.shape}")
            self._feature_std = std
            self._depth_std = 0.25 if calibration is None else float(calibration.depth_std)
            self._dyn_cols = _columns(schema, "dynamic")
            self.mm_scale = 1.0 if calibration is None else float(calibration.mm_scale)

    @classmethod
    def opaque(cls, W: int, calibration: Calibration | None = None, **kw) -> "SceneDistance":
        return cls(opaque_schema(W), calibration, **kw)

    # -- matching ------------------------------------------------------------------------

    def match(self, obs, target) -> np.ndarray:
        """`[K_target]`: the observed slot matched to each target slot, or -1."""
        obs, target = _as_frame(obs), _as_frame(target)
        cost = appearance_cost(obs, target, self.schema, self._feature_std)
        perm = np.full(target.K, -1, dtype=np.int64)
        finite = np.isfinite(cost)
        if not finite.any():
            return perm
        big = 1e6
        r, c = linear_sum_assignment(np.where(finite, cost, big))
        for i, j in zip(r, c):
            if finite[i, j] and cost[i, j] <= self.max_cost:
                perm[j] = i
        return perm

    def align(self, obs, target, perm: np.ndarray | None = None):
        """Observed rows in the order of `target`: `(rows [K, W], depth [K], matched [K])`.

        Unmatched target slots get a zero row and depth 0.5, the empty-slot convention."""
        obs, target = _as_frame(obs), _as_frame(target)
        perm = self.match(obs, target) if perm is None else np.asarray(perm)
        rows = np.zeros((target.K, obs.W), dtype=np.float32)
        depth = np.full(target.K, 0.5, dtype=np.float32)
        ok = perm >= 0
        rows[ok] = obs.rows[perm[ok]]
        depth[ok] = obs.depth[perm[ok]]
        return rows, depth, ok

    # -- distance --------------------------------------------------------------------------

    def __call__(self, obs, target, *, perm: np.ndarray | None = None,
                 held: int | None = None) -> DistanceResult:
        """`D(obs, target)`. `perm` reuses an assignment; `held` is the target slot of the
        brick in the hand (named schema only)."""
        obs, target = _as_frame(obs), _as_frame(target)
        if obs.W != self.schema.W or target.W != self.schema.W:
            raise ValueError(f"rows are W={obs.W}/{target.W}; schema is W={self.schema.W}")
        perm = self.match(obs, target) if perm is None else np.asarray(perm, dtype=np.int64)
        matched = perm >= 0
        n_missing = int(np.sum(target.objects & ~matched))
        if self.named:
            d_abs, d_rel = self._named(obs, target, perm, held)
        else:
            d_abs, d_rel = self._opaque(obs, target, perm)
        parts = {"abs": self.mm_scale * d_abs, "rel": self.mm_scale * d_rel,
                 "missing": self.missing_mm * n_missing}
        d_mm = parts["abs"] + parts["rel"] + parts["missing"]
        return DistanceResult(d_mm=float(d_mm), parts={k: float(v) for k, v in parts.items()},
                              perm=perm, matched=matched, n_missing=n_missing)

    def _weights(self, obs: SlotFrame, target: SlotFrame, perm: np.ndarray) -> np.ndarray:
        """`[K_target]` w = 1 - border of the more cut-off side of each matched pair."""
        w = np.zeros(target.K)
        ok = perm >= 0
        if not self.schema.has("border"):
            w[ok] = 1.0
            return w
        bt = self.schema.get(target.rows, "border")[:, 0].astype(np.float64)
        bo = np.zeros(target.K)
        bo[ok] = self.schema.get(obs.rows[perm[ok]], "border")[:, 0]
        w[ok] = np.clip(1.0 - np.maximum(bt[ok], bo[ok]), 0.0, 1.0)
        return w

    def _named(self, obs, target, perm, held):
        ok = perm >= 0
        if not ok.any():
            return 0.0, 0.0
        j = np.nonzero(ok)[0]
        i = perm[j]
        co = obs.rows[i][:, self._centre_cols].astype(np.float64)
        ct = target.rows[j][:, self._centre_cols].astype(np.float64)
        e = np.sum((co - ct) ** 2, axis=1)
        if self._other_cols.size:
            df = (obs.rows[i][:, self._other_cols].astype(np.float64)
                  - target.rows[j][:, self._other_cols]) * (CENTRE_SCALE / self._other_scale)
            e = e + OTHER_WEIGHT * np.sum(df ** 2, axis=1)
        dd = (obs.depth[i].astype(np.float64) - target.depth[j]) * (CENTRE_SCALE / NAMED_SCALES["depth"])
        e = e + OTHER_WEIGHT * dd ** 2
        w = self._weights(obs, target, perm)[j]
        d_abs = float(math.sqrt(np.sum(w * e) / np.sum(w))) if np.sum(w) > 0 else 0.0

        d_rel = 0.0
        if held is not None and 0 <= held < target.K and perm[held] >= 0:
            refs = [(jj, ii, ww) for jj, ii, ww in zip(j, i, w) if jj != held and ww > 0]
            if refs:
                h_o = obs.rows[perm[held], self._centre_cols].astype(np.float64)
                h_t = target.rows[held, self._centre_cols].astype(np.float64)
                num = den = 0.0
                for jj, ii, ww in refs:
                    r = (h_o - obs.rows[ii, self._centre_cols]) - (h_t - target.rows[jj, self._centre_cols])
                    num += ww * float(np.dot(r, r))
                    den += ww
                d_rel = math.sqrt(num / den)
        return d_abs, d_rel

    def _opaque(self, obs, target, perm):
        ok = perm >= 0
        if not ok.any():
            return 0.0, 0.0
        j = np.nonzero(ok)[0]
        i = perm[j]
        cols = self._dyn_cols
        std = self._feature_std[cols]
        zo = np.concatenate([obs.rows[i][:, cols] / std,
                             (obs.depth[i] / self._depth_std)[:, None]], axis=1).astype(np.float64)
        zt = np.concatenate([target.rows[j][:, cols] / std,
                             (target.depth[j] / self._depth_std)[:, None]], axis=1).astype(np.float64)
        w = self._weights(obs, target, perm)[j]
        e = np.mean((zo - zt) ** 2, axis=1)
        d_abs = float(math.sqrt(np.sum(w * e) / np.sum(w))) if np.sum(w) > 0 else 0.0
        d_rel = 0.0
        n = len(j)
        if n >= 2:
            acc, cnt = 0.0, 0
            for a in range(n):
                for b in range(a + 1, n):
                    r = (zo[a] - zo[b]) - (zt[a] - zt[b])
                    acc += float(np.mean(r ** 2))
                    cnt += 1
            d_rel = math.sqrt(acc / cnt)
        return d_abs, d_rel
