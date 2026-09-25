"""Logged episodes -> training tensors for the predictor and the inverse.

Every pair this module builds stays inside one episode and one step: a step boundary changes
which objects the loop lines itself up on, so a slate row at tick `i` and the same row at tick
`i + j` are only the same object while the step is the same.

Three sets come out of the same log.

- **Predictor.** `(slate_i, mech_i, action_i) -> slate_{i+1} - slate_i`, where `action_i` is the
  tool target's own change over the tick, `cmd_{i+1} - cmd_i`, which is what the world saw.
- **Inverse, hindsight.** `(slate_i - slate_{i+j}, mech_i) -> the correction increment commanded
  at tick i`, with `j` drawn in 1..8. The correction increment is read back from the log as
  `corr_{i+1} - corr_i`, so it includes the probe as well as the loop's own answer -- both were
  commanded, and both are in the target state `j` ticks later.
- **Inverse, teacher.** The same inputs against the target *reference frame* the controller was
  tracking, with the analytic inverse's own (unsaturated) answer as the label. Used for the
  held-out action-error metric, and as the label for the DAgger-style remedy.

`corr` is recovered as `cmd - ff` rather than logged twice: the commanded tool target is the
recorded one plus the correction, by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.features import (
    ACTION_W, INVERSE_IN, INVERSE_OUT, N_SLATE, ROW_W, encode_inverse, encode_predictor,
)

CENTRE = slice(1, 3)
"""Where the two centre coordinates sit inside one slate row's eight fields."""

MAX_HINDSIGHT = 8


@dataclass
class Episodes:
    """Every logged tick of a set of episodes, with the episode each one belongs to."""

    arrays: dict
    episode: np.ndarray
    """`[n]` int: which episode each tick came from."""

    @property
    def n(self) -> int:
        return int(self.episode.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def flag(self, key: str) -> np.ndarray:
        """A logged gate as a boolean mask, whatever dtype the archive stored it as."""
        return np.asarray(self.arrays[key]).astype(bool)

    def corrections(self) -> np.ndarray:
        """`[n, 3]`: the correction in force at each tick, in (x, y, spin)."""
        d = self.arrays["cmd"] - self.arrays["ff"]
        return np.stack([d[:, 0], d[:, 1], d[:, 3]], axis=1).astype(np.float64)


def load(paths, limit: int | None = None) -> Episodes:
    """Read every `ep_*.npz` under each of `paths`."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.glob("ep_*.npz")) if p.is_dir() else [p])
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no logged episodes under {[str(p) for p in paths]}")

    per_tick = ("obs_rows", "obs_valid", "obs_weight", "tgt_rows", "tgt_valid", "slot_ids",
                "mech", "cmd", "ff", "analytic", "delta", "measured", "err_mm", "k",
                "step_index", "dwelling", "trusted", "frozen", "in_hand", "seen", "t", "step")
    chunks: dict[str, list] = {key: [] for key in per_tick}
    episode: list[np.ndarray] = []
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            n = int(data["k"].shape[0])
            if n == 0:
                continue
            for key in per_tick:
                chunks[key].append(data[key])
            episode.append(np.full(n, index, dtype=np.int64))
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    return Episodes(arrays=arrays, episode=np.concatenate(episode))


# ------------------------------------------------------------------------------------ pairing ---


def _same(data: Episodes, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Which index pairs stay inside one episode and one step."""
    return ((data.episode[i] == data.episode[j])
            & (data["step_index"][i] == data["step_index"][j]))


def predictor_pairs(data: Episodes) -> dict:
    """`x [m, PREDICTOR_IN]`, `y [m, N_SLATE * ROW_W]`, `w [m, N_SLATE * ROW_W]`."""
    i = np.arange(data.n - 1)
    j = i + 1
    keep = _same(data, i, j)
    i, j = i[keep], j[keep]

    obs, val, wgt = data["obs_rows"], data["obs_valid"], data["obs_weight"]
    action = (data["cmd"][j] - data["cmd"][i]).astype(np.float64)[:, :ACTION_W]
    x = np.stack([encode_predictor(obs[a], val[a], wgt[a], data["mech"][a], action[t],
                                   data["in_hand"][a])
                  for t, a in enumerate(i)])
    both = (val[i] > 0) & (val[j] > 0)
    y = ((obs[j] - obs[i]) * both[:, :, None]).reshape(len(i), -1)
    w = np.repeat(both.astype(np.float64), ROW_W, axis=1)
    return {"x": x, "y": y, "w": w, "i": i, "j": j, "action": action}


def hindsight_pairs(data: Episodes, rng: np.random.Generator, *, repeats: int = 2,
                    max_j: int = MAX_HINDSIGHT, max_ff_mm: float | None = None,
                    log=print) -> dict:
    """`x [m, INVERSE_IN]`, `y [m, 3]` -- the correction increment commanded at tick `i`.

    `max_ff_mm` keeps only windows over which the recorded target barely moves, which is what a
    dwell is: there the change of the scene between `i` and `i + j` is the correction's doing and
    not the feed-forward's. Leaving it `None` uses every acting tick.

    Empty, not an error, when nothing in the log produces a usable window -- whether because
    nothing passes the `trusted & ~frozen` gate, or because everything that does is then dropped
    by the same-step or travel filter: `x`/`y` come back with zero rows and `log` gets one line
    saying which case it was, so a round with nothing to learn from skips training instead of
    failing on `np.stack([])`.
    """
    corr = data.corrections()
    obs, val, wgt = data["obs_rows"], data["obs_valid"], data["obs_weight"]
    acting = data.flag("trusted") & ~data.flag("frozen")

    xs, ys, idx, js = [], [], [], []
    for _ in range(repeats):
        i = np.arange(data.n - 1)
        j = i + rng.integers(1, max_j + 1, size=i.shape)
        j = np.minimum(j, data.n - 1)
        keep = _same(data, i, j) & acting[i] & _same(data, i, i + 1)
        if max_ff_mm is not None:
            travel = 1e3 * np.linalg.norm(
                (data["ff"][np.minimum(j, data.n - 1)] - data["ff"][i])[:, :3], axis=1)
            keep &= travel <= max_ff_mm
        i, j = i[keep], j[keep]
        for t, a in enumerate(i):
            b = j[t]
            xs.append(encode_inverse(obs[a], val[a], wgt[a], obs[b], val[b], data["mech"][a],
                                     data["in_hand"][a], float(np.sum(val[a][:-1]))))
            ys.append(corr[a + 1] - corr[a])
        idx.append(i)
        js.append(j)
    if not xs:
        gate = int(acting.sum())
        if gate == 0:
            log(f"hindsight_pairs: 0 of {data.n} logged ticks pass the trusted-and-not-frozen "
                f"gate; no hindsight pairs")
        else:
            log(f"hindsight_pairs: {gate} of {data.n} logged ticks pass the "
                f"trusted-and-not-frozen gate, but same-step and travel filtering dropped every "
                f"window; no hindsight pairs")
        x = np.zeros((0, INVERSE_IN), dtype=np.float64)
        y = np.zeros((0, INVERSE_OUT), dtype=np.float64)
    else:
        x = np.stack(xs)
        y = np.stack(ys).astype(np.float64)
    return {"x": x, "y": y, "i": np.concatenate(idx), "j": np.concatenate(js)}


def teacher_pairs(data: Episodes, *, log=print) -> dict:
    """`x [m, INVERSE_IN]`, `y [m, 3]` -- the analytic inverse's own answer at that tick.

    The target is the reference frame the controller was tracking, which is what the deployed
    inverse is given, so this is the set the held-out action error is measured on. Empty, not an
    error, when nothing passes the gate -- see `hindsight_pairs`.
    """
    obs, val, wgt = data["obs_rows"], data["obs_valid"], data["obs_weight"]
    tgt, tval = data["tgt_rows"], data["tgt_valid"]
    keep = np.nonzero(data.flag("trusted") & ~data.flag("frozen"))[0]
    if keep.size == 0:
        log(f"teacher_pairs: 0 of {data.n} logged ticks pass the trusted-and-not-frozen gate; "
            f"no teacher pairs")
        x = np.zeros((0, INVERSE_IN), dtype=np.float64)
    else:
        x = np.stack([encode_inverse(obs[a], val[a], wgt[a], tgt[a], tval[a], data["mech"][a],
                                     data["in_hand"][a], float(np.sum(val[a][:-1])))
                      for a in keep])
    return {"x": x, "y": data["analytic"][keep].astype(np.float64), "i": keep}


def feed_forward(data: Episodes, i: np.ndarray) -> np.ndarray:
    """`[m, ACTION_W]`: the recorded tool-target change over tick `i`, without any correction."""
    return (data["ff"][i + 1] - data["ff"][i]).astype(np.float64)[:, :ACTION_W]


# ------------------------------------------------------------------------------------- scaling ---


def mm_per_centre_unit(data: Episodes) -> float:
    """Millimetres of tool motion per unit of slot-centre difference, measured on the log.

    The analytic inverse already answers this question every tick: it maps the same centre
    differences through the step's measured Jacobian into metres of tool motion. The median ratio
    of the two norms, over ticks where the difference is large enough to be signal, is the
    conversion the error tables are quoted in.
    """
    obs, val = data["obs_rows"], data["obs_valid"]
    tgt, tval = data["tgt_rows"], data["tgt_valid"]
    both = (val > 0) & (tval > 0)
    diff = (obs[:, :, CENTRE] - tgt[:, :, CENTRE]) * both[:, :, None]
    feature = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    tool_mm = 1e3 * np.linalg.norm(data["measured"][:, :2], axis=1)
    ok = (feature > 1e-3) & (tool_mm > 0.2) & data.flag("trusted")
    if not np.any(ok):
        return 190.0
    return float(np.median(tool_mm[ok] / feature[ok]))


def split(n: int, rng: np.random.Generator, holdout: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Indices of a train/held-out split."""
    order = rng.permutation(n)
    cut = int(round(n * (1.0 - holdout)))
    return order[:cut], order[cut:]


def episode_split(data: Episodes, rng: np.random.Generator, holdout: float = 0.15) -> np.ndarray:
    """`[n]` bool: True for ticks belonging to held-out **episodes**, so a held-out tick is never
    the neighbour of a trained one."""
    ids = np.unique(data.episode)
    held = set(rng.permutation(ids)[: max(1, int(round(len(ids) * holdout)))].tolist())
    return np.array([e in held for e in data.episode], dtype=bool)
