"""Episode logs and multi-step open-loop rollout windows for `F`; synthetic control-affine toy.

**Episode log** (`ep_*.npz`, arrays only, read with `allow_pickle=False`), one row per vision
tick. Row 0 may be the reset observation (its action is then NaN).

| Key | Shape | Contents |
|---|---|---|
| `z`, `depth`, `valid`, `slot_ids` | [N, K, W], [N, K] x 3 | observed slot rows in the source's order |
| `mech` | [N, 18] | joint positions, speeds, loads (servo bus) |
| `tool` | [N, 13] | commanded tool target (5), bus-FK tool pose (4), accumulated offset (4) |
| `exec_action` | [N, 4] | executed tool-target change over the tick that ended at this row |
| `grip`, `compliance` | [N], [N, 2] | grip and (c_vert, c_lat) of that tick's last write |
| `step_onehot` | [N, 9] | optional (else `progress[:, 3:12]`, else zeros) |
| `meta` | JSON | episode id, level, world seed, target key, source, weights sha256, W, controller, round |

Other per-tick arrays of the transition record (`u`, `k`, `d_mm`, `gt_z`, ...) are kept in
`EpisodeLog.extras`.

**Window** `(t, H)`: the frame `z_t` exactly as logged (every slot, background included, with
its validity and slot ids), the rows of the same slot ids at `t+1..t+H` (slot correspondence
from the source's `slot_ids`), contexts `c_t..c_{t+H-1}`, actions `a_t..a_{t+H-1}` and joint
positions `q_{t+1..t+H}`. The context of tick `t -> t+1` is `mech_t`, `tool_t`, `step_t` and
the grip and compliance of row `t+1`; its action is `exec_action[t+1]`. `Windows.frame(i)` is
the first tick as a `SlotFrame`, the same frame `EpisodeLog.frame(t)` gives.

**Split.** Every 7th episode by id (`id % 7 == 6`) is held out of `F` training
(`Windows.split`, `split_logs`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SlotSchema
from .codec import SlotFrame
from .model import (
    ACTION_W, CONTEXT_W, MECH_W, Q_W, STEP_W, TOOL_W, SlotLayout, context_vector,
)

__all__ = [
    "EpisodeLog", "HOLDOUT_EVERY", "ToyConfig", "ToyDynamics", "Windows", "episode_ids",
    "load_episodes", "split_logs", "synthetic_logs",
]

HOLDOUT_EVERY = 7
REQUIRED = ("z", "depth", "valid", "slot_ids", "mech", "tool", "exec_action")
KNOWN = REQUIRED + ("grip", "compliance", "step_onehot", "meta")


# --------------------------------------------------------------------------- episode log ---


@dataclass
class EpisodeLog:
    """One episode, one row per vision tick."""

    z: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    slot_ids: np.ndarray
    mech: np.ndarray
    tool: np.ndarray
    exec_action: np.ndarray
    grip: np.ndarray
    compliance: np.ndarray
    step_onehot: np.ndarray
    meta: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=np.float32)
        N, K, _ = self.z.shape
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.slot_ids = np.asarray(self.slot_ids, dtype=np.int64)
        self.mech = np.asarray(self.mech, dtype=np.float64)
        self.tool = np.asarray(self.tool, dtype=np.float64)
        self.exec_action = np.asarray(self.exec_action, dtype=np.float64)
        self.grip = np.asarray(self.grip, dtype=np.float64).reshape(N)
        self.compliance = np.asarray(self.compliance, dtype=np.float64).reshape(N, 2)
        self.step_onehot = np.asarray(self.step_onehot, dtype=np.float64).reshape(N, STEP_W)
        for name, shape in (("depth", (N, K)), ("valid", (N, K)), ("slot_ids", (N, K)),
                            ("mech", (N, MECH_W)), ("tool", (N, TOOL_W)),
                            ("exec_action", (N, ACTION_W))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must be {shape}; got {getattr(self, name).shape}")

    @property
    def n_ticks(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_transitions(self) -> int:
        return max(self.n_ticks - 1, 0)

    @property
    def episode_id(self) -> int:
        return int(self.meta.get("episode_id", -1))

    def frame(self, t: int) -> SlotFrame:
        """Tick `t` as logged: every slot, background included."""
        return SlotFrame(self.z[t], self.depth[t], self.valid[t], self.slot_ids[t])

    def contexts(self) -> np.ndarray:
        """`[N-1, 43]`: the context of each tick `t -> t+1` (the layout of `context_vector`)."""
        out = np.concatenate([self.mech[:-1], self.tool[:-1], self.step_onehot[:-1],
                              self.compliance[1:], self.grip[1:, None]], axis=1)
        assert out.shape[1] == CONTEXT_W
        return out.astype(np.float32)

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_steps(cls, reset_obs: dict, reset_info: dict, steps: Sequence[tuple[dict, dict]],
                   meta: dict | None = None) -> "EpisodeLog":
        """From `env.reset` and the `(obs, info)` of each `env.step` (`info["transition"]`)."""
        a = reset_obs["actor"]
        rows = [np.asarray(a["z"], dtype=np.float32)]
        depth = [np.asarray(a["z_depth"], dtype=np.float32)]
        valid = [np.asarray(a["z_valid"]) > 0.5]
        ids = [np.asarray(reset_info["slot_ids"], dtype=np.int64)]
        mech = [np.asarray(a["mech"], dtype=np.float64)]
        tool = [np.asarray(a["tool"], dtype=np.float64)]
        act = [np.full(ACTION_W, np.nan)]
        grip = [float(a["tool"][4])]
        comp = [np.ones(2)]
        step = [np.asarray(a["progress"], dtype=np.float64)[3:3 + STEP_W]]
        extra_keys = ("u", "k", "step_index", "held", "d_mm", "gt_z", "gt_tool_err_mm")
        extras: dict[str, list] = {k: [] for k in extra_keys}
        for obs, info in steps:
            tr = info["transition"]
            rows.append(np.asarray(tr["z"], dtype=np.float32))
            depth.append(np.asarray(tr["depth"], dtype=np.float32))
            valid.append(np.asarray(tr["valid"], dtype=bool))
            ids.append(np.asarray(tr["slot_ids"], dtype=np.int64))
            mech.append(np.asarray(tr["mech"], dtype=np.float64))
            tool.append(np.asarray(tr["tool"], dtype=np.float64))
            act.append(np.asarray(tr["exec_action"], dtype=np.float64))
            grip.append(float(tr["grip"]))
            comp.append(np.asarray(tr["compliance"], dtype=np.float64))
            step.append(np.asarray(obs["actor"]["progress"], dtype=np.float64)[3:3 + STEP_W])
            for k in extra_keys:
                if k in tr:
                    extras[k].append(np.asarray(tr[k]))
        ex = {k: np.stack(v) for k, v in extras.items() if v and len(v) == len(steps)}
        m = dict(meta or {})
        for k in ("world_seed", "target_seed", "level", "source", "weights_sha256"):
            if k in reset_info and k not in m:
                m[k] = reset_info[k]
        m.setdefault("W", int(rows[0].shape[1]))
        return cls(z=np.stack(rows), depth=np.stack(depth), valid=np.stack(valid),
                   slot_ids=np.stack(ids), mech=np.stack(mech), tool=np.stack(tool),
                   exec_action=np.stack(act), grip=np.asarray(grip), compliance=np.stack(comp),
                   step_onehot=np.stack(step), meta=m, extras=ex)

    # -- storage ---------------------------------------------------------------------------

    def arrays(self) -> dict:
        out = {"z": self.z, "depth": self.depth, "valid": self.valid, "slot_ids": self.slot_ids,
               "mech": self.mech, "tool": self.tool, "exec_action": self.exec_action,
               "grip": self.grip, "compliance": self.compliance, "step_onehot": self.step_onehot,
               "meta": np.asarray(json.dumps(self.meta, sort_keys=True), dtype=np.str_)}
        for k, v in self.extras.items():
            if k not in out:
                out[k] = np.asarray(v)
        return out

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **self.arrays())
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> "EpisodeLog":
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            raise ValueError(f"{path}: episode log lacks {missing}")
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        if "episode_id" not in meta:
            digits = re.findall(r"\d+", path.stem)
            if digits:
                meta["episode_id"] = int(digits[-1])
        N = data["z"].shape[0]
        tool = data["tool"]
        grip = data["grip"] if "grip" in data else tool[:, 4]
        comp = data["compliance"] if "compliance" in data else np.ones((N, 2))
        if "step_onehot" in data:
            step = data["step_onehot"]
        elif "progress" in data:
            step = np.asarray(data["progress"])[:, 3:3 + STEP_W]
        else:
            step = np.zeros((N, STEP_W))
        extras = {k: v for k, v in data.items() if k not in KNOWN}
        return cls(z=data["z"], depth=data["depth"], valid=data["valid"],
                   slot_ids=data["slot_ids"], mech=data["mech"], tool=tool,
                   exec_action=data["exec_action"], grip=grip, compliance=comp,
                   step_onehot=step, meta=meta, extras=extras)


def load_episodes(directory, pattern: str = "ep_*.npz") -> list[EpisodeLog]:
    """Every episode log in `directory`, in file-name order."""
    return [EpisodeLog.load(p) for p in sorted(Path(directory).glob(pattern))]


def episode_ids(logs: Sequence[EpisodeLog]) -> np.ndarray:
    """`[n]` episode id per log: `meta["episode_id"]`, else the position in `logs`."""
    return np.asarray([log.episode_id if log.episode_id >= 0 else n
                       for n, log in enumerate(logs)], dtype=np.int64)


def split_logs(logs: Sequence[EpisodeLog], every: int = HOLDOUT_EVERY
               ) -> tuple[list[EpisodeLog], list[EpisodeLog]]:
    """`(training, held out)` logs by episode id, the rule of `Windows.split`."""
    held = (episode_ids(logs) % int(every)) == int(every) - 1
    return ([log for log, h in zip(logs, held) if not h],
            [log for log, h in zip(logs, held) if h])


# ------------------------------------------------------------------------------- windows ---


@dataclass
class Windows:
    """Multi-step open-loop rollout windows, slots in the order of the window's first tick."""

    rows: np.ndarray
    """`[M, H+1, K, W]`: `z_t` as logged (every slot), then the rows of the same slot ids at
    `t+1..t+H` (objects of `t` only; zeros if absent)."""
    depth: np.ndarray
    """`[M, H+1, K]` (as logged at `t`; 0.5 if absent later)."""
    present: np.ndarray
    """`[M, H+1, K]`: an object at `t` (valid, slot id >= 0) observed with the same id at `t+h`;
    `present[:, 0]` is `F`'s slot set."""
    valid0: np.ndarray
    """`[M, K]` valid at `t` (background included)."""
    ids0: np.ndarray
    """`[M, K]` slot ids at `t` (background -1)."""
    ctx: np.ndarray
    """`[M, H, 43]` contexts `c_t..c_{t+H-1}`."""
    action: np.ndarray
    """`[M, H, 4]` executed actions `a_t..a_{t+H-1}` [m, rad]."""
    q_next: np.ndarray
    """`[M, H, 6]` joint positions `q_{t+1..t+H}`."""
    episode: np.ndarray
    start: np.ndarray
    horizon: int

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    @property
    def K(self) -> int:
        return int(self.rows.shape[2])

    @property
    def W(self) -> int:
        return int(self.rows.shape[3])

    @classmethod
    def from_logs(cls, logs: Sequence[EpisodeLog], horizon: int = 3) -> "Windows":
        H = int(horizon)
        if H < 1:
            raise ValueError("horizon must be at least 1")
        parts: dict[str, list] = {k: [] for k in ("rows", "depth", "present", "valid0", "ids0",
                                                  "ctx", "action", "q_next", "episode",
                                                  "start")}
        eids = episode_ids(logs)
        for n, log in enumerate(logs):
            N = log.n_ticks
            K, W = log.z.shape[1], log.z.shape[2]
            n_win = N - H
            if n_win <= 0:
                continue
            obj = log.valid & (log.slot_ids >= 0)
            al_rows = np.zeros((n_win, H + 1, K, W), dtype=np.float32)
            al_depth = np.full((n_win, H + 1, K), 0.5, dtype=np.float32)
            present = np.zeros((n_win, H + 1, K), dtype=bool)
            al_rows[:, 0] = log.z[:n_win]
            al_depth[:, 0] = log.depth[:n_win]
            present[:, 0] = obj[:n_win]
            for h in range(1, H + 1):
                ia, ib = log.slot_ids[:N - h], log.slot_ids[h:]
                eq = ((ia[:, :, None] == ib[:, None, :]) & obj[:N - h, :, None]
                      & log.valid[h:, None, :])
                j = eq.argmax(axis=-1)
                found = eq.any(axis=-1)
                r = np.take_along_axis(log.z[h:], j[..., None], axis=1)
                d = np.take_along_axis(log.depth[h:], j, axis=1)
                al_rows[:, h] = np.where(found[..., None], r, 0.0)[:n_win]
                al_depth[:, h] = np.where(found, d, 0.5)[:n_win]
                present[:, h] = found[:n_win]
            call = log.contexts()
            aall = log.exec_action[1:]
            qall = log.mech[1:, :Q_W]
            idx = np.arange(n_win)[:, None] + np.arange(H)[None, :]
            ctx, act, q = call[idx], aall[idx], qall[idx]
            ok = (np.all(np.isfinite(act), axis=(1, 2)) & np.all(np.isfinite(ctx), axis=(1, 2))
                  & np.all(np.isfinite(al_rows), axis=(1, 2, 3)))
            eid = int(eids[n])
            parts["rows"].append(al_rows[ok])
            parts["depth"].append(al_depth[ok])
            parts["present"].append(present[ok])
            parts["valid0"].append(log.valid[:n_win][ok])
            parts["ids0"].append(log.slot_ids[:n_win][ok])
            parts["ctx"].append(ctx[ok].astype(np.float32))
            parts["action"].append(act[ok].astype(np.float32))
            parts["q_next"].append(q[ok].astype(np.float32))
            parts["episode"].append(np.full(int(ok.sum()), eid, dtype=np.int64))
            parts["start"].append(np.arange(n_win, dtype=np.int64)[ok])
        if not parts["rows"]:
            raise ValueError(f"no episode is longer than the horizon ({H} ticks)")
        return cls(**{k: np.concatenate(v) for k, v in parts.items()}, horizon=H)

    def subset(self, idx) -> "Windows":
        idx = np.asarray(idx)
        return Windows(rows=self.rows[idx], depth=self.depth[idx], present=self.present[idx],
                       valid0=self.valid0[idx], ids0=self.ids0[idx], ctx=self.ctx[idx],
                       action=self.action[idx], q_next=self.q_next[idx],
                       episode=self.episode[idx], start=self.start[idx], horizon=self.horizon)

    def frame(self, i: int) -> SlotFrame:
        """The first tick of window `i` as logged (background included)."""
        return SlotFrame(self.rows[i, 0], self.depth[i, 0], self.valid0[i], self.ids0[i])

    def held_out(self, every: int = HOLDOUT_EVERY) -> np.ndarray:
        """`[M]` bool: the window's episode is held out (`id % every == every - 1`)."""
        return (self.episode % int(every)) == int(every) - 1

    def split(self, every: int = HOLDOUT_EVERY) -> tuple["Windows", "Windows"]:
        """`(training, held out)` by episode id."""
        held = self.held_out(every)
        return self.subset(np.nonzero(~held)[0]), self.subset(np.nonzero(held)[0])

    def fingerprint(self) -> str:
        """sha256 over every array and the horizon."""
        h = hashlib.sha256()
        h.update(f"H={self.horizon}".encode())
        for name in ("rows", "depth", "present", "valid0", "ids0", "ctx", "action", "q_next",
                     "episode", "start"):
            a = np.ascontiguousarray(getattr(self, name))
            h.update(name.encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def tensors(self) -> dict:
        import torch

        return {"rows": torch.from_numpy(np.ascontiguousarray(self.rows)),
                "depth": torch.from_numpy(np.ascontiguousarray(self.depth)),
                "present": torch.from_numpy(np.ascontiguousarray(self.present)),
                "valid0": torch.from_numpy(np.ascontiguousarray(self.valid0)),
                "ids0": torch.from_numpy(np.ascontiguousarray(self.ids0)),
                "ctx": torch.from_numpy(np.ascontiguousarray(self.ctx)),
                "action": torch.from_numpy(np.ascontiguousarray(self.action)),
                "q_next": torch.from_numpy(np.ascontiguousarray(self.q_next))}

    def batches(self, batch: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
        """Index batches of one shuffled epoch."""
        order = rng.permutation(len(self))
        for s in range(0, len(order), int(batch)):
            yield order[s:s + int(batch)]


# --------------------------------------------------------------------------- synthetic toy ---


@dataclass(frozen=True)
class ToyConfig:
    """Synthetic control-affine slot dynamics."""

    episode_ticks: int = 50
    action_range: tuple[float, float, float, float] = (2e-3, 2e-3, 1e-3, math.radians(1.5))
    """Training action range per tick [m, m, m, rad]; actions uniform in +-range."""
    gain: float = 0.33
    """Window units of centre motion per metre of tool motion, times the tool height [m]."""
    height_range: tuple[float, float] = (0.045, 0.095)
    centre_noise: float = 2e-4
    mass_noise: float = 2e-5
    moments_noise: float = 2e-5
    orient_noise: float = 1e-3
    drift: float = 2e-3
    """Centre drift per tick [window units] times tanh of the first two joint speeds."""
    shuffle_p: float = 0.1
    """Probability per tick of a new slot order."""
    dropout_p: float = 0.01
    """Probability per object and tick of a missed detection."""
    K: int = 8


class ToyDynamics:
    """Known `f` and `B` of the synthetic toy, named schema, physical units."""

    def __init__(self, cfg: ToyConfig = ToyConfig(),
                 schema: SlotSchema = PLACEHOLDER_SLOT_SCHEMA) -> None:
        self.cfg = cfg
        self.schema = schema
        self.layout = SlotLayout.for_schema(schema)
        names = self.layout.dyn_names
        self.i = {n: names.index(n) for n in names}

    def affine(self, rows: np.ndarray, depth: np.ndarray, ctx: np.ndarray):
        """`f [K, D]` and `B [K, D, 4]` for every slot (callers mask objects)."""
        L, i = self.layout, self.i
        dyn = L.dyn(np.asarray(rows, dtype=np.float64), np.asarray(depth, dtype=np.float64))
        K, D = dyn.shape
        tool = np.asarray(ctx, dtype=np.float64)[MECH_W:MECH_W + TOOL_W]
        mech = np.asarray(ctx, dtype=np.float64)[:MECH_W]
        h, psi = float(tool[2]), float(tool[3])
        G = self.cfg.gain / h
        c, s = math.cos(psi), math.sin(psi)
        RT = np.array([[c, s], [-s, c]])
        cx, cy = i["centre_0"], i["centre_1"]
        B = np.zeros((K, D, ACTION_W))
        B[:, cx:cy + 1, 0:2] = -G * RT[None]
        B[:, cx, 2] = -dyn[:, cx] / h
        B[:, cy, 2] = -dyn[:, cy] / h
        B[:, i["mass_0"], 2] = -2.0 * dyn[:, i["mass_0"]] / h
        for n in ("moments_0", "moments_1", "moments_2"):
            B[:, i[n], 2] = -2.0 * dyn[:, i[n]] / h
        B[:, cx, 3] = dyn[:, cy]
        B[:, cy, 3] = -dyn[:, cx]
        B[:, i["orient4_0"], 3] = 4.0 * dyn[:, i["orient4_1"]]
        B[:, i["orient4_1"], 3] = -4.0 * dyn[:, i["orient4_0"]]
        f = np.zeros((K, D))
        f[:, cx] = self.cfg.drift * math.tanh(mech[6])
        f[:, cy] = self.cfg.drift * math.tanh(mech[7])
        return f, B


def _toy_object(rng: np.random.Generator, schema: SlotSchema, depth: float):
    r = np.zeros(schema.W)
    r[schema.fields["mass"]] = rng.uniform(0.004, 0.05)
    r[schema.fields["centre"]] = rng.uniform(-0.6, 0.6, 2)
    r[schema.fields["moments"]] = (rng.uniform(0.001, 0.01), rng.uniform(-0.002, 0.002),
                                   rng.uniform(0.001, 0.01))
    r[schema.fields["colour"]] = rng.uniform(-1.0, 1.0, 3)
    r[schema.fields["colour_spread"]] = rng.uniform(0.0, 0.05)
    r[schema.fields["border"]] = rng.uniform(0.0, 0.1)
    r[schema.fields["age"]] = 0.5
    r[schema.fields["orient4"]] = rng.uniform(-0.2, 0.2, 2)
    return r, depth


def synthetic_logs(n_transitions: int, *, seed: int = 0, cfg: ToyConfig = ToyConfig(),
                   first_episode_id: int = 0) -> tuple[list[EpisodeLog], ToyDynamics]:
    """Episodes of the toy, at least `n_transitions` transitions in total.

    Per tick: `z' = z + f(z, c) + B(z, c) a + noise` on every object; the tool target moves by
    `a`; joint positions follow `q' = q + 0.1 speed`. Slots keep ids but change order and miss
    detections at random; slot 0 of the scene is a static background (valid, id -1)."""
    toy = ToyDynamics(cfg)
    schema, L = toy.schema, toy.layout
    rng = np.random.default_rng(seed)
    K, W = cfg.K, schema.W
    rngA = np.asarray(cfg.action_range)
    noise = np.zeros(L.D)
    for n, i in toy.i.items():
        noise[i] = (cfg.centre_noise if n.startswith("centre") else cfg.mass_noise if n == "mass_0"
                    else cfg.moments_noise if n.startswith("moments")
                    else cfg.orient_noise if n.startswith("orient4") else 0.0)
    logs = []
    total = 0
    ep = first_episode_id
    while total < n_transitions:
        T = cfg.episode_ticks
        n_obj = int(rng.integers(3, K - 1))
        depths = rng.permutation(np.linspace(0.1, 0.9, n_obj))
        objs = [_toy_object(rng, schema, float(d)) for d in depths]
        rows_true = np.zeros((K, W))
        depth_true = np.full(K, 0.5)
        ids_true = np.full(K, -1, dtype=np.int64)
        valid_true = np.zeros(K, dtype=bool)
        bg = np.zeros(W)
        bg[schema.fields["mass"]] = 0.5
        bg[schema.fields["colour"]] = (0.0, -0.05, -0.1)
        bg[schema.fields["moments"]] = (0.3, 0.0, 0.3)
        rows_true[0], depth_true[0], valid_true[0] = bg, 1.0, True
        for n, (r, d) in enumerate(objs):
            rows_true[1 + n], depth_true[1 + n] = r, d
            ids_true[1 + n], valid_true[1 + n] = 10 + n, True
        mech = np.concatenate([rng.uniform(-1.0, 1.0, 6), rng.normal(0.0, 0.5, 6),
                               rng.normal(0.0, 0.1, 6)])
        h0 = rng.uniform(*cfg.height_range)
        cmd = np.array([rng.uniform(0.15, 0.3), rng.uniform(-0.1, 0.1), h0,
                        rng.uniform(-0.3, 0.3)])
        offset = np.zeros(4)
        order = rng.permutation(K)

        z, dp, va, ids, M, TL, A = [], [], [], [], [], [], []
        a_prev = np.full(ACTION_W, np.nan)
        for t in range(T + 1):
            if t > 0 and rng.random() < cfg.shuffle_p:
                order = rng.permutation(K)
            miss = (rng.random(K) < cfg.dropout_p) & (ids_true >= 0)
            z.append(np.where(miss[:, None], 0.0, rows_true)[order])
            dp.append(np.where(miss, 0.5, depth_true)[order])
            va.append((valid_true & ~miss)[order])
            ids.append(np.where(miss, -1, ids_true)[order])
            M.append(mech.copy())
            fk = cmd + rng.normal(0.0, 2e-4, 4)
            TL.append(np.concatenate([cmd, [0.0], fk, offset]))
            A.append(a_prev)
            if t == T:
                break
            a = rng.uniform(-1.0, 1.0, ACTION_W) * rngA
            a[2] = float(np.clip(a[2], cfg.height_range[0] - cmd[2], cfg.height_range[1] - cmd[2]))
            ctx = context_vector(mech, TL[-1], None, (1.0, 1.0), 0.0)
            f, B = toy.affine(rows_true, depth_true, ctx)
            dyn = L.dyn(rows_true, depth_true)
            obj = ids_true >= 0
            step = f + np.einsum("kdj,j->kd", B, a) + rng.normal(0.0, 1.0, (K, L.D)) * noise
            rows_true, depth_true = L.with_dyn(rows_true, np.where(obj[:, None], dyn + step, dyn))
            a_prev = a
            mech = mech.copy()
            mech[0:6] = mech[0:6] + 0.1 * mech[6:12]
            mech[6:12] = 0.8 * mech[6:12] + rng.normal(0.0, 0.3, 6)
            mech[12:18] = rng.normal(0.0, 0.1, 6)
            cmd = cmd + a
            offset = offset + a
        exec_action = np.stack(A)
        N = len(z)
        logs.append(EpisodeLog(
            z=np.stack(z), depth=np.stack(dp), valid=np.stack(va), slot_ids=np.stack(ids),
            mech=np.stack(M), tool=np.stack(TL), exec_action=exec_action, grip=np.zeros(N),
            compliance=np.ones((N, 2)), step_onehot=np.zeros((N, STEP_W)),
            meta={"episode_id": ep, "source": "synthetic", "weights_sha256": "", "W": W,
                  "level": "synthetic", "seed": int(seed)}))
        total += N - 1
        ep += 1
    return logs, toy
