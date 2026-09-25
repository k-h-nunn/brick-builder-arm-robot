"""Next-scene-state predictor `F`: control-affine, per slot.

    z'_i = z_i + f_i(z, c) + B_i(z, c) a        dynamic fields and depth
    z'_i = z_i                                  appearance fields and flags (copied)

- `z`: slot rows `[K, W]` and ordinal depth `[K]` in the source's own slot order. Identity is
  kept: row i of the prediction is slot i.
- `c`: context `[43]` (`context_vector`): bus joint positions, speeds and loads (18), tool vector
  (13), step one-hot (9), then c_vert, c_lat and grip of the tick's command.
- `a`: executed tool-target change over the tick `[4]`: dx, dy, dz [m], dyaw [rad]
  (`info["exec_action"]`).

`a` enters only through `B`, so `dz'/da = B(z, c)` at any action size. Nothing in the
differentiable path clamps or clips.

**Slot set.** `F`'s set is the objects: valid slots with slot id >= 0 (`SlotFrame.objects`).
The background slot (valid, id -1) and empty slots are neither network inputs nor predicted;
they are copied. Training (`train_model`) and every numpy entry point (`predict`, `affine`,
`rollout`, `action_jacobian`) use the same set, so no background row reaches the network.

**Body (M1).** Permutation-equivariant slot-set MLP: `h_i = MLP1([phi(z_i), c])` (64), masked
mean and max pool over the objects (128), `g_i = MLP2([h_i, pool, c])` (64). Heads per slot:
`f_i [D]`, `B_i [D, 4]`, validity logit. Head on the pool: next joint positions [6].

**Units.** Heads are in scaled units: dynamic component d changes by
`s_d (f_d + sum_j B_dj a_j / s_a,j)`.

| Quantity | Scale |
|---|---|
| centre, mass, moments, orient4, depth (named schema) | 0.01, 0.002, 0.002, 0.2, 0.25 (the codec's) |
| opaque schema | per-feature std of one-tick changes (calibration), depth 0.25 |
| action dx, dy, dz, dyaw | 1 mm, 1 mm, 1 mm, 1 deg |
| network inputs | affine (mean, std) of the first training set, frozen, stored with the weights |

A checkpoint records source, W and weights sha256 and refuses to load across a mismatch.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ...perception.scene_model import SlotSchema
from .codec import NAMED_SCALES, OTHER_WEIGHT, SlotFrame
from .config import STEP_TITLES

__all__ = [
    "ACTION_NAMES", "ACTION_SCALE", "ACTION_W", "CONTEXT_W", "InputNorm", "MECH_W", "ModelConfig",
    "Q_W", "STEP_W", "ScenePredictor", "SlotLayout", "Step", "TOOL_W", "context_from_obs",
    "context_vector", "fit_input_norm",
]

MECH_W = 18
TOOL_W = 13
STEP_W = len(STEP_TITLES)
CONTEXT_W = MECH_W + TOOL_W + STEP_W + 3
"""Context width: mech (18), tool (13), step one-hot (9), c_vert, c_lat, grip."""
ACTION_NAMES = ("dx", "dy", "dz", "dyaw")
ACTION_W = len(ACTION_NAMES)
ACTION_SCALE = (1e-3, 1e-3, 1e-3, math.radians(1.0))
"""Action input unit [m, m, m, rad]: 1 mm, 1 mm, 1 mm, 1 deg."""
Q_W = 6
FORMAT = "scene_target_predictor"
FORMAT_VERSION = 1

NAMED_DYNAMIC = ("centre", "mass", "moments", "orient4")
NAMED_APPEARANCE = ("colour", "colour_spread")
STD_FLOOR = 1e-6


# ------------------------------------------------------------------------------- context ---


def context_vector(mech, tool, step_onehot=None, compliance=(1.0, 1.0), grip=None) -> np.ndarray:
    """`[43]` context: mech (18), tool (13), step one-hot (9), c_vert, c_lat, grip.

    `compliance` and `grip` are those of the tick being predicted; `grip` defaults to the
    commanded grip in `tool[4]`."""
    mech = np.asarray(mech, dtype=np.float64).reshape(-1)
    tool = np.asarray(tool, dtype=np.float64).reshape(-1)
    if mech.shape != (MECH_W,) or tool.shape != (TOOL_W,):
        raise ValueError(f"mech must be [{MECH_W}] and tool [{TOOL_W}]; got {mech.shape}, {tool.shape}")
    step = np.zeros(STEP_W) if step_onehot is None else np.asarray(step_onehot, dtype=np.float64)
    if step.shape != (STEP_W,):
        raise ValueError(f"step one-hot must be [{STEP_W}]; got {step.shape}")
    comp = np.asarray(compliance, dtype=np.float64).reshape(2)
    g = float(tool[4]) if grip is None else float(grip)
    return np.concatenate([mech, tool, step, comp, [g]]).astype(np.float32)


def context_from_obs(actor: dict, compliance=(1.0, 1.0), grip=None) -> np.ndarray:
    """`[43]` context from the environment's actor observation."""
    progress = np.asarray(actor["progress"], dtype=np.float64)
    return context_vector(actor["mech"], actor["tool"], progress[3:3 + STEP_W], compliance, grip)


# -------------------------------------------------------------------------------- layout ---


@dataclass(frozen=True)
class SlotLayout:
    """Column roles of a slot row for `F`: predicted, appearance input, border."""

    W: int
    dyn_cols: tuple[int, ...]
    """Row columns predicted by `f + B a`; the depth is the last dynamic component."""
    app_cols: tuple[int, ...]
    """Row columns used as appearance inputs (copied in the prediction)."""
    border_col: int
    """Border-fraction column, or -1."""
    dyn_scale: tuple[float, ...]
    """`[D]` scale per dynamic component, depth last."""
    dyn_weight: tuple[float, ...]
    """`[D]` loss weight per dynamic component (the distance's: centre 1, others 0.25)."""
    dyn_names: tuple[str, ...]
    named: bool

    @property
    def D(self) -> int:
        return len(self.dyn_cols) + 1

    @property
    def A(self) -> int:
        return len(self.app_cols)

    @property
    def centre(self) -> tuple[int, ...]:
        """Positions of the centre components in the dynamic vector (named schema)."""
        return tuple(i for i, n in enumerate(self.dyn_names) if n.startswith("centre"))

    @classmethod
    def for_schema(cls, schema: SlotSchema, *, feature_std: Sequence[float] | None = None,
                   depth_std: float | None = None) -> "SlotLayout":
        """Named schema: the codec's fields and scales. Opaque: every column dynamic unless the
        schema declares `dynamic`; scales from `feature_std` (calibration), else 1."""
        W = int(schema.W)
        border = int(np.arange(W)[schema.fields["border"]][0]) if schema.has("border") else -1
        if schema.has("centre"):
            cols, scales, weights, names = [], [], [], []
            for name in NAMED_DYNAMIC:
                if not schema.has(name):
                    continue
                idx = np.arange(W)[schema.fields[name]]
                for n, c in enumerate(idx.tolist()):
                    cols.append(c)
                    scales.append(NAMED_SCALES[name])
                    weights.append(1.0 if name == "centre" else OTHER_WEIGHT)
                    names.append(f"{name}_{n}")
            app = []
            for name in NAMED_APPEARANCE:
                if schema.has(name):
                    app.extend(np.arange(W)[schema.fields[name]].tolist())
            return cls(W=W, dyn_cols=tuple(cols), app_cols=tuple(app), border_col=border,
                       dyn_scale=tuple(scales) + (NAMED_SCALES["depth"],),
                       dyn_weight=tuple(weights) + (OTHER_WEIGHT,),
                       dyn_names=tuple(names) + ("depth",), named=True)
        dyn = (np.arange(W)[schema.fields["dynamic"]].tolist() if schema.has("dynamic")
               else list(range(W)))
        if schema.has("match"):
            app = [c for c in np.arange(W)[schema.fields["match"]].tolist() if c not in dyn]
        else:
            app = [c for c in range(W) if c not in dyn]
        std = np.ones(W) if feature_std is None or len(feature_std) == 0 else np.asarray(
            feature_std, dtype=np.float64)
        if std.shape != (W,):
            raise ValueError(f"feature_std must be [{W}]; got {std.shape}")
        dstd = 0.25 if depth_std is None else float(depth_std)
        return cls(W=W, dyn_cols=tuple(dyn), app_cols=tuple(app), border_col=border,
                   dyn_scale=tuple(float(std[c]) for c in dyn) + (dstd,),
                   dyn_weight=(1.0,) * (len(dyn) + 1),
                   dyn_names=tuple(f"f{c}" for c in dyn) + ("depth",), named=False)

    def dyn(self, rows, depth):
        """`[..., D]` dynamic vector (numpy or torch)."""
        cols = list(self.dyn_cols)
        if isinstance(rows, torch.Tensor):
            return torch.cat([rows[..., cols], depth[..., None]], dim=-1)
        rows = np.asarray(rows)
        return np.concatenate([rows[..., cols], np.asarray(depth)[..., None]], axis=-1)

    def app(self, rows):
        """`[..., A]` appearance inputs."""
        return rows[..., list(self.app_cols)]

    def border(self, rows):
        """`[...]` border fraction (0 without a border column)."""
        if self.border_col < 0:
            return rows[..., 0] * 0.0
        return rows[..., self.border_col]

    def with_dyn(self, rows, dyn):
        """`(rows, depth)` with the dynamic components replaced (numpy or torch)."""
        cols = list(self.dyn_cols)
        if isinstance(rows, torch.Tensor):
            out = rows.clone()
            out[..., cols] = dyn[..., :-1]
            return out, dyn[..., -1]
        out = np.array(rows, copy=True)
        out[..., cols] = dyn[..., :-1]
        return out, np.asarray(dyn[..., -1])

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "SlotLayout":
        return cls(W=int(d["W"]), dyn_cols=tuple(int(c) for c in d["dyn_cols"]),
                   app_cols=tuple(int(c) for c in d["app_cols"]), border_col=int(d["border_col"]),
                   dyn_scale=tuple(float(s) for s in d["dyn_scale"]),
                   dyn_weight=tuple(float(s) for s in d["dyn_weight"]),
                   dyn_names=tuple(d["dyn_names"]), named=bool(d["named"]))


# ------------------------------------------------------------------------- normalization ---


@dataclass
class InputNorm:
    """Affine input normalization: `(x - mean) / std`; never clipped."""

    slot_mean: np.ndarray
    """`[D + A]` dynamic then appearance inputs."""
    slot_std: np.ndarray
    ctx_mean: np.ndarray
    ctx_std: np.ndarray


def _std(x: np.ndarray) -> np.ndarray:
    s = np.std(x, axis=0)
    return np.where(s > STD_FLOOR, s, 1.0)


def fit_input_norm(layout: SlotLayout, rows: np.ndarray, depth: np.ndarray, objects: np.ndarray,
                   ctx: np.ndarray) -> InputNorm:
    """Mean and std of the per-slot inputs over objects (`F`'s slot set) and of the context
    over ticks.

    A component with std below 1e-6 keeps unit scale."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, layout.W)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    objects = np.asarray(objects, dtype=bool).reshape(-1)
    ctx = np.asarray(ctx, dtype=np.float64).reshape(-1, np.asarray(ctx).shape[-1])
    x = np.concatenate([layout.dyn(rows, depth), layout.app(rows)], axis=-1)[objects]
    if x.shape[0] == 0:
        raise ValueError("no object slot to fit the input normalization on")
    return InputNorm(slot_mean=x.mean(0), slot_std=_std(x), ctx_mean=ctx.mean(0),
                     ctx_std=_std(ctx))


# ------------------------------------------------------------------------------ network ---


@dataclass(frozen=True)
class ModelConfig:
    """`F` architecture (M1)."""

    hidden: int = 64
    action_scale: tuple[float, ...] = ACTION_SCALE
    context_w: int = CONTEXT_W
    n_q: int = Q_W

    @property
    def n_a(self) -> int:
        return len(self.action_scale)

    def to_dict(self) -> dict:
        return {"hidden": self.hidden, "action_scale": list(self.action_scale),
                "context_w": self.context_w, "n_q": self.n_q}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(hidden=int(d["hidden"]), action_scale=tuple(float(s) for s in d["action_scale"]),
                   context_w=int(d["context_w"]), n_q=int(d["n_q"]))


class SlotSetNet(nn.Module):
    """Per-slot MLP with a masked mean and max pool."""

    def __init__(self, n_slot: int, n_ctx: int, D: int, n_a: int, hidden: int, n_q: int) -> None:
        super().__init__()
        self.D, self.n_a = D, n_a
        self.mlp1 = nn.Sequential(nn.Linear(n_slot + n_ctx, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.mlp2 = nn.Sequential(nn.Linear(3 * hidden + n_ctx, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, D + D * n_a + 1)
        self.q_head = nn.Linear(2 * hidden + n_ctx, n_q)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, c: torch.Tensor):
        """`x [B, K, n_slot]`, `mask [B, K]` bool (the pooled set), `c [B, n_ctx]` ->
        `f [B, K, D]`, `B [B, K, D, n_a]`, validity logit `[B, K]`, `q [B, n_q]`."""
        nb, K, _ = x.shape
        ck = c[:, None, :].expand(nb, K, c.shape[-1])
        h = self.mlp1(torch.cat([x, ck], dim=-1))
        n = mask.to(h.dtype)[..., None].sum(dim=1)
        mean = torch.where(mask[..., None], h, torch.zeros_like(h)).sum(dim=1) / (
            n + (n == 0).to(h.dtype))
        mx = h.masked_fill(~mask[..., None], float("-inf")).amax(dim=1)
        mx = torch.where(n > 0, mx, torch.zeros_like(mx))
        pool = torch.cat([mean, mx], dim=-1)
        g = self.mlp2(torch.cat([h, pool[:, None, :].expand(nb, K, pool.shape[-1]), ck], dim=-1))
        out = self.head(g)
        D, n_a = self.D, self.n_a
        f = out[..., :D]
        Bm = out[..., D:D + D * n_a].reshape(nb, K, D, n_a)
        v = out[..., D + D * n_a]
        q = self.q_head(torch.cat([pool, c], dim=-1))
        return f, Bm, v, q


@dataclass
class Step:
    """One predicted tick (tensors)."""

    rows: torch.Tensor
    """`[B, K, W]` predicted rows."""
    depth: torch.Tensor
    """`[B, K]` predicted depth."""
    dyn: torch.Tensor
    """`[B, K, D]` predicted dynamic vector."""
    valid_logit: torch.Tensor
    """`[B, K]` logit that the slot is still present."""
    q_norm: torch.Tensor
    """`[B, 6]` next joint positions, normalized like the context."""


class ScenePredictor(nn.Module):
    """Control-affine next-scene-state predictor `F`."""

    def __init__(self, layout: SlotLayout, cfg: ModelConfig = ModelConfig(),
                 norm: InputNorm | None = None, meta: dict | None = None) -> None:
        super().__init__()
        self.layout = layout
        self.cfg = cfg
        self.meta = dict(meta or {})
        """`source`, `weights_sha256`, `W`, training record."""
        D, A, C = layout.D, layout.A, cfg.context_w
        f32 = torch.float32
        self.register_buffer("dyn_scale", torch.tensor(layout.dyn_scale, dtype=f32))
        self.register_buffer("dyn_weight", torch.tensor(layout.dyn_weight, dtype=f32))
        self.register_buffer("act_scale", torch.tensor(cfg.action_scale, dtype=f32))
        self.register_buffer("slot_mean", torch.zeros(D + A, dtype=f32))
        self.register_buffer("slot_std", torch.ones(D + A, dtype=f32))
        self.register_buffer("ctx_mean", torch.zeros(C, dtype=f32))
        self.register_buffer("ctx_std", torch.ones(C, dtype=f32))
        self.net = SlotSetNet(D + A + 2, C, D, cfg.n_a, cfg.hidden, cfg.n_q)
        if norm is not None:
            self.set_norm(norm)

    # -- construction ------------------------------------------------------------------------

    @classmethod
    def create(cls, layout: SlotLayout, cfg: ModelConfig = ModelConfig(), *, seed: int = 0,
               norm: InputNorm | None = None, meta: dict | None = None) -> "ScenePredictor":
        """Initialized from `seed` alone; the global torch generator is left unchanged."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            return cls(layout, cfg, norm, meta)

    def set_norm(self, norm: InputNorm) -> None:
        f32 = torch.float32
        self.slot_mean.copy_(torch.as_tensor(np.asarray(norm.slot_mean), dtype=f32))
        self.slot_std.copy_(torch.as_tensor(np.asarray(norm.slot_std), dtype=f32))
        self.ctx_mean.copy_(torch.as_tensor(np.asarray(norm.ctx_mean), dtype=f32))
        self.ctx_std.copy_(torch.as_tensor(np.asarray(norm.ctx_std), dtype=f32))

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    # -- tensor path ---------------------------------------------------------------------------

    def _inputs(self, rows, depth, objects, ctx):
        """Per-slot inputs (zero outside `objects`, whatever the row holds) and the context."""
        L = self.layout
        x = torch.cat([L.dyn(rows, depth), L.app(rows)], dim=-1)
        x = (x - self.slot_mean) / self.slot_std
        of = objects.to(rows.dtype)
        x = torch.cat([x, of[..., None], (1.0 - L.border(rows))[..., None]], dim=-1)
        x = torch.where(objects[..., None], x, torch.zeros_like(x))
        c = (ctx - self.ctx_mean) / self.ctx_std
        return x, c

    def heads(self, rows: torch.Tensor, depth: torch.Tensor, objects: torch.Tensor,
              ctx: torch.Tensor):
        """`f [B, K, D]`, `B [B, K, D, n_a]` (scaled units), validity logit, normalized `q`.

        `objects [B, K]` bool: `F`'s slot set (valid, slot id >= 0; `SlotFrame.objects`). Other
        slots, the background included, are not inputs."""
        x, c = self._inputs(rows, depth, objects, ctx)
        return self.net(x, objects, c)

    def affine_t(self, rows, depth, objects, ctx):
        """`f [B, K, D]` and `B [B, K, D, n_a]` in physical units (row units per tick; per m or rad)."""
        f, Bm, _, _ = self.heads(rows, depth, objects, ctx)
        return f * self.dyn_scale, Bm * self.dyn_scale[:, None] / self.act_scale

    def step_t(self, rows, depth, objects, ctx, action) -> Step:
        """One tick: `z + s (f + B a / s_a)` on `objects`; other slots copied."""
        f, Bm, v, q = self.heads(rows, depth, objects, ctx)
        a = action / self.act_scale
        delta = self.dyn_scale * (f + torch.einsum("bkdj,bj->bkd", Bm, a))
        dyn0 = self.layout.dyn(rows, depth)
        dyn = torch.where(objects[..., None], dyn0 + delta, dyn0)
        new_rows, new_depth = self.layout.with_dyn(rows, dyn)
        return Step(rows=new_rows, depth=new_depth, dyn=dyn, valid_logit=v, q_norm=q)

    def rollout_t(self, rows, depth, objects, ctxs, actions, *, teacher=None) -> list[Step]:
        """Open-loop rollout over `H = ctxs.shape[1]` ticks from `(rows, depth)`; the slot set
        stays `objects` (the first tick's).

        `teacher = (rows [B, H, K, W], depth [B, H, K], present [B, H, K])`: the observed state
        at t+h replaces the prediction as the input of tick h+1 where present (teacher forcing)."""
        out = []
        r, d = rows, depth
        for h in range(ctxs.shape[1]):
            s = self.step_t(r, d, objects, ctxs[:, h], actions[:, h])
            out.append(s)
            r, d = s.rows, s.depth
            if teacher is not None and h + 1 < ctxs.shape[1]:
                tr, td, tp = teacher
                r = torch.where(tp[:, h, :, None], tr[:, h], r)
                d = torch.where(tp[:, h], td[:, h], d)
        return out

    # -- numpy path ----------------------------------------------------------------------------

    def _frame_tensors(self, frame: SlotFrame):
        """`rows`, `depth` and the slot set (`frame.objects`) as batch-of-one tensors."""
        if frame.W != self.layout.W:
            raise ValueError(f"frame is W={frame.W}; the predictor is for W={self.layout.W}")
        rows = torch.as_tensor(np.asarray(frame.rows, dtype=np.float32))[None]
        depth = torch.as_tensor(np.asarray(frame.depth, dtype=np.float32))[None]
        objects = torch.as_tensor(np.asarray(frame.objects, dtype=bool))[None]
        return rows, depth, objects

    @staticmethod
    def _vec(x, n: int, name: str) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1))
        if t.shape != (n,):
            raise ValueError(f"{name} must be [{n}]; got {tuple(t.shape)}")
        return t[None]

    @torch.no_grad()
    def predict(self, frame: SlotFrame, context, action) -> SlotFrame:
        """`z_{t+1}` for one frame: same slot order and ids; background and empty slots copied
        (and never network inputs)."""
        rows, depth, objects = self._frame_tensors(frame)
        c = self._vec(context, self.cfg.context_w, "context")
        a = self._vec(action, self.cfg.n_a, "action")
        s = self.step_t(rows, depth, objects, c, a)
        return SlotFrame(s.rows[0].numpy(), s.depth[0].numpy(), frame.valid.copy(),
                         frame.slot_ids.copy())

    @torch.no_grad()
    def affine(self, frame: SlotFrame, context) -> tuple[np.ndarray, np.ndarray]:
        """`f [K, D]` [row units per tick] and `B [K, D, n_a]` [row units per m or rad] at `frame`;
        zero on background and empty slots."""
        rows, depth, objects = self._frame_tensors(frame)
        f, Bm = self.affine_t(rows, depth, objects,
                              self._vec(context, self.cfg.context_w, "context"))
        m = objects[0].numpy()
        f, Bm = f[0].numpy() * m[:, None], Bm[0].numpy() * m[:, None, None]
        return f.astype(np.float64), Bm.astype(np.float64)

    @torch.no_grad()
    def rollout(self, frame: SlotFrame, contexts, actions) -> list[SlotFrame]:
        """Open-loop `z_{t+1..t+H}` from `contexts [H, C]` and `actions [H, n_a]`."""
        rows, depth, objects = self._frame_tensors(frame)
        ctxs = torch.as_tensor(np.asarray(contexts, dtype=np.float32))[None]
        acts = torch.as_tensor(np.asarray(actions, dtype=np.float32))[None]
        if ctxs.shape[1] != acts.shape[1]:
            raise ValueError("contexts and actions must have the same number of ticks")
        steps = self.rollout_t(rows, depth, objects, ctxs, acts)
        return [SlotFrame(s.rows[0].numpy(), s.depth[0].numpy(), frame.valid.copy(),
                          frame.slot_ids.copy()) for s in steps]

    def action_jacobian(self, frame: SlotFrame, context, action) -> np.ndarray:
        """`d dyn' / d a` `[K, D, n_a]` by automatic differentiation at `action`."""
        rows, depth, objects = self._frame_tensors(frame)
        c = self._vec(context, self.cfg.context_w, "context")
        a0 = self._vec(action, self.cfg.n_a, "action")

        def fn(a):
            return self.step_t(rows, depth, objects, c, a[None]).dyn[0]

        J = torch.autograd.functional.jacobian(fn, a0[0])
        return J.detach().numpy().astype(np.float64)

    # -- storage -------------------------------------------------------------------------------

    def describe(self) -> dict:
        return {"format": FORMAT, "version": FORMAT_VERSION, "layout": self.layout.to_dict(),
                "model": self.cfg.to_dict(), "meta": self.meta}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"describe": json.dumps(self.describe(), sort_keys=True),
                    "state_dict": self.state_dict()}, tmp)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path, *, source: str | None = None, W: int | None = None,
             weights_sha256: str | None = None) -> "ScenePredictor":
        """Loads a checkpoint; refuses one recorded for a different source, width or weights."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        d = json.loads(blob["describe"])
        if d.get("format") != FORMAT:
            raise ValueError(f"{path} is not a scene-target predictor checkpoint")
        layout = SlotLayout.from_dict(d["layout"])
        model = cls(layout, ModelConfig.from_dict(d["model"]), None, d.get("meta", {}))
        model.load_state_dict(blob["state_dict"])
        model.check_compatible(source=source, W=W, weights_sha256=weights_sha256)
        model.eval()
        return model

    def check_compatible(self, *, source: str | None = None, W: int | None = None,
                         weights_sha256: str | None = None) -> None:
        """Raises `ValueError` when the recorded source, width or weights differ (a value that
        was never recorded counts as different)."""
        if W is not None and int(W) != self.layout.W:
            raise ValueError(f"predictor is for W={self.layout.W}, not W={W}")
        for name, want in (("source", source), ("weights_sha256", weights_sha256)):
            got = self.meta.get(name)
            if want is not None and want != got:
                raise ValueError(f"predictor is for {name}={got!r}, not {want!r}")
