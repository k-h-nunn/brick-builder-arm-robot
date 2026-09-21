"""A small tanh MLP with a frozen input normalizer, in plain NumPy.

The control path never imports torch. Training (`brick_builder.action.learn.train`) builds the
same layer stack in torch and writes the weights here as one `.npz`; the loop, the offline
metrics and the C exporter all read that file and run this forward pass, so there is exactly one
definition of what the network computes.

The normalizer is part of the network, not of the caller: inputs are centred and scaled by
statistics measured on the training set and then clipped to `+-clip` standard deviations. That
clipping is what makes a far target read as a direction -- an input ten times further out than
anything in the data arrives at the first layer as the same saturated value, so the output is a
direction of the right sign rather than an extrapolated magnitude. Outputs are scaled the same
way, so the network's own units are order one whatever the physical units are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mlp:
    """`clip((x - mean) / std) -> Linear/tanh stack -> y * out_std + out_mean`."""

    layers: list[tuple[np.ndarray, np.ndarray]]
    """`(W [out, in], b [out])` per layer, the last one without an activation."""
    in_mean: np.ndarray
    in_std: np.ndarray
    out_mean: np.ndarray
    out_std: np.ndarray
    clip: float = 3.0
    meta: dict = None

    def __post_init__(self) -> None:
        self.layers = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32))
                       for w, b in self.layers]
        self.in_mean = np.asarray(self.in_mean, dtype=np.float32)
        self.in_std = np.asarray(self.in_std, dtype=np.float32)
        self.out_mean = np.asarray(self.out_mean, dtype=np.float32)
        self.out_std = np.asarray(self.out_std, dtype=np.float32)
        self.meta = dict(self.meta or {})

    # -- shape and cost -------------------------------------------------------------------------

    @property
    def sizes(self) -> tuple[int, ...]:
        return (int(self.layers[0][0].shape[1]), *[int(w.shape[0]) for w, _ in self.layers])

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in self.layers))

    @property
    def macs(self) -> int:
        """Multiply-accumulates of one forward pass, weights only -- the figure the guide quotes."""
        return int(sum(w.size for w, _ in self.layers))

    @property
    def weight_bytes(self) -> int:
        """Bytes of float32 weights, biases and normalizer constants."""
        return int(4 * (self.n_params + self.in_mean.size + self.in_std.size
                        + self.out_mean.size + self.out_std.size))

    # -- forward --------------------------------------------------------------------------------

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.clip((x - self.in_mean) / self.in_std, -self.clip, self.clip)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """`x [n, in]` or `[in]` -> `[n, out]` or `[out]`, in the output's own physical units."""
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        h = self.normalize(x.reshape(1, -1) if single else x)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        y = h * self.out_std + self.out_mean
        return y[0] if single else y

    __call__ = forward

    # -- storage --------------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"n_layers": np.array(len(self.layers)),
                  "in_mean": self.in_mean, "in_std": self.in_std,
                  "out_mean": self.out_mean, "out_std": self.out_std,
                  "clip": np.array(self.clip, dtype=np.float32)}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        for key, value in self.meta.items():
            arrays[f"meta_{key}"] = np.array(value)
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Mlp":
        with np.load(Path(path), allow_pickle=False) as data:
            n = int(data["n_layers"])
            layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(n)]
            meta = {k[len("meta_"):]: data[k].tolist() for k in data.files if k.startswith("meta_")}
            return cls(layers=layers, in_mean=data["in_mean"], in_std=data["in_std"],
                       out_mean=data["out_mean"], out_std=data["out_std"],
                       clip=float(data["clip"]), meta=meta)


def from_torch(module, in_mean, in_std, out_mean, out_std, *, clip: float = 3.0,
               meta: dict | None = None) -> Mlp:
    """Copy a `torch.nn.Sequential(Linear, Tanh, ..., Linear)` into an `Mlp`."""
    layers = []
    for child in module:
        w = getattr(child, "weight", None)
        if w is None:
            continue
        layers.append((w.detach().cpu().numpy().astype(np.float32),
                       child.bias.detach().cpu().numpy().astype(np.float32)))
    return Mlp(layers=layers, in_mean=in_mean, in_std=in_std, out_mean=out_mean,
               out_std=out_std, clip=clip, meta=meta or {})


def std_floor(std: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Standard deviations with a floor, so a constant input column does not divide by zero."""
    std = np.asarray(std, dtype=np.float64)
    return np.maximum(std, floor).astype(np.float32)
