"""Turn a PPO or SAC checkpoint's actor into plain C data.

`write_header` reads a checkpoint written by `brick_builder.rl.ppo_asym.train` or
`brick_builder.rl.sac_asym.train` (both share the same `actor`, `norm_actor`, `cfg`, `obs_spec`
checkpoint layout -- see either module for the exact format) and emits `policy_theta.h`: the
actor's weight matrices and biases, the frozen input normalizer's mean/var, the action
bounds, the checkpoint's squashing mode, and a few macros the hand-written `c/policy.c`
needs to size its buffers. `write_tanh_table` emits the optional lookup-table tanh
(`c/tanh_table.h`), which does not depend on any checkpoint and only needs to be written once.

**Squashing mode.** A PPO actor's deployed action is its raw mean, clamped to the action
bounds (`brick_builder.rl.ppo_asym.evaluate`); a SAC actor's is `tanh(mean)` mapped onto the
bounds (`brick_builder.rl.sac_asym.select_action`). Both actors are the
same tanh-MLP-plus-state-independent-log-std architecture (`brick_builder.rl.nets.Actor`), so the
only thing that differs is this last step -- recorded in the checkpoint's `cfg["squash"]`
(absent, or any value other than `"tanh"`, means the PPO/clamp convention, which keeps every
checkpoint written before this field existed exporting exactly as before) and emitted here as
`POLICY_SQUASH_TANH` for `c/policy.c` to branch on.

Both headers are generated data, not source: `build.py` writes them into a build directory
(by default `brick_builder/export/_build/`, which is git-ignored) rather than into `c/`, which
holds only the hand-written, checkpoint-independent `policy.c` / `policy.h`.

This is the usual pattern for a weights-to-C generator
(read a container, emit `static const` C arrays plus a digest and dimension macros) but the
container here is an ordinary `torch.save` checkpoint dict, since that is what
`brick_builder.rl.ppo_asym.train` writes; there is no separate binary container format to parse.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _c_float(v: float) -> str:
    """Format a float32 as a C99 floating constant. `%.9g` round-trips a float32 exactly;
    a value with neither a '.' nor an exponent (e.g. an integer-valued float) needs a '.'
    added before the 'f' suffix, or '0f' would be an illegal integer-with-float-suffix."""
    s = f"{float(np.float32(v)):.9g}"
    if not any(ch in s for ch in ".eEnN"):
        s += ".0"
    return s + "f"


def _wrapped(values: Sequence[str], per_line: int = 8, indent: str = "    ") -> list[str]:
    out = []
    for i in range(0, len(values), per_line):
        out.append(indent + ", ".join(values[i : i + per_line]) + ",")
    if out:
        out[-1] = out[-1][:-1]  # drop the trailing comma on the last line
    return out


def load_actor_arrays(checkpoint: Path) -> dict:
    """Pull everything `write_header` needs out of a checkpoint, as plain Python/NumPy
    objects (no torch types survive past this function, so the rest of the module has no
    torch dependency).

    Layer sizes come from `cfg["hidden_actor"]` and `obs_spec["actor_dim"]`/`["act_dim"]`,
    exactly as `brick_builder.rl.nets._mlp` builds them: `nn.Sequential` alternates
    `Linear, Tanh, Linear, Tanh, ..., Linear` (no trailing activation), so the i-th Linear
    layer's parameters are always at state-dict keys `net.{2*i}.weight` / `net.{2*i}.bias`.
    """
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    obs_spec = ckpt["obs_spec"]
    cfg = ckpt["cfg"]
    hidden = tuple(cfg["hidden_actor"])
    sizes = (int(obs_spec["actor_dim"]), *hidden, int(obs_spec["act_dim"]))
    n_layers = len(sizes) - 1

    actor_state = ckpt["actor"]
    layers = []
    for i in range(n_layers):
        w = actor_state[f"net.{2 * i}.weight"].detach().numpy().astype(np.float32)
        b = actor_state[f"net.{2 * i}.bias"].detach().numpy().astype(np.float32)
        assert w.shape == (sizes[i + 1], sizes[i]), (w.shape, sizes, i)
        layers.append((w, b))

    norm_state = ckpt["norm_actor"]
    mean = norm_state["mean"].detach().numpy().astype(np.float32)
    var = norm_state["var"].detach().numpy().astype(np.float32)

    return {
        "obs_dim": sizes[0],
        "act_dim": sizes[-1],
        "hidden": hidden,
        "layers": layers,
        "norm_mean": mean,
        "norm_var": var,
        "norm_eps": 1e-8,  # brick_builder.rl.nets.RunningNorm's fixed default; not itself a buffer
        "norm_clip": float(cfg["norm_clip"]),
        "act_low": np.asarray(obs_spec["act_low"], dtype=np.float32),
        "act_high": np.asarray(obs_spec["act_high"], dtype=np.float32),
        # "clamp" (PPO's convention) unless the checkpoint's own cfg says otherwise; a PPO
        # checkpoint's cfg (dataclasses.asdict(PPOConfig)) simply has no "squash" key.
        "squash": cfg.get("squash", "clamp"),
        "digest": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint": str(Path(checkpoint).resolve()),
    }


def weight_bytes(arrays: dict) -> int:
    """Total bytes of `float` weights and biases the header stores (the figure the export
    part of the guide and the parity report quote)."""
    return sum((w.size + b.size) * 4 for w, b in arrays["layers"])


def write_header(checkpoint: Path, out_dir: Path) -> Path:
    """Write `<out_dir>/policy_theta.h` from a PPO checkpoint. Returns its path."""
    a = load_actor_arrays(checkpoint)
    n_layers = len(a["layers"])
    sizes = [a["obs_dim"], *a["hidden"], a["act_dim"]]
    max_hidden = max(sizes) if sizes else 1
    squash_tanh = 1 if a["squash"] == "tanh" else 0

    lines = [
        "/* policy_theta.h: GENERATED by brick_builder/export/actor_to_c.py. DO NOT EDIT BY HAND.",
        " *",
        " * The actor MLP of one PPO or SAC checkpoint (brick_builder.rl.ppo_asym.train's or",
        " * brick_builder.rl.sac_asym.train's format -- both share this layout), as plain C data:",
        " * weight matrices and biases (row-major, out x in, matching torch.nn.Linear.weight),",
        " * the frozen input normalizer's mean/var, the action bounds, and the checkpoint's",
        " * squashing mode (POLICY_SQUASH_TANH: 0 = clamp the mean to the bounds, PPO's",
        " * convention; 1 = tanh(mean) mapped onto the bounds, SAC's).",
        " *",
        f" * checkpoint    {a['checkpoint']}",
        f" * sha256        {a['digest']}",
        f" * layer sizes   {' -> '.join(str(s) for s in sizes)}",
        f" * weight bytes  {weight_bytes(a)}",
        f" * squash mode   {a['squash']}",
        " */",
        "#ifndef POLICY_THETA_H",
        "#define POLICY_THETA_H",
        "",
        f"#define POLICY_OBS_DIM {a['obs_dim']}",
        f"#define POLICY_ACT_DIM {a['act_dim']}",
        f"#define POLICY_N_LAYERS {n_layers}",
        f"#define POLICY_MAX_WIDTH {max_hidden}",
        f'#define POLICY_CKPT_SHA256 "{a["digest"]}"',
        f'#define POLICY_SQUASH_MODE "{a["squash"]}"',
        f"#define POLICY_SQUASH_TANH {squash_tanh}",
        "",
    ]

    for i, (w, b) in enumerate(a["layers"]):
        out_dim, in_dim = w.shape
        lines.append(f"/* layer {i}: Linear({in_dim} -> {out_dim}) */")
        lines.append(f"static const float policy_w{i}[{out_dim * in_dim}] = {{")
        lines += _wrapped([_c_float(v) for v in w.reshape(-1)])
        lines.append("};")
        lines.append(f"static const float policy_b{i}[{out_dim}] = {{")
        lines += _wrapped([_c_float(v) for v in b])
        lines.append("};")
        lines.append("")

    lines.append(f"static const int policy_layer_in[POLICY_N_LAYERS] = {{{', '.join(str(w.shape[1]) for w, _ in a['layers'])}}};")
    lines.append(f"static const int policy_layer_out[POLICY_N_LAYERS] = {{{', '.join(str(w.shape[0]) for w, _ in a['layers'])}}};")
    lines.append("static const float *const policy_weights[POLICY_N_LAYERS] = {" + ", ".join(f"policy_w{i}" for i in range(n_layers)) + "};")
    lines.append("static const float *const policy_biases[POLICY_N_LAYERS] = {" + ", ".join(f"policy_b{i}" for i in range(n_layers)) + "};")
    lines.append("")

    # RunningNorm is (x - mean) / sqrt(var + eps), clipped to +-clip. The 1/sqrt(var + eps)
    # scale is precomputed here in Python rather than with a runtime sqrtf, so the only
    # libm call anywhere in policy.c is the (switchable) tanhf.
    inv_std = 1.0 / np.sqrt(a["norm_var"].astype(np.float64) + a["norm_eps"])
    lines.append("/* RunningNorm: (x - mean) * inv_std, clipped to +-POLICY_NORM_CLIP.")
    lines.append(" * inv_std = 1/sqrt(var + eps), precomputed here so policy.c needs no sqrtf. */")
    lines.append(f"static const float policy_norm_mean[POLICY_OBS_DIM] = {{{', '.join(_c_float(v) for v in a['norm_mean'])}}};")
    lines.append(f"static const float policy_norm_inv_std[POLICY_OBS_DIM] = {{{', '.join(_c_float(v) for v in inv_std)}}};")
    lines.append(f"#define POLICY_NORM_CLIP {_c_float(a['norm_clip'])}")
    lines.append("")
    lines.append("/* POLICY_SQUASH_TANH == 0: the mean action is clamped to these bounds (PPO,")
    lines.append(" * brick_builder.rl.ppo_asym.evaluate). POLICY_SQUASH_TANH == 1: tanh(mean) is mapped onto")
    lines.append(" * [low, high] as tanh(mean) * (high-low)/2 + (high+low)/2 (SAC, brick_builder.rl.sac_asym.select_action). */")
    lines.append(f"static const float policy_act_low[POLICY_ACT_DIM] = {{{', '.join(_c_float(v) for v in a['act_low'])}}};")
    lines.append(f"static const float policy_act_high[POLICY_ACT_DIM] = {{{', '.join(_c_float(v) for v in a['act_high'])}}};")
    lines.append("")
    lines.append("#endif /* POLICY_THETA_H */")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy_theta.h"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_tanh_table(out_dir: Path, n_knots: int = 4097, lo: float = -8.0, hi: float = 8.0) -> Path:
    """Write `<out_dir>/tanh_table.h`: a lookup table of `math.tanh` at `n_knots` evenly
    spaced points on `[lo, hi]`, for `c/policy.c`'s linear-interpolation tanh variant
    (`POLICY_TANH_TABLE`). This table does not depend on any checkpoint, so it is generated
    once and shared by every export.
    """
    xs = np.linspace(lo, hi, n_knots, dtype=np.float64)
    ys = np.tanh(xs).astype(np.float32)

    lines = [
        "/* tanh_table.h: GENERATED by brick_builder/export/actor_to_c.py. DO NOT EDIT BY HAND.",
        " *",
        f" * tanh(x) at {n_knots} evenly spaced knots on [{lo}, {hi}]. c/policy.c linearly",
        " * interpolates between the two knots bracketing x, and saturates to +-1 outside",
        " * the table's range (tanh is already within 1e-6 of +-1 at |x| = 8).",
        " */",
        "#ifndef TANH_TABLE_H",
        "#define TANH_TABLE_H",
        "",
        f"#define TANH_TABLE_N {n_knots}",
        f"#define TANH_TABLE_LO {_c_float(lo)}",
        f"#define TANH_TABLE_HI {_c_float(hi)}",
        f"static const float tanh_table[TANH_TABLE_N] = {{",
    ]
    lines += _wrapped([_c_float(v) for v in ys])
    lines += ["};", "", "#endif /* TANH_TABLE_H */"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tanh_table.h"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def table_tanh_max_error(n_knots: int = 4097, lo: float = -8.0, hi: float = 8.0, n_probe: int = 200_001) -> float:
    """Max absolute error of the linear-interpolation table against `math.tanh`, sampled
    densely (including off-knot points, where the error is largest) between the knots.
    Used by the export report and by the test that gates the table variant."""
    knots_x = np.linspace(lo, hi, n_knots, dtype=np.float64)
    knots_y = np.tanh(knots_x)
    probe = np.linspace(lo, hi, n_probe, dtype=np.float64)
    interp = np.interp(probe, knots_x, knots_y)
    exact = np.tanh(probe)
    return float(np.max(np.abs(interp - exact)))
