"""Target sequences: arrays, `.npz` cache, generators.

A `TargetSequence` holds the target scene states `S*_0..S*_{T-1}` and the feed-forward commands
that take the nominal twin from each frame to the next. It is arrays only and is stored as `.npz`
without pickled objects, so a source whose `SceneState` carries a closure (the convolutional
model's decode) round-trips.

| Field | Shape | Contents |
|---|---|---|
| `rows`, `depth`, `valid`, `slot_ids` | [T, K, W], [T, K], [T, K], [T, K] | target slot states, source order |
| `commands` | [T, 5, 7] | per frame, the 20 ms writes: pos (3), spin, grip, c_vert, c_lat |
| `n_commands` | [T] | writes recorded in each frame (5, fewer only on a recording's last frame) |
| `action` | [T, 4] | net tool-target change over the frame (dx, dy, dz, dyaw) |
| `step_index`, `titles` | [T], [n] | step of each frame, step titles |
| `freeze_k` | scalar | first frame whose correction is frozen (T when never) |
| `tool_pose` | [T, 4] | tool target (x, y, z, yaw) when `S*_k` was seen; privileged |
| `meta` | JSON | family, seed or cell, source, weights sha256, W, schema, generator |

Cache: `<cache_dir>/targets/<family>/seed<seed>_<source>_<sha8>.npz`.

Generate the cache from the command line (one process):

    python -m brick_builder.rl.scene_target.targets --source ideal --seeds 0-9,20-49
    python -m brick_builder.rl.scene_target.targets --source learned \
        --weights brick_builder/perception/weights/vision_model.pt --seeds 0-9,20-49

Generators:

- **Viewing pose (L0a, L0b)**: a kinematic twin path. Target pose relative to the nominal plate
  centre: xy +-30 mm, TCP height 45-95 mm (the convolutional model's training range), yaw
  +-15 deg. Start pose 10-40 mm and 0-10 deg from it. Minimum-jerk path
  `s = 10 tau^3 - 15 tau^4 + 6 tau^5`, one frame per 100 ms, T = 20-40, five interpolated writes
  per frame. Every frame is rendered in the twin and passed, in order, through one fresh
  instance of the configured source. A draw is rejected when inverse kinematics fails on any
  frame or when any frame has fewer than 2 reference slots matched to its neighbour.
- **Placement stages (L1-L3)**: `from_ideal_sequence` converts a recording of
  `action.place.record_ideal`; `record_placement` makes that recording (imports
  `brick_builder.action` and is never called by the environment at run time).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA, SlotSchema
from .codec import SceneDistance, SlotFrame, reference_mask

__all__ = [
    "GENERATOR_VERSION", "MIN_REFERENCES", "RejectedPose", "TargetSequence",
    "cache_path", "default_cache_dir", "encode_frames", "from_ideal_sequence",
    "load_or_generate", "min_jerk", "parse_seeds", "record_placement", "viewing_pose_draw",
    "viewing_pose_sequence",
]

GENERATOR_VERSION = 1
SCHEMA_VERSION = 1
MIN_REFERENCES = 2
"""Fewest reference slots, matched to the neighbouring frame, a target frame may have."""

TARGET_XY_RANGE = 0.030
TARGET_Z_RANGE = (0.045, 0.095)
TARGET_YAW_RANGE = math.radians(15.0)
START_DIST_RANGE = (0.010, 0.040)
START_YAW_RANGE = (0.0, math.radians(10.0))
T_RANGE = (20, 40)
MAX_DRAWS = 40


class RejectedPose(RuntimeError):
    """A drawn viewing-pose path fails inverse kinematics or the reference-slot test."""


def default_cache_dir() -> Path:
    """`data_cache/scene_target` at the repository root."""
    return Path(__file__).resolve().parents[3] / "data_cache" / "scene_target"


def cache_path(cache_dir, family: str, seed: int, source: str, weights_sha256: str,
               variant: str = "") -> Path:
    """`<cache_dir>/targets/<family>/seed<seed>_<source>_<sha8>[_<variant>].npz`."""
    sha8 = weights_sha256[:8] if weights_sha256 else "none"
    tail = f"_{variant}" if variant else ""
    return Path(cache_dir) / "targets" / family / f"seed{int(seed):05d}_{source}_{sha8}{tail}.npz"


def min_jerk(tau):
    """Minimum-jerk position profile `10 t^3 - 15 t^4 + 6 t^5` on [0, 1]."""
    tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    return 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5


# ------------------------------------------------------------------------------ the arrays ---


@dataclass
class TargetSequence:
    """One target sequence as arrays."""

    rows: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    slot_ids: np.ndarray
    commands: np.ndarray
    n_commands: np.ndarray
    action: np.ndarray
    step_index: np.ndarray
    titles: tuple[str, ...]
    freeze_k: int
    tool_pose: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.float32)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.slot_ids = np.asarray(self.slot_ids, dtype=np.int64)
        self.commands = np.asarray(self.commands, dtype=np.float64)
        self.n_commands = np.asarray(self.n_commands, dtype=np.int64)
        self.action = np.asarray(self.action, dtype=np.float64)
        self.step_index = np.asarray(self.step_index, dtype=np.int64)
        self.titles = tuple(str(t) for t in self.titles)
        self.freeze_k = int(self.freeze_k)
        self.tool_pose = np.asarray(self.tool_pose, dtype=np.float64)
        T, K, W = self.rows.shape
        for name, arr, shape in (("depth", self.depth, (T, K)), ("valid", self.valid, (T, K)),
                                 ("slot_ids", self.slot_ids, (T, K)),
                                 ("commands", self.commands, (T, 5, 7)),
                                 ("n_commands", self.n_commands, (T,)),
                                 ("action", self.action, (T, 4)),
                                 ("step_index", self.step_index, (T,)),
                                 ("tool_pose", self.tool_pose, (T, 4))):
            if arr.shape != shape:
                raise ValueError(f"{name} must be {shape}; got {arr.shape}")
        if np.any(self.n_commands < 1) or np.any(self.n_commands > 5):
            raise ValueError("n_commands must be in 1..5")

    @property
    def T(self) -> int:
        return int(self.rows.shape[0])

    @property
    def K(self) -> int:
        return int(self.rows.shape[1])

    @property
    def W(self) -> int:
        return int(self.rows.shape[2])

    def frame(self, k: int) -> SlotFrame:
        """`S*_k` as a `SlotFrame`; `k` is clamped to [0, T-1]."""
        k = int(min(max(k, 0), self.T - 1))
        return SlotFrame(self.rows[k], self.depth[k], self.valid[k], self.slot_ids[k])

    def frame_commands(self, k: int) -> np.ndarray:
        """`[n, 7]` the writes recorded for frame `k`."""
        k = int(min(max(k, 0), self.T - 1))
        return self.commands[k, : int(self.n_commands[k])]

    def feedforward(self) -> np.ndarray:
        """`[T, 5]` recorded change per frame: (dx, dy, dz, dyaw, dgrip)."""
        out = np.zeros((self.T, 5))
        out[:, :4] = self.action
        for k in range(self.T):
            c = self.frame_commands(k)
            out[k, 4] = c[-1, 4] - c[0, 4]
        return out

    # -- storage -------------------------------------------------------------------------------

    def arrays(self) -> dict:
        return {
            "rows": self.rows, "depth": self.depth, "valid": self.valid, "slot_ids": self.slot_ids,
            "commands": self.commands, "n_commands": self.n_commands, "action": self.action,
            "step_index": self.step_index, "titles": np.asarray(self.titles, dtype=np.str_),
            "freeze_k": np.asarray(self.freeze_k, dtype=np.int64), "tool_pose": self.tool_pose,
            "meta": np.asarray(json.dumps(self.meta, sort_keys=True), dtype=np.str_),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        }

    def save(self, path) -> Path:
        """Write `.npz` atomically (temp file, then rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **self.arrays())
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> "TargetSequence":
        """Read `.npz` with `allow_pickle=False`."""
        with np.load(Path(path), allow_pickle=False) as z:
            version = int(z["schema_version"])
            if version != SCHEMA_VERSION:
                raise ValueError(f"{path}: schema version {version}, expected {SCHEMA_VERSION}")
            return cls(
                rows=z["rows"], depth=z["depth"], valid=z["valid"], slot_ids=z["slot_ids"],
                commands=z["commands"], n_commands=z["n_commands"], action=z["action"],
                step_index=z["step_index"], titles=tuple(z["titles"].tolist()),
                freeze_k=int(z["freeze_k"]), tool_pose=z["tool_pose"],
                meta=json.loads(str(z["meta"])),
            )

    def bitwise_equal(self, other: "TargetSequence") -> bool:
        a, b = self.arrays(), other.arrays()
        return a.keys() == b.keys() and all(
            a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
            for k in a)


# ------------------------------------------------------------------------ viewing-pose paths ---


@dataclass(frozen=True)
class ViewingPoseDraw:
    """One drawn viewing-pose path, before rendering."""

    start: np.ndarray
    """(x, y, z, yaw) of the start pose."""
    target: np.ndarray
    """(x, y, z, yaw) of the target pose."""
    T: int
    draw: int
    """How many draws this seed took (1 = the first was accepted)."""

    def pose(self, tau: float) -> np.ndarray:
        s = float(min_jerk(tau))
        return self.start + s * (self.target - self.start)

    def frame_poses(self) -> np.ndarray:
        """`[T, 4]`: the pose of every frame."""
        return np.stack([self.pose(k / (self.T - 1)) for k in range(self.T)])

    def frame_commands(self, grip: float) -> np.ndarray:
        """`[T, 5, 7]`: five writes per frame along the path; the last frame holds the target."""
        out = np.zeros((self.T, 5, 7))
        for k in range(self.T):
            for i in range(5):
                tau = min((k + (i + 1) / 5.0) / (self.T - 1), 1.0)
                p = self.pose(tau)
                out[k, i, :4] = p
                out[k, i, 4] = grip
                out[k, i, 5:] = 1.0
        return out


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0x5CE7]))


def viewing_pose_draw(seed: int, rng: np.random.Generator | None = None, draw: int = 1,
                      plate_xy=(0.17, 0.0)) -> ViewingPoseDraw:
    """One path draw for `seed` (target pose, start pose, T); no rendering."""
    rng = _rng(seed) if rng is None else rng
    tx = plate_xy[0] + rng.uniform(-TARGET_XY_RANGE, TARGET_XY_RANGE)
    ty = plate_xy[1] + rng.uniform(-TARGET_XY_RANGE, TARGET_XY_RANGE)
    tz = rng.uniform(*TARGET_Z_RANGE)
    tyaw = rng.uniform(-TARGET_YAW_RANGE, TARGET_YAW_RANGE)
    for _ in range(100):
        v = rng.normal(size=3)
        v /= max(np.linalg.norm(v), 1e-12)
        dist = rng.uniform(*START_DIST_RANGE)
        start_xyz = np.array([tx, ty, tz]) + dist * v
        if TARGET_Z_RANGE[0] <= start_xyz[2] <= TARGET_Z_RANGE[1]:
            break
    dyaw = rng.choice([-1.0, 1.0]) * rng.uniform(*START_YAW_RANGE)
    T = int(rng.integers(T_RANGE[0], T_RANGE[1] + 1))
    start = np.array([*start_xyz, tyaw + dyaw])
    target = np.array([tx, ty, tz, tyaw])
    return ViewingPoseDraw(start=start, target=target, T=T, draw=draw)


def encode_frames(frames, source) -> list:
    """Pass rendered frames, in order, through `source`; returns the `SceneState`s."""
    from .world import observe, reset_source

    reset_source(source)
    return [observe(source, f, t=0.1 * k) for k, f in enumerate(frames)]


def _stack_states(states, K: int, W: int):
    T = len(states)
    rows = np.zeros((T, K, W), dtype=np.float32)
    depth = np.zeros((T, K), dtype=np.float32)
    valid = np.zeros((T, K), dtype=bool)
    ids = np.zeros((T, K), dtype=np.int64)
    for k, s in enumerate(states):
        f = SlotFrame.from_state(s)
        rows[k], depth[k], valid[k], ids[k] = f.rows, f.depth, f.valid, f.slot_ids
    return rows, depth, valid, ids


def reference_counts(rows, depth, valid, slot_ids, schema: SlotSchema = PLACEHOLDER_SLOT_SCHEMA):
    """`[T]` reference slots of each frame matched to its neighbour (the next; the previous for
    the last frame)."""
    dist = SceneDistance(schema)
    T = rows.shape[0]
    frames = [SlotFrame(rows[k], depth[k], valid[k], slot_ids[k]) for k in range(T)]
    out = np.zeros(T, dtype=np.int64)
    for k in range(T):
        other = frames[k + 1] if k + 1 < T else frames[k - 1] if T > 1 else frames[k]
        perm = dist.match(frames[k], other)            # target = other, obs = frame k
        ref = reference_mask(frames[k], schema)
        matched_obs = set(int(i) for i in perm if i >= 0)
        out[k] = sum(1 for i in np.nonzero(ref)[0] if int(i) in matched_obs)
    return out


def viewing_pose_sequence(seed: int, *, source, source_info, twin=None, T: int | None = None,
                          keep_frames: bool = False, max_draws: int = MAX_DRAWS):
    """Generate the viewing-pose target sequence of `seed` through `source`.

    Returns `(sequence, stats)`; `stats` has the twin frame cost and, with `keep_frames`, the
    rendered frames (for re-encoding tests). `T` overrides the drawn length.
    """
    from .world import NominalTwin

    own = twin is None
    twin = NominalTwin() if own else twin
    try:
        rng = _rng(seed)
        rejected: list[str] = []
        for draw in range(1, max_draws + 1):
            d = viewing_pose_draw(seed, rng, draw)
            if T is not None:
                d = ViewingPoseDraw(start=d.start, target=d.target, T=int(T), draw=draw)
            poses = d.frame_poses()
            twin.reset_seed_pose()
            try:
                for p in poses:
                    twin.pose(p[:3], p[3])
            except RuntimeError as exc:
                rejected.append(f"draw {draw}: inverse kinematics ({exc})")
                continue
            twin.reset_seed_pose()
            t0 = time.perf_counter()
            frames = []
            for p in poses:
                twin.pose(p[:3], p[3])
                frames.append(twin.render())
            t_render = time.perf_counter() - t0
            t0 = time.perf_counter()
            states = encode_frames(frames, source)
            t_encode = time.perf_counter() - t0
            schema = states[0].schema
            rows, depth, valid, ids = _stack_states(states, states[0].K, states[0].W)
            counts = reference_counts(rows, depth, valid, ids, schema)
            if counts.min() < MIN_REFERENCES:
                rejected.append(f"draw {draw}: frame {int(np.argmin(counts))} has "
                                f"{int(counts.min())} matched reference slot(s)")
                continue
            commands = d.frame_commands(twin.grip)
            action = commands[:, -1, :4] - commands[:, 0, :4]
            meta = {
                "family": "viewing_pose", "seed": int(seed), "source": source_info.name,
                "weights_sha256": source_info.weights_sha256, "W": int(rows.shape[2]),
                "schema": "named" if schema.has("centre") else "opaque",
                "generator": "viewing_pose", "generator_version": GENERATOR_VERSION,
                "start_pose": d.start.tolist(), "target_pose": d.target.tolist(), "T": d.T,
                "draw": draw, "rejected": rejected, "grip": twin.grip,
                "min_references": int(counts.min()),
            }
            seq = TargetSequence(
                rows=rows, depth=depth, valid=valid, slot_ids=ids, commands=commands,
                n_commands=np.full(d.T, 5), action=action, step_index=np.zeros(d.T),
                titles=("Viewing pose",), freeze_k=d.T, tool_pose=poses, meta=meta,
            )
            stats = {"render_s": t_render, "encode_s": t_encode, "frames": d.T,
                     "frame_ms": 1e3 * (t_render + t_encode) / d.T, "draws": draw}
            if keep_frames:
                stats["frames_rendered"] = frames
            return seq, stats
        raise RejectedPose(f"seed {seed}: no acceptable path in {max_draws} draws: {rejected[-3:]}")
    finally:
        if own:
            twin.close()


def load_or_generate(seed: int, *, source_name: str, source_kwargs=None, cache_dir=None,
                     family: str = "viewing_pose", source_info=None, source_factory=None,
                     twin_factory=None, variant: str = ""):
    """Cached target sequence of `seed`, generated and cached on a miss.

    `source_factory()` builds a fresh source and `twin_factory()` the twin, only on a miss.
    Returns `(sequence, stats)`; `stats["cached"]` says whether it came from disk."""
    from .world import build_source

    if family != "viewing_pose":
        raise NotImplementedError(f"target family {family!r} is not generated by this version")
    source = None
    if source_info is None:
        source, source_info = build_source(source_name, source_kwargs)
    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    path = cache_path(cache_dir, family, seed, source_info.name, source_info.weights_sha256,
                      variant)
    if path.exists():
        seq = TargetSequence.load(path)
        if (seq.meta.get("generator_version") == GENERATOR_VERSION
                and seq.meta.get("weights_sha256", "") == source_info.weights_sha256
                and seq.meta.get("source") == source_info.name):
            return seq, {"cached": True, "path": str(path)}
    if source is None:
        source = (source_factory() if source_factory is not None
                  else build_source(source_name, source_kwargs)[0])
    twin = twin_factory() if twin_factory is not None else None
    seq, stats = viewing_pose_sequence(seed, source=source, source_info=source_info, twin=twin)
    seq.meta["variant"] = variant
    seq.save(path)
    stats.update({"cached": False, "path": str(path)})
    return seq, stats


# --------------------------------------------------------------------------- placement stages ---


def from_ideal_sequence(ideal, *, meta: dict | None = None, freeze_after: str = "Align"
                        ) -> TargetSequence:
    """Arrays of an `action.sequence.IdealSequence` (duck-typed: `frames` with `state`,
    `commands`, `action`, `step`, `step_index`; `titles()`).

    Commands are padded to five by repeating the last one; `n_commands` keeps the count.
    `tool_pose[k]` is the last write of frame k-1 (the first write of frame 0 for k = 0).
    `freeze_k` is the first frame after the last `freeze_after` frame."""
    frames = list(ideal.frames)
    T = len(frames)
    states = [f.state for f in frames]
    K, W = states[0].K, states[0].W
    rows, depth, valid, ids = _stack_states(states, K, W)
    commands = np.zeros((T, 5, 7))
    n_commands = np.zeros(T, dtype=np.int64)
    for k, f in enumerate(frames):
        cs = list(f.commands)
        if not cs:
            raise ValueError(f"frame {k} has no commands")
        n_commands[k] = min(len(cs), 5)
        for i in range(5):
            c = cs[min(i, len(cs) - 1)]
            commands[k, i, :3] = np.asarray(c.pos, dtype=float)
            commands[k, i, 3:] = (float(c.spin), float(c.grip), float(c.c_vert), float(c.c_lat))
    action = np.stack([np.asarray(f.action, dtype=float) for f in frames])
    step_index = np.array([int(f.step_index) for f in frames])
    tool_pose = np.zeros((T, 4))
    tool_pose[0] = commands[0, 0, :4]
    for k in range(1, T):
        tool_pose[k] = commands[k - 1, int(n_commands[k - 1]) - 1, :4]
    align = [k for k, f in enumerate(frames) if f.step == freeze_after]
    freeze_k = (max(align) + 1) if align else T
    return TargetSequence(rows=rows, depth=depth, valid=valid, slot_ids=ids, commands=commands,
                          n_commands=n_commands, action=action, step_index=step_index,
                          titles=tuple(ideal.titles()), freeze_k=freeze_k, tool_pose=tool_pose,
                          meta=dict(meta or {}))


def record_placement(level: str, *, cell=(4, 3), source=None, source_info=None, pick: bool = False):
    """Record the placement script in the nominal world through `source` and convert it.

    Imports `brick_builder.action` (the recording script lives there); the environment never
    calls this at run time -- cached sequences are loaded instead."""
    from ...action.place import PlaceConfig, record_ideal

    ideal, world = record_ideal(PlaceConfig(pick=pick), target=tuple(cell), source=source)
    world.close()
    meta = {"family": "pick_and_place" if pick else "placement", "cell": list(cell),
            "level": level, "source": getattr(source_info, "name", "ideal"),
            "weights_sha256": getattr(source_info, "weights_sha256", ""),
            "generator": "record_ideal", "generator_version": GENERATOR_VERSION}
    return from_ideal_sequence(ideal, meta=meta)


# ---------------------------------------------------------------------------------- command ---


def parse_seeds(text: str) -> list[int]:
    """`"0-9,20-49"` -> `[0, ..., 9, 20, ..., 49]`."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    """Generate and cache viewing-pose target sequences."""
    import argparse

    from .world import NominalTwin, build_source

    p = argparse.ArgumentParser(description="Viewing-pose target sequences")
    p.add_argument("--source", default="ideal", choices=["ideal", "learned"])
    p.add_argument("--weights", default=None, help="checkpoint of the learned source (explicit path)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", default="0-9,20-49")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args(argv)

    kwargs: dict = {}
    if args.source == "learned":
        kwargs = {"threads": args.threads}
        if args.weights:
            kwargs["weights"] = args.weights
    _, info = build_source(args.source, kwargs)
    print(f"source {info.name} | weights {info.weights_path or '-'} | sha256 "
          f"{info.weights_sha256 or '-'}", flush=True)
    twin = NominalTwin()
    frame_ms = []
    try:
        for seed in parse_seeds(args.seeds):
            seq, st = load_or_generate(
                seed, source_name=args.source, source_kwargs=kwargs, cache_dir=args.cache_dir,
                source_info=info, source_factory=lambda: build_source(args.source, kwargs)[0],
                twin_factory=lambda: twin)
            cost = "cached" if st["cached"] else f"{st['frame_ms']:.1f} ms per frame"
            if not st["cached"]:
                frame_ms.append(st["frame_ms"])
            print(f"seed {seed} | T {seq.T} | draw {seq.meta.get('draw')} | min references "
                  f"{seq.meta.get('min_references')} | {cost} | {st['path']}", flush=True)
    finally:
        twin.close()
    if frame_ms:
        print(f"generated {len(frame_ms)} | frame cost [ms] mean {np.mean(frame_ms):.1f} "
              f"max {np.max(frame_ms):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
