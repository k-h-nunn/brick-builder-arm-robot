"""`SceneTargetEnv`: reach a target sequence of scene states, one vision tick per step.

Gymnasium id `BrickBuilder/SceneTarget-v0`. One `step` is one vision tick (100 ms): five 20 ms
writes to the servo bus, one wrist frame, one scene-model pass. The observation is the scene
model's slot state, the mechanical state and a window of the target sequence `S*_{k+1..k+H}`;
reward and representation success are computed in the scene model's representation
(`codec.SceneDistance`), physical success against simulator ground truth.

Observation (`Dict`):

| Key | Shape | Contents |
|---|---|---|
| `actor/z`, `z_depth`, `z_valid` | [K, W], [K], [K] | observed slot rows (source order), ordinal depth, non-empty flag |
| `actor/z_matched`, `matched_valid` | [K, W], [K] | observed rows in the order of `S*_{k+1}` |
| `actor/target`, `target_depth`, `target_valid` | [H, K, W], [H, K], [H, K] | `S*_{k+1..k+H}`, last frame repeated |
| `actor/mech` | [18] | joint positions, speeds, loads (servo bus) |
| `actor/tool` | [13] | commanded tool target (x, y, z, yaw, grip), bus-FK tool pose (x, y, z, yaw), accumulated offset (x, y, z, yaw) |
| `actor/feedforward` | [H, 5] | recorded change `a*_{k..k+H-1}` (dx, dy, dz, dyaw, dgrip); zeros with the feed-forward off |
| `actor/progress` | [12] | k/(T-1), step/(n_steps-1), holds/max_hold, step one-hot (9) |
| `actor/image` | [256, 256, 3] | with `include_image` only |
| `privileged/gt_z` | [K, 14] | ground-truth rows of the same frame; zeros when `info["gt_z_available"]` is false |
| `privileged/critic` | [32] | plate displacement (3), held-brick error (3), in-hand offset (4), TCP pose (4), TCP error to the current target pose in the plate frame (4), contacts (2), joint loads (6), damage level (1), level one-hot (5) |

The actor half is what the arm can measure; the privileged half is for a critic, evaluation and
debugging and is dropped at export.

Ground-truth rows: on `ideal` they are the source's own rows. On other sources they come from a
second, ground-truth pass over the same frame (`privileged_gt_z=True`, the default). With
`privileged_gt_z=False` on another source there are none: `gt_z` is zeros, `info["gt_slot_ids"]`
is -1, `transition["gt_z"]` is NaN and `info["gt_z_available"]` is false. The scene model's own
rows are never put in their place.

Action: `Box(-1, 1, (n,))`, the change of the tool-target offset in units of the stage's
per-tick scale (L0a: dx, dy, dz, dyaw at 3 mm, 3 mm, 2 mm, 2 deg; L0b: dx, dy, dyaw at 2 mm,
2 mm, 1.5 deg). With the feed-forward on, the recorded writes follow the progress index
(`feedforward_clock="progress"`, the default: they pause while it holds) or the recording's own
clock (`"recording"`: one frame per tick). `u = 0` with the recording clock is open-loop replay
of the recording, the camera-off control condition.

Stages in this version: L0a `viewing_pose` and L0b `viewing_pose_displaced`. The environment
imports nothing under `brick_builder.action` (see `world.py`).
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...bricks.damage import STATES as DAMAGE_STATES
from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA
from .codec import Calibration, SceneDistance, SlotFrame, calibration_path
from .config import LEVELS, STEP_TITLES, SceneTargetConfig, resolve_level
from .executor import apply_action, tick_commands
from .rules import (
    ProgressState, pose_error, progress_update, step_reward, target_in_plate_frame,
    terminal_reward, tube_width,
)
from .targets import TargetSequence, default_cache_dir, load_or_generate
from .world import (
    NOMINAL_PLATE_XY, WRIST_FULL, NominalTwin, ViewingPoseWorld, build_source, observe,
    plate_displacement, reset_source,
)

__all__ = ["CRITIC_W", "MECH_W", "PROGRESS_W", "SceneTargetEnv", "TOOL_W"]

MECH_W = 18
TOOL_W = 13
PROGRESS_W = 3 + len(STEP_TITLES)
CRITIC_W = 32
GT_W = PLACEHOLDER_SLOT_SCHEMA.W
RES = 256
CRACKED = DAMAGE_STATES.index("cracked")
LEVEL_ORDER = tuple(LEVELS)


class SceneTargetEnv(gym.Env):
    """Target-sequence reaching in the scene model's representation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: SceneTargetConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config if config is not None else SceneTargetConfig()
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        self.cfg = cfg
        stage = cfg.stage
        if cfg.action_space != "tool":
            raise NotImplementedError(
                "action_space='joint' (five joint-position increments) needs the joint executor, "
                "which is not part of this package version; use action_space='tool'")
        if not stage.implemented:
            raise NotImplementedError(
                f"stage {stage.code} ({stage.name}) is declared but not implemented in this version; "
                "implemented: " + ", ".join(f"{s.code} {n}" for n, s in LEVELS.items() if s.implemented))
        if cfg.render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode must be None or 'rgb_array'; got {cfg.render_mode!r}")
        self.render_mode = cfg.render_mode

        self._source, self.source_info = build_source(cfg.source, cfg.source_kwargs, K=cfg.K)
        self._gt_source = None
        if cfg.privileged_gt_z and cfg.source != "ideal":
            self._gt_source, _ = build_source("ideal", {}, K=cfg.K)
        self.gt_available = cfg.source == "ideal" or self._gt_source is not None
        """Whether `gt_z` and `gt_slot_ids` hold ground-truth rows (see the module docstring)."""
        self.schema = getattr(self._source, "schema", PLACEHOLDER_SLOT_SCHEMA)
        self.W = int(self.schema.W)
        self.K = int(cfg.K)
        self.H = int(cfg.horizon)

        self.cache_dir = Path(cfg.cache_dir) if cfg.cache_dir is not None else default_cache_dir()
        calib_dir = Path(cfg.calib_dir) if cfg.calib_dir is not None else self.cache_dir / "calib"
        self.calibration: Calibration | None = None
        self.calibration_file = ""
        cpath = calibration_path(calib_dir, self.source_info.name, self.source_info.weights_sha256,
                                 cfg.stage.family)
        if cpath.exists():
            self.calibration = Calibration.load(cpath, source=self.source_info.name,
                                                weights_sha256=self.source_info.weights_sha256,
                                                W=self.W, family=cfg.stage.family)
            self.calibration_file = str(cpath)
        self.distance = SceneDistance(self.schema, self.calibration)
        self.sigma_obs_mm = (0.0 if self.calibration is None or self.calibration.sigma_obs_mm is None
                             else float(self.calibration.sigma_obs_mm))
        if cfg.tol_mm is not None:
            self.tol_mm, self.tol_source = float(cfg.tol_mm), "config"
        elif self.calibration is not None:
            self.tol_mm, self.tol_source = self.calibration.tol_mm, "calibration"
        else:
            self.tol_mm, self.tol_source = 0.5, "default"

        self._configure_stage()

        K, W, H = self.K, self.W, self.H
        f32 = np.float32
        actor = {
            "z": spaces.Box(-np.inf, np.inf, (K, W), f32),
            "z_depth": spaces.Box(0.0, 1.0, (K,), f32),
            "z_valid": spaces.Box(0.0, 1.0, (K,), f32),
            "z_matched": spaces.Box(-np.inf, np.inf, (K, W), f32),
            "matched_valid": spaces.Box(0.0, 1.0, (K,), f32),
            "target": spaces.Box(-np.inf, np.inf, (H, K, W), f32),
            "target_depth": spaces.Box(0.0, 1.0, (H, K), f32),
            "target_valid": spaces.Box(0.0, 1.0, (H, K), f32),
            "mech": spaces.Box(-np.inf, np.inf, (MECH_W,), f32),
            "tool": spaces.Box(-np.inf, np.inf, (TOOL_W,), f32),
            "feedforward": spaces.Box(-np.inf, np.inf, (H, 5), f32),
            "progress": spaces.Box(0.0, 1.0, (PROGRESS_W,), f32),
        }
        if cfg.include_image:
            actor["image"] = spaces.Box(0, 255, (RES, RES, 3), np.uint8)
        self.observation_space = spaces.Dict({
            "actor": spaces.Dict(actor),
            "privileged": spaces.Dict({
                "gt_z": spaces.Box(-np.inf, np.inf, (K, GT_W), f32),
                "critic": spaces.Box(-np.inf, np.inf, (CRITIC_W,), f32),
            }),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (len(self._dims),), dtype=np.float32)

        self._cache_variant = ("" if cfg.render_samples is None
                               else f"ms{int(cfg.render_samples)}")
        self._world = None
        self._twin = None
        self._targets: dict[int, TargetSequence] = {}
        self._ref_steps: dict[int, np.ndarray] = {}
        self.seq: TargetSequence | None = None
        self.target_stats: dict = {}
        self._reset_state()

    # ------------------------------------------------------------------------------ setup ---

    def _configure_stage(self) -> None:
        cfg = self.cfg
        self.level = resolve_level(cfg.level)
        self._dims = cfg.stage.action_dims
        self._scale = cfg.resolved_scale()
        self._ceiling = cfg.resolved_ceiling()
        self._feedforward = cfg.resolved_feedforward()
        self._max_hold = cfg.resolved_max_hold()

    def _reset_state(self) -> None:
        self.k = 0
        self.holds = 0
        self.blind_holds = 0
        self.settled = 0
        self.lost = 0
        self.ticks = 0
        self.offset = np.zeros(4)
        self.u_prev = np.zeros(len(self._dims))
        self.damage = 0
        self.done = True
        self.failure = ""
        self._z: SlotFrame | None = None
        self._gt: SlotFrame | None = None
        self._rgb: np.ndarray | None = None
        self._decode: np.ndarray | None = None
        self._last_cmd = np.zeros(7)
        self._episode_info: dict = {}

    def _ensure_world(self):
        if self._world is None:
            self._world = ViewingPoseWorld(render_samples=self.cfg.render_samples)
        return self._world

    def _sequence(self, target_seed: int) -> TargetSequence:
        if target_seed not in self._targets:
            seq, stats = load_or_generate(
                target_seed, source_name=self.cfg.source, cache_dir=self.cache_dir,
                family=self.cfg.stage.family, source_info=self.source_info,
                source_factory=self._fresh_target_source, twin_factory=self._target_twin,
                variant=self._cache_variant)
            self._targets[target_seed] = seq
            self.target_stats[target_seed] = {k: v for k, v in stats.items()
                                              if k != "frames_rendered"}
            steps = np.zeros(seq.T)
            for k in range(seq.T - 1):
                steps[k] = self.distance(seq.frame(k + 1), seq.frame(k)).d_mm
            self._ref_steps[target_seed] = steps
        return self._targets[target_seed]

    def _fresh_target_source(self):
        """Fresh source for target recording: same weights, separate slot assigner."""
        src, info = build_source(self.cfg.source, self.cfg.source_kwargs, K=self.cfg.K)
        if info.weights_sha256 != self.source_info.weights_sha256:
            raise RuntimeError("scene-model weights changed on disk during the run")
        return src

    def _target_twin(self):
        if self._twin is None:
            self._twin = NominalTwin(render_samples=self.cfg.render_samples)
        return self._twin

    # ------------------------------------------------------------------------------ reset ---

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        opts = dict(options or {})
        level = resolve_level(opts.get("level", self.level))
        if level != self.level:
            if (LEVELS[level].action_dims != self._dims or not LEVELS[level].implemented
                    or LEVELS[level].family != self.cfg.stage.family):
                raise ValueError(f"cannot switch from {self.level} to {level} in one environment: "
                                 "the action space, the target family or the implementation differs")
            self.cfg = self.cfg.with_overrides(level=level)
            self._configure_stage()
        stage = self.cfg.stage
        world_seed = int(opts["world_seed"]) if "world_seed" in opts else int(
            self.np_random.choice(np.asarray(self.cfg.world_seeds)))
        target_seed = int(opts.get("target_seed", world_seed))
        disp_scale = float(opts.get("displacement_scale", self.cfg.displacement_scale))
        start_mm = float(opts.get("start_offset_mm", self.cfg.start_offset_mm))
        start_deg = float(opts.get("start_offset_deg", self.cfg.start_offset_deg))

        self._reset_state()
        seq = self._sequence(target_seed)
        self.seq = seq
        self._ref_step = self._ref_steps[target_seed]

        if stage.displaced:
            d_xy, d_yaw = plate_displacement(world_seed, disp_scale)
        else:
            d_xy, d_yaw = np.zeros(2), 0.0
        self.plate_offset = np.asarray(d_xy, dtype=float)
        self.plate_yaw = float(d_yaw)

        offset = np.zeros(4)
        if start_mm > 0.0 or start_deg > 0.0:
            th = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            offset[:2] = 1e-3 * start_mm * np.array([math.cos(th), math.sin(th)])
            offset[3] = math.radians(start_deg) * float(self.np_random.choice([-1.0, 1.0]))
        self.offset = offset
        start = seq.tool_pose[0]
        grip = float(seq.commands[0, 0, 4])
        self._start_command = np.array([start[0], start[1], start[2], start[3], grip, 1.0, 1.0])

        world = self._ensure_world()
        world.reset(plate_offset_xy=self.plate_offset, plate_yaw=self.plate_yaw,
                    arm_tcp=start[:3] + offset[:3], arm_yaw=start[3] + offset[3],
                    seed=world_seed, settle_ticks=self.cfg.settle_control_ticks)
        self._last_cmd = self._start_command.copy()
        self._last_cmd[:3] += offset[:3]
        self._last_cmd[3] += offset[3]
        reset_source(self._source)
        if self._gt_source is not None:
            reset_source(self._gt_source)
        # Cleanup of reference cycles left by the previous episode's objects.
        gc.collect()

        timings = self._observe()
        self.damage = world.damage_level()
        self.done = False
        self._episode_info = {
            "world_seed": world_seed, "target_seed": target_seed, "level": self.level,
            "level_code": stage.code, "source": self.source_info.name,
            "weights_sha256": self.source_info.weights_sha256,
            "weights_path": self.source_info.weights_path,
            "plate_offset_mm": float(1e3 * np.linalg.norm(self.plate_offset)),
            "plate_yaw_deg": float(math.degrees(self.plate_yaw)),
            "T": seq.T, "tol_mm": self.tol_mm, "tol_source": self.tol_source,
            "mm_scale": float(self.distance.mm_scale), "calibration_file": self.calibration_file,
            "feedforward": self._feedforward, "feedforward_clock": self.cfg.feedforward_clock,
            "max_ticks": self.cfg.max_ticks(seq.T), "gt_z_available": bool(self.gt_available),
        }
        res = self.distance(self._z, seq.frame(1))
        info = dict(self._episode_info)
        info.update(self._progress_info(res))
        info["slot_ids"] = self._z.slot_ids.copy()
        info["gt_slot_ids"] = self._gt_slot_ids()
        if self.cfg.timings_in_info:
            info["timings"] = timings
        return self._observation(res), info

    # ------------------------------------------------------------------------------- step ---

    def step(self, action):
        if self.done or self.seq is None:
            raise RuntimeError("step() called before reset() or after the episode ended")
        seq, world, cfg = self.seq, self._world, self.cfg
        u = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if u.shape != (len(self._dims),):
            raise ValueError(f"action must have {len(self._dims)} components; got {u.shape}")

        k_before = self.k
        target = seq.frame(k_before + 1)
        d_prev = self.distance(self._z, target).d_mm

        self.offset = apply_action(self.offset, u, dims=self._dims, scale=self._scale,
                                   ceiling=self._ceiling)
        cmds = tick_commands(seq.commands, seq.n_commands, k_before, held=self.holds > 0,
                             offset=self.offset, feedforward=self._feedforward,
                             start_command=self._start_command)
        t0 = time.perf_counter()
        for c in cmds:
            world.command(c[:3], c[3], c[4], c_vert=c[5], c_lat=c[6])
        physics_ms = 1e3 * (time.perf_counter() - t0)
        exec_action = (cmds[-1, :4] - self._last_cmd[:4]).astype(np.float64)
        self._last_cmd = cmds[-1].copy()

        timings = self._observe()
        timings["physics_ms"] = physics_ms
        t0 = time.perf_counter()
        res = self.distance(self._z, target)
        d_next = res.d_mm
        tube = tube_width(float(self._ref_step[min(k_before, seq.T - 1)]),
                          floor_mm=cfg.advance_floor_mm, sigma_obs_mm=self.sigma_obs_mm,
                          tube_scale=cfg.tube_scale)
        state, advanced, held = progress_update(
            ProgressState(self.k, self.holds, self.settled), T=seq.T, d_mm=d_next,
            tube_mm=tube, tol_mm=self.tol_mm,
            play_forward=cfg.feedforward_clock == "recording")
        self.k, self.holds, self.settled = state.k, state.holds, state.settled
        n_matched = int(np.sum(res.matched & target.objects))
        blind = n_matched == 0
        self.blind_holds = (self.blind_holds + int(blind)) if held else 0
        self.lost = self.lost + 1 if (int(np.sum(target.objects)) > 0 and blind) else 0
        timings["codec_ms"] = 1e3 * (time.perf_counter() - t0)

        damage = world.damage_level()
        damage_rise = max(damage - self.damage, 0)
        self.damage = max(self.damage, damage)
        self.ticks += 1

        reward = step_reward(cfg.reward, d_prev=d_prev, d_next=d_next, advanced=advanced, u=u,
                             u_prev=self.u_prev, held=held, damage_rise=damage_rise,
                             tol_mm=self.tol_mm)
        self.u_prev = u.copy()

        success_repr = self.k >= seq.T - 1 and self.settled >= cfg.settle_ticks
        failure = ""
        terminated = truncated = False
        if success_repr:
            terminated = True
        elif self.holds > self._max_hold:
            terminated = True
            failure = ("stalled blind" if self.blind_holds * 2 > self.holds
                       else "stalled with a scene-state error")
        elif self.lost >= cfg.lost_view_ticks:
            terminated, failure = True, "lost view"
        elif self.damage >= CRACKED:
            terminated, failure = True, "damage"
        elif self.ticks >= cfg.max_ticks(seq.T):
            truncated, failure = True, "timeout"
        if terminated or truncated:
            reward += terminal_reward(cfg.reward, success=success_repr, failure=failure)
            self.done = True
            self.failure = failure

        gt_now = self._gt_errors(k_before + 1)
        gt_final = self._gt_errors(seq.T - 1)
        success_gt = gt_final.success()
        info = dict(self._episode_info)
        info.update(self._progress_info(res))
        info.update({
            "advanced": bool(advanced), "held": bool(held), "tube_mm": tube,
            "success_repr": bool(success_repr), "success_gt": bool(success_gt),
            "agree": bool(success_repr == success_gt),
            "gt_tool_err_mm": gt_now.xyz_mm, "gt_yaw_err_deg": gt_now.yaw_deg,
            "gt_final_xy_mm": gt_final.xy_mm, "gt_final_z_mm": gt_final.z_mm,
            "gt_final_yaw_deg": gt_final.yaw_deg,
            "exec_action": exec_action, "offset": self.offset.copy(),
            "failure": failure, "damage_state": DAMAGE_STATES[self.damage],
            "slot_ids": self._z.slot_ids.copy(), "gt_slot_ids": self._gt_slot_ids(),
        })
        if cfg.timings_in_info:
            info["timings"] = timings
        info["transition"] = self._transition(res, u, exec_action, gt_now.xyz_mm,
                                              timings if cfg.timings_in_info else None)
        return self._observation(res), float(reward), bool(terminated), bool(truncated), info

    # ----------------------------------------------------------------------------- pieces ---

    def _observe(self) -> dict:
        """Render the wrist view and run the scene model (and the ground-truth pass)."""
        world = self._world
        t0 = time.perf_counter()
        frame = world.render()
        t1 = time.perf_counter()
        impl = getattr(self._source, "impl", None)
        if self.render_mode == "rgb_array" and impl is not None and hasattr(impl, "feature_maps"):
            fm = impl.feature_maps(frame.rgb, window=WRIST_FULL)
            state = fm["state"]
            self._decode = fm["instance_map"]
        else:
            state = observe(self._source, frame, t=world.t)
            self._decode = np.asarray(frame.instance) if self.render_mode == "rgb_array" else None
        t2 = time.perf_counter()
        self._z = SlotFrame.from_state(state)
        if self._gt_source is not None:
            self._gt = SlotFrame.from_state(observe(self._gt_source, frame, t=world.t))
        elif self.gt_available:
            self._gt = self._z  # the `ideal` source's rows are the ground truth
        else:
            self._gt = None  # no ground-truth pass; never substitute the scene model's rows
        t3 = time.perf_counter()
        self._rgb = np.asarray(frame.rgb)
        return {"render_ms": 1e3 * (t1 - t0), "source_ms": 1e3 * (t2 - t1),
                "gt_ms": 1e3 * (t3 - t2)}

    def _gt_slot_ids(self) -> np.ndarray:
        if self._gt is None:
            return np.full(self.K, -1, dtype=np.int64)
        return self._gt.slot_ids.copy()

    def _gt_errors(self, k: int):
        pos, yaw = self._world.tool_pose_truth()
        return pose_error(pos, yaw, self.seq.tool_pose[min(max(k, 0), self.seq.T - 1)],
                          plate_nominal_xy=NOMINAL_PLATE_XY, plate_offset_xy=self.plate_offset,
                          plate_yaw=self.plate_yaw)

    def _progress_info(self, res) -> dict:
        step = int(self.seq.step_index[min(self.k, self.seq.T - 1)])
        titles = self.seq.titles
        return {"k": int(self.k), "T": int(self.seq.T), "ticks": int(self.ticks),
                "step": titles[step] if step < len(titles) else "",
                "holds": int(self.holds), "d_mm": float(res.d_mm),
                "d_abs_mm": res.parts["abs"], "d_rel_mm": res.parts["rel"],
                "d_missing_mm": res.parts["missing"], "n_matched": int(np.sum(res.matched))}

    def _window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq = self.seq
        idx = [min(self.k + 1 + h, seq.T - 1) for h in range(self.H)]
        return seq.rows[idx], seq.depth[idx], seq.valid[idx].astype(np.float32)

    def _feedforward_window(self) -> np.ndarray:
        out = np.zeros((self.H, 5), dtype=np.float32)
        if not self._feedforward:
            return out
        ff = self.seq.feedforward()
        for h in range(self.H):
            out[h] = ff[min(self.k + h, self.seq.T - 1)]
        return out

    def _progress(self) -> np.ndarray:
        seq = self.seq
        out = np.zeros(PROGRESS_W, dtype=np.float32)
        out[0] = self.k / max(seq.T - 1, 1)
        step = int(seq.step_index[min(self.k, seq.T - 1)])
        out[1] = step / max(len(seq.titles) - 1, 1)
        out[2] = min(self.holds / max(self._max_hold, 1), 1.0)
        title = seq.titles[step] if step < len(seq.titles) else ""
        if title in STEP_TITLES:
            out[3 + STEP_TITLES.index(title)] = 1.0
        return np.clip(out, 0.0, 1.0)

    def _tool(self) -> np.ndarray:
        fb = self._world.feedback(float(self._last_cmd[4]))
        return np.concatenate([self._last_cmd[:5], np.asarray(fb.tcp_pos, dtype=np.float64)[:3],
                               [float(fb.tcp_spin)], self.offset]).astype(np.float32)

    def _critic(self) -> np.ndarray:
        out = np.zeros(CRITIC_W, dtype=np.float32)
        world = self._world
        out[0:2] = self.plate_offset
        out[2] = self.plate_yaw
        # 3:6 held-brick error and 6:10 in-hand offset stay zero on the viewing-pose stages.
        pos, yaw = world.tool_pose_truth()
        out[10:13] = pos
        out[13] = yaw
        tgt = target_in_plate_frame(self.seq.tool_pose[min(self.k + 1, self.seq.T - 1)],
                                    NOMINAL_PLATE_XY, self.plate_offset, self.plate_yaw)
        out[14:17] = np.asarray(pos) - tgt[:3]
        out[17] = math.remainder(yaw - tgt[3], 2.0 * math.pi)
        out[18:20] = world.contact_flags()
        out[20:26] = np.asarray(world.driver.reading.load, dtype=np.float32)[:6]
        out[26] = self.damage / (len(DAMAGE_STATES) - 1)
        out[27 + LEVEL_ORDER.index(self.level)] = 1.0
        return out

    def _observation(self, res) -> dict:
        z = self._z
        zm, _, ok = self.distance.align(z, self.seq.frame(self.k + 1))
        tr, td, tv = self._window()
        actor = {
            "z": z.rows.astype(np.float32), "z_depth": z.depth.astype(np.float32),
            "z_valid": z.valid.astype(np.float32),
            "z_matched": zm.astype(np.float32), "matched_valid": ok.astype(np.float32),
            "target": tr.astype(np.float32), "target_depth": td.astype(np.float32),
            "target_valid": tv.astype(np.float32),
            "mech": self._world.bus_state().astype(np.float32),
            "tool": self._tool(),
            "feedforward": self._feedforward_window(),
            "progress": self._progress(),
        }
        if self.cfg.include_image:
            actor["image"] = np.ascontiguousarray(self._rgb[..., :3], dtype=np.uint8)
        gt = np.zeros((self.K, GT_W), dtype=np.float32)
        if self._gt is not None and self._gt.rows.shape == gt.shape:
            gt[:] = self._gt.rows
        return {"actor": actor, "privileged": {"gt_z": gt, "critic": self._critic()}}

    def _transition(self, res, u, exec_action, gt_err_mm, timings) -> dict:
        z = self._z
        zm, _, ok = self.distance.align(z, self.seq.frame(self.k + 1))
        rec = {
            "z": z.rows.copy(), "depth": z.depth.copy(), "valid": z.valid.copy(),
            "slot_ids": z.slot_ids.copy(), "z_matched": zm, "matched_valid": ok,
            "k": int(self.k), "step_index": int(self.seq.step_index[min(self.k, self.seq.T - 1)]),
            "held": int(self.holds > 0),
            "mech": self._world.bus_state(), "tool": self._tool().astype(np.float64),
            "exec_action": exec_action, "u": np.asarray(u, dtype=np.float64).copy(),
            "grip": float(self._last_cmd[4]),
            "compliance": np.asarray(self._last_cmd[5:7], dtype=np.float64).copy(),
            "d_mm": float(res.d_mm),
            "gt_z": (self._gt.rows.copy() if self._gt is not None
                     else np.full((self.K, GT_W), np.nan, dtype=np.float32)),
            "gt_z_available": bool(self._gt is not None),
            "gt_tool_err_mm": float(gt_err_mm),
        }
        if timings is not None:
            rec["timings"] = dict(timings)
        return rec

    # ------------------------------------------------------------------------------ misc ---

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return None if self._rgb is None else np.asarray(self._rgb[..., :3], dtype=np.uint8).copy()

    @property
    def decode(self) -> np.ndarray | None:
        """The last frame's dense scene-model output (instance map) with `render_mode='rgb_array'`."""
        return self._decode

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None
        if self._twin is not None:
            self._twin.close()
            self._twin = None
        gc.collect()
