"""A Gymnasium environment around the closed-loop placement task.

One environment step is one vision tick of `action.place.TrackingEpisode` -- five 20 ms control
writes, one wrist frame, one reading, one correction update -- so the environment runs at the
10 Hz the controller acts at and at the same cost per step the published tables were measured at.

The action is a **residual** on the analytic controller, not a replacement for it:

    correction <- correction + G_analytic(observed, target) + residual

with the residual clipped to the loop's own per-tick saturation and the sum held to the loop's own
ceiling. A zero action therefore reproduces today's controller tick for tick, which is what
`tests/rl/vision_control/test_env.py` asserts to 1e-9; a trained policy is today's
controller plus whatever it has learned to add. The residual is applied only while the loop's own
correction is live -- not after the freeze -- which is the same gate the collector's probe uses.

The observation is the raw 256x256 wrist frame plus the mechanical state, the target scene state
for this tick and the progress index. Nothing decoded, nothing from the simulator: the policy's
own scene model is what turns the frame into slot rows, and a different source registered through
`brick_builder.perception.sources` can take that place without this file changing. The critic's
half of the observation *is* privileged -- the plate's real displacement, the held brick's real
error against the target cell, the damage state -- and is dropped at export, as in
`brick_builder/rl/ppo_asym.py`.

Episodes can start part-way along the target sequence (`start_step="Align"`) and end when the
loop freezes its correction (`stop_when_frozen`): the prefix is run with a zero residual and the
tail is played out with one, and neither is part of the episode. Together they keep a training
episode to the forty-odd steps the action can do something in, out of about two hundred and
twenty, while the terminal reward is still the placement the whole sequence ended with.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Iterator

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...action.place import PlaceConfig, TickRecord, TrackingEpisode, record_ideal
from ...action.worlds import SUPPLY_BODY, build_pick_world, build_world
from ...bricks.damage import STATES as DAMAGE_STATES
from ...control.pick_place import verify_placement
from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA

__all__ = ["VisionControlConfig", "VisionControlEnv", "make_env"]

RES = 256
"""Side of the wrist window the arm camera publishes, in pixels."""

K_SLOTS = 8
W_SLOT = PLACEHOLDER_SLOT_SCHEMA.W
MECH_W = 12
"""Six joint positions and six joint speeds, as the servo bus reports them -- the same mechanical
state the learned inverse reads."""

PROGRESS_W = 3
CRITIC_W = 30

MAX_DAMAGE = len(DAMAGE_STATES) - 1


@dataclass(frozen=True)
class VisionControlConfig:
    """Task, episode bounds and reward weights. Distances in millimetres unless a name says else."""

    task: str = "placement"
    """`placement` (the brick starts in the hand) or `pick` (the whole pick-and-place)."""

    start_step: str | None = "Align"
    """First step of the episode. The sequence before it is played with a zero residual and is not
    scored; `None` starts at the first frame. `Align` is where the placement task's feedback does
    its work, and starting there takes a placement episode from about 220 vision ticks to about
    185; `stop_when_frozen` takes it the rest of the way, to about 40."""

    seeds: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    """The seed pool an episode is drawn from. Seeds 0-9 are the published evaluation table's and
    are deliberately absent, so a trained policy is measured on worlds it never trained in."""

    stop_when_frozen: bool = True
    """End the episode when the loop freezes its correction, and play the rest of the sequence out
    with no residual before scoring it.

    The residual can only do something while the correction is live: past the freeze the loop
    deliberately stops moving the estimate and `_apply` is a no-op, so asking the policy for an
    action there is asking it to act into a hundred and fifty ticks of dead time and then take the
    blame for the outcome. With this on the episode is the forty-odd ticks the action matters for,
    and the terminal reward is still the real placement -- the press, the release and the retreat
    are run out and `verify_placement` scores what actually ended up on the plate."""

    max_ticks: int = 220
    """Vision ticks before the episode truncates. The placement sequence is about 220 and about 185
    from the align dwell, so this only fires on a run that is holding its progress index."""

    residual_xy: float = 0.0012
    """Largest residual in x and y per tick [m] -- the loop's own per-tick saturation."""

    residual_spin: float = math.radians(0.7)
    """Largest residual in spin per tick [rad] -- the loop's own."""

    w_progress: float = 1.0
    """Reward per millimetre the tracked scene-state error falls in one tick."""

    w_action: float = 0.05
    """Penalty on the action's size, in units of the saturation (so a full-scale action costs
    `w_action` per axis)."""

    w_seated: float = 10.0
    """Paid once, at the end, when the brick is seated on the target cell."""

    w_lateral: float = 2.0
    """Paid once, at the end, as `w_lateral * max(0, 1 - lateral_mm)`, so a seated run that is
    nearer the cell centre is worth more than one that only just counts."""

    w_damage: float = 5.0
    """Penalty per step of the damage monitor's own state ladder (`bricks.damage.STATES`) that the
    run climbs. Reading the monitor is reading the simulator, so it is a reward term and never an
    observation the actor sees."""

    w_stall: float = 0.02
    """Penalty per tick the progress index is held."""

    w_failure: float = 5.0
    """Paid once, at the end, when the episode ended in a stall or a failed grasp."""

    controller_source: str = "ideal"
    """Which scene model drives the **controller** underneath the residual: `ideal` is the
    ground-truth source the published tables were measured with, `conv` the trained convolutional
    one. The policy's own view of the world is the raw frame either way."""

    render_mode: str | None = None


def make_env(seed: int = 0, cfg: VisionControlConfig | None = None) -> "VisionControlEnv":
    """One environment, seeded -- the factory `ppo_asym.make_vector_env` wants."""
    return VisionControlEnv(cfg=cfg or VisionControlConfig(), seed=seed)


class VisionControlEnv(gym.Env):
    """One closed-loop episode per `reset`, one vision tick per `step`."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: VisionControlConfig | None = None, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg or VisionControlConfig()
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

        self.observation_space = spaces.Dict({
            "actor": spaces.Dict({
                "image": spaces.Box(0, 255, (RES, RES, 3), dtype=np.uint8),
                "mech": spaces.Box(-np.inf, np.inf, (MECH_W,), dtype=np.float32),
                "target": spaces.Box(-np.inf, np.inf, (K_SLOTS, W_SLOT), dtype=np.float32),
                "progress": spaces.Box(0.0, 1.0, (PROGRESS_W,), dtype=np.float32),
            }),
            "critic": spaces.Box(-np.inf, np.inf, (CRITIC_W,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        """dx, dy and spin, in units of the loop's own per-tick saturation.

        Three, not four: the controller has no z correction channel -- it commands the recorded
        height unchanged -- and neither has the learned inverse, whose
        `models.features.INVERSE_OUT` has no z column. A fourth action that nothing could apply
        would be a dead dimension, so the residual has exactly the three the loop commands."""

        self._ideal_cache: dict[int, object] = {}
        self._episode: TrackingEpisode | None = None
        self._gen: Iterator[TickRecord] | None = None
        self._scene = None
        self._brick = ""
        self._prev_err = float("nan")
        self._damage = 0
        self._ticks = 0
        self._last: TickRecord | None = None
        self._prev_cmd = np.zeros(4)
        self._world_info: dict = {}
        self._done_info: dict = {}
        self._source = None

    # -- episode setup -----------------------------------------------------------------------

    def _place_config(self) -> PlaceConfig:
        return PlaceConfig(pick=self.cfg.task == "pick")

    def _ideal(self, seed: int):
        """The target sequence for this seed, recorded once per process and reused.

        Recording runs the task in the nominal world and costs as much as an episode, so a
        training run that recorded one per reset would spend half its budget on the reference.
        It depends on the task and the driver's own noise, not on the world the episode runs in.
        """
        if seed not in self._ideal_cache:
            ideal, nominal = record_ideal(self._place_config(), seed=seed, source=self._make_source())
            nominal.close()
            self._ideal_cache[seed] = ideal
        return self._ideal_cache[seed]

    def _make_source(self):
        if self.cfg.controller_source == "ideal":
            return None
        from ...perception.cnn.source import CnnSceneModel

        if self._source is None:
            self._source = CnnSceneModel(build_decode=False, threads=1)
        self._source.reset()
        return self._source

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.close_episode()

        world_seed = int((options or {}).get("world_seed", self._rng.choice(self.cfg.seeds)))
        cfg = self._place_config()
        ideal = self._ideal(world_seed)
        if self.cfg.task == "pick":
            scene, d_xy, d_yaw, b_xy, _ = build_pick_world(world_seed)
            self._brick = SUPPLY_BODY
        else:
            scene, d_xy, d_yaw = build_world(world_seed)
            self._brick = scene.held_name
            b_xy = np.zeros(2)
        self._scene = scene
        self._world_info = {"world_seed": world_seed,
                            "plate_offset_mm": float(1e3 * np.linalg.norm(d_xy)),
                            "plate_yaw_deg": float(math.degrees(d_yaw)),
                            "brick_offset_mm": float(1e3 * np.linalg.norm(b_xy))}

        self._episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=world_seed,
                                        source=self._make_source())
        self._gen = self._episode.run()
        self._ticks = 0
        self._damage = 0
        self._done_info = {}
        self._last = self._advance()
        if self._last is not None and self.cfg.start_step is not None:
            while self._last is not None and self._last.step != self.cfg.start_step:
                self._last = self._advance()
        self._prev_err = float("nan") if self._last is None else self._last.err_mm
        self._prev_cmd = (np.zeros(4) if self._last is None
                          else np.array([self._last.cmd_pos[0], self._last.cmd_pos[1],
                                         self._last.cmd_pos[2], self._last.spin]))
        return self._observation(), {**self._world_info, "tool_delta": np.zeros(4, dtype=np.float32)}

    def close_episode(self) -> None:
        if self._episode is not None:
            self._episode.close()
        self._episode, self._gen, self._scene = None, None, None
        self._last = None
        # One episode builds three simulator models and three render contexts. They are closed,
        # but they sit in reference cycles the generational collector would only reach much
        # later, and a training run of a few hundred episodes reaches several gigabytes before it
        # does -- measured, it was killed by the machine at about forty episodes per process.
        # `action/learn/collect.py` collects here for the same reason.
        gc.collect()

    def close(self) -> None:
        self.close_episode()

    # -- the tick ----------------------------------------------------------------------------

    def _advance(self) -> TickRecord | None:
        """Run the loop forward to the next vision tick, or `None` when the episode is over."""
        if self._gen is None:
            return None
        for rec in self._gen:
            if rec.vision:
                return rec
        self._gen = None
        return None

    def _apply(self, action: np.ndarray) -> np.ndarray:
        """Add the residual to the correction the analytic inverse has just set.

        Clipped to the loop's own per-tick saturation and then held to the loop's own ceiling, so
        the residual can never command something the analytic term could not have commanded. Not
        applied once the correction is frozen: past that point the arm is leaning on something and
        the loop deliberately stops moving the estimate.
        """
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        residual = np.array([a[0] * self.cfg.residual_xy, a[1] * self.cfg.residual_xy,
                             a[2] * self.cfg.residual_spin])
        ep = self._episode
        if ep is None or ep.frozen:
            return residual * 0.0
        corr = np.asarray(ep.correction, dtype=float) + residual
        n = float(np.linalg.norm(corr[:2]))
        if n > ep.cfg.max_correction:
            corr[:2] *= ep.cfg.max_correction / n
        ep.correction = corr
        return residual

    def step(self, action):
        if self._episode is None:
            raise RuntimeError("step() before reset()")
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        self._apply(a)

        rec = self._advance()
        self._ticks += 1
        finished = rec is None
        if not finished:
            self._last = rec
        if not finished and self.cfg.stop_when_frozen and self._episode.frozen:
            self._play_out()
            finished = True

        err = float("nan") if self._last is None else self._last.err_mm
        damage = _damage_level(self._scene)
        reward = step_reward(self.cfg, prev_err_mm=self._prev_err, err_mm=err, action=a,
                             holding=(not finished) and self._episode.holds > 0,
                             damage_rise=max(damage - self._damage, 0),
                             scored=finished)
        if err == err:
            self._prev_err = err
        self._damage = max(self._damage, damage)

        truncated = (not finished) and self._ticks >= self.cfg.max_ticks
        terminated = finished
        if terminated or truncated:
            reward += self._score()
        info = {**self._world_info, **self._done_info, "ticks": self._ticks,
                "tool_delta": self._tool_delta()}
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def _tool_delta(self) -> np.ndarray:
        """`[4]`: the tool target's own change over this tick, (dx, dy, dz [m], spin [rad]).

        What the world saw, feed-forward plus correction -- the same quantity the tuple log calls
        the action and the predictor `F` reads."""
        rec = self._last
        if rec is None:
            return np.zeros(4, dtype=np.float32)
        now = np.array([rec.cmd_pos[0], rec.cmd_pos[1], rec.cmd_pos[2], rec.spin])
        out = (now - self._prev_cmd).astype(np.float32)
        self._prev_cmd = now
        return out

    def _play_out(self) -> None:
        """Run the rest of the sequence with no residual, so the placement can be scored.

        Nothing is asked of the policy here and nothing is added to the correction: the loop is
        already frozen, so these ticks are exactly the ones today's controller would run.
        """
        while self._advance() is not None:
            pass

    def _score(self) -> float:
        """The terminal reward, and the scored result the info carries.

        `verify_placement` reads the simulator, which is why it is here and not in the loop.
        """
        ep, scene = self._episode, self._scene
        check = verify_placement(scene, self._brick, (4.5, 3.5, 0))
        seated = bool(check.seated) and bool(check.at_target_cell)
        errs = [e for e in ep.errors if e == e]
        preds = [p for p in ep.pred_errors if p == p]
        self._done_info = {
            "success": seated,
            "seated": bool(check.seated),
            "at_target_cell": bool(check.at_target_cell),
            "lateral_mm": float(check.lateral_mm),
            "seat_depth_error": float(check.depth_error_mm),
            "damage_state": DAMAGE_STATES[self._damage],
            "failure": ep.failure or "",
            "median_tracking_mm": float(np.median(errs)) if errs else float("nan"),
            "median_prediction_mm": float(np.median(preds)) if preds else float("nan"),
            "vision_ticks": len(ep.errors),
            "structure_intact": self._damage == 0,
            "params": dict(self._world_info),
        }
        reward = 0.0
        if seated:
            reward += self.cfg.w_seated
            reward += self.cfg.w_lateral * max(0.0, 1.0 - float(check.lateral_mm))
        if ep.failure:
            reward -= self.cfg.w_failure
        return reward

    # -- what the two halves see --------------------------------------------------------------

    def _observation(self) -> dict:
        rec, ep = self._last, self._episode
        image = np.zeros((RES, RES, 3), dtype=np.uint8)
        if rec is not None and rec.rgb is not None:
            frame = np.asarray(rec.rgb)
            image = (frame if frame.dtype == np.uint8
                     else np.clip(frame * 255.0, 0, 255).astype(np.uint8))[..., :3]
        target = np.zeros((K_SLOTS, W_SLOT), dtype=np.float32)
        if rec is not None and rec.target is not None:
            target = np.asarray(rec.target.slots, dtype=np.float32)
        mech = np.zeros(MECH_W, dtype=np.float32)
        if ep is not None:
            mech = np.concatenate([ep.world.joint_position,
                                   ep.world.joint_speed]).astype(np.float32)[:MECH_W]
        return {"actor": {"image": image, "mech": mech, "target": target,
                          "progress": self._progress()},
                "critic": self._critic()}

    def _progress(self) -> np.ndarray:
        """`[k / (T - 1), step_index / (n_steps - 1), holds / max_hold]`, each clipped to `[0, 1]`."""
        ep = self._episode
        if ep is None:
            return np.zeros(PROGRESS_W, dtype=np.float32)
        frames = ep.ideal.frames
        titles = ep.ideal.titles()
        step_index = 0 if self._last is None else self._last.step_index
        return np.clip(np.array([
            ep.k / max(len(frames) - 1, 1),
            step_index / max(len(titles) - 1, 1),
            ep.holds / max(ep.cfg.max_hold, 1),
        ], dtype=np.float32), 0.0, 1.0)

    def _critic(self) -> np.ndarray:
        """The privileged half: what the simulator knows and the arm does not.

        Dropped at export with the critic that reads it. Kept deliberately small and cheap --
        nothing here renders or re-scores a placement, which would cost more than the tick.
        """
        out = np.zeros(CRITIC_W, dtype=np.float32)
        ep, scene = self._episode, self._scene
        if ep is None or scene is None:
            return out
        plate = np.asarray(scene.cfg.plate_xy, dtype=np.float64)
        pos, _ = scene.body_pose(self._brick)
        tool = np.asarray(ep.world.feedback(0.0).tcp_pos, dtype=np.float64)
        err = float("nan") if self._last is None else self._last.err_mm
        out[:2] = plate
        out[2] = float(scene.cfg.plate_yaw)
        out[3:6] = np.asarray(pos, dtype=np.float32)
        out[6:9] = tool.astype(np.float32)
        out[9:12] = np.asarray(ep.correction, dtype=np.float32)
        out[12] = 0.0 if err != err else float(err)
        out[13] = float(self._damage) / MAX_DAMAGE
        out[14] = float(ep.frozen)
        out[15] = float(ep.trusted)
        out[16] = float(ep.gain)
        out[17:20] = self._progress()
        out[20:26] = np.asarray(ep.world.joint_load, dtype=np.float32)[:6]
        out[26:30] = np.asarray(
            [ep.dwelling, ep.blind_frames / max(len(ep.errors), 1), ep.lag, ep.regrasps],
            dtype=np.float32)
        return out

    def render(self):
        return None if self._last is None else self._last.rgb


def step_reward(cfg: VisionControlConfig, *, prev_err_mm: float, err_mm: float,
                action: np.ndarray, holding: bool, damage_rise: int, scored: bool) -> float:
    """The per-tick reward, as a function of numbers alone.

    Split out of `step` so the signs can be checked on synthetic sequences without a simulator
    (`tests/rl/vision_control/test_reward.py`). The terminal bonus is not here: it needs the
    scored placement, which is `_score`'s business.

    | Term | Sign | Units |
    |---|---|---|
    | Progress along the sequence | + when the tracked error falls | `w_progress` per mm |
    | Action size | - | `w_action` per unit of saturation |
    | Holding the progress index | - | `w_stall` per tick |
    | Damage state rising | - | `w_damage` per step of the ladder |
    """
    reward = -cfg.w_action * float(np.abs(np.asarray(action, dtype=np.float64)).mean())
    if not scored and err_mm == err_mm and prev_err_mm == prev_err_mm:
        reward += cfg.w_progress * (prev_err_mm - err_mm)
    if holding:
        reward -= cfg.w_stall
    if damage_rise > 0:
        reward -= cfg.w_damage * float(damage_rise)
    return float(reward)


def _damage_level(scene) -> int:
    """The damage monitor's own state as a level in `bricks.damage.STATES`, 0 = intact."""
    monitor = getattr(scene, "monitor", None)
    if monitor is None:
        return 0
    try:
        return int(DAMAGE_STATES.index(monitor.state))
    except (ValueError, AttributeError):
        return 0
