"""``BrickBuilder/AlignFromTokens-v0``: bring the TCP above a table-brick using only vision tokens.

A single brick sits loosely near the plate's centre, at a continuous, randomized (x, y, yaw). The
policy never sees its true pose -- only an :class:`~brick_builder.perception.tokens.ObjectToken`
produced by one of the three interchangeable :mod:`brick_builder.perception.producers`, refreshed only
while the arm sits at a fixed ``LOOK`` pose and turned into a **metric**
world position by :mod:`brick_builder.perception.backproject` -- the pixel-to-position step every
producer's raw output needs before it means anything to a controller.

Where the target actually is
-----------------------------
The spec's own wording is "a brick lies on the table"; what is built here is a brick lying loose
on top of the baseplate instead, near its centre, and that is a deliberate, measured deviation
documented in the guide, not a shortcut: the SO-101's wrist camera is
mounted a few centimetres from the fixed jaw with a fairly narrow field of view, and the *reachable
tool-down envelope* for this arm has a dead zone directly over the plate between roughly 9 cm and
30 cm of hover height (checked by a grid IK sweep, not assumed) -- the only place a tool-down wrist
view actually shows the plate clearly is a low hover a few centimetres above it. The baseplate
already occupies that reachable, visible neighbourhood, so the target brick rests there (bare-table
resting height would need a wrist pose this arm and camera cannot deliver a wide, well-behaved view
from). The brick is not on the stud grid -- it is teleported to a continuous pose overlapping the
plate, in the studs-block-it "resting on top of the studs" regime `bricks/generator.py` describes,
not a snapped, grid-aligned placement.

The LOOK pose and refresh rule
-------------------------------
``LOOK_TCP`` is a fixed tool-down pose the arm starts every episode at (`SceneConfig.home_tcp`). A
token is captured there at reset (age 0) and again whenever the arm's *read* TCP pose (not
simulator truth -- the same signal the actor observes) comes back within ``LOOK_TOL_POS``/
``LOOK_TOL_ANG`` of ``LOOK_TCP``; otherwise the last token is kept and its age grows by one control
tick. The scripted primitive is `brick_builder.control.primitives.MoveTo`, re-aimed (and re-started from
wherever the tool currently is, at the current speed and heading) at the *current* token estimate
every time a fresh one arrives -- a straight-line ramp, never an instant jump, which is both what a
real position servo needs and what :func:`~brick_builder.envs.base_env.ArmEnv`'s own per-tick IK
warm-start assumes ("the target moves by at most a couple of millimetres/degrees a tick"). The
target height is the brick's assumed resting height (``self._table_z``, a fixed calibration
constant a real system would also have to assume, not simulator truth). With the residual scale at
zero this is exactly the "scripted go-to-token controller" the guide measures against token
noise, age and producer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from brick_builder.bricks.spec import BrickSpec
from brick_builder.bricks.structures import Placement
from brick_builder.control.primitives import Feedback, MoveTo, feedback_from_readings
from brick_builder.envs import obs_spec
from brick_builder.envs.base_env import CONTROL_DT, ArmEnv, PrimitiveTarget
from brick_builder.perception.backproject import (
    CameraGeometry,
    backproject_to_plane,
    camera_geometry,
    project_point,
)
from brick_builder.perception.producers import (
    BlobProducer,
    GroundTruthObject,
    GroundTruthProducer,
    ObjectToken,
    PALETTE,
)
from brick_builder.robot.kinematics import euler_zyx
from brick_builder.robot.patch_so101 import WRIST_CAM, gripper_angle_for_width
from brick_builder.scene.compose import SceneConfig, grasp_pose

SUCCESS_TOL_POS = 0.003             # m
SUCCESS_TOL_YAW = math.radians(5.0)
SUCCESS_TICKS = 10

LOOK_TCP = (0.19, -0.01, 0.075)      # tool-down; see the module docstring for why here
LOOK_YAW = 0.0
LOOK_TOL_POS = 0.006                # m; wide enough for the arm's own ~5 mm settling deadband
LOOK_TOL_ANG = math.radians(5.0)

SEARCH_RADIUS = 0.010                # m; the brick's (x, y) is drawn in a box this size around LOOK_TCP
MOVE_SPEED = 0.05                    # m/s; the scripted MoveTo primitive's ramp speed
SPIN_RATE = math.radians(180.0)      # rad/s; how fast the commanded tool spin is allowed to ramp
RETARGET_POS_TOL = 0.0015            # m; a token re-estimate closer than this does not reset the ramp
RETARGET_ANG_TOL = math.radians(3.0)  # rad
BRICK_NX, BRICK_NY = 2, 4            # studs across (jaw axis) x along (length)
BRICK_GRID_CELL = (3, 2)             # (x, y) footprint placement used only to create the body/joint

RENDER_W, RENDER_H = 160, 120        # wrist-cam capture size for the blob producer

_ALIGN_TOKEN = obs_spec.register(obs_spec.ObsEntry(
    "align_token", 5, "token", "actor",
    note="x, y (m, relative to LOOK_TCP's x/y), yaw (rad), confidence, age (s) of the last "
    "backprojected token; zeros with confidence 0 before the first reading ever arrives",
))
_ALIGN_TIME_FRAC = obs_spec.register(obs_spec.ObsEntry(
    "align_time_frac", 1, "task", "actor", note="tick / horizon",
))
_ALIGN_TRUE_ERROR = obs_spec.register(obs_spec.ObsEntry(
    "align_true_error", 2, "sim", "critic",
    note="true TCP distance to the grasp-ready pose (m) and true yaw error (rad, brick's "
    "180-degree grasp symmetry already folded in)",
))


def _tool_down_rot(spin: float) -> np.ndarray:
    """Tool-down rotation matrix (pitch 0, roll pi) at spin ``spin``, in closed form.

    Same convention as ``scene.compose``/``base_env`` (roll = pi is tool-down): with pitch fixed
    at 0, ``Rz(spin) @ Ry(0) @ Rx(pi)`` collapses to ``Rz(spin) @ diag(1, -1, -1)`` -- written out
    directly rather than composed from three general rotation matrices, since this env only ever
    needs this one, fixed orientation family.
    """
    c, s = math.cos(spin), math.sin(spin)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _wrap_symmetric(angle: float, period: float) -> float:
    """``angle`` folded into ``(-period/2, period/2]``."""
    return float(math.remainder(angle, period))


@dataclass
class _Token:
    x: float
    y: float
    yaw: float
    confidence: float


class AlignFromTokensEnv(ArmEnv):
    """``BrickBuilder/AlignFromTokens-v0``. Horizon 150 ticks; success within 3 mm / 5 deg for 10 ticks.

    ``producer`` selects which :mod:`brick_builder.perception.producers` source builds the token
    this episode: ``"surrogate"`` (:class:`GroundTruthProducer`, the default -- fast, used for bulk
    measurement) or ``"blob"`` (:class:`BlobProducer`, classical colour thresholding on the
    rendered wrist frame). ``token_noise_mm`` only affects ``"surrogate"``: it is converted to a
    pixel jitter using this env's own LOOK-pose camera geometry (:meth:`_px_per_m`), so "noise"
    always means the same physical thing regardless of render resolution.
    """

    frozen = frozenset({"grip", "compliance"})

    def __init__(
        self,
        *,
        horizon: int = 150,
        seed: int | None = None,
        producer: str = "surrogate",
        colour: str = "blue",
        token_noise_mm: float = 4.0,
        yaw_jitter_rad: float = 0.05,
        dropout_p: float = 0.02,
    ) -> None:
        if producer not in ("surrogate", "blob"):
            raise ValueError(f"unknown producer {producer!r}; must be surrogate or blob")
        if colour not in PALETTE:
            raise ValueError(f"unknown colour {colour!r}; must be one of {sorted(PALETTE)}")
        self.producer_name = producer
        self.colour = colour
        self._token_noise_mm = float(token_noise_mm)
        self._yaw_jitter_rad = float(yaw_jitter_rad)
        self._dropout_p = float(dropout_p)
        self._renderer = None

        super().__init__(horizon=horizon, seed=seed)

        self._grasp = grasp_pose(BRICK_NX, BRICK_NY, self.scene.spec)
        self._table_z = float(self.scene.plate_top + self.scene.spec.stud_h)
        self._brick_name = self.scene.placed_names[0]
        self._brick_free_adr = self.scene.free_joint_adr(self._brick_name)
        self._open_grip = gripper_angle_for_width(0.05)
        self._look_tcp = np.asarray(LOOK_TCP, dtype=float)

        self._px_per_m = self._measure_px_per_m()
        self._producer = self._build_producer()

    # -- scene ------------------------------------------------------------------------------

    def scene_config(self, rng: np.random.Generator) -> SceneConfig:
        placement = Placement(
            h=BRICK_NY, w=BRICK_NX, x=BRICK_GRID_CELL[0], y=BRICK_GRID_CELL[1], z=0, colour=self.colour
        )
        return SceneConfig(
            held=None, placed=[placement], cameras=(WRIST_CAM,),
            home_tcp=LOOK_TCP, home_yaw=LOOK_YAW,
        )

    def _build_producer(self):
        if self.producer_name == "blob":
            return BlobProducer()
        return GroundTruthProducer(
            jitter_std_px=self._token_noise_mm * 1e-3 * self._px_per_m,
            yaw_jitter_rad=self._yaw_jitter_rad,
            dropout_p=self._dropout_p,
            merge_radius_px=1.0,  # only one ground-truth object ever exists in this scene
        )

    def _measure_px_per_m(self) -> float:
        """Pixels per metre at the LOOK pose, at the brick's assumed resting height.

        A one-off geometric calibration (not simulator truth about the *brick*): it only uses the
        fixed, known LOOK camera pose, exactly the way a real system would calibrate "how many
        pixels is a millimetre here" once, from known camera geometry.
        """
        geom = self._camera_geometry()
        centre = np.array([self._look_tcp_xy()[0], self._look_tcp_xy()[1], self._table_z])
        p0 = project_point(geom, centre)
        p1 = project_point(geom, centre + np.array([0.001, 0.0, 0.0]))
        return float(np.linalg.norm(p1 - p0)) / 0.001

    def _look_tcp_xy(self) -> np.ndarray:
        return np.asarray(LOOK_TCP[:2], dtype=float)

    def _camera_geometry(self) -> CameraGeometry:
        return camera_geometry(self.scene.model, self.scene.data, WRIST_CAM, RENDER_W, RENDER_H)

    def _render_wrist_frame(self) -> np.ndarray:
        if self._renderer is None:
            from brick_builder.render.renderer import SceneRenderer

            self._renderer = SceneRenderer(self.scene.model, width=RENDER_W, height=RENDER_H)
        frame = self._renderer.render(self.scene.data, WRIST_CAM)
        return frame.rgb.astype(np.float32) / 255.0

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    # -- task -------------------------------------------------------------------------------

    def task_actor_entries(self) -> list[str]:
        return ["align_token", "align_time_frac"]

    def task_critic_entries(self) -> list[str]:
        return ["align_true_error"]

    def task_obs_values(self) -> dict[str, np.ndarray]:
        look_xy = self._look_tcp_xy()
        pos_err, yaw_err = self._pose_error()
        return {
            "align_token": np.array([
                self._token.x - look_xy[0], self._token.y - look_xy[1],
                self._token.yaw, self._token.confidence, self._token_age,
            ]),
            "align_time_frac": np.array([self._tick / self.horizon]),
            "align_true_error": np.array([pos_err, yaw_err]),
        }

    def task_reset(self, rng: np.random.Generator) -> None:
        dx, dy = rng.uniform(-SEARCH_RADIUS, SEARCH_RADIUS, size=2)
        yaw = float(rng.uniform(-math.pi, math.pi))
        look_xy = self._look_tcp_xy()
        x, y = float(look_xy[0] + dx), float(look_xy[1] + dy)
        quat = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])

        # `reset_scene` (already run before `task_reset`) zeroed every qvel via `mj_resetData`,
        # so only qpos needs setting here -- the brick starts perfectly still at its new pose.
        adr = self._brick_free_adr
        self.scene.data.qpos[adr:adr + 3] = [x, y, self._table_z]
        self.scene.data.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_forward(self.scene.model, self.scene.data)

        self._token = _Token(x=float(look_xy[0]), y=float(look_xy[1]), yaw=0.0, confidence=0.0)
        self._token_age = 0.0
        self._within_streak = 0
        self._spin_now = 0.0  # matches the tool's actual spin at LOOK_TCP (home_yaw=0)
        self._mover = MoveTo(
            target=self._look_tcp.copy(), speed=MOVE_SPEED, tilt=0.0, spin=0.0,
            grip=self._open_grip, c_vert=1.0, c_lat=1.0,
        )
        self._mover.reset(self._current_feedback())
        # A measurement script studying the effect of token *age* (the tool sits at LOOK_TCP the
        # whole time in this scene, so nothing about the true pose actually changes while it
        # waits -- see the guide) sets this to defer the moment the natural "at LOOK -> refresh"
        # check in `primitive_target` is allowed to fire, simulating a vision pipeline that took
        # this many ticks to produce its first reading. 0 means "never frozen" (the normal case).
        self._refresh_frozen_until_tick = 0
        self._refresh_token()

    # -- token refresh ------------------------------------------------------------------------

    def _pick_token(self, tokens: list[ObjectToken]) -> ObjectToken | None:
        candidates = [t for t in tokens if t.colour == self.colour]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.confidence)

    @staticmethod
    def _grasp_yaw(quat: np.ndarray) -> float:
        """The brick's z-rotation, folded into ``(-pi/2, pi/2]``.

        A rectangular brick grasped across its width looks the same rotated by 180 degrees (the
        jaws close on the same two long walls either way), and the SO-101's wrist-roll joint only
        covers about +-160 degrees -- short of a full turn. Picking the symmetric representative
        every time a yaw is turned into a commanded tool spin is what keeps every commanded grasp
        orientation inside the joint's actual range, not just numerically equivalent on paper.
        """
        raw_yaw = 2.0 * math.atan2(quat[3], quat[0])
        return _wrap_symmetric(raw_yaw, math.pi)

    def _current_feedback(self) -> Feedback:
        return feedback_from_readings(self.kin, self._last_reading, grip=float(self._cmd_echo["grip"]))

    def _retarget_mover(self) -> None:
        """Point :attr:`_mover` at the current token estimate, ramping from wherever the tool is
        right now -- never an instant jump, so the per-tick IK warm-start never has to chase a
        target that moved 5 cm in one control tick.

        Only actually resets the ramp (latching a fresh start point and restarting its clock) when
        the estimate moved by more than a token's own noise floor, or on the very first call.
        Otherwise the target/spin are updated in place and the ramp already under way keeps
        running: while the tool sits at the LOOK pose, *every* tick's refresh sees essentially the
        same estimate (the arm has not moved the brick), and resetting on every one of those would
        freeze the ramp at ``elapsed == 0`` forever -- it would never get anywhere to leave the
        LOOK-pose tolerance band that gates the next refresh, a self-inflicted deadlock this guard
        avoids.
        """
        target_pt = np.array([self._token.x, self._token.y, self._table_z])
        tool_rot = _tool_down_rot(self._token.yaw)
        tcp_target = target_pt - tool_rot @ self._grasp.pos_in_tcp

        moved = float(np.linalg.norm(tcp_target - self._mover.target))
        turned = abs(_wrap_symmetric(self._token.yaw - self._mover.spin, 2.0 * math.pi))
        first_call = self._mover.phase == "idle"

        self._mover.target = tcp_target
        self._mover.spin = self._token.yaw
        if first_call or moved > RETARGET_POS_TOL or turned > RETARGET_ANG_TOL:
            self._mover.reset(self._current_feedback())

    def _refresh_token(self) -> None:
        rng = self.np_random
        if self.producer_name == "surrogate":
            true_pos, true_quat = self.scene.body_pose(self._brick_name)
            true_yaw = self._grasp_yaw(true_quat)
            geom = self._camera_geometry()
            try:
                px = project_point(geom, true_pos)
            except ValueError:
                return  # brick behind the camera this instant; keep the previous token
            gt = [GroundTruthObject(x=float(px[0]), y=float(px[1]), yaw=true_yaw, layer=0,
                                     colour=self.colour, size=20.0)]
            tokens = self._producer.produce(rng, ground_truth=gt)
        else:
            frame = self._render_wrist_frame()
            tokens = self._producer.produce(rng, frame=frame)

        match = self._pick_token(tokens)
        if match is None:
            return
        geom = self._camera_geometry()
        try:
            world = backproject_to_plane(geom, np.array([match.x, match.y]), plane_z=self._table_z)
        except ValueError:
            return
        self._token = _Token(
            x=float(world[0]), y=float(world[1]),
            yaw=_wrap_symmetric(match.yaw, math.pi), confidence=float(match.confidence),
        )
        self._token_age = 0.0
        self._retarget_mover()

    # -- primitive --------------------------------------------------------------------------

    def primitive_target(self, tick: int) -> PrimitiveTarget:
        if tick > 0 and tick >= self._refresh_frozen_until_tick:
            pos, rot = self.read_tcp_pose()
            yaw_now, _, _ = euler_zyx(rot)
            pos_err = float(np.linalg.norm(pos - self._look_tcp))
            ang_err = abs(_wrap_symmetric(yaw_now - LOOK_YAW, 2.0 * math.pi))
            if pos_err < LOOK_TOL_POS and ang_err < LOOK_TOL_ANG:
                self._refresh_token()
            else:
                self._token_age += CONTROL_DT
        elif tick > 0:
            self._token_age += CONTROL_DT

        target = self._mover.update(self._current_feedback())

        # `MoveTo` only ramps position; a brick can be up to 90 degrees (its folded grasp
        # symmetry range) away in orientation, and handing `_pose_ik` a spin that jumps that far
        # in one tick is exactly the "target moved 5 cm in one control tick" case its own
        # docstring says it is not built for (its inner solve gets a handful of iterations,
        # tuned for a couple of degrees a tick). So spin is ramped here, separately, at a fixed
        # rate, and this ramped value -- not `target.spin` -- is what actually gets commanded.
        diff = _wrap_symmetric(target.spin - self._spin_now, 2.0 * math.pi)
        step = float(np.clip(diff, -SPIN_RATE * CONTROL_DT, SPIN_RATE * CONTROL_DT))
        self._spin_now += step

        return PrimitiveTarget(
            pos=target.pos, tilt=target.tilt, spin=self._spin_now, grip=target.grip,
            c_vert=target.c_vert, c_lat=target.c_lat,
        )

    # -- reward / termination -----------------------------------------------------------------

    def _pose_error(self) -> tuple[float, float]:
        true_pos, true_quat = self.scene.body_pose(self._brick_name)
        true_yaw = self._grasp_yaw(true_quat)
        target_pt = np.array([true_pos[0], true_pos[1], true_pos[2]])
        tool_rot = _tool_down_rot(true_yaw)
        true_tcp_target = target_pt - tool_rot @ self._grasp.pos_in_tcp

        tcp_pos, tcp_rot = self.tcp_pose()
        yaw_now, _, _ = euler_zyx(tcp_rot)
        pos_err = float(np.linalg.norm(tcp_pos - true_tcp_target))
        yaw_err = abs(_wrap_symmetric(yaw_now - true_yaw, math.pi))
        return pos_err, yaw_err

    def success(self) -> bool:
        pos_err, yaw_err = self._pose_error()
        ok = pos_err < SUCCESS_TOL_POS and yaw_err < SUCCESS_TOL_YAW
        self._within_streak = self._within_streak + 1 if ok else 0
        return self._within_streak >= SUCCESS_TICKS

    def reward(self) -> float:
        pos_err, yaw_err = self._pose_error()
        r = -20.0 * pos_err - 1.0 * yaw_err
        if self._within_streak >= SUCCESS_TICKS:
            r += 10.0
        return r

    def terminated(self) -> bool:
        return False

    def episode_result_extra(self) -> dict:
        pos_err, yaw_err = self._pose_error()
        return {"params": {
            "final_pos_error_m": pos_err, "final_yaw_error_deg": math.degrees(yaw_err),
            "producer": self.producer_name, "token_confidence": self._token.confidence,
            "token_age_s": self._token_age,
        }}


# ================================================================== measurement / figures =====
#
# Success of the scripted go-to-token controller as a function
# of token noise, token age and producer; the measured token error (mm) of each producer on 50
# random scenes; ms per observation for each producer. This section is the entry point
# `scripts/make_media.py` calls, and is also what the guide's own numbers came from.


def _rollout_scripted(env: AlignFromTokensEnv, seed: int, *, defer_ticks: int = 0) -> dict:
    """One zero-residual episode. ``defer_ticks`` simulates a vision pipeline that only produces
    its first reading after that many ticks: the token this env got for free at reset is thrown
    away and the tool is held hovering at ``LOOK_TCP`` (where it already is) until then, so this
    measures the cost of *not having a reading yet*, not of the (static, in this scene) target
    having moved -- see the guide section this function's numbers feed.
    """
    obs, _ = env.reset(seed=seed)
    if defer_ticks > 0:
        look_xy = env._look_tcp_xy()
        env._token = _Token(x=float(look_xy[0]), y=float(look_xy[1]), yaw=0.0, confidence=0.0)
        env._token_age = 0.0
        env._mover.target = env._look_tcp.copy()
        env._mover.spin = 0.0
        env._mover.reset(env._current_feedback())
        env._refresh_frozen_until_tick = defer_ticks

    zero = np.zeros(8, dtype=np.float32)
    info: dict = {}
    for _ in range(env.horizon):
        obs, _, terminated, truncated, info = env.step(zero)
        if terminated or truncated:
            break
    return info


def measure_success_vs_noise_age_producer(
    n_episodes: int = 20,
    noise_levels_mm: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0),
    age_levels_s: tuple[float, ...] = (0.0, 0.2, 0.5, 1.0),
    producers: tuple[str, ...] = ("surrogate", "blob"),
    seed0: int = 0,
) -> dict:
    """Success rate of the scripted go-to-token controller across the three sweep axes.

    Returns ``{"noise": {mm: successes/n}, "age": {s: successes/n}, "producer": {name: (rate,
    mean_token_error_mm, ms_per_observation)}}``. The noise sweep uses the surrogate producer
    (it is the only one whose error is a controllable dial); the age sweep uses the surrogate
    producer at a fixed, modest noise (2 mm) since it needs a producer fast enough to run many
    episodes; the producer comparison runs each real producer at its own, unmodified error
    profile. A producer that cannot be constructed on this machine is skipped, with its slot left
    out of the returned dict.
    """
    import time

    results: dict = {"noise": {}, "age": {}, "producer": {}}

    for mm in noise_levels_mm:
        env = AlignFromTokensEnv(horizon=150, seed=seed0, producer="surrogate", token_noise_mm=mm)
        ok = sum(int(_rollout_scripted(env, seed0 + i)["success"]) for i in range(n_episodes))
        results["noise"][mm] = ok / n_episodes
        env.close()

    for age_s in age_levels_s:
        env = AlignFromTokensEnv(horizon=150, seed=seed0, producer="surrogate", token_noise_mm=2.0)
        defer_ticks = int(round(age_s / CONTROL_DT))
        ok = sum(
            int(_rollout_scripted(env, seed0 + i, defer_ticks=defer_ticks)["success"])
            for i in range(n_episodes)
        )
        results["age"][age_s] = ok / n_episodes
        env.close()

    for name in producers:
        try:
            env = AlignFromTokensEnv(horizon=150, seed=seed0, producer=name)
        except ValueError:  # not a producer this build knows about -- skip, do not fail the sweep
            continue
        successes = 0
        errors_mm = []
        obs_times_s = []
        for i in range(n_episodes):
            env.reset(seed=seed0 + i)
            true_pos, true_quat = env.scene.body_pose(env._brick_name)
            # "ms per observation" means one token-producing call (render, if needed, plus
            # `producer.produce`), not a whole episode -- timed here, right after `reset` already
            # did the first one for free, so this is a second, freshly timed call at the same pose.
            t0 = time.perf_counter()
            env._refresh_token()
            obs_times_s.append(time.perf_counter() - t0)
            # token error: where this reading landed vs. true position, in mm
            errors_mm.append(1000.0 * math.hypot(
                env._token.x - true_pos[0] if env._token.confidence > 0 else float("nan"),
                env._token.y - true_pos[1] if env._token.confidence > 0 else float("nan"),
            ))
            info = _rollout_scripted(env, seed0 + i)
            successes += int(info["success"])
        results["producer"][name] = {
            "success_rate": successes / n_episodes,
            "mean_token_error_mm": float(np.nanmean(errors_mm)) if errors_mm else float("nan"),
            "ms_per_observation": 1000.0 * float(np.mean(obs_times_s)),
        }
        env.close()

    return results


def make_align_figures(out_dir: str = "docs/media/envs", width: int = 800, height: int = 600) -> dict:
    """Render this module's two figures: the LOOK-pose wrist view, and success vs. noise/
    age/producer. This is the entry point `scripts/make_media.py` calls.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from brick_builder.render.renderer import SceneRenderer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    # -- 1. the LOOK pose, wrist camera --------------------------------------------------------
    env = AlignFromTokensEnv(horizon=5, seed=0)
    env.reset(seed=0)
    renderer = SceneRenderer(env.scene.model, width=width, height=height)
    try:
        frame = renderer.render(env.scene.data, WRIST_CAM)
        fig, ax = plt.subplots(figsize=(width / 130, height / 130), dpi=130)
        ax.imshow(frame.rgb)
        ax.set_title(
            "AlignFromTokens LOOK pose: wrist_cam view of the target brick near the plate centre\n"
            "(the fixed jaw fills the right of frame at every wrist-cam pose -- WP-SCENE)",
            fontsize=8,
        )
        ax.axis("off")
        fig.tight_layout()
        path = out_dir / "align_look_pose.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        out["align_look_pose"] = str(path)
    finally:
        renderer.close()
        env.close()

    # -- 2. success vs noise / age / producer --------------------------------------------------
    results = measure_success_vs_noise_age_producer(n_episodes=20)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=130)

    noise = results["noise"]
    axes[0].plot(list(noise.keys()), list(noise.values()), "o-")
    axes[0].set_xlabel("token noise (mm, surrogate producer)")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("vs. noise")

    age = results["age"]
    axes[1].plot(list(age.keys()), list(age.values()), "o-", color="tab:orange")
    axes[1].set_xlabel("token age (s) before the first reading")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("vs. age (static target: mostly a time-budget cost)")

    prod = results["producer"]
    names = list(prod.keys())
    axes[2].bar(names, [prod[n]["success_rate"] for n in names], color="tab:green")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("vs. producer")

    fig.suptitle("AlignFromTokens: scripted go-to-token controller success", fontsize=10)
    fig.tight_layout()
    path = out_dir / "align_success_vs_noise.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    out["align_success_vs_noise"] = str(path)
    out["results"] = results
    return out
