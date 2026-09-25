"""Empty-hand world, kinematic twin and scene-model sources for the viewing-pose stages.

The viewing-pose world is the baseplate with the two reference bricks (`REF_YELLOW`,
`REF_RED`) and an empty hand, built from the same scene configuration as
`brick_builder.action.worlds` and displaced by the same `displacement(seed)` recipe. The
constants are restated here, and `tests/rl/scene_target/test_world.py` checks them against
`action.worlds` and `action.adapter`, because importing anything under `brick_builder.action`
runs that package's `__init__`, which loads the analytic servo and `TrackingEpisode`. The
environment must not load either (import-guard test).

`ViewingPoseWorld` compiles the scene once and re-places the baseplate for every episode by
writing its body pose into the model, the way `scene.randomize` writes plate poses; `reset_scene`
then lays the reference bricks out against it. One compiled model and one render context per
environment instance.

`ViewingPoseWorld` exposes the same four verbs as `action.adapter.WorldAdapter` (write a tool
target, read the bus, render the wrist view, let time pass) plus a scoring interface
(`tool_pose_truth`, `contact_flags`, `damage_level`) that only the environment's privileged half
and the ground-truth scorer read.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ...bricks.damage import STATES as DAMAGE_STATES
from ...bricks.structures import Placement
from ...control.pick_place import ArmDriver
from ...control.primitives import CONTROL_DT
from ...perception.views import ViewingPose, render_view
from ...render.renderer import SceneRenderer
from ...robot.kinematics import euler_zyx
from ...robot.patch_so101 import (
    GRIPPER_BODY, WRIST_CAM, WRIST_CAM_DIAGONAL_FOV_DEG, ArmConfig, fovy_from_diagonal_fov,
)
from ...scene.compose import (
    Scene, SceneConfig, build_scene, gripper_command, reset_scene, tool_down_ik,
)

__all__ = [
    "HOME_TCP", "NOMINAL_PLATE_XY", "NominalTwin", "REF_RED", "REF_YELLOW", "SourceInfo",
    "VISION_EVERY", "ViewingPoseWorld", "WRIST_FULL", "WRIST_RENDER", "build_source",
    "empty_hand_config", "plate_displacement", "sha256_file",
]

NOMINAL_PLATE_XY = (0.17, 0.0)
REF_YELLOW = Placement(h=2, w=4, x=0, y=5, z=0, colour="yellow")
REF_RED = Placement(h=2, w=2, x=6, y=0, z=0, colour="red")
HOME_TCP = (NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075)
"""Equal to `action.worlds.NOMINAL_PLATE_XY`, `REF_YELLOW`, `REF_RED` and the home TCP of
`action.worlds.scene_cfg` (checked by the tests)."""

WRIST_RENDER = (512, 512)
WRIST_FULL = ViewingPose(
    name="wrist",
    camera=WRIST_CAM,
    render_size=WRIST_RENDER,
    crop=(0, 0, WRIST_RENDER[0], WRIST_RENDER[1]),
    fovy_deg=fovy_from_diagonal_fov(WRIST_CAM_DIAGONAL_FOV_DEG, *WRIST_RENDER),
    output_size=256,
    tcp=None,
)
"""The full wrist sensor, resized to 256; equal to `action.adapter.WRIST_FULL`."""

VISION_EVERY = 5
"""Control ticks (20 ms) per vision tick (100 ms)."""

CRACKED = DAMAGE_STATES.index("cracked")


def empty_hand_config(plate_xy=NOMINAL_PLATE_XY, plate_yaw: float = 0.0) -> SceneConfig:
    """Scene configuration of the viewing-pose world: plate, two reference bricks, empty hand.

    Field for field what `action.worlds.scene_cfg` builds with `held=None` and no cameras."""
    return SceneConfig(
        plate_xy=tuple(plate_xy),
        plate_yaw=float(plate_yaw),
        placed=[REF_YELLOW, REF_RED],
        held=None,
        grasp_mode="friction",
        tray=False,
        cameras=(),
        home_tcp=tuple(HOME_TCP),
        home_hover=0.045,
        arm=ArmConfig(pad_friction=0.5),
    )


def plate_displacement(seed: int, scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Baseplate displacement of world `seed`: 4-8 mm and 2-3 deg, times `scale`.

    The draw of `action.worlds.displacement` (same generator, same order), scaled."""
    rng = np.random.default_rng(1000 + int(seed))
    r = rng.uniform(0.004, 0.008)
    th = rng.uniform(0.0, 2.0 * math.pi)
    yaw = rng.choice([-1.0, 1.0]) * rng.uniform(math.radians(2.0), math.radians(3.0))
    d_xy = np.array([r * math.cos(th), r * math.sin(th)])
    return float(scale) * d_xy, float(scale) * float(yaw)


def set_render_samples(scene: Scene, samples: int | None) -> None:
    """Offscreen multisample count, before the render context exists; `None` keeps the model's."""
    if samples is not None:
        scene.model.vis.quality.offsamples = int(samples)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def _arm_body_ids(model) -> set[int]:
    """Bodies rigidly carried by the gripper (the jaws and the wrist camera mount)."""
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    out = {grip}
    for i in range(model.nbody):
        j = i
        for _ in range(32):
            if j <= 0:
                break
            if j == grip:
                out.add(i)
                break
            j = int(model.body_parentid[j])
    return out


# --------------------------------------------------------------------------------- the world ---


class ViewingPoseWorld:
    """One compiled empty-hand world, re-placed per episode."""

    def __init__(self, *, render_samples: int | None = None) -> None:
        self.scene: Scene = build_scene(empty_hand_config())
        set_render_samples(self.scene, render_samples)
        self.renderer = SceneRenderer(self.scene.model, width=WRIST_RENDER[0],
                                      height=WRIST_RENDER[1])
        self.renderer.instance_lut(self.scene.instance_ids)
        self.grip = float(gripper_command(self.scene))
        """The gripper command of the empty hand; every recorded viewing-pose command uses it."""
        self.driver: ArmDriver | None = None
        self.plate_xy = np.asarray(NOMINAL_PLATE_XY, dtype=float)
        self.plate_yaw = 0.0
        model = self.scene.model
        self._plate_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.scene.plate.name)
        self._arm = _arm_body_ids(model)
        self._bricks = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                        for n in self.scene.bricks}

    def close(self) -> None:
        self.renderer.close()
        self.driver = None

    # -- episode setup (scene construction; not reachable from the actor) --------------------

    def reset(self, *, plate_offset_xy, plate_yaw: float, arm_tcp, arm_yaw: float,
              seed: int, settle_ticks: int = 0, settle_tcp=None, settle_yaw=None) -> None:
        """Place the plate, put the arm at `arm_tcp`/`arm_yaw`, and run `settle_ticks` control
        ticks holding the settle pose (default: the start pose)."""
        scene = self.scene
        self.plate_xy = np.asarray(NOMINAL_PLATE_XY, dtype=float) + np.asarray(plate_offset_xy, float)
        self.plate_yaw = float(plate_yaw)
        scene.model.body_pos[self._plate_bid][:2] = self.plate_xy
        scene.model.body_quat[self._plate_bid] = _yaw_quat(self.plate_yaw)
        q = tool_down_ik(scene.kin, np.asarray(arm_tcp, dtype=float), float(arm_yaw),
                         q0=scene.home_q)
        reset_scene(scene, np.random.default_rng(int(seed)), arm_q=q)
        self.driver = ArmDriver(scene, rng=np.random.default_rng(int(seed)))
        tcp = np.asarray(arm_tcp if settle_tcp is None else settle_tcp, dtype=float)
        yaw = float(arm_yaw if settle_yaw is None else settle_yaw)
        for _ in range(int(settle_ticks)):
            self.driver.step(tcp, yaw, self.grip)

    # -- the four verbs ------------------------------------------------------------------------

    def command(self, tcp, spin: float, grip: float, *, c_vert: float = 1.0,
                c_lat: float = 1.0) -> None:
        """One 20 ms control tick: tool target through the arm's IK onto the servo bus."""
        self.driver.step(np.asarray(tcp, dtype=float), float(spin), float(grip),
                         c_vert=c_vert, c_lat=c_lat)

    def feedback(self, grip: float):
        return self.driver.feedback(grip)

    def bus_state(self) -> np.ndarray:
        """`[18]` joint positions, speeds and loads as the bus reports them."""
        r = self.driver.reading
        return np.concatenate([np.asarray(r.position, dtype=np.float64)[:6],
                               np.asarray(r.speed, dtype=np.float64)[:6],
                               np.asarray(r.load, dtype=np.float64)[:6]])

    def render(self):
        """One wrist frame (`render.renderer.Frame`: rgb, instance, depth) at 256 x 256."""
        frame, _ = render_view(self.scene, WRIST_FULL, self.renderer)
        return frame

    @property
    def t(self) -> float:
        return 0.0 if self.driver is None else self.driver.tick * CONTROL_DT

    # -- scoring interface (privileged) ----------------------------------------------------------

    def tool_pose_truth(self) -> tuple[np.ndarray, float]:
        """Simulator TCP position [m] and tool yaw [rad]."""
        pos, rot = self.scene.tcp_pose()
        yaw, _, _ = euler_zyx(rot)
        return pos, float(yaw)

    def contact_flags(self) -> np.ndarray:
        """`[2]`: arm touching a brick, arm touching the baseplate."""
        model, data = self.scene.model, self.scene.data
        out = np.zeros(2, dtype=np.float32)
        for c in data.contact[:data.ncon]:
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            for a, b in ((b1, b2), (b2, b1)):
                if a in self._arm:
                    root = int(model.body_rootid[b])
                    if b in self._bricks or root in self._bricks:
                        out[0] = 1.0
                    if b == self._plate_bid or root == self._plate_bid:
                        out[1] = 1.0
        return out

    def damage_level(self) -> int:
        """Index into `bricks.damage.STATES`; 0 = intact."""
        monitor = getattr(self.scene, "monitor", None)
        if monitor is None:
            return 0
        try:
            return int(DAMAGE_STATES.index(monitor.state))
        except (ValueError, AttributeError):
            return 0


# ---------------------------------------------------------------------------------- the twin ---


class NominalTwin:
    """Kinematic copy of the nominal viewing-pose world: pose the arm, render the wrist view.

    The same operations as `action.twin.DigitalTwin.pose` and `observe_rgb`: tool-down inverse
    kinematics seeded from the previous pose, joint angles written, no physics."""

    def __init__(self, *, render_samples: int | None = None) -> None:
        self.scene: Scene = build_scene(empty_hand_config())
        set_render_samples(self.scene, render_samples)
        reset_scene(self.scene, np.random.default_rng(0))
        self.renderer = SceneRenderer(self.scene.model, width=WRIST_RENDER[0],
                                      height=WRIST_RENDER[1])
        self.renderer.instance_lut(self.scene.instance_ids)
        self.grip = float(gripper_command(self.scene))
        self._q = self.scene.home_q.copy()

    def close(self) -> None:
        self.renderer.close()

    def reset_seed_pose(self) -> None:
        self._q = self.scene.home_q.copy()

    def solve(self, tcp, yaw: float) -> np.ndarray:
        """Joint angles for a tool-down pose; raises `RuntimeError` when unreachable."""
        return tool_down_ik(self.scene.kin, np.asarray(tcp, dtype=float), float(yaw), q0=self._q)

    def pose(self, tcp, yaw: float) -> np.ndarray:
        scene = self.scene
        q = self.solve(tcp, yaw)
        self._q = q.copy()
        scene.data.qpos[scene.arm_qpos_adr[:5]] = q
        scene.data.qpos[scene.arm_qpos_adr[5]] = self.grip
        mujoco.mj_forward(scene.model, scene.data)
        return q

    def render(self):
        frame, _ = render_view(self.scene, WRIST_FULL, self.renderer)
        return frame


# ------------------------------------------------------------------------------- the sources ---


@dataclass(frozen=True)
class SourceInfo:
    """Which scene model produced a slot state, and with which weights."""

    name: str
    weights_path: str = ""
    weights_sha256: str = ""
    threads: int | None = None
    device: str = "cpu"

    @property
    def sha8(self) -> str:
        return self.weights_sha256[:8] if self.weights_sha256 else "none"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_source(name: str, kwargs=None, *, K: int = 8, build_decode: bool = False):
    """A fresh scene-model source and its `SourceInfo`.

    `learned` loads its checkpoint by explicit path (`weights`; default the file shipped in
    `brick_builder/perception/weights/`) and records the file's sha256. CPU only in this
    version; `threads` caps torch's intra-op threads."""
    from ...perception import sources

    kwargs = dict(kwargs or {})
    if name == "ideal":
        src = sources.get("ideal")(K=K, build_decode=build_decode)
        return src, SourceInfo(name="ideal")
    if name == "learned":
        import torch

        from ...perception.cnn.source import DEFAULT_WEIGHTS

        device = str(kwargs.pop("device", "cpu"))
        if device != "cpu":
            raise ValueError(f"device {device!r}: this environment runs the scene model on the CPU")
        weights = Path(kwargs.pop("weights", None) or DEFAULT_WEIGHTS).resolve()
        threads = kwargs.pop("threads", None)
        if kwargs:
            raise ValueError(f"unknown learned-source arguments: {sorted(kwargs)}")
        if threads is not None:
            torch.set_num_threads(int(threads))
        src = sources.get("learned")(checkpoint=str(weights), device="cpu", K=K)
        impl = getattr(src, "impl", None)
        if impl is not None:
            impl.build_decode = bool(build_decode)
        return src, SourceInfo(name="learned", weights_path=str(weights),
                               weights_sha256=sha256_file(weights),
                               threads=None if threads is None else int(threads))
    weights = kwargs.get("weights") or kwargs.get("checkpoint")
    src = sources.get(name)(**kwargs)
    sha = sha256_file(weights) if weights else ""
    return src, SourceInfo(name=name, weights_path=str(weights or ""), weights_sha256=sha)


def observe(source, frame, *, t: float | None = None):
    """One scene-model pass over a rendered wrist frame."""
    return source.observe(frame.rgb, window=WRIST_FULL, instance=frame.instance,
                          depth=frame.depth, t=t)


def reset_source(source) -> None:
    """Clear the source's temporal slot assigner."""
    reset = getattr(source, "reset", None)
    if callable(reset):
        reset()
