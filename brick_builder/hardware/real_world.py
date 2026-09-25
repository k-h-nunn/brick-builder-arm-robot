"""The real world behind the four adapter verbs, and the hybrids that let each half run alone.

``brick_builder/action/adapter.py`` is the only window the controller has on the world: write a
tool target, read what the bus reports back, ask the wrist camera for one frame turned into slot
states, and let time pass.  The real system has to answer the same four verbs, and it does here:

``RealArmWorld``
    Inverse kinematics and the 50 Hz cadence of ``control/pick_place.py``'s ``ArmDriver``, wall
    clock paced by a monotonic scheduler, with :class:`~brick_builder.hardware.bus.Bus` in place of
    ``ServoBus`` and every tick written to a :class:`~brick_builder.hardware.twin_sync.TwinLog`.
``RealCamera``
    A UVC camera through OpenCV, undistorted with the calibrated intrinsics from the parameter
    file, cut to the 256x256 window and handed to the scene model that runs on the host.
``HybridWorld``
    One arm and one camera, each of which may be real or simulated, so a real arm can be tested
    against a rendered wrist view and a real camera against a simulated arm before the two are put
    together.

The camera is opened lazily and never at import time, and the frame source is an interface: a
recorded image or a rendered array stands in for a device wherever one is not present, which is how
the whole camera path is tested with no hardware attached.

Status: verified against the virtual bus, the simulator and rendered frames; untested on hardware.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import mujoco
import numpy as np

from ..perception.backproject import CameraGeometry
from ..perception.scene_model import SceneState
from ..perception.views import ViewingPose, crop_and_resize, cropped_camera_geometry
from ..robot.kinematics import ArmKinematics
from ..robot.patch_so101 import (
    ARM_JOINTS,
    ArmConfig,
    WRIST_CAM,
    build_arm_spec,
    fovy_from_diagonal_fov,
    servo_kp,
)
from .bus import Bus, BusState, CONTROL_PERIOD_S
from .twin_sync import TwinLog

N_JOINTS = 6
N_ARM = 5
"""Joints the inverse kinematics solves for; the gripper moves no tool axis."""

UVC_RESOLUTION = (640, 480)
"""The wrist module's target operating resolution, and what the parameter file ships."""

WINDOW_SIZE = 256
"""The scene model's input resolution -- the same 256 the simulator's wrist window uses."""


# ------------------------------------------------------------------------------- the camera ---


@dataclass(frozen=True)
class CameraIntrinsics:
    """Calibrated pinhole intrinsics and OpenCV distortion coefficients, at one resolution.

    ``k1, k2, p1, p2, k3`` are the Brown-Conrady coefficients ``cv2.calibrateCamera`` returns --
    a different parametrization from ``render.domain_rand.PostProcessConfig``'s image-space remap,
    which is why the parameter file carries both and says so.  The shipped values are derived from
    the lens's nominal 115 degree diagonal field of view at 640x480, not measured; guide 08 has
    the calibration procedure that replaces them.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    width: int = UVC_RESOLUTION[0]
    height: int = UVC_RESOLUTION[1]

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=float)

    @property
    def diagonal_fov_deg(self) -> float:
        half = math.hypot(self.width / (2.0 * self.fx), self.height / (2.0 * self.fy))
        return math.degrees(2.0 * math.atan(half))

    @classmethod
    def nominal(cls, diagonal_fov_deg: float = 115.0,
                size: tuple[int, int] = UVC_RESOLUTION) -> "CameraIntrinsics":
        """Intrinsics implied by a diagonal field of view and a centred principal point."""
        width, height = size
        fovy = math.radians(fovy_from_diagonal_fov(diagonal_fov_deg, width, height))
        f = (0.5 * height) / math.tan(0.5 * fovy)
        return cls(fx=f, fy=f, cx=0.5 * (width - 1), cy=0.5 * (height - 1),
                   width=width, height=height)

    @classmethod
    def from_params(cls, params) -> "CameraIntrinsics":
        """Read ``camera.*`` out of a loaded parameter file."""
        width = int(params.value("camera.resolution_width_px"))
        height = int(params.value("camera.resolution_height_px"))
        return cls(
            fx=float(params.value("camera.fx_px")),
            fy=float(params.value("camera.fy_px")),
            cx=float(params.value("camera.cx_px")),
            cy=float(params.value("camera.cy_px")),
            k1=float(params.value("camera.opencv_k1")),
            k2=float(params.value("camera.opencv_k2")),
            p1=float(params.value("camera.opencv_p1")),
            p2=float(params.value("camera.opencv_p2")),
            k3=float(params.value("camera.opencv_k3")),
            width=width, height=height,
        )


def camera_window(intr: CameraIntrinsics, output_size: int = WINDOW_SIZE) -> ViewingPose:
    """The 256x256 window cut out of an undistorted frame: the largest centred square, resized.

    A 4:3 sensor is not square and the scene model's input is, so something has to be thrown away.
    The centred square keeps the full vertical field of view and the middle of the horizontal one,
    which is where the wrist camera looks; ``crop`` records exactly which pixels survived, so
    ``views.output_to_full`` maps a slot centre back to a sensor pixel and the calibrated
    intrinsics still apply.
    """
    side = min(intr.width, intr.height)
    x0 = (intr.width - side) // 2
    y0 = (intr.height - side) // 2
    fovy = math.degrees(2.0 * math.atan(0.5 * intr.height / intr.fy))
    return ViewingPose(
        name="wrist_real", camera=WRIST_CAM, render_size=(intr.width, intr.height),
        crop=(x0, y0, x0 + side, y0 + side), fovy_deg=fovy, output_size=output_size, tcp=None,
    )


class FrameSource(Protocol):
    """Where a frame comes from.  A device, a file on disk, or an array a renderer produced."""

    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class ArraySource:
    """A fixed frame, or a sequence of them.  What a test uses instead of a device."""

    def __init__(self, frames) -> None:
        frames = [frames] if isinstance(frames, np.ndarray) else list(frames)
        if not frames:
            raise ValueError("ArraySource needs at least one frame")
        self.frames = [np.asarray(f) for f in frames]
        self.index = 0

    def read(self) -> np.ndarray:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def close(self) -> None:
        pass


class ImageFileSource(ArraySource):
    """Frames read off disk, in RGB.  The recorded stand-in for the camera."""

    def __init__(self, paths) -> None:
        import cv2

        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        frames = []
        for path in paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"cannot read {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        super().__init__(frames)


class UvcCamera:
    """A USB UVC camera through OpenCV.  Opened on the first :meth:`read`, never at import.

    ``index`` is the OpenCV device index.  The resolution is requested rather than assumed, and
    :meth:`read` raises if what comes back is a different size, because intrinsics calibrated at
    one resolution do not apply at another.
    """

    def __init__(self, index: int = 0, size: tuple[int, int] = UVC_RESOLUTION,
                 *, fourcc: str = "MJPG", warmup_frames: int = 5) -> None:
        self.index = int(index)
        self.size = (int(size[0]), int(size[1]))
        self.fourcc = fourcc
        self.warmup_frames = int(warmup_frames)
        self._cap = None

    def open(self):
        import cv2

        if self._cap is not None:
            return self._cap
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open UVC device {self.index}")
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        # The first frames off a UVC module are usually mid-exposure-adjustment; drop them.
        for _ in range(self.warmup_frames):
            cap.read()
        self._cap = cap
        return cap

    def read(self) -> np.ndarray:
        import cv2

        cap = self.open()
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"UVC device {self.index} returned no frame")
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            raise RuntimeError(
                f"device {self.index} gave {bgr.shape[1]}x{bgr.shape[0]}, "
                f"intrinsics are for {self.size[0]}x{self.size[1]}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RealCamera:
    """Frame source to slot states: undistort, cut the window, run the scene model.

    ``source`` is anything with :meth:`read`; ``model`` is the scene model that runs on the host
    (``perception.cnn.source.CnnSceneModel`` by default, built lazily so importing this module
    does not load a checkpoint).
    """

    def __init__(self, source: FrameSource, intrinsics: CameraIntrinsics | None = None, *,
                 model=None, undistort: bool = True, output_size: int = WINDOW_SIZE) -> None:
        self.source = source
        self.intrinsics = intrinsics if intrinsics is not None else CameraIntrinsics.nominal()
        self.window = camera_window(self.intrinsics, output_size)
        self.undistort = bool(undistort)
        self._model = model
        self.frames = 0
        self._maps: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def model(self):
        if self._model is None:
            from ..perception.cnn.source import CnnSceneModel

            self._model = CnnSceneModel()
        return self._model

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        """Remove lens distortion, keeping the same intrinsics so the window maths still holds."""
        if not self.undistort or not np.any(self.intrinsics.distortion):
            return frame
        import cv2

        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(
                self.intrinsics.matrix, self.intrinsics.distortion, None,
                self.intrinsics.matrix, (self.intrinsics.width, self.intrinsics.height),
                cv2.CV_16SC2,
            )
        return cv2.remap(frame, self._maps[0], self._maps[1], interpolation=cv2.INTER_LINEAR)

    def window_frame(self, frame: np.ndarray | None = None) -> np.ndarray:
        """One rectified, cropped, resized ``output_size`` square, RGB."""
        import cv2

        frame = self.source.read() if frame is None else frame
        frame = np.asarray(frame)
        if (frame.shape[1], frame.shape[0]) != (self.intrinsics.width, self.intrinsics.height):
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, intrinsics are for "
                f"{self.intrinsics.width}x{self.intrinsics.height}")
        return crop_and_resize(self.rectify(frame), self.window.crop, self.window.output_size,
                               interpolation=cv2.INTER_LINEAR)

    def observe(self, t: float | None = None) -> SceneState:
        return self.observe_rgb(t)[0]

    def observe_rgb(self, t: float | None = None) -> tuple[SceneState, np.ndarray]:
        rgb = self.window_frame()
        state = self.model.observe(rgb, window=self.window, t=t)
        self.frames += 1
        return state, rgb

    def geometry(self, pos: np.ndarray, rot: np.ndarray) -> CameraGeometry:
        """The window's camera geometry: calibrated intrinsics, extrinsics from the arm's pose."""
        full = CameraGeometry(fx=self.intrinsics.fx, fy=self.intrinsics.fy,
                              cx=self.intrinsics.cx, cy=self.intrinsics.cy,
                              pos=np.asarray(pos, dtype=float).copy(),
                              rot=np.asarray(rot, dtype=float).copy())
        return cropped_camera_geometry(full, self.window.crop, self.window.output_size)

    def close(self) -> None:
        self.source.close()


class SimulatedCamera:
    """The twin's wrist view, rendered from whatever joint angles the real arm reports.

    This is the simulated half of ``HybridWorld``: the arm is real, the picture is not.  The scene
    is posed and ``mj_forward`` run before each render, so the camera moves with the reported
    joints rather than with a simulated arm of its own.
    """

    def __init__(self, scene, *, renderer=None, model=None, output_size: int = WINDOW_SIZE) -> None:
        from ..action.adapter import WRIST_FULL, WRIST_RENDER
        from ..render.renderer import SceneRenderer

        self.scene = scene
        self.window = WRIST_FULL if output_size == WRIST_FULL.output_size else ViewingPose(
            name="wrist", camera=WRIST_CAM, render_size=WRIST_RENDER,
            crop=(0, 0, WRIST_RENDER[0], WRIST_RENDER[1]), fovy_deg=WRIST_FULL.fovy_deg,
            output_size=output_size, tcp=None,
        )
        self._renderer = renderer if renderer is not None else SceneRenderer(
            scene.model, width=WRIST_RENDER[0], height=WRIST_RENDER[1])
        self._renderer.instance_lut(scene.instance_ids)
        self._model = model
        self.frames = 0

    @property
    def model(self):
        if self._model is None:
            from ..perception.ideal_scene_model import IdealSceneModel

            self._model = IdealSceneModel()
        return self._model

    def set_joints(self, q: Sequence[float]) -> None:
        q = np.asarray(q, dtype=float)
        self.scene.data.qpos[self.scene.arm_qpos_adr[:len(q)]] = q
        mujoco.mj_forward(self.scene.model, self.scene.data)

    def observe(self, t: float | None = None) -> SceneState:
        return self.observe_rgb(t)[0]

    def observe_rgb(self, t: float | None = None) -> tuple[SceneState, np.ndarray]:
        from ..perception.views import render_view

        frame, _ = render_view(self.scene, self.window, self._renderer)
        state = self.model.observe(frame.rgb, window=self.window, instance=frame.instance,
                                   depth=frame.depth, t=t)
        self.frames += 1
        return state, np.asarray(frame.rgb)

    def geometry(self) -> CameraGeometry:
        from ..perception.views import full_frame_geometry

        return cropped_camera_geometry(full_frame_geometry(self.scene, self.window),
                                       self.window.crop, self.window.output_size)

    def close(self) -> None:
        self._renderer.close()


# --------------------------------------------------------------------------------- the clock ---


class Pacer:
    """A monotonic scheduler: one tick every ``period`` seconds, however long the work took.

    ``wait`` returns how late the tick was.  Deadlines are counted from a fixed origin rather than
    from the previous tick, so a slow tick does not push every later one back; a tick that
    overruns by more than one period gives up on catching the lost ones and re-bases, which is
    what keeps a stalled vision frame from turning into a burst of six back-to-back bus cycles.
    """

    def __init__(self, period: float = CONTROL_PERIOD_S) -> None:
        self.period = float(period)
        self.t0 = time.monotonic()
        self.tick = 0
        self.overruns = 0
        self.worst_overrun_s = 0.0

    def wait(self) -> float:
        self.tick += 1
        deadline = self.t0 + self.tick * self.period
        now = time.monotonic()
        late = now - deadline
        if late <= 0.0:
            time.sleep(-late)
            return 0.0
        self.overruns += 1
        self.worst_overrun_s = max(self.worst_overrun_s, late)
        if late > self.period:
            self.t0 = now - self.tick * self.period
        return late


# ------------------------------------------------------------------------------------ the arm ---


def kinematic_arm(cfg: ArmConfig = ArmConfig()) -> tuple[mujoco.MjModel, mujoco.MjData,
                                                         ArmKinematics]:
    """The patched arm on its own: a compiled model for inverse kinematics and camera extrinsics.

    No table, no bricks, no plate -- nothing that would step.  The real arm's physics happens in
    the real world; this model exists to answer "which joint angles put the tool there" and "where
    is the wrist camera at these joint angles", both of which are geometry.
    """
    spec = build_arm_spec(cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, ArmKinematics(model)


@dataclass
class ArmWorldConfig:
    """Cadence, compliance and safety for :class:`RealArmWorld`."""

    period_s: float = CONTROL_PERIOD_S
    arm: ArmConfig = field(default_factory=ArmConfig)
    paced: bool = True
    """Wall-clock pacing.  ``False`` runs as fast as the bus answers, which is what a virtual-bus
    run wants: the simulator behind it already sets the pace."""
    gravity_feedforward: bool = True
    log_every_tick: bool = True
    source: str = "real"
    """Which entry of ``twin_sync.SOURCES`` this run's log is labelled with."""


class RealArmWorld:
    """The four adapter verbs against :class:`~brick_builder.hardware.bus.Bus`.

    ``command`` solves the same inverse kinematics ``ArmDriver`` solves, adds the same gravity
    feed-forward, writes one sync-write and reads one sync-read, then waits out the rest of the
    20 ms.  ``feedback`` builds the same :class:`~brick_builder.control.primitives.Feedback` from
    what came back over the wire, so a scripted skill cannot tell which world it is driving.
    """

    def __init__(
        self,
        bus: Bus,
        *,
        cfg: ArmWorldConfig | None = None,
        camera: RealCamera | SimulatedCamera | None = None,
        kin: ArmKinematics | None = None,
        log: TwinLog | None = None,
    ) -> None:
        self.bus = bus
        self.cfg = cfg or ArmWorldConfig()
        self.camera = camera
        if kin is None:
            self.model, self.data, self.kin = kinematic_arm(self.cfg.arm)
        else:
            self.model, self.data, self.kin = kin.model, mujoco.MjData(kin.model), kin
        self.kp = servo_kp(self.cfg.arm.p_gain, self.cfg.arm.vin)
        self.pacer = Pacer(self.cfg.period_s)
        self.log = log if log is not None else TwinLog({"source": self.cfg.source})
        self.state: BusState = bus.cycle(self._read_joints())
        self.q_goal = self.state.position[:N_ARM].copy()
        self.tick = 0
        self.ik_failures = 0
        self.last_target = self.state.position.copy()
        self.last_torque = np.ones(N_JOINTS)
        self.last_overrun_s = 0.0
        self._grip = float(self.state.position[N_ARM])
        self._cmd_tcp = np.full(3, np.nan)

    # -- write ---------------------------------------------------------------------------------

    def command(self, tcp, spin: float, grip: float, *, c_vert: float = 1.0,
                c_lat: float = 1.0) -> None:
        """One tool target: inverse kinematics, one bus cycle, and the rest of the 20 ms."""
        from ..envs.base_env import _pose_ik
        from ..scene.compose import tool_down_ik

        tcp = np.asarray(tcp, dtype=float)
        q5, ok = _pose_ik(self.kin, tcp, 0.0, float(spin), self.q_goal)
        if not ok:
            try:
                q5, ok = tool_down_ik(self.kin, tcp, float(spin), q0=self.q_goal), True
            except RuntimeError:
                ok = False
        if ok:
            self.q_goal = q5
        else:
            self.ik_failures += 1
        self._grip = float(grip)
        self._cmd_tcp = tcp.copy()
        goal = np.concatenate([self._feedforward(self.q_goal), [self._grip]])

        torque = np.ones(N_JOINTS)
        torque[[ARM_JOINTS.index(n) for n in ("shoulder_lift", "elbow_flex", "wrist_flex")]] = c_vert
        torque[[ARM_JOINTS.index(n) for n in ("shoulder_pan", "wrist_roll")]] = c_lat

        self.last_target = goal
        self.last_torque = torque
        self.state = self.bus.cycle(goal, torque_limit=torque)
        self.last_overrun_s = self.pacer.wait() if self.cfg.paced else 0.0
        if self.cfg.log_every_tick:
            self._log_tick()
        self.tick += 1

    def _feedforward(self, q5: np.ndarray) -> np.ndarray:
        """The command that puts the arm at ``q5``, not ``q5`` itself -- ``ArmDriver``'s own."""
        if not self.cfg.gravity_feedforward:
            return q5.copy()
        comp = q5.copy()
        for _ in range(2):
            comp = q5 + self.kin.gravity_torque(comp) / self.kp
        return comp

    def _log_tick(self) -> None:
        self.log.append(
            tick=self.tick, t=self.tick * self.cfg.period_s,
            cmd_tcp=self._cmd_tcp, cmd_grip=self._grip,
            joint_target=self.last_target, joint_measured=self.state.position,
            joint_speed=self.state.speed, joint_load=self.state.load,
            bus_valid=self.state.valid.astype(float), bus_rtt_s=self.state.rtt_s,
            tool_measured=self.kin.fk(self.state.position[:N_ARM])[0],
            overrun_s=self.last_overrun_s,
        )

    def _read_joints(self) -> np.ndarray:
        """A goal that commands no motion: whatever the servos say they are at right now."""
        pos, _, _, _ = self.bus.read_ticks()
        return np.array([cal.to_rad(p) for cal, p in zip(self.bus.calibration, pos)])

    # -- read the bus --------------------------------------------------------------------------

    def feedback(self, grip: float):
        from ..control.primitives import feedback_from_readings

        return feedback_from_readings(self.kin, self.state, grip=grip)

    @property
    def joint_position(self) -> np.ndarray:
        return self.state.position.copy()

    @property
    def joint_speed(self) -> np.ndarray:
        return self.state.speed.copy()

    @property
    def joint_load(self) -> np.ndarray:
        return self.state.load.copy()

    @property
    def t(self) -> float:
        return self.tick * self.cfg.period_s

    @property
    def cmd_tcp(self) -> np.ndarray:
        """The tool target of the last :meth:`command`, NaN before the first one."""
        return self._cmd_tcp.copy()

    # -- look ----------------------------------------------------------------------------------

    def observe(self) -> SceneState:
        return self.observe_rgb()[0]

    def observe_rgb(self) -> tuple[SceneState, np.ndarray]:
        if self.camera is None:
            raise RuntimeError("this world has no camera; give RealArmWorld one to look through")
        if isinstance(self.camera, SimulatedCamera):
            self.camera.set_joints(self.state.position)
        return self.camera.observe_rgb(self.t)

    def camera_geometry(self) -> CameraGeometry:
        """Where the wrist camera is, from the joint angles the bus just reported."""
        if self.camera is None:
            raise RuntimeError("this world has no camera")
        if isinstance(self.camera, SimulatedCamera):
            self.camera.set_joints(self.state.position)
            return self.camera.geometry()
        pos, rot = self.wrist_camera_pose()
        return self.camera.geometry(pos, rot)

    def wrist_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Wrist-camera position and rotation in the arm's base frame at the reported joints."""
        self.data.qpos[:] = 0.0
        adr = [int(self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ARM_JOINTS]
        self.data.qpos[adr] = self.state.position
        mujoco.mj_forward(self.model, self.data)
        cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
        return self.data.cam_xpos[cam].copy(), self.data.cam_xmat[cam].reshape(3, 3).copy()

    # -- shutdown ------------------------------------------------------------------------------

    def close(self) -> None:
        """Soft-stop the arm and let go of the port and the camera, in that order."""
        try:
            self.bus.disable_torque()
        finally:
            self.bus.close()
            if self.camera is not None:
                self.camera.close()

    def __enter__(self) -> "RealArmWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ----------------------------------------------------------------------------------- hybrids ---


class HybridWorld:
    """One arm and one camera, each real or simulated, behind the same four verbs.

    ``arm`` is anything that answers ``command``/``feedback``/``joint_*`` -- a
    :class:`RealArmWorld` or the simulator's own ``WorldAdapter`` -- and ``camera`` is anything
    that answers ``observe_rgb``.  The two constructors below are the pair worth having:
    :func:`real_arm_simulated_camera` checks the bus, the inverse kinematics and the cadence with
    a picture nobody has to trust, and :func:`simulated_arm_real_camera` checks the camera path
    with an arm nobody has to trust.
    """

    def __init__(self, arm, camera) -> None:
        self.arm = arm
        self.camera = camera

    def command(self, tcp, spin: float, grip: float, *, c_vert: float = 1.0,
                c_lat: float = 1.0) -> None:
        self.arm.command(tcp, spin, grip, c_vert=c_vert, c_lat=c_lat)

    def feedback(self, grip: float):
        return self.arm.feedback(grip)

    @property
    def joint_position(self) -> np.ndarray:
        return self.arm.joint_position

    @property
    def joint_speed(self) -> np.ndarray:
        return self.arm.joint_speed

    @property
    def joint_load(self) -> np.ndarray:
        return self.arm.joint_load

    @property
    def tick(self) -> int:
        return int(self.arm.tick)

    @property
    def t(self) -> float:
        return float(self.arm.t)

    def observe(self) -> SceneState:
        return self.observe_rgb()[0]

    def observe_rgb(self) -> tuple[SceneState, np.ndarray]:
        if isinstance(self.camera, SimulatedCamera):
            self.camera.set_joints(self.arm.joint_position)
        return self.camera.observe_rgb(self.t)

    def close(self) -> None:
        for part in (self.arm, self.camera):
            close = getattr(part, "close", None)
            if close is not None:
                close()


def real_arm_simulated_camera(bus: Bus, twin, *, cfg: ArmWorldConfig | None = None,
                              renderer=None, model=None) -> HybridWorld:
    """Real arm, rendered wrist view: the twin is posed from the joint angles the bus reports."""
    camera = SimulatedCamera(twin, renderer=renderer, model=model)
    return HybridWorld(RealArmWorld(bus, cfg=cfg, camera=camera), camera)


def simulated_arm_real_camera(adapter, camera: RealCamera) -> HybridWorld:
    """Simulated arm, real camera: ``adapter`` is the simulator's own ``WorldAdapter``."""
    return HybridWorld(adapter, camera)
