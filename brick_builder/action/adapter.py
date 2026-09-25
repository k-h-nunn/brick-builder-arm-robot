"""The narrow window between the controller and the world.

The controller is not allowed to know where anything really is. `WorldAdapter` is the whole of
what it may do: write a tool target onto the servo bus, read what the bus reports back, ask the
wrist camera for one frame turned into slot states, and let time pass. Deliberately no accessor
for a body pose, a contact, or the snap model -- if a number is not on this class, control code
cannot have it.

The wrist *window* is the full sensor: `WRIST_FULL` crops nothing, so a slot centre in its
normalized `[-1, 1]` coordinates already *is* a full-sensor normalized coordinate and
`views.output_to_full` is a plain scale. That matters because the controller servos on centres
and must not have a moving crop quietly changing what a centre means.
"""

from __future__ import annotations

import numpy as np

from ..control.pick_place import CONTROL_DT, ArmDriver
from ..perception.ideal_scene_model import IdealSceneModel
from ..perception.scene_model import SceneState
from ..perception.views import ViewingPose, render_view
from ..render.renderer import SceneRenderer
from ..robot.patch_so101 import WRIST_CAM, WRIST_CAM_DIAGONAL_FOV_DEG, fovy_from_diagonal_fov
from ..scene.compose import Scene

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
"""The whole wide-angle wrist sensor, resized to 256 -- no crop, and `tcp=None` so it renders
from wherever the arm happens to be rather than driving the arm somewhere."""

VISION_EVERY = 5
"""Control ticks between vision ticks: 5 x 20 ms = 10 Hz, the rate the loop acts at."""


class WorldAdapter:
    """One running world, seen through four verbs."""

    def __init__(self, scene: Scene, *, rng: np.random.Generator | None = None,
                 renderer: SceneRenderer | None = None, K: int = 8,
                 build_decode: bool = False, source=None) -> None:
        self._scene = scene
        rng = np.random.default_rng(0) if rng is None else rng
        self._driver = ArmDriver(scene, rng=rng)
        self._renderer = renderer if renderer is not None else SceneRenderer(
            scene.model, width=WRIST_RENDER[0], height=WRIST_RENDER[1]
        )
        # Without this the renderer labels every body 0 and the scene model sees one big
        # background blob: `instance_lut` is what gives each brick its own stable instance id.
        self._renderer.instance_lut(scene.instance_ids)
        # `source` swaps the scene model without the controller noticing: the trained
        # convolutional model and the simulator's ground-truth one publish the same slot states,
        # so either can drive the loop and the video says which one did.
        self._vision = source if source is not None else IdealSceneModel(
            K=K, build_decode=build_decode)
        self._frames = 0

    # -- write ---------------------------------------------------------------------------------

    def command(self, tcp, spin: float, grip: float, *, c_vert: float = 1.0,
                c_lat: float = 1.0, q_goal=None) -> None:
        """Write one tool target (solved to joint targets by the arm's own inverse kinematics,
        tool pointing down) and let one 20 ms control tick of physics run.

        `q_goal` seeds that solve with joint angles a controller has already worked out, so a
        controller running off-process drives the arm with its own answer and the driver's solve
        only checks it. `joint_goal` afterwards says whether it was kept.
        """
        if q_goal is not None:
            self._driver.q_goal = np.asarray(q_goal, dtype=float).copy()
        self._driver.step(np.asarray(tcp, dtype=float), float(spin), float(grip),
                          c_vert=c_vert, c_lat=c_lat)

    # -- read the bus --------------------------------------------------------------------------

    def feedback(self, grip: float):
        """What the servo bus says the arm is doing: joint positions, speeds, loads, and the tool
        pose those positions imply. Everything here came back over the bus."""
        return self._driver.feedback(grip)

    @property
    def joint_load(self) -> np.ndarray:
        return np.asarray(self._driver.reading.load, dtype=float).copy()

    @property
    def joint_speed(self) -> np.ndarray:
        """Output-shaft speeds as the bus reports them -- how the controller knows it is still."""
        return np.asarray(self._driver.reading.speed, dtype=float).copy()

    @property
    def joint_position(self) -> np.ndarray:
        return np.asarray(self._driver.reading.position, dtype=float).copy()

    @property
    def joint_goal(self) -> np.ndarray:
        """The joint angles the bus was last told to hold, before the gravity term."""
        return np.asarray(self._driver.q_goal, dtype=float).copy()

    @property
    def kinematics(self):
        """The arm's own forward/inverse kinematics -- geometry, not a view of the world."""
        return self._driver.kin

    @property
    def tick(self) -> int:
        return int(self._driver.tick)

    @property
    def t(self) -> float:
        return self._driver.tick * CONTROL_DT

    # -- look ----------------------------------------------------------------------------------

    def observe(self) -> SceneState:
        """One wrist-camera frame, as slot states. The controller's only view of the world."""
        frame, _ = render_view(self._scene, WRIST_FULL, self._renderer)
        state = self._vision.observe(
            frame.rgb, window=WRIST_FULL, instance=frame.instance, depth=frame.depth,
            t=self.t,
        )
        self._frames += 1
        return state

    def observe_rgb(self) -> tuple[SceneState, np.ndarray]:
        """As `observe`, and also hands back the raw wrist image -- for the video's inset only.
        Control code calls `observe`."""
        frame, _ = render_view(self._scene, WRIST_FULL, self._renderer)
        state = self._vision.observe(
            frame.rgb, window=WRIST_FULL, instance=frame.instance, depth=frame.depth,
            t=self.t,
        )
        self._frames += 1
        return state, np.asarray(frame.rgb)

    def close(self) -> None:
        self._renderer.close()

    # -- the one back door, for scoring and recording only ----------------------------------------

    @property
    def scene_for_scoring(self) -> Scene:
        """The simulator itself. **Not for control**: `place.py` never touches this, and the
        guard test in `tests/action/` checks it does not. The scorer (`verify_placement`) and
        the video's outside camera do, because both are allowed to know the truth."""
        return self._scene

    @property
    def driver_for_scoring(self):
        return self._driver
