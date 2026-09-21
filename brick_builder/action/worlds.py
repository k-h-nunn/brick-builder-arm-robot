"""The worlds this package is measured in, and the errors nobody tells the controller about.

Scene construction, not control. Everything here is allowed to write joint angles and free-joint
positions straight into the simulator, because that is what setting a scene up *is*; the
controller in `place.py` has no way to reach any of it and the guard test in `tests/action`
checks that it does not.

Two worlds per seed, from the same description:

- the **nominal** one, where the baseplate is where the task says it is and the brick to pick up
  lies where the supply row says. The digital twin runs the job here, and the recording of that
  run is the reference the real run tracks.
- the **real** one, with the baseplate 4-8 mm and 2-3 degrees out, the brick to pick 8-15 mm and
  10-20 degrees out, and the held brick sitting a couple of millimetres off-centre along the jaw
  faces. None of it is told to the controller and none of it is readable through the adapter.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..bricks.structures import Placement
from ..robot.patch_so101 import ArmConfig
from ..scene.compose import HeldBrick, Scene, SceneConfig, build_scene, reset_scene

NOMINAL_PLATE_XY = (0.17, 0.0)
"""Where the baseplate is *supposed* to be. The controller believes this; the world disagrees."""

REF_YELLOW = Placement(h=2, w=4, x=0, y=5, z=0, colour="yellow")
REF_RED = Placement(h=2, w=2, x=6, y=0, z=0, colour="red")
"""The two bricks already on the plate, which the loop lines itself up on."""

TARGET_CELL = (4, 3)
"""Column 5, row 4 counting from one -- grid cell (4, 3) counting from zero."""


def scene_cfg(*, plate_xy, plate_yaw, placed, held, home_tcp, cameras) -> SceneConfig:
    return SceneConfig(
        plate_xy=tuple(plate_xy),
        plate_yaw=float(plate_yaw),
        placed=list(placed),
        held=held,
        grasp_mode="friction",
        tray=False,
        cameras=tuple(cameras),
        home_tcp=tuple(home_tcp),
        home_hover=0.045,
        arm=ArmConfig(pad_friction=0.5),
    )


def displacement(seed: int) -> tuple[np.ndarray, float]:
    """The world's unknown error: the plate 4-8 mm off and 2-3 deg round, direction drawn from
    `seed`. The controller never sees this; only `build_world` and the scorer do."""
    rng = np.random.default_rng(1000 + seed)
    r = rng.uniform(0.004, 0.008)
    th = rng.uniform(0.0, 2.0 * math.pi)
    yaw = rng.choice([-1.0, 1.0]) * rng.uniform(math.radians(2.0), math.radians(3.0))
    return np.array([r * math.cos(th), r * math.sin(th)]), float(yaw)


def build_world(seed: int, *, displaced: bool = True, cameras=()) -> tuple[Scene, np.ndarray, float]:
    """The scene the arm really runs in, with the plate moved where nobody told the controller."""
    d_xy, d_yaw = displacement(seed) if displaced else (np.zeros(2), 0.0)
    # The brick's unknown offset in the hand is ALONG the jaw faces, not across the gap:
    # closing the jaws squares up an error across the gap and can do nothing about one along
    # them. (It also has to be that way here -- a reset that starts the brick off-centre across
    # the gap begins the episode with one pad already inside it, and the solver fires it out.)
    held = HeldBrick(nx=2, ny=2, colour="blue",
                     offset_jitter=(0.0, 0.002, 0.0) if displaced else (0.0, 0.0, 0.0))
    cfg = scene_cfg(
        plate_xy=np.asarray(NOMINAL_PLATE_XY) + d_xy, plate_yaw=d_yaw,
        placed=[REF_YELLOW, REF_RED], held=held,
        home_tcp=(NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075), cameras=cameras,
    )
    scene = build_scene(cfg)
    reset_scene(scene, np.random.default_rng(seed))
    return scene, d_xy, d_yaw


WANTED = (2, 2, "blue")
"""The brick to pick up and place. Square, so which way round it is gripped does not matter --
which is also why its own blob only ever says which way it is turned to within a quarter turn."""

DISTRACTOR = (2, 4, "white")
"""A second brick lying beside it, so "find the one you were asked for" is a real question.
White, because a colour near the red or the yellow already on the plate would turn a test of the
loop into a test of the palette."""

SUPPLY_XY = (0.13, -0.17)


def build_pick_world(seed: int, *, displaced: bool = True, cameras=()):
    """The world for the whole job: two bricks lying loose on the table, an empty hand.

    Two things are wrong with it and nobody tells the controller either: the baseplate is 4-8 mm
    and 2-3 degrees from nominal, and the brick to be picked up is 8-15 mm and 10-20 degrees from
    where the supply row says it should be.
    """
    d_xy, d_yaw = displacement(seed) if displaced else (np.zeros(2), 0.0)
    cfg = scene_cfg(
        plate_xy=np.asarray(NOMINAL_PLATE_XY) + d_xy, plate_yaw=d_yaw,
        placed=[REF_YELLOW, REF_RED], held=None,
        home_tcp=(NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075), cameras=cameras,
    )
    cfg.supply = (WANTED, DISTRACTOR)
    cfg.supply_xy = SUPPLY_XY
    scene = build_scene(cfg)
    reset_scene(scene, np.random.default_rng(seed))
    b_xy, b_yaw = brick_displacement(seed) if displaced else (np.zeros(2), 0.0)
    if displaced:
        nudge_brick(scene, SUPPLY_BODY, b_xy, b_yaw)
    return scene, d_xy, d_yaw, b_xy, b_yaw


SUPPLY_BODY = "supply_brick_0"
SPARE_BODY = "supply_brick_1"
"""The two loose bricks, named for the part each plays rather than for what is on the table.

Both names are fixed by the row position, not by how many bricks are standing in the row, so
"the spare" is the same body in the world the sequence was recorded in and in the twin that has
already picked the other one up.
"""


def brick_displacement(seed: int) -> tuple[np.ndarray, float]:
    """Where the brick to pick really lies: 8-15 mm and 10-20 degrees off the supply row."""
    rng = np.random.default_rng(7000 + seed)
    r = rng.uniform(0.008, 0.015)
    th = rng.uniform(0.0, 2.0 * math.pi)
    yaw = rng.choice([-1.0, 1.0]) * rng.uniform(math.radians(10.0), math.radians(20.0))
    return np.array([r * math.cos(th), r * math.sin(th)]), float(yaw)


def nudge_brick(scene: Scene, body: str, d_xy, d_yaw: float) -> None:
    """Move a loose brick to where it really is. Scene setup, not control: the controller has no
    way to ask and is never told."""
    adr = scene.free_joint_adr(body)
    scene.data.qpos[adr:adr + 2] += np.asarray(d_xy, dtype=float)
    c, s_ = math.cos(0.5 * d_yaw), math.sin(0.5 * d_yaw)
    q = scene.data.qpos[adr + 3:adr + 7].copy()
    scene.data.qpos[adr + 3:adr + 7] = [
        q[0] * c - q[3] * s_, q[1] * c - q[2] * s_,
        q[2] * c + q[1] * s_, q[3] * c + q[0] * s_,
    ]
    mujoco.mj_forward(scene.model, scene.data)


def nominal_cfg(*, held: HeldBrick | None, placed=None, supply=()) -> SceneConfig:
    """What the controller believes the world looks like: the plate where it is supposed to be."""
    cfg = scene_cfg(
        plate_xy=NOMINAL_PLATE_XY, plate_yaw=0.0,
        placed=[REF_YELLOW, REF_RED] if placed is None else placed, held=held,
        home_tcp=(NOMINAL_PLATE_XY[0], NOMINAL_PLATE_XY[1], 0.075), cameras=(),
    )
    cfg.supply = tuple(supply)
    cfg.supply_xy = SUPPLY_XY
    return cfg




def build_nominal_world(pick: bool, *, held, prepare=None) -> Scene:
    """The world the twin runs the job in: everything exactly where the task says it is."""
    cfg = nominal_cfg(held=held, supply=(WANTED, DISTRACTOR) if pick else ())
    scene = build_scene(cfg)
    if prepare is not None:
        prepare(scene.model)
    reset_scene(scene, np.random.default_rng(0))
    return scene


def supply_nominal_xy(scene: Scene) -> np.ndarray:
    """Where the supply row says the brick to pick up is, in the nominal world.

    Nominal task knowledge, looked up here rather than in the controller: reading a body's pose
    is a thing scene setup is allowed to do and control code is not, even when the scene in
    question is the twin's.
    """
    pos = np.asarray(scene.body_pose(SUPPLY_BODY)[0], dtype=float).copy()
    pos[2] = 0.0
    return pos
