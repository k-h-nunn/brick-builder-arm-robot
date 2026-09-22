"""Scene setup for the page, and the disturbances it can inject.

Everything here writes joint angles, free-joint positions and model fields straight into the
simulator, which is what setting a scene up is; it is the same licence `brick_builder.action.
worlds` takes and this module builds on that one rather than repeating it. The controller reads
none of it: a disturbance posted from the page lands in the world, never in the loop.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..action import worlds
from ..bricks.structures import Placement, format_placement, parse_placement
from ..params import DEFAULT_PARAMS_PATH, ParamFile, ParamsError, load_params, to_scene_config
from ..scene.compose import HeldBrick, Scene, build_scene, reset_scene, set_brick_pose

PARAMS_DIR = Path(DEFAULT_PARAMS_PATH).resolve().parent

JOBS = ("place", "pick and place")
CONTROLLERS = ("analytic", "learned")
OFFSETS = ("nominal", "seed", "typed")
MODES = ("sim", "twin (sim)", "real arm", "microcontroller")
CONNECTED_MODES = ("sim", "twin (sim)")

DEFAULT_PLACED = (worlds.REF_YELLOW, worlds.REF_RED)
"""The two bricks the recorded sequence lines up on. A scene may carry more; it may not carry
fewer without the loop losing the references it was recorded against."""

PARK_XY = (0.60, 0.60)
"""Where a removed brick goes: off the table, out of every camera, still a legal free body."""

MIN_OFFSCREEN = (1280, 960)
"""Smallest offscreen framebuffer a scene is built with, whatever the parameter file says the
camera's resolution is. The wrist window renders at 512x512 and the free camera up to 1920x1080;
both go through this buffer."""


# ================================================================================== the setup ===


@dataclass
class Setup:
    """One scene, as the page posts it. Every field is a JSON scalar or a list of them."""

    instruction: str = "put the blue 2x2 at column 5 row 4"
    job: str = "place"
    seed: int = 1
    placed: list[str] = field(
        default_factory=lambda: [format_placement(p) for p in DEFAULT_PLACED])
    plate_offset: str = "seed"
    plate_dx_mm: float = 0.0
    plate_dy_mm: float = 0.0
    plate_yaw_deg: float = 0.0
    supply_offset: str = "seed"
    supply_dx_mm: float = 0.0
    supply_dy_mm: float = 0.0
    supply_yaw_deg: float = 0.0
    params_file: str = "default.yaml"
    param_edits: dict[str, Any] = field(default_factory=dict)
    source: str = "ideal"
    controller: str = "analytic"
    breakable: bool = False
    damage_monitor: bool = True
    mode: str = "sim"
    twin_params_file: str = "default.yaml"

    @classmethod
    def from_json(cls, blob: dict) -> "Setup":
        """Whatever the page posted, coerced to the type each field's own default has.

        A new field needs nothing here: the default it is declared with says what it is.
        """
        out = cls()
        for f in dataclasses.fields(cls):
            if f.name not in blob:
                continue
            value = blob[f.name]
            default = getattr(out, f.name)
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            elif isinstance(default, list):
                value = [str(v) for v in value]
            elif isinstance(default, dict):
                value = dict(value)
            else:
                value = str(value)
            setattr(out, f.name, value)
        return out

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def pick(self) -> bool:
        return self.job == "pick and place"


def param_files() -> list[str]:
    """The parameter files on the shelf, by file name."""
    return sorted(p.name for p in PARAMS_DIR.glob("*.yaml"))


def params_path(name: str) -> Path:
    """The path a parameter file name resolves to. An absolute path is taken as it stands."""
    p = Path(name)
    return p if p.is_absolute() else PARAMS_DIR / p.name


def load_edited(name: str, edits: dict[str, Any], scratch: Path) -> tuple[ParamFile, list[str]]:
    """A parameter file with the page's typed values applied, validated by `params.py`.

    The edits are written into the file's own nested form and re-read through `load_params`, so
    a typed value is checked by the same schema a file on disk is: a value out of range comes
    back as a list of problems and the unedited file is returned.
    """
    base = load_params(params_path(name))
    if not edits:
        return base, []
    nested = base.as_nested_dict()
    for path, value in edits.items():
        node = nested
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict) or parts[-1] not in node:
            return base, [f"no parameter named {path!r}"]
        leaf = node[parts[-1]]
        if isinstance(leaf, dict):
            leaf["value"] = value
            leaf["status"] = "measured"
        else:
            node[parts[-1]] = value
    import yaml

    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(yaml.safe_dump(nested, sort_keys=True, allow_unicode=True))
    try:
        return load_params(scratch), []
    except ParamsError as exc:
        return base, list(exc.problems)


def parse_placed(lines) -> tuple[list[Placement], list[str]]:
    out: list[Placement] = []
    problems: list[str] = []
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        try:
            out.append(parse_placement(text))
        except ValueError:
            problems.append(f"not a placement: {text!r}")
    return out, problems


def plate_displacement(setup: Setup) -> tuple[np.ndarray, float]:
    if setup.plate_offset == "nominal":
        return np.zeros(2), 0.0
    if setup.plate_offset == "typed":
        return (np.array([setup.plate_dx_mm, setup.plate_dy_mm]) * 1e-3,
                math.radians(setup.plate_yaw_deg))
    return worlds.displacement(int(setup.seed))


def supply_displacement(setup: Setup) -> tuple[np.ndarray, float]:
    if setup.supply_offset == "nominal":
        return np.zeros(2), 0.0
    if setup.supply_offset == "typed":
        return (np.array([setup.supply_dx_mm, setup.supply_dy_mm]) * 1e-3,
                math.radians(setup.supply_yaw_deg))
    return worlds.brick_displacement(int(setup.seed))


def build(setup: Setup, params: ParamFile, *, prepare=None) -> dict:
    """The world the arm really runs in.

    Returns the scene and the offsets nobody tells the controller about. `prepare` is handed the
    compiled model before the first reset, which is where the render look is applied.
    """
    placed, problems = parse_placed(setup.placed)
    d_xy, d_yaw = plate_displacement(setup)
    held = None if setup.pick else HeldBrick(
        nx=worlds.WANTED[0], ny=worlds.WANTED[1], colour=worlds.WANTED[2],
        offset_jitter=(0.0, 0.002, 0.0) if setup.plate_offset != "nominal" else (0.0, 0.0, 0.0))
    cfg = worlds.scene_cfg(
        plate_xy=np.asarray(worlds.NOMINAL_PLATE_XY) + d_xy, plate_yaw=d_yaw,
        placed=placed, held=held,
        home_tcp=(worlds.NOMINAL_PLATE_XY[0], worlds.NOMINAL_PLATE_XY[1], 0.075), cameras=(),
    )
    if setup.pick:
        cfg.supply = (worlds.WANTED, worlds.DISTRACTOR)
        cfg.supply_xy = worlds.SUPPLY_XY
    cfg, skipped = to_scene_config(
        params, base=cfg,
        plate_xy=(float(cfg.plate_xy[0]), float(cfg.plate_xy[1])), plate_yaw=float(cfg.plate_yaw),
        breakable=bool(setup.breakable), damage_monitor=bool(setup.damage_monitor),
        # `offscreen` is the framebuffer every camera in the scene renders into, not a camera
        # resolution: the parameter file's camera fields are smaller than the 512x512 wrist
        # window the loop reads, and a framebuffer under that size refuses the render.
        offscreen=(max(int(cfg.offscreen[0]), MIN_OFFSCREEN[0]),
                   max(int(cfg.offscreen[1]), MIN_OFFSCREEN[1])),
    )
    scene = build_scene(cfg)
    if prepare is not None:
        prepare(scene.model)
    reset_scene(scene, np.random.default_rng(int(setup.seed)))
    b_xy, b_yaw = supply_displacement(setup)
    if setup.pick and (np.any(b_xy) or b_yaw):
        worlds.nudge_brick(scene, worlds.SUPPLY_BODY, b_xy, b_yaw)
    return {"scene": scene, "plate_d_xy": d_xy, "plate_d_yaw": d_yaw,
            "supply_d_xy": b_xy, "supply_d_yaw": b_yaw,
            "problems": problems, "skipped": skipped}


def target_cell(goal) -> tuple[int, int]:
    """The grid cell the compiled command's first step asks for."""
    p = goal.steps[0].placement
    return int(p.x), int(p.y)


def brick_names(scene: Scene) -> list[str]:
    """Every loose brick a disturbance may act on, by body name."""
    names = list(scene.placed_names)
    for name in (worlds.SUPPLY_BODY, worlds.SPARE_BODY):
        if name in scene.bricks:
            names.append(name)
    if scene.held_name:
        names.append(scene.held_name)
    return names


# ============================================================================== disturbances ===


def nudge(scene: Scene, body: str, dx_mm: float, dy_mm: float, dyaw_deg: float) -> None:
    """Move one loose brick, in millimetres and degrees."""
    worlds.nudge_brick(scene, body, np.array([dx_mm, dy_mm]) * 1e-3, math.radians(dyaw_deg))


def remove(scene: Scene, body: str) -> None:
    """Take one brick off the plate: parked off the table, at rest."""
    if body == scene.held_name:
        raise ValueError("the brick in the jaws is not removable; nudge it instead")
    adr = scene.free_joint_adr(body)
    scene.data.qpos[adr:adr + 3] = [PARK_XY[0], PARK_XY[1], 0.02]
    scene.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_free")
    dof = int(scene.model.jnt_dofadr[jid])
    scene.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(scene.model, scene.data)


def displace_plate(scene: Scene, dx_mm: float, dy_mm: float, dyaw_deg: float) -> None:
    """Move the baseplate, and everything standing on it, as one rigid body.

    The plate is a static body, so this writes `model.body_pos` and `model.body_quat`; the bricks
    already placed are free bodies and are carried by the same transform, which is what moving a
    plate with a build on it does.
    """
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    centre = scene.model.body_pos[bid].copy()
    d = np.array([dx_mm, dy_mm, 0.0]) * 1e-3
    yaw = math.radians(dyaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)

    quat = scene.model.body_quat[bid].copy()
    turn = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, turn, quat)
    scene.model.body_pos[bid] = centre + d
    scene.model.body_quat[bid] = new_quat

    for name in scene.placed_names:
        pos, bq = scene.body_pose(name)
        local = np.asarray(pos, dtype=float) - centre
        moved = centre + d + np.array([c * local[0] - s * local[1],
                                       s * local[0] + c * local[1], local[2]])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, turn, np.asarray(bq, dtype=float))
        set_brick_pose(scene, name, moved, out)
    mujoco.mj_forward(scene.model, scene.data)


class ArmHold:
    """Torque off on the arm for a number of vision ticks: the servos stop driving and let go.

    A position actuator holds its joint through its own gain, so switching one off is writing
    that gain to zero and putting it back afterwards. Scene setup, like every other disturbance
    here -- the controller keeps writing tool targets throughout and is never told.
    """

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.left = 0
        self._gain = scene.model.actuator_gainprm.copy()
        self._bias = scene.model.actuator_biasprm.copy()

    def start(self, ticks: int) -> None:
        if self.left == 0:
            self.scene.model.actuator_gainprm[:6, 0] = 0.0
            self.scene.model.actuator_biasprm[:6, 1] = 0.0
        self.left = max(int(ticks), 0)

    def tick(self) -> None:
        if self.left <= 0:
            return
        self.left -= 1
        if self.left == 0:
            self.release()

    def release(self) -> None:
        self.scene.model.actuator_gainprm[:] = self._gain
        self.scene.model.actuator_biasprm[:] = self._bias
        self.left = 0
