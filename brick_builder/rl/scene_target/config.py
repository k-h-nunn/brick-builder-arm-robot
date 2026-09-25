"""Configuration of the scene-target environment: stages, action bounds, reward weights.

Distances are in metres and angles in radians inside the code; names ending in `_mm` or `_deg`
are the exceptions and say so. Scene-state distances are in mm-eq (see `codec.py`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "ACTION_SPACES", "FEEDFORWARD_CLOCKS", "LEVELS", "LEVEL_ALIASES", "LevelSpec", "RewardWeights", "STEP_TITLES",
    "SceneTargetConfig", "level_spec", "resolve_level",
]

STEP_TITLES = (
    "Approach", "Grasp", "Lift", "Transport", "Align",
    "Press the brick down", "Release", "Retract", "Verify",
)
"""The nine step titles of the placement script (`action.place.STEP_TITLES`); the progress
observation carries a one-hot over them. Viewing-pose stages have no step and leave it zero."""

FEEDFORWARD_CLOCKS = ("progress", "recording")
"""`progress`: the recorded writes follow the progress index and pause while it holds (the
default). `recording`: one recorded frame per tick whatever the distance, the camera-off control
condition; the index follows the recording up to the last frame, where the settle and hold rule
applies."""

ACTION_SPACES = ("tool", "joint")
"""`tool`: change of the tool-target offset (default). `joint`: 5 joint-position increments, the
joint-space variant; its executor is not part of this package version."""


@dataclass(frozen=True)
class LevelSpec:
    """One curriculum stage."""

    name: str
    code: str
    title: str
    family: str
    """Target-sequence family: stages that share a family share cached target sequences."""
    feedforward: bool
    """Whether the recorded commands drive the arm (`u = 0` is then open-loop replay)."""
    displaced: bool
    """Whether the baseplate is displaced from nominal in the world the episode runs in."""
    action_dims: tuple[str, ...]
    action_scale: tuple[float, ...]
    """Largest change of the offset per tick, per action dimension [m, m, m, rad]."""
    ceiling_xy: float
    ceiling_z: float
    ceiling_yaw: float
    max_hold: int
    max_ticks_extra: int
    """Episode length bound: `T + max_ticks_extra` (L0) or `1.5 T + max_ticks_extra` (L1-L3)."""
    max_ticks_factor: float
    implemented: bool


def _deg(x: float) -> float:
    return math.radians(x)


LEVELS: dict[str, LevelSpec] = {
    "viewing_pose": LevelSpec(
        name="viewing_pose", code="L0a", title="Viewing pose", family="viewing_pose",
        feedforward=False, displaced=False,
        action_dims=("dx", "dy", "dz", "dyaw"),
        action_scale=(0.003, 0.003, 0.002, _deg(2.0)),
        ceiling_xy=0.050, ceiling_z=0.030, ceiling_yaw=_deg(20.0),
        max_hold=30, max_ticks_extra=30, max_ticks_factor=1.0, implemented=True,
    ),
    "viewing_pose_displaced": LevelSpec(
        name="viewing_pose_displaced", code="L0b", title="Viewing pose, displaced",
        family="viewing_pose", feedforward=True, displaced=True,
        action_dims=("dx", "dy", "dyaw"),
        action_scale=(0.002, 0.002, _deg(1.5)),
        ceiling_xy=0.020, ceiling_z=0.0, ceiling_yaw=_deg(10.0),
        max_hold=30, max_ticks_extra=30, max_ticks_factor=1.0, implemented=True,
    ),
    "align": LevelSpec(
        name="align", code="L1", title="Align", family="placement",
        feedforward=True, displaced=True,
        action_dims=("dx", "dy", "dyaw"),
        action_scale=(0.002, 0.002, _deg(1.5)),
        ceiling_xy=0.020, ceiling_z=0.0, ceiling_yaw=_deg(10.0),
        max_hold=70, max_ticks_extra=70, max_ticks_factor=1.5, implemented=False,
    ),
    "placement": LevelSpec(
        name="placement", code="L2", title="Placement", family="placement",
        feedforward=True, displaced=True,
        action_dims=("dx", "dy", "dyaw"),
        action_scale=(0.002, 0.002, _deg(1.5)),
        ceiling_xy=0.020, ceiling_z=0.0, ceiling_yaw=_deg(10.0),
        max_hold=70, max_ticks_extra=70, max_ticks_factor=1.5, implemented=False,
    ),
    "pick_and_place": LevelSpec(
        name="pick_and_place", code="L3", title="Pick and place", family="pick_and_place",
        feedforward=True, displaced=True,
        action_dims=("dx", "dy", "dyaw"),
        action_scale=(0.002, 0.002, _deg(1.5)),
        ceiling_xy=0.020, ceiling_z=0.0, ceiling_yaw=_deg(10.0),
        max_hold=70, max_ticks_extra=70, max_ticks_factor=1.5, implemented=False,
    ),
}
"""The five stages of the curriculum. L1-L3 are declared so the observation layout (level
one-hot) is fixed now; their executor path (prefix, freeze, play-out, `verify_placement`) is not
in this package version and `reset` refuses them."""

LEVEL_ALIASES = {stage.code: name for name, stage in LEVELS.items()}


def resolve_level(level: str) -> str:
    """Stage name from a name or a code (`"L0a"` -> `"viewing_pose"`)."""
    if level in LEVELS:
        return level
    if level in LEVEL_ALIASES:
        return LEVEL_ALIASES[level]
    raise KeyError(f"unknown level {level!r}; known: {sorted(LEVELS)} or {sorted(LEVEL_ALIASES)}")


def level_spec(level: str) -> LevelSpec:
    return LEVELS[resolve_level(level)]


@dataclass(frozen=True)
class RewardWeights:
    """Reward weights."""

    w_p: float = 0.2
    """Per mm-eq of scene-state distance closed toward the pre-step target."""
    w_d: float = 0.1
    """Tracking term, per multiple of `tol_mm`, capped at 5."""
    w_adv: float = 0.5
    w_h: float = 0.02
    w_u: float = 0.02
    w_s: float = 0.02
    w_dmg: float = 5.0
    success: float = 10.0
    lateral: float = 2.0
    failure: float = 5.0


@dataclass(frozen=True)
class SceneTargetConfig:
    """Scene-target environment configuration."""

    level: str = "viewing_pose"
    source: str = "ideal"
    """`perception.sources` name: `ideal` (ground truth) or `learned` (the convolutional model)."""
    source_kwargs: Mapping = field(default_factory=dict)
    """For `learned`: `weights` (explicit checkpoint path; default the shipped file), `threads`,
    `device` (`cpu` only in this version)."""
    horizon: int = 5
    """H: target frames exposed, `S*_{k+1..k+H}`."""
    action_space: str = "tool"
    """`tool` (default) or `joint` (the joint-space variant; not implemented here)."""
    action_scale: tuple[float, ...] | None = None
    """Per-tick bound per action dimension; `None` = the stage's."""
    action_ceiling: tuple[float, float, float] | None = None
    """(xy norm [m], |z| [m], |yaw| [rad]) bound on the accumulated offset; `None` = the stage's."""
    tol_mm: float | None = None
    """Success tolerance [mm-eq]; `None` = from the calibration file, else 0.5."""
    advance_floor_mm: float = 1.0
    tube_scale: float = 1.6
    settle_ticks: int = 3
    max_hold: int | None = None
    """`None` = the stage's (30 on L0, 70 otherwise)."""
    lost_view_ticks: int = 20
    displacement_scale: float = 1.0
    """Scales the plate displacement (4-8 mm, 2-3 deg) on displaced stages."""
    start_offset_mm: float = 0.0
    start_offset_deg: float = 0.0
    world_seeds: tuple[int, ...] = tuple(range(1000, 1100))
    eval_seeds: tuple[int, ...] = tuple(range(0, 10)) + tuple(range(20, 50))
    target_cell: tuple[int, int] = (4, 3)
    include_image: bool = False
    render_mode: str | None = None
    reward: RewardWeights = field(default_factory=RewardWeights)
    feedforward: bool | None = None
    """`None` = the stage's (off on L0a, on elsewhere). On L0a `True` plays the recorded path
    from the start pose."""
    feedforward_clock: str = "progress"
    """`progress` or `recording` (see `FEEDFORWARD_CLOCKS`). `recording` with `u = 0` is
    open-loop replay of the recording."""
    settle_control_ticks: int = 25
    """Control ticks (20 ms each) of physics at the start pose before the first observation."""
    privileged_gt_z: bool = True
    """Compute the ground-truth rows `gt_z` of every frame (free on `ideal`, one extra
    ground-truth pass on other sources)."""
    timings_in_info: bool = True
    """Per-tick wall-clock timings in `info`. With bit-exact renders (`render_samples=0`) the
    only non-deterministic `info` entry."""
    cache_dir: str | None = None
    """Root of the target cache; `None` = `data_cache/scene_target` in the repository."""
    calib_dir: str | None = None
    """Root of the calibration files; `None` = `<cache_dir>/calib`."""
    render_samples: int | None = None
    """Multisample count of the offscreen render; `None` = the model's own (4). With 4 samples
    two renders of one state differ on about 50 of 65,536 output pixels. On `ideal` only the
    colour fields of the slot rows change (by up to 4e-5); on `learned` the CNN reads the RGB, so
    every slot field, the reward and `d_mm` change (measured up to 5e-2 and 0.15 mm-eq). 0 gives
    bit-exact renders and exact repeats on both sources."""
    K: int = 8

    def __post_init__(self) -> None:
        resolve_level(self.level)
        if self.feedforward_clock not in FEEDFORWARD_CLOCKS:
            raise ValueError(f"feedforward_clock must be one of {FEEDFORWARD_CLOCKS}; "
                             f"got {self.feedforward_clock!r}")
        if self.action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}; got {self.action_space!r}")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")

    @property
    def stage(self) -> LevelSpec:
        return level_spec(self.level)

    def with_overrides(self, **overrides) -> "SceneTargetConfig":
        return dataclasses.replace(self, **overrides)

    # -- resolved per-stage values -----------------------------------------------------------

    def resolved_feedforward(self) -> bool:
        return self.stage.feedforward if self.feedforward is None else bool(self.feedforward)

    def resolved_scale(self) -> tuple[float, ...]:
        scale = self.stage.action_scale if self.action_scale is None else tuple(self.action_scale)
        if len(scale) != len(self.stage.action_dims):
            raise ValueError(f"action_scale has {len(scale)} entries; the stage has "
                             f"{len(self.stage.action_dims)} action dimensions")
        return tuple(float(s) for s in scale)

    def resolved_ceiling(self) -> tuple[float, float, float]:
        if self.action_ceiling is not None:
            xy, z, yaw = self.action_ceiling
            return float(xy), float(z), float(yaw)
        s = self.stage
        return s.ceiling_xy, s.ceiling_z, s.ceiling_yaw

    def resolved_max_hold(self) -> int:
        return self.stage.max_hold if self.max_hold is None else int(self.max_hold)

    def max_ticks(self, T: int) -> int:
        s = self.stage
        return int(math.ceil(s.max_ticks_factor * T)) + s.max_ticks_extra
