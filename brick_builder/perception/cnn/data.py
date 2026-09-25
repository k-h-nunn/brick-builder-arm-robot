"""Training frames for the convolutional scene model, from the simulator alone.

Every sample is exactly what the ideal source sees at run time: one 256x256 RGB window of the
wrist camera, cropped and resized by `perception.views` from a wide render, with the renderer's
own per-pixel instance ids and metric depth cropped the same way. Nothing here reads a real
photograph; nothing here is hand-labelled.

What the scenes cover
---------------------
The controller only ever looks at a handful of situations, and the training set is built to cover
all of them rather than to be large:

- the supply area: loose bricks standing on the table, with the optional tray present and with it
  taken away;
- the baseplate: empty, and carrying one to six placed bricks, including a second layer;
- a brick held in the jaws over the plate, at several heights above it;
- an approach: the open jaws coming down onto a brick;
- same-coloured bricks touching each other, which colour alone cannot separate;
- bricks running off the edge of the window;
- the arm's own links filling part of the frame.

Poses come from two places: randomly sampled viewing poses (a tool-down inverse-kinematics
solution over the region being looked at, plus a small joint perturbation so the camera is not
always perfectly level), and real `control.pick_place.PickPlace` episodes sampled every few
control ticks, which is where the awkward mid-motion poses and the held-brick-over-the-plate views
come from.

What is randomized
------------------
Lights (position, brightness, colour), the camera (diagonal field of view over the lens's stated
100-130 degree range, position/orientation jitter, barrel distortion, exposure, white balance,
gamma, sensor noise, blur), brick colours with a small shade jitter, the table's colour, the
baseplate's pose, and the colour of the arm's printed parts -- the real arm's colour is not known
and the upstream meshes are all one colour, so the network must not be allowed to learn it. Green
bricks on the green baseplate are included and labelled like any other brick; the evaluation
reports their accuracy separately because they are the hardest case by construction.

Storage
-------
Shards of 64 samples as compressed `.npz` under `data_cache/vision_model/<split>/`, each with a
sidecar `.json` describing what went into them. Splits are separated by seed: `train` and `val`
draw from disjoint seed lists, so no scene layout, colour draw or camera model is shared.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .labels import ARM_INSTANCE, FIRST_BRICK_INSTANCE, PLATE_INSTANCE, class_map_from_instance

DEFAULT_CACHE = Path("data_cache/vision_model")
"""Where shards live by default. Git-ignored: this is a rebuildable cache, not a source file."""

SHARD_SIZE = 64
OUTPUT_SIZE = 256
RENDER_SIZE = (512, 512)

ARM_BODIES: tuple[str, ...] = (
    "base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1",
)
"""Every body that belongs to the arm, including the fixed jaw (part of `gripper`) and the moving
jaw. `scene.compose` names them; they are listed rather than discovered so that a body added to a
scene later cannot quietly change what "arm" means to a trained checkpoint."""

PRINTED_PART_RGB = (1.0, 0.82, 0.12)
"""The colour the upstream SO-101 meshes are all shipped in. Every material matching it is a
printed part, and gets repainted to one of `PRINTED_PART_COLOURS` per scene."""

PRINTED_PART_COLOURS: dict[str, tuple[float, float, float]] = {
    "white": (0.92, 0.92, 0.90),
    "black": (0.12, 0.12, 0.13),
    "grey": (0.55, 0.55, 0.57),
    "orange": (0.95, 0.45, 0.08),
    "blue": (0.10, 0.28, 0.70),
    "yellow": (1.00, 0.82, 0.12),
    "red": (0.75, 0.10, 0.10),
}
"""Plausible filament colours for a printed arm. The real arm's colour is unknown, so the model
is shown all of them."""

BRICK_COLOURS: tuple[str, ...] = ("red", "blue", "yellow", "green", "orange", "white")
"""The palette `bricks.spec.PALETTE` defines. Green is included on purpose: green bricks on the
green baseplate are a real case, and the evaluation scores them separately."""

TABLE_RGB_RANGE = ((0.30, 0.30, 0.30), (0.78, 0.76, 0.72))
"""The table is a plain, texture-free plane; its colour is drawn uniformly between these two."""


# ------------------------------------------------------------------------------ instance map ---


def perception_instance_lut(scene) -> dict[str, int]:
    """Body name -> instance id, in `labels.py`'s convention, for `SceneRenderer.instance_lut`.

    Bodies left out of the returned map render as id 0, which both this package and the ideal
    source read as "unlabelled background" -- that is where the table plane, the tray and the
    snap model's invisible collet bodies end up.

    This is not `scene.instance_ids`: that map assigns the *world* body id 1, so the table plane
    carries the arm's id and the ideal source reports table-plus-arm as a single object. Feeding
    both producers the map built here is what makes their slot states comparable.
    """
    lut: dict[str, int] = {}
    for name in ARM_BODIES:
        lut[name] = ARM_INSTANCE
    lut[scene.plate.name] = PLATE_INSTANCE
    next_id = FIRST_BRICK_INSTANCE
    for name in scene.bricks:
        if "_collet_" in name:
            continue
        lut[name] = next_id
        next_id += 1
    return lut


# ------------------------------------------------------------------------- visual randomizing ---


def _jitter(rgb, rng: np.random.Generator, amount: float = 0.06) -> tuple[float, float, float]:
    arr = np.asarray(rgb, dtype=float) * (1.0 + rng.uniform(-amount, amount, size=3))
    return tuple(float(v) for v in np.clip(arr, 0.0, 1.0))


def paint_arm(scene, rng: np.random.Generator) -> str:
    """Repaint every printed part of the arm one colour, drawn from `PRINTED_PART_COLOURS`.

    The servo casings keep their own (black) material: they are injection-moulded parts that come
    with the motors, not something a builder chooses.
    """
    model = scene.model
    name = str(rng.choice(list(PRINTED_PART_COLOURS)))
    rgb = _jitter(PRINTED_PART_COLOURS[name], rng, 0.05)
    base = np.asarray(PRINTED_PART_RGB)
    for m in range(model.nmat):
        if np.allclose(model.mat_rgba[m, :3], base, atol=2e-2):
            model.mat_rgba[m, :3] = rgb
    return name


def paint_table(scene, rng: np.random.Generator) -> tuple[float, float, float]:
    """Give the table plane a fresh flat colour (no texture: the real table has none modelled)."""
    import mujoco

    from ...scene.compose import TABLE_GEOM

    gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    if gid < 0:
        return (0.0, 0.0, 0.0)
    lo, hi = np.asarray(TABLE_RGB_RANGE[0]), np.asarray(TABLE_RGB_RANGE[1])
    rgb = tuple(float(v) for v in rng.uniform(lo, hi))
    scene.model.geom_rgba[gid, :3] = rgb
    return rgb


def paint_brick(scene, body: str, rgb: Sequence[float]) -> None:
    """Repaint one brick's visible shell, studs included.

    Only the geoms that were built in that brick's own colour are touched, the same rule
    `scene.randomize._apply_visual` follows, so the model's own nominal colours stay the
    reference and nothing else on the body (a marker dot, a collision proxy) is repainted.
    """
    import mujoco

    from ...bricks.spec import PALETTE

    handle = scene.bricks[body] if body in scene.bricks else scene.plate
    base = np.asarray(PALETTE.get(handle.colour, PALETTE["white"]), dtype=float)
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"unknown body {body!r}")
    start = int(scene.model.body_geomadr[bid])
    for gid in range(start, start + int(scene.model.body_geomnum[bid])):
        if np.allclose(scene.nominal["geom_rgba"][gid, :3], base, atol=2e-3):
            scene.model.geom_rgba[gid, :3] = np.asarray(rgb, dtype=float)


def repaint_bricks(scene, rng: np.random.Generator, colours: dict[str, str] | None = None) -> dict[str, str]:
    """Give every brick (and the baseplate) a palette colour with a small shade jitter.

    `colours` fixes the palette name per body when the caller wants a particular arrangement --
    two touching bricks of the same colour, for instance. Anything not named is drawn at random.
    """
    from ...bricks.spec import PALETTE

    chosen = dict(colours or {})
    for name in list(scene.bricks) + [scene.plate.name]:
        if "_collet_" in name:
            continue
        if name == scene.plate.name:
            chosen.setdefault(name, "green")
        else:
            chosen.setdefault(name, str(rng.choice(BRICK_COLOURS)))
        paint_brick(scene, name, _jitter(PALETTE[chosen[name]], rng, 0.07))
    return chosen


def randomize_lights(scene, rng: np.random.Generator) -> None:
    """Move and recolour the lights the scene was compiled with.

    A compiled model cannot gain or lose lights without recompiling (which costs far more than a
    whole episode of physics), so the rig's *size* is fixed and its position, direction, colour
    and ambient level are what move.
    """
    model = scene.model
    nominal_pos = scene.nominal["light_pos"]
    for i in range(model.nlight):
        model.light_pos[i] = nominal_pos[i] + rng.uniform(-0.25, 0.25, size=3) * np.array([1.0, 1.0, 0.5])
        level = float(rng.uniform(0.45, 1.10))
        tint = 1.0 + rng.uniform(-0.12, 0.12, size=3)
        model.light_diffuse[i] = np.clip(level * tint, 0.0, 1.0)
        model.light_specular[i] = np.clip(0.3 * level * tint, 0.0, 1.0)
        model.light_ambient[i] = np.full(3, float(rng.uniform(0.06, 0.22)))
        d = np.asarray(model.light_dir[i], dtype=float) + rng.uniform(-0.2, 0.2, size=3)
        n = np.linalg.norm(d)
        if n > 1e-9:
            model.light_dir[i] = d / n


def jitter_camera(scene, rng: np.random.Generator) -> None:
    """Small mount error on the wrist camera: a few millimetres and a few degrees."""
    import mujoco

    from ...robot.patch_so101 import WRIST_CAM

    cam = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    if cam < 0:
        return
    scene.model.cam_pos[cam] = scene.nominal["cam_pos"][cam] + rng.uniform(-0.005, 0.005, size=3)
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    angle = math.radians(float(rng.uniform(-3.0, 3.0)))
    dq = np.array([math.cos(0.5 * angle), *(math.sin(0.5 * angle) * axis)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, scene.nominal["cam_quat"][cam], dq)
    scene.model.cam_quat[cam] = out


def jitter_plate(scene, rng: np.random.Generator, max_xy: float = 0.02, max_yaw_deg: float = 10.0) -> None:
    """Move the baseplate. Wider than `scene.randomize`'s +-3 mm / +-2 degrees on purpose: that
    range models the *build* error a taught pose has to survive, while this one models not knowing
    where the plate is at all, which is what a vision model has to cope with."""
    import mujoco

    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    yaw = scene.cfg.plate_yaw + math.radians(float(rng.uniform(-max_yaw_deg, max_yaw_deg)))
    scene.model.body_pos[bid, 0] = scene.cfg.plate_xy[0] + float(rng.uniform(-max_xy, max_xy))
    scene.model.body_pos[bid, 1] = scene.cfg.plate_xy[1] + float(rng.uniform(-max_xy, max_xy))
    scene.model.body_quat[bid] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def randomize_appearance(scene, rng: np.random.Generator) -> dict:
    """Everything visual, in one call: arm colour, table colour, brick colours, lights, camera."""
    arm_colour = paint_arm(scene, rng)
    table_rgb = paint_table(scene, rng)
    brick_colours = repaint_bricks(scene, rng)
    randomize_lights(scene, rng)
    jitter_camera(scene, rng)
    return {"arm_colour": arm_colour, "table_rgb": table_rgb, "brick_colours": brick_colours}


# ------------------------------------------------------------------------------- the windows ---


def sample_camera_model(rng: np.random.Generator):
    """One draw of the lens and sensor: a diagonal field of view in the stated range, plus the
    post-process `render.domain_rand` applies (barrel distortion, exposure, white balance, gamma,
    shot and read noise, blur)."""
    from ...render.domain_rand import sample_postprocess_config
    from ...robot.patch_so101 import (
        WRIST_CAM_DIAGONAL_FOV_RANGE_DEG,
        WRIST_CAM_DISTORTION_K1_RANGE,
        WRIST_CAM_DISTORTION_K2_RANGE,
    )

    diagonal = float(rng.uniform(*WRIST_CAM_DIAGONAL_FOV_RANGE_DEG))
    post = sample_postprocess_config(
        rng, ranges={"k1": WRIST_CAM_DISTORTION_K1_RANGE, "k2": WRIST_CAM_DISTORTION_K2_RANGE}
    )
    return diagonal, post


def window_for_roi(
    scene,
    roi_world: np.ndarray,
    diagonal_fov_deg: float,
    rng: np.random.Generator | None = None,
    *,
    margin: float = 0.25,
    name: str = "train",
):
    """A `views.LookPose` framing `roi_world` from wherever the wrist camera currently is, at a
    chosen diagonal field of view.

    `views.attention_view` does the same thing at the nominal field of view only; this re-does its
    two steps (set the camera's vertical fov for the render resolution, then ask
    `views.attention_crop` for the square crop) so the lens can be randomized too. Returns `None`
    when the region is not in front of the camera, which is a real answer and not an error.
    """
    import mujoco

    from ..backproject import camera_geometry
    from ..views import LookPose, attention_crop
    from ...robot.patch_so101 import WRIST_CAM, fovy_from_diagonal_fov

    fovy = fovy_from_diagonal_fov(diagonal_fov_deg, *RENDER_SIZE)
    cam = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
    scene.model.cam_fovy[cam] = fovy
    mujoco.mj_forward(scene.model, scene.data)
    geom_full = camera_geometry(scene.model, scene.data, WRIST_CAM, *RENDER_SIZE)
    try:
        crop = attention_crop(geom_full, roi_world, RENDER_SIZE, margin=margin, min_crop_px=64.0)
    except ValueError:
        return None
    if rng is not None:
        # Slide the crop around a little so objects fall off the edge of the window sometimes --
        # `SceneState`'s border fraction only means something if the model has seen cut objects.
        x0, y0, x1, y1 = crop
        side = x1 - x0
        shift = rng.integers(-side // 4, side // 4 + 1, size=2)
        x0 = int(np.clip(x0 + shift[0], 0, RENDER_SIZE[0] - side))
        y0 = int(np.clip(y0 + shift[1], 0, RENDER_SIZE[1] - side))
        crop = (x0, y0, x0 + side, y0 + side)
    return LookPose(
        name=name, camera=WRIST_CAM, render_size=RENDER_SIZE, crop=crop, fovy_deg=fovy,
        output_size=OUTPUT_SIZE, tcp=None,
    )


def roi_around(centre, half=(0.06, 0.06, 0.035)) -> np.ndarray:
    from ..views import roi_box_corners

    return roi_box_corners(np.asarray(centre, dtype=float), np.asarray(half, dtype=float))


# ------------------------------------------------------------------------------- one sample ---


@dataclass
class Sample:
    """One training frame and its labels."""

    rgb: np.ndarray            # uint8 [256, 256, 3]
    instance: np.ndarray       # uint8 [256, 256], `labels.py`'s convention
    depth: np.ndarray          # float16 [256, 256], metres, NaN where invalid
    meta: dict = field(default_factory=dict)

    @property
    def classes(self) -> np.ndarray:
        return class_map_from_instance(self.instance)


def render_sample(scene, pose, renderer, post_cfg, rng: np.random.Generator, meta: dict) -> Sample:
    """Render `pose` and apply the camera post-process, returning labels aligned to the image.

    The post-process remaps instance and depth with the same distortion map as the colour channel
    (nearest-neighbour, never blended), so a label always belongs to the pixel it sits on.
    """
    from ...render.domain_rand import apply_postprocess
    from ..views import render_view

    frame, geometry = render_view(scene, pose, renderer)
    frame = apply_postprocess(frame, post_cfg, rng)
    inst = np.clip(frame.instance, 0, 255).astype(np.uint8)
    return Sample(
        rgb=np.ascontiguousarray(frame.rgb),
        instance=inst,
        depth=frame.depth.astype(np.float16),
        meta={**meta, "crop": list(pose.crop), "fovy_deg": float(pose.fovy_deg),
              "fx": float(geometry.fx), "fy": float(geometry.fy)},
    )


# --------------------------------------------------------------------------- posing the arm ---


def _set_arm(scene, q: np.ndarray, grip: float | None = None) -> None:
    import mujoco

    scene.data.qpos[scene.arm_qpos_adr[:5]] = q
    scene.data.ctrl[:5] = q
    if grip is not None:
        scene.data.qpos[scene.arm_qpos_adr[5]] = grip
        scene.data.ctrl[5] = grip
    mujoco.mj_forward(scene.model, scene.data)


def _viewing_pose(scene, target_xy, rng: np.random.Generator, z_range=(0.045, 0.095), tries: int = 6):
    """Joint angles for a tool-down look at `target_xy`, plus a small joint perturbation.

    The perturbation is what stops every training frame being taken from a perfectly level
    camera: the real arm's wrist is never exactly where the solver says it is.

    The height band is deliberately low. This arm has an unreachable dead zone directly over the
    plate from roughly 9 cm upwards (`perception.views` measured it), so asking for a tool-down
    pose higher than that simply fails; a few draws are tried and `None` comes back if none of
    them is reachable.
    """
    from ...scene.compose import tool_down_ik

    lo, hi = scene.kin.jnt_range[:5, 0], scene.kin.jnt_range[:5, 1]
    for _ in range(tries):
        tcp = np.array([
            float(target_xy[0]) + float(rng.uniform(-0.03, 0.03)),
            float(target_xy[1]) + float(rng.uniform(-0.03, 0.03)),
            float(rng.uniform(*z_range)),
        ])
        yaw = float(rng.uniform(-math.pi, math.pi))
        try:
            q = tool_down_ik(scene.kin, tcp, yaw, q0=np.array([0.0, -0.5, 0.9, 1.2, 0.0]))
        except RuntimeError:
            continue
        return np.clip(q + rng.normal(0.0, 0.03, size=5), lo, hi)
    return None


# --------------------------------------------------------------------------- the scene kinds ---


def _plate_centre(scene) -> np.ndarray:
    import mujoco

    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
    return scene.model.body_pos[bid].copy()


def _random_structure(rng: np.random.Generator, touching_same_colour: bool):
    """One to six placed bricks on an 8x8 plate, sometimes two layers, sometimes two of the same
    colour side by side (the case colour alone cannot separate)."""
    from ...bricks.structures import Placement, validate_structure

    n = int(rng.integers(1, 7))
    placed: list = []
    occupied: set[tuple[int, int, int]] = set()
    colours = list(BRICK_COLOURS)
    shared = str(rng.choice(colours))
    for i in range(n):
        for _ in range(30):
            # `Placement` is (h, w, ...): `h` runs along y, `w` along x.
            h, w = (2, 4) if rng.random() < 0.6 else (2, 2)
            if rng.random() < 0.5:
                h, w = w, h
            x = int(rng.integers(0, 9 - w))
            y = int(rng.integers(0, 9 - h))
            z = 1 if (placed and rng.random() < 0.25) else 0
            cells = {(x + a, y + b, z) for a in range(w) for b in range(h)}
            if cells & occupied:
                continue
            if z == 1 and not {(cx, cy, 0) for cx, cy, _ in cells} <= occupied:
                continue
            colour = shared if (touching_same_colour and i < 2) else str(rng.choice(colours))
            placed.append(Placement(h=h, w=w, x=x, y=y, z=z, colour=colour))
            occupied |= cells
            break
    while placed and validate_structure(placed, grid_shape=(8, 8)):
        placed.pop()
    return placed


def _hide_tray(scene) -> None:
    """Take the tray out of every render without taking it out of the physics.

    Moving its visual geoms into the collision group is enough: `SceneRenderer` hides that group,
    so the loose bricks look like they are standing on the bare table (they are 2 mm higher than
    they would be, which is well inside the pose jitter the scenes already carry).
    """
    import mujoco

    from ...bricks.generator import COLLISION_GROUP

    for g in range(scene.model.ngeom):
        name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("tray_") and "brick" not in name:
            scene.model.geom_group[g] = COLLISION_GROUP


def _scatter_loose_bricks(scene, rng: np.random.Generator, centre_xy) -> None:
    """Put the tray's bricks down at random spots and angles around the supply area."""
    import mujoco

    names = [n for n in scene.bricks if n.startswith("tray_brick_")]
    spots: list[np.ndarray] = []
    for name in names:
        for _ in range(40):
            xy = np.asarray(centre_xy, dtype=float) + rng.uniform(-0.055, 0.055, size=2)
            if all(np.linalg.norm(xy - s) > 0.045 for s in spots):
                spots.append(xy)
                break
        else:
            spots.append(np.asarray(centre_xy, dtype=float) + rng.uniform(-0.06, 0.06, size=2))
        yaw = float(rng.uniform(-math.pi, math.pi))
        adr = scene.free_joint_adr(name)
        scene.data.qpos[adr:adr + 3] = [spots[-1][0], spots[-1][1], 0.004]
        scene.data.qpos[adr + 3:adr + 7] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
    mujoco.mj_forward(scene.model, scene.data)


def _supply_scene(rng: np.random.Generator, frames: int) -> Iterator[Sample]:
    """Loose bricks in the supply area, tray shown or hidden."""
    from ...render.renderer import SceneRenderer
    from ...robot.patch_so101 import ArmConfig
    from ...scene.compose import SceneConfig, build_scene, reset_scene, settle

    with_tray = bool(rng.random() < 0.5)
    cfg = SceneConfig(
        held=None, placed=[], tray=True, tray_bricks=int(rng.integers(2, 5)),
        cameras=(), arm=ArmConfig(),
    )
    scene = build_scene(cfg)
    if not with_tray:
        _hide_tray(scene)
    jitter_plate(scene, rng)
    reset_scene(scene, rng)
    randomize_appearance(scene, rng)
    centre = np.asarray(cfg.tray_xy, dtype=float)
    if not with_tray:
        _scatter_loose_bricks(scene, rng, centre)
        settle(scene, 0.25)

    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        yield from _frames_at(
            scene, renderer, rng, frames,
            look_at=lambda r: centre + r.uniform(-0.04, 0.04, size=2),
            roi_centre=lambda r: np.array([centre[0] + r.uniform(-0.05, 0.05),
                                           centre[1] + r.uniform(-0.05, 0.05), 0.02]),
            meta={"kind": "supply", "tray": with_tray},
            repaint=lambda r: randomize_appearance(scene, r),
        )
    finally:
        renderer.close()


def _plate_scene(rng: np.random.Generator, frames: int) -> Iterator[Sample]:
    """The baseplate, empty or carrying up to six bricks, seen from a range of viewing poses."""
    from ...render.renderer import SceneRenderer
    from ...scene.compose import SceneConfig, build_scene, reset_scene, settle

    touching = bool(rng.random() < 0.45)
    placed = [] if rng.random() < 0.15 else _random_structure(rng, touching)
    cfg = SceneConfig(held=None, placed=placed, tray=bool(rng.random() < 0.4), cameras=())
    scene = build_scene(cfg)
    jitter_plate(scene, rng)
    reset_scene(scene, rng)
    forced = {}
    if touching and len(placed) >= 2:
        shared = placed[0].colour
        for name, p in zip(scene.placed_names, placed):
            if p.colour == shared:
                forced[name] = shared
    randomize_appearance(scene, rng)
    if forced:
        repaint_bricks(scene, rng, colours=forced)
    settle(scene, 0.2)
    plate_xy = _plate_centre(scene)[:2]

    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        yield from _frames_at(
            scene, renderer, rng, frames,
            look_at=lambda r: plate_xy + r.uniform(-0.04, 0.04, size=2),
            roi_centre=lambda r: np.array([plate_xy[0] + r.uniform(-0.05, 0.05),
                                           plate_xy[1] + r.uniform(-0.05, 0.05), 0.02]),
            meta={"kind": "plate", "n_placed": len(placed), "touching_same_colour": touching},
            repaint=lambda r: _repaint_with(scene, r, forced),
        )
    finally:
        renderer.close()


def _bias_table_bright(scene, rng: np.random.Generator) -> tuple[float, float, float]:
    """Table colour drawn from the pale half of `TABLE_RGB_RANGE` only.

    Plain `paint_table` covers the whole range uniformly; this scene kind exists to add density
    at the end of it that the failure analysis flagged (see the guide's "White bricks on a
    bright table" section), so it draws from the pale half only.
    """
    import mujoco

    from ...scene.compose import TABLE_GEOM

    gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    if gid < 0:
        return (0.0, 0.0, 0.0)
    lo, hi = np.asarray(TABLE_RGB_RANGE[0]), np.asarray(TABLE_RGB_RANGE[1])
    mid = 0.5 * (lo + hi)
    rgb = tuple(float(v) for v in rng.uniform(mid, hi))
    scene.model.geom_rgba[gid, :3] = rgb
    return rgb


def _brighten_lights(scene, rng: np.random.Generator, boost_range=(1.0, 1.45)) -> None:
    """An extra brightness pass on top of `randomize_lights`, part of this scene kind's bias
    towards the pale-table case (docs/guide/06b, "White bricks on a bright table"). The inspected
    failure frames were not themselves unusually bright; this boost widens lighting coverage
    around the targeted case, it does not reproduce a brightness pattern seen in those frames."""
    model = scene.model
    for i in range(model.nlight):
        boost = float(rng.uniform(*boost_range))
        model.light_diffuse[i] = np.clip(model.light_diffuse[i] * boost, 0.0, 1.0)
        model.light_specular[i] = np.clip(model.light_specular[i] * boost, 0.0, 1.0)


def _repaint_bright_light(scene, rng: np.random.Generator, forced: dict[str, str]) -> dict:
    """A fresh appearance draw biased towards this scene kind's target case: a pale table, a
    forced light-coloured brick, and an added lighting-brightness boost."""
    meta = randomize_appearance(scene, rng)
    meta["table_rgb"] = _bias_table_bright(scene, rng)
    if forced:
        meta["brick_colours"] = repaint_bricks(scene, rng, colours=forced)
    _brighten_lights(scene, rng)
    return meta


def _bright_light_plate_scene(rng: np.random.Generator, frames: int) -> Iterator[Sample]:
    """The baseplate with at least one light-coloured brick, on a pale table, with an added
    lighting-brightness boost.

    Targeted coverage for the failure documented in the guide ("White bricks on a bright
    table"): a light-coloured brick loses contrast against a pale table, worst where its
    apparent size is smallest; the lighting boost adds coverage rather than reproducing a
    brightness pattern the guide's inspected failure frames actually showed. Not part of
    `DEFAULT_MIX`; generated on its own into a separate split and mixed into training as extra
    data, so the general-purpose split stays exactly what it was.
    """
    from ...bricks.structures import Placement
    from ...render.renderer import SceneRenderer
    from ...scene.compose import SceneConfig, build_scene, reset_scene, settle

    placed: list = []
    for _ in range(5):
        placed = _random_structure(rng, touching_same_colour=False)
        if placed:
            break
    if not placed:
        placed = [Placement(h=2, w=4, x=2, y=2, z=0, colour="white")]
    cfg = SceneConfig(held=None, placed=placed, tray=bool(rng.random() < 0.4), cameras=())
    scene = build_scene(cfg)
    jitter_plate(scene, rng)
    reset_scene(scene, rng)

    forced = {scene.placed_names[0]: "white" if rng.random() < 0.7 else "yellow"}
    if len(placed) >= 3 and rng.random() < 0.35:
        forced[scene.placed_names[1]] = "white" if rng.random() < 0.7 else "yellow"
    _repaint_bright_light(scene, rng, forced)
    settle(scene, 0.2)
    plate_xy = _plate_centre(scene)[:2]

    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        yield from _frames_at(
            scene, renderer, rng, frames,
            look_at=lambda r: plate_xy + r.uniform(-0.04, 0.04, size=2),
            roi_centre=lambda r: np.array([plate_xy[0] + r.uniform(-0.05, 0.05),
                                           plate_xy[1] + r.uniform(-0.05, 0.05), 0.02]),
            meta={"kind": "bright_plate", "n_placed": len(placed),
                  "forced_light": sorted(forced.values())},
            repaint=lambda r: _repaint_bright_light(scene, r, forced),
        )
    finally:
        renderer.close()


def _held_scene(rng: np.random.Generator, frames: int) -> Iterator[Sample]:
    """A brick held in the jaws, hovering over the plate at a range of heights."""
    import mujoco

    from ...render.renderer import SceneRenderer
    from ...scene.compose import HeldBrick, SceneConfig, build_scene, reset_scene, tool_down_ik

    placed = [] if rng.random() < 0.4 else _random_structure(rng, bool(rng.random() < 0.4))
    cfg = SceneConfig(
        held=HeldBrick(nx=2, ny=int(rng.choice([2, 4])), colour=str(rng.choice(BRICK_COLOURS))),
        placed=placed, tray=bool(rng.random() < 0.3), cameras=(),
    )
    scene = build_scene(cfg)
    jitter_plate(scene, rng)
    reset_scene(scene, rng)
    meta_vis = randomize_appearance(scene, rng)
    plate_xy = _plate_centre(scene)[:2]
    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        for _ in range(frames):
            height = float(rng.uniform(0.004, 0.055))
            tcp = np.array([
                plate_xy[0] + float(rng.uniform(-0.03, 0.03)),
                plate_xy[1] + float(rng.uniform(-0.03, 0.03)),
                scene.plate_top + height + float(scene.grasp.pos_in_tcp[2]),
            ])
            try:
                q = tool_down_ik(scene.kin, tcp, float(rng.uniform(-math.pi, math.pi)),
                                 q0=scene.home_q)
            except RuntimeError:
                continue
            reset_scene(scene, rng, arm_q=q)
            meta_vis = randomize_appearance(scene, rng)
            mujoco.mj_forward(scene.model, scene.data)
            diagonal, post = sample_camera_model(rng)
            roi = roi_around([plate_xy[0], plate_xy[1], scene.plate_top + 0.5 * height],
                             (0.055, 0.055, 0.04))
            pose = window_for_roi(scene, roi, diagonal, rng, name="held")
            if pose is None:
                continue
            yield render_sample(scene, pose, renderer, post, rng,
                                {"kind": "held", "hover_mm": round(height * 1e3, 1), **meta_vis})
    finally:
        renderer.close()


def _approach_scene(rng: np.random.Generator, frames: int) -> Iterator[Sample]:
    """The open jaws coming down onto a loose brick: the pick, seen from the wrist."""
    import mujoco

    from ...render.renderer import SceneRenderer
    from ...scene.compose import (
        SceneConfig, build_scene, grasp_pose, jaw_angle_for_clearance, reset_scene, settle,
        tool_down_ik,
    )

    cfg = SceneConfig(held=None, placed=[], tray=True, tray_bricks=int(rng.integers(2, 4)),
                      cameras=())
    scene = build_scene(cfg)
    if rng.random() < 0.5:
        _hide_tray(scene)
    reset_scene(scene, rng)
    meta_vis = randomize_appearance(scene, rng)
    centre = np.asarray(cfg.tray_xy, dtype=float)
    _scatter_loose_bricks(scene, rng, centre)
    settle(scene, 0.25)

    names = [n for n in scene.bricks if n.startswith("tray_brick_")]
    grasp = grasp_pose(cfg.tray_brick[0], cfg.tray_brick[1], scene.cfg.spec)
    grip_open = jaw_angle_for_clearance(grasp.contact_point, 0.004)
    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        for _ in range(frames):
            target = str(rng.choice(names))
            pos, quat = scene.body_pose(target)
            yaw = 2.0 * math.atan2(float(quat[3]), float(quat[0]))
            above = float(rng.uniform(0.004, 0.045))
            tcp = np.array([pos[0], pos[1], pos[2] + above + float(grasp.pos_in_tcp[2])])
            tcp[:2] += rng.uniform(-0.008, 0.008, size=2)
            try:
                q = tool_down_ik(scene.kin, tcp, yaw + float(rng.uniform(-0.15, 0.15)),
                                 q0=scene.home_q)
            except RuntimeError:
                continue
            meta_vis = randomize_appearance(scene, rng)
            _set_arm(scene, q, grip=grip_open)
            mujoco.mj_forward(scene.model, scene.data)
            diagonal, post = sample_camera_model(rng)
            pose = window_for_roi(scene, roi_around(pos, (0.045, 0.045, 0.03)), diagonal, rng,
                                  name="approach")
            if pose is None:
                continue
            yield render_sample(scene, pose, renderer, post, rng,
                                {"kind": "approach", "above_mm": round(above * 1e3, 1), **meta_vis})
    finally:
        renderer.close()


def _episode_scene(rng: np.random.Generator, frames: int) -> Iterator[Sample]:
    """Frames taken along a real pick-and-place episode, every few control ticks.

    This is where the mid-motion poses come from: the arm tilted, the brick half in the jaws, the
    plate sliding through the frame. The controller will see exactly these.
    """
    from ...control.pick_place import PickPlace, make_scene
    from ...render.renderer import SceneRenderer

    scene = make_scene(tray_bricks=int(rng.integers(2, 4)), cameras=())
    jitter_plate(scene, rng, max_xy=0.01, max_yaw_deg=5.0)
    meta_vis = randomize_appearance(scene, rng)
    target = (float(rng.integers(1, 7)) + 0.5, float(rng.integers(1, 7)) + 0.5, 0)
    runner = PickPlace(scene, brick=f"tray_brick_{int(rng.integers(0, 2))}", target=target,
                       rng=np.random.default_rng(int(rng.integers(1 << 30))))
    renderer = SceneRenderer(scene.model, width=RENDER_SIZE[0], height=RENDER_SIZE[1])
    plate_xy = _plate_centre(scene)[:2]
    try:
        renderer.instance_lut(perception_instance_lut(scene))
        taken, every = 0, int(rng.integers(9, 18))
        for step in runner.run():
            if taken >= frames:
                break
            if step.tick % every:
                continue
            meta_vis = randomize_appearance(scene, rng)
            brick_pos = scene.body_pose(runner.brick)[0]
            focus = brick_pos if step.phase in ("to_tray", "open", "descend", "close", "lift") \
                else np.array([plate_xy[0], plate_xy[1], scene.plate_top])
            diagonal, post = sample_camera_model(rng)
            pose = window_for_roi(scene, roi_around(focus, (0.06, 0.06, 0.04)), diagonal, rng,
                                  name="episode")
            if pose is None:
                continue
            yield render_sample(scene, pose, renderer, post, rng,
                                {"kind": "episode", "phase": step.phase, **meta_vis})
            taken += 1
    finally:
        renderer.close()


def _repaint_with(scene, rng: np.random.Generator, forced: dict[str, str]) -> dict:
    """A fresh appearance draw that keeps a chosen same-colour arrangement intact."""
    meta = randomize_appearance(scene, rng)
    if forced:
        meta["brick_colours"] = repaint_bricks(scene, rng, colours=forced)
    return meta


def _frames_at(scene, renderer, rng, frames, *, look_at, roi_centre, meta, repaint=None) -> Iterator[Sample]:
    """Shared loop for the static scenes: move the arm to a viewing pose, frame a region, render.

    Appearance is redrawn for every frame, not once per scene: building and settling a scene is
    the expensive part, so a scene is reused for a dozen frames, and if the lighting and the
    colours were fixed per scene the set would carry only as many appearance draws as it has
    scenes.
    """
    for _ in range(frames):
        q = _viewing_pose(scene, look_at(rng), rng)
        if q is None:
            continue
        if repaint is not None:
            meta = {**meta, **repaint(rng)}
        _set_arm(scene, q)
        diagonal, post = sample_camera_model(rng)
        # From a tight 6 cm box (one brick and its neighbours) out to a 20 cm one (the whole
        # plate with room to spare), so the model sees both the dense and the wide framing the
        # three named views in `perception.views` represent.
        side = float(rng.uniform(0.03, 0.10))
        half = np.array([side, side * float(rng.uniform(0.8, 1.25)), 0.03])
        pose = window_for_roi(scene, roi_around(roi_centre(rng), half), diagonal, rng,
                              name=str(meta.get("kind", "train")))
        if pose is None:
            continue
        yield render_sample(scene, pose, renderer, post, rng, dict(meta))


SCENE_KINDS = {
    "supply": _supply_scene,
    "plate": _plate_scene,
    "held": _held_scene,
    "approach": _approach_scene,
    "episode": _episode_scene,
    "bright_plate": _bright_light_plate_scene,
}

DEFAULT_MIX: dict[str, float] = {
    "plate": 0.30, "held": 0.20, "episode": 0.20, "supply": 0.18, "approach": 0.12,
}
"""How many scenes of each kind. The plate is the largest share because it is what the controller
looks at most; episodes are expensive (a whole run of physics) so they are capped."""


# ----------------------------------------------------------------------------- shard storage ---


def shard_paths(root: Path | str, split: str) -> list[Path]:
    return sorted(Path(root).joinpath(split).glob("shard_*.npz"))


def write_shard(path: Path, samples: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=np.stack([s.rgb for s in samples]),
        instance=np.stack([s.instance for s in samples]),
        depth=np.stack([s.depth for s in samples]),
    )
    path.with_suffix(".json").write_text(
        json.dumps([s.meta for s in samples], default=float, indent=1), encoding="utf-8"
    )


def read_shard(path: Path | str) -> list[Sample]:
    path = Path(path)
    with np.load(path) as z:
        rgb, instance, depth = z["rgb"], z["instance"], z["depth"]
    meta_path = path.with_suffix(".json")
    metas = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else [{}] * len(rgb)
    return [Sample(rgb=rgb[i], instance=instance[i], depth=depth[i], meta=metas[i])
            for i in range(len(rgb))]


def load_samples(root: Path | str = DEFAULT_CACHE, split: str = "train", limit: int | None = None) -> list[Sample]:
    out: list[Sample] = []
    for shard in shard_paths(root, split):
        out.extend(read_shard(shard))
        if limit is not None and len(out) >= limit:
            return out[:limit]
    return out


def dataset_size(root: Path | str = DEFAULT_CACHE, split: str = "train") -> int:
    total = 0
    for shard in shard_paths(root, split):
        with np.load(shard) as z:
            total += int(z["instance"].shape[0])
    return total


# -------------------------------------------------------------------------------- generation ---

SEED_LISTS: dict[str, tuple[int, int]] = {
    "train": (1000, 1999),
    "train_b": (3000, 3999),
    "val": (7000, 7999),
    # Targeted "bright_plate" frames (white/light bricks on a pale table), disjoint from every
    # other split, mixed into training as extra data rather than merged into `train` itself.
    "bright_white_v1": (5000, 5999),
}
"""Disjoint seed ranges per split: a scene layout, colour draw or camera model from one split can
never appear in the other. A scene's seed is `base + index`, so a given dataset is reproducible
from its split name and size alone. `train_b` is a second helping of training scenes, rendered by
a separate process and then merged into `train` by `merge_split` -- rendering is single-threaded
and this machine has cores to spare."""


def merge_split(root: Path | str, source: str, target: str = "train") -> int:
    """Move `source`'s shards into `target`, renumbering them. Returns the shards moved."""
    root = Path(root)
    existing = shard_paths(root, target)
    next_index = 1 + max((int(p.stem.split("_")[1]) for p in existing), default=-1)
    moved = 0
    for shard in shard_paths(root, source):
        new = root / target / f"shard_{next_index:04d}.npz"
        shard.replace(new)
        meta = shard.with_suffix(".json")
        if meta.exists():
            meta.replace(new.with_suffix(".json"))
        next_index += 1
        moved += 1
    return moved


def generate(
    out_dir: Path | str = DEFAULT_CACHE,
    split: str = "train",
    n_scenes: int = 160,
    frames_per_scene: int = 14,
    mix: dict[str, float] | None = None,
    progress: bool = True,
) -> dict:
    """Render `n_scenes` randomized scenes into compressed shards and return a summary.

    Deterministic: every scene is driven by one generator seeded from this split's own seed range,
    so re-running with the same arguments writes the same frames.
    """
    out_dir = Path(out_dir)
    mix = dict(mix or DEFAULT_MIX)
    base = SEED_LISTS.get(split, (0, 999))[0]
    kinds = list(mix)
    weights = np.asarray([mix[k] for k in kinds], dtype=float)
    weights /= weights.sum()
    plan = [kinds[int(i)] for i in np.repeat(np.arange(len(kinds)), np.maximum(
        1, np.round(weights * n_scenes).astype(int)))][:n_scenes]

    (out_dir / split).mkdir(parents=True, exist_ok=True)
    for stale in (out_dir / split).glob("shard_*"):
        stale.unlink()

    buffer: list[Sample] = []
    counts: dict[str, int] = {}
    shard_index, total = 0, 0
    t0 = time.perf_counter()
    for i, kind in enumerate(plan):
        rng = np.random.default_rng(base + i)
        try:
            for sample in SCENE_KINDS[kind](rng, frames_per_scene):
                buffer.append(sample)
                counts[kind] = counts.get(kind, 0) + 1
                total += 1
                if len(buffer) >= SHARD_SIZE:
                    write_shard(out_dir / split / f"shard_{shard_index:04d}.npz", buffer)
                    shard_index += 1
                    buffer = []
        except Exception as exc:  # one bad scene must not lose a whole run
            print(f"  scene {i} ({kind}) failed: {type(exc).__name__}: {exc}")
        if progress and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(plan)} scenes, {total} frames, "
                  f"{time.perf_counter() - t0:.0f}s")
    if buffer:
        write_shard(out_dir / split / f"shard_{shard_index:04d}.npz", buffer)
        shard_index += 1

    summary = {
        "split": split, "scenes": len(plan), "frames": total, "shards": shard_index,
        "by_kind": counts, "seconds": round(time.perf_counter() - t0, 1),
        "megabytes": round(sum(p.stat().st_size for p in shard_paths(out_dir, split)) / 1e6, 1),
    }
    (out_dir / split / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ------------------------------------------------------------------------------ contact sheet ---


def contact_sheet(samples: Sequence[Sample], out_path: Path | str, columns: int = 8,
                  title: str = "Training frames - instance labels") -> Path:
    """A grid of frames with their class labels drawn over them -- the picture to look at before
    training anything. Brick instances get their own outline colour so two touching bricks of one
    colour can be told apart by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(math.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.35 * rows), dpi=130)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, sample in enumerate(samples):
        ax = axes[i // columns, i % columns]
        ax.imshow(overlay_labels(sample))
        ax.set_title(f"{sample.meta.get('kind', '?')}", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{title}   (magenta: arm, white: baseplate, other: one per brick)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


_OUTLINE_COLOURS = np.array([
    [255, 0, 255],   # arm
    [255, 255, 255],  # baseplate
    [255, 64, 64], [64, 255, 64], [64, 160, 255], [255, 220, 0],
    [255, 128, 0], [0, 255, 220], [200, 120, 255], [160, 255, 120],
], dtype=np.uint8)


def overlay_labels(sample: Sample) -> np.ndarray:
    """The frame with every labelled instance outlined, for a human to check."""
    import cv2

    img = sample.rgb.copy()
    for k, inst_id in enumerate(np.unique(sample.instance)):
        if inst_id == 0:
            continue
        mask = (sample.instance == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = _OUTLINE_COLOURS[min(int(inst_id) - 1, len(_OUTLINE_COLOURS) - 1)]
        cv2.drawContours(img, contours, -1, tuple(int(c) for c in colour), 1)
    return img
