"""Domain randomization for a compiled :class:`~brick_builder.scene.compose.Scene`.

One function per group, exactly the groups design report 01 sections 4 and 6 list:

============  ===========================================================================
``dynamics``  servo gain, torque limit, joint damping, friction loss, armature, backlash,
              supply voltage, link mass
``command``   the command path: `ServoConfig` and `BusConfig` fields (delay, dead band,
              rate limit, read skew and noise, packet loss, period jitter)
``snap``      the snap model's parameters, including the force-curve shape of WP-SNAP2
``contact``   brick, pad and table friction, and the stud diameter
``plate``     the build plate's pose (+-3 mm, +-2 degrees)
``in_hand``   the in-hand offset of the held brick and the soft weld's compliance
``visual``    lights, camera extrinsics and field of view, brick colour, and the image
              post-process (which is `render.domain_rand`'s job)
============  ===========================================================================

``sample(rng, level)`` returns one flat dict of plain numbers and strings, keyed
``"<group>.<field>"``.  That dict is what gets appended to the critic observation, logged in
``EpisodeResult.params``, and handed back to :func:`apply`, so a run can always be reproduced from
what was recorded rather than from a seed plus a version of this file.

``level`` is a number in [0, 1] and means "how much of the range to use": 0 returns the nominal
value of every parameter exactly, 1 draws from the full published range, and anything between
interpolates from the nominal towards the drawn value.  That gives the curriculum a single knob
and makes "level 0 reproduces the un-randomized scene" true by construction rather than by
convention.  (The task curriculum's own ``level`` 0..5, from spec section 4, is a different
number and lives in the environments.)

Apply order matters: :func:`apply` writes the model, then
:func:`~brick_builder.scene.compose.reset_scene` lays the bricks out against the plate pose that was
just chosen and re-binds the snap model.  Randomize, then reset.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable

import mujoco
import numpy as np

from brick_builder.bricks.generator import COLLISION_GROUP, VISUAL_GROUP
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.spec import PALETTE
from brick_builder.render.domain_rand import (
    ColourJitterConfig,
    PostProcessConfig,
    jitter_rgb_hsv,
)
from brick_builder.robot.bus_emulator import BusConfig
from brick_builder.robot.patch_so101 import (
    ARM_JOINTS,
    FIXED_PAD_GEOM,
    MOVING_PAD_GEOM,
    NOMINAL_VIN,
    ArmConfig,
    BAM_ARMATURE,
    BAM_FRICTION_BASE,
    servo_damping,
    servo_kp,
)
from brick_builder.robot.servo_model import ERROR_DEADBAND_TICKS_RANGE, ServoConfig
from brick_builder.scene.compose import TABLE_GEOM, Scene

# ------------------------------------------------------------------------------------ ranges ---

Range = tuple[float, float]

RANGES: dict[str, Range] = {
    # -- dynamics (design report 01 section 6) ---------------------------------------------
    "dynamics.kp": (7.0, 20.0),                 # N*m/rad
    "dynamics.forcerange": (1.5, 3.4),          # N*m; the datasheet/BAM disagreement of ruling 7
    "dynamics.damping": (0.45, 0.8),            # N*m*s/rad
    "dynamics.frictionloss": (0.03, 0.10),      # N*m
    "dynamics.armature": (0.018, 0.032),        # kg*m^2
    "dynamics.backlash_deg": (0.2, 1.0),        # half-range of the free-play joints
    "dynamics.vin": (6.8, 7.6),                 # V
    "dynamics.link_mass_scale": (0.9, 1.1),
    # -- command path (design report 01 section 6, WP-ROBOT's ServoConfig/BusConfig) --------
    "command.deadband_ticks": (0.0, 10.0),      # 10 ticks of 4096 is 0.88 degrees
    "command.error_deadband_ticks": ERROR_DEADBAND_TICKS_RANGE,  # firmware dead band, regs 26/27
    "command.rate_limit": (4.0, 6.0),           # rad/s
    "command.action_delay_s": (0.0, 0.04),      # 0-2 control steps at 50 Hz
    "command.action_delay_jitter_s": (0.0, 0.004),
    "command.read_skew_s": (0.0, 0.0025),       # BusEmulator measures 2.05 ms over six servos
    "command.read_noise_ticks": (0.0, 2.0),
    "command.obs_delay_ticks": (0.0, 1.0),      # consumed by the env's observation wrapper
    "command.period_jitter_s": (0.0, 0.002),
    "command.packet_loss": (0.0, 0.01),
    # -- snap model (design report 01 section 4, plus WP-SNAP2's curve shapes) --------------
    "snap.f_clutch": (0.3, 2.5),                # log-uniform: see `_log_uniform`
    "snap.beta": (0.0, 0.4),
    "snap.exponent": (0.5, 2.0),
    "snap.capture_lateral": (0.0001, 0.0004),
    "snap.capture_tilt_deg": (3.0, 10.0),
    "snap.engage_depth": (0.0003, 0.0010),
    "snap.plateau_gain": (0.25, 1.0),
    "snap.detent_gain": (0.2, 1.6),
    "snap.detent_depth": (0.0006, 0.0016),
    "snap.detent_width": (0.0004, 0.0010),
    # -- contact ----------------------------------------------------------------------------
    "contact.brick_friction": (0.2, 0.5),
    "contact.pad_friction": (0.4, 1.2),
    "contact.table_friction": (0.4, 1.0),
    "contact.stud_radius_delta": (-0.000025, 0.000025),   # +-0.05 mm on the diameter
    # -- plate pose -------------------------------------------------------------------------
    "plate.dx": (-0.003, 0.003),
    "plate.dy": (-0.003, 0.003),
    "plate.dyaw": (-math.radians(2.0), math.radians(2.0)),
    # -- in-hand ----------------------------------------------------------------------------
    "in_hand.dx": (-0.002, 0.002),
    "in_hand.dy": (-0.002, 0.002),
    "in_hand.dz": (-0.001, 0.001),
    "in_hand.dyaw": (-math.radians(3.0), math.radians(3.0)),
    "in_hand.weld_timeconst": (0.004, 0.016),
    "in_hand.weld_torquescale": (0.3, 3.0),
    # -- visuals ----------------------------------------------------------------------------
    "visual.light_dx": (-0.15, 0.15),
    "visual.light_dy": (-0.15, 0.15),
    "visual.light_dz": (-0.10, 0.10),
    "visual.light_diffuse": (0.45, 0.95),
    "visual.cam_dx": (-0.005, 0.005),
    "visual.cam_dy": (-0.005, 0.005),
    "visual.cam_dz": (-0.005, 0.005),
    "visual.cam_rot_deg": (-3.0, 3.0),
    "visual.cam_fovy_deg": (-3.0, 3.0),
    "visual.hue_deg": (-10.0, 10.0),
    "visual.sat": (-0.15, 0.15),
    "visual.val": (-0.15, 0.15),
    "visual.post_k1": (-0.15, 0.15),
    "visual.post_k2": (-0.05, 0.05),
    "visual.post_exposure": (0.8, 1.25),
    "visual.post_gamma": (0.85, 1.2),
    "visual.post_shot_noise": (0.0, 0.02),
    "visual.post_read_noise": (0.0, 0.01),
    "visual.post_blur_sigma": (0.0, 1.0),
}

NOMINAL: dict[str, float] = {
    "dynamics.kp": servo_kp(16, NOMINAL_VIN),
    "dynamics.forcerange": 1.9,
    "dynamics.damping": servo_damping(),
    "dynamics.frictionloss": BAM_FRICTION_BASE,
    "dynamics.armature": BAM_ARMATURE,
    "dynamics.backlash_deg": ArmConfig.backlash_deg,
    "dynamics.vin": NOMINAL_VIN,
    "dynamics.link_mass_scale": 1.0,
    "command.deadband_ticks": 0.0,
    "command.error_deadband_ticks": ServoConfig().error_deadband_ticks,
    "command.rate_limit": ServoConfig().rate_limit,
    "command.action_delay_s": ServoConfig().action_delay_s,
    "command.action_delay_jitter_s": ServoConfig().action_delay_jitter_s,
    "command.read_skew_s": ServoConfig().read_skew_s,
    "command.read_noise_ticks": 0.0,
    "command.obs_delay_ticks": 0.0,
    "command.period_jitter_s": 0.0,
    "command.packet_loss": BusConfig().packet_loss,
    "snap.f_clutch": SnapParams().f_clutch,
    "snap.beta": SnapParams().beta,
    "snap.exponent": SnapParams().exponent,
    "snap.capture_lateral": SnapParams().capture_lateral,
    "snap.capture_tilt_deg": SnapParams().capture_tilt_deg,
    "snap.engage_depth": SnapParams().engage_depth,
    "snap.plateau_gain": getattr(SnapParams(), "plateau_gain", 0.35),
    "snap.detent_gain": getattr(SnapParams(), "detent_gain", 1.0),
    "snap.detent_depth": getattr(SnapParams(), "detent_depth", 0.001),
    "snap.detent_width": getattr(SnapParams(), "detent_width", 0.0006),
    "contact.brick_friction": 0.35,    # `SceneConfig.brick_friction`, from design report 01 s4
    "contact.pad_friction": 1.0,
    "contact.table_friction": 0.6,
    "contact.stud_radius_delta": 0.0,
    "plate.dx": 0.0,
    "plate.dy": 0.0,
    "plate.dyaw": 0.0,
    "in_hand.dx": 0.0,
    "in_hand.dy": 0.0,
    "in_hand.dz": 0.0,
    "in_hand.dyaw": 0.0,
    "in_hand.weld_timeconst": 0.008,
    "in_hand.weld_torquescale": 1.0,
    "visual.light_dx": 0.0,
    "visual.light_dy": 0.0,
    "visual.light_dz": 0.0,
    "visual.light_diffuse": 0.7,
    "visual.cam_dx": 0.0,
    "visual.cam_dy": 0.0,
    "visual.cam_dz": 0.0,
    "visual.cam_rot_deg": 0.0,
    "visual.cam_fovy_deg": 0.0,
    "visual.hue_deg": 0.0,
    "visual.sat": 0.0,
    "visual.val": 0.0,
    "visual.post_k1": 0.0,
    "visual.post_k2": 0.0,
    "visual.post_exposure": 1.0,
    "visual.post_gamma": 1.0,
    "visual.post_shot_noise": 0.0,
    "visual.post_read_noise": 0.0,
    "visual.post_blur_sigma": 0.0,
}

_LOG_UNIFORM = ("snap.f_clutch",)
"""Parameters drawn log-uniformly: a clutch force is a scale, and 0.3-2.5 N spans a factor of 8."""

SNAP_CURVES = ("ramp", "plateau", "detent")
"""The three force-curve shapes `bricks.snap` offers; sampled until a real measurement picks one."""


def _draw(key: str, rng: np.random.Generator, level: float) -> float:
    lo, hi = RANGES[key]
    nominal = NOMINAL[key]
    if key in _LOG_UNIFORM:
        value = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        # Interpolate in log space too, or level=0.5 would not be the geometric middle.
        value = float(np.exp(math.log(nominal) + level * (math.log(value) - math.log(nominal))))
    else:
        value = float(rng.uniform(lo, hi))
        value = nominal + level * (value - nominal)
    return float(np.clip(value, min(lo, nominal), max(hi, nominal)))


def _group(prefix: str, rng: np.random.Generator, level: float) -> dict:
    return {k: _draw(k, rng, level) for k in RANGES if k.startswith(prefix + ".")}


# ------------------------------------------------------------------------------ the samplers ---


def sample_dynamics(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Servo gain, torque limit, joint damping/friction/armature, backlash, voltage, link mass.

    ``vin`` is sampled but is *not* applied a second time to ``kp`` and ``forcerange``.  Design
    report 01 lists the supply voltage as scaling both, and it does - but the published ranges for
    gain (7-20) and torque limit (1.5-3.4) already span the whole 6.8-7.6 V band and the
    datasheet-vs-BAM disagreement on top of it, so multiplying again would push the model outside
    the ranges this module promises to stay inside.  The voltage still reaches the servo
    emulation, where it changes the current and load readings, through :func:`servo_config`.
    """
    return _group("dynamics", rng, level)


def sample_command(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Dead band, slew limit, command delay and jitter, read skew and noise, bus packet loss."""
    return _group("command", rng, level)


def sample_snap(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Snap-fit parameters, including which force-curve shape this episode uses.

    The curve is a categorical draw over ``ramp``/``plateau``/``detent`` because nobody has
    measured a real DUPLO brick's force-against-depth shape.  At ``level`` 0 it
    stays on whatever the scene was built with.
    """
    out = _group("snap", rng, level)
    if level > 0.0 and any(f.name == "curve" for f in dataclasses.fields(SnapParams)):
        out["snap.curve"] = str(rng.choice(SNAP_CURVES))
    return out


def sample_contact(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Sliding friction of the bricks, the fingertip pads and the table, and the stud diameter."""
    return _group("contact", rng, level)


def sample_plate(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Build-plate pose error: +-3 mm and +-2 degrees, as the design report specifies."""
    return _group("plate", rng, level)


def sample_in_hand(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Where the brick really is in the hand, and how compliant the soft weld holding it is."""
    return _group("in_hand", rng, level)


def sample_visual(rng: np.random.Generator, level: float = 1.0) -> dict:
    """Lights, camera extrinsics and fov, brick colour jitter, and the image post-process."""
    return _group("visual", rng, level)


GROUPS: dict[str, Callable[[np.random.Generator, float], dict]] = {
    "dynamics": sample_dynamics,
    "command": sample_command,
    "snap": sample_snap,
    "contact": sample_contact,
    "plate": sample_plate,
    "in_hand": sample_in_hand,
    "visual": sample_visual,
}


def sample(
    rng: np.random.Generator, level: float = 1.0, groups: tuple[str, ...] | None = None
) -> dict:
    """Draw every group and return one flat dict, keyed ``"<group>.<field>"``.

    ``level`` 0 gives the nominal value of everything, so ``apply(scene, sample(rng, 0))`` is a
    no-op in effect and an easy control condition.
    """
    names = tuple(GROUPS) if groups is None else tuple(groups)
    unknown = [g for g in names if g not in GROUPS]
    if unknown:
        raise ValueError(f"unknown randomization group(s) {unknown}; have {tuple(GROUPS)}")
    out: dict = {}
    for name in names:
        out.update(GROUPS[name](rng, level))
    return out


# ------------------------------------------------------------------ turning a sample into one ---
#                                                                     of the project's configs


def servo_config(sample: dict, base: ServoConfig = ServoConfig()) -> ServoConfig:
    """Build the `ServoConfig` a sampled command path describes.

    ``command.error_deadband_ticks`` (the firmware position dead band, registers 26/27) is
    sampled and applied here like every other command-path field, but it is deliberately left out
    of ``envs.base_env.DR_PARAM_KEYS`` -- the critic's ``dr_params`` observation entry is a fixed
    17-wide vector, and growing it is an environment-owned decision (`brick_builder/envs/**`) this
    package does not make. The episode still randomizes over it and `Scene.randomization` still
    records the drawn value; only the critic's dedicated summary vector does not grow to show it.
    """
    return dataclasses.replace(
        base,
        deadband_ticks=int(round(sample.get("command.deadband_ticks", base.deadband_ticks))),
        error_deadband_ticks=float(
            sample.get("command.error_deadband_ticks", base.error_deadband_ticks)
        ),
        rate_limit=float(sample.get("command.rate_limit", base.rate_limit)),
        action_delay_s=float(sample.get("command.action_delay_s", base.action_delay_s)),
        action_delay_jitter_s=float(
            sample.get("command.action_delay_jitter_s", base.action_delay_jitter_s)
        ),
        read_skew_s=float(sample.get("command.read_skew_s", base.read_skew_s)),
        read_noise_ticks=float(sample.get("command.read_noise_ticks", base.read_noise_ticks)),
        vin=float(sample.get("dynamics.vin", base.vin)),
    )


def bus_config(sample: dict, base: BusConfig = BusConfig()) -> BusConfig:
    """Build the `BusConfig` a sampled command path describes."""
    return dataclasses.replace(
        base,
        packet_loss=float(sample.get("command.packet_loss", base.packet_loss)),
        period_jitter_s=float(sample.get("command.period_jitter_s", base.period_jitter_s)),
    )


def snap_params(sample: dict, base: SnapParams = SnapParams()) -> SnapParams:
    """Build the `SnapParams` a sampled snap group describes."""
    fields = {f.name for f in dataclasses.fields(SnapParams)}
    changes = {}
    for key, value in sample.items():
        if not key.startswith("snap."):
            continue
        name = key.split(".", 1)[1]
        if name in fields:
            changes[name] = value
    return dataclasses.replace(base, **changes)


def postprocess_config(sample: dict, base: PostProcessConfig = PostProcessConfig()) -> PostProcessConfig:
    """Build the image post-process `render.domain_rand` applies to a rendered frame."""
    return dataclasses.replace(
        base,
        k1=float(sample.get("visual.post_k1", base.k1)),
        k2=float(sample.get("visual.post_k2", base.k2)),
        exposure=float(sample.get("visual.post_exposure", base.exposure)),
        gamma=float(sample.get("visual.post_gamma", base.gamma)),
        shot_noise=float(sample.get("visual.post_shot_noise", base.shot_noise)),
        read_noise=float(sample.get("visual.post_read_noise", base.read_noise)),
        blur_sigma=float(sample.get("visual.post_blur_sigma", base.blur_sigma)),
    )


# ------------------------------------------------------------------------------------- apply ---


def _arm_joint_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(joint ids, dof addresses, backlash joint ids) for the six actuated joints."""
    jids, dofs, backlash = [], [], []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        jids.append(jid)
        dofs.append(int(model.jnt_dofadr[jid]))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_backlash")
        if bid >= 0:
            backlash.append(bid)
    return (
        np.asarray(jids, dtype=np.int32),
        np.asarray(dofs, dtype=np.int32),
        np.asarray(backlash, dtype=np.int32),
    )


def _geoms_of_body(model: mujoco.MjModel, body: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return np.zeros(0, dtype=np.int32)
    start = int(model.body_geomadr[bid])
    return np.arange(start, start + int(model.body_geomnum[bid]), dtype=np.int32)


def brick_geoms(scene: Scene, visual: bool = False) -> np.ndarray:
    """Ids of every collision (or visual) geom belonging to a brick or the build plate."""
    tag = "_vis_" if visual else "_col_"
    want = VISUAL_GROUP if visual else COLLISION_GROUP
    names = list(scene.bricks) + [scene.plate.name]
    out = []
    for name in names:
        for gid in _geoms_of_body(scene.model, name):
            gname = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
            if tag in gname and int(scene.model.geom_group[gid]) == want:
                out.append(int(gid))
    return np.asarray(out, dtype=np.int32)


def apply(scene: Scene, sample: dict) -> None:
    """Write a sample into the scene's model, in place.

    Call this *before* :func:`~brick_builder.scene.compose.reset_scene`: the reset lays the bricks out
    against the plate pose chosen here and re-binds the snap model, which is also what puts back
    the collet regularization that `mj_setConst` recomputes when link masses change.

    `scene.randomization` is merged with (not replaced by) `sample` (scene-bricks-1): a key this
    call does not mention keeps whatever value an earlier `apply()` call gave it, because that is
    still the value live in the model -- `apply()` only ever writes the fields a given `sample`
    actually names (see every `if "<group>.<field>" in sample` below), so a wholesale replace
    would make the record silently stop matching the model the moment two calls use different
    `groups`. This is what makes reproducing an episode from `EpisodeResult.params`/
    `scene.randomization` alone still correct after any number of same- or different-groups
    `apply()` calls in a row -- the documented contract this record exists for. Use
    :func:`reset_to_nominal` to actually clear it back to nothing.
    """

    model, data = scene.model, scene.data
    scene.randomization = {**scene.randomization, **sample}
    jids, dofs, backlash = _arm_joint_ids(model)
    touched_mass = False

    # -- dynamics --------------------------------------------------------------------------
    if "dynamics.kp" in sample:
        kp = float(sample["dynamics.kp"])
        model.actuator_gainprm[:len(ARM_JOINTS), 0] = kp
        model.actuator_biasprm[:len(ARM_JOINTS), 1] = -kp
    if "dynamics.forcerange" in sample:
        limit = float(sample["dynamics.forcerange"])
        model.actuator_forcerange[:len(ARM_JOINTS), 0] = -limit
        model.actuator_forcerange[:len(ARM_JOINTS), 1] = +limit
    if "dynamics.damping" in sample:
        model.dof_damping[dofs] = float(sample["dynamics.damping"])
    if "dynamics.frictionloss" in sample:
        model.dof_frictionloss[dofs] = float(sample["dynamics.frictionloss"])
    if "dynamics.armature" in sample:
        model.dof_armature[dofs] = float(sample["dynamics.armature"])
    if "dynamics.backlash_deg" in sample and backlash.size:
        play = math.radians(float(sample["dynamics.backlash_deg"]))
        model.jnt_range[backlash, 0] = -play
        model.jnt_range[backlash, 1] = +play
    if "dynamics.link_mass_scale" in sample:
        scale = float(sample["dynamics.link_mass_scale"])
        link_bodies = np.unique(model.jnt_bodyid[jids])
        # `body_mass` is what the sample names, so it is what gets scaled; the inertia scales with
        # it because the shape does not change.
        model.body_mass[link_bodies] = scene.nominal["body_mass"][link_bodies] * scale
        model.body_inertia[link_bodies] = scene.nominal["body_inertia"][link_bodies] * scale
        touched_mass = True

    # -- snap ------------------------------------------------------------------------------
    if any(k.startswith("snap.") for k in sample):
        scene.snap.params = snap_params(sample, scene.snap.params)

    # -- contact ---------------------------------------------------------------------------
    if "contact.brick_friction" in sample:
        model.geom_friction[brick_geoms(scene), 0] = float(sample["contact.brick_friction"])
    if "contact.pad_friction" in sample:
        for pad in (FIXED_PAD_GEOM, MOVING_PAD_GEOM):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad)
            if gid >= 0:
                model.geom_friction[gid, 0] = float(sample["contact.pad_friction"])
    if "contact.table_friction" in sample:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
        if gid >= 0:
            model.geom_friction[gid, 0] = float(sample["contact.table_friction"])
    if "contact.stud_radius_delta" in sample:
        delta = float(sample["contact.stud_radius_delta"])
        studs = stud_geoms(scene)
        model.geom_size[studs, 0] = scene.nominal["geom_size"][studs, 0] + delta

    # -- plate pose ------------------------------------------------------------------------
    if any(k.startswith("plate.") for k in sample):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.plate.name)
        yaw = scene.cfg.plate_yaw + float(sample.get("plate.dyaw", 0.0))
        model.body_pos[bid, 0] = scene.cfg.plate_xy[0] + float(sample.get("plate.dx", 0.0))
        model.body_pos[bid, 1] = scene.cfg.plate_xy[1] + float(sample.get("plate.dy", 0.0))
        model.body_quat[bid] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]

    # -- in-hand ---------------------------------------------------------------------------
    if any(k.startswith("in_hand.") for k in sample) and scene.held_name is not None:
        scene.in_hand_override = np.array(
            [
                float(sample.get("in_hand.dx", 0.0)),
                float(sample.get("in_hand.dy", 0.0)),
                float(sample.get("in_hand.dz", 0.0)),
                float(sample.get("in_hand.dyaw", 0.0)),
            ]
        )
        if scene.weld_eq >= 0:
            model.eq_solref[scene.weld_eq, 0] = float(
                sample.get("in_hand.weld_timeconst", scene.cfg.weld_solref[0])
            )
            scene.weld_torquescale = float(
                sample.get("in_hand.weld_torquescale", scene.cfg.weld_torquescale)
            )

    # -- visuals ---------------------------------------------------------------------------
    _apply_visual(scene, sample)

    if touched_mass:
        # Subtree masses and the constraint inverse-weights are precomputed from body_mass.
        mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)


def reset_to_nominal(scene: Scene) -> None:
    """Restore every randomized field to its nominal value and clear the recorded sample
    (scene-bricks-1).

    Undoes any number of prior `apply()` calls -- whole-groups or partial, all at once or
    spread across several resets -- in one step: `apply(scene, dict(NOMINAL))` writes every
    field `NOMINAL` names through the exact same code paths `apply()` always uses (so this
    needs no separate "undo" logic to keep in sync), then `scene.randomization` is set to `{}`
    rather than left holding every key at its nominal value -- `apply`'s own merge behaviour
    would otherwise report "randomized, but at nominal values" instead of "nothing is
    randomized", which is the whole point of calling this rather than just sampling at level 0.
    """
    apply(scene, dict(NOMINAL))
    scene.randomization = {}


def stud_geoms(scene: Scene) -> np.ndarray:
    """Ids of every collision stud geom, on the bricks and on the plate."""

    out = []
    for name in list(scene.bricks) + [scene.plate.name]:
        for gid in _geoms_of_body(scene.model, name):
            gname = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
            if "_col_stud_" in gname:
                out.append(int(gid))
    return np.asarray(out, dtype=np.int32)


def _apply_visual(scene: Scene, sample: dict) -> None:
    """Lights, cameras and brick colour, written straight into the compiled model.

    `render.domain_rand` has the physics of all three, but its scene-side helpers take an
    uncompiled `MjSpec` (they add lights, they do not move existing ones) and a scene here is
    already compiled - re-compiling per episode would cost far more than an episode of physics.
    So the *sampling* and the colour maths come from that module and only the writes are local.
    """

    model = scene.model
    if "visual.light_diffuse" in sample or "visual.light_dx" in sample:
        shift = np.array([
            float(sample.get("visual.light_dx", 0.0)),
            float(sample.get("visual.light_dy", 0.0)),
            float(sample.get("visual.light_dz", 0.0)),
        ])
        model.light_pos[:] = scene.nominal["light_pos"] + shift
        if "visual.light_diffuse" in sample:
            model.light_diffuse[0] = float(sample["visual.light_diffuse"])

    if any(k.startswith("visual.cam_") for k in sample):
        shift = np.array([
            float(sample.get("visual.cam_dx", 0.0)),
            float(sample.get("visual.cam_dy", 0.0)),
            float(sample.get("visual.cam_dz", 0.0)),
        ])
        model.cam_pos[:] = scene.nominal["cam_pos"] + shift
        model.cam_fovy[:] = scene.nominal["cam_fovy"] + float(sample.get("visual.cam_fovy_deg", 0.0))
        angle = math.radians(float(sample.get("visual.cam_rot_deg", 0.0)))
        if angle:
            # One small rotation about the camera's own x axis: the tilt a loose mount gives.
            tilt = np.array([math.cos(0.5 * angle), math.sin(0.5 * angle), 0.0, 0.0])
            for c in range(model.ncam):
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, scene.nominal["cam_quat"][c], tilt)
                model.cam_quat[c] = out

    hue = float(sample.get("visual.hue_deg", 0.0))
    sat = float(sample.get("visual.sat", 0.0))
    val = float(sample.get("visual.val", 0.0))
    # Gate on whether the sample names these keys at all, not on whether their value happens to
    # be nonzero: a level-0 (or `NOMINAL`) sample legitimately draws exactly 0.0 for all three,
    # and must still repaint every brick/plate geom back to its true base colour -- e.g. after a
    # previous, non-zero-jitter `apply()` on the same scene (`reset_to_nominal` depends on this).
    # Gating on the value instead would silently leave a previously-jittered colour in place
    # whenever the new draw's jitter happened to be exactly zero.
    if "visual.hue_deg" in sample or "visual.sat" in sample or "visual.val" in sample:
        cfg = ColourJitterConfig(hue_jitter_deg=abs(hue), sat_jitter=abs(sat), val_jitter=abs(val))
        # One generator seeded from the sample itself, so the colours are a function of the
        # recorded parameters and not of whatever generator state `apply` happens to be handed.
        seed = abs(hash((round(hue, 6), round(sat, 6), round(val, 6)))) % (2**32)
        rng = np.random.default_rng(seed)
        no_jitter = hue == 0.0 and sat == 0.0 and val == 0.0
        handles = [(scene.plate.name, scene.plate)] + list(scene.bricks.items())
        for name, handle in handles:
            base_rgb = np.asarray(PALETTE.get(handle.colour, PALETTE["white"]), dtype=np.float32)
            if no_jitter:
                # Exactly the nominal colour: skip the lossy RGB->HSV->RGB round trip rather than
                # ask it to reproduce base_rgb from scratch, which loses lossless bit-exactness
                # (jitter_rgb_hsv is exercised by the sat/hue/val>0 branch and its own tests
                # already, so nothing about its correctness is untested by skipping it here).
                rgb = tuple(float(c) for c in base_rgb)
            else:
                rgb = jitter_rgb_hsv(tuple(float(c) for c in base_rgb), cfg, rng)
            for gid in _geoms_of_body(model, name):
                # Only the geoms that were built in this brick's own colour (shell and studs);
                # anything the generator coloured differently keeps its colour.
                if not np.allclose(scene.nominal["geom_rgba"][gid, :3], base_rgb, atol=2e-3):
                    continue
                model.geom_rgba[gid, :3] = rgb
