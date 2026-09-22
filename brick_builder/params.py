"""The member-editable parameter file: schema, loader, validation, and conversion.

Four students measure real DUPLO bricks and a real SO-101 arm with a kitchen scale, a ruler and
a caliper, and type what they find into ``configs/params/default.yaml`` (or their own copy of
it). This module is the only thing that reads that file. It does four jobs:

1. **Schema** (:data:`SCHEMA`). Every parameter the file may contain, its unit, its type, a
   sane range, and (for a string parameter) its allowed values. This is what lets a typo be
   caught with a clear message instead of silently doing nothing.
2. **Loading** (:func:`load_params`). Reads a YAML file where every leaf is either a bare value
   (shorthand for a measured value with no further detail: ``{value: v, status: 'measured'}``) or a block carrying its
   own ``value``, ``status`` (``assumed`` | ``derived`` | ``measured``), and optional
   ``measured_by``, ``measured_on`` and ``notes``. A member's file is merged **over**
   ``configs/params/default.yaml`` (:data:`DEFAULT_PARAMS_PATH`): a key the member's file does
   not mention keeps the shipped default, status and all.
3. **Validation**. Unknown keys, out-of-range values, wrong types and bad enum choices are all
   collected and raised together as one :class:`ParamsError`, so a member fixing a file sees
   every problem at once rather than one per run.
4. **Conversion**. ``to_brick_spec``, ``to_snap_params``, ``to_arm_config``, ``to_servo_config``,
   ``to_camera_settings`` and ``to_scene_config`` turn a loaded :class:`ParamFile` into the
   simulator's own config objects. Every one of them is built through
   :func:`filter_dataclass_kwargs`, which only ever passes fields that exist on the target
   dataclass *right now* -- those dataclasses are still under active development elsewhere in
   the codebase, so a field this module wants to set may not exist yet, or may stop existing
   between one call and the next.

Nothing here imports anything that constructs a :class:`~mujoco.MjModel` at import time, and
loading a parameter file never mutates any module-level default (``SnapParams()`` still
constructs the same object it always did) -- everything a converter builds is a fresh instance
handed back to the caller.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .bricks.snap_api import SnapParams
from .bricks.spec import DUPLO, BrickSpec
from .robot.patch_so101 import ArmConfig, fovy_from_diagonal_fov
from .robot.servo_model import ServoConfig

Status = Literal["assumed", "derived", "measured"]
STATUSES: tuple[Status, ...] = ("assumed", "derived", "measured")

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "params" / "default.yaml"
"""``configs/params/default.yaml``, resolved from this file's location rather than the cwd."""


class ParamsError(ValueError):
    """One or more problems with a parameter file, all listed at once.

    ``problems`` is the list of individual, human-readable messages; ``str(err)`` joins them
    with a blank line between so they read well printed straight to a terminal.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        header = f"{len(self.problems)} problem(s) in the parameter file:" if len(self.problems) != 1 else "Problem in the parameter file:"
        super().__init__(header + "\n\n" + "\n\n".join(f"  - {p}" for p in self.problems))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """What one leaf of the parameter file is allowed to hold.

    ``kind`` is ``float``, ``int`` or ``str``. ``lo``/``hi`` bound a numeric field (inclusive);
    ``choices`` restricts a string field. ``nullable`` allows ``value: null`` -- used for the raw
    kitchen-scale readings that start out uncollected. ``default``/``default_status`` are what a
    field gets if :data:`DEFAULT_PARAMS_PATH` itself somehow omits it (defensive: the shipped
    file is expected to set every field explicitly, and a test checks that it does).
    """

    section: str
    unit: str
    kind: type
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] | None = None
    nullable: bool = False
    default: Any = None
    default_status: Status = "assumed"
    doc: str = ""


SCHEMA: dict[str, FieldSpec] = {
    # -- bricks: the snap-fit model --------------------------------------------------------
    "bricks.clutch_force_per_stud_N": FieldSpec(
        "bricks", "N", float, 0.05, 50.0, default=1.0, default_status="assumed",
        doc="Clutch force of one fully-seated stud -- the average of push-on and pull-off force "
            "per stud (push = clutch*(1+beta), pull = clutch*(1-beta)). The most important "
            "unknown in the project; the plausible range is about 0.5-5 N. The shipped default "
            "matches the simulator's own bare placeholder; a first-principles estimate of 4.79 N "
            "(band 2.7-10.0) is documented in guide 02c as an upper scenario, not used here. "
            "Sets SnapParams.f_clutch."),
    "bricks.push_pull_asymmetry_beta": FieldSpec(
        "bricks", "-", float, 0.0, 0.9, default=0.15,
        doc="Push/pull asymmetry: pushing a stud in costs f_clutch*(1+beta), pulling it out "
            "releases at f_clutch*(1-beta). Sets SnapParams.beta."),
    "bricks.press_curve_shape": FieldSpec(
        "bricks", "-", str, choices=("ramp", "plateau", "detent"), default="ramp",
        doc="Shape of the force-against-depth curve. Sets SnapParams.curve."),
    "bricks.press_curve_exponent": FieldSpec(
        "bricks", "-", float, 0.3, 4.0, default=1.91, default_status="derived",
        doc="Exponent of the 'ramp'/'detent' curve's rising part. Sets SnapParams.exponent."),
    "bricks.press_curve_plateau_gain": FieldSpec(
        "bricks", "-", float, 0.0, 1.0, default=0.35,
        doc="Height of the 'detent' curve's flat part, as a fraction of f_clutch. Sets "
            "SnapParams.plateau_gain. Unused unless press_curve_shape is 'detent'."),
    "bricks.press_curve_detent_gain": FieldSpec(
        "bricks", "-", float, 0.0, 3.0, default=1.0,
        doc="Height of the 'detent' curve's entry bump above its plateau, as a fraction of "
            "f_clutch. Sets SnapParams.detent_gain."),
    "bricks.press_curve_detent_depth_mm": FieldSpec(
        "bricks", "mm", float, 0.0, 4.5, default=1.0,
        doc="Depth the 'detent' curve's entry bump is centred on. Sets SnapParams.detent_depth."),
    "bricks.press_curve_detent_width_mm": FieldSpec(
        "bricks", "mm", float, 0.0, 4.5, default=0.6,
        doc="Width of the 'detent' curve's entry bump. Sets SnapParams.detent_width."),
    # -- bricks: geometry (BrickSpec) -------------------------------------------------------
    "bricks.pitch_mm": FieldSpec("bricks", "mm", float, 8.0, 24.0, default=16.0,
                                  doc="Centre-to-centre stud spacing. Sets BrickSpec.pitch."),
    "bricks.stud_outer_diameter_mm": FieldSpec(
        "bricks", "mm", float, 4.0, 14.0, default=9.4,
        doc="Outer diameter of one stud. Sets BrickSpec.stud_d."),
    "bricks.stud_inner_diameter_mm": FieldSpec(
        "bricks", "mm", float, 0.0, 10.0, default=6.5,
        doc="Inner (hollow) diameter of one stud, 0 for a solid stud. Sets BrickSpec.stud_inner_d."),
    "bricks.stud_height_mm": FieldSpec("bricks", "mm", float, 1.0, 8.0, default=4.5,
                                        doc="Height a stud projects above the roof. Sets BrickSpec.stud_h."),
    "bricks.wall_thickness_mm": FieldSpec("bricks", "mm", float, 0.5, 4.0, default=1.55,
                                           doc="Thickness of the four outer side walls. Sets BrickSpec.wall."),
    "bricks.tube_outer_diameter_mm": FieldSpec(
        "bricks", "mm", float, 4.0, 20.0, default=13.4,
        doc="Outer diameter of an interior support tube. Sets BrickSpec.tube_od."),
    "bricks.tube_inner_diameter_mm": FieldSpec(
        "bricks", "mm", float, 2.0, 18.0, default=10.7,
        doc="Inner diameter of an interior support tube. Sets BrickSpec.tube_id."),
    "bricks.roof_thickness_mm": FieldSpec("bricks", "mm", float, 0.5, 4.0, default=1.5,
                                           doc="Thickness of the top plate. Sets BrickSpec.roof."),
    "bricks.brick_height_mm": FieldSpec(
        "bricks", "mm", float, 8.0, 30.0, default=19.2,
        doc="Outer height of the brick body, not counting the studs. Sets BrickSpec.height."),
    "bricks.clearance_mm": FieldSpec(
        "bricks", "mm", float, 0.0, 1.0, default=0.2,
        doc="Radius removed from every stud so it can physically enter a socket. Sets "
            "BrickSpec.clearance."),
    "bricks.chamfer_mm": FieldSpec(
        "bricks", "mm", float, 0.0, 2.0, default=0.5,
        doc="Height of the tapered lead-in at a stud's tip. Sets BrickSpec.chamfer."),
    "bricks.density_kg_m3": FieldSpec(
        "bricks", "kg/m^3", float, 700.0, 1600.0, default=1050.0,
        doc="Bulk material density used to compute brick mass and inertia. Sets BrickSpec.density."),
    "bricks.mass_2x2_g": FieldSpec(
        "bricks", "g", float, 0.5, 30.0, nullable=True, default=6.60, default_status="derived",
        doc="Mass of an assembled 2x2 brick. Cross-check only: compare against what "
            "density_kg_m3 plus the dimensions above actually generate."),
    "bricks.mass_2x4_g": FieldSpec(
        "bricks", "g", float, 1.0, 60.0, nullable=True, default=12.52, default_status="derived",
        doc="Mass of an assembled 2x4 brick. Cross-check only, same as mass_2x2_g."),
    "bricks.brick_on_brick_friction": FieldSpec(
        "bricks", "-", float, 0.02, 2.0, default=0.35,
        doc="Sliding friction coefficient, ABS brick on ABS brick/baseplate. Sets "
            "SceneConfig.brick_friction and the press-bench's BenchConfig.friction."),
    "bricks.jaw_on_brick_friction": FieldSpec(
        "bricks", "-", float, 0.02, 2.0, default=0.40,
        doc="Sliding friction coefficient, the bare printed jaw face on an ABS brick. Sets "
            "ArmConfig.pad_friction."),
    "bricks.lateral_capture_tolerance_mm": FieldSpec(
        "bricks", "mm", float, 0.01, 2.0, default=0.30,
        doc="How far off-centre a stud may be and still be captured by a socket. Sets "
            "SnapParams.capture_lateral."),
    # -- bricks: contact model, damage and breakable bricks ---------------------------------
    "bricks.contact_model": FieldSpec(
        "bricks", "-", str, choices=("material", "derived", "mujoco"), default="material",
        default_status="derived",
        doc="Where a contact's solref/solimp come from. All three derive a flat-punch contact "
            "stiffness from the ABS contact modulus and a stated patch size and then clip it. "
            "'material' clips at the solver's floor (4 ms at a 2 ms step) on the jaw pads and the "
            "arm links, and leaves brick-on-brick, brick-on-baseplate and brick-on-table on "
            "MuJoCo's own settings, because the derived ones stop the press 4.50 mm short and "
            "the pick-and-place 0.39 mm short of seated. 'derived' clips "
            "at the solver floor on every pair, which is what the material asks for. 'mujoco' "
            "leaves MuJoCo's shipped defaults everywhere. Sets SceneConfig.contact_model."),
    "bricks.youngs_modulus_GPa": FieldSpec(
        "bricks", "GPa", float, 0.5, 6.0, default=2.3, default_status="derived",
        doc="ABS Young's modulus at 23 C. Datasheet figure for unfilled injection-moulding ABS; "
            "generic range 1.8 to 2.7. Feeds the contact stiffness, the press-fit stiffness and "
            "every damage threshold. Measure with a three-point bend on a brick wall (guide "
            "02c section 7)."),
    "bricks.tensile_yield_MPa": FieldSpec(
        "bricks", "MPa", float, 10.0, 120.0, default=44.0, default_status="derived",
        doc="ABS tensile yield at 23 C. The permanent-set ('yielded') damage threshold and the "
            "marking threshold are both written against it. Generic range 36 to 52."),
    "bricks.flexural_strength_MPa": FieldSpec(
        "bricks", "MPa", float, 10.0, 200.0, default=70.0, default_status="derived",
        doc="ABS flexural strength at 23 C. The 'cracked' damage threshold is written against "
            "it, and 'broken' at 1.5 times it. Generic range 60 to 95."),
    "bricks.damage_monitor": FieldSpec(
        "bricks", "-", str, choices=("on", "off"), default="on",
        doc="Attach a DamageMonitor to every scene. It rides on the snap model's own update, so "
            "nothing in a controller has to know about it, and reports a damage state and a peak "
            "wall stress per brick. Sets SceneConfig.damage_monitor."),
    "bricks.damage_monitor_every": FieldSpec(
        "bricks", "steps", int, 1, 100, default=10,
        doc="Physics steps between damage passes. 10 is one control tick and measures at about "
            "2 % of step time; 1 costs about 17 %. Sets SceneConfig.damage_monitor_every."),
    "bricks.breakable": FieldSpec(
        "bricks", "-", str, choices=("on", "off"), default="off",
        doc="Build bricks out of welded pieces that can come apart instead of one rigid body. "
            "Off by default: a breakable 2x4 steps 6.8 times slower. Sets "
            "SceneConfig.breakable."),
    "bricks.breakable_limit_scale": FieldSpec(
        "bricks", "-", float, 0.01, 10.0, default=1.0,
        doc="Multiplies every weld limit of a breakable brick at once, for randomization or for "
            "making a demonstration fail at a convenient load. Sets "
            "SceneConfig.breakable_limit_scale."),
    # -- bricks: raw measurements, used only to fit f_clutch/beta in the press-bench experiment
    "bricks.measured.push_on_2x2_N": FieldSpec(
        "bricks", "N", float, 0.0, 500.0, nullable=True, default=None,
        doc="Peak force to press a whole 2x2 fully home, kitchen-scale test (guide 10). Leave "
            "null until measured."),
    "bricks.measured.push_on_2x4_N": FieldSpec(
        "bricks", "N", float, 0.0, 500.0, nullable=True, default=None,
        doc="Same, for a 2x4."),
    "bricks.measured.pull_off_2x2_N": FieldSpec(
        "bricks", "N", float, 0.0, 500.0, nullable=True, default=None,
        doc="Peak force to pull a whole seated 2x2 straight off, hanging-bottle test (guide 10)."),
    "bricks.measured.pull_off_2x4_N": FieldSpec(
        "bricks", "N", float, 0.0, 500.0, nullable=True, default=None,
        doc="Same, for a 2x4."),
    # -- arm ----------------------------------------------------------------------------------
    "arm.supply_voltage_V": FieldSpec(
        "arm", "V", float, 3.0, 15.0, default=7.4,
        doc="Servo supply voltage, read off the power supply or the servo tool. Sets "
            "ArmConfig.vin and ServoConfig.vin."),
    "arm.servo_p_gain": FieldSpec(
        "arm", "-", int, 1, 255, default=16,
        doc="STS3215 P register (address 21), read from the servo tool. Sets ArmConfig.p_gain."),
    "arm.torque_limit_Nm": FieldSpec(
        "arm", "N*m", float, 0.2, 8.0, default=1.9,
        doc="Servo stall torque at supply_voltage_V. Sets ArmConfig.torque_limit_nm."),
    "arm.backlash_total_deg": FieldSpec(
        "arm", "deg", float, 0.0, 10.0, default=0.87,
        doc="TOTAL gear free-play at the output, felt with a ruler at the tool (a dial gauge on a "
            "lever, servo powered, is the bench-accurate version). ArmConfig.backlash_deg is a "
            "half-range, so to_arm_config halves this value before setting it."),
    "arm.servo_dead_band_counts": FieldSpec(
        "arm", "counts", int, 0, 10, default=0,
        doc="Firmware position dead band, STS3215 registers 26 (CW) / 27 (CCW): while the position "
            "error stays inside this many encoder counts, the firmware commands nothing. Default 0 "
            "(nominal, matches the identified law with nothing added); documented value is about 2 "
            "counts. Sets ServoConfig.error_deadband_ticks."),
    "arm.joint_friction_Nm": FieldSpec(
        "arm", "N*m", float, 0.0, 1.0, default=0.0515,
        doc="Dry friction in one servo's gear train. Recorded for reference only: ArmConfig has "
            "no field for this today, so it does not reach the simulator. See the loader's "
            "report for the hook this is missing."),
    # -- camera -------------------------------------------------------------------------------
    "camera.diagonal_fov_deg": FieldSpec(
        "camera", "deg", float, 30.0, 170.0, default=115.0,
        doc="Wrist camera lens field of view, corner to corner. Converted to a vertical fovy "
            "for a given resolution by fovy_from_diagonal_fov."),
    "camera.resolution_width_px": FieldSpec(
        "camera", "px", int, 32, 4096, default=640,
        doc="Wrist camera operating resolution, width. Also sets SceneConfig.offscreen."),
    "camera.resolution_height_px": FieldSpec(
        "camera", "px", int, 32, 4096, default=480,
        doc="Wrist camera operating resolution, height."),
    "camera.distortion_k1": FieldSpec(
        "camera", "-", float, -1.0, 1.0, default=-0.225,
        doc="2nd-order radial lens distortion, render.domain_rand.PostProcessConfig convention."),
    "camera.distortion_k2": FieldSpec(
        "camera", "-", float, -1.0, 1.0, default=0.04,
        doc="4th-order radial lens distortion, same convention."),
    "camera.fx_px": FieldSpec(
        "camera", "px", float, 10.0, 8000.0, default=254.828,
        doc="Calibrated focal length in pixels, x, at resolution_width_px x resolution_height_px. "
            "Read from cv2.calibrateCamera; the procedure is in docs/guide/08."),
    "camera.fy_px": FieldSpec(
        "camera", "px", float, 10.0, 8000.0, default=254.828,
        doc="Calibrated focal length in pixels, y.  See fx_px."),
    "camera.cx_px": FieldSpec(
        "camera", "px", float, 0.0, 4096.0, default=319.5,
        doc="Calibrated principal point, x [px].  See fx_px."),
    "camera.cy_px": FieldSpec(
        "camera", "px", float, 0.0, 4096.0, default=239.5,
        doc="Calibrated principal point, y [px].  See fx_px."),
    "camera.opencv_k1": FieldSpec(
        "camera", "-", float, -2.0, 2.0, default=0.0,
        doc="OpenCV (Brown-Conrady) 2nd-order radial distortion, as cv2.calibrateCamera returns "
            "it.  A different parametrization from distortion_k1, which is the image-space remap "
            "render.domain_rand uses; the two are not interchangeable."),
    "camera.opencv_k2": FieldSpec(
        "camera", "-", float, -2.0, 2.0, default=0.0,
        doc="OpenCV 4th-order radial distortion.  See opencv_k1."),
    "camera.opencv_p1": FieldSpec(
        "camera", "-", float, -1.0, 1.0, default=0.0,
        doc="OpenCV 1st tangential distortion coefficient.  See opencv_k1."),
    "camera.opencv_p2": FieldSpec(
        "camera", "-", float, -1.0, 1.0, default=0.0,
        doc="OpenCV 2nd tangential distortion coefficient.  See opencv_k1."),
    "camera.opencv_k3": FieldSpec(
        "camera", "-", float, -2.0, 2.0, default=0.0,
        doc="OpenCV 6th-order radial distortion.  See opencv_k1."),
    "camera.mount_x_mm": FieldSpec(
        "camera", "mm", float, -200.0, 200.0, default=2.5,
        doc="Camera position in the gripper body frame, x.  Hand-eye result from "
            "cv2.calibrateHandEye, converted to MuJoCo's camera convention; compare against "
            "patch_so101.WRIST_CAM_POS_IN_GRIPPER."),
    "camera.mount_y_mm": FieldSpec(
        "camera", "mm", float, -200.0, 200.0, default=73.44,
        doc="Camera position in the gripper body frame, y.  See mount_x_mm."),
    "camera.mount_z_mm": FieldSpec(
        "camera", "mm", float, -200.0, 200.0, default=6.59,
        doc="Camera position in the gripper body frame, z.  See mount_x_mm."),
    "camera.mount_roll_deg": FieldSpec(
        "camera", "deg", float, -180.0, 180.0, default=-25.073,
        doc="Camera orientation in the gripper body frame, rotation about x.  The vendored mount "
            "is a pure x rotation, so the other two are nominally zero; see mount_x_mm."),
    "camera.mount_pitch_deg": FieldSpec(
        "camera", "deg", float, -180.0, 180.0, default=0.0,
        doc="Camera orientation in the gripper body frame, rotation about y.  See mount_roll_deg."),
    "camera.mount_yaw_deg": FieldSpec(
        "camera", "deg", float, -180.0, 180.0, default=0.0,
        doc="Camera orientation in the gripper body frame, rotation about z.  See mount_roll_deg."),
    # -- workspace ------------------------------------------------------------------------------
    "workspace.baseplate_x_m": FieldSpec(
        "workspace", "m", float, -1.0, 1.0, default=0.17,
        doc="Baseplate centre, x, in the arm's base frame. Sets SceneConfig.plate_xy."),
    "workspace.baseplate_y_m": FieldSpec(
        "workspace", "m", float, -1.0, 1.0, default=0.0,
        doc="Baseplate centre, y."),
    "workspace.baseplate_yaw_deg": FieldSpec(
        "workspace", "deg", float, -180.0, 180.0, default=0.0,
        doc="Baseplate rotation about its own centre. Sets SceneConfig.plate_yaw (radians)."),
    "workspace.tray_x_m": FieldSpec(
        "workspace", "m", float, -1.0, 1.0, default=0.13,
        doc="Spare-brick tray centre, x. Sets SceneConfig.tray_xy."),
    "workspace.tray_y_m": FieldSpec(
        "workspace", "m", float, -1.0, 1.0, default=-0.17,
        doc="Spare-brick tray centre, y."),
}


# ---------------------------------------------------------------------------
# Values and the loaded file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamValue:
    """One leaf of a loaded parameter file."""

    path: str
    value: Any
    status: Status
    unit: str
    measured_by: str | None = None
    measured_on: str | None = None
    notes: str = ""

    def __float__(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class ParamFile:
    """Every parameter, loaded, validated and merged. Immutable once built.

    ``source`` is the path this was loaded from (the override file, if there was one, else
    :data:`DEFAULT_PARAMS_PATH`); ``base_source`` is always :data:`DEFAULT_PARAMS_PATH`.
    """

    values: dict[str, ParamValue]
    source: Path
    base_source: Path

    def get(self, path: str) -> ParamValue:
        if path not in self.values:
            raise KeyError(f"{path!r} is not a known parameter (see brick_builder.params.SCHEMA)")
        return self.values[path]

    def value(self, path: str) -> Any:
        """The bare value at ``path`` (what most callers want)."""
        return self.get(path).value

    def section(self, name: str) -> dict[str, ParamValue]:
        """Every parameter whose dotted path starts with ``name.``."""
        prefix = name + "."
        return {p: v for p, v in self.values.items() if p.startswith(prefix)}

    def as_nested_dict(self) -> dict:
        """The full file as a plain nested dict, suitable for ``yaml.safe_dump``."""
        out: dict = {}
        for path, pv in sorted(self.values.items()):
            parts = path.split(".")
            node = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = {
                "value": pv.value,
                "status": pv.status,
                "measured_by": pv.measured_by,
                "measured_on": pv.measured_on,
                "notes": pv.notes,
            }
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.as_nested_dict(), sort_keys=True, allow_unicode=True)


# ---------------------------------------------------------------------------
# Reading and flattening YAML
# ---------------------------------------------------------------------------


_LEAF_MARKER_KEYS = {"value", "status", "measured_by", "measured_on", "notes"}
"""Keys that only ever appear on a parameter block, never on a section name in this schema.  A
non-empty mapping using only these keys is treated as a (possibly malformed) leaf rather than a
section, even without a ``value`` key -- e.g. ``{status: measured}`` with a forgotten ``value``
-- so :func:`_normalize_leaf` gets to report the specific, helpful "missing value" problem
instead of ``_flatten`` silently recursing into it and reporting a confusing "unknown parameter
'foo.status'" instead."""


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Walk a nested dict down to every leaf that looks like a parameter entry.

    A leaf is either a bare scalar (``6.9``) or a mapping containing a ``value`` key (or, for a
    forgotten ``value``, a non-empty mapping using only :data:`_LEAF_MARKER_KEYS`). Anything else
    -- including an empty mapping, e.g. ``bricks: {}`` meaning "no overrides here" -- is a
    section and is recursed into.
    """
    out: dict[str, Any] = {}
    if not isinstance(node, dict):
        out[prefix] = node
        return out
    if "value" in node or (node and set(node) <= _LEAF_MARKER_KEYS):
        out[prefix] = node
        return out
    for key, child in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        out.update(_flatten(child, path))
    return out


def _read_yaml(path: Path) -> dict:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ParamsError([f"could not read {path}: {exc}"]) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParamsError([f"{path} is not valid YAML: {exc}"]) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ParamsError([f"{path} must contain a mapping at the top level, got {type(data).__name__}"])
    return data


def _normalize_leaf(
    path: str, raw: Any
) -> tuple[Any, Status | None, str | None, str | None, str, str | None]:
    """A leaf's raw YAML value to ``(value, status, measured_by, measured_on, notes, problem)``.

    A bare scalar is shorthand for a measured value: ``status`` comes back ``None`` (the
    caller fills in ``"measured"``, or ``"assumed"`` for a bare ``null``) so a member's terse
    override (``supply_voltage_V: 6.9``) does not have to spell out the block form. A mapping
    must at least have ``value``. Never raises: a problem is returned as the last element so the
    caller can collect every problem in the file instead of stopping at the first.
    """
    if isinstance(raw, dict):
        if "value" not in raw:
            return (None, None, None, None, "",
                     f"{path}: a parameter block must have a 'value' key, got keys {sorted(raw)}")
        status = raw.get("status")
        unknown = set(raw) - _LEAF_MARKER_KEYS
        if unknown:
            return (None, None, None, None, "",
                     f"{path}: unknown field(s) {sorted(unknown)} (allowed: value, status, "
                     "measured_by, measured_on, notes)")
        # measured_by/measured_on are free-form provenance text (a name, a date, "bench #3" -
        # whatever). YAML happily parses an unquoted 2026-09-20 into a datetime.date rather than
        # a string, which would then fail to JSON-encode wherever a report writes this out, so
        # both are normalized to plain strings (or None) right here rather than at every call
        # site that might serialize a ParamValue.
        measured_by = raw.get("measured_by")
        measured_on = raw.get("measured_on")
        return (
            raw["value"], status,
            None if measured_by is None else str(measured_by),
            None if measured_on is None else str(measured_on),
            str(raw.get("notes") or ""), None,
        )
    return (raw, None, None, None, "", None)


def _suggest(path: str, known: list[str]) -> str:
    close = difflib.get_close_matches(path, known, n=1)
    return f" did you mean {close[0]!r}?" if close else ""


def _validate_leaf(path: str, spec: FieldSpec, value: Any, status: Any) -> list[str]:
    problems: list[str] = []
    if status not in STATUSES:
        problems.append(f"{path}: status {status!r} must be one of {STATUSES}")
    if value is None:
        if not spec.nullable:
            problems.append(f"{path}: value is null, but this parameter is required")
        return problems
    if spec.kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number ({spec.unit}), got {value!r}")
            return problems
        value = float(value)
    elif spec.kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected an integer, got {value!r}")
            return problems
    elif spec.kind is str:
        if not isinstance(value, str):
            problems.append(f"{path}: expected a string, got {value!r}")
            return problems
        if spec.choices is not None and value not in spec.choices:
            problems.append(f"{path}: {value!r} is not one of {spec.choices}")
            return problems
    if spec.kind in (float, int) and (spec.lo is not None or spec.hi is not None):
        lo = -math.inf if spec.lo is None else spec.lo
        hi = math.inf if spec.hi is None else spec.hi
        if not (lo <= value <= hi):
            problems.append(
                f"{path}: {value} {spec.unit} is outside the plausible range [{lo}, {hi}] {spec.unit}"
            )
    return problems


def _parse_leaves(
    raw_flat: dict[str, Any], *, require_all: bool
) -> tuple[dict[str, ParamValue], list[str]]:
    """Every problem in ``raw_flat`` (unknown keys, bad blocks, range/type errors), collected.

    A leaf that has a problem is left out of the returned ``values``; the caller is expected to
    raise on a non-empty problem list rather than use a partially-valid result.
    """
    problems: list[str] = []
    values: dict[str, ParamValue] = {}
    known = list(SCHEMA)
    for path, raw in raw_flat.items():
        spec = SCHEMA.get(path)
        if spec is None:
            problems.append(f"unknown parameter {path!r}.{_suggest(path, known)}")
            continue
        value, status, measured_by, measured_on, notes, leaf_problem = _normalize_leaf(path, raw)
        if leaf_problem is not None:
            problems.append(leaf_problem)
            continue
        if status is None:
            # A bare scalar is shorthand for "I put this number in myself" -- except a bare
            # null, which cannot have been measured; it just has not been filled in yet.
            status = "assumed" if value is None else "measured"
        leaf_problems = _validate_leaf(path, spec, value, status)
        if leaf_problems:
            problems.extend(leaf_problems)
            continue
        values[path] = ParamValue(
            path=path, value=value, status=status, unit=spec.unit,
            measured_by=measured_by, measured_on=measured_on, notes=notes,
        )
    if require_all:
        for path in SCHEMA:
            if path not in values and path not in raw_flat:
                problems.append(f"{path}: missing from the default parameter file (schema/file drift)")
    return values, problems


def _load_defaults() -> ParamFile:
    raw = _read_yaml(DEFAULT_PARAMS_PATH)
    flat = _flatten(raw)
    values, problems = _parse_leaves(flat, require_all=True)
    if problems:
        raise ParamsError(problems)
    return ParamFile(values=values, source=DEFAULT_PARAMS_PATH, base_source=DEFAULT_PARAMS_PATH)


def load_params(path: str | Path | None = None) -> ParamFile:
    """Load a parameter file, merged over :data:`DEFAULT_PARAMS_PATH`.

    ``path=None`` (or ``path`` pointing at the default file itself) returns the shipped defaults
    unchanged. Otherwise, ``path`` is read, validated against :data:`SCHEMA`, and every leaf it
    mentions replaces the matching default; a leaf it does not mention -- including its
    ``status`` -- is left exactly as the default file has it. Raises :class:`ParamsError` with
    every problem found, not just the first.
    """
    defaults = _load_defaults()
    if path is None:
        return defaults
    path = Path(path)
    if path.resolve() == DEFAULT_PARAMS_PATH.resolve():
        return defaults

    raw = _read_yaml(path)
    flat = _flatten(raw)
    override_values, problems = _parse_leaves(flat, require_all=False)
    if problems:
        raise ParamsError(problems)

    merged = dict(defaults.values)
    merged.update(override_values)
    return ParamFile(values=merged, source=path, base_source=DEFAULT_PARAMS_PATH)


# ---------------------------------------------------------------------------
# Listing assumptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assumption:
    """One parameter that is not yet backed by a measurement on this team's own hardware."""

    path: str
    value: Any
    status: Status
    unit: str
    doc: str


def list_assumptions(params: ParamFile, include_derived: bool = True) -> list[Assumption]:
    """Every parameter whose status is not ``"measured"``, worst-justified first.

    ``"assumed"`` (a placeholder or a borrowed number from someone else's part) sorts before
    ``"derived"`` (worked out from other assumed numbers, so better than a guess but still not a
    measurement on this arm or these bricks). Pass ``include_derived=False`` to see only the
    bare guesses.
    """
    order = {"assumed": 0, "derived": 1, "measured": 2}
    wanted: tuple[Status, ...] = ("assumed", "derived") if include_derived else ("assumed",)
    out = [
        Assumption(path=pv.path, value=pv.value, status=pv.status, unit=pv.unit,
                   doc=SCHEMA[pv.path].doc if pv.path in SCHEMA else "")
        for pv in params.values.values()
        if pv.status in wanted
    ]
    out.sort(key=lambda a: (order[a.status], a.path))
    return out


# ---------------------------------------------------------------------------
# Converting into the simulator's own config objects
# ---------------------------------------------------------------------------


def filter_dataclass_kwargs(cls: type, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only the keys of ``kwargs`` that are fields of ``cls`` right now.

    Fields are still actively being added and removed on some of these dataclasses
    (``SceneConfig`` in particular). Every converter in this module builds its target through
    this function instead of calling the constructor with a fixed kwarg list, so a field that
    has been renamed or removed is silently left out rather than raising a ``TypeError`` at a
    random call site. Returns ``(kept_kwargs, skipped_names)`` so a caller can report what it
    could not set.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    kept = {k: v for k, v in kwargs.items() if k in valid}
    skipped = sorted(set(kwargs) - valid)
    return kept, skipped


def to_brick_spec(params: ParamFile, base: BrickSpec = DUPLO) -> BrickSpec:
    """A :class:`~brick_builder.bricks.spec.BrickSpec` from the ``bricks.*_mm``/``density_kg_m3`` fields."""
    mm = lambda path: float(params.value(path)) / 1000.0  # noqa: E731
    kwargs = {
        "pitch": mm("bricks.pitch_mm"),
        "height": mm("bricks.brick_height_mm"),
        "stud_d": mm("bricks.stud_outer_diameter_mm"),
        "stud_h": mm("bricks.stud_height_mm"),
        "stud_inner_d": mm("bricks.stud_inner_diameter_mm"),
        "wall": mm("bricks.wall_thickness_mm"),
        "tube_od": mm("bricks.tube_outer_diameter_mm"),
        "tube_id": mm("bricks.tube_inner_diameter_mm"),
        "roof": mm("bricks.roof_thickness_mm"),
        "clearance": mm("bricks.clearance_mm"),
        "chamfer": mm("bricks.chamfer_mm"),
        "density": float(params.value("bricks.density_kg_m3")),
    }
    kept, _ = filter_dataclass_kwargs(BrickSpec, kwargs)
    return dataclasses.replace(base, **kept)


def to_snap_params(params: ParamFile, base: SnapParams = SnapParams()) -> SnapParams:
    """A :class:`~brick_builder.bricks.snap_api.SnapParams` from the ``bricks.*`` snap-fit fields."""
    kwargs = {
        "f_clutch": float(params.value("bricks.clutch_force_per_stud_N")),
        "beta": float(params.value("bricks.push_pull_asymmetry_beta")),
        "curve": str(params.value("bricks.press_curve_shape")),
        "exponent": float(params.value("bricks.press_curve_exponent")),
        "plateau_gain": float(params.value("bricks.press_curve_plateau_gain")),
        "detent_gain": float(params.value("bricks.press_curve_detent_gain")),
        "detent_depth": float(params.value("bricks.press_curve_detent_depth_mm")) / 1000.0,
        "detent_width": float(params.value("bricks.press_curve_detent_width_mm")) / 1000.0,
        "capture_lateral": float(params.value("bricks.lateral_capture_tolerance_mm")) / 1000.0,
    }
    kept, _ = filter_dataclass_kwargs(SnapParams, kwargs)
    return dataclasses.replace(base, **kept)


def to_arm_config(params: ParamFile, base: ArmConfig = ArmConfig()) -> ArmConfig:
    """An :class:`~brick_builder.robot.patch_so101.ArmConfig` from the ``arm.*`` fields.

    ``arm.backlash_total_deg`` is the TOTAL free play a teammate measures at the tool;
    ``ArmConfig.backlash_deg`` is a half-range, so this halves it before setting the field.

    ``arm.joint_friction_Nm`` is recorded in the parameter file but has no matching field on
    ``ArmConfig`` today (``patch_so101._patch_joints`` hard-codes the BAM-identified dry
    friction); :func:`filter_dataclass_kwargs` simply leaves it out, and it never reaches this
    function's kwargs in the first place because there is nothing to put it in.
    """
    kwargs = {
        "vin": float(params.value("arm.supply_voltage_V")),
        "p_gain": int(params.value("arm.servo_p_gain")),
        "torque_limit_nm": float(params.value("arm.torque_limit_Nm")),
        "backlash_deg": float(params.value("arm.backlash_total_deg")) / 2.0,
        "pad_friction": float(params.value("bricks.jaw_on_brick_friction")),
    }
    kept, _ = filter_dataclass_kwargs(ArmConfig, kwargs)
    return dataclasses.replace(base, **kept)


def to_servo_config(params: ParamFile, base: ServoConfig = ServoConfig()) -> ServoConfig:
    """A :class:`~brick_builder.robot.servo_model.ServoConfig` with the measured supply voltage
    and firmware dead band."""
    kwargs = {
        "vin": float(params.value("arm.supply_voltage_V")),
        "error_deadband_ticks": float(params.value("arm.servo_dead_band_counts")),
    }
    kept, _ = filter_dataclass_kwargs(ServoConfig, kwargs)
    return dataclasses.replace(base, **kept)


@dataclass(frozen=True)
class CameraSettings:
    """The wrist camera's field of view and resolution, plus its lens distortion.

    There is no ``CameraConfig`` dataclass anywhere in the simulator to build -- the wrist
    camera's fovy and pose are module-level constants in ``patch_so101.py``, computed once at
    import time. This is a params-module-local stand-in so a member's field-of-view entry has
    somewhere typed to land; see the accompanying report for why a real ``CameraConfig`` would
    be a useful hook.
    """

    diagonal_fov_deg: float
    width_px: int
    height_px: int
    fovy_deg: float
    distortion_k1: float
    distortion_k2: float


def to_camera_settings(params: ParamFile) -> CameraSettings:
    """Camera parameters, with the diagonal fov converted to MuJoCo's vertical ``fovy``."""
    diag = float(params.value("camera.diagonal_fov_deg"))
    width = int(params.value("camera.resolution_width_px"))
    height = int(params.value("camera.resolution_height_px"))
    return CameraSettings(
        diagonal_fov_deg=diag,
        width_px=width,
        height_px=height,
        fovy_deg=fovy_from_diagonal_fov(diag, width, height),
        distortion_k1=float(params.value("camera.distortion_k1")),
        distortion_k2=float(params.value("camera.distortion_k2")),
    )


def to_postprocess_config(params: ParamFile, base: Any = None) -> Any:
    """A ``render.domain_rand.PostProcessConfig`` with this file's lens distortion.

    Imported lazily so that loading a parameter file never requires ``cv2``
    (``render.domain_rand`` imports it at module level) unless a caller actually wants the
    post-process config.
    """
    from .render.domain_rand import PostProcessConfig

    base = PostProcessConfig() if base is None else base
    cam = to_camera_settings(params)
    kwargs = {"k1": cam.distortion_k1, "k2": cam.distortion_k2}
    kept, _ = filter_dataclass_kwargs(PostProcessConfig, kwargs)
    return dataclasses.replace(base, **kept)


def to_scene_config(params: ParamFile, base: Any = None, **overrides: Any) -> tuple[Any, list[str]]:
    """A :class:`~brick_builder.scene.compose.SceneConfig` from the workspace/arm/brick fields.

    Imported lazily (``scene.compose`` pulls in the whole robot+bricks+kinematics stack) and
    built entirely through :func:`filter_dataclass_kwargs`, per this module's own docstring:
    fields such as ``SceneConfig.press_shoe`` are still in flux elsewhere in the codebase, and
    this function must keep working whether a given field exists or not. ``overrides`` lets a
    caller add fields this module does not know about (e.g. ``held=...``, ``tray=True``) without
    having to reconstruct the result afterwards; they go through the same filter. Returns
    ``(scene_config, skipped_field_names)``.
    """
    from .scene.compose import SceneConfig

    base = SceneConfig() if base is None else base
    yaw = math.radians(float(params.value("workspace.baseplate_yaw_deg")))
    kwargs = {
        "arm": to_arm_config(params, base.arm if hasattr(base, "arm") else ArmConfig()),
        "snap": to_snap_params(params, base.snap if hasattr(base, "snap") else SnapParams()),
        "spec": to_brick_spec(params, base.spec if hasattr(base, "spec") else DUPLO),
        "plate_xy": (
            float(params.value("workspace.baseplate_x_m")),
            float(params.value("workspace.baseplate_y_m")),
        ),
        "plate_yaw": yaw,
        "tray_xy": (
            float(params.value("workspace.tray_x_m")),
            float(params.value("workspace.tray_y_m")),
        ),
        "brick_friction": float(params.value("bricks.brick_on_brick_friction")),
        "contact_model": str(params.value("bricks.contact_model")),
        "damage_monitor": params.value("bricks.damage_monitor") == "on",
        "damage_monitor_every": int(params.value("bricks.damage_monitor_every")),
        "breakable": params.value("bricks.breakable") == "on",
        "breakable_limit_scale": float(params.value("bricks.breakable_limit_scale")),
        "offscreen": (
            int(params.value("camera.resolution_width_px")),
            int(params.value("camera.resolution_height_px")),
        ),
        **overrides,
    }
    kept, skipped = filter_dataclass_kwargs(SceneConfig, kwargs)
    return dataclasses.replace(base, **kept), skipped


def expected_brick_mass_g(spec: BrickSpec, nx: int, ny: int) -> float:
    """What a generated ``nx`` x ``ny`` brick actually masses at ``spec.density`` [g].

    Compiles a throwaway one-brick model and reads MuJoCo's own computed ``body_mass`` -- the
    same check ``tests/bricks/test_materials.py::test_density_reproduces_the_catalogue_mass``
    makes for the shipped defaults. Used by the ``params`` experiment to compare a member's
    ``mass_2x2_g``/``mass_2x4_g`` measurement against what their ``density_kg_m3`` predicts.
    """
    import mujoco

    from .bricks.generator import add_brick

    world = mujoco.MjSpec()
    add_brick(world, "b", nx, ny, "red", (0.0, 0.0, 0.0), spec=spec)
    model = world.compile()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    return float(model.body_mass[bid]) * 1000.0
