"""Writing the target bundle: one little-endian blob the C controller reads in place.

Section order and every field's type are fixed here and in `firmware/control_lib/include/
bb_bundle.h`; the version number in the header changes whenever the two must be rebuilt
together. Sections are padded to eight bytes so the controller can point doubles straight at
them without copying.

Nothing here is control. It is the host writing down what its digital twin measured.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from ..action.place import PlaceConfig
from ..action.sequence import IdealSequence
from ..action.servo import VisualServo
from .plan import Plan

MAGIC = 0x42425442
VERSION = 2
K = 8
W = 14
HEADER = struct.Struct("<16I")

HELD_NONE = -1
"""A frame's placed-brick name when the task places nothing: relative mode checks nothing."""
HELD_UNLISTED = -2
"""When the task places a brick but this frame does not list it: never matched."""

#: Order of the doubles in `bb_cfg`, then of its int32s. Changing either changes VERSION.
CFG_DOUBLES = (
    "lam", "tol_mm", "advance_mm", "tube_scale", "dwell_travel_mm", "dwell_speed",
    "min_gain", "max_correction", "stall_load", "grasp_yaw_deg", "grasp_miss",
    "pick_freeze_below", "grasp_z", "servo_max_xy", "servo_max_spin", "servo_near",
    "servo_smooth_in_hand", "servo_dead_band_mm", "servo_dead_band_single_mm",
    "servo_min_yaw_weight", "servo_truncate_at", "servo_max_in_hand", "delta_xy_cap",
    "delta_spin_cap", "jc_ki", "jc_clamp", "jc_deadzone", "jc_dt", "grip_lo", "grip_hi",
)
CFG_INTS = (
    "jacobian_every", "match_window", "max_hold", "settle_ticks", "grasp_ticks", "grasp_wait",
    "servo_freeze_in_hand", "servo_min_refs", "descent_k", "grasp_k", "freeze_index", "pick",
    "closed_loop", "use_yaw", "dwell_only", "vision_every", "jc_joints", "pad0",
)

#: Saturation of one correction increment, as `TrackingEpisode._feedback` applies it.
DELTA_XY_CAP = 0.0012
DELTA_SPIN_CAP = math.radians(0.7)


class _Writer:
    """Bytes with eight-byte section alignment."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.size = 0

    def align(self) -> int:
        pad = (-self.size) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        return self.size

    def write(self, data: bytes) -> int:
        at = self.size
        self.parts.append(data)
        self.size += len(data)
        return at

    def bytes(self) -> bytes:
        return b"".join(self.parts)


def config_values(cfg: PlaceConfig, plan: Plan, *, grip_range, jc_ki: float = 0.0,
                  jc_joints: int = 0) -> dict:
    """The scalars the controller runs on, gathered from the config and the recording.

    The servo's own constants come from `VisualServo`'s defaults, so one place still declares
    them and a change there reaches the C controller through the bundle.
    """
    servo = VisualServo(step=None)
    dwell_only = cfg.pick if cfg.dwell_only is None else cfg.dwell_only
    return {
        "lam": cfg.lam,
        "tol_mm": cfg.tol_mm,
        "advance_mm": cfg.advance_mm,
        "tube_scale": cfg.tube_scale,
        "dwell_travel_mm": cfg.dwell_travel_mm,
        "dwell_speed": cfg.dwell_speed,
        "min_gain": cfg.min_gain,
        "max_correction": cfg.max_correction,
        "stall_load": cfg.stall_load,
        "grasp_yaw_deg": cfg.grasp_yaw_deg,
        "grasp_miss": cfg.grasp_miss,
        "pick_freeze_below": cfg.pick_freeze_below,
        "grasp_z": plan.grasp_z,
        "servo_max_xy": servo.max_xy,
        "servo_max_spin": servo.max_spin,
        "servo_near": servo.near,
        "servo_smooth_in_hand": servo.smooth_in_hand,
        "servo_dead_band_mm": servo.dead_band_mm,
        "servo_dead_band_single_mm": servo.dead_band_single_mm,
        "servo_min_yaw_weight": servo.min_yaw_weight,
        "servo_truncate_at": servo.truncate_at,
        "servo_max_in_hand": 0.012 if cfg.pick else servo.max_in_hand,
        "delta_xy_cap": DELTA_XY_CAP,
        "delta_spin_cap": DELTA_SPIN_CAP,
        "jc_ki": jc_ki,
        "jc_clamp": 40.0,
        "jc_deadzone": 1.0,
        "jc_dt": 0.02,
        "grip_lo": float(grip_range[0]),
        "grip_hi": float(grip_range[1]),
        "jacobian_every": cfg.jacobian_every,
        "match_window": cfg.match_window,
        "max_hold": cfg.max_hold,
        "settle_ticks": cfg.settle_ticks,
        "grasp_ticks": cfg.grasp_ticks,
        "grasp_wait": cfg.grasp_wait,
        "servo_freeze_in_hand": servo.freeze_in_hand,
        "servo_min_refs": 1,
        "descent_k": plan.descent_k,
        "grasp_k": plan.grasp_k,
        "freeze_index": plan.freeze_index,
        "pick": int(bool(cfg.pick)),
        "closed_loop": int(bool(cfg.closed_loop)),
        "use_yaw": int(bool(cfg.use_yaw)),
        "dwell_only": int(bool(dwell_only)),
        "vision_every": 5,
        "jc_joints": int(jc_joints),
        "pad0": 0,
    }


def _pack_cfg(values: dict) -> bytes:
    out = struct.pack("<%dd" % len(CFG_DOUBLES), *[float(values[n]) for n in CFG_DOUBLES])
    out += struct.pack("<%di" % len(CFG_INTS), *[int(values[n]) for n in CFG_INTS])
    return out + b"\0" * ((-len(out)) % 8)


def _pack_arm(arm: dict) -> bytes:
    out = b""
    out += np.asarray(arm["body_pos"], dtype="<f8").tobytes()
    out += np.asarray(arm["body_quat"], dtype="<f8").tobytes()
    out += np.asarray(arm["axis"], dtype="<f8").tobytes()
    out += np.asarray(arm["site_pos"], dtype="<f8").tobytes()
    out += np.asarray(arm["site_quat"], dtype="<f8").tobytes()
    out += np.asarray(arm["jnt_lo"], dtype="<f8").tobytes()
    out += np.asarray(arm["jnt_hi"], dtype="<f8").tobytes()
    return out


def pack_state(state) -> bytes:
    """One scene state as the wire and the bundle both carry it."""
    slots = np.ascontiguousarray(state.slots, dtype="<f4")
    depth = np.ascontiguousarray(state.depth, dtype="<f4")
    ids = np.ascontiguousarray(state.slot_ids, dtype="<i4")
    if slots.shape != (K, W):
        raise ValueError(f"expected a [{K},{W}] scene state, got {slots.shape}")
    return slots.tobytes() + depth.tobytes() + ids.tobytes()


def name_table(ideal: IdealSequence, plan: Plan) -> dict[str, int]:
    """One integer per reference title, so a frame can name the placed brick's reference and the
    controller can find it among whichever step's references are active at the time."""
    names: dict[str, int] = {}
    for fp in plan.frames:
        if fp.step is not None:
            for ref in list(fp.step.refs) + ([fp.step.brick_ref] if fp.step.brick_ref else []):
                names.setdefault(ref.name, len(names))
    for frame in ideal.frames:
        for title in (frame.bodies or {}):
            names.setdefault(title, len(names))
    return names


def held_title(ideal: IdealSequence, frame) -> str | None:
    """The title under which `frame` lists the placed brick, as `TrackingEpisode._feedback`
    looks it up in relative mode."""
    if ideal.held_body is None:
        return None
    titles = [t for t, body in (frame.bodies or {}).items() if body == ideal.held_body]
    return titles[0] if titles else None


def _pack_ref(ref, name_id: int) -> bytes:
    size = ref.size if ref.size is not None else (0.0, 0.0)
    out = struct.pack(
        "<9d",
        float(ref.colour[0]), float(ref.colour[1]), float(ref.colour[2]),
        float(size[0]), float(size[1]),
        float(ref.colour_tol), float(ref.size_tol),
        float(ref.min_mass if ref.min_mass is not None else 0.0),
        float(ref.max_border),
    )
    out += struct.pack("<4i", int(ref.size is not None), int(ref.min_mass is not None),
                       int(ref.require_inside), int(name_id))
    return out


def _pack_step(plan_frame, name_ids: dict[str, int]) -> bytes:
    step = plan_frame.step
    if step is None:
        return struct.pack("<12i", 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0) + struct.pack("<6d", *[0.0] * 6)

    names = list(step.feature_names)
    ref_names = [r.name for r in step.refs]
    feat = [ref_names.index(n) for n in names]
    n_feat = len(names)
    brick_dim = 0 if step.b_star is None else int(np.asarray(step.b_star).shape[0])
    brick_dof = 0 if step.J_brick is None else int(np.asarray(step.J_brick).shape[1])
    j_cols = int(np.asarray(step.J).shape[1])
    angle_mode = 0 if step.angle_mode == "pair" else 1
    angle_slot = ref_names.index(step.angle_slot) if step.angle_slot in ref_names else -1
    has_spin = int(step.spin_gain is not None)
    spin_re, spin_im = 0.0, 0.0
    if step.spin_star is not None:
        if isinstance(step.spin_star, complex):
            spin_re, spin_im = float(step.spin_star.real), float(step.spin_star.imag)
        else:
            spin_re = float(step.spin_star)

    out = struct.pack(
        "<12i", 1, len(step.refs), n_feat, brick_dim, brick_dof, angle_mode, angle_slot,
        int(step.angle_period), int(plan_frame.trusted), j_cols, has_spin,
        int(step.brick_ref is not None),
    )
    out += struct.pack("<6d", float(step.tol), float(plan_frame.gain),
                       float(step.spin_gain if step.spin_gain is not None else 0.0),
                       spin_re, spin_im, 0.0)
    feat_bytes = struct.pack("<%di" % n_feat, *feat) if n_feat else b""
    out += feat_bytes + b"\0" * ((-len(feat_bytes)) % 8)

    # `J`'s rows are already two per entry of `feature_names`, in that order, and so is `s_star`.
    j_rows = np.asarray(step.J, dtype=float)
    if j_rows.shape != (2 * n_feat, j_cols):
        raise ValueError(f"step Jacobian is {j_rows.shape}, expected ({2 * n_feat}, {j_cols})")
    out += np.ascontiguousarray(j_rows, dtype="<f8").tobytes()
    out += np.ascontiguousarray(np.asarray(step.s_star, dtype=float), dtype="<f8").tobytes()
    if brick_dim:
        out += np.ascontiguousarray(np.asarray(step.J_brick, dtype=float), dtype="<f8").tobytes()
        out += np.ascontiguousarray(np.asarray(step.b_star, dtype=float), dtype="<f8").tobytes()
    for ref in step.refs:
        out += _pack_ref(ref, name_ids[ref.name])
    if step.brick_ref is not None:
        out += _pack_ref(step.brick_ref, name_ids[step.brick_ref.name])
    return out


@dataclass(frozen=True)
class BundleInfo:
    bytes: int
    n_frames: int
    n_commands: int
    crc: int


def write_bundle(ideal: IdealSequence, plan: Plan, cfg: PlaceConfig, arm: dict, *,
                 grip_range, jc_ki: float = 0.0, jc_joints: int = 0) -> bytes:
    """The whole bundle, ready to hand to `bb_control_init` or to chunk onto the wire."""
    values = config_values(cfg, plan, grip_range=grip_range, jc_ki=jc_ki, jc_joints=jc_joints)
    body = _Writer()
    body.write(b"\0" * HEADER.size)

    off_cfg = body.align()
    body.write(_pack_cfg(values))
    off_arm = body.align()
    body.write(_pack_arm(arm))

    frames = ideal.frames
    names = name_table(ideal, plan)
    commands: list[bytes] = []
    frame_recs: list[bytes] = []
    cmd_index = 0
    for i, frame in enumerate(frames):
        fp = plan.frames[i]
        title = held_title(ideal, frame)
        if ideal.held_body is None:
            held_name = HELD_NONE
        else:
            held_name = names[title] if title is not None else HELD_UNLISTED
        for command in frame.commands:
            commands.append(struct.pack(
                "<8d", float(command.pos[0]), float(command.pos[1]), float(command.pos[2]),
                float(command.spin), float(command.grip), float(command.c_vert),
                float(command.c_lat), 0.0))
        frame_recs.append(struct.pack(
            "<8i", int(frame.k), int(frame.step_index), int(fp.step_id), len(frame.commands),
            cmd_index, int(fp.held), int(fp.is_freeze_step), held_name)
            + struct.pack("<2d", float(frame.travel), 0.0))
        cmd_index += len(frame.commands)

    off_frames = body.align()
    body.write(b"".join(frame_recs))
    off_commands = body.align()
    body.write(b"".join(commands))
    off_states = body.align()
    body.write(b"".join(pack_state(f.state) for f in frames))

    step_blobs = [_pack_step(fp, names) for fp in plan.frames]
    off_step_off = body.align()
    body.write(b"\0" * (4 * len(step_blobs)))
    offsets = []
    for blob in step_blobs:
        offsets.append(body.align())
        body.write(blob)
    body.align()

    blob = bytearray(body.bytes())
    blob[off_step_off:off_step_off + 4 * len(offsets)] = struct.pack(
        "<%dI" % len(offsets), *offsets)
    total = len(blob)
    header = HEADER.pack(MAGIC, VERSION, total, 0, len(frames), cmd_index, K, W,
                         off_cfg, off_arm, off_frames, off_commands, off_states,
                         off_step_off, offsets[0] if offsets else 0, 0)
    blob[0:HEADER.size] = header
    crc = zlib.crc32(bytes(blob[16:])) & 0xFFFFFFFF
    struct.pack_into("<I", blob, 12, crc)
    return bytes(blob)


def bundle_info(blob: bytes) -> BundleInfo:
    fields = HEADER.unpack_from(blob, 0)
    return BundleInfo(bytes=fields[2], n_frames=fields[4], n_commands=fields[5], crc=fields[3])
