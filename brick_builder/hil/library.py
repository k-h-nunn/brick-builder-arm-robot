"""Build `firmware/control_lib` with the system gcc and load it through ctypes.

Same flags as the actor and action exports (`brick_builder/export/build.py`): `-O2 -std=c99
-ffp-contract=off`, so no fused multiply-add changes the rounding and the C build's arithmetic is
the naive multiply-then-add numpy also does. The library is compiled into a cache directory and
rebuilt whenever a source file is newer than it.

The ctypes structures below mirror the C ones field for field. `check_layout` asks the compiled
library for its own `sizeof` of each and compares, rather than trusting that the two agree.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = REPO_ROOT / "firmware" / "control_lib"
SRC_DIR = LIB_DIR / "src"
INC_DIR = LIB_DIR / "include"
RUNNER_SRC = REPO_ROOT / "firmware" / "host_runner" / "main.c"
DEFAULT_BUILD_DIR = Path(os.environ.get("BB_CONTROL_BUILD_DIR", LIB_DIR / "_build"))
"""Where the library and the runner are compiled; `BB_CONTROL_BUILD_DIR` moves it, so a test
run does not rebuild a library another process has loaded."""

CFLAGS = ["-O2", "-std=c99", "-ffp-contract=off", "-Wall", "-Wextra"]

K = 8
W = 14
NJ = 5
NJOINT = 6
MAX_CMDS = 12
MAX_REFS = 8

_D = ctypes.c_double
_I = ctypes.c_int32


class CArm(ctypes.Structure):
    _fields_ = [
        ("body_pos", (_D * 3) * NJ),
        ("body_quat", (_D * 4) * NJ),
        ("axis", (_D * 3) * NJ),
        ("site_pos", _D * 3),
        ("site_quat", _D * 4),
        ("jnt_lo", _D * NJ),
        ("jnt_hi", _D * NJ),
    ]


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


class CCfg(ctypes.Structure):
    _fields_ = ([(n, _D) for n in CFG_DOUBLES] + [(n, _I) for n in CFG_INTS])


class CCommandRec(ctypes.Structure):
    _fields_ = [("pos", _D * 3), ("spin", _D), ("grip", _D), ("c_vert", _D), ("c_lat", _D),
                ("pad", _D)]


class CFrameRec(ctypes.Structure):
    _fields_ = [("k", _I), ("step_index", _I), ("step_id", _I), ("n_commands", _I),
                ("cmd_offset", _I), ("held", _I), ("is_freeze_step", _I), ("held_name", _I),
                ("travel", _D), ("pad", _D)]


class CSlots(ctypes.Structure):
    _fields_ = [("slots", ctypes.c_float * (K * W)), ("depth", ctypes.c_float * K),
                ("ids", _I * K)]


class CRefRec(ctypes.Structure):
    _fields_ = [("colour", _D * 3), ("size", _D * 2), ("colour_tol", _D), ("size_tol", _D),
                ("min_mass", _D), ("max_border", _D), ("has_size", _I), ("has_min_mass", _I),
                ("require_inside", _I), ("name", _I)]


class CStepHdr(ctypes.Structure):
    _fields_ = [("valid", _I), ("n_refs", _I), ("n_feat", _I), ("brick_dim", _I),
                ("brick_dof", _I), ("angle_mode", _I), ("angle_slot", _I), ("angle_period", _I),
                ("trusted", _I), ("j_cols", _I), ("has_spin", _I), ("has_brick_ref", _I),
                ("tol", _D), ("gain", _D), ("spin_gain", _D), ("spin_star_re", _D),
                ("spin_star_im", _D), ("pad", _D)]


FRAME_MAX = 8 + 2048 + 4


class CDecoder(ctypes.Structure):
    """`bb_decoder`, so the C frame decoder can be fed the same bytes as `protocol.Decoder`."""

    _fields_ = [("buf", ctypes.c_uint8 * FRAME_MAX), ("have", ctypes.c_uint32),
                ("consumed", ctypes.c_uint32), ("bad_crc", ctypes.c_uint32),
                ("resyncs", ctypes.c_uint32), ("type", ctypes.c_uint8),
                ("flags", ctypes.c_uint8), ("seq", ctypes.c_uint16), ("len", ctypes.c_uint16),
                ("payload", ctypes.c_void_p)]


class CBusReadings(ctypes.Structure):
    _fields_ = [("position", _D * NJOINT), ("speed", _D * NJOINT), ("load", _D * NJOINT)]


class CCommand(ctypes.Structure):
    _fields_ = [("pos", _D * 3), ("spin", _D), ("grip", _D), ("c_vert", _D), ("c_lat", _D),
                ("q", _D * NJOINT), ("ik_ok", _I)]


class CTickOut(ctypes.Structure):
    _fields_ = [
        ("n_commands", _I), ("cmd", CCommand * MAX_CMDS), ("k", _I), ("step_index", _I),
        ("step_id", _I), ("holds", _I), ("refs_seen", _I), ("vision", _I), ("done", _I),
        ("failure", _I), ("regrasps", _I), ("matched", _I), ("correction", _D * 3),
        ("err_mm", _D), ("tol_mm", _D), ("absolute_mm", _D), ("predict_mm", _D), ("yaw_err", _D),
    ]


#: Struct indices `bb_struct_size` answers to, in its own order.
_SIZE_NAMES = ("bb_cfg", "bb_arm", "bb_frame_rec", "bb_command_rec", "bb_state_rec",
               "bb_ref_rec", "bb_step_hdr", "bb_controller", "bb_tick_out", "bb_command",
               "bb_bus_readings", "bb_slots", "bb_joint_correction", "bb_decoder")

_PY_SIZES = {
    "bb_cfg": CCfg, "bb_arm": CArm, "bb_frame_rec": CFrameRec, "bb_command_rec": CCommandRec,
    "bb_state_rec": CSlots, "bb_ref_rec": CRefRec, "bb_step_hdr": CStepHdr,
    "bb_tick_out": CTickOut, "bb_command": CCommand, "bb_bus_readings": CBusReadings,
    "bb_slots": CSlots, "bb_decoder": CDecoder,
}

FAILURES = {0: None, 1: "stalled: nothing to line up on", 2: "stalled: error not coming down",
            3: "the grasp stopped short twice"}


@dataclass(frozen=True)
class Build:
    library: Path
    runner: Path


def _newer_than(target: Path, sources) -> bool:
    if not target.exists():
        return True
    stamp = target.stat().st_mtime
    return any(Path(s).stat().st_mtime > stamp for s in sources)


def sources() -> list[Path]:
    return sorted(SRC_DIR.glob("*.c"))


def build(build_dir: Path = DEFAULT_BUILD_DIR, *, cc: str = "gcc", runner: bool = True) -> Build:
    """Compile the shared library (and the native runner) if anything has changed."""
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    src = sources()
    headers = sorted(INC_DIR.glob("*.h"))
    lib = build_dir / "control_lib.so"
    exe = build_dir / "control_runner"

    if _newer_than(lib, src + headers):
        _run([cc, *CFLAGS, "-shared", "-fPIC", "-I", str(INC_DIR),
              *[str(p) for p in src], "-o", str(lib), "-lm"])
    if runner and _newer_than(exe, src + headers + [RUNNER_SRC]):
        _run([cc, *CFLAGS, "-I", str(INC_DIR), *[str(p) for p in src], str(RUNNER_SRC),
              "-o", str(exe), "-lm"])
    return Build(library=lib, runner=exe)


def _run(cmd) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed (exit {result.returncode}) on: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


class ControlLib:
    """The compiled controller, called in this process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lib = ctypes.CDLL(str(self.path.resolve()))
        lib = self._lib
        lib.bb_struct_size.argtypes = [ctypes.c_int]
        lib.bb_struct_size.restype = _I
        lib.bb_control_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
                                        ctypes.POINTER(_D)]
        lib.bb_control_init.restype = ctypes.c_int
        lib.bb_tick_begin.argtypes = [ctypes.c_void_p, ctypes.POINTER(CTickOut)]
        lib.bb_tick_begin.restype = ctypes.c_int
        lib.bb_tick_end.argtypes = [ctypes.c_void_p, ctypes.POINTER(CSlots),
                                    ctypes.POINTER(CBusReadings), ctypes.POINTER(CTickOut)]
        lib.bb_tick_end.restype = ctypes.c_int
        lib.bb_crc32.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        lib.bb_crc32.restype = ctypes.c_uint32
        for name in ("bb_fk",):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(CArm), ctypes.POINTER(_D), ctypes.POINTER(_D),
                           ctypes.POINTER(_D)]
            fn.restype = None
        lib.bb_pose_ik.argtypes = [ctypes.POINTER(CArm), ctypes.POINTER(_D), _D, _D,
                                   ctypes.POINTER(_D), ctypes.POINTER(_D)]
        lib.bb_pose_ik.restype = ctypes.c_int
        lib.bb_tool_down_ik.argtypes = lib.bb_pose_ik.argtypes
        lib.bb_tool_down_ik.restype = ctypes.c_int
        lib.bb_jc_init.argtypes = [ctypes.c_void_p, ctypes.c_int, _D, _D, _D, _D, ctypes.c_int]
        lib.bb_jc_init.restype = None
        lib.bb_jc_update.argtypes = [ctypes.c_void_p, ctypes.POINTER(_D), ctypes.POINTER(_D),
                                     ctypes.c_int]
        lib.bb_jc_update.restype = None
        lib.bb_decoder_init.argtypes = [ctypes.POINTER(CDecoder)]
        lib.bb_decoder_init.restype = None
        lib.bb_decoder_feed.argtypes = [ctypes.POINTER(CDecoder), ctypes.c_uint8]
        lib.bb_decoder_feed.restype = ctypes.c_int
        lib.bb_decoder_next.argtypes = [ctypes.POINTER(CDecoder)]
        lib.bb_decoder_next.restype = ctypes.c_int
        self.check_layout()

    def sizeof(self, name: str) -> int:
        return int(self._lib.bb_struct_size(_SIZE_NAMES.index(name)))

    def check_layout(self) -> None:
        """Compare this build's `sizeof` against the ctypes mirror, field group by field group."""
        bad = []
        for name, cls in _PY_SIZES.items():
            want = self.sizeof(name)
            got = ctypes.sizeof(cls)
            if want != got:
                bad.append(f"{name}: C {want} B, Python {got} B")
        if bad:
            raise RuntimeError("control_lib struct layout does not match: " + "; ".join(bad))

    def decode(self, data: bytes) -> tuple[list[tuple[int, int, bytes]], int, int]:
        """Feed `data` to the C decoder a byte at a time, as `bb_session_run` does. Returns the
        frames as (type, seq, payload) and the bad-CRC and resynchronisation counts."""
        dec = CDecoder()
        self._lib.bb_decoder_init(ctypes.byref(dec))
        frames = []
        for byte in data:
            ready = self._lib.bb_decoder_feed(ctypes.byref(dec), byte)
            while ready:
                frames.append((int(dec.type), int(dec.seq),
                               ctypes.string_at(dec.payload, dec.len) if dec.len else b""))
                ready = self._lib.bb_decoder_next(ctypes.byref(dec))
        return frames, int(dec.bad_crc), int(dec.resyncs)

    def crc32(self, data: bytes) -> int:
        return int(self._lib.bb_crc32(data, len(data)))

    def fk(self, arm: CArm, q) -> tuple[np.ndarray, np.ndarray]:
        qc = (_D * NJ)(*[float(v) for v in q])
        pos = (_D * 3)()
        rot = (_D * 9)()
        self._lib.bb_fk(ctypes.byref(arm), qc, pos, rot)
        return np.array(pos), np.array(rot).reshape(3, 3)

    def pose_ik(self, arm: CArm, pos, tilt: float, spin: float, q0):
        pc = (_D * 3)(*[float(v) for v in pos])
        q0c = (_D * NJ)(*[float(v) for v in q0])
        out = (_D * NJ)()
        ok = self._lib.bb_pose_ik(ctypes.byref(arm), pc, float(tilt), float(spin), q0c, out)
        return np.array(out), bool(ok)

    def tool_down_ik(self, arm: CArm, pos, spin: float, q0):
        pc = (_D * 3)(*[float(v) for v in pos])
        q0c = (_D * NJ)(*[float(v) for v in q0])
        out = (_D * NJ)()
        ok = self._lib.bb_tool_down_ik(ctypes.byref(arm), pc, float(spin), q0c, out)
        return np.array(out), bool(ok)


_CACHE: dict[str, ControlLib] = {}


def load(build_dir: Path = DEFAULT_BUILD_DIR, *, cc: str = "gcc") -> tuple[ControlLib, Build]:
    """Build if needed and load, caching one handle per built path in this process."""
    built = build(build_dir, cc=cc)
    key = str(built.library)
    if key not in _CACHE:
        _CACHE[key] = ControlLib(built.library)
    return _CACHE[key], built
