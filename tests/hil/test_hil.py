"""The fast half of the hardware-in-the-loop checks: layout, framing, and C-against-Python.

Nothing here renders or steps physics. The slow end-to-end comparison lives in
`brick_builder.hil.equivalence`, which is run by hand and whose numbers are quoted in
`docs/guide/13-hardware-in-the-loop.md`.
"""

from __future__ import annotations

import math
import shutil
import struct
import zlib

import numpy as np
import pytest

from brick_builder.hil import protocol as P
from brick_builder.hil.library import CArm, load

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="no gcc on this machine")

IK_BOUND = 1e-6
"""Radians the C inverse kinematics may differ from the Python one on the same pose.

Measured: 1.9e-9 rad over a thousand poses. The gap starts as the last bits of the site Jacobian
(2.6e-15) and is amplified by the damped-least-squares solve, whose three same-axis joints make
it ill-conditioned.
"""


@pytest.fixture(scope="module")
def lib():
    handle, _ = load()
    return handle


@pytest.fixture(scope="module")
def arm(lib):
    """The arm-only kinematic chain, read off the same model `ArmKinematics` evaluates on."""
    import mujoco

    from brick_builder.hil.plan import arm_constants
    from brick_builder.robot.kinematics import ArmKinematics, _arm_template

    kin = ArmKinematics(_arm_template())
    values = arm_constants(kin)
    out = CArm()
    for i in range(5):
        for j in range(3):
            out.body_pos[i][j] = values["body_pos"][i][j]
            out.axis[i][j] = values["axis"][i][j]
        for j in range(4):
            out.body_quat[i][j] = values["body_quat"][i][j]
    for j in range(3):
        out.site_pos[j] = values["site_pos"][j]
    for j in range(4):
        out.site_quat[j] = values["site_quat"][j]
    for i in range(5):
        out.jnt_lo[i] = values["jnt_lo"][i]
        out.jnt_hi[i] = values["jnt_hi"][i]
    return out, kin


# ----------------------------------------------------------------------------- the library ---


def test_the_compiled_structs_match_the_ones_the_host_packs(lib):
    lib.check_layout()


def test_the_two_crc_implementations_agree(lib):
    rng = np.random.default_rng(0)
    for n in (0, 1, 7, 64, 1000):
        data = bytes(rng.integers(0, 256, n, dtype=np.uint8))
        assert lib.crc32(data) == (zlib.crc32(data) & 0xFFFFFFFF)


# ------------------------------------------------------------------------------ kinematics ---


def test_forward_kinematics_matches_mujoco(lib, arm):
    chain, kin = arm
    rng = np.random.default_rng(1)
    lo, hi = kin.jnt_range[:, 0], kin.jnt_range[:, 1]
    worst_pos = worst_rot = 0.0
    for _ in range(200):
        q = lo + rng.random(5) * (hi - lo)
        pos, rot = lib.fk(chain, q)
        want_pos, want_rot = kin.fk(q)
        worst_pos = max(worst_pos, float(np.max(np.abs(pos - want_pos))))
        worst_rot = max(worst_rot, float(np.max(np.abs(rot - want_rot))))
    assert worst_pos < 1e-12, f"tool position differs by {worst_pos:.3e} m"
    assert worst_rot < 1e-12, f"tool rotation differs by {worst_rot:.3e}"


def test_inverse_kinematics_parity_on_a_thousand_poses(lib, arm):
    """The same warm-started solve `ArmDriver` runs, C against Python, over the workspace."""
    from brick_builder.envs.base_env import _pose_ik

    chain, kin = arm
    rng = np.random.default_rng(7)
    q_py = np.array([0.0, -0.5, 0.9, 1.2, 0.0])
    q_c = q_py.copy()
    worst = 0.0
    disagreements = 0
    for _ in range(1000):
        r = rng.uniform(0.13, 0.22)
        bearing = rng.uniform(-0.6, 0.6)
        pos = np.array([r * math.cos(bearing), r * math.sin(bearing), rng.uniform(0.03, 0.08)])
        spin = rng.uniform(-0.3, 0.3)
        want, ok_py = _pose_ik(kin, pos, 0.0, spin, q_py)
        got, ok_c = lib.pose_ik(chain, pos, 0.0, spin, q_c)
        worst = max(worst, float(np.max(np.abs(got - want))))
        disagreements += int(ok_py != ok_c)
        if ok_py:
            q_py = want
        if ok_c:
            q_c = got
    assert disagreements == 0
    assert worst < IK_BOUND, f"joint angles differ by {worst:.3e} rad"


def test_the_joint_trim_matches_the_python_integrator(lib):
    import ctypes

    from brick_builder.robot.joint_correction import JointCorrection, JointCorrectionConfig

    cfg = JointCorrectionConfig()
    python = JointCorrection(5, cfg)
    state = ctypes.create_string_buffer(lib.sizeof("bb_joint_correction"))
    lib._lib.bb_jc_init(ctypes.cast(state, ctypes.c_void_p), 5, cfg.ki, cfg.dt,
                        cfg.clamp_ticks, cfg.deadzone_ticks, cfg.ticks_per_rev)
    rng = np.random.default_rng(3)
    target = (ctypes.c_double * 5)()
    for tick in range(400):
        desired = rng.normal(0.0, 0.6, 5)
        measured = desired + rng.normal(0.0, 0.02, 5)
        freeze = tick % 37 == 0
        want = python.update(desired, measured, freeze=freeze).copy()
        d = (ctypes.c_double * 5)(*desired)
        m = (ctypes.c_double * 5)(*measured)
        lib._lib.bb_jc_update(ctypes.cast(state, ctypes.c_void_p), d, m, int(freeze))
        ctypes.memmove(target, ctypes.byref(ctypes.c_double.from_buffer(
            state, lib.sizeof("bb_joint_correction") - 5 * 8)), 5 * 8)
        got = np.array(target)
        assert np.max(np.abs(got - want)) == 0.0, f"tick {tick}: {got} vs {want}"


# -------------------------------------------------------------------------------- protocol ---


def test_a_frame_round_trips():
    payload = bytes(range(64))
    decoder = P.Decoder()
    frames = decoder.feed(P.encode(P.MSG_TICK_LOG, payload, seq=7))
    assert len(frames) == 1
    assert frames[0].type == P.MSG_TICK_LOG
    assert frames[0].seq == 7
    assert frames[0].payload == payload


def test_frames_split_across_reads_still_decode():
    data = P.encode(P.MSG_STATUS, b"abc") + P.encode(P.MSG_HELLO, b"de")
    decoder = P.Decoder()
    got = []
    for i in range(len(data)):
        got.extend(decoder.feed(data[i:i + 1]))
    assert [f.type for f in got] == [P.MSG_STATUS, P.MSG_HELLO]


def test_a_corrupted_payload_fails_the_crc_and_the_next_frame_still_arrives():
    bad = bytearray(P.encode(P.MSG_STATUS, b"\0" * 8))
    bad[12] ^= 0xFF
    decoder = P.Decoder()
    assert decoder.feed(bytes(bad)) == []
    assert decoder.bad_crc == 1
    got = decoder.feed(P.encode(P.MSG_HELLO, b"ok"))
    assert [f.type for f in got] == [P.MSG_HELLO]


def test_a_truncated_frame_is_dropped_and_the_stream_recovers():
    whole = P.encode(P.MSG_TARGETS, b"\1" * 32)
    decoder = P.Decoder()
    assert decoder.feed(whole[:-3]) == []
    # The cut frame's own bytes are still in the buffer, so the next frame completes its length
    # and fails its CRC. The decoder reports that and resynchronises rather than delivering a
    # frame made of two halves.
    got = decoder.feed(P.encode(P.MSG_HELLO, b"ok"))
    assert decoder.bad_crc >= 1
    assert [f.type for f in got] == [P.MSG_HELLO]
    assert all(f.type != P.MSG_TARGETS for f in got)


def test_sync_bytes_inside_a_payload_do_not_split_a_frame():
    decoder = P.Decoder()
    got = decoder.feed(P.encode(P.MSG_TICK_LOG, P.SYNC * 8))
    assert len(got) == 1 and got[0].payload == P.SYNC * 8


def test_the_bundle_chunks_cover_the_blob_exactly():
    blob = bytes(range(256)) * 40
    seen = bytearray(len(blob))
    for chunk in P.bundle_chunks(blob, chunk=1000):
        total, off = struct.unpack_from("<II", chunk, 0)
        assert total == len(blob)
        seen[off:off + len(chunk) - 8] = chunk[8:]
    assert bytes(seen) == blob


def test_the_targets_payload_round_trips():
    body = struct.pack("<4i", 1, 5, 2, 3)
    body += struct.pack("<13d", 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.0, *[0.7] * 6)
    body += struct.pack("<2i", 1, 0)
    parsed = P.parse_targets(body)
    assert parsed.k == 5 and len(parsed.commands) == 1
    assert parsed.commands[0].ik_ok is True
    assert parsed.commands[0].pos.tolist() == [0.1, 0.2, 0.3]


def test_the_tick_log_payload_round_trips():
    body = struct.pack("<10i", 4, 1, 2, 0, 3, 0, 0, 0, 4, 1)
    body += struct.pack("<8d", 0.001, 0.002, 0.003, 1.5, 0.4, 2.0, 0.25, -0.01)
    log = P.parse_tick_log(body)
    assert log.k == 4 and log.refs_seen == 3 and log.ik_failures == 1
    assert log.err_mm == pytest.approx(1.5)


def test_the_c_and_python_decoders_agree_on_a_damaged_stream(lib):
    """Decoder parity on flipped bytes, cut frames, garbage and sync pairs inside payloads."""
    rng = np.random.default_rng(11)
    stream = bytearray()
    for i in range(300):
        payload = bytes(rng.integers(0, 256, int(rng.integers(0, 120)), dtype=np.uint8))
        if i % 17 == 0:
            payload = P.SYNC * 3 + payload
        frame = bytearray(P.encode(int(rng.integers(1, 13)), payload, seq=i))
        roll = rng.random()
        if roll < 0.1:
            frame[int(rng.integers(2, len(frame)))] ^= 0xFF
        elif roll < 0.18:
            frame = frame[:int(rng.integers(1, len(frame)))]
        elif roll < 0.24:
            stream.extend(rng.integers(0, 256, int(rng.integers(1, 9)), dtype=np.uint8).tobytes())
        stream.extend(frame)
    python = P.Decoder()
    want = [(f.type, f.seq, f.payload) for f in python.feed(bytes(stream))]
    got, bad_crc, _ = lib.decode(bytes(stream))
    assert got == want
    assert bad_crc == python.bad_crc
    assert len(want) > 200


# ------------------------------------------------------ the native process (second rung) ---

N_FRAMES = 24
J_SYNTH = np.array([[9.0, 0.5, 0.0], [0.3, 8.0, 0.0], [9.5, -0.4, 0.0], [-0.2, 8.5, 0.0]])
"""Feature change per metre of tool motion, two rows per blob."""


def _synthetic_bundle():
    """Synthetic target bundle.

    No recording behind it: two coloured blobs to line up on, a step boundary where the
    correction freezes, and five 50 Hz writes per frame."""
    from types import SimpleNamespace

    from brick_builder.action.place import PlaceConfig
    from brick_builder.action.sequence import Command
    from brick_builder.action.servo import Reference
    from brick_builder.hil import bundle as B
    from brick_builder.hil.plan import FramePlan, Plan, arm_constants
    from brick_builder.robot.kinematics import ArmKinematics, _arm_template

    yellow, red = (0.95, 0.8, 0.1), (0.8, 0.1, 0.1)
    J = J_SYNTH
    step = SimpleNamespace(
        feature_names=("yellow", "red"),
        refs=[Reference("yellow", colour=yellow), Reference("red", colour=red)],
        b_star=None, J_brick=None, J=J, s_star=np.zeros(4), angle_mode="pair",
        angle_slot=None, angle_period=1, spin_gain=0.8, spin_star=0.0, tol=0.0004,
        brick_ref=None)
    frames, plans = [], []
    for k in range(N_FRAMES):
        index = 0 if k < 16 else 1
        title = "Align" if index == 0 else "Release"
        x = 0.17 + 0.0004 * k
        commands = [Command(pos=np.array([x + 0.00008 * i, 0.01, 0.05]), spin=0.02,
                            grip=0.3, c_vert=1.0, c_lat=1.0) for i in range(5)]
        slots = np.zeros((K_SLOTS, W_SLOTS), dtype=np.float32)
        for row, (colour, cx) in enumerate(((yellow, -0.2), (red, 0.25))):
            slots[row, 0] = 0.02
            slots[row, 1:3] = (cx - 0.004 * k, 0.1 + 0.002 * k)
            slots[row, 3:6] = (0.002, 0.0, 0.001)
            slots[row, 6:9] = [2.0 * c - 1.0 for c in colour]
        depth = np.full(K_SLOTS, 0.5, dtype=np.float32)
        depth[:2] = (0.1, 0.2)
        frames.append(SimpleNamespace(
            k=k, step=title, step_index=index, commands=commands, travel=0.0004,
            state=SimpleNamespace(slots=slots, depth=depth,
                                  slot_ids=np.arange(K_SLOTS, dtype=np.int32)),
            bodies={"yellow": "plate_a", "red": "plate_b"}))
        plans.append(FramePlan(step=step, trusted=True, gain=8.0, held=False,
                               is_freeze_step=title == "Align",
                               step_id=("Align", "Release").index(title) * 2 + 4))
    ideal = SimpleNamespace(frames=frames, held_body=None)
    plan = Plan(frames=plans, descent_k=-1, grasp_k=-1, grasp_z=0.0, freeze_index=0)
    arm = arm_constants(ArmKinematics(_arm_template()))
    return B.write_bundle(ideal, plan, PlaceConfig(), arm, grip_range=(-0.2, 1.7)), frames


K_SLOTS, W_SLOTS = 8, 14
Q_HOME = np.array([0.05, -0.45, 0.85, 1.25, 0.02])


@pytest.fixture(scope="module")
def synthetic():
    return _synthetic_bundle()


def _drive(controller, blob, frames, *, ticks: int = 60, damage=None):
    """Lock-step run against a plant with a fixed offset.

    The correction has to take the offset out. Returns every number the controller sent back,
    in order."""
    from types import SimpleNamespace

    offset = np.array([0.002, -0.0015])
    controller.start(blob, Q_HOME)
    out = []
    for tick in range(ticks):
        if damage is not None and tick == damage[0]:
            damage[1](controller)
        targets = controller.tick_begin()
        if not targets.commands:
            break
        frame = frames[targets.k]
        slots = frame.state.slots.copy()
        # Where the tool really is against where the recording had it: the plate's offset plus
        # the correction the controller added, seen through the step's own Jacobian.
        moved = offset + (targets.commands[-1].pos[:2] - frame.commands[-1].pos[:2])
        for row in range(2):
            slots[row, 1:3] += (J_SYNTH[2 * row:2 * row + 2, :2] @ moved).astype(np.float32)
        state = SimpleNamespace(slots=slots, depth=frame.state.depth,
                                slot_ids=frame.state.slot_ids, t=0.1 * tick, frame_id=tick)
        q = targets.commands[-1].q
        bus = (q, np.full(6, 0.001 * (tick % 3)), np.zeros(6))
        log = controller.tick_end(state, bus)
        out.append((targets.k, [c.q.tobytes() + c.pos.tobytes() for c in targets.commands],
                    log.k, log.holds, log.matched, np.float64(log.err_mm).tobytes(),
                    log.correction.tobytes(), log.done, log.failure))
        if log.done:
            break
    return out


def _process(kind: str, timeout: float = 10.0):
    from brick_builder.hil.controller import ProcessController
    from brick_builder.hil.library import build

    runner = build().runner
    if kind == "tcp":
        return ProcessController(runner, tcp_port=58431, timeout=timeout)
    return ProcessController(runner, timeout=timeout)


def test_the_synthetic_run_moves_the_index_and_the_correction(lib, synthetic):
    from brick_builder.hil.controller import SilController

    blob, frames = synthetic
    got = _drive(SilController(lib), blob, frames)
    ks = [row[2] for row in got]
    assert max(ks) >= 16, "the index never crossed the step boundary"
    assert any(np.frombuffer(row[6])[0] != 0.0 for row in got), "the correction never moved"


@pytest.mark.parametrize("kind", ["pipe", "tcp"])
def test_the_native_process_is_bit_identical_to_the_library_in_process(lib, synthetic, kind):
    from brick_builder.hil.controller import SilController

    blob, frames = synthetic
    want = _drive(SilController(lib), blob, frames)
    runner = _process(kind)
    try:
        got = _drive(runner, blob, frames)
        status = runner.status()
    finally:
        runner.close()
    assert got == want
    assert status.bad_crc == 0 and status.ticks == len(want)


@pytest.mark.parametrize("kind", ["crc", "drop"])
def test_a_damaged_frame_between_ticks_is_dropped_and_the_run_carries_on(lib, synthetic, kind):
    """Stray damaged frame: dropped, and every later tick bit-identical to the clean run."""
    from brick_builder.hil.controller import SilController

    blob, frames = synthetic
    want = _drive(SilController(lib), blob, frames)
    runner = _process("pipe")
    try:
        got = _drive(runner, blob, frames, damage=(7, lambda c: c.inject_corruption(kind)))
        status = runner.status()
    finally:
        runner.close()
    assert got == want
    assert status.bad_crc + status.resyncs >= 1


@pytest.mark.parametrize("kind", ["crc", "lose"])
def test_a_damaged_request_fails_loudly(lib, synthetic, kind):
    """Damaged or lost request: not acted on, and the host raises instead of carrying on."""
    from brick_builder.hil import protocol as proto
    from brick_builder.hil.controller import ControllerError

    blob, frames = synthetic
    runner = _process("pipe", timeout=1.0)
    try:
        with pytest.raises(ControllerError, match="timed out waiting for TICK_LOG"):
            _drive(runner, blob, frames,
                   damage=(5, lambda c: c.damage_next(proto.MSG_SCENE_STATE, kind)))
    finally:
        runner.close()


def test_the_runner_greets_a_host_that_asks(lib):
    runner = _process("pipe", timeout=5.0)
    try:
        runner._await(P.MSG_HELLO)
        runner._write(P.MSG_HELLO)
        frame = runner._await(P.MSG_HELLO)
        version, capacity = struct.unpack_from("<II", frame.payload, 0)
    finally:
        runner.close()
    assert version == 2 and capacity >= 1 << 20


def test_a_damaged_bundle_is_refused(lib, synthetic):
    import ctypes

    blob, _ = synthetic
    bad = bytearray(blob)
    bad[len(bad) // 2] ^= 0x01
    ctrl = ctypes.create_string_buffer(lib.sizeof("bb_controller"))
    q = (ctypes.c_double * 5)(*Q_HOME)
    good_rc = lib._lib.bb_control_init(ctypes.cast(ctrl, ctypes.c_void_p),
                                       ctypes.create_string_buffer(blob, len(blob)), len(blob), q)
    bad_rc = lib._lib.bb_control_init(ctypes.cast(ctrl, ctypes.c_void_p),
                                      ctypes.create_string_buffer(bytes(bad), len(bad)),
                                      len(bad), q)
    assert good_rc == 0
    assert bad_rc == -3  # BB_ERR_CRC
