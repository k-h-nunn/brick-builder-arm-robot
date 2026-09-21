"""`DetentSnap`, the simple fallback: the four bench tests it is supposed to pass, and why.

PLAN section 6.3 asks the fallback model to pass tests 2, 3, 5 and 8 - capture failure,
pull-off, stability and save/restore. It is not asked to pass 1, 4, 6 or 7 because those are
all about the shape of the insertion force curve, and a weld does not have one. The last test
in this file measures that gap rather than leaving it as an assertion in a docstring: the same
press that costs 8.5 N against the collet model costs essentially nothing here.
"""

from __future__ import annotations

import numpy as np
import pytest

import mujoco

from brick_builder.bricks.bench import BenchConfig, PressBench, build_stack_scene
from brick_builder.bricks.generator import add_baseplate, add_brick
from brick_builder.bricks.snap import ColletSnap
from brick_builder.bricks.snap_api import SnapModel, SnapParams
from brick_builder.bricks.snap_detent import DetentSnap, _write_weld_pose
from brick_builder.bricks.spec import DUPLO

N_STUDS = 8
SETTLE_STEPS = 5000


def _bench(params: SnapParams | None = None, **kw) -> PressBench:
    return PressBench(DetentSnap(params or SnapParams()), BenchConfig(**kw))


def _drift(snap, offsets, steps=5000, **kw):
    model, data, bricks = build_stack_scene(snap, offsets, **kw)
    adr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{h.name}_free")]
        for h in bricks
    ]
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        snap.update(model, data)
    start = data.qpos.copy()
    for _ in range(steps):
        mujoco.mj_step(model, data)
        snap.update(model, data)
    drift = max(float(np.abs(data.qpos[a : a + 3] - start[a : a + 3]).max()) for a in adr)
    return drift, int(data.warning.number.sum()), snap.connections()


# ---------------------------------------------------------------------------
# The one MuJoCo fact this model depends on
# ---------------------------------------------------------------------------


def test_weld_activation_does_not_move_the_brick():
    """`_write_weld_pose` must freeze two bodies exactly where they are, with no jump.

    This is the check that the `eq_data` layout for a weld really is
    `[anchor in body2, the same point in body1, body2's orientation relative to body1,
    torquescale]` in the installed MuJoCo, rather than something the docs were read to say.
    A wrong layout would teleport the brick the moment it seated.
    """

    world = mujoco.MjSpec()
    world.option.gravity = [0.0, 0.0, 0.0]
    plate = add_baseplate(world, "plate", 4, 4, pos=(0.0, 0.0, 0.0))
    brick = add_brick(world, "b", 2, 2, "red", pos=(0.013, -0.007, 0.05),
                      quat=(0.966, 0.0, 0.0, 0.259))
    eq = world.add_equality()
    eq.name = "w"
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = "b"
    eq.name2 = "plate"
    eq.active = False
    eq.data = np.zeros(mujoco.mjNEQDATA)
    eq.data[6] = 1.0
    eq.data[10] = 1.0
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    before_pos = data.xpos[bid].copy()
    before_quat = data.xquat[bid].copy()
    _write_weld_pose(model, data, 0, bid, pid)
    data.eq_active[0] = 1
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.allclose(data.xpos[bid], before_pos, atol=1e-6), data.xpos[bid] - before_pos
    assert np.allclose(data.xquat[bid], before_quat, atol=1e-6)
    assert int(data.warning.number.sum()) == 0
    assert plate.socket_sites == []      # a plate has studs only, by construction


def test_detent_snap_satisfies_the_protocol():
    assert isinstance(DetentSnap(), SnapModel)
    assert isinstance(ColletSnap(), SnapModel)


# ---------------------------------------------------------------------------
# 2. Capture failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, label",
    [
        ({"offset": (0.003, 0.0)}, "3 mm sideways"),
        ({"yaw_deg": 10.0}, "10 degrees of yaw"),
    ],
)
def test_a_misaligned_brick_never_seats_even_at_twice_the_force(kw, label):
    """Test 2 for the fallback: 3 mm of offset or 10 degrees of yaw never weld."""

    bench = _bench(**kw)
    trace = bench.press(travel=0.003, until_seated=False)
    nominal = N_STUDS * SnapParams().f_clutch * (1.0 + SnapParams().beta)
    assert trace.peak_force > 2.0 * nominal, f"{label}: only reached {trace.peak_force:.2f} N"
    assert bench.snap.connections() == [], label
    assert not bench.seated(), label
    assert float(np.max(trace.depth)) < 0.3 * DUPLO.stud_h, label


# ---------------------------------------------------------------------------
# 3. Pull-off
# ---------------------------------------------------------------------------


def test_pull_off_releases_at_the_configured_force_and_clears_every_weld():
    """Test 3 for the fallback: the weld breaks at `n * f_c * (1 - beta)` within 10 %."""

    params = SnapParams()
    bench = _bench(params)
    bench.press()
    assert bench.seated()
    conns = bench.snap.connections()
    assert len(conns) == 1 and conns[0].n_engaged == N_STUDS and conns[0].seated
    bench.hold(0.2)
    trace = bench.pull(speed=0.002, travel=0.008)
    expected = N_STUDS * params.f_clutch * (1.0 - params.beta)
    rel = abs(trace.peak_pull - expected) / expected
    assert trace.warnings == 0
    assert rel < 0.10, f"peak pull {trace.peak_pull:.3f} N against {expected:.3f} N"
    assert bench.snap.connections() == []
    assert not np.any(bench.data.eq_active[bench.snap._eq_ids])


def test_a_broken_weld_does_not_immediately_reform():
    """Once it lets go it stays let go until the brick has come back out of the socket."""

    bench = _bench()
    bench.press()
    bench.hold(0.2)
    bench.pull(speed=0.002, travel=0.004)
    assert bench.snap.connections() == []
    bench.hold(0.5)                      # still up near the studs, but out of them
    assert bench.snap.connections() == []


# ---------------------------------------------------------------------------
# 5. Stability
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_three_layer_overhanging_stack_barely_moves_in_ten_seconds():
    """Test 5a for the fallback: a welded staircase is rigid, so it should hardly move at all."""

    drift, warnings, conns = _drift(DetentSnap(SnapParams()), [(0, 0), (0, 1), (0, 2)])
    assert warnings == 0
    assert len(conns) == 3
    assert drift < 1e-4, f"drift {drift * 1e3:.4f} mm"


@pytest.mark.slow
def test_a_brick_hanging_upside_down_creeps_less_than_a_third_of_a_millimetre():
    """Test 5b for the fallback: an inverted brick hangs from its weld without creeping."""

    drift, warnings, conns = _drift(DetentSnap(SnapParams()), [(0, 0)], inverted=True)
    assert warnings == 0
    assert conns and conns[0].n_engaged == N_STUDS
    assert drift < 3e-4, f"creep {drift * 1e3:.4f} mm"


# ---------------------------------------------------------------------------
# 8. Save and restore
# ---------------------------------------------------------------------------


def test_save_and_restore_reproduce_the_trajectory_exactly():
    """Test 8 for the fallback: `mjSTATE_INTEGRATION` plus `SnapState` replays bit for bit."""

    bench = _bench()
    bench.press()
    bench.hold(0.05)
    model, data, snap = bench.model, bench.data, bench.snap
    assert snap.connections()

    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    saved = np.empty(size)
    mujoco.mj_getState(model, data, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
    snap_state = snap.get_state().copy()
    start_ctrl = float(data.ctrl[0])

    def replay(steps=500):
        out = []
        t0 = float(data.time)
        for _ in range(steps):
            data.ctrl[0] = start_ctrl + 0.002 * (data.time - t0)   # pull it back off again
            mujoco.mj_step(model, data)
            snap.update(model, data)
            out.append(data.qpos.copy())
        return np.array(out)

    first = replay()
    assert snap.connections() == []          # the replay really does break the weld
    mujoco.mj_setState(model, data, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
    snap.set_state(snap_state)
    second = replay()
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# The funnel, and what the fallback cannot do
# ---------------------------------------------------------------------------


def test_the_funnel_pulls_a_free_brick_into_line():
    """A brick dropped slightly off the grid is drawn onto it, instead of landing askew."""

    params = SnapParams()
    snap = DetentSnap(params)
    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    plate = add_baseplate(world, "plate", 6, 6, pos=(0.0, 0.0, 0.0))
    offset = 0.0012                        # 1.2 mm off, inside `funnel_lateral`
    brick = add_brick(
        world, "b", 2, 4, "red", pos=(offset, 0.0, DUPLO.roof * 2.0 + DUPLO.stud_h * 0.7)
    )
    snap.declare(world, [brick], [plate])
    model = world.compile()
    data = mujoco.MjData(model)
    snap.bind(model, data)
    mujoco.mj_forward(model, data)
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "b_free")]

    for _ in range(600):
        data.qfrc_applied[:] = 0.0
        snap.update(model, data)
        mujoco.mj_step(model, data)
    final = abs(float(data.qpos[adr]))
    assert final < offset, f"the funnel pushed it further out: {final * 1e3:.3f} mm"
    assert int(data.warning.number.sum()) == 0


def test_the_fallback_has_no_force_against_depth_at_all():
    """The measurement that makes the collet model worth its complexity.

    Pressing the same brick onto the same studs costs 8.5 N against the collet model and
    essentially nothing against the weld model, because a weld only exists once the brick is
    already home. A press policy trained against this model would have no force to learn to
    apply, which is exactly why `snap.py` exists.
    """

    params = SnapParams()
    detent = _bench(params)
    detent_trace = detent.press()
    collet_trace = PressBench(ColletSnap(params), BenchConfig()).press()

    assert detent.seated() and detent.snap.connections()
    assert collet_trace.peak_force > 8.0
    assert detent_trace.peak_force < 0.1 * collet_trace.peak_force, (
        f"detent peak {detent_trace.peak_force:.3f} N, collet peak {collet_trace.peak_force:.3f} N"
    )
