"""The damage monitor: what drives it, what it costs, and the two failures it is for.

The monitor is a :class:`~brick_builder.bricks.damage.DamageModel` attached to the snap model's
observer list, which every stepping loop in this project already calls. That is the whole
mechanism, and the first test here is the one that matters: a scene built by ``build_scene``
accumulates damage readings without any runner, environment or controller knowing it exists.

The last two tests are the failure cases this arm can actually cause - a collision and an
over-squeeze - run through both the monitor and breakable bricks. Their numbers are quoted in
`docs/guide/02d-damage-and-breakable-bricks.md`.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
import pytest

from brick_builder.bricks import damage as dmg
from brick_builder.bricks import materials as mat
from brick_builder.bricks.damage import STATES, DamageMonitor
from brick_builder.scene.compose import HeldBrick, SceneConfig, build_scene


# ---------------------------------------------------------------------------
# what drives it
# ---------------------------------------------------------------------------


def test_a_composed_scene_carries_a_monitor_and_nobody_has_to_drive_it():
    scene = build_scene(SceneConfig())
    assert scene.monitor is not None
    assert scene.monitor in scene.snap.observers
    before = scene.monitor.damage._n_calls
    scene.step(200)
    assert scene.monitor.damage._n_calls > before
    assert scene.monitor.state in STATES
    assert scene.monitor.line().startswith("damage: ")


def test_the_switch_leaves_the_scene_without_one():
    scene = build_scene(SceneConfig(damage_monitor=False))
    assert scene.monitor is None
    assert scene.snap.observers == []


def test_the_rate_subsamples_and_the_guard_still_means_seconds():
    """``every`` changes how often the pass runs, not what a latched state means.

    The persistence timer measures the span between the first and the last call that saw the
    condition, so one sampled tick cannot latch a permanent state however long the tick is.
    """

    scene = build_scene(SceneConfig(damage_monitor_every=10))
    scene.step(200)
    assert scene.monitor.damage._n_calls == pytest.approx(20, abs=2)

    fast = build_scene(SceneConfig(damage_monitor_every=1))
    fast.step(200)
    assert fast.monitor.damage._n_calls == pytest.approx(200, abs=2)


def test_a_reset_clears_the_episode_but_keeps_the_high_water_mark():
    scene = build_scene(SceneConfig())
    scene.monitor.damage._peak_by_mode[:] = 5e6
    seen = scene.monitor.peak_wall_stress_Pa
    assert seen > 0.0
    scene.monitor.reset()
    assert scene.monitor.peak_wall_stress_Pa == 0.0
    assert scene.monitor.worst_wall_stress_Pa == pytest.approx(seen)


def test_the_pass_costs_a_few_per_cent_of_a_step_at_the_default_rate():
    """Measured, not asserted from a docstring. The bound is loose because the machine is not.

    On an unloaded machine the pass is about 22 us a call against a 500 to 700 us step, so 2 %
    at one call per control tick and 17 % at one call per step. The assertion is at 10 % so a
    loaded CI box does not fail it, but a regression that made the pass ten times more expensive
    would still be caught.
    """

    scene = build_scene(SceneConfig())
    scene.snap.observers.clear()
    scene.step(400)
    monitor = scene.monitor
    n, spent = 800, 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        scene.snap.update(scene.model, scene.data)
        a = time.perf_counter()
        monitor.update()
        spent += time.perf_counter() - a
        mujoco.mj_step(scene.model, scene.data)
    total = time.perf_counter() - t0
    fraction = spent / max(total - spent, 1e-9)
    assert fraction < 0.10, f"monitor cost {fraction:.1%} of step time at every={monitor.every}"


# ---------------------------------------------------------------------------
# the permanent-set rung
# ---------------------------------------------------------------------------


def test_the_state_ladder_has_a_permanent_set_rung_between_loosened_and_cracked():
    assert STATES == ("intact", "marked", "loosened", "yielded", "cracked", "broken")


def test_yielded_arrives_before_cracked_as_the_load_rises():
    """Permanent set at the tensile yield, cracking at the flexural strength, in that order."""

    seen = {}
    for force in (20.0, 30.0, 40.0, 60.0, 90.0):
        rig = dmg.build_wall_rig()
        seen[force] = dmg.run_wall_push(rig, force)["state"]
    ranks = [STATES.index(seen[f]) for f in sorted(seen)]
    assert ranks == sorted(ranks), seen
    assert "yielded" in seen.values()
    assert seen[90.0] in ("cracked", "broken")


# ---------------------------------------------------------------------------
# the two failures this arm can cause
# ---------------------------------------------------------------------------


COLLISION_SPEEDS = (0.1, 0.3, 0.6)


def test_a_collision_reads_higher_the_faster_the_tool_is_going():
    """0.1, 0.3 and 0.6 m/s of tool against a wall, with the monitor alone.

    The peak wall stress rises with speed, and no permanent state latches: a rigid-body impact
    at a 2 ms step is resolved in one impulsive step, and the persistence guard is there exactly
    so a one-step spike cannot latch. Impacts are what breakable bricks are for; the monitor's
    job here is the peak and the clutch wear.
    """

    peaks = []
    for v in COLLISION_SPEEDS:
        rig = dmg.build_wall_rig(rod_damping=0.0, rod_mass=0.5)
        out = dmg.run_wall_impact(rig, v)
        peaks.append(out["peak_wall_stress_Pa"])
        assert out["longest_overload_s"] < dmg.DamageThresholds().persist_s
        assert out["state"] in ("intact", "loosened")
    assert peaks[0] < peaks[1] < peaks[2]
    # 0.6 m/s is past the wall's flexural strength on the instantaneous reading.
    assert peaks[2] > mat.ABS_BRICK_23C.flexural_strength.nominal


def test_a_collision_breaks_a_breakable_wall_at_the_speed_the_arithmetic_says():
    """The same three speeds with the pieces welded rather than rigid.

    The weld criterion is instantaneous by design - a piece that has come apart is not a stress
    that might be noise - so this is the path that catches an impact. 0.1 and 0.3 m/s hold,
    0.6 m/s releases a weld and latches the brick to ``broken``.
    """

    results = {}
    for v in COLLISION_SPEEDS:
        rig = dmg.build_wall_rig(rod_damping=0.0, rod_mass=0.5, breakable=True)
        out = dmg.run_wall_impact(rig, v, breakable=rig.breakable)
        out["n_released"] = rig.breakable.n_released
        out["utilisation"] = rig.breakable.summary()["peak_utilisation"]
        results[v] = out

    assert results[0.1]["n_released"] == 0
    assert results[0.3]["n_released"] == 0
    assert results[0.6]["n_released"] >= 1
    assert results[0.6]["state"] == "broken"
    utils = [results[v]["utilisation"] for v in COLLISION_SPEEDS]
    assert utils[0] < utils[1] < utils[2]
    assert utils[2] > 1.0


def _squeeze(breakable: bool, squeeze_m: float = 0.006) -> dict:
    """Close the jaws on a 2x4's walls and report what the monitor saw."""

    cfg = SceneConfig(
        held=HeldBrick(nx=2, ny=4, colour="red"),
        grasp_mode="friction",
        grip_squeeze=squeeze_m,
        breakable=breakable,
        damage_monitor_every=1,
        table=True,
    )
    scene = build_scene(cfg)
    scene.step(1200)
    out = {
        "state": scene.monitor.state,
        "peak_wall_stress_Pa": scene.monitor.peak_wall_stress_Pa,
        "peak_lateral_N": max(
            scene.monitor.damage.brick(n).peak_lateral_N for n in scene.monitor.damage.names
        ),
    }
    if scene.breakable is not None:
        out["n_released"] = scene.breakable.n_released
        out["utilisation"] = scene.breakable.summary()["peak_utilisation"]
    return out


def test_an_over_squeeze_is_reported_and_stays_short_of_breaking_the_brick():
    """The jaws closed hard onto a 2x4's walls, at the servo's own torque limit.

    The stock gripper reaches about 17 N on a brick face. That is an order of magnitude under
    the 942 N a stud needs and three orders under the 13.2 kN a brick needs, so the brick must
    survive - and with breakable pieces no weld may let go. What the monitor reports is the wall
    stress the squeeze produces, which is the number a grip policy should be shaped around.
    """

    rigid = _squeeze(breakable=False)
    assert rigid["state"] == "intact"
    assert rigid["peak_lateral_N"] < 3.0 * mat.ARM_GRIP_FORCE_N
    # The pad is 10 by 5 mm, so 10 N spread over it bends the wall by about 1.4 MPa against a
    # 44 MPa yield. The squeeze is a force the arm has; it is not a stress the brick minds.
    assert rigid["peak_wall_stress_Pa"] < 0.1 * mat.ABS_BRICK_23C.tensile_yield.nominal

    piecewise = _squeeze(breakable=True)
    assert piecewise["state"] == "intact"
    assert piecewise["n_released"] == 0, "the stock gripper broke a brick, which it cannot do"
    assert piecewise["utilisation"] < 0.1
