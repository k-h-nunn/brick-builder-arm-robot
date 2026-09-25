"""Tests for brick_builder/control/skill_fsm.py: sequencing, timeouts and the VERIFY tug.

These need physics -- the whole point of VERIFY is what a real brick does when you pull on it --
so they build one scene and reuse it. The expensive one (the 40-trial discrimination rate the
spec asks for) is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.control.primitives import Feedback, PressConfig
from brick_builder.control.skill_fsm import (
    STATES,
    FsmConfig,
    SkillFSM,
    ground_truth_occupancy,
    run_skill,
    verify_trials,
)
from brick_builder.scene.compose import (
    HeldBrick,
    SceneConfig,
    build_scene,
    reset_scene,
    tool_reach_below_tcp,
)


@pytest.fixture(scope="module")
def scene():
    return build_scene(SceneConfig(held=HeldBrick(nx=2, ny=4), cameras=(), home_hover=0.005))


def seat_and_cell(scene):
    return float(scene.plate_top + tool_reach_below_tcp(scene)), np.array(scene.cfg.plate_xy)


def fb(z: float = 0.05, load: float = 0.0) -> Feedback:
    return Feedback(tcp_pos=np.array([0.17, 0.0, z]), load=np.full(6, load))


# --------------------------------------------------------------------------------- sequencing ---


def test_the_fsm_walks_look_align_press_verify(scene):
    seat_z, cell = seat_and_cell(scene)
    reset_scene(scene, np.random.default_rng(0))
    look = np.array([cell[0] - 0.02, cell[1], seat_z + 0.03])
    fsm = SkillFSM(seat_z, cell, cfg=FsmConfig(look_pos=tuple(look)))
    report = run_skill(scene, fsm, max_ticks=500)
    visited = [state for state, _ in fsm.history]
    assert visited[:4] == ["look", "align", "press", "verify"]
    assert report["state"] in ("done", "failed")
    assert not report["warnings"], report["warnings"]


def test_without_a_look_pose_the_fsm_starts_at_align(scene):
    seat_z, cell = seat_and_cell(scene)
    fsm = SkillFSM(seat_z, cell)
    fsm.reset(fb())
    assert fsm.state == "align"


def test_a_state_that_runs_out_of_time_fails_and_names_itself(scene):
    seat_z, cell = seat_and_cell(scene)
    cfg = FsmConfig(timeouts={"look": 2.0, "align": 0.04, "press": 6.0, "verify": 1.5})
    # A target a metre away cannot be reached inside 40 ms, so align must time out.
    fsm = SkillFSM(seat_z, cell + 1.0, cfg=cfg)
    fsm.reset(fb())
    for _ in range(20):
        fsm.update(fb())
        if fsm.done:
            break
    assert fsm.failed and fsm.timed_out == "align" and fsm.state == "failed"


def test_every_state_it_reports_is_declared(scene):
    seat_z, cell = seat_and_cell(scene)
    reset_scene(scene, np.random.default_rng(0))
    fsm = SkillFSM(seat_z, cell)
    run_skill(scene, fsm, max_ticks=500)
    assert {state for state, _ in fsm.history} <= set(STATES)


# ---------------------------------------------------------------------------- placing a brick ---


def test_run_skill_runs_end_to_end_and_verify_agrees_with_the_depths(scene):
    """The whole skill, end to end, and VERIFY telling the truth about what it found.

    The blind press does not seat a 2x4: the tool settles within about 0.4 mm of its commanded
    pose now that the encoder is modelled on the output shaft, and a 0.25 mm capture window
    leaves no lateral motion to find the sockets with (rates in
    ``docs/guide/02-simulator-and-bricks.md``, "Brick seating with the stock gripper"; closed-loop
    placement is ``brick_builder.action`` and ``tests/action/``). What must hold here is that the
    skill completes cleanly and that VERIFY's height verdict agrees with the stud depths rather
    than reporting a seat that is not there. Depth is read from the snapshot taken *before* the
    tug -- see ``skill_fsm``'s module docstring for why a tug big enough to be informative can
    also undo what it is measuring.
    """
    seat_z, cell = seat_and_cell(scene)
    reset_scene(scene, np.random.default_rng(0))
    fsm = SkillFSM(seat_z, cell)
    report = run_skill(scene, fsm, max_ticks=500)
    depths = report["depths_before_verify"]
    stud_h = scene.cfg.spec.stud_h

    assert depths.size == 8, depths.size
    assert not report["warnings"], report["warnings"]
    assert report["seconds"] < 6.0
    assert report["state"] in STATES, report["state"]

    really_seated = bool(depths.min() >= 0.9 * stud_h)
    assert report["seated_by_height"] is really_seated, (
        f"VERIFY said seated={report['seated_by_height']} at a settled tool height of "
        f"{1e3 * report['verify_height']:.2f} mm, but the stud depths are "
        f"{np.round(depths * 1e3, 2)} mm"
    )
    assert not really_seated, (
        "a blind 2x4 press seated here -- if that is now reliable rather than luck, re-measure "
        "the table in docs/guide/02-simulator-and-bricks.md and re-specify this test"
    )


# ------------------------------------------------------------------------ occupancy cross-check ---


def test_with_no_occupancy_check_given_the_field_stays_none(scene):
    seat_z, cell = seat_and_cell(scene)
    fsm = SkillFSM(seat_z, cell)
    fsm.reset(fb())
    assert fsm.seated_by_occupancy is None


def test_the_occupancy_check_is_read_once_at_the_settled_moment_and_never_overrides_height(scene):
    """The scene-model cross-check the module docstring describes: read once, reported
    alongside the height verdict, never used to change it. A producer's own occupancy call has
    failure modes (dropout, wrong cell) this module cannot audit, so a reading that disagrees
    with the settled height must not flip that verdict -- see the module docstring."""
    seat_z, cell = seat_and_cell(scene)
    reset_scene(scene, np.random.default_rng(0))
    calls = []

    def occupancy_check():
        calls.append(1)
        # Deliberately the opposite of the height verdict this exact scene/seed produces (a
        # blind 2x4 press does not seat here -- see the end-to-end test above).
        return True

    fsm = SkillFSM(seat_z, cell, occupancy_check=occupancy_check)
    report = run_skill(scene, fsm, max_ticks=500)

    assert len(calls) == 1, "the occupancy readout should be taken exactly once, at the settle"
    assert fsm.seated_by_occupancy is True
    assert report["seated_by_occupancy"] is True
    assert fsm.seated_by_height is False, "a disagreeing occupancy reading must not override this"
    assert report["seated_by_height"] is False


def test_ground_truth_occupancy_reads_the_scene_it_is_given(scene):
    """The sim-side placeholder binding, exercised against the real scene rather than a stub."""
    seat_z, cell = seat_and_cell(scene)
    reset_scene(scene, np.random.default_rng(0))
    fsm = SkillFSM(seat_z, cell, occupancy_check=lambda: ground_truth_occupancy(scene))
    report = run_skill(scene, fsm, max_ticks=500)

    depths = report["depths_before_verify"]
    stud_h = scene.cfg.spec.stud_h
    expected = bool(depths.size and float(depths.max()) > 0.1 * stud_h)
    assert report["seated_by_occupancy"] is expected


def test_ground_truth_occupancy_is_none_with_nothing_held():
    """Matches a real occupancy sensor that never got a reading: nothing to ask about, not a
    False that could be mistaken for "confirmed empty"."""

    class _NoHeldBrick:
        held_name = None

    assert ground_truth_occupancy(_NoHeldBrick()) is None


def test_verify_tells_a_seated_brick_from_a_loose_one(scene):
    """A small, fast version of the discrimination check: the two groups have to separate with a
    gap, not merely land on either side of a threshold by luck."""
    data = verify_trials(trials=3, seed=0)
    runs = data["runs"]
    # Split by what actually happened, not by which group the trial was set up as: a badly
    # aligned press occasionally does find a socket, and then "loose" is the wrong label for it.
    seated = np.array([r["verify_height"] for r in runs if r["really_seated"]])
    loose = np.array([r["verify_height"] for r in runs if not r["really_seated"]])
    assert seated.size and loose.size, [r["really_seated"] for r in runs]
    assert np.max(seated) < np.min(loose), (seated, loose)
    assert np.max(seated) < FsmConfig().verify_height_tol < np.min(loose)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spec section 3's acceptance number for VERIFY, applied to the verdict that actually "
        "works on this arm (settled tool height) rather than the load-rise signal it specifies. "
        "Measured on this rig (2026-09-25, seed 7): right 37/40, Wilson [0.801, 0.974] -- short "
        "of 95 %, not by an aggregation artefact but by a real, traced failure mode: near the "
        "capture gate's own edge, a case whose believed cell was close enough on paper to expect "
        "a seat can still come up short (a stud shallow, or the brick missing the cell entirely "
        "with the empty tool coincidentally settling inside tolerance), and there is no second "
        "signal here to catch that -- see the module docstring. This is not loosened to pass: an "
        "occupancy cross-check exists (`SkillFSM.seated_by_occupancy`) but only helps when a "
        "caller has a scene model bound to it, which is not the case in this trial harness."
    ),
)
def test_verify_is_right_in_at_least_ninety_five_percent_of_cases():
    """Spec section 3's acceptance number, over 20 seated and 20 unseated cases, using the
    verdict that works on this arm (the settled tool height). See the xfail reason: it falls
    short, honestly, not by a loosened bound."""
    from brick_builder.eval.wilson import wilson_interval

    data = verify_trials(trials=20, seed=7)
    right = sum(1 for run in data["runs"] if bool(run["seated_by_height"]) == run["really_seated"])
    total = len(data["runs"])
    interval = wilson_interval(right, total)
    assert right / total >= 0.95, (
        f"{right}/{total} correct, Wilson {interval.low:.3f}-{interval.high:.3f}; "
        f"heights {np.round([r['verify_height'] * 1e3 for r in data['runs']], 2)} mm"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spec section 3 defines VERIFY as an upward tug whose *load rise* says whether the brick "
        "is attached. Measured on this arm it does not: Present_Load is the joint position error "
        "scaled, and this arm's own joint friction is enough that a tug this small does not "
        "reliably move the tool whether or not anything is holding it. Over 20 trials at the "
        "4 mm / 0.8 torque-limit tug the load-rise verdict was right 6-8 times out of 20 (seated "
        "rises 0.017-0.053, loose 0.042-0.068 -- overlapping, and the wrong way round). An 8 mm "
        "tug does beat the stiction but pulls a seated 2x4 back off the plate, so it destroys "
        "what it measures. The settled-tool-height verdict in the test above is right 37/40 on "
        "held-out seeds (also short of 95 %, but for a different, traced reason -- see its own "
        "xfail reason) and is what the FSM reports alongside this one."
    ),
)
def test_the_load_rise_verdict_the_spec_specifies_reaches_ninety_five_percent():
    data = verify_trials(trials=20, seed=7)
    right = sum(1 for run in data["runs"] if bool(run["verified"]) == run["really_seated"])
    assert right / len(data["runs"]) >= 0.95, (
        f"{right}/{len(data['runs'])}; "
        f"seated rises {np.round(data['seated'], 3)}, loose {np.round(data['loose'], 3)}"
    )


def test_the_tug_releases_the_press_before_it_measures(scene):
    """Without the release phase the baseline is taken while the servos still carry the press
    error, and the tug then reads that error *relaxing*: measured on this rig, a seated brick
    came out at -0.07 and a loose one at +0.05, i.e. exactly backwards."""
    seat_z, cell = seat_and_cell(scene)
    cfg = FsmConfig()
    fsm = SkillFSM(seat_z, cell, cfg=cfg, press=PressConfig())
    fsm.reset(fb())
    fsm._enter("verify", fb(z=0.02, load=0.6))
    first = fsm.update(fb(z=0.02, load=0.6))
    assert first.c_vert == 1.0, "the release holds at full torque"
    assert first.pos[2] == pytest.approx(0.02), "and at the pose the tool is really at"
    for _ in range(int((cfg.verify_release + cfg.verify_settle) / fsm.dt) + 1):
        target = fsm.update(fb(z=0.02, load=0.1))
    assert target.c_vert == pytest.approx(cfg.verify_torque)
    assert target.pos[2] == pytest.approx(0.02 + cfg.verify_lift)
