"""Tests for brick_builder/control/primitives.py.

Most of these need no simulator at all: a primitive is a state machine over
:class:`~brick_builder.control.primitives.Feedback`, so the way to test what it does under a stall, a
find or a saturated load is to hand it exactly that feedback and read the target it produces.
The tests that do need physics live in ``test_skill_fsm.py`` and ``tests/envs/test_press_env.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from brick_builder.control import primitives
from brick_builder.control.primitives import (
    CONTROL_DT,
    PHASES,
    Feedback,
    MoveTo,
    PressConfig,
    PressScripted,
)

SOURCE = Path(primitives.__file__).read_text()


def fb(z: float, x: float = 0.0, y: float = 0.0, load: float = 0.0) -> Feedback:
    return Feedback(tcp_pos=np.array([x, y, z]), load=np.full(6, load))


# ------------------------------------------------------------------------------- deployability ---


def test_the_control_package_never_touches_the_simulator():
    """Section 3's binding requirement: a scripted skill runs on deployable signals only. The
    cheapest honest check is that the module cannot even reach simulator state -- it does not
    import mujoco, the scene or the environments, so there is nothing for it to peek at."""
    for forbidden in ("import mujoco", "from brick_builder.scene", "from brick_builder.envs", "brick_builder.bricks"):
        top = SOURCE.split("# ----------------------------------------------------------------------------------- figures ---")[0]
        assert forbidden not in top, f"{forbidden!r} appears in the per-tick half of primitives.py"


def test_feedback_only_carries_signals_a_real_servo_bus_gives():
    fields = set(Feedback.__dataclass_fields__)
    assert fields == {"tcp_pos", "tcp_tilt", "tcp_spin", "load", "grip", "tripped"}


def test_feedback_from_readings_is_forward_kinematics_of_the_read_positions():
    import mujoco

    from brick_builder.robot.kinematics import ArmKinematics
    from brick_builder.robot.patch_so101 import build_arm_spec
    from brick_builder.robot.servo_model import ServoReadings

    model = build_arm_spec().compile()
    kin = ArmKinematics(model)
    q = np.array([0.1, -0.4, 0.8, 0.9, 0.2])
    readings = ServoReadings(
        position=np.concatenate([q, [0.3]]), speed=np.zeros(6), load=np.full(6, 0.2),
        current=np.zeros(6), tripped=np.zeros(6, dtype=bool),
    )
    out = primitives.feedback_from_readings(kin, readings, grip=0.3)
    pos, _ = kin.fk(q)
    assert np.allclose(out.tcp_pos, pos)
    assert out.peak_load == pytest.approx(0.2)
    assert mujoco is not None  # the import above is what compiled the model


# -------------------------------------------------------------------------------------- MoveTo ---


def test_moveto_walks_a_straight_line_at_the_commanded_speed():
    move = MoveTo([0.1, 0.0, 0.0], speed=0.05)
    move.reset(fb(0.0))
    targets = [move.update(fb(0.0)).pos.copy() for _ in range(10)]
    for k, pos in enumerate(targets):
        assert pos[0] == pytest.approx(0.05 * k * CONTROL_DT)
        assert pos[1] == 0.0 and pos[2] == 0.0


def test_moveto_finishes_once_the_measured_tool_is_inside_the_tolerance():
    move = MoveTo([0.01, 0.0, 0.0], speed=0.05, tol=0.001, settle_s=10.0)
    move.reset(fb(0.0))
    for _ in range(20):
        move.update(fb(0.0))          # tool never moves: cannot finish on tolerance
    assert not move.done
    move.update(Feedback(tcp_pos=np.array([0.0102, 0.0, 0.0]), load=np.zeros(6)))
    assert move.done


def test_moveto_gives_up_waiting_after_the_settle_time():
    move = MoveTo([0.01, 0.0, 0.0], speed=0.05, tol=1e-9, settle_s=0.1)
    move.reset(fb(0.0))
    for _ in range(int(0.2 / CONTROL_DT) + int(0.1 / CONTROL_DT) + 2):
        move.update(fb(0.0))
    assert move.done, "a soft servo never lands exactly; the settle timeout has to end the move"


# ------------------------------------------------------------------------------- PressScripted ---


def test_the_descent_ramp_is_ten_millimetres_a_second_and_stops_at_the_press_depth():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.02))
    z = []
    for _ in range(300):
        z.append(float(press.update(fb(0.02)).pos[2]))   # tool never follows: no seat, no stall
    assert z[0] == pytest.approx(0.02)
    assert z[5] == pytest.approx(0.02 - cfg.descend_speed * 5 * CONTROL_DT)
    assert min(z) == pytest.approx(-cfg.press_depth)


def test_a_stall_runs_a_check_that_stops_pushing_before_it_judges():
    """The whole point of the ``check`` phase: a press can only tell whether the brick went in
    once it has let go. While pushing, this arm's own 2 mm of deflection under 10 N sits on top
    of the reading and the two cases are indistinguishable (measured: jammed and seated presses
    both read between -1.5 and +3.2 mm of the seated height)."""
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.02))
    for _ in range(400):
        target = press.update(fb(0.008, load=0.5))     # stuck at 8 mm, under load
        if press.phase == "check":
            break
    assert press.phase == "check"
    assert target is not None
    held = press.update(fb(0.008, load=0.5))
    assert held.pos[2] == pytest.approx(0.0), "the check commands the seated height, not below it"


def test_the_check_calls_it_seated_when_the_released_tool_stands_low_enough():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.02))
    press._enter("check")
    for _ in range(int(cfg.check_time / CONTROL_DT) + 2):
        press.update(fb(cfg.check_height - 0.0005))
    assert press.phase == "seated" and press.seated and press.done


def test_the_check_starts_a_search_when_the_released_tool_still_stands_high():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.02))
    press._enter("check")
    for _ in range(int(cfg.check_time / CONTROL_DT) + 2):
        press.update(fb(cfg.check_height + 0.002))
    assert press.phase == "lift" and not press.seated


def test_a_stall_with_no_load_is_not_a_stall():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.01))
    for _ in range(int((cfg.stall_time + 0.5) / CONTROL_DT)):
        press.update(fb(0.01, load=0.0))
    assert press.phase == "descend"


def test_a_saturated_load_makes_the_press_give_up_instead_of_pushing_harder():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.01))
    for _ in range(int((cfg.saturated_time + 0.1) / CONTROL_DT)):
        press.update(fb(0.01, load=0.95))
    assert press.phase == "stuck" and press.done and not press.seated


def test_the_search_lifts_then_spirals_and_presses_again_where_the_tool_drops():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    stuck_z = cfg.check_height + 0.003
    press.reset(fb(stuck_z))
    for _ in range(400):
        press.update(fb(stuck_z, load=0.5))
        if press.phase == "lift":
            break
    assert press.phase == "lift"
    lifted = press.update(fb(stuck_z, load=0.5)).pos
    assert lifted[2] == pytest.approx(stuck_z + cfg.lift)

    for _ in range(int(cfg.lift_time / CONTROL_DT) + 1):
        press.update(fb(stuck_z, load=0.5))
    assert press.phase == "search" and press.searches == 1

    radii = []
    for _ in range(20):
        pos = press.update(fb(stuck_z, load=0.2)).pos
        radii.append(math.hypot(pos[0], pos[1]))
        assert pos[2] == pytest.approx(stuck_z - cfg.skim_press)
    assert radii == sorted(radii), "the spiral has to grow outwards"

    # The tool drops: the studs found their sockets, so press from here.
    dropped = press.update(fb(stuck_z - cfg.drop_detect - 1e-4, load=0.2))
    assert press.phase == "descend"
    assert dropped is not None


def test_the_search_gives_up_once_it_has_swept_its_whole_disc():
    cfg = PressConfig()
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.01))
    press._skim_z = 0.01
    press._enter("search")
    for _ in range(5000):
        press.update(fb(0.01, load=0.2))
        if press.phase != "search":
            break
    assert press.phase == "stuck" and press.done


def test_the_default_spiral_covers_its_disc_inside_one_episode_and_the_specs_does_not():
    """Why ``spiral_pitch`` deviates from the spec's 0.5 mm, and what the coarse pitch does and
    does not guarantee: through the interior it is within half a pitch of every point (swept
    going out, and again one turn further out coming back nearby), but the spiral ends mid-turn,
    so its outer rim is only swept from one side and the worst gap there is bigger than half a
    pitch -- see :class:`~brick_builder.control.primitives.PressConfig`'s docstring."""
    coarse_x, coarse_y = primitives._spiral_path(PressConfig())
    fine_x, fine_y = primitives._spiral_path(PressConfig(spiral_pitch=0.0005))
    assert len(coarse_x) < 100, len(coarse_x)
    assert len(fine_x) > 200, len(fine_x)   # more than the whole 200-tick episode, on its own

    path = np.column_stack([coarse_x, coarse_y])

    # Interior of the disc (up to one half-pitch short of the outer radius): every point is
    # within about half a pitch of the nearest sample of the sweep.
    rng = np.random.default_rng(0)
    worst_inner = 0.0
    for _ in range(2000):
        angle = rng.uniform(0, 2 * math.pi)
        radius = math.sqrt(rng.uniform()) * (4.0 - 0.5 * 2.5)   # mm, stays off the outer rim
        point = np.array([radius * math.cos(angle), radius * math.sin(angle)])
        gap = float(np.min(np.linalg.norm(path - point, axis=1)))
        worst_inner = max(worst_inner, gap)
    assert worst_inner <= 0.5 * 2.5 + 0.2, worst_inner

    # The outer rim itself, swept on a fine angular grid (deterministic, not a random sample --
    # this is exactly the boundary case the claim is about). The spiral's last ring is not
    # closed, so this is not "within half a pitch": measured worst gap 2.23 mm, not 1.25 mm.
    # Still comfortably inside the measured 3 mm capture basin of a 2x4
    # (docs/guide/04-environments.md, "Scripted press and the press environment"), which is the
    # claim that actually matters for the search to still find the sockets from there.
    thetas = np.linspace(0.0, 2.0 * math.pi, 3600, endpoint=False)
    ring = np.column_stack([4.0 * np.cos(thetas), 4.0 * np.sin(thetas)])
    gaps = np.min(np.linalg.norm(path[None, :, :] - ring[:, None, :], axis=2), axis=1)
    worst_outer = float(np.max(gaps))
    assert 2.0 <= worst_outer <= 2.5, worst_outer
    assert worst_outer < 3.0, "the outer-rim gap must still stay inside the capture basin"
    assert len(fine_y) == len(fine_x)


def test_rocking_tilts_first_and_levels_out_after_one_row_engages():
    cfg = PressConfig(rocking=True)
    press = PressScripted(0.0, cfg)
    press.reset(fb(0.02))
    assert press.phase == "rock"
    first = press.update(fb(0.02))
    assert first.tilt == pytest.approx(cfg.rock_tilt)

    # The tool goes down by rock_engage: one stud row is in, so level out.
    tilts = []
    for _ in range(int((cfg.rock_level_time + 0.1) / CONTROL_DT)):
        tilts.append(press.update(fb(0.02 - cfg.rock_engage - 1e-4)).tilt)
    assert tilts[0] > tilts[-1]
    assert tilts[-1] == pytest.approx(0.0, abs=1e-9)
    assert press.phase == "descend"


def test_every_phase_the_press_can_report_is_declared():
    """``press_phase`` is a fixed-width one-hot in the observation, so an undeclared phase would
    be an index error at exactly the wrong moment."""
    cfg = PressConfig()
    seen = set()
    for scenario in ("seat", "stall", "saturate"):
        press = PressScripted(0.0, cfg)
        press.reset(fb(0.02))
        seen.add(press.phase)
        for k in range(400):
            if scenario == "seat":
                feedback = fb(0.02 if k < 5 else 0.0002, load=0.5)
            elif scenario == "stall":
                feedback = fb(0.01, load=0.5)
            else:
                feedback = fb(0.01, load=0.95)
            press.update(feedback)
            seen.add(press.phase)
    assert seen <= set(PHASES), seen - set(PHASES)
    assert {"descend", "check", "lift", "search", "seated", "stuck"} <= seen
