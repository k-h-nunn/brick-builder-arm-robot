"""Render the bring-up kit's findings as a parameter-file fragment.

``configs/params/default.yaml`` gives every measured field a
``value``/``status``/``measured_by``/``measured_on``/``notes`` block under its group; this module
writes exactly that block, under ``arm:``, from a
:class:`~brick_builder.calib.servo_selftest.SelfTestReport` and a sysid
:class:`~scripts.hardware.servo_sysid.ServoFit`-shaped mapping, so the text is ready to paste
rather than retyped from a printed table.  Nothing here writes to ``configs/params/`` itself;
pasting the printed block in is a manual step.
"""

from __future__ import annotations

import datetime as _dt
import statistics
from dataclasses import dataclass

_DEG_PER_TICK = 360.0 / 4096.0

GRAVITY_LOADED_JOINTS = frozenset({"shoulder_lift", "elbow_flex", "wrist_flex"})
"""The three joints the droop line can identify at all (guide 08): their own axis carries enough
gravity torque to fit ``kp``/friction from bus readings. Whether a *given* gravity-loaded joint's
ramp also sees the firmware dead band is not fixed by this set -- it depends on how that droop
compares to the registered dead band on this particular unit, computed per fit by
``ServoFit.dead_band_reachable`` (see :func:`build_fields`)."""


@dataclass(frozen=True)
class Field:
    value: float | int
    notes: str
    status: str | None = None
    """Overrides the value/real default status (:func:`render_param_fragment`) when a field's own
    reliability does not track whether the bus was virtual or real -- ``joint_friction_Nm`` is
    biased low by a fixed mechanism (the settling transient) whether or not the arm is real."""


def _p_gain_and_dead_band(report) -> tuple[Field, Field]:
    """The self-test's own register reads, across every present joint -- flagged if they
    disagree, since the P register and the dead band are set per unit at the factory."""
    present = [j for j in report.joints if j.present and j.p_gain >= 0]
    if not present:
        return (Field(16, "no self-test register reads available; simulator default."),
                Field(0, "no self-test register reads available; simulator default."))
    p_gains = [j.p_gain for j in present]
    p_mode = statistics.mode(p_gains)
    p_note = (f"read from register 21 on all {len(present)} present servos."
             if len(set(p_gains)) == 1 else
             f"registers 21 disagree across servos ({sorted(set(p_gains))}); "
             f"{p_mode} is the most common, from the self-test table.")
    dbs = [max(j.dead_band_ticks) if j.dead_band_ticks else 0 for j in present]
    db_mode = statistics.mode(dbs)
    db_note = (f"read from registers 26/27 (CW/CCW) on all {len(present)} present servos."
              if len(set(dbs)) == 1 else
              f"registers 26/27 disagree across servos ({sorted(set(dbs))}); "
              f"{db_mode} is the most common, from the self-test table.")
    return Field(int(p_mode), p_note), Field(int(db_mode), db_note)


def build_fields(report, fit=None) -> dict[str, Field]:
    """Every ``arm.*`` field this kit can fill, from a self-test report and (optionally) a sysid
    fit.  ``arm.torque_limit_Nm`` is left out: this kit does not run the stall-torque lever test.
    """
    present = [j for j in report.joints if j.present]
    voltages = [j.voltage_v for j in present if j.voltage_v == j.voltage_v]  # drop NaN
    p_gain, dead_band = _p_gain_and_dead_band(report)
    fields = {
        "supply_voltage_V": Field(
            round(statistics.mean(voltages), 2) if voltages else 7.4,
            f"mean of {len(voltages)} self-test voltage reads." if voltages
            else "no self-test voltage reads available; simulator default.",
        ),
        "servo_p_gain": p_gain,
        "servo_dead_band_counts": dead_band,
    }
    if fit is not None:
        # `fit.dead_band_ticks` is `servo_sysid.py`'s own attempt to split the ramp's combined
        # hysteresis in two from bus behaviour alone (`fit_dead_band_ticks`'s zero-`Present_Load`
        # pocket); `--validate` shows it fits 0 on every draw on a gravity-loaded joint -- not
        # because the pocket is too short to sample, but because the joint's own P-gain droop
        # against gravity keeps the ramp's position error outside the dead band at every point on
        # the ramp, so there is no pocket there to find (guide 08). The register read is what
        # actually fills `servo_dead_band_counts`.
        registered_ticks = dead_band.value
        fields["servo_dead_band_counts"] = Field(
            registered_ticks,
            dead_band.notes + " (the ramp's own hysteresis cannot fill this field on a "
            "gravity-loaded joint -- see guide 08 -- so this is the register read.)",
        )
        # Computed from the fit's own two numbers (not `ServoFit.dead_band_reachable`'s property
        # directly), so this also works when `fit` is a namespace loaded back from a saved
        # `servo_sysid.py --out` JSON file, which only round-trips dataclass fields.
        dead_band_reachable = fit.registered_dead_band_ticks >= fit.min_droop_ticks
        if not dead_band_reachable:
            # This joint's own smallest droop (min_droop_ticks) stayed at or above the registered
            # dead band everywhere it was swept, so nothing measured by the ramp needs the
            # registered dead band subtracted back out: the raw combined figure already is the
            # free-play estimate. Computed per fit (ServoFit.dead_band_reachable), not assumed from
            # which joint this is.
            backlash_value = fit.deadband_and_play_deg
            backlash_note = (
                f"system identification, joint {fit.joint}: ramp hysteresis "
                f"({fit.deadband_and_play_deg:.3f} deg, the droop part already taken out) taken "
                f"directly as free play -- this joint's own droop ({fit.min_droop_ticks:.1f} "
                f"ticks) never got as small as the registered dead band "
                f"({fit.registered_dead_band_ticks:.1f} ticks), so its ramp never saw the dead "
                f"band and nothing is subtracted (guide 08)."
            )
        else:
            backlash_value = max(
                fit.deadband_and_play_deg - 2.0 * registered_ticks * _DEG_PER_TICK, 0.0)
            backlash_note = (
                f"system identification, joint {fit.joint}: ramp hysteresis "
                f"({fit.deadband_and_play_deg:.3f} deg, the droop part already taken out) minus "
                f"this fragment's own servo_dead_band_counts above -- this joint's own droop "
                f"({fit.min_droop_ticks:.1f} ticks) got as small as the registered dead band "
                f"({fit.registered_dead_band_ticks:.1f} ticks) somewhere in the swept range, so "
                f"its ramp can see the dead band; the subtraction is only a partial correction "
                f"(guide 08 -- the kp/friction fit above is itself biased in this case, not just "
                f"this field)."
            )
        fields["backlash_total_deg"] = Field(round(backlash_value, 3), backlash_note)
        fields["joint_friction_Nm"] = Field(
            round(fit.friction_nm, 4),
            f"system identification, joint {fit.joint}, from the hold-from-below/from-above "
            f"asymmetry; known biased low by the settling transient crossing the friction dead "
            f"zone (guide 08, roughly 3x low in validation) -- a lower bound, not a measurement.",
            status="derived",
        )
    return fields


def render_param_fragment(
    fields: dict[str, Field], *, measured_by: str = "arm bring-up kit",
    measured_on: str | None = None, virtual: bool = True,
) -> str:
    """The ``arm:`` YAML block, in ``configs/params/default.yaml``'s own field format."""
    date = measured_on or _dt.date.today().isoformat()
    source_note = (
        " Measured against the virtual bus (no arm connected yet): the values below are the "
        "kit's self-consistency check, not a hardware measurement -- re-run unchanged against "
        "the real bus and re-paste." if virtual else ""
    )
    lines = ["arm:", ""]
    order = ["supply_voltage_V", "servo_p_gain", "backlash_total_deg",
            "servo_dead_band_counts", "joint_friction_Nm"]
    for name in order:
        if name not in fields:
            continue
        field = fields[name]
        status = field.status or ("assumed" if virtual else "measured")
        lines.append(f"  {name}:")
        lines.append(f"    value: {field.value}")
        lines.append(f"    status: {status}")
        lines.append(f"    measured_by: {measured_by}")
        lines.append(f"    measured_on: {date}")
        notes = field.notes + source_note
        lines.append(f"    notes: \"{notes}\"")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
