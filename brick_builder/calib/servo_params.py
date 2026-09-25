"""Render the bring-up kit's findings as a parameter-file fragment.

``configs/params/default.yaml`` (the ``parameters`` package's file, not this one's) gives every
measured field a ``value``/``status``/``measured_by``/``measured_on``/``notes`` block under its
group; this module writes exactly that block, under ``arm:``, from a
:class:`~brick_builder.calib.servo_selftest.SelfTestReport` and a sysid
:class:`~scripts.hardware.servo_sysid.ServoFit`-shaped mapping, so the text is ready to paste
rather than retyped from a printed table.  Nothing here writes to ``configs/params/`` itself --
this package does not own that path.
"""

from __future__ import annotations

import datetime as _dt
import statistics
from dataclasses import dataclass

_DEG_PER_TICK = 360.0 / 4096.0


@dataclass(frozen=True)
class Field:
    value: float | int
    notes: str


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
        # `fit.dead_band_ticks`/`fit.backlash_deg` are `servo_sysid.py`'s own attempt to split
        # the ramp's combined hysteresis in two from bus behaviour alone; `--validate` shows that
        # split is close to 100 % wrong (the zero-load pocket it looks for is shorter than one
        # control tick).  The register read is what actually fills the dead band, and free play
        # is what is left of the *combined* (reliable) figure once that registered dead band is
        # taken out -- not `fit.backlash_deg`, which assumes its own unreliable split.
        registered_ticks = dead_band.value
        backlash_from_combined = max(
            fit.deadband_and_play_deg - 2.0 * registered_ticks * _DEG_PER_TICK, 0.0)
        fields["servo_dead_band_counts"] = Field(
            registered_ticks,
            dead_band.notes + " (sysid's own bus-behaviour split of the combined figure is "
            "validated unreliable at the 50 Hz control rate -- see guide 08 -- so this is the "
            "register read, not that split.)",
        )
        fields["backlash_total_deg"] = Field(
            round(backlash_from_combined, 3),
            f"system identification, joint {fit.joint}: ramp hysteresis "
            f"({fit.deadband_and_play_deg:.3f} deg combined dead band + free play, with the "
            f"droop part already taken out) minus this fragment's own servo_dead_band_counts "
            f"above (the self-test's register read, not sysid's unreliable split -- guide 08).",
        )
        fields["joint_friction_Nm"] = Field(
            round(fit.friction_nm, 4),
            f"system identification, joint {fit.joint}, from the hold-from-below/from-above "
            f"asymmetry; recorded for reference, biased low by the settling transient (guide 08) "
            f"-- treat as a lower bound.",
        )
    return fields


def render_param_fragment(
    fields: dict[str, Field], *, measured_by: str = "servo-bringup kit",
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
        status = "assumed" if virtual else "measured"
        lines.append(f"  {name}:")
        lines.append(f"    value: {field.value}")
        lines.append(f"    status: {status}")
        lines.append(f"    measured_by: {measured_by}")
        lines.append(f"    measured_on: {date}")
        notes = field.notes + source_note
        lines.append(f"    notes: \"{notes}\"")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
