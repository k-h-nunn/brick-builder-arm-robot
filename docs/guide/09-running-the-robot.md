# Running the robot

**Status: planned, except the pre-flight checklist below, which runs today against the virtual
bus and unchanged against the real bus once it arrives.** End-to-end operation of the real robot
starts once Stages G and H land. No real closed-loop run has happened yet. This page will gain a
step-by-step runbook and a recorded demo video once that stage lands.

## Purpose

Running Brick Builder Arm Robot on the real arm, from power-on to a finished LEGO DUPLO structure.

## Pre-flight checklist

The bring-up kit ([Hardware, wiring, calibration](08-hardware-wiring-calibration.md)), run in
order, before anything else touches the bus. Steps 1-3 take `--virtual` today (a simulated arm
behind a pseudo-terminal) and `--port /dev/ttyACM0` (or whatever the adapter board enumerates as)
the day the arm arrives -- same commands, same order, same pass criteria; step 4 takes neither
(it reads the JSON steps 1-2 wrote with `--out`).

1. **Self-test.** `scripts/hardware/arm_selftest.py` -- ping, identity and safety registers, a
   torque-off range read, one small commanded move per joint. Pass: every present joint's ping,
   registers and gate (voltage/temperature/status) read `PASS`, and the commanded-move column
   reads `PASS` for every joint except one whose own downstream chain was still drooped from the
   range read when its turn came (a known shape, not a fault -- see guide 08's self-test section).
   A commanded-move `FAIL` from a missing servo, the gate, or an overload trip is a real fault:
   stop and fix it before continuing.
2. **System identification.** `scripts/hardware/servo_sysid.py --joint <name>`, on
   `shoulder_lift`, `elbow_flex` and `wrist_flex`, one joint at a time with the others holding.
   `shoulder_pan` and `wrist_roll` raise (their own axis carries too little gravity torque to fit
   this way -- guide 08); their `kp` comes from step 1's P-register read instead. Pass: the three
   joints above each print a finite `kp`/friction/lost-motion row with no exception; there is no
   virtual-bus known value to check against on real hardware, so there is no numeric bar here
   beyond "it ran" -- guide 08's per-joint table is what those three numbers looked like against
   the virtual bus's own known values.
3. **Settling-error map.** `scripts/hardware/servo_workspace_map.py` -- commanded vs settled tool
   position over a workspace grid, tool down. Diagnostic, not a parameter fill: pass is a per-point
   error the same order of magnitude as guide 08's virtual-bus map (single-digit to low-teens mm);
   an error much larger, or many more unreachable points than the virtual bus's own map shows, is
   the sign to stop and debug before anything else. This does not check the bus's own cycle rate --
   guide 08's round-trip measurement (`Bus.measure_cycle_time`) is the timing check.
4. **Parameter-file fragment.** `scripts/hardware/servo_params_fragment.py`, fed steps 1 and 2's
   `--out` JSON, prints the `arm:` block to paste into `configs/params/default.yaml`. Pass: the
   printed block loads through `brick_builder.params.load_params` with no error (checked by
   `tests/calib/test_servo_params.py` against the virtual-bus fragment).

None of this is the physical safety checklist (supply polarity, seated servo leads, no joint
against a stop) guide 08's "Before power" section already gives -- that runs first, by hand, and
nothing here substitutes for it.

## Not yet written

Bring-up sequence past the checklist above (shadow mode, then ramped authority, then full
closed-loop operation); build-plan execution (a placement list reaching the skill state machine,
running the look-pick-transport-align-press-verify cycle with automatic repair attempts on an
unexpected placement); a fallback ladder (full autonomy down through a laptop-assisted mode to a
simpler perception source, with what triggers each drop and how to force a specific rung for a
demo); reading a log afterward (the twin-log schema, shared by simulation, hardware-in-the-loop,
and real runs, and replay of a real run's commands back through the simulator to check for
discrepancies); a demo-day runbook.
