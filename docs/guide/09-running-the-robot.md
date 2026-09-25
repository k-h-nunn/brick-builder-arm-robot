# Running the robot

**Status: planned, except the pre-flight checklist below, which runs today against the virtual
bus and unchanged against the real bus once it arrives.** End-to-end operation of the real robot
starts once Stages G and H land. No real closed-loop run has happened yet. This page will gain a
step-by-step runbook and a recorded demo video once that stage lands.

## Purpose

Running Brick Builder Arm Robot on the real arm, from power-on to a finished LEGO DUPLO structure.

## Pre-flight checklist

The bring-up kit ([Hardware, wiring, calibration](08-hardware-wiring-calibration.md)), run in
order, before anything else touches the bus. Every command below takes `--virtual` today (a
simulated arm behind a pseudo-terminal) and `--port /dev/ttyACM0` (or whatever the adapter board
enumerates as) the day the arm arrives -- same commands, same order, same pass criteria.

1. **Self-test.** `scripts/hardware/arm_selftest.py` -- ping, identity and safety registers, a
   torque-off range read, one small commanded move per joint. Nothing past this step runs until
   every joint's row reads `PASS`.
2. **System identification.** `scripts/hardware/servo_sysid.py --joint <name>`, once per arm
   joint (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`), one joint at
   a time with the others holding. Confirms `kp` against the P register read in step 1 and gives
   the friction and free-play figures guide 08 explains how to read.
3. **Settling-error map.** `scripts/hardware/servo_workspace_map.py` -- commanded vs settled tool
   position over a workspace grid, tool down. This is the frame-rate/settling-accuracy check: a
   bus that cannot hold a 50 Hz cycle (guide 08's round-trip measurement) or a servo law far off
   the simulator's shows up here as a large or uneven error across the grid, before any brick is
   involved.
4. **Parameter-file fragment.** `scripts/hardware/servo_params_fragment.py`, fed steps 1 and 2's
   `--out` JSON, prints the `arm:` block to paste into `configs/params/default.yaml`.

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
