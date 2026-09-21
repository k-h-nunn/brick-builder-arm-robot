# Running the robot

**Status: planned.** End-to-end operation of the real robot starts once Stages G and H land. No
real closed-loop run has happened yet. This page will gain a step-by-step runbook and a recorded
demo video once that stage lands.

## Purpose

Running Brick Builder Arm Robot on the real arm, from power-on to a finished LEGO DUPLO structure.

## Not yet written

Pre-flight checklist (self-test script, bus scan, voltage check, frame-rate check, and the
physical safety checklist from [Hardware, wiring, calibration](08-hardware-wiring-calibration.md));
bring-up sequence (shadow mode, then ramped authority, then full closed-loop operation); build-plan
execution (a placement list reaching the skill state machine, running the
look-pick-transport-align-press-verify cycle with automatic repair attempts on an unexpected
placement); a fallback ladder (full autonomy down through a laptop-assisted mode to a simpler
perception source, with what triggers each drop and how to force a specific rung for a demo);
reading a log afterward (the twin-log schema, shared by simulation, hardware-in-the-loop, and real
runs, and replay of a real run's commands back through the simulator to check for discrepancies);
a demo-day runbook.
