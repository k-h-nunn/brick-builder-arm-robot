# Running the robot

**Status: planned.** This page describes end-to-end operation of the real robot, which starts once
Stages G and H land. No real closed-loop run has happened yet. This page will gain a step-by-step runbook
and a recorded demo video once that stage lands.

## Purpose

This is the page you follow to actually run Brick Builder Arm Robot on the real arm, from power-on to a finished LEGO
DUPLO structure — the payoff of every earlier stage.

## Planned capabilities

- Bring the real system up safely, in a fixed, rehearsed order (self-test, homing, a shadow-mode check
  before any real motion is authorized).
- Ramp the learned policy's authority up gradually rather than trusting it fully on the first run, with a
  documented fallback ladder if something is not working
  on the day: dropping from full autonomy down through a laptop-assisted mode, then a simpler perception
  source, in that order.
- Feed the robot a build plan (from the LLM planner or a fixed script) and watch it execute the
  look-pick-transport-align-press-verify cycle, with automatic repair attempts if a placement does not
  come out as expected.
- Review a run afterward using the project's unified logging format, which records the same information
  whether the run happened in simulation, in a hardware-in-the-loop test, or for real.

## Sections to be filled in as the code lands

- **Pre-flight checklist** — self-test script, bus scan, voltage check, frame-rate check, and the
  physical safety checklist from [Hardware, wiring, calibration](08-hardware-wiring-calibration.md).
- **Bring-up sequence** — the order operations happen in: shadow mode, then a ramped authority increase,
  then full closed-loop operation, matching the hardware-in-the-loop ladder
  section 8.
- **Giving it a build plan** — how a placement list reaches the skill state machine, and what the
  look-pick-transport-align-press-verify cycle looks like from the outside.
- **The fallback ladder** — every rung, what triggers dropping to it, and how to force a specific rung for
  a demo.
- **Reading a log afterward** — the twin-log schema, and how to replay a real run's commands back through
  the simulator to check for discrepancies.
- **Demo-day runbook** — the fixed script and rehearsed contingencies for a live demonstration.
