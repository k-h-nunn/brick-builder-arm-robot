# Hardware, wiring, calibration

**Status: planned.** This page describes work that starts once the arm and bricks physically arrive
(Stage G). No hardware has been purchased or connected yet. This page will gain wiring diagrams, photos,
and calibration results once that stage lands.

## Purpose

This is the bridge from simulation to the real robot: how the physical parts connect, and how the
simulator gets tuned to match what the real hardware actually does.

## Planned capabilities

- Wire the servo bus, the camera, and the microcontroller together, following one of a small number of
  vetted options (with a documented fallback if the preferred option does not pan out).
- Run the calibration procedures that turn "a camera and an arm sitting on a table" into "a system that
  knows where its own hand and camera are, in the same coordinate frame as the bricks it is trying to
  place": camera intrinsics, hand-eye calibration, and a taught grid of known stud locations.
- Fit the simulator's servo model (stiffness, backlash, delay) to real measurements taken from the actual
  arm, using system identification, so that a policy trained in simulation transfers as cleanly as
  possible to the real robot.
- Follow the project's safety checklist (torque and angle limits, an emergency stop, watchdogs) before
  running anything on real hardware.

## Three findings from simulation to check on real hardware

Nothing below is a hardware measurement yet — these are simulation findings (
items 13-15) that this page's own procedures need to check once the arm exists, recorded here so
the checklist has somewhere to live before the hardware does.

- **Friction grasp with the stock gripper.** The hardware is the stock SO-101 gripper: fixed jaw and
  moving jaw, bare printed plastic, nothing attached. The jaw gap is a wedge (31.8 mm at the hinge end,
  39.9 mm at the tips for a 2x N brick), and the brick is held near the tips. Simulation measures
  2.08 mm of slip under a 5 N downward load ([Simulator and bricks](02-simulator-and-bricks.md)).
  **Check on hardware:** repeat the 5 N, 5 s slip measurement on the real jaws and enter the measured
  jaw-on-brick friction in the parameter file ([Measure and enter parameters](10-measure-and-enter-parameters.md)).
  Pressing while gripping pushes the brick into the narrowing part of the gap, which tightens the grip.
- **A load-rise tug cannot verify a seated brick on this arm.** `Present_Load` is the joint
  position error scaled, and this arm's identified Coulomb-friction dead band (about 4-5 mm at the
  TCP) is larger than a tug gentle enough to leave a seated brick alone; simulation gets the right
  answer only 6-8 times out of 20, with the seated and loose load-rise distributions overlapping
  the wrong way round ([Environments](04-environments.md) has the full measurement).
  **Action:** verify a seated brick by settled tool height and by vision (`SkillFSM.seated_by_height`,
  40/40 on held-out seeds in simulation), and confirm both signals still separate seated from loose
  once real servo noise and real friction are in the loop — do not rely on a load-rise tug.
- **The servo's real stall torque is unknown by a factor of 1.8.** The datasheet says 1.9 N·m at
  7.4 V; the servo law identified from step/chirp data in simulation saturates at 3.41 N·m at full
  duty ([Robot and servo model](03-robot-and-servo-model.md) has both derivations). Every
  press-force and torque-limit number computed so far uses whichever of the two the calling code
  happened to pick, not a measured value. **Action:** the system-identification section below is
  exactly the procedure that resolves this — run it early, since press-force budgets, the torque
  limit safety checklist, and the `c_vert`/`c_lat` compliance mapping all depend on the answer.

## Sections to be filled in as the code lands

- **Wiring** — the preferred servo-bus connection (a vendor adapter board), a documented fallback using
  discrete level-shifting parts, and the camera path (a ribbon-cable CSI camera vs. a USB camera),
  including which one was chosen once both have been tried.
- **System identification** — the procedure for fitting the corrected servo law (see
  [Robot and servo model](03-robot-and-servo-model.md)) to a specific real servo, using step, chirp, and
  ramp tests.
- **Camera calibration** — intrinsics from a calibration board, and extrinsics ("hand-eye" calibration)
  from bricks placed at known positions, since a 5-degree-of-freedom wrist makes standard hand-eye
  calibration only weakly solvable on its own and needs to be combined with a joint bundle adjustment.
- **Teaching the grid** — physically touching the gripper to known stud locations to fit a
  grid-to-joint-space correction, which is what actually bounds real-world placement accuracy.
- **Safety checklist** — servo-side and controller-side limits, watchdogs, and the emergency-stop wiring,
  to be reviewed before every session with real hardware.
