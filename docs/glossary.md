# Glossary

Plain-language definitions of terms used throughout this site.

**Action model** — the part of the software that decides what the robot should do next; in this
project, a small neural network that outputs a correction on top of scripted motion. See
[Environments](guide/04-environments.md).

**Backlash** — mechanical play or slack in a joint's gear train: the amount the joint can move without
the motor turning. A real SO-101 servo has more backlash than its specification promises, which
matters because that play can be larger than the tolerance needed to seat a brick. See
[Robot and servo model](guide/03-robot-and-servo-model.md#gear-free-play-and-encoder-position).

**Clutch (DUPLO)** — the grip that holds a stud inside a tube once it is pressed in, coming from a tiny
manufacturing interference between the stud's and tube's actual sizes.

**Decode (of a scene model's output)** — a function that answers, for a queried pixel location,
which object (if any) the model thinks is there: how a coarse per-object summary becomes a
specific answer about a specific place in the image, such as "is there a brick at this stud
location." Viewer and evaluation only: the controller reads slot states, never a decode. See
[Perception](guide/06-perception.md).

**Digital twin** — a simulation that mirrors the real robot's actual measured state (joint angles, and so
on) so that a real run can be replayed and compared against simulated behavior afterward.

**Domain randomization** — varying simulation parameters (friction, servo stiffness, camera
lighting, etc.) across training so a learned policy does not overfit to the simulator's exact
assumptions and generalizes across a range of possibilities, including the real world.

**DUPLO clutch / stud / tube** — the three parts of a DUPLO brick connection: the **stud** is the raised
bump on top of a brick, the **tube** is the hollow cylinder on the underside of the brick above it that
the stud presses into, and the **clutch** is the grip that results (see above). See the
[DUPLO dimensions table](guide/02-simulator-and-bricks.md#duplo-dimensions).

**Hardware-in-the-loop (HIL)** — testing where some parts of the system are real hardware and other
parts are still simulated, used as a gradual path from "everything simulated" to "everything real"
without changing the software in between. See
[Export and firmware](guide/07-export-and-firmware.md).

**Object token** — the standard way perception hands information to the control code: a per-object record
of position, orientation, layer, color, size, confidence, and age. See
[Perception](guide/06-perception.md).

**Press-fit** — a connection that holds because two parts are sized to interfere slightly and grip when
pressed together, rather than through a separate fastener like a screw or clip. This is how DUPLO bricks
connect, and it is what Brick Builder Arm Robot's [snap-fit model](guide/02-simulator-and-bricks.md)
simulates.

**Scene model** — the perception component: it reads one camera frame and publishes a `SceneState`,
a small set of per-object "slot states". Three sources ship with the repository (simulator ground
truth, colour segmentation, a trained convolutional model) and others can be registered. See
[Perception](guide/06-perception.md).

**Residual policy** — a policy that outputs a small correction on top of an existing, simpler controller
(here, a scripted motion primitive), rather than deciding the whole action from scratch. See
[Environments](guide/04-environments.md).

**Sim-to-real** — the general problem of making something that works in simulation also work on the
real robot, and the gap between the two when it does not.

**Slot** — one object's row in a `SceneState`: mass, centre, second moments, colour, colour spread,
border fraction, age and a fourth-order orientation moment, plus an ordinal depth and a persistent
identity. Up to eight per frame. See [Perception](guide/06-perception.md).

**SO-101** — the specific robot arm design this project builds on: a small, low-cost arm using Feetech
STS3215 servo motors.

**STS3215** — the specific servo motor model used in the SO-101 arm.

**Token interface** — see **object token**.

**Vision-language-action (VLA) model** — a large neural network that takes in camera images and a text
instruction and directly outputs robot actions, without a separately engineered perception or planning
stage. This project's control stack — a small scene model plus scripted motion with a learned
residual — is the opposite end of that spectrum. See [the home page](index.md).
