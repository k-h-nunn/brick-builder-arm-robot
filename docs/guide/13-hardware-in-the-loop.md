# Hardware in the loop

The run-time controller is one C99 library, `firmware/control_lib`. It carries target-sequence
tracking, the inverse and forward kinematics, the host-side joint trim and the servo bus behind
two function pointers, and it has no dependency beyond libm. The same library runs inside the
Python process, as a native process on the other end of a byte stream, and — once that work
resumes — on a microcontroller. Everything the digital twin computes is data the host sends it in
one binary **target bundle**.

## The ladder

| Rung | What runs where | Status |
|---|---|---|
| Software in the loop | `control_lib` called by ctypes inside the simulator's process | built; runs a placement end to end, does not yet reproduce the Python loop tick for tick |
| Native process | the same library in a C `main`, wire protocol over a pipe or loopback TCP, host in lock step | built; protocol under unit test, not yet run against a whole episode |
| Processor in the loop | the library on an ESP32-P4, UART transport, simulated plant on the host | not started |
| Bus in the loop | the real STS3215 driver behind `bb_bus`, servos on the bench | not started |
| Arm in the loop | the arm moving, the host still scoring in simulation | not started |
| Shadow | the controller commanding nothing, its targets compared against the simulated ones | not started |
| Closed loop | the arm driven by the controller's own targets | not started |

## What the library owns, and what it does not

`bb_tick_begin` produces the 50 Hz writes for one vision tick: the tool target with the
correction added, the gripper command, the compliance scalars, and the six joint targets its own
inverse kinematics solved for. `bb_tick_end` takes one scene state and one set of bus readings
and runs the feedback, the progress index, the dwell rules, the grasp check and the end
conditions.

Two things are deliberately outside it:

- **The gravity feedforward.** `ArmDriver` adds a gravity term to the joint goal before writing
  it, from the arm's link inertias through MuJoCo's recursive Newton-Euler. The C library
  publishes the plain joint goal and the host adds that term. Moving it across needs the link
  inertias in the bundle and an RNE port.
- **The learned inverse G and the predictor F.** `brick_builder/export/c/action_net.c` already
  holds both forward passes in C99 with a ctypes parity test. Wiring G into the tick needs the
  feature encoder (`action/models/features.py: encode_inverse`) ported as well; the analytic
  inverse is the default and is what the C tick runs today.

## The target bundle

One little-endian blob, parsed in place — the controller points doubles straight at the caller's
buffer and copies nothing. It must be 8-byte aligned and outlive the run.

| Section | Contents |
|---|---|
| header, 64 B | magic `BBTB`, version, total bytes, CRC-32, frame and command counts, K and W, one offset per section |
| cfg | 30 doubles then 18 int32s: the tracking gains and tolerances, the servo's own constants, the joint-trim setting, the gripper's command range, and the indices the recording fixes (`descent_k`, `grasp_k`, `freeze_index`) |
| arm | the five-joint chain: parent-relative body frame and joint axis per link, the tool site's offset, the joint ranges |
| frames | one record per reference frame: progress index, step index, step title id, its command span, whether a brick is in the hand, whether this is the freeze step, the held body's reference index, and the frame's own travel |
| commands | every 50 Hz write of the recording: tool position, spin, gripper, and the two compliance scalars, as doubles |
| states | the reference scene state per frame: `[8, 14]` float32 slot rows, 8 float32 depth orders, 8 int32 slot ids |
| step offsets | one `uint32` per frame into the step records |
| step records | per frame: the feature Jacobian and its `s*`, the in-hand Jacobian and its `b*`, the yaw estimator's mode, period, gain and target, the reference descriptions (`select_slot`'s colour, size, tolerances and mass floor), and the host's own `trusted`/`gain` verdict |

The CRC covers everything after byte 16 and is CRC-32/ISO-HDLC, the same polynomial `zlib.crc32`
uses; a unit test checks the C and Python implementations against each other.

A step is measured at **every** reference frame rather than only where the live loop would
re-measure, because which frames those are depends on the run's own history. Both controllers are
then given the same bundle, so what a comparison shows is the control logic and not two different
twins.

## The wire protocol

Framed, CRC-checked, little-endian over any byte stream — a pipe, a TCP socket or a UART.

```
B1 C5 | type u8 | flags u8 | len u16 | seq u16 | payload[len] | crc32 u32
```

The CRC covers `type` through the end of the payload, so a frame whose sync pair appeared inside
an earlier payload fails it and the decoder resynchronises on the next pair. Maximum payload
2048 B.

| Type | Name | Direction | Payload |
|---|---|---|---|
| 0x01 | HELLO | controller → host | version, bundle capacity |
| 0x02 | TARGET_BUNDLE | host → controller | total bytes, this chunk's offset, bytes (chunked at 2040 B) |
| 0x03 | BUNDLE_ACK | controller → host | status, bytes received |
| 0x04 | SCENE_STATE | host → controller | frame id, host timestamp, `[8,14]` float32 slots, 8 depths, 8 ids |
| 0x05 | BUS_READINGS | host → controller in PIL; controller → host as telemetry when the bus is real | six positions, speeds and loads as doubles |
| 0x06 | TARGETS | controller → host | count, progress index, step index and id, then per 50 Hz write: tool target, spin, gripper, compliance, six joint targets, whether the solve converged |
| 0x07 | TICK_LOG | controller → host | progress index, step, holds, references seen, done, failure class, regrasps, matched frame, correction, tracking error, tolerance, absolute error, prediction error, yaw error |
| 0x08 | STATUS | either | version, frames, ticks, bad CRCs, resynchronisations, IK failures, mean and worst microseconds per tick |
| 0x09 | ERROR | controller → host | code, detail |
| 0x0A | TICK_BEGIN | host → controller | empty |
| 0x0B | RESET | host → controller | empty |

Lock step: the host sends TICK_BEGIN and waits for TARGETS, runs the plant for exactly those
writes, sends BUS_READINGS and SCENE_STATE, and waits for TICK_LOG. The controller never advances
on its own clock, and `bb_session_run` has no timer in it.

## Commands

```bash
# one placement, the repository's own Python loop (the default, unchanged)
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/run_placement.py --seed 1

# the same placement, the C library called in this process
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/run_placement.py \
    --seed 1 --controller c

# the same library as a native process, wire protocol over a pipe
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/run_placement.py \
    --seed 1 --controller process

# ... or over a loopback TCP socket
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/run_placement.py \
    --seed 1 --controller process:tcp

# both controllers on the same seeds, every difference reported
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.hil.equivalence \
    --seeds 0-9

# the fast checks
.venv/bin/python -m pytest tests/hil -q
```

The library and the native runner are compiled on demand into `firmware/control_lib/_build/`
with `gcc -O2 -std=c99 -ffp-contract=off`, the same flags the actor and action exports use, so no
fused multiply-add changes the rounding.

## What to measure at each rung

| Rung | Measure | Pass criterion |
|---|---|---|
| Software in the loop | progress index and step boundaries against the Python loop on the same bundle; joint targets per 50 Hz tick; seated-on-target counts over ten seeds | identical index, joint targets within 1e-6 rad, the same seeds seat |
| Native process | the same numbers against software in the loop; a corrupted frame and a dropped frame injected | identical to the last bit, and the protocol either recovers or reports an error |
| Processor in the loop | the same, over the UART; microseconds per tick from STATUS | identical targets; worst tick under 20 ms |
| Bus in the loop | commanded against reported joint angles on the bench | the servos reach their targets within the servo's own dead band |
| Arm in the loop | tool pose from forward kinematics against the simulated one | agreement within the measured encoder gap |
| Shadow | the controller's targets against the simulator's, arm not driven | no divergence over a whole episode |
| Closed loop | seated-on-target rate on hardware | measured, with a Wilson interval |

## Measured so far

- Forward kinematics and the site Jacobian against MuJoCo, 2000 random configurations inside the
  joint ranges: tool position within 3.3e-16 m, rotation within 1.4e-15, Jacobian within
  2.6e-15.
- The per-tick inverse kinematics (`envs.base_env._pose_ik`) C against Python, 1000 warm-started
  solves over the tool-down workspace: joint angles within 1.9e-9 rad, no disagreement on whether
  a solve was usable. The gap is the last bits of the Jacobian amplified by the damped
  least-squares step, whose three same-axis joints make it ill-conditioned.
- The joint trim, 400 ticks with the freeze input exercised: bit-identical.
- CRC-32 and the frame decoder: C and Python agree, and the decoder drops a corrupted or
  truncated frame and recovers on the next good one.
- One placement with `--controller c`, seed 1: the brick seated on the target cell, 0.07 mm
  lateral, 0.00 mm depth error. The progress index does **not** yet follow the Python loop's —
  247 vision ticks against 211, diverging from the first tick. That is an open defect in the
  port, not a property of the approach.

## Not done

- The progress index divergence above. The kinematics, the joint trim and the protocol are
  checked piece by piece; the tracking loop's port is not yet checked against the Python one.
- `firmware/p4_control/`, the ESP-IDF application: not started. `bb_session_run` and `bb_bus` are
  the two seams it needs, and both exist.
- A target-bundle round-trip test. The protocol and layout tests are in `tests/hil`; a bundle
  test needs a recorded sequence and so belongs with the slow tests.
- The gravity feedforward and the learned inverse, as above.
