# Hardware in the loop

The run-time controller is one C99 library, `firmware/control_lib`. It carries target-sequence
tracking, the inverse and forward kinematics, the host-side joint trim and the servo bus behind
two function pointers, and it has no dependency beyond libm. The same library runs inside the
Python process, as a native process on the other end of a byte stream, and in the ESP-IDF
application `firmware/p4_control` for the ESP32-P4 (built, not yet flashed). Everything the
digital twin computes is data the host sends it in one binary **target bundle**.

## The ladder

| Rung | What runs where | Status |
|---|---|---|
| Software in the loop | `control_lib` called by ctypes inside the simulator's process | passes: progress index and step boundaries identical, joint targets within 2.6e-15 rad (criterion 1e-6), placement seeds 0-9 and pick-and-place seeds 0-9 |
| Native process | the same library in a C `main`, wire protocol over a pipe or loopback TCP, host in lock step | passes: bit-identical to software in the loop on the same 20 episodes, both transports |
| Processor in the loop | the library on an ESP32-P4, UART0, simulated plant on the host | application built (`firmware/p4_control`); not flashed, not run |
| Bus in the loop | the real STS3215 driver behind `bb_bus`, servos on the bench | not started; `bb_bus` has a loopback stub |
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

### Port details

The Python loop (`action/place.py: TrackingEpisode`) is the reference, and three of its
properties are reproduced exactly:

- **No in-hand reading on the first measured step.** `_settle_servo` builds the `VisualServo` on
  the first usable step without that step's brick reference and sets `brick_ref` only when a
  later step is swapped in (the next `jacobian_every` frame or step boundary). Until then the
  loop lines up the tool alone. `bb_servo.brick` carries the same switch.
- **The matched frame is the last one matched.** `TICK_LOG` reports the frame the last
  successful reading matched, which is what `TrackingEpisode.matched` holds and what the learned
  predictor reads; a tick that returns early leaves it unchanged.
- **The placed brick is looked up by name.** In relative mode the placed brick's reference is
  found by its title among the *active* step's references, which were measured at an earlier
  frame and need not be listed in the same order. The bundle carries one integer per title for
  this.

## The target bundle

One little-endian blob, parsed in place — the controller points doubles straight at the caller's
buffer and copies nothing. It must be 8-byte aligned and outlive the run.

| Section | Contents |
|---|---|
| header, 64 B | magic `BBTB`, version (2), total bytes, CRC-32, frame and command counts, K and W, one offset per section |
| cfg | 30 doubles then 18 int32s: the tracking gains and tolerances, the servo's own constants, the joint-trim setting, the gripper's command range, and the indices the recording fixes (`descent_k`, `grasp_k`, `freeze_index`) |
| arm | the five-joint chain: parent-relative body frame and joint axis per link, the tool site's offset, the joint ranges |
| frames | one record per reference frame: progress index, step index, step title id, its command span, whether a brick is in the hand, whether this is the freeze step, the placed brick's title id (-1: the task places nothing, -2: this frame does not list it), and the frame's own travel |
| commands | every 50 Hz write of the recording: tool position, spin, gripper, and the two compliance scalars, as doubles |
| states | the reference scene state per frame: `[8, 14]` float32 slot rows, 8 float32 depth orders, 8 int32 slot ids |
| step offsets | one `uint32` per frame into the step records |
| step records | per frame: the feature Jacobian and its `s*`, the in-hand Jacobian and its `b*`, the yaw estimator's mode, period, gain and target, the reference descriptions (`select_slot`'s colour, size, tolerances, mass floor and title id), and the host's own `trusted`/`gain` verdict |

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

The CRC covers `type` through the end of the payload. A frame with a bad length or a bad CRC
loses its sync pair and the bytes behind it are scanned again, so a damaged frame never swallows
a good one that starts inside it. The C decoder (`bb_proto.c`) and the Python one
(`protocol.Decoder`) return the same frames on the same damaged stream. Maximum payload 2048 B.

| Type | Name | Direction | Payload |
|---|---|---|---|
| 0x01 | HELLO | controller → host at start; host → controller asks for it again | version, bundle capacity |
| 0x02 | TARGET_BUNDLE | host → controller | total bytes, this chunk's offset, bytes (chunked at 2040 B) |
| 0x03 | BUNDLE_ACK | controller → host | status, bytes received |
| 0x04 | SCENE_STATE | host → controller | frame id, host timestamp, `[8,14]` float32 slots, 8 depths, 8 ids |
| 0x05 | BUS_READINGS | host → controller in PIL; controller → host as telemetry when the bus is real | six positions, speeds and loads as doubles |
| 0x06 | TARGETS | controller → host | count, progress index, step index and id, then per 50 Hz write: tool target, spin, gripper, compliance, six joint targets, whether the solve converged |
| 0x07 | TICK_LOG | controller → host | progress index, step, holds, references seen, done, failure class, regrasps, matched frame, correction, tracking error, tolerance, absolute error, prediction error, yaw error |
| 0x08 | STATUS | either | version, frames, ticks, bad CRCs, resynchronisations, IK failures, mean and worst microseconds per vision tick (`bb_tick_begin` plus `bb_tick_end`, transport excluded), bus writes, bus targets dropped, longest bus period in microseconds |
| 0x09 | ERROR | controller → host | code, detail |
| 0x0A | TICK_BEGIN | host → controller | empty |
| 0x0B | RESET | host → controller | empty |
| 0x0C | START | host → controller | the five joint angles the arm starts at; the controller restarts the sequence with them as the first inverse-kinematics seed and answers BUNDLE_ACK |

Start: HELLO, the bundle in chunks (each acknowledged), START. Lock step: the host sends
TICK_BEGIN and waits for TARGETS, runs the plant for exactly those writes, sends BUS_READINGS and
SCENE_STATE, and waits for TICK_LOG. The controller never advances on its own clock, and
`bb_session_run` has no timer in it.

A damaged frame that is not a request (a stray frame between ticks) is dropped and costs
nothing. A request that is damaged or lost is never acted on; the host waits for the reply until
its timeout and raises `ControllerError` naming the message it waited for. Nothing retries: a
second TICK_BEGIN would advance the inverse kinematics twice.

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

# the Python loop against the C library, and the native process against the C library,
# on the same seeds; every difference reported, rows written as JSON
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.hil.equivalence \
    --seeds 0-9 --against c,process,process:tcp --out results/hil/equivalence_placement.json
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.hil.equivalence \
    --seeds 0-9 --pick --against c,process,process:tcp --out results/hil/equivalence_pick.json

# the fast checks: kinematics, framing, the decoders on a damaged stream, the three port
# details, and the native process against the library on a synthetic bundle, with damaged and
# lost frames in both directions
.venv/bin/python -m pytest tests/hil -q
```

`BB_CONTROL_BUILD_DIR` moves the build directory, so a test run does not rebuild a library
another process has loaded.

### Processor in the loop

Build, flash and the run are in [Export and firmware](07-export-and-firmware.md#controller-firmware-firmwarep4_control).
Once the board is flashed and its USB-UART is `/dev/ttyACM0` (the `hardware` extra installs
pyserial):

```bash
# the board against the C library in this process, bit for bit, ten placements
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.hil.equivalence \
    --seeds 0-9 --against c,serial:/dev/ttyACM0 --out results/hil/equivalence_board.json

# ... and ten pick-and-place runs
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.hil.equivalence \
    --seeds 0-9 --pick --against c,serial:/dev/ttyACM0 --out results/hil/equivalence_board_pick.json

# one placement driven by the board
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/run_placement.py \
    --seed 1 --controller serial:/dev/ttyACM0
```

The per-row output line carries the board's own STATUS timing, mean and worst microseconds per
vision tick; the JSON rows carry it as `tick_us_mean` and `tick_us_max`.

The library and the native runner are compiled on demand into `firmware/control_lib/_build/`
with `gcc -O2 -std=c99 -ffp-contract=off`, the same flags the actor and action exports use, so no
fused multiply-add changes the rounding.

## What to measure at each rung

| Rung | Measure | Pass criterion |
|---|---|---|
| Software in the loop | progress index and step boundaries against the Python loop on the same bundle; joint targets per 50 Hz tick; seated-on-target counts over ten seeds | identical index, joint targets within 1e-6 rad, the same seeds seat |
| Native process | the same numbers against software in the loop; a corrupted frame and a dropped frame injected | identical to the last bit, and the protocol either recovers or reports an error |
| Processor in the loop | the same, over the UART; microseconds per vision tick from STATUS | identical targets; worst tick under 20 ms |
| Bus in the loop | commanded against reported joint angles on the bench | the servos reach their targets within the servo's own dead band |
| Arm in the loop | tool pose from forward kinematics against the simulated one | agreement within the measured encoder gap |
| Shadow | the controller's targets against the simulator's, arm not driven | no divergence over a whole episode |
| Closed loop | seated-on-target rate on hardware | measured, with a Wilson interval |

## Measured so far

### Software in the loop and native process, 20 episodes

`python -m brick_builder.hil.equivalence --seeds 0-9 [--pick] --against c,process,process:tcp`:
per seed one recording, one target bundle, and four runs of the same world (the Python loop, the
library in process, the runner over a pipe, the runner over loopback TCP). Rows are in
`results/hil/equivalence_placement.json` and `results/hil/equivalence_pick.json`.

| Task, seeds 0-9 | `c` against Python: progress index and step boundaries | Max joint target difference [rad] | Max tracking error difference [mm] | Seated on the target cell, Python / `c` | Pipe and TCP against `c` |
|---|---|---|---|---|---|
| Placement | identical 10/10; 2294 vision ticks, 8690 writes at 50 Hz | 2.2e-15 | 5.3e-15 | 10/10 / 10/10 | bit-identical 10/10 each |
| Pick and place | identical 10/10; 3664 vision ticks, 16706 writes at 50 Hz | 2.6e-15 | 2.1e-14 | 9/10 / 9/10 | bit-identical 10/10 each |

Seated on the target cell, 95 % Wilson intervals: placement 10/10, 72.2-100 %; pick and place
9/10, 59.6-98.2 %. The miss is seed 4 on every side alike: the brick ends 16.1 mm lateral of the
target cell with no failure reported. The joint-target difference is the last bits of the inverse
kinematics; the progress index, the step boundaries, the number of 50 Hz writes and the end
condition are the same on every tick of all 20 episodes. Bit-identical means the joint targets,
the tracking errors and the corrections compare equal as doubles, tick for tick. Wall time for
the 20 episodes, four sides each, one process: 4481 s (placement 1295 s, pick and place 3186 s).

Provenance: the 20 episodes ran on the working tree before it was committed. Library sources
(among them `bb_servo.c` and `bb_bundle.h`) were edited after the placement run had loaded the
library, and `bb_control.c`, `bb_session.c` and the host modules in `brick_builder/hil` after
both runs had. Placement seed 1 and pick-and-place seed 4 were run again on commit
28fb383 with the same command (`results/hil/equivalence_placement_head.json`,
`results/hil/equivalence_pick_head.json`, 369 s): all six rows equal the published ones in every
compared field (progress index, step boundaries, vision and 50 Hz tick counts, maximum joint
target and tracking error differences, first divergent tick, outcome, lateral and depth error,
bit identity). The other 18 episodes have not been run on the committed code.

Controller time per vision tick on the development laptop (x86-64, `-O2`), from STATUS:

| Side | Mean [us] | Worst [us] |
|---|---|---|
| Library in process (ctypes marshalling included) | 38-116 | 6111 |
| Native process, pipe | 21-48 | 171 |
| Native process, loopback TCP | 24-49 | 546 |

Ranges are over the 20 episodes. The upper means and the worst ticks in process (6111 us) and
over TCP (546 us) are from pick-and-place seeds 7-9, whose wall times were 494-549 s against
217-248 s for seeds 0-6 on the same code; the runs were at nice 19 on a laptop shared with other
jobs. The other episodes have means of 21-53 us.

### Divergence root cause

The Python loop's inputs per vision tick (scene state, bus readings) were recorded for placement
seed 1 and fed to the library directly, which separates the control logic from the plant. The
first differing quantity was the tracking error at tick 0: 8.24 mm in C, 9.38 mm in Python, and
from there the correction and the index. The Python `VisualServo` has no brick reference on the
first measured step (see [Port details](#port-details)); the C servo read the
held brick from frame 0. With that and the two further details fixed, the same replay agrees on
every tick of placement seed 1 (211 ticks) and pick-and-place seed 0 (370 ticks): index, matched
frame and references seen identical, tracking error within 1e-9 mm, correction within 1e-12,
joint targets within 2.2e-15 rad.

### Component checks

- Forward kinematics and the site Jacobian against MuJoCo, 2000 random configurations inside the
  joint ranges: tool position within 3.3e-16 m, rotation within 1.4e-15, Jacobian within
  2.6e-15.
- The per-tick inverse kinematics (`envs.base_env._pose_ik`) C against Python, 1000 warm-started
  solves over the tool-down workspace: joint angles within 1.9e-9 rad, no disagreement on whether
  a solve was usable. The gap is the last bits of the Jacobian amplified by the damped
  least-squares step, whose three same-axis joints make it ill-conditioned.
- The joint trim, 400 ticks with the freeze input exercised: bit-identical.
- CRC-32: C and Python agree. The frame decoders: the same frames, the same bad-CRC count, on a
  300-frame stream with flipped bytes, cut frames, inserted garbage and sync pairs inside
  payloads.
- The runner against the library on a synthetic bundle (`tests/hil`): bit-identical over a pipe
  and over TCP. A corrupted or a truncated frame injected between ticks is dropped and the run
  stays bit-identical; a SCENE_STATE that is corrupted or never sent ends in `ControllerError`
  after the host's timeout, and no tick is run on it. A TARGETS or TICK_LOG reply that is
  corrupted or lost on its way back ends in the same `ControllerError`. A bundle with one flipped
  byte is refused (`BB_ERR_CRC`).
- The three [port details](#port-details), on synthetic bundles (`tests/hil`): a held brick
  shifted in the jaws changes nothing before the first swap and the tracking error after it, and
  tick 0 matches the Python `VisualServo` without a brick reference to 1e-9 mm; with the
  references listed in alternating order, a hidden or unlisted placed brick gives an unknown
  (NaN) tracking error and a hidden second reference does not; a blind tick keeps the last
  matched frame. Each test fails on a library with the corresponding detail reverted (brick
  reference from frame 0, placed brick by index into the current frame's step, matched frame from
  the current tick) and passes on the committed one.
- `firmware/p4_control` builds with ESP-IDF v5.5.5 for `esp32p4`: 257 776 B application image,
  93 504 B of internal RAM statically used; sizes in
  [Export and firmware](07-export-and-firmware.md#controller-firmware-firmwarep4_control).

## Not done

- **Processor in the loop.** The application is built and has never been flashed or run; the
  commands above are the next step. Its tick time is not measured. The P4's FPU is single
  precision, so the controller's doubles run in software floating point: addition,
  multiplication, division and square root round as on the host, but `atan2`, `sin`, `cos`,
  `hypot` and `exp` come from newlib rather than glibc and may differ in the last bit. The board
  may therefore agree with `c` to 1e-6 rad rather than bit for bit; that is measured, not
  assumed, when the board runs.
- **The STS3215 driver behind `bb_bus`.** `bb_bus_loopback` stands in. In a lock-step run that is
  faster than real time the 50 Hz writer cannot drain five writes per vision tick and drops the
  oldest; STATUS counts them. Only the loopback stub is affected.
- **The first-step brick reference.** Reproduced from the Python loop, not changed there: on the
  first measured step of a placement the loop lines up the tool without the in-hand reading.
  Open: whether the first measured step should carry the in-hand reading. Changing it belongs in
  `action/place.py` and would change every published controller measurement; the reference is
  kept as is.
- A fast test on a recorded bundle. Synthetic bundles are parsed and run in `tests/hil`,
  including the three port details; recorded bundles are exercised end to end by the
  equivalence run only.
- The gravity feedforward and the learned inverse, as above.
