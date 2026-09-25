# Hardware, wiring, calibration

The servo-bus driver, the virtual bus it is proven against, the world adapters and digital-twin
synchronisation. **Status: verified against the virtual bus and the simulator; untested on
hardware.** No arm is connected. Every path below runs today against a pseudo-terminal and a
MuJoCo scene; the measurement procedures are what a teammate follows on the day the arm arrives,
with the parameter-file entry each one fills.

Code: `brick_builder/hardware/` (`sts3215.py`, `bus.py`, `virtual_bus.py`, `real_world.py`,
`twin_sync.py`), `scripts/hardware/servo_sysid.py`, tests in `tests/hardware/`.

## Hardware

| Part | What | Connection |
|---|---|---|
| SO-101 follower arm | Six Feetech STS3215 serial bus servos, ids 1-6 in the order `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` | Daisy-chained on one three-wire bus (V+, GND, signal) |
| Servo-bus adapter board | The kit's own USB adapter: half-duplex, automatic direction control | USB to the host; one servo lead to the chain |
| Supply | 7.4 V to the adapter board's barrel jack, not from USB | Common ground with the bus |
| Wrist camera | 32x32 mm USB UVC module on TheRobotStudio's SO101 Hex-Nut mount | USB to the host |

The arm is stock. Nothing is added to either jaw, and the camera is the only sensor beyond the
servos' own encoders.

**Before power.** Check the supply polarity at the barrel jack, that every servo lead is seated,
and that no joint is against a stop. The driver refuses to enable torque on a bus that does not
answer (below), but it cannot detect a reversed supply.

## The bus driver

`brick_builder/hardware/sts3215.py` implements the Feetech SCS/STS serial protocol and the STS3215
memory table; `bus.py` is the driver over pyserial (the `hardware` extra).

**Protocol.** Dynamixel-1.0 frame layout — `FF FF ID LEN INSTR PARAM... CHK`, with
`CHK = (~(ID + LEN + INSTR + sum(PARAM))) & 0xFF`. PING, READ, WRITE, REG WRITE, ACTION, SYNC READ
(0x82) and SYNC WRITE (0x83), addressed to one id or to the broadcast id 0xFE. Two STS-specific
conventions: 16-bit registers are **little-endian** (the SCS series is big-endian), and
`Goal_Speed`, `Present_Speed` and `Present_Current` are **sign-magnitude** with the sign in bit 15,
`Present_Load` with the sign in bit 10 over a 0-1000 magnitude. Source: Feetech's published STS3215
memory table (STS/SMS series datasheet), as reproduced in Feetech's SCServo SDK (`SCServo_Python`,
`sms_sts.py`).

**Registers the driver uses.**

| Register | Address | Area | Used for |
|---|---|---|---|
| `P_Coefficient` | 21 | EPROM | Servo gain; sets `ArmConfig.p_gain` |
| `CW_Dead_Band` / `CCW_Dead_Band` | 26 / 27 | EPROM | `ServoConfig.error_deadband_ticks` |
| `Torque_Enable` | 40 | RAM | Soft stop |
| `Goal_Position` | 42 | RAM | One sync-write per tick |
| `Goal_Speed` | 46 | RAM | 0 at startup: the host's trajectory sets the speed |
| `Torque_Limit` | 48 | RAM | Per-joint compliance, 0-1000 = 0-100 % |
| `Lock` | 55 | RAM | Opened and re-closed around an EPROM write |
| `Present_Position` … `Present_Current` | 56-70 | RAM | One 15-byte sync-read per tick |

The whole table is in `sts3215.REGISTERS`, keyed by name, with each entry's size, area, unit and
sign convention.

**Position conversion.** The encoder is 12 bit over one *output-shaft* turn, so one count is
`2*pi/4096` rad of link angle (guide 03: the encoder sits past the 1:345 gearbox).
`JointCalibration` holds a per-joint centre count, direction and travel limits;
`so101_calibration()` ships every joint centred at 2048 with direction +1 and the travel limits of
`third_party/so101/so101_new_calib.xml`. A test compiles the arm and checks those limits still
match the model. **The real arm's per-unit homing offsets and drive directions are a measurement**
(below), not a constant.

**One tick.** `Bus.cycle(goal_rad)` clips each goal to that joint's travel, converts to counts,
sends one sync-write of `Goal_Position` and one sync-read of 56-70, and returns a `BusState` with
positions, speeds, loads (PWM duty, not torque), currents, voltages, temperatures, a per-joint
reply-valid flag and the measured round trip. A lost reply holds the previous value rather than
stalling the loop. `BusSettings.sync_read = False` falls back to one READ per servo for firmware
without SYNC READ.

**Error counters.** `Bus.counters` counts instructions, replies, timeouts, checksum errors, framing
errors, retries, resynchronization bytes, status errors and missing replies. Nothing on the wire is
silently swallowed.

**Safety, all in the driver.**

- *Angle limits.* Every goal is clipped to the joint's own travel before it reaches the wire, in
  radians in `cycle` and again in counts in `write_ticks`.
- *Torque limit.* `Torque_Limit` (48) is written per joint as a fraction of the servo's own limit,
  which is how the `c_vert`/`c_lat` compliance scalars reach hardware.
- *Soft stop.* `disable_torque()` broadcasts `Torque_Enable = 0` to all six.
- *Startup self-test.* `self_test()` pings all six, reads voltage, temperature, P and both dead
  bands from each, and fails on a missing servo, a supply outside 6.0-8.4 V, a servo above 55 °C or
  a non-zero status byte. `enable_torque()` runs it first and raises rather than energizing;
  it then seeds each servo's goal with its own present position, so an arm pushed by hand does not
  snap back to a stale target.

**Contract for the microcontroller's bus stub.** `bus_write(targets[6])` and
`bus_read(pos[6], speed[6], load[6])`, both in encoder counts. `Bus.write_ticks` and
`Bus.read_ticks` are those two functions on the host side; `read_ticks` also returns a per-servo
valid flag.

## The virtual bus

`virtual_bus.py` speaks the same protocol on the other end of an `os.openpty()` pair. `Bus(port)`
opens the slave with pyserial and cannot tell it from the adapter board: every byte crosses a real
file descriptor, so framing, partial reads, timeouts and resynchronization are exercised rather
than mocked.

Two backends. `RegisterServos` is a register file with the STS3215 power-on defaults and no
physics. `SimulatedServos` puts a MuJoCo `Scene` behind it: one sync-write of `Goal_Position` is
one `ServoBus.write` plus ten 2 ms physics steps — one 20 ms control tick — after which 56-70 is
refreshed from `ServoBus.read()`. The whole existing servo model is therefore behind the wire:
encoder quantization, the firmware's rate limit and dead band, free play, and the duty cycle
`Present_Load` really reports. `WireConfig` injects packet loss, turnaround and per-servo latency
with the same meaning those fields have in `robot/bus_emulator.py`.

```python
from brick_builder.hardware.bus import Bus, BusSettings
from brick_builder.hardware.virtual_bus import simulated_bus

vbus = simulated_bus(scene).start()          # scene is any brick_builder Scene with the arm in it
bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
bus.enable_torque()
state = bus.cycle(goal_rad)                  # drives the simulated arm
```

**Measured.** A conformance run against the register-file backend
(`virtual_bus.conformance_run(1_000_000)`) drives 125 000 cycles — one sync-write, one sync-read
request and six replies each — through the driver:

| | |
|---|---|
| Packets on the wire | 1 000 000 (125 000 cycles x 8) |
| Protocol errors, driver side | 0 (no timeouts, checksum errors, framing errors or missing replies) |
| Bad frames, bus side | 0 |
| Round trip, mean / p95 | 0.24 ms / under 20 ms, so a 50 Hz tick has margin to spare |

`tests/hardware/test_bus.py` runs a short version of the same thing on every suite run. Under
injected loss (`WireConfig(packet_loss=0.25)`) the driver retries, then holds the previous reading
for the servos that did not answer and flags them `valid=False`; the error counters rise and the
control loop does not stall.

Against the **simulated** backend the measured round trip is 3-9 ms, because the physics of one
control tick runs inside the exchange. That is a property of the virtual bus, not of the protocol:
the protocol's own round trip is the register-file number above.

## World adapters

`real_world.py` answers the four `WorldAdapter` verbs — `command`, `feedback`, the `joint_*`
readings, and `observe`/`observe_rgb` — against the driver and a real camera.

`RealArmWorld` solves the same inverse kinematics `control/pick_place.py`'s `ArmDriver` solves
(`envs.base_env._pose_ik` with the full Levenberg-Marquardt fallback), adds the same gravity
feed-forward, writes one bus cycle and waits out the rest of the 20 ms on a monotonic scheduler
(`Pacer`, which reports overruns and re-bases rather than firing a burst of catch-up ticks). Every
tick is logged. `close()` soft-stops the arm before releasing the port.

`RealCamera` is a UVC camera through OpenCV: undistort with the calibrated intrinsics, cut the
largest centred square, resize to the scene model's 256x256 window, run the model on the host.
The frame source is an interface — `UvcCamera`, `ImageFileSource`, `ArraySource` — so the whole
chain is tested with a recorded or rendered frame and no device. The camera is opened lazily, never
at import.

`HybridWorld` pairs one arm with one camera, each real or simulated:

- `real_arm_simulated_camera(bus, twin)` — the twin renders the wrist view from the joint angles
  the bus reports, so the bus, the inverse kinematics and the cadence can be checked with a picture
  nobody has to trust yet.
- `simulated_arm_real_camera(adapter, camera)` — the simulator's own `WorldAdapter` with a real
  camera, so the camera path can be checked with an arm nobody has to trust yet.

## Twin synchronisation and the discrepancy metrics

`twin_sync.py` keeps a simulator `Scene` aligned with whatever is on the other side of the bus.

**The arm.** Each tick both twins are advanced with the tick's joint targets, then compared with
what the bus reported. The *tracking* twin is re-seated from `Present_Position` afterwards —
through the free-play partner joints, so the link angle matches the encoder — which makes its next
output a one-step prediction of the next reading. The *open-loop* twin is never re-seated, so its
divergence is the accumulated modelling error. The pair separates a twin whose one-step dynamics
are wrong (prediction error moves immediately) from one whose steady state is wrong (open-loop
error grows).

**Bricks and the baseplate.** A slot state carries a blob centre in normalized window coordinates.
With the camera's intrinsics and its extrinsics from forward kinematics, that centre plus a known
plane height is one world point (`perception.backproject.backproject_to_plane`); the blob's fourth
moment gives the footprint's yaw, modulo 90°, with the quadrant taken from the pose the twin
already holds. Estimates are associated with the nearest twin body, and a body no slot claims keeps
its pose and counts the ticks it has been held.

**Per tick.** `DiscrepancyRecord`: open-loop joint RMSE and max, tool position and rotation error,
one-step prediction RMSE and max, scene-state distance, bus round trip.

**Scene-state distance.** Object slots are matched greedily on blob centre; each matched pair
contributes the Euclidean distance between its geometry fields (centre, mass, second moments,
fourth moment) and each unmatched slot a fixed penalty. Colour is excluded: a lighting change moves
it without the scene having moved.

**The log.** One format for every source — `sim`, `virtual_bus`, `hybrid`, `real` — because a
discrepancy is only interesting beside the same number from another run. `TwinLog.write(dir)`
produces `meta.json` (source, perturbation, seeds, counters) and `ticks.npz` (one array per column
of `TWIN_LOG_COLUMNS`); `TwinLog.read` reverses it and `to_csv` writes the scalar columns. No new
dependency. `discrepancy_report(log)` returns a markdown table and a four-panel figure.

### Evidence: the metrics detect a perturbed far end

`twin_sync.perturbation_study()` runs six "real versus twin" comparisons over the virtual bus. The
far end is a second simulator with perturbed parameters, driven only through `Bus` over a
pseudo-terminal; the twin is nominal and sees nothing but the joint readings and the wrist view.
The motion is a scripted press: hover over a 2x4 brick resting on the stud tops at the plate's
centre, push down on its roof with the jaws together, retreat — 200 ticks. Both twins start from
the far end's state at tick 0, and the servo model's random draws are off on both sides, so the
baseline is identical by construction and every non-zero entry is the perturbation.

| perturbation | open-loop joint RMSE [mdeg] | open-loop tool position [mm] | one-step prediction RMSE [mdeg] | scene-state distance |
|---|---|---|---|---|
| nominal | 9.9 | 0.027 | 47 | 0.015 |
| clutch force 3x | 148 (14.9x) | 0.68 (25.4x) | 51 (1.1x) | 0.013 (0.9x) |
| joint friction 3x | 347 (34.9x) | 1.58 (59.0x) | 42 (0.9x) | 0.013 (0.9x) |
| free play 3x | 281 (28.2x) | 0.48 (18.1x) | 60 (1.3x) | 0.014 (0.9x) |
| servo gain P=32 | 784 (78.9x) | 5.56 (207.8x) | 62 (1.3x) | 0.016 (1.1x) |
| baseplate +3 mm | 187 (18.8x) | 1.10 (41.1x) | 47 (1.0x) | 0.030 (2.0x) |

Every perturbation moves the open-loop metrics by at least an order of magnitude over the baseline.
The baseplate offset is the one that also moves the scene-state distance, which is what
distinguishes it: the arm's own dynamics are nominal, the world it is pressing into is not. The
one-step prediction error is dominated by encoder quantization at these amplitudes and separates
the perturbations only weakly — it is the metric to watch on hardware, where quantization is the
same but the dynamics are not.

Reproduce (about 40 s):

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -c "
from brick_builder.hardware.twin_sync import perturbation_study
print(perturbation_study(out_dir='results/twin')[0])"
```

## System identification

`scripts/hardware/servo_sysid.py` runs step, ramp and chirp tests on one joint and fits the servo
law of guide 03. It drives the virtual bus today and the real bus the day the arm arrives — same
code, same tests, same fit.

- **kp** from the static droop line: the joint holds at several angles whose gravity torque the arm
  model knows, and `1/kp` is the slope of commanded-minus-measured against torque. Each hold is
  approached from below and from above; the mean of the two cancels the dry-friction dead zone.
- **Friction** from half the difference between those two approaches.
- **Peak step speed** from a step — a lower bound on the firmware's slew, not an estimate of it.
- **Lost motion** from the hysteresis of a slow triangular ramp; with kp and friction already
  fitted, the droop part is known and what is left is free play plus the firmware dead band.

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/servo_sysid.py \
    --virtual --joint shoulder_lift
```

**Recovery against the virtual bus**, whose parameters are known (`ArmConfig()`: P = 16 at 7.4 V,
`backlash_deg` 0.435 half-range, `frictionloss` 0.0515 N·m, dead band 0):

| quantity | known | fitted | error |
|---|---|---|---|
| kp | 9.384 N·m/rad | 9.32 | 0.7 % |
| dead band + free play | 0.87° (twice the half-range) | 1.22° | 41 % |
| friction | 0.0515 N·m | 0.0149 | 71 % low |

kp is the number the press-force budget and the compliance mapping depend on, and it comes back to
within a percent. Friction is under-recovered because the settling transient of each hold crosses
the friction dead zone rather than stopping inside it; treat the fitted friction as a lower bound
and the lost motion as free play plus dead band plus an unresolved friction term. Improving the
friction fit needs a slower approach, which is the first thing to change if the number matters.

**On hardware:** run it per joint, one joint at a time, with the others holding. Enter `kp`'s
implied P value in `arm.p_gain`, the dead band read straight off registers 26/27 in
`arm.servo_dead_band_counts`, and the lost motion in `arm.backlash_total_deg`. Re-run the sag and
step-response figures of guide 03 afterwards.

## Measurements for the day the arm arrives

Each row is one procedure and the parameter-file entry it fills.

| Measurement | Procedure | Fills |
|---|---|---|
| Self-test | `Bus.self_test()` on the assembled bus. All six answer; supply inside 6.0-8.4 V; no servo above 55 °C; no status bits set. Nothing else runs until this passes. | — (a gate) |
| Servo registers | Read P (21) and both dead bands (26/27) off each servo. They are set per unit at the factory; a second-hand arm may not be uniform. | `arm.p_gain`, `arm.servo_dead_band_counts` |
| Homing offsets and directions | Drive each joint to both stops with torque off, log `Present_Position` at each, and set `JointCalibration.centre`/`direction` so the model's zero and sign match. **This is the one calibration that has no shipped value worth trusting.** | `JointCalibration` (code), to be mirrored into the parameter file |
| Servo system identification | `servo_sysid.py --port …`, per joint. | `arm.p_gain`, `arm.backlash_total_deg`, `arm.servo_dead_band_counts`, `arm.joint_friction_Nm` |
| Stall torque | Load a joint through a lever with known masses until the duty saturates at 0.97 and the joint gives way. Resolves the 1.9 vs 3.41 N·m disagreement of guide 03, which scales every press-force number. | `arm.torque_limit_Nm` |
| Press-force ramp | Tool-down onto a kitchen scale, jaws together; command the tool down in 1 mm steps and record the scale reading against `Present_Load` and `Present_Current` at each step. Gives the duty-to-force map the compliance scalars assume. Real only — there is no scale in the simulator. | `arm.torque_limit_Nm` (cross-check), guide 03's force map |
| Camera intrinsics + extrinsics | `scripts/calibrate_camera.py --images DIR --poses poses.json --out FRAGMENT.yaml`: 15-20+ shots of a 9x6, 15 mm chessboard at varied *position and orientation*, each with the logged **`gripper` body** pose in the base frame (not `ArmKinematics.fk`'s TCP-site pose -- see "Camera calibration" below). Runs `cv2.calibrateCamera` for intrinsics/distortion and a hand-eye `AX = XB` solve for the mount pose, converts the hand-eye result from OpenCV's convention to MuJoCo's, prints a 180°-sensor-flip check, and writes the fragment directly. | `camera.fx_px`, `fy_px`, `cx_px`, `cy_px`, `opencv_k1`, `k2`, `p1`, `p2`, `k3`, `mount_x_mm`, `mount_y_mm`, `mount_z_mm`, `mount_roll_deg`, `mount_pitch_deg`, `mount_yaw_deg` |
| Baseplate and tray survey | Ruler from the arm's base mounting point to the plate centre and the tray centre. | `workspace.baseplate_x_m`, `baseplate_y_m`, `baseplate_yaw_deg`, `tray_x_m`, `tray_y_m` |
| Twin baseline | One `TwinSync` run over the real bus on the same press motion, logged with `source: "real"`, compared against the `virtual_bus` log from the same motion. | the discrepancy table above, with a real row |

`camera.diagonal_fov_deg`, `camera.distortion_k1` and `camera.distortion_k2` stay as they are: the
first is the lens's nominal figure and the last two are the image-space remap
`render/domain_rand.py` uses, which is a **different parametrization** from the OpenCV coefficients
the calibration produces. Do not copy one into the other.

## Camera calibration

`brick_builder/calib/camera.py` implements checkerboard corner detection, `cv2.calibrateCamera`
intrinsics + distortion, a hand-eye `AX = XB` solve, and a 180°-sensor-flip check;
`scripts/calibrate_camera.py` is the command-line wrapper (see the table row above for real-hardware
use). The pose logged for each shot must be the **`gripper` body**'s own pose in the base frame,
not `ArmKinematics.fk`'s return: `fk` reports the TCP site, offset from the gripper body by a
fixed, non-trivial transform (about 97 mm and 180° at a typical pose). A forward-kinematics
reading that only gives the TCP pose must be converted first with
`scripts.calibrate_camera.gripper_pose_from_tcp_pose`.

This project's installed OpenCV (5.0.0) exposes the `CALIB_HAND_EYE_*` method constants but not
the `cv2.calibrateHandEye` Python function itself (absent from `dir(cv2)` and from the installed
`cv2/__init__.pyi` stub) — `calibrate_hand_eye` calls it when present and otherwise falls back to
a from-scratch closed-form axis-correspondence solve that does not implement any of Tsai/Park/
Horaud/Andreff/Daniilidis and ignores the requested `method` (`brick_builder.calib.camera`'s own
module docstring has the convention notes; re-check this once OpenCV is upgraded). The written
parameter-file fragment's `notes` field names whichever one actually ran
(`HandEyeResult.solver_description`), not always `cv2.calibrateHandEye`.

**`--simulate`**: dry-runs the whole pipeline with no hardware, rendering a test-only checkerboard
scene with MuJoCo at a known `CameraConfig` (never reachable from the default scene builder) and
reporting recovery error against that known truth. Its output fragment's `measured_by` field is
always `simulate`, never a person's name, since the numbers are a synthetic round-trip, not a
measurement:

```
.venv/bin/python scripts/calibrate_camera.py --simulate --out-dir sim_calib --out sim.yaml
```

Validated (`tests/calib/test_camera_sim.py`, slow): 65 known arm poses (position, pitch and roll,
not just position and yaw — real rotational spread is what makes the hand-eye solve
well-conditioned) at a default, a perturbed-pose (a few mm/degrees) and a perturbed-field-of-view
`CameraConfig`:

| Case | poses (found/requested) | reprojection RMS [px] | focal error [%] | position error [mm] | rotation error [deg] |
|---|---|---|---|---|---|
| default | 16/65 | 0.668 | 1.26 | 5.85 | 4.050 |
| perturbed pose (+4/-3/+2 mm, ~2°) | 35/65 | 1.021 | 6.15 | 31.85 | 1.515 |
| perturbed fov (100° diagonal) | 22/65 | 0.742 | 0.32 | 5.42 | 0.968 |

(`tsai`, seeds 0/1/2; the 180° flip check correctly reports "not flipped" in all three. This table
predates a corner-detection fix and has not yet been re-run against it -- see `docs/guide/03`'s
wrist-camera section for the full account and why the perturbed-pose row is expected to improve.)
The perturbed-pose case's reprojection RMS and focal error are both worse than the other two
cases' (1.021 px/6.15% against 0.668-0.742 px/0.32-1.26%). The root cause is not the intrinsics
fit's own conditioning, and a genuine `cv2.calibrateHandEye` binding would not change it: given
clean corners both the intrinsics fit and the closed-form hand-eye solve are accurate. It traces
to `find_corners`' checkerboard corner refinement using a single fixed search window that overruns
neighbouring corners on this pose set's foreshortened far rows; `find_corners` now sizes that
window from each image's own detected corner spacing instead.

**This does not yet demonstrate resolving a real few-mm mount offset.** Every case's recovered
position error in the (not yet re-measured) table above is at or above both the
perturbation being tested (5.4 mm) and the CAD-nominal mount pose's own prior uncertainty
(`configs/params/default.yaml`, `camera.mount_x_mm` etc., "about 3-5 mm"). What is validated: the
pipeline runs end to end, recovers a camera in roughly the right place and orientation, and the
180°-flip check works.

## Findings from simulation to check on hardware

- **Friction grasp with the stock gripper.** Bare printed jaws, a wedge-shaped gap, the brick held
  near the tips; simulation measures 2.08 mm of slip under a 5 N downward load
  ([Simulator and bricks](02-simulator-and-bricks.md)). Repeat the 5 N, 5 s slip measurement on the
  real jaws and enter the measured jaw-on-brick friction
  ([Measure and enter parameters](10-measure-and-enter-parameters.md)).
- **A load-rise tug cannot verify a seated brick on this arm.** `Present_Load` is a PWM duty cycle,
  and this arm's Coulomb-friction dead band (about 4-5 mm at the TCP) is larger than a tug gentle
  enough to leave a seated brick alone ([Environments](04-environments.md)). Verify a seated brick
  by settled tool height and by vision, and confirm both still separate seated from loose with real
  servo noise in the loop.
- **The servo's real stall torque is unknown by a factor of 1.8.** Datasheet 1.9 N·m at 7.4 V
  against the identified law's 3.41 N·m at full duty
  ([Robot and servo model](03-robot-and-servo-model.md)). The press-force ramp above settles it.

## Not yet written

- `scripts/hardware/arm_selftest.py`, `force_ramp.py` and `camera_calibration.py`: the procedures
  are in the table above and the driver exposes what each needs (`Bus.self_test`,
  `Bus.cycle` with a per-joint torque limit, `CameraIntrinsics`), but the three command-line
  wrappers do not exist yet.
- Chessboard image generation for a camera-calibration self-check, and the hand-eye fit itself.
- Watchdog and emergency-stop wiring. The driver's soft stop is a software stop on a live bus; it
  is not a substitute for cutting power.
