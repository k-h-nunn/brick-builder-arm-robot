# Hardware, wiring, calibration

The servo-bus driver, the virtual bus it is proven against, the world adapters and digital-twin
synchronisation. **Status: verified against the virtual bus and the simulator; untested on
hardware.** No arm is connected. Every path below runs today against a pseudo-terminal and a
MuJoCo scene; the measurement procedures are what a teammate follows on the day the arm arrives,
with the parameter-file entry each one fills.

Code: `brick_builder/hardware/` (`sts3215.py`, `bus.py`, `virtual_bus.py`, `real_world.py`,
`twin_sync.py`), `brick_builder/calib/servo_bench.py`, `servo_selftest.py`, `servo_workspace.py`,
`servo_params.py`, `scripts/hardware/arm_selftest.py`, `servo_sysid.py`, `servo_workspace_map.py`,
`servo_params_fragment.py`, tests in `tests/hardware/` and `tests/calib/`.

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

## Self-test

`scripts/hardware/arm_selftest.py` (`brick_builder/calib/servo_selftest.py`) is the bring-up kit's
first step: ping every id, read the identity and safety registers, read `Present_Position` across
each joint's own range with torque off, then one small commanded move per joint with the reading
checked — a pass/fail table, one row per servo.

| Check | What | Against the virtual bus | Against real hardware |
|---|---|---|---|
| ping | the servo answers its own id | direct | direct |
| registers | model number (`servo_major`/`_minor`), firmware version (`firmware_major`/`_minor`), supply voltage, temperature, `P_Coefficient`, both dead-band registers (26/27), both angle-limit registers (9/11) | direct | direct |
| range read (torque off) | `Present_Position` tracks the joint across its calibrated travel with no torque | the scene's joint is posed directly and `SimulatedServos.resync()` pushes it into 56-70, the equivalent of backdriving it by hand | polls `Present_Position` for a few seconds while a person moves the joint; passes on a healthy bus and every reading inside the calibrated limits — how much of the range that covers is the person's job |
| commanded move | torque on, settle at 0 rad, then one ~5 deg step per joint; passes at 10 % of the commanded delta reached -- a loop-responds check, not a settling-accuracy one (no gravity feed-forward; `shoulder_lift` alone reaches only 16 % at 0 rad) | direct | direct |

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/arm_selftest.py --virtual
```

Against the virtual bus, all six joints pass every check (`results/servo_bringup/selftest.json`).
Nothing here replaces `Bus.enable_torque`'s own gate (voltage window, temperature, a non-zero
status byte); this is the longer report a teammate reads once before trusting the bus at all, and
its JSON (`--out`) is one of the two inputs the parameter-file fragment below is built from.

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
the friction dead zone rather than stopping inside it; treat the fitted friction as a lower bound.
Improving the friction fit needs a slower approach, which is the first thing to change if the
number matters.

### Validation: 5 randomized draws, and which quantities are not identifiable from the bus

The nominal recovery above is one case, with no free play and no dead band. `--validate N` draws
`N` sets of `dynamics.kp`, `dynamics.frictionloss`, `dynamics.backlash_deg` and
`command.error_deadband_ticks` from `brick_builder.scene.randomize` (`level=1.0`, the full
published range — the fields the `parameters` package added the dead band to), applies them with
`randomize.apply` exactly as training does, and identifies `shoulder_lift` through each one:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/servo_sysid.py \
    --validate 5 --joint shoulder_lift --seed0 1000
```

| quantity | mean \|error\| over 5 draws | identifiable from the bus? |
|---|---|---|
| kp | 2.6 % | yes — the droop line is the reliable part of this kit |
| friction | 64.5 % low | yes in principle, biased low in practice (see above) |
| dead band + free play, combined, against the TRUE combined figure | 48.7 % | **no** — see below |
| dead band + free play, combined, against free play ALONE (dead band excluded) | 12.2 % | — a coincidence, not a finding (see below) |
| dead band (the split attempt, `fit_dead_band_ticks`) | 100 % (fits 0 every draw) | **no** |

**The dead band cannot be separated from free play, and the ramp's own combined figure does not
reliably include the dead band's contribution either.** `fit_dead_band_ticks` looks for a
zero-`Present_Load` pocket at the ramp's turning point — inside the firmware's dead band the duty
cycle should read exactly zero — but every one of the 5 draws (true dead bands of 1.8-8.6 ticks)
fits 0. A closer look at the servo model (`brick_builder/robot/servo_model.py`) shows why: the
duty this bus reads never actually reaches zero near a reversal on a gravity-loaded joint, because
gravity and dry friction keep it away from zero even where the position error genuinely sits inside
the dead band. Position still stalls near a reversal — the ramp genuinely does see *something* —
but the stall is dominated by free play and by the friction-droop sign flip, not linearly by the
dead band's width: the combined figure lands close to the free-play-alone truth by coincidence,
not because it correctly adds the dead band on top. **Fill `arm.servo_dead_band_counts` from the
self-test's direct register read (26/27) instead of this fit** — that register is the one number
this kit reads rather than infers — and take the ramp's combined figure only as a rough, order-of-
magnitude diagnostic of total slop, not a precise free-play measurement; guide 10's dial-gauge test
is the trustworthy source for `arm.backlash_total_deg` if the two disagree by much.

**On hardware:** run system identification per joint, one joint at a time, with the others
holding. Enter `kp`'s implied P value in `arm.p_gain` (cross-checked against the self-test's own
register read), and see the parameter-file fragment below for how the dead band and free-play
fields are actually filled. Re-run the sag and step-response figures of guide 03 afterwards.

## Settling-error map

`scripts/hardware/servo_workspace_map.py` (`brick_builder/calib/servo_workspace.py`) commands a
tool-down pose over a polar workspace grid (100-200 mm radius, 5 headings, one fixed height) and
compares the commanded target against the settled tool position from the *measured* joint angles
(bus readings) and forward kinematics — the same P-gain droop, friction, free play and dead band
the per-joint sysid isolates, but all five arm joints together, at wherever the task needs the
tool, with no gravity feed-forward (this drives `Bus.cycle` directly, not `RealArmWorld`, which
does add it).

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/servo_workspace_map.py \
    --virtual --out results/servo_bringup/workspace_map.json \
    --figure results/servo_bringup/workspace_map.png
```

Against the virtual bus, nominal parameters, 15 grid points (3 radii x 5 headings): 12 reached, 3
unreachable at the 100 mm radius (the wrist geometry does not fold tightly enough at the two
side headings and straight ahead at that radius with the tool pointing straight down). Settling
error grows with radius — 3.8-11.3 mm at 100 mm, 8.7-9.7 mm at 150 mm, 11.7-13.5 mm at 200 mm, mean
10.5 mm, p95 13.3 mm over the 12 reached points — which is the expected shape for an
open-loop-in-gravity command: the further out, the larger the joint angles' contribution to tool
error for the same angular droop. `RealArmWorld`'s gravity feed-forward (guide 08's world adapters
section) is what the campaign's own controller runs with; this map deliberately does not use it,
because the point is to characterize the bare bus and servo law before anything compensates for it.

## Parameter-file fragment

`scripts/hardware/servo_params_fragment.py` reads the self-test's and the sysid's `--out` JSON and
prints the `arm:` block in `configs/params/default.yaml`'s own
`value`/`status`/`measured_by`/`measured_on`/`notes` format, ready to paste. This package does not
own `configs/params/`, so nothing pastes it there automatically.

```bash
.venv/bin/python scripts/hardware/arm_selftest.py --virtual --out results/servo_bringup/selftest.json
.venv/bin/python scripts/hardware/servo_sysid.py --virtual --joint shoulder_lift \
    --out results/servo_bringup/sysid_shoulder_lift.json
.venv/bin/python scripts/hardware/servo_params_fragment.py \
    --selftest results/servo_bringup/selftest.json \
    --sysid results/servo_bringup/sysid_shoulder_lift.json \
    --out results/servo_bringup/params_fragment.yaml
```

`arm.servo_p_gain` and `arm.servo_dead_band_counts` come from the self-test's register reads
(flagged in the notes if the six servos disagree); `arm.backlash_total_deg` is the sysid's combined
figure with that same registered dead band subtracted (not the sysid's own unreliable split, per
the validation above); `arm.joint_friction_Nm` is the sysid's friction fit, noted as a lower bound.
Against the virtual bus this is the kit's self-consistency check, not a claim that every number
comes back exact: `servo_p_gain` (16) and `servo_dead_band_counts` (0) match `ArmConfig()`'s
defaults exactly, because those two are register reads; `backlash_total_deg` comes out 1.224
against a true 0.87 and `joint_friction_Nm` 0.0149 against a true 0.0515, the same 41 %-high and
71 %-low biases the recovery table above already reports, carried through rather than hidden.
`results/servo_bringup/params_fragment.yaml` has the run this guide's numbers come from.

## Measurements for the day the arm arrives

Each row is one procedure and the parameter-file entry it fills.

| Measurement | Procedure | Fills |
|---|---|---|
| Self-test | `arm_selftest.py --port …` on the assembled bus. All six rows `PASS`. Nothing else runs until this passes. | `arm.servo_p_gain`, `arm.servo_dead_band_counts` (also read here, see below) |
| Homing offsets and directions | Drive each joint to both stops with torque off, log `Present_Position` at each, and set `JointCalibration.centre`/`direction` so the model's zero and sign match. **This is the one calibration that has no shipped value worth trusting.** | `JointCalibration` (code), to be mirrored into the parameter file |
| Servo system identification | `servo_sysid.py --port … --joint <name>`, per joint. | `arm.supply_voltage_V` (cross-check), `arm.joint_friction_Nm`; `arm.backlash_total_deg` combined with the self-test's own registers (see the parameter-file fragment) |
| Settling-error map | `servo_workspace_map.py --port …`. Diagnostic, not a parameter fill: a large or uneven error against the virtual bus's own map above is the sign to stop and debug before anything else. | — (a cross-check) |
| Parameter-file fragment | `servo_params_fragment.py`, fed the two `--out` files above. | the `arm:` block, ready to paste |
| Stall torque | Load a joint through a lever with known masses until the duty saturates at 0.97 and the joint gives way. Resolves the 1.9 vs 3.41 N·m disagreement of guide 03, which scales every press-force number. | `arm.torque_limit_Nm` |
| Press-force ramp | Tool-down onto a kitchen scale, jaws together; command the tool down in 1 mm steps and record the scale reading against `Present_Load` and `Present_Current` at each step. Gives the duty-to-force map the compliance scalars assume. Real only — there is no scale in the simulator. | `arm.torque_limit_Nm` (cross-check), guide 03's force map |
| Camera intrinsics | 15-20 shots of a 9x6, 15 mm chessboard at varied poses, `cv2.findChessboardCorners` then `cv2.calibrateCamera` at 640x480. | `camera.fx_px`, `fy_px`, `cx_px`, `cy_px`, `opencv_k1`, `k2`, `p1`, `p2`, `k3` |
| Camera extrinsics (hand-eye) | Same shots. Feed each shot's gripper-to-base transform (`ArmKinematics.fk`) and board-to-camera transform (`cv2.solvePnP`) to `cv2.calibrateHandEye`. Convert from OpenCV's convention (+z forward, +y down) to MuJoCo's (-z forward, +y up): a fixed 180° flip about the camera's own x axis. | `camera.mount_x_mm`, `mount_y_mm`, `mount_z_mm`, `mount_roll_deg`, `mount_pitch_deg`, `mount_yaw_deg` |
| Baseplate and tray survey | Ruler from the arm's base mounting point to the plate centre and the tray centre. | `workspace.baseplate_x_m`, `baseplate_y_m`, `baseplate_yaw_deg`, `tray_x_m`, `tray_y_m` |
| Twin baseline | One `TwinSync` run over the real bus on the same press motion, logged with `source: "real"`, compared against the `virtual_bus` log from the same motion. | the discrepancy table above, with a real row |

`camera.diagonal_fov_deg`, `camera.distortion_k1` and `camera.distortion_k2` stay as they are: the
first is the lens's nominal figure and the last two are the image-space remap
`render/domain_rand.py` uses, which is a **different parametrization** from the OpenCV coefficients
the calibration produces. Do not copy one into the other.

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

- `scripts/hardware/force_ramp.py` and `camera_calibration.py`: the procedures are in the table
  above and the driver exposes what each needs (`Bus.cycle` with a per-joint torque limit,
  `CameraIntrinsics`), but the two command-line wrappers do not exist yet.
- Chessboard image generation for a camera-calibration self-check, and the hand-eye fit itself.
- Watchdog and emergency-stop wiring. The driver's soft stop is a software stop on a live bus; it
  is not a substitute for cutting power.
