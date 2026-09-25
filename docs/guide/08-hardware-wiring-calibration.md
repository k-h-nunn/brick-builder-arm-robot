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
checked — a pass/fail table, one row per servo. The range read and the commanded move run one
joint at a time, interleaved: joint `k`'s range read, then its commanded move, then it is driven
back toward the centre of its own travel before joint `k + 1`'s range read starts, so a joint
already checked does not sit near its own extreme while the next joint sweeps its full range next
to it (that compounding used to let one joint's saturated loop drag a neighbour past its own
calibrated limit through the kinematic chain). That parking step cannot rescue the joint currently
under test, though: a full-range sweep leaves *that* joint at its own extreme, with no gravity
feed-forward anywhere in this check, so a gravity-loaded joint's own commanded move can genuinely
fail there on the arm's own weight alone — see the result below. Nothing here goes around
`Bus.enable_torque`'s own gate (voltage window, temperature, a non-zero status byte): the commanded
move calls it exactly like any other caller, and a gate failure fails every joint's row, the same
as it would on the wire. Torque is written off before anything else runs, so a servo left energized
from a previous session does not fight the range read.

Torque is bus-wide: every `disable_torque()` call (before the first range read, and after every
joint's own commanded move) drops holding torque on all six servos at once, not just the one under
test. Against the virtual bus this costs at most one 20 ms free-fall tick, because nothing here
steps physics except a written goal. On the real arm every one of those calls leaves the whole arm
limp — support it (or have it resting on something that will not damage it) for the full self-test
run, not only during the range reads.

| Check | What | Against the virtual bus | Against real hardware |
|---|---|---|---|
| ping | the servo answers its own id | direct | direct |
| registers | model number (`servo_major`/`_minor`), firmware version (`firmware_major`/`_minor`), supply voltage, temperature, `P_Coefficient`, both dead-band registers (26/27), both angle-limit registers (9/11) | direct | direct |
| gate | supply 6.0-8.4 V, temperature <= 55 °C, no status bits (`BusSettings`, the same window `Bus.enable_torque` always checks) | direct | direct |
| range read (torque off) | `Present_Position` tracks the joint across its calibrated travel with no torque | the scene's joint is posed directly and `SimulatedServos.resync()` pushes it into 56-70, the equivalent of back-driving it by hand | announces the joint by name, then polls `Present_Position` for a few seconds while a person moves it; passes on a healthy bus, every reading inside the calibrated limits, and at least 5° of actual motion seen — a PASS with no motion is refused |
| commanded move | torque on, one ~5 deg step per joint *from its own reading just before the step* (not an assumed baseline), aimed back toward the middle of its travel; passes between 10 % and 150 % of the commanded delta reached, settled, no overload trip — a loop-responds-and-stays-within-rating check, not a settling-accuracy one (no gravity feed-forward) | direct | direct |

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/arm_selftest.py --virtual
```

Against the virtual bus, seed 0: **4 of 6 joints pass** (`results/servo_bringup/selftest.json`).
`shoulder_lift` and `elbow_flex` fail the commanded-move check: `shoulder_lift` moves −2 % of the
commanded step, `elbow_flex` moves −219 % and reads 117° against a ±96.8° calibrated range.
`wrist_flex` passes at 115 % (ceiling 150 %).

Cause: open. `servo_selftest.py` and `tests/calib/test_servo_selftest.py` attribute the two
failures to gravity load at the joint's range extreme with no gravity feed-forward. A contact probe
on the same seed contradicts this. The virtual range read writes joint positions directly, and
after `shoulder_lift`'s range read the lower arm, wrist and gripper penetrate the shoulder link by
16-31 mm (18 contacts; 48 after `elbow_flex`'s). During the failed move `shoulder_lift`'s duty is
−0.22 (limit 0.97), so the joint is jammed, not torque-limited. With the park step skipped,
`shoulder_lift` passes (104 %). The test that expects `overall: FAIL` encodes the unresolved
result, so the kit cannot pass its own nominal self-test until the range read is made
collision-free.

**On real hardware a commanded-move FAIL on any joint stops the bring-up.** The virtual-bus result
above does not predict the real arm: there the range read is done by hand and the links cannot pass
through each other. Support the arm through every range read and commanded move.

This is the longer report a teammate reads once before trusting the bus at all; its JSON (`--out`)
is one of the two inputs the parameter-file fragment below is built from.

## System identification

`scripts/hardware/servo_sysid.py` runs step, ramp and chirp tests on one joint and fits the servo
law of guide 03. It drives the virtual bus today and the real bus the day the arm arrives — same
code, same tests, same fit.

- **kp** from the static droop line: the joint holds at several angles whose gravity torque the arm
  model knows, and `1/kp` is the slope of commanded-minus-measured against torque. Each hold is
  approached from below and from above; the mean of the two cancels the dry-friction dead zone.
  Only identifiable on a joint whose own axis carries meaningful gravity torque — see the
  per-joint table below.
- **Friction** from half the difference between those two approaches. Same restriction as kp.
- **Peak step speed** from a step — a lower bound on the firmware's slew, not an estimate of it.
- **Lost motion** from the hysteresis of a slow triangular ramp; with kp and friction already
  fitted, the droop part is known and what is left is free play plus the firmware dead band.

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/servo_sysid.py \
    --virtual --joint shoulder_lift
```

**Every arm joint, nominal virtual bus** (`ArmConfig()`: P = 16 at 7.4 V, `backlash_deg` 0.435
half-range, `frictionloss` 0.0515 N·m, dead band 0 — `results/servo_bringup/sysid_<joint>.json`):

| joint | gravity torque range | kp | friction | dead band + free play (combined) | dead band split [ticks] | free play (split) [deg] |
|---|---|---|---|---|---|---|
| shoulder_pan | 0 N·m (vertical axis) | not identifiable — raises | not identifiable | not identifiable | not identifiable | not identifiable |
| shoulder_lift | 0.74 N·m | 9.32 vs 9.384, 0.7 % | 0.0149 vs 0.0515, 71 % low | 1.22° vs 0.87°, 41 % | 0.0 vs true 0-9.2, correct sign (no pocket) | 1.22° (= combined, nothing subtracted) |
| elbow_flex | 0.26 N·m | 9.48 vs 9.384, 1.0 % | 0.0221 vs 0.0515, 57 % low | 1.11° vs 0.87°, 28 % | 0.0, same reasoning as shoulder_lift | 1.11° (= combined, nothing subtracted) |
| wrist_flex | 0.066 N·m | 7.81 vs 9.384, 17 % | 0.0236 vs 0.0515, 54 % low | 1.03° vs 0.87°, 19 % | 18.5, **spurious** (true dead band is 0 here) | 0.00° (100 % low — see below) |
| wrist_roll | 0.0026 N·m | not identifiable — raises | not identifiable | not identifiable | not identifiable | not identifiable |

`shoulder_pan` (a vertical axis: exactly zero gravity torque at every angle) and `wrist_roll` (the
wrist mass sits a lever arm under a millimetre from the roll axis) fail with an explicit
`ValueError` naming the gravity-torque range that was too small
(`MIN_GRAVITY_TORQUE_RANGE_NM = 0.02` N·m) rather than returning a number a noise-dominated fit
happened to produce — this is the identifiability answer for those two joints: their own `kp` has
to come from the self-test's P-register read, not from a droop line. Of the three joints the
method does work on, `kp` recovers to within about 1 % on the nominal draw for the two with the
most gravity torque (`shoulder_lift`, `elbow_flex`). Only `shoulder_lift` is validated over 5
randomized draws (below, 2.6 % mean error); on `elbow_flex` a 5-draw run (seeds 1000-1004) gave
2.4-55.7 % error, mean 16.4 %, so its `kp` is not validated. `wrist_flex` recovers to 17 % on a
single nominal draw (not separately validated); its torque range is roughly an
order of magnitude smaller. Friction is under-recovered on all three (54-71 % low) because the
settling transient of each hold crosses the friction dead zone rather than stopping inside it;
treat the fitted friction as a lower bound, not a measurement. Improving the friction fit needs a
slower approach, which is the first thing to change if the number matters.

`wrist_flex`'s own dead-band split is the outlier worth flagging on its own: against the nominal
virtual bus (true dead band 0), `fit_dead_band_ticks` fits an 18.5-tick pocket that is not there,
which zeroes out the split's own free-play estimate (0.00° against a true 0.87°, 100 % low). This
does not reach `arm.backlash_total_deg` — `servo_params.py` fills that field from the *combined*
(unsplit) figure whenever `ServoFit.dead_band_reachable` is `False`, which it is here (registered
dead band 0 ticks under nominal conditions), so the parameter fragment uses the combined column
above (1.03° vs 0.87°, 19 %), not this split. It is still a real identifiability gap worth
recording: `wrist_flex`'s own droop is small enough, relative to the dead bands this project
randomizes, that neither the split nor (on a large enough draw) the droop-line `kp`/`friction` fit
itself is backed by the same margin `shoulder_lift` and `elbow_flex` have — see the validation
section below.

### Validation: 5 randomized draws

`--validate N` draws `N` sets of `dynamics.kp`, `dynamics.frictionloss`, `dynamics.backlash_deg`
and `command.error_deadband_ticks` from `brick_builder.scene.randomize` (`level=1.0`, the full
published range), applies them with `randomize.apply` exactly as training does, and identifies one
named joint through each draw; `identify` runs with the nominal kinematic model here too (not the
randomized scene's own link masses), so `dynamics.link_mass_scale`'s confound with kp shows up in
the error below rather than being hidden. `kp`, friction, backlash and the dead band are single
values shared across every servo (one firmware configuration, guide 10), but the *recovery error*
below is only run on `shoulder_lift` — it does not automatically carry over to `elbow_flex` or
`wrist_flex`, which fit against their own gravity torque and their own droop-to-dead-band margin
(the per-joint table above already shows `wrist_flex`'s single-draw fit diverging from
`shoulder_lift`'s and `elbow_flex`'s). Read this table as `shoulder_lift`'s own randomized-draw
recovery, not a blanket number for every gravity-loaded joint:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/hardware/servo_sysid.py \
    --validate 5 --joint shoulder_lift --seed0 1000
```

| quantity | mean \|error\| over 5 draws, `shoulder_lift` | identifiable from the bus? |
|---|---|---|
| kp | 6.1 % | yes, on `shoulder_lift` (validated) and, by the per-joint table's single-draw fit, `elbow_flex` — not separately validated on `wrist_flex` (below) |
| friction | 63.4 % low | yes in principle, biased low in practice (see above) |
| free play, from the raw (unsplit) ramp hysteresis | 12.7 % | yes, on a gravity-loaded joint whose droop clears the dead band — see below |
| dead band (the split attempt, `fit_dead_band_ticks`) | 100 % (fits 0 every draw) | **no** |

**Dead band.** Not identifiable on `shoulder_lift` (validated over 5 draws) or, by the same
mechanism, `elbow_flex` — not because of a sampling-rate limit but because gravity droop keeps the
ramp's own position error outside the dead band at every point, over any dead band this project
randomizes. `fit_dead_band_ticks` looks for a zero-`Present_Load` pocket at the ramp's turning
point (inside the firmware's dead band the duty cycle should read exactly zero — `servo_model.py`'s
`_update_overload` sets `duty = 0.0` exactly whenever the position error sits inside the dead band),
and on `shoulder_lift` every one of the 5 draws fits 0, over true dead bands of 3.4-9.2 ticks;
checked directly on an 8-tick dead band, the ramp's steady-state position error stays 19-46 ticks
off target the whole way, never once inside ±8. `wrist_flex` is different: its own gravity droop
(3.5-8 ticks over the angles swept) is small enough to be comparable to a large randomized dead
band, so the split's "no pocket" answer there is not backed by the same margin, and `identify`'s
`registered_dead_band_ticks`/`min_droop_ticks` fields (`ServoFit.dead_band_reachable`) exist to
flag exactly this, per joint and per unit, rather than assume it from which joint it is. Separately,
against the *nominal* virtual bus (true dead band 0), `fit_dead_band_ticks` still fits a spurious
18.5-tick pocket on `wrist_flex` (`results/servo_bringup/sysid_wrist_flex.json`) where none exists —
a false positive from the split heuristic on this joint's particular ramp shape, not from a real
dead band. The split does find a real pocket, tens of encoder ticks wide, on a joint with
negligible gravity torque and a nonzero dead band — but that is `shoulder_pan` or `wrist_roll`,
exactly the two joints this method cannot identify kp on either (above). No joint gives this kit a
working dead-band split; fill `arm.servo_dead_band_counts` from the self-test's direct register
read (26/27) instead — that register is the one number this kit reads rather than infers, and
`servo_params.py` never uses `fit.backlash_deg` (the split's own output) to fill
`arm.backlash_total_deg`, so the spurious wrist_flex pocket above does not reach the parameter
fragment either way (next).

**Free play.** Recovers from the raw (unsplit) ramp hysteresis on `shoulder_lift` (validated, 12.7
% mean error). `elbow_flex` is not validated (25.5 % mean error on seeds 1000-1004). The ramp's own
combined hysteresis figure (droop already taken out) already is a free-play estimate on its own,
with nothing to subtract, because the dead band has zero effect on `shoulder_lift`'s ramp regardless
of what it randomizes to. Subtracting the registered dead band, as if it had contributed to the
ramp, would remove something the ramp never measured. `wrist_flex` is not validated this way — its
own droop can be small enough for the dead band to matter, so whether "nothing to subtract" holds
depends on the actual unit's registered dead band versus its own measured droop, computed per fit
(`ServoFit.dead_band_reachable`) rather than assumed from the joint's name. `servo_params.py` takes
the raw combined figure directly as `arm.backlash_total_deg` when `dead_band_reachable` is `False`;
when it is `True` it subtracts the registered dead band instead, itself only a partial correction
(the kp/friction fit above is biased in that case too, not just this field).

**Confound.** `dynamics.link_mass_scale` (0.9-1.1) is not separable from `kp` in the droop line —
both change the fitted `1/kp` the same way a real, unmeasured link mass would, and this kit has no
independent read of the arm's own link masses. The 6.1 % mean `kp` error above already carries this
confound (see the validation note); on real hardware, where nothing can be checked against a
ground-truth mass either, the same margin applies.

**On hardware:** run system identification on `shoulder_lift`, `elbow_flex` and `wrist_flex`, one
joint at a time, with the others holding (`shoulder_pan` and `wrist_roll` will raise — read their P
gain from the self-test's register table instead). Enter `kp`'s implied P value in
`arm.servo_p_gain` (cross-checked against the self-test's own register read), and see the
parameter-file fragment below for how the dead band and free-play fields are actually filled.
Re-run the sag and step-response figures of guide 03 afterwards.

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
unreachable, all three at the 100 mm radius — and specifically the three *centre* headings (-25°,
0°, +25°); both wide headings (±50°) reach at 100 mm. `tool_down_ik` (`brick_builder/scene/
compose.py`) fails its own convergence check at those three: position residual 0.91 mm at -25°,
7.73 mm at 0°, 0.67 mm at +25°. Two of the three residuals are under a millimetre, which does not
by itself distinguish a genuine kinematic reachability limit from an artifact of the solver's own
seeding or iteration count at that particular pose — this map does not run the solver with a
different seed or more iterations to tell those apart, so the cause of the gap is open rather than
asserted. Settling error grows with radius — 3.8-11.3 mm at 100 mm,
8.7-9.7 mm at 150 mm, 11.7-13.5 mm at 200 mm, mean 10.5 mm, p95 13.3 mm over the 12 reached points —
which is the expected shape for an open-loop-in-gravity command: the further out, the larger the
joint angles' contribution to tool error for the same angular droop. `RealArmWorld`'s gravity
feed-forward (the world adapters section above) is what the campaign's own controller runs with;
this map deliberately does not use it, because the point is to characterize the bare bus and servo
law before anything compensates for it.

## Parameter-file fragment

`scripts/hardware/servo_params_fragment.py` reads the self-test's and the sysid's `--out` JSON and
prints the `arm:` block in `configs/params/default.yaml`'s own
`value`/`status`/`measured_by`/`measured_on`/`notes` format, ready to paste. Nothing here writes to
`configs/params/` itself; pasting the printed block in is a manual step, and it loads through
`brick_builder.params.load_params` unchanged (checked by `tests/calib/test_servo_params.py`).

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
(flagged in the notes if the six servos disagree). `arm.backlash_total_deg` is the sysid's raw
(unsplit) ramp figure when `ServoFit.dead_band_reachable` is `False` for that fit — nothing
subtracted, because this joint's own droop never got close enough to the registered dead band for
it to matter (see the validation above) — and the registered dead band subtracted out as a partial
correction when it is `True`. Computed per fit, not assumed from which joint it is. Known defect:
on `shoulder_lift`, `dead_band_reachable` is `True` on 2 of the 5 validation draws (seeds 1000 and 1001), and the
fragment then writes `backlash_total_deg = 0.0` against a true 0.71-0.72°; `min_droop_ticks` comes
from the droop sweep, not from the ramp the rule judges. Check `backlash_total_deg` by hand until
this is fixed. `arm.joint_friction_Nm` is the sysid's friction fit,
always written with `status: derived` (never `measured`, even against real hardware — the
settling-transient bias is a property of the method, not of whether the bus is virtual), noted as a
lower bound.

Against the virtual bus this is the kit's self-consistency check, not a claim that every number
comes back exact: `servo_p_gain` (16) and `servo_dead_band_counts` (0) match `ArmConfig()`'s
defaults exactly, because those two are register reads; `results/servo_bringup/params_fragment.yaml`
has the run this guide's numbers come from, with `backlash_total_deg` and `joint_friction_Nm` from
the `shoulder_lift` fit above (41 % and 71 %-low against their true values, carried through rather
than hidden).

## Measurements for the day the arm arrives

Each row is one procedure and the parameter-file entry it fills.

| Measurement | Procedure | Fills |
|---|---|---|
| Self-test | `arm_selftest.py --port …` on the assembled bus, arm supported throughout. Ping/registers/gate `PASS` on every present joint before anything else runs; any `FAIL` (commanded move on any joint, missing servo, voltage/temperature gate, overload trip) stops the bring-up until explained; the virtual-bus `FAIL` on `shoulder_lift` and `elbow_flex` is an open simulator defect (self-test section above). | `arm.servo_p_gain`, `arm.servo_dead_band_counts` (also read here, see below) |
| Homing offsets and directions | Drive each joint to both stops with torque off, log `Present_Position` at each, and set `JointCalibration.centre`/`direction` so the model's zero and sign match. **This is the one calibration that has no shipped value worth trusting.** | `JointCalibration` (code), to be mirrored into the parameter file |
| Servo system identification | `servo_sysid.py --port … --joint <name>`, on `shoulder_lift`, `elbow_flex` and `wrist_flex` (`shoulder_pan`/`wrist_roll` raise -- their kp comes from the self-test's register read instead). | `arm.joint_friction_Nm`; `arm.backlash_total_deg` from the raw ramp figure on these three joints (see the parameter-file fragment) |
| Settling-error map | `servo_workspace_map.py --port …`. Diagnostic, not a parameter fill: a large or uneven error against the virtual bus's own map above is the sign to stop and debug before anything else. | — (a cross-check) |
| Parameter-file fragment | `servo_params_fragment.py`, fed the two `--out` files above. | the `arm:` block, ready to paste |
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
| default | 16/65 | 0.594 | 1.05 | 7.46 | 1.498 |
| perturbed pose (+4/-3/+2 mm, ~2°) | 35/65 | 0.528 | 0.59 | 5.15 | 0.451 |
| perturbed fov (100° diagonal) | 22/65 | 1.003 | 0.40 | 6.46 | 2.027 |

(`tsai`, seeds 0/1/2; the 180° flip check correctly reports "not flipped" in all three -- 1.5°,
0.5°, 2.0° to the expected pose, in table order.) No case's reprojection RMS or focal error stands
out from the other two's here. An earlier measurement had the perturbed-pose case visibly worse on
both (1.021 px/6.15%, against 0.668-0.742 px/0.32-1.26% for the other two), attributed at the time
to the checkerboard intrinsics fit; that attribution was wrong, and so was the remedy it proposed (a
genuine `cv2.calibrateHandEye` binding or a better-conditioned intrinsics fit) -- given clean
corners both the intrinsics fit and the closed-form hand-eye solve are accurate. The actual cause
was `find_corners`' checkerboard corner refinement using a single fixed search window that overran
neighbouring corners on that pose set's foreshortened far rows; `find_corners` now sizes that
window from each image's own detected corner spacing instead, which brought the perturbed-pose row
back in line with the other two above (`docs/guide/03`'s wrist-camera section has the full account,
including the isolated hand-eye-solve measurement that confirms the solve itself was never the
weak step).

**This does not yet demonstrate resolving a real few-mm mount offset.** Every case's recovered
position error in the full-pipeline table above (5.15-7.46 mm) is at or above both the
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

- `scripts/hardware/force_ramp.py`: the procedure is in the table above and the driver exposes
  what it needs (`Bus.cycle` with a per-joint torque limit), but the command-line wrapper does not
  exist yet.
- Watchdog and emergency-stop wiring. The driver's soft stop is a software stop on a live bus; it
  is not a substitute for cutting power.
