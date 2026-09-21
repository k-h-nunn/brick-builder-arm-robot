# Robot and servo model

This is the part of the project that models the SO-101 arm itself: its geometry, the collision shapes
physics actually uses, and a corrected model of how its servos behave. The manufacturer's own
simulation file gets that last part wrong in a way that matters a great deal for pressing bricks.

Code: `brick_builder/robot/patch_so101.py`, `brick_builder/robot/servo_model.py`,
`brick_builder/robot/bus_emulator.py`, `brick_builder/robot/kinematics.py`, `brick_builder/robot/force_map.py`.

## Model overview

The arm is vendored, not authored. `third_party/so101/so101_new_calib.xml` comes straight from
TheRobotStudio's SO-ARM100 repository at a pinned commit, and nothing here ever edits it. Every
correction is applied in memory to a `mujoco.MjSpec` when the model is loaded, so a new upstream
release can be dropped in and the same patches re-applied without a merge.

```python
from brick_builder.robot.patch_so101 import ArmConfig, build_arm_spec, attach_arm

spec = build_arm_spec(ArmConfig())        # patched, uncompiled MjSpec
model = spec.compile()

# or drop the arm into a scene you are already building
import mujoco
world = mujoco.MjSpec()
attach_arm(world, ArmConfig(), pos=(0, 0, 0))
```

What `build_arm_spec` changes:

| Patch | Why |
|---|---|
| Position actuators at `kp = servo_kp(P, Vin)`, `kv = 0` | Upstream's 998 is about 100x too stiff (below) |
| `forcerange = ±torque_limit_nm · Vin/7.4` | Upstream's ±3.35 is a 12 V number |
| Joint damping 0.62, frictionloss 0.052, armature 0.026 | Rhoban BAM's identified values for this servo |
| Optional backlash joint beside each actuated joint | Real gear play is larger than a stud's capture window |
| Boxes instead of collision meshes, plus two fingertip pads | A mesh's convex hull fills the jaw gap (below) |
| `tcp_pinch` site, `wrist_cam` camera on the `gripper` body | The tool point and the camera the policy sees through |
| implicitfast, elliptic cone, impratio 10, Newton, noslip 0 | Design report 01 section 1; noslip 0 also keeps a GPU port possible |

Names you can rely on: `ARM_JOINTS` (the six actuated joints in servo-ID order), `TCP_SITE`,
`WRIST_CAM`, `FIXED_PAD_GEOM`, `MOVING_PAD_GEOM`. `attach_arm` keeps those names unprefixed.

## Upstream servo gain error

The official file sets a position gain of 998.22. That number comes from an upstream derivation that
assumes the servo firmware measures its position error in degrees. It does not: the firmware works in
encoder ticks and its internal constant was measured with an oscilloscope by the Rhoban BAM project.
The measured law is

```
duty = clip(error_rad · P · 0.166 · ratio, ±0.97)
tau  = kt · (Vin · duty − kt · dq) / R
```

Near zero error that is a plain spring, `tau = kp · error`, with

```
kp = (kt / R) · Vin · 0.166 · ratio · P
```

With BAM's identified constants for the 7.4 V STS3215 (`kt = 1.1776`, `R = 2.4787`,
`ratio = 1.0049`) this gives **9.38 N·m/rad at P = 16**, the value LeRobot writes on connect, and
**18.77 at P = 32**, the factory default. Upstream's 998 is roughly a hundred times stiffer than the
part it claims to model.

| Source | kp at P=16 | kp at P=32 |
|---|---|---|
| Upstream simulation file | 998.22 (for any P) | 998.22 |
| Rhoban BAM measured law (what this model uses) | 9.38 | 18.77 |
| Independent bench test at 12 V | not measured | about 38 (BAM predicts about 30) |

This matters because press force is `kp · error` up to saturation. Get the gain wrong by 100x and a
simulated press seats every brick effortlessly, which is exactly the sim-to-real gap that would only
show up on real hardware.

Two more numbers fall out of the same constants and both match the design report:

- **Joint damping** is the back-EMF term `kt²/R = 0.559` plus BAM's mechanical viscous friction
  `0.060`, so `0.619`. Putting it in the joint rather than in the actuator's `kv` reproduces BAM's
  torque law exactly and leaves the actuator a pure position spring, which is what ports to a GPU
  pipeline later.
- **Saturation.** At full duty the law reaches `kt · Vin · 0.97 / R = 3.41 N·m`, while the datasheet
  says the 7.4 V part stalls at 1.9. The two disagree and nobody has measured ours, so `forcerange`
  defaults to the conservative 1.9 and the randomization range spans both.

### Sag: the servo as a spring

![Static sag against the predicted tau/kp](../media/robot/sag.png)

Hang a known torque on `shoulder_lift` and let it settle. The joint gives way by `tau / kp` to within
a few percent over 0.4 to 1.6 N·m, and doubling P halves the sag. The dashed line at the bottom is
what upstream's gain would give: 0.06 degrees at 1 N·m, where the real servo gives about 6.

Reproduce it with `brick_builder.robot.servo_model.measure_sag`.

### Step response

![Step response: the rate limit on the left, the gear play on the right](../media/robot/step_response.png)

A 0.5 rad step on `shoulder_lift` settles into a 2% band in about 0.50 s and overshoots by about
10%. The original design spec assumed an over-damped response. That assumption does not hold for the
identified servo constants: over-damped would need damping of at least `2·sqrt(kp·I)`, which is
0.99 N·m·s/rad for the 0.026 kg·m² armature alone, against the 0.62 the BAM law actually gives. No
joint of this arm is over-damped at P = 16. The firmware's own rate limit does not rescue it either, because 5.2 rad/s
crosses a 0.5 rad step in 0.1 s while the natural period is around 0.3 s. `wrist_flex` is milder:
about 4.6% overshoot, 0.43 s settling. `tests/robot/test_servo_model.py` now asserts the measured
character directly (overshoot at most 15%, 2% settling under 0.65 s for a 0.5 rad step) instead of
keeping the over-damped assumption as an xfail; system identification on the real arm will calibrate
these numbers further once hardware is available.

The right-hand panel is the free play. With the loop closed on the output shaft the play leaves no
standing offset — the trace crosses zero and keeps crossing it — but it does leave a hunt across
the gap with a period of about a second. That hunt decays: at the nominal gain the per-second
peak-to-peak falls from 9.2 counts at one second to 0.7 at eight. At the top of the randomized gain
range (20 N·m/rad) it stops decaying and settles into a flat limit cycle of 11.8 counts, 1.02
degrees at the joint, which is steady to within 0.2 counts over twenty seconds. That is a real
buzz, measured with gravity off, where nothing preloads the play against a stop; with gravity on
the same joint buzzes about half as far. Account for it before reading a tracking error off a
stationary arm. `servo_model.step_response_sweep` reports it for every
joint at every corner of the gain range, and `tests/robot/test_servo_model.py` bounds it.

## Changing the voltage and the gain

Everything lives in `ArmConfig`, which is frozen, so make a new one rather than mutating it.

```python
from brick_builder.robot.patch_so101 import ArmConfig, build_arm_spec, servo_kp

servo_kp(16, 7.4)                 # 9.38  N*m/rad
servo_kp(32, 7.4)                 # 18.77
servo_kp(16, 12.0)                # 15.22, because kp is linear in supply voltage

cfg = ArmConfig(p_gain=32, vin=6.9, torque_limit_nm=1.9)
model = build_arm_spec(cfg).compile()
```

- `p_gain` is the servo's P register (address 21). It lives in EPROM, so on the real arm it is not
  something to rewrite every tick. Change it at skill boundaries only.
- `vin` scales both `kp` and `forcerange`, which is what a sagging battery does. The randomization
  range for training is 6.8 to 7.6 V.
- `torque_limit_nm` is the stall torque **at 7.4 V**; the forcerange written into the model is scaled
  by `vin / 7.4` from there.
- `backlash_deg` is the **half-range** of the free-play joints, so total lost motion is twice it;
  `0` removes them entirely, which is useful when you want to isolate the actuator law in a test
  and is also the configuration under which this model is bit-for-bit what it was before the
  encoder correction.

The per-tick compliance knob is a different register. `Torque_Limit` (address 48) is RAM and can be
written every tick, so it is exposed through `ServoBus.write(goal, torque_limit=...)` as a fraction of
the forcerange, per joint. That is what a policy uses to press gently and then firmly.

## Collision geometry and grasp failure

Upstream uses the full visual mesh as collision geometry. MuJoCo collides against a mesh's **convex
hull**, and the hull of `wrist_roll_follower` (the fixed jaw) spans the opening between the jaws. The
pinch point is therefore inside solid geometry and a brick placed there starts about 8 mm
interpenetrated. It gets squirted out rather than held, and it never touches the fingertips at all.
`tests/robot/test_patch_so101.py::test_mesh_collision_cannot_grasp_because_the_hull_fills_the_jaw`
keeps that on the record; set `ArmConfig(primitive_collision=False)` to watch it happen.

With `primitive_collision=True` (the default) every collision mesh is replaced by convex primitives
fitted to the STL vertices at load time:

- Each arm link part gets one box in its own mesh frame. Boxes are much tighter than capsules on
  these printed parts, which are boxy rather than round.
- Each jaw is cut into slabs along the finger and fitted with one box per slab, so the staircase
  follows the taper and cannot bridge the gap the way a single hull does.
- The tip of each finger gives up its inner 1.2 mm to a **pad**: `condim 4`, friction from
  `ArmConfig.pad_friction`, and `priority 1` so the pad's friction wins over whatever it touches.

The fit is done by a small binary-STL reader in `patch_so101.py` (`read_binary_stl`). No trimesh, and
nothing is read at import time. A test checks the reader lands vertices exactly where MuJoCo's own
loader does.

### The TCP and the grasp it assumes

`tcp_pinch` sits between the fingertips, with its z axis pointing outwards along the approach
direction. It is placed at the centre of a nominal 32 mm brick held by the fingertips, hanging 8 mm
past them.

That overhang is not arbitrary. The moving jaw swings on a hinge, so the two pad faces are never
parallel: at a 32 mm opening the gap is widest right at the tips and narrows going up the finger. A
32 mm brick cannot be pushed deep into the jaw, and the tips are where it sits. `gripper_angle_for_width`
returns the joint angle at which the **narrowest** part of the gap equals a given width, which is the
number you want when deciding whether something fits.

With the primitive geometry the arm closes on a 32 mm, 12.2 g brick, squeezes it with about 17 N, and
holds it against gravity with a slip of half a millimetre over two seconds. Pulling with 2 N barely
moves it and 20 N rips it out, so the hold is friction rather than anything glued.

## The wrist camera

The real hardware is a 32x32 mm USB UVC camera module on TheRobotStudio's own **SO101 Wrist Cam
Hex-Nut Mount** -- upstream's `README.md` (section "Wrist-Mount Cameras") lists three 32x32-UVC
wrist mounts and labels only this one "(SO101)"; the other two are a one-piece replacement for the
wrist-roll part itself (shared SO100/101) or SO100-only. Vendored at
`third_party/so101/wrist_camera_mount/` (STL, STEP, the upstream README, an installed-hardware
photo, `VENDORED.md`).

A camera pose picked earlier in this project's history (a position aimed straight at `tcp_pinch`,
41.4 deg vertical fov -- an OV5647 number unrelated to this hardware) put the fixed jaw at 80-85% of
the frame. That guess is now replaced by geometry read directly off TheRobotStudio's own
camera-equipped assembly for this exact hardware: `Simulation/SO101/so101_new_calib_camera.xml` /
`.urdf`, vendored at the same commit as `so101_new_calib.xml`, chains `wrist_camera_mount` then
`wrist_camera` off `gripper` as two fixed (jointless) bodies -- both welded, so composing them into
one gripper-relative camera pose is exact *for the CAD-nominal design*. `patch_so101.py`'s
`WRIST_CAM_POS_IN_GRIPPER` / `WRIST_CAM_QUAT_IN_GRIPPER` are that composed transform; their own
docstrings carry the full chain and the two independent checks used to confirm the lens/forward
axis (z-profiling the board mesh, and a probe camera rendered down the derived axis).

**Pose, in the gripper body frame:** position `(2.5, 73.4, 6.6) mm`; forward axis
`(0, -0.424, -0.906)` -- mostly along the fixed jaw's own pointing direction (`-z`), tilted about
25 degrees toward the palm side (`-y`). At the nominal grip this axis passes about 11 degrees from
`tcp_pinch` itself: close, but the real mount is bolted on with a fixed geometric relationship to
the jaw, not independently aimed at the grasp point the way the old guess was --
`tests/robot/test_patch_so101.py::test_wrist_camera_pose_matches_the_derived_mount_constants` pins
the exact axis and `..._is_on_the_gripper_body_and_sees_the_tcp` checks the (now merely
approximate) 20-degree bound.

**Uncertainty.** This is a CAD-nominal derivation, not a measurement on assembled hardware:

- *Position*, roughly +-3-5 mm: FDM print shrinkage is negligible (well under a millimetre over a
  66 mm bracket) but the M3-screw-through-hex-nut mounting has real clearance, and the sensor's
  exact placement inside a specific purchased "32x32 mm UVC module" board is not part of the CAD
  (that class of module is sourced from more than one vendor).
- *Orientation*, roughly +-3-5 degrees pitch/yaw for the same reasons; the lever arm from the
  gripper origin to the camera (~74 mm) turns a couple of degrees of play at the mount into a few
  mm at the lens.
- *Roll* (which way is "up" in the image) is fixed by the rigid mount and not independently
  chosen, except for one binary risk this derivation cannot rule out: a specific purchased module
  with its sensor die soldered rotated 180 degrees on its own board would flip the image upside
  down. That is a software/firmware fix (rotate the frame), not a pose error.

**Calibration procedure, for once real hardware exists.** Print a small checkerboard (9x6 internal
corners, 15 mm squares fits this arm's short working distance) and mount it rigidly at a known
survey position. Drive the arm to 15-20 varied poses (`ArmKinematics.fk` gives the gripper body's
pose in the base frame at each one) and capture a wrist-cam frame at each. `cv2.calibrateCamera`
against the checkerboard corners gives the real intrinsics (fx, fy, cx, cy and distortion) --
replacing `WRIST_CAM_FOVY_DEG`'s class-level estimate with a measured one and giving real
coefficients to compare against `WRIST_CAM_DISTORTION_K1_RANGE`/`_K2_RANGE`. For the extrinsic pose
(the number this section is actually about), feed each shot's gripper-to-base transform (from FK)
and its checkerboard-to-camera transform (`cv2.solvePnP` per image) into `cv2.calibrateHandEye`,
which solves exactly the `AX = XB` problem this is -- a constant camera-in-gripper transform seen
through many different gripper poses. Convert its result from OpenCV's camera convention (+z
forward, +y down) to MuJoCo's (-z forward, +y up; a fixed 180-degree flip about the camera's own x
axis) before comparing it to `WRIST_CAM_POS_IN_GRIPPER`/`WRIST_CAM_QUAT_IN_GRIPPER`, and use the
difference to refine the uncertainty budget above with a real number.

**Lens.** The specified lens is in the "~120 degree" wide-angle class, stated range 100-130
degrees measured on the sensor **diagonal**. `WRIST_CAM_DIAGONAL_FOV_RANGE_DEG = (100.0, 130.0)` is
that range (use it for domain randomization); `WRIST_CAM_DIAGONAL_FOV_DEG = 115.0` (the midpoint)
is the nominal. A MuJoCo camera only takes a vertical `fovy` and derives the horizontal extent from
the render's own aspect ratio, so a diagonal spec has to be converted for a specific resolution
before it means anything to MuJoCo; `fovy_from_diagonal_fov(diagonal_deg, width, height)` does that
for any aspect ratio (4:3, 16:9, or the square renders the scene model takes), from
`tan(diag/2)**2 == tan(fovx/2)**2 + tan(fovy/2)**2` with `tan(fovx/2) == aspect * tan(fovy/2)`. The
compiled model's own default, `WRIST_CAM_FOVY_DEG` (86.6 deg -- 115 deg diagonal at 640x480 works
out to a *smaller* vertical number than the diagonal because 4:3 spends part of that diagonal
horizontally), is that formula at 640x480 (4:3) -- the Optional-folder README's own recommended
operating resolution for this class of module, and also `brick_builder/envs/align_env.py`'s wrist-cam
render size (160x120, also 4:3). At the range's endpoints the same formula gives 71.1 deg (100 deg
diagonal) to 104.3 deg (130 deg diagonal). A caller rendering at a different aspect ratio must
recompute fovy for it, not reuse `WRIST_CAM_FOVY_DEG` as-is.

Barrel distortion is represented the way the rest of the codebase already represents it --
`render.domain_rand.PostProcessConfig.k1`/`k2`, the existing image-space radial remap -- rather
than modelled again here. A lens this wide typically still shows substantial barrel distortion
even after any in-lens correction; `WRIST_CAM_DISTORTION_K1_RANGE = (-0.35, -0.10)` and
`WRIST_CAM_DISTORTION_K2_RANGE = (0.0, 0.08)` are an engineering estimate for that (predominantly
negative `k1`, in `domain_rand`'s own convention, is what makes image content compress toward the
edges), not a measured number -- refit them once the calibration procedure above has real images.
`render/domain_rand.py`'s own default range is untouched (this module does not edit that file);
a caller wanting this camera's distortion represented more accurately passes these two ranges
through `sample_postprocess_config`'s own `ranges` argument.

**Measured image composition**, rendered at 1280x960 (full resolution, not a thumbnail) and
measured by instance segmentation (`brick_builder.render.renderer.SceneRenderer`, gripper + moving-jaw
bodies counted together as "jaws"):

| Scene | jaws | held brick | plate mesh | notes |
|---|---|---|---|---|
| `wrist_camera_home.png` -- arm at qpos 0, empty gripper | 14.6% | -- | 0% | the arm's rest configuration is not aimed at anything in particular |
| `wrist_camera_baseplate_view.png` -- tool-down, 7 cm above the plate centre, 6 bricks placed | 14.6% | -- | 3.8% (bricks cover much more of the visible plate area) | |
| `wrist_camera_tray_view.png` -- tool-down, 7 cm above the loose-brick tray, 3 bricks | 14.6% | -- | 4.5% | tray + 3 bricks clearly visible |
| `wrist_camera_held_brick.png` -- default scene, a 2x4 brick held | 14.4% | 2.8% | 8.9% | |

![Arm camera POV, home pose](../media/robot/wrist_camera_home.png)
![Arm camera POV, baseplate viewing pose](../media/robot/wrist_camera_baseplate_view.png)
![Arm camera POV, tray viewing pose](../media/robot/wrist_camera_tray_view.png)
![Arm camera POV, holding a 2x4 brick](../media/robot/wrist_camera_held_brick.png)

The jaw share is close to constant across every pose (14.4-14.6%) because it is an *angular*
constant of a camera rigidly attached to the same body as the jaw. An earlier viewing-pose
calibration, done before the mount geometry was known, put the jaw share at 80-85%; the 14-15%
here is what it becomes once the pose and the wide-angle field of view both match the real mount
instead of a guess.

## Gear free play and encoder position

A real STS3215 was measured with about 0.87 degrees of total gear play at the output shaft, against
a specification of 0.5 degrees. The model gives every actuated joint a second hinge on the same
axis, limited to `±backlash_deg`; two hinges about one axis compose to a single angle, so this is
upstream's dummy-body pattern with one fewer body. The actuator drives the first hinge, the link
hangs off the second, and motor torque reaches the link only once the slack has been taken up.

**`backlash_deg` is a half-range.** Total lost motion is twice it. The 0.87 degrees is a total — a
dial gauge on an 86 mm lever read 1.3 mm end to end — so the default is 0.435. The old default of
0.5 was the datasheet's own total being read as a half-range, which put twice the specified play in
the model.

### Encoder position, relative to the gearbox

The STS3215's 12-bit magnetic encoder sits on the **output shaft**, past the 1:345 reduction. That
is the only arrangement in which one absolute 0–4095 reading can cover one output turn, and it is
what independent bench measurements of the part say outright. So `Present_Position` is the angle of
the *link*, the firmware's proportional law is closed on the link, and `ServoBus` reports and
controls the driven joint **plus** its free-play partner.

This model used to put the encoder on the motor side. The difference is not a detail:

| | encoder on the motor side (wrong) | encoder on the output shaft (hardware) |
|---|---|---|
| Steady-state placement error at the tool, nominal servo | 3.6 mm (2.5–4.5) | 0.43 mm (0.07–0.96) |
| ditto, randomized servos | 6.0 mm (2.9–8.1) | 0.42 mm (0.05–1.3) |
| How much of that error the robot can see | none: the encoder gap is 4.0 mm of a 3.6 mm error | all but 0.16 mm |
| Loaded clockwise-vs-counter-clockwise deviation | 2·τ/kp, play invisible | 2·τ/kp, and the play shows up inside the dead band |

Measured over 49 tool-down poses, 0.13–0.22 m reach, three heights, five bearings. The row that
matters is the third: with the loop closed on the output shaft the servo **takes the slack up itself**, so
gear play is not a standing placement error that nothing on the robot can see. What play costs is
dead time — the motor crosses the gap before any torque reaches the link — and freedom inside the
firmware's dead band, where the servo commands nothing at all and the link can drift anywhere in
the slack. That freedom is exactly what the dial-gauge measurement reads.

What *is* still invisible on the real arm is link and horn flex, spline play outboard of the
encoder, and calibration error. Gear play is not on that list any more.

At the home pose one joint's worth of play moves the tool 6.1 mm for `shoulder_pan`, 6.0 mm for
`shoulder_lift`, 5.1 for `elbow_flex`, 2.8 for `wrist_flex` and essentially nothing for `wrist_roll`
and `gripper`, so a DUPLO stud's 0.3 mm capture window is still an order of magnitude tighter than
one gap. That number now bounds a transient, not a resting offset.

### Checking the model against the bench

`tests/robot/test_servo_model.py` reproduces the published measurements on the 12 V part at what
the numbers imply is the factory default P = 32 (kp = 30.4 N·m/rad):

| measurement | bench | this model |
|---|---|---|
| droop at 10 kg·cm | ≈ 22.5 counts | 21 counts (τ/kp = 1.85°) |
| droop at 15 kg·cm | ≈ 30 counts | 31 counts (τ/kp = 2.77°) |
| CW-vs-CCW deviation, loaded ≈ 3 kg·cm | 14.78 counts | 12 counts at 3.0 kg·cm; 14.78 implies 3.5 kg·cm |
| free play, gentle push, dead band open | 0.87° | 1.06° (2·`backlash_deg` plus two counts of solver compliance) |
| unloaded repeatability | ±2 counts | quantization, ±0.5 count |

The CW/CCW row needs a careful reading. With the encoder on the output shaft a *loaded,
powered, settled* deviation measures twice the elastic droop and nothing else — sweeping
`backlash_deg` from 0 to 1.0 changes it by less than one count. The bench's own 14.78 is the
identified law at a slightly heavier load than "about 3 kg·cm", not a measurement of the gear play.

## The servo bus

`ServoBus` sits between a policy and `data.ctrl` and makes the arm behave like six serial servos
rather than six ideal joints.

```python
from brick_builder.robot.servo_model import ServoBus, ServoConfig

bus = ServoBus(model, data, cfg=ServoConfig(), rng=np.random.default_rng(0))
for tick in range(n_control_ticks):
    bus.write(goal_rad, torque_limit=compliance)   # once per control tick, 50 Hz
    for _ in range(10):                            # ten physics steps per tick at dt 2 ms
        bus.substep()
        mujoco.mj_step(model, data)
    readings = bus.read()
```

A command goes: quantized to encoder ticks, dropped if it moved less than the dead zone, queued
behind the wire delay, then chased by the firmware's own internal target which can only slew at about
5.2 rad/s. What comes back is quantized too, with each servo sampled a little later than the one
before it. One count is 0.088 degrees, both ways.

`data.ctrl` is not the firmware's target. MuJoCo's position actuator computes
`kp · (ctrl − qpos_driven)` and the error the firmware works on is `target − (qpos_driven +
qpos_play)`, so the bus writes `target − qpos_play` and the two agree; `ServoBus.target` is the
firmware number. With no free-play joints the two are the same and the write is unchanged, which is
what makes `backlash_deg = 0` bit-for-bit identical to the model before the encoder correction.

| Field | Default | Randomization range | What it is |
|---|---|---|---|
| `ticks_per_rev` | 4096 | fixed | Encoder resolution |
| `deadband_ticks` | 0 | 0 to 10 | Command dead zone: a *goal change* smaller than this is dropped |
| `error_deadband_ticks` | 0 | 0 to 10 | Firmware position dead band, registers 26/27: no correction while the error is inside it |
| `encoder_on_output` | `True` | fixed | Where the encoder is. `False` is the motor-side model this project ran before the correction, kept only so the two can be measured against each other |
| `rate_limit` | 5.29 rad/s | 4 to 6 | Firmware target slew (BAM's identified `max_velocity`) |
| `action_delay_s` | 0.008 | with jitter, 4 to 12 ms | Command latency |
| `action_delay_jitter_s` | 0.004 | | Uniform, per write |
| `read_skew_s` | 0.0005 | from the bus model | Per-servo sampling offset inside one read |
| `load_lsb` | 0.001 | fixed | `Present_Load` step, 0.1% of duty |
| `current_lsb` | 0.0065 | fixed | `Present_Current` step, 6.5 mA |
| `overload_duty` / `overload_time_s` / `overload_derate` | 0.95 / 1.5 s / 0.2 | | Emulated overload protection |
| `read_noise_ticks` | 0 | 0 to a few | Extra read noise |
| `vin` | 7.4 | 6.8 to 7.6 | Supply the duty estimate assumes |

Two details to know before reading any of this off the real robot:

- **`Present_Load` is a PWM duty cycle, not a torque.** It saturates at 0.97 while the joint torque
  saturates at whatever `forcerange` says. `ServoReadings.load` reports the duty, in 0.1% steps, and
  a test pins both facts down.
- **The dead band is a guess until somebody reads it.** `error_deadband_ticks` defaults to 0, the
  same convention the command dead zone uses: nominal means the identified law with nothing added.
  `servo_model.STS3215_ERROR_DEADBAND_TICKS` is 2 counts, the value the published measurements
  support — the bench report says 10 counts, but its own unloaded repeatability of ±2 counts rules
  that out, because a servo that stopped correcting anywhere inside ±10 counts could not return to
  the same count twice. Read the real value off your own servos (below) before trusting either.
- **Overload protection latches.** Duty above 0.95 held for 1.5 s trips the servo, after which its
  forcerange is derated to 20% until `reset_overload()` is called. In the press environment that is a
  termination condition, not something to train through.

## The outer correction

The firmware is a proportional law with nothing behind it, so it leaves a standing error wherever
something pushes back: dry friction in the gear train (about 0.05 N·m against a 9.4 N·m/rad gain, a
dead zone of roughly ±0.32 degrees per joint) and its own dead band, inside which it commands
nothing at all. Gravity has a feedforward term already; neither of the others can have one, because
where inside a dead zone a joint stops depends on which way it came from.
`brick_builder/robot/joint_correction.py` closes that at the host, at the 50 Hz control tick, on the
angles the bus reports — which is only meaningful now that those angles are the link's.

```python
from brick_builder.robot.joint_correction import JointCorrection, JointCorrectionConfig

corr = JointCorrection(5, JointCorrectionConfig())     # five joints: the gripper is left alone
for tick in range(n):
    corr.update(q_desired, readings.position[:5], freeze=pressing_or_gripping)
    bus.write(np.concatenate([gravity_feedforward(q_desired) + corr.correction, [grip]]))
```

One clamped integrator per joint and nothing else. `update` returns `desired + correction`; a
caller with its own feedforward adds `corr.correction` to that instead, so the integrator sees the
plain tracking error rather than re-deriving the gravity term. It is written for a C99 port: fixed
arrays, one loop, no allocation, and the returned buffer is reused between calls.

| Field | Default | What it is |
|---|---|---|
| `ki` | 1.0 /s | Integral gain. Swept over the workspace grid; 1.0 is the lowest mean error of anything tried, and 2.0 and 4.0 give most of it back as a limit cycle |
| `clamp_ticks` | 40 | Largest trim any joint may accumulate (3.5°). Settled it is 2–6 counts; the clamp binds transiently on 5 of 49 poses during a step across the whole workspace, which is the windup it exists to bound |
| `deadzone_ticks` | 1.0 | Below one count the reading does not carry the sign of the error |
| `dt` | 0.02 s | Seconds per call |

Three things to know before switching it on.

- **The gripper is not in it.** Its command is a squeeze past the object's face, so its position
  error is large, deliberate and permanent. Build the corrector with five joints.
- **Freeze it while the arm leans on something.** An error the arm is *meant* to have — a tool
  pushed into a brick, jaws closing on a face — would otherwise wind the trim up and release it as
  a lurch when the contact ends. `ArmDriver` freezes automatically on any tick that commands a
  compliance below 1.
- **The dead zone has to clear the read noise.** With `read_noise_ticks` at the top of its
  randomization range (2 counts) and the one-count dead zone, the trim chases noise and the mean
  settling error goes from 0.42 mm untrimmed to 0.80 mm trimmed. Raise `deadzone_ticks` above the
  noise or leave the trim off.

Where it earns its place is a firmware dead band, which is the one error the proportional law
cannot remove by itself:

| firmware dead band | untrimmed | trimmed | settling time |
|---|---|---|---|
| 0 counts | 0.43 mm | 0.35 mm | 0.84 s → 1.79 s |
| 2 counts | 0.51 mm | 0.36 mm | 0.84 s → 1.77 s |
| 7 counts | 1.12 mm | 0.44 mm | 0.76 s → 1.80 s |

Mean over the same 49 poses, nominal servo. The trim roughly doubles the time to arrive and adds a
visible buzz — mean 0.33 mm of peak-to-peak tool motion in the last second against 0.19 mm — which
is the price of integral action against a friction dead zone and is not going away.

![Tool settling error, before and after](../media/robot/settling_error.png)

Reproduce the whole thing with:

```bash
.venv/bin/python -c "
from brick_builder.robot.joint_correction import measure_settling, summarize
print(summarize(measure_settling()))"
```

## Reading the real numbers off your own servos

Four registers decide everything on this page, and none of them has been read off the hardware yet.
All four are in the STS3215's memory table and all four are reachable from either Feetech's own
tool or the LeRobot driver the project already depends on.

| Register | Address | What it decides here |
|---|---|---|
| P coefficient | 21 | `ArmConfig.p_gain`, hence `kp` and every press force |
| CW dead band | 26 | `ServoConfig.error_deadband_ticks` |
| CCW dead band | 27 | the same, in the other direction |
| Torque limit | 48 | the per-tick compliance knob |

With FD.exe / FT SCServo Debug (Feetech's own Windows tool): connect at 1 Mbps, select each servo
ID in turn, open the EPROM page and read addresses 21, 26 and 27 straight off it. Write them down
per servo — they are set per unit at the factory and a second-hand arm may not be uniform.

From Python, with LeRobot installed:

```python
from lerobot.common.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors={...})   # your six, in ID order
bus.connect()
for name in bus.motors:
    print(name,
          bus.read("P_Coefficient", name),
          bus.read("CW_Dead_Band", name),
          bus.read("CCW_Dead_Band", name))
```

If the register names differ in your version, read the raw addresses instead: 21, 26, 27, one byte
each. Then put them in the parameter file — `arm.p_gain` exists already; the dead band wants a new
entry — and re-run the sag and step-response measurements against your own arm.

Two measurements are worth making at the same time, because they are what the model is fitted to:

- **Dead band, directly.** Command a hold, then push the output shaft by hand with a torque small
  enough not to back-drive the gearbox, and log `Present_Position` at both extremes. What you read
  is the free motion: the gear play if the dead band is wider than it, the dead band if it is not.
  Repeat with a dial gauge on a lever to separate horn and spline play, which the encoder cannot
  see, from gear play, which it can.
- **Droop.** Hang known masses on a lever and log `Present_Position` against torque. The slope is
  `1/kp`, and `kp` from the law at the P value you just read should land on it. If it does not, the
  law's `0.166` firmware factor is the thing to refit.

## Bus timing

`bus_emulator.py` is the timing half, with no MuJoCo in it at all. It models one 1 Mbps half-duplex
line carrying one sync-write and one sync-read per control tick.

```python
from brick_builder.robot.bus_emulator import BusConfig, BusEmulator

bus = BusEmulator(BusConfig(), rng=np.random.default_rng(0))
timing = bus.tick(t_now)
timing.command_latency_s   # 260 us
timing.read_skew_s         # 2.05 ms across six servos
timing.reply_lost          # per servo
```

The durations are consequences of the frame sizes rather than tuned constants. A 2-byte
Goal_Position sync-write to six servos is 26 bytes, which at 10 µs per byte is **260 µs**. A
sync-read of addresses 56 to 70 is a 14-byte request plus six 21-byte replies, so 140 bytes and
**1.4 ms**. With a 200 µs direction turnaround before each reply the whole exchange takes 3.06 ms,
comfortably inside a 20 ms control tick, and servo 5 is sampled 2.05 ms after servo 0. Packet loss
defaults to 0.5% per packet; a lost request takes every reply in that tick with it.

## Figures

The sag, step-response and settling-error plots on this page come from one function, so they can be
regenerated after any change:

```bash
.venv/bin/python -c "from brick_builder.robot.servo_model import make_figures; make_figures('docs/media/robot')"
```

The settling-error panel drives the arm to 49 poses four times over and takes a few minutes; the
other two are seconds.

## Still to come

- **Servo system identification** against a real arm, once hardware arrives, which is what will
  settle whether saturation is 1.9 or 3.4 N·m and what the real overload time constant is. Three
  numbers disagree and none of them is ours: the datasheet says the 7.4 V part stalls at 1.9 N·m,
  the identified law at full duty implies 3.41 N·m at 7.4 V, and the independent bench measured
  35 kg·cm (3.43 N·m) on the 12 V variant, where the same law predicts 5.53. `forcerange` keeps
  the conservative 1.9 and the randomization range spans 1.5 to 3.4.
- **The firmware dead band**, registers 26 and 27, read off our own servos. `error_deadband_ticks`
  defaults to 0 and the documented value of 2 counts is a bracket between the bench report's "10
  counts" and its own ±2-count repeatability, not a measurement.
- **`error_deadband_ticks` is not in the randomization ranges yet**, because adding it to
  `scene/randomize.py` grows `DR_PARAM_KEYS` and therefore the critic observation. Adding it
  means one entry in `DR_PARAM_KEYS`, one line sampling it in `sample_params`, and one line
  applying it in `apply_params`.

## Kinematics

`kinematics.py` implements forward and inverse kinematics for the arm's five non-gripper joints
(`shoulder_pan` .. `wrist_roll`; the gripper does not move the TCP along any axis this module
solves for). Everything is built on two MuJoCo calls, `mj_forward` and `mj_jacSite`, and nothing
else -- no separate DH-parameter or screw-axis table is kept in sync with the MJCF, so if the
vendored geometry changes, `ArmKinematics` picks it up automatically.

```python
from brick_builder.robot.kinematics import ArmKinematics, euler_zyx

kin = ArmKinematics(model)               # model is a compiled mujoco.MjModel with the arm attached
pos, rot = kin.fk(q)                     # TCP position (3,) and rotation matrix (3, 3)
jac = kin.jacobian(q)                    # (6, 5): linear velocity rows, then angular velocity rows
tau_g = kin.gravity_torque(q)            # (5,): torque needed to hold q against gravity alone

yaw, pitch, roll = euler_zyx(rot)        # read a pose's orientation in the module's own convention
result = kin.ik(pos, pitch, roll, q0)    # IKResult(q, converged, pos_error, pitch_error, roll_error, iterations)
```

**Orientation convention.** A pose's rotation is decomposed as `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`
(world axes). `ik` only takes a target position, pitch and roll -- five numbers for five joints --
because there is no independent yaw joint at the wrist: yaw is whatever `shoulder_pan` needs to be
to reach the target x/y, not something a caller picks. Internally, IK pins the target frame's yaw
to the arm's own current yaw at every evaluation and reads the angular error off with the usual
cross-product formula, which keeps the linear algebra free of the gimbal lock a plain
"solve for a full target rotation" approach would hit exactly where a press needs it to work: tool
pointing straight down. Because that makes the residual a (mildly) moving target, its Jacobian is
built from finite differences of the exact residual rather than an analytic derivative through the
yaw extraction, and each Newton step is only kept if it actually lowers the re-evaluated residual
(Levenberg-Marquardt) -- both guard against the fact that `shoulder_lift`, `elbow_flex` and
`wrist_flex` share one rotation axis, so the orientation half of the Jacobian cannot tell them
apart on its own.

**Accuracy.** `fk` matches MuJoCo's own site pose to floating-point precision (it is the same call).
Seeded close to the true joint angles (the round-trip scenario: FK a random pose, perturb the seed
by up to ~3 degrees per joint, IK back), 2496 of 2500 draws across five seeds (99.84%) recover the
pose to under 0.5 mm and 0.5 degrees, with a median error near 0.01 mm / 0.0001 degrees; a target
outside the workspace, or a seed far enough from any solution, comes back `converged=False` with
whatever residual the solver bottomed out at, never a plausible-looking but wrong answer. The
handful of misses are genuine local minima of the damped least squares (not numerical noise close
to the tolerance), which is why the residual is reported rather than papered over.

## Force map

`force_map.py` answers the question that decides where the build plate goes and what supply
voltage the team should buy: how much downward force can the arm actually deliver at the TCP,
tool-down, across the workspace, at 5.0, 7.4 and 12.0 V?

By the principle of virtual work, a joint torque `tau_i` acting alone would produce a vertical TCP
force `tau_i / J_z,i`, where `J_z,i` is the z-row of the position Jacobian's i-th column; the joint
that saturates first sets the ceiling for the whole arm, so

```
F = min_i (tau_max_i - tau_gravity_i) / |J_z,i|
```

`tau_max_i` is the per-voltage stall torque budget, and `tau_gravity_i` is the torque needed to hold
the arm still at that pose (`ArmKinematics.gravity_torque`, i.e. `qfrc_bias` at zero velocity). This
is a per-joint torque-budget bound, not a full multi-joint optimisation -- it is the simplified
estimate the design report calls for.

The datasheet stall torque (1.9 N·m at 7.4 V) and the BAM servo law's own
saturation ceiling (3.41 N·m at full duty, `patch_so101.servo_stall_torque`) disagree by a factor of
1.8, and neither has been measured on the real arm, so `stall_torque(vin, torque_limit_nm=...)` takes
which one to use explicitly, and every force number below says which it used.

```python
from brick_builder.robot.force_map import press_force_map, figure_force_map

result = press_force_map(model, kin, xs, ys, z=0.02, vin_list=(5.0, 7.4, 12.0))
# result: xs, ys, z, reachable (ny, nx bool), q (ny, nx, 5), force ({vin: (ny, nx) N})
figure_force_map(result, "docs/media/robot/force_map.png")
```

Unreachable grid points are left `NaN` in `force` and `False` in `reachable`, not filled in from a
neighbour. Each cell is solved from up to three seeds -- the already-solved cell to its left, the
already-solved cell above it, and a fresh start -- and, when more than one converges, the solution
closest in joint space to its solved neighbours is kept, so a whole row or column stays on one elbow
branch instead of a lone cell jumping to a different configuration. A repair pass then retries any
cell still unreached now that its right and below neighbours are solved too: on
the grid below this took the count of unreachable cells from 39 to 17 out of 483 and removed every
isolated single-cell hole, leaving only the genuinely unreachable band close to the base.

![Force map](../media/robot/force_map.png)

Available downward force along the arm's centreline (`y = 0`, `z = 0.02 m`, tool-down), at four
reaches, under both torque-limit assumptions:

**1.9 N·m datasheet stall torque:**

| reach (m) | 5.0 V | 7.4 V | 12.0 V |
|---|---|---|---|
| 0.12 | 10.16 N | 13.99 N | 21.34 N |
| 0.16 | 8.71 N | 11.95 N | 18.16 N |
| 0.20 | 7.30 N | 9.98 N | 15.12 N |
| 0.24 | 7.71 N | 10.69 N | 16.39 N |

**3.41 N·m identified-servo-law stall torque (BAM law's full-duty ceiling at 7.4 V, scaled the same
way with voltage):**

| reach (m) | 5.0 V | 7.4 V | 12.0 V |
|---|---|---|---|
| 0.12 | 16.51 N | 23.39 N | 36.58 N |
| 0.16 | 14.07 N | 19.89 N | 31.03 N |
| 0.20 | 11.74 N | 16.55 N | 25.77 N |
| 0.24 | 12.64 N | 17.98 N | 28.21 N |

Force falls with reach out to about 0.20 m as expected, then ticks back up at 0.24 m: the IK solver
lands on a different elbow configuration there, with a longer lever arm from `shoulder_lift` to the
TCP, and the map faithfully reports whichever branch it actually finds. At the design's chosen
supply (7.4 V, datasheet limit) this is 14.0-10.0 N across x = 0.12-0.20 m, somewhat under the
12-16 N the design report's own back-of-envelope estimate expected for that range -- worth
rechecking once the torque-limit uncertainty above is resolved by a real measurement, since it
scales every number in both tables directly.

Near full extension (0.28-0.36 m at this height) `shoulder_lift` is consistently the limiting
joint -- the "shoulder-limited region" the spec's `torque / reach` rule of thumb refers to -- but the
rule itself runs 55-59% low there (measured 8.4-10.5 N at 7.4 V, datasheet limit, against 5.3-6.8 N
predicted), because it implicitly assumes a single rigid link with the whole torque budget free for
the load. On the real chain, `shoulder_lift`'s own gravity torque already spends a sixth to a third
of the 1.9 N·m budget before any press force is applied, and its lever arm to the TCP is well under
the straight-line reach because the forearm folds down toward the low press height rather than
staying collinear with the upper arm. `tests/robot/test_force_map.py` checks this instead with an
independent statics test rather than loosening that tolerance: it
applies the map's own reported force back at the TCP as a real Cartesian wrench in an actual
simulation (`mj_applyFT` plus forward-dynamics settling, not the Jacobian arithmetic that produced
the number) and confirms the limiting joint's actuator torque lands on its own limit within 5% --
the qualitative trend (force falls with reach, `shoulder_lift` saturates first far out) holds, and
the map's own arithmetic is checked against simulated statics rather than against a rule of
thumb it was never going to match.
