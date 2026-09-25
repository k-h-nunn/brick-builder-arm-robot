# Measure and enter parameters

This page is for whoever is holding actual DUPLO bricks and the actual arm, not just reading
code. It walks through every measurement the simulator needs, in the order that matters most,
and for each one says: what to do, what to type into a parameter file, which experiment to run
afterwards, and how to read what comes back.

The parameter file is `configs/params/default.yaml`, loaded by `brick_builder/params.py`. **Do not
edit that file.** Copy it and edit your copy:

```bash
cp configs/params/default.yaml my_params.yaml
```

Every entry in it looks like this:

```yaml
bricks:
  clutch_force_per_stud_N:
    value: 1.0
    status: assumed
    measured_by:
    measured_on:
    notes: "..."
```

`status` is one of `assumed` (a placeholder or a guess), `derived` (worked out from other
assumed numbers — better than a guess, still not a measurement), or `measured` (you measured it,
on this team's own bricks or this team's own arm, and wrote down how). Every experiment's report
prints this status next to every value it used, so nobody mistakes a guess for a fact.

You don't have to use the long form. A one-line override in your own file —

```yaml
arm:
  supply_voltage_V: 6.9
```

— is shorthand for `{value: 6.9, status: measured}`. Use the long form when you want to record
who measured it and when, or a note about how; use the short form when you're in a hurry. Your
file only needs to mention what you've actually changed — everything else is filled in from
`configs/params/default.yaml` automatically.

The loop, for every measurement below, is the same four steps: **measure it, edit
`my_params.yaml`, run the experiment named, read the report.** All commands assume
`MUJOCO_GL=egl` per [Install](01-install.md#rendering-backend); prefix it explicitly if
you haven't exported it in your shell already.

## Equipment

- A handful of **new, light-coloured** DUPLO bricks: at least two 2x2s and two 2x4s. Not from the
  toy box — old or dark-coloured bricks are measurably a different, more brittle material (see
  [Materials and damage](02c-materials-and-damage.md#11-the-brick-plastic)), and this whole page
  is about measuring the material the simulator is supposed to represent.
- A kitchen scale, ideally reading to 0.1 g, ideally with a peak-hold mode. If it doesn't have
  one, your phone's camera at 60 fps does: record the display and step through the frames.
- A ruler.
- The arm, connected, with whatever tool LeRobot or the servo manufacturer gives you for reading
  a servo's registers (voltage, P gain).
- A printed checkerboard for the camera measurement (see that section for the exact pattern).

Calipers help for the brick dimension fields (`bricks.pitch_mm`, `stud_outer_diameter_mm`, and
so on) but are not required to get useful results — those dimensions are the least uncertain
numbers in the file, and the shipped defaults are already within a caliper's worth of correct.
Measuring them is worthwhile but is not where the priority is; the sections below are in
priority order, and this isn't one of them.

---

## 1. Clutch force per stud — push-on and pull-off with a kitchen scale

**This is the single most important measurement in the project.** It decides whether the arm can
seat a brick at all. Twenty minutes, no arm needed.

`bricks.clutch_force_per_stud_N` means one specific thing: the **average** of push-on and
pull-off force per stud, not either one alone. The simulator's own snap-fit model
(`SnapParams.f_clutch`) is built that way — push-on costs `f_clutch * (1+beta)`, pull-off
releases at `f_clutch * (1-beta)` — so measuring push-on and pull-off separately, the way this
section does, gives you exactly what the parameter needs, plus the asymmetry `beta` as a bonus.

**Push-on.** Put the kitchen scale on the table and a baseplate flat on the scale. Tare it. With
one finger on the middle of a new 2x2, press it down onto the plate slowly and squarely until it
is fully home. Read the **peak** force the scale showed on the way down (record it if your scale
doesn't hold a peak). Do this once for a 2x2 and once for a 2x4. If you have time, repeat five
times each and use the mean — the spread across repeats is itself useful (it's the width you'd
randomize this parameter over).

**Pull-off.** Press the same brick fully on, then turn the plate upside down, hold it by the
edges, and hang a small bag from the brick. Add water a little at a time until the brick lets
go. Weigh the bag: grams × 0.00981 = newtons. Do this for a 2x2 and a 2x4.

While you're doing it, notice: does the force build up smoothly, or is there a click partway
in? A click means `bricks.press_curve_shape` should be `detent`, not the default `ramp`.

**The arithmetic, by hand.** For a brick with `n` studs, peak push-on force `F_push` and peak
pull-off force `F_pull`:

```
clutch_force_per_stud_N = (F_push + F_pull) / (2 * n)
push_pull_asymmetry_beta = (F_push - F_pull) / (F_push + F_pull)
```

Do this once per brick size — a 2x2 and a 2x4 should give the same `clutch_force_per_stud_N` and
the same `beta`, within the spread of your repeats, because the model treats both as one per-stud
property. `press-bench` (below) fits the same two numbers automatically, by an algebraically
identical route (it gets `beta` from the push/pull ratio first, then `clutch = pull_off / n /
(1 - beta)`, which is the same fraction rearranged); the [worked example](#a-worked-example)
checks that the by-hand arithmetic above and `press-bench`'s automated fit agree on one set of
numbers, so you can trust either.

**Type into `my_params.yaml`:**

```yaml
bricks:
  measured:
    push_on_2x2_N:
      value: 9.8
      status: measured
      measured_by: your_name
      measured_on: 2026-09-21
    push_on_2x4_N:
      value: 19.1
      status: measured
    pull_off_2x2_N:
      value: 7.2
      status: measured
    pull_off_2x4_N:
      value: 14.6
      status: measured
```

(Those four numbers are made up for illustration — see the [worked example](#a-worked-example)
below for one carried all the way through.)

**Run:**

```bash
.venv/bin/python -m brick_builder.experiments press-bench --params my_params.yaml --out results/
```

**Read the report:** look for "Fit from your measurements." It converts your four raw readings
into `clutch_force_per_stud_N` and `push_pull_asymmetry_beta` the same way the worksheet in
[Materials and damage](02c-materials-and-damage.md#70-the-one-to-do-first-exactly-as-written)
does (and the same way the by-hand arithmetic above does — see the worked example), tells you
whether your 2x4 pull-off is close to twice your 2x2 pull-off (it should be, if the force really
is per-stud), and prints the two values to paste back into your file — as `status: measured` this
time, since now they are. The report also shows the simulated push-on and pull-off curve at
whatever `clutch_force_per_stud_N` your file currently has (the shipped default, 1.0 N, unless
you've already changed it), so you can compare "what the simulator currently assumes" against
"what you just measured" on the same figure before you make the edit.

---

## 2. Friction — the angle at which a brick starts to slide

**The second-biggest unknown.** No source anywhere gives a coefficient of friction for ABS on
ABS, or for a printed PLA jaw on an ABS brick, and every holding-force number in the simulator
scales linearly with it.

**Procedure.** Rest one DUPLO brick flat on top of another (or on a wide flat DUPLO plate), on a
board you can tilt. Slowly raise one end until the top brick just starts to slide. Measure the
angle from horizontal (a phone's built-in level/inclinometer app works well) — the tangent of
that angle is the static coefficient of friction. Repeat a few times and average.

For the jaw-on-brick number, do the same thing with a flat offcut of PLA (the material the
gripper's fingers are printed from) in place of the lower brick. A scrap from any PLA print will
do; it does not have to be a genuine spare part.

**Type into `my_params.yaml`:**

```yaml
bricks:
  brick_on_brick_friction: 0.31
  jaw_on_brick_friction: 0.52
```

**Run:** `press-bench` (brick-on-brick friction feeds the press simulation directly) and
`grasp-hold` (jaw-on-brick friction is the only thing holding a brick in the stock gripper):

```bash
.venv/bin/python -m brick_builder.experiments press-bench --params my_params.yaml --out results/
.venv/bin/python -m brick_builder.experiments grasp-hold --params my_params.yaml --out results/
```

**Read the report:** `press-bench`'s peak push-on/pull-off numbers move with this value —
compare the new report against the one from step 1. `grasp-hold`'s summary table says, for a
2x2 and a 2x4, in each direction (hanging, or pushed up the way a press reaction loads it), the
largest load that stays under a millimetre of slip. Read its "Reading" section before assuming a
low number is entirely this parameter's fault — the stock jaw's geometry (a closing wedge) does
a lot of the work independently of friction, and the report says which direction that applies to.

---

## 3. Brick mass

**Procedure.** Weigh a new 2x2 and a new 2x4 on the kitchen scale.

**Type into `my_params.yaml`:**

```yaml
bricks:
  mass_2x2_g: 6.4
  mass_2x4_g: 12.1
```

**Run:**

```bash
.venv/bin/python -m brick_builder.experiments params --params my_params.yaml --out results/
```

**Read the report:** the "Brick mass cross-check" table compares your weighed mass against what
`bricks.density_kg_m3` plus the dimension fields predict for the same shape. If they agree
within 10%, leave `density_kg_m3` alone — the default (generic ABS) is fine. If they don't, the
discrepancy is telling you the actual material is denser or lighter than generic ABS; adjust
`density_kg_m3` until `params`'s report shows agreement, and mark it `derived` (you didn't
measure density directly — you measured mass and backed density out from it).

---

## 4. Supply voltage and servo gain — read from the servo tool

**Two of the four biggest unknowns.** The simulator's whole force budget scales with supply
voltage, and the two competing torque-limit numbers in the servo model (1.9 N·m from the
datasheet, 3.41 N·m from the identified law — see
[Robot and servo model](03-robot-and-servo-model.md#upstream-servo-gain-error)) have
never been checked against this team's actual arm.

**Procedure.** With the arm connected to its servo configuration tool (whatever LeRobot or the
Feetech/Waveshare tool your kit came with provides), read:

- the supply voltage the servos are actually running at (off the power supply under load is more
  honest than the supply's nameplate rating);
- the P register (address 21) each servo is set to — LeRobot writes 16 on connect, the factory
  default is 32; confirm which this team's servos actually have;
- the CW/CCW dead band (registers 26/27) each servo is set to — the same tool session reads
  these, and [Robot and servo model](03-robot-and-servo-model.md#reading-the-servo-registers) has
  the exact commands. `arm.servo_dead_band_counts` wants one number for both registers; if they
  differ, write down the larger of the two.

**Type into `my_params.yaml`:**

```yaml
arm:
  supply_voltage_V: 7.35
  servo_p_gain: 16
  servo_dead_band_counts: 2
```

**Run:**

```bash
.venv/bin/python -m brick_builder.experiments seating-feasibility --params my_params.yaml --out results/
.venv/bin/python -m brick_builder.experiments force-map --params my_params.yaml --out results/
```

**Read the report:** `seating-feasibility` says, in words, whether a straight or a tilted press
of a 2x4 and a 2x2 is possible anywhere on the build plate, not possible anywhere, or possible
only in the best part of it, at your voltage. `force-map` shows the whole workspace as a heatmap
and reports the force specifically at your configured baseplate position. If either report says
a press is "not possible anywhere," raising `arm.supply_voltage_V` (if your power supply allows
it) is the most direct lever you have; the servo's stall torque is proportional to it.

`arm.torque_limit_Nm` — which of the two candidate numbers above is real — is not something a
kitchen scale settles. It needs a lever arm and a scale, or a calibrated stall test with the
servo tool, and is out of scope for this page; the parameter file ships the conservative 1.9 N·m
figure, and both `seating-feasibility` and `force-map` show what changes if the truth turns out
to be the higher, identified-law number instead.

`arm.servo_dead_band_counts` converts to `ServoConfig.error_deadband_ticks` through
`to_servo_config`, but no experiment or training run builds its `ServoConfig` from the loaded
parameter file today — nothing in the codebase calls `to_servo_config`. A value entered here has
no effect on any run yet; it is recorded for the day a caller passes it through. It also does not
change `brick_builder/scene/randomize.py`'s dead-band randomization range or nominal value: those
are the fixed constants `ERROR_DEADBAND_TICKS_RANGE` and `ServoConfig()`'s default, independent
of this file — a training run always randomizes and centres on the same dead band regardless of
what is measured here.

---

## 5. Backlash — by feel, with a ruler at the tool

**Procedure.** With the arm powered and holding still (not actively moving), grip the tool
firmly and gently rock it side to side without commanding any motion — you're feeling for the
gear play in the joints, not driving them. Hold a ruler next to the tool tip and read how far it
moves before you feel resistance from the motor itself. `shoulder_pan` and `shoulder_lift` have
the longest lever arm to the tool and dominate what you feel (see
[Robot and servo model](03-robot-and-servo-model.md#gear-free-play-and-encoder-position) for the per-joint breakdown).
Convert the play at the tool into a joint angle with the small-angle approximation:

```
backlash_total_deg ≈ degrees( play_at_tool_m / distance_from_shoulder_to_tool_m )
```

`distance_from_shoulder_to_tool_m` is roughly the reach you're working at — 0.15 to 0.25 m for
most press poses. A few millimetres of play at the tool over a 0.2 m reach is under a degree —
the [worked example](#a-worked-example) below computes 0.86 deg for 3 mm at 0.2 m.

`arm.backlash_total_deg` wants that whole number — the TOTAL play you just felt, not half of
it. The simulator's own free-play joints use a half-range internally, so `to_arm_config` halves
this value for you; type in exactly what you measured.

**Type into `my_params.yaml`:**

```yaml
arm:
  backlash_total_deg: 1.1
```

**Run:**

```bash
.venv/bin/python -m brick_builder.experiments pick-and-place --params my_params.yaml --out results/
```

(If `pick-and-place` prints "not available yet," the sequence module it depends on
(`brick_builder/control/pick_place.py`) isn't finished yet — see that experiment's own report for
what to do; everything else on this page still works.)

**Read the report:** backlash mostly shows up as placement error the arm can't feel through its
own encoders (the encoder reads the motor side of the gear, not the link side), so its clearest
effect is on `pick-and-place`'s success rate and the "wrong cell"/"partial seat" failure counts.
There isn't a simulator dial that shows backlash's effect in isolation — it's a background
contributor, not something with its own report section — so treat a change here as something to
re-check the whole pick-and-place success rate against, not something with an immediate number
of its own to read.

---

## 6. Camera field of view — with a printed checkerboard

**The fourth big unknown**, and the one furthest from "measure it with things already in the
kitchen." Two versions, quick and proper.

**Quick version**, no extra software: `diagonal_fov_deg` wants the corner-to-corner angle, so
measure it corner to corner. Take a sheet of A4 paper (diagonal 364 mm — √(210² + 297²) mm) and
fold it once, corner to corner, to mark the diagonal, or just work out its length with a ruler on
the flat sheet. Tape the sheet to a wall, back the wrist camera away until the sheet's two
opposite corners just touch the two opposite corners of the live frame (whatever live-preview the
camera driver gives you), and measure the distance from the lens to the paper. The diagonal field
of view is

```
diagonal_fov_deg = 2 * degrees( atan( (paper_diagonal_m / 2) / distance_m ) )
```

If you'd rather measure width and height instead of a diagonal, `fovx_from_diagonal_fov` and
`fovy_from_diagonal_fov` in `brick_builder/robot/patch_so101.py` do the conversion the other way —
the point is not to hand-wave a factor between horizontal and diagonal, because they are not
close enough for that to be a good approximation at a wide-angle lens like this one.

**Proper version**, once you have five minutes with a laptop and OpenCV: print a 9x6-internal-
corner checkerboard at 15 mm squares (fits this arm's short working distance), take 15-20+ photos
of it from varied positions *and orientations* with the wrist camera, log the `gripper` body's
base-frame pose at each shot, and run

```bash
.venv/bin/python scripts/calibrate_camera.py --images captures/ --poses captures/poses.json \
    --out camera_measured.yaml
```

This gives you real intrinsics + distortion (`cv2.calibrateCamera`) and, from the same shots, the
camera's mount *position* (a hand-eye `AX = XB` solve, not just field of view), and writes both
straight into a `configs/params/` fragment ready to paste into `my_params.yaml` -- see
[Robot and servo model](03-robot-and-servo-model.md#the-wrist-camera) for the full procedure and
[Hardware, wiring, calibration](08-hardware-wiring-calibration.md#camera-calibration) for what the
script prints and how to read it. `--simulate` runs the whole pipeline against a rendered
checkerboard with no camera or arm needed, as a self-check of the pipeline itself.

**Type into `my_params.yaml`:**

```yaml
camera:
  diagonal_fov_deg: 121.0
  resolution_width_px: 640
  resolution_height_px: 480
```

(Use whatever resolution you actually captured at — the file's `fovy` conversion depends on it.)

**Run:**

```bash
.venv/bin/python -m brick_builder.experiments params --params my_params.yaml --out results/
```

**Read the report:** the camera section of the parameter table shows the vertical `fovy` MuJoCo
will actually compile the camera with, computed for your resolution. None of the six experiments
(`press-bench`, `seating-feasibility`, `force-map`, `grasp-hold`, `pick-and-place`) render an
image or read the camera fields at all, so a `diagonal_fov_deg`-only entry from the quick version
above has no feedback loop besides "does the number in the report look like what your lens
actually does." `scripts/calibrate_camera.py`'s own printout (reprojection RMS, focal-length
percent) is the real feedback loop for the proper version's `fx_px`/etc., and its `--simulate`
mode's recovery-error table (guides 03 and 08) is how that pipeline's own accuracy was checked
before trusting it on real photographs. If a vision-based task reads the camera model at runtime,
this is where it comes from.

---

## 7. Joint dry friction

`arm.joint_friction_Nm` is the dry friction in one servo's gear train. It sets
`ArmConfig.joint_friction_nm`, which every actuated joint's MuJoCo `frictionloss` is drawn from.
The shipped default is the BAM-identified value the model already used, so leaving this alone
changes nothing.

There is no kitchen-scale procedure for it — the number comes from a system identification
against a real servo (a droop or step-response fit, like
[Robot and servo model](03-robot-and-servo-model.md#checking-the-model-against-the-bench)
describes), which is arm bring-up work, not this page's. Until that identification runs on this
team's own servos, leave it at the shipped default and `status: assumed`.

**Run**, if you do change it: `params` shows the value the report will use; there is no
dedicated report section, the same way `arm.backlash_total_deg` has none. It does reach
`pick-and-place` the same way `backlash_total_deg` does — that experiment builds its scene from
this file's `ArmConfig` and always calls `pick_place.bench` at `dr_level=0.0`. A change here does
not appear in the step-response, sag or settling-error figures
[Robot and servo model](03-robot-and-servo-model.md#figures) regenerates — `make_figures` always
compiles a default `ArmConfig`, not one built from a parameter file.

A value entered here does not survive `brick_builder/scene/randomize.py`'s `dynamics` group being
applied at a randomization level above 0: `apply()` then writes `dof_frictionloss` from a draw
between `NOMINAL["dynamics.frictionloss"]` (the shipped default, used exactly at level 0) and
`RANGES["dynamics.frictionloss"]` (0.03-0.10 N·m), replacing whatever the scene was built with.
`pick-and-place` calls `bench` at `dr_level=0.0`, so it keeps a measured value; an `ArmEnv`/
`PressEnv` reset at `dr_level > 0` does not. `backlash_deg` and `kp` have the same property.

---

## 8. Arm print colour

`arm.print_color_r`/`_g`/`_b` set the RGBA this team's own printed arm parts render in
(`ArmConfig.print_color_rgba`, applied to every printed-part material except the STS3215 servos'
own black plastic casing). Unlike every other section on this page, this is not a physical
measurement — it is a fact only whoever printed the parts, or has the arm in hand, can state. The
shipped default reproduces the upstream mesh's colour, a golden yellow; whether that matches this
team's own printed parts is not yet known.

**Type into `my_params.yaml`**, once the colour is known (values 0-1, not 0-255):

```yaml
arm:
  print_color_r: 0.85
  print_color_g: 0.85
  print_color_b: 0.85
```

**Run:** `params` reflects the new values immediately. No physics report changes — this is a
render-only parameter. Anything that builds a scene from this file through `to_scene_config`
picks it up, in particular the GUI worlds (`brick_builder/gui/world.py`) and `pick-and-place`'s
own scene; the convolutional scene model's own training-data generator
(`brick_builder/perception/cnn/data.py`) does not read the parameter file at all — it always
builds a default `ArmConfig`, so a colour entered here does not reach it.

---

## A worked example

Suppose a teammate spent an afternoon and came back with:

- 2x2: push-on 9.8 N, pull-off 7.2 N. 2x4: push-on 19.1 N, pull-off 14.6 N.
- Friction: brick-on-brick slid at 17°, jaw-on-brick at 27°.
- Masses: 2x2 = 6.4 g, 2x4 = 12.1 g.
- Supply voltage read as 7.35 V, P gain 16.
- Backlash: about 3 mm of play at the tool at a 0.2 m reach.
- Camera: A4 paper's diagonal (364 mm) exactly filled the frame corner to corner at 0.15 m.

Their `my_params.yaml`:

```yaml
bricks:
  measured:
    push_on_2x2_N: 9.8
    push_on_2x4_N: 19.1
    pull_off_2x2_N: 7.2
    pull_off_2x4_N: 14.6
  brick_on_brick_friction: 0.31    # tan(17 deg)
  jaw_on_brick_friction: 0.51      # tan(27 deg)
  mass_2x2_g: 6.4
  mass_2x4_g: 12.1

arm:
  supply_voltage_V: 7.35
  servo_p_gain: 16
  backlash_total_deg: 0.86         # degrees(atan(0.003 / 0.2))

camera:
  diagonal_fov_deg: 101.0          # 2*degrees(atan(0.182 / 0.15))
```

Running `press-bench --params my_params.yaml` fits `clutch_force_per_stud_N ≈ 2.12 N` and
`push_pull_asymmetry_beta ≈ 0.14` from the push/pull readings above — check it both ways, since
guide 10 promises they agree.

By hand: `beta = (F_push - F_pull) / (F_push + F_pull)` gives 2x2 (9.8-7.2)/(9.8+7.2) = 0.153,
2x4 (19.1-14.6)/(19.1+14.6) = 0.134, averaged 0.143; and `clutch = (F_push + F_pull) / (2n)`
gives 2x2 17.0/8 = 2.125 N, 2x4 33.7/16 = 2.106 N, averaged **2.12 N**.

`press-bench`'s automated route gets there differently but lands on the same numbers: it fits
`beta` from each brick's push/pull ratio first (the same 0.153 and 0.134, averaged 0.143), then
`clutch = pull_off / n / (1 - beta)` (2x2: 7.2/(4×0.857) ≈ 2.10 N; 2x4: 14.6/(8×0.857) ≈ 2.13 N,
averaged **2.12 N**). Both routes agree, which is the point of doing it both ways.

That measured 2.12 N is inside the plausible 0.5-5 N range: more pessimistic than the file's
shipped default (1.0 N, `assumed`) and far more optimistic than the first-principles upper
scenario (4.79 N, guide 02c section 2.3). Pasting the fitted numbers in and re-running
`seating-feasibility` tells a different story at each of those three clutch values for a
**straight** press of a 2x4: possible everywhere on the plate at the shipped default (9.2 N
needed, just under the arm's worst-case 9.5 N at 7.4 V), not possible anywhere at the measured
clutch (19.4 N needed) or at the first-principles scenario (44 N needed). A **tilted** press
fares better at all three: 2.3 N at the shipped default and 4.8 N at the measured clutch, both
comfortably inside the arm's 9.5-14.0 N range, so tilted is possible everywhere on the plate at
either; even at the pessimistic first-principles scenario a tilted press needs only 11.0 N, which
still fits inside that range (in the best part of the plate, not yet everywhere). That is a real,
measured answer to part of the project's central question, in about twenty minutes of
kitchen-scale work: an optimistic shipped default can make a straight press look free, a tilted
press is the one that keeps working across the whole plausible range, and only a real measurement
tells you which of those stories your own bricks are actually in.

## Recording results

Use `docs/experiments/TEMPLATE.md` as a one-page lab notebook entry for anything you measure or
run: what you were trying to find out, how you set it up, which parameter file and command you
used, what happened, and what you concluded. Copy it to a new file per session — it's meant to
be filled in before and after, not just after.

## Quick reference

| Parameter | Measured how | Experiment to run afterwards |
|---|---|---|
| `bricks.clutch_force_per_stud_N`, `push_pull_asymmetry_beta` | kitchen-scale push-on/pull-off | `press-bench` |
| `bricks.press_curve_shape` | feel for a click while pressing | `press-bench` |
| `bricks.brick_on_brick_friction` | tilt-and-slide angle, brick on brick | `press-bench` |
| `bricks.jaw_on_brick_friction` | tilt-and-slide angle, PLA on brick | `grasp-hold` |
| `bricks.mass_2x2_g`, `mass_2x4_g` | kitchen scale | `params` |
| `bricks.*_mm` (dimensions) | caliper | `params` |
| `arm.supply_voltage_V` | servo tool / power supply | `seating-feasibility`, `force-map` |
| `arm.servo_p_gain` | servo tool | `pick-and-place` |
| `arm.torque_limit_Nm` | not a kitchen-scale measurement — see step 4 | `seating-feasibility`, `force-map` |
| `arm.backlash_total_deg` | ruler at the tool, by feel (total play, not half) | `pick-and-place` |
| `arm.servo_dead_band_counts` | servo tool, registers 26/27 | `params` (no run consumes it yet) |
| `arm.joint_friction_Nm` | not a kitchen-scale measurement — system identification, see guide 03 | `pick-and-place` (no dedicated report section); overwritten by domain randomization's `dynamics` group above level 0 |
| `arm.print_color_r`/`_g`/`_b` | look at the printed arm — not a physics measurement | `params` (render only) |
| `camera.diagonal_fov_deg`, resolution | printed checkerboard or a sheet of paper | `params` |
| `camera.fx_px`/`fy_px`/`cx_px`/`cy_px`/`opencv_*`, `mount_*_mm`/`_deg` | `scripts/calibrate_camera.py` against checkerboard photos | `params` |
| `bricks.youngs_modulus_GPa` | three-point bend on a wall cut from a brick | `params` |
| `bricks.tensile_yield_MPa`, `flexural_strength_MPa` | destructive: push a wall sideways to a visible set, then to a crack (guide 02c section 7.7) | `grasp-hold` |
| `bricks.contact_model`, `damage_monitor`, `breakable` | switches, not measurements — see guide 02d | `pick-and-place` |

And after any change, `python -m brick_builder.experiments params --params my_params.yaml` is the
cheapest way to see the whole file's current state — every value, every status, and which of the
four headline unknowns are still unmeasured — before spending time on the heavier
experiments.
