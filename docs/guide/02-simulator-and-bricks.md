# Simulator and bricks

**Status: built.** The brick generator and the snap-fit model (`brick_builder/bricks/`), the scene
composer (`brick_builder/scene/`) and the renderer (`brick_builder/render/`) all exist, and the numbers on
this page are measured from them.

## Purpose

This is the part of Brick Builder Arm Robot that turns a description of a LEGO DUPLO structure into a physically
simulated scene: brick geometry, a simulated camera, and — the hard part — a physically believable model
of how a stud presses and clicks into a tube.

## Capabilities

- Generate any DUPLO (or LEGO System, as a configuration variant) brick shape as a MuJoCo body, with
  separate collision geometry (a handful of convex shapes, for speed) and visual geometry (thin walls,
  ribs, hollow studs, for a realistic picture).
- Compose a scene of bricks, a baseplate, and a tray, and render it the way a camera on the robot would
  see it — RGB, depth, and per-object segmentation.
- Run the "press bench": drive a ram down onto a brick along a slide joint, log force and depth, and
  check the result against the acceptance tests in [Bench tests](#bench-tests) below.
- Apply domain randomization (lighting, brick color jitter, camera pose, snap-fit strength, and more) so
  that a policy trained in simulation is not fooled by physics or camera details that turn out to be
  slightly wrong.

## DUPLO dimensions

These numbers come from three independent sources that agree with each other (a brick-generator
project's parameters, a caliper measurement, and a parametric-brick modeling tool), and set the
defaults for the brick generator.

| Dimension | Value (mm) |
|---|---|
| Stud pitch | 16 |
| Brick height | 19.2 (measured range 19.09-19.17) |
| Footprint | `n * 16 - 0.2` for an n-stud-long brick |
| Stud outer diameter | 9.40 (measured 9.38) |
| Stud height | 4.5 (measured 4.55) |
| Stud hollow inner diameter | 6.5 (measured 6.44) |
| Wall thickness | 1.55-1.6 |
| Wall rib width | about 1.5, touching the studs |
| Underside tube outer diameter | 13.4 (measured 13.48) |
| Underside tube inner diameter | 10.7 |
| Underside tubes | one at each interior grid intersection — 1 for a 2x2, 3 for a 2x4 |

The diagonal gap between two adjacent studs is `16 * sqrt(2) - 9.4 = 13.23` mm, and the tube's measured
outer diameter is 13.48 mm — a 0.125 mm radial interference, in line with the commonly cited 0.1-0.2 mm
LEGO manufacturing tolerance. That tiny interference is the entire physical basis of the "click."

A 2x4 brick masses 12.2 g (from a brick-cataloguing site); a 2x2 is assumed to be about 6.5 g until it is
weighed.

## Press-fit model design

Rigid-body simulators, MuJoCo included, cannot resolve the 0.125 mm interference fit the way a
real plastic part deforms and grips. Three simpler approaches were considered and rejected
before landing on the design below:

- Tuning MuJoCo's ordinary contact stiffness parameters - contact force in MuJoCo does not
  scale like a simple spring, so this could not be calibrated to real press-force curves.
- Applying a custom hand-written force law - the damping a light 12 g brick would need puts the
  contact solver in a regime where nothing actually holds in place.
- Modeling the connection as a single rigid "weld" that breaks past some force - this cannot
  model the insertion process itself, only an instantaneous connect/disconnect. This one is
  implemented anyway, as `DetentSnap` in `bricks/snap_detent.py`, because it is a useful
  control: it answers "connected or not" cheaply and correctly, and measuring what it cannot do
  is what justifies the complexity of the real model.

**The chosen design: a per-stud friction joint** (`bricks/snap.py`, class `ColletSnap`). Every
socket on the underside of a brick gets a tiny extra body. It has no geometry, it
never collides with anything, and it is not a part of a real brick: it exists so that the
relative motion between a brick and the stud it is being pressed onto becomes a single joint
coordinate that MuJoCo's constraint solver can put **dry friction** on. That body hangs off
the brick on three short slide joints - one along the brick's own z axis, which is the clutch,
and two sideways ones limited to the socket's clearance, which is the play a stud really has
inside a socket.

Then, once per physics step:

1. **Pairing and capture.** Every free socket is checked against every stud in the scene, in
   one vectorised pass. A socket that is within 0.3 mm sideways, within 6 degrees of parallel,
   and at least 0.5 mm onto the stud gets connected: a `connect` equality that was declared
   inactive at compile time is switched on through `data.eq_active`, with its anchors written
   into `model.eq_data`, pinning that body to that stud's tip. The residual sideways offset is
   taken up by its own play and pulled to the middle of the socket by a weak spring -
   the self-centring a real chamfer does mechanically.
2. **The clutch.** From then on, that body's axial coordinate *is* the insertion depth, and
   the brick can only move relative to the one below it by sliding that joint. The model
   schedules the joint's dry friction with depth, `frictionloss(d) = f_clutch * g(d)`. That is
   the entire force law, and `g` is the only place the shape of the curve is defined.
   What shape `g` has is a *parameter*, `SnapParams.curve`, because nobody has yet put a real
   DUPLO brick on a force gauge. See [Tilted press vs. flat press](#tilted-press-vs-flat-press)
   below for whether the shape makes tilting the brick during the press worthwhile:

   | `curve` | `g(d)` for an engaged stud | what it says a brick is like |
   |---|---|---|
   | `ramp` (default) | `clip((d - engage) / (h - engage), 0, 1) ** exponent` | resistance grows all the way in and is highest just as the brick goes home |
   | `plateau` | `1` | a stud is as hard to push at 1 mm as at 4 mm |
   | `detent` | a plateau at `plateau_gain` rising to 1 at full seat, plus a Gaussian bump of height `detent_gain` centred at `detent_depth` | the interference snap: hardest to push past the entry bump, easier afterwards - the click |

   All three reach `g = 1` at full seat, so a seated stud always holds `f_clutch * (1 - beta)`
   and the pull-off, stability and peel numbers do not depend on which shape is chosen. Only
   the work of getting the brick on does, and whether tilting the brick on is worth anything.
3. **Push/pull asymmetry.** The same bound is scaled by `1 + beta` while a stud is going in and
   `1 - beta` while it is coming out or standing still, so pressing costs
   `n * f_clutch * (1 + beta)` and pulling off releases at `n * f_clutch * (1 - beta)` - a real
   brick is stiffer going on than coming off.
4. **Seating** is `d >= h` for every engaged socket. It is a label, not a change of physics.
5. **Breakage** is the friction saturating: a straight pull, or a peel that unzips the
   connection one stud at a time, both fall out of the per-stud bookkeeping. A connection is
   released when its stud comes back out of the socket mouth, or when the sideways constraint
   force tears a still-shallow one out of line.

Nothing in the model knows what a structure is: a structure is just the graph of live
connections, and compressive load travels through ordinary contacts.

### Measured vs. configured force curve

![Measured press force against the configured curve, for all three curve shapes](../media/bricks/force_vs_depth.png)

A 2x4 brick pressed on at 2 mm/s under each of the three shapes, against
`n * f_clutch * (1 + beta) * g(d)` with the defaults below. The pale thick line is what the
parameters asked for and the thin line is what the simulation did: 0.24 % relative RMS error
for `ramp`, 0.35 % for `plateau` and 0.28 % for `detent`, with under 0.1 % difference between a
2 ms and a 0.5 ms timestep. Two things about the trace:

- The trace is drawn in the order it happened rather than sorted by depth, because a press
  through a detent is not monotone in depth. When the ram gets over the bump, the energy stored
  in its own stiffness throws the brick forward: the force dips, briefly goes negative (the ram
  is being dragged), and then the descent catches up about 1.8 mm later. That is the click, and
  it is why the comparison with the quasi-static curve is only made outside that window
  (`bench.quasi_static_band`). `plateau` has a smaller version of the same thing at the step.
- Every trace stops slightly short of the model's end point, because a press run stops the
  instant the brick is home rather than continuing to squash it into the plate.

### Tilted press vs. flat press

![Tilted press against a flat press, for all three curve shapes](../media/bricks/rocking_vs_flat.png)

A tilted press is not special-cased anywhere. Tilt the brick and its sockets are simply at different
depths at the same instant: the four rows of a 2x4 go in one after another, spread over 2.2 mm,
while a flat press keeps every socket within a rounding error of every other. The row that went
in first becomes the hinge, and the wrist rotation that levers the rest down costs 0.27 to
0.33 N.m - comfortably inside one STS3215's 1.9 N.m. All of that is true for every curve shape.

Whether it *pays* - whether the peak force at the tool comes down - is decided entirely by the
shape, and bench test 4 is now three measurements rather than one threshold (2° of tilt, tool
at the leading edge):

| `curve` | flat peak | tilted-press peak | ratio |
|---|---|---|---|
| `ramp` | 8.54 N | 8.61 N | **1.01** |
| `plateau` | 9.28 N | 9.64 N | **1.04** |
| `detent` | 13.21 N | 8.83 N | **0.67** |

The reason is Newton's third law. The bench gauge reads the sum of the vertical forces on the
brick, so `F_tool = sum(F_stud)` plus any contact reaction, which is never smaller than
`sum(F_stud)`. A rigid brick on a flat plate can only have every socket at full depth at the
same instant, so the last moment of *any* press has all eight studs sliding at `g = 1`. If that
is also where the curve is highest - `ramp`, `plateau` - no amount of staging can beat it, and
a ratio of about 1 is the answer, not a failure. If instead the curve peaks early, at an entry
bump, then tilting the brick sends its four rows over that bump at four different times and the
tool never has to push all eight over it together: 0.67, which is what the plan's 0.65 was
reaching for.

Whether to tilt the brick during the press is not a question about the controller. It is a
question about real bricks, settled by putting one on a force gauge (open question 2 of the
design report). Until then the default stays `ramp`, which is the pessimistic
one, and the snap randomization group is to sample all three shapes so a trained policy never
learns to rely on a click that may not be there.

## Calibration

Everything tunable lives in one frozen dataclass, `bricks/snap_api.SnapParams`, and the
defaults are the "until we can measure real bricks" values of the design report. Nothing here
has been calibrated against hardware.

| Parameter | Default | What it does | How to measure it |
|---|---|---|---|
| `f_clutch` | 1.0 N | holding force of one fully inserted stud | push a 2x4 onto a kitchen scale in 0.5 mm shims; divide the plateau by 8 |
| `curve` | `"ramp"` | which family `g(d)` comes from | look at the shape of the measured force-against-depth curve: rising to the end, flat, or a bump near the start |
| `exponent` | 1.0 | curvature of the `ramp` (and of the `detent`'s base) | fit the shape of the same force-against-depth curve |
| `plateau_gain` | 0.35 | height of the `detent` curve's flat part, x `f_clutch` | the force between the click and full seat, over the seated holding force |
| `detent_gain` | 1.0 | height of the `detent` bump above that plateau, x `f_clutch` | the peak of the click over the seated holding force, minus `plateau_gain` |
| `detent_depth`, `detent_width` | 1.0 mm, 0.6 mm | where the bump is and how wide | read them off the same curve |
| `engage_depth` | 0.5 mm | how far in before a stud grips at all | the depth at which the scale first reads anything |
| `beta` | 0.15 | push/pull asymmetry | `(push peak - pull peak) / (push peak + pull peak)` |
| `capture_lateral` | 0.3 mm | sideways capture gate | offset jig and a feeler gauge: the largest offset that still seats |
| `capture_tilt_deg` | 6 deg | tilt capture gate | wedge shims |
| `lateral_play`, `lateral_ratio` | 1.0, 1.0 | sideways play and its grip, relative to `clearance` and `f_clutch` | how far a seated brick can be pushed sideways before it moves |

The workflow is:

1. Run the real measurement and write the numbers into a profile under `configs/snap/*.yaml`
   (one profile per brick population: new DUPLO, worn DUPLO, Mega Bloks, and so on).
2. Reproduce it on the bench. `bricks/bench.py` is the virtual force gauge: a stiff ram on a
   vertical slide carrying a brick down onto a baseplate, with an optional wrist hinge for
   tilting and peeling, logging force against depth every physics step.

   ```python
   from brick_builder.bricks.bench import BenchConfig, PressBench
   from brick_builder.bricks.snap import ColletSnap
   from brick_builder.bricks.snap_api import SnapParams

   bench = PressBench(ColletSnap(SnapParams(f_clutch=1.4, beta=0.2)), BenchConfig())
   press = bench.press(speed=0.002)     # down until the brick is home
   bench.hold(0.3)
   pull = bench.pull(speed=0.002)       # and back off again
   print(press.peak_force, pull.peak_pull)
   ```

3. Compare. `bench.predicted_press_force(depth, n_studs, params, stud_h)` gives the curve the
   parameters ask for, and `trace.at_depths(grid)` gives what the simulator produced;
   `bench.quasi_static_band(params, stud_h)` says over which part of the stroke the two are
   comparable. `brick_builder.bricks.media.make_force_vs_depth_figure(path)` draws all three shapes,
   and `bench.measure_rocking_ratio(params)` runs the flat-against-tilted pair of presses.
4. Re-run the bench tests: `.venv/bin/python -m pytest -q tests/bricks`.

Two numbers matter before tuning:

- The snap update costs about 100 microseconds per physics step for one brick over an 8x8
  baseplate and 320 microseconds for six bricks, against 36 and 3510 microseconds for
  `mj_step` itself on the same scenes. It is significant for a single brick and negligible once
  a scene has a few.
- A seated connection creeps about 0.01 mm per second at 90 % of its nominal friction. That is
  a property of how MuJoCo regularizes friction constraints, and `ColletSnap` already
  overwrites `dof_invweight0` on its own friction joints to get there from the 5 mm per second
  the default would give.


## Bench tests

Stage B's exit gate is eight bench tests (`tests/bricks/test_snap_bench.py`), run on the press
bench. Every test that can depend on the shape of the force curve is run for all three shapes;
where the three differ, all three numbers are given, in the order ramp / plateau / detent.

| # | Test | Acceptance | Measured |
|---|---|---|---|
| 1 | force against depth | within 10 % RMS of `n * f_c * (1 + beta) * g(d)`, under 5 % between dt 2 ms and 0.5 ms | 0.24 / 0.35 / 0.28 % RMS, under 0.1 % between timesteps |
| 2 | capture failure | 3 mm offset or 10 deg yaw never seats at twice nominal force | nothing connects at 31 N (nominal is 9.2 N); the brick stops 0.25 mm in |
| 3 | pull-off | `n * f_c * (1 - beta)` within 10 %, every connect cleared | break-away 6.87 / 6.89 / 6.88 N against 6.80 N. With `detent` the whole extraction peaks higher, at 9.77 N, because the stud has to be dragged back over the entry bump |
| 4 | tilted press | three measured cases (spec section 6b) | ratio 1.01 / 1.04 / **0.67** - see [Tilted press vs. flat press](#tilted-press-vs-flat-press) |
| 5 | stability | 3-layer overhang under 0.1 mm in 10 s; inverted brick under 0.3 mm | 0.079 / **0.241 (`xfail`)** / 0.087 mm, and 0.0013 mm inverted for all three |
| 6 | peel | failure moment within 20 % of `sum(f_c * lever)` | 0.236 / 0.256 / 0.238 N.m against 0.256 N.m |
| 7 | sanity | no energy gain, penetration under 0.1 mm at 10 N, zero warnings | no gain on a settled stack for any shape; the overhanging staircase trends 2.0e-5 / **7.4e-5 (`xfail`)** / -1.2e-5 J over 6 s; 0.006 mm of penetration at 9.8-13.2 N; zero warnings |
| 8 | save and restore | `mjSTATE_INTEGRATION` plus `SnapState` replays the trajectory | identical, bit for bit, for all three |

The two `xfail`s are the same scene and the same cause, written up in full in the tests' own
`reason` strings: the overhanging top brick of the three-layer staircase rattles inside the
friction joints' sideways play, nothing ever comes loose, and re-measuring the same scene over
successive 5 s windows gives anything from 0.02 to 0.38 mm. The 0.1 mm line is inside this
rig's own noise for every curve shape, so `ramp` passing it is partly luck. Quieting the rattle
means damping the friction joints and the brick-to-brick contact, not changing the force law.

The fallback model is held to tests 2, 3, 5 and 8 only
(`tests/bricks/test_snap_detent.py`), because the other four are about the shape of a force
curve it does not have; it passes all four.

Bricks default to a per-stud clutch force of 1.0 N (randomized log-uniformly between 0.3 and
2.5 N during training, since the real value has not been measured yet), a 0.20 mm clearance,
and a 0.5 mm chamfer; all three are replaced with measured values once a real brick has been
measured.


## Sections to be filled in as the code lands

- **Brick generator API** — `BrickSpec`, how to declare a new brick shape or a different brick system
  (LEGO System is a configuration change, not a new code path).
- **Scene composition** is written up in [Scenes and randomization](#scenes-and-randomization) below.
- **Rendering** is the only remaining gap for this page; the press bench and calibration
  workflow are written up in [Calibration](#calibration) above.


## Scenes and randomization

A *scene* is one complete world: the table, the arm, the build plate, whatever is already
standing on the plate, an optional tray of spare bricks, and usually a brick already in the
gripper. `brick_builder/scene/compose.py` builds one from a `SceneConfig`; `brick_builder/scene/randomize.py`
perturbs it between episodes.

```python
from brick_builder.scene.compose import SceneConfig, build_scene, reset_scene, settle
from brick_builder.bricks.structures import Placement

cfg = SceneConfig(
    plate_studs=(8, 8), plate_xy=(0.17, 0.0),
    placed=[Placement(2, 4, 1, 2, 0, "blue")],   # BrickGPT text format, already parsed
    grasp_mode="weld",                            # or "friction"
)
scene = build_scene(cfg)        # compiled, snap model bound, arm at its home pose
print(settle(scene, 2.0))       # warnings, drift, live connections
```

![The default scene](../media/scene/overview.png)

### Coordinate frame and layout

World x is forward, y is left, z is up. The arm's base sits at the origin and the table top is
z = 0. The plate's centre defaults to (0.17, 0.0), which keeps the whole 8x8 plate between 0.10 m
and 0.24 m of the base - the band where the force map says the arm has 12-16 N of downward push
at 7.4 V (that number uses the 1.9 N.m datasheet torque limit, not the 3.4 N.m the servo law
implies; see the note in the robot guide).

`reset_scene` puts everything back: the arm at `scene.home_q`, the placed bricks laid out against
the plate's *current* pose, the held brick back in the hand, and the snap model re-bound so no
stale connection survives. The home pose hovers the lowest part of the tool - the held brick's
bottom face, or the jaw tips when the gripper is empty - `home_hover` above the tallest thing
on the plate. That distinction matters: the jaw tips reach 8 mm below the TCP, so a home pose
worked out for a held brick drives an empty gripper straight through a brick standing on the
plate.

### Build order

`mujoco.MjSpec` does not forgive the wrong order here, and getting it wrong fails silently - you
get a scene where nothing ever connects:

1. `attach_arm`, then `add_baseplate` / `add_brick` / `build_structure`;
2. the snap model's `declare()`, which adds a friction-joint body per socket and a pile of inactive
   equalities that have to exist before compilation;
3. `world.compile()`;
4. the snap model's `bind()`.

`build_scene` is the one place that order is written down; use it rather than assembling a world
by hand.

### Grasp geometry

The brick is gripped across its 31.8 mm width with the tool pointing straight down, so the brick's
own +z (the way its studs point) is the gripper body's +z. Every number in `GraspPose` comes from
`patch_so101.jaw_geometry()` plus the brick's own `BrickSpec`:

| Quantity | Value for a 2x4 DUPLO brick |
|---|---|
| jaw opening | 31.8 mm (`nx * pitch - clearance`) |
| gripper joint angle where the jaws just touch | 0.2298 rad |
| brick wall between the jaw faces | 11.2 mm of its 19.2 mm height |
| brick hangs below the jaw tips | 8.0 mm |
| brick origin in the TCP frame | (-0.1, 0.0, +16.0) mm |

![The grasp, close up](../media/scene/grasp_closeup.png)

The moving jaw swings on a hinge, so the two working faces are never parallel: **the gap is
narrowest high up between the jaws and widest at the tips**. At the joint angle that leaves 31.8 mm at the narrow
end, the tips are 39.9 mm apart. `patch_so101.gripper_angle_for_width` deliberately measures the
narrow end, because that makes it a guarantee for the whole face; a grasp needs the opposite
question answered, so `compose.jaw_clearance(point, angle)` reconstructs the gap at any point and
`jaw_angle_for_clearance` inverts it. Closing to "31.8 mm" as `gripper_angle_for_width` measures it
leaves a brick held at the tips completely loose.

That wedge is not only bad news. It opens downwards, so a load that pushes the brick *up* between
the jaws drives it into a narrowing gap and the pinch tightens; a load that pulls it *down* lets it
into a widening one and it slides. Pressing a brick onto the plate is the first of those, which is
why a grip the acceptance test calls a failure can still seat bricks. The
[pick and place](#brick-seating-with-the-stock-gripper) below is that measurement.

### Two grasp modes

`grasp_mode="weld"` (the default) ties the brick to the gripper with a soft weld equality and
commands the jaws half a millimetre wider than the brick so they never touch it. It is
compliant - about 0.17 mm of give under 5 N - and its time constant and torque scale are
randomized, so it stands in for a grip without anyone having to tune finger contact first. That is
the whole point: Deliverable 1 is about the press.

`grasp_mode="friction"` closes the jaws 6 mm past the brick's face (about 19 N of pinch) and lets
friction between the bare printed jaw faces and the brick do the work. That is the only grip the
real arm has: the SO-101 gripper is a fixed jaw and a moving jaw, both printed, with nothing
attached to either.

Nobody has measured the friction coefficient between a printed jaw and an ABS brick, here or on
hardware, so it is a parameter (`ArmConfig.pad_friction`) rather than a fact, and its default of
1.0 is a placeholder. The one related number this project does have is design report 01's ABS on
ABS, 0.35, randomized over 0.2-0.5. Dry printed plastic on dry moulded plastic is the same kind
of contact, so **0.3-0.5 is the band to design against and 0.3 is the one to design for.** The
table below is the same 5 N pull, held 5 s, at four coefficients, so you can see how much of the
answer depends on the parameter nobody has measured:

| 5 N for 5 s | weld | friction, 0.3 | friction, 0.5 | friction, 0.8 | friction, 1.0 |
|---|---|---|---|---|---|
| downward (pulling the brick out) | 0.17 mm | **28.4 mm** | 2.32 mm | 2.11 mm | 2.08 mm |
| upward (the press direction) | 0.08 mm | 1.15 mm | 1.10 mm | 1.06 mm | 1.04 mm |
| sideways, across the jaws | 0.02 mm | brick ejected | brick ejected | 1.25 mm | 1.29 mm |
| along the brick | 0.03 mm | brick ejected | 6.10 mm | 8.53 mm | 9.28 mm |
| no load at all, 5 s | 0.000 mm | 0.84 mm | 0.88 mm | 0.86 mm | 0.85 mm |

The bottom row is the one to worry about. That creep is the brick sliding out of the wedge under
nothing but its own 12 g, and it does not stop: the same hold run for 10 s instead of 5 gives
1.43 mm at a coefficient of 0.3 and 1.55 mm at 0.5, so the brick is still leaking out at about a
tenth of a millimetre a second when the clock runs out. **A stock-gripper pick and place has to be
quick.** The scripted sequence below carries a brick for about six seconds and loses roughly
0.7 mm of grip in the process.

Read the rest of the table by column, not by row. Downwards the grasp is hopeless below a coefficient of
about 0.5 and never better than 2 mm above it, and the reason is the wedge rather than Coulomb
friction: squeezing harder makes it worse, not better (0.85 mm of unloaded creep at 18.8 N of
pinch, 1.22 mm at 33 N). Upwards - the direction a press loads it - it is about 1 mm at every
coefficient tested, including 0.3, because the gap closes on the brick as it rides up. "Brick
ejected" means the 5 N sideways pull squirted the brick out of the jaws and into the mechanism,
which MuJoCo reports as a bad-acceleration warning.

`tests/scene/test_compose.py` keeps the 1 mm downward test as a strict xfail with these numbers
in it, and the weld mode stays the default for training. But the pull-out test is not the task.

### Brick seating with the stock gripper

`brick_builder.control.pick_place` drives the whole job through the servo bus at 50 Hz: home, over the
tray, open, down onto a brick, close on it, lift, carry, line up over the target studs, seat it,
let go, retreat, home. It is a generator - one step per control tick, carrying the live scene and
the phase name - so a video script can film it, and `verify_placement` reads the answer back out
of the simulator afterwards: connected, at the intended cell, within 0.3 mm of full depth, and
still there once the arm has left.

The fixed jaw cannot move, so arriving with the brick flush against it - which is where the brick
ends up once the jaws close - leaves a brick a couple of millimetres off in that direction outside
the gripper. The descent therefore aims to put the brick in the *middle* of the open gap, half the
open clearance back, and lets the moving jaw shove it flush as it closes. Error across the jaws is
corrected by the act of gripping; error along the brick, and yaw, are not.

Three ways of seating it were built and measured against each other.

**Press while gripping.** Keep hold of the brick and drive the tool down until the studs go in.
The reaction pushes the brick up into the narrowing part of the gap, so the press tightens its own
grip.

**Place, let go, then push.** Set the brick on the stud tops, open the jaws, lift clear, bring the
jaws together, and push straight down on the brick's roof with the tip of the fixed jaw - which
reaches 7.2 mm lower than the moving one, so it is the only thing that touches. No grip force is
involved in the press at all.

**Both, in that order.** Press it in while holding it, let go, and push once more on the roof
with the jaws together. The second push exists because of what the first one leaves behind when
it fails.

Thirty runs per cell, a 2x4 brick, a friction coefficient of 0.5 between the jaws and
the brick, and a fresh draw each run of where the brick really sits in the tray
(+-2 mm, +-2 degrees, which the arm is not told about) and which way the placement error
points. "Seated" means every stud within 0.3 mm of full depth, on the intended cell, and
still there after the arm has gone home.

**These numbers were corrected after first publication.** The servo bus used to report the
*motor*-side encoder position, which hides the gear backlash between the motor and the output
shaft from the robot; the STS3215's actual encoder is on the output shaft (see
[Gear free play and encoder position](03-robot-and-servo-model.md#gear-free-play-and-encoder-position)).
With the encoder moved to where it really is, the arm itself now settles about 0.4 mm from its
commanded pose, against 3.6-6.0 mm before. The seating rates below turn out to have depended on
the hidden play: the old, motor-side-encoder tool arrived several millimetres off target and crept
sideways into the cell while it pressed, dragging the brick with it. An accurate arm no longer
does that. Instead the brick lands on the stud tops about 1 mm off centre - its position in the
jaws varies by that much along the jaw axis - and a blind press with a 0.25 mm capture window has
no lateral motion left to find the sockets.

#### 2x4

| placement error the arm does not know about | press while gripping | place, let go, push | both, in that order |
|---|---|---|---|
| 0 mm | 86 % → **0 %** | 46 % → **17 %** | 92 % → **17 %** |
| 1 mm | 84 % → **0 %** | 54 % → **3 %** | 90 % → **7 %** |
| 2 mm | 66 % → **3 %** | 28 % → **10 %** | 72 % → **10 %** |

Each cell reads old (motor-side encoder) → corrected (output-shaft encoder). Timing is unchanged
for the first two strategies - about 15 s pressing while gripping, 18.1 s placing and pushing -
and runs 20-23 s for both together; the arm joints still peak at about 0.9 N.m against the 1.9 N.m
limit, with only the gripper reaching its own limit. Sustained press force on the 2x4 drops from
about 9.5 N to 3-6 N: a symptom of the corrected tracking, not a new failure.

#### 2x2 (both, in that order)

| placement error | old | corrected |
|---|---|---|
| 0 mm | 100 % | 83 % |
| 1 mm | 98 % | 53 % |
| 2 mm | 98 % | 43 % |

A 2x2 fails less often than a 2x4 at the same placement error: half the studs need half the
force, and a 32 mm square footprint has nowhere near the leverage of a 64 mm one to stand itself
up on one row of studs.

Blind scripted pressing is therefore not enough on its own: **closed-loop visual alignment is
required** to seat a 2x4 reliably. The spiral search that `PressScripted` already has does not
rescue a friction grip the way it rescues a welded one: its trigger waits for the tool to drop
into a socket, and a friction grip lets the brick move in the hand instead, so the trigger never
fires.

Every one of these numbers comes from `brick_builder.control.pick_place.bench`, which is the same code path the tests run.

### Wrist camera field of view

![Arm camera POV, two poses](../media/scene/wrist_view.png)

The wrist camera's pose comes from TheRobotStudio's own official SO-101 wrist camera mount
(`so101_new_calib_camera.xml`), and its lens is modelled at 115 degrees diagonal field of view,
randomized over 100-130 degrees. At that field of view the fixed jaw covers 7-15% of the frame,
and the whole 8x8 baseplate is visible at a 5.8 cm hover. See
[Robot and servo model](03-robot-and-servo-model.md#the-wrist-camera) for the full pose, lens and
measured image-composition detail.

### Randomization

`randomize.sample(rng, level)` returns one flat dictionary keyed `"<group>.<field>"` - the same
dictionary that goes into the critic observation and into `EpisodeResult.params`, so a run is
reproducible from what was logged rather than from a seed plus a version of the source.
`randomize.apply(scene, sample)` writes it into the compiled model in place.

| group | what it moves |
|---|---|
| `dynamics` | servo gain, torque limit, joint damping, friction loss, armature, backlash, supply voltage, link mass |
| `command` | dead band, slew limit, command delay and jitter, read skew and noise, bus packet loss and period jitter |
| `snap` | clutch force (log-uniform), asymmetry, curve exponent, capture gates, and which of `ramp`/`plateau`/`detent` this episode uses |
| `contact` | brick, pad and table friction, and the stud diameter |
| `plate` | the build plate's pose, +-3 mm and +-2 degrees |
| `in_hand` | the in-hand offset and yaw of the held brick, and the soft weld's compliance |
| `visual` | lights, camera extrinsics and field of view, brick colour, and the image post-process |

`level` is a number in [0, 1] meaning "how much of the range to use". Level 0 gives every
parameter its nominal value exactly, so `apply(scene, sample(rng, 0))` leaves the model bit-for-bit
as built and is the control condition every sweep should include. (The task curriculum's own
`level` 0..5 is a different number and lives in the environments.)

Order matters: **randomize, then reset.** `apply` writes the model; `reset_scene` then lays the
bricks out against the plate pose that was just chosen and re-binds the snap model, which is also
what restores the friction-joint regularization after a link-mass change forces MuJoCo to recompute its
constants.

Three things the sampler does not do the obvious way, each for a reason:

- The supply voltage is sampled but is *not* applied a second time to the gain and torque limit.
  The published ranges for gain (7-20 N.m/rad) and torque limit (1.5-3.4 N.m) already span the
  whole 6.8-7.6 V band plus the datasheet-versus-BAM disagreement, so multiplying again would push
  the model outside the ranges this module promises to stay inside. The voltage still reaches the
  servo emulation through `randomize.servo_config`, where it changes the current and load readings.
- The visual group writes lights, cameras and colours straight into the compiled model rather than
  going through `render.domain_rand`'s scene-side helpers, which take an uncompiled `MjSpec` and
  would mean recompiling once per episode. The sampling and the colour maths still come from that
  module; only the writes are local.
- Brick friction is 0.35 (design report 01 section 4), set by the scene on the compiled geoms.
  `bricks/generator.py` leaves them at MuJoCo's default of 1.0, which is a placeholder rather than
  a measurement, and changing it there would move the bench tests that are calibrated against it.

### Measured, on this machine

| | |
|---|---|
| build + compile, 8x8 plate + 2 placed + held brick | 0.15 s |
| `reset_scene` | 3.8 ms |
| physics steps/s, arm + 8x8 plate + held 2x4 | 1810 |
| ... + 2 placed bricks | 1500 |
| ... + a tray of 3 more bricks | 560 |
| settle, 2 s, any of the above | zero MuJoCo warnings, every placed brick seated, residual speed under 0.6 mm/s |

Those rates are physics steps, not environment steps: at the project's 10 physics steps per
control tick, and before the servo bus and any rendering are added, the middle row is about 150
control ticks per second in one process.
