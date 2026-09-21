# Perception

**Status: implemented.** The **slot-state interface** (`scene_model.py`, `ideal_scene_model.py`,
`conventional_scene_model.py`, `sources.py`, `slot_ops.py`, `views.py`) is what the controller
consumes: a small, per-object set of numbers per frame, read and written by name, never decoded
pixel by pixel on the control path. The **per-pixel decode machinery** (`tokens.py`,
`producers.py`, `readout.py`) still exists and is still tested, but is **viewer and evaluation
only** — a way to draw what a scene model currently believes, or to measure one against ground
truth, never something `brick_builder.control` or `brick_builder.action` may import. A trained
convolutional scene model ships with the repository (see
[Convolutional scene model](06b-conventional-vision-model.md)).

## Purpose

The **scene model** is a handful of per-object "slot states" the controller reads directly,
produced either from simulator ground truth (training and testing) or from a real image (on the
robot).

## The scene-model interface: `SceneState`

The controller does not look at pixels. Every vision tick it reads a `scene_model.SceneState`: up
to `K = 8` **slots**, each a small, named row of numbers for one object — position, rough size and
orientation, colour — plus one ordinal depth value per slot and a persistent identity so the same
object keeps the same slot as long as it stays in view. How the numbers were produced (rendered
ground truth, classical colour thresholding, a future learned model) is invisible in the
interface; every source publishes the same shape.

```python
from brick_builder.perception import SceneState, PLACEHOLDER_SLOT_SCHEMA

state: SceneState = some_source.observe(rgb, window=view, ...)
state.K                       # 8
state.slot_ids                # [K] a persistent id per slot, -1 = empty/unclaimed
state.depth                   # [K] ordinal, nearer = smaller, 0.5 = empty
for k in range(state.K):
    if state.is_empty(k):
        continue
    state.mass(k)              # fraction of the window's pixels this object claims
    state.centre(k)            # (x, y), normalized window coordinates in [-1, 1]
    state.orientation_and_axes(k)  # (angle, (major, minor)) from the object's own second moments
    state.colour(k)             # mean colour, 2*rgb - 1
```

Every accessor goes through `state.schema` (a `SlotSchema`): a source declares which named fields
its row has (`schema.has("colour")`), and `SceneState.mass`/`centre`/`moments`/`colour`/
`colour_spread`/`border_fraction`/`age`/`orientation_and_axes` read through that declaration
rather than a hard-coded column index — a source with a different row width or field set plugs
into the same interface, since a consumer asks the schema rather than assuming.

### `PLACEHOLDER_SLOT_SCHEMA`: the shared slot layout

Every source publishes one shared, conventional layout — a colour-and-position blob per object,
the kind of output a classical instance-segmentation pipeline would give:

| columns | field | meaning |
|---|---|---|
| `[0]` | `mass` | fraction of the window's pixels this object claims |
| `[1:3]` | `centre` | `(x, y)`, normalized window coordinates, `[-1, 1]` |
| `[3:6]` | `moments` | `sxx, sxy, syy` — second central moments of the object's own pixel `(x, y)` |
| `[6:9]` | `colour` | mean colour, `2 * rgb - 1` |
| `[9]` | `colour_spread` | mean squared deviation of pixel colour from the mean |
| `[10]` | `border` | fraction of the object's pixels on the window's own edge (it is cut off) |
| `[11]` | `age` | `tanh(frames_active / 10)` — 0 when just seen, rising toward 1 |
| `[12:14]` | `orient4` | fourth-order complex moment, real and imaginary parts. Second moments give an orientation only for an elongated blob; this carries the angle of a four-fold symmetric one (a 2×2 brick's footprint), modulo 90°, with `abs(c4)` as its strength |

`orientation_and_axes` is not a stored field: it is the eigen-decomposition of the 2x2 moment
matrix, computed on request, so no two representations of the same ellipse need to stay in sync.
An empty slot is all zeros, `depth` exactly `0.5`.

### Sources: who fills in `PLACEHOLDER_SLOT_SCHEMA`

All behind one `scene_model.SceneModelSource` protocol — `observe(rgb, *, window, instance=None,
depth=None, present_mask=None, t=None) -> SceneState`:

1. **`ideal_scene_model.IdealSceneModel`** (the simulator's default). Built from
   `brick_builder.render.renderer.Frame`'s ground-truth instance segmentation and metric depth: every
   distinct instance id is a candidate object (except the renderer's "unlabelled" ids, `-1` and
   `0`), each mapped to its own slot up to the `K - 1` cap (the largest kept, the rest folded into
   one background slot), with a per-pixel ground-truth label map exposed through
   `SceneState.decode` for evaluation/figures. Identity is free: the simulator's own instance id
   is the stable id `TemporalSlotAssigner` needs. Four degradation knobs (amodal statistics,
   boundary noise, same-colour merging, label noise) are off by default.
2. **`conventional_scene_model.ColourSegmentationSceneModel`** — a real, working, RGB-only source:
   classical HSV/RGB colour thresholding (`producers.hsv_colour_distance`'s embedding) plus
   connected components, no learned model and no simulator ground truth. What the real robot can
   run today. No simulator id, so identity comes from matching this frame's blobs against the
   previous frame's slot rows (`scipy.optimize.linear_sum_assignment`, a proper one-to-one match, not
   greedy nearest-neighbour).
3. **`conventional_scene_model.LearnedSceneModel`** (alias `ConventionalGpuSceneModel`) — the
   trained convolutional scene model (`brick_builder.perception.cnn.source.CnnSceneModel`): RGB
   only, 2.37 M parameters, trained on simulator renders, runs on CPU or GPU. Architecture,
   training, evaluation and the viewer API: [Convolutional scene model](06b-conventional-vision-model.md).

### Registering another source

`brick_builder.perception.sources` maps a name to a factory: `sources.register("my_model", lambda
**kw: MyModelSceneModel(**kw))` makes it reachable as `sources.get("my_model")(...)`; the three
sources above register themselves the same way under `ideal`, `colour` and `learned`. The factory
must return something satisfying `scene_model.SceneModelSource`; a source whose row layout is not
a documented set of named fields publishes `scene_model.opaque_schema(W)` rather than claiming
`PLACEHOLDER_SLOT_SCHEMA`'s fields.

### Identity: `TemporalSlotAssigner`

Slot order carries no meaning frame to frame, so every source runs its candidate objects through
one shared `scene_model.TemporalSlotAssigner`: first come, first served; an object keeps its slot
while visible; a vanished object's slot is held a few frames (brief occlusion) then freed; when
more than `K - 1` objects are visible, the largest are kept; the unclaimed slot is background.
Identity comes from the simulator's own id when there is one (`IdealSceneModel`), or a
minimum-cost match against the previous frame's rows when there is not
(`ColourSegmentationSceneModel`).

### `views.py`: the window a scene state is computed in

Every `SceneState` carries the `views.LookPose` (`window`) its 256x256 frame was cropped from.
Three fixed, pre-measured poses/crops exist (`PLATE_VIEW`, the whole 8x8 build plate;
`PLATE_VIEW_DENSE`, a denser 4x4-stud window; `TRAY_VIEW`, the loose-brick tray). The wrist camera
moves every tick under closed-loop control, so `attention_view` picks the smallest square crop
that keeps a given 3D region of interest in frame from wherever the camera currently is, with a
margin; the three fixed poses are a special case of this mechanism, computed once offline for a
fixed ROI at a fixed pose. `window_from_slots` is the slot-state-only counterpart: it picks the
next crop from a `SceneState`'s own published slot centres and extents alone (no decode, no ground
truth) — what a controller reading only slot states needs to narrow in on "the target cell and the
brick in the gripper."

### `slot_ops.py`: controller operations on a `SceneState`

- `slot_delta(intended, actual)` / `slot_distance(intended, actual)` — per-slot and scalar
  comparison between a predicted or target state and the observed state.
- `select_slot(state, colour=..., size=...)` — finds "the red 2x4" from slot numbers alone (mean
  colour, mass, the moment ellipse's axis lengths), never a decode.
- `set_slot_blob(state, k, centre=..., ...)` / `set_slot_offset(state, k, reference_slot=...,
  offset=...)` — build an edited, *intended* state from the current one; `set_slot_offset`
  expresses "the held brick belongs over stud cell `(x, y)` of the plate" purely from the plate
  blob's own centre, orientation and extent, with no camera geometry involved.
- `LogRecord`/`write_log`/`read_log` — `(slot state, joint state, action)` tuples at the vision
  rate, written to and read from one `.npz` archive, for training a future action model.

## Pixel units, not metric units

Everything in `tokens.py` and `producers.py` is in **pixel** units: an `ObjectToken`'s `x`, `y`
and `size` are pixel coordinates in the source frame, not millimetres. A metric position needs the
calibrated camera pose (`perception/backproject.py`).

## `readout.py`: viewer and evaluation only

`readout.py` builds on the per-pixel decode (`SceneState.decode`, when a source provides one —
`IdealSceneModel` always does, `ColourSegmentationSceneModel` never does) plus the RGB frame and a
view's camera geometry to read **metric** brick positions, occupancy grids, and tray contents —
`read_objects`, `read_plate_grid`, `read_tray`. Not what the controller reads: a pixel-precise
reader for a human-facing figure or for measuring a scene model's accuracy against ground truth.
`tests/perception/test_readout.py` enforces the boundary with a static source scan: nothing under
`brick_builder/control/` or `brick_builder/action/` may import `readout.py`.

Two DUPLO colours are hard to separate by colour alone. The baseplate's green is close enough to a
green *brick* that only a size/position check tells them apart. The stock jaw plastic (`rgba 1
0.82 0.12`, measured from the vendored SO-101 MJCF materials) sits almost exactly on DUPLO
"yellow" in the HSV-cone distance `producers.hsv_colour_distance` uses (~0.055, against a ~0.40
separation for every other colour pair); every candidate region is therefore also checked
geometrically — does its footprint, back-projected onto a legal stacking plane, come out a legal
brick size at a plate position? A patch of yellow robot casing fails that check despite passing on
colour. Per-region colour matching is a **per-pixel plurality vote**, not the region's mean
colour: MuJoCo's Phong shading puts a lit brick's pixels anywhere from a near-white specular
highlight to near-black shadow, and averaging RGB first (an earlier version) washes a saturated
colour toward grey before comparison — a lit red brick's mean pixel measures 0.44+ from "red" in
this metric, outside the 0.40 tolerance, though most individual pixels match confidently.

## `tokens.py`: query sets and slot statistics

- `grid_queries(step, res=256) -> np.ndarray [n,2]` — an evenly spaced `(col, row)` pixel grid,
  `step` pixels apart, centred in the frame. For broad, task-aligned coverage: a baseplate stud
  grid, a coarse global scan.
- `region_queries(mask) -> np.ndarray [n,2]` — every `(col, row)` pixel inside a boolean `[H, W]`
  mask. For a specific region a LOOK skill has already picked out: the gripper's footprint, a tray
  pocket.
- `slot_statistics(logits, px, frames) -> SlotStats` — turns a batch of decode logits (`[B,n,K]`) at
  query pixels `px` (`[B,n,2]`) into per-slot `mass`, `centroid`, `var`/`covar` (position spread),
  and `mean_colour`, all `[B,K,...]`, by soft-assigning each query to a slot with `softmax` over
  the K logits. A slot with negligible mass reports `nan` statistics rather than a meaningless
  centroid.
- `ObjectToken` — the dataclass the control code consumes: `x, y, yaw, layer, colour, size,
  confidence, age, slot`.
- `plot_decode_overlay(frame, px, logits, save_path=None)` — scatters query pixels over the frame,
  coloured by their argmax slot; a figure function for inspecting a decode.

## `producers.py`: the token interface and its producers

Nothing in the control code (skills, the RL policy, the FSM) talks to a vision model directly:
everything consumes **object tokens** from one of two interchangeable producers behind the
`TokenProducer` protocol (`produce(rng, *, frame=None, ground_truth=None) -> list[ObjectToken]`):

1. **`GroundTruthProducer`** (bulk RL training). Reads a list of `GroundTruthObject`s (the
   simulator's own object state, already projected to pixels by the caller) and applies a
   parametric noise model — `bias_px`, `jitter_std_px`, `yaw_jitter_rad`, `dropout_p`, same-colour
   `merge_radius_px` — fitted to a real vision model's measured error statistics, so the policy
   trains against that error profile without paying the model's compute every step. Two objects of
   the same colour within `merge_radius_px` are reported as one merged token, modelling the
   structural limit that same-colour touching objects are separable only by position and that K=8
   can run out.
2. **`BlobProducer`** (classical fallback). HSV/RGB colour thresholding against a small palette
   plus connected components, no learned model. Reports confidence 1.0 always (no model
   uncertainty) and layer 0 always (a flat colour mask cannot tell layers apart).

A further producer is added out of tree: implement `TokenProducer` and hand the instance to
whatever selects one.

Since control code only sees tokens, never which producer generated them, switching producers —
or evaluating with one and training with another — never touches the control code. The RL
environment `AlignFromTokens-v0` (see [Environments](04-environments.md)) trains with the fast
ground-truth-with-noise producer and evaluates with a real perception stack in the loop,
exercising the complete camera-to-tokens-to-policy-to-servo-to-physics path end to end.

**Task-aligned queries**, rather than decoding every pixel, build object tokens: baseplate
stud-grid queries become an occupancy grid, tray-pocket queries an inventory count, gripper-region
queries reveal what (if anything) is held and its offset in the gripper, and a coarse global-grid
query flags unknown objects or a human disturbance in frame.

## Measured performance: the scene-model interface

![Slot blobs on the baseplate view, coloured by slot with index and mass](../media/perception/slot_blobs.png)

`IdealSceneModel` decodes ground truth exactly by construction
(`tests/perception/test_ideal_scene_model.py::test_decode_reproduces_ground_truth_partition_exactly`),
so the numbers of interest are about identity and occlusion, not label accuracy:

- **Identity stability.** On a rendered scene observed twice with unchanged ground truth,
  `IdealSceneModel` and `ColourSegmentationSceneModel` both keep every object's slot index
  unchanged (`test_identity_persists_across_frames_on_a_rendered_scene`,
  `test_identity_persists_across_frames_via_hungarian_matching`); `ColourSegmentationSceneModel`
  also keeps a single moving blob in its own slot across a swept sequence of frames
  (`test_identity_tracks_a_slowly_moving_blob`). The figure below shows this visually: the same
  five bricks keep the same slot colours and indices across a five-tick camera sweep
  (`attention_view` re-cropping every tick).

![Slot identity persisting across a wrist-camera sweep](../media/perception/identity_sweep.png)

- **Occlusion bias, held brick over the plate** (12 randomized trials, default 2x4 held grasp with
  ±6 mm/±0.3 rad in-hand jitter, `IdealSceneModel`, `PLATE_VIEW`): the held brick was visible in
  every trial; its ideal-slot centroid sat a median **3.9 mm** from its true, projected centre
  (`SceneState.centre` vs. the true body pose projected through the same camera geometry) — not
  sensor error (the ideal source is exact per-pixel), but the second-moment centroid not coinciding
  with the geometric centre once the brick's visible silhouette is cropped/skewed by the view. The
  held brick's own visible fraction (modal mass over an amodal, convex-hull estimate of its
  un-occluded extent) had a median of **0.96**. A stud cell near the plate's centre — the natural
  target for "place the held brick here" — was almost never visible as bare plate underneath the
  held brick and gripper: median visible fraction **0.03** across the same trials. A controller
  aligning a held brick over a target cell therefore cannot use the cell itself as a visual
  reference, since the brick and jaws cover it; what stays in view the whole way down is the held
  brick's own blob and, beyond and beside it, the baseplate's blob and the bricks already placed —
  alignment is servoed on the held brick relative to those, every frame, never on the hidden cell.
  The 3.9 mm centroid bias does not hurt that comparison: the target scene state for the step is
  rendered with the same jaws and the same brick in the same place, so it carries the same bias and
  the two cancel.

![Held-brick centroid error and visible fraction under occlusion](../media/perception/occlusion_bias.png)

**`readout.py` (viewer/evaluation only), measured on 40 randomized rendered scenes** (1-6 bricks,
up to 3 layers, all six colours except a baseplate-coloured brick directly on the baseplate,
`PLATE_VIEW`, `IdealSceneModel`, CPU; `tests/perception/test_readout.py`) — fewer scenes than the
original target of 200, to keep the test suite bounded on shared hardware, but real,
un-cherry-picked numbers from the current implementation:

| | overall | unoccluded | occluded |
|---|---|---|---|
| bricks (n) | 145 | 97 | 48 |
| detected | 58% | 55% | 65% |
| position error, median | 8.2 mm | 6.8 mm | 8.4 mm |
| position error, 95th pct | 24.8 mm | 20.4 mm | 25.3 mm |
| yaw error, median | 2.0° | 2.1° | 1.9° |
| colour accuracy | 54% | 81% | 6% |
| size accuracy | 69% | 77% | 55% |
| layer accuracy | 44% | 55% | 26% |

Mean **87 ms/readout**. None of this meets the original acceptance targets (median position error
under 1 mm / 2.5 mm at the 95th percentile, yaw under 2 degrees, size/colour/layer at least 99%,
all for unoccluded bricks) — `tests/perception/test_readout.py` keeps those as
`xfail(strict=False)` with these numbers as the documented reason, rather than deleting or
lowering the target. Detection rate, not just accuracy given a detection, is the first-order
problem: a many-brick scene routinely leaves a legitimate on-plate brick undetected, most often
because its colour vote fell just short of confidence or its fitted footprint, at the query grid's
resolution, missed rounding to a legal stud count. Colour accuracy has the sharpest
occluded-vs-unoccluded gap (81% vs. 6%): a partly-buried brick's remaining visible pixels are
disproportionately side faces and edge/shadow pixels, the pixels the per-pixel colour vote is
least confident about. This module is secondary to the slot-state interface above; the numbers are
for whoever next tunes it, not a target this deliverable's acceptance depends on.

## Where the rest of it lives

- **`backproject.py`** — pixel to metric position; see "From tokens to a target: AlignFromTokens"
  below.
- **Slot identity** — `scene_model.TemporalSlotAssigner`: first-come-first-served slot claims,
  vanish-then-reuse, Hungarian matching for sources without a simulator id. See
  "Identity: `TemporalSlotAssigner`" above.
- **Learned per-slot heads** — an alternative to closed-form decoding, not built. Superseded in
  intent by the trained convolutional scene model (`LearnedSceneModel`).

## From tokens to a target: AlignFromTokens

`brick_builder/perception/backproject.py` turns a producer's pixel into a world position, using the
camera's actual geometry. `CameraGeometry` reads intrinsics off `model.cam_fovy` and the render
resolution (`fx = fy = (height / 2) / tan(fovy / 2)` in pixels — MuJoCo stretches the horizontal
field of view to match the render's aspect ratio, which is what makes `fx == fy` true) and
extrinsics off `data.cam_xpos` / `data.cam_xmat` (the same `R @ local = world` convention
`ArmKinematics` and `scene/compose.py` use elsewhere). `pixel_rays` turns a `(col, row)` into a
world-frame ray; `backproject_to_plane` intersects it with a horizontal plane at a chosen height;
`project_point` is the exact inverse (world point to pixel), needed by the ground-truth-with-noise
producer as much as the real path. Both raise `ValueError` for a point or plane behind the camera,
or a ray parallel to the plane, rather than a large or `nan` number.

`tests/perception/test_backproject.py` checks this against an actual compiled MuJoCo camera (a
real `model`/`data`, forwarded once) and an independently written pinhole-projection formula, for
a camera 0.25 m above a plane: under 0.5 mm of error at the image centre, under 1.5 mm at the
corners of a 25-degree-tilted camera — both inside the spec's bound.

### `BrickBuilder/AlignFromTokens-v0`

A single brick lies loose near the plate's centre, at a continuous (not stud-grid-quantized)
random `(x, y, yaw)`. The policy never sees its true pose: only a token from one of the three
producers, turned into a metric position and handed to the actor as `align_token = (x, y, yaw,
confidence, age)` — `x`/`y` relative to the fixed `LOOK_TCP` pose the episode starts at. The
critic additionally sees `align_true_error = (pos_err_m, yaw_err_rad)`, the true
distance/orientation error to the brick's actual grasp-ready pose. Horizon 150 ticks (3 s);
success is the spec's 3 mm / 5 degrees for 10 ticks.

**Target placement.** The spec says "a brick lies on the table"; the built scene instead places it
loose on top of the baseplate, near its centre. A grid IK sweep of tool-down poses over the plate
found a dead zone: nothing between roughly 9 cm and 30 cm of hover height is reachable with the
tool pointing straight down at the plate's centre, so the only tool-down wrist view showing the
plate at all is a low hover a few centimetres above it. The wrist camera, mounted a few
centimetres from the fixed jaw, shows only a modest (several-centimetre) patch of world at any
pose regardless of height (the jaw fills the rest of the frame), and the baseplate already
occupies that one reachable, visible neighbourhood — hence the target brick sits there, resting on
the studs (the `bricks/generator.py` "misaligned enough to sit stud_h higher than flush" regime),
not snapped to the grid. `docs/media/envs/align_look_pose.png` is the actual rendered wrist-camera
view at `LOOK_TCP`.

![AlignFromTokens viewing pose, arm camera POV](../media/envs/align_look_pose.png)

**The scripted go-to-token controller.** The primitive is `brick_builder.control.primitives.MoveTo`,
re-aimed at the current token estimate every time a fresh one arrives, ramped from wherever the
tool is now — never an instant jump: `ArmEnv`'s per-tick IK warm-start is tuned for target motion
of at most a couple of millimetres/degrees a tick, and cannot follow a five-centimetre, one-tick
jump. Orientation needs the same ramping, for a separate reason: a brick's yaw is drawn from the
full circle, but a rectangular brick grasped across its width looks the same rotated 180°, and the
SO-101's wrist-roll joint covers only about ±160° — short of a full turn. Every yaw this
environment turns into a commanded tool spin is first folded into `(-90, 90]` degrees using that
grasp symmetry (`AlignFromTokensEnv._grasp_yaw`), keeping every request inside the joint's range;
the commanded spin is ramped at a fixed rate (separately from `MoveTo`, which only ramps position),
for the same small-step reason. `AlignFromTokensEnv._retarget_mover` resets `MoveTo`'s ramp only
when the new token estimate has moved past the token's own noise floor, so re-detecting the same
estimate every tick while the tool sits still at the viewing pose does not reset the ramp's clock
and strand the arm at the viewing pose.

**Measured performance** (`brick_builder.envs.align_env.measure_success_vs_noise_age_producer`, 12
episodes per condition, zero-residual scripted controller, CPU, on a shared machine — the timings
are not clean single-tenant numbers):

![Success vs. token noise, age and producer](../media/envs/align_success_vs_noise.png)

| token noise (mm, ground truth with noise) | success rate (n=12) |
|---|---|
| 0 | 16.7 % |
| 1 | 33.3 % |
| 2 | 25.0 % |
| 4 | 8.3 % |
| 8 | 0.0 % |

| token age before first reading (s) | success rate (n=12) |
|---|---|
| 0.0 | 25.0 % |
| 0.2 | 16.7 % |
| 0.5 | 16.7 % |
| 1.0 | 16.7 % |

| producer | success rate | mean token error (mm) | ms / observation |
|---|---|---|---|
| ground truth with noise (4 mm nominal, n=12) | 8.3 % | 5.8 | 0.4 |
| blob (n=12) | 0.0 % | 11.3 | 13.9 |

Twelve scenes per condition is a small sample: the noise curve's rise from 0 to 1 mm before it
falls is sampling noise (a difference of two episodes out of twelve), not a real
non-monotonicity, and none of these numbers should be read to more than about one significant
figure. The overall trend still holds: success falls off as noise grows, from a modest floor at
0-2 mm to zero at 8 mm. Even at zero noise the scripted controller's success rate is well under
100%: the same servo-level Coulomb-friction deadband (about 4-5 mm at this arm's gains) documented
for `Reach`'s hand-written P controller, which a purely scripted approach — no learned dithering
residual — cannot beat either. The age sweep is close to flat by construction: the brick is static
in this scene, so an old-but-correct reading is exactly as good as a fresh one, and the small
change with age is almost entirely the horizon eating into the controller's own travel-and-settle
time rather than positional drift — a limitation of this particular scene (nothing here moves on
its own) that a future moving-target version of this task would need to re-measure.

No policy was trained for this environment: the scripted go-to-token controller above is the
specified deliverable ("training a policy here is optional").
