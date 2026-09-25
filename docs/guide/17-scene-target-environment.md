# Scene-target environment

`BrickBuilder/SceneTarget-v0` (`brick_builder/rl/scene_target/`) is a Gymnasium environment for
reaching a target **sequence** of scene states. One step is one vision tick (100 ms): five 20 ms
writes to the servo bus, one wrist frame, one scene-model pass. At tick k the target is
`S*_{k+1}`. Reward, the progress index and representation success are computed in the scene
model's own representation; ground-truth success is scored against the simulator.

This version implements the two viewing-pose stages (L0a, L0b), the scene-state codec, the
target-sequence cache and the per-source calibration. The placement stages (L1-L3) are declared
and refused. No predictor or controller is trained here; `u = 0` is the only controller
exercised.

```python
import gymnasium as gym
import numpy as np
import brick_builder.rl.scene_target  # registers the id

env = gym.make("BrickBuilder/SceneTarget-v0", level="viewing_pose", source="ideal")
obs, info = env.reset(seed=0, options={"world_seed": 0})
obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, np.float32))
```

The convolutional scene model: `source="learned", source_kwargs={"weights":
"brick_builder/perception/weights/vision_model.pt", "threads": 2}`. The checkpoint is loaded by
explicit path (the shipped file when `weights` is omitted) and its sha256 is in every `info`.
CPU only.

## Package

| Module | Contents |
|---|---|
| `codec.py` | `SlotFrame`, appearance matching, `SceneDistance` [mm-eq], `Calibration` files |
| `targets.py` | `TargetSequence` (arrays, `.npz`), viewing-pose generator, cache and its command, placement-recording converter |
| `world.py` | empty-hand world, kinematic twin, scene-model sources |
| `executor.py` | action to offset, offset to the tick's five writes |
| `rules.py` | progress index, reward, ground-truth pose error |
| `env.py` | `SceneTargetEnv` |
| `wrappers.py` | `PredictionScoring` |
| `bench.py` | calibration, throughput, open-loop replay |
| `config.py` | `SceneTargetConfig`, stages, reward weights |

## Stages

| Code | `level` | World | Feed-forward | Action | Scale per tick | Offset ceiling | Max hold | Max ticks | Status |
|---|---|---|---|---|---|---|---|---|---|
| L0a | `viewing_pose` | plate, yellow 2x4, red 2x2, empty hand; nominal | off | dx, dy, dz, dyaw | 3 mm, 3 mm, 2 mm, 2 deg | 50 mm (xy), 30 mm (z), 20 deg | 30 | T + 30 | implemented |
| L0b | `viewing_pose_displaced` | same, plate 4-8 mm and 2-3 deg off, times `displacement_scale` | on | dx, dy, dyaw | 2 mm, 2 mm, 1.5 deg | 20 mm, 10 deg | 30 | T + 30 | implemented |
| L1 | `align` | `build_world` | on | dx, dy, dyaw | 2 mm, 2 mm, 1.5 deg | 20 mm, 10 deg | 70 | 1.5 T + 70 | declared |
| L2 | `placement` | `build_world` | on | dx, dy, dyaw | same | same | 70 | 1.5 T + 70 | declared |
| L3 | `pick_and_place` | `build_pick_world` | on | dx, dy, dyaw | same | same | 70 | 1.5 T + 70 | declared |

The plate displacement is `action.worlds.displacement(seed)`. L0a and L0b share one target
sequence per seed, recorded in the nominal twin.

## Target sequences

A `TargetSequence` is arrays only and is read with `allow_pickle=False`:

| Field | Shape | Contents |
|---|---|---|
| `rows`, `depth`, `valid`, `slot_ids` | [T, 8, W], [T, 8] x 3 | `S*_0..S*_{T-1}` in the source's slot order |
| `commands`, `n_commands` | [T, 5, 7], [T] | per frame, the 20 ms writes: pos (3), spin, grip, c_vert, c_lat |
| `action` | [T, 4] | net tool-target change over the frame (dx, dy, dz, dyaw) |
| `step_index`, `titles`, `freeze_k` | [T], [n], scalar | step of each frame, step titles, first frozen frame |
| `tool_pose` | [T, 4] | tool target when `S*_k` was seen; ground-truth scoring only |
| `meta` | JSON | family, seed, source, weights sha256, W, schema, generator, draw |

Cache: `data_cache/scene_target/targets/<family>/seed<seed>_<source>_<sha8>.npz`.

**Viewing-pose generator.** Target pose relative to the nominal plate centre: xy within
+-30 mm, TCP height 45-95 mm (the convolutional model's training range), yaw within +-15 deg.
Start pose 10-40 mm and 0-10 deg from it. Minimum-jerk path `s = 10 t^3 - 15 t^4 + 6 t^5`,
T = 20-40 frames at 100 ms, five interpolated writes per frame. Every frame is rendered in the
kinematic twin and passed in order through one fresh instance of the configured source. A draw
is rejected when inverse kinematics fails on any frame or when a frame has fewer than two
reference slots (border fraction at most 0.25, mass 0.0012-0.5) matched to its neighbour.

**Placement recordings.** `targets.from_ideal_sequence` converts an `IdealSequence` from
`action.place.record_ideal`; the arrays equal its fields (tested on a real placement recording).

| Twin frame cost | Frames | Per frame [ms] | Command |
|---|---|---|---|
| Ground truth | 40 sequences (seeds 0-9, 20-49) | mean 25.0, max 34.7 | `targets --source ideal --seeds 0-9,20-49` |
| Convolutional, 2 threads | 40 sequences (seeds 0-9, 20-49) | mean 89.2, max 117.5 | `targets --source learned --threads 2 --weights .../vision_model.pt --seeds 0-9,20-49` |
| Ground truth | seed 7, T = 40 | 24.5 (render 11.9, scene model 12.6) | `test_ground_truth_sequence_of_40_frames_round_trips_bitwise` |
| Convolutional, 1 thread | seed 2, T = 20 | 102.7 | `test_convolutional_sequence_round_trips_without_pickling_and_reencodes_bitwise` |

Both sources accepted the same draw and length on every seed (seed 24 needed seven draws).
Fewest matched reference slots per sequence, convolutional: 4 on 31 seeds, 3 on 6, 2 on 3. The
convolutional caches are for the shipped weights (sha256 `3e07a382...`).

## Scene-state distance

```text
e_i   = |d centre_i|^2 + 0.25 sum_f (s_c d f_i / s_f)^2        f: mass, moments, orient4, depth
D_abs = sqrt( sum_i w_i e_i / sum_i w_i ),   w_i = 1 - border_i
D_rel = RMS_j |(c_held - c_j) - (c*_held - c*_j)|               only with a matched held brick
D_mm  = mm_scale (D_abs + D_rel) + 2.0 n_missing
```

| Field | Scale |
|---|---|
| centre (s_c) | 0.01 window units |
| mass | 0.002 |
| moments | 0.002 |
| orient4 | 0.2 |
| depth | 0.25 |

- **Matching.** Hungarian assignment on colour (scale 0.3), log mass (1.0) and normalized
  second-moment shape (0.5), plus 0.05 per window unit of centre distance as a tie-break
  between objects of identical appearance. Pairs costing more than 1.5 stay unmatched; the
  background slot (valid, negative slot id) is never matched. One assignment per call.
- **Missing slots.** `n_missing` counts valid, non-background target slots without a match.
- **Opaque schema.** Any width W: whitened matched RMS plus the RMS of whitened pairwise
  differences, times `mm_scale`, with per-feature scales from the calibration file.
- **`mm_scale`.** 87.4 mm per window unit (the align-hover value) without a calibration file;
  the calibrated value otherwise (next section).

## Calibration

One file per source, weights and target family:
`data_cache/scene_target/calib/<source>_<sha8>_<family>.json`, written by `bench.py` before any
other measurement and loaded by the environment when present (`info["calibration_file"]`,
`info["mm_scale"]`, `info["tol_mm"]`).

| Quantity | Measurement | Use |
|---|---|---|
| `mm_scale` | kinematic twin at the target pose of training seeds 1000-1039 (five offsets each) and at that pose offset 0.5-5 mm in the plane; tool offset [mm] against the distance [window units], least squares through the origin, ground-truth source | distance unit [mm-eq] |
| `sigma_obs_mm` | arm held at the start pose of training seeds 1000-1003, 50 ticks each; standard deviation of the distance between consecutive frames with every target slot matched | `tol_mm = max(0.5, 2 sigma_obs)`; tube floor `3 sigma_obs` |

On the named schema a slot centre is a window coordinate whatever the source, so every source
uses the ground-truth fit. The convolutional source's own fit is kept in its file as a record.

| Source | `mm_scale` [mm per window unit] | Fit samples | Ratio p10-p50-p90 | Fit residual RMS [mm] | Distance per mm of height [mm-eq] | Distance per deg of yaw [mm-eq] | `sigma_obs` [mm-eq] | `tol_mm` [mm-eq] |
|---|---|---|---|---|---|---|---|---|
| Ground truth | 210.5 | 165 | 184-216-250 | 0.39 | 0.53 | 0.87 | 0.047 | 0.50 |
| Convolutional (`3e07a382`) | 210.5 (ground-truth fit) | 165 | 184-216-250 | 0.39 | 0.53 | 0.87 | 1.182 | 2.36 |

With the fitted scale (2.41 times the align-hover value), 1 mm-eq is 1 mm of tool motion in the
plane at viewing-pose heights; height and yaw read weaker (0.53 mm-eq per mm, 0.87 mm-eq per
deg). The convolutional source's own fit is 28.0 mm per window unit (n 141, ratio p10-p50-p90
21-74-138, residual 2.32 mm; 34 samples skipped for an unmatched slot): the fit through the
origin absorbs the source's noise floor. Per fit, 49 of 280 twin poses (in-plane, height and yaw
samples) were unreachable and skipped.

## Observation

| Key | Shape | Contents |
|---|---|---|
| `actor/z`, `z_depth`, `z_valid` | [8, W], [8], [8] | observed slot rows (source order), ordinal depth, non-empty flag |
| `actor/z_matched`, `matched_valid` | [8, W], [8] | observed rows in the order of `S*_{k+1}` |
| `actor/target`, `target_depth`, `target_valid` | [H, 8, W], [H, 8], [H, 8] | `S*_{k+1..k+H}`, last frame repeated |
| `actor/mech` | [18] | joint positions, speeds, loads (servo bus) |
| `actor/tool` | [13] | commanded tool target (x, y, z, yaw, grip), bus-FK tool pose (x, y, z, yaw), accumulated offset (x, y, z, yaw) |
| `actor/feedforward` | [H, 5] | recorded change (dx, dy, dz, dyaw, dgrip); zeros with the feed-forward off |
| `actor/progress` | [12] | k/(T-1), step/(n_steps-1), holds/max_hold, step one-hot (9) |
| `actor/image` | [256, 256, 3] | with `include_image=True` |
| `privileged/gt_z` | [8, 14] | ground-truth rows of the same frame |
| `privileged/critic` | [32] | plate displacement (3), held-brick error (3), in-hand offset (4), TCP pose (4), TCP error to the current target pose in the plate frame (4), contacts (2), joint loads (6), damage level (1), stage one-hot (5) |

H = 5 by default. The actor half contains nothing from the simulator beyond what the bus and the
wrist camera report; a test checks that the plate displacement does not appear in it.

## Action and execution

The action `u` in `[-1, 1]^n` changes the accumulated tool-target offset by `scale * u`; the
offset is held inside the stage's ceiling (norm in the plane, height, yaw). Offsets are in the
nominal-plate world frame. Every tick issues five 20 ms writes:

| Tick | Writes |
|---|---|
| feed-forward on, index advancing | frame k's five recorded writes, each plus the offset |
| feed-forward on, index held | frame k's last recorded write, five times |
| feed-forward off (L0a) | start pose plus the offset, five times |

| `feedforward_clock` | Index before the last frame | `u = 0` |
|---|---|---|
| `progress` (default) | advances when `D <= tube`, holds otherwise; the recorded writes pause while it holds | the recording, paused wherever the arm leaves the tube |
| `recording` | advances every tick | open-loop replay of the recording (the camera-off control condition) |

The last frame follows the settle and hold rule under both clocks. `TrackingEpisode` has the same
split: its camera-off condition plays the index forward. `TrackingEpisode` issues one write on a
held tick; here a step is always 100 ms of physics, so the predictor's time step is fixed.
`info["exec_action"]` is the executed tool-target change over the tick. `action_space="joint"` is
declared and refused in this version.

## Progress index, reward, termination

```text
tube = max(advance_floor_mm, 3 sigma_obs, tube_scale * D(S*_k, S*_{k+1}))     1.0, -, 1.6
k <- k + 1  when D(z_{t+1}, S*_{k+1}) <= tube, else holds += 1                (progress clock)
last frame: settled += 1 when D <= tol_mm (0 otherwise); success at settle_ticks = 3
```

The rule is `TrackingEpisode.run`'s with the tube in mm-eq of scene-state distance;
`rules.progress_update` is tested against it on synthetic error sequences. The index never
advances by more than one frame per tick.

```text
r = 0.2 (D_prev - D_next) - 0.1 min(D_next / tol_mm, 5) + 0.5 [advanced] - 0.02 [held]
    - 0.02 mean(u^2) - 0.02 mean((u - u_prev)^2) - 5 (damage level rise)
terminal: +10 on representation success; -5 on stall, lost view or damage; 0 on timeout
```

Both distances are against the pre-step target `S*_{k+1}`.

| Condition | Flag | `info["failure"]` |
|---|---|---|
| last frame settled | terminated | empty |
| holds > max_hold | terminated | `stalled with a scene-state error`, or `stalled blind` when most held ticks matched no target slot |
| no target slot matched for 20 ticks | terminated | `lost view` |
| damage level at least `cracked` | terminated | `damage` |
| ticks = max ticks | truncated | `timeout` |

## Ground-truth scoring

Simulator TCP pose against the final target pose: within 1.0 mm in the plane, 1.5 mm in
height and 1.0 deg in yaw. On L0b the target pose is carried with the displaced plate.
`info` carries `success_repr`, `success_gt`, `agree`, `gt_tool_err_mm` (against the current
target pose) and `gt_final_xy_mm`, `gt_final_z_mm`, `gt_final_yaw_deg`.

## Determinism

The GPU's 4x multisample resolve is not bit-reproducible: two renders of one state differ on
44-72 of 65,536 output pixels (256 x 256; five twin poses, seeds 0, 1, 2, 5, 6). Instance and
depth maps are exact. The slot rows then differ by 1.1e-5 to 4.2e-5 at most, above the env
checker's 1e-5 tolerance.

| Setting | `gymnasium.utils.env_checker.check_env` |
|---|---|
| default (`render_samples=None`, 4 samples), with or without `timings_in_info` | fails: `env.reset(seed=123)` observations not equivalent |
| `render_samples=0`, `timings_in_info=False` | passes, L0a and L0b (`test_env_checker_with_bit_exact_renders`) |

With the default rendering, two runs of one seed agree exactly in every field except colour and
colour spread (within 1e-3), and in reward and distance. `timings_in_info=False` removes the
wall-clock timings, the only non-deterministic `info` entry.

## Import guard

Importing anything under `brick_builder.action` runs that package's `__init__`, which loads
the analytic servo and `TrackingEpisode`. The environment therefore restates the world
constants, the wrist view and the twin posing in `world.py`; `tests/rl/scene_target/test_world.py`
checks them against `action.worlds`, `action.adapter` and `action.twin.DigitalTwin` (same
configuration, same displacement draw, same joint angles, same instance maps).
`test_guard.py` checks statically that no module imports the action package (except the
offline placement recorder) and, in a fresh interpreter, that a running environment on both
sources has loaded no module under `brick_builder.action`.

## Measured

`python -m brick_builder.rl.scene_target.bench --threads 2 --weights
brick_builder/perception/weights/vision_model.pt`, one process, nice 19, 2 threads
(`OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`), CPU only, commit `c12fca5`. Load average 9.4
at the start and 14.9 at the end on 16 hardware threads (other jobs running). Output:
`results/scene_target/bench.json`.

### Open-loop replay, L0a with the feed-forward on

`u = 0` from the start pose, nominal world, seeds 0-4, calibrated distance.

| Source | Clock | Seed | T | Ticks | Held | End | D [mm-eq] | xy [mm] | z [mm] | yaw [deg] | Representation | Ground truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ground truth | recording | 0 | 40 | 70 | 0 | timeout | 0.49 | 0.47 | -0.12 | 0.08 | no | yes |
| Ground truth | recording | 1 | 36 | 66 | 24 | timeout | 0.87 | 0.66 | 0.10 | 0.03 | no | yes |
| Ground truth | recording | 2 | 28 | 58 | 15 | timeout | 0.55 | 0.50 | 0.06 | -0.01 | no | yes |
| Ground truth | recording | 3 | 27 | 57 | 15 | timeout | 0.68 | 0.70 | 0.08 | -0.13 | no | yes |
| Ground truth | recording | 4 | 27 | 47 | 0 | settled | 0.49 | 0.61 | -0.08 | -0.04 | yes | yes |
| Ground truth | progress | 0 | 40 | 70 | 52 | timeout | 1.10 | 8.44 | -3.11 | -1.55 | no | no |
| Ground truth | progress | 1 | 36 | 66 | 56 | timeout | 1.57 | 19.17 | -0.26 | 2.41 | no | no |
| Ground truth | progress | 2 | 28 | 58 | 38 | timeout | 1.93 | 3.28 | 0.72 | 0.40 | no | no |
| Ground truth | progress | 3 | 27 | 57 | 48 | timeout | 2.41 | 17.93 | -5.36 | -4.24 | no | no |
| Ground truth | progress | 4 | 27 | 57 | 24 | timeout | 0.71 | 0.83 | -0.05 | -0.04 | no | yes |
| Convolutional | recording | 0 | 40 | 70 | 15 | timeout | 5.29 | 0.47 | -0.12 | 0.08 | no | yes |
| Convolutional | recording | 1 | 36 | 66 | 31 | stalled with a scene-state error | 83.85 | 0.66 | 0.10 | 0.03 | no | yes |
| Convolutional | recording | 2 | 28 | 57 | 24 | settled | 1.83 | 0.51 | 0.06 | -0.01 | yes | yes |
| Convolutional | recording | 3 | 27 | 29 | 0 | settled | 1.53 | 1.56 | 0.14 | -0.12 | yes | no |
| Convolutional | recording | 4 | 27 | 46 | 9 | settled | 2.19 | 0.62 | -0.08 | -0.04 | yes | yes |
| Convolutional | progress | 0 | 40 | 31 | 31 | stalled with a scene-state error | 3.64 | 13.61 | -5.88 | -2.86 | no | no |
| Convolutional | progress | 1 | 36 | 31 | 31 | stalled with a scene-state error | 14.04 | 21.57 | -0.91 | 2.81 | no | no |
| Convolutional | progress | 2 | 28 | 31 | 31 | stalled with a scene-state error | 5.43 | 24.36 | 10.09 | 2.98 | no | no |
| Convolutional | progress | 3 | 27 | 35 | 6 | settled | 1.59 | 1.62 | 0.14 | -0.06 | yes | no |
| Convolutional | progress | 4 | 27 | 57 | 36 | timeout | 19.42 | 2.06 | 0.17 | -0.19 | no | no |

| Source | Clock | Ground truth | Representation |
|---|---|---|---|
| Ground truth | recording | 5/5 | 1/5 |
| Ground truth | progress | 1/5 | 0/5 |
| Convolutional | recording | 4/5 | 3/5 |
| Convolutional | progress | 0/5 | 1/5 |

- **Arm lag.** Measured on seeds 0 and 3 with the index forced forward
  (`feedforward=True, advance_floor_mm=1e6, tol_mm=0`), ground-truth source. The physical arm
  trails the kinematic twin: 1.8 mm at the first tick of seed 3 (after the 0.5 s reset settle),
  2.4-2.8 mm while the path moves at 1.3-1.6 mm per tick, then a slow approach after the last
  frame (seed 3: 1.7 mm at T, 1.0 mm at T + 17 ticks, 0.7 mm at T + 30). Held at one command,
  the tool error falls from 1.45 mm to 0.05-0.08 mm over 300 control ticks (6 s). After the
  last frame the distance and the tool error agree to within 0.2 mm; while the arm moves the
  distance reads up to 0.4 mm lower.
- **Recording clock.** The recorded writes play out exactly; every seed ends inside the
  ground-truth tolerance, and four of five run out of ticks before the distance holds under
  `tol_mm = 0.5` for three ticks.
- **Progress clock.** The lag exceeds the 1.0 mm tube floor, the index holds, the recorded
  writes pause, and the episode times out part-way along the path.
- **Convolutional source.** Seed 3 settles in the representation at 1.53 mm-eq with 1.56 mm of
  tool error (`tol_mm = 2.36`); seed 1 ends with 83.85 mm-eq after a slot-matching failure on the
  last frame while the tool is 0.66 mm from the target.

### L0b, `u = 0`

World seed 5 (plate 4.33 mm, 2.64 deg), ground truth, no calibration:

| Clock | Ticks | End | Final xy [mm] | Final yaw [deg] |
|---|---|---|---|---|
| recording | 70 | stalled with a scene-state error | 3.26 | -2.88 |
| progress | 31 | stalled with a scene-state error | 34.89 | - |

With the recording clock the arm lands where the recording goes in the nominal world, off the
displaced target by the plate's yaw; with the progress clock the index holds at the first frame.

### Static arm

Arm held at the start pose of training seeds 1000-1003, 50 ticks each; distance between
consecutive frames.

| Source | Ticks | Mean [mm-eq] | Std [mm-eq] | p50 [mm-eq] | p90 [mm-eq] | Max [mm-eq] | Frames with an unmatched target slot |
|---|---|---|---|---|---|---|---|
| Ground truth | 200 | 0.068 | 0.047 | 0.060 | 0.115 | 0.374 | 0.000 |
| Convolutional | 200 | 1.017 | 1.182 | 0.592 | 2.352 | 8.302 | 0.000 |

### Throughput

Uniform random action, 500 ticks per row, world seeds 0-9 (targets cached), timings are p50
per tick.

| Stage | Source | Ticks/s | Ticks/s with resets | Planning rate [ticks/s] | Measured / planning | Physics [ms] | Render [ms] | Scene model [ms] | Ground-truth pass [ms] | Codec [ms] |
|---|---|---|---|---|---|---|---|---|---|---|
| L0a | Ground truth | 15.98 | 14.37 | 5 | 3.20 | 23.2 | 10.0 | 11.6 | 0.0 | 0.40 |
| L0b | Ground truth | 13.06 | 11.89 | 5 | 2.61 | 23.7 | 9.8 | 11.6 | 0.0 | 0.41 |
| L0a | Convolutional | 4.47 | 4.25 | 2 | 2.24 | 30.0 | 12.8 | 85.3 | 14.7 | 0.53 |
| L0b | Convolutional | 4.75 | 4.49 | 2 | 2.37 | 30.1 | 13.1 | 85.6 | 15.1 | 0.54 |

Physics alone at its p50: 42-43 ticks/s (ground-truth rows) and 33 ticks/s (convolutional rows),
against the 15 ticks/s planning rate. Physics means are 37.5-106.4 ms per tick: load spikes.
Resident memory 1327-1530 MB per process.

### Memory

RSS 1497 MB after 10 and after 50 L0a episodes (`test_memory_stays_flat_over_50_episodes`).

## Limitations

- L1-L3 (prefix, freeze, play-out, `verify_placement`, the grasp-miss rule) are not implemented.
- The joint-space action is not implemented.
- Calibration covers the viewing-pose family only; placement families need their own fit.
- Target frames come from the kinematic twin; the physical arm lags them (see "Open-loop
  replay").

## Commands

```bash
# tests (slow ones render and step the simulator)
.venv/bin/python -m pytest -q tests/rl/scene_target
# target caches, evaluation seeds
.venv/bin/python -m brick_builder.rl.scene_target.targets --source ideal --seeds 0-9,20-49
.venv/bin/python -m brick_builder.rl.scene_target.targets --source learned \
    --weights brick_builder/perception/weights/vision_model.pt --seeds 0-9,20-49
# calibration, throughput, open-loop replay
.venv/bin/python -m brick_builder.rl.scene_target.bench --out results/scene_target/bench.json \
    --weights brick_builder/perception/weights/vision_model.pt
```
