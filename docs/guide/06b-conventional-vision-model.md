# Convolutional scene model

**Status: implemented and trained.** A 2.37 M-parameter convolutional network reads one 256x256
wrist-camera window and publishes the same `SceneState` as every other producer. Code in
`brick_builder/perception/cnn/`, weights in `brick_builder/perception/weights/vision_model.pt`
(4.8 MB).

## Purpose

The controller consumes slot states, not pixels: up to eight rows of named numbers per frame plus
an ordinal per-slot depth ([Perception](06-perception.md)). Two producers filled that interface
before this one: `IdealSceneModel` (simulator ground truth, unavailable on hardware) and
`ColourSegmentationSceneModel` (CPU colour thresholding, fails on touching same-coloured bricks,
lighting change, and green bricks over the green baseplate).

The laptop-scale third option: an ordinary U-Net encoder-decoder trained on simulator frames only.
46 ms/frame on three CPU threads against 21 ms for colour thresholding, for a 2.7x smaller centre
error and a working answer where colour cannot resolve. Fallback when the microcontroller path is
unavailable, second opinion when it is not; any further source plugs into the same interface
unchanged.

## Architecture

Five-stage strided encoder (16/32/64/128/192 channels), dilated bottleneck, four-stage decoder
with skip connections, batch normalization and ReLU throughout, four 1x1 output heads. The decoder
stops at half resolution and the head outputs are bilinearly upsampled to 256x256; a 1x1 head
commutes with bilinear upsampling, so this is equivalent to a full-resolution head at a quarter of
the arithmetic. 1.27 GMAC per frame.

| head | output | purpose |
| --- | --- | --- |
| class | 4 channels | background/table, baseplate, arm (links and both jaws), brick |
| centre | 1 channel | heat map peaking at each brick centre |
| offset | 2 channels | vector from each brick pixel to its own brick's centre, in pixels |
| depth | 1 channel | relative depth, per-frame normalized |

Instance grouping is bottom-up, as in Panoptic-DeepLab: non-maximum suppression over the centre
heat map gives the instance centres; each brick pixel joins the centre its own offset points
nearest to. Colour plays no part, which is what separates two touching bricks of one colour. The
arm and the baseplate carry one renderer instance id each, so for those the class mask is the
instance.

Mask-to-slot conversion reuses the ideal source: `ideal_scene_model._blob_row` for the slot-row
statistics, `TemporalSlotAssigner` for identity across frames, `rank_scale` for the ordinal depth
field. This model is a different way of obtaining masks, not a different scene model.

## Usage

```python
from brick_builder.perception.cnn.source import CnnSceneModel

source = CnnSceneModel()                    # shipped weights, device="cpu"
state = source.observe(rgb, window=pose)    # SceneState, PLACEHOLDER_SLOT_SCHEMA
source.reset()                              # new episode: drop slot identity
```

`rgb` is a 256x256 image and the only input used. `device="cuda"` selects a GPU; `threads=` caps
intra-operation threads. `ConvolutionalSceneModel` is an alias, and
`conventional_scene_model.LearnedSceneModel` (older name `ConventionalGpuSceneModel`) forwards to
this class.

### Viewer API

Display and measurement only; never called from `brick_builder.control`.

```python
maps = source.feature_maps(rgb, window=pose)                       # one forward pass
inst_rgb = source.colourize_instances(maps["instance_map"], maps["state"])
heat_rgb = source.colourize_scalar(maps["centre_heatmap"], 0.0, 1.0)
scores = source.accuracy(maps["instance_map"], gt_instance, gt_class)
```

`feature_maps` returns `class_prob [4,H,W]`, `centre_heatmap [H,W]`, `offset [2,H,W]` in pixels,
`depth [H,W]`, `instance_map [H,W]` (int32, `0` = none, otherwise slot index + 1), `class_map`,
and the `SceneState` for the same frame in the same slot order. `colourize_instances` paints each
slot in the mean colour its own row publishes; `colourize_scalar` applies a perceptually uniform
colour map. `accuracy` returns mIoU over the four classes, per-class IoU, brick recall and
precision at IoU 0.5, and pixel accuracy, in 2.4 ms.

```bash
python -m brick_builder.perception.cnn.demo
```

![Convolutional scene model, six-panel demo](../media/perception_cnn/demo.png)

## Evaluation

Checkpoint `3e07a382`. 300 held-out frames from scenes, colours, lighting and lens settings absent
from training, plus one full pick-and-place episode replayed in order so that slot identity is
exercised. The colour-
threshold source runs on the same frames; it publishes no class map and no masks, so its rows are
matched at slot level by centroid and the pixel-level rows are marked n/a rather than filled with
a different quantity.

| metric | CNN | colour threshold |
| --- | --- | --- |
| IoU background/table | 0.963 | n/a |
| IoU baseplate | 0.862 | n/a |
| IoU arm | 0.925 | n/a |
| IoU brick | 0.760 | n/a |
| mIoU, 4 classes | 0.878 | n/a |
| Brick recall @ IoU 0.5 | 0.690 | 0.744 (centroid match) |
| Brick precision @ IoU 0.5 | 0.830 | n/a |
| Brick recall, touching same colour (n=307) | 0.567 | n/a |
| Brick recall, green on green plate (n=183) | 0.290 | n/a |
| Centre error (px) | 3.20 | 8.31 |
| Centre error at plate (mm) | 3.12 | 8.49 |
| Centre error at plate, p90 (mm) | 8.01 | n/a |
| Orientation error, elongated bricks (deg, n=363) | 8.6 | n/a |
| Depth-order accuracy, brick pairs | 0.795 | n/a |
| Slot match rate vs ideal source | 0.851 | 0.600 |
| Slot-row error: mass | 0.0081 | 0.0326 |
| Slot-row error: centre | 0.0318 | 0.0706 |
| Slot-row error: second moments | 0.0150 | 0.0360 |
| Slot-row error: colour | 0.0495 | 0.2964 |
| Slot changes per 100 tracked frames | 0.0 | 1.9 |
| Latency, 1 CPU thread (ms/frame, median and best) | 78 / 62 | - |
| Latency, 3 CPU threads (ms/frame, median and best) | 46 / 41 | - |
| Latency, 1 CPU thread (ms/frame) | - | 21 |
| Parameters (M) | 2.37 | 0 |
| MAC per frame (G) | 1.27 | - |
| Checkpoint (MB) | 4.8 | - |

![Per-class IoU](../media/perception_cnn/class_overlap.png)

![Held-out frames, ground truth and prediction](../media/perception_cnn/samples.png)

Episode replay, 90 frames in order: mIoU 0.919, brick IoU 0.876, brick recall 0.908, precision
0.921, centre error 2.6 mm, 0 slot changes per 100 tracked frames. Episode frames are easier than
the held-out set: the camera is close and the bricks are large.

### Failure modes

![Failure cases](../media/perception_cnn/failures.png)

![Brick detection vs apparent size](../media/perception_cnn/error_vs_size.png)

- **Green bricks on the green baseplate.** Recall 0.290 against 0.690 overall. The class head
  assigns them to the baseplate, so no centre survives and no instance is formed. This is the
  dominant weakness and the reason such bricks are excluded from tasks by default.
- **Small bricks.** Below 0.6 % of the frame (about 400 px) recall is 0.10. Mask area falls faster
  than the centre-head response, and the 60 px area floor in `GroupingConfig` removes what is
  left.
- **Bricks filling the window.** Above 12 % of the frame (n=7) recall is 0.57 and centre error
  11.3 mm. Such views are rare in the training mix.
- **Touching same-coloured bricks.** Recall 0.567. The grouping separates them more often than
  colour thresholding can, but a shared boundary still merges a fraction of pairs.
- **Depth order.** 0.795 on held-out frames and 0.689 on the episode, where the bricks sit on one
  plane and the true ordering is close to a tie. The field is ordinal and is not a distance.
- **Brick recall against the colour source.** The table's 0.690 (IoU 0.5) and the colour source's
  0.744 are different criteria: the colour source is scored by a 32 px centroid match, which
  passes on blobs whose mask overlap is far below 0.5.

## White bricks on a bright table

### Diagnosis

Per-instance brick IoU by palette colour, 543 held-out frames: blue 0.77, red 0.77, orange 0.72,
yellow 0.69, white 0.61, green 0.34. White is the second-weakest colour, and the gap widens
sharply with distance and table shade: within the 0.09-0.15 m working-distance band white IoU is
0.71-0.74 regardless of the table, but past 0.15 m on a table drawn from the pale half of the
domain-randomization range (mean channel value >= 0.55) it falls to 0.49 (recall 0.61, n=57).
Four inspected frames show the mechanism: a small white or near-white brick sits on top of a
taller stack, framed wide under a light with boosted brightness and specular glare; white's RGB
(0.95, 0.95, 0.95) sits almost on top of the pale end of the table's own randomization range
(0.78, 0.76, 0.72) and of some of the arm's own randomized paint colours (white, grey), so
neither hue nor lightness separates brick from background, and only weak shading and stud-texture
edges are left for the class head to key on -- the same mechanism already documented for green
bricks on the green baseplate, here against the table instead of the baseplate. The effect
compounds with apparent size: white bricks under 400 px are recalled only 4.5% of the time, worse
than the 11.9% cross-colour rate at that size.

![Brick IoU by colour, shipped vs fine-tuned candidate](../media/perception_cnn/white_bricks_fix.png)

### Fix

Targeted data: a new scene kind (`data.py`'s `_bright_light_plate_scene`, split name
`bright_white_v1`, not part of the default generation mix) forces one or two placed bricks white
or yellow, draws the table colour from the pale half of `TABLE_RGB_RANGE` only, and boosts every
light's brightness and specular level -- 876 frames from 55 scenes. Photometric augmentation:
`_augment`'s jitter half-widths (brightness, contrast, per-channel gain, pixel noise) are now a
`photometric_strength` multiplier on the original fixed ranges (see "Fine-tuning from the shipped
checkpoint with extra data" below); this run used 1.8x.

CPU fine-tune from the shipped checkpoint (`--init-weights`), the extra split mixed in at the
frame level (`--extra-split`, never written into the shared `train` split on disk), lr 3e-4 (a
tenth of the original rate, as this guide's real-frame fine-tuning procedure already
recommends), 3 threads. Speed measured first: 3.1-3.3 steps/s at 3 threads, batch 8, crop 160.
Sized for a 45-minute budget (of the 120 allowed); 16 epochs actually ran (19.1 minutes) before
the session that produced this page had to stop it, so the number below is an early read on a
fine-tune that did not use its full budget.

A training run's `_save_weights` used to overwrite the shipped
`brick_builder/perception/weights/vision_model.pt` on every new best epoch, unconditionally --
found while sizing this run's speed probe, which briefly wrote an under-trained 3-epoch
checkpoint there before any gate had run (caught immediately; the shipped file was restored from
git before anything else touched it). Fixed: `TrainConfig.save_shipped` (default on, so a plain
training run behaves exactly as before) gates the shipped-file write separately from a new
per-run `out/best.pt`, and `scripts/train_vision_model.py` defaults it off whenever `--out` is
not `results/vision_model`, so an experimental run can no longer clobber the shipped checkpoint
ahead of its own evaluation.

### Result

543 held-out frames, both checkpoints, same set:

| metric | shipped (`3e07a382`) | candidate (`373f51c3`) | delta | gate |
| --- | --- | --- | --- | --- |
| IoU blue | 0.773 | 0.782 | +0.009 | pass |
| IoU red | 0.774 | 0.766 | -0.008 | pass |
| IoU orange | 0.723 | 0.728 | +0.004 | pass |
| IoU yellow | 0.686 | 0.679 | -0.007 | pass |
| IoU white | 0.614 | 0.638 | +0.025 | pass |
| IoU green | 0.342 | 0.379 | +0.037 | pass |
| White subset IoU | 0.614 | 0.638 | +0.025 | **fail (needs >= +0.05)** |
| White, far + pale table (n=57) | 0.485 | 0.493 | +0.008 | informational |
| mIoU, 4 classes | 0.872 | 0.875 | +0.004 | -- |
| Closed loop, seeds 0-9 (`run_placement.py --cnn`), success | 3/10 | 9/10 | +6 | pass |
| Closed loop, mean lateral error | 4.87 mm | 0.88 mm | -3.99 mm | -- |
| Closed loop, mean depth error | 1.49 mm | 0.51 mm | -0.98 mm | -- |

No colour drops by more than a point and the closed loop is far from worse -- seed by seed, the
candidate's placement failures on the shipped checkpoint (large lateral error, brick not seated
at the target cell) mostly resolve. But the white-subset pixel gate this fix was written against
does not clear its own bar: +2.5 points on 543 held-out frames, not the required 5. The held-out
white-subset number is the narrower, harder-to-move target; the closed loop is the thing the
scene model is actually for, and it moved a great deal. That gap is reported rather than
resolved by judgement call.

### Decision

**Kept the shipped checkpoint** (`3e07a382`). The white-subset gate is the deciding one by the
letter of the acceptance criteria, and it was not met on the numbers actually measured. The
fine-tune used less than half its 45-minute budget; a full run, or a second helping of targeted
data, is the natural next step before trying this gate again. The data-generation code, the
extra-split/fine-tune infrastructure and this page's numbers are committed regardless, so that
next attempt starts from measured ground rather than from scratch.

## Retraining

```bash
# render ~2 700 training frames and ~550 held-out frames (~8 min)
python scripts/train_vision_model.py --generate --no-train

# inspect labels before training
python scripts/train_vision_model.py --contact-sheet --no-train

# train against a wall-clock budget; resumable
python scripts/train_vision_model.py --minutes 120
```

![Training frames with instance labels](../media/perception_cnn/training_frames.png)

Check two things on that sheet: every object that should be labelled carries an outline, and studs
carry their brick's own colour. Grey studs indicate the brick generator is not painting them.

Rendering is single-threaded; a second helping can be produced in parallel and merged:

```bash
python -c "from brick_builder.perception.cnn.data import generate; \
  print(generate(split='train_b', n_scenes=170, frames_per_scene=16))"
python -c "from brick_builder.perception.cnn.data import merge_split, dataset_size; \
  merge_split('data_cache/vision_model', 'train_b'); \
  print(dataset_size('data_cache/vision_model', 'train'))"
```

`train_b` draws from its own seed range, disjoint from `train` and from `val`. The shipped
checkpoint was trained on 4 546 frames built this way, with 543 held-out frames.

Outputs land in `results/vision_model/` (git-ignored): `checkpoint.pt` for resuming,
`training_log.csv`, `loss_curves.png`, `train_summary.json`. The best held-out checkpoint is
written to `brick_builder/perception/weights/vision_model.pt`, which is the shipped file. Re-running
the same command resumes. Frames are cached as compressed shards under `data_cache/` (also
git-ignored) and are rebuildable from their seeds.

![Training and held-out curves](../media/perception_cnn/loss_curves.png)

### Domain randomization

| quantity | range |
| --- | --- |
| lights | position, brightness, tint, ambient level, direction |
| lens | diagonal field of view 100-130 deg; barrel distortion k1 in [-0.35, -0.10], k2 in [0, 0.08] |
| sensor | exposure, white balance, gamma, shot and read noise, blur |
| brick colour | palette colour with +/-7 % shade jitter |
| table | flat colour, dark grey to pale wood; no texture |
| baseplate pose | +/-20 mm, +/-10 deg |
| arm colour | white, black, grey, orange, blue, yellow, red |

The arm's colour is randomized because the real arm's colour is unknown and every upstream mesh is
the same yellow; without it the network can learn "the arm is the yellow object".

Scene coverage: supply area with loose bricks (tray present and absent), baseplate empty and with
one to six bricks including a second layer, a brick held in the jaws at several heights above the
plate, an approach with the jaws open, same-coloured bricks touching, bricks cut by the window
edge, and the arm's own links in frame. Poses come from sampled tool-down viewing poses and from
`PickPlace` episodes sampled every few control ticks.

### Training recipe

Cross-entropy on the class map with inverse-square-root class weights measured on the training
split (bricks are 8.8 % of pixels, background 55.2 %). Mean-squared error on the centre heat map,
drawn as a Gaussian whose width follows each brick's apparent size. L1 on offsets over brick pixels
only. L1 on depth after per-frame median/MAD normalization, which keeps the target scale-invariant.
AdamW, one cosine decay across the wall-clock budget, gradient norm clipped at 5. Augmentation:
horizontal and vertical flips applied to image and labels together, plus small brightness, contrast
and per-channel gain jitter. Each step runs on a random 160x160 crop of the 256x256 window, which
the fully convolutional network makes equivalent to a whole-window step at 40 % of the cost;
validation always runs on the full window.

Knobs: `--scenes`, `--frames-per-scene`, `--width`, `--batch-size`, `--threads`, `--device cuda`.

### Fine-tuning from the shipped checkpoint with extra data

For a targeted fix (new data aimed at one failure mode, not a from-scratch run):
`--init-weights <path>` starts a fresh `--out` from a shipped snapshot instead of a random one
(ignored once that `--out` already has a `checkpoint.pt` to resume); `--extra-split <name>`
(repeatable) mixes an already-generated split into the training set in memory, on top of
`train`, without ever touching `train` on disk; `--photometric-strength <x>` scales the
photometric jitter's half-widths (1.0 reproduces the fixed ranges above, 2.0 doubles them).

```bash
python scripts/train_vision_model.py --out results/vision_model_experiment \
  --init-weights brick_builder/perception/weights/vision_model.pt \
  --extra-split my_new_split --photometric-strength 1.8 --lr 3e-4 --minutes 45
```

A training run only overwrites the shipped `vision_model.pt` when `--ship-weights` is on; the
default is on for `--out results/vision_model` (an ordinary run) and off for any other `--out`,
so an experimental run cannot leave the shipped checkpoint in a partially-trained, ungated state.
Every run also writes `out/best.pt`, the file to evaluate and, only once its numbers pass the
guide's gate, copy over the shipped one by hand.

### Code reuse

This package holds no copy of the blob statistics. It calls `ideal_scene_model._blob_row` and
`_background_mask` directly, because a second copy would be a second thing to keep in step with
`PLACEHOLDER_SLOT_SCHEMA` and there is already one copy too many
(`conventional_scene_model._blob_row_from_mask`). Suggested refactor: lift the closing block of
`IdealSceneModel.observe` — masks, frame, depth map and a `TemporalSlotAssigner` in, `SceneState`
out — into one public function and have all three producers call it.
`perception/cnn/postprocess.py:slots_from_masks` already has that shape and would shrink to a call.

## Fine-tuning on real frames

Rendered frames differ from photographs in gloss, vignetting and table texture. Procedure for
closing the gap once real frames exist:

1. **Collect.** Drive the arm through the same viewing poses the simulator uses and save the raw
   256x256 windows, the wrist camera pose from forward kinematics, and the joint angles. A few
   hundred frames covering the same situations is sufficient to adapt a model that already knows
   the task.
2. **Label.** Four classes and one instance id per brick. The arm mask can be rendered from the
   joint angles and the baseplate mask projected from a measured plate pose, leaving only the
   bricks to draw by hand. Depth may be left unlabelled; the loss masks invalid pixels, so an
   unlabelled depth channel does not train that head.
3. **Mix.** Roughly one real frame to four rendered ones. Fine-tuning on real frames alone
   discards the variation the rendered set supplies.
4. **Rate.** Start from the shipped checkpoint at about a tenth of the training rate and stop
   after a few thousand steps.
5. **Measure.** Hold out real frames that are never trained on and compare the same table. A
   centre error worse than the simulator figure indicates a situation missing from collection.

No code change is required: `data.py` stores `(rgb, instance, depth)` shards, and a labelled real
frame is the same three arrays.

## Limitations

- Four classes only. The tray and anything else on the table are background.
- No simulator object id, so identity comes from matching slot rows frame to frame, as in the
  colour source. An object hidden for more than `vanish_after` frames may return in a different
  slot.
- The slot contract keeps the seven largest objects; an eighth brick falls into the background
  slot. This is the interface's limit, not the model's.
- Relative depth is ordinal. Metric positions come from the camera geometry and
  `perception/backproject.py`.
