# Closed-loop placement film

<video controls preload="metadata" width="100%"
       poster="../../media/hd/closed-loop-verify.png">
  <source src="../../media/hd/brick-builder-arm-robot-closed-loop-placement-1080p.mp4" type="video/mp4">
</video>

[1920x1080 file](../media/hd/brick-builder-arm-robot-closed-loop-placement-1080p.mp4)

| Step | Frame |
|---|---|
| Align | [closed-loop-align.png](../media/hd/closed-loop-align.png) |
| Press the brick down | [closed-loop-press.png](../media/hd/closed-loop-press.png) |
| Verify | [closed-loop-verify.png](../media/hd/closed-loop-verify.png) |

## Run conditions

| Item | Value |
|---|---|
| Task | Placement of a DUPLO 2x2 brick held in the stock SO-101 gripper; two bricks already on the baseplate |
| Baseplate offset from nominal | 6.1 mm, 2.2 deg (unknown to the controller) |
| Scene model (control input) | Convolutional scene model, checkpoint `3e07a382` |
| Controller | Target-sequence tracking at 10 Hz: feed-forward target action plus feedback on target minus observed scene state |
| Target sequence | Recorded in the digital twin from the goal and nominal parameters |
| Closed-loop steps | Transport, Align (x, y, yaw of the held brick over the cell) |
| Correction after Align | Held: it estimates a static unknown (baseplate pose, brick's offset in the jaws) |
| Open-loop steps | Press the brick down, Release, Retract (run from the converged pose) |
| Video | 1920x1080, 30 fps, H.264, 24.5 s, 734 frames, single take |

## On-screen elements

| Element | Content |
|---|---|
| Arm camera POV | 256x256 wrist-camera window (model input) |
| Observed scene state | Predicted segmentation of the current frame |
| Target scene state | Segmentation of the next frame of the target sequence |
| Ground truth segmentation | Simulator labels; reference for mIoU |
| Scene-state error [mm] | Feature error between target and observed scene state; dashed line: 0.40 mm tolerance; vertical rules: step boundaries |
| Segmentation mIoU [%] | Prediction vs. ground truth, per frame |
| Prediction error [mm] | Observed minus predicted next scene state |
| Press force [N], Insertion depth [mm] | Simulator measurements, display only |

## Results

Filmed run: brick seated on the target cell; lateral error 0.11 mm; depth error 0.01 mm; Align converged to 0.85 mm over 157 vision ticks.

Ten seeds, identical displaced worlds:

| Control input | Seated on target cell | Lateral error |
|---|---|---|
| Ground-truth scene model | 10/10 | median 0.10 mm |
| Convolutional scene model | 8/10 | 0.06-0.11 mm on seated runs |
| Convolutional scene model, target sequence recorded through it | 4/10 | — |
| Camera corrections off | 4/10 | median 0.19 mm, max 16.1 mm |

The camera-off column is not a clean zero, and the reason is worth stating: the stud-and-tube
joint has a capture basin of roughly half a stud pitch, so a 4-8 mm baseplate offset is partly
inside what the press corrects by itself. Six of the ten land on the target cell and four seat
fully; the failures are two bricks pressed into the NEIGHBOURING cell (16.0 and 16.1 mm) and two
that stop half a pitch out and do not seat at all. What the loop adds is not the last
half-millimetre alone -- it is removing the whole-cell mistakes.

(An earlier version of this table read 0/10. That measurement was wrong: with corrections off the
progress index still waited for an error nothing was going to reduce, so the runs stalled part
way instead of playing through. The control condition now replays the recorded actions to the
end, which is what it always should have meant.)

Recording the target sequence through the same convolutional model measures worse, not better:
its per-frame noise enters the target as well as the observation and the median tracking error
rises from 1.4-1.9 mm to 3.2-6.2 mm.

## Scene-state error after Release

Once the brick is let go the correction is held, so camera-frame feature positions along the
retract path no longer track the recording. From Release onward the displayed error is measured
on relative geometry: the placed brick's blob against the baseplate's and the already-placed
bricks', through the same Jacobians. The absolute tracking error is logged separately. A tick
where the placed brick is not matched reports no value rather than a small one.

## Reproduction

```
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=4 \
  python scripts/make_demo_video.py --quality hd1080 --seed 0 --cnn-drives \
    --lead 2.2 --hold 3.0 --stills 6 \
    --stills-dir media_out/hd/closed_loop_stills \
    --out media_out/hd/brick_builder_arm_robot_closed_loop_placement_1080p.mp4
```

The film uses checkpoint `3e07a382`, which is the shipped
`brick_builder/perception/weights/vision_model.pt`. A local copy at
`results/film/vision_model_frozen.pt` (git-ignored) takes precedence when present; without it the
script loads the shipped weights.

# Closed-loop pick-and-place film

<video controls preload="metadata" width="100%"
       poster="../../media/hd/closed-loop-pick-press.png">
  <source src="../../media/hd/brick-builder-arm-robot-closed-loop-pick-and-place-1080p.mp4" type="video/mp4">
</video>

[1920x1080 file](../media/hd/brick-builder-arm-robot-closed-loop-pick-and-place-1080p.mp4)

| Step | Frame |
|---|---|
| Approach | [closed-loop-pick-approach.png](../media/hd/closed-loop-pick-approach.png) |
| Align | [closed-loop-pick-align.png](../media/hd/closed-loop-pick-align.png) |
| Press the brick down | [closed-loop-pick-press.png](../media/hd/closed-loop-pick-press.png) |

## Run conditions

| Item | Value |
|---|---|
| Task | Whole job: fetch a DUPLO 2x2 off the table and place it; two bricks already on the baseplate, one spare brick beside the one to pick |
| Baseplate offset from nominal | 6.1 mm, 2.2 deg (unknown to the controller) |
| Brick offset from the supply row | 11.3 mm, 12.6 deg (unknown to the controller) |
| Scene model (control input) | Simulator ground truth |
| Scene model (panels) | Convolutional scene model, checkpoint `3e07a382` |
| Controller | Target-sequence tracking at 10 Hz: feed-forward target action plus feedback on target minus observed scene state |
| Target sequence | Recorded in the digital twin from the goal and nominal parameters |
| Dwells that estimate | Hover 55 mm above the brick, second dwell 20 mm above it, lift dwell over the bare table, align dwell over the target cell |
| Unknown per dwell | Where the brick lies (both approach dwells), how it sits in the jaws (lift), where the baseplate is (align) |
| Open-loop steps | Final descent onto the brick, Press the brick down, Release, Retract |
| Video | 1920x1080, 30 fps, H.264, 38.4 s, 1152 frames, single take |

On-screen elements are the same as the placement film.

## Results

Filmed run: brick seated on the target cell; lateral error 0.04 mm; depth error 0.25 mm; Align converged to 0.44 mm over 69 vision ticks.

Ten seeds, identical displaced worlds:

| Control input | Seated on target cell | Lateral error |
|---|---|---|
| Ground-truth scene model | 8/10 | 0.02-0.23 mm on seated runs |
| Convolutional scene model | 8/10 | 0.07-0.33 mm on seated runs |
| Camera corrections off | 0/10 | median 16.2 mm |

Both scene models reach the same count on these ten seeds, and they do not fail on the same
ones: ground truth misses seeds 1 and 4, the convolutional model seeds 0 and 2, and on three of
those four the brick is on the target cell but short of full depth rather than on the wrong
cell. The take above is driven by ground truth.

The camera-off column is a clean zero here, unlike the placement task's. The brick's own
8-15 mm offset on the table is added to the baseplate's, and the grasp is scripted from the
supply row rather than from where the brick is, so the errors land outside the stud-and-tube
joint's capture basin instead of partly inside it.

Per-seed, ground truth driving, with the brick's pose in the jaws measured at the lift:

| Seed | Seated | On cell | Lateral [mm] | In the jaws, across / along [mm] |
|---|---|---|---|---|
| 0 | yes | yes | 0.15 | -0.30 / 3.28 |
| 1 | no | yes | 0.19 | -0.16 / 7.71 |
| 2 | yes | yes | 0.17 | -0.26 / 6.26 |
| 3 | yes | yes | 0.23 | -0.21 / -7.87 |
| 4 | no | no | 16.09 | 0.03 / -8.49 |
| 5 | yes | yes | 0.05 | -0.27 / 2.42 |
| 6 | yes | yes | 0.08 | -0.26 / -0.44 |
| 7 | yes | yes | 0.02 | -0.28 / 1.92 |
| 8 | yes | yes | 0.20 | -0.30 / -1.70 |
| 9 | yes | yes | 0.11 | -0.29 / -3.48 |

What the jaws close on is what decides a run. Every seed that misses closes on a brick sitting
more than 8 mm along the jaws; the approach's second dwell is what brought that from 8.3-13.7 mm
down to 0.4-7.9 and the count from 5/10 to 8/10.

## Reproduction

```
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=3 \
  python scripts/make_demo_video.py --quality hd1080 --seed 0 --pick \
    --lead 2.2 --hold 3.0 --stills 6 \
    --stills-dir media_out/hd/closed_loop_pick_stills \
    --out media_out/hd/brick_builder_arm_robot_closed_loop_pick_and_place_1080p.mp4
```

One run of the same loop, headless:

```
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl python scripts/run_placement.py --pick --seed 0
```
