# Closed-loop campaign

The arm performs brick tasks in simulation while the two models on the control path are trained
from what it did, and the closed-loop result is measured after every training round. Nothing is
scripted between rounds: the same loop that places the brick produces the tuples the next round
fits, and the same held-out seeds score every version.

## The loop

```
        +-------------------------------------------------------------+
        |                                                             |
        v                                                             |
   PERFORM ------------> TRAIN ------------> REDEPLOY ------------> EVALUATE
   N episodes per task   F: next scene       version id,           seeds 0-9,
   on fresh seeds,       state | mech, a     campaign weights      every task,
   current models        G: DAgger on the    directory             learned and
   driving; every        analytic inverse                          analytic inverse
   tick logged           at the logged
                         states
```

- **Perform.** `N` episodes of each task on seeds drawn from a stream that never meets the
  evaluation set. The learned inverse drives once the campaign has deployed one; the analytic
  inverse drives before that. Every vision tick is logged through `TrackingEpisode.observer`, in
  the layout `action.learn.dataset` reads.
- **Train.** `action.learn.rounds.train_round` fits the next-scene-state predictor `F` on the
  change of the slot rows given the mechanical state and the action, then the one-step inverse
  `G` on three terms: the hindsight target, the distal teacher through that `F`, and regression
  on the analytic inverse's answer at the states the round itself visited. Driving with the
  learned inverse and labelling with the analytic one is what makes a round a DAgger round.
- **Redeploy.** The round's two checkpoints become the pair the campaign's own runs load, under a
  version id. The campaign writes them to `results/campaign/<name>/weights/` and never to the
  pair the repository ships.
- **Evaluate.** Every task on seeds 0-9, with the learned inverse and with the analytic one, and
  with both prediction rules on `placement` so the two are compared on identical trajectories.

The analytic inverse does not change between rounds, so it is measured once, in round 0, and
carried into every table as the fixed baseline; the last round re-measures it on `placement` as a
check that it really did not move.

The campaign deploys every round to its own directory unconditionally. A campaign that deployed
only the rounds that already won would never collect the states a losing round visits, and those
are the states the next round is relabelled on. Whether a round earned the checkpoints the
repository *ships* is a separate question, and every round's table answers it in the **Would
ship** column: the same rule `action.learn.rounds.deploy_if_better` uses -- the learned inverse
at least matching the analytic baseline on every task, not winning on one and losing on another --
applied to whichever tasks the round measured.

## The task set

| Task | Cycles | Starts | Unknown to the controller | Scored |
|---|---|---|---|---|
| `placement` | 1 | brick already held | baseplate 4-8 mm, 2-3 deg | one brick |
| `pick_and_place` | 1 | brick loose on the table | the above, and the brick 8-15 mm, 10-20 deg | one brick |
| `build_2` | 2 | as above, twice | the above, per cycle | both bricks |
| `stack_2` | 2 | as above, twice; the second brick on the first | the above, per cycle | both bricks, cell and layer |

`build_2` is one English command, compiled by `brick_builder.language`:

```
put a blue 2x2 at column 5 row 4 then put a blue 2x2 left of the blue brick
```

which becomes two placements, `2x2 (4,3,0) blue` and `2x2 (2,3,0) blue`, executed as consecutive
pick-and-place cycles. After each cycle `Goal.remaining_steps` is recomputed against what is
really on the plate; a brick that did not land on its cell leaves its own step outstanding and
that cycle is run once more. Both bricks are blue 2x2, because the supply row offers a blue 2x2.
The compiler has no pronouns, so the second brick names the first by colour.

The second brick goes to the left-hand neighbour (column 3, row 4; 154 mm from the base). The
right-hand neighbour (column 7, row 4; 218 mm) lies outside the 100-200 mm band over which the
gripper's downward force was characterised: on five seeds there, lateral placement landed within
0.02-0.29 mm of the cell on four of them (not separately measured on the fifth), and the press
reached full insertion depth on one of the five; on the other four it stopped 0.43-4.50 mm short.

The second cycle's world carries the first brick on the plate; so does the nominal plate its
target sequence is recorded against (`record_ideal(..., placed=...)`) and so do the twins its
Jacobians are measured in (`TrackingEpisode(..., placed=...)`). All three have to agree: a
recording made against a bare plate matches nothing where the run has just put a brick, and a
twin that lacks a brick the recording lines itself up on drops that reference silently.

`build_2`'s second cell, (2, 3), has two flush neighbours: the first brick at +x and the yellow 2x4
at +y. `action.place.Geometry` aligns that cycle from a hover raised from 20.0 to 25.7 mm above the
seat, clear of the neighbours' stud tops, with the aim offset 0.3 mm away from them
(`PlaceConfig.neighbour_clear`, `PlaceConfig.neighbour_bias`;
[Closed-loop control, Flush neighbour](12-closed-loop-control.md#flush-neighbour)). Recorded target
sequences in `ideals/` are keyed by task, cycle, seed, source, cell, layer and align hover (e.g.
`build_2_c1_s3_ideal_x2y3z0_h25.7`), so the published campaign's files (`build_2_c1_s0_ideal.pkl`)
are not reused; they are recorded again under the new name. The `build_2` rows under Results were
measured without the raised hover and the aim offset. With them, on seeds 0-9 (ground-truth scene
model, analytic inverse, re-plan off), the second brick seats on 1/10 (seed 8; Wilson 95 % 2-40 %);
the fixed analytic baseline in round 0 (re-plan on, without them) seated it on 1/10 (seed 1).

`stack_2` is the stacked build:

```
put a blue 2x2 at column 5 row 4 then put a blue 2x2 on top of the blue brick
```

which becomes `2x2 (4,3,0) blue` and `2x2 (4,3,1) blue`. The second cycle seats one brick height
up, on the first brick's studs
([Closed-loop control, Stacked placements](12-closed-loop-control.md#stacked-placements)).
`stack_2` is last in `TASKS`, so the older tasks keep their performing-seed offsets, and it is not
in `scripts/run_campaign.py`'s default `--tasks`.

### Scoring

`control.pick_place.verify_placement` reads the simulator after the arm has left: every stud
within 0.3 mm of full insertion depth is `seated`, and a brick within 0.4 of a stud pitch
(6.4 mm) of the cell centre is `at_target_cell`. An episode succeeds when **every** brick the
task asked for is seated on the cell it was sent to, so a build_2 episode with one brick down is
a failure even though one brick landed; the per-brick count is reported beside it.

Each episode's record carries the task, the seed, the models' version ids, per brick the seated
flag and the lateral and depth errors, the vision ticks per step, the failure class, the median
tracking and prediction errors, the damage state the monitor reached, and the wall clock. Success
rates carry Wilson 95 % intervals (`eval.wilson`).

Every brick is also scored on its target layer, layer 0 included: `at_target_layer` holds when the
brick's bottom face is within half a brick height (9.6 mm) of the layer's height. A brick sent to
the plate that ends on top of another brick fails it, and so does a brick sent to layer 1 that ends
on the plate; a brick seated on another layer has the failure class `seated on the wrong layer`.
The episode CSV carries the check as `brick{i}_at_layer`; a file written before that column existed
reads back as on the layer.

## Commands

Start a campaign:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=3 \
  python scripts/run_campaign.py --rounds 5 --episodes 3 --seeds 0-9 \
    --sparse-tasks build_2 --train-minutes 3 --budget-minutes 265
```

`--sparse-tasks build_2` evaluates the build in the first and last round only, and performs it in
every one. Ten held-out builds are twenty closed-loop cycles and about half of what a round of
evaluation costs; this is the knob the budget turns before it gives up a round.

Continue one that stopped, for any reason, from the round it reached:

```bash
python scripts/run_campaign.py --rounds 5 --episodes 3 --budget-minutes 200 --resume
```

Score one round's deployed models without performing or training, and with the convolutional
scene model driving:

```bash
python scripts/run_campaign.py --evaluate-only --round 0 --source cnn --name closed_loop_cnn
```

Redraw the figures and the tables from what is already measured:

```bash
python scripts/run_campaign.py --curves-only
python -m brick_builder.campaign.curves --campaign results/campaign/closed_loop
python -m brick_builder.campaign.tables --campaign results/campaign/closed_loop
```

Two episode processes at a time with three threads each, no GPU, and a wall-clock budget checked
before every round, so a campaign that runs out of time stops between rounds with its state
consistent. `results/campaign/<name>/campaign.json` is that state; deleting a phase's entry from
it re-runs that phase.

## Results

Five rounds, ten held-out seeds per task per round, `--sparse-tasks build_2`. Tables from
`python -m brick_builder.campaign.tables --campaign results/campaign/closed_loop`, figures from
`python -m brick_builder.campaign.curves --campaign results/campaign/closed_loop`. `published` is
the ground-truth-driving number for the same task from
[Closed-loop control](12-closed-loop-control.md); `recheck` is the round-5 re-measurement of the
analytic baseline described in [The loop](#the-loop).

### Placement, seeds 0-9

| Round | Inverse | Version | Seated on target cell | Wilson 95 % | Bricks | Lateral error [mm] | Median tracking [mm] | Median prediction [mm] | Ticks at Align | Tuples trained on | Would ship |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | analytic | -- | 10/10 | 72-100 % | 10/10 | 0.07 | 1.28 | 0.24 | 73 | -- | -- |
| published | ground truth | -- | 10/10 | -- | -- | -- | -- | -- | -- | -- | -- |
| recheck (round 5) | analytic | -- | 10/10 | 72-100 % | 10/10 | 0.07 | 1.28 | 0.24 | 73 | -- | -- |
| 0 | learned | shipped | 10/10 | 72-100 % | 10/10 | 0.07 | 1.21 | 0.17 | 77 | -- | meets |
| 1 | learned | r1 | 5/10 | 24-76 % | 5/10 | 0.10 | 1.39 | 0.47 | 93 | 3448 | no |
| 2 | learned | r2 | 9/10 | 60-98 % | 9/10 | 0.07 | 1.43 | 0.28 | 101 | 7845 | no |
| 3 | learned | r3 | 6/10 | 31-83 % | 6/10 | 0.11 | 1.41 | 0.19 | 108 | 11869 | no |
| 4 | learned | r4 | 9/10 | 60-98 % | 9/10 | 0.07 | 1.24 | 0.23 | 88 | 16334 | no |
| 5 | learned | r5 | 10/10 | 72-100 % | 10/10 | 0.07 | 1.25 | 0.24 | 85 | 20444 | yes |

### Pick and place, seeds 0-9

| Round | Inverse | Version | Seated on target cell | Wilson 95 % | Bricks | Lateral error [mm] | Median tracking [mm] | Median prediction [mm] | Ticks at Align | Tuples trained on | Would ship |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | analytic | -- | 9/10 | 60-98 % | 9/10 | 0.10 | 4.12 | 0.70 | 69 | -- | -- |
| published | ground truth | -- | 8/10 | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | learned | shipped | 7/10 | 40-89 % | 7/10 | 0.07 | 3.48 | 0.50 | 84 | -- | no |
| 1 | learned | r1 | 3/10 | 11-60 % | 3/10 | 0.05 | 4.20 | 0.46 | 106 | 3448 | no |
| 2 | learned | r2 | 5/10 | 24-76 % | 5/10 | 0.10 | 4.11 | 0.27 | 86 | 7845 | no |
| 3 | learned | r3 | 7/10 | 40-89 % | 7/10 | 0.13 | 4.27 | 0.46 | 83 | 11869 | no |
| 4 | learned | r4 | 7/10 | 40-89 % | 7/10 | 0.11 | 4.41 | 0.48 | 74 | 16334 | no |
| 5 | learned | r5 | 9/10 | 60-98 % | 9/10 | 0.07 | 3.78 | 0.38 | 70 | 20444 | yes |

### Build 2, seeds 0-9

| Round | Inverse | Version | Seated on target cell | Wilson 95 % | Bricks | Lateral error [mm] | Median tracking [mm] | Median prediction [mm] | Ticks at Align | Tuples trained on | Would ship |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | analytic | -- | 1/10 | 2-40 % | 10/20 | 0.10 | 4.19 | 0.73 | 138 | -- | -- |
| 0 | learned | shipped | 1/10 | 2-40 % | 9/20 | 0.07 | 3.67 | 0.55 | 185 | -- | meets |
| 5 | learned | r5 | 2/10 | 6-51 % | 12/20 | 0.07 | 4.17 | 0.45 | 162 | 20444 | yes |

`Would ship` applies the same rule `action.learn.rounds.deploy_if_better` uses --
`yes` when the round's learned inverse matched the fixed analytic baseline on every task it was
measured on, `meets` when it matched on this task but not on all of them, `no` when it did not
match here -- to whichever tasks the round measured. `deploy_if_better` itself is not called by
the campaign, and its own default task pair does not cover `build_2`, which this column does. The
campaign deploys every round to its own weights directory whatever this says.

### Cost per round

| Round | Perform [min] | Train [min] | Evaluate [min] | Episodes logged | Tuples |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 61.5 | 0 | 0 |
| 1 | 11.1 | 0.1 | 14.4 | 17 | 3448 |
| 2 | 9.8 | 0.2 | 12.2 | 35 | 7845 |
| 3 | 7.8 | 0.3 | 9.9 | 52 | 11869 |
| 4 | 8.5 | 0.4 | 10.2 | 69 | 16334 |
| 5 | 7.5 | 0.5 | 38.9 | 85 | 20444 |

### Failures

| Round | Task | Seated on target cell | Failures |
|---|---|---|---|
| 0 | build 2 | 1/10 | partial seat=6;seated on the wrong cell=2;stalled with a tracked error=1 |
| 0 | pick and place | 7/10 | partial seat=1;seated on the wrong cell=1;stalled with a tracked error=1 |
| 0 | placement | 10/10 | none |
| 1 | pick and place | 3/10 | partial seat=3;seated on the wrong cell=3;stalled with a tracked error=1 |
| 1 | placement | 5/10 | stalled with a tracked error=5 |
| 2 | pick and place | 5/10 | partial seat=2;seated on the wrong cell=2;stalled with a tracked error=1 |
| 2 | placement | 9/10 | stalled with a tracked error=1 |
| 3 | pick and place | 7/10 | partial seat=2;stalled with a tracked error=1 |
| 3 | placement | 6/10 | stalled with a tracked error=4 |
| 4 | pick and place | 7/10 | seated on the wrong cell=3 |
| 4 | placement | 9/10 | partial seat=1 |
| 5 | build 2 | 2/10 | partial seat=5;seated on the wrong cell=2;stalled with a tracked error=1 |
| 5 | pick and place | 9/10 | stalled with a tracked error=1 |
| 5 | placement | 10/10 | none |

![Closed-loop success against training round [%]](../media/campaign/campaign_success.png)

![One-step prediction error against training round [mm]](../media/campaign/campaign_prediction_error.png)

![Action error against the analytic inverse [mm/tick, deg/tick]](../media/campaign/campaign_action_error.png)

![Training data by round [ticks, episodes]](../media/campaign/campaign_data.png)

Both tasks dip in round 1 and recover by round 5. Pick-and-place goes 7/10 (round 0) to 3/10
(round 1, the campaign's low point, Wilson 11-60 %), then 5/10, 7/10, 7/10, and 9/10 at round 5,
matching the fixed analytic baseline (9/10, 60-98 %). Placement goes 10/10 to 5/10 (round 1,
24-76 %), then 9/10, 6/10, 9/10, and back to 10/10 at round 5. `build_2`, scored only at round 0
and round 5, goes from 1/10 (2-40 %) to 2/10 (6-51 %) seated, bricks on target 9/20 to 12/20.

Two-sided Fisher exact tests against the fixed analytic baseline (10/10 on `placement`, 9/10 on
`pick_and_place`) find round 1 the only round that differs from it at the uncorrected 5 % level on
either task: `placement` 5/10 against 10/10, p = 0.033; `pick_and_place` 3/10 against 9/10,
p = 0.020. No other round reaches p < 0.05 on either task; `placement` round 3 (6/10 against
10/10) comes closest, at p = 0.087. These are twelve round-versus-baseline tests (rounds 0-5 on
both tasks), none corrected for running more than one: at a Bonferroni threshold of
0.05 / 12 = 0.0042, neither round-1 result clears it, so nothing on this page is significant once
that correction is applied. Overlapping Wilson intervals at n = 10 do not by themselves rule out a
significant difference and are not evidence of one.

Median one-step prediction error does not move the same way on the two tasks. On `placement` it
is lowest at round 0 (0.17 mm, F shipped) and round 3 (0.19 mm), and highest at round 1
(0.47 mm). On `pick_and_place` it is highest at round 0 (0.50 mm) and lowest at round 2
(0.27 mm); round 1 (0.46 mm) sits close to rounds 3 and 4 (0.46 mm, 0.48 mm), not at the peak.
Neither task's prediction error falls monotonically with more data.

The learned inverse's disagreement with the analytic teacher (`campaign_action_error.png`) is
measured on each round's own held-out ticks: a 15 % split by episode that round's own training run
did not use. That sample is smaller and different at every round -- 449 ticks at round 1, up to
2642 at round 5, about 13 % of that round's 20444-tick log -- so the analytic answer's own size on
it also varies (0.54 to 0.89 mm/tick in x across rounds 1-4) and the raw RMS millimetre numbers
are not directly comparable round to round. Scaled by that round's own baseline, the learned
inverse's error on `y` falls in every round, 31 % to 16 % of the baseline. On `x` and `spin` the
lowest round is round 4, not round 5: `spin` falls from 47.0 % (round 1) to 20.4 % (round 4) and
rises to 23.2 % at round 5; `x` falls to 21.0 % at round 4 (after a small round 2-to-3 rise,
24.2 % to 24.7 %) and rises again to 21.2 % at round 5 -- the same last-round direction as `spin`,
not a smooth curve on either axis.

Whether a round would ship, by the rule in [The loop](#the-loop) -- the learned inverse's seated
fraction at least matching the fixed analytic baseline on every task the round measured -- only
round 5 does. The `Would ship` column above applies the same rule
`action.learn.rounds.deploy_if_better` uses; `deploy_if_better` itself is not called here, and its
own default task pair does not cover `build_2`, which this column does. Round 0 (the shipped
pair) fails only on `pick_and_place` (7/10 against 9/10); `placement`
(10/10) and `build_2` (1/10 against 1/10) both meet. Rounds 1-4 fail on both `placement` and
`pick_and_place` in every round measured: `placement` 5/10, 9/10, 6/10, 9/10 and `pick_and_place`
3/10, 5/10, 7/10, 7/10, all below the baseline's 10/10 and 9/10. Round 5 is the only round that
meets the bar on every task it measured, `build_2` included (2/10 against 1/10), and is what the
campaign would ship.

## Limits

- The success rates are ten seeds per task per round. A Wilson interval on ten trials is wide:
  8/10 and 10/10 overlap, and a one-seed change moves the point estimate by ten points.
- The evaluation seeds are fixed, so the same ten worlds are scored every round. That is what
  makes the rounds comparable and it is also the whole of the sample.
- `build_2` was evaluated in the first and the last round only (`--sparse-tasks build_2`), so its
  curve has two points where the other two tasks have one per round. It was performed in every
  round, so the training set grew with it throughout.
- `build_2`'s two bricks go side by side on the bottom layer. The stacked build is `stack_2`, which
  the campaign's rounds did not perform or evaluate. Measured on its own on seeds 0-9 with the
  ground-truth scene model, the analytic inverse and re-plan off, each cycle a pick-and-place in
  its own world: the second brick seats on 10/10 (Wilson 95 % 72-100 %) and both bricks on 9/10
  (60-98 %)
  ([Closed-loop control, Two-brick builds](12-closed-loop-control.md#two-brick-builds-seeds-0-9)).
  `stack_2` with the convolutional scene model as the control input (`--source cnn`) was not
  evaluated. Headless runs of the two-brick recording's world with that model driving (cycle 1 a
  placement from the jaws, cycle 2 a pick-and-place;
  `scripts/make_campaign_video.py --task stack_2 --seed S --cnn-drives [--targets truth] --probe`,
  one run per seed) built both bricks on none of the seeds tried: seeds 0 and 1 with ground-truth
  target sequences, seeds 0, 1, 2, 3 and 5 with target sequences recorded through the model
  ([Closed-loop control, Two-brick recording](12-closed-loop-control.md#two-brick-recording)).
- The build's two cells are flush neighbours, and the second brick is pressed against a wall that
  is already there. That is what the build measures beyond two pick-and-places, and it is where
  it fails: see the table.
- Each of the build's cycles runs in its own world, built with the bricks the earlier cycles were
  sent to place, on the cells they were sent to. The plan carries over, the physics does not: a
  brick that missed is rebuilt on its cell for the next cycle, so the second cycle's result is a
  measurement of the second cycle rather than of the first one's luck.
- The campaign's own deployed pair is written under `results/`, which is git-ignored. Nothing here
  changes what `PlaceConfig(inverse="learned")` loads by default.

## Next steps

- More performing episodes per round. The log is the thing the curve is a function of, and a
  round of three episodes per task is a few thousand ticks.
- A round that trains on the convolutional scene model's own states rather than the simulator's.
  [Campaign on the scene model](16b-campaign-on-the-scene-model.md) has a two-round attempt; it
  logs a predictor checkpoint and does not reach a trained inverse.
- The predictor in the loop as a signal rather than a report: `PlaceConfig.predictor` selects
  which one-step prediction the error is measured against, and nothing reads it.
