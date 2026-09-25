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
ship** column: that is what `action.learn.rounds.deploy_if_better` would have decided -- the
learned inverse at least matching the analytic baseline on every task, not winning on one and
losing on another.

## The task set

| Task | Cycles | Starts | Unknown to the controller | Scored |
|---|---|---|---|---|
| `placement` | 1 | brick already held | baseplate 4-8 mm, 2-3 deg | one brick |
| `pick_and_place` | 1 | brick loose on the table | the above, and the brick 8-15 mm, 10-20 deg | one brick |
| `build_2` | 2 | as above, twice | the above, per cycle | both bricks |

`build_2` is one English command, compiled by `brick_builder.language`:

```
put a blue 2x2 at column 5 row 4 then put a blue 2x2 next to the blue brick
```

which becomes two placements, `2x2 (4,3,0) blue` and `2x2 (2,3,0) blue`, executed as consecutive
pick-and-place cycles. After each cycle `Goal.remaining_steps` is recomputed against what is
really on the plate; a brick that did not land on its cell leaves its own step outstanding and
that cycle is run once more. Both bricks are on the bottom layer and both are blue 2x2, because
the seat height the geometry places on is the plate's own top face and the supply row offers a
blue 2x2. The compiler has no pronouns, so the second brick names the first by colour.

Left rather than right, and the press decided it rather than the loop. The right-hand neighbour,
column 7 row 4, is 218 mm from the arm's base, past the 100-200 mm band the gripper's downward
force was characterised over. Measured there on five seeds, the second brick reached its cell to
within 0.02-0.29 mm on four of them and full insertion depth on one; the shortfalls were
0.43-4.50 mm. The left-hand neighbour is at 154 mm.

The second cycle's world carries the first brick on the plate; so does the nominal plate its
target sequence is recorded against (`record_ideal(..., placed=...)`) and so do the twins its
Jacobians are measured in (`TrackingEpisode(..., placed=...)`). All three have to agree: a
recording made against a bare plate matches nothing where the run has just put a brick, and a
twin that lacks a brick the recording lines itself up on drops that reference silently.

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

<!-- RESULTS -->

## Limits

- The success rates are ten seeds per task per round. A Wilson interval on ten trials is wide:
  8/10 and 10/10 overlap, and a one-seed change moves the point estimate by ten points.
- The evaluation seeds are fixed, so the same ten worlds are scored every round. That is what
  makes the rounds comparable and it is also the whole of the sample.
- `build_2` was evaluated in the first and the last round only (`--sparse-tasks build_2`), so its
  curve has two points where the other two tasks have one per round. It was performed in every
  round, so the training set grew with it throughout.
- `build_2`'s two bricks go side by side on the bottom layer. A stacked build needs a seat height
  above the plate's top face, which `action.place.Geometry` has no way to express.
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
- The predictor in the loop as a signal rather than a report: `PlaceConfig.predictor` selects
  which one-step prediction the error is measured against, and nothing reads it.
