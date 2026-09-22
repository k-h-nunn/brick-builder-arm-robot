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

which becomes two placements, `2x2 (4,3,0) blue` and `2x2 (6,3,0) blue`, executed as consecutive
pick-and-place cycles. After each cycle `Goal.remaining_steps` is recomputed against what is
really on the plate; a brick that did not land on its cell leaves its own step outstanding and
that cycle is run once more. Both bricks are on the bottom layer and both are blue 2x2, because
the seat height the geometry places on is the plate's own top face and the supply row offers a
blue 2x2. The compiler has no pronouns, so the second brick names the first by colour.

The second cycle's world carries the first brick on the plate, and its target sequence is
recorded against a nominal plate that carries it too (`record_ideal(..., placed=...)`). Without
that, the brick the run has just placed matches nothing in the recording and the loop lines
itself up on it.

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
  python scripts/run_campaign.py --rounds 5 --episodes 3 --budget-minutes 200
```

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
- `build_2`'s two bricks go side by side on the bottom layer. A stacked build needs a seat height
  above the plate's top face, which `action.place.Geometry` has no way to express.
- Each of the build's cycles runs in its own world, built with the bricks the earlier cycles
  landed. The plan carries over, the physics does not: a brick that seated slightly off its cell
  is rebuilt on the cell for the next cycle.
- The campaign's own deployed pair is written under `results/`, which is git-ignored. Nothing here
  changes what `PlaceConfig(inverse="learned")` loads by default.

## Next steps

- More performing episodes per round. The log is the thing the curve is a function of, and a
  round of three episodes per task is a few thousand ticks.
- A round that trains on the convolutional scene model's own states rather than the simulator's.
- The predictor in the loop as a signal rather than a report: `PlaceConfig.predictor` selects
  which one-step prediction the error is measured against, and nothing reads it.
