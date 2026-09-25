# Campaign on the scene model

The closed-loop campaign (`docs/guide/16-closed-loop-campaign.md`) drives with simulator ground
truth and scores the analytic and learned inverse against it. This page drives the same loop with
the convolutional scene model instead: the loop reads `SceneState` rows the scene model
produced from the rendered frame, not the simulator's own state. Two questions follow from that
substitution:

- How much does the ground-truth-trained r5 pair lose when the states it sees at test time come
  from the scene model instead of the simulator -- `closed_loop_cnn`, evaluate-only, round 0 (analytic
  baseline) and the round-5 learned pair.
- What a short campaign that performs **and** trains on the scene model's own states looks like --
  `closed_loop_cnn_train`, two rounds asked for, three episodes per task per round. It does not
  reach a trained pair; see Results for why.

Both campaigns reuse `brick_builder/campaign/**` unchanged; only `--source cnn` and the campaign
name differ from `docs/guide/16-closed-loop-campaign.md`.

## Scene-model weights

`brick_builder/perception/weights/vision_model.pt`, sha256
`3e07a382f963145318479f81b61dbe273ffd421ea2e772dd2c0df1bbe40bcafb`. Recorded here
because another package may ship new weights later, and every number on this page is a number
about *this* checkpoint.

## Evaluate-only against the r5 pair

`--evaluate-only` scores whatever a campaign has deployed. Pointed at a fresh campaign name it
finds nothing under `results/campaign/<name>/weights/` and, at round 0, falls back to the pair the
repository ships. To score the closed-loop campaign's own round-5 pair under a different name (its
own `results/campaign/closed_loop/weights/` is another package's evaluation in progress and is not
touched here), its two checkpoints and `deployed.json` are copied into
`results/campaign/closed_loop_cnn/weights/` before the round-5 call -- that directory is exactly
what `Campaign.deployed_paths()` reads, so nothing in `campaign.py` changes.

```bash
PYTHONPATH=$PWD heavy .../python scripts/run_campaign.py \
  --evaluate-only --round 0 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --name closed_loop_cnn --root results/campaign

cp results/campaign/closed_loop/weights/{inverse.npz,predictor.npz,deployed.json} \
   results/campaign/closed_loop_cnn/weights/

PYTHONPATH=$PWD heavy .../python scripts/run_campaign.py \
  --evaluate-only --round 5 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --name closed_loop_cnn --root results/campaign
```

`--no-baseline-learned` drops the repository-shipped pair's own column so the budget goes to the
analytic baseline and the r5 pair, the two this page is about; `--processes 1` throughout, the
machine-wide limit while other packages are also running. `build_2` was left out of both calls: it
needs its own recorded target sequence per cycle under `--source cnn` (see the workaround below),
and the two single-cycle tasks already used most of the time budget this step had.

## A round that trains on what the scene model saw

`closed_loop_cnn_train` is a full, short `run_campaign.py` call, not `--evaluate-only`: it performs
the tasks with the scene model driving, logs the tuples `TrackingEpisode.observer` records at every
vision tick -- which are the scene model's own slot rows, not the simulator's -- trains `F` and `G` on
them, redeploys, and scores the held-out seeds, exactly as `docs/guide/16-closed-loop-campaign.md`
describes, with `--source cnn` the only substantive difference.

```bash
PYTHONPATH=$PWD heavy .../python scripts/run_campaign.py \
  --rounds 2 --episodes 3 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --budget-minutes 150 --round-minutes 30 \
  --name closed_loop_cnn_train --root results/campaign
```

Two rounds and three episodes per task, not five and three like the ground-truth campaign: this
size was chosen from the time this step had and the throughput the evaluate-only step measured. It
did not run out of budget -- see Results for what stopped it instead.

## Results

### Ground truth against the scene model, seeds 0-9

Same two checkpoint pairs (the fixed analytic inverse, and `closed_loop`'s round-5 learned pair)
on the same ten held-out worlds; the only thing that changes between the two right-hand columns
and the ground-truth column is which scene model the loop reads from. `docs/guide/16-closed-loop-campaign.md`
is the ground-truth column's source; the other two are `closed_loop_cnn`'s round 0 and round 5.

| Task | Inverse | Ground truth (`--source ideal`) | Scene model (`--source cnn`) |
|---|---|---|---|
| Placement | analytic | 10/10 (72-100 %) | 4/10 (17-69 %) |
| Placement | learned (r5) | 10/10 (72-100 %) | 2/10 (6-51 %) |
| Pick and place | analytic | 9/10 (60-98 %) | 0/1 (0-79 %, see below) |
| Pick and place | learned (r5) | 9/10 (60-98 %) | 0/1 (0-79 %, see below) |

Placement holds together enough to score -- every episode finishes -- and drops hard: the fixed
analytic baseline goes from 10/10 to 4/10, and the r5 pair, ground-truth-trained and unconditionally
deployed at round 5, goes from 10/10 to 2/10, below the analytic baseline under the scene model
even though it is not below it under ground truth. By the rule in
`docs/guide/16-closed-loop-campaign.md`'s "The loop" (the learned inverse's seated fraction at
least matching the fixed analytic baseline) the r5 pair does not clear it here: 2/10 against 4/10.
(`campaign.would_ship`'s own column reads `yes` for both rows regardless -- the same
`seated_on_target_cell` key mismatch `docs/guide/16-closed-loop-campaign.md` already reports for
`rounds.seated_fraction`; the comparison above is by hand from the `Seated on target cell` values,
not from that column.)

Pick and place does not hold together: nine of the ten held-out episodes raise before the episode
finishes, for both inverses, always the same seven seeds at `Approach` and the same two at `Lift`,
never seed 7 -- see below. `0/1` is the one seed that ran to completion, not nine failures and one
success; the Wilson interval and every other number in that row is one episode's worth, not ten.

![Closed-loop success against training round, percent](../media/campaign/campaign_cnn_success.png)

The analytic baseline (dashed) and the r5 learned pair (round 5, with its Wilson bar) against the
ground-truth-driven numbers `docs/guide/16-closed-loop-campaign.md` publishes (dotted); round 0 is
not drawn for the learned series because `closed_loop_cnn` measured the r5 pair only at round 5,
never at round 0.

### Cost per round

| Campaign | Round | Evaluate [min] | Episodes scored | Episodes crashed |
|---|---|---|---|---|
| `closed_loop_cnn` | 0 (analytic) | 18.8 | 21 / 30 | 9 |
| `closed_loop_cnn` | 5 (r5 learned) | 49.3 | 41 / 50 | 9 |

Total heavy time for this step: 68.1 min inside `run_campaign.py` (both rounds, cumulative, the
figure the campaign itself reports) plus roughly 10 min of one-off recording outside it, seeds 0-9
for both tasks (see the workaround below) -- about 78 min against the 60-minute target. The
overrun is the crashed episodes' own cost (each one still renders, runs the network and fails
partway through `Approach` or `Lift` before raising) plus replanning on the ones that did finish:
several placement episodes took 150-220 s, roughly three to four times the 30-60 s a clean
single-pass recording cost during pre-caching.

### The scene model loses track during the pick

Every `pick_and_place` crash is the same exception, raised before the episode's own scoring runs,
so the campaign's `runner.py` catches it, counts it, and drops the job -- it never reaches
`evaluate.csv`:

```
RuntimeError: step 'Approach': the twin only shows 0 of the 0 reference blobs, which is not enough to line up on
RuntimeError: step 'Lift': the twin only shows 0 of the 1 reference blobs, which is not enough to line up on
```

(`brick_builder/action/twin.py:214`, called from `brick_builder/action/place.py`'s
`_prepare_step` -> `_build`.) Both round 0 (analytic) and round 5 (r5 learned) hit it on the exact
same seven seeds at `Approach` and the same two at `Lift`, and both leave the exact same seed (7)
as the one that completes -- the pattern tracks the seed, not which inverse is driving, which
points at the scene model's own read of that seed's frame during the pick rather than at either
inverse. `placement` never starts held, so it never asks the scene model to find the brick on the
table before a grasp, and it does not crash. Diagnosing why the scene model reports zero
detections there is perception work, outside `brick_builder/campaign/**` and
`brick_builder/perception/**`, neither of which this package owns; this page only measures and
reports it.

### The scene model's own training round

`closed_loop_cnn_train`, two rounds, three episodes per task per round, `--source cnn`. Round 0
(analytic, no learned pair yet) reproduces `closed_loop_cnn`'s own round 0 on the same seeds --
placement 4/10, pick and place 0/1, the same seven-and-two crash pattern -- which is expected,
since it is the same weights, the same scene model and the same held-out worlds.

Round 1 does not finish. `perform` ran six episodes (three placement, three pick and place) with
the analytic inverse: placement's three completed (all three failed the task -- two "seated on
the wrong cell", one "partial seat" -- 0/3, consistent with round 0's 4/10 on a small sample); all
three pick-and-place episodes hit the same `twin.build_step` crash the evaluate-only step measured
(`ideal_seed=900000` for every one of them, since `--processes 1` collapses the pool to one
recording -- see Limits), so pick-and-place logged nothing. `train` then raised before writing any
checkpoint:

```
ValueError: need at least one array to stack
```

in `brick_builder/action/learn/dataset.py`'s `hindsight_pairs`, called from `train_inverse`. Every
tick placement's three episodes logged -- 10 ticks per attempt, 6 attempts (two per episode, the
campaign's own replan retry), 60 rows total -- carries `trusted=True` and `frozen=True`, so
`acting = trusted & ~frozen` is empty on all 60 and `hindsight_pairs` has nothing to stack.
`brick_builder/action/place.py` latches `self.frozen = True` once `frame.step_index` passes
`self._freeze_index` and never clears it; the ticks the collector kept here are all past that
point in every one of the six attempts. That the ground-truth campaign's own round 1 logs 3448
ticks from 9 episodes and trains in 8 s (`docs/guide/16-closed-loop-campaign.md`'s cost table)
while three whole `--source cnn` episodes here contribute zero usable ticks is the finding: not a
sample-size problem, six-for-six.

`campaign.json` records `r1.perform` as done and `r1.train` as not; `--resume` would repeat the
same six episodes and the same crash, since `perform_seeds` is deterministic in the round index.
Fixing this is `brick_builder/action/place.py` and `brick_builder/action/learn/dataset.py`, neither
owned by this package -- this page stops at reporting it. Zero of the two rounds asked for
completed; there is no round-1 or round-2 pair to evaluate, deploy, or draw a training curve from.

## The pickling workaround

The first evaluate-only attempt (one seed, one task, before any of the numbers on this page) failed
every episode:

```
AttributeError: Can't pickle local object 'slots_from_masks.<locals>.decode_fn'
```

`record_for` (`brick_builder/campaign/tasks.py`) records the nominal target sequence through the
*same* scene model that will drive the run (see its docstring), so under `--source cnn` the cached
`IdealSequence` carries `SceneState` objects from the scene model, and `SceneState._decode_fn`
(`brick_builder/perception/scene_model.py`) is a closure `perception/cnn/postprocess.slots_from_masks`
builds locally -- not picklable, and `record_for` unconditionally writes its cache to
`results/campaign/<name>/ideals/*.pkl`. Under `--source ideal` this path is not exercised the same
way and nothing before this page had run the closed loop with `--source cnn` end to end.

`decode` is documented as "viewer/evaluation only... not read by anything on the control path"
(`scene_model.py`), so dropping `_decode_fn` before the cache is written changes nothing the loop
reads. Fixing `record_for` or `SceneState` itself is outside what this package owns
(`brick_builder/campaign/**` and `brick_builder/perception/**` are other packages' files); the
numbers on this page instead come from a one-off script, run outside the package and never
imported by it, that monkey-patches `SceneState.__getstate__` to drop the field and pre-populates
the ideal-sequence cache before the unmodified `run_campaign.py` / `campaign.runner` pipeline reads
it back with a plain `pickle.load`. The cache files it wrote are ordinary pickles with
`_decode_fn: None`; nothing downstream needed the patch to read them. This should be fixed at the
source so a fresh `--source cnn` run does not need it.

## Limits

- `pick_and_place` under `--source cnn` crashes on 9 of 10 held-out seeds (`RuntimeError`, "the
  twin only shows 0 of the N reference blobs" -- see above); its numbers on this page are one
  episode, not ten. `placement` does not crash and is measured on the full ten.
- Ten seeds per task per round, same as `docs/guide/16-closed-loop-campaign.md`; the Wilson
  interval on ten trials is wide and its limits apply here unchanged.
- `build_2` was not measured under `--source cnn`: its recorded target sequence needs the same
  workaround per cycle, and two single-cycle tasks across two campaigns already used most of the
  step's time budget.
- `--processes 1` throughout (the machine-wide limit while other packages are running), so the
  performing rounds' recorded-sequence pool (`ideal_pool_seed`) is one recording, not the two a
  `--processes 2` run would keep; every performing episode of a task in `closed_loop_cnn_train`
  drove against the same recorded reference, varying only the world's own seed noise.
- `closed_loop_cnn_train` did not complete a round: `perform` ran (three placement, three pick and
  place), `train` raised before writing a checkpoint (see Results), so there is no round-1 pair,
  no round 2, and nothing to compare against `docs/guide/16-closed-loop-campaign.md`'s own five
  rounds of ground-truth training.
