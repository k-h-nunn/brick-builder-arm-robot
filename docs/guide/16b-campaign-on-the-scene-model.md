# Campaign on the scene model

The closed-loop campaign (`docs/guide/16-closed-loop-campaign.md`) drives with simulator ground
truth and scores the analytic and learned inverse against it. This page drives the same loop with
the convolutional scene model instead: the loop reads `SceneState` rows the scene model produced
from the rendered frame, not the simulator's own state. Two runs are reported.

- `closed_loop_cnn`, evaluate-only: how much the ground-truth-trained round-5 pair loses when the
  states it sees at test time come from the scene model instead of the simulator, against the
  analytic baseline measured the same way. Round 0 (analytic) and round 5 (learned).
- `closed_loop_cnn_train`, two rounds asked for, three episodes per task per round: a campaign
  that performs and trains on the scene model's own states rather than the simulator's. It logs a
  predictor checkpoint at round 1 and does not reach a trained inverse; see Results.

Both campaigns reuse `brick_builder/campaign/**` unchanged; only `--source cnn` and the campaign
name differ from `docs/guide/16-closed-loop-campaign.md`.

## Scene-model weights

`brick_builder/perception/weights/vision_model.pt`, sha256
`3e07a382f963145318479f81b61dbe273ffd421ea2e772dd2c0df1bbe40bcafb`. Every measured number on this
page is against this checkpoint. A later checkpoint at the same path needs its own sha256 line
here and its own numbers; a changed weights file changes what the scene model reports, which is
exactly what this page measures.

`brick_builder/action/twin.py` and `brick_builder/action/learn/dataset.py` changed after the
numbers below were measured (see Fixes to the crashes); the sha256 above did not change, and the
tables and the figure are unchanged, measured-before-the-fix numbers. They are not re-run here.

## Evaluate-only against the round-5 pair

`--evaluate-only` scores whatever a campaign has deployed. Pointed at a fresh campaign name it
finds nothing under `results/campaign/<name>/weights/` and, at round 0, falls back to the pair the
repository ships. To score the closed-loop campaign's own round-5 pair under a different name (its
own `results/campaign/closed_loop/weights/` holds a round-5 evaluation still in progress elsewhere
and is not touched here), its two checkpoints and `deployed.json` are copied into
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
analytic baseline and the round-5 pair, the two this page compares; `--processes 1` throughout,
the machine-wide limit while other work is also running. `build_2` was left out of both calls: it
needs its own recorded target sequence per cycle under `--source cnn` (see The pickling
workaround), and the two single-cycle tasks already used most of the time budget this step had.

These commands do not reproduce the numbers below on a fresh checkout by themselves -- see The
pickling workaround.

## Training campaign

`closed_loop_cnn_train` is a full, short `run_campaign.py` call, not `--evaluate-only`: it performs
the tasks with the scene model driving, logs the tuples `TrackingEpisode.observer` records at every
vision tick -- the scene model's own slot rows, not the simulator's -- trains `F` and `G` on them,
redeploys, and scores the held-out seeds, as `docs/guide/16-closed-loop-campaign.md` describes,
with `--source cnn` the only substantive difference.

```bash
PYTHONPATH=$PWD heavy .../python scripts/run_campaign.py \
  --rounds 2 --episodes 3 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --budget-minutes 150 --round-minutes 30 \
  --name closed_loop_cnn_train --root results/campaign
```

Two rounds and three episodes per task, not five and three like the ground-truth campaign: this
size was set from the time this step had and the throughput the evaluate-only step measured. It
did not run out of budget; round 1's train phase raised (see Results).

## Results

### Ground truth against the scene model, seeds 0-9

Same two checkpoint pairs (the fixed analytic inverse, and `closed_loop`'s round-5 learned pair)
on the same ten held-out worlds. Ground truth is `docs/guide/16-closed-loop-campaign.md`'s own
measured numbers for the same pairs, the `fixed` and round-5 rows in its per-task tables; scene
model is `closed_loop_cnn`'s round 0 and round 5. Both columns' reference sequences and driving
states change together with `--source`: `record_for` records the target sequence through the same
scene model that will drive the run, so the scene-model column is not the ground-truth column with
only the driving state swapped (see Mislabeled reference bodies).

| Task | Inverse | Ground truth (`--source ideal`) | Scene model (`--source cnn`) |
|---|---|---|---|
| Placement | analytic | 10/10 (72-100 %) | 4/10 (17-69 %) |
| Placement | learned (r5) | 10/10 (72-100 %) | 2/10 (6-51 %) |
| Pick and place | analytic | 9/10 (60-98 %) | 0/1 (0-79 %, see below) |
| Pick and place | learned (r5) | 9/10 (60-98 %) | 0/1 (0-79 %, see below) |

Placement scores every episode under `--source cnn`: 10/10 to 4/10 for the analytic baseline,
10/10 to 2/10 for the round-5 pair. The round-5 pair is below the analytic baseline under the
scene model (2/10 against 4/10) though not under ground truth (10/10 against 10/10); by the rule
in `docs/guide/16-closed-loop-campaign.md`'s "The loop" (the learned inverse's seated fraction at
least matching the fixed analytic baseline) it does not clear it here.
(`campaign.would_ship`'s own column reads `yes` for both rows regardless -- the same
`seated_on_target_cell` key mismatch `docs/guide/16-closed-loop-campaign.md` already reports; the
comparison above is by hand from the `Seated on target cell` values, not from that column.)

Pick and place raises before nine of the ten held-out episodes finish, for both inverses, on the
same seven seeds at `Approach` and the same two at `Lift`, never seed 7 -- see Mislabeled reference
bodies. `0/1` is the one seed that ran to completion, not nine failures and one success; the
Wilson interval and every other number in that row is one episode's worth.

![Closed-loop success, seeds 0-9: ground truth vs scene model, percent](../media/campaign/campaign_cnn_success.png)

Both panels plot the table above: the fixed analytic inverse at round 0, the round-5 learned pair
at round 5, each pair of bars the ground-truth number beside the scene-model number. Pick and
place's two scene-model bars are the same one-episode measurement described above, marked `n=1`.

### Cost per round

| Campaign | Round | Evaluate [min] | Episodes scored | Episodes crashed |
|---|---|---|---|---|
| `closed_loop_cnn` | 0 (analytic) | 18.8 | 21 / 30 | 9 |
| `closed_loop_cnn` | 5 (r5 learned) | 49.3 | 41 / 50 | 9 |

Total heavy time for this step: 68.1 min inside `run_campaign.py` (both rounds, cumulative, the
figure the campaign itself reports) plus roughly 10 min of one-off recording outside it, seeds 0-9
for both tasks (see The pickling workaround) -- about 78 min against the 60-minute target.

Placement's per-episode wall time spans widely and does not move the same way between rounds:
round 0's ten placement episodes run 27-63 s; round 5's run 17-232 s, with its five slowest at
151, 152, 163, 206 and 232 s. What made the slow round-5 episodes slower than round 0's is not
measured here -- both replanning (a second attempt when the first cycle's brick does not land) and
machine load from other work running at the same time are possible causes, and this page does not
have per-episode replan counts or load samples to tell them apart.

### Mislabeled reference bodies

Every `pick_and_place` crash before the fix in twin.py (see Fixes to the crashes) raised the same
exception at `Approach` or `Lift`:

```
RuntimeError: step 'Approach': the twin only shows 0 of the 0 reference blobs, which is not enough to line up on
```

The exception comes from `brick_builder/action/twin.py`'s `build_step`, called from
`brick_builder/action/place.py`'s `_prepare_step` while building the twin's expectation for that
step -- not from the scene model reporting nothing during the pick, which is a different, working
part of the loop. `TrackingEpisode` always measures the twin's own expectation with
`IdealSceneModel`, ground truth rendered from the twin's nominal scene (`twin.py:107`), regardless
of `--source`.

What changes under `--source cnn` is the recorded reference labels `build_step` is asked to line
up on. `record_ideal` names each step's reference bodies with `reference_bodies()`
(`place.py:317`), which reads a state's `slot_ids` as renderer instance ids -- true of the
simulator's own `SceneState`, not of the convolutional model's: its `slot_ids` are plain slot
indices, with a comment on that line in `postprocess.py:209` saying so. Under `--source cnn` the
recorded sequence carries the scene model's states throughout (`record_for` records the target
sequence through the same source that will drive the run), so `reference_bodies()` reads slot
indices as if they were instance ids and names the wrong body, or none, at essentially every step.

The recorded caches show this directly. A ground-truth `pick_and_place` recording's `Approach`
step names an empty reference set (nothing to line up on is asked of the twin there by design);
the matching `--source cnn` recording for seed 0 names `{'brick to pick up': 'supply_brick_0'}` at
`Approach` -- a real body in the twin's scene, just not the one at that pose, so `calibrate()`
matches nothing and `build_step` raises. `placement` recordings carry the same defect but it
surfaces differently: their per-step body sets are frequently empty outright (`Transport {}`,
`Align {}`), which `_prepare_step` already treats as an untrusted step without raising (a body
dict that is empty, rather than non-empty and wrong, hits a different, pre-existing branch); other
`placement` recordings carry non-empty but arbitrary sets (`Transport {'plate'}`,
`Align {'placed brick 1'}`, `'held'` named as a reference at `Release`/`Retract`, where nothing is
held). Both tasks' recorded reference sets are affected; `placement` does not raise because its
sets are more often empty than wrong.

The pattern that stays fixed across both inverses in the table above (seven seeds crash at
`Approach`, two at `Lift`, seed 7 is the one that completes) is the recorded label, not either
inverse: both rounds replay the same cached `pick_and_place` recordings. Fixing the label mapping
is `record_ideal`/`reference_bodies` in `place.py`, a file this page does not edit; this page
measures and reports it.

### Training round

`closed_loop_cnn_train`, two rounds, three episodes per task per round, `--source cnn`. Round 0
(analytic, no learned pair yet) reproduces `closed_loop_cnn`'s own round 0 on the same seeds --
placement 4/10, pick and place 0/1, the same crash pattern -- which is expected: same weights,
same scene model, same held-out worlds.

Round 1's `perform` phase ran six episodes (three placement, three pick and place) with the
analytic inverse: placement's three completed and none landed the brick (two "seated on the wrong
cell", one "partial seat"), consistent with round 0's 4/10 on a small sample; all three
pick-and-place episodes hit the reference-body crash above (`ideal_seed=900000` for every one of
them, since `--processes 1` collapses the recorded-sequence pool to one recording -- see Limits),
so pick-and-place logged nothing. `train_round` (`brick_builder/action/learn/rounds.py`) then
trained the predictor `F` and wrote `predictor.npz` to disk before training the inverse `G`, which
raised:

```
ValueError: need at least one array to stack
```

in `brick_builder/action/learn/dataset.py`'s `hindsight_pairs`, called from `train_inverse`. Every
tick placement's three episodes logged (10 ticks per attempt, 6 attempts -- two per episode, the
campaign's own replan retry -- 60 rows total) carried `trusted=True` and `frozen=True`, so
`acting = trusted & ~frozen` was empty on all 60 and `hindsight_pairs` had nothing to stack. Why
only 10, already-frozen ticks were logged per attempt rather than a full trajectory is not traced
further here: `place.py` latches `frozen` once `frame.step_index` passes a fixed index and does
not clear it again within an episode, which is documented, working behaviour, not itself the
defect; whether the short, already-frozen log is downstream of the same reference-body labelling
above, of the collector, or of something else in `place.py`/`action/learn/collect.py` (neither
edited here) is unverified.

### Fixes to the crashes

Two changes, made so the two crashes above no longer end a whole episode or a whole campaign.
Each has a unit test that does not need a campaign run (`tests/action/test_twin_blind_step.py`,
`tests/action/learn/test_dataset_empty_gate.py`; `PYTHONPATH=$PWD heavy .../python -m pytest -q
tests/action` also passes, 47 tests, no regressions).

- **`brick_builder/action/twin.py`.** `build_step`'s reference-count check raised a bare
  `RuntimeError` when it saw fewer than `min_refs` blobs before any Jacobian is measured; the check
  just below it, once probing starts, already raises `BlindStep`, and `place.py`'s
  `_prepare_step` already catches `BlindStep` and marks the step untrusted instead of losing the
  episode. The first check now raises `BlindStep` too, and prints the count (blobs seen, blobs
  needed, the step name) to stderr before raising, since nothing downstream of the catch sees the
  message again. A `pick_and_place` episode that hits this at `Approach` or `Lift` now plays that
  step open-loop rather than raising; it is scored by `verify_placement` at the end like any other
  episode. This was not re-run through a campaign (see Scene-model weights); the crash counts in
  the table above are from before this change.
- **`brick_builder/action/learn/dataset.py`.** `hindsight_pairs` and `teacher_pairs` both crashed
  on `np.stack([])` when zero ticks passed the `trusted & ~frozen` gate. Both now return zero-row
  `x`/`y` arrays and log one line saying how many of how many ticks passed the gate, through an
  optional `log` argument (`print` by default). This does not, on its own, make
  `closed_loop_cnn_train`'s round 1 train cleanly: `train_inverse`
  (`brick_builder/action/learn/train.py`, a file this page does not edit) reshapes
  `obs[j] - obs[i]` against `len(i)` two lines after calling `hindsight_pairs`, and numpy refuses
  to reshape a zero-size array into `(0, -1)` -- a different `ValueError`
  (`cannot reshape array of size 0 into shape (0,newaxis)`), at `train.py:194`, confirmed by
  calling `train_inverse` directly on a synthetic zero-acting-tick log. The round still ends the
  campaign on this input; the crash moves one file and 17 lines, and fixing it needs a change in
  `train.py`, which this page does not edit.

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
way, and nothing before this page had run the closed loop with `--source cnn` end to end.

`decode` is documented as "viewer/evaluation only... not read by anything on the control path"
(`scene_model.py`), so dropping `_decode_fn` before the cache is written changes nothing the loop
reads. Fixing `record_for` or `SceneState` itself needs changes in `brick_builder/campaign/**` and
`brick_builder/perception/**`, neither of which this page edits; the numbers on this page instead
come from a one-off script, run separately and never imported by the repository, that
monkey-patches `SceneState.__getstate__` to drop the field and
pre-populates the ideal-sequence cache before the unmodified `run_campaign.py` / `campaign.runner`
pipeline reads it back with a plain `pickle.load`. The cache files it wrote are ordinary pickles
with `_decode_fn: None`; nothing downstream needed the patch to read them.

The commands in Evaluate-only against the round-5 pair and in Training campaign do not reproduce
the numbers on this page by themselves on a fresh checkout: they need this pre-caching step first,
and the script that does it is scratch, uncommitted, and not part of this repository. Fixing the
pickling defect at its source, in `record_for`/`SceneState`, would let a fresh `--source cnn` run
populate the cache itself and remove the need for it.

## Limits

- `pick_and_place` under `--source cnn` raised on 9 of 10 held-out seeds before the twin.py fix
  above (see Mislabeled reference bodies); its numbers in the table are one episode, not ten.
  `placement` does not raise and is measured on the full ten. Neither number reflects a re-run
  after the fix (see Scene-model weights).
- Ten seeds per task per round, same as `docs/guide/16-closed-loop-campaign.md`; the Wilson
  interval on ten trials is wide and its limits apply here unchanged.
- `build_2` was not measured under `--source cnn`: its recorded target sequence needs the same
  workaround per cycle, and two single-cycle tasks across two campaigns already used most of the
  step's time budget.
- `--processes 1` throughout (the machine-wide limit while other work is running), so the
  performing rounds' recorded-sequence pool (`ideal_pool_seed`) is one recording, not the two a
  `--processes 2` run would keep; every performing episode of a task in `closed_loop_cnn_train`
  drove against the same recorded reference, varying only the world's own seed noise.
- `closed_loop_cnn_train` did not complete a round: `perform` ran (three placement, three pick and
  place), `train` wrote a predictor checkpoint and then raised training the inverse (see Training
  round and Fixes to the crashes), so there is no round-1 pair, no round 2, and nothing to compare
  against `docs/guide/16-closed-loop-campaign.md`'s own five rounds of ground-truth training.
- The dataset.py fix (see Fixes to the crashes) does not by itself get `closed_loop_cnn_train`
  past round 1: `train_inverse` (`brick_builder/action/learn/train.py`) still raises on the same
  zero-tick input, two lines further into training, in a file this page does not edit.
