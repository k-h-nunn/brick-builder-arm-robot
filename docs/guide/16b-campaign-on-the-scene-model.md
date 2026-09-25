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
here and its own numbers.

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
.venv/bin/python scripts/run_campaign.py \
  --evaluate-only --round 0 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --name closed_loop_cnn --root results/campaign

cp results/campaign/closed_loop/weights/{inverse.npz,predictor.npz,deployed.json} \
   results/campaign/closed_loop_cnn/weights/

.venv/bin/python scripts/run_campaign.py \
  --evaluate-only --round 5 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --name closed_loop_cnn --root results/campaign
```

`--no-baseline-learned` drops the repository-shipped pair's own column so the budget goes to the
analytic baseline and the round-5 pair, the two this page compares; `--processes 1` throughout,
the machine-wide limit while other work is also running. `build_2` was left out of both calls: it
needs its own recorded target sequence per cycle under `--source cnn` (see The pickling fix), and
the two single-cycle tasks already used most of the time budget this step had.

These commands did not reproduce the numbers below on a fresh checkout by themselves when they
were first measured -- see The pickling fix, and Re-evaluation after the fixes for the numbers
measured after it landed.

## Training campaign

`closed_loop_cnn_train` is a full, short `run_campaign.py` call, not `--evaluate-only`: it performs
the tasks with the scene model driving, logs the tuples `TrackingEpisode.observer` records at every
vision tick -- the scene model's own slot rows, not the simulator's -- trains `F` and `G` on them,
redeploys, and scores the held-out seeds, as `docs/guide/16-closed-loop-campaign.md` describes,
with `--source cnn` the only substantive difference.

```bash
.venv/bin/python scripts/run_campaign.py \
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
| `closed_loop_cnn` | 5 (learned, plus the analytic recheck) | 49.3 | 41 / 50 | 9 |

Total heavy time for this step: 68.1 min inside `run_campaign.py` (both rounds, cumulative, the
figure the campaign itself reports) plus roughly 10 min of one-off recording outside it, seeds 0-9
for both tasks (see The pickling workaround) -- about 78 min against the 60-minute target.

Placement's per-episode wall time spans widely and does not move the same way between rounds, or
even between two rows of the same round. Round 0 scores placement twenty times (ten seeds against
each of two `predictor` columns, the analytic inverse throughout): 27-63 s. Round 5 scores it forty
times (ten seeds times two inverses times two `predictor` columns), 17-232 s, but the two halves
do not behave alike: the twenty rows using the round-5 learned inverse run 17-56 s, inside round
0's own span; every row above 150 s uses the analytic inverse that `recheck_analytic`
(`campaign.py:402`) reruns at the final round for comparison -- its five slowest are 151.6, 162.8,
205.6, 220.8 and 231.8 s. The 49.3 min in the table above includes this recheck; it is not the cost
of the learned pair.

`evaluate.csv`'s `replanned` and `vision_ticks` columns separate replanning from machine load for
these analytic-inverse rows. The same seed carries the same `vision_ticks` and the same `replanned`
flag in round 0 and round 5 (seed 0: 163 ticks, replanned, in both), and within round 5 itself the
two `predictor` columns for the same seed and inverse also carry identical `vision_ticks`/`replanned`
values yet differ in wall time by up to 4x -- seed 0: 54.0 s against 205.6 s; seed 9: 71.6 s against
220.8 s, the slower column swapping sides between the two seeds. The recorded work is identical and
the wall time is not, so machine load from the other work running on the machine at the same time,
not replanning, accounts for the spread.

### Mislabeled reference bodies

Every `pick_and_place` crash before the fix in twin.py (see Fixes to the crashes) raised
`RuntimeError` from the same check in `build_step`, at `Approach` or `Lift`, with a different
reference count each time -- Approach:

```
RuntimeError: step 'Approach': the twin only shows 0 of the 0 reference blobs, which is not enough to line up on
```

and Lift:

```
RuntimeError: step 'Lift': the twin only shows 0 of the 1 reference blobs, which is not enough to line up on
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

The recorded caches show this only partially. A ground-truth `pick_and_place` recording's
`Approach` step names `{'brick to pick up': 'supply_brick_0'}` in 520 of 620 frames across the ten
seeds -- every frame after the first ten of each 62-frame step. The matching `--source cnn`
recordings name the same body in 353 of 620 Approach frames and an empty set in the other 267,
never any other body: `place.py`'s pick-and-place filter keeps only entries that already equal
`SUPPLY_BODY` at `Approach`/`Grasp` (`place.py:595-603`), so whatever else the slot-id/instance-id
mismatch mislabels there is dropped rather than shown. Why the label is present in some frames and
absent in others under `--source cnn` is not traced further here.

`placement` recordings carry the same slot-id/instance-id mismatch, but under `--source cnn` most
of their per-step body sets are not empty: `Align` is empty in none of 400 cnn frames, `Transport`
in 14 of 80, `Press the brick down` in 144 of 728, `Verify` in 8 of 150 -- 166 of 1597 frames
overall, about 10 %. Some non-empty sets are arbitrary (`Transport {'plate'}`,
`Align {'placed brick 1'}`), a direct instance of the mismatch reading a real body's name off a
slot index that only coincides with that body's instance id by chance. Why `placement`'s
`build_step` calls do not raise despite these wrong, non-empty sets is not analyzed further here.

`held` named as a reference at `Release`, `Retract` and `Verify` in ground-truth `placement`
recordings, in all ten seeds, is not a mislabeling: the released brick keeps the MuJoCo body name
`held` (`compose.py:917`), and `reference_bodies()` only excludes it while `_in_hand(step)`
(`place.py:594`) -- true through `Grasp`/`Lift`/`Transport`/`Align`/`Press the brick down`, false
once the brick is let go, so it names the released brick's own body.

The pattern that stays fixed across both inverses in the table above (seven seeds crash at
`Approach`, two at `Lift`, seed 7 is the one that completes) is the recorded label, not either
inverse: both rounds replay the same cached `pick_and_place` recordings. Fixing the label mapping
needs a change in `record_ideal`/`reference_bodies` (`place.py`); this page reports the effect, not
the fix.

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
Each has unit tests that do not need a campaign run (`tests/action/test_twin_blind_step.py`,
`tests/action/learn/test_dataset_empty_gate.py`; `.venv/bin/python -m pytest -q
tests/action` also passes, 48 tests, no regressions).

- **`brick_builder/action/twin.py`.** `build_step`'s reference-count check raised a bare
  `RuntimeError` when it saw fewer than `min_refs` blobs before any Jacobian is measured; the check
  just below it, once probing starts, already raises `BlindStep`, and `place.py`'s
  `_prepare_step` already catches `BlindStep` and marks the step untrusted instead of losing the
  episode. The first check now raises `BlindStep` too, and prints the count (blobs seen, blobs
  needed, the step name) to stderr before raising, since nothing downstream of the catch sees the
  message again. Reading `_prepare_step`, a `pick_and_place` episode that hits this at `Approach`
  or `Lift` should play that step open-loop and reach `verify_placement` at the end like any other
  episode -- confirmed by Re-evaluation after the fixes: all ten `pick_and_place` seeds complete
  under `--source cnn` with this fix in place, where nine of ten raised before it. Neither
  `place.py` nor `campaign/tasks.py`'s `EpisodeRecord` (both outside this page's files) give the
  blind step its own failure class, so a blind step's effect on the reported numbers is
  indistinguishable from any other cause of a low seated fraction in `evaluate.csv`. The crash
  counts in the Results table above are from before this change; Re-evaluation after the fixes has
  the numbers measured with it applied.
- **`brick_builder/action/learn/dataset.py`.** `hindsight_pairs` and `teacher_pairs` both crashed
  on `np.stack([])` when zero ticks passed the `trusted & ~frozen` gate. Both now return zero-row
  `x`/`y` arrays and log one line saying how many of how many ticks passed the gate, through an
  optional `log` argument (`print` by default). This does not, on its own, make
  `closed_loop_cnn_train`'s round 1 train cleanly: `train_inverse`
  (`brick_builder/action/learn/train.py`) reshapes `obs[j] - obs[i]` against `len(i)` at
  `train.py:194`, 18 lines after calling `hindsight_pairs` at `train.py:176`, and numpy refuses to
  reshape a zero-size array into `(0, -1)` -- a different `ValueError`
  (`cannot reshape array of size 0 into shape (0,newaxis)`), confirmed by calling `train_inverse`
  directly on a synthetic zero-acting-tick log. The round still ends the campaign on this input;
  fixing it needs a change in `train.py`.

## The pickling fix

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
(`scene_model.py`), so dropping `_decode_fn` before pickling changes nothing the loop reads.
`SceneState` now does this itself: `__getstate__` copies `__dict__` and replaces `_decode_fn` with
`None`, `__setstate__` restores the rest unchanged. A round-tripped state answers every
control-path query exactly as before; it just has no decode function any more (`has_decode` is
`False`, `decode` raises the same "no decode function" `RuntimeError` it already raises for any
producer that never built one) -- see
`test_cnn_source_state_round_trips_through_pickle` in `tests/perception/test_scene_model.py`.
`record_for`'s plain `pickle.dump`/`pickle.load` needed no change: a fresh `--source cnn` run now
populates its own ideal-sequence cache directly.

The numbers on this page below "Re-evaluation after the fixes" were measured before this fix
landed, using a one-off script (scratch, uncommitted, never imported by the repository) that
monkey-patched `SceneState.__getstate__` the same way, outside the process that ran
`run_campaign.py`. Nothing downstream needed the patch to read the cache files it wrote --
they are ordinary pickles with `_decode_fn: None`, exactly what the fix above now writes directly.
The commands in Evaluate-only against the round-5 pair and in Training campaign reproduce those
numbers on a fresh checkout by themselves now that the fix is in `SceneState`; they did not before.

## Re-evaluation after the fixes

`closed_loop_cnn_v2`: the same two commands as Evaluate-only against the round-5 pair (`--source
cnn --evaluate-only --processes 1`, seeds 0-9, placement and pick and place, round 0 against the
fixed analytic baseline and round 5 against `closed_loop`'s own round-5 pair), renamed and re-run
with the twin.py and dataset.py fixes (see Fixes to the crashes) and the pickling fix (see The
pickling fix) in place -- the first `--source cnn` campaign run this project has not needed a
workaround for. Same checkpoints throughout: `vision_model.pt` is still `3e07a382` (see
Scene-model weights and `docs/guide/06b-conventional-vision-model.md`'s Decision, which records
the candidate checkpoint was not shipped), and the round-5 `inverse.npz`/`predictor.npz` pair
copied from `results/campaign/closed_loop/weights/` is byte-identical to the one
`closed_loop_cnn` used -- sha256 `d7be3b0eeafdb925482fd81049d644b921c0c289ef8e378bed47df2e195274fa`
(`inverse.npz`) and `b2c735fe598cefe59c5f908094b41446e1ef9b441cfe3d46693d93ee209728a0`
(`predictor.npz`) in both `closed_loop_cnn/weights/` and `closed_loop_cnn_v2/weights/`. The
difference below is the code fixes, not a different model.

| Task | Round | Inverse | `closed_loop_cnn` (before) | `closed_loop_cnn_v2` (after) |
|---|---|---|---|---|
| Placement | 0 | analytic | 4/10 (17-69 %) | 4/10 (17-69 %) |
| Placement | 5 | learned (r5) | 2/10 (6-51 %) | 2/10 (6-51 %) |
| Pick and place | 0 | analytic | 0/1 (0-79 %) | 5/10 (24-76 %) |
| Pick and place | 5 | learned (r5) | 0/1 (0-79 %) | 2/10 (6-51 %) |

Placement is unaffected either way -- it never hit the reference-body crash -- and reproducing its
exact seated fraction and Wilson interval on a second, independent run is itself a check that the
fixes changed nothing on a path that was already working. Pick and place is the one the crash was
hiding: all ten seeds now complete instead of nine raising, and the completed baseline is a real
ten-episode measurement (5/10) rather than the one seed that happened not to crash (0/1, itself a
failure). The round-5 learned pair reaches 2/10, below the ten-seed analytic baseline (5/10); by
`docs/guide/16-closed-loop-campaign.md`'s "The loop" rule (the learned inverse's seated fraction
at least matching the fixed analytic baseline) pick and place does not clear it here either, the
same conclusion placement already reported, now on a full sample for both tasks. Failure classes,
`closed_loop_cnn_v2`, seeds 0-9: pick and place round 0 (5 failures) -- seated on the wrong
cell=4, partial seat=1; round 5 (8 failures) -- seated on the wrong cell=7, partial seat=1.

Commands (identical to Evaluate-only against the round-5 pair, `--name closed_loop_cnn_v2`):

```bash
.venv/bin/python scripts/run_campaign.py \
  --evaluate-only --round 0 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --name closed_loop_cnn_v2 --root results/campaign

cp results/campaign/closed_loop/weights/{inverse.npz,predictor.npz,deployed.json} \
   results/campaign/closed_loop_cnn_v2/weights/

.venv/bin/python scripts/run_campaign.py \
  --evaluate-only --round 5 --seeds 0-9 --tasks placement pick_and_place \
  --source cnn --processes 1 --no-baseline-learned \
  --name closed_loop_cnn_v2 --root results/campaign
```

## Limits

- `pick_and_place` under `--source cnn` raised on 9 of 10 held-out seeds before the twin.py fix
  above (see Mislabeled reference bodies); the `closed_loop_cnn` numbers in the Results table
  above are one episode, not ten. `placement` did not raise and was measured on the full ten
  throughout. Re-evaluation after the fixes reports the same measurement at n=10 for both tasks,
  after the fix.
- Ten seeds per task per round, same as `docs/guide/16-closed-loop-campaign.md`; the Wilson
  interval on ten trials is wide and its limits apply here unchanged.
- `build_2` was not measured under `--source cnn`: it needs its own recorded target sequence per
  cycle, and two single-cycle tasks across two campaigns already used most of the step's time
  budget.
- `--processes 1` throughout (the machine-wide limit while other work is running), so the
  performing rounds' recorded-sequence pool (`ideal_pool_seed`) is one recording, not the two a
  `--processes 2` run would keep; every performing episode of a task in `closed_loop_cnn_train`
  drove against the same recorded reference, varying only the world's own seed noise.
- `closed_loop_cnn_train` did not complete a round: `perform` ran (three placement, three pick and
  place), `train` wrote a predictor checkpoint and then raised training the inverse (see Training
  round and Fixes to the crashes), so there is no round-1 pair, no round 2, and nothing to compare
  against `docs/guide/16-closed-loop-campaign.md`'s own five rounds of ground-truth training.
- The dataset.py fix (see Fixes to the crashes) does not by itself get `closed_loop_cnn_train`
  past round 1: `train_inverse` (`brick_builder/action/learn/train.py:194`) still raises on the
  same zero-tick input, 18 lines further into training than the `hindsight_pairs` call.
- The twin.py fix's effect on a `pick_and_place` episode is now observed, not only inferred from
  reading `_prepare_step` (see Fixes to the crashes and Re-evaluation after the fixes): all ten
  seeds complete under `--source cnn`. Neither `place.py` nor `campaign/tasks.py` give a blind
  step its own failure class in `evaluate.csv`, so which completed episodes hit a blind step is
  still not visible in the per-episode record.
