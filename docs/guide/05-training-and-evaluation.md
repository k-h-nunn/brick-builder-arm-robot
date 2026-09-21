# Training and evaluation

**Status: wired end to end.** The RL core (`brick_builder/rl/nets.py`, `brick_builder/rl/ppo_asym.py`,
`brick_builder/rl/sac_asym.py`) trains against the three real `brick_builder/envs/` tasks now; `scripts/train.py`
and `scripts/eval.py` are the one command each way the project's definition of done calls for (see
[Command-line training](#command-line-training) and [Command-line evaluation](#command-line-evaluation)
below). Both RL cores' own acceptance tests still use a throwaway Dict-wrapped `Pendulum-v1` (see
[Reproducing the example run](#reproducing-the-example-run) and
[SAC: the off-policy fallback recipe](#sac-the-off-policy-fallback-recipe)) — those tests predate any
brick_builder environment and are about the two implementations themselves, not about a real skill.

## Purpose

This is where a policy actually gets trained (reinforcement learning) and where a trained policy gets
scored against the project's acceptance criteria (evaluation).

## Capabilities

- Train the primary policy — a small residual-RL network layered on top of scripted motion, trained with
  PPO using an asymmetric critic — with one command, and watch its progress in TensorBoard.
- Switch to an off-policy fallback recipe (SAC with an asymmetric *twin* critic, LayerNorm, automatic
  entropy tuning and 3-step returns — not a distributional critic; see
  [SAC: the off-policy fallback recipe](#sac-the-off-policy-fallback-recipe)) with one config field
  (`algo: sac`) if the primary recipe is not learning fast enough, without changing the environment, the
  actor network, or how the resulting checkpoint is evaluated or exported.
- Run the standard evaluation suite against a trained policy: held-out random seeds, the full range of
  domain randomization, hundreds of episodes per condition, and a report (JSON and a readable summary)
  with success rates and confidence intervals, sweeps over misalignment and other difficulty knobs, and
  an automatically labeled taxonomy of *why* each failed episode failed (wrong cell, partial seat, jam,
  structural damage, slip, overload trip, timeout, or a kinematic problem).
- Compare three tiers of policy quality on the same task: a privileged "teacher" (sees everything,
  including information the real robot cannot access), the deployable "student," and a hand-scripted
  baseline — the ordering `teacher >= student >= scripted` is itself one of the acceptance checks.

## The learning algorithm

PPO with an asymmetric critic (Pinto 2018), implemented from scratch in `brick_builder/rl/ppo_asym.py` in
one CleanRL-style file — no Stable-Baselines3, because its Dict-observation asymmetric policies are
awkward and export needs to stay simple. Any environment used with it must expose

```python
observation_space = gymnasium.spaces.Dict({"actor": Box(...), "critic": Box(...)})
```

The actor (`brick_builder.rl.nets.Actor`) only ever sees `"actor"`: a plain MLP, tanh hidden activations,
64x64 by default, with a *state-independent* log-std (one learned vector, not a function of the
observation). It stays this small and this plain because it will eventually be hand-mirrored into
about 80 lines of C for the microcontroller (design report 02, section 6) — see
[the export page](07-export-and-firmware.md). The critic (`brick_builder.rl.nets.Critic`, 256x256 by
default) only ever sees `"critic"`: privileged simulator state a real robot cannot measure (contact
forces, per-stud insertion depth, the true in-hand offset, and so on for the press skill). It exists
purely to reduce the variance of the actor's gradient and is discarded at export time, along with its
own normalizer.

Both halves of the observation are normalized on the way in by `brick_builder.rl.nets.RunningNorm`, a
running mean/variance tracker (Welford's algorithm) that keeps updating during training and gets
`.frozen = True` set on it for evaluation and export, so a deployed actor always sees the same
normalization it was trained under.

### Configuration — `PPOConfig`

`brick_builder.rl.ppo_asym.PPOConfig` is a keyword-only dataclass. The fields and their starting values
(design report 02, section 3 — tuned for the press skill, not for the toy environment used in the
tests below):

| Field | Default | Meaning |
|---|---|---|
| `n_envs` | 16 | parallel environments |
| `n_steps` | 512 | steps collected per environment per update |
| `epochs` | 5 | passes over each batch of collected data |
| `minibatches` | 8 | minibatches per epoch (minibatch size = `n_envs*n_steps/minibatches`) |
| `lr` | 3e-4 | Adam learning rate |
| `gamma`, `lam` | 0.99, 0.95 | discount and GAE trace decay |
| `clip` | 0.2 | PPO clipping range |
| `target_kl` | 0.02 | epochs for the current update stop early once mean approx-KL exceeds 1.5x this |
| `ent_coef`, `vf_coef` | 0.0, 0.5 | entropy bonus and value-loss weight |
| `max_grad_norm` | 0.5 | global gradient-norm clip |
| `init_log_std` | -1.0 | initial action noise (std ~= 0.37) |
| `total_steps` | *(required)* | training stops once this many env-steps have been collected |
| `seed` | 0 | seeds `torch.manual_seed` (network init, action sampling) and the numpy `Generator` used for minibatch shuffling, and is passed to every sub-environment's constructor and `reset(seed=...)` |
| `hidden_actor`, `hidden_critic` | (64,64), (256,256) | network widths |
| `async_envs` | `True` | `True` -> `gymnasium.vector.AsyncVectorEnv` (forkserver context); `False` -> `SyncVectorEnv`, used by the test suite so pytest is not spawning subprocesses |
| `vector_context` | `"forkserver"` | multiprocessing start method for `AsyncVectorEnv` |
| `norm_clip` | 10.0 | clamp applied by `RunningNorm.normalize` |
| `log_every` | 1 | updates between TensorBoard scalar writes |

`n_envs`, `lr`, `gamma` and the rest are ordinary keyword arguments — nothing stops a specific skill's
training config from overriding any of them; the table above is a starting point, not a constant.

### Training entry point

The whole training loop is one function call — `scripts/train.py` (below) is a thin YAML-driven wrapper
around exactly this:

```python
from pathlib import Path
from brick_builder.rl.ppo_asym import PPOConfig, train

def make_env(seed: int):
    ...  # build and return one Dict-observation gymnasium.Env, using `seed` however it likes

cfg = PPOConfig(total_steps=10_000_000, seed=0)
checkpoint_path = train(make_env, cfg, run_dir=Path("runs/press-v0"))
```

`train` builds the vector env, runs PPO to `total_steps`, and returns the path to the one checkpoint it
writes at the end, `<run_dir>/checkpoint.pt`. Set `CUDA_VISIBLE_DEVICES=""` before importing torch in
whatever process calls this — training only ever runs on CPU, deliberately, since the GPU may be busy
with someone else's job.

## Command-line training

```bash
.venv/bin/python scripts/train.py --config configs/train/reach.yaml
.venv/bin/python scripts/train.py --config configs/train/press.yaml
.venv/bin/python scripts/train.py --config configs/train/align.yaml
```

One config per task, under `configs/train/`. A config is a plain mapping:

```yaml
env: BrickBuilder/Reach-v0        # any id brick_builder.envs registers with gymnasium
env_kwargs: {}                 # passed straight to gym.make(env, **env_kwargs)
run_name: reach                 # runs/<run_name>/ is where everything lands
ppo:                             # every field of brick_builder.rl.ppo_asym.PPOConfig
  n_envs: 6
  n_steps: 256
  total_steps: 307200
  seed: 0
  # ...
```

`scripts/train.py --config <path>` reads this, builds `make_env(seed) = lambda: gym.make(env,
**env_kwargs)` and calls `train()` exactly as above. `--total-steps`, `--seed` and `--n-envs` override
the config's own `ppo` section without editing the file (useful for a quick smoke run); `--sync-envs`
forces `SyncVectorEnv` instead of the config's `async_envs` — this is what the test suite uses so pytest
is not left spawning subprocess workers. `--run-dir` overrides the default `runs/<run_name>/`.

Every run writes, in `runs/<run_name>/`:

- `config.yaml` — an exact copy of the config this run actually used,
- `ppo_config.json` — the resolved `PPOConfig` after any command-line overrides,
- `tb/`, `checkpoint.pt`, `returns.json` — `ppo_asym.train`'s own outputs (see above),
- `learning_curve.png` — `plot_learning_curve` applied to `returns.json`, written only
  once at least one episode has finished and there is a return to draw.

`configs/train/reach.yaml` reproduces `runs/reach_smoke/` exactly (see
[Reach: a trained skill](#reach-a-trained-skill) below). `configs/train/press.yaml` and
`configs/train/align.yaml` are real training configurations (millions of steps — hours on this
16-thread, 6-worker-process machine), not smoke tests; what was actually measured with a much smaller
budget, for each, is in the sections below. **The project's own hard rule — never start more than 6
worker processes on this shared machine — is why every shipped config's `ppo.n_envs` is 6, not the
`PPOConfig` default of 16.**

### Logged outputs

Two things land in `run_dir` on every run:

- `tb/` — a TensorBoard event file with `train/pg_loss`, `train/v_loss`, `train/entropy`,
  `train/approx_kl` and `train/mean_return` (average return of episodes that finished during that
  update), one point per PPO update. `tensorboard --logdir runs/press-v0/tb` to view it.
- `returns.json` — the same `(global_step, mean_return)` pairs as a plain JSON list, so a learning-curve
  plot does not require parsing TensorBoard's event format. `brick_builder.rl.ppo_asym.plot_learning_curve`
  turns this list straight into a PNG (matplotlib, `Agg` backend, no GUI window) — this is the
  figure-producing function `scripts/make_media.py` calls for an RL run, and it produced
  `docs/media/rl/pendulum_learning_curve.png` below.

`train` also accepts an optional `callback(update_index, metrics)`, called after every update with the
same numbers that get logged; this is what the test suite uses to check that two runs with the same
seed produce identical losses, without parsing the event file.

### Checkpoint format

`checkpoint.pt` is one `torch.save`d dict:

```python
{
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
    "norm_actor": norm_actor.state_dict(),
    "norm_critic": norm_critic.state_dict(),
    "cfg": dataclasses.asdict(cfg),
    "obs_spec": {"actor_dim": int, "critic_dim": int, "act_dim": int, "act_low": [...], "act_high": [...]},
}
```

`brick_builder.rl.ppo_asym.load_actor(checkpoint_path)` rebuilds just the actor and its frozen normalizer
from this (the critic is never needed after training, so nothing downstream reconstructs it). Loading
the same checkpoint twice and running the same observation through both copies reproduces the action
bit-for-bit — this is one of the four acceptance tests in `tests/rl/test_ppo_asym.py`.

### Evaluation procedure

`brick_builder.rl.ppo_asym.evaluate(checkpoint_path, make_env, episodes, seed)` loads the actor, sets its
normalizer to `frozen = True`, and rolls out `episodes` full episodes taking the *mean* action (no
sampling noise) at every step, one environment at a time, seeded `seed, seed+1, seed+2, ...`. It
returns a `list[brick_builder.eval.results.EpisodeResult]` — the same record type the rest of `brick_builder`
uses, so this plugs directly into `brick_builder.eval.report` and the Wilson-interval machinery once a real
task environment exists. An environment that has nothing richer to report than a raw return (like the
Pendulum example here) just gets the `EpisodeResult` defaults (`success=False` unless `info["success"]`
is set, `seat_depth_error=None`, and so on); an environment that does report seat depth, peak load,
structure-intact and a `FailureClass` in `info` gets those carried straight through.

## Command-line evaluation

```bash
.venv/bin/python scripts/eval.py --checkpoint runs/reach/checkpoint.pt --env BrickBuilder/Reach-v0
```

Rolls out `--episodes` held-out episodes (default 100, seeded `--seed, --seed+1, ...`, default
`--seed 10000` so held-out seeds never overlap a training run's own `env.reset` seeds), writes
`<out-dir>/report.json` and `report.md` (`brick_builder.eval.report`: Wilson intervals, the failure
taxonomy — `--out-dir` defaults to `<checkpoint's directory>/eval`) and records `--videos` clips (the
first few episodes, plus one more per distinct failure class actually seen, so a checkpoint's evaluation
always includes a look at *how* it fails, not only how often).

- `--env-kwargs '{"level": 3}'` — any JSON object, merged into `gym.make(env, **env_kwargs)`.
- `--producer surrogate|blob` — shorthand for `BrickBuilder/AlignFromTokens-v0`'s own `env_kwargs`
  key: `--env BrickBuilder/AlignFromTokens-v0 --producer blob`.
- `--actor torch|c` — `c` builds the exported C actor fresh from the checkpoint with the system `gcc`
  (`brick_builder.export`) and evaluates through that instead, so the same command checks that the exported
  policy reproduces the torch one's episode outcomes under the same seed (the acceptance criterion
  checked by `tests/scripts/test_eval.py::test_c_actor_matches_torch_actor_outcomes`).
- `--sweep press-misalign` — instead of a flat episode batch, runs `BrickBuilder/PressFromHeld-v0`'s own
  misalignment sweep (`brick_builder.envs.press_env.baseline_jobs`/`sweep`) against the checkpoint and writes
  the sweep's JSON and success-curve figure. This is how "beats the scripted baseline by >=15 points at
  4-6 mm" (Deliverable 1's acceptance target) gets measured, by pointing the same sweep the
  scripted baseline is measured with at a checkpoint instead: `--actor torch` only (the sweep's own
  `evaluate_point` loads a torch checkpoint directly, once per worker process).

## SAC: the off-policy fallback recipe

The decision on record: fall back from PPO to an off-policy learner if throughput falls
below about 2,000 environment steps/s or a task needs more than about 50 million steps to converge.
An earlier measurement put Reach at 208 steps/s single-process and Press at 41-82 steps/s — one to
two orders of magnitude under that gate — which is what `brick_builder/rl/sac_asym.py` exists to answer.

`brick_builder.rl.sac_asym.train` implements soft actor-critic (Haarnoja et al. 2018) with the same
asymmetric-observation contract as PPO (`{"actor": Box, "critic": Box}`) and deliberately reuses PPO's
pieces where the two algorithms mean the same thing:

- **The actor** is exactly `brick_builder.rl.nets.Actor` — the same tanh MLP, the same state-independent
  `log_std`, no new parameters — so a SAC checkpoint's `actor` state dict has the identical layer-key
  layout a PPO checkpoint's does. This is what keeps `brick_builder/export`'s C exporter (which only ever
  reads a checkpoint's `actor`/`norm_actor`/`cfg`/`obs_spec` keys) working unchanged for either learner.
- **The input normalizer** is `brick_builder.rl.nets.RunningNorm`, used exactly as PPO uses it (frozen for
  eval/export).
- **The checkpoint layout** is the same top-level keys plus twin critics (`critic1`, `critic2` instead
  of PPO's single `critic`) and one new `cfg` field, `"squash"`, that is not PPO's concern.

What differs, because SAC needs it and PPO does not:

- **A tanh-squashed action.** PPO's deployed action is the actor's raw mean, clamped to the action
  bounds. SAC's is `tanh(mean)` mapped onto the bounds (`action = tanh(mean) * scale + bias`, `scale =
  (high-low)/2`, `bias = (high+low)/2`) — the deployed action is `tanh(mean)`, and it is what the
  checkpoint's `cfg["squash"] = "tanh"` field records for the C exporter (see
  [Exporting a SAC actor](#exporting-a-sac-actor) below).
- **Twin critics with LayerNorm** (`brick_builder.rl.sac_asym.QCritic`, not `nets.Critic`): `Linear ->
  LayerNorm -> Tanh` per hidden layer over the concatenated (critic observation, action), plus Polyak-
  averaged target networks. LayerNorm is what keeps the critic loss stable under this recipe's
  update-to-data ratio; PPO's on-policy critic never needs it. Never exported.
- **A replay buffer with 3-step returns**, CPU-only numpy, and **automatic entropy tuning** (a learned
  temperature, target entropy `-act_dim` by default).

### Configuration — `SACConfig`

| Field | Default | Meaning |
|---|---|---|
| `n_envs` | 8 | parallel environments |
| `n_step` | 3 | return length before bootstrapping from the target critics |
| `gamma`, `tau` | 0.99, 0.005 | discount and target-critic Polyak averaging rate |
| `lr_actor`, `lr_critic`, `lr_alpha` | 3e-4 each | Adam learning rates |
| `init_alpha`, `target_entropy` | 0.2, `None` (-> `-act_dim`) | entropy temperature starting point and target |
| `buffer_size`, `batch_size` | 200 000, 256 | replay buffer capacity and sample size per update |
| `warmup_steps` | 1 000 | env-steps of uniform-random action before any gradient update |
| `utd_ratio` | 0.5 | gradient updates per environment step, aggregated over `n_envs` |
| `init_log_std` | -0.5 | initial actor exploration noise |
| `total_steps` | *(required)* | training stops once this many env-steps have been collected |
| `seed` | 0 | seeds `torch.manual_seed` and the `numpy.random.Generator` the replay buffer samples with |
| `hidden_actor`, `hidden_critic` | (64,64), (256,256) | network widths (same defaults as `PPOConfig`) |
| `async_envs`, `vector_context`, `norm_clip`, `log_every` | same defaults and meaning as `PPOConfig` | |
| `squash` | `"tanh"` | recorded in the checkpoint; the only value `brick_builder/export` currently implements the SAC side of |

`brick_builder.rl.sac_asym.train(make_env, cfg, run_dir, callback=None)` has the identical call shape as
`ppo_asym.train`, and `brick_builder.rl.sac_asym.evaluate(checkpoint, make_env, episodes, seed)` the identical
shape as `ppo_asym.evaluate` (deterministic `tanh(mean)` rollouts, the same `EpisodeResult` fallbacks).
`brick_builder.rl.sac_asym.plot_learning_curve` is not a separate implementation — it re-exports
`ppo_asym.plot_learning_curve`, since turning a `(step, mean_return)` list into a PNG does not depend on
which algorithm produced the list.

### Command line: `algo: sac`

```bash
.venv/bin/python scripts/train.py --config configs/train/reach_sac.yaml
.venv/bin/python scripts/train.py --config configs/train/press_sac.yaml
.venv/bin/python scripts/train.py --config configs/train/align_sac.yaml
```

A config selects SAC with two changes from the PPO shape: `algo: sac` and an `sac:` section (in place
of `ppo:`) with `SACConfig`'s fields. Everything else — `env`, `env_kwargs`, `run_name`,
`--total-steps`/`--seed`/`--n-envs`/`--sync-envs` overrides — works exactly as for
PPO. The run directory gets `sac_config.json` (the resolved `SACConfig`) in place of `ppo_config.json`;
everything else (`config.yaml`, `tb/`, `checkpoint.pt`, `returns.json`, `learning_curve.png`) is named
the same either way.

`configs/train/reach_sac.yaml`, `press_sac.yaml` and `align_sac.yaml` mirror their PPO counterparts'
tasks and randomization settings with SAC hyperparameters in place of PPO ones. **None of the three
has been trained to convergence.** A run against the
real Brick Builder Arm Robot press task was capped at a five-minute smoke test; the full training campaign is
future work. What a short smoke run of `press_sac.yaml` actually
measured is in the open issues below.

### Exporting a SAC actor

`scripts/eval.py --actor c` and `brick_builder.export.build.build_library` work on a SAC checkpoint exactly
as on a PPO one — the checkpoint layout is compatible by construction (see above) — except for the
final step `c/policy.c` takes: `policy_theta.h`'s `POLICY_SQUASH_TANH` macro (from the checkpoint's
`cfg["squash"]`, `0` for a checkpoint with no such field, i.e. every existing PPO checkpoint) picks
between clamping the raw output (PPO) and `tanh(raw) * scale + bias` (SAC), reusing the same
`policy_tanh` the lookup-table build variant already provides. `brick_builder.export.parity.check_parity`
picks the matching reference computation on the torch side from the same field, so the same function
validates both learners.

Measured on this machine (`tests/export/test_sac_export.py`, a checkpoint from a few SAC updates on the
Pendulum wrapper, same fixture pattern as the PPO export tests): the SAC checkpoint's header carries
`POLICY_SQUASH_TANH 1` (a PPO checkpoint's carries `0`, unchanged from before this field existed);
parity between the torch actor and the compiled C actor is within the same `<= 1e-5` bound as PPO's, for
both the float-`tanhf` and lookup-table builds, over 10,000 random observations.

### Measured: Pendulum learning time and update throughput

`tests/rl/test_sac_asym.py::test_learns_pendulum` (`.venv/bin/python -m pytest -q -m slow
tests/rl/test_sac_asym.py::test_learns_pendulum -s`), same Dict-wrapped `Pendulum-v1` wrapper as PPO's
own acceptance test, `SACConfig(n_envs=4, total_steps=8_000, warmup_steps=300, batch_size=256,
buffer_size=50_000, utd_ratio=1.0, lr_actor=1e-3, lr_critic=1e-3, gamma=0.98, hidden_critic=(128, 128),
async_envs=False)`, seed 0:

![SAC Pendulum learning curve](../media/rl/sac_pendulum_learning_curve.png)

Measured on this project's dev machine (CPU only, `SyncVectorEnv`, **under sustained load from an
unrelated CPU-heavy job already running on this shared machine** — `cat /proc/loadavg` read 10.37 12.83
10.21 at the moment the run finished): mean return over 10 held-out episodes reaches **-168.3** (up from
about -1200 for a random policy), in **69.2 seconds** of wall-clock time for training plus evaluation —
well inside the 3-minute acceptance budget even under this load. `hidden_critic=(128, 128)` here (rather
than `SACConfig`'s (256, 256) default) is a deliberate choice for this 3-dimensional toy task, not a
general recommendation — see the update-throughput measurement below for why.

> **Predates rl-eval-export-1.** This number was measured before the fix to `_flush_n_step`'s
> truncation handling (a time-limit truncation no longer stores `done=1` and drops the bootstrap
> the way a true termination correctly does; it now stores `done=0` and bootstraps from the
> episode's real final observation). Re-running the same command post-fix (seed 0, same config)
> still clears the acceptance bar comfortably — mean return **-168.0** over 10 held-out episodes —
> so the number above happens to hold up almost unchanged for this particular toy task; it has not
> been re-measured carefully enough to replace the plot or the "-168.3" figure as the page's
> canonical number, and PPO's number below did change materially (see its own note).

`tests/rl/test_sac_asym.py::test_update_throughput_on_brick_builder_sized_networks` measures `sac_update`'s
own cost directly (random tensors, no environment stepping), at the observation/action sizes
relevant to the actual press skill: actor obs ~150, critic obs ~250, action 8 (the
real numbers, measured from a five-minute BrickBuilder/PressFromHeld-v0 smoke run — see below — were 133,
192 and 8), `hidden_actor=(64,64)`, `hidden_critic=(256,256)`, batch 256. Measured on this machine, same
loaded conditions (`cat /proc/loadavg`: 8.83 12.20 10.09): **71.6 updates/second**. This barely changes
with critic width (a separate measurement at `hidden_critic=(64,64)`/`(128,128)`/`(256,256)`, same
dims otherwise, gave 65-85 updates/s across all three) — at this batch size and network scale, the
per-call Python/torch overhead of five forward/backward passes (two critics, two targets, the actor)
dominates over the actual FLOPs, so shrinking the network does not meaningfully speed up training; the
Pendulum acceptance test's own budget was tuned by controlling `total_steps * utd_ratio` (the number of
gradient updates), not network width.

A five-minute smoke run of `press_sac.yaml` (`--sync-envs`, `n_envs=4`, `warmup_steps=20`,
`batch_size=32`, `buffer_size=500`, `utd_ratio=0.5`, `total_steps=400` — deliberately tiny, to avoid
a real training run on the press task) ran end to end in
about 11 seconds on this machine, confirmed the real `PressFromHeld-v0` dimensions (actor obs 133,
critic obs 192, action 8, level 0 with domain randomization on) flow through `sac_update` without error,
and produced a checkpoint that loads and exports the same way the Pendulum one does. It is a plumbing
check, not a learning result — see the open issues below.

## Reproducing the example run

Every number and the plot on this page come from the actual acceptance test,
`tests/rl/test_ppo_asym.py::test_learns_pendulum`, run with `.venv/bin/python -m pytest -q -m slow
tests/rl/test_ppo_asym.py::test_learns_pendulum`. It wraps `Pendulum-v1` so its plain `Box` observation
becomes `{"actor": obs, "critic": obs}` (both halves identical — Pendulum's observation already is the
full state, so there is no extra privileged information to hand the critic), trains for 400,000 steps
with `gamma=0.9` and `lr=1e-3` (tuned for this short-horizon toy task, not the arm's `PPOConfig`
defaults), and evaluates for 10 held-out episodes.

![Pendulum learning curve](../media/rl/pendulum_learning_curve.png)

Measured on this project's dev machine (CPU only, `n_envs=8`, `SyncVectorEnv`): mean return improves
from about -1200 (random policy) to -271.6 over held-out episodes, in about 137 seconds of wall-clock
time for training plus evaluation together — comfortably inside the acceptance test's 4-minute budget.

> **Predates rl-eval-export-1.** This -271.6 figure (and the plot above) was measured before the fix
> to PPO's GAE bootstrap: every episode here truncates at Pendulum's 200-step time limit, and the
> pre-fix code treated that truncation as a true termination, zeroing the value bootstrap instead of
> bootstrapping from the truncated episode's real final observation. This page's plot has not been
> regenerated against the fixed code (that needs `scripts/make_media.py`, outside this fix's scope),
> but a controlled before/after comparison at reduced scale (150,000 of the full 400,000 steps, same
> seed and hyperparameters, run against both the pre-fix loop and the current code) measured mean
> return over 20 held-out episodes at **-499.5 (fixed) vs. -990.9 (pre-fix)** — nearly 2x better at
> the same step count, with the two learning curves identical through the first update (same actions,
> since only the value target differs) and diverging in the fixed run's favor from there on. The full
> acceptance test still passes post-fix (mean return well past the -300 bar), but -271.6 is a pre-fix
> number and should be re-measured (and the plot regenerated) the next time this page's media is
> rebuilt.

## Reach: a trained skill

`configs/train/reach.yaml` reproduces `runs/reach_smoke/` exactly: 6 async worker processes, CPU only,
`total_steps=307200`. Measured on this machine: training finished in 505 s (8.4 min); evaluating the
result over 200 held-out episodes (`scripts/eval.py --checkpoint runs/reach_smoke/checkpoint.pt --env
BrickBuilder/Reach-v0 --episodes 200 --seed 10000`) gives **200/200 successes**, Wilson interval
`[0.981, 1.000]`, mean final TCP error 1.02 mm — well past the 90% target. This is the checkpoint the
`policy` demo plays by default (`docs/demos/README.md`).

## Press: measured results

`configs/train/press.yaml` is a real training configuration (2,000,000 steps — realistically hours on
this shared, 6-worker-process machine); running it to convergence and measuring the "beats scripted by
>=15 points at 4-6 mm" acceptance target is future work. What was actually run, with a tiny smoke
budget (`--sync-envs --n-envs 4 --total-steps 6144`, i.e. 6 PPO updates — a few minutes on this
machine):

| policy | level | episodes | successes | 95% Wilson CI |
|---|---|---|---|---|
| scripted baseline (`press_env.scripted_action`) | 0 | 20 | 16 | [0.583, 0.927] |
| 6-update smoke checkpoint | 0 | 20 | 15 | [0.531, 0.888] |

The two overlap completely — six PPO updates move the actor's weights off their random init and nothing
more; this is a check that the training path (`scripts/train.py` -> `PressFromHeld-v0` ->
`scripts/eval.py`) runs correctly end to end, not a claim that the residual policy has learned anything
yet. A real run needs `configs/train/press.yaml`'s full budget (or several times it — see the open issues
below) before the sweep comparison (`scripts/eval.py --sweep press-misalign`) means anything.

## AlignFromTokens: measured results

Same story, `configs/train/align.yaml`, same tiny smoke budget (6 PPO updates): 0/20 successes at the
default 3 mm / 5 deg tolerance, against the scripted (zero-residual) go-to-token controller's own 2/20
at the same seeds — both are small-sample noise around a genuinely hard threshold (see
`brick_builder/envs/align_env.py`'s own module docstring for the measured effect of token noise, age and
producer on this number). `configs/train/align.yaml`'s real budget is 600,000 steps.

## Open issues

- **Neither `press.yaml` nor `align.yaml` has been trained to convergence.** Both configs are real
  (not smoke) training budgets; actually running them, on this shared machine, under the project's
  6-worker-process limit, is realistically an overnight job not yet done. The numbers above are only
  a correctness check of the tooling.
- **This machine's throughput for `PressFromHeld-v0` was noticeably worse than `Reach-v0`'s** (measured
  here at roughly 30 single-process steps/s at level 0 with domain randomization on, versus `Reach-v0`'s
  reported ~180-600 steps/s combined across 6 workers) — some of this is the heavier scene (a press
  scene has more bodies/equality constraints than Reach's bare gripper, see `docs/guide/02-simulator-
  and-bricks.md`'s own physics-rate measurements), and some of it was simply CPU contention with other
  work running on the same 16-thread machine at measurement time. Budget training time generously.
- **`scripts/train.py` has no automatic teacher/student/scripted comparison report** — `PressEnv`'s own
  `scripted_action`/`oracle_action` (a privileged, cheating residual) and `scripts/eval.py --sweep
  press-misalign` give the pieces (`brick_builder.envs.press_env.evaluate_point` already accepts
  `policy="oracle"`), but nothing wires the three into one comparison table yet.
- **`--sweep press-misalign` only supports `--actor torch`** — its `evaluate_point` loads a torch
  checkpoint directly inside a worker process for pickling reasons; a C-actor sweep would need its own
  entry point.
- **None of the SAC configs (`reach_sac.yaml`, `press_sac.yaml`, `align_sac.yaml`) has been trained to
  convergence, or even tuned against a real Brick Builder Arm Robot task's reward scale.** The algorithm and export
  path are verified on Pendulum (learns) and on a five-minute Press smoke run
  (plumbing only, not learning); choosing between PPO and SAC for the actual Deliverable-1 campaign, and
  tuning whichever is chosen, is still open.
- **`scripts/eval.py`'s `--actor torch` path does not know about SAC's tanh squash.** It calls
  `brick_builder.demo.common.load_actor_fn`, which reconstructs a checkpoint's actor with
  `brick_builder.rl.ppo_asym.load_actor`/`select_action` — the PPO clamp convention — regardless of which
  learner wrote the checkpoint. A SAC
  checkpoint's C export (`--actor c`) and `brick_builder.rl.sac_asym.evaluate` both apply the correct tanh
  squash; only the flat `scripts/eval.py --checkpoint <sac-checkpoint> --actor torch` path does not.
  Fixing this needs a small, squash-aware change to `brick_builder/demo/common.py` (check the checkpoint's
  `cfg["squash"]` and branch) — not a technical blocker, just not yet done.
- **`utd_ratio` and network width trade off wall-clock cost in a way this page's own throughput
  measurement should inform** — see "Measured: Pendulum learning time and
  update throughput" above: at Pendulum/press scale, per-call torch overhead dominates FLOPs, so a
  faster training budget should first cut `total_steps * utd_ratio` (the update count), not shrink
  `hidden_critic`.
