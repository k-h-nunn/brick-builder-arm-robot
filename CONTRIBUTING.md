# Contributing

This is a four-person team project (CS 6341, Robotics, UT Dallas). These are the working rules
for all four of us, written down so they do not depend on anyone's memory.

## Workflow

- **Branch per task**, off `main`. Name it however makes sense to you (`press-bench-tests`,
  `align-env`, whatever); there is no enforced convention.
- **Pull requests into `main`.** Nobody pushes directly to `main`.
- **At least one review** from another teammate before merging. Small, obviously-correct PRs
  (a typo fix, a doc link) still go through a PR, but do not need to wait long for review.
- Use [`.github/pull_request_template.md`](.github/pull_request_template.md)'s checklist — it is
  filled in automatically when you open a PR.

## Requirements to merge

Both of these, run locally and (once CI is set up on the repo) in the `CI` GitHub Actions check:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q -m "not slow"
.venv/bin/mkdocs build --strict
```

If either fails, the PR is not ready, no exceptions for "it's just docs" or "it's just a small
change" — both commands are fast enough (well under ten minutes combined) that there is no good
reason to skip them.

## The project's rules

These apply to everyone, not just whoever wrote a rule down first:

1. **Docs change in the same pull request as the code they describe.** A PR that changes what a
   module does, without touching the guide page that describes it, is not done yet. Stale docs are
   worse than no docs, because they actively mislead the next person.
2. **Every stage needs a demo or a figure**, not just passing tests. If you finish a piece of a
   stage that has a visible effect (a plot, a video, a rendered scene), regenerate it and check it
   in under `docs/media/`. See "Adding a demo or a figure" below.
3. **Fix files in place. Never add a copy** (`foo_v2.py`, `press_env_new.py`, `bench_fixed.py`).
   If a module needs reworking, edit it; if you truly need to keep the old version around for
   comparison, that is what git history is for. Duplicate near-identical files are one of the
   fastest ways this repo would stop making sense to the next person who opens it.
4. **Tests assert real behaviour.** If a target is not met yet, the test for it is a
   diagnosed `xfail` with the actual measured numbers in the reason string (see any test in
   `tests/bricks/test_snap_bench.py` or `tests/control/test_skill_fsm.py` for the pattern), not a
   loosened threshold and not a test that silently passes. An honest "this doesn't work yet, and
   here is the number" is useful; a quietly weakened assertion is not.
5. **Plain English.** No coined terms, no unexplained jargon. If you introduce a new term, add it
   to [the glossary](docs/glossary.md). This also means: write the docs for someone who was not in
   the room when the decision was made.

## Running the tests

```bash
# Fast subset (about fourteen minutes) — run this before every push
.venv/bin/python -m pytest -q -m "not slow"

# Slow tests too (PPO training smoke tests, longer physics settles, the full
# press baseline sweep) — several minutes; run before merging anything that
# touches training, physics timing, or anything the slow tests cover
.venv/bin/python -m pytest -q -m slow

# Everything
.venv/bin/python -m pytest -q
```

Always `MUJOCO_GL=egl` (or whatever backend your machine uses — see
[Install](docs/guide/01-install.md)) and `CUDA_VISIBLE_DEVICES=""` unless you deliberately want a
GPU run; `tests/conftest.py` sets sane defaults for both, but an explicit `CUDA_VISIBLE_DEVICES=""`
on a shared machine is still good manners.

## Adding a demo and a figure

- Demos are subcommands of `python -m brick_builder.demo <name>` (see `brick_builder/demo/`). A new demo is
  a new module there plus an entry in `brick_builder/demo/__init__.py`'s `DEMOS`; it should run headless and
  support `--record <path>`, matching the existing demos.
- Figures live under `docs/media/<subsystem>/` and are produced by a plain function inside the
  package, not by a one-off script — `scripts/make_media.py` just calls each subsystem's
  figure-producing function, with a fixed seed, so every picture in the docs stays reproducible
  from the code that made it. Add your function, then wire it into the matching section (or a new
  one) in `scripts/make_media.py`'s `SECTIONS` mapping.
- Regenerate and check the result before committing it:

  ```bash
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/make_media.py --only <your-section>
  ```

- See [docs/demos/README.md](docs/demos/README.md) for the existing gallery and its format.

## Large files

- `*.mp4` and `runs/` are git-ignored (see `.gitignore`) — videos and training runs do not belong
  in git history. A demo's `--record` output is for local viewing; if it is worth keeping in the
  docs, convert the point you want to a short GIF instead.
- Keep anything under `docs/media/` **under 3 MB**. Most figures are well under this already
  (PNGs are tens to low hundreds of KB; the existing GIFs are 0.3-1.2 MB). If a GIF comes out
  bigger than that, shorten it, drop the frame rate, or reduce it in size before committing.
- `.gitattributes` marks `*.stl`, `*.pt`, `*.png`, `*.gif`, and `*.mp4` as binary, so git never
  tries to diff or line-ending-normalize them.

## Adding a scene-model source

`brick_builder.perception.sources` maps a name to a factory. A source that lives in another
package registers itself and is then reachable by name, and nothing in this repository has to
import it:

```python
from brick_builder.perception import sources

sources.register("my_model", lambda **kw: MyModelSceneModel(**kw))
```

The factory returns anything satisfying `perception.scene_model.SceneModelSource`: one method,
`observe(rgb, *, window, instance=None, depth=None, present_mask=None, t=None) -> SceneState`.
A source whose row layout is not a documented set of named fields publishes
`scene_model.opaque_schema(W)` rather than claiming `PLACEHOLDER_SLOT_SCHEMA`'s fields; consumers
ask the schema what is available. See [Perception](docs/guide/06-perception.md).


## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). It is short.
