## What does this PR do?

<!-- One or two sentences. Link the issue or task, if there is one. -->

## Checklist

Project rules, from [CONTRIBUTING.md](../CONTRIBUTING.md):

- [ ] Docs are updated in this same PR, if this PR changes anything a doc page describes
- [ ] If this PR finishes a stage (or a visible piece of one), it includes a demo or a figure
      under `docs/media/`, regenerated via `scripts/make_media.py`
- [ ] No new `_v2` / `_new` / copy-of-a-file — existing files were fixed in place
- [ ] New or changed tests assert real behaviour; any unmet target is a diagnosed `xfail` with the
      measured numbers in the reason string, not a weakened assertion
- [ ] Plain English; any new term is added to [the glossary](../docs/glossary.md)

Before requesting review:

- [ ] `.venv/bin/python -m pytest -q -m "not slow"` passes locally
- [ ] `.venv/bin/mkdocs build --strict` passes locally
- [ ] At least one reviewer requested

## Notes for the reviewer

<!-- Anything that needs context: known gaps, follow-ups, things you're unsure about. -->
