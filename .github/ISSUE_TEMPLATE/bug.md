---
name: Bug report
about: Something does not work the way its tests or docs say it should
title: "[bug] "
labels: bug
---

## What happened

<!-- What you ran, and what happened. Paste the actual error/output if there is one. -->

## What you expected instead

<!-- What a test, a doc page, or a design report says should happen. Link it if you can. -->

## How to reproduce

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python ...
```

## Environment

- Branch / commit:
- Rendering backend (`MUJOCO_GL=egl` or `osmesa`):
- Anything non-default about your setup (GPU vs CPU, OS, etc.):

## Anything else

<!-- Logs, a figure from docs/media/, related issues. -->
