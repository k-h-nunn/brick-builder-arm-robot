# Install

A working environment for the simulator, the training code, the docs and the test suite. Real
hardware needs the extra steps in [Hardware, wiring, calibration](08-hardware-wiring-calibration.md).

## Requirements

| | |
|---|---|
| OS | Linux (Ubuntu 24.04 recommended) or Windows via WSL2 (Ubuntu 24.04). |
| Python | 3.12. The package declares `>=3.10`; 3.12 is what CI runs. |
| GPU | Not required. Everything here runs on CPU. |
| Disk | A few gigabytes: MuJoCo is small, PyTorch is not. |
| Rendering | Off-screen only, through EGL or OSMesa. No display server is used. |

## Virtual environment

One environment per project, created from the system interpreter, in the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
```

Every command in these docs calls `.venv/bin/python` or `.venv/bin/pip` explicitly. On a shared
machine the shell's `python3` and `pip` may belong to something else entirely.

## PyTorch

Install the CPU wheel before the project, so pip does not pull the CUDA build:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The CUDA build is several gigabytes and buys nothing here. Skip this step to get the default
wheel if you have a GPU and want to use it.

## The project

```bash
.venv/bin/pip install -e ".[dev,docs]"
```

Editable, so code changes take effect without reinstalling. The extras:

| Extra | Contents | Needed for |
|---|---|---|
| (base) | `mujoco`, `gymnasium`, `numpy`, `scipy`, `torch`, `opencv-python-headless`, `pillow`, `pyyaml`, `tensorboard`, `imageio[ffmpeg]`, `matplotlib` | the package itself |
| `dev` | `pytest` | the test suite |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | building this site |

`imageio[ffmpeg]` is what writes the demo mp4s; `pillow` is what draws the video's text panels;
`tensorboard` is optional at runtime (training skips the event log if it is missing) but is
declared so a fresh install gets it.

## Rendering backend

MuJoCo needs an off-screen GL backend, named through `MUJOCO_GL`:

| Value | Needs | Use when |
|---|---|---|
| `egl` | a GPU or DRM device and its EGL driver | a workstation with a graphics card |
| `osmesa` | `libosmesa6` (`sudo apt-get install -y libosmesa6`) | a CI runner, a container, a VM |

Check it:

```bash
MUJOCO_GL=egl .venv/bin/python -c "import mujoco; print(mujoco.__version__)"
```

If that raises, use `osmesa` instead. `tests/conftest.py` sets `MUJOCO_GL=egl` only when the
variable is unset, so exporting it wins:

```bash
export MUJOCO_GL=osmesa
```

## Environment variables

Every command in these docs is prefixed with the same three:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=4 .venv/bin/python ...
```

- `CUDA_VISIBLE_DEVICES=""` keeps training and evaluation on CPU, so a shared GPU is never
  claimed. Leave it unset to use one.
- `MUJOCO_GL` as above.
- `OMP_NUM_THREADS=4` stops PyTorch and BLAS from spawning a thread per core, which is slower
  than four threads on a machine running anything else.

## Verify

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q -m "not slow"
.venv/bin/mkdocs build --strict
```

A green fast suite and a clean strict docs build is this repository's definition of working. The
fast suite takes about fourteen minutes; `-m slow` adds the PPO training smoke tests, the longer
physics settles and the press baseline sweep.

Next: [Quickstart](../quickstart.md).

## Conventions

- Never `pip install` with the shell's default `pip`. Check `which pip` first.
- Fix files in place. No `_v2` copies.
- An unmet target is a measured `xfail` with the number in its reason, not a weakened assertion.
