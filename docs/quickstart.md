# Quickstart

Fifteen minutes from a clean clone to a closed-loop placement in simulation. [Install](guide/01-install.md)
has the full requirements and the reasoning behind each step.

## Install

```bash
git clone <repo-url> brick-builder-arm-robot
cd brick-builder-arm-robot
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,docs]"
```

Installing the CPU wheel of PyTorch first keeps the CUDA build (several gigabytes) out of the
environment. Drop that line to get the default build if you have a GPU.

Every command below uses `.venv/bin/python` explicitly, never a bare `python`.

## Check the renderer

MuJoCo renders off-screen through EGL or OSMesa. EGL needs a GPU or a DRM device; OSMesa is pure
software and works anywhere.

```bash
MUJOCO_GL=egl .venv/bin/python -c "import mujoco; print(mujoco.__version__)"
```

If that fails, use `MUJOCO_GL=osmesa` in every command on this page.

## Run the tests

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q -m "not slow"
```

About fourteen minutes. Drop `-m "not slow"` to include the PPO training smoke tests, the longer
physics settles and the press baseline sweep.

## Run a closed-loop placement

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/run_placement.py --cnn
```

A 2x2 DUPLO brick held in the gripper, a baseplate moved a few millimetres and a couple of degrees
from where the controller believes it is, and a target sequence recorded in the digital twin. The
controller tracks that sequence on scene-state error at 10 Hz. About forty seconds, and it prints
the per-step convergence and whether the brick ended up seated at the target cell:

```
baseplate offset from nominal: 6.45 mm, 2.86 deg (unknown to the controller)
placement: seated=True at_target_cell=True lateral=0.15 mm depth_error=0.00 mm
RESULT: placed
```

`--cnn` drives the loop from the trained convolutional scene model, which is what the film shows.
Without it the same loop runs on simulator ground truth instead. `--seed` redraws the baseplate
offset and the brick's offset in the jaws; `--pick` fetches the brick from the table first.

## Watch a scripted press

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.demo press --record press.mp4
```

A few seconds, and writes an mp4 of a 2x4 brick being pressed onto an 8x8 baseplate. It prints
whether the brick seated: blind, at the default 1 mm of unknown misalignment, usually it does not.
That is the limit this simulator is built to measure, and the reason the closed loop above exists.

Every demo records to a file rather than opening a window: `scene`, `teleop`, `press`, `rocking`,
`force-map`, `policy`, `tokens`. See [the demo gallery](demos/README.md).

## Render the film

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/make_demo_video.py \
    --quality draft --cnn-drives --stills 0 --out media_out/placement.mp4
```

`--cnn-drives` runs the loop on the convolutional scene model, which is what the published film
shows. The annotated 1920x1080 version is `--quality hd1080`, which takes considerably longer;
[the film page](demos/videos.md) has the exact command it was made with, and what every on-screen
panel means.

## Train a policy

`BrickBuilder/Reach-v0` is the fastest skill to train to its acceptance target, about 8 minutes on
6 worker processes:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/train.py --config configs/train/reach.yaml
```

That writes `runs/reach/checkpoint.pt`, a TensorBoard log and a learning curve. Play it back:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python -m brick_builder.demo policy \
    --record reach_policy.mp4 --checkpoint runs/reach/checkpoint.pt
```

Or score it, with a JSON and markdown report and a Wilson interval on the success rate:

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl .venv/bin/python scripts/eval.py \
    --checkpoint runs/reach/checkpoint.pt --env BrickBuilder/Reach-v0 --episodes 100
```

`--actor scripted` runs the same report path with no checkpoint: the zero-residual baseline every
evaluation is measured against. `configs/train/press.yaml` and `configs/train/align.yaml` train the
two harder skills; see [Training and evaluation](guide/05-training-and-evaluation.md) for the budget
each really needs.

## Read the docs

```bash
.venv/bin/mkdocs serve
```

Then open <http://127.0.0.1:8000>. `mkdocs build --strict` builds the static site into `site/` and
fails on a broken link.

## Assumptions

- Linux, Python 3.12, a few gigabytes of free disk for MuJoCo and PyTorch.
- No GPU. `CUDA_VISIBLE_DEVICES=""` is set deliberately in every command above: training and
  evaluation run on CPU by default so a shared GPU is never claimed. Leave it unset if one is free.
- Simulation only. Real hardware is [Hardware, wiring, calibration](guide/08-hardware-wiring-calibration.md)
  and [Running the robot](guide/09-running-the-robot.md).
