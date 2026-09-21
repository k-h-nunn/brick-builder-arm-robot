# Vendored: SO-101 robot model (SO-ARM100)

- **Source:** https://github.com/TheRobotStudio/SO-ARM100
- **Commit:** `eecbe3e0a9ebb23e25ad7b2759b03884c6660903`
- **Date recorded:** 2026-09-20
- **License:** Apache-2.0 (see `LICENSE` in this directory, copied from the repo root)
- **Fetched with:** `git clone --depth 1 --filter=blob:none --sparse <url>`, then
  `git sparse-checkout set Simulation/SO101`.

## Files in this directory

- `so101_new_calib.xml` — MJCF, current calibration (from `Simulation/SO101/so101_new_calib.xml`)
- `so101_old_calib.xml` — MJCF, old calibration (from `Simulation/SO101/so101_old_calib.xml`)
- `scene.xml` — MJCF scene wrapper (from `Simulation/SO101/scene.xml`)
- `assets/` — STL meshes (28 files, from `Simulation/SO101/assets/`)
- `UPSTREAM_README.md` — the folder's own README (from `Simulation/SO101/README.md`)
- `LICENSE` — the upstream repo's Apache-2.0 license file

## Warning

Upstream actuator kp (~998) is roughly 50-100x stiffer than the real STS3215 firmware gain
and forcerange is the 12 V spec; `brick_builder/robot/patch_so101.py` corrects both
programmatically; never hand-edit these files.

## `so101_new_calib.xml` contents (for the next wave)

**Joints** (all `type="hinge"`, `axis="0 0 1"`, default class `sts3215`):
- `shoulder_pan` range [-1.9199, 1.9199] rad
- `shoulder_lift` range [-1.7453, 1.7453] rad
- `elbow_flex` range [-1.69, 1.69] rad
- `wrist_flex` range [-1.6581, 1.6581] rad
- `wrist_roll` range [-2.7438, 2.8412] rad
- `gripper` range [-0.1745, 1.7453] rad

**Actuators** (all `<position>` type, one per joint, named after the joint):
- `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- Each has `forcerange="-3.35 3.35"` set directly on the actuator element.
- `kp` is not set per-actuator; it is inherited from the `sts3215` default class:
  `kp="998.22"`, `kv="2.731"`, and that class's own `forcerange="-2.94 2.94"` (overridden
  per-actuator as above, so the effective forcerange is ±3.35, not ±2.94).

**Default classes (hierarchy):**
- (root, unnamed) → `joint damping=1 frictionloss=0.1 armature=0.005`, `position kp=50`
  - `visual` → `geom type=mesh contype=0 conaffinity=0 group=2`
  - `collision` → `geom group=3`
- (root, unnamed, second block) → `sts3215` → `geom contype=0 conaffinity=0`,
  `joint damping=0.60 frictionloss=0.052 armature=0.028`,
  `position kp=998.22 kv=2.731 forcerange="-2.94 2.94"`
  - `backlash` → `joint damping=0.01 frictionloss=0 armature=0.01 limited=true
    range=[-0.008726646259971648, 0.008726646259971648]` (~±0.5 degrees)

Note: the file defines two separate unnamed top-level `<default>` blocks (one holding
`so101_new_calib`/`visual`/`collision`, one holding `sts3215`/`backlash`); the values above
are read straight off the XML, not merged or reinterpreted.

**Sites:** `baseframe`, `gripperframe` (no `class` attribute set on either).

**Cameras:** none defined in `so101_new_calib.xml` itself.

## Sanity check

Well-formedness of `so101_new_calib.xml` was verified with `/usr/bin/python3`'s stdlib
`xml.etree.ElementTree` (parses cleanly, root tag `mujoco`). This is a stdlib-only syntax
check; it does not validate MuJoCo compilation.
