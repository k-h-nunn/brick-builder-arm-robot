# Vendored: SO-101 wrist camera mount (32x32 mm USB UVC module, Hex-Nut Recess variant)

- **Source:** https://github.com/TheRobotStudio/SO-ARM100
- **Path:** `Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/`
- **Commit:** `eecbe3e0a9ebb23e25ad7b2759b03884c6660903` (same commit as the rest of
  `third_party/so101/`; see `../VENDORED.md`)
- **Date recorded:** 2026-09-21
- **License:** Apache-2.0 -- see `../LICENSE` (copied from the same upstream repo root; not
  duplicated here since it is the same repo and commit)
- **Fetched with:** `git clone --depth 1 --filter=blob:none <url>` (full non-sparse clone, read
  from the working tree), then `cp` of the four files below.

## Why this variant

Upstream's own root `README.md` (section "Wrist-Mount Cameras") lists three 32x32 mm-UVC wrist
mounts side by side and labels only one of them "(SO101)":

| Mount | Upstream's own label |
|---|---|
| `SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/` (this one) | **32x32 UVC Hex Nut (SO101)** |
| `Wrist_Cam_Mount_32x32_UVC_Module/` | 32x32 UVC Integrated (SO100/101) -- a one-piece
  replacement for the wrist-roll-follower part itself, not an add-on bracket |
| `Wrist_Cam_Plug_Mount_32x32_UVC_Module/` | 32x32 UVC Plug-on -- SO-100 only per its own README
  title |

This is the mount the project's stated hardware matches (bolt-on bracket + separate 32x32 UVC
module on a stock, unmodified SO101 gripper), and the one `brick_builder/robot/patch_so101.py`'s
`WRIST_CAM_POS_IN_GRIPPER` / `WRIST_CAM_QUAT_IN_GRIPPER` are derived from.

## Files in this directory

- `README.md` -- the mount's own upstream installation guide (hardware list: 1x 32x32 UVC module,
  1x 3D-printed adapter, 4x M2 screws, 2x M3x8mm screws, 2x M3 hex nuts; assembly steps).
- `stl/SO-ARM101_camera_wrist_mount.stl` -- the printed adapter, binary STL, **millimetres** (not
  metres -- unlike the rest of `third_party/so101/assets/`, this file is upstream's plain
  print-ready CAD export; anything reading it into the same scene as the arm's own meter-scaled
  meshes must scale by 0.001, exactly as `Simulation/SO101/so101_new_calib_camera.xml`'s own
  `<mesh ... scale="0.001 0.001 0.001"/>` does).
- `stl/SO-ARM101_camera_wrist_mount.step` -- the same part as a STEP solid, for anyone who wants
  exact CAD geometry (edges/faces/screw-hole axes) rather than a triangle soup.
- `media/UVC_cam_mount_so101.jpg` -- upstream's own installed-hardware photo (root `README.md`'s
  thumbnail for the "(SO101)" table cell above), full resolution, not a thumbnail crop. Shows the
  mount bolted to the top of the gripper (behind/above the fixed-jaw hinge), leaning forward, with
  the 32x32 module's lens visible at the top pointing forward and down over the jaws.

The camera-board mesh itself (`wrist_camera_so101_v1.stl`, the 32x32 UVC module) is **not**
duplicated here -- it already lives in `third_party/so101/assets/` (it ships in upstream's main
`Simulation/SO101/assets/` folder, in metres, already in the same convention as the rest of the
arm; see `../VENDORED.md`). Only the mount bracket, which is Optional-folder-only, is new here.

## How the camera pose was actually derived (short version; full derivation and uncertainty in
`docs/guide/03-robot-and-servo-model.md`)

Neither this STL nor the STEP file is used directly to compute
`WRIST_CAM_POS_IN_GRIPPER`/`WRIST_CAM_QUAT_IN_GRIPPER` by matching screw holes -- doing that blind,
from a bare mesh with no known reference frame, is exactly how the pose this work replaces (a
hand-picked position aimed at the TCP, filling 80-85% of the frame with the fixed jaw) was
produced. Upstream itself ships the answer: `Simulation/SO101/so101_new_calib_camera.xml` and
`.urdf` (same commit, alongside `so101_new_calib.xml`, which `patch_so101.py` already patches) are
a version of the arm **with this exact mount and camera already wired in**, chained as two fixed
(jointless) bodies off `gripper`. `patch_so101.py`'s constants are that chain, composed into one
gripper-relative pose and cross-checked by z-profiling `wrist_camera_so101_v1.stl` (to confirm
which local axis is the lens direction) and by rendering the composed pose both from outside (does
the assembly look like the photo above?) and from the camera's own eye (does the fixed jaw now
occupy a small, plausible share of frame instead of 80-85%?).

## Sanity check

Well-formedness of the STEP file was not machine-checked beyond a successful `cp`; the STL was read
successfully with `brick_builder.robot.patch_so101.read_binary_stl` (this project's own binary-STL
reader, already used for every other mesh in `third_party/so101/assets/`) as part of the pose
derivation, and its vertex count/bounding box were confirmed to match
`third_party/so101/assets/wrist_camera_mount_so101_v1.stl` (an earlier copy of this same part,
unwired into any MJCF body -- byte-for-byte identical geometry, confirmed by comparing vertex
arrays, not just file size).
