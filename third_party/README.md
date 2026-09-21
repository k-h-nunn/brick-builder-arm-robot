# third_party/

## third_party/so101/

`third_party/so101/` is vendored from `TheRobotStudio/SO-ARM100` (`Simulation/SO101`), licensed
Apache-2.0. `third_party/so101/VENDORED.md` records the exact source commit, the file list, and the
licence; `third_party/so101/UPSTREAM_README.md` is the upstream README as vendored. The wrist-camera
mount under `third_party/so101/wrist_camera_mount/` has its own `VENDORED.md` with its own source
and licence.

Tracked in git normally: it is redistributable third-party code.

Do not hand-edit the vendored files. `brick_builder/robot/patch_so101.py` applies every correction
this project needs programmatically, at load time, so the vendored tree stays byte-identical to
upstream and a future re-vendor is a plain copy. See `VENDORED.md`'s own warning.
