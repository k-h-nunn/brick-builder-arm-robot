#!/usr/bin/env python
"""Wrist camera calibration: checkerboard images + arm poses in, a `configs/params/` YAML
fragment out.

Real-hardware mode (`docs/guide/08-hardware-wiring-calibration.md`, "Camera intrinsics"/"Camera
extrinsics"):

    .venv/bin/python scripts/calibrate_camera.py \\
        --images captures/ --poses captures/poses.json --out camera_measured.yaml

``--images`` is a folder of checkerboard photographs (any `cv2.imread`-readable format);
``--poses`` is a JSON file mapping each image's filename (with or without its extension) to the
**`gripper` body's** pose in the base frame at the moment it was shot --

    {"shot001.png": {"position_m": [0.21, -0.03, 0.31], "quat_wxyz": [0.98, 0.0, 0.0, 0.19]}, ...}

-- MuJoCo's `(w, x, y, z)` quaternion convention. This is `patch_so101.GRIPPER_BODY`'s own pose,
not `ArmKinematics.fk`'s return: `fk` reports the TCP site (`patch_so101.TCP_SITE`), offset from
the gripper body by a fixed, non-trivial transform (about 97 mm and a 180-degree rotation at a
typical pose) that hand-eye calibration would otherwise silently bake into the mount pose it
writes. A forward-kinematics reading that only gives the TCP pose -- real hardware or `fk` alike --
must be converted first with `gripper_pose_from_tcp_pose` below. ``--simulate`` writes gripper-body
poses directly (`camera_sim.py` reads them off the `gripper` body, never through `fk`), so it
doubles as the reference for what a real logging script should produce.

Dry-run / self-test mode, no hardware needed -- renders its own checkerboard set with MuJoCo at a
known `CameraConfig` and reports recovery error against that known truth, in addition to writing
the same fragment a real run would:

    .venv/bin/python scripts/calibrate_camera.py --simulate --out-dir sim_calib --out sim.yaml
    .venv/bin/python scripts/calibrate_camera.py --simulate --out-dir sim_calib --out sim.yaml \\
        --perturb-pos-mm 3 -2 1 --perturb-rot-deg 2 -1 0.5 --fov-deg 108

Either mode prints a 180-degree sensor-flip check (`brick_builder.calib.camera.check_sensor_flip`):
if it fires, the fix is rotating every input image 180 degrees (not editing the pose fragment) and
re-running -- see the printed message.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

_TCP_SITE_QUAT_IN_GRIPPER = np.array([0.0, 1.0, 0.0, 0.0])
"""`patch_so101._add_sites_and_camera`'s own TCP-site quaternion (180 degrees about x) -- the
fixed rotation part of the gripper-body-to-TCP-site transform this module's `fk`-to-gripper-body
conversion inverts."""


def gripper_pose_from_tcp_pose(tcp_pos: np.ndarray, tcp_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` of the `gripper` body in the base frame, from a TCP-site pose in the same frame
    (`ArmKinematics.fk`'s return, or an equivalent real-hardware forward-kinematics reading).

    `patch_so101.TCP_SITE` sits at a fixed, known offset from `patch_so101.GRIPPER_BODY`
    (`patch_so101.tcp_position()`, plus `_TCP_SITE_QUAT_IN_GRIPPER`'s 180-degree-about-x
    rotation) -- this inverts that fixed transform. Hand-eye calibration needs the gripper body's
    own pose (what `camera_sim.py` reads directly off `data.xpos`/`data.xmat[gripper_id]`, and what
    `--simulate` writes to `poses.json`), not the TCP site's: composing camera-to-gripper with a
    TCP-site pose instead would silently write a mount pose about 97 mm and 180 degrees off (the
    site's own offset and its 180-degree rotation relative to the body it sits on).
    """
    from brick_builder.calib.camera import quat_wxyz_to_matrix
    from brick_builder.robot.patch_so101 import tcp_position

    r_site_in_gripper = quat_wxyz_to_matrix(_TCP_SITE_QUAT_IN_GRIPPER)
    t_site_in_gripper = tcp_position()
    tcp_rot = np.asarray(tcp_rot, dtype=float)
    r_gripper = tcp_rot @ r_site_in_gripper.T
    t_gripper = np.asarray(tcp_pos, dtype=float) - r_gripper @ t_site_in_gripper
    return r_gripper, t_gripper


def _load_poses(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{stem: (R, t)}``, MuJoCo convention, from a ``poses.json`` (see the module docstring)."""
    from brick_builder.calib.camera import quat_wxyz_to_matrix

    raw = json.loads(path.read_text())
    poses = {}
    for key, entry in raw.items():
        stem = Path(key).stem
        pos = np.asarray(entry["position_m"], dtype=float)
        rot = quat_wxyz_to_matrix(np.asarray(entry["quat_wxyz"], dtype=float))
        poses[stem] = (rot, pos)
    return poses


def _load_images_and_poses(images_dir: Path, poses_path: Path):
    poses = _load_poses(poses_path)
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"no images ({', '.join(_IMAGE_EXTS)}) found in {images_dir}")

    images, gripper_to_base, used_names, missing = [], [], [], []
    for p in image_paths:
        if p.stem not in poses:
            missing.append(p.name)
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            missing.append(p.name)
            continue
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        gripper_to_base.append(poses[p.stem])
        used_names.append(p.name)
    if missing:
        print(f"skipped {len(missing)} image(s) with no pose entry or unreadable: {missing}", file=sys.stderr)
    if len(images) < 3:
        raise SystemExit(f"only {len(images)} usable (image, pose) pair(s); need at least 3")
    return images, gripper_to_base, used_names


def _run_calibration(images, gripper_to_base, target, hand_eye_method, expected_quat_wxyz):
    from brick_builder.calib import camera as calib
    import cv2 as _cv2

    intrinsics = calib.calibrate_intrinsics(images, target=target)
    target_to_cam, g2b_used = [], []
    for idx, rvec, tvec in zip(intrinsics.used_image_indices, intrinsics.rvecs, intrinsics.tvecs):
        r_t2c, _ = _cv2.Rodrigues(rvec)
        target_to_cam.append((r_t2c, tvec))
        g2b_used.append(gripper_to_base[idx])
    hand_eye = calib.calibrate_hand_eye(g2b_used, target_to_cam, method=hand_eye_method)
    flip = calib.check_sensor_flip(hand_eye.rotation_mujoco(), expected_quat_wxyz)
    return intrinsics, hand_eye, flip


def _print_summary(intrinsics, hand_eye, flip) -> None:
    print(f"intrinsics: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} cx={intrinsics.cx:.2f} "
          f"cy={intrinsics.cy:.2f} px, RMS reprojection {intrinsics.rms_reprojection_px:.4f} px "
          f"({len(intrinsics.used_image_indices)} image(s) used)")
    print(f"distortion (OpenCV): k1={intrinsics.k1:.5f} k2={intrinsics.k2:.5f}")
    t_mm = hand_eye.translation_m_mujoco() * 1000.0
    print(f"hand-eye ({hand_eye.solver_description}, {hand_eye.n_poses} pose(s), MuJoCo convention): "
          f"position [{t_mm[0]:.2f}, {t_mm[1]:.2f}, {t_mm[2]:.2f}] mm")
    print(f"180-degree sensor-flip check: angle to expected pose "
          f"{flip.angle_to_expected_deg:.1f} deg, to expected-flipped {flip.angle_to_expected_flipped_deg:.1f} deg")
    if flip.likely_flipped:
        print(
            "  ** LIKELY 180-DEGREE FLIP ** -- the calibrated orientation matches the expected "
            "mount pose rotated 180 degrees about the camera's own optical axis far better than "
            "it matches the expected pose directly. Rotate every input image 180 degrees "
            "(brick_builder.calib.camera.rotated_180) and re-run rather than trusting this pose.",
            file=sys.stderr,
        )


def _write_fragment(path: Path, intrinsics, hand_eye, measured_by: str) -> None:
    from brick_builder.calib.camera import parameter_fragment
    import datetime

    text = parameter_fragment(
        intrinsics, hand_eye, measured_by=measured_by,
        measured_on=datetime.date.today().isoformat(),
    )
    path.write_text(text)
    print(f"wrote {path}")


def _cmd_real(args: argparse.Namespace) -> None:
    from brick_builder.calib.camera import CheckerboardTarget
    from brick_builder.robot.patch_so101 import WRIST_CAM_QUAT_IN_GRIPPER

    target = CheckerboardTarget(pattern=_parse_pattern(args.pattern), square_m=args.square_mm / 1000.0)
    images, gripper_to_base, _names = _load_images_and_poses(Path(args.images), Path(args.poses))
    intrinsics, hand_eye, flip = _run_calibration(
        images, gripper_to_base, target, args.hand_eye_method, WRIST_CAM_QUAT_IN_GRIPPER,
    )
    _print_summary(intrinsics, hand_eye, flip)
    _write_fragment(Path(args.out), intrinsics, hand_eye, measured_by=args.measured_by)


def _cmd_simulate(args: argparse.Namespace) -> None:
    from brick_builder.calib import camera as calib
    from brick_builder.calib import camera_sim as sim
    from brick_builder.robot.patch_so101 import CameraConfig

    base = CameraConfig(diagonal_fov_deg=args.fov_deg, resolution=tuple(args.resolution))
    pos = np.array(base.pos_in_gripper) + np.array(args.perturb_pos_mm) / 1000.0
    if any(args.perturb_rot_deg):
        wiggle = calib.quat_wxyz_to_matrix(
            np.array([1.0, *(np.radians(args.perturb_rot_deg) / 2.0)])  # small-angle quat approx
        )
        quat = calib.matrix_to_quat_wxyz(calib.quat_wxyz_to_matrix(np.array(base.quat_in_gripper)) @ wiggle)
    else:
        quat = np.array(base.quat_in_gripper)
    camera_cfg = dataclasses.replace(base, pos_in_gripper=tuple(pos), quat_in_gripper=tuple(quat))

    print(f"simulating {args.n_poses} checkerboard shots (seed {args.seed}) at:")
    print(f"  diagonal fov {camera_cfg.diagonal_fov_deg:.1f} deg, resolution {camera_cfg.resolution}")
    print(f"  pose (mm) {tuple(round(float(v) * 1000, 3) for v in camera_cfg.pos_in_gripper)}, "
          f"quat_wxyz {tuple(round(float(v), 5) for v in camera_cfg.quat_in_gripper)}")

    calib_set = sim.render_calibration_set(camera_cfg=camera_cfg, n_poses=args.n_poses, seed=args.seed)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    poses = {}
    for i, shot in enumerate(calib_set.shots):
        name = f"sim_{i:03d}.png"
        cv2.imwrite(str(images_dir / name), cv2.cvtColor(shot.rgb, cv2.COLOR_RGB2BGR))
        rot, pos_i = shot.gripper_to_base
        poses[name] = {
            "position_m": [float(v) for v in pos_i],
            "quat_wxyz": [float(v) for v in calib.matrix_to_quat_wxyz(rot)],
        }
    poses_path = out_dir / "poses.json"
    poses_path.write_text(json.dumps(poses, indent=2))
    print(f"wrote {len(calib_set.shots)} image(s) + {poses_path}")

    intrinsics, hand_eye = sim.calibrate_from_calibration_set(
        calib_set, hand_eye_method=args.hand_eye_method, min_corners_found=args.min_corners,
    )
    flip = calib.check_sensor_flip(hand_eye.rotation_mujoco(), np.array(camera_cfg.quat_in_gripper))
    _print_summary(intrinsics, hand_eye, flip)

    report = sim.report_from_calibration(camera_cfg, intrinsics, hand_eye, args.n_poses, label="simulate")
    print(
        f"recovery vs. known ground truth: reprojection {report.reprojection_rms_px:.4f} px, "
        f"focal error {report.focal_error_pct:.3f}%, position error {report.position_error_mm:.3f} mm, "
        f"rotation error {report.rotation_error_deg:.4f} deg"
    )
    _write_fragment(Path(args.out), intrinsics, hand_eye, measured_by=args.measured_by)


def _parse_pattern(text: str) -> tuple[int, int]:
    cols, rows = text.lower().split("x")
    return int(cols), int(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default="9x6", help="checkerboard internal corners, COLSxROWS (default 9x6)")
    parser.add_argument("--square-mm", type=float, default=15.0, help="checkerboard square size [mm] (default 15)")
    parser.add_argument("--hand-eye-method", default="tsai",
                         choices=("tsai", "park", "horaud", "andreff", "daniilidis"))
    parser.add_argument("--out", required=True, help="output parameter-file fragment (YAML)")
    parser.add_argument("--measured-by", default=None,
                         help="parameter-file 'measured_by' provenance field (default: $USER for a "
                              "real run; 'simulate', never a person's name, for --simulate)")

    parser.add_argument("--simulate", action="store_true", help="render a synthetic checkerboard set instead of reading photographs")
    parser.add_argument("--images", help="folder of checkerboard photographs (real mode)")
    parser.add_argument("--poses", help="poses.json for --images (real mode)")

    parser.add_argument("--out-dir", help="where --simulate writes images/ and poses.json")
    parser.add_argument("--n-poses", type=int, default=90,
                         help="--simulate: arm poses to render (default 90, with headroom above the "
                              "--min-corners default of 15: the tilted, close-hover views a real "
                              "hand-eye calibration needs typically only yield a usable checkerboard "
                              "detection in roughly a quarter to a third of shots -- see "
                              "brick_builder.calib.camera_sim.render_calibration_set)")
    parser.add_argument("--seed", type=int, default=0, help="--simulate: RNG seed for pose sampling")
    parser.add_argument("--fov-deg", type=float, default=115.0, help="--simulate: ground-truth diagonal fov")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    parser.add_argument("--perturb-pos-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--perturb-rot-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("RX", "RY", "RZ"))
    parser.add_argument("--min-corners", type=int, default=15, help="--simulate: minimum usable renders required")

    args = parser.parse_args(argv)

    if args.simulate:
        if not args.out_dir:
            parser.error("--simulate requires --out-dir")
        if args.measured_by is None:
            # Never default to $USER here: a --simulate fragment's numbers are a synthetic
            # self-test round-trip, not anything a person measured, and the file's own
            # `measured_by` field is the only thing telling a later reader that.
            args.measured_by = "simulate"
        _cmd_simulate(args)
    else:
        if not (args.images and args.poses):
            parser.error("real mode requires --images and --poses (or pass --simulate)")
        if args.measured_by is None:
            args.measured_by = os.environ.get("USER", "unknown")
        _cmd_real(args)


if __name__ == "__main__":
    main()
