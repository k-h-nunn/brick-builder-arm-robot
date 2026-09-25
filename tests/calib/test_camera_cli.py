"""Tests for `scripts/calibrate_camera.py`.

The pose-file round trip, argument parsing and the TCP-to-gripper-body pose conversion are fast
(a MuJoCo model compile, no rendering); the `--simulate` end-to-end run renders its own images and
is `@pytest.mark.slow`.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import numpy as np
import pytest

from scripts import calibrate_camera as cc


def test_parse_pattern():
    assert cc._parse_pattern("9x6") == (9, 6)
    assert cc._parse_pattern("7X5") == (7, 5)


def test_gripper_pose_from_tcp_pose_inverts_the_fixed_site_offset():
    """`ArmKinematics.fk`'s TCP-site pose, converted back to the `gripper` body's own pose, must
    exactly match the body's actual pose (`data.xpos`/`data.xmat`) -- not merely something in the
    right ballpark. The fixed offset is about 97 mm and 180 degrees at this pose, so a wrong sign
    or a forgotten conversion would be large, not subtle."""
    import mujoco

    from brick_builder.robot.kinematics import ArmKinematics
    from brick_builder.robot.patch_so101 import ArmConfig, GRIPPER_BODY, build_arm_spec

    model = build_arm_spec(ArmConfig()).compile()
    kin = ArmKinematics(model)
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    q = np.array([0.3, -0.4, 0.6, 0.5, -0.2])
    tcp_pos, tcp_rot = kin.fk(q)

    data = mujoco.MjData(model)
    for name, val in zip(joints, q):
        data.qpos[model.joint(name).qposadr[0]] = val
    mujoco.mj_forward(model, data)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    true_pos = data.xpos[gripper_id].copy()
    true_rot = data.xmat[gripper_id].reshape(3, 3).copy()

    rec_rot, rec_pos = cc.gripper_pose_from_tcp_pose(tcp_pos, tcp_rot)
    np.testing.assert_allclose(rec_pos, true_pos, atol=1e-9)
    np.testing.assert_allclose(rec_rot, true_rot, atol=1e-9)

    # The offset this inverts is not negligible -- prove the test would catch a forgotten
    # conversion (using the raw TCP pose as if it were the gripper body's).
    assert np.linalg.norm(tcp_pos - true_pos) > 0.05  # > 5 cm


def test_load_poses_round_trips_position_and_orientation(tmp_path: Path):
    from brick_builder.calib.camera import matrix_to_quat_wxyz, quat_wxyz_to_matrix

    rot = quat_wxyz_to_matrix(np.array([0.9, 0.1, 0.2, 0.3]))
    quat = matrix_to_quat_wxyz(rot)
    poses_path = tmp_path / "poses.json"
    poses_path.write_text(json.dumps({
        "shot001.png": {"position_m": [0.2, -0.01, 0.31], "quat_wxyz": [float(v) for v in quat]},
    }))

    loaded = cc._load_poses(poses_path)
    assert set(loaded.keys()) == {"shot001"}
    r_loaded, t_loaded = loaded["shot001"]
    np.testing.assert_allclose(t_loaded, [0.2, -0.01, 0.31])
    np.testing.assert_allclose(r_loaded, rot, atol=1e-9)


def test_load_images_and_poses_skips_images_with_no_pose_entry(tmp_path: Path, capsys):
    import cv2

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    for name in ("a.png", "b.png", "c.png", "d.png"):
        cv2.imwrite(str(images_dir / name), img)
    poses_path = tmp_path / "poses.json"
    poses_path.write_text(json.dumps({
        n: {"position_m": [0.0, 0.0, 0.3], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
        for n in ("a.png", "b.png", "c.png")  # "d.png" deliberately missing
    }))

    images, gripper_to_base, names = cc._load_images_and_poses(images_dir, poses_path)
    assert names == ["a.png", "b.png", "c.png"]
    assert len(images) == 3
    assert "no pose entry" in capsys.readouterr().err


def test_main_requires_images_and_poses_or_simulate():
    with pytest.raises(SystemExit):
        cc.main(["--out", "x.yaml"])


def test_main_simulate_requires_out_dir():
    with pytest.raises(SystemExit):
        cc.main(["--simulate", "--out", "x.yaml"])


def test_simulate_never_defaults_measured_by_to_the_real_user(monkeypatch):
    """A `--simulate` fragment's numbers are a synthetic round-trip, not anything a person
    measured -- its `measured_by` must never silently default to `$USER` the way real mode's
    does."""
    seen = {}
    monkeypatch.setattr(cc, "_cmd_simulate", lambda args: seen.update(measured_by=args.measured_by))
    monkeypatch.setenv("USER", "kurt")
    cc.main(["--simulate", "--out-dir", "d", "--out", "x.yaml"])
    assert seen["measured_by"] == "simulate"


def test_real_mode_defaults_measured_by_to_the_user(monkeypatch):
    seen = {}
    monkeypatch.setattr(cc, "_cmd_real", lambda args: seen.update(measured_by=args.measured_by))
    monkeypatch.setenv("USER", "kurt")
    cc.main(["--images", "i", "--poses", "p.json", "--out", "x.yaml"])
    assert seen["measured_by"] == "kurt"


def test_explicit_measured_by_is_never_overridden_for_simulate(monkeypatch):
    seen = {}
    monkeypatch.setattr(cc, "_cmd_simulate", lambda args: seen.update(measured_by=args.measured_by))
    cc.main(["--simulate", "--out-dir", "d", "--out", "x.yaml", "--measured-by", "alice"])
    assert seen["measured_by"] == "alice"


@pytest.mark.slow
def test_simulate_end_to_end_writes_images_poses_and_fragment(tmp_path: Path):
    out_dir = tmp_path / "sim"
    frag_path = tmp_path / "fragment.yaml"
    cc.main([
        "--simulate", "--out-dir", str(out_dir), "--out", str(frag_path),
        "--n-poses", "45", "--seed", "0", "--min-corners", "8",
    ])

    images = sorted((out_dir / "images").glob("*.png"))
    assert len(images) == 45
    poses = json.loads((out_dir / "poses.json").read_text())
    assert len(poses) == 45
    assert set(poses) == {p.name for p in images}

    assert frag_path.exists()
    yaml = pytest.importorskip("yaml")
    frag = yaml.safe_load(frag_path.read_text())
    assert "fx_px" in frag["camera"]
    assert "mount_x_mm" in frag["camera"]  # hand-eye ran too (>= 8 usable images)


@pytest.mark.slow
def test_simulate_then_real_mode_on_the_written_files_agree(tmp_path: Path):
    """The files `--simulate` writes are exactly what real mode reads (module docstring): feeding
    them back into real mode must reproduce the *same calibration numbers* `--simulate` itself
    reported, not just produce some fragment file that happens to exist."""
    yaml = pytest.importorskip("yaml")
    out_dir = tmp_path / "sim"
    sim_frag = tmp_path / "sim_fragment.yaml"
    cc.main([
        "--simulate", "--out-dir", str(out_dir), "--out", str(sim_frag),
        "--n-poses", "45", "--seed", "3", "--min-corners", "8",
    ])
    real_frag = tmp_path / "real_fragment.yaml"
    cc.main([
        "--images", str(out_dir / "images"), "--poses", str(out_dir / "poses.json"),
        "--out", str(real_frag), "--measured-by", "test",
    ])
    assert real_frag.exists()

    sim_cam = yaml.safe_load(sim_frag.read_text())["camera"]
    real_cam = yaml.safe_load(real_frag.read_text())["camera"]
    # rel=1e-3, not exact: both runs re-detect corners and re-run cv2.calibrateCamera/hand-eye
    # from scratch (real mode never reads sim's fitted numbers, only its images/poses -- module
    # docstring), and a multi-threaded BLAS solve is not bit-reproducible run to run. A few
    # thousandths agrees on the calibration and would catch a real format/frame-convention bug.
    for key in ("fx_px", "fy_px", "cx_px", "cy_px", "mount_x_mm", "mount_y_mm", "mount_z_mm"):
        assert real_cam[key]["value"] == pytest.approx(sim_cam[key]["value"], rel=1e-3), key
