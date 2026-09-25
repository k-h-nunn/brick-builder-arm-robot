"""Tests for `scripts/calibrate_camera.py`.

The pose-file round trip and argument parsing are fast (no MuJoCo); the `--simulate` end-to-end
run renders its own images and is `@pytest.mark.slow`.
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
    """The files `--simulate` writes are exactly what real mode reads (module docstring) --
    checked directly by feeding them back in."""
    out_dir = tmp_path / "sim"
    cc.main([
        "--simulate", "--out-dir", str(out_dir), "--out", str(tmp_path / "sim_fragment.yaml"),
        "--n-poses", "45", "--seed", "3", "--min-corners", "8",
    ])
    real_frag = tmp_path / "real_fragment.yaml"
    cc.main([
        "--images", str(out_dir / "images"), "--poses", str(out_dir / "poses.json"),
        "--out", str(real_frag),
    ])
    assert real_frag.exists()
