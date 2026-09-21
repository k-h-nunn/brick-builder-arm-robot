"""Tests for `python -m brick_builder.demo <name>`.

The acceptance criterion: each demo runs headless with `--record` and writes
a playable mp4/PNG. Every demo here is called directly through `module.run(...)` (what
`brick_builder.demo.main` itself calls after argument parsing) with a tiny/default scenario and a
`tmp_path` output file, and the file is checked for existing and being non-empty; `force-map` is
marked slow (a multi-point IK force-map sweep takes tens of seconds).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import pytest

import brick_builder.demo as demo_pkg
import brick_builder.demo.force_map_demo as force_map_demo
import brick_builder.demo.policy_demo as policy_demo
import brick_builder.demo.press_demo as press_demo
import brick_builder.demo.rocking_demo as rocking_demo
import brick_builder.demo.scene_demo as scene_demo
import brick_builder.demo.teleop_demo as teleop_demo
import brick_builder.demo.tokens_demo as tokens_demo


def _assert_playable(path: Path) -> None:
    assert path.exists(), path
    assert path.stat().st_size > 0


def test_registry_lists_every_spec_demo() -> None:
    assert set(demo_pkg.DEMOS) == {
        "scene", "teleop", "press", "rocking", "force-map", "policy", "tokens",
    }


def test_scene_demo_writes_a_png_by_default(tmp_path: Path) -> None:
    out = tmp_path / "scene.png"
    result = scene_demo.run(record=str(out), seed=0, extra=())
    assert result == out
    _assert_playable(out)


def test_scene_demo_writes_a_video_when_asked(tmp_path: Path) -> None:
    out = tmp_path / "scene.mp4"
    scene_demo.run(record=str(out), seed=0, extra=["--seconds", "0.5"])
    _assert_playable(out)


def test_teleop_demo_falls_back_to_the_scripted_jog(tmp_path: Path) -> None:
    out = tmp_path / "teleop.mp4"
    teleop_demo.run(record=str(out), seed=0, extra=["--seconds", "0.5"])
    _assert_playable(out)


def test_press_demo_scripted_baseline_seats_a_2x4_at_level_0(tmp_path: Path) -> None:
    out = tmp_path / "press.mp4"
    press_demo.run(record=str(out), seed=0, extra=["--level", "0"])
    _assert_playable(out)


def test_rocking_demo_runs(tmp_path: Path) -> None:
    out = tmp_path / "rocking.mp4"
    rocking_demo.run(record=str(out), seed=0, extra=["--level", "1"])
    _assert_playable(out)


def test_policy_demo_plays_a_checkpoint(tmp_path: Path, tiny_reach_checkpoint: Path) -> None:
    out = tmp_path / "policy.mp4"
    policy_demo.run(record=str(out), seed=0, extra=["--checkpoint", str(tiny_reach_checkpoint)])
    _assert_playable(out)


def test_policy_demo_complains_about_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        policy_demo.run(record=str(tmp_path / "x.mp4"), seed=0,
                         extra=["--checkpoint", str(tmp_path / "does_not_exist.pt")])


def test_tokens_demo_overlays_the_surrogate_producer(tmp_path: Path) -> None:
    out = tmp_path / "tokens.mp4"
    tokens_demo.run(record=str(out), seed=0, extra=["--producer", "surrogate"])
    _assert_playable(out)


def test_main_dispatches_by_name(tmp_path: Path) -> None:
    out = tmp_path / "scene_via_main.png"
    demo_pkg.main(["scene", "--record", str(out), "--seed", "1"])
    _assert_playable(out)


@pytest.mark.slow
def test_force_map_demo_writes_a_heatmap(tmp_path: Path) -> None:
    out = tmp_path / "force_map.png"
    force_map_demo.run(record=str(out), seed=0, extra=())
    _assert_playable(out)
