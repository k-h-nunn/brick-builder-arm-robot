"""Tests for scripts/eval.py, against a tiny (but real) BrickBuilder/Reach-v0 checkpoint."""

from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import numpy as np

import brick_builder.envs  # noqa: F401
from scripts import eval as eval_script


def test_run_flat_eval_writes_report_and_videos(tmp_path: Path, tiny_reach_checkpoint: Path) -> None:
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    results = eval_script.run_flat_eval(
        "BrickBuilder/Reach-v0", {}, tiny_reach_checkpoint, "torch", episodes=6, seed0=0,
        n_videos=2, out_dir=out_dir, fps=25,
    )
    assert len(results) == 6
    assert all(r.steps > 0 for r in results)

    report = json.loads((out_dir / "report.json").read_text())
    assert report["n_episodes"] == 6
    assert 0.0 <= report["overall"]["success_rate"] <= 1.0
    assert (out_dir / "report.md").exists()

    videos = sorted(out_dir.glob("episode_*.mp4"))
    assert len(videos) >= 2
    for v in videos:
        assert v.stat().st_size > 0


def test_c_actor_matches_torch_actor_outcomes(tmp_path: Path, tiny_reach_checkpoint: Path) -> None:
    """The exported C actor reproduces the torch actor's episode
    outcomes under the same seed. Checked here at the level scripts/eval.py actually cares
    about (per-episode success and length), not just raw weight parity (tests/export's job)."""
    out_torch = tmp_path / "torch_eval"
    out_c = tmp_path / "c_eval"
    out_torch.mkdir()
    out_c.mkdir()

    results_torch = eval_script.run_flat_eval(
        "BrickBuilder/Reach-v0", {}, tiny_reach_checkpoint, "torch", episodes=5, seed0=100,
        n_videos=0, out_dir=out_torch, fps=25,
    )
    results_c = eval_script.run_flat_eval(
        "BrickBuilder/Reach-v0", {}, tiny_reach_checkpoint, "c", episodes=5, seed0=100,
        n_videos=0, out_dir=out_c, fps=25,
    )

    assert [r.success for r in results_torch] == [r.success for r in results_c]
    assert [r.steps for r in results_torch] == [r.steps for r in results_c]


def test_press_sweep_writes_a_json_file_and_figures(tmp_path: Path, tiny_reach_checkpoint: Path) -> None:
    del tiny_reach_checkpoint  # this sweep needs a PressFromHeld checkpoint, trained below
    import gymnasium as gym

    from brick_builder.rl.ppo_asym import PPOConfig, train

    def make_press_env(seed: int):
        del seed
        return gym.make("BrickBuilder/PressFromHeld-v0", level=0, dr=False)

    press_ckpt = train(
        make_press_env,
        PPOConfig(n_envs=2, n_steps=16, epochs=1, minibatches=2, total_steps=2 * 16, seed=0, async_envs=False),
        tmp_path / "press_run",
    )

    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    results = eval_script.run_press_sweep(press_ckpt, {"level": 0, "dr": False}, episodes=2, workers=1, out_dir=out_dir)
    assert results
    assert all(r["n"] == 2 for r in results)
    assert (out_dir / "press_sweep.json").exists()
    assert (out_dir / "press_baseline_curve.png").exists()


def test_main_cli_end_to_end(tmp_path: Path, tiny_reach_checkpoint: Path) -> None:
    out_dir = tmp_path / "cli_eval"
    eval_script.main([
        "--checkpoint", str(tiny_reach_checkpoint), "--env", "BrickBuilder/Reach-v0",
        "--episodes", "3", "--videos", "1", "--out-dir", str(out_dir),
    ])
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()


def test_sweep_rejects_the_c_actor() -> None:
    import pytest

    with pytest.raises(SystemExit):
        eval_script.main([
            "--checkpoint", "does-not-matter.pt", "--env", "BrickBuilder/PressFromHeld-v0",
            "--sweep", "press-misalign", "--actor", "c",
        ])


def test_scripted_actor_needs_no_checkpoint(tmp_path: Path) -> None:
    """`--actor scripted` reaches the zero-residual baseline through the
    same `run_flat_eval` path a checkpoint evaluation uses, with `checkpoint=None`."""
    out_dir = tmp_path / "scripted_eval"
    out_dir.mkdir()
    results = eval_script.run_flat_eval(
        "BrickBuilder/Reach-v0", {}, None, "scripted", episodes=5, seed0=0,
        n_videos=0, out_dir=out_dir, fps=25,
    )
    assert len(results) == 5
    assert all(r.steps > 0 for r in results)
    report = json.loads((out_dir / "report.json").read_text())
    assert report["n_episodes"] == 5


def test_scripted_actor_reproduces_an_all_zero_rollout_on_reach() -> None:
    """Reach freezes grip and compliance, so the scripted baseline is literally the all-zero
    action -- check `_scripted_action_fn` actually produces that, not just "runs"."""
    import gymnasium as gym

    env_a = gym.make("BrickBuilder/Reach-v0")
    env_b = gym.make("BrickBuilder/Reach-v0")
    try:
        obs_a, info_a = env_a.reset(seed=7)
        obs_b, info_b = env_b.reset(seed=7)
        action_fn = eval_script._scripted_action_fn("BrickBuilder/Reach-v0")
        zero = np.zeros(env_a.unwrapped.action_space.shape, dtype=np.float32)
        for _ in range(20):
            a_scripted = action_fn(env_a.unwrapped)
            assert np.array_equal(a_scripted, zero)
            obs_a, r_a, t_a, tr_a, info_a = env_a.step(a_scripted)
            obs_b, r_b, t_b, tr_b, info_b = env_b.step(zero)
            assert np.array_equal(obs_a["actor"], obs_b["actor"])
            assert r_a == r_b
    finally:
        env_a.close()
        env_b.close()


def test_scripted_action_fn_uses_press_scripted_action_for_press() -> None:
    from brick_builder.envs.press_env import scripted_action

    assert eval_script._scripted_action_fn("BrickBuilder/PressFromHeld-v0") is scripted_action


def test_press_sweep_scripted_baseline_needs_no_checkpoint(tmp_path: Path) -> None:
    out_dir = tmp_path / "scripted_sweep"
    out_dir.mkdir()
    results = eval_script.run_press_sweep(None, {"level": 0, "dr": False}, episodes=2, workers=1, out_dir=out_dir)
    assert results
    assert all(r["n"] == 2 for r in results)
    assert all(r["learned"] is False for r in results)
    assert (out_dir / "press_sweep.json").exists()


def test_main_cli_requires_checkpoint_unless_scripted() -> None:
    import pytest

    with pytest.raises(SystemExit):
        eval_script.main(["--env", "BrickBuilder/Reach-v0", "--episodes", "1"])


def test_main_cli_scripted_actor(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_scripted"
    eval_script.main([
        "--env", "BrickBuilder/Reach-v0", "--actor", "scripted",
        "--episodes", "3", "--videos", "0", "--out-dir", str(out_dir),
    ])
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
