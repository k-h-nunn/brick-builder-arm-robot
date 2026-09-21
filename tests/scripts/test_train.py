"""Tests for scripts/train.py: config loading, `make_env_factory`, and one real (tiny) run."""

from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import pytest
import yaml

import brick_builder.envs  # noqa: F401
from scripts import train as train_script

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "train"


@pytest.mark.parametrize("name", ["reach.yaml", "press.yaml", "align.yaml"])
def test_shipped_configs_parse_and_build_a_ppo_config(name: str) -> None:
    raw = train_script.load_config(CONFIG_DIR / name)
    assert raw["env"].startswith("BrickBuilder/")
    cfg = train_script.build_ppo_config(raw, overrides={})
    assert cfg.total_steps > 0
    assert 1 <= cfg.n_envs <= 6, "the shared machine's own worker-process limit"


def test_load_config_rejects_a_config_with_no_env(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("ppo:\n  total_steps: 10\n")
    with pytest.raises(ValueError, match="env"):
        train_script.load_config(bad)


def test_build_ppo_config_requires_total_steps() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        train_script.build_ppo_config({"ppo": {}}, overrides={})


def test_overrides_win_over_the_config_file() -> None:
    raw = {"ppo": {"total_steps": 1000, "seed": 1, "n_envs": 8}}
    cfg = train_script.build_ppo_config(raw, overrides={"total_steps": 50, "seed": None, "n_envs": None})
    assert cfg.total_steps == 50   # overridden
    assert cfg.seed == 1           # untouched override (None) falls back to the file
    assert cfg.n_envs == 8


def test_make_env_factory_builds_the_requested_gym_env() -> None:
    make_env = train_script.make_env_factory("BrickBuilder/Reach-v0", {})
    env = make_env(seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"actor", "critic"}
    finally:
        env.close()


def test_train_end_to_end_writes_a_run_directory(tmp_path: Path) -> None:
    """One full pass through `main()`: tiny budget, `BrickBuilder/Reach-v0`, `SyncVectorEnv`."""
    config = tmp_path / "reach_tiny.yaml"
    config.write_text(
        yaml.safe_dump({
            "env": "BrickBuilder/Reach-v0",
            "env_kwargs": {},
            "run_name": "reach_tiny",
            "ppo": {"n_envs": 4, "n_steps": 32, "epochs": 2, "minibatches": 4, "seed": 0},
        })
    )
    run_dir = tmp_path / "run"
    ckpt = train_script.main([
        "--config", str(config), "--run-dir", str(run_dir), "--total-steps", str(4 * 32 * 2),
        "--sync-envs",
    ])

    assert ckpt == run_dir / "checkpoint.pt"
    assert ckpt.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "returns.json").exists()

    # 256 env steps across 4 environments is 64 ticks each, under Reach's 100-tick horizon, so
    # no episode finishes and there is no return to plot. The curve is written exactly when
    # there is something to draw, and not otherwise -- an empty axes is not an artefact.
    history = json.loads((run_dir / "returns.json").read_text())
    assert history == []
    assert not (run_dir / "learning_curve.png").exists()

    written_ppo_cfg = json.loads((run_dir / "ppo_config.json").read_text())
    assert written_ppo_cfg["total_steps"] == 4 * 32 * 2
    assert written_ppo_cfg["async_envs"] is False
