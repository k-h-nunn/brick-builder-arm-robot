#!/usr/bin/env python
"""Train a BrickBuilder skill: one command, one YAML config.

    .venv/bin/python scripts/train.py --config configs/train/reach.yaml
    .venv/bin/python scripts/train.py --config configs/train/reach_sac.yaml

Reads a config (`env`: a registered gymnasium id; `env_kwargs`: passed to `gym.make`; `algo`:
`"ppo"` (default) or `"sac"`, selecting `brick_builder.rl.ppo_asym.PPOConfig`/`.train` or
`brick_builder.rl.sac_asym.SACConfig`/`.train`; `ppo:`/`sac:` -- whichever field matches `algo` --
supplies that dataclass's fields), builds the vector env and hands it to the selected
algorithm's `train`. Everything it produces lands in `runs/<run_name>/` (default `--run-dir
runs/<run_name>`, where `run_name` is the config's own `run_name` field, or the config file's
stem if it does not set one):

- `config.yaml` -- an exact copy of the config this run actually used,
- `ppo_config.json` / `sac_config.json` -- the resolved `PPOConfig`/`SACConfig` (after any
  `--total-steps`/`--seed`/`--n-envs` overrides), for a quick "what was this run's config" read
  without parsing YAML,
- `tb/` -- a TensorBoard event log (`train.py` writes nothing here directly; the algorithm's
  own `train` does),
- `checkpoint.pt` -- the trained actor (+ critics + normalizers) (`train`'s own output; both
  algorithms share the same `actor`/`norm_actor`/`cfg`/`obs_spec` checkpoint keys, so
  `scripts/eval.py`'s torch/C actor loading works unchanged for either),
- `returns.json` -- `(global_step, mean_return)` pairs (`train`'s own output),
- `learning_curve.png` -- `plot_learning_curve` applied to the above (the same function for
  either algorithm; `brick_builder.rl.sac_asym.plot_learning_curve` re-exports
  `brick_builder.rl.ppo_asym.plot_learning_curve` since the plot itself is algorithm-agnostic),
CPU only, per project policy: the GPU may be busy with someone else's job, so
`CUDA_VISIBLE_DEVICES` is cleared before `torch` (imported transitively by `brick_builder.rl.ppo_asym`)
is ever imported in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium as gym
import yaml

import brick_builder.envs  # noqa: F401 -- registers BrickBuilder/* gymnasium ids
from brick_builder.rl.ppo_asym import PPOConfig, plot_learning_curve, train
from brick_builder.rl.sac_asym import SACConfig
from brick_builder.rl.sac_asym import train as sac_train

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    """Read and lightly validate one `configs/train/*.yaml` file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "env" not in raw:
        raise ValueError(f"{path}: config must be a mapping with at least an 'env' key")
    return raw


def make_env_factory(env_id: str, env_kwargs: dict):
    """`make_env(seed) -> gym.Env`, the closure `ppo_asym.train`'s vector env wants.

    `seed` is unused at construction: every BrickBuilder task only consumes a construction-time seed
    to build its initial, un-randomized scene geometry (default randomization ranges are zero, so
    this has no effect on any config here); the per-episode randomness a training run actually
    depends on comes from `env.reset(seed=...)`, which `ppo_asym.train` itself calls once per
    sub-environment at the start of training.
    """

    def make_env(seed: int) -> gym.Env:
        del seed
        return gym.make(env_id, **env_kwargs)

    return make_env


def build_ppo_config(raw: dict, overrides: dict) -> PPOConfig:
    kwargs = dict(raw.get("ppo", {}))
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    if "total_steps" not in kwargs:
        raise ValueError("config's 'ppo' section (or --total-steps) must set total_steps")
    return PPOConfig(**kwargs)


def build_sac_config(raw: dict, overrides: dict) -> SACConfig:
    """`algo: sac` counterpart of `build_ppo_config`: same override semantics, reading the
    config's `sac:` section instead of `ppo:`."""
    kwargs = dict(raw.get("sac", {}))
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    if "total_steps" not in kwargs:
        raise ValueError("config's 'sac' section (or --total-steps) must set total_steps")
    return SACConfig(**kwargs)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="e.g. configs/train/reach.yaml")
    parser.add_argument("--run-dir", type=Path, default=None, help="default: runs/<run_name>")
    parser.add_argument("--total-steps", type=int, default=None, help="override <algo>.total_steps")
    parser.add_argument("--seed", type=int, default=None, help="override <algo>.seed")
    parser.add_argument("--n-envs", type=int, default=None, help="override <algo>.n_envs")
    parser.add_argument(
        "--sync-envs", action="store_true",
        help="force SyncVectorEnv (<algo>.async_envs=False); used by the test suite so pytest is "
        "not left spawning subprocesses",
    )
    args = parser.parse_args(argv)

    raw = load_config(args.config)
    env_id = raw["env"]
    env_kwargs = dict(raw.get("env_kwargs", {}))
    run_name = raw.get("run_name", args.config.stem)
    run_dir = args.run_dir if args.run_dir is not None else REPO_ROOT / "runs" / run_name
    algo = raw.get("algo", "ppo")
    if algo not in ("ppo", "sac"):
        raise ValueError(f"{args.config}: 'algo' must be 'ppo' or 'sac', got {algo!r}")

    overrides: dict = {"total_steps": args.total_steps, "seed": args.seed, "n_envs": args.n_envs}
    if args.sync_envs:
        overrides["async_envs"] = False

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    make_env = make_env_factory(env_id, env_kwargs)

    if algo == "sac":
        sac_cfg = build_sac_config(raw, overrides)
        (run_dir / "sac_config.json").write_text(json.dumps(asdict(sac_cfg), indent=1) + "\n")

        def log(update: int, metrics: dict) -> None:
            if update == 0 or update % 100 == 0:
                mean_return = metrics.get("mean_return", float("nan"))
                print(
                    f"[{run_name}] update {update:6d}  step {int(metrics['global_step']):>9d}  "
                    f"return {mean_return:+9.3f}  critic_loss {metrics.get('critic_loss', float('nan')):.4f}  "
                    f"alpha {metrics.get('alpha', float('nan')):.4f}"
                )

        print(
            f"training {env_id} -> {run_dir}  (algo=sac, "
            f"{sac_cfg.total_steps} steps, n_envs={sac_cfg.n_envs}, async={sac_cfg.async_envs})"
        )
        ckpt_path = sac_train(make_env, sac_cfg, run_dir, callback=log)
    else:
        ppo_cfg = build_ppo_config(raw, overrides)
        (run_dir / "ppo_config.json").write_text(json.dumps(asdict(ppo_cfg), indent=1) + "\n")

        def log(update: int, metrics: dict) -> None:
            if update == 0 or update % 10 == 0:
                print(
                    f"[{run_name}] update {update:5d}  step {int(metrics['global_step']):>9d}  "
                    f"return {metrics['mean_return']:+9.3f}  kl {metrics['approx_kl']:.4f}"
                )

        print(
            f"training {env_id} -> {run_dir}  (algo=ppo, "
            f"{ppo_cfg.total_steps} steps, n_envs={ppo_cfg.n_envs}, async={ppo_cfg.async_envs})"
        )
        ckpt_path = train(make_env, ppo_cfg, run_dir, callback=log)

    history = json.loads((run_dir / "returns.json").read_text())
    if history:
        plot_learning_curve(history, run_dir / "learning_curve.png", title=f"{run_name} learning curve")

    print(f"wrote {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    main()
