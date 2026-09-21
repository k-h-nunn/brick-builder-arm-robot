"""Tests for brick_builder/export/actor_to_c.py and c/policy.c, against spec section 5's
acceptance criterion: parity <= 1e-5 against torch on 10 000 random observations.

Uses the `pendulum_checkpoint` fixture (tests/export/conftest.py): a few PPO updates on the
Pendulum wrapper, per spec section 5's own instruction for testing export before a real
BrickBuilder checkpoint exists.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

from brick_builder.export.actor_to_c import load_actor_arrays, table_tanh_max_error, weight_bytes, write_header
from brick_builder.export.build import build_library
from brick_builder.export.c_policy import CPolicy
from brick_builder.export.parity import benchmark_inference, check_parity, plot_parity_and_speed, sample_observations


def test_write_header_matches_checkpoint_shapes(pendulum_checkpoint, tmp_path):
    header = write_header(pendulum_checkpoint, tmp_path)
    text = header.read_text()

    arrays = load_actor_arrays(pendulum_checkpoint)
    assert "#define POLICY_OBS_DIM 3" in text
    assert "#define POLICY_ACT_DIM 1" in text
    assert f'#define POLICY_CKPT_SHA256 "{arrays["digest"]}"' in text
    # One Linear layer per hidden layer plus the output layer: hidden=(64, 64) -> 3 layers.
    assert "#define POLICY_N_LAYERS 3" in text
    assert weight_bytes(arrays) == sum(w.size * 4 + b.size * 4 for w, b in arrays["layers"])


@pytest.fixture(scope="module")
def built_policies(pendulum_checkpoint, tmp_path_factory):
    """Both tanh variants, compiled once per test module (compiling gcc twice per test
    would be wasteful and the checkpoint does not change)."""
    build_dir_f = tmp_path_factory.mktemp("build_float")
    build_dir_t = tmp_path_factory.mktemp("build_table")
    lib_f, _ = build_library(pendulum_checkpoint, build_dir=build_dir_f, tanh_table=False)
    lib_t, _ = build_library(pendulum_checkpoint, build_dir=build_dir_t, tanh_table=True)
    cp_f = CPolicy(lib_f, obs_dim=3, act_dim=1)
    cp_t = CPolicy(lib_t, obs_dim=3, act_dim=1)
    return {"float": cp_f, "table": cp_t}


def test_build_produces_a_shared_library(built_policies):
    for cp in built_policies.values():
        obs = np.zeros(3, dtype=np.float32)
        act = cp.act(obs)
        assert act.shape == (1,)
        assert np.all(np.isfinite(act))


@pytest.mark.parametrize("variant", ["float", "table"])
def test_parity_within_1e_5_on_10000_observations(pendulum_checkpoint, built_policies, variant):
    """Spec section 5's acceptance criterion, verbatim: parity <= 1e-5 against torch on
    10 000 random observations, for both the float-tanhf and lookup-table builds."""
    report = check_parity(pendulum_checkpoint, built_policies[variant], n=10_000, seed=0)
    assert report.n == 10_000
    assert report.max_abs_error <= 1e-5, (
        f"{variant} tanh: max abs error {report.max_abs_error} exceeds the 1e-5 acceptance bound"
    )


def test_c_policy_matches_pattern_of_torch_evaluate(pendulum_checkpoint, built_policies):
    """A sanity check independent of check_parity's own machinery: actions must actually
    respect the action bounds (a bug that skipped the clamp would still often pass a loose
    parity check on in-range random observations)."""
    obs = sample_observations(pendulum_checkpoint, 200, seed=3)
    for cp in built_policies.values():
        acts = cp.act_batch(obs)
        assert np.all(acts >= -2.0 - 1e-6) and np.all(acts <= 2.0 + 1e-6)  # Pendulum-v1 action space


def test_table_tanh_error_and_speed(pendulum_checkpoint, built_policies, capsys):
    """Reports (does not just assert) the numbers spec section 5 asks for: parity, weight
    bytes and nanoseconds per inference for float vs. table tanh."""
    table_err = table_tanh_max_error()
    assert table_err < 1e-5, f"linear-interpolation tanh table error {table_err} is too coarse"

    arrays = load_actor_arrays(pendulum_checkpoint)
    n_bytes = weight_bytes(arrays)
    assert n_bytes > 0

    obs = sample_observations(pendulum_checkpoint, 1, seed=5)[0]
    ns_float = benchmark_inference(built_policies["float"], obs, n_calls=2000, n_warmup=200)
    ns_table = benchmark_inference(built_policies["table"], obs, n_calls=2000, n_warmup=200)
    assert ns_float > 0 and ns_table > 0

    print(f"\n[export] weight bytes: {n_bytes}")
    print(f"[export] table tanh max abs error vs math.tanh: {table_err:.3e}")
    print(f"[export] median ns/call (float tanhf): {ns_float:.0f}")
    print(f"[export] median ns/call (table tanh):  {ns_table:.0f}")


def test_plot_parity_and_speed_produces_a_figure(pendulum_checkpoint, built_policies, tmp_path):
    """Exercises the figure-producing function scripts/make_media.py will later call for
    this package. Small n so the test stays fast; the real figure
    under docs/media/export/ is regenerated with the default, larger n."""
    out = plot_parity_and_speed(
        pendulum_checkpoint, built_policies["float"], built_policies["table"],
        tmp_path / "fig.png", n_parity=200, n_calls=200,
    )
    assert out.exists()
    assert out.stat().st_size > 0
