"""The training loop's own options: the shipped-weights gate, extra-split mixing, fine-tuning
from a snapshot, and the photometric-augmentation identity at strength 1.0."""

from __future__ import annotations

import numpy as np
import torch

from brick_builder.perception.cnn import data as D
from brick_builder.perception.cnn.model import SceneModelNet, pack_weights, unpack_weights
from brick_builder.perception.cnn.train import (
    TrainConfig,
    WEIGHTS_DIR,
    WEIGHTS_NAME,
    _augment,
    train,
)


def _write_split(cache, split: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        instance = np.zeros((256, 256), dtype=np.uint8)
        depth = np.full((256, 256), 0.2, dtype=np.float16)
        samples.append(D.Sample(rgb=rgb, instance=instance, depth=depth, meta={"kind": "test"}))
    D.write_shard(cache / split / "shard_0001.npz", samples)


def _tiny_cfg(cache, out, **overrides) -> TrainConfig:
    kwargs = dict(
        cache=str(cache), out=str(out), minutes=60.0, epochs=1, batch_size=2, lr=3e-3,
        threads=1, crop=0, steps_per_epoch=1, val_frames=1, device="cpu",
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


# --------------------------------------------------------------------- photometric strength ---


def _replicate_jitter(seed: int, strength: float, shape=(4, 4, 3)):
    """Redraws the same rng sequence `_augment` consumes (flips, gain, brightness, contrast,
    noise) and returns the image `_augment` should produce for a mid-grey input, so the
    fixed ranges the docstring promises (brightness +/-0.06, contrast 0.88-1.14, gain +/-0.10,
    noise std 0.012 at strength 1.0) are pinned down rather than merely described."""
    rng = np.random.default_rng(seed)
    rng.random()  # horizontal-flip draw; a constant image is unaffected either way
    rng.random()  # vertical-flip draw
    gain = 1.0 + rng.uniform(-0.10 * strength, 0.10 * strength, size=3).astype(np.float32)
    brightness = float(rng.uniform(-0.06 * strength, 0.06 * strength))
    contrast = float(rng.uniform(1.0 - 0.12 * strength, 1.0 + 0.14 * strength))
    img = np.full(shape, 128.0 / 255.0, dtype=np.float32)  # rgb fixture is a constant 128
    img = (img - 0.5) * contrast + 0.5 + brightness
    img = np.clip(img * gain, 0.0, 1.0)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0.0, 0.012 * strength, size=shape).astype(np.float32), 0.0, 1.0)
    return img


def test_augment_strength_1_reproduces_the_documented_fixed_ranges():
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    instance = np.zeros((4, 4), dtype=np.uint8)
    depth = np.full((4, 4), 0.2, dtype=np.float16)
    for seed in range(8):
        expected = _replicate_jitter(seed, strength=1.0, shape=(4, 4, 3))
        actual, _, _ = _augment(rgb, instance, depth, np.random.default_rng(seed), crop=0, strength=1.0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_augment_strength_scales_the_jitter_half_widths():
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    instance = np.zeros((4, 4), dtype=np.uint8)
    depth = np.full((4, 4), 0.2, dtype=np.float16)
    for seed in range(8):
        expected = _replicate_jitter(seed, strength=2.0, shape=(4, 4, 3))
        actual, _, _ = _augment(rgb, instance, depth, np.random.default_rng(seed), crop=0, strength=2.0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


# ------------------------------------------------------------------------------ save_shipped ---


def test_save_shipped_false_never_touches_the_shipped_weights_file(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    shipped = weights_dir / WEIGHTS_NAME
    shipped.write_bytes(b"unchanged")
    monkeypatch.setattr("brick_builder.perception.cnn.train.WEIGHTS_DIR", weights_dir)

    cache = tmp_path / "cache"
    _write_split(cache, "train", 4, seed=1)
    _write_split(cache, "val", 2, seed=2)
    out = tmp_path / "out"
    cfg = _tiny_cfg(cache, out, save_shipped=False)

    train(cfg, verbose=False)

    assert shipped.read_bytes() == b"unchanged", "save_shipped=False must not write the shipped file"
    assert (out / "best.pt").exists(), "the per-run checkpoint must still be written"


def test_save_shipped_true_writes_the_shipped_weights_file(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setattr("brick_builder.perception.cnn.train.WEIGHTS_DIR", weights_dir)

    cache = tmp_path / "cache"
    _write_split(cache, "train", 4, seed=1)
    _write_split(cache, "val", 2, seed=2)
    out = tmp_path / "out"
    cfg = _tiny_cfg(cache, out, save_shipped=True)

    summary = train(cfg, verbose=False)

    assert (weights_dir / WEIGHTS_NAME).exists()
    assert summary["weights"] == str(weights_dir / WEIGHTS_NAME)


# -------------------------------------------------------------------------------- extra_splits ---


def test_extra_splits_are_concatenated_into_the_training_set(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("brick_builder.perception.cnn.train.WEIGHTS_DIR", tmp_path / "weights")
    cache = tmp_path / "cache"
    _write_split(cache, "train", 2, seed=1)
    _write_split(cache, "extra", 3, seed=2)
    _write_split(cache, "val", 1, seed=3)
    out = tmp_path / "out"
    cfg = _tiny_cfg(cache, out, save_shipped=False, extra_splits=("extra",))

    train(cfg, verbose=True)

    out_text = capsys.readouterr().out
    assert "train frames 5  val frames 1" in out_text, out_text


# -------------------------------------------------------------------------------- init_weights ---


def test_init_weights_is_loaded_when_the_run_starts_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr("brick_builder.perception.cnn.train.WEIGHTS_DIR", tmp_path / "weights")
    cache = tmp_path / "cache"
    _write_split(cache, "train", 4, seed=1)
    _write_split(cache, "val", 2, seed=2)

    torch.manual_seed(0)
    init_model = SceneModelNet(width=1.0)
    init_path = tmp_path / "init.pt"
    torch.save({"model": pack_weights(init_model.state_dict())}, init_path)

    out = tmp_path / "out"
    # lr=0.0 makes AdamW's update exactly zero every step, so one epoch from `init_weights`
    # must leave the checkpoint bit-for-bit equal to the snapshot it started from -- direct
    # evidence the snapshot was actually loaded, not just that training ran without error.
    cfg = _tiny_cfg(cache, out, save_shipped=False, init_weights=str(init_path), lr=0.0)

    train(cfg, verbose=False)

    ckpt = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    trained_state = ckpt["model"]
    init_state = unpack_weights(pack_weights(init_model.state_dict()))
    # Only the trainable parameters: at lr=0.0 the optimizer makes an exactly-zero update to
    # these, but batch-norm running_mean/running_var are buffers that the forward pass alone
    # updates by exponential average, on every call, with no dependence on lr.
    param_names = {name for name, _ in init_model.named_parameters()}
    assert param_names, "the model must have at least one trainable parameter to check"
    for name in param_names:
        torch.testing.assert_close(trained_state[name], init_state[name], atol=0.0, rtol=0.0)
