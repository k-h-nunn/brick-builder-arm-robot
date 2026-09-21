"""Tests for scripts/make_media.py.

Every test passes `--out-root`/`out_root=tmp_path` so nothing here ever writes into the real
`docs/media/` tree: most of it belongs to other packages (this package's own media path is
`docs/media/demos/**`), and several of those directories may be actively written by another
worker while this suite runs.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import pytest

from scripts import make_media


def test_eval_section_is_fast_and_writes_a_synthetic_heatmap(tmp_path: Path) -> None:
    status = make_media.run(["eval"], seed=0, run_all=False, out_root=tmp_path)
    assert status == {"eval": "ok"}
    assert (tmp_path / "docs/media/eval/example_heatmap.png").exists()


def test_control_section_writes_its_three_figures(tmp_path: Path) -> None:
    status = make_media.run(["control"], seed=0, run_all=False, out_root=tmp_path)
    assert status == {"control": "ok"}
    out_dir = tmp_path / "docs/media/control"
    for name in ("spiral_search.png", "press_recovery.png", "verify_tug.png"):
        assert (out_dir / name).exists(), name


def test_bricks_section_creates_its_own_output_directory(tmp_path: Path) -> None:
    """Regression check: brick_builder.bricks.media's figure functions do not mkdir their own parent
    (unlike most other figure functions in the codebase), which make_media.py has to do itself."""
    status = make_media.run(["bricks"], seed=0, run_all=False, out_root=tmp_path)
    assert status == {"bricks": "ok"}
    out_dir = tmp_path / "docs/media/bricks"
    for name in ("exploded_collision_vs_visual.png", "force_vs_depth.png", "rocking_vs_flat.png"):
        assert (out_dir / name).exists(), name


def test_unknown_section_raises() -> None:
    with pytest.raises(ValueError, match="unknown section"):
        make_media.run(["not-a-real-section"], seed=0, run_all=False, out_root=Path("/tmp"))


def test_a_failing_section_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:
    def _boom(seed, run_all, out_root):
        raise RuntimeError("boom")

    monkeypatch.setitem(make_media.SECTIONS, "scene", _boom)
    status = make_media.run(["scene"], seed=0, run_all=False, out_root=tmp_path)
    assert status == {"scene": "failed"}


def test_main_strict_flag_propagates_a_failure_to_the_exit_code(tmp_path: Path, monkeypatch) -> None:
    def _boom(seed, run_all, out_root):
        raise RuntimeError("boom")

    monkeypatch.setitem(make_media.SECTIONS, "scene", _boom)
    code = make_media.main(["--only", "scene", "--out-root", str(tmp_path), "--strict"])
    assert code == 1

    code_lenient = make_media.main(["--only", "scene", "--out-root", str(tmp_path)])
    assert code_lenient == 0


@pytest.mark.slow
def test_scene_and_render_sections(tmp_path: Path) -> None:
    status = make_media.run(["scene", "render"], seed=0, run_all=False, out_root=tmp_path)
    assert status == {"scene": "ok", "render": "ok"}
    assert (tmp_path / "docs/media/scene/overview.png").exists()
    assert (tmp_path / "docs/media/render/contact_sheet.png").exists()


@pytest.mark.slow
def test_demos_section_runs_every_demo(tmp_path: Path) -> None:
    status = make_media.run(["demos"], seed=0, run_all=False, out_root=tmp_path)
    assert status == {"demos": "ok"}
    out_dir = tmp_path / "docs/media/demos"
    written = {p.stem for p in out_dir.iterdir()}
    assert written == {"scene", "teleop", "press", "rocking", "force_map", "policy", "tokens"}
