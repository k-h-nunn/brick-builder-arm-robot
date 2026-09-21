"""This package stands on its own.

The convolutional scene model is an ordinary, self-contained convolutional network trained on
this project's own simulator frames. It depends on nothing outside this project but the
dependencies `pyproject.toml` already declares, and nothing outside its own package but the
scene-model interface it publishes through. Another scene-model source plugs into that same
interface (`brick_builder.perception.sources`); this package never reaches for one.

The test reads the source text rather than importing, so a name mentioned inside a function body
or a lazily imported module is caught too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "brick_builder" / "perception" / "cnn"
TESTS = Path(__file__).resolve().parent

FORBIDDEN_MODULES = ("third_party",)
"""Vendored trees are data, not importable packages: nothing here may import one."""

FORBIDDEN_TEXT = ("third_party/",)
"""Nor reach into one by path."""


def _python_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py")) + sorted(TESTS.rglob("*.py"))


def test_there_is_something_to_check():
    assert len(_python_files()) >= 8


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_forbidden_import(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    for name in imported:
        for bad in FORBIDDEN_MODULES:
            assert not (name == bad or name.startswith(bad + ".")), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_forbidden_name_anywhere_in_the_text(path: Path):
    text = path.read_text(encoding="utf-8")
    if path == Path(__file__):
        return  # this file names them on purpose
    for bad in FORBIDDEN_TEXT:
        assert bad not in text, f"{path.name} mentions {bad!r}"


def test_another_scene_model_source_can_be_registered_without_this_package_knowing():
    """The seam this package is isolated behind: a source registered from outside is usable by
    name, and registering it touches nothing in `brick_builder.perception.cnn`."""
    from brick_builder.perception import sources
    from brick_builder.perception.scene_model import SceneModelSource

    class _Outside:
        def observe(self, rgb, *, window, instance=None, depth=None, present_mask=None, t=None):
            raise NotImplementedError

    sources.register("_test_outside", lambda **kw: _Outside())
    try:
        assert "_test_outside" in sources.available()
        assert isinstance(sources.get("_test_outside")(), SceneModelSource)
    finally:
        sources.unregister("_test_outside")


def test_the_package_imports_with_nothing_but_torch_numpy_and_this_project():
    """A relative import inside `brick_builder` is fine; an import of anything outside the project,
    beyond the dependencies the project already declares, is not."""
    allowed_roots = {
        "brick_builder", "numpy", "torch", "scipy", "cv2", "matplotlib", "mujoco", "json", "math",
        "time", "csv", "os", "re", "sys", "hashlib", "argparse", "dataclasses", "pathlib", "typing",
        "__future__", "collections", "itertools", "functools", "random", "warnings",
    }
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root in allowed_roots, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                assert root in allowed_roots, f"{path.name} imports from {node.module}"
