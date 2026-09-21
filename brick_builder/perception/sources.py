"""Registry of scene-model sources, keyed by name.

A scene-model source is anything satisfying `scene_model.SceneModelSource`: one method,
`observe(rgb, *, window, instance=None, depth=None, present_mask=None, t=None) -> SceneState`.
The sources that ship with this package register themselves here at import time, under the names
`ideal`, `colour` and `learned`.

A source that lives in another package registers itself the same way, and nothing in this package
needs to know it exists:

    from brick_builder.perception import sources

    sources.register("my_model", lambda **kw: MyModelSceneModel(**kw))
    source = sources.get("my_model")(device="cpu")

`register` takes a factory, not an instance, so a source whose construction is expensive (loading
a checkpoint) is not built until something asks for it. A name may be registered only once unless
`replace=True` is passed, so two packages claiming the same name is an error rather than a silent
last-one-wins.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .scene_model import SceneModelSource

__all__ = ["SceneModelFactory", "available", "get", "register", "unregister"]


@runtime_checkable
class SceneModelFactory(Protocol):
    """A callable that builds a `SceneModelSource`. Keyword arguments are the source's own."""

    def __call__(self, **kwargs) -> SceneModelSource: ...


_REGISTRY: dict[str, SceneModelFactory] = {}


def register(name: str, factory: SceneModelFactory, *, replace: bool = False) -> None:
    """Registers `factory` under `name`.

    Raises `ValueError` if `name` is already taken and `replace` is False, or if `factory` is not
    callable.
    """
    if not callable(factory):
        raise TypeError(f"factory for {name!r} is not callable: {factory!r}")
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"scene-model source {name!r} is already registered; pass replace=True to override"
        )
    _REGISTRY[name] = factory


def unregister(name: str) -> None:
    """Removes `name`. Raises `KeyError` if it was never registered."""
    del _REGISTRY[name]


def get(name: str) -> SceneModelFactory:
    """The factory registered under `name`. Raises `KeyError` naming what is available."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no scene-model source named {name!r}; registered: {available()}"
        ) from None


def available() -> tuple[str, ...]:
    """Every registered name, sorted."""
    return tuple(sorted(_REGISTRY))


def _register_builtins() -> None:
    def _ideal(**kwargs):
        from .ideal_scene_model import IdealSceneModel

        return IdealSceneModel(**kwargs)

    def _colour(**kwargs):
        from .conventional_scene_model import ColourSegmentationSceneModel

        return ColourSegmentationSceneModel(**kwargs)

    def _learned(**kwargs):
        from .conventional_scene_model import LearnedSceneModel

        return LearnedSceneModel(**kwargs)

    register("ideal", _ideal, replace=True)
    register("colour", _colour, replace=True)
    register("learned", _learned, replace=True)


_register_builtins()
