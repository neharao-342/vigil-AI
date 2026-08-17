"""L1 — Block registry.

Blocks register themselves by name so graphs can be built by reference
(e.g. from a declarative spec) without importing concrete classes directly.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from core.blocks.block import Block

_REGISTRY: dict[str, type[Block]] = {}

B = TypeVar("B", bound=type[Block])


def register(name: str | None = None) -> Callable[[B], B]:
    """Class decorator that registers a Block subclass under `name`.

    If `name` is omitted, the block's `name` attribute is used.
    """

    def _decorator(cls: B) -> B:
        key = name or getattr(cls, "name", None)
        if not key:
            raise ValueError(f"{cls.__name__}: block must declare a name")
        if key in _REGISTRY:
            raise ValueError(f"duplicate block name: {key!r}")
        _REGISTRY[key] = cls
        return cls

    return _decorator


def get(name: str) -> type[Block]:
    """Return the registered Block class for `name`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown block: {name!r}") from exc


def create(name: str, **config: object) -> Block:
    """Instantiate a registered block by name with the given config."""
    return get(name)(**config)


def available() -> list[str]:
    """Return the sorted list of registered block names."""
    return sorted(_REGISTRY)
