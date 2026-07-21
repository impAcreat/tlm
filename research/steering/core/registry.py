"""Small explicit registry used by config-driven experiment scripts."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class MethodRegistry:
    def __init__(self, kind: str):
        self.kind = kind
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        if name in self._constructors:
            raise ValueError(f"duplicate {self.kind} method: {name}")
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            constructor = self._constructors[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._constructors)) or "<none>"
            raise ValueError(f"unknown {self.kind} method {name!r}; known: {known}") from exc
        return constructor(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._constructors))
