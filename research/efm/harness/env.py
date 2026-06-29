"""The per-benchmark contract every environment implements to plug into EFM.

A bench connects to the harness by providing an :class:`EFMEnv`.  The harness
runs the (frozen) agent and the EFM; the env only *loads tasks*, *executes the
agent's action text*, and *reports success*.  Action parsing/execution is the
env's job — that keeps the agent and runner bench-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


@dataclass
class Reset:
    """Initial state of one episode, shown identically to every arm."""

    task_id: str
    task_description: str
    observation: str
    system_prompt: str = ""
    task_type: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class Step:
    """Outcome of executing one agent action in the environment."""

    raw_observation: str
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@runtime_checkable
class EFMEnv(Protocol):
    """Minimal interactive environment contract.

    ``environment_id`` is a stable bench tag (e.g. ``"tau2"``) recorded in the
    EFM state.  ``is_success`` reads the final ``Step.info`` (or ``Reset.info``
    if the episode produced no step).
    """

    environment_id: str

    def iter_tasks(self) -> Iterable[Any]:
        """Yield the task objects this run should evaluate."""

    def reset(self, task: Any) -> Reset:
        """Begin an episode for ``task`` and return its opening observation."""

    def step(self, action: str) -> Step:
        """Execute one agent action (raw text) and return the result."""

    def is_success(self, info: dict) -> bool:
        """Whether the episode succeeded, given the last observed ``info``."""

    def close(self) -> None:  # optional; default no-op providers may omit
        """Release any per-run resources."""
