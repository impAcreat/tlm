"""Frozen, skill-free chat agent used for EFM Phase-1 experiments.

Per ``IDEA.md`` Phase 1, the agent is frozen and carries no skill — only the
feedback module learns.  The agent keeps a running transcript and returns raw
action text; the environment is responsible for parsing/executing it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ._client import chat, make_client
from .config import EndpointConfig


@runtime_checkable
class Agent(Protocol):
    def reset(self, system_prompt: str) -> None: ...
    def act(self, observation: str) -> str: ...


class LLMAgent:
    """Stateless-per-episode chat agent over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: EndpointConfig,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        history_turns: int = 40,
    ) -> None:
        self.endpoint = endpoint
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.history_turns = int(history_turns)
        self._client = make_client(endpoint)
        self._system = ""
        self._messages: list[dict] = []

    def reset(self, system_prompt: str) -> None:
        self._system = str(system_prompt or "")
        self._messages = []

    def act(self, observation: str) -> str:
        self._messages.append({"role": "user", "content": str(observation)})
        window = self._messages[-2 * self.history_turns:]
        messages = [{"role": "system", "content": self._system}, *window] if self._system else window
        text = chat(
            self._client,
            self.endpoint,
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self._messages.append({"role": "assistant", "content": text})
        return text
