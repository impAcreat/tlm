"""LLM clients for real agent execution.

References:
- OpenAI-compatible chat-completions APIs used by OpenAI, vLLM, Ollama proxies,
  and many hosted inference providers.
- Agent-R1/AgentGym style harnesses, where the model emits an action and the
  environment returns the next observation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


Message = dict[str, str]


class ChatModel(Protocol):
    def complete(self, messages: list[Message]) -> str:
        """Return assistant text for a chat prompt."""


@dataclass
class OpenAICompatibleChatModel:
    """Minimal OpenAI-compatible client using only the Python standard library."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    timeout_s: int = 60

    def complete(self, messages: list[Message]) -> str:
        base_url = (
            self.base_url
            or os.environ.get("CLASSROOM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        api_key = self.api_key or os.environ.get("CLASSROOM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if _looks_local(base_url):
                api_key = "EMPTY"
            else:
                raise RuntimeError("Set CLASSROOM_API_KEY or OPENAI_API_KEY for hosted LLM calls.")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc

        return str(data["choices"][0]["message"]["content"])


class ScriptedChatModel:
    """Deterministic test model; production code should use OpenAICompatibleChatModel."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("ScriptedChatModel has no responses left.")
        return self._responses.pop(0)


def _looks_local(base_url: str) -> bool:
    return "localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url
