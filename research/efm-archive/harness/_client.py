"""OpenAI-compatible chat helper shared by the harness.

Kept dependency-free of any benchmark code so every bench (ALFWorld via
SkillOpt, tau2, appworld, ...) can reach the same EFM core without importing a
particular benchmark framework.
"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing openai at module import time
    from openai import OpenAI

    from .config import EndpointConfig

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def make_client(endpoint: "EndpointConfig") -> "OpenAI":
    """Construct an OpenAI client pointed at a local vLLM endpoint."""
    from openai import OpenAI

    return OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout,
        max_retries=0,  # retries handled here so we control backoff + logging
    )


def strip_think(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block if a reasoning model leaks one."""
    return _THINK_BLOCK.sub("", text or "", count=1).strip()


def chat(
    client: "OpenAI",
    endpoint: "EndpointConfig",
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float | None = None,
) -> str:
    """Single chat completion with bounded retries; returns post-think text."""
    extra_body: dict = {}
    if endpoint.disable_thinking:
        # Qwen3 chat template honours this; harmless for non-reasoning models.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    temp = endpoint.temperature if temperature is None else temperature
    last_error: Exception | None = None
    for attempt in range(max(1, endpoint.max_retries)):
        try:
            response = client.chat.completions.create(
                model=endpoint.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                extra_body=extra_body or None,
            )
            content = response.choices[0].message.content or ""
            return strip_think(content)
        except Exception as error:  # network / server hiccups: retry with backoff
            last_error = error
            if attempt + 1 < max(1, endpoint.max_retries):
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"chat completion failed after retries: {last_error}") from last_error
