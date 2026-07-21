from __future__ import annotations


def reflection_prefix(text: str, *, index: int = 1) -> str:
    return (
        "## Reflections from your previous failed attempt(s)\n"
        f"Reflection {index}:\n{text.strip()}\n\n"
        "Use these reflections in this retry.\n\n"
    )
