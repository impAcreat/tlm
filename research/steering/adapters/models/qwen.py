"""Qwen-specific surface normalization; no steering logic belongs here."""
from __future__ import annotations

import json
import re


def normalize_alfworld_action(raw: str) -> str:
    if re.search(r"<action>.*?</action>", raw, flags=re.S | re.I):
        if re.search(r"<think>.*?</think>", raw, flags=re.S | re.I):
            return raw
        return f"<think>Action selected by the model.</think>\n{raw}"
    payload = raw.split("<tool_call>", 1)[1] if "<tool_call>" in raw else raw
    payload = payload.replace("</tool_call>", "").strip()
    candidate = next((line.strip() for line in payload.splitlines() if line.strip()), "look")
    if candidate.startswith("{"):
        try:
            obj = json.loads(candidate)
            candidate = str(obj.get("action") or obj.get("arguments") or obj.get("name") or "look")
        except json.JSONDecodeError:
            pass
    return f"<think>Action selected by the model.</think>\n<action>{candidate}</action>"
