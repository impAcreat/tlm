"""Model configuration loaded from a local JSON file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    student_model: str
    teacher_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    timeout_s: int = 60


def load_model_config(path: str | Path) -> ModelConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ModelConfig(
        student_model=str(data["student_model"]),
        teacher_model=data.get("teacher_model"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        temperature=float(data.get("temperature", 0.0)),
        timeout_s=int(data.get("timeout_s", 60)),
    )
