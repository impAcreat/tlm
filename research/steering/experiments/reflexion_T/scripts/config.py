"""Load and merge Reflexion-T YAML configs without embedding experiment logic."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(base_path: str | Path, *override_paths: str | Path) -> dict:
    with Path(base_path).open() as stream:
        config = yaml.safe_load(stream) or {}
    for path in override_paths:
        with Path(path).open() as stream:
            config = deep_merge(config, yaml.safe_load(stream) or {})
    return config
