"""Interfaces shared by validated steering-vector extraction methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class HiddenStateProvider(Protocol):
    """Model adapter contract; implementations return [state, layer, hidden]."""

    def encode(self, prompts: Sequence[str], *, pooling: str) -> torch.Tensor: ...


@dataclass
class ExtractionResult:
    mean_vector: torch.Tensor
    state_deltas: torch.Tensor | None
    consistency: torch.Tensor
    layers: tuple[int, ...]
