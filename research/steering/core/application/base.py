"""Application interfaces; model layout is supplied by an adapter resolver."""
from __future__ import annotations

from collections.abc import Callable

import torch

LayerResolver = Callable[[torch.nn.Module, int], torch.nn.Module]
TokenGate = Callable[[torch.Tensor], bool]


def all_forwards(_: torch.Tensor) -> bool:
    return True
