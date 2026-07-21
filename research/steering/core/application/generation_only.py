"""Gate additive steering to cached decode forwards only."""
from __future__ import annotations

import torch


def generation_only(hidden: torch.Tensor) -> bool:
    return hidden.ndim == 3 and hidden.shape[1] == 1
