"""Unvalidated hidden-relative generation-only steering policy.

This remains experiment-local until task-level causal validation is positive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from research.steering.adapters.models.hf_causal import resolve_decoder_layer


@dataclass
class HiddenRelativeSteerer:
    model: torch.nn.Module
    layer: int
    vector: torch.Tensor
    natural_rho: float
    multiplier: float
    min_addition_norm: float
    max_addition_norm: float
    _handle: torch.utils.hooks.RemovableHandle | None = field(default=None, init=False)

    def __enter__(self):
        if self.vector.ndim != 1 or not torch.isfinite(self.vector).all():
            raise ValueError("vector must be a finite rank-1 tensor")
        self._handle = resolve_decoder_layer(self.model, self.layer).register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] != 1 or self.multiplier == 0:
            return output
        direction = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        direction = direction / direction.float().norm().clamp_min(1e-12).to(hidden.dtype)
        hidden_norm = hidden.float().norm(dim=-1, keepdim=True)
        magnitude = abs(float(self.multiplier))
        sign = 1.0 if self.multiplier >= 0 else -1.0
        target = hidden_norm * float(self.natural_rho) * magnitude
        lower = float(self.min_addition_norm) * magnitude
        upper = float(self.max_addition_norm) * magnitude
        if upper > 0:
            target = target.clamp(min=max(0.0, lower), max=max(lower, upper))
        steered = hidden + direction.view(1, 1, -1) * target.to(hidden.dtype) * sign
        return (steered, *output[1:]) if isinstance(output, tuple) else steered
