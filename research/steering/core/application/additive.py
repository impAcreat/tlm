"""Validated additive activation steering, single- and multi-layer."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from research.steering.core.application.base import LayerResolver, TokenGate, all_forwards


@dataclass
class AdditiveSteerer:
    model: torch.nn.Module
    resolver: LayerResolver
    layer: int
    vector: torch.Tensor
    alpha: float
    gate: TokenGate = all_forwards
    token_slice: slice | None = None
    _handle: torch.utils.hooks.RemovableHandle | None = field(default=None, init=False)

    def __enter__(self):
        if self.vector.ndim != 1:
            raise ValueError("vector must be rank 1")
        self._handle = self.resolver(self.model, self.layer).register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not self.gate(hidden):
            return output
        steered = self._steer_hidden(hidden)
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    def _steer_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        if vector.numel() != hidden.shape[-1]:
            raise ValueError(f"vector dim {vector.numel()} != hidden dim {hidden.shape[-1]}")
        addition = self.alpha * vector.view(1, 1, -1)
        if self.token_slice is None:
            return hidden + addition
        steered = hidden.clone()
        steered[:, self.token_slice, :] = steered[:, self.token_slice, :] + addition
        return steered


@dataclass
class MultiLayerAdditiveSteerer:
    model: torch.nn.Module
    resolver: LayerResolver
    vectors: dict[int, torch.Tensor]
    alphas: float | dict[int, float]
    gate: TokenGate = all_forwards
    token_slice: slice | None = None
    _handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list, init=False)

    def __enter__(self):
        for layer, vector in sorted(self.vectors.items()):
            alpha = self.alphas[layer] if isinstance(self.alphas, dict) else self.alphas
            steerer = AdditiveSteerer(
                self.model, self.resolver, layer, vector, float(alpha), self.gate, self.token_slice
            )
            self._handles.append(self.resolver(self.model, layer).register_forward_hook(steerer._hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
