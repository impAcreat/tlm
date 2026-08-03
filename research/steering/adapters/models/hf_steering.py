"""HF-layout wrappers, including the legacy multi-layer subclass contract."""
from __future__ import annotations

import torch

from research.steering.adapters.models.hf_causal import resolve_decoder_layer
from research.steering.core.application.additive import AdditiveSteerer


class ActivationSteerer(AdditiveSteerer):
    def __init__(self, model, layer, vector, alpha, token_slice=None):
        super().__init__(model, resolve_decoder_layer, layer, vector, alpha, token_slice=token_slice)


class MultiLayerActivationSteerer:
    """Backward-compatible HF wrapper used by the established 4B scripts.

    Keeping ``_steer_hidden(hidden, vector)`` overridable preserves the mature
    generation/prefill steering modes while model layout stays outside core.
    """

    def __init__(self, model, vectors, alpha, token_slice=None):
        self.model = model
        self.vectors = vectors
        self.alpha = alpha
        self.token_slice = token_slice
        self._handles = []

    def __enter__(self):
        for layer, vector in sorted(self.vectors.items()):
            module = resolve_decoder_layer(self.model, int(layer))
            self._handles.append(module.register_forward_hook(self._make_hook(vector)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, vector):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            steered = self._steer_hidden(hidden, vector)
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        return hook

    def _steer_hidden(self, hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(device=hidden.device, dtype=hidden.dtype)
        if vector.numel() != hidden.shape[-1]:
            raise ValueError(f"vector dim {vector.numel()} != hidden dim {hidden.shape[-1]}")
        addition = float(self.alpha) * vector.view(1, 1, -1)
        if self.token_slice is None:
            return hidden + addition
        steered = hidden.clone()
        steered[:, self.token_slice, :] += addition
        return steered
