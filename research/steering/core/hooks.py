from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _get_transformer_layer(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer]
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers[layer]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    raise ValueError(
        "unsupported model layout: expected model.model.layers, "
        "model.language_model.layers, model.model.language_model.layers, or model.transformer.h"
    )


@dataclass
class ActivationSteerer:
    """Temporarily add a steering vector to one transformer layer output."""

    model: torch.nn.Module
    layer: int
    vector: torch.Tensor
    alpha: float
    token_slice: slice | None = None
    _handle: torch.utils.hooks.RemovableHandle | None = field(default=None, init=False)

    def __enter__(self) -> "ActivationSteerer":
        module = _get_transformer_layer(self.model, self.layer)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            return (self._steer_hidden(hidden), *output[1:])
        return self._steer_hidden(output)

    def _steer_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        addition = self.alpha * vector.view(1, 1, -1)
        if self.token_slice is None:
            return hidden + addition

        steered = hidden.clone()
        steered[:, self.token_slice, :] = steered[:, self.token_slice, :] + addition
        return steered


@dataclass
class MultiLayerActivationSteerer:
    """Temporarily add steering vectors to multiple transformer layer outputs."""

    model: torch.nn.Module
    vectors: dict[int, torch.Tensor]
    alpha: float
    token_slice: slice | None = None
    _handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list, init=False)

    def __enter__(self) -> "MultiLayerActivationSteerer":
        for layer, vector in sorted(self.vectors.items()):
            module = _get_transformer_layer(self.model, int(layer))
            self._handles.append(module.register_forward_hook(self._make_hook(vector)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, vector: torch.Tensor):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                return (self._steer_hidden(hidden, vector), *output[1:])
            return self._steer_hidden(output, vector)

        return hook

    def _steer_hidden(self, hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(device=hidden.device, dtype=hidden.dtype)
        addition = self.alpha * vector.view(1, 1, -1)
        if self.token_slice is None:
            return hidden + addition
        steered = hidden.clone()
        steered[:, self.token_slice, :] = steered[:, self.token_slice, :] + addition
        return steered
