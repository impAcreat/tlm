"""Hugging Face adapter implementing the core hidden-state provider contract."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_decoder_layer(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers[layer]
    raise ValueError("unsupported decoder layout")


class HFHiddenStateProvider:
    def __init__(self, model, tokenizer, *, device: str, system_prompt: str = "",
                 enable_thinking: bool = False, chat_template: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.chat_template = chat_template

    def _ids(self, prompt: str) -> torch.Tensor:
        if not self.chat_template:
            return self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self.device)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(add_generation_prompt=True, return_tensors="pt")
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(self.device)

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], *, pooling: str) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            ids = self._ids(prompt)
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
            )
            layers = output.hidden_states[1:]
            if pooling == "last_token":
                reps = torch.stack([hidden[0, -1] for hidden in layers])
            elif pooling == "mean":
                reps = torch.stack([hidden[0].mean(0) for hidden in layers])
            else:
                raise ValueError(f"unsupported pooling: {pooling}")
            rows.append(reps.float().cpu())
        return torch.stack(rows)
