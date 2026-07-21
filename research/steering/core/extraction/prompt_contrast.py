"""Paired prompt-conditioned mean-delta extraction.

This is the validated extraction family used by the 4B experiments and the
32B Reflexion study: identical environment states, with only the conditioning
text changed, followed by a mean over paired last-token activation deltas.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from research.steering.core.extraction.base import ExtractionResult, HiddenStateProvider
from research.steering.core.metrics.consistency import directional_consistency


class PromptContrastExtractor:
    name = "prompt_contrast"

    def __init__(self, *, pooling: str = "last_token", keep_state_deltas: bool = False):
        if pooling not in {"last_token", "mean"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.pooling = pooling
        self.keep_state_deltas = keep_state_deltas

    @torch.no_grad()
    def extract(
        self,
        provider: HiddenStateProvider,
        base_prompts: Sequence[str],
        conditioned_prompts: Sequence[str],
        *,
        layers: Sequence[int] | None = None,
    ) -> ExtractionResult:
        if len(base_prompts) != len(conditioned_prompts):
            raise ValueError("paired prompt lists must have equal length")
        if not base_prompts:
            raise ValueError("at least one paired state is required")
        base = provider.encode(base_prompts, pooling=self.pooling).float()
        conditioned = provider.encode(conditioned_prompts, pooling=self.pooling).float()
        return self.extract_representations(base, conditioned, layers=layers)

    def extract_representations(
        self,
        base: torch.Tensor,
        conditioned: torch.Tensor,
        *,
        layers: Sequence[int] | None = None,
    ) -> ExtractionResult:
        if base.shape != conditioned.shape or base.ndim != 3:
            raise ValueError(
                "provider must return matching [state, layer, hidden] tensors; "
                f"got {tuple(base.shape)} and {tuple(conditioned.shape)}"
            )
        selected = tuple(range(base.shape[1])) if layers is None else tuple(int(x) for x in layers)
        delta = conditioned[:, selected, :] - base[:, selected, :]
        return ExtractionResult(
            mean_vector=delta.mean(0),
            state_deltas=delta if self.keep_state_deltas else None,
            consistency=directional_consistency(delta),
            layers=selected,
        )
