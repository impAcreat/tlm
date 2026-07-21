from __future__ import annotations

import torch


def directional_consistency(state_deltas: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean off-diagonal cosine per layer for [state, layer, hidden] deltas."""
    if state_deltas.ndim != 3:
        raise ValueError("state_deltas must have shape [state, layer, hidden]")
    n = state_deltas.shape[0]
    if n < 2:
        return torch.full((state_deltas.shape[1],), float("nan"))
    unit = state_deltas.float() / state_deltas.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    sims = torch.einsum("slh,tlh->lst", unit, unit)
    mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
    return sims[:, mask].mean(1)
