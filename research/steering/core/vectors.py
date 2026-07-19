from __future__ import annotations

import torch


def mean_pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token activations while ignoring padding."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [batch, seq, dim], got {tuple(hidden.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, seq], got {tuple(attention_mask.shape)}"
        )
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden/attention shape mismatch: {tuple(hidden.shape[:2])} vs {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / counts


def compute_contrast_vector(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    method: str = "pca",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a unit steering vector oriented from negative states to positive states.

    The default follows the intra-trajectory prototype used in the paper's spirit:
    collect positive/negative step representations, compute the dominant centered
    contrast direction, and orient it using the positive-minus-negative mean.
    """
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative must be rank-2 [n, dim] tensors")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError(f"dimension mismatch: {positive.shape[1]} vs {negative.shape[1]}")
    if positive.shape[0] == 0 or negative.shape[0] == 0:
        raise ValueError("positive and negative must both contain at least one vector")

    positive = positive.float()
    negative = negative.float()
    mean_diff = positive.mean(dim=0) - negative.mean(dim=0)

    if method == "mean_diff":
        vector = mean_diff
    elif method == "pca":
        centered = torch.cat([positive, negative], dim=0)
        centered = centered - centered.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        vector = vh[0]
        if torch.dot(vector, mean_diff) < 0:
            vector = -vector
    else:
        raise ValueError(f"unknown method: {method}")

    norm = vector.norm()
    if not torch.isfinite(norm) or norm < eps:
        raise ValueError("contrast vector is degenerate")
    return vector / norm
