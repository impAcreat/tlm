from __future__ import annotations

import numpy as np


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("a and b must be matching rank-2 arrays")
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return (an * bn).sum(1)


def residual_cosine_rows(predicted: np.ndarray, target: np.ndarray, reference_mean: np.ndarray) -> np.ndarray:
    mean = np.asarray(reference_mean).reshape(1, -1)
    return cosine_rows(np.asarray(predicted) - mean, np.asarray(target) - mean)
