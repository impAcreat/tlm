"""Dual-form ridge compiler validated by the 4B and exploratory 32B runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeCompiler:
    alpha: float = 100.0
    x_mean_: np.ndarray | None = None
    y_mean_: np.ndarray | None = None
    x_centered_: np.ndarray | None = None
    dual_: np.ndarray | None = None

    def fit(self, text_representations: np.ndarray, vectors: np.ndarray):
        x = np.asarray(text_representations, dtype=np.float32)
        y = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("x and y must be rank-2 arrays with equal row counts")
        self.x_mean_ = x.mean(0, keepdims=True)
        self.y_mean_ = y.mean(0, keepdims=True)
        self.x_centered_ = x - self.x_mean_
        yc = y - self.y_mean_
        gram = self.x_centered_ @ self.x_centered_.T
        self.dual_ = np.linalg.solve(
            gram + np.eye(len(x), dtype=np.float32) * self.alpha,
            yc,
        )
        return self

    def predict(self, text_representations: np.ndarray) -> np.ndarray:
        if any(x is None for x in (self.x_mean_, self.y_mean_, self.x_centered_, self.dual_)):
            raise RuntimeError("compiler is not fitted")
        x = np.asarray(text_representations, dtype=np.float32)
        return (x - self.x_mean_) @ self.x_centered_.T @ self.dual_ + self.y_mean_
