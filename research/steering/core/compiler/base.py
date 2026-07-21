from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorCompiler(Protocol):
    def fit(self, text_representations: np.ndarray, vectors: np.ndarray): ...
    def predict(self, text_representations: np.ndarray) -> np.ndarray: ...
