"""Stable, model-independent metadata for steering artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class VectorMetadata:
    model_id: str
    layer: int
    hidden_size: int
    extraction_method: str
    representation: str
    aggregation: str
    conditioning: str
    unit_id: str = ""
    config_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, vector: torch.Tensor) -> None:
        if vector.ndim != 1:
            raise ValueError(f"steering vector must be rank 1, got {tuple(vector.shape)}")
        if vector.numel() != self.hidden_size:
            raise ValueError(
                f"hidden-size mismatch: metadata={self.hidden_size}, vector={vector.numel()}"
            )
        if not torch.isfinite(vector).all():
            raise ValueError("steering vector contains non-finite values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SteeringVectorArtifact:
    vector: torch.Tensor
    metadata: VectorMetadata

    def __post_init__(self) -> None:
        self.metadata.validate(self.vector)

    def state_dict(self) -> dict[str, Any]:
        return {"vector": self.vector.detach().cpu(), "metadata": self.metadata.to_dict()}
