"""Generic :class:`~research.efm.models.FeedbackModel` over HTTP endpoints.

This is the decoupling piece: it gives any benchmark the same EFM backend
behaviour as ``skillopt.integrations.efm.SkillOptFeedbackModel`` without taking
a dependency on SkillOpt.  Role routing mirrors that adapter exactly so results
are comparable across benches.
"""
from __future__ import annotations

from ._client import chat, make_client
from .config import EndpointConfig

# Online stages run on the (small) target backend; everything else — trajectory
# review, policy proposal, gate, memory eval — runs on the strong optimizer.
ONLINE_STAGES = {"efm_step", "efm_reflect"}


class OpenAIChatFeedbackModel:
    """Feedback model backed by one or two OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        target: EndpointConfig,
        optimizer: EndpointConfig | None = None,
        role: str = "split",
    ) -> None:
        normalized = str(role or "split").strip().lower()
        if normalized not in {"split", "target", "optimizer"}:
            raise ValueError("role must be 'split', 'target', or 'optimizer'")
        self.role = normalized
        self._target_ep = target
        self._optimizer_ep = optimizer or target
        self._target_client = make_client(target)
        self._optimizer_client = (
            self._target_client if optimizer is None else make_client(optimizer)
        )

    def _route(self, stage: str) -> tuple[EndpointConfig, object]:
        if self.role == "split":
            use_optimizer = stage not in ONLINE_STAGES
        else:
            use_optimizer = self.role == "optimizer"
        if use_optimizer:
            return self._optimizer_ep, self._optimizer_client
        return self._target_ep, self._target_client

    def complete(self, system: str, user: str, *, max_tokens: int, stage: str) -> str:
        endpoint, client = self._route(stage)
        return chat(
            client,
            endpoint,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=int(max_tokens),
        )
