"""Run configuration for the bench-agnostic EFM harness.

Everything benchmark-specific (task loading, action execution, success check)
lives in a per-bench :class:`~research.efm.harness.env.EFMEnv`.  This module
only carries endpoints and budgets, so tau2 / appworld / alfworld share one
runner and one output contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from research.efm.models import FeedbackRuntimeConfig

Arm = Literal["raw", "handcrafted", "efm"]


@dataclass(frozen=True)
class EndpointConfig:
    """An OpenAI-compatible chat endpoint (local vLLM by default)."""

    base_url: str
    model: str
    api_key: str = "token-abc123"
    timeout: float = 180.0
    temperature: float = 0.0
    max_retries: int = 3
    disable_thinking: bool = True


@dataclass(frozen=True)
class RunConfig:
    """One arm of one benchmark run.

    ``target`` serves the (frozen) agent and the online Step EFM; ``optimizer``
    (a stronger model) serves the offline skill-update stages.  When
    ``optimizer`` is omitted the target is reused for every stage.
    """

    run_dir: str
    target: EndpointConfig
    optimizer: EndpointConfig | None = None

    arm: Arm = "efm"
    max_steps: int = 30
    agent_temperature: float = 0.2
    agent_max_tokens: int = 512
    handcraft_char_limit: int = 600

    # EFM runtime knobs (Phase A-C default: single rollout, no self-evolution).
    feedback_role: Literal["split", "target", "optimizer"] = "split"
    step_max_tokens: int = 192
    trajectory_max_tokens: int = 2048
    feedback_workers: int = 1
    policy_update_enabled: bool = False
    reflect_enabled: bool = True
    extra_feedback_config: dict = field(default_factory=dict)

    @property
    def optimizer_endpoint(self) -> EndpointConfig:
        return self.optimizer or self.target

    def feedback_config(self) -> FeedbackRuntimeConfig:
        """Translate harness knobs into a core EFM runtime config."""
        return FeedbackRuntimeConfig(
            step_max_tokens=int(self.step_max_tokens),
            trajectory_max_tokens=int(self.trajectory_max_tokens),
            feedback_workers=int(self.feedback_workers),
            policy_update_enabled=bool(self.policy_update_enabled),
            reflect_enabled=bool(self.reflect_enabled),
            **dict(self.extra_feedback_config),
        )
