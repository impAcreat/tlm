"""Public data contracts for the Environment Feedback Module (EFM)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol


SignalType = Literal[
    "progress", "constraint_violated", "tool_error", "ambiguity", "state_change",
]


class FeedbackModel(Protocol):
    """Minimal, backend-agnostic interface used by the feedback layer.

    The return value may be either response text or ``(response_text, metadata)``.
    This makes adapters for a local agent, OpenAI-compatible client, or SkillOpt's
    optimizer model equally small.
    """

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        stage: str,
    ) -> str | tuple[str, dict]: ...


@dataclass(frozen=True)
class FeedbackRuntimeConfig:
    """Budget and retention controls for a :class:`FeedbackRuntime`."""

    step_max_tokens: int = 192
    reflect_enabled: bool = True
    reflect_every_k_steps: int = 3
    reflect_max_tokens: int = 256
    reflection_max_notes: int = 3
    memory_eval_enabled: bool = True
    memory_context_steps: int = 3
    trajectory_max_tokens: int = 2048
    policy_max_tokens: int = 1024
    gate_max_tokens: int = 1024
    recent_actions_limit: int = 4
    history_limit: int = 200
    feedback_workers: int = 4
    raw_observation_char_limit: int = 6_000
    trace_char_limit: int = 6_000
    policy_update_enabled: bool = True
    policy_window_episodes: int = 20
    policy_analysis_episodes: int = 12
    policy_validation_transitions: int = 8
    policy_validation_fraction: float = 0.25
    policy_min_support: int = 3
    policy_max_edits: int = 2
    policy_max_rules: int = 8
    policy_max_examples: int = 8
    policy_gate_min_win_rate: float = 0.60
    policy_gate_mode: Literal["deterministic", "llm", "outcome"] = "deterministic"
    policy_gate_min_pivotal_gain: int = 1


@dataclass(frozen=True)
class StepFeedback:
    """The only step-level observation that should be shown to an agent."""

    core_signal: str
    signal_type: SignalType = "ambiguity"
    filtered_out: str = ""
    fallback: bool = False
    intention_status: Literal["fulfilled", "unfulfilled", "unclear"] = "unclear"

    def agent_text(self) -> str:
        """Render the deliberately narrow online-facing observation."""
        return f"Environment feedback [{self.signal_type}]: {self.core_signal}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryCorrection:
    step_id: int
    original_feedback: str
    problem: str
    better_feedback: str
    episode_id: str = ""
    event_type: str = ""
    pivotal: bool = False
    importance_gap: str = ""
    whole_picture_feedback: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PolicyUpdateDecision:
    accepted: bool
    reason: str
    base_version: int
    candidate_version: int | None = None
    corrections: list[TrajectoryCorrection] = field(default_factory=list)
    candidate_patch: dict | None = None
    gate_diagnostics: dict | None = None

    def to_dict(self) -> dict:
        value = {
            "accepted": self.accepted,
            "reason": self.reason,
            "base_version": self.base_version,
            "candidate_version": self.candidate_version,
            "corrections": [item.to_dict() for item in (self.corrections or [])],
        }
        if self.candidate_patch is not None:
            value["candidate_patch"] = self.candidate_patch
        if self.gate_diagnostics is not None:
            value["gate_diagnostics"] = self.gate_diagnostics
        return value
