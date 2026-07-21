"""Behavior checks for the deterministic EFM policy gate."""
from __future__ import annotations

from research.efm.models import FeedbackRuntimeConfig, StepFeedback
from research.efm.policy import EFMPolicy, PolicyRule
from research.efm.updater import PolicyUpdater


def _window() -> list[dict]:
    return [{
        "episode_id": "ep-good",
        "task_description": "Find the apple.",
        "environment_id": "alfworld",
        "task_type": "look",
        "split": "validation",
        "trace": [{
            "step_id": 0,
            "action": "look",
            "raw_observation": "You see apple 1 on the table. You are not carrying anything.",
            "step_feedback": {
                "core_signal": "No verified environment state change was extracted.",
                "signal_type": "ambiguity",
                "filtered_out": "",
                "fallback": False,
            },
        }],
    }]


def _updater_for(feedback: StepFeedback) -> PolicyUpdater:
    return PolicyUpdater(
        state={},
        config=FeedbackRuntimeConfig(
            policy_gate_mode="deterministic",
            policy_validation_transitions=1,
        ),
        complete=lambda *args, **kwargs: "{}",
        render_candidate=lambda context, policy: feedback,
    )


def test_deterministic_gate_accepts_more_complete_candidate() -> None:
    updater = _updater_for(StepFeedback(core_signal="You see apple 1 on the table.", signal_type="progress"))
    candidate = EFMPolicy(version=1, rules=[PolicyRule(
        id="rule-good",
        scope={"environment_id": "alfworld"},
        instruction="Report concrete visible objects from the current observation.",
        avoid="Do not default to ambiguity when the observation names a visible object.",
        support_episode_ids=["ep-good", "ep-2", "ep-3"],
    )])
    assert updater._gate(candidate, _window()) == "accepted"


def test_deterministic_gate_rejects_ungrounded_transform_candidate() -> None:
    updater = _updater_for(StepFeedback(core_signal="apple 1 was cleaned.", signal_type="progress"))
    candidate = EFMPolicy(version=1)
    assert updater._gate(candidate, _window()) == "deterministic_quality_regressed"


if __name__ == "__main__":
    test_deterministic_gate_accepts_more_complete_candidate()
    test_deterministic_gate_rejects_ungrounded_transform_candidate()
    print("deterministic gate tests passed")
