"""Environment Feedback Module (EFM).

EFM sits between an environment and an agent.  It refines each environment
observation for online use, then uses completed trajectories to improve its
own future refinements.  It deliberately does not decide how a SkillOpt skill
should be edited.
"""

from .models import (
    FeedbackModel,
    FeedbackRuntimeConfig,
    PolicyUpdateDecision,
    StepFeedback,
    TrajectoryCorrection,
)
from .policy import EFMPolicy, PolicyExample, PolicyPatch, PolicyRule
from .runtime import EpisodeFeedbackSession, FeedbackRuntime

__all__ = [
    "EpisodeFeedbackSession",
    "FeedbackModel",
    "FeedbackRuntime",
    "FeedbackRuntimeConfig",
    "EFMPolicy",
    "PolicyExample",
    "PolicyPatch",
    "PolicyRule",
    "PolicyUpdateDecision",
    "StepFeedback",
    "TrajectoryCorrection",
]
