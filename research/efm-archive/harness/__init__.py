"""Bench-agnostic harness for running EFM experiments.

A benchmark plugs in by implementing :class:`EFMEnv` (load tasks, execute the
agent's action text, report success).  The shared runner then drives a frozen
:class:`LLMAgent` and the EFM core, comparing arms (``raw`` / ``handcrafted`` /
``efm``) and emitting the output contract consumed by ``research.efm.bench.eval``.

No benchmark framework (e.g. SkillOpt) is imported here, so tau2 / appworld /
alfworld all reach the *same* EFM core through this layer.
"""
from __future__ import annotations

from .agent import Agent, LLMAgent
from .config import Arm, EndpointConfig, RunConfig
from .env import EFMEnv, Reset, Step
from .feedback_model import OpenAIChatFeedbackModel
from .runner import run_episodes, run_one

__all__ = [
    "Agent",
    "LLMAgent",
    "Arm",
    "EndpointConfig",
    "RunConfig",
    "EFMEnv",
    "Reset",
    "Step",
    "OpenAIChatFeedbackModel",
    "run_episodes",
    "run_one",
]
