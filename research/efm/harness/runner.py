"""The single rollout loop shared by every benchmark.

It drives ``env`` + ``agent`` + EFM and emits the common output contract read
by :mod:`research.efm.bench.eval`:

    <run_dir>/results.jsonl        one JSON per episode: {id, hard, soft, ...}
    <run_dir>/feedback_state.json  EFM runtime state (efm arm only)
    <run_dir>/feedback/<id>.efm.json  per-episode EFM audit trace (efm arm)

The ``arm`` controls only what observation text the agent sees after each
action — raw observation, a handcrafted compression, or the EFM refinement —
holding the agent, env, tasks and budgets fixed.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.efm import FeedbackRuntime

from .agent import Agent
from .config import RunConfig
from .env import EFMEnv
from .feedback_model import OpenAIChatFeedbackModel


def _handcraft(observation: str, limit: int) -> str:
    """Trivial template arm: collapse whitespace and cap length."""
    text = " ".join(str(observation).split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _build_runtime(cfg: RunConfig, run_dir: Path) -> FeedbackRuntime:
    model = OpenAIChatFeedbackModel(
        target=cfg.target,
        optimizer=cfg.optimizer,
        role=cfg.feedback_role,
    )
    return FeedbackRuntime(
        model,
        state_path=str(run_dir / "feedback_state.json"),
        config=cfg.feedback_config(),
    )


def run_one(env: EFMEnv, agent: Agent, task, cfg: RunConfig, runtime, feedback_dir: Path) -> dict:
    reset = env.reset(task)
    agent.reset(reset.system_prompt)
    session = None
    if runtime is not None:
        session = runtime.start_episode(
            reset.task_id,
            reset.task_description,
            environment_id=getattr(env, "environment_id", ""),
            task_type=reset.task_type,
        )

    observation = reset.observation
    last_info: dict = dict(reset.info or {})
    steps = 0
    for steps in range(1, cfg.max_steps + 1):
        action = agent.act(observation)
        step = env.step(action)
        last_info = dict(step.info or {})
        if session is not None:
            observation = session.refine(action, step.raw_observation).agent_text()
        elif cfg.arm == "handcrafted":
            observation = _handcraft(step.raw_observation, cfg.handcraft_char_limit)
        else:  # raw
            observation = str(step.raw_observation)
        if step.done:
            break

    success = env.is_success(last_info)
    if session is not None:
        session.finish(success=success, outcome=last_info, artifact_dir=str(feedback_dir))

    soft = last_info.get("soft")
    return {
        "id": reset.task_id,
        "arm": cfg.arm,
        "task_type": reset.task_type,
        "hard": 1.0 if success else 0.0,
        "soft": float(soft) if soft is not None else (1.0 if success else 0.0),
        "steps": steps,
        "done": bool(last_info.get("done", success)),
    }


def run_episodes(env: EFMEnv, agent: Agent, cfg: RunConfig, tasks=None, runtime=None) -> list[dict]:
    """Run every task through one arm and write the output contract.

    ``runtime`` may be supplied to inject a pre-built (or stubbed) EFM runtime;
    otherwise one is constructed from ``cfg`` when the arm is ``"efm"``.
    """
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    feedback_dir = run_dir / "feedback"
    if runtime is None and cfg.arm == "efm":
        runtime = _build_runtime(cfg, run_dir)

    task_iter = list(tasks) if tasks is not None else list(env.iter_tasks())
    results: list[dict] = []
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for task in task_iter:
            result = run_one(env, agent, task, cfg, runtime, feedback_dir)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    close = getattr(env, "close", None)
    if callable(close):
        close()
    return results
