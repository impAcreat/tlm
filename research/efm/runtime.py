"""Online Step EFM and windowed, gated prompt-policy updates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from .constitution import build_step_system
from .models import FeedbackModel, FeedbackRuntimeConfig, PolicyUpdateDecision, StepFeedback, TrajectoryCorrection
from .policy import EFMPolicy
from .prompts import REFLECT_SYSTEM, reflect_user_prompt, step_user_prompt
from .updater import PolicyUpdater


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _truncate(value: Any, limit: int) -> str:
    text = _as_text(value)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated for EFM budget]"


def _json_value(response: str) -> Any:
    text = str(response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char in "[{":
                try:
                    return decoder.raw_decode(text[index:])[0]
                except json.JSONDecodeError:
                    continue
    raise ValueError("EFM model did not return JSON")


class _JsonStateStore:
    """Atomic JSON persistence for policy versions and private audit traces."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None

    def load(self) -> dict:
        if self.path is None or not self.path.exists():
            return self._empty_state()
        with self.path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"EFM state must be a JSON object: {self.path}")
        value.setdefault("schema_version", 2)
        value.setdefault("policy", EFMPolicy().to_dict())
        value.setdefault("episodes", [])
        value.setdefault("corrections", [])
        value.setdefault("policy_updates", [])
        value.setdefault("policy_cursor", 0)
        return value

    @staticmethod
    def _empty_state() -> dict:
        return {
            "schema_version": 2,
            "policy": EFMPolicy().to_dict(),
            "episodes": [],
            "corrections": [],
            "policy_updates": [],
            "policy_cursor": 0,
        }

    def save(self, state: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".efm-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


class EpisodeFeedbackSession:
    """Private trace for one episode; public callers see only StepFeedback."""

    def __init__(
        self,
        runtime: "FeedbackRuntime",
        episode_id: str,
        task_description: str,
        *,
        environment_id: str = "",
        task_type: str = "",
    ) -> None:
        self.runtime = runtime
        self.episode_id = str(episode_id)
        self.task_description = str(task_description or "")
        self.environment_id = str(environment_id or "")
        self.task_type = str(task_type or "")
        self.policy_version = runtime.policy.version
        self.trace: list[dict] = []
        self._actions: list[str] = []
        self.reflection: list[str] = []
        self.reflection_history: list[dict] = []

    def refine(self, action: str, raw_observation: Any, intention: str = "") -> StepFeedback:
        feedback = self.runtime._refine_step(
            task_description=self.task_description,
            action=action,
            raw_observation=raw_observation,
            step_id=len(self.trace),
            recent_actions=self._actions,
            environment_id=self.environment_id,
            task_type=self.task_type,
            intention=intention,
            reflection=self.reflection,
        )
        self._append(action, raw_observation, feedback, intention)
        self.runtime._maybe_reflect(self)
        return feedback

    def _append(
        self, action: str, raw_observation: Any, feedback: StepFeedback,
        intention: str = "",
    ) -> None:
        self.trace.append({
            "step_id": len(self.trace),
            "action": _as_text(action),
            "intention": _as_text(intention),
            "raw_observation": _as_text(raw_observation),
            "step_feedback": feedback.to_dict(),
        })
        self._actions.append(_as_text(action))

    def finish(
        self,
        *,
        success: bool,
        outcome: dict | None = None,
        artifact_dir: str | None = None,
    ) -> list[TrajectoryCorrection]:
        decision = self.runtime._finish_episode(self, success=success, outcome=outcome or {})
        if artifact_dir:
            directory = Path(artifact_dir)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{self.episode_id}.efm.json").open("w", encoding="utf-8") as handle:
                json.dump({
                    "episode_id": self.episode_id,
                    "task_description": self.task_description,
                    "environment_id": self.environment_id,
                    "task_type": self.task_type,
                    "success": bool(success),
                    "policy_version": self.policy_version,
                    "reflection": list(self.reflection),
                    "reflection_history": list(self.reflection_history),
                    "trace": self.trace,
                    "policy_update": decision.to_dict() if decision else None,
                }, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return decision.corrections if decision else []


class FeedbackRuntime:
    """Composable EFM runtime with immutable constitution and versioned policy."""

    def __init__(
        self,
        model: FeedbackModel,
        *,
        state_path: str | os.PathLike[str] | None = None,
        config: FeedbackRuntimeConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or FeedbackRuntimeConfig()
        self._store = _JsonStateStore(state_path)
        self._state = self._store.load()
        self._updater = PolicyUpdater(
            state=self._state,
            config=self.config,
            complete=self._complete,
            render_candidate=self._render_candidate,
        )

    @property
    def policy(self) -> EFMPolicy:
        return EFMPolicy.from_dict(self._state["policy"])

    def start_episode(
        self,
        episode_id: str,
        task_description: str,
        *,
        environment_id: str = "",
        task_type: str = "",
    ) -> EpisodeFeedbackSession:
        return EpisodeFeedbackSession(
            self,
            episode_id,
            task_description,
            environment_id=environment_id,
            task_type=task_type,
        )

    def refine_many(self, requests: Iterable[tuple]) -> list[StepFeedback]:
        """Run independent Step EFM calls concurrently while preserving order.

        Each request is ``(session, action, observation)`` or
        ``(session, action, observation, intention)``.
        """
        items = [
            tuple(item) if len(item) == 4 else (item[0], item[1], item[2], "")
            for item in requests
        ]
        if not items:
            return []
        workers = min(max(1, self.config.feedback_workers), len(items))
        if workers == 1:
            return [
                session.refine(action, observation, intention)
                for session, action, observation, intention in items
            ]
        outputs: list[StepFeedback | None] = [None] * len(items)

        def work(index: int, session: EpisodeFeedbackSession, action: str, observation: Any, intention: str):
            return index, self._refine_step(
                task_description=session.task_description,
                action=action,
                raw_observation=observation,
                step_id=len(session.trace),
                recent_actions=session._actions,
                environment_id=session.environment_id,
                task_type=session.task_type,
                intention=intention,
                reflection=session.reflection,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(work, index, *item) for index, item in enumerate(items)]
            for future in as_completed(futures):
                index, feedback = future.result()
                outputs[index] = feedback
        for (session, action, observation, intention), feedback in zip(items, outputs):
            assert feedback is not None
            session._append(action, observation, feedback, intention)
        for session in {id(item[0]): item[0] for item in items}.values():
            self._maybe_reflect(session)
        return [feedback for feedback in outputs if feedback is not None]

    def _complete(self, system: str, user: str, *, max_tokens: int, stage: str) -> str:
        response = self.model.complete(system, user, max_tokens=max_tokens, stage=stage)
        return str(response[0] if isinstance(response, tuple) else response or "")

    def _refine_step(
        self,
        *,
        task_description: str,
        action: Any,
        raw_observation: Any,
        step_id: int,
        recent_actions: list[str],
        environment_id: str,
        task_type: str,
        policy: EFMPolicy | None = None,
        intention: str = "",
        reflection: list[str] | None = None,
    ) -> StepFeedback:
        """Online step-level feedback: best local delivery of THIS observation,
        conditioned on the constitution, the current durable policy, and the
        transient within-episode reflection (which the agent never sees)."""
        active_policy = policy or self.policy
        try:
            value = _json_value(self._complete(
                build_step_system(active_policy, environment_id=environment_id, task_type=task_type, action=_as_text(action)),
                step_user_prompt(
                    task_description=_truncate(task_description, 2_000),
                    action=_truncate(action, 1_000),
                    raw_observation=_truncate(raw_observation, self.config.raw_observation_char_limit),
                    step_id=step_id,
                    recent_actions=recent_actions[-self.config.recent_actions_limit:],
                    agent_intention=_truncate(intention, 1_000),
                    episode_reflection=list(reflection or []),
                ),
                max_tokens=self.config.step_max_tokens,
                stage="efm_step",
            ))
            if not isinstance(value, dict):
                raise ValueError("step output is not an object")
            signal_type = str(value.get("signal_type", "ambiguity"))
            allowed = {"progress", "constraint_violated", "tool_error", "ambiguity", "state_change"}
            intention_status = str(value.get("intention_status", "unclear"))
            allowed_intent = {"fulfilled", "unfulfilled", "unclear"}
            return StepFeedback(
                core_signal=str(value.get("core_signal", "")).strip() or "No verified environment state change was extracted.",
                signal_type=signal_type if signal_type in allowed else "ambiguity",  # type: ignore[arg-type]
                filtered_out=str(value.get("filtered_out", "")).strip(),
                intention_status=intention_status if intention_status in allowed_intent else "unclear",  # type: ignore[arg-type]
            )
        except Exception as exc:
            return StepFeedback(
                core_signal="The environment response could not be reliably refined; no verified state change is available.",
                signal_type="ambiguity",
                filtered_out=f"EFM unavailable ({type(exc).__name__}); raw observation retained only in the private audit trace.",
                fallback=True,
            )

    def _maybe_reflect(self, session: "EpisodeFeedbackSession") -> None:
        """Online step-level optimization: every K steps, update the episode's
        reflection so later feedback in the SAME task improves. Never edits the
        durable skill -- graduation to the skill is decided later by the gate."""
        if not self.config.reflect_enabled:
            return
        k = max(1, self.config.reflect_every_k_steps)
        if session.trace and len(session.trace) % k == 0:
            before = list(session.reflection)
            session.reflection = self._reflect(session)
            if session.reflection != before:
                session.reflection_history.append({
                    "at_step": len(session.trace),
                    "notes": list(session.reflection),
                    "new_notes": [n for n in session.reflection if n not in before],
                })

    def _reflect(self, session: "EpisodeFeedbackSession") -> list[str]:
        try:
            window = max(2, self.config.reflect_every_k_steps * 2)
            recent = [
                {
                    "step_id": row["step_id"],
                    "action": row["action"],
                    "raw_observation": _truncate(row["raw_observation"], 800),
                    "step_feedback": row["step_feedback"],
                }
                for row in session.trace[-window:]
            ]
            value = _json_value(self._complete(
                REFLECT_SYSTEM,
                reflect_user_prompt(
                    task_description=_truncate(session.task_description, 2_000),
                    recent_steps=recent,
                    current_reflection=list(session.reflection),
                    max_notes=self.config.reflection_max_notes,
                ),
                max_tokens=self.config.reflect_max_tokens,
                stage="efm_reflect",
            ))
            notes = value.get("reflection", []) if isinstance(value, dict) else []
            cleaned = [str(note).strip() for note in notes if str(note).strip()]
            return cleaned[: self.config.reflection_max_notes] or session.reflection
        except Exception:
            return session.reflection

    def _render_candidate(self, context: dict, candidate: EFMPolicy) -> StepFeedback:
        episode, row = context["episode"], context["row"]
        previous_actions = [item["action"] for item in episode["trace"][:row["step_id"]]]
        return self._refine_step(
            task_description=episode.get("task_description", ""),
            action=row["action"],
            raw_observation=row["raw_observation"],
            step_id=int(row["step_id"]),
            recent_actions=previous_actions,
            environment_id=episode.get("environment_id", ""),
            task_type=episode.get("task_type", ""),
            policy=candidate,
            intention=row.get("intention", ""),
            reflection=None,
        )

    def _finish_episode(self, session: EpisodeFeedbackSession, *, success: bool, outcome: dict) -> PolicyUpdateDecision | None:
        self._updater.observe_episode({
            "episode_id": session.episode_id,
            "task_description": session.task_description,
            "environment_id": session.environment_id,
            "task_type": session.task_type,
            "policy_version": session.policy_version,
            "success": bool(success),
            "outcome": outcome,
            "reflection": list(session.reflection),
            "reflection_history": list(session.reflection_history),
            "trace": session.trace,
        })
        decision = self._updater.maybe_update()
        self._trim_history()
        self._store.save(self._state)
        return decision

    def _trim_history(self) -> None:
        limit = max(1, self.config.history_limit)
        episodes = self._state["episodes"]
        dropped = max(0, len(episodes) - limit)
        if dropped:
            self._state["episodes"] = episodes[-limit:]
            self._state["policy_cursor"] = max(0, int(self._state["policy_cursor"]) - dropped)
        self._state["corrections"] = self._state["corrections"][-limit:]
        self._state["policy_updates"] = self._state["policy_updates"][-limit:]
