"""Low-cost, validation-gated prompt-policy updates for EFM."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

from .models import FeedbackRuntimeConfig, PolicyUpdateDecision, StepFeedback, TrajectoryCorrection
from .policy import EFMPolicy, PolicyPatch, apply_patch, validate_patch
from .quality import score_feedback
from .prompts import GATE_SYSTEM, MEMORY_EVAL_SYSTEM, PIVOTAL_GATE_SYSTEM, POLICY_SYSTEM, TRAJECTORY_SYSTEM, gate_user_prompt, memory_eval_user_prompt, pivotal_gate_user_prompt, policy_user_prompt, trajectory_batch_user_prompt


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
    raise ValueError("model did not return JSON")


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


_CORRECTION_ADVICE_PATTERN = re.compile(
    r"\b(you can|consider|should|must|need to|next|further inspection|"
    r"searching other|turn it on|might be)\b",
    re.IGNORECASE,
)


def _valid_correction(row: dict[str, Any]) -> bool:
    better_feedback = str(row.get("better_feedback", "")).strip()
    problem = str(row.get("problem", "")).strip()
    if not better_feedback or not problem:
        return False
    return _CORRECTION_ADVICE_PATTERN.search(better_feedback) is None


class PolicyUpdater:
    """Collect trajectories, propose one small patch, then gate it offline."""

    def __init__(
        self,
        *,
        state: dict,
        config: FeedbackRuntimeConfig,
        complete: Callable[..., str],
        render_candidate: Callable[[dict, EFMPolicy], StepFeedback],
    ) -> None:
        self.state = state
        self.config = config
        self.complete = complete
        self.render_candidate = render_candidate
        self.state.setdefault("policy", EFMPolicy().to_dict())
        self.state.setdefault("episodes", [])
        self.state.setdefault("corrections", [])
        self.state.setdefault("policy_updates", [])
        self.state.setdefault("policy_cursor", 0)

    def observe_episode(self, episode: dict) -> None:
        digest = hashlib.sha256(str(episode["episode_id"]).encode()).digest()[0] / 255
        episode["split"] = "validation" if digest < self.config.policy_validation_fraction else "train"
        self.state["episodes"].append(episode)

    def maybe_update(self) -> PolicyUpdateDecision | None:
        if not self.config.policy_update_enabled:
            return None
        policy = EFMPolicy.from_dict(self.state["policy"])
        cursor = int(self.state.get("policy_cursor", 0))
        episodes = self.state["episodes"]
        window = [item for item in episodes[cursor:] if item.get("policy_version") == policy.version]
        if len(window) < self.config.policy_window_episodes:
            return None

        # Hold out half of the window's FAILED episodes for the gate, so policy
        # acceptance is tested on failures the proposal never saw. Independent of
        # the hash train/val split, this works at any window size.
        traced = [ep for ep in window if ep.get("trace")]
        failures = [ep for ep in traced if not ep.get("success")]
        successes = [ep for ep in traced if ep.get("success")]
        if len(failures) >= 4:
            # Enough failures for a clean held-out split.
            proposal_pool = failures[0::2] + successes[0::2]
            gate_episodes = failures[1::2] + successes[1::2]
        else:
            # Small window: share failures so both proposal (needs support) and
            # gate (needs failures to test) can run. Not fully held-out -- a
            # larger window restores the clean split above.
            proposal_pool = failures + successes[0::2]
            gate_episodes = failures + successes[1::2]

        analysis = self._select_analysis(proposal_pool)
        corrections = self._review(analysis)
        self.state["corrections"].extend(correction.to_dict() for correction in corrections)
        graduated = self._evaluate_memories(proposal_pool)
        decision, candidate = self._propose_and_gate(policy, corrections, window, gate_episodes, graduated)
        self.state["policy_updates"].append(decision.to_dict())
        self.state["policy_cursor"] = len(episodes)
        if decision.accepted and candidate is not None:
            self.state["policy"] = candidate.to_dict()
        return decision

    def _select_analysis(self, episodes: list[dict]) -> list[dict]:
        traced = [item for item in episodes if item.get("trace")]
        failures = [item for item in traced if not item.get("success")]
        successes = [item for item in traced if item.get("success")]
        # Spread selection across task types so corrections span multiple types,
        # which lets the policy proposer write environment-scoped rules instead
        # of narrow task-type-scoped ones.
        budget = self.config.policy_analysis_episodes
        task_types: dict[str, list[dict]] = {}
        for ep in failures + successes:
            tt = str(ep.get("task_type") or "unknown")
            task_types.setdefault(tt, []).append(ep)
        if len(task_types) <= 1:
            return (failures + successes)[:budget]
        # Round-robin across task types (failures-first within each bucket)
        result: list[dict] = []
        buckets = list(task_types.values())
        i = 0
        while len(result) < budget and any(buckets):
            bucket = buckets[i % len(buckets)]
            if bucket:
                result.append(bucket.pop(0))
            i += 1
        return result

    @staticmethod
    def _mem_step(row: dict) -> dict:
        return {
            "step_id": row.get("step_id"),
            "action": _clip(row.get("action", ""), 200),
            "raw_observation": _clip(row.get("raw_observation", ""), 400),
            "feedback": str((row.get("step_feedback") or {}).get("core_signal", ""))[:200],
        }

    def _evaluate_memories(self, episodes: list[dict]) -> list[dict]:
        """Judge each within-episode memory by its before/after window + outcome.

        A memory only becomes a graduation candidate if the optimizer rules it a
        grounded, reusable feedback-delivery lesson. Fail-safe: any error on an
        episode simply yields no graduated memories for it.
        """
        if not self.config.memory_eval_enabled:
            return []
        graduated: list[dict] = []
        kctx = max(1, self.config.memory_context_steps)
        for episode in episodes:
            history = episode.get("reflection_history") or []
            trace = episode.get("trace") or []
            if not history or not trace:
                continue
            items = []
            for event in history:
                at = int(event.get("at_step", 0))
                before = trace[max(0, at - kctx):at]
                after = trace[at:at + kctx]
                items.append({
                    "id": f"{episode.get('episode_id', '')}:{at}",
                    "memory": event.get("new_notes") or event.get("notes") or [],
                    "before_steps": [self._mem_step(r) for r in before],
                    "after_steps": [self._mem_step(r) for r in after],
                })
            if not items:
                continue
            outcome = episode.get("outcome", {}) or {}
            try:
                value = _json_value(self.complete(
                    MEMORY_EVAL_SYSTEM,
                    memory_eval_user_prompt(
                        task_description=_clip(episode.get("task_description", ""), 800),
                        outcome={
                            "hard": outcome.get("hard"),
                            "soft": outcome.get("soft"),
                            "success": bool(episode.get("success")),
                            "fail_reason": _clip(outcome.get("fail_reason", ""), 200),
                        },
                        memories=items,
                    ),
                    max_tokens=self.config.policy_max_tokens,
                    stage="efm_memory_eval",
                ))
            except Exception:
                continue
            verdicts = value.get("memories", []) if isinstance(value, dict) else []
            for verdict in verdicts:
                if not isinstance(verdict, dict):
                    continue
                if str(verdict.get("verdict", "")).strip().lower() != "graduate":
                    continue
                lesson = str(verdict.get("lesson", "")).strip()
                if not lesson:
                    continue
                graduated.append({
                    "episode_id": str(episode.get("episode_id", "")),
                    "memory_id": str(verdict.get("id", "")),
                    "lesson": lesson,
                })
        return graduated

    def _review(self, episodes: list[dict]) -> list[TrajectoryCorrection]:
        corrections: list[TrajectoryCorrection] = []
        for start in range(0, len(episodes), 4):
            payload = []
            valid_steps: dict[str, set[int]] = {}
            for episode in episodes[start:start + 4]:
                valid_steps[str(episode["episode_id"])] = {
                    int(row["step_id"]) for row in episode["trace"]
                }
                trace = [
                    {
                        "step_id": row["step_id"],
                        "intention": _clip(row.get("intention", ""), 500),
                        "action": row["action"],
                        "raw_observation": _clip(row["raw_observation"], 2_000),
                        "step_feedback": row["step_feedback"],
                    }
                    for row in episode["trace"]
                ]
                outcome = episode.get("outcome", {}) or {}
                payload.append({
                    "episode_id": episode["episode_id"],
                    "task_description": _clip(episode.get("task_description", ""), 2_000),
                    "success": bool(episode.get("success")),
                    "outcome": {
                        "hard": outcome.get("hard"),
                        "soft": outcome.get("soft"),
                        "fail_reason": _clip(outcome.get("fail_reason", ""), 300),
                    },
                    "trace": _clip(trace, self.config.trace_char_limit),
                })
            try:
                value = _json_value(self.complete(
                    TRAJECTORY_SYSTEM,
                    trajectory_batch_user_prompt(episodes=payload),
                    max_tokens=self.config.trajectory_max_tokens,
                    stage="efm_trajectory_review",
                ))
                rows = value.get("corrections", []) if isinstance(value, dict) else []
                for row in rows:
                    if not isinstance(row, dict) or not _valid_correction(row):
                        continue
                    episode_id = str(row.get("episode_id", ""))
                    step_id = int(row.get("step_id", -1))
                    if episode_id not in valid_steps or step_id not in valid_steps[episode_id]:
                        continue
                    correction = TrajectoryCorrection(
                        episode_id=episode_id,
                        step_id=step_id,
                        original_feedback=str(row.get("original_feedback", "")).strip(),
                        problem=str(row.get("problem", "")).strip(),
                        better_feedback=str(row.get("better_feedback", "")).strip(),
                        event_type=str(row.get("event_type", "")).strip(),
                        pivotal=bool(row.get("pivotal", False)),
                        importance_gap=str(row.get("importance_gap", "")).strip(),
                        whole_picture_feedback=str(row.get("whole_picture_feedback", "")).strip(),
                    )
                    corrections.append(correction)
            except Exception:
                continue
        return corrections

    def _propose_and_gate(
        self,
        policy: EFMPolicy,
        corrections: list[TrajectoryCorrection],
        window: list[dict],
        gate_episodes: list[dict] | None = None,
        graduated_memories: list[dict] | None = None,
    ) -> tuple[PolicyUpdateDecision, EFMPolicy | None]:
        if not corrections:
            return PolicyUpdateDecision(False, "no_corrections", policy.version, corrections=[]), None
        scope_by_episode = {
            str(episode.get("episode_id", "")): (
                str(episode.get("environment_id", "")),
                str(episode.get("task_type", "")),
            )
            for episode in window
        }
        correction_rows = []
        for correction in corrections:
            row = correction.to_dict()
            environment_id, task_type = scope_by_episode.get(correction.episode_id, ("", ""))
            row["environment_id"] = environment_id
            row["task_type"] = task_type
            correction_rows.append(row)
        try:
            patch = PolicyPatch.from_dict(_json_value(self.complete(
                POLICY_SYSTEM,
                policy_user_prompt(
                    policy=policy.to_dict(),
                    corrections=correction_rows,
                    edit_budget=self.config.policy_max_edits,
                    min_support=self.config.policy_min_support,
                    graduated_memories=graduated_memories or [],
                ),
                max_tokens=self.config.policy_max_tokens,
                stage="efm_policy_proposal",
            )))
        except Exception:
            return PolicyUpdateDecision(False, "proposal_unavailable", policy.version, corrections=corrections), None
        invalid_reason = validate_patch(
            patch,
            policy,
            min_support=self.config.policy_min_support,
            max_edits=self.config.policy_max_edits,
        )
        dropped_edits = []
        if invalid_reason:
            valid_edits = []
            for index, edit in enumerate(patch.edits):
                single = PolicyPatch(base_version=patch.base_version, edits=[edit], reasoning=patch.reasoning)
                edit_reason = validate_patch(
                    single,
                    policy,
                    min_support=self.config.policy_min_support,
                    max_edits=1,
                )
                if edit_reason:
                    dropped_edits.append({"index": index, "reason": edit_reason, "edit": edit})
                    continue
                valid_edits.append(edit)
            if valid_edits:
                patch = PolicyPatch(
                    base_version=patch.base_version,
                    edits=valid_edits[:self.config.policy_max_edits],
                    reasoning=patch.reasoning,
                )
                invalid_reason = validate_patch(
                    patch,
                    policy,
                    min_support=self.config.policy_min_support,
                    max_edits=self.config.policy_max_edits,
                )
            if invalid_reason:
                return PolicyUpdateDecision(
                    False,
                    invalid_reason,
                    policy.version,
                    corrections=corrections,
                    candidate_patch=patch.to_dict(),
                    gate_diagnostics={
                        "mode": "not_run",
                        "invalid_reason": invalid_reason,
                        "dropped_edits": dropped_edits,
                    },
                ), None
        candidate = apply_patch(
            policy,
            patch,
            max_rules=self.config.policy_max_rules,
            max_examples=self.config.policy_max_examples,
        )
        gate_reason = self._gate(candidate, window, gate_episodes or [])
        decision = PolicyUpdateDecision(
            accepted=gate_reason == "accepted",
            reason=gate_reason,
            base_version=policy.version,
            candidate_version=candidate.version if gate_reason == "accepted" else None,
            corrections=corrections,
            candidate_patch=patch.to_dict(),
            gate_diagnostics={
                **(getattr(self, "_last_gate_diagnostics", None) or {}),
                **({"dropped_edits": dropped_edits} if dropped_edits else {}),
            },
        )
        return decision, candidate if decision.accepted else None

    def _gate(self, candidate: EFMPolicy, window: list[dict], gate_episodes: list[dict]) -> str:
        self._last_gate_diagnostics = None
        if self.config.policy_gate_mode == "outcome":
            return self._outcome_gate(candidate, gate_episodes)
        transitions = self._validation_transitions(window)
        if len(transitions) < self.config.policy_validation_transitions:
            return "insufficient_validation_transitions"
        if self.config.policy_gate_mode == "llm":
            return self._llm_gate(candidate, transitions)
        return self._deterministic_gate(candidate, transitions)

    def _validation_transitions(self, window: list[dict]) -> list[tuple[dict, dict]]:
        transitions = []
        for episode in window:
            if episode.get("split") != "validation":
                continue
            for row in episode.get("trace", []):
                transitions.append((episode, row))
        return transitions[:self.config.policy_validation_transitions]

    def _quality_totals(
        self, candidate: EFMPolicy, transitions: list[tuple[dict, dict]],
    ) -> tuple[list[int], list[int], dict | None]:
        """Score baseline vs candidate quality over transitions.

        Returns ``(baseline_totals, candidate_totals, hard_fail)`` where
        ``hard_fail`` is a diagnostics dict if any candidate feedback breaks
        consistency (an unconditional regression), else ``None``.
        """
        baseline_totals = [0, 0, 0]
        candidate_totals = [0, 0, 0]
        for episode, row in transitions:
            previous_actions = [item["action"] for item in episode.get("trace", [])[:int(row["step_id"])]]
            baseline = score_feedback(
                row["step_feedback"],
                action=row["action"],
                raw_observation=row["raw_observation"],
                recent_actions=previous_actions,
                task_description=episode.get("task_description", ""),
            )
            candidate_feedback = self.render_candidate({"episode": episode, "row": row}, candidate)
            candidate_quality = score_feedback(
                candidate_feedback,
                action=row["action"],
                raw_observation=row["raw_observation"],
                recent_actions=previous_actions,
                task_description=episode.get("task_description", ""),
            )
            if candidate_quality.consistency == 0:
                return baseline_totals, candidate_totals, {
                    "failed_transition": f"{episode.get('episode_id')}:{row.get('step_id')}",
                    "failed_issues": candidate_quality.issues,
                }
            for index, score in enumerate(baseline.as_tuple()):
                baseline_totals[index] += score
            for index, score in enumerate(candidate_quality.as_tuple()):
                candidate_totals[index] += score
        return baseline_totals, candidate_totals, None

    def _deterministic_gate(self, candidate: EFMPolicy, transitions: list[tuple[dict, dict]]) -> str:
        baseline_totals, candidate_totals, hard_fail = self._quality_totals(candidate, transitions)
        self._last_gate_diagnostics = {
            "mode": "deterministic",
            "transitions": len(transitions),
            "baseline_totals": _totals_dict(baseline_totals),
            "candidate_totals": _totals_dict(candidate_totals),
            **(hard_fail or {}),
        }
        if hard_fail is not None:
            return "deterministic_quality_regressed"
        if any(cand < base for cand, base in zip(candidate_totals, baseline_totals)):
            return "deterministic_quality_regressed"
        if any(cand > base for cand, base in zip(candidate_totals, baseline_totals)):
            return "accepted"
        return "gate_not_improved"

    def _outcome_gate(self, candidate: EFMPolicy, gate_episodes: list[dict]) -> str:
        """Accept iff the candidate improves feedback on the pivotal steps of
        HELD-OUT failed episodes (judged against the whole-picture target by a
        reference-visible LLM), with no quality regression on the held-out set.
        Ties acceptance to actual task outcomes, not a structural-quality proxy.
        """
        guard = [
            (ep, row)
            for ep in gate_episodes
            for row in ep.get("trace", [])
        ][: max(1, self.config.policy_validation_transitions) * 2]
        baseline_totals, candidate_totals, hard_fail = self._quality_totals(candidate, guard)
        diagnostics = {
            "mode": "outcome",
            "gate_episodes": len(gate_episodes),
            "guard_transitions": len(guard),
            "baseline_totals": _totals_dict(baseline_totals),
            "candidate_totals": _totals_dict(candidate_totals),
        }
        if hard_fail is not None:
            self._last_gate_diagnostics = {**diagnostics, "stage": "no_regression", **hard_fail}
            return "deterministic_quality_regressed"
        if any(cand < base for cand, base in zip(candidate_totals, baseline_totals)):
            self._last_gate_diagnostics = {**diagnostics, "stage": "no_regression"}
            return "deterministic_quality_regressed"

        failed = [ep for ep in gate_episodes if not ep.get("success") and ep.get("trace")]
        diagnostics["failed_gate_episodes"] = len(failed)
        if not failed:
            self._last_gate_diagnostics = {**diagnostics, "pivotal_steps": 0, "note": "no_failed_gate_episodes"}
            return "accepted" if any(c > b for c, b in zip(candidate_totals, baseline_totals)) else "gate_not_improved"

        pivotal = [cor for cor in self._review(failed) if cor.pivotal]
        ep_by_id = {str(ep["episode_id"]): ep for ep in failed}
        pairs = []
        for cor in pivotal[:self.config.policy_validation_transitions]:
            episode = ep_by_id.get(cor.episode_id)
            if episode is None:
                continue
            row = next((r for r in episode["trace"] if int(r["step_id"]) == cor.step_id), None)
            if row is None:
                continue
            candidate_feedback = self.render_candidate({"episode": episode, "row": row}, candidate)
            pairs.append({
                "id": f"{cor.episode_id}:{cor.step_id}",
                "task_description": _clip(episode.get("task_description", ""), 800),
                "action": _clip(row["action"], 500),
                "raw_observation": _clip(row["raw_observation"], 1_500),
                "target_feedback": _clip(cor.whole_picture_feedback or cor.better_feedback, 500),
                "baseline_feedback": row["step_feedback"],
                "candidate_feedback": candidate_feedback.to_dict(),
            })
        diagnostics["pivotal_steps"] = len(pairs)
        if not pairs:
            self._last_gate_diagnostics = {**diagnostics, "note": "no_pivotal_steps"}
            return "gate_not_improved"
        try:
            result = _json_value(self.complete(
                PIVOTAL_GATE_SYSTEM,
                pivotal_gate_user_prompt(transitions=pairs),
                max_tokens=self.config.gate_max_tokens,
                stage="efm_pivotal_gate",
            ))
        except Exception:
            self._last_gate_diagnostics = {**diagnostics, "note": "pivotal_judge_unavailable"}
            return "gate_unavailable"
        wins = _count(result.get("candidate_wins", []))
        losses = _count(result.get("baseline_wins", []))
        unsafe = result.get("unsafe_candidate_ids", [])
        diagnostics.update({
            "pivotal_candidate_wins": wins,
            "pivotal_baseline_wins": losses,
            "unsafe_candidate_ids": list(unsafe) if isinstance(unsafe, list) else [],
        })
        self._last_gate_diagnostics = diagnostics
        if unsafe:
            return "unsafe_candidate_feedback"
        if wins - losses >= self.config.policy_gate_min_pivotal_gain:
            return "accepted"
        return "gate_not_improved"

    def _llm_gate(self, candidate: EFMPolicy, transitions: list[tuple[dict, dict]]) -> str:
        pairs = []
        for episode, row in transitions:
            candidate_feedback = self.render_candidate({"episode": episode, "row": row}, candidate)
            pairs.append({
                "id": f"{episode['episode_id']}:{row['step_id']}",
                "task_description": _clip(episode.get("task_description", ""), 800),
                "action": _clip(row["action"], 500),
                "raw_observation": _clip(row["raw_observation"], 1_500),
                "baseline_feedback": row["step_feedback"],
                "candidate_feedback": candidate_feedback.to_dict(),
            })
        try:
            result = _json_value(self.complete(
                GATE_SYSTEM,
                gate_user_prompt(transitions=pairs),
                max_tokens=self.config.gate_max_tokens,
                stage="efm_policy_gate",
            ))
            wins = _count(result.get("candidate_wins", []))
            losses = _count(result.get("baseline_wins", []))
            unsafe = result.get("unsafe_candidate_ids", [])
            decisive = wins + losses
            if unsafe:
                return "unsafe_candidate_feedback"
            if decisive == 0 or wins / decisive < self.config.policy_gate_min_win_rate or wins <= losses:
                return "gate_not_improved"
            return "accepted"
        except Exception:
            return "gate_unavailable"


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list) else max(0, int(value or 0))


def _totals_dict(totals: list[int]) -> dict:
    return {"consistency": totals[0], "completeness": totals[1], "efficiency": totals[2]}
