"""Offline replay of one EFM policy-update window on the saved 06-24 state.

Drives the real pipeline (trajectory review -> proposal -> validate -> apply ->
frozen-transition gate) against a copy of feedback_state.json, using the same
Qwen3.5-4B backend.  Mutates only the scratch copy.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request

sys.path.insert(0, "/data5/ninghan/tlm")
from research.efm.runtime import FeedbackRuntime
from research.efm.models import FeedbackRuntimeConfig

SRC = ("/data5/ninghan/tlm/benchmarks/skillopt/outputs/"
       "qwen35_4b_alfworld_efm_policy_smoke_gpu1_20260624/feedback_state.json")
SCRATCH = "/data5/ninghan/tlm/research/efm/diagnostics/_offline_state.json"
ENDPOINT = "http://localhost:8007/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-4B"


class VLLMModel:
    def complete(self, system: str, user: str, *, max_tokens: int, stage: str) -> str:
        body = json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer token-abc123"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]


def main() -> None:
    shutil.copyfile(SRC, SCRATCH)
    runtime = FeedbackRuntime(VLLMModel(), state_path=SCRATCH, config=FeedbackRuntimeConfig())
    # The saved run already advanced the cursor past its window; rewind so this
    # replay re-evaluates the same 20-episode window against the fixed code.
    runtime._state["policy_cursor"] = 0
    before = runtime.policy
    splits = {}
    for ep in runtime._state["episodes"]:
        splits[ep.get("split", "?")] = splits.get(ep.get("split", "?"), 0) + 1
    print(f"episodes={len(runtime._state['episodes'])}  splits={splits}  "
          f"policy.version(before)={before.version} rules={len(before.rules)}")

    decision = runtime._updater.maybe_update()
    runtime._store.save(runtime._state)
    after = runtime.policy

    print("\n===== DECISION =====")
    if decision is None:
        print("maybe_update returned None (window/precondition not met)")
    else:
        print(json.dumps({
            "accepted": decision.accepted,
            "reason": decision.reason,
            "base_version": decision.base_version,
            "candidate_version": decision.candidate_version,
            "n_corrections": len(decision.corrections),
        }, ensure_ascii=False, indent=2))
    print(f"\npolicy.version(after)={after.version}  rules={len(after.rules)}")
    for rule in after.rules:
        print(f"  - [{rule.status}] scope={rule.scope} support={rule.support_episode_ids}")
        print(f"    instruction: {rule.instruction[:160]}")


if __name__ == "__main__":
    main()
