"""Open up the EFM gate: does the candidate rule actually fire on the
validation transitions, and what does the gate judge see?
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request

sys.path.insert(0, "/data5/ninghan/tlm")
from research.efm.runtime import FeedbackRuntime
from research.efm.models import FeedbackRuntimeConfig
from research.efm.policy import EFMPolicy, PolicyPatch, validate_patch, apply_patch
from research.efm.prompts import POLICY_SYSTEM, policy_user_prompt
from research.efm.constitution import build_step_system

SRC = ("/data5/ninghan/tlm/benchmarks/skillopt/outputs/"
       "qwen35_4b_alfworld_efm_policy_smoke_gpu1_20260624/feedback_state.json")
SCRATCH = "/data5/ninghan/tlm/research/efm/diagnostics/_probe_state.json"
ENDPOINT = "http://localhost:8007/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-4B"


class VLLMModel:
    def complete(self, system, user, *, max_tokens, stage):
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(ENDPOINT, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer token-abc123"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]


def main():
    shutil.copyfile(SRC, SCRATCH)
    rt = FeedbackRuntime(VLLMModel(), state_path=SCRATCH, config=FeedbackRuntimeConfig())
    rt._state["policy_cursor"] = 0
    up = rt._updater
    policy = EFMPolicy.from_dict(rt._state["policy"])
    window = [e for e in rt._state["episodes"] if e.get("policy_version") == policy.version]

    analysis = up._select_analysis(window)
    corrections = up._review(analysis)
    print(f"corrections={len(corrections)}")

    # build correction rows with scope (mirror _propose_and_gate)
    scope_by_ep = {str(e.get("episode_id", "")): (str(e.get("environment_id", "")),
                   str(e.get("task_type", ""))) for e in window}
    rows = []
    for c in corrections:
        r = c.to_dict()
        env, tt = scope_by_ep.get(c.episode_id, ("", ""))
        r["environment_id"], r["task_type"] = env, tt
        rows.append(r)

    raw = up.complete(POLICY_SYSTEM, policy_user_prompt(
        policy=policy.to_dict(), corrections=rows,
        edit_budget=2, min_support=3), max_tokens=1024, stage="probe")
    from research.efm.updater import _json_value
    patch = PolicyPatch.from_dict(_json_value(raw))
    err = validate_patch(patch, policy, min_support=3, max_edits=2)
    print(f"patch valid? {err!r}; edits scopes={[e.get('scope') for e in patch.edits]}")
    if err:
        return
    candidate = apply_patch(policy, patch, max_rules=8, max_examples=8)
    print(f"candidate v{candidate.version} rules={len(candidate.rules)}")

    # gate transitions: validation split
    transitions = []
    for ep in window:
        if ep.get("split") != "validation":
            continue
        for row in ep.get("trace", []):
            transitions.append((ep, row))
    transitions = transitions[:rt.config.policy_validation_transitions]
    print(f"gate transitions={len(transitions)} from validation episodes\n")

    for ep, row in transitions[:4]:
        sel = candidate.select_rules(environment_id=ep.get("environment_id", ""),
                                     task_type=ep.get("task_type", ""))
        cand_fb = up.render_candidate({"episode": ep, "row": row}, candidate)
        print(f"[{ep['episode_id']}:{row['step_id']}] env={ep.get('environment_id')} "
              f"task={ep.get('task_type')} | rule_fires={len(sel)>0}")
        print(f"   action      : {row['action']}")
        print(f"   baseline_fb : {row['step_feedback']['core_signal'][:140]}")
        print(f"   candidate_fb: {cand_fb.core_signal[:140]}")
        print()


if __name__ == "__main__":
    main()
