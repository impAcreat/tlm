"""Aggregate EFM step-feedback quality + meta-evolution signals from saved runs."""
from __future__ import annotations

import collections
import glob
import json
import os
import re

RUNS = {
    "policy_smoke(20ep)":  "benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_policy_smoke_gpu1_20260624/feedback_state.json",
    "parallel(24ep)":      "benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_gpu1_parallel_20260624/feedback_state.json",
}

# crude fidelity flags
META_LEAK = re.compile(r"untrusted|data artifact|grammatical|step_id|recent_actions|task_description was treated", re.I)
HALLUCINATION_HINT = re.compile(r"\bcleaned\b|\bwas cleaned\b|\bplaced on the\b|\bheated\b|will\b", re.I)


def audit(path):
    state = json.load(open(path))
    eps = state["episodes"]
    sig = collections.Counter()
    n_steps = fallback = meta_leak = success = 0
    leak_examples, halluc_examples = [], []
    for ep in eps:
        success += 1 if ep.get("success") else 0
        for row in ep.get("trace", []):
            n_steps += 1
            fb = row["step_feedback"]
            sig[fb.get("signal_type", "?")] += 1
            if fb.get("fallback"):
                fallback += 1
            core = fb.get("core_signal", "")
            if META_LEAK.search(core) or META_LEAK.search(fb.get("filtered_out", "")):
                meta_leak += 1
                if len(leak_examples) < 3:
                    leak_examples.append((ep["episode_id"], row["step_id"], row["action"], core))
    return {
        "n_ep": len(eps), "n_steps": n_steps, "success": success,
        "sig": dict(sig), "fallback": fallback, "meta_leak": meta_leak,
        "corrections": len(state.get("corrections", [])),
        "policy_version": state.get("policy", {}).get("version"),
        "policy_rules": len(state.get("policy", {}).get("rules", [])),
        "policy_updates": state.get("policy_updates", []),
        "leak_examples": leak_examples,
    }


for name, path in RUNS.items():
    if not os.path.exists(path):
        print(f"[{name}] MISSING {path}"); continue
    a = audit(path)
    print(f"========== {name} ==========")
    print(f"episodes={a['n_ep']} steps={a['n_steps']} task_success={a['success']}/{a['n_ep']}")
    print(f"signal_type dist: {a['sig']}")
    print(f"fallback_steps={a['fallback']}  meta_leak/hedge_steps={a['meta_leak']}  corrections_logged={a['corrections']}")
    print(f"policy: version={a['policy_version']} rules={a['policy_rules']} "
          f"updates={[(u.get('reason')) for u in a['policy_updates']]}")
    for eid, sid, act, core in a["leak_examples"]:
        print(f"   leak [{eid}:{sid}] action='{act}'\n        core: {core[:160]}")
    print()
