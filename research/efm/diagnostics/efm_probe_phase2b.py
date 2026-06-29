"""Diagnose Phase 2b `no_corrections`: was v0 feedback genuinely clean, or did
the 30B trajectory review silently error?"""
from __future__ import annotations

import json
import sys
import urllib.request

sys.path.insert(0, "/data5/ninghan/tlm")
from research.efm import quality
from research.efm.prompts import TRAJECTORY_SYSTEM, trajectory_batch_user_prompt

STATE = ("/data5/ninghan/tlm/benchmarks/skillopt/outputs/"
         "qwen35_4b_alfworld_efm_detgate_phase2b_20260625/feedback_state.json")
OPT = ("http://localhost:8001/v1/chat/completions", "Qwen/Qwen3-30B")


def chat(system, user, max_tokens=2048):
    body = json.dumps({"model": OPT[1],
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "max_tokens": max_tokens, "temperature": 0.3,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(OPT[0], data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer token-abc123"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    s = json.load(open(STATE))
    eps = s["episodes"]
    # 1) quality of online v0 feedback
    tot = [0, 0, 0]; n = 0
    sig = {}
    for ep in eps:
        for row in ep.get("trace", []):
            q = quality.score_feedback(row["step_feedback"], action=row["action"],
                                       raw_observation=row["raw_observation"],
                                       recent_actions=[r["action"] for r in ep["trace"][:row["step_id"]]])
            t = q.as_tuple(); tot = [a + b for a, b in zip(tot, t)]; n += 1
            st = row["step_feedback"].get("signal_type", "?"); sig[st] = sig.get(st, 0) + 1
    print(f"online v0 feedback: n_steps={n}  consistency={100*tot[0]/n:.0f}% "
          f"completeness={100*tot[1]/n:.0f}% efficiency={100*tot[2]/n:.0f}%")
    print(f"signal_type dist: {sig}")

    # 2) does 30B trajectory review actually return corrections? (errors surfaced)
    splits = {}
    for ep in eps:
        splits[ep.get("split", "?")] = splits.get(ep.get("split", "?"), 0) + 1
    print(f"\nsplits={splits}")
    train = [e for e in eps if e.get("split") == "train" and e.get("trace")]
    batch = train[:4]
    payload = [{"episode_id": e["episode_id"], "task_description": e.get("task_description", "")[:1500],
                "success": bool(e.get("success")),
                "trace": [{"step_id": r["step_id"], "action": r["action"],
                           "raw_observation": str(r["raw_observation"])[:1500],
                           "step_feedback": r["step_feedback"]} for r in e["trace"]]}
               for e in batch]
    print(f"reviewing {len(batch)} train episodes via 30B ...")
    try:
        raw = chat(TRAJECTORY_SYSTEM, trajectory_batch_user_prompt(episodes=payload))
        print("--- 30B raw (first 1200 chars) ---")
        print(raw[:1200])
        t = raw.strip()
        val = json.JSONDecoder().raw_decode(t[t.index("{"):])[0]
        corr = val.get("corrections", []) if isinstance(val, dict) else []
        print(f"\n>>> parsed corrections: {len(corr)}")
        for c in corr[:3]:
            print("   -", json.dumps(c, ensure_ascii=False)[:200])
    except Exception as exc:
        print(f">>> REVIEW ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
