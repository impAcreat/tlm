"""Offline reproduction of the EFM policy proposal that was rejected as
`unsupported_edit`, to capture the raw candidate patch the optimizer emits.

Faithful to the run: same POLICY_SYSTEM + policy_user_prompt, same v0 policy,
same saved corrections, same Qwen3.5-4B backend (temp 0.7, no thinking).
"""
from __future__ import annotations

import json
import sys
import urllib.request

sys.path.insert(0, "/data5/ninghan/tlm")
from research.efm.prompts import POLICY_SYSTEM, policy_user_prompt
from research.efm.policy import EFMPolicy, PolicyPatch, validate_patch
from research.efm.runtime import _json_value

STATE = ("/data5/ninghan/tlm/benchmarks/skillopt/outputs/"
         "qwen35_4b_alfworld_efm_policy_smoke_gpu1_20260624/feedback_state.json")
ENDPOINT = "http://localhost:8007/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-4B"
N = 3


def call(prompt: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": POLICY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
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
    state = json.load(open(STATE))
    policy = EFMPolicy.from_dict(state["policy"])
    corrections = state["corrections"]
    prompt = policy_user_prompt(policy=policy.to_dict(), corrections=corrections, edit_budget=2)
    print(f"policy.version={policy.version}  n_corrections={len(corrections)}\n")
    for i in range(N):
        raw = call(prompt)
        print(f"===== SAMPLE {i} RAW =====")
        print(raw)
        try:
            value = _json_value(raw)
            patch = PolicyPatch.from_dict(value)
            err = validate_patch(patch, policy, min_support=3, max_edits=2)
            ops = [str(e.get("op", "<MISSING>")) for e in patch.edits]
            keys = [sorted(e.keys()) for e in patch.edits]
            print(f">>> base_version={patch.base_version}  n_edits={len(patch.edits)}")
            print(f">>> edit ops   = {ops}")
            print(f">>> edit keys  = {keys}")
            print(f">>> validate_patch -> {err!r}")
        except Exception as exc:  # noqa: BLE001
            print(f">>> parse error: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
