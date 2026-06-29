"""Disentangle model vs skill for EFM feedback quality (offline, no rollouts).

Cells:
  A = 4B  + current skill (reuse saved baseline feedback)
  B = 4B  + hand-authored policy rules (the 'learnable skill' lever)
  C = 30B + current skill
Blind judge = 30B scores faithful / signal_ok / vacuous per candidate.
"""
from __future__ import annotations

import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data5/ninghan/tlm")
from research.efm.constitution import build_step_system
from research.efm.prompts import step_user_prompt
from research.efm.policy import EFMPolicy, PolicyRule

STATE = ("/data5/ninghan/tlm/benchmarks/skillopt/outputs/"
         "qwen35_4b_alfworld_efm_gpu1_parallel_20260624/feedback_state.json")
SMALL = ("http://localhost:8007/v1/chat/completions", "Qwen/Qwen3.5-4B")
STRONG = ("http://localhost:8001/v1/chat/completions",
          "/data2/qinhao/LLM_weight/Qwen/Qwen3-30B-A3B-Instruct-2507/")
N = 30
ALLOWED = {"progress", "constraint_violated", "tool_error", "ambiguity", "state_change"}


def chat(endpoint_model, system, user, max_tokens):
    endpoint, model = endpoint_model
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer token-abc123"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def parse_fb(raw):
    t = str(raw or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    try:
        v = json.JSONDecoder().raw_decode(t[t.index("{"):])[0]
    except Exception:
        return {"core_signal": "(unparseable)", "signal_type": "ambiguity"}
    st = str(v.get("signal_type", "ambiguity"))
    return {"core_signal": str(v.get("core_signal", "")).strip(),
            "signal_type": st if st in ALLOWED else "ambiguity"}


# --- B: hand-authored policy rules targeting over-hedge + signal misclassification ---
B_POLICY = EFMPolicy(version=1, rules=[
    PolicyRule(
        id="b-anti-hedge", scope={"environment_id": "alfworld"},
        instruction=("When the raw observation literally states a new location, an "
                     "open/closed state, or lists the items in a receptacle/container, "
                     "report that concrete state or the presence/absence of the "
                     "task-relevant object as the core_signal and label it state_change "
                     "or progress."),
        avoid=("Returning ambiguity when the observation explicitly states a location, "
               "an open/closed state, or an item list."),
        support_episode_ids=["x"]),
    PolicyRule(
        id="b-signal-discipline", scope={"environment_id": "alfworld"},
        instruction=("Use constraint_violated only when the observation itself reports a "
                     "failed, blocked, or no-effect action. An action whose observation "
                     "confirms its intended change is progress or state_change."),
        avoid=("Labeling a successfully executed, observation-confirmed action as "
               "constraint_violated because it looks inconsistent with the task goal."),
        support_episode_ids=["x"]),
])

JUDGE_SYS = (
    "You strictly evaluate environment-feedback quality for ALFWorld (a text game). "
    "You are given the agent action, the raw environment observation, and candidate "
    "one-line feedbacks labeled X, Y, Z. For EACH candidate return three booleans: "
    "faithful (every claim is supported by the raw observation; inventing facts such as "
    "'cleaned', 'heated', or future outcomes is unfaithful), signal_ok (signal_type fits: "
    "state_change/progress for a successful observed change; constraint_violated ONLY if "
    "the observation reports a failed/blocked action; ambiguity ONLY if the observation is "
    "empty or contradictory), vacuous (true if it restates nothing useful or claims it "
    "cannot determine despite a clear observation). Return one JSON object: "
    '{"X":{"faithful":bool,"signal_ok":bool,"vacuous":bool}, "Y":{...}, "Z":{...}}.')


def build_transitions():
    eps = json.load(open(STATE))["episodes"]
    rows = []
    for ep in eps:
        prior = []
        for row in ep.get("trace", []):
            rows.append({
                "task": ep.get("task_description", ""), "env": ep.get("environment_id", ""),
                "task_type": ep.get("task_type", ""), "action": row["action"],
                "raw": row["raw_observation"], "recent": prior[-4:],
                "baseline": row["step_feedback"], "step_id": row["step_id"]})
            prior.append(row["action"])
    random.Random(7).shuffle(rows)
    return rows[:N]


def gen(endpoint_model, policy, tr):
    sys_p = build_step_system(policy, environment_id=tr["env"], task_type=tr["task_type"], action=tr["action"])
    user_p = step_user_prompt(task_description=tr["task"][:2000], action=tr["action"][:1000],
                              raw_observation=tr["raw"][:6000], step_id=tr["step_id"], recent_actions=tr["recent"])
    return parse_fb(chat(endpoint_model, sys_p, user_p, 192))


def main():
    trs = build_transitions()

    def work(tr):
        a = parse_fb(json.dumps(tr["baseline"]))
        a = {"core_signal": tr["baseline"].get("core_signal", ""), "signal_type": tr["baseline"].get("signal_type", "ambiguity")}
        b = gen(SMALL, B_POLICY, tr)
        c = gen(STRONG, EFMPolicy(), tr)
        cells = {"A": a, "B": b, "C": c}
        # judge
        labels = ["X", "Y", "Z"]
        order = labels[:]
        random.Random(tr["step_id"] * 13 + len(tr["action"])).shuffle(order)
        mapping = dict(zip(order, ["A", "B", "C"]))
        cand = {lab: f'[{cells[mapping[lab]]["signal_type"]}] {cells[mapping[lab]]["core_signal"]}' for lab in labels}
        ju = json.dumps({"action": tr["action"], "raw_observation": tr["raw"][:1500], "candidates": cand}, ensure_ascii=False)
        raw = chat(STRONG, JUDGE_SYS, ju, 400)
        try:
            t = raw.strip()
            verdict = json.JSONDecoder().raw_decode(t[t.index("{"):])[0]
        except Exception:
            verdict = {}
        scored = {}
        for lab, cell in mapping.items():
            v = verdict.get(lab, {}) if isinstance(verdict, dict) else {}
            scored[cell] = {"faithful": bool(v.get("faithful")), "signal_ok": bool(v.get("signal_ok")),
                            "vacuous": bool(v.get("vacuous"))}
        return cells, scored

    agg = {c: {"faithful": 0, "signal_ok": 0, "vacuous": 0, "hedge": 0} for c in "ABC"}
    n_ok = 0
    records = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for tr, (cells, scored) in zip(trs, ex.map(work, trs)):
            n_ok += 1
            records.append({
                "action": tr["action"], "raw_observation": tr["raw"],
                "task": tr["task"], "recent_actions": tr["recent"],
                "feedback": {c: cells[c] for c in "ABC"},
                "judge_30b": scored,
            })
            for c in "ABC":
                agg[c]["faithful"] += scored[c]["faithful"]
                agg[c]["signal_ok"] += scored[c]["signal_ok"]
                agg[c]["vacuous"] += scored[c]["vacuous"]
                agg[c]["hedge"] += 1 if cells[c]["signal_type"] == "ambiguity" else 0
    with open("research/efm/diagnostics/eval_records.json", "w") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    print(f"n={n_ok}  (A=4B+cur skill, B=4B+authored policy, C=30B+cur skill)\n")
    print(f"{'cell':<6}{'faithful%':>10}{'signal_ok%':>12}{'vacuous%':>10}{'hedge%':>9}")
    for c in "ABC":
        d = agg[c]
        print(f"{c:<6}{100*d['faithful']/n_ok:>10.0f}{100*d['signal_ok']/n_ok:>12.0f}"
              f"{100*d['vacuous']/n_ok:>10.0f}{100*d['hedge']/n_ok:>9.0f}")


if __name__ == "__main__":
    main()
