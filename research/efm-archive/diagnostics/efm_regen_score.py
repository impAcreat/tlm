"""Regenerate step feedback under the OPTIMIZED initial skill (4B + empty policy +
optimized constitution) on the same 30 transitions, score with eval_v2, and
compare to saved A (old skill) and B (authored policy)."""
from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data5/ninghan/tlm")
import research.efm.constitution as C
importlib.reload(C)
from research.efm.prompts import step_user_prompt
from research.efm.policy import EFMPolicy
import research.efm.diagnostics.efm_eval_v2 as V

REC = "research/efm/diagnostics/eval_records.json"
SMALL = ("http://localhost:8007/v1/chat/completions", "Qwen/Qwen3.5-4B")
ALLOWED = {"progress", "constraint_violated", "tool_error", "ambiguity", "state_change"}


def chat(system, user):
    body = json.dumps({"model": SMALL[1],
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "max_tokens": 192, "temperature": 0.3,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(SMALL[0], data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer token-abc123"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(raw):
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


def score_cell(fb, rec):
    cons, _ = V.consistency(fb["core_signal"], rec["action"], rec["raw_observation"], rec.get("recent_actions", []))
    comp = V.completeness(fb["signal_type"], rec["raw_observation"])
    eff = V.efficiency(fb["core_signal"])
    return cons, comp, eff


def gen_opt(rec):
    sysp = C.build_step_system(EFMPolicy(), environment_id=rec.get("environment_id", "alfworld"),
                               task_type=rec.get("task_type", ""), action=rec["action"])
    usr = step_user_prompt(task_description=rec.get("task", "")[:2000], action=rec["action"][:1000],
                           raw_observation=rec["raw_observation"][:6000], step_id=0,
                           recent_actions=rec.get("recent_actions", []))
    return parse(chat(sysp, usr))


def main():
    records = json.load(open(REC))
    agg = {k: {"cons": 0, "comp": 0, "eff": 0} for k in ("A_old", "B_authored", "A2_optskill")}
    with ThreadPoolExecutor(max_workers=4) as ex:
        opt_fbs = list(ex.map(gen_opt, records))
    for rec, optfb in zip(records, opt_fbs):
        for key, fb in (("A_old", rec["feedback"]["A"]),
                        ("B_authored", rec["feedback"]["B"]),
                        ("A2_optskill", optfb)):
            c, p, e = score_cell(fb, rec)
            agg[key]["cons"] += c; agg[key]["comp"] += p; agg[key]["eff"] += e
    n = len(records)
    print(f"n={n}  (effectiveness=N/A online)\n")
    print(f"{'cell':<14}{'consistency%':>14}{'completeness%':>15}{'efficiency%':>13}")
    for k in ("A_old", "A2_optskill", "B_authored"):
        d = agg[k]
        print(f"{k:<14}{100*d['cons']/n:>14.0f}{100*d['comp']/n:>15.0f}{100*d['eff']/n:>13.0f}")


if __name__ == "__main__":
    main()
