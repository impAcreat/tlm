"""EFM feedback quality eval v2 — 4 deterministic-leaning dimensions.

Consistency : every asserted entity/state is grounded in the CURRENT raw
              observation (or the current action). Absence claims ("no X") are
              legitimate, not hallucinations. Reciting earlier history as if it
              were the current observation counts as inconsistent.
Completeness: when the observation states a concrete fact, the feedback must
              report it (defaulting to ambiguity on a concrete obs = under-report).
Efficiency  : concise, no verbatim restatement, no meta phrases.
Effectiveness: ONLINE-ONLY (needs the agent counterfactual) -> not scored here.

Runs on a records file (feedback + raw_observation + action + recent_actions),
so it needs no model calls and is fully reproducible.
"""
from __future__ import annotations

import json
import re
import sys

ENTITY = re.compile(r"\b([a-z]{3,}) (\d+)\b")
STOP_ENTITY = {"step"}
# Only state-TRANSFORMATIONS that are real hallucination risks; locomotion ("moved
# to <place>"), pick/take and open/close are handled by entity + possession checks.
TRANSFORM = ["cleaned", "cooled", "heated", "sliced"]
OBS_VERB = {"clean": "cleaned", "cool": "cooled", "heat": "heated", "slice": "sliced",
            "open": "opened", "close": "closed", "pick up": "picked up", "take": "took",
            "move": "moved", "put": "put", "place": "placed", "arrive": "arrived"}
META = re.compile(r"raw[_ ]observation|supplied data|step[_ ]?id|recent[_ ]actions|"
                  r"in step \d+|the observation confirms|the task('s| ) ", re.I)
CONCRETE_OBS = re.compile(r"you (arrive|open|close|pick up|take|move|put|clean|cool|heat|see)|"
                          r"is (open|closed)|you see nothing|not carrying anything", re.I)


def entities(text):
    return {f"{n} {d}" for n, d in ENTITY.findall(text.lower()) if n not in STOP_ENTITY}


def negated(text, ent):
    """True if entity appears only in an absence/negation context."""
    low = text.lower()
    for m in re.finditer(re.escape(ent), low):
        window = low[max(0, m.start() - 30):m.end() + 20]
        if not re.search(r"\bno\b|\bnot\b|n't|absent|without|no longer|"
                         r"is not|are not|not present|not observed|not visible", window):
            return False  # at least one non-negated mention
    return True


def current_verbs(action, obs):
    low = (action + " " + obs).lower()
    found = set()
    for raw, norm in OBS_VERB.items():
        if raw in low:
            found.add(norm)
    return found


def consistency(fb_text, action, obs, recent):
    obs_ents = entities(obs)
    cur_verbs = current_verbs(action, obs)
    fb = fb_text.lower()
    issues = []
    # 1) hallucinated entity: in feedback, not in obs, not negated
    for ent in entities(fb_text):
        if ent in obs_ents:
            continue
        if negated(fb_text, ent):
            continue
        issues.append(f"entity '{ent}' not in obs")
    # 2) ungrounded transform claim: asserts a transform not in current action/obs
    for verb in TRANSFORM:
        if verb in fb and verb not in cur_verbs:
            # allow if clearly negated ("not moved")
            if re.search(r"(no|not|n't)\s+\w*\s*" + re.escape(verb), fb):
                continue
            issues.append(f"transform '{verb}' not in current step")
    # 3) possession contradiction: a POSITIVE carrying claim vs "not carrying anything"
    if "not carrying anything" in obs.lower():
        positive_carry = re.search(r"\b(carrying|holding|in .{0,12}possession)\b", fb)
        denies = re.search(r"not\s+carry|isn't\s+carry|no\s+items|nothing", fb)
        if positive_carry and not denies:
            issues.append("claims carrying but obs says not carrying")
    return (1 if not issues else 0), issues


def completeness(fb_signal_type, obs):
    if CONCRETE_OBS.search(obs) and fb_signal_type == "ambiguity":
        return 0
    return 1


def efficiency(fb_text):
    if len(fb_text) > 180:
        return 0
    if META.search(fb_text):
        return 0
    return 1


def main(path):
    records = json.load(open(path))
    agg = {c: {"cons": 0, "comp": 0, "eff": 0} for c in "ABC"}
    rows = []
    for i, rec in enumerate(records):
        line = {"i": i, "action": rec["action"]}
        for c in "ABC":
            fb = rec["feedback"][c]
            cons, issues = consistency(fb["core_signal"], rec["action"], rec["raw_observation"], rec.get("recent_actions", []))
            comp = completeness(fb["signal_type"], rec["raw_observation"])
            eff = efficiency(fb["core_signal"])
            agg[c]["cons"] += cons; agg[c]["comp"] += comp; agg[c]["eff"] += eff
            line[c] = {"cons": cons, "comp": comp, "eff": eff, "issues": issues,
                       "judge30b_faithful": rec["judge_30b"][c]["faithful"]}
        rows.append(line)
    n = len(records)
    print(f"n={n}  (deterministic; effectiveness=N/A online)\n")
    print(f"{'cell':<5}{'consistency%':>14}{'completeness%':>15}{'efficiency%':>13}")
    for c in "ABC":
        d = agg[c]
        print(f"{c:<5}{100*d['cons']/n:>14.0f}{100*d['comp']/n:>15.0f}{100*d['eff']/n:>13.0f}")

    # validation: where v2 consistency disagrees with 30B faithful
    print("\n--- v2 consistency vs 30B faithful (disagreements) ---")
    for line in rows:
        for c in "ABC":
            v2 = line[c]["cons"]; j = 1 if line[c]["judge30b_faithful"] else 0
            if v2 != j:
                tag = "v2=OK/30b=halluc" if v2 else "v2=halluc/30b=OK"
                print(f"#{line['i']:>2}{c} [{tag}] issues={line[c]['issues']} | act='{line['action']}'")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "research/efm/diagnostics/eval_records.json")
