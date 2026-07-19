"""Build the paired-trajectory manifest for latent-space analysis.

Joins the three selection_eval result sets (same 140 valid_seen tasks under
rough_v1 / step-1 candidate / step-2 candidate skills) and records pair
categories for the skill-driven contrasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN_ROOT = Path("/data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714")

CONDITIONS = {
    "v0000": RUN_ROOT / "selection_eval_baseline",
    "step1": RUN_ROOT / "steps/step_0001/selection_eval",
    "step2": RUN_ROOT / "steps/step_0002/selection_eval",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cond: dict[str, dict[str, dict]] = {}
    for cond, base in CONDITIONS.items():
        rows = read_jsonl(base / "results.jsonl")
        entries = {}
        for row in rows:
            tid = row["id"]
            conv = base / "predictions" / tid / "conversation.json"
            if not conv.exists():
                raise FileNotFoundError(conv)
            trace = json.loads(conv.read_text())
            entries[tid] = {
                "conversation": str(conv),
                "hard": int(row["hard"]),
                "n_turns": int(row["n_turns"]),
                "trace_steps": len(trace),
                "task_type": row["task_type"],
                "task_description": row["task_description"],
                "gamefile": row["gamefile"],
            }
        per_cond[cond] = entries

    ids = sorted(per_cond["v0000"])
    for cond in CONDITIONS:
        assert sorted(per_cond[cond]) == ids, f"id mismatch in {cond}"
    for tid in ids:
        gfs = {per_cond[c][tid]["gamefile"] for c in CONDITIONS}
        assert len(gfs) == 1, f"gamefile mismatch for {tid}"

    def pair_category(a: int, b: int) -> str:
        return {(0, 1): "repaired", (1, 0): "broken", (1, 1): "both_success", (0, 0): "both_fail"}[(a, b)]

    pairs = {}
    for other in ("step1", "step2"):
        cats: dict[str, list[str]] = {"repaired": [], "broken": [], "both_success": [], "both_fail": []}
        for tid in ids:
            cats[pair_category(per_cond["v0000"][tid]["hard"], per_cond[other][tid]["hard"])].append(tid)
        pairs[f"v0000_vs_{other}"] = cats

    manifest = {
        "run_root": str(RUN_ROOT),
        "conditions": {c: str(p) for c, p in CONDITIONS.items()},
        "ids": ids,
        "tasks": {tid: {c: per_cond[c][tid] for c in CONDITIONS} for tid in ids},
        "pairs": pairs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {c: sum(v["hard"] for v in per_cond[c].values()) for c in CONDITIONS}
    print("successes per condition (n=%d):" % len(ids), summary)
    for key, cats in pairs.items():
        print(key, {k: len(v) for k, v in cats.items()})
    lengths = {c: sorted(v["trace_steps"] for v in per_cond[c].values()) for c in CONDITIONS}
    for c, ls in lengths.items():
        print(c, "steps min/med/max:", ls[0], ls[len(ls) // 2], ls[-1])
    by_type: dict[str, int] = {}
    for tid in ids:
        by_type[per_cond["v0000"][tid]["task_type"]] = by_type.get(per_cond["v0000"][tid]["task_type"], 0) + 1
    print("task types:", by_type)


if __name__ == "__main__":
    main()
