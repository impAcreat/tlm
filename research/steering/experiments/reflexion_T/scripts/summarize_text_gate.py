#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = Path(args.out_dir)
    rows = []
    for path in sorted(glob.glob(str(root / "results_shard*.jsonl"))):
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())

    duplicate_ids = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
    errors = []
    for row in rows:
        for arm, attempts in (
            ("initial", [row["initial"]]),
            ("control", row["control_retries"]),
            ("reflex", row["reflex_retries"]),
        ):
            for i, attempt in enumerate(attempts):
                for step in attempt.get("trace", []):
                    if "error" in step:
                        errors.append({"id": row["id"], "arm": arm, "attempt": i, "error": step["error"]})

    eligible = [r for r in rows if r["initial_failed"] and not any(e["id"] == r["id"] for e in errors)]
    n = len(eligible)
    control = sum(r["control_any"] for r in eligible)
    reflex = sum(r["reflex_any"] for r in eligible)
    reflex_only = sum(r["reflex_any"] and not r["control_any"] for r in eligible)
    control_only = sum(r["control_any"] and not r["reflex_any"] for r in eligible)
    both = sum(r["control_any"] and r["reflex_any"] for r in eligible)
    neither = n - reflex_only - control_only - both

    by_type = defaultdict(lambda: {"n": 0, "control": 0, "reflex": 0})
    for r in eligible:
        d = by_type[r["task_type"]]
        d["n"] += 1
        d["control"] += r["control_any"]
        d["reflex"] += r["reflex_any"]

    summary = {
        "rows": len(rows),
        "initial_successes": sum(r["initial"]["hard"] for r in rows),
        "eligible_initial_failures": n,
        "control_successes": control,
        "control_rate": control / n if n else None,
        "reflex_successes": reflex,
        "reflex_rate": reflex / n if n else None,
        "absolute_lift": (reflex - control) / n if n else None,
        "gate_G1_pass": bool(n and (reflex - control) / n >= 0.15),
        "paired": {
            "reflex_only": reflex_only,
            "control_only": control_only,
            "both": both,
            "neither": neither,
        },
        "by_task_type": dict(by_type),
        "duplicate_ids": duplicate_ids,
        "runtime_errors": errors,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
