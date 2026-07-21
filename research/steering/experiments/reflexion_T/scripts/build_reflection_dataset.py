#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--phase1", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(str(Path(a.phase1)/"results_shard*.jsonl"))):
    rows += [json.loads(x) for x in open(f) if x.strip()]
out = []
for r in rows:
    for i, (text, retry) in enumerate(zip(r["reflections"], r["reflex_retries"])):
        out.append({
            "unit_id": f"hint_{r['id'].replace(':', '')}_{i}",
            "episode_id": r["id"],
            "group_id": r.get("group_id", r["id"]),
            "task_id": r.get("task_id", r["id"]),
            "task_seed": r.get("task_seed", 0),
            "split": r.get("split", "legacy"),
            "source": "reflexion",
            "application": "reflection_prefix",
            "text": text,
            "task_type": r["task_type"],
            "retry_index": i,
            "text_success": bool(retry["hard"]),
            "paired_effective": bool(retry["hard"] and not r["control_any"]),
        })
td = Path(a.out_dir) / "t_dataset"
td.mkdir(parents=True, exist_ok=True)
with (td/"hints.jsonl").open("w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
summary = {
    "units": len(out),
    "groups": len(set(x["group_id"] for x in out)),
    "tasks": len(set(x["task_id"] for x in out)),
    "text_success": sum(x["text_success"] for x in out),
    "paired_effective": sum(x["paired_effective"] for x in out),
    "by_split": {
        split: {
            "units": sum(x["split"] == split for x in out),
            "text_success": sum(x["split"] == split and x["text_success"] for x in out),
            "paired_effective": sum(x["split"] == split and x["paired_effective"] for x in out),
        }
        for split in sorted(set(x["split"] for x in out))
    },
}
(td/"hints_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary)
