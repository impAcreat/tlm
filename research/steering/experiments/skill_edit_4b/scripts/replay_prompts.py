"""Replay recorded v0000 selection rollouts through the real ALFWorld env to
recover the exact per-step observation text (prompt body) the agent saw.

Runs in the skillopt cu128 env (has alfworld). CPU only. Output: prompts.jsonl
with one row per (task, step<=MAX_STEP) state, plus a replay fidelity report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for p in (str(ROOT), str(SKILLOPT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("ALFWORLD_DATA", str(SKILLOPT_ROOT / "data" / "alfworld_data"))
os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")

from skillopt.envs.alfworld.rollout import build_alfworld_env  # noqa: E402

MAX_STEP = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", default="v0000")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    ids = manifest["ids"]
    if args.limit:
        ids = ids[: args.limit]

    cat_of = {}
    for cat, tids in manifest["pairs"]["v0000_vs_step2"].items():
        for t in tids:
            cat_of[t] = cat

    out_path = out_dir / f"prompts_{args.condition}.jsonl"
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["task_id"])

    report = {"episodes": 0, "steps": 0, "feedback_matches": 0, "feedback_total": 0, "errors": []}
    with out_path.open("a") as fout:
        for n, tid in enumerate(ids):
            if tid in done_ids:
                continue
            entry = manifest["tasks"][tid][args.condition]
            trace = json.loads(Path(entry["conversation"]).read_text())
            gamefile = entry["gamefile"]
            eval_dataset = "eval_in_distribution" if "/valid_seen/" in gamefile else "eval_out_of_distribution"
            try:
                env = build_alfworld_env(env_num=1, eval_dataset=eval_dataset, seed=42,
                                         is_train=False, specific_gamefiles=[gamefile])
                obs, infos = env.reset({})
                rows = []
                for step in trace:
                    sidx = int(step["step"])
                    if sidx <= MAX_STEP:
                        rows.append({
                            "task_id": tid,
                            "step": sidx,
                            "obs_text": obs["text"][0],
                            "category": cat_of[tid],
                            "task_type": entry["task_type"],
                            "hard": entry["hard"],
                        })
                    if sidx >= MAX_STEP and sidx >= min(len(trace) - 1, MAX_STEP):
                        pass
                    obs, rewards, dones, infos = env.step([step["model_response"]])
                    fb = str(step.get("env_feedback") or "").strip()
                    anchor = str(obs.get("anchor", [""])[0]).strip()
                    report["feedback_total"] += 1
                    if fb and fb in anchor or fb == anchor:
                        report["feedback_matches"] += 1
                    if bool(dones[0].item() if hasattr(dones[0], "item") else dones[0]):
                        break
                    if sidx >= MAX_STEP:
                        break
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                report["episodes"] += 1
                report["steps"] += len(rows)
            except Exception as exc:  # noqa: BLE001
                report["errors"].append({"task_id": tid, "error": str(exc)[:300]})
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            if (n + 1) % 10 == 0:
                print(f"{n + 1}/{len(ids)} episodes; steps={report['steps']} "
                      f"fb_match={report['feedback_matches']}/{report['feedback_total']}", flush=True)

    (out_dir / f"replay_report_{args.condition}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
    print("errors:", len(report["errors"]))


if __name__ == "__main__":
    main()
