"""Generate reflexion-style unit hints from failed rollouts with GPT-5.5.

For each failed episode (v0000 + step1 selection rollouts), ask the optimizer
model for two DISTINCT, generalizable, unit-granularity hints (1-3 sentences,
imperative, no task-instance specifics). Output: t_dataset/hints.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

RUN_ROOT = Path(__file__).resolve().parents[5] / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"

SYSTEM = (
    "You analyze failed episodes of a household agent (ALFWorld) and write skill hints. "
    "A hint must be a GENERAL, reusable behavioral rule (imperative, 1-3 sentences) in the style of a "
    "skill-document bullet point. Never mention instance-specific ids (like 'drawer 3'), only object/"
    "receptacle categories and strategies. The two hints must target different failure aspects."
)

PROMPT = """Task: {task}

Trajectory (action -> observation, truncated):
{digest}

Result: FAILED (ran out of steps).

Write exactly two distinct generalizable hints that would have prevented this failure.
Reply as JSON: {{"hints": ["...", "..."]}}"""


def digest(trace: list[dict], max_steps: int = 25, max_obs: int = 90) -> str:
    rows = []
    for s in trace[:max_steps]:
        a = str(s.get("action") or "").strip()
        o = str(s.get("env_feedback") or "").strip().replace("\n", " ")[:max_obs]
        rows.append(f"{a} -> {o}")
    if len(trace) > max_steps:
        rows.append(f"... ({len(trace) - max_steps} more steps, mostly repetitive)")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) / "t_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hints.jsonl"

    client = OpenAI(base_url=os.environ["OPTIMIZER_BASE_URL"],
                    api_key=os.environ["OPTIMIZER_API_KEY"])

    manifest = json.loads((Path(args.out_dir) / "manifest.json").read_text())
    episodes = []
    for cond in ("v0000", "step1"):
        for tid in manifest["ids"]:
            e = manifest["tasks"][tid][cond]
            if e["hard"] == 0:
                episodes.append((cond, tid, e))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"{len(episodes)} failed episodes", flush=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["episode"])

    def work(item):
        cond, tid, e = item
        key = f"{cond}/{tid}"
        if key in done:
            return None
        trace = json.loads(Path(e["conversation"]).read_text())
        msg = PROMPT.format(task=e["task_description"], digest=digest(trace))
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": msg}],
                    max_completion_tokens=4096,
                )
                text = resp.choices[0].message.content or ""
                m = re.search(r"\{.*\}", text, re.DOTALL)
                hints = json.loads(m.group(0))["hints"]
                assert isinstance(hints, list) and len(hints) >= 2
                return key, cond, tid, e["task_type"], [str(h).strip() for h in hints[:2]]
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    return key, cond, tid, e["task_type"], {"error": str(exc)[:200]}
                time.sleep(3 * (attempt + 1))

    seen_hash = set()
    n_ok = n_err = n_dup = 0
    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, it) for it in episodes]
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res is None:
                continue
            key, cond, tid, ttype, hints = res
            if isinstance(hints, dict):
                n_err += 1
                fout.write(json.dumps({"episode": key, "error": hints["error"]}) + "\n")
                continue
            for j, h in enumerate(hints):
                hh = hashlib.md5(re.sub(r"\W+", "", h.lower()).encode()).hexdigest()
                if hh in seen_hash:
                    n_dup += 1
                    continue
                seen_hash.add(hh)
                fout.write(json.dumps({
                    "unit_id": f"hint_{cond}_{tid.replace(':', '')}_{j}",
                    "episode": key, "source": "reflexion", "condition": cond,
                    "task_type": ttype, "text": h,
                }, ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(episodes)} ok={n_ok} dup={n_dup} err={n_err}", flush=True)
    print(f"done: hints={n_ok} dup={n_dup} err={n_err}")


if __name__ == "__main__":
    main()
