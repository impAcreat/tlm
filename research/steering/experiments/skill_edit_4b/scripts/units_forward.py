"""Batch-extract ground-truth conditioning vectors + text reps for T-dataset units.

Units = reflexion hints (t_dataset/hints.jsonl) + skillopt edit units (all ranked
edits of steps 1-3, uniform treatment). Every unit is applied the same way:
appended to the step-1 base skill under an "## Additional Hints" section.

Per unit we store (fp16):
  vector   [32, 2560]  mean over states of h(base+unit) - h(base), last token
  cons_l14/l18         cross-state consistency of the per-state deltas
  text_mean [32, 2560] mean-pooled hidden of the unit text alone
  text_last [32, 2560] last-token hidden of the unit text alone

Supports --shard k/n for multi-GPU splitting; shards write separate files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for p in (str(ROOT), str(SKILLOPT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.steering.adapters.models.loading import load_causal_lm  # noqa: E402
from research.steering.experiments.skill_edit_4b.scripts.prompt_forward import build_skill_prompt, last_token_layers  # noqa: E402

RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
N_STATES = 200


def unit_text_block(text: str) -> str:
    return f"\n\n## Additional Hints\n- {text.strip()}\n"


@torch.no_grad()
def text_reps(model, tokenizer, device, text):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    mean = torch.stack([h[0].mean(dim=0) for h in out.hidden_states[1:]], dim=0)
    last = torch.stack([h[0, -1] for h in out.hidden_states[1:]], dim=0)
    return mean.float().cpu(), last.float().cpu()


def collect_units(out_dir: Path) -> list[dict]:
    units = []
    hints_path = out_dir / "t_dataset" / "hints.jsonl"
    for line in hints_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "unit_id" in row:
            units.append({"unit_id": row["unit_id"], "source": "reflexion",
                          "text": row["text"], "task_type": row.get("task_type")})
    for step in ("step_0001", "step_0002", "step_0003"):
        ranked = json.loads((RUN_ROOT / "steps" / step / "ranked_edits.json").read_text())
        for i, e in enumerate(ranked["edits"]):
            content = str(e.get("content") or "").strip()
            if content:
                units.append({"unit_id": f"skillopt_{step}_e{i}", "source": "skillopt",
                              "text": content, "task_type": None})
    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", default="0/1", help="k/n")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    tdir = out_dir / "t_dataset"
    k, n = (int(x) for x in args.shard.split("/"))

    units = collect_units(out_dir)
    units = [u for i, u in enumerate(units) if i % n == k]
    out_path = tdir / f"unit_vectors_shard{k}.pt"
    done = {}
    if out_path.exists():
        done = torch.load(out_path, weights_only=False)
    print(f"shard {k}/{n}: {len(units)} units, {len(done)} done", flush=True)

    lad = torch.load(out_dir / "ladder_reps.pt", weights_only=False)
    idx600 = np.array(lad["subsample_indices"])
    sel = np.arange(N_STATES)  # first 200 of the 600-subsample
    base_reps = lad["reps"]["s1"].float()[sel]  # [200, 32, d]

    states = [json.loads(l) for l in (out_dir / "prompts_v0000.jsonl").read_text().splitlines() if l.strip()]
    sub = [states[i] for i in idx600[sel]]

    s1_text = (RUN_ROOT / "steps/step_0001/candidate_skill.md").read_text()

    model, tokenizer = load_causal_lm(args.model_path, args.device)

    t0 = time.time()
    n_done = 0
    for u in units:
        if u["unit_id"] in done:
            continue
        full = s1_text + unit_text_block(u["text"])
        buf = []
        for st in sub:
            user = build_skill_prompt(full) + "\n" + st["obs_text"]
            buf.append(last_token_layers(model, tokenizer, args.device, user))
        rp = torch.stack(buf)  # [200, 32, d]
        D = rp - base_reps
        rec = {"unit_id": u["unit_id"], "source": u["source"], "task_type": u["task_type"],
               "text": u["text"], "vector": D.mean(0).half()}
        for L in (14, 18):
            uD = D[:, L] / D[:, L].norm(dim=1, keepdim=True).clamp_min(1e-12)
            sim = uD @ uD.T
            rec[f"cons_l{L}"] = float(sim[~torch.eye(len(uD), dtype=bool)].mean())
        tm, tl = text_reps(model, tokenizer, args.device, u["text"])
        rec["text_mean"] = tm.half()
        rec["text_last"] = tl.half()
        done[u["unit_id"]] = rec
        n_done += 1
        if n_done % 5 == 0:
            torch.save(done, out_path)
            print(f"[shard {k}] {n_done} new, total {len(done)}, {time.time() - t0:.0f}s", flush=True)
    torch.save(done, out_path)
    print(f"[shard {k}] finished: {len(done)} units ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
