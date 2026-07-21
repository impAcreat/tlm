"""Round 4 forwards: step-ladder and unit-edit skill variants + skill-text self encodings.

Variants forwarded over a fixed 600-state subsample of the replayed states:
  s1            step-1 candidate text
  s1_e{i}       step-1 text + single ranked edit i of step 2 (skillopt apply_edit)
Also encodes each skill text alone (token mean pooling, all layers).

Existing reps reused elsewhere: bad/good(s2)/none in prompt_deltas.pt, good(s3) in
prompt_deltas_step3.pt (same state order; subsample indices saved here).
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
from research.steering.experiments.skill_edit_4b.scripts.prompt_forward import (  # noqa: E402
    SYSTEM_PROMPT,
    build_skill_prompt,
    last_token_layers,
)
from skillopt.optimizer.skill import apply_edit  # noqa: E402

RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
N_SUB = 600


@torch.no_grad()
def text_mean_layers(model, tokenizer, device: str, text: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    return torch.stack([h[0].mean(dim=0) for h in out.hidden_states[1:]], dim=0).float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    s1_text = (RUN_ROOT / "steps/step_0001/candidate_skill.md").read_text()
    s2_text = (RUN_ROOT / "steps/step_0002/candidate_skill.md").read_text()
    ranked = json.loads((RUN_ROOT / "steps/step_0002/ranked_edits.json").read_text())
    edits = ranked["edits"]

    texts = {"s1": s1_text}
    for i, edit in enumerate(edits):
        texts[f"s1_e{i}"] = apply_edit(s1_text, edit)
    seq = s1_text
    for edit in edits:
        seq = apply_edit(seq, edit)
    print("sequential-apply reproduces step-2 candidate:", seq == s2_text, flush=True)
    (out_dir / "ladder_apply_check.json").write_text(json.dumps({
        "sequential_equals_step2": seq == s2_text,
        "n_edits": len(edits),
        "unit_lengths": {k: len(v) for k, v in texts.items()},
        "s2_length": len(s2_text),
    }, indent=2))

    states = []
    with (out_dir / "prompts_v0000.jsonl").open() as f:
        for line in f:
            if line.strip():
                states.append(json.loads(line))
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(states), min(N_SUB, len(states)), replace=False))
    sub = [states[i] for i in idx]
    print(f"{len(sub)} / {len(states)} states", flush=True)

    model, tokenizer = load_causal_lm(args.model_path, args.device)

    reps = {}
    t0 = time.time()
    for name, content in texts.items():
        buf = []
        for j, st in enumerate(sub):
            user = build_skill_prompt(content) + "\n" + st["obs_text"]
            buf.append(last_token_layers(model, tokenizer, args.device, user).half())
            if (j + 1) % 100 == 0:
                print(f"[{name}] {j + 1}/{len(sub)} elapsed {time.time() - t0:.0f}s", flush=True)
        reps[name] = torch.stack(buf)
        print(f"[done] {name} ({time.time() - t0:.0f}s)", flush=True)

    skill_files = {
        "v0000": RUN_ROOT / "skills/skill_v0000.md",
        "s1": RUN_ROOT / "steps/step_0001/candidate_skill.md",
        "s2": RUN_ROOT / "steps/step_0002/candidate_skill.md",
        "s3": RUN_ROOT / "steps/step_0003/candidate_skill.md",
    }
    text_reps = {k: text_mean_layers(model, tokenizer, args.device, p.read_text()).half()
                 for k, p in skill_files.items()}
    for i in range(len(edits)):
        text_reps[f"s1_e{i}"] = text_mean_layers(model, tokenizer, args.device, texts[f"s1_e{i}"]).half()
    text_reps["s1_full"] = text_reps["s1"]
    # the edit contents alone (semantic units)
    for i, edit in enumerate(edits):
        text_reps[f"edit{i}_content"] = text_mean_layers(
            model, tokenizer, args.device, str(edit.get("content", ""))).half()

    torch.save({"subsample_indices": idx.tolist(), "reps": reps, "text_reps": text_reps,
                "edit_meta": [{"op": e.get("op"), "target": str(e.get("target"))[:120]} for e in edits]},
               out_dir / "ladder_reps.pt")
    print(f"saved ladder_reps.pt ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
