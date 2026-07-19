"""Extract the protocol-domain unit vector v(P): step-3 ranked edit #1
("direct heat/cool/clean action priority at appliances") applied to the
step-1 base text, prompt-conditioned contrast on the same 600-state subsample.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data5/ninghan/tlm")
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for p in (str(ROOT), str(SKILLOPT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.steering.core.hf import load_causal_lm  # noqa: E402
from research.steering.latent.prompt_forward import build_skill_prompt, last_token_layers  # noqa: E402
from skillopt.optimizer.skill import apply_edit  # noqa: E402

RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
LAYERS = (14, 18, 22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    vec_dir = out_dir / "vectors"

    s1_text = (RUN_ROOT / "steps/step_0001/candidate_skill.md").read_text()
    edits = json.loads((RUN_ROOT / "steps/step_0003/ranked_edits.json").read_text())["edits"]
    edit_p = edits[1]
    text_p = apply_edit(s1_text, edit_p)
    applied_inline = edit_p["target"] in s1_text
    print("target-section exists in s1 text:", applied_inline,
          "| added chars:", len(text_p) - len(s1_text), flush=True)

    lad = torch.load(out_dir / "ladder_reps.pt", weights_only=False)
    idx = np.array(lad["subsample_indices"])
    s1_reps = lad["reps"]["s1"].float()  # [600, 32, d]

    states = []
    with (out_dir / "prompts_v0000.jsonl").open() as f:
        for line in f:
            if line.strip():
                states.append(json.loads(line))
    sub = [states[i] for i in idx]

    model, tokenizer = load_causal_lm(args.model_path, args.device)
    buf = []
    t0 = time.time()
    for j, st in enumerate(sub):
        user = build_skill_prompt(text_p) + "\n" + st["obs_text"]
        buf.append(last_token_layers(model, tokenizer, args.device, user).half())
        if (j + 1) % 150 == 0:
            print(f"{j + 1}/{len(sub)} elapsed {time.time() - t0:.0f}s", flush=True)
    rp = torch.stack(buf).float()  # [600, 32, d]

    V = rp - s1_reps  # per-state deltas
    norms = {}
    geo = {}
    for L in LAYERS:
        mu = V[:, L].mean(0)
        norms[f"P_l{L}"] = float(mu.norm())
        torch.save({"vector": mu / mu.norm().clamp_min(1e-12), "layer": L, "family": "unit_P_step3e1"},
                   vec_dir / f"v_unit_P_l{L}.pt")
        u = V[:, L] / V[:, L].norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = (u @ u.T)
        geo[f"P_consistency_l{L}"] = float(sim[~torch.eye(len(u), dtype=bool)].mean())
        for other in ("e0", "e1"):
            art = torch.load(vec_dir / f"v_unit_{other}_l{L}.pt", weights_only=False)
            geo[f"cos_P_{other}_l{L}"] = float(torch.dot(mu / mu.norm(), art["vector"].float()))
    (vec_dir / "unit_P_meta.json").write_text(json.dumps(
        {"applied_inline": applied_inline, "norms": norms, "geometry": geo,
         "edit_content": str(edit_p.get("content"))[:400]}, indent=2))
    torch.save({"reps": rp.half()}, out_dir / "unit_P_reps.pt")
    print(json.dumps({**{k: round(v, 3) for k, v in norms.items()},
                      **{k: round(v, 3) for k, v in geo.items()}}, indent=1))


if __name__ == "__main__":
    main()
