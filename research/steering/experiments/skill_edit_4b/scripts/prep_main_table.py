"""Prep for the full main table.

1) Extend manifest with the 134 valid_unseen tasks (gamefiles/types from
   test_eval_baseline; reference hard labels: baseline + best).
2) Train T heads for L14/18/22 on all 298 hints, compile the ENTIRE step-2
   skill document text into per-layer vectors, report cos vs extracted gmb,
   and save injection-ready files (rows scaled to extracted gmb per-layer
   norms; dose calibration is deployment-side, direction is T's).
3) Save a matching random-direction file (seed 7) at the same norms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
LAYERS = (14, 18, 22)
LAYERS_IN = (14, 18, 22)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # ---- 1) manifest extension ----
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if not any(t.startswith("test:") for t in manifest["ids"]):
        base_rows = {}
        for line in (RUN_ROOT / "test_eval_baseline/results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                base_rows[r["id"]] = r
        best_rows = {}
        for line in (RUN_ROOT / "test_eval/results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                best_rows[r["id"]] = r
        for tid, r in sorted(base_rows.items()):
            manifest["tasks"][tid] = {"v0000": {
                "gamefile": r["gamefile"], "task_type": r["task_type"],
                "task_description": r["task_description"], "hard": int(r["hard"]),
                "conversation": "", "n_turns": int(r["n_turns"]), "trace_steps": 0,
            }, "unseen_best_hard": int(best_rows[tid]["hard"]) if tid in best_rows else None}
            manifest["ids"].append(tid)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"manifest extended: +{len(base_rows)} unseen tasks "
              f"(baseline {sum(r['hard'] for r in base_rows.values())}/134, "
              f"best {sum(r['hard'] for r in best_rows.values())}/134)")
    else:
        print("manifest already extended")
    seen = [t for t in manifest["ids"] if t.startswith("val:")]
    unseen = [t for t in manifest["ids"] if t.startswith("test:")]
    (out_dir / "tasks_seen140.txt").write_text("\n".join(seen))
    (out_dir / "tasks_unseen134.txt").write_text("\n".join(unseen))

    # ---- 2) T doc-level compilation ----
    from research.steering.adapters.models.loading import load_causal_lm  # noqa: E402

    tdir = out_dir / "t_dataset"
    units = {}
    for f in sorted(tdir.glob("unit_vectors_shard*.pt")):
        units.update(torch.load(f, weights_only=False))
    hints = [r for r in units.values() if r["source"] == "reflexion"]
    X = np.concatenate([np.stack([r["text_mean"][L].float().numpy() for r in hints])
                        for L in LAYERS_IN], axis=1)

    model, tokenizer = load_causal_lm(args.model_path, args.device)
    doc = (RUN_ROOT / "steps/step_0002/candidate_skill.md").read_text()
    ids = tokenizer(doc, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)
    with torch.no_grad():
        outp = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    doc_rep = {L: outp.hidden_states[L + 1][0].float().mean(dim=0).cpu().numpy() for L in LAYERS_IN}
    xdoc = np.concatenate([doc_rep[L] for L in LAYERS_IN])[None]

    gmb = torch.load(out_dir / "vectors/gmb_raw_means.pt", weights_only=False)["mean_delta"].float().numpy()

    t_means = np.zeros_like(gmb)
    rnd_means = np.zeros_like(gmb)
    g = torch.Generator().manual_seed(7)
    report = {}
    for L in LAYERS:
        Y = np.stack([r["vector"][L].float().numpy() for r in hints])
        head = Ridge(alpha=100.0)
        head.fit(X, Y)
        pred = head.predict(xdoc)[0]
        c = float(np.dot(unit(pred), unit(gmb[L])))
        norm = float(np.linalg.norm(gmb[L]))
        t_means[L] = unit(pred) * norm
        rv = torch.randn(gmb.shape[1], generator=g).numpy()
        rnd_means[L] = unit(rv) * norm
        report[f"L{L}"] = {"cos_Tdoc_vs_extracted_gmb": c, "gmb_norm": norm,
                           "pred_norm_raw": float(np.linalg.norm(pred))}
    torch.save({"mean_delta": torch.tensor(t_means)}, out_dir / "vectors/t_doc_means.pt")
    torch.save({"mean_delta": torch.tensor(rnd_means)}, out_dir / "vectors/random_means.pt")
    (out_dir / "vectors/t_doc_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
