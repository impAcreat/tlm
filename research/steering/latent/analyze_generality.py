"""Cross-skill-text generality of the prompt-conditioned direction.

Compares the step-2-based deltas (prompt_deltas.pt) with the step-3-based
deltas (prompt_deltas_step3.pt) on identical states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

NUM_LAYERS = 32


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"

    p2 = torch.load(out_dir / "prompt_deltas.pt", weights_only=False)
    p3 = torch.load(out_dir / "prompt_deltas_step3.pt", weights_only=False)
    assert [s["task_id"] for s in p2["states"]] == [s["task_id"] for s in p3["states"]]

    out = {"per_layer": []}
    for L in range(NUM_LAYERS):
        row = {"layer": L}
        for key, name in (("good_minus_bad", "gmb"), ("good_minus_none", "gmn")):
            v2 = unit(p2["deltas"][key][:, L].float().numpy().mean(0))
            v3 = unit(p3["deltas"][key][:, L].float().numpy().mean(0))
            row[f"cos_{name}2_{name}3"] = float(np.dot(v2, v3))
        # controls: bad_minus_none should be identical by construction (same prompts)
        b2 = unit(p2["deltas"]["bad_minus_none"][:, L].float().numpy().mean(0))
        b3 = unit(p3["deltas"]["bad_minus_none"][:, L].float().numpy().mean(0))
        row["cos_bmn_control"] = float(np.dot(b2, b3))
        # cross-check: gmn2 vs gmb3 etc.
        g2n = unit(p2["deltas"]["good_minus_none"][:, L].float().numpy().mean(0))
        g3b = unit(p3["deltas"]["good_minus_bad"][:, L].float().numpy().mean(0))
        row["cos_gmn2_gmb3"] = float(np.dot(g2n, g3b))
        out["per_layer"].append(row)

    (ana / "generality_metrics.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ls = list(range(NUM_LAYERS))
    ax.plot(ls, [r["cos_gmb2_gmb3"] for r in out["per_layer"]], label="gmb(step2) vs gmb(step3)")
    ax.plot(ls, [r["cos_gmn2_gmn3"] for r in out["per_layer"]], label="gmn(step2) vs gmn(step3)")
    ax.plot(ls, [r["cos_bmn_control"] for r in out["per_layer"]], "--", color="#7f7f7f",
            label="bmn identity control (should be 1)")
    ax.set_xlabel("layer"); ax.set_ylabel("cosine"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Does the skill-conditioning direction generalize across skill texts?")
    fig.tight_layout()
    fig.savefig(ana / "generality_step2_vs_step3.png", dpi=150)

    mids = [r for r in out["per_layer"] if 10 <= r["layer"] <= 22]
    print("mid-layer (10-22) means:")
    for k in ("cos_gmb2_gmb3", "cos_gmn2_gmn3", "cos_bmn_control", "cos_gmn2_gmb3"):
        print(" ", k, round(float(np.mean([r[k] for r in mids])), 4))


if __name__ == "__main__":
    main()
