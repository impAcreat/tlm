"""Redemption test prep: use T (trained on 298 reflexion hints only) to
COMPILE the strong skillopt unit vectors from their TEXT, cross-domain.

For units S = step_0002_e0 (search) and P = step_0003_e1 (protocol):
  extracted_<u>   ground-truth append-style vector (same extraction as target)
  predicted_<u>   T(text_mean of the unit)  -- genuine cross-domain
Writes per-task vector maps for T_search / T_protocol, single-layer L18,
dose 3.9x extracted L18 norm (matching round 6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

LAYERS_IN = (14, 18, 22)
L_OUT = 18
UNITS = {"S": "skillopt_step_0002_e0", "P": "skillopt_step_0003_e1"}
TASKSETS = {
    "S": ["val:0001","val:0003","val:0014","val:0015","val:0023","val:0024","val:0038","val:0047",
          "val:0116","val:0117","val:0118","val:0122","val:0129","val:0130","val:0135","val:0138"],
    "P": ["val:0048","val:0050","val:0051","val:0052","val:0056","val:0057","val:0060","val:0068",
          "val:0076","val:0077","val:0080","val:0085","val:0095","val:0102","val:0103","val:0112",
          "val:0113","val:0114","val:0115"],
}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    tdir = out_dir / "t_dataset"
    vdir = tdir / "causal_vectors"

    units = {}
    for f in sorted(tdir.glob("unit_vectors_shard*.pt")):
        units.update(torch.load(f, weights_only=False))
    hints = [r for r in units.values() if r["source"] == "reflexion"]

    # train T on ALL hints (skillopt units were never in training)
    X = np.concatenate([np.stack([r["text_mean"][L].float().numpy() for r in hints])
                        for L in LAYERS_IN], axis=1)
    Y = np.stack([r["vector"][L_OUT].float().numpy() for r in hints])
    T = Ridge(alpha=100.0).fit(X, Y)

    report = {}
    g = torch.Generator().manual_seed(23)
    for name, uid in UNITS.items():
        rec = units[uid]
        ex = rec["vector"][L_OUT].float().numpy()
        x = np.concatenate([rec["text_mean"][L].float().numpy() for L in LAYERS_IN])[None]
        pred = T.predict(x)[0]
        alpha = float(3.9 * np.linalg.norm(ex))
        report[name] = {"unit": uid, "extracted_norm": float(np.linalg.norm(ex)),
                        "cos_pred_extracted": float(np.dot(unit(pred), unit(ex))), "alpha": alpha}
        for arm, vec in (("extracted", unit(ex)), ("predicted", unit(pred))):
            m = {}
            for tid in TASKSETS[name]:
                p = vdir / f"redeem_{name}_{arm}.pt"
                torch.save({"vector": torch.tensor(vec, dtype=torch.float32), "layer": L_OUT}, p)
                m[tid] = {"path": str(p), "alpha": alpha}
            (vdir / f"map_redeem_{name}_{arm}.json").write_text(json.dumps(m, indent=1))
    (vdir / "redeem_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
