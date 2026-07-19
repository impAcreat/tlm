"""Save unit-edit and step-increment steering vectors from ladder_reps.pt.

Files (load_steering_vector-compatible):
  vectors/v_unit_e{i}_l{L}.pt   unit edit i conditioning direction
  vectors/v_d12_l{L}.pt         full step1->step2 increment direction
  vectors/multi_gmb_calib.pt    per-layer gmb unit vectors pre-scaled to each
                                layer's mean delta norm, for MultiLayer use
Also prints per-layer norms for alpha calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

LAYERS = (14, 18, 22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    vec_dir = out_dir / "vectors"

    lad = torch.load(out_dir / "ladder_reps.pt", weights_only=False)
    p2 = torch.load(out_dir / "prompt_deltas.pt", weights_only=False)
    idx = np.array(lad["subsample_indices"])

    s1 = lad["reps"]["s1"].float().numpy()
    s2 = p2["reps_last_token"]["good"][idx].float().numpy()
    d12 = s2 - s1
    units = {k: lad["reps"][k].float().numpy() - s1 for k in lad["reps"] if k.startswith("s1_e")}

    norms = {}
    for L in LAYERS:
        m = d12[:, L].mean(0)
        norms[f"d12_l{L}"] = float(np.linalg.norm(m))
        torch.save({"vector": torch.tensor(m / np.linalg.norm(m)), "layer": L, "family": "d12"},
                   vec_dir / f"v_d12_l{L}.pt")
        for k, V in units.items():
            mu = V[:, L].mean(0)
            norms[f"{k}_l{L}"] = float(np.linalg.norm(mu))
            torch.save({"vector": torch.tensor(mu / (np.linalg.norm(mu) + 1e-12)), "layer": L,
                        "family": f"unit_{k}"},
                       vec_dir / f"v_unit_{k.replace('s1_', '')}_l{L}.pt")

    # calibrated multi-layer gmb bundle: vector per layer scaled to that layer's mean-of-delta norm
    gmb_raw = torch.load(vec_dir / "gmb_raw_means.pt", weights_only=False)
    means = gmb_raw["mean_delta"].float()  # [32, d]
    calib = {L: means[L] / means[L].norm() * means[L].norm() for L in range(32)}  # == means, explicit
    torch.save({"mean_delta": means}, vec_dir / "multi_gmb_calib.pt")

    for L in range(32):
        norms[f"gmb_norm_of_mean_l{L}"] = float(means[L].norm())

    (vec_dir / "unit_vector_norms.json").write_text(json.dumps(norms, indent=2))
    for k in sorted(norms):
        if any(f"l{L}" == k.rsplit("_l", 1)[-1] and f"_l{L}" in k for L in LAYERS) or "d12" in k or "e" in k[:6]:
            pass
    print(json.dumps({k: round(v, 3) for k, v in norms.items() if "gmb" not in k}, indent=1))
    print("gmb norm-of-mean L14/L18/L22:",
          round(norms["gmb_norm_of_mean_l14"], 2),
          round(norms["gmb_norm_of_mean_l18"], 2),
          round(norms["gmb_norm_of_mean_l22"], 2))


if __name__ == "__main__":
    main()
