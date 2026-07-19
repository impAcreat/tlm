"""Visualize the prompt-conditioned latent space: same 1213 states under
bad / good / none skill prompts, last-token reps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RNG = np.random.default_rng(42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--sub", type=int, default=600)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"
    pd = torch.load(out_dir / "prompt_deltas.pt", weights_only=False)
    states = pd["states"]
    L = args.layer

    reps = {k: v[:, L, :].float().numpy() for k, v in pd["reps_last_token"].items()}
    n = len(states)
    idx = RNG.choice(n, min(args.sub, n), replace=False)
    X = np.concatenate([reps[k][idx] for k in ("bad", "good", "none")])
    lab = ["bad"] * len(idx) + ["good"] * len(idx) + ["none"] * len(idx)
    ttypes = [states[i]["task_type"] for i in idx] * 3

    ts = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(X)
    pc = PCA(n_components=2).fit_transform(X - X.mean(0))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    colors = {"bad": "#1f77b4", "good": "#2ca02c", "none": "#7f7f7f"}
    for ax, emb, name in ((axes[0], pc, "PCA"), (axes[1], ts, "t-SNE")):
        for k in colors:
            m = np.array(lab) == k
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, c=colors[k], label=f"{k} skill")
        ax.set_title(f"{name} @ L{L}: same states, three skill prompts")
        ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    tt_list = sorted(set(ttypes))
    cmap = plt.get_cmap("tab10")
    for i, tt in enumerate(tt_list):
        m = np.array(ttypes) == tt
        axes[2].scatter(ts[m, 0], ts[m, 1], s=8, alpha=0.6, color=cmap(i), label=tt[:18])
    axes[2].set_title("same t-SNE, colored by task type")
    axes[2].legend(fontsize=6); axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.suptitle("Prompt-conditioned latent space (last token before generation)")
    fig.tight_layout()
    fig.savefig(ana / f"prompt_space_l{L}.png", dpi=150)
    print("saved", ana / f"prompt_space_l{L}.png")

    # quantify: within-state spread vs cross-condition shift
    d_gb = np.linalg.norm(reps["good"] - reps["bad"], axis=1)
    centroid_shift = np.linalg.norm(reps["good"].mean(0) - reps["bad"].mean(0))
    state_spread = np.linalg.norm(reps["bad"] - reps["bad"].mean(0), axis=1).mean()
    print(json.dumps({
        "layer": L,
        "mean_state_delta_norm_good_bad": float(d_gb.mean()),
        "centroid_shift_good_bad": float(centroid_shift),
        "state_spread_bad": float(state_spread),
        "shift_to_spread_ratio": float(centroid_shift / state_spread),
    }, indent=2))


if __name__ == "__main__":
    main()
