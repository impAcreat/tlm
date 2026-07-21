from __future__ import annotations

from pathlib import Path


def plot_causal_arms(success_rates: dict[str, float], output: str | Path, *, title="Causal arms"):
    import matplotlib.pyplot as plt

    labels = list(success_rates)
    values = [success_rates[x] for x in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="#4f46e5")
    ax.set(ylabel="Task success rate", ylim=(0, 1), title=title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
