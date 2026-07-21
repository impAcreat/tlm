from __future__ import annotations

from pathlib import Path


def plot_layer_curves(series: dict[str, list[float]], output: str | Path, *, title: str = "Layer analysis"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in series.items():
        ax.plot(range(len(values)), values, marker=".", linewidth=1.5, label=label)
    ax.set(xlabel="Transformer layer", ylabel="Metric", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
