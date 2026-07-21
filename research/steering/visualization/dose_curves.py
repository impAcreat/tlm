from __future__ import annotations

from pathlib import Path


def plot_dose_curves(doses, success, invalid, output: str | Path, *, title="Dose calibration"):
    import matplotlib.pyplot as plt

    fig, left = plt.subplots(figsize=(7, 4.5))
    right = left.twinx()
    left.plot(doses, success, "o-", color="#2563eb", label="success")
    right.plot(doses, invalid, "s--", color="#dc2626", label="invalid")
    left.set(xlabel="Dose", ylabel="Success rate", title=title)
    right.set_ylabel("Invalid-action rate")
    left.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
