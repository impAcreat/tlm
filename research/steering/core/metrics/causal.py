from __future__ import annotations

from collections.abc import Mapping


def paired_flips(baseline: Mapping[str, int], treatment: Mapping[str, int]) -> dict[str, int]:
    ids = sorted(set(baseline).intersection(treatment))
    return {
        "n": len(ids),
        "treatment_only": sum(not baseline[i] and bool(treatment[i]) for i in ids),
        "baseline_only": sum(bool(baseline[i]) and not treatment[i] for i in ids),
        "both": sum(bool(baseline[i]) and bool(treatment[i]) for i in ids),
        "neither": sum(not baseline[i] and not treatment[i] for i in ids),
    }
