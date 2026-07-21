from __future__ import annotations

from collections.abc import Mapping

from research.steering.core.metrics.causal import paired_flips


def compare_arms(baseline: Mapping[str, int], arms: Mapping[str, Mapping[str, int]]) -> dict:
    base_rate = sum(baseline.values()) / max(1, len(baseline))
    result = {"baseline": {"n": len(baseline), "success_rate": base_rate}}
    for name, values in arms.items():
        common = sorted(set(baseline).intersection(values))
        rate = sum(values[x] for x in common) / max(1, len(common))
        result[name] = {
            "n": len(common),
            "success_rate": rate,
            "absolute_gain": rate - base_rate,
            "paired": paired_flips(baseline, values),
        }
    return result
