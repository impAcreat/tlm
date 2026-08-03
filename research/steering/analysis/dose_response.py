"""Aggregate and select development-time steering doses by task performance."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def adaptive_dose_value(rows: Iterable[Mapping], *, arms: tuple[str, ...] = ("extracted", "random")) -> dict:
    """Measure how much a per-unit dose would buy over the best single global dose.

    ``best_global`` is the highest success rate reachable by one multiplier applied
    to every unit.  ``oracle_per_unit`` lets each unit pick its own best multiplier
    after the fact, so it is optimistically biased: with binary outcomes a unit
    counts as a success if *any* dose worked.  That bias applies equally to a
    control arm, so the control's own gap is the floor the real arm must clear —
    only ``excess_over_control`` supports a claim that dose must be adaptive.
    """
    by_arm: dict[str, dict[float, dict[str, int]]] = {}
    for row in rows:
        arm = row["arm"]
        if arm not in arms:
            continue
        multiplier = float(row["multiplier"])
        success = bool(row.get("success", row.get("hard", 0)))
        by_arm.setdefault(arm, {}).setdefault(multiplier, {})[row["unit_id"]] = int(success)

    result: dict = {"arms": {}}
    for arm, per_dose in by_arm.items():
        units = sorted({unit for successes in per_dose.values() for unit in successes})
        if not units:
            continue
        per_dose_rate = {
            dose: sum(successes.get(unit, 0) for unit in units) / len(units)
            for dose, successes in sorted(per_dose.items())
        }
        best_dose = max(per_dose_rate, key=lambda d: (per_dose_rate[d], -abs(d)))
        oracle = sum(
            max(successes.get(unit, 0) for successes in per_dose.values()) for unit in units
        ) / len(units)
        result["arms"][arm] = {
            "n_units": len(units),
            "per_dose_success": per_dose_rate,
            "best_global_dose": best_dose,
            "best_global_success": per_dose_rate[best_dose],
            "oracle_per_unit_success": oracle,
            "adaptive_gap": oracle - per_dose_rate[best_dose],
        }

    if len(result["arms"]) > 1 and "random" in result["arms"]:
        control = result["arms"]["random"]["adaptive_gap"]
        result["excess_over_control"] = {
            arm: metrics["adaptive_gap"] - control
            for arm, metrics in result["arms"].items()
            if arm != "random"
        }
    return result


def summarize_dose(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    n = len(rows)
    return {
        "n": n,
        "success_rate": sum(bool(x.get("success", x.get("hard", 0))) for x in rows) / max(1, n),
        "invalid_rate": sum(float(x.get("invalid_rate", 0.0)) for x in rows) / max(1, n),
        "repeat_rate": sum(float(x.get("repeat_rate", 0.0)) for x in rows) / max(1, n),
        "runtime_errors": sum(bool(x.get("runtime_error")) for x in rows),
    }


def rank_doses(rows: Iterable[Mapping], *, max_invalid_rate: float) -> list[dict]:
    """Rank fixed doses; success is primary and invalid generation is a guard."""
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["dose"]), []).append(row)
    summaries = []
    for dose, dose_rows in grouped.items():
        summary = summarize_dose(dose_rows)
        summary.update(dose=dose, safe=summary["invalid_rate"] <= max_invalid_rate)
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda item: (
            item["safe"],
            item["success_rate"],
            -item["invalid_rate"],
            -abs(item["dose"]),
        ),
        reverse=True,
    )
