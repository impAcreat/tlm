"""Aggregate and select development-time steering doses by task performance."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


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
