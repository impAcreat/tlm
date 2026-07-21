from __future__ import annotations

from collections.abc import Iterable, Mapping


def behavior_quality(rows: Iterable[Mapping]) -> dict[str, float | int]:
    rows = list(rows)
    n = len(rows)
    return {
        "n": n,
        "invalid_rate": sum(bool(x.get("invalid")) for x in rows) / max(1, n),
        "repeat_rate": sum(bool(x.get("repeat")) for x in rows) / max(1, n),
        "runtime_errors": sum(bool(x.get("runtime_error")) for x in rows),
    }
