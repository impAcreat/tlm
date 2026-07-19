from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


class JsonSplitLoader(SplitDataLoader):
    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path)
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no JSON split in {path}")
        rows = json.loads(files[0].read_text())
        for row in rows:
            row["id"] = str(row["id"])
        return rows

