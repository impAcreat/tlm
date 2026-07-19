from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, path = value.split("=", 1)
        result[key.strip()] = Path(path.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare semantic skill components with full best skill.")
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--component-root", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-success-drop", type=int, default=1)
    args = parser.parse_args()

    full = {str(row["id"]): row for row in read_jsonl(args.full_root / "results.jsonl")}
    report: dict[str, Any] = {"full_root": str(args.full_root), "components": {}}
    for name, root in parse_mapping(args.component_root).items():
        rows = read_jsonl(root / "results.jsonl")
        component = {str(row["id"]): row for row in rows}
        ids = sorted(component.keys() & full.keys())
        full_success = sum(int(full[item]["hard"]) for item in ids)
        component_success = sum(int(component[item]["hard"]) for item in ids)
        repairs = [item for item in ids if not full[item]["hard"] and component[item]["hard"]]
        regressions = [item for item in ids if full[item]["hard"] and not component[item]["hard"]]
        report["components"][name] = {
            "root": str(root),
            "n": len(ids),
            "full_success": full_success,
            "component_success": component_success,
            "success_delta": component_success - full_success,
            "outcome_agreement": sum(bool(full[item]["hard"]) == bool(component[item]["hard"]) for item in ids),
            "repairs_vs_full": repairs,
            "regressions_vs_full": regressions,
            "accepted": component_success >= full_success - args.max_success_drop,
        }
    report["all_accepted"] = all(item["accepted"] for item in report["components"].values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "component_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 1 semantic skill validation",
        "",
        "| Component | Full | Split | Delta | Agreement | Accepted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["components"].items():
        lines.append(
            f"| {name} | {item['full_success']}/{item['n']} | "
            f"{item['component_success']}/{item['n']} | {item['success_delta']:+d} | "
            f"{item['outcome_agreement']}/{item['n']} | {item['accepted']} |"
        )
    (args.out_dir / "component_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
