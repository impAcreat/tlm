"""Round 9 prep: (a) per-task skill texts = s1 + hint (exactly the extraction
format) for the text-prepend control; (b) per-task multi-layer raw vectors
{14,18,22} for calibrated multi-layer injection of the extracted hint vectors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

RUN_ROOT = Path(__file__).resolve().parents[5] / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
LAYERS = (14, 18, 22)


def unit_text_block(text: str) -> str:
    return f"\n\n## Additional Hints\n- {text.strip()}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    tdir = out_dir / "t_dataset"
    vdir = tdir / "causal_vectors"
    sdir = tdir / "hint_skill_texts"
    sdir.mkdir(exist_ok=True)

    units = {}
    for f in sorted(tdir.glob("unit_vectors_shard*.pt")):
        units.update(torch.load(f, weights_only=False))
    meta = json.loads((vdir / "eval_meta.json").read_text())
    s1_text = (RUN_ROOT / "steps/step_0001/candidate_skill.md").read_text()

    skill_map = {}
    multi_map = {}
    for m in meta:
        tid = m["task_id"]
        rec = units[m["unit_id"]]
        p = sdir / f"{tid.replace(':', '_')}.md"
        p.write_text(s1_text + unit_text_block(str(rec["text"])))
        skill_map[tid] = str(p)
        vecs = {int(L): rec["vector"][L].float() for L in LAYERS}
        vp = vdir / f"extracted_multi_{tid.replace(':', '_')}.pt"
        torch.save({"vectors": vecs, "family": "T_causal_extracted_multi", "unit_id": m["unit_id"]}, vp)
        multi_map[tid] = {"path": str(vp), "alpha": 1.5}
    (vdir / "skill_map_text.json").write_text(json.dumps(skill_map, indent=1))
    (vdir / "map_extracted_multi.json").write_text(json.dumps(multi_map, indent=1))
    norms = {m["task_id"]: {str(L): float(units[m["unit_id"]]["vector"][L].float().norm()) for L in LAYERS}
             for m in meta}
    print("built", len(skill_map), "skill texts and multi vector files")
    print("example norms:", json.dumps(list(norms.items())[0]))


if __name__ == "__main__":
    main()
