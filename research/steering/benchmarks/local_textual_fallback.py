from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILLOPT = ROOT / "benchmarks/skillopt"
sys.path[:0] = [str(ROOT), str(SKILLOPT)]

from research.steering.benchmarks.common import HFSkillPolicy
from skillopt.gradient.reflect import fmt_minibatch_trajectories


SYSTEM = """You are a textual-skill optimizer for an autonomous agent.
Study the failed trajectories and rewrite the current skill into concise, reusable procedures.
Focus on error recovery, state verification, and benchmark-valid actions or API use.
Do not include task-specific answers. Return only the complete replacement skill in Markdown."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-skill", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-chars", type=int, default=30000)
    args = parser.parse_args()

    results = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    trajectories = fmt_minibatch_trajectories(results, args.prediction_dir)[: args.max_chars]
    current = Path(args.current_skill).read_text()
    user = f"## Current Skill\n{current}\n\n## Rollout Trajectories\n{trajectories}"
    policy = HFSkillPolicy(str(Path(args.model_path).resolve()), args.device, max_new_tokens=384)
    skill = policy.generate(SYSTEM, user, "")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(skill.strip() + "\n")
    print(f"wrote {out} ({len(skill)} chars, {len(results)} trajectories)")


if __name__ == "__main__":
    main()
