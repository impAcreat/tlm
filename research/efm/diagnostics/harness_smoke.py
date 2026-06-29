"""Offline structural smoke for research.efm.harness.

No GPU / network: a stub feedback model, a stub agent, and a tiny scripted env
exercise the whole loop and the output contract, then we re-read the run with
research.efm.bench.eval to prove the contract parses.

Run:  PYTHONPATH=<tlm_root> python -m research.efm.diagnostics.harness_smoke
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from research.efm import FeedbackRuntime
from research.efm.harness import RunConfig, run_episodes
from research.efm.harness.config import EndpointConfig
from research.efm.harness.env import Reset, Step


class ScriptedEnv:
    """A 3-step toy env where success depends on issuing the 'take' action."""

    environment_id = "smoke"

    def __init__(self) -> None:
        self._tasks = [f"task-{i}" for i in range(2)]
        self._took = False

    def iter_tasks(self):
        return list(self._tasks)

    def reset(self, task):
        self._took = False
        return Reset(
            task_id=str(task),
            task_description="pick up the mug",
            observation="You are in a room. A mug is on the table.",
            system_prompt="You are an agent. Emit one action per turn.",
            task_type="pick",
        )

    def step(self, action: str) -> Step:
        if "take" in action.lower():
            self._took = True
            return Step(raw_observation="You take the mug.", reward=1.0, done=True,
                        info={"success": True, "soft": 1.0})
        return Step(raw_observation="Nothing happens. The mug is still on the table.",
                    info={"success": False})

    def is_success(self, info: dict) -> bool:
        return bool(info.get("success"))


class StubAgent:
    """Takes the mug on its second turn (so we get >1 step of trace)."""

    def __init__(self) -> None:
        self._turn = 0

    def reset(self, system_prompt: str) -> None:
        self._turn = 0

    def act(self, observation: str) -> str:
        self._turn += 1
        return "look around" if self._turn == 1 else "take mug from table"


class StubFeedbackModel:
    """Deterministic FeedbackModel; returns minimal valid JSON per stage."""

    def complete(self, system: str, user: str, *, max_tokens: int, stage: str) -> str:
        if stage == "efm_step":
            return json.dumps({
                "core_signal": "The mug remains on the table.",
                "signal_type": "state_change",
                "filtered_out": "room description",
                "intention_status": "unclear",
            })
        return json.dumps({})


def _run(arm: str, run_dir: Path) -> list[dict]:
    ep = EndpointConfig(base_url="http://unused", model="stub")
    cfg = RunConfig(run_dir=str(run_dir), target=ep, arm=arm, max_steps=5,
                    policy_update_enabled=False, reflect_enabled=False)
    runtime = None
    if arm == "efm":
        runtime = FeedbackRuntime(
            StubFeedbackModel(),
            state_path=str(run_dir / "feedback_state.json"),
            config=cfg.feedback_config(),
        )
    return run_episodes(ScriptedEnv(), StubAgent(), cfg, runtime=runtime)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="efm_harness_smoke_"))
    raw_dir = root / "raw"
    efm_dir = root / "efm"

    raw_results = _run("raw", raw_dir)
    efm_results = _run("efm", efm_dir)

    # Contract files exist.
    assert (raw_dir / "results.jsonl").exists(), "raw results.jsonl missing"
    assert (efm_dir / "results.jsonl").exists(), "efm results.jsonl missing"
    assert (efm_dir / "feedback_state.json").exists(), "feedback_state.json missing"
    assert list(efm_dir.glob("feedback/*.efm.json")), "per-episode efm artifact missing"

    # Both arms solved the scripted task in 2 steps.
    for results in (raw_results, efm_results):
        assert len(results) == 2, results
        assert all(r["hard"] == 1.0 for r in results), results
        assert all(r["steps"] == 2 for r in results), results

    # EFM state carries refined step feedback (what bench/eval reads).
    state = json.loads((efm_dir / "feedback_state.json").read_text())
    episodes = state.get("episodes", [])
    assert episodes, "no episodes recorded in feedback_state.json"
    trace = episodes[0]["trace"]
    assert trace and "step_feedback" in trace[0], "trace missing step_feedback"

    # The bench-agnostic aggregator parses both runs.
    from research.efm.bench.eval import load_run, summarize
    for d in (raw_dir, efm_dir):
        summary = summarize(load_run(str(d)))
        assert summary["n"] == 2, summary

    print("HARNESS SMOKE PASS")
    print(f"  raw run : {raw_dir}")
    print(f"  efm run : {efm_dir}")
    print(f"  efm episodes={len(episodes)} steps0={len(trace)} "
          f"core_signal0={trace[0]['step_feedback'].get('core_signal')!r}")


if __name__ == "__main__":
    main()
