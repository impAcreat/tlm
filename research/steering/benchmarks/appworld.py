from __future__ import annotations

import json
import os
import re
from pathlib import Path

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter

from .common import HFSkillPolicy
from .loaders import JsonSplitLoader


SYSTEM = """You are an autonomous coding agent in AppWorld. Use Python and the `apis` object.
Return exactly one focused Python code block per turn. Inspect API documentation before guessing.
When finished, call apis.supervisor.complete_task(answer=<answer>) for questions, or complete_task() otherwise."""
CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


class AppWorldAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str,
        model_path: str,
        data_root: str,
        device: str = "cuda:2",
        max_steps: int = 10,
        max_new_tokens: int = 192,
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 3,
        edit_budget: int = 2,
        **_: object,
    ) -> None:
        self.data_root = str(Path(data_root).resolve())
        self.dataloader = JsonSplitLoader(split_dir=split_dir, split_mode="split_dir", seed=42)
        self.policy = HFSkillPolicy(str(Path(model_path).resolve()), device, max_new_tokens)
        self.max_steps = int(max_steps)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        return self.build_env_from_batch(self.dataloader.build_train_batch(batch_size, seed, **kwargs))

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        return self.build_env_from_batch(self.dataloader.build_eval_batch(env_num, split, seed, **kwargs))

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        return run_appworld(env_manager, skill_content, self.policy, self.max_steps, self.data_root, out_dir)

    def get_task_types(self) -> list[str]:
        return sorted({str(x.get("task_type", "appworld")) for x in self.dataloader.train_items})


def _extract_code(text: str) -> str:
    match = CODE.search(text or "")
    return (match.group(1) if match else text or "").strip()


def _user(instruction: str, apps: str, history: list[dict]) -> str:
    recent = history[-6:]
    transcript = "\n".join(f"Python:\n{x['code']}\nOutput:\n{x['output']}" for x in recent)
    return f"Task: {instruction}\n\nAvailable apps:\n{apps}\n\nRecent interaction:\n{transcript or '(none)'}\n\nWrite the next Python code block."


def run_appworld(items, skill, policy, max_steps, data_root, out_dir, vectors=None, alpha=1.0):
    old_cwd = os.getcwd()
    out_dir = str(Path(out_dir).resolve())
    os.chdir(data_root)
    try:
        from appworld import AppWorld
        results = []
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for item in items:
            world = AppWorld(task_id=item["task_id"], experiment_name="skillopt_steering", max_interactions=max_steps * 3 + 10)
            history, prompt_records = [], []
            try:
                instruction = str(world.task.instruction)
                apps = world.execute("print(apis.api_docs.show_app_descriptions())")
                for _step in range(max_steps):
                    user = _user(instruction, apps, history)
                    prompt_records.append({"base_system": SYSTEM, "user": user})
                    raw = policy.generate(SYSTEM, user, skill, vectors=vectors, alpha=alpha)
                    code = _extract_code(raw)
                    try:
                        output = world.execute(code)
                    except Exception as error:
                        output = f"[execute error] {error}"
                    history.append({"code": code, "raw": raw, "output": output})
                    if world.task_completed():
                        break
                report = world.evaluate().to_dict()
            finally:
                world.close()
            n_tests = int(report.get("num_tests") or 0)
            n_pass = len(report.get("passes") or [])
            row = {
                "id": item["id"], "task_id": item["task_id"], "hard": int(bool(report.get("success"))),
                "soft": n_pass / n_tests if n_tests else 0.0, "task_type": item.get("task_type", "appworld"),
                "task_description": instruction, "trajectory": history, "prompt_records": prompt_records,
                "num_tests": n_tests, "num_passed": n_pass,
                "n_turns": len(history),
                "fail_reason": "" if report.get("success") else json.dumps(report.get("failures") or [], ensure_ascii=False),
            }
            results.append(row)
            conversation = [
                {
                    "step": idx,
                    "action": step["code"],
                    "reasoning": step["raw"],
                    "env_feedback": step["output"],
                }
                for idx, step in enumerate(history, 1)
            ]
            conv_dir = Path(out_dir) / "predictions" / str(item["id"])
            conv_dir.mkdir(parents=True, exist_ok=True)
            (conv_dir / "conversation.json").write_text(
                json.dumps(conversation, ensure_ascii=False, indent=2)
            )
        path = Path(out_dir) / "results.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results))
        return results
    finally:
        os.chdir(old_cwd)
