from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter

from .common import HFSkillPolicy, extract_tag
from .loaders import JsonSplitLoader


SYSTEM = """You are an agent in ScienceWorld. Solve the task through exact text actions.
Return exactly one action inside <action>...</action>. Choose from the valid actions shown.
Do not explain the action and do not invent commands."""


class ScienceWorldAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str,
        model_path: str,
        device: str = "cuda:1",
        max_steps: int = 20,
        max_new_tokens: int = 64,
        simplification: str = "easy",
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 3,
        edit_budget: int = 2,
        **_: object,
    ) -> None:
        self.dataloader = JsonSplitLoader(split_dir=split_dir, split_mode="split_dir", seed=42)
        self.policy = HFSkillPolicy(model_path, device, max_new_tokens)
        self.max_steps = int(max_steps)
        self.simplification = simplification
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
        return run_scienceworld(env_manager, skill_content, self.policy, self.max_steps, self.simplification, out_dir)

    def get_task_types(self) -> list[str]:
        return sorted({str(x.get("task_type", "scienceworld")) for x in self.dataloader.train_items})


def _user(task: str, observation: str, history: list[dict], valid: list[str]) -> str:
    recent = history[-8:]
    transcript = "\n".join(f"Action: {x['action']}\nObservation: {x['observation']}" for x in recent)
    actions = "\n".join(f"- {x}" for x in valid[:180])
    return f"Task: {task}\n\nRecent history:\n{transcript or '(none)'}\n\nCurrent observation:\n{observation}\n\nValid actions:\n{actions}"


def run_scienceworld(items, skill, policy, max_steps, simplification, out_dir, vectors=None, alpha=1.0):
    from scienceworld import ScienceWorldEnv

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for item in items:
        env = ScienceWorldEnv("", envStepLimit=max_steps + 1)
        history, prompt_records = [], []
        try:
            env.load(item["task_name"], int(item["variation"]), simplification)
            observation, info = env.reset()
            task = env.get_task_description()
            score = float(info.get("score", 0.0))
            done = False
            for _step in range(max_steps):
                valid = [x["action"] for x in env.get_valid_action_object_combinations_with_templates()]
                user = _user(task, observation, history, valid)
                prompt_records.append({"base_system": SYSTEM, "user": user})
                raw = policy.generate(SYSTEM, user, skill, vectors=vectors, alpha=alpha)
                action = extract_tag(raw, "action").lower().strip()
                observation, _reward, done, info = env.step(action)
                score = float(info.get("score", score))
                history.append({"action": action, "raw": raw, "observation": observation, "score": score})
                if done:
                    break
        finally:
            env.close()
        row = {
            "id": item["id"], "hard": int(score >= 100.0), "soft": max(0.0, score) / 100.0,
            "score": score, "task_type": item.get("task_type", item["task_name"]),
            "task_description": task, "trajectory": history, "prompt_records": prompt_records,
        }
        results.append(row)
    path = Path(out_dir) / "results.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results))
    return results

