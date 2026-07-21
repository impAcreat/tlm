"""Thin ALFWorld lifecycle adapter used by experiment scripts."""
from __future__ import annotations

import os
from pathlib import Path


class AlfworldAdapter:
    def __init__(self, data_root: str | Path, *, seed: int = 42):
        self.data_root = Path(data_root)
        self.seed = seed
        os.environ.setdefault("ALFWORLD_DATA", str(self.data_root))
        os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")

    def local_gamefile(self, path: str) -> str:
        marker = "/json_2.1.1/"
        if marker in path:
            return str(self.data_root / "json_2.1.1" / path.split(marker, 1)[1])
        return path

    def build(self, gamefile: str):
        from skillopt.envs.alfworld.rollout import build_alfworld_env

        local = self.local_gamefile(gamefile)
        dataset = "eval_in_distribution" if "/valid_seen/" in local else "eval_out_of_distribution"
        return build_alfworld_env(
            env_num=1,
            eval_dataset=dataset,
            seed=self.seed,
            is_train=False,
            specific_gamefiles=[local],
        )
