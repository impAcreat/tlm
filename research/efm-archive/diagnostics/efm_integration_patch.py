"""Patch SkillOpt EFM integration: stage-based model routing.

Step EFM (online) -> target backend (small 4B); skill-update stages
(trajectory review, policy proposal, llm gate) -> optimizer backend (strong 30B).
"""
p = "benchmarks/skillopt/skillopt/integrations/efm.py"
s = open(p).read()

old_cls = '''    def __init__(self, role: str = "optimizer") -> None:
        normalized = str(role or "optimizer").strip().lower()
        if normalized not in {"optimizer", "target"}:
            raise ValueError("feedback_model_role must be 'optimizer' or 'target'")
        self.role = normalized

    def complete(self, system: str, user: str, *, max_tokens: int, stage: str):
        from skillopt.model import chat_optimizer, chat_target

        call = chat_optimizer if self.role == "optimizer" else chat_target
        return call(
            system=system,
            user=user,
            max_completion_tokens=max_tokens,
            retries=3,
            stage=stage,
            timeout=None,
        )'''

new_cls = '''    def __init__(self, role: str = "split") -> None:
        normalized = str(role or "split").strip().lower()
        if normalized not in {"optimizer", "target", "split"}:
            raise ValueError("feedback_model_role must be 'optimizer', 'target', or 'split'")
        self.role = normalized

    def complete(self, system: str, user: str, *, max_tokens: int, stage: str):
        from skillopt.model import chat_optimizer, chat_target

        # "split" realizes the research design: a small model runs the online
        # Step EFM, while a strong optimizer runs the skill-update stages
        # (trajectory review, policy proposal, llm gate).
        if self.role == "split":
            use_optimizer = stage != "efm_step"
        else:
            use_optimizer = self.role == "optimizer"
        call = chat_optimizer if use_optimizer else chat_target
        return call(
            system=system,
            user=user,
            max_completion_tokens=max_tokens,
            retries=3,
            stage=stage,
            timeout=None,
        )'''

assert s.count(old_cls) == 1, "SkillOptFeedbackModel anchor not unique/found"
s = s.replace(old_cls, new_cls)
s = s.replace('    model_role: str = "optimizer",', '    model_role: str = "split",')
open(p, "w").write(s)
print("integration patched")

import subprocess
print(subprocess.run(["grep", "-n", "role", p], capture_output=True, text=True).stdout)
