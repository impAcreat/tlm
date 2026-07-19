from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from research.steering.core.hf import load_causal_lm
from research.steering.core.hooks import MultiLayerActivationSteerer


SKILL_HEADER = "\n\n## Skill Knowledge\nUse these learned procedures when deciding what to do:\n\n"


def system_with_skill(base: str, skill: str) -> str:
    return base + (SKILL_HEADER + skill.strip() if skill.strip() else "")


class GenOnlyMultiLayerSteerer(MultiLayerActivationSteerer):
    """Leave prompt prefill untouched and steer decode-token forwards only."""

    def _steer_hidden(self, hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden
        return super()._steer_hidden(hidden, vector)


@dataclass
class HFSkillPolicy:
    model_path: str
    device: str
    max_new_tokens: int = 96
    model: object | None = field(default=None, init=False)
    tokenizer: object | None = field(default=None, init=False)

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.model, self.tokenizer = load_causal_lm(self.model_path, self.device)

    def _ids(self, system: str, user: str) -> torch.Tensor:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
            )
        except TypeError:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        base_system: str,
        user: str,
        skill: str,
        *,
        vectors: dict[int, torch.Tensor] | None = None,
        alpha: float = 1.0,
    ) -> str:
        self.ensure_loaded()
        ids = self._ids(system_with_skill(base_system, skill), user)
        kwargs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if vectors:
            with GenOnlyMultiLayerSteerer(self.model, vectors=vectors, alpha=alpha):
                out = self.model.generate(**kwargs)
        else:
            out = self.model.generate(**kwargs)
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    @torch.no_grad()
    def last_token_layers(self, base_system: str, user: str, skill: str) -> torch.Tensor:
        self.ensure_loaded()
        ids = self._ids(system_with_skill(base_system, skill), user)
        out = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
            use_cache=False,
        )
        return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states[1:]])


def extract_prompt_vectors(
    policy: HFSkillPolicy,
    prompt_records: list[dict],
    bad_skill: str,
    good_skill: str,
    layers: list[int],
) -> dict:
    deltas = []
    for row in prompt_records:
        bad = policy.last_token_layers(row["base_system"], row["user"], bad_skill)
        good = policy.last_token_layers(row["base_system"], row["user"], good_skill)
        deltas.append(good - bad)
    stacked = torch.stack(deltas)
    means = stacked.mean(0)
    return {
        "vectors": {int(layer): means[layer] for layer in layers},
        "layers": layers,
        "num_states": len(prompt_records),
        "mean_delta_norm": {str(layer): float(stacked[:, layer].norm(dim=1).mean()) for layer in layers},
        "cross_state_cos": {
            str(layer): _mean_pairwise_cos(stacked[:, layer]) for layer in layers
        },
    }


def _mean_pairwise_cos(x: torch.Tensor) -> float:
    if len(x) < 2:
        return float("nan")
    u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = u @ u.T
    mask = ~torch.eye(len(x), dtype=torch.bool)
    return float(sim[mask].mean())


def random_matched_vectors(vectors: dict[int, torch.Tensor], seed: int = 7) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer, vector in vectors.items():
        random = torch.randn(vector.shape, generator=g)
        out[layer] = random * vector.norm() / random.norm().clamp_min(1e-12)
    return out


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_prompt_records(results: list[dict], limit: int) -> list[dict]:
    rows = []
    for result in results:
        rows.extend(result.get("prompt_records") or [])
        if len(rows) >= limit:
            break
    return rows[:limit]


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.I | re.S)
    return match.group(1).strip() if match else (text or "").strip().splitlines()[0].strip()

