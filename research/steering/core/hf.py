from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_alphas(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def load_causal_lm(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()
    return model, tokenizer


def chat_ids(tokenizer, text: str, *, add_generation_prompt: bool) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        if isinstance(rendered, torch.Tensor):
            return rendered
        if hasattr(rendered, "input_ids"):
            return rendered.input_ids
        rendered_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        return tokenizer(rendered_text, return_tensors="pt", add_special_tokens=False).input_ids
    return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
