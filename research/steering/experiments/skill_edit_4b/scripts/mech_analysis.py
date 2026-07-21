"""Mechanistic evidence for the skill-conditioning vectors.

(a) Logit-lens: project each steering vector through the unembedding; report
    top promoted / suppressed tokens per layer.
(b) Teacher-forced likelihood shift: at identical step-0 states, does injecting
    the vector (response positions only == gen-only analogue) raise the
    log-likelihood of the GOOD-skill rollout's actual step-0 response while the
    prompt still contains the bad skill? Controls: random vector, good-prompt
    upper reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from research.steering.adapters.models.loading import load_causal_lm  # noqa: E402
from research.steering.adapters.models.hf_steering import ActivationSteerer, MultiLayerActivationSteerer  # noqa: E402
from research.steering.experiments.skill_edit_4b.scripts.prompt_forward import SYSTEM_PROMPT, build_skill_prompt  # noqa: E402

RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"


def get_unembed(model):
    head = model.get_output_embeddings()
    return head.weight.detach().float()  # [vocab, d]


def chat_ids(tokenizer, user, device):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    try:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                            enable_thinking=False)
    except TypeError:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return ids.to(device)


@torch.no_grad()
def response_logprob(model, tokenizer, device, user, response, steerer_args=None, multi=False):
    p_ids = chat_ids(tokenizer, user, device)
    r_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids = torch.cat([p_ids, r_ids], dim=1)
    p_len = p_ids.shape[1]
    sl = slice(p_len, None)  # steer response positions only (gen-only analogue)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    if steerer_args is None:
        out = model(**kwargs)
    elif multi:
        with MultiLayerActivationSteerer(model, token_slice=sl, **steerer_args):
            out = model(**kwargs)
    else:
        with ActivationSteerer(model, token_slice=sl, **steerer_args):
            out = model(**kwargs)
    logits = out.logits[0, p_len - 1:-1].float()
    targets = ids[0, p_len:]
    lp = torch.log_softmax(logits, dim=-1)
    tok_lp = lp[torch.arange(len(targets)), targets]
    return float(tok_lp.sum()), int(len(targets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-tasks", type=int, default=100)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"
    vec_dir = out_dir / "vectors"

    model, tokenizer = load_causal_lm(args.model_path, args.device)
    W = get_unembed(model)

    # ---------- (a) logit-lens ----------
    lens = {}
    for fam, layers in (("v_prompt_good_minus_bad", (10, 14, 18, 22)),
                        ("v_unit_e0", (18,)), ("v_unit_e1", (18,)), ("v_unit_P", (18,))):
        for L in layers:
            f = vec_dir / f"{fam}_l{L}.pt"
            if not f.exists():
                continue
            v = torch.load(f, weights_only=False)["vector"].float().to(W.device)
            logits = (W @ v).cpu()
            top = torch.topk(logits, 25)
            bot = torch.topk(-logits, 25)
            lens[f"{fam}_l{L}"] = {
                "promoted": [tokenizer.decode([i]) for i in top.indices.tolist()],
                "suppressed": [tokenizer.decode([i]) for i in bot.indices.tolist()],
            }
    (ana / "logit_lens.json").write_text(json.dumps(lens, ensure_ascii=False, indent=1))

    # ---------- (b) teacher-forced likelihood shift ----------
    manifest = json.loads((out_dir / "manifest.json").read_text())
    skills = {
        "bad": (RUN_ROOT / "skills/skill_v0000.md").read_text(),
        "good": (RUN_ROOT / "steps/step_0002/candidate_skill.md").read_text(),
    }
    states = [json.loads(l) for l in (out_dir / "prompts_v0000.jsonl").read_text().splitlines()
              if l.strip()]
    step0 = {s["task_id"]: s for s in states if s["step"] == 0}
    tids = sorted(step0)[: args.n_tasks]

    gmb = torch.load(vec_dir / "v_prompt_good_minus_bad_l18.pt", weights_only=False)
    v18 = gmb["vector"].float()
    g = torch.Generator().manual_seed(7)
    rnd = torch.randn(v18.shape, generator=g)
    rnd = rnd / rnd.norm()
    calib = torch.load(vec_dir / "multi_gmb_calib.pt", weights_only=False)["mean_delta"].float()
    multi_vecs = {L: calib[L] for L in (14, 18, 22)}

    arms = {
        "bad_prompt": (None, False),
        "bad_gmb_l18_a8": (dict(layer=18, vector=v18, alpha=8.0), False),
        "bad_rnd_l18_a8": (dict(layer=18, vector=rnd, alpha=8.0), False),
        "bad_multi_a1": (dict(vectors=multi_vecs, alpha=1.0), True),
    }

    rows = []
    t0 = time.time()
    for n, tid in enumerate(tids):
        obs = step0[tid]["obs_text"]
        conv = json.loads(Path(manifest["tasks"][tid]["step2"]["conversation"]).read_text())
        good_resp = str(conv[0].get("model_response") or "").strip()
        if not good_resp:
            continue
        row = {"task_id": tid, "n_resp_tokens": None}
        for arm, (sa, multi) in arms.items():
            user = build_skill_prompt(skills["bad"]) + "\n" + obs
            lp, ntok = response_logprob(model, tokenizer, args.device, user, good_resp, sa, multi)
            row[arm] = lp
            row["n_resp_tokens"] = ntok
        user_g = build_skill_prompt(skills["good"]) + "\n" + obs
        row["good_prompt"], _ = response_logprob(model, tokenizer, args.device, user_g, good_resp)
        rows.append(row)
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(tids)} elapsed {time.time() - t0:.0f}s", flush=True)

    agg = {}
    base = np.array([r["bad_prompt"] for r in rows])
    ntoks = np.array([r["n_resp_tokens"] for r in rows])
    for arm in list(arms) + ["good_prompt"]:
        vals = np.array([r[arm] for r in rows])
        d = (vals - base) / ntoks  # per-token delta logprob vs bad prompt
        agg[arm] = {"mean_per_token_delta_vs_bad": float(d.mean()),
                    "sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                    "frac_improved": float((d > 0).mean())}
    (ana / "teacher_forced_shift.json").write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
