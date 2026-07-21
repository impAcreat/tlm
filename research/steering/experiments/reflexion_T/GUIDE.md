# Guide: Big-model reflexion quality + T compiler validation

Audience: an agent working in the canonical lab-130 checkout
(`/sdc/ninghan/tlm`) with no context from prior sessions.
Read `experiments/EXPERIMENT_LOG.md` entries 2026-07-17 → 2026-07-20 and
`research/steering/experiments/skill_edit_4b/PIPELINE_20260717.md` before touching anything.

## Mission (in priority order)

The primary purpose of the bigger model is **not** generic scale validation. It is:
1. Find the **minimum model size at which faithful Reflexion works well** on ALFWorld
   (large text-effect on retries), because high-quality reflexion hints are the
   training data for a better text→steering-vector compiler T.
2. With that model: regenerate a high-quality hint dataset, re-extract labels,
   retrain T, and re-run the pre-registered T validations.
3. Secondary: replicate three frozen-pipeline facts on that model (extraction
   consistency, unit additivity, best-recipe injection causality) at reduced scale.

## Hard constraints (violations have burned days before)

- SSH alias `lab-130`, root `/sdc/ninghan/tlm`. Check
  `nvidia-smi` owners first — other users (e.g. qinhao) share 0/3; do not kill
  or contend with their jobs; prefer free GPUs; our jobs are ~10 GB (4B) each.
- Python env: `envs/skillopt-qwen35-vllm` (transformers 5.x) is the ONLY env that
  loads Qwen3.5 for HF forwards. `-cu128` is for env replay only. New models may
  need their own env — verify `AutoModelForCausalLM` + `output_hidden_states`
  before building anything.
- Qwen chat template needs explicit `enable_thinking=False`.
- Long jobs in tmux with logs in the run dir; every eval loop must be resumable
  (append to results.jsonl, skip done ids).
- Shut down any vllm server you start once its run is finished.

## Stable method boundary

Reusable methods live in `research/steering/core/`, integration in `adapters/`,
and this experiment's orchestration in `experiments/reflexion_T/scripts/`.
The 4B reference scripts are under `experiments/skill_edit_4b/scripts/`.
Run-dir pattern:
`research/steering/runs/latent_<model>_<date>/`.

1. **States**: reuse `runs/latent_skillopt_repro42_20260717/prompts_v0000.jsonl`
   (1,213 replayed decision states; env-side text, model-independent). No new replay needed.
2. **Extraction**: prompt-conditioned contrast, byte-identical prompts except the
   skill section, last-token hidden at all layers, mean over states
   (`scripts/extract_vectors.py`; see the 4B reference scripts for provenance). Units are applied with
   `skillopt.optimizer.skill.apply_edit` when they are skillopt edits
   (append-style "## Additional Hints" is a WEAKER manipulation — fine for hints,
   wrong for edit units; see R10 caveat A).
3. **Injection**: gen-only (decode steps only), greedy decoding. Layer and dose
   must be calibrated on development groups for each model; never transfer 4B
   layer indices or strengths to 32B by proportional depth alone.
4. **Metrics discipline**: task-level paired performance is primary. Residual
   cosine diagnoses extracted-vs-predicted representation fidelity but cannot
   establish steering effectiveness. Every causal test needs matched baseline,
   text, extracted, predicted, and norm-matched random arms.

## Phase 1 — Reflexion-size sweep (the new part; ~1 day/model size)

Faithful Reflexion reproduction FIRST, our modifications only later:
- Loop: attempt task → on failure, model writes a reflection about ITS OWN failed
  trajectory (specific, not "generalizable" — do not reuse the kimi hint prompt
  from `reflexion_hints.py`, which deliberately distilled generic advice and
  produced only +2/20 text effect) → retry SAME task with the reflection
  prepended → up to 2 retries, temperature 0.7.
- Candidate sizes (all local or one download): Qwen3.5-4B (baseline, expect weak),
  Qwen3-32B (`models/Qwen3-32B`, local), plus one mid size (e.g. Qwen3-14B,
  download) if 32B works and 4B doesn't — bisect to the minimum workable size.
- Task set: 40 v0000-failure tasks from `manifest.json` (stratified by task type).
- Gate G1: a size "works" if reflexion lifts retry success by ≥15 pp over
  no-reflection retry at the same temperature/attempts. Pick the SMALLEST size
  that passes. Reflection AUTHOR and retry EXECUTOR are the same model here
  (that is the Reflexion setting; do not mix models silently).

## Phase 2 — Model-native layer and dose calibration

Before collecting compiler-scale labels, run a model-native calibration on
development tasks only:

1. Extract all-layer prompt-contrast vectors from text-effect-validated units.
2. Screen layers using consistency plus a small matched extracted/random causal
   test; geometry alone cannot select the layer.
3. At shortlisted layers, estimate dose-response curves in relative-to-vector
   norm units, including zero and sign controls. Record success, invalid-action
   rate and action-change rate.
4. Select either one robust fixed dose or a predeclared dynamic rule from
   development data. Freeze it before compiler evaluation.

## Phase 3 — Hint dataset + T on the chosen model

1. Generate enough reflections for stable group-held-out training and an
   untouched causal test via the Phase-1 loop (only from genuine failures;
   keep the full reflection text, no distillation). Store as
   `t_dataset/hints.jsonl` (schema: unit_id `hint_<cond>_<valXXXX>_<n>`, text, task fields).
2. Labels: `scripts/extract_vectors.py` uses the frozen state sample and paired
   prompt contrast, saves all layers in fp16, and stores text representations.
   State count is configured from the dataset-size and variance analysis rather
   than copied blindly from the 4B run.
3. Train/evaluate T with episode-grouped folds after layer/dose calibration.
   Report held-out residual cosine and shuffled-null only as G2 diagnostics.
4. Primary causal validation uses untouched tasks and the arms baseline / text /
   extracted / T-predicted / norm-matched random. Report success, paired flips,
   invalid-action rate, and action changes across a predeclared dose policy.
   No fixed 3.9× dose or proportionally mapped layer is inherited from 4B.

## Phase 4 — Frozen-fact replication at reduced scale

Only three facts, nothing else (config grids, confound analyses, mechanism probes
are DONE at 4B and must not be repeated):
- gmb cross-state consistency curve + step2-vs-step3 text stability (forwards only).
- Unit additivity: cos(Σ v(eᵢ), d12) with the three step-2 edits (forwards only).
- Best-recipe causal: 3 arms (bad / bad+gmb multi@1× / random) × 24 repaired
  tasks, gen-only, greedy.

## Reporting

- Each phase appends a dated entry to `experiments/EXPERIMENT_LOG.md`:
  question / protocol / numbers / evidence boundary. State gate pass/fail explicitly.
- Durable artifacts in the new run dir; update this experiment's README/configs
  with any pipeline delta the new model forced.
- Do not overwrite anything under `runs/latent_skillopt_repro42_20260717/`.

## Known traps (from 10 rounds of experience)

1. Trajectory-contrast extraction is a length artifact — never use it.
2. Injecting per-layer deltas at many layers unscaled destroys generation;
   norm-calibrated few-layer is the only validated multi-layer mode.
3. Dose cliffs were sharp on 4B; treat those numbers only as evidence that a
   model-native dose curve and invalid-generation guard are necessary.
4. Temperature sampling drowns single-vector effects — causal arms are greedy.
5. `best_skill.md` in the skillopt output dir is the step-3 candidate now;
   the 62.14% "good" skill is `steps/step_0002/candidate_skill.md`.
6. The optimizer gateway's GPT channels have failed before; kimi-k2.5 worked.
   For Phase 1 the reflection author is the local model itself — no API needed.
