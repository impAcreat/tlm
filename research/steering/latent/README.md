# Latent-space analysis + steering vectors from SkillOpt selection rollouts

Pipeline built 2026-07-17 against the clean seed42 SkillOpt ALFWorld repro
(`benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714`), which provides the same
140 `valid_seen` tasks rolled out under three skill versions (rough_v1 41.43%, step-1 candidate
52.14%, step-2 candidate 62.14%). Environment: `envs/skillopt-qwen35-vllm` (transformers 5.12 —
the only env that loads `models/Qwen3.5-4B` for HF forwards; the `-cu128` env cannot).
Replay uses `envs/skillopt-qwen35-vllm-cu128`.

Run dir with all artifacts: `research/steering/runs/latent_skillopt_repro42_20260717/`.
Findings and evidence boundary: see `experiments/EXPERIMENT_LOG.md` entry 2026-07-17.

## Pipeline (in execution order)

| script | env | what it does |
|---|---|---|
| `build_dataset.py` | any | join the 3 selection_eval result sets into `manifest.json`, pair categories (repaired/broken/both_*) |
| `extract_reps.py` | tf5, GPU | behavior-only trajectory serialization, action-token span mean pooling, all 32 layers -> `reps/{cond}.pt` |
| `analyze_latent.py` | tf5 | NPM-style inter success/fail probes + PCA/t-SNE, paired skill deltas, intra effective/degenerate steps |
| `analyze_controlled.py` | tf5 | length-controlled contrasts (early5, matched prefix), skill-identifiability probe |
| `replay_prompts.py` | cu128 | exact env replay of recorded rollouts to recover per-step prompts -> `prompts_v0000.jsonl` |
| `prompt_forward.py` | tf5, GPU | same state under bad/good/none skill prompts, last-token hidden all layers -> `prompt_deltas.pt` |
| `extract_vectors.py` | tf5 | steering-vector families + cross-family alignment -> `vectors/*.pt` |
| `analyze_prompt_space.py` | tf5 | PCA/t-SNE of the three prompt conditions |
| `steered_eval.py` | tf5, GPU | HF greedy ALFWorld rollouts with skillopt-faithful prompts, optional single- or multi-layer steering |
| `behav_effect.py` | any | graded behavioral movement of steered arms (Jaccard vs good/bad reference rollouts, repeat/invalid rates) |

## Key cautions (learned here, do not rediscover)

- Full-trajectory success/fail separation is a length artifact (failures = 50-step timeouts;
  `n_steps` alone matches the 0.96 probe). Length-control before interpreting.
- Skill effects are a global condition signature (probe 0.83 from first-5-step reps), not
  task-specific repair directions (those collapse to shuffled-null under length control).
- The prompt-conditioned good-minus-bad contrast is the most consistent vector family
  (cross-state cos 0.86 @L14 / 0.76 @L18) and is near-orthogonal to the behavioral and
  outcome directions.
- Multi-layer injection of raw per-layer mean deltas (L6-22, alpha 1) destroys generation —
  deltas compound across layers. Use single-layer injection; alpha on the order of 2x the
  target layer's mean delta norm (L18: mean norm ~4, effective alpha 8).
