# Skill-Edit Steering Prototype

This directory keeps only the current v2 prototype for connecting textual skill edits to activation steering.

The active research question is narrow:

```text
Can an old/new ALFWorld skill difference be converted into an activation vector
that moves the old-skill agent toward the new-skill behavior?
```

Current validated task family:

```text
ALFWorld look_at_obj_in_light
```

## Directory Layout

```text
research/steering/
  core/
    hf.py                 HF model loading and chat-template tokenization
    hooks.py              activation injection hooks
    vectors.py            hidden-state pooling and contrast-vector construction
  rollouts/
    rollout_hf.py         reusable HF ALFWorld rollout helpers
    run_alfworld_hf_rollout.py
                          CLI for baseline/steered ALFWorld rollouts
  skill_edit/
    intra_step_mean.py    raw-trajectory serialization, exact step-token masks, and mean pooling
    run_intra_step_mean.py
                          shared paper-style / atomic skill-edit intra analysis runner
    run_intra_text_alignment.py
                          edit text/full-skill delta vs behavioral intra-vector alignment
    npm_skill_memory.py   old/new rollout trace loader and step/traj dataset builder
    run_npm_skill_memory.py
                          CLI for inter-trajectory and intra-step vector extraction
    edit_suite.py         atomic SkillOpt accepted-edit definitions and S-/S+ skill materialization
    run_materialize_skill_edits.py
                          writes per-edit minus.md/plus.md skill pairs
    run_paired_skill_rollouts.py
                          runs the same gamefiles under S- and S+ for each edit
    run_label_skill_edit_nodes.py
                          asks a flagship model to label counterfactual intra-step pairs
    run_extract_skill_edit_vectors.py
                          extracts inter/intra edit vectors and clustering figures
  skills/
    manual_light_procedural_v2.md
                          manually constructed new skill used as current positive control
  artifacts/
    skill_edit_v2/
      summary_20260706.md durable result summary and evidence boundary
```

## What Was Removed From The Main Path

The earlier `tiny`, clean ALFWorld, offline action-choice, SkillEdit v1, random-control, and unrelated-vector scripts are intentionally not part of this directory anymore. They were useful debugging attempts, but they do not support the current claim because their effects were either synthetic, offline-only, or not specific enough.

The active path starts from real old/new ALFWorld rollouts, extracts trajectory-level and step-level memory vectors, then evaluates them through real HF ALFWorld rollouts.

## Active Data Flow

```text
old skill rollout traces
new skill rollout traces
        |
        v
skill_edit/npm_skill_memory.py
        |
        +-- inter trajectory dataset:
        |      positive = successful new trajectories
        |      negative = failed old trajectories
        |
        +-- intra step dataset:
               grouped by failed old trajectory
               positive = effective steps
               negative = repeated/invalid/no-op steps
        |
        v
skill_edit/run_npm_skill_memory.py
        |
        +-- inter_traj_vector.pt
        +-- intra_step_vector.pt
        |
        v
rollouts/run_alfworld_hf_rollout.py
        |
        +-- baseline old skill rollout
        +-- old skill + vector rollout
```

## Known Evidence Boundary

Current Qwen3.5 HF positive control holds:

```text
old skill:    1/6 success
new skill:    3/6 success
```

The current NPM-style mean-difference vectors do not yet transfer the skill edit into task success. The best current steering result only slightly reduces repeated actions under last-token intervention, without improving success rate.

The next method change should be paired skill-delta extraction:

```text
same task + same history + old skill prompt
same task + same history + new skill prompt
v = h(new_skill_prompt) - h(old_skill_prompt)
```

Then run the 2x2 directionality check:

```text
old skill + no vector
old skill + +v
new skill + no vector
new skill + -v
```

## SkillOpt Accepted-Edit Vector Extraction

The current code supports the narrower mechanism test:

```text
one atomic SkillOpt edit -> many paired S-/S+ rollouts -> instance vectors -> one pooled edit vector
```

The default accepted edits come from the GPT-5.5 SkillOpt ALFWorld run:

```text
benchmarks/skillopt/outputs/skillopt_gpt55_raw_full_paperish_gate_w2_20260707_1
```

Main edit ids:

```text
e_search_checked
e_light_hold_then_lamp
e_clean_empty_sink
e_heat_microwave
e_pickup_precise
e_cool_fridge
```

Recommended first subset:

```text
e_light_hold_then_lamp,e_pickup_precise,e_cool_fridge
```

Materialize atomic S-/S+ skills:

```bash
CUDA_VISIBLE_DEVICES= envs/skillopt-qwen35-vllm-cu128/bin/python -m research.steering.skill_edit.run_materialize_skill_edits \
  --out-dir research/steering/runs/skill_edit_suite_gpt55_20260708/skills
```

Run paired rollouts after reviewing the materialized skills:

```bash
CUDA_VISIBLE_DEVICES=6 envs/skillopt-qwen35-vllm-cu128/bin/python -m research.steering.skill_edit.run_paired_skill_rollouts \
  --edits-json research/steering/runs/skill_edit_suite_gpt55_20260708/skills/edits.json \
  --results-jsonl benchmarks/skillopt/outputs/eval_gpt55_raw_initial_test32_20260707/results.jsonl \
  --edit-ids e_light_hold_then_lamp,e_pickup_precise,e_cool_fridge \
  --model-path models/Qwen3.5-4B \
  --max-episodes-per-edit 8 \
  --max-steps 25 \
  --out-dir research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts
```

Label intra-step pairs with the OpenAI-compatible flagship optimizer endpoint:

```bash
OPENAI_BASE_URL=https://newapi.metamind.work/v1 \
SKILL_EDIT_LABEL_MODEL=gpt-5.5 \
envs/skillopt-qwen35-vllm-cu128/bin/python -m research.steering.skill_edit.run_label_skill_edit_nodes \
  --paired-rollouts-jsonl research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts/paired_rollouts.jsonl \
  --edit-ids e_light_hold_then_lamp,e_pickup_precise,e_cool_fridge \
  --out-jsonl research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts/llm_node_labels.jsonl
```

Extract vectors and clustering figures:

```bash
CUDA_VISIBLE_DEVICES=6 envs/skillopt-qwen35-vllm-cu128/bin/python -m research.steering.skill_edit.run_extract_skill_edit_vectors \
  --paired-rollouts-jsonl research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts/paired_rollouts.jsonl \
  --node-labels-jsonl research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts/llm_node_labels.jsonl \
  --edit-ids e_light_hold_then_lamp,e_pickup_precise,e_cool_fridge \
  --model-path models/Qwen3.5-4B \
  --layer 16 \
  --pooling last \
  --out-dir research/steering/runs/skill_edit_suite_gpt55_20260708/vectors_l16
```

The extraction output keeps both instance-level and pooled vectors:

```text
vectors/all_instances.pt
vectors/<edit_id>_inter_instances.pt
vectors/<edit_id>_intra_instances.pt
figures/pca_skill_edit_instances.png
figures/pooled_cosine_heatmap.png
summary.json
```

`run_intra_text_alignment.py` additionally compares each behavioral edit vector against (1) the mean hidden representation of the atomic edit text and (2) the full `plus skill - minus skill` hidden-state difference. Its cross-edit heatmaps test whether diagonal text/edit matches are stronger than mismatched edits.

## Step-Token Mean INTRA Analysis

The current representation path uses a shared extractor for two scientifically distinct contrasts:

```text
paper mode:      effective steps - degenerate steps within one failed trajectory
skill-edit mode: aligned S+ step - S- step for one atomic edit on the same game
```

Both modes encode only the raw task/trajectory transcript. History is context; mean pooling covers only the current action tokens. This keeps the representation identical across old traces that differ in whether full model reasoning was saved. Skill text, side labels, outcome, reward, and done are excluded from the representation input.

Primary outputs:

```text
intra_states_and_vectors.pt
span_audit.json
intra_hidden_and_vector_pca.png
summary.json
```

## Commands

Use the Qwen3.5 HF-compatible transformers-main path when loading `models/Qwen3.5-4B`:

```bash
cd /data5/ninghan/tlm
export PYTHONPATH=/data5/ninghan/tlm/.tmp/transformers_main_20260706:/data5/ninghan/tlm:/data5/ninghan/tlm/benchmarks/skillopt
export ALFWORLD_DATA=/data5/ninghan/tlm/benchmarks/skillopt/data/alfworld_data
export ALFWORLD_WORKER_START_METHOD=spawn
```

Extract current NPM-style vectors:

```bash
CUDA_VISIBLE_DEVICES=6 envs/skillopt-qwen35-vllm-cu128/bin/python -m research.steering.skill_edit.run_npm_skill_memory \
  --old-results-jsonl research/steering/runs/qwen35_hf_old_light_6ep_20260706/baseline/results.jsonl \
  --new-results-jsonl research/steering/runs/qwen35_hf_manual_new_light_6ep_20260706/baseline/results.jsonl \
  --old-conversation-root research/steering/runs/qwen35_hf_old_light_6ep_20260706/baseline \
  --new-conversation-root research/steering/runs/qwen35_hf_manual_new_light_6ep_20260706/baseline \
  --task-type look_at_obj_in_light \
  --model-path models/Qwen3.5-4B \
  --layer 16 \
  --method mean_diff \
  --out-dir research/steering/runs/qwen35_npm_skill_memory_light_l16_mean_diff_6ep_20260706
```

Run a baseline or steered rollout:

```bash
CUDA_VISIBLE_DEVICES=6 envs/skillopt-qwen35-vllm-cu128/bin/python -m research.steering.rollouts.run_alfworld_hf_rollout \
  --model-path models/Qwen3.5-4B \
  --results-jsonl benchmarks/skillopt/outputs/skill_edit_v2_old_light_test18_20260706/results.jsonl \
  --skill benchmarks/skillopt/skillopt/envs/alfworld/skills/old_light_general_v2.md \
  --vector-path research/steering/runs/qwen35_npm_skill_memory_light_l16_mean_diff_6ep_20260706/inter_traj_vector.pt \
  --mode steered \
  --alpha 1 \
  --steer-token-slice last \
  --max-episodes 6 \
  --max-steps 25 \
  --out-dir research/steering/runs/qwen35_inter_last_rollout_old_light_alpha1_6ep_20260706
```
