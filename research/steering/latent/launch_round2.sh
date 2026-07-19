#!/bin/bash
cd /data5/ninghan/tlm
RUN=research/steering/runs/latent_skillopt_repro42_20260717
PY=envs/skillopt-qwen35-vllm/bin/python
GMN18=$RUN/vectors/v_prompt_good_minus_none_l18.pt
GMN14=$RUN/vectors/v_prompt_good_minus_none_l14.pt
GMB18=$RUN/vectors/v_prompt_good_minus_bad_l18.pt

# GPU0 chain: step-3 generality forward first (short), then none-skill arms
tmux new-session -d -s r2_gpu0 bash -c "
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/prompt_forward.py --out-dir $RUN --good-skill-path benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714/steps/step_0003/candidate_skill.md --out-name prompt_deltas_step3.pt > $RUN/prompt_forward_step3.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name none_base --skill none --categories repaired --limit 12 > $RUN/none_base_1.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name none_base --skill none --categories both_fail --limit 6 > $RUN/none_base_2.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name none_gmn_l18_a10 --skill none --vector-path $GMN18 --alpha 10 --categories repaired --limit 8 > $RUN/none_gmn_l18_a10.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name genonly_l18_a8 --skill bad --vector-path $GMB18 --alpha 8 --gen-only --categories repaired --limit 8 > $RUN/genonly_l18_a8.log 2>&1
"

# GPU3 chain: gmn main arm, then L14 dose, then gen-only high alpha
tmux new-session -d -s r2_gpu3 bash -c "
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name none_gmn_l18_a6 --skill none --vector-path $GMN18 --alpha 6 --categories repaired --limit 12 > $RUN/none_gmn_l18_a6_1.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name none_gmn_l18_a6 --skill none --vector-path $GMN18 --alpha 6 --categories both_fail --limit 6 > $RUN/none_gmn_l18_a6_2.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name none_gmn_l14_a7.5 --skill none --vector-path $GMN14 --alpha 7.5 --categories repaired --limit 8 > $RUN/none_gmn_l14_a7.5.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name genonly_l18_a16 --skill bad --vector-path $GMB18 --alpha 16 --gen-only --categories repaired --limit 8 > $RUN/genonly_l18_a16.log 2>&1
"
echo launched-round2
