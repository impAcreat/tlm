#!/bin/bash
cd /data5/ninghan/tlm
RUN=research/steering/runs/latent_skillopt_repro42_20260717
PY=envs/skillopt-qwen35-vllm/bin/python
GMB18=$RUN/vectors/v_prompt_good_minus_bad_l18.pt

tmux new-session -d -s r3_gpu0 bash -c "
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name genonly_random_l18_a8 --skill bad --vector-path $GMB18 --alpha 8 --gen-only --random-vector --categories repaired --limit 8 > $RUN/genonly_random_l18_a8.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name genonly_l18_a12 --skill bad --vector-path $GMB18 --alpha 12 --gen-only --categories repaired --limit 8 > $RUN/genonly_l18_a12.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name genonly_l18_a8_t07_s2 --skill bad --vector-path $GMB18 --alpha 8 --gen-only --temperature 0.7 --sample-tag s2 --categories repaired --limit 12 > $RUN/genonly_t07_s2.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name bad_base_t07_s2 --skill bad --temperature 0.7 --sample-tag s2 --categories repaired --limit 12 > $RUN/bad_t07_s2.log 2>&1
"

tmux new-session -d -s r3_gpu3 bash -c "
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name genonly_l18_a8_t07_s1 --skill bad --vector-path $GMB18 --alpha 8 --gen-only --temperature 0.7 --sample-tag s1 --categories repaired --limit 12 > $RUN/genonly_t07_s1.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name bad_base_t07_s1 --skill bad --temperature 0.7 --sample-tag s1 --categories repaired --limit 12 > $RUN/bad_t07_s1.log 2>&1
"
echo launched-round3
