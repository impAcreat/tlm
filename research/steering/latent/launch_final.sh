#!/bin/bash
cd /data5/ninghan/tlm
RUN=research/steering/runs/latent_skillopt_repro42_20260717
PY=envs/skillopt-qwen35-vllm/bin/python
VEC=$RUN/vectors/v_prompt_good_minus_bad_l18.pt

tmux new-session -d -s latent_final_gpu0 bash -c "
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name random_l18_a8 --skill bad --vector-path $VEC --alpha 8 --random-vector --categories repaired --limit 8 > $RUN/random_l18_a8.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name pilot_steer_l18_a8 --skill bad --vector-path $VEC --alpha 8 --categories repaired --limit 12 > $RUN/steer_l18_a8_ext1.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name pilot_steer_l18_a8 --skill bad --vector-path $VEC --alpha 8 --categories both_fail --limit 6 > $RUN/steer_l18_a8_ext2.log 2>&1
"

tmux new-session -d -s latent_final_gpu3 bash -c "
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name steer_l18_a12 --skill bad --vector-path $VEC --alpha 12 --categories repaired --limit 8 > $RUN/steer_l18_a12.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name pilot_bad_base --skill bad --categories repaired --limit 12 > $RUN/bad_base_ext1.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name pilot_bad_base --skill bad --categories both_fail --limit 6 > $RUN/bad_base_ext2.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name pilot_good_base --skill good --categories repaired --limit 12 > $RUN/good_base_ext1.log 2>&1
CUDA_VISIBLE_DEVICES=3 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name pilot_good_base --skill good --categories both_fail --limit 6 > $RUN/good_base_ext2.log 2>&1
"
echo launched-final
