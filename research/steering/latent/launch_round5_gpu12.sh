#!/bin/bash
cd /data5/ninghan/tlm
RUN=research/steering/runs/latent_skillopt_repro42_20260717
PY=envs/skillopt-qwen35-vllm/bin/python
SO=benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714
S1=$SO/steps/step_0001/candidate_skill.md
U16="val:0001,val:0003,val:0015,val:0024,val:0038,val:0050,val:0051,val:0052,val:0076,val:0077,val:0080,val:0095,val:0102,val:0103,val:0113,val:0114"

tmux new-session -d -s r5_gpu1 bash -c "
CUDA_VISIBLE_DEVICES=1 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_base --skill-path $S1 --task-ids $U16 > $RUN/s1_base.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_unit_e1_l18_a2.2 --skill-path $S1 --vector-path $RUN/vectors/v_unit_e1_l18.pt --alpha 2.2 --gen-only --task-ids $U16 > $RUN/s1_unit_e1.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name multi_gmb_141822_a1_genonly --skill bad --multi-vector-path $RUN/vectors/multi_gmb_calib.pt --multi-layers 14,18,22 --alpha 1.0 --gen-only --categories repaired --limit 8 > $RUN/multi_gmb_141822_a1.log 2>&1
"

tmux new-session -d -s r5_gpu2 bash -c "
CUDA_VISIBLE_DEVICES=2 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_unit_e0_l18_a2.2 --skill-path $S1 --vector-path $RUN/vectors/v_unit_e0_l18.pt --alpha 2.2 --gen-only --task-ids $U16 > $RUN/s1_unit_e0.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_d12_l18_a3.3 --skill-path $S1 --vector-path $RUN/vectors/v_d12_l18.pt --alpha 3.3 --gen-only --task-ids $U16 > $RUN/s1_d12.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name multi_gmb_141822_a1.5_genonly --skill bad --multi-vector-path $RUN/vectors/multi_gmb_calib.pt --multi-layers 14,18,22 --alpha 1.5 --gen-only --categories repaired --limit 8 > $RUN/multi_gmb_141822_a1.5.log 2>&1
"
echo launched-round5
