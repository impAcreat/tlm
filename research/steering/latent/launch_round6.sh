#!/bin/bash
cd /data5/ninghan/tlm
RUN=research/steering/runs/latent_skillopt_repro42_20260717
PY=envs/skillopt-qwen35-vllm/bin/python
SO=benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714
S1=$SO/steps/step_0001/candidate_skill.md
ALL35="val:0001,val:0003,val:0014,val:0015,val:0023,val:0024,val:0038,val:0047,val:0116,val:0117,val:0118,val:0122,val:0129,val:0130,val:0135,val:0138,val:0048,val:0050,val:0051,val:0052,val:0056,val:0057,val:0060,val:0068,val:0076,val:0077,val:0080,val:0085,val:0095,val:0102,val:0103,val:0112,val:0113,val:0114,val:0115"

tmux new-session -d -s r6_gpu1 bash -c "
CUDA_VISIBLE_DEVICES=1 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_unit_P_l18 --skill-path $S1 --vector-path $RUN/vectors/v_unit_P_l18.pt --alpha 1.4 --gen-only --task-ids $ALL35 > $RUN/s1_unit_P.log 2>&1
"
tmux new-session -d -s r6_gpu2 bash -c "
CUDA_VISIBLE_DEVICES=2 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_base --skill-path $S1 --task-ids $ALL35 > $RUN/s1_base_ext.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY research/steering/latent/steered_eval.py --out-dir $RUN --arm-name s1_unit_e0_l18_a2.2 --skill-path $S1 --vector-path $RUN/vectors/v_unit_e0_l18.pt --alpha 2.2 --gen-only --task-ids $ALL35 > $RUN/s1_unit_e0_ext.log 2>&1
"
echo launched-round6
