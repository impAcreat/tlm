#!/usr/bin/env bash
set -euo pipefail

ROOT="/sdc/ninghan/lm"
export TMPDIR="${ROOT}/.tmp"
export HF_HOME="${ROOT}/models/hf"
export HUGGINGFACE_HUB_CACHE="${ROOT}/models/hf/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XDG_CACHE_HOME="${ROOT}/.cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${ROOT}/logs"

MODEL_PATH="${MODEL_PATH:-${ROOT}/models/modelscope/Qwen/Qwen3-32B}"

exec /sdc/ninghan/miniforge3/bin/conda run --no-capture-output -n qwen \
  python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-32b \
  --host 0.0.0.0 \
  --port "${PORT:-8008}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
  --trust-remote-code
