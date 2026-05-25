#!/usr/bin/env bash
set -euo pipefail

ROOT="/sdc/ninghan/lm"
mkdir -p "${ROOT}/logs"

nohup "${ROOT}/scripts/serve_qwen3_32b.sh" > "${ROOT}/logs/qwen3-32b-vllm.log" 2>&1 &
echo $! > "${ROOT}/logs/qwen3-32b-vllm.pid"
echo "Started qwen3-32b vLLM server. PID=$(cat "${ROOT}/logs/qwen3-32b-vllm.pid")"
echo "Log: ${ROOT}/logs/qwen3-32b-vllm.log"
