#!/usr/bin/env bash
set -euo pipefail

PID_FILE="/sdc/ninghan/lm/logs/qwen3-32b-vllm.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file found: ${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill "${PID}" 2>/dev/null; then
  echo "Stopped qwen3-32b vLLM server PID=${PID}"
else
  echo "Process ${PID} was not running"
fi
rm -f "${PID_FILE}"
