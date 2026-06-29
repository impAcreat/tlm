# Terminal-Bench

## Location

- Source and Core runner: `/data5/ninghan/tlm/benchmarks/terminal_bench`
- Core v0.1.1 dataset: `datasets/terminal-bench-core-0.1.1`
- Core runner environment: `.venv`
- Terminal-Bench 2.0 / Harbor runner: `.harbor-venv`
- Durable outputs: `experiment_runs/`
- Durable logs: `logs/`

## Core runner

```bash
cd /data5/ninghan/tlm/benchmarks/terminal_bench
export UV_CACHE_DIR="$PWD/.uv-cache"
uv run tb run \
  --dataset-path datasets/terminal-bench-core-0.1.1 \
  --task-id hello-world \
  --agent nop \
  --n-concurrent 1 \
  --output-path experiment_runs/<run_name>
```

Provided Core agents include `terminus`, `terminus-1`, `terminus-2`, `claude-code`, `codex`, `openhands`, `mini-swe-agent`, and `qwen-coder`. Do not use `oracle` for a research result.

## Terminal-Bench 2.0

```bash
cd /data5/ninghan/tlm/benchmarks/terminal_bench
.harbor-venv/bin/harbor --help
.harbor-venv/bin/harbor run --dataset terminal-bench@2.0 --agent <agent> --model <provider/model> --n-concurrent 1
```

## Resource guard

`/data5` has limited remaining space. Start with one task and `--n-concurrent 1`; inspect Docker and run size before any multi-task or parallel run. Keep the feedback interface restricted to agent-visible shell output, exit status, file/process/service observations, and permitted test output. Hidden verifier tests and oracle solutions are evaluation-only.

## Verified smoke

On 2026-06-24, `tb run` completed `hello-world` with `--agent nop` in 54 seconds. The expected 0/1 result confirms task container setup, session creation, verifier execution, lockfile, and result artifact creation; it is not an agent-performance result.

Artifact: `experiment_runs/terminal_bench_nop_smoke_20260624/2026-06-24__08-44-28/`.
