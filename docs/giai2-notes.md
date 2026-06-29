# GIAI2 Notes

This note condenses the two project-local docs that were previously under `giai2/`: `GIAI2_RESEARCH_DEV_GUIDE.md` and `GAIA2_ENV_AND_AGENT.md`.

## Current Paths

```text
workspace:      /data5/ninghan/tlm
giai2 source:   /data5/ninghan/tlm/benchmarks/giai2
experiment out: /data5/ninghan/tlm/benchmarks/giai2/experiment_runs
dataset:        /data5/ninghan/tlm/datasets/giai2/gaia2-cli
qwen service:   /data5/ninghan/tlm/services/qwen
model service:  http://127.0.0.1:8006/v1
current model:  qwen3-32b-awq-tool
model path:     /data5/ninghan/tlm/models/Qwen3-32B-AWQ
service tmux:   qwen32_awq_service
service log:    /data5/ninghan/tlm/services/qwen/logs/qwen3-32b-awq-tool-8006.log
```

Check the service:

```bash
tmux attach -t qwen32_awq_service
tail -f /data5/ninghan/tlm/services/qwen/logs/qwen3-32b-awq-tool-8006.log
```

## Environment Setup

ARE Python environment:

```bash
cd /data5/ninghan/tlm/benchmarks/giai2
uv venv -p 3.12
source .venv/bin/activate
uv pip install -e .
```

With GUI support:

```bash
BUILD_GUI=1 uv pip install -e ".[gui]"
```

Main commands:

- `are-run`: run one scenario.
- `are-benchmark`: run benchmark or GIAI2 dataset batches.
- `are-gui`: start the web GUI, usually on port `8080`.

Gaia2 CLI container runner:

```bash
cd /data5/ninghan/tlm/benchmarks/giai2/gaia2-cli
cp .env.example .env
make gaia2-hermes
# or
make gaia2-oc

uv run --project runner --python 3.12 gaia2-runner run-config \
  --config runner/examples/quickstart_hermes.toml
```

Serve a run trace:

```bash
uv run --project runner --python 3.12 gaia2-runner serve \
  --output-dir /data5/ninghan/tlm/benchmarks/giai2/experiment_runs/<run_name>
```

## ARE Interaction Model

ARE is the Meta Agents Research Environments framework.

- `are/` is the general Python simulation framework.
- `gaia2-cli/` is the containerized Gaia2 benchmark stack.
- A scenario includes apps, events, validation logic, and a task prompt.
- The environment exposes app methods as tools and records actions, observations, events, and validation artifacts.
- The agent acts through tool interfaces; it should not directly mutate environment state.

Important files:

- `giai2/are/simulation/environment.py`: event loop, app registration, tool collection, logs.
- `giai2/are/simulation/scenario_runner.py`: connects scenario, environment, agent, validation, and trace export.
- `giai2/are/simulation/agents/default_agent/`: default ReAct-style agent loop.
- `giai2/are/simulation/tool_utils.py`: app tool metadata and schema conversion.
- `giai2/gaia2-cli/runner/gaia2_runner/runner.py`: containerized scenario runner.
- `giai2/gaia2-cli/runner/examples/`: Hermes, OpenClaw, and OpenAI-compatible config templates.

## Agent Flow

The default agent loop:

1. `ScenarioRunner._run_with_agent()` creates agent config and injects model/provider/endpoint.
2. `AgentBuilder` constructs `ARESimulationAgent` with the same `Environment`.
3. `ARESimulationAgent.init_tools()` collects app tools and wraps them for the agent.
4. The system prompt includes tool descriptions, current time, and notification-system details.
5. The agent receives user messages and environment notifications from the notification queue.
6. Each LLM turn should produce a structured tool action.
7. `JsonActionExecutor` parses the action, finds the tool, and executes the app method.
8. Observations return to the agent loop until `final_answer` or limits are reached.
9. The environment records actions and validation-relevant events.

For GIAI2 feedback research, the key is whether the agent receives feedback/notifications and then changes its later reasoning or tool use.

## Visualization

ARE GUI is useful for scenario and environment inspection, not historical trajectory viewing:

```bash
ssh lab-50 '
tmux kill-session -t are_gui 2>/dev/null || true

RUN_DIR=/data5/ninghan/tlm/benchmarks/giai2/experiment_runs/scenario_visual_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

tmux new-session -d -s are_gui "
cd /data5/ninghan/tlm/benchmarks/giai2 &&
export HF_DATASETS_OFFLINE=1 &&
.venv/bin/are-gui \
  --hostname 0.0.0.0 \
  --port 8080 \
  --scenario_id scenario_find_image_file \
  --agent default \
  --provider local \
  --model openai/qwen-local \
  --endpoint http://127.0.0.1:8000/v1 \
  --ui_view SCENARIOS \
  --dataset-path /data5/ninghan/tlm/datasets/giai2/gaia2-cli \
  > $RUN_DIR/are_gui.log 2>&1
"

echo "log: $RUN_DIR/are_gui.log"
'
```

Open:

```text
http://<lab-50-host>:8080
```

If direct access fails:

```bash
ssh -L 8080:127.0.0.1:8080 lab-50
```

Run traces are generated per run/scenario:

```text
/data5/ninghan/tlm/benchmarks/giai2/experiment_runs/<run_name>/index.html
/data5/ninghan/tlm/benchmarks/giai2/experiment_runs/<run_name>/<split>/<scenario_id>/trace.html
```

Trace inspection checklist:

- Are tool calls structured, or emitted as literal `<tool_call>` text?
- Does the run reach environment feedback or notification turns?
- Does the agent update state, plan, or tool choice after feedback?
- Is failure caused by model ability, protocol mismatch, tool/runtime error, or judge mismatch?

## Experiment Gate

Before feedback-quality ablation, require:

1. GIAI2 native multi-turn/notification path.
2. Mature agent scaffold, preferably OpenClaw.
3. Verified local model service with structured tool calls.
4. Small scenario set with real feedback opportunities.
5. At least one run that reaches feedback/notification turns.
6. Manual trace inspection showing the agent reacts to feedback.

Then compare:

- baseline/default feedback
- random feedback
- all/noisy feedback
- high-quality feedback

