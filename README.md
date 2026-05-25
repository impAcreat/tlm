# Classroom

Research prototype for **teacher-student self-evolving LLM agents**.

The project studies harness evolution: a teacher agent tests heterogeneous student agents, diagnoses failures, proposes structured patches to their harnesses, and updates its own teaching memory based on whether those patches helped. Students also maintain their own memories and skills, so learning happens at the harness layer before any model fine-tuning.

## Layout

```text
src/classroom/
  agent/
    student.py          # StudentAgent: solve tasks and absorb advice
    teacher.py          # TeacherAgent: diagnose, patch, self-improve
    types.py            # StudentFeature, StudentAttempt, TeachingAdvice
  harness/
    state.py            # prompt / skills / memory / tool policy / patch history
  interaction/
    classroom.py        # multi-student, multi-generation orchestration
    workflow.py         # PocketFlow node graph for one classroom round
    peer.py             # peer experience bank
  benchmark/
    humaneval.py        # adapter over OpenAI human-eval package
  factory/
    builders.py         # default teacher/student/model initialization
  model/
    client.py           # OpenAI-compatible API client
```

## Harness Status

This is now closer to a research prototype than a toy demo:

- benchmark execution uses the official `human-eval` package;
- interaction is represented as a PocketFlow workflow: attempt -> teach -> revise -> teacher reflection;
- harness state is inspectable and patchable at multiple layers;
- teacher and students are separate agents with separate harnesses;
- peer experience is a first-class input to teacher advice.

PocketFlow is intentionally used only for orchestration. OpenHarness looks useful for production-style tool/memory runtimes, but it is too heavy for the first experimental core because it would hide the harness variables we want to study.

## Models

Put model URL/key in a local JSON file:

```bash
cp configs/model.example.json configs/model.local.json
```

Then edit `configs/model.local.json`:

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-REPLACE_ME",
  "student_model": "gpt-4.1-mini",
  "teacher_model": "gpt-4.1",
  "temperature": 0.0
}
```

Hosted GPT-style run:

```bash
python scripts/run_teacher_student_classroom.py \
  --config configs/model.local.json \
  --students 3 \
  --tasks 3 \
  --generations 1
```

Local OpenAI-compatible deployment, such as vLLM:

```bash
scripts/start_qwen3_32b.sh
python scripts/run_teacher_student_classroom.py --config configs/model.qwen3-32b.local.json
```

Ollama OpenAI-compatible endpoint is similar:

```bash
export CLASSROOM_BASE_URL=http://localhost:11434/v1
export CLASSROOM_API_KEY=EMPTY
python scripts/run_teacher_student_classroom.py --model qwen2.5-coder:7b
```

See `configs/model.example.json`, `configs/model.local.example.json`, and `.env.example` for placeholders.

## Benchmark

Initial benchmark: **HumanEval** through the `human-eval` package.

The current classroom trajectory output is JSONL:

```text
runs/classroom/rounds.jsonl
```

Each row stores before/after scores, teacher advice, and the patch payloads applied to student harnesses.

There is also a readable transcript:

```text
runs/classroom/transcript.md
```

It shows each round's student, task, before/after result, teacher diagnosis, and applied harness patches.

## Notes

- No large-model test is run by default; fill API/local model settings first.
- `human-eval` uses multiprocessing for isolation, so run from a normal shell if a sandbox blocks local socket/process creation.
- The next natural step is to make patch acceptance explicit: students can accept, reject, or rewrite teacher patches instead of always applying them.
