# Activation Steering Research

This package separates reusable steering methods from models, benchmarks, and
individual experiments.  Both the mature Qwen3.5-4B line and exploratory
Qwen3-32B work use the same core interfaces.

## Layout

```text
core/           validated extraction, application, compiler, and metrics
adapters/       model, benchmark, and conditioning-format integration
analysis/       reusable analysis over saved artifacts
visualization/  deterministic figures from analysis outputs
experiments/    experiment-specific scripts, configs, and evidence notes
runs/           immutable outputs; source code must never import a run
tests/          fast method-level regression tests
```

See `ARCHITECTURE.md` for boundaries and artifact requirements.

## Validated core methods

- Extraction: paired prompt-conditioned mean activation delta.
- Application: additive activation steering, including generation-only and
  calibrated multi-layer use.
- Compiler: ridge text-representation-to-vector mapping.
- Evaluation: geometry, cross-state consistency, task-level paired effects,
  and behavior safety metrics.

Methods without positive controlled evidence do not belong in `core/`.
They remain inside the owning experiment's `experimental/` directory.

## Active experiments

- `experiments/skill_edit_4b/`: mature 4B SkillOpt/ALFWorld evidence and
  regression reference.
- `experiments/reflexion_T/`: faithful Reflexion text validation, model-native
  layer/strength calibration, compiler training, and causal evaluation on 4B
  and 32B configurations.

## Development rule

Experiment scripts are thin orchestration.  If a script reimplements vector
extraction, hooks, compiler fitting, or metrics, that logic must first be moved
into the corresponding reusable module and covered by a test.
