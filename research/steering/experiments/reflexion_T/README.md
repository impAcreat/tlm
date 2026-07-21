# Reflexion-to-steering compiler experiments

This directory contains only experiment orchestration and configuration.  All
extraction, application, compiler, and metric implementations live in
`research/steering/core`; model and benchmark details live in `adapters`.

## Evidence stages

1. Validate full, same-model Reflexion text against matched no-reflection retries.
2. Select layers and calibrate steering strength on development groups only.
3. Extract vectors and train the compiler on pre-declared training groups.
4. Evaluate baseline/text/extracted/predicted/random on untouched test groups.

## Reusable entry points

- `scripts/collect_reflections.py`: matched textual Reflexion collection.
- `scripts/summarize_text_gate.py`: G1 text-effect summary.
- `scripts/build_reflection_dataset.py`: convert validated traces to compiler units.
- `scripts/extract_vectors.py`: sharded, resumable paired prompt-contrast extraction.
- `scripts/train_compiler.py`: group-held-out ridge fitting and diagnostics.

Layer choice and steering dose are intentionally absent from model defaults.
They must be selected on development groups before compiler and causal test runs.

Unvalidated methods stay under `experimental/` and must not be registered as
core methods until a causal arm beats its matched controls.
