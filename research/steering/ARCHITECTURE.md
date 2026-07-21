# Steering research architecture

## Boundaries

- `core/`: only methods with positive evidence in at least one controlled run.
- `adapters/`: model, benchmark, and conditioning-format dependencies.
- `analysis/`: reusable artifact-only analysis; never launches models.
- `visualization/`: deterministic plots from analysis outputs.
- `experiments/`: thin scripts and immutable configuration for one research question.
- `runs/`: outputs and traces; never imported by source code.

Core modules must not hard-code model paths, layer counts, benchmark task ids,
or run directories.  A method that is being explored remains in an experiment's
`experimental/` directory until causal validation justifies promotion.

## Artifact discipline

Every saved vector records model id, layer, hidden size, extraction method,
representation, aggregation, conditioning format, unit id, and config hash.
Analysis and evaluation reject artifacts whose metadata do not match the model.
