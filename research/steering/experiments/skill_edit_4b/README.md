# Qwen3.5-4B Skill-Edit steering reference

This is the maintained reference experiment, not a deprecated model line. It
contains the evidence and reproducibility scripts from the iterative 4B work.

## Scientific boundary

- The validated extraction family is a paired, same-state prompt contrast:
  conditioned hidden state minus matched base hidden state, averaged across
  states.
- Additive activation steering has positive controlled evidence in this line.
- Trajectory success/failure mean differences were length-confounded and are
  not part of the reusable core.
- Layer indices and steering dose are model- and protocol-specific. They are
  reference results here, not defaults for 32B.

## Layout

- `scripts/`: original 4B data preparation, extraction, analysis, compiler and
  causal-evaluation entry points, retained for reproducibility.
- `PIPELINE_20260717.md`: exact 2026-07-17 run order, environments and evidence
  cautions.
- `research/steering/core/`: reusable validated method implementations.
- `research/steering/analysis/` and `visualization/`: reusable artifact-level
  analysis and plotting.

Historical run artifacts remain under
`research/steering/runs/latent_skillopt_repro42_20260717/`; durable claims are
recorded in `experiments/EXPERIMENT_LOG.md`.
