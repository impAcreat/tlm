# ScienceWorld + AppWorld steering pilot — 2026-07-19

## Goal

Move the verified ALFWorld steering stack from lab50 to lab130 and test the same
`SkillOpt textual delta -> prompt-conditioned activation delta -> gen-only multi-layer injection`
pipeline on two new interactive benchmarks. This is a mechanism pilot, not a benchmark-scale run.

## Environment and migration

- Fresh checkout: `/sdc/ninghan/tlm-steering`, branch `exp/sciworld-appworld-steering`.
- Python 3.12 env: `envs/steering-hf`; torch 2.6.0+cu124, transformers 5.12.1.
- Qwen3.5-4B was copied over the private LAN after host-to-host SSH trust failed. The two shard
  SHA-256 values match lab50: `26a93f...c61` and `cb544b...e188`.
- ScienceWorld 1.2.3 and AppWorld 0.1.3.post1 real task/evaluator smokes passed. AppWorld data
  (193 MB) was copied from lab50 and verified.
- Lab130 `/` is full, so HOME, cache, and tmp are redirected under `/sdc/ninghan/tlm-steering`.

## ScienceWorld

- SkillOpt: train 6 / selection 3 / test 3, one update step, max 20 steps, soft-score gate.
- Metamind GPT-5.5 intermittently returned 500/EOF. The first text stage generated no patch;
  retrying the saved trajectories recovered one two-edit failure patch and a 1255-character skill.
- Extraction: 12 training states; layers 14/18/22; good-minus-bad mean delta; gen-only; greedy;
  alpha 1.0. Cross-state cosine: 0.815/0.789/0.779. Mean norm: 1.512/2.413/4.385.
- Held-out 3-task arms, all hard=0: bad soft 0.0067; textual-good 0.1233; steered 0.0067;
  matched-random 0.0233. Steered changed every trajectory and 1/3 first decisions, but did not
  transfer the textual gain or beat random.
- Artifacts: `outputs/skillopt_scienceworld_light/retry_text/` and
  `outputs/steering_scienceworld_light/summary.json`.

## AppWorld

- SkillOpt: train 3 / selection 3, one update step, max 10 interactions, soft-score gate.
- The analyst produced one failure patch with two applied edits; candidate length was 89 -> 1385.
  Candidate and initial selection soft were both 0.5, so the candidate was rejected.
- The redundant best-skill test (best == initial) was stopped after the baseline held-out test.
  The rejected candidate is used only as a nonzero textual contrast for the mechanism test.
- Extraction: 9 training states; layers 14/18/22; gen-only; greedy; alpha 1.0. Cross-state cosine:
  0.852/0.791/0.810. Mean norm: 1.619/2.848/9.576.
- Held-out 3-task arms: bad/textual-good/steered/random all hard=0 and soft=0.5. All three
  interventions changed every first decision, so this shows controllability but not directional or
  task-level advantage.
- Artifacts: `outputs/skillopt_appworld_light/steps/step_0001/` and
  `outputs/steering_appworld_light/summary.json`.

## Evidence boundary

- Cross-benchmark extraction and hook application are operational, and both benchmarks yield coherent
  cross-state directions. ScienceWorld also shows a sizeable textual-skill soft-score improvement.
- These pilots do not establish steering-vector task gain outside ALFWorld: alpha=1 failed to reproduce
  the ScienceWorld textual gain, and AppWorld remained at its no-op partial-credit floor.
- No alpha sweep was run, to keep the requested experiment light and avoid result fishing.
- AppWorld needs dynamic compilation disabled. Custom adapters must persist
  `predictions/<id>/conversation.json`; upstream SkillOpt reflection does not read trajectories directly
  from `results.jsonl`.
