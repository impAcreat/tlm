# Experiment Log

Restarted 2026-08-03.  Everything before that date is distilled into the
"Established record" sections below; the full 70-entry history is preserved in
`archive/EXPERIMENT_LOG_full_through_20260803.md` and must not be edited.

New entries are appended under "Log" with the usual shape: question / protocol /
numbers / evidence boundary, and an explicit gate pass/fail.

---

# Established record (through 2026-08-03)

## 0. What the project is

Turn an agent's own textual experience — SkillOpt skill edits, Reflexion
reflections — into **activation-space interventions** on a multi-step
interactive agent (ALFWorld), and measure **task success**, not geometry.

Two lines exist.  **Line A (4B / SkillOpt)** is the mature methodological
reference and is frozen.  **Line B (32B / Reflexion)** is the active line.

Two claims are being pursued, in dependency order:

- **(a) vectorizable** — a vector extracted from the text raises task success.
  *Never demonstrated at scale; largest n ever run is 20.*
- **(b) compilable (T)** — the vector can be predicted from the text alone.
  Geometry established, small-n causal support exists.  **(b) is worthless
  unless (a) holds**, so (a) has priority.

## 1. Frozen methods — do not re-litigate

- **Extraction**: prompt-conditioned contrast — byte-identical prompts except
  the skill/reflection section, last-token hidden state, mean over a frozen
  state sample.  Cross-state cosine 0.86 @L14 / 0.76 @L18 (4B).
- **Trajectory contrast is a length artifact.**  Full-trajectory success/fail
  separation is reproduced by `n_steps` alone (0.96 probe).  Never use it;
  length-control every representation claim.
- **Injection**: generation-only (decode steps only, prompt encoding untouched)
  strictly beats full-position.  Norm-calibrated multi-layer — {14,18,22}, each
  at 1× its own mean-delta norm — beats single-layer overdrive at a fraction of
  the invalid-action cost.  **Dose cliff is sharp**: 1.5× per layer collapses.
- **Greedy only.**  Temperature 0.7 drowns single-vector effects.
- **Unit format matters**: `skillopt.optimizer.skill.apply_edit` section
  insertion is the correct manipulation for edit units; appending under
  "## Additional Hints" is materially weaker and produces an anomalously weak
  extraction reference.
- **Shared component**: 92–94% of any unit vector is a shared
  "instruction present" direction.  All geometry metrics must be mean-removed.
- **Hidden-relative dose (Line B)**:
  `‖Δh‖ = multiplier × natural_rho × ‖h_t‖`, clamped to `[q10·m, q90·m]`, where
  `natural_rho = ‖Δh_text‖/‖h_base‖` is how far the real text moves the hidden
  state.  **m = 1.0 therefore means "as large as the text's own perturbation",
  not "weak".**  Layer indices and doses never transfer across models.
- **Env replay**: recorded `model_response`s replay exactly (1188/1188 feedback
  matches), so per-step prompts are recoverable for any recorded rollout.

## 2. Line A results (4B / SkillOpt / ALFWorld) — frozen

- **Skill improvements are a global condition signature**, not per-task repair
  directions.  The good steering object is the prompt-conditioned contrast.
- **Unit edits compose linearly**: step2 = step1 + 3 ranked edits (byte-exact);
  the three per-edit vectors are near-orthogonal and sum to the step increment
  (cos 0.97).  **Unit-level injection beats sum-direction injection** — the
  whole increment `d12` produced zero flips.
- **Unit vectors are causally active**: 2–5/16 vs base 1–2/16 across arms.
- **Partial double dissociation (R6)**: search unit S gives +3 in-domain with
  exactly zero cross-domain effect; protocol unit P is weak everywhere.
- **Compiler T is learnable (R7)**: ridge from unit-text representations
  (L14+18+22 concat, text_mean) to conditioning vectors, 298 reflexion hints,
  grouped held-out residual cosine **0.59 vs permutation null 0.00**.
  Cross-domain (hints → skillopt edits) 0.28–0.33.
- **Hint-level causal transfer failed (R8/R9)**: extracted and predicted hint
  vectors inert on retry (1–2/20 ≈ base ≈ random).  **But the hint TEXT itself
  was only +2/20** — there was no effect to transfer.  Boundary hypothesis:
  mean-delta vectorization moves global behavioral-mode content, not weak
  episodic advice.  **Always run the text-prepend control before any vector arm.**
- **Compiled strong units are causally active (R10)**: T-predicted S reaches
  5/16 vs base 2/16, T-predicted P 3/19 vs 1/19.  Caveats: the append-style
  extraction reference was anomalously weak, and the predicted-vector
  specificity 2×2 was never run.
- **R11 — the most important negative, and the reason for the current direction.**
  On the full 140-task shared set, undirected injection is **not inert but
  net-zero**:

  | arm | success | net | churn |
  |---|---|---|---|
  | `mt_gmb_x` vs `mt_bad` | 41/140 vs 40/140 | +1 | **fixed +13 / broke −12** |
  | `mt_s1_TS` vs `mt_s1` | 57/140 vs 56/140 | +1 | **fixed +11 / broke −10** |

  Injection flips ~25 of 140 outcomes but fixes and breaks in near-equal
  numbers.  **The broken tasks are ones the base skill already solved** —
  injecting a "good direction" into tasks needing no repair is pure
  perturbation.  Consequently the earlier "recovers ~75% of the text effect"
  holds **only on repaired subsets** and must not be extrapolated to a
  heterogeneous full set.  **Value requires selectivity**: route the vector to
  states that have repair headroom, skip the rest.
- **Mechanism**: the vector does **not** mimic the good prompt's local
  next-token policy (teacher-forced shift −0.093/token vs good prompt +0.305);
  its effect accumulates over the generated reasoning stream across steps.
  Logit-lens decoding of the conditioning direction is semantically void — the
  direction is not vocabulary-aligned (reported as a null result).

## 3. Line B results (32B / faithful Reflexion) — active

- **Exploratory port is frozen as exploratory.**  Proportional-depth layer
  mapping (4B L18 → 32B L36) plus a fixed 3.9×-extracted-norm dose gave
  extracted 0/12, predicted 0/12, random 2/12, invalid-action rates 34–41% vs
  5.0% baseline.  Classified as an off-manifold dose/protocol failure, not
  evidence against vector content.  No number from that run may be promoted.
- **Clean run is pre-registered.**  Task-disjoint split frozen before
  generation: Train 66 tasks × 4 seeds, Dev 16 × 2, Test 85 × 1.
  Split SHA256 `5fa0a3c1fae7f3464726a1d0e808550cf131f435621ff8fd93033dc895064057`.
  **Test is untouched and stays untouched until G3.**
- **G1 (text gate) — PASS.**  296/296 Train+Dev groups collected, no missing or
  duplicate ids, no runtime errors.  75 initial successes, 221 eligible initial
  failures.  On those failures: matched two-retry no-memory control 42/221
  (19.00%), faithful same-model full Reflexion 85/221 (38.46%) —
  **absolute lift +19.46 pp**, over the pre-registered +15 pp threshold.
  Paired cells: Reflexion-only 59, control-only 16, both 26, neither 120.
- **Effect heterogeneity is material and must stay visible.**  Strong for
  `look_at_obj_in_light` (13/21 vs 2/21) and `pick_and_place` (24/32 vs 6/32);
  weak for cooling (12/53 vs 9/53), heating (12/34 vs 9/34), two-object
  placement (6/38 vs 4/38), cleaning (18/43 vs 12/43).  A vector compiled from
  text cannot beat text on the families where the text itself does nothing.
- **Dataset**: 386 reflections / 221 failed groups / 73 distinct tasks;
  85 `text_success`, 59 `paired_effective` (Train 73/53, Dev 12/6).  Both
  pre-registered sufficiency thresholds pass.
- **Extraction complete**: all 64 layers over the 200 frozen states for all 85
  `text_success` units (`phase2/vectors_text_success_shard{0,1}.pt`).
- **G2 (layer/dose) — incomplete and underpowered.**  The Dev pilot ran
  **only m = 0.25** on 8 units.  Baseline 0.375, text upper bound 0.750.
  L23 was the sole condition marked content-specific (+0.125, beating random
  and mismatched), but at n = 8 one flipped task is ±0.125 and every CI
  includes zero.  A neighbour sweep (L21/22/24/25, m = 0.25) finished
  collecting on 2026-07-23 and **was never summarized**.
- **Dev-A / Dev-B are frozen** (`phase3/causal_task_splits.json`): the 37
  eligible Train `text_success` tasks are split by task type via
  SHA256(seed|task_id) into **Dev-A 19 / Dev-B 18**; the original 8 Dev units
  are recorded as `pilot`.  Content SHA256 `4b18f32d…`.  Selection happens on
  Dev-A, confirmation on Dev-B, Test remains untouched.

## 4. Dead ends — do not repeat

- Trajectory-contrast extraction (length artifact).
- Naive text-semantics → vector mapping (alignment ≈ 0.00); requires empirical
  extraction or a learned map.
- Vectorizing weak episodic advice (4B hints).
- Whole-increment / sum-direction injection.
- Unscaled multi-layer injection (deltas compound and destroy generation).
- Proportional-depth layer mapping and cross-model dose transfer.
- Logit-lens interpretation of conditioning directions.
- Temperature-sampled causal arms.

## 5. Open questions, ranked

1. **(a) at scale**: does an extracted reflection vector raise task success on a
   properly powered set?  Never tested above n = 20.
2. **Adaptive dose**: is a single global multiplier viable, or must strength be
   per-unit / per-state?  Listed as a lever since Round 1, never run.
   Measurable as `oracle-per-unit-m` minus `best-global-m`.
3. **Selectivity / routing** (from R11): can churn be avoided by injecting only
   where there is repair headroom?  R11's follow-ups (b) analytical repaired-23
   table and (c) skip-control were never finished.
4. Predicted-vector specificity 2×2 (R10 caveat B).
5. G3 on the untouched Test split.

**Reporting rule adopted 2026-08-03**: every causal arm reports
**fixed / broke decomposition**, not just net success delta.  R11 showed net
hides churn.

## 6. Operations

- **Canonical steering checkout: `lab-130:/sdc/ninghan/tlm`**, in sync with
  `origin/main`.  `lab-50:/data5/ninghan/tlm` holds the Qwen3-32B weights, the
  ALFWorld benchmark data, and a copy of the clean run dir; only GPUs **0–3**
  are available there.
- Extracted 32B vectors live in the clean run's `phase2/` — copying them beats
  re-extraction (~3–4 GPU-hours saved).
- Env: `envs/skillopt-qwen35-vllm` (transformers 5.x) is the only env that loads
  Qwen3.5 for HF forwards; `-cu128` is for env replay.  Qwen chat template needs
  explicit `enable_thinking=False`.
- Throughput reference: one ALFWorld rollout ≈ 54 s median / 70 s mean on an
  A100-80GB at 32B bf16, one worker per card.
- Long jobs in tmux, logs in the run dir, every eval loop resumable (append to
  `results.jsonl`, skip done ids).  Shut down vllm servers when a run finishes.
  Check `nvidia-smi` owners before taking GPUs.
- Never overwrite anything under `runs/latent_skillopt_repro42_20260717/`.

---

# Log

## 2026-08-03 Dev-A dose sweep at L23 — effect appears only above the old grid; no evidence dose must be adaptive

**Question.** On a properly powered set, does an extracted reflection vector raise
task success over matched controls, and must the steering strength be per-unit?

**Protocol.** Dev-A (19 task-disjoint Train `text_success` tasks from the frozen
`phase3/causal_task_splits.json`, SHA `4b18f32d…`), Qwen3-32B, layer 23,
generation-only, greedy, max 35 steps.  Arms baseline / text / extracted /
norm-matched random / deterministic mismatched donor, at multipliers
**0.5, 1.0, 2.0, 4.0** of the hidden-relative natural dose.  266/266 evals
completed, 19/19 units complete, zero runtime errors.  Test untouched.
Dev-B untouched.

**Results.**  Baseline 1/19 (0.053).  Text upper bound 14/19 (0.737), paired
delta +0.684 — high by construction, since these are `text_success` units.

| multiplier | extracted | random | mismatched |
|---:|---|---|---|
| 0.5 | 1/19 | 1/19 | 2/19 |
| 1.0 | 1/19 | 0/19 | 0/19 |
| **2.0** | **4/19 (0.211)** | 2/19 (0.105) | 1/19 (0.053) |
| 4.0 | 1/19 | 1/19 | 2/19 |

At m=2.0: extracted paired delta **+0.158 [-0.053, +0.368]**, W/L **4/1**;
random +0.053 (W/L 2/1); mismatched +0.000 (W/L 1/1).  The summarizer marks
m=2.0 content-specific — extracted beats both controls — but **the interval
crosses zero at n=19, so this is directional, not significant.**

**Finding 1 — the previous grid could not have found this.**  The dose response
is non-monotone with a single peak at 2x: 0.5x and 1.0x sit exactly at baseline,
4.0x collapses back.  The abandoned grid was {0.25, 0.5, 1.0}, and the earlier
Dev pilot ran only m=0.25.  **A null at those doses was uninformative, not
evidence against vector content.**  Matching the magnitude of the text's own
perturbation (m=1.0) is not enough; the mean direction averaged over states
carries less useful component per unit norm than the state-specific natural
delta, so overdrive is required — consistent with the 4B line needing 2-3x.

**Finding 2 — no evidence that dose must be adaptive.**  `adaptive_dose_value`:
extracted best-global 0.211 @ m2.0 vs oracle-per-unit 0.263, gap **+0.053**.
The norm-matched random arm has an **identical** gap of +0.053, so
`excess_over_control` is **exactly 0.000**.  The apparent per-unit dose
heterogeneity is fully explained by after-the-fact selection over four doses.
One global multiplier is adequate at this sample size; a per-unit dose policy is
not currently justified.

**Finding 3 — the vector recovers about a quarter of the text effect.**
+0.158 of +0.684 = 23%.  The 4B line's "recovers ~75% of the text effect" figure
came from cherry-picked repaired subsets; on a pre-registered, non-cherry-picked
set the honest number is much smaller.

**Evidence boundary.**
- n=19, differences of 1-3 tasks; no statistical claim.  Ordering
  (extracted > random > mismatched = baseline) is consistent at the peak dose
  only.
- Floor effect: baseline is 1/19 with 18/19 rollouts hitting the 35-step
  timeout.  Conclusions are about a near-non-functional base agent.
- **Harness concern**: `format_repair_rate` is 0.47-0.59 at m<=2.0, i.e. the
  model fails to emit `<action>` tags on roughly half of all steps and is being
  repaired by normalization.  At m=4.0 it drops to 0.045 while success stays at
  baseline — well-formed but wrong actions.  This asymmetry is unexplained and
  should be understood before the format-repair path is trusted.
- m=1.0 is flagged content-specific only because both controls fell to 0/19;
  that is specificity by the controls degrading, not by extracted improving.
  Do not report it as a positive condition.

**Artifacts.** `phase2/calibration_devA_L23_shard{0..3}.jsonl`,
`phase2/calibration_devA_L23_summary.{json,md}`,
`phase2/launch_devA_L23.sh`.

**Next.** Dev-B (18 held-out tasks) as an independent confirmation of the
Dev-A-selected configuration L23 @ m=2.0, arms baseline/text/extracted/random/
mismatched — 90 rollouts, about 30 minutes on four A100s.  Selection is now
frozen; Dev-B is one-shot and must not be reused.
