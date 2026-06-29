# EFM 自进化 — 执行方案（交接给执行 agent）

> 目标：实现**初步可行的 EFM 自进化（EFM-self-evolution）**。
> 本文件是可冷启动执行的实验计划。先读完"背景"和"环境"，再按 Phase 顺序做。
> 每个 Phase 完成后，按 `experiments/EXPERIMENT_LOG.md` 的格式记录，再进下一步。
> 纪律遵循 `.claude/skills/tlm-remote/SKILL.md` 的实验闭环：每步先写清问题/判据，先 smoke 再放大，先看 trace 再信聚合指标。

---

## 0. 背景（必读）

EFM = Environment Feedback Module（`research/efm/`），独立于 agent/环境/SkillOpt，只优化"该把哪些环境事实暴露给 agent"。
- **constitution**（`constitution.py`）：固定核，不在自进化中改。
- **policy**（`policy.py`）：可学 skill = 少量规则 + few-shot，**带版本、有上限**。
- 链路：`env.step → Step EFM → agent`；每 20 episode → 轨迹审计 → 提 policy patch → 冻结轨迹 gate → 新 policy。

**已确立的结论（不要重复验证）**：
1. 反馈质量短板是 **skill 问题，不是模型问题**：30B+旧skill ≈ 4B+旧skill；4B+好skill ≫ 二者。EFM 用小模型（4B）即可，**优化 skill 才是杠杆**。
2. 质量主轴是**完备性（over-hedge）**，不是一致性。
3. 初始 skill 已优化：仅改 constitution，30 条评测 一致性/完备性/高效性 = **90/87/77**（旧版 90/33/80）。
4. 自进化的契约 bug 已修；闭环能端到端到 gate；但**当前 gate 是噪声 LLM-A/B，不可信，必须先换**（见 Phase 1）。

**"初步可行"的完成定义（Definition of Done）**：
- (A) 在**可信（确定性）gate** 下，至少发生一次 policy 升级 v0→v1，且 4 维质量（除有效性）**不回退**；
- (B) **在线反事实**证明 feedback 对 agent 有用：v0-skill 臂在 interaction success / 步效率上**优于 raw 臂**；
- 二者同时满足即达成初步可行的 EFM 自进化。

---

## 1. 环境与资源（lab-50）

- 连接：`ssh lab-50`，根目录 `/data5/ninghan/tlm`。命令模板：`ssh lab-50 'cd /data5/ninghan/tlm && <cmd>'`。
- Python：`/data5/ninghan/tlm/envs/skillopt-qwen35-vllm-cu128/bin/python`
- 模型端点（OpenAI 兼容，api-key `token-abc123`）：
  - **Target（ALFWorld agent）+ 小 EFM**：Qwen3.5-4B @ `http://localhost:8007/v1`（备 8008），model id `Qwen/Qwen3.5-4B`
  - **Optimizer（写 skill / 审计 / 强裁判）**：Qwen3-30B @ `http://localhost:8001/v1`，model id 见 `/v1/models`
- GPU：项目仅用 `CUDA_VISIBLE_DEVICES=0,1,2,3`；8001 是他人服务，**只发推理、不重启不占显存**。
- 长任务用 `tmux`/`nohup` 并报告日志路径；ALFWorld worker 需 `ALFWORLD_WORKER_START_METHOD=spawn`。

### 关键文件
- EFM 核心：`research/efm/{constitution,policy,updater,runtime,prompts,models}.py`
- 备份：`research/efm/.bak_20260625_053644/`、`research/efm/constitution.py.bak_preopt`
- SkillOpt 接入：`benchmarks/skillopt/skillopt/integrations/efm.py`、`skillopt/envs/alfworld/{adapter,rollout}.py`
- 配置：`benchmarks/skillopt/configs/alfworld/efm.yaml`（继承 `default.yaml`）
- 启动：`benchmarks/skillopt/scripts/run_alfworld.sh` → `scripts/train.py --config configs/alfworld/efm.yaml`；离线评测脚手架 `scripts/eval_only.py`
- 诊断/评测脚本：`research/efm/diagnostics/`
  - `efm_eval_v2.py`：4 维确定性评测（一致/完备/高效；有效性 online-only）
  - `efm_quality_eval.py`：A/B/C 三格生成 + 打分；`eval_records.json`：30 条固定评测集
  - `repro_policy_proposal.py` / `efm_offline_loop.py` / `efm_gate_probe.py`：proposal/loop/gate 复现
- 关键事实：rollout 中 **agent 只看到精炼后的 StepFeedback，看不到 raw observation**（`rollout.py` 注释确认）。所以低完备反馈=对 agent 隐藏信息。

### 评测 4 维（口径固定，见 `efm_eval_v2.py`）
| 维度 | 定义 | 测法 |
|---|---|---|
| 一致性 Consistency | 只陈述当前 obs 支持的事实，无幻觉 | 确定性 grounding（实体/否定/transform 动词）+ 30B 仅兜底关系型幻觉 |
| 完备性 Completeness | 不漏当前 obs 的决定性事实（hedge=漏报） | obs 有明确事实却输出 ambiguity → 不完备 |
| 高效性 Efficiency | 简洁、不复述、不掺 meta | 长度 + meta 短语 |
| 有效性 Effectiveness | 反馈是否真帮到 agent | **只能在线反事实**，离线 N/A |

确定性 checker 已知边界：无数字实体（如 "lettuce"）和"X 现在在 Y"型关系幻觉测不到 → 这类用 30B 兜底，且**始终以 trace 抽检校准裁判**。

---

## Phase 0 — 环境自检 + 复现基线（~30 min）

**问题**：环境是否就绪、当前数字是否可复现。
**做**：
1. 三个端点 `/v1/models` 可达；`python -c "import research.efm"` 通过。
2. 跑 `efm_eval_v2.py`，确认 A/B/C ≈ 一致 90/90/83、完备 33/63/40。
3. 跑 `efm_regen_score.py`，确认优化后初始 skill A2 ≈ 90/87/77。
**判据**：数字与上文 ±5% 吻合 → 通过。否则先排查端点/代码漂移，不要往下走。
**记录**：仅在偏差时记 log。

---

## Phase 1 — 把 gate 换成确定性判据（核心前置）

**问题**：能否用 eval_v2 的确定性 4 维替代噪声 LLM-A/B gate，使"接受/拒绝"可复现、可信？
**为什么**：已证明 30B 当裁判在"列举+缺失"上系统误判；gate 不可信则整个自进化结论不可信。
**做**：
1. 把 `efm_eval_v2.py` 里的 `consistency/completeness/efficiency` 提升为正式模块 `research/efm/quality.py`（让 runtime 与离线评测共用，不要从 diagnostics import）。
2. 在 `updater.py` 增加 gate 模式开关（config 加 `gate_mode: "deterministic" | "llm"`，默认 deterministic）。deterministic gate：对冻结 validation transitions，用 quality 对 **baseline vs candidate** 打 3 维分；**接受条件 = 三维均不回退且至少一维净增**；关系型幻觉用 30B 兜底（candidate 出现 unsafe 幻觉则一票否决）。
3. 单测：构造一个"明显更完备且不掉一致性"的 candidate policy（参考 `efm_quality_eval.py` 里的 B_POLICY），确认 deterministic gate 判 accept；构造一个"臆造事实"的 candidate，确认判 reject。
**判据**：两个单测方向正确，且 gate 决策对同输入可复现（确定性部分无随机）。
**falsification**：若确定性指标无法区分好/坏 candidate（例如冻结 transition 太少、信号饱和）→ 增大 validation transition 数或换更有区分度的 transition，并在 log 说明。
**产物/记录**：`research/efm/quality.py`、`updater.py` diff、单测脚本入 `research/efm/diagnostics/`；log 一条。

---

## Phase 2 — 跑通一窗在线自进化（v0→v1）

**问题**：从优化后的 v0 出发，强 optimizer（30B）能否学出 policy patch，经确定性 gate 接受、且 4 维不回退？
**前提**：Phase 1 完成；确认 `efm.yaml` 的 `feedback_model_role=optimizer` 指向 30B（target=4B）。**先确认 train.py/router 的 env 变量**（OPTIMIZER 指 8001/30B，TARGET 指 8007/4B），不要照搬旧 run 的端点。
**做**：
1. **Smoke**（≥20 episode，触发一次更新窗口）：
   `tmux` 内跑 `scripts/train.py --config configs/alfworld/efm.yaml`（OPTIMIZER→30B、TARGET→4B、合适 out_root），`ALFWORLD_WORKER_START_METHOD=spawn`。
2. 看 `feedback_state.json`：`policy.version`、`policy_updates[].reason`、`corrections` 数。
3. 若 `reason != accepted`：按类别归因——`gate_not_improved`（v0 已很好、真无可升，是健康结果）/ `*_not_object`/`*_support`（schema）/ `proposal_unavailable`（模型/解析）。先看 trace 再下结论。
4. 用 `efm_eval_v2.py` 在固定 30 条上对比 **v0 vs 学到的 policy**：4 维是否不回退、完备/一致是否净增。
**判据（达成 DoD-A）**：出现一次 `accepted` 的 v0→v1，且新 policy 在评测集上 3 维不回退。
**重要**：**不得为凑 accept 放宽 gate**。若 v0 太强导致一直 `gate_not_improved`，这是合理结论——记录"静态 skill 已饱和，policy 的增量需来自 env 特异/longitudinal 模式"，转 Phase 3/4。
**产物/记录**：run 目录路径、policy 版本轨迹、4 维对比表、失败归因；log 一条。

---

## Phase 3 — 在线反事实：测有效性（回答 IDEA 问题 1）

**问题**：feedback 是否真的帮到 agent？evolved skill 是否优于 v0？
**为什么**：有效性是唯一离线测不了、也是 IDEA 的核心监督信号；agent 只看精炼反馈，所以这步直接检验"反馈机制有没有用"。
**做**：固定任务子集（建议 ≥50 episode）、**固定 seed/采样**、**放开 step budget**（旧 smoke 10 步封顶是 0/20 的主因，clean/cool 类任务需 >10；用足够大的 max_steps）。三臂同条件，只换 agent 可见信息：
- **raw**：`feedback_enabled=false`（agent 看原始 obs）
- **v0-skill**：`feedback_enabled=true` + 优化后的 v0
- **evolved**：`feedback_enabled=true` + Phase 2 学到的 policy（若有）
记录每臂：task success、错误恢复（收到反馈后是否停止重复无效动作）、平均步数；并抽 3–5 条 trace 人工核因。
**判据（达成 DoD-B）**：v0-skill 臂在 success 或步效率上**显著优于 raw**。evolved vs v0 给出方向即可。
**falsification**：若 v0-skill ≤ raw → 反馈机制对该 agent 无效，需回看是 completeness 之外的问题（如反馈太抽象、agent 不消费），记录并定位，**不要**当作模型/环境结论。
**产物/记录**：三臂 run 路径、对比表、trace 抽检结论；log 一条。

---

## Phase 4 — 跨窗 longitudinal（净正向，不回退）

**问题**：连续多个更新窗口，policy 是否保持净正向（repair 不带 regression）？
**做**：连续跑 ≥3 个窗口（≥60 episode），每窗后用 `efm_eval_v2.py` 记 4 维 + policy 版本；观察是否单调不降、是否触发 retire 规则、是否出现质量回退。
**判据**：多窗后 4 维不低于 v0，且至少一维持续改善或稳定在高位。
**产物/记录**：版本-质量曲线；log 一条。这对应 IDEA 的 longitudinal 层。

---

## 迭代与停止逻辑

- **每个 Phase 都是一个 gate**：判据不过不要进下一步；失败先归因（protocol/path、runtime/tool、environment/judge、model-capability、hypothesis），再决定回退修哪一层。
- **达成 DoD-A + DoD-B 即"初步可行"达成**，停下来汇报，不要擅自扩 benchmark 或加 longitudinal 复杂度。
- **裁判校准是持续动作**：任何用到 30B 打分的地方，都要定期抽 trace 人工核，发现系统偏差就改成确定性检查或修 rubric。
- **红线**：不放宽 gate 凑结果；不动他人服务；不把单次/小样本失败写成研究结论；改源码先备份。

## 当前基线速查
| | 一致性 | 完备性 | 高效性 |
|---|---|---|---|
| 旧 skill (A) | 90 | 33 | 80 |
| **优化后 v0 (A2)** | 90 | 87 | 77 |
| 手写 policy (B) | 90 | 63 | 83 |
（n=30 固定集，effectiveness=online-only。）
