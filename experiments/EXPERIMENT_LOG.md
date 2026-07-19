# Experiment Log

短记录即可。每条只回答四件事：目标、做成了什么、发现了什么问题、下一步做什么。

## 模板

```text
## YYYY-MM-DD Title

目标：

做成了什么：

发现的问题：

下一步：
```

## 2026-06-12 Workspace Reorganization

目标：
把 learning-machine 科研相关的 code、dataset、model service、notes 和实验记录收进 `/data5/ninghan/tlm`，让后续工作从 research idea 出发。

做成了什么：
整理了目录，创建了 `README.md`、`docs/giai2-notes.md`、`experiments/EXPERIMENT_LOG.md` 和项目 skill。

发现的问题：
长期 working memory 应该放在 `README.md` 和本 log；细节命令放 `docs/`，不要塞进根 README。

下一步：
跑一个最小 Qwen/GIAI2 smoke，确认 structured tool call 和 agent-env 交互链路。

## 2026-06-13 Phase 0 Hermes + Qwen Tool-Call Smoke

目标：
证明本地 Qwen、Hermes agent、Docker GAIA2 env、runner trace 这条链路能完成真实 tool call 和多步交互。

做成了什么：
修了 Qwen/vLLM thinking 控制，给 Hermes 请求加入 `chat_template_kwargs.enable_thinking=false`；`contacts` smoke PASS，真实产生 `terminal` tool call 和 `Contacts.search_contacts` env event。`search` 和 `time` 虽然 FAIL，但已经能进入真实多步工具交互。

发现的问题：
失败不再是 tool-call protocol，而是 harness/agent 策略问题：agent 会拉取过大的 emails/contacts observation，超过 Qwen 8192 context；time 场景还暴露出长程等待/通知/调度能力弱。

下一步：
先做 tool observation budget / reducer，避免 raw env observation 原样塞进 LLM context。

## 2026-06-13 Phase 0 Tool Observation Budget

目标：
验证对长 tool observation 做截断后，search/time 不再因 context overflow 直接失败。

做成了什么：
在 Hermes worker 增加 `MAX_TOOL_OBSERVATION_CHARS`，对过长 `role="tool"` observation 截断并标记 `_gaia2_truncated=true`。重跑后 search events 从 4 到 13，time events 到 9，均不再出现 context-length BadRequest。

发现的问题：
简单截断只能止血，不能解决信息保真。search 变成 agent 把“继续分析”当最终答；time 完成首次邮件发送和检查，但没有完成后续 disappointment email / message fallback。附件 base64、长邮件正文、大列表需要结构化 reducer。

下一步：
做 tool-specific observation reducer：emails/contacts/files 只保留决策字段，删除附件 base64 和长正文；再重跑 search/time，观察 agent 是否能稳定完成多步任务。

## 2026-06-16 Qwen32 GIAI2 Bench Trace Sample

目标：
跑当前本地 `qwen3-32b-awq-tool` + Hermes scaffold 在 GIAI2 bench 上的小样本，记录真实交互流程，定位模型在解题、发现问题/改进方案、执行改进方案三方面的问题。

做成了什么：
运行 `execution/search/time` 各 1 条，输出目录：`/data5/ninghan/tlm/benchmarks/giai2/experiment_runs/qwen32_bench_trace_20260616_081135`。生成简化交互记录：`interaction_records_clean.json` 和 `interaction_records_clean.md`。结果 0/3 pass，但三条都有真实 structured tool calls 和 trace。

发现的问题：
三条都暴露 `tool/interface discovery` 问题：模型会先调用不存在的命令或参数，例如 `cloud-drive search`、`cloud-drive ls --folder`、`contacts get-contacts --limit`，之后能部分查看 help 但恢复效率低。三条都出现 Qwen 8192 context overflow，说明当前 raw observation/历史压缩仍不够。execution/time 还存在 `execution completeness`：执行动作数量和 oracle 不匹配；search 在收集部分证据后最终 answer 为空。

下一步：
先不要扩大 bench。下一轮优先做最小 observation reducer / context compactor，并保留同样的 `interaction_records_clean` 记录格式；重跑同三条场景，比较是否减少 context overflow，以及是否提升最终动作完整性。

## 2026-06-19 Qwen32 40k Context Rerun

目标：
把本地 `qwen3-32b-awq-tool` 服务从 8192 context 开到模型上限 40960，并重跑同一组 `execution/search/time` 各 1 条，判断之前失败是否主要来自 context overflow。

做成了什么：
备份并修改 `/data5/ninghan/tlm/services/qwen/configs/qwen3-32b-awq-tool-8006.env`：`VLLM_MAX_MODEL_LEN=40960`、`VLLM_MAX_NUM_BATCHED_TOKENS=40960`、`VLLM_MAX_NUM_SEQS=2`。服务 `/v1/models` 返回 `max_model_len=40960`，10k-token smoke 通过。新 run 输出目录：`/data5/ninghan/tlm/benchmarks/giai2/experiment_runs/qwen32_bench_40k_20260619_033738`，生成 `interaction_records_clean.json/md`。新增本地提取/对比脚本：`extract_giai2_interaction_records.py`、`compare_giai2_interaction_records.py`。

发现的问题：
40k 后三条都不再出现 LLM context overflow：old `llm_errors=2/2/2` -> new `0/0/0`。但结果仍是 0/3 pass。execution 从 20 events 到 23 events，走到 `rm` 等后半段动作，但最终答成了无关的 Family 文件统计。search 不再空答，给出 `39`，但 judge 仍不匹配。time 从 26 events 降到 10 events，模型误删/移动 `llama2.pdf` 路线，完全偏离邮件和 message fallback 主线。结论：扩 context 解除了硬崩溃，但没有解决 terminal-only scaffold 下的工具语义理解、任务主线保持和动作安全问题。本轮不是模型能力最终结论，因为采样非确定且只跑 3 条小样本。

下一步：
不要只靠继续加 context。下一轮应做两件事之一：1) 固定同三条任务，降低采样随机性/固定生成参数后复跑，判断 40k 的稳定收益；2) 更关键地，做 app-specific action interface 或 observation reducer，把 CLI help/大列表/附件输出转成可执行、可归因反馈，再比较 task completion 和 unsafe action 是否改善。

## 2026-06-24 Terminal-Bench Core Setup and NOP Smoke

目标：
在 TLM 中建立可复现的 Terminal-Bench Core 与 Terminal-Bench 2.0/Harbor 运行入口，并只验证 benchmark plumbing。

做成了什么：
官方 Terminal-Bench source、Core v0.1.1 task dataset、isolated `.venv` 和 isolated Harbor `.harbor-venv` 已配置在 `benchmarks/terminal_bench/`。Core 数据集落在 `datasets/terminal-bench-core-0.1.1/`。用 `hello-world + nop` 完成了一个 54 秒的 Docker smoke，生成 run metadata、trial result、commands、results 与 lockfile。

发现的问题：
用户全局 `~/.cache/uv` 是指向已删除 scratch 路径的断链符号链接；所有 Terminal-Bench 命令必须设置项目内 `UV_CACHE_DIR=$PWD/.uv-cache`。前台 SSH 下载任务可能被会话窗口清理，长下载应使用 tmux。当前 `/data5` 余量约 100 GB，不能直接并行或全量构建镜像。

下一步：
先以固定、非 oracle 的 `terminus` 或同一 Qwen backend agent 跑 3--5 个轻量任务，保存完整 shell trace；随后将 EvoFeedback interface 只接入 agent 可见 command output / exit code / state observations，比较 raw、random、selected feedback 的 retry 与 transfer，不把隐藏 verifier/oracle 注入 agent。

## 2026-06-25 SkillOpt ALFWorld Step EFM Small Runs

目标：
验证新 EFM 代码在 SkillOpt/ALFWorld + Qwen3.5-4B 服务上的小规模在线链路：Step EFM 生成、trajectory audit、policy patch proposal、静态校验与 replay gate 前置流程。

做成了什么：
已有小规模输出位于 `benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_policy_smoke_gpu1_20260624` 和 `benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_gpu1_parallel_20260624`。前者完成 20 条 test episode，生成 20 个 `.efm.json`、`feedback_state.json` 和 `eval_summary.json`；后者完成 24 条 test episode，生成 24 个 `.efm.json` 与 `feedback_state.json`。两者均触发 1 次 policy update 尝试。

初步结果：
20-episode smoke 的 `eval_summary.json` 为 `n_items=20, hard=0.0, soft=0.0`；`feedback_state.json` 记录 `episodes=20, corrections=7, policy_updates=1`。24-episode parallel run 记录 `episodes=24, corrections=18, policy_updates=1`。两次 policy update 均未接受，原因均为 `unsupported_edit`，base policy 保持 version 0，规则数为 0；因此本轮没有进入 replay gate，也没有提交新 policy version。

发现的问题：
在线 Step EFM 与轨迹审计链路可运行并能产出 corrections，但 Qwen 生成的 policy patch 未满足 `research/efm/policy.py` 允许的 edit schema（仅允许 add_rule/add_example/retire_rule/retire_example），被静态校验提前拒绝。当前 `PolicyUpdateDecision` 只保存 reason 和 corrections，不保存原始候选 patch，因此无法仅从产物精确还原非法 edit 内容。ALFWorld 小样本任务成功率仍为 0/20；这不能解释为 EFM policy 无效，因为 policy 实际未升级。

下一步：
先做一个最小诊断 run 或离线复现 policy proposal，记录原始候选 patch；根据候选形态修正 policy proposal prompt/JSON schema 约束，确认至少一次合法 patch 能进入 replay gate。随后再重跑 20-episode 小样本，比较 policy version 是否升级以及 Step EFM correction 数是否下降。


## 2026-06-25 EFM Policy-Update Loop Contract Fix

目标：
诊断 06-24 SkillOpt/ALFWorld run 中 policy patch 被 `unsupported_edit` 拒的根因，把自进化闭环修到能端到端经过 replay gate（先合上 loop，不追求最终性能）。

做成了什么：
离线复现 policy proposal（`research/efm/diagnostics/repro_policy_proposal.py`）定位根因为 prompt 契约问题：Qwen 输出判别键 `edit_type`，而 `validate_patch` 读 `op`，且 `POLICY_SYSTEM` 从未声明 edit 的 JSON schema。最小修复：`prompts.py` 的 `POLICY_SYSTEM` 写明 `op` 判别键 + scope 必须是对象 + support≥min_support + 示例；`policy_user_prompt` 注入 `min_support` 与 `available_episode_ids`；`updater.py` 把每条 correction 的真实 environment_id/task_type 传给 proposer；`policy.py` 的 `validate_patch` 加 scope 非对象守卫。修后离线复现 3/3 `validate_patch=None`。整窗离线复现（`efm_offline_loop.py`）跑通 proposal→validate→apply→gate，结果 `gate_not_improved`（v0 未升级）。gate probe（`efm_gate_probe.py`）确认 candidate 规则 scope=`{environment_id: alfworld}` 已正确 fire，candidate feedback 与 baseline 有差异。源文件备份在 `research/efm/.bak_20260625_053644/`。

发现的问题：
契约 bug 已消除，gate 从"崩溃前拒绝"变成"正常履职拒绝"。但 candidate feedback 保真度不稳：probe 中 `test:0069:3` 的 candidate 臆造"soapbar 被清洗并放置"等当前 observation 不存在的事实，违反 constitution 第 4 条。Qwen-4B 同时当 proposer 与 gate judge，能力偏弱，故 not_improved 合理。结论：自进化闭环机制已合上；拿到真实 v0→v1 接受属反馈质量/模型能力问题，不是协议问题，不应靠放宽 gate 强行达成。

下一步（三选一，待定）：
1) 用更强模型当 gate judge / proposer，复测 gate 是否接受；
2) 收紧 step EFM "只报当前 observation 事实" 的约束，降低臆造，再复测；
3) 接受"loop 通 + gate 正常"，转 Step 2：冻结 v0 policy，做 interaction value 基线对比（EFM step feedback vs raw observation，同子集同采样，先放宽 step budget）。

## 2026-06-25 EFM Feedback Quality: Model vs Skill Ablation

目标：
在 EFM 模型固定为小模型（Qwen3.5-4B）的约束下，判别当前反馈质量差是模型能力问题还是 skill（policy）问题。

做成了什么：
离线评测（`research/efm/diagnostics/efm_quality_eval.py`，不跑新 rollout），从 parallel run 抽 30 条固定 transition，Qwen3-30B(8001) 当盲裁判，三格对比：A=4B+当前skill（复用saved baseline）、B=4B+手写policy规则（专治over-hedge与signal误判）、C=30B+当前skill。结果 faithful/signal_ok/vacuous/hedge = A:60/40/23/67，B:83/73/7/33，C:63/43/20/63。

发现的问题：
C≈A（换30B、skill不变几乎无改善）→ 不是模型能力问题；B≫A（同4B、仅换policy大幅提升）→ 是 skill 问题。结论：反馈质量头部空间在可学的 policy/skill 层，小 EFM 模型够用，验证"EFM小模型+强optimizer优化skill"的框架。边界：n=30、单一30B裁判、B规则系手写（强optimizer的人工替身）、仅ALFWorld；证明"skill有头部空间"，未证明"自进化能自动学到B"。

下一步：
以手写 B 为 oracle/靶子，测强 optimizer 能否把 policy 自动学到接近 B 的质量并过 gate（接 06-25 已修通的自进化闭环）。评测指标沿用 faithful/signal_ok/vacuous/hedge。

## 2026-06-25 EFM Eval v2 (4-dim) + Initial Skill Optimization

目标：
重做反馈质量评测（不再单靠 30B 打分），并优化初始 EFM skill 得到一个合理反馈机制。

做成了什么：
评测改为 4 维（`research/efm/diagnostics/efm_eval_v2.py`）：一致性 Consistency（确定性 grounding：实体/否定/transform 动词，对 ALFWorld 模板 obs 可复现）、完备性 Completeness（obs 有明确事实却输出 ambiguity = 漏报）、高效性 Efficiency（长度+meta 短语），有效性 Effectiveness 标为 online-only 不离线测。用手判过的 30 条验证：v2 修掉了 30B 在"列举+缺失"句式上的系统误判（#14A/#15A/#25C/#27B 等），残留少量关系型幻觉（如 #6C）需 LLM 兜底。v2 下三 cell 一致性都 ~83-90%，真正差距在完备性（A=33/B=63/C=40）。随后优化初始 skill：仅改 constitution（加 anti-hedge 规则5、signal_type 定义、禁止复述历史），重测 30 条：A_old 一致/完备/高效=90/33/80 → A2_optskill=90/87/77，超过手写 policy B 的 90/63/83。constitution 备份在 `research/efm/constitution.py.bak_preopt`。

发现的问题：
反馈质量的主轴是完备性（over-hedge），不是一致性；而完备性可由静态 constitution 大幅修复（33→87），不需大模型、不需 policy。含义：env-agnostic 的通用质量靠固定 skill 就能解决；留给可学 policy/自进化 的，是 env/task 特异、无法预先写死的涌现模式 + 有效性（只能在线测）。一致性的确定性 checker 有已知边界：无数字实体（如 "lettuce"）和关系型"X 现在在 Y"幻觉测不到。

下一步：
跑 EFM skill 自进化：1) 把 gate 从噪声 LLM-A/B 改成 eval_v2 确定性指标做接受判据；2) 以优化后的 v0 为起点跑一窗在线自进化（optimizer=30B），看 policy 能否升级且 4 维不回退；3) 在线反事实测有效性（raw vs v0-skill vs evolved-skill 对 agent 的 interaction success/recovery/步数）。

## 2026-06-25 EFM Deterministic Gate + Phase 2b Online Smoke

目标：
依据 `research/efm/SELFEVOLVE_PLAN.md` 推进后续 EFM 自进化实验：先把 policy gate 从噪声 LLM-A/B 替换为 eval_v2 口径的确定性质量判据，再启动一窗在线自进化 smoke，目标是验证 v0→v1 policy update 是否能在可信 gate 下发生。

做成了什么：
- 新增正式质量模块 `research/efm/quality.py`，复用一致性、完备性、高效性三维判据。
- `FeedbackRuntimeConfig` 增加 `policy_gate_mode = deterministic | llm`，默认 deterministic；`PolicyUpdater._gate` 现在按模式走 deterministic gate 或保留旧 LLM gate。
- deterministic gate 判据：冻结 validation transitions 上三维 aggregate 不回退且至少一维净增则 accept；candidate 出现 consistency=0 的未 grounded 幻觉则拒绝为 `deterministic_quality_regressed`。
- SkillOpt ALFWorld/EFM 配置链路透传 `feedback_policy_gate_mode`；`eval_only.py` 补齐 qwen_chat 分角色配置，避免 target/optimizer 端点依赖隐式环境变量。
- 单测通过：`tests/test_efm.py` 6 passed；`tests/test_qwen_backend.py tests/test_efm.py` 10 passed；`research/efm/diagnostics/test_deterministic_gate.py` passed。
- Phase 0 复核：固定 A/B/C eval_v2 仍为 A=90/33/80、B=90/63/83、C=83/40/100；A2 online regen 两次为 97/73/73、93/70/83，说明 regenerated A2 受 4B 采样影响，但仍显著优于旧 skill 完备性 33，并略高于手写 B 完备性 63。
- 已启动 Phase 2b run：tmux `efm_detgate_phase2b_20260625`；输出目录 `benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2b_20260625/`；命令记录在 `run_command.sh`，stdout 在 `run.log`。设置为 20 个 test episodes、`max_steps=10`、target=Qwen3.5-4B@8007、optimizer=Qwen3-30B@8001、deterministic gate、target max tokens=1024、optimizer max tokens=2048。

发现的问题：
- 用户给出的本地路径 `/Users/fanninghan/agent/tlm/research/efm/SELFEVOLVE_PLAN.md` 不存在；实际计划文件在远端 `/data5/ninghan/tlm/research/efm/SELFEVOLVE_PLAN.md`。
- 训练入口 `scripts/train.py` 会引入 selection baseline / slow-update 等额外流程，不适合只验证 EFM policy window；Phase 2 smoke 改用 `scripts/eval_only.py` 复用 prior 20-episode smoke 路径。
- 初始 probe 若不设置 `ALFWORLD_DATA=/data5/ninghan/tlm/benchmarks/skillopt/data/alfworld_data` 会落到 `~/.cache/alfworld` 并找不到 game 文件。
- Phase 2b 当前仍在第 1 个 ALFWorld rollout，尚未产出 `results.jsonl` 或 `feedback_state.json`；进程仍运行、8007 小请求 0.16s 返回，暂不判定为服务不可用。

下一步：
1. 等 `efm_detgate_phase2b_20260625` 完成或至少产出首条结果后，读取 `feedback_state.json` 的 `policy.version`、`policy_updates[].reason`、`corrections`。
2. 若 run 卡在首条超过合理时间，做单步 trace：确认 target 请求 payload/返回、ALFWorld worker pipe、以及是否 agent 生成了不可执行/超长动作。
3. 若出现 `accepted`，用 eval_v2 对 v0 vs learned policy 做 3 维不回退检查；若是 `gate_not_improved`，记录为 v0 静态 skill 已较强或 candidate 无净增，不放宽 gate 凑结果。

## 2026-06-26 EFM Phase 2b Hourly Check: Invalid Run Due to Optimizer Model ID

目标：
巡检 `efm_detgate_phase2b_20260625`，判断 deterministic gate self-evolution smoke 是否完成，以及是否达到 v0→v1 accepted 的 Phase 2 目标。

做成了什么：
- Phase 2b run 已完成，输出目录：`benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2b_20260625/`。
- `eval_summary.json`: `n_items=20, hard=0.2, soft=0.2`；`results.jsonl` 中 4/20 hard success。
- `feedback_state.json`: `episodes=20, train=15, validation=5, success=4, failure=16, corrections=0, policy_updates=1`。
- policy 未升级：`policy.version=0`，`policy_updates[0].reason=no_corrections`。
- 抽查失败 train traces 发现 Step EFM 全部 fallback：184/184 step feedback 为 `EFM unavailable (RuntimeError)`。
- endpoint probe 定位根因：8001 不接受 `Qwen/Qwen3-30B`，返回 404；8001 接受 `/data2/qinhao/LLM_weight/Qwen/Qwen3-30B-A3B-Instruct-2507/`，返回 200。

发现的问题：
Phase 2b 不是有效的 self-evolution 证据。EFM 在线调用 optimizer 时 model id 配错，导致所有 Step EFM 输出 fallback ambiguity；trajectory review 没有可学习 correction，policy update 自然 `no_corrections`。这属于配置/协议问题，不是 deterministic gate、policy proposal 或 EFM 自进化机制失败。

修复动作：
启动修正版 Phase 2c：tmux `efm_detgate_phase2c_20260625`，输出目录 `benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2c_20260625/`。核心修正：`model.optimizer=/data2/qinhao/LLM_weight/Qwen/Qwen3-30B-A3B-Instruct-2507/`；其它条件保持同 Phase 2b（20 test episodes、max_steps=10、deterministic gate、target=Qwen3.5-4B@8007）。自动化巡检目标已更新到 Phase 2c。

下一步：
下次巡检 Phase 2c 时首先统计 StepFeedback fallback rate；若仍有大量 fallback，优先读具体 RuntimeError/请求 payload；若 fallback 消失，再看 corrections、policy update reason，以及是否出现 v0→v1 accepted。

## 2026-06-25 Phase 2b Invalid (404) + Stage-Routing Fix

目标：
推进 Phase 2 在线自进化，核查 detgate_phase2b run 的 no_corrections 结果。

做成了什么：
诊断 phase2b（`efm_probe_phase2b.py`）：在线 v0 反馈 184/184 步全是 ambiguity、完备性仅 32%，离线重跑 30B 审计直接 HTTP 404。根因 = optimizer model id 写成 `Qwen/Qwen3-30B`，而 8001 实际 served id 是 `/data2/qinhao/LLM_weight/Qwen/Qwen3-30B-A3B-Instruct-2507/`，导致 EFM 每步/每次审计调用全 404 → fallback ambiguity + 审计吞异常 → corrections=0 → no_corrections。**phase2b 作为自进化测试无效。** 另发现设计问题：efm.yaml `feedback_model_role=optimizer` 会让 Step EFM 也走 30B，违反"小 EFM"约束。修复：集成 `SkillOptFeedbackModel` 增加 stage 路由（新 role `split`，默认）：`efm_step`→target(4B)，审计/提案/llm-gate→optimizer(30B)，落实"小 EFM + 强 optimizer"。备份 `research/efm/.bak_20260625_053644/efm_integration.py.bak`。2-ep split smoke 验证：0 fallback，signal 分布 state_change5/progress3/constraint_violated1/ambiguity1，反馈准确——404 修好、split 生效、在线即达优化后 constitution 质量。

发现的问题：
所有用到本地共享端点的 run 必须用 `/v1/models` 实际 id，不能猜模型名。并发存在另一 agent 的 phase2c run（20-ep，role=optimizer 即 step 也用 30B，非 faithful），占用 8001。

下一步：
跑 canonical split 20-ep（step=4B/optimize=30B）拿 DoD-A 结果；避免与 phase2c 抢 8001。


## 2026-06-26 EFM Phase 2c Hourly Check: Valid Run, Gate Rejected Correction Quality

- Question: After fixing the optimizer model id from `Qwen/Qwen3-30B` to `/data2/qinhao/LLM_weight/Qwen/Qwen3-30B-A3B-Instruct-2507/`, can Phase 2 produce a v0->v1 accepted EFM policy update without quality regression?
- Run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2c_20260625/`; command: `run_command.sh`; log: `run.log`; artifacts: `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Status: tmux session `efm_detgate_phase2c_20260625` has exited and the run completed 20 ALFWorld test items.
- Metrics: `eval_summary.json` reports `hard=0.2`, `soft=0.2`, `n_items=20`; `results.jsonl` contains 20 rows with 4 hard successes.
- EFM runtime boundary: the optimizer endpoint/config issue from Phase 2b is fixed. StepFeedback fallback rate is `0/178 = 0.0`, with signal types `state_change=149`, `ambiguity=25`, `progress=4`.
- Policy result: `feedback_state.json` has `policy.version=0`; one policy update was proposed but not accepted: `policy_updates[0].reason=deterministic_quality_regressed`; corrections count is 14.
- Failure classification: hypothesis/protocol-quality boundary, not endpoint failure, ALFWorld worker stall, data path failure, or deterministic gate looseness. The deterministic gate correctly rejected the candidate policy because correction quality regressed.
- Evidence: 6/14 trajectory-audit corrections contain agent-advice or speculative wording such as `You can examine it`, `Consider searching other locations`, `You can turn it on`, `Further inspection may be needed`, and `The pencil might be on...`. These violate the EFM feedback contract because corrections should describe grounded feedback quality, not tell the agent what to do or hallucinate unseen object locations.
- Decision: Do not relax the deterministic gate to force acceptance. Next smallest useful action is to harden the trajectory audit/correction contract and/or add a pre-policy correction filter that rejects advice-like or ungrounded corrections, then rerun a small Phase 2d validation before a larger run.

## 2026-06-26 EFM Phase 2c Hourly Recheck 02:56 CST

- Run: /data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2c_20260625/; tmux session efm_detgate_phase2c_20260625 alive: False; run.log size 8796, mtime 2026-06-25 17:35:06 +0000.
- Artifacts: {'run.log': True, 'run_command.sh': True, 'results.jsonl': True, 'feedback_state.json': True, 'eval_summary.json': True}.
- Metrics: eval_summary.json = {'skill': '/dev/null', 'split': 'test', 'n_items': 20, 'hard': 0.2, 'soft': 0.2}; results.jsonl rows 20, hard successes 4.
- EFM state: policy.version=0, policy updates 1, last update accepted False, reason deterministic_quality_regressed, corrections 14.
- StepFeedback fallback: 0/178 = 0.0.
- Failure classification: repeated confirmation of Phase 2c valid-runtime but rejected-policy outcome. Endpoint/config failure is resolved; the remaining blocker is trajectory-audit/correction quality, with advice-like or speculative corrections 11/14.
- Decision: keep deterministic gate unchanged. Next action remains to harden correction generation/validation and rerun as Phase 2d; no safe in-place continuation exists for the completed Phase 2c artifacts.


## 2026-06-26 EFM Phase 2d Launch: Correction Validation Filter

- Question: Can filtering trajectory-audit corrections that contain agent advice or speculative ungrounded claims remove the Phase 2c policy-quality blocker without weakening the deterministic gate?
- Root-cause evidence from Phase 2c: runtime/config was valid (`fallback=0/178`), but policy stayed at v0 with `deterministic_quality_regressed`; the correction stream included advice-like/speculative `better_feedback` text such as `You can...`, `consider...`, `further inspection...`, and `might be...`.
- Code change: added a focused trajectory-review validation test in `benchmarks/skillopt/tests/test_efm.py`; the test first failed because `_review()` admitted an advice/speculative correction. Added `_valid_correction()` filtering in `research/efm/updater.py` before constructing `TrajectoryCorrection`, and tightened `research/efm/prompts.py` to require declarative grounded feedback rather than agent next-action advice.
- Verification: red test failed as expected, then passed after the fix; focused regression `tests/test_qwen_backend.py tests/test_efm.py -q` passed `11 passed` using `/data5/ninghan/tlm/envs/skillopt-qwen35-vllm-cu128/bin/python` with `PYTHONPATH=/data5/ninghan/tlm`.
- Phase 2d run: started tmux session `efm_detgate_phase2d_20260626`; output directory `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2d_20260626/`; command `run_command.sh`; log `run.log`.
- Launch check: tmux is alive and `run.log` has entered ALFWorld rollout chunk 1/20. No immediate model endpoint, config path, data path, or ALFWorld worker failure observed.
- Success criterion: after completion, require `policy.version` to advance v0->v1 with accepted update and no deterministic quality regression; also check StepFeedback fallback rate, corrections count/filter effect, and `hard/soft/n_items` for non-regression.


## 2026-06-26 EFM Phase 2d Result and Phase 2e Diagnostic Launch

- Phase 2d run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2d_20260626/`; tmux exited normally; artifacts present: `run.log`, `run_command.sh`, `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Metrics: `eval_summary.json` reports `hard=0.2`, `soft=0.2`, `n_items=20`; `results.jsonl` rows `20`, hard successes `4`.
- Runtime boundary: StepFeedback fallback `0/186 = 0.0`; endpoint/config/data path/ALFWorld worker were not the failure boundary.
- Correction filter result: policy update retained `14` corrections and advice/speculative phrase residual was `0/14`; deterministic correction audit scored `12/14` as consistency/completeness/efficiency all passing. The Phase 2c advice-contamination issue is fixed.
- Policy result: `policy.version=0`; one update was rejected with `reason=gate_not_improved`, not `deterministic_quality_regressed`; policy rules/examples remain empty. This means the candidate policy was valid enough to reach the deterministic gate but did not improve scored validation feedback.
- Remaining diagnostic gap: current `PolicyUpdateDecision` did not persist the raw candidate patch or deterministic gate totals, so Phase 2d cannot distinguish weak proposal, rule non-application, or candidate render equivalence.
- Code change for next run: added a red test `test_rejected_policy_update_persists_candidate_and_gate_diagnostics`, then implemented optional `candidate_patch` and `gate_diagnostics` fields in `research/efm/models.py` and deterministic gate totals in `research/efm/updater.py`. Focused regression passed: `tests/test_qwen_backend.py tests/test_efm.py -q` -> `12 passed`.
- Phase 2e diagnostic run: started tmux session `efm_detgate_phase2e_diag_20260626`; output directory `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2e_diag_20260626/`; launch check shows tmux alive and `run.log` entered ALFWorld rollout chunk 1/20.
- Next criterion: if Phase 2e still rejects with `gate_not_improved`, inspect saved `candidate_patch` and `gate_diagnostics` to decide whether the research mechanism needs better policy proposal support, better rule application, or a different validation-transition construction. Do not relax deterministic gate.


## 2026-06-26 EFM Phase 2e Result and Phase 2f Target-Aware Gate Launch

- Phase 2e run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2e_diag_20260626/`; tmux exited normally; artifacts present: `run.log`, `run_command.sh`, `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Metrics: `eval_summary.json` reports `hard=0.15`, `soft=0.15`, `n_items=20`; `results.jsonl` rows `20`, hard successes `3`.
- Runtime/correction boundary: StepFeedback fallback `0/185 = 0.0`; corrections `11`; advice/speculative residual `0/11`. Endpoint, config path, data path, ALFWorld worker, and correction filter were not the failure boundary.
- Policy result: `policy.version=0`; one policy update rejected with `reason=gate_not_improved`; no rules/examples accepted.
- New diagnostics: candidate patch was valid and focused on `look_at_obj_in_light`, adding rules to explicitly state the absence/non-visibility of task-relevant target objects. Deterministic gate diagnostics showed `baseline_totals={consistency:8, completeness:8, efficiency:8}` and `candidate_totals={consistency:8, completeness:8, efficiency:8}` across 8 validation transitions.
- Failure classification: deterministic gate scorer saturation / missing task-target completeness dimension. The validation transitions included cases like task `examine the bowl with the desklamp` where the observation listed visible desk objects but omitted the bowl; baseline feedback also omitted that the bowl was not visible, yet the old scorer still gave completeness credit.
- Code change: added red test `test_deterministic_gate_rewards_task_target_absence_feedback`; implemented optional `task_description` in `score_feedback`, target extraction for ALFWorld task descriptions, and target-absence completeness checking when a concrete observation lists objects but omits the task target. Updated deterministic gate to pass episode `task_description` to both baseline and candidate scoring. Focused regression passed: `tests/test_qwen_backend.py tests/test_efm.py -q` -> `13 passed`.
- Phase 2f run: started tmux session `efm_detgate_phase2f_targetgate_20260626`; output directory `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2f_targetgate_20260626/`; launch check shows tmux alive and `run.log` entered ALFWorld rollout chunk 1/20.
- Next criterion: Phase 2f should show whether target-aware deterministic gate can accept a valid v0->v1 policy without quality regression. If rejected, inspect `candidate_patch` and `gate_diagnostics`; do not relax the gate.


## 2026-06-26 EFM Phase 2f Result and Phase 2g Invalid-Patch Diagnostic Launch

- Phase 2f run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2f_targetgate_20260626/`; tmux exited normally; artifacts present: `run.log`, `run_command.sh`, `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Metrics: `eval_summary.json` reports `hard=0.15`, `soft=0.15`, `n_items=20`; `results.jsonl` rows `20`, hard successes `3`.
- Runtime/correction boundary: StepFeedback fallback `0/185 = 0.0`; corrections `10`; advice/speculative residual `0/10`. Endpoint, config path, data path, ALFWorld worker, correction filter, and Step EFM runtime were not the failure boundary.
- Policy result: `policy.version=0`; one policy update rejected with `reason=rule_lacks_content_or_support`; no rules/examples accepted. `candidate_patch` and `gate_diagnostics` were absent because invalid patches were not persisted before validation return.
- Failure classification: policy proposal/schema support boundary plus diagnostic persistence gap. The target-aware gate did not run because proposal validation failed before gate evaluation.
- Evidence: retained corrections were distributed across 10 distinct episodes with mixed `state_change`, `ambiguity`, and `observation` event types; the proposal likely failed the policy min-support/content validation, but Phase 2f cannot reconstruct the raw patch.
- Code change: added red test `test_invalid_policy_update_persists_candidate_patch_diagnostics`; implemented invalid-patch persistence in `research/efm/updater.py`, storing `candidate_patch` and `gate_diagnostics={mode:not_run, invalid_reason:<reason>}` when `validate_patch` rejects a proposal. Focused regression passed: `tests/test_qwen_backend.py tests/test_efm.py -q` -> `14 passed`.
- Phase 2g run: started tmux session `efm_detgate_phase2g_invalidpatchdiag_20260626`; output directory `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2g_invalidpatchdiag_20260626/`; launch check shows tmux alive and `run.log` entered ALFWorld rollout chunk 1/20.
- Next criterion: if Phase 2g rejects with `rule_lacks_content_or_support`, inspect saved `candidate_patch` to distinguish weak support ids, empty instruction/avoid, wrong schema, or proposal over-fragmentation. Do not relax deterministic gate or policy support requirements without evidence.


## 2026-06-26 EFM Phase 2g Result and Phase 2h Filtered-Edits Launch

- Phase 2g run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2g_invalidpatchdiag_20260626/`; tmux exited normally; artifacts present: `run.log`, `run_command.sh`, `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Metrics: `eval_summary.json` reports `hard=0.15`, `soft=0.15`, `n_items=20`; `results.jsonl` rows `20`, hard successes `3`.
- Runtime/correction boundary: StepFeedback fallback `0/193 = 0.0`; corrections `10`; advice/speculative residual `0/10`. Endpoint/config/data path/ALFWorld worker and correction filter are healthy.
- Policy result: `policy.version=0`; one update rejected with `reason=rule_lacks_content_or_support`; no rules/examples accepted; `gate_diagnostics={mode:not_run, invalid_reason:rule_lacks_content_or_support}` confirms target-aware gate did not run.
- Candidate patch diagnosis: proposal contained two `add_rule` edits. Edit 0 was valid with 5 distinct support episodes: `test:0006`, `test:0007`, `test:0009`, `test:0010`, `test:0015`. Edit 1 had duplicated support IDs `test:0002` repeated and only 2 distinct support episodes, below `policy_min_support=3`. The whole patch was rejected because one edit was under-supported.
- Failure classification: policy proposal over-fragmentation / edit-level validation boundary. A valid rule was available, but the updater rejected the entire mixed patch instead of dropping the invalid edit.
- Code change: added red test `test_policy_update_filters_invalid_edits_and_gates_valid_subset`; implemented edit-level validation in `research/efm/updater.py`, dropping invalid edits while retaining and gating a valid subset. This does not relax `policy_min_support` or deterministic gate. Focused regression passed: `tests/test_qwen_backend.py tests/test_efm.py -q` -> `15 passed`.
- Phase 2h run: started tmux session `efm_detgate_phase2h_filterededits_20260626`; output directory `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2h_filterededits_20260626/`; launch check shows tmux alive and `run.log` entered ALFWorld rollout chunk 1/20.
- Next criterion: Phase 2h should either accept a valid subset patch and advance v0->v1, or reach deterministic gate with saved `candidate_patch`, `dropped_edits`, and `gate_diagnostics` for the next diagnosis. Do not relax support or gate criteria.


## 2026-06-26 EFM Phase 2h Result: First Accepted v0-to-v1 Policy

- Phase 2h run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_detgate_phase2h_filterededits_20260626/`; tmux exited normally; artifacts present: `run.log`, `run_command.sh`, `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Metrics: `eval_summary.json` reports `hard=0.15`, `soft=0.15`, `n_items=20`; `results.jsonl` rows `20`, hard successes `3`. This is not an ALFWorld score improvement relative to Phase 2c/2d, but it is enough to test whether the EFM policy update mechanism can produce an accepted, quality-gated v1.
- Runtime/correction boundary: StepFeedback fallback `0/188 = 0.0`; corrections `8`; advice/speculative residual `0/8`. Endpoint/config/data path/ALFWorld worker and correction filter are healthy.
- Policy result: `policy.version=1`; one update accepted with `reason=accepted`; `candidate_version=1`; policy now has two active rules scoped to `environment_id=alfworld`, `task_type=look_at_obj_in_light`.
- Accepted rule 1: visibility claims must be based on explicit current observational evidence, including confirmed presence, absence, or occlusion; avoid assuming visibility/spatial relations without direct observation. Support episodes: `test:0000`, `test:0002`, `test:0007`, `test:0009`, `test:0010`, `test:0015`, `test:0017`.
- Accepted rule 2: mention current desklamp lighting state when it changed and may affect visibility. Support episodes: `test:0010`, `test:0015`, `test:0017`.
- Gate evidence: deterministic gate ran on 8 validation transitions. Baseline totals were `{consistency:8, completeness:1, efficiency:8}`; candidate totals were `{consistency:8, completeness:6, efficiency:8}`. The accepted policy improved task-target completeness without regressing consistency or efficiency.
- Failure-boundary resolution: the chain of blockers was resolved in order: optimizer model id -> correction advice/speculative contamination -> missing gate diagnostics -> target-aware completeness saturation -> invalid patch diagnostics -> mixed-patch edit filtering. Phase 2h is the first run satisfying the v0->v1 accepted and no deterministic quality regression criterion.
- Evidence boundary: this establishes a viable EFM self-evolution mechanism on the small ALFWorld 20-item setting, not a benchmark performance gain. The aggregate hard/soft score remains `0.15`, so the next research step is held-out or second-window validation of the v1 policy: check whether it keeps fallback low, avoids correction contamination, and does not reduce task success or feedback quality.
- Next action: run a Phase 2i validation/replay using the accepted v1 state or continue one more policy window from Phase 2h to test whether v1 is stable and whether accepted feedback-quality gains transfer beyond the validation transitions.


## 2026-06-26 EFM Phase 2i Launch: Seeded v1 Stability Validation

- Question: Does the accepted Phase 2h v1 EFM policy remain stable in a fresh validation window, keeping feedback-quality gains without runtime fallback or correction contamination?
- Prior evidence: Phase 2h accepted v0->v1 with deterministic gate improvement from baseline `{consistency:8, completeness:1, efficiency:8}` to candidate `{consistency:8, completeness:6, efficiency:8}`, `fallback=0/188`, corrections `8`, advice residual `0/8`, but task hard/soft remained `0.15/0.15`.
- Setup: created `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_phase2i_v1_validation_20260626/` and seeded `feedback_state.json` from Phase 2h policy only: `policy.version=1`, active rules copied, `episodes=[]`, `corrections=[]`, `policy_updates=[]`, `policy_cursor=0`. This avoids mixing Phase 2h history while validating v1 behavior.
- Command: copied Phase 2h `run_command.sh` and changed only `env.out_root` to the Phase 2i directory. Endpoint/model/worker/gate settings are otherwise unchanged.
- Run: started tmux session `efm_phase2i_v1_validation_20260626`; launch check shows tmux alive and `run.log` entered ALFWorld rollout chunk 1/20. Seed check confirmed `policy_version=1`, `episodes=0`, `updates=0` at launch.
- Success criterion: after completion, require low StepFeedback fallback, advice/speculative residual `0`, deterministic quality no regression, and hard/soft not below Phase 2h baseline `0.15/0.15`. If a v1->v2 update is attempted, inspect `candidate_patch`, `dropped_edits`, and `gate_diagnostics` separately from the v1 rollout metrics.


## 2026-06-26 EFM Phase 2i Result: v1 Stability Validation Passed

- Phase 2i run: `/data5/ninghan/tlm/benchmarks/skillopt/outputs/qwen35_4b_alfworld_efm_phase2i_v1_validation_20260626/`; tmux exited normally; artifacts present: `run.log`, `run_command.sh`, `results.jsonl`, `feedback_state.json`, `eval_summary.json`, `feedback/*.efm.json`.
- Setup check: `feedback_state.json` was seeded from Phase 2h accepted v1 policy only: `policy.version=1`, two active rules, old `episodes/corrections/policy_updates` cleared at launch. All 20 Phase 2i episodes ran with `policy_version=1`.
- Metrics: `eval_summary.json` reports `hard=0.25`, `soft=0.25`, `n_items=20`; `results.jsonl` rows `20`, hard successes `5`. This is above the Phase 2h reference `0.15/0.15`.
- Runtime/correction boundary: StepFeedback fallback `0/181 = 0.0`; corrections `13`; advice/speculative residual `0/13`. Endpoint/config/data path/ALFWorld worker/state seeding were healthy.
- v1 stability result: passed. The accepted Phase 2h v1 policy kept fallback low, avoided correction contamination, and did not reduce task score relative to Phase 2h; observed hard/soft improved from `0.15/0.15` to `0.25/0.25` in this 20-item validation window.
- v1->v2 proposal: one update was attempted from base version 1 but rejected with `reason=deterministic_quality_regressed`; `policy.version` remains `1`. Candidate patch tried to add another visibility rule for unobserved locations, but deterministic gate showed baseline totals `{consistency:8, completeness:6, efficiency:8}` and candidate totals `{consistency:8, completeness:5, efficiency:8}`. The gate correctly blocked a completeness regression.
- Evidence boundary: Phase 2 now establishes a complete and protected EFM self-evolution loop on the small ALFWorld setting: v0 generated feedback traces, trajectory audit produced corrections, policy proposal generated rules, deterministic gate accepted v1, and an independent v1-seeded validation window preserved quality and improved observed hard/soft. It is still a small-sample result, not a final benchmark claim.
- Next research decision: promote this into the first feasible research scheme: (1) report v0->v1 accepted as mechanism proof; (2) report Phase 2i as small-window stability evidence; (3) run a larger held-out A/B next if resources allow, comparing v0 vs accepted v1 on more ALFWorld items and possibly a second task type. Hourly Phase 2 rescue monitoring is no longer necessary unless a new larger run is launched.

## 2026-06-26 Idea-Validation P1 Launched (ALFWorld raw vs efm, 32B target)

目标：
验证"小 EFM 提炼环境反馈能否提升 agent 任务表现"——P1 交互价值，ALFWorld，raw vs efm 单一固定 skill。

做成了什么：
- 确认 ALFWorld 原生 budget=50（旧 10 步是地板元凶）。
- 起 32B-AWQ 服务 @ 8006/GPU2（用户授权 2/3，GPU2 当时空闲）：`services/qwen/configs/qwen3-32b-awq-tool-8006-gpu2.env`，tmux `efm-32b`。
- 配置：target=32B@8006、EFM=4B@8007（经 optimizer 后端复用，`feedback_model_role=optimizer`、`feedback_policy_update_enabled=false` 即固定 skill 不进化）。修正 model id 教训沿用。
- 新增 bench 无关评测/监控模块 `research/efm/bench/eval.py`（读 results.jsonl+feedback_state.json 出 hard/soft+质量4维+CI），满足"不一 bench 一份代码"。
- 2-ep 32B 冒烟通过：budget50 生效、0 fallback、signal 多样、32B 在 look_at_obj_in_light 50 步仍失败 → 非天花板、有头部空间。
- 正式 P1 启动：tmux `idea-exp`，顺序跑 raw(100)→efm(100)，输出 `outputs/idea_alfworld_32b_{raw,efm}_run1`，wrapper 日志 `benchmarks/skillopt/logs/idea_exp_run1.log`。

发现的问题：
results.jsonl 仅在每臂跑完才写；运行中实时分数看 tmux 日志的 cumulative=X/Y。网络抖动频繁，长任务必须 tmux（已用）。

下一步：
等两臂跑完，用 eval.py 比 raw vs efm 的 hard/soft（±CI）；若 efm 显著 > raw → P1 成立，扩 seed/上 Terminal-Bench；若不成立，先查头部空间 vs 反馈消费。

## 2026-06-26 Idea Validation P1: ALFWorld Interaction Value (raw vs EFM)

目标：
验证 idea 核心命题——一个小 EFM 对环境反馈做提炼，能否提升 agent 在 bench 上的任务表现（交互价值）。target=Qwen3-32B-AWQ@8006，EFM=Qwen3.5-4B@8007，ALFWorld 原生 budget=50，单一固定 skill（优化后 constitution + 空 policy），关闭自进化。EFM≤target 且各自独立端点，避免能力走私。

做成了什么：
新增 bench 无关评测/聚合层 `research/efm/bench/eval.py`（读 results.jsonl + feedback_state.json，出 hard/soft + 质量4维 + bootstrap CI，兼监控）。参数化 runner `benchmarks/skillopt/run_idea_validation.sh <raw|efm> <N> <suf>`（DRY，后续 bench 复用）。各跑 n=100（同一批 test 任务，配对）。结果：raw hard=0.260，efm hard=0.380。配对 Δhard=+0.120，95%CI=[+0.030,+0.220]，bootstrap P(Δ≤0)=0.006；翻转 efm独赢18 / raw独赢6 / 持平76。efm 在线反馈质量 cons=96 comp=68 eff=79 fallback=0% avg_steps=39.7。

发现的问题：
交互价值假设成立且统计显著（p<0.01）：4B 小 EFM 使 32B agent ALFWorld 成功率 0.26→0.38（+46% 相对）。32B 裸跑 0.26 处于反馈甜区（非地板/天花板），头部空间充足。注意：早期首-34 读数 0.62 是高光区间虚高，全量回落到稳定 +0.12，说明中途单点不可信、必须全量配对。在线 completeness 仅 68%（离线 87%），反馈本身仍有提升空间。非配对 CI 重叠、配对 CI 不重叠——必须用配对检验。运维：32B-AWQ 起在 GPU2（用户授权 2/3），8006。

下一步：
1) 复跑第二批/seed 验证 +0.12 的稳健性与跨任务泛化；2) 之后用共享 bench 层接 Terminal-Bench 验证跨环境泛化；3) completeness 68→更高 仍是反馈侧可优化点。

## 2026-06-27 Stage 1+2: Constitution Fix + Self-Evolution Improvements

目标：
1. (Stage 1) 修复在线 completeness 68% 差距：诊断发现 91% 的 completeness 失败来自 EFM 在 concrete obs 上误用 signal_type="ambiguity"（"you see nothing"、"you see [items]"、到达新位置等）。修改 `research/efm/constitution.py` rule 5 和 rule 8，明确列出不属于 ambiguity 的具体情况，要求在 task target 可见/缺席时显式说明。
2. (Stage 2) 改进自进化机制：(a) `prompts.py` TRAJECTORY_SYSTEM 增加 top-3 常见失败模式指引；POLICY_SYSTEM 引导 proposer 优先写 environment-scoped 规则（跨 task type）而非窄 task-type-scoped 规则，并鼓励 add_example。(b) `updater.py _select_analysis` 改为按 task_type 轮询采样，确保 analysis window 覆盖多种任务类型，让 proposer 有足够跨类型证据支持 env-scoped rule。

做成了什么：
- 诊断：P1 EFM run1 总 3967 步，completeness 64.8%；type-A (ambiguity on concrete obs) 失败 1238 步 (31.2%)，type-B (未报告 target 存在/缺席) 123 步 (3.1%)；实体/变换一致性 issues 仅 28 步。
- constitution.py rule 5 新增 4 条具体说明（"you see nothing"→state_change、item list→state_change+target mention、新位置→state_change、confirmed action→progress/state_change）；rule 8 增加 target 显式报告要求。
- prompts.py 两处改进：TRAJECTORY_SYSTEM 增加 3 条 failure mode 列表；POLICY_SYSTEM 增加 scope 策略指引和 add_example 使用指引。
- updater.py `_select_analysis` 改为 round-robin across task_types，优先覆盖多类型。
- 测试：15 passed（benchmarks/skillopt/tests/）。
- 理论 completeness 上限：修复两类失败后可达 99.1%（仅剩 entity/transform 一致性 28 步）。
- 验证 run：`efm_constfix_val_20260627`，n=20，output `idea_alfworld_32b_efm_constfix_val_20260627`，tmux 已启动。

发现的问题：
- 在线 completeness 低的根本原因是 constitution 对 "negative evidence"（location visited but empty、target not in visible list）没有明确说明，导致 EFM 系统性地选择 ambiguity。
- type-A 失败均可由 constitution 修复，不需要 policy rule 或 fine-tuning。
- type-B 失败（123 步）由 rule 8 新增的 target mention 要求覆盖。

下一步：
1. 等 constfix_val_20260627 (n=20) 完成，用 bench/eval.py 对比 completeness 是否提升到 ~90%+。
2. 若验证通过，启动 constfix P1 full run (n=100 efm，与 run1 raw arm 配对) 确认 task score 提升。
3. 跑一轮 Phase 2 style self-evolution run，用 constfix constitution + 改进的 prompts/updater，观察 policy proposer 是否能产出 env-scoped 规则（而非 task-type-scoped）和 add_example。


## 2026-06-27 Two-Level Co-Working Self-Evolution: Intention + Outcome-Grounded Gate

目标：
回应 constfix 的核心发现——结构质量指标（completeness 68→98%）提升但 task hard/soft 不动（0.37≈0.38），说明 quality proxy 不等于 downstream 表现。按 IDEA.md 三层优化框架，实现 trajectory-level 与 step-level 自进化的协同机制，让 evolution 由实际 outcome（hard/soft）驱动而非 quality proxy。用户决策：gate 用 LLM pivotal judge（在 failed validation episodes 的 pivotal step 上盲判 baseline vs candidate），non-pivotal validation step 上做 no-regression 守卫。

设计（co-working）：
- Trajectory-level（全局，每 window）：rollout 后已有 hard/soft/fail_reason，做 credit assignment——标出 failed episode 的 pivotal step（feedback 若正确将最可能改变轨迹）+ intention_gap。
- Step-level（局部，每 action）：EFM 现在能看到 agent 的 `<think>` intention，判断 intention 是否被当前 observation fulfilled（intention_status: fulfilled/unfulfilled/unclear）。
- 协同：trajectory 用真实 outcome 告诉 step-level "哪里重要"（pivotal）+ "差在哪"（intention gap）；step-level 的进化（policy rules/examples）针对这些 pivotal 修复；gate 只在 pivotal step 上用实际 outcome 验收 → 解决 step-level 短期/历史增益过拟合（cadence=windowed，验收=outcome-grounded）。

做成了什么（最小改动，复用全部 window/split/patch/validation 基础设施）：
- models.py：StepFeedback +intention_status；TrajectoryCorrection +pivotal +intention_gap；config +gate_mode "outcome" +policy_gate_min_pivotal_gain=1。
- constitution.py：rule 6 要求输出 intention_status；新增 rule 9（agent_intention 为 untrusted data，只判 observation 是否达成 intention，不给 advice）。
- prompts.py：TRAJECTORY_SYSTEM 增加 outcome-grounded credit assignment（pivotal + intention_gap）；新增 PIVOTAL_GATE_SYSTEM + pivotal_gate_user_prompt；step_user_prompt 加 agent_intention。
- runtime.py：refine/_refine_step/_append/refine_many 全链路透传 intention；解析 intention_status；trace row 增加 intention 字段；_render_candidate 透传 intention。refine_many 兼容 3- 与 4-tuple。
- updater.py：_review 注入 episode outcome + 每步 intention/intention_status，解析 pivotal/intention_gap；重构出 _quality_totals 复用；新增 _outcome_gate（no-regression 守卫 + 在 failed validation episodes 上重跑 credit assignment 找 pivotal step + LLM pivotal judge 验收）；_gate 增加 outcome 分发；+_totals_dict helper。
- rollout.py：feedback_requests 增加 `_extract_think(model_responses[i])`（intention，此前被丢弃）。
- 配置 plumbing：adapter.py + integrations/efm.py 透传 feedback_policy_min_pivotal_gain。
- 测试：新增 test_outcome_gate_accepts_pivotal_judge_win（demonstrates outcome gate 接受 deterministic 不会接受的 pivotal-win）+ test_outcome_gate_rejects_when_baseline_not_beaten。全套 17 passed（原 15 + 2）。
- 备份：*.bak_20260627。

发现的问题：
- 关键发现（来自先前 constfix）：agent 读 core_signal 文本而非 signal_type 字段，且 completeness 不是 task success 的 binding constraint，所以纯 quality 进化不动分。这正是需要 outcome-grounded gate 的原因。
- 之前 agent 的 `<think>` intention 一直在 rollout.py 被丢弃（_extract_think 已存在但未使用）；hard/soft 已 stored 但 review 只用 success bool、gate 完全不用 → proxy 问题的根因。

下一步：
1. 2-ep intent smoke（idea runner，32B agent + 4B EFM，policy 关）确认 trace row 带 intention 且 intention_status 被填充。
2. 跑一个 outcome-gate 自进化 window（split role：4B step EFM + 30B optimizer，gate_mode=outcome，window=20，budget=50）观察是否产出 outcome-grounded v1，检查 gate_diagnostics 的 pivotal_candidate_wins / pivotal_baseline_wins。
3. 若 v1 accepted，用 v0 vs v1 配对 rollout 验证实际 hard/soft 是否提升（这才是 constfix 没做到的）。


## 2026-06-28 Co-Evolution Redesign: Within-Episode Reflection + Outcome-Gated Skill Delta

目标（用户主导的系统设计定稿）：
按 IDEA.md 三层框架，把 step-level 与 trajectory-level 自进化做成真正的协同（co-evolution），并确保 evolution 由实际 outcome 驱动。用户明确三点：(1) step-level optim 方向 = 把当前环境反馈以最好的方式喂给 agent，且可在线（task 进行中）；(2) trajectory-level = step 的方向（但来自整条 rollout）+ 捕捉"整体来看什么更重要"，在某些 env feedback 点把 local-best 升级为 whole-picture-best；(3) skill 更新参考 SkillOpt：改 delta 不重写整份 skill；step-level 的在线优化不直接改 EFM skill，而是写 mem/reflection，是否升级进 skill 由 gate 决定。

设计（三时间尺度）：
- Within-episode（每 K=3 步，在线）：EFM 维护 per-episode reflection（≤3 notes），condition 后续 step 反馈；不改 durable skill；reflection 是"毕业候选"。
- Window（每 N=8 episodes）：optimizer 读 window 的 corrections+reflections+outcomes，提一个 SkillOpt 式 delta（add_rule/add_example/retire，不重写）。
- Gate（outcome-grounded, reference-visible）：在 failed validation episodes 的 pivotal step 上，judge 看 target_feedback（whole_picture_feedback）判 candidate 是否比 baseline 更接近 target；non-pivotal 上 no-regression；通过才 v+1。

co-evolution 闭环：traj 算出 local→global delta（importance_gap）与目标（whole_picture_feedback）→ 进 proposal（importance_gap→rule, whole_picture→example）+ gate（target 作为 judge 参照）→ skill delta → step 在线应用，使其 local-best 逐步携带 global importance。收敛信号：pivotal 缺口逐窗缩小。

做成了什么（代码，最小改动，复用 window/split/patch/validation 全套）：
- models.py：config 用 reflect_enabled/reflect_every_k_steps=3/reflect_max_tokens/reflection_max_notes=3 取代 self_refine；StepFeedback 保留 intention_status；TrajectoryCorrection +pivotal +importance_gap +whole_picture_feedback；gate +"outcome" 模式 +policy_gate_min_pivotal_gain。
- constitution.py：PRIMARY OBJECTIVE = best local delivery；新增 intention 子句 + reflection 使用子句。
- prompts.py：step_user_prompt +agent_intention +episode_reflection；新增 REFLECT_SYSTEM + reflect_user_prompt；TRAJECTORY_SYSTEM 双轴（local correction + global importance/pivotal/whole_picture）；POLICY_SYSTEM 消费 pivotal target + 升级 recurring reflection；PIVOTAL_GATE_SYSTEM reference-visible；policy_user_prompt +episode_reflections。
- runtime.py：EpisodeFeedbackSession.reflection buffer；_refine_step 单遍、condition reflection；新增 _maybe_reflect/_reflect（efm_reflect, 每 K 步）；_render_candidate 用 reflection=None（gate 测 durable skill）；episode/artifact 记录 reflection。
- updater.py：_outcome_gate 在 pivotal pairs 注入 target_feedback；_propose_and_gate 收集 window reflections 传入 proposal。
- integrations/efm.py：split role 把 efm_reflect 归为 online（target backend）。
- 配置 plumbing：adapter + integrations 透传 feedback_policy_min_pivotal_gain。
- 测试：移除 self-refine 两个用例，新增 reflection 两个（K 触发、condition、不改 skill；disabled 不触发）。全套 19 passed。
- 备份 *.bak_20260627。

验证：
- 2-ep pipeline smoke（32B agent + 32B EFM/optimizer @8006）：0 fallback；reflection 正确填充且有用（"desklamp 不响应 turn on"、"alarmclock 只在 desk 2"、"重复 look 无新信息"）；intention 流入；3-note 上限生效。
- 正式 run：tmux efm_selfevo_n8，N=8 window=8 budget=50，outcome gate，min_support=2，min_pivotal_gain=1，输出 outputs/efm_selfevo_n8_20260628。

模型/角色：agent=target=qwen3-32b-awq-tool@8006；EFM(step+reflect)+optimizer(review/proposal/gate)=optimizer backend=同 32B@8006（role=optimizer）。弃用来历不明的 30B；full exp 再换小 EFM / 更强 optimizer。

下一步：
1. 等 n8 跑完，读 feedback_state.json 的 policy_updates[-1]：reason、gate_diagnostics（pivotal_total/candidate_wins/baseline_wins）、是否 v0->v1 accepted、graduated 的 rule/example 是否来自 reflection/whole_picture。
2. 若 accept，做 v0 vs v1 配对 rollout 看实际 hard/soft（constfix 没做到的因果验证）。
3. 据 gate 是否拿到 failed validation episodes 决定是否把 N 调大一点以稳定 gate。


## 2026-06-28 Terminal-Bench Qwen32B Terminus-2 Baseline

目标：
用本地 32B 服务在 Terminal-Bench Core 上跑出第一个非 oracle、非 nop 的可复现 baseline，并定位从 smoke 到正式 baseline 的工程 gap。

做成了什么：
- 32B 服务确认可用：`qwen3-32b-awq-tool` @ `http://localhost:8006/v1`，`max_model_len=40960`，api-key `token-abc123`。
- Terminal-Bench base images 本地化，避免 GHCR/Docker Hub 抖动：`ghcr.io/laude-institute/t-bench/python-3-13:latest`、`ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest`。
- 给 `terminal_bench/agents/terminus_2/terminus_2.py` 加了两个可选 agent kwargs：`max_tokens` 与 `enable_thinking`。原因是默认 LiteLLM/vLLM 会给 Qwen 接近 40k 的 completion budget，首轮 task 会长时间生成；`max_tokens=4096` 后 hello-world smoke 通过。`enable_thinking=False` 虽然更快，但 hello-world 写出额外空行而失败，本次正式 baseline 未使用。
- 有效 smoke：`experiment_runs/terminal_bench_qwen32_terminus2_xml_maxtok_smoke_20260628/2026-06-28__06-56-51/`，`hello-world` 1/1。
- 正式 5-task local baseline：`experiment_runs/terminal_bench_qwen32_terminus2_xml_5task_baseline_maxtok4096_20260628/2026-06-28__06-59-14/`。
  - 命令核心：`terminus-2`, `parser_name=xml`, `temperature=0.2`, `max_tokens=4096`, `n-concurrent=1`。
  - 结果：2/5 resolved，accuracy=40%。
  - resolved：`fix-permissions`, `hello-world`。
  - unresolved：`grid-pattern-transform`（pattern 解错）、`sqlite-db-truncate`（test_timeout）、`fibonacci-server`（server endpoint tests 全 fail）。

发现的问题：
- 已解决的环境 gap：之前只有 `nop` smoke；现在 32B+agent+Docker+verifier 全链路能跑正式任务。
- Docker gap：任务 Dockerfile 依赖 `ghcr.io/laude-institute/t-bench/*` base image，本地缺失时会因 registry TLS timeout 全部 build fail。必须先本地构建/缓存 base images。
- 任务选择 gap：`count-dataset-tokens` 会触发 HuggingFace/pip 外网依赖，不适合第一版 baseline；已改用 local subset。
- Agent gap：Qwen + terminus-2/xml 经常输出 `<response>` 前额外文本、命令缺换行、嵌套 response/重复 task_complete；parser 可恢复但影响效率与正确性。
- 参数 gap：不给 `max_tokens` 会让 vLLM 分配接近 40k completion，导致首轮调用过慢。`max_tokens=4096` 是当前可跑配置；更好的做法是 prompt/adapter 层强约束简短 XML。
- Caveat：baseline 启动前 8006 上有一个 ALFWorld `eval_only.py` 共享服务；未擅自停止。accuracy 可作初版参考，latency 不应视为干净测量。

下一步：
1. 固化 Terminal-Bench Qwen baseline runner 脚本，显式设置 `UV_CACHE_DIR`、api key、`max_tokens=4096`、task subset，避免手敲参数漂移。
2. 针对 Qwen/terminus-2 改 prompt 或 parser：禁止前置文本、要求每个 command 以 newline 结束、用 `printf` 替代 `echo -e`，降低格式恢复开销。
3. 在无共享 32B 负载时复跑同一 5-task subset；若稳定，再扩到 10-20 个不依赖外网下载的 task。

## 2026-06-28 Self-Evo N=8 Result + Gate Data-Flow Fix

做成了什么：
- N=8 自进化 run（efm_selfevo_n8_20260628）跑通整链：within-episode reflection（24 notes，内容有用）、trajectory credit assignment（9 corrections，4 pivotal 带 importance_gap+whole_picture_feedback）、proposal 产出 candidate（add_rule "state target absence" + add_example 由 reflection/whole_picture 毕业）。eval hard=0.75（8 items，偏易）。
- 首次 gate 失败：insufficient_validation_transitions。根因：hash split 对这 8 个 episode_id 给出 0 个 validation episode；且 _select_analysis 失败优先 → 2 个失败全进 proposal，gate 无失败可测。
- 修复（不依赖 hash split）：maybe_update 显式 hold out 一半失败给 gate；≥4 失败做干净 split，<4 失败共享（轻微 leakage，flagged，大窗口恢复干净 split）。_gate outcome 模式直接用 held-out gate_episodes，不再卡 validation-transition 阈值。_outcome_gate：guard transitions 来自 held-out，no-regression 守卫 + 在 held-out 失败的 pivotal step 上 reference-visible judge（看 target_feedback）。
- 用 8006/32B 对已 rollout 的 8 episodes 离线重跑新 gate（efm_regate_n8）：gate 正常运行；no-regression 守卫在 held-out step test:0000:5 上发现 candidate 会让 EFM 断言 obs 中不存在的实体 "alarmclock 1"（consistency=0）→ 正确拒绝 deterministic_quality_regressed。
- 测试：全套 19 passed（含 3 个直接调用 _gate 的用例补第三参）。

发现的问题：
- 这是 gate 的健康行为：候选规则 "always state target absence" 过度泛化，会让 EFM 在不相关/成功步骤命名未 grounding 的目标实体；consistency 守卫拦截、拒绝毕业，符合 EFM "只报 grounded 事实" 契约。pivotal judge 因 no-regression 先拒未触发。
- 小窗口张力：outcome-grounded 自进化需要足够失败（proposal 需 ≥2 支持，gate 需 ≥1 失败）。N=8、本 slice success=0.75 仅 2 失败 → 走 leakage 路径。干净 split 需 ≥4 失败。

下一步：
1. N=16 run（efm_selfevo_n16_20260628）：更多失败→干净 hold-out，看正向 accept 是否出现（pivotal judge 真正触发对 target 的比较）。
2. 若仍因 no-regression 被拒，说明 proposal 规则过宽——这是 proposal 质量问题（可在 POLICY_SYSTEM 要求更窄 scope/前置条件），非 gate bug。
3. accept 后做 v0 vs v1 配对 rollout 看实际 hard/soft。

## 2026-06-28 First Outcome-Grounded v0->v1 Accept (N=24, real data)

做成了什么：
- N=24 自进化 run（efm_selfevo_n24_8006_20260628，8006/GPU2）首次在真实数据上 v0->v1 ACCEPTED，完整正向路径触发。
- 24 episodes，success 18/24（6 失败→干净 hold-out split：proposal 3 失败、gate 3 失败）。6 corrections，2 pivotal。
- gate_diagnostics（outcome 模式）：no-regression 通过（baseline {cons16,comp16,eff16} == candidate {16,16,16}，零回退）；failed_gate_episodes=3，pivotal_steps=3；reference-visible pivotal judge **candidate_wins=3 / baseline_wins=0 / unsafe=0** → accepted。
- 毕业进 durable skill（v1）：add_rule "When reporting presence/absence of the target object, explicitly confirm its absence..."；add_example "You arrive at desk 1... the book is not among them. This confirms the book is not..."。二者均源自 reflection/whole_picture target → 闭环 coupling 全程生效。
- 对照：N=8（2 失败，leakage 路径，candidate consistency 幻觉被拒）、N=16（6 失败但 review 仅 1 correction、0 pivotal，example-only candidate completeness 5->4 被 no-regression 拒）。N=24 因更多失败+干净 split，optimizer 产出 scope 更窄、可泛化的规则，过 no-regression 且 pivotal 3-0。

发现的问题：
- 决定 accept 成败的是 proposal 质量与失败样本量：失败少→proposal 过宽/支持不足→被 no-regression 或 support 拦截；失败足够+干净 hold-out→可得无回退且 pivotal 占优的候选。gate 全程诚实（N=8/16 正确拒绝有害候选，N=24 正确接受净优候选）。
- 这与"full exp 用更强 optimizer"判断一致：32B optimizer 在小窗口产出弱/泄漏 delta；样本与 optimizer 能力是放大维度。

下一步：
1. v0 vs v1 配对 rollout（固定 skill、关 reflection 以隔离 policy 因果）看真实 hard/soft 是否提升——constfix 未做到的因果验证。两臂并行：v0@8006、v1@8008。
2. 之后扩 seed/连续多窗口（v1->v2）看是否稳定累积。

## 2026-06-28 Memory as an Evaluated Role in Evolution

目标（用户设计点）：reflection/mem 只持久化 final buffer 是缺陷——成功且有意义的 mem 应成为 evolution 中的一个角色：可审计、可评估、可决定是否毕业。直接做法：给 optimizer 每条 mem 的前后步（before/after window）+ outcome，让它判断该 mem 是否值得进 skill。用户选 bounded before/after window。

做成了什么：
- 记忆事件持久化（runtime.py）：session.reflection_history 记录每次 _reflect 触发的事件 {at_step, notes, new_notes}；写入 episode dict 与 artifact。修复"只存 final buffer"缺陷，可审计 mem 在 episode 内如何演化。
- 新离线阶段 efm_memory_eval（prompts.py MEMORY_EVAL_SYSTEM + memory_eval_user_prompt；updater.py _evaluate_memories）：对每条 mem 取 before=trace[at-K:at]、after=trace[at:at+K]（K=memory_context_steps=3）+ episode outcome，让 optimizer 判 verdict=graduate|discard 并给一行可复用 lesson（仅 feedback-selection，不含 agent advice）。fail-safe：异常/无 history 则该 episode 无毕业候选。
- 毕业候选进 proposal（updater._propose_and_gate + POLICY_SYSTEM）：用 graduated_memories（带 lesson + 来源 episode_id）替代原先扁平 reflection notes，作为与 pivotal corrections 并列的一等毕业候选。proposal 优先把 graduated lesson 变 add_rule/add_example。
- 评估在 proposal_pool（held out from gate）上做，不泄漏给 gate；gate 不变（no-regression + reference-visible pivotal judge）。
- config：memory_eval_enabled=True, memory_context_steps=3。stage efm_memory_eval 经 split role 归 optimizer（离线）。
- 测试：+3（reflection_history 持久化、_evaluate_memories 毕业有用 mem、无 history 跳过），全套 22 passed。

设计闭环（现状）：
- step-level（在线）：每 K 步写 mem，condition 后续 feedback；mem 事件带位置持久化。
- mem 评估（离线）：optimizer 用 before/after+outcome 判每条 mem 能否毕业 → 这就是"mem 作为 evolution 的角色"。
- proposal：graduated mem + pivotal corrections → delta。
- gate：held-out 失败上 no-regression + 对 whole-picture target 的 pivotal judge → 决定 v+1。

下一步（待用户指示，不擅自跑实验）：
1. 多窗口连续进化（v0→v1→v2…）让 skill 跨轮累积后再评测——正确的因果实验。
2. 第三 backend 实现 agent / 小 EFM / 强 optimizer 三模型分离。
3. 可选：持久化每步 reflection 快照以研究在线动态（目前存事件级，已足够审计毕业）。

## 2026-06-29 Shared EFM Harness (Phase A — multi-bench scaffold)

目标（用户主导）：把 tau2、appworld 接入与 alfworld 相同的 EFM 核心；不同 bench 共享 efm core；保留 SkillOpt 式 skill 迭代但不让 skillopt 约束其它 bench 代码。

做成了什么（research/efm/harness/，纯加法，benchmark-agnostic，无 skillopt 依赖）：
- _client.py：OpenAI 兼容 chat 封装（重试 + 关闭 thinking 的 chat_template_kwargs + 去 <think> 兜底）。
- feedback_model.py：OpenAIChatFeedbackModel，复刻 SkillOptFeedbackModel 的 split 路由（online efm_step/efm_reflect→target，其余 trajectory/proposal/gate/memory_eval→optimizer），但不 import skillopt。
- env.py：每个 bench 实现的最小契约 EFMEnv（iter_tasks/reset→Reset/step→Step/is_success）；动作解析与执行归 env，agent 与 runner 保持 bench 无关。
- agent.py：LLMAgent（冻结、无 skill，符合 IDEA Phase 1）。
- runner.py：唯一 rollout loop，arm∈{raw,handcrafted,efm} 只切换 agent 看到的 observation；产出共享契约 results.jsonl + feedback_state.json + feedback/<id>.efm.json。
- config.py：EndpointConfig/RunConfig，feedback_config() 转 FeedbackRuntimeConfig。
- diagnostics/harness_smoke.py：离线结构 smoke（stub model/agent/env）。

验证（path，非研究结论）：
- 离线结构 smoke PASS：raw+efm 两臂均产出契约文件；feedback_state.json episodes[].trace[].step_feedback 完整；research.efm.bench.eval.summarize 正常解析。
- 在线 smoke：OpenAIChatFeedbackModel split 路由实测 efm_step→4B@8007、efm_trajectory→32B@8006，thinking 关闭、输出简短；LLMAgent@8007 正常出动作。
- 回归：benchmarks/skillopt/tests/test_efm.py 仍 19 passed；未触碰 alfworld/skillopt 任何文件。

边界：仅证明共享 harness 通路可用；尚未接任何真实 bench。endpoints 现状：8006/8008=32B(qwen3-32b-awq-tool)、8007=4B(Qwen/Qwen3.5-4B)，旧 8001/30B 已下线。AppWorld 未安装（仅 terminal_bench 下有 converter，无 venv）。

下一步：Phase B tau2（vendoring + EFMEnv 适配 + 2-task raw-vs-efm smoke），Phase C appworld（安装原版 appworld + native 交互 env）。

## 2026-06-29 tau2 (τ²-bench) on shared EFM harness (Phase B — green smoke)

目标：把 τ²-bench 接入共享 EFM harness，与 alfworld 共用 efm core，不依赖 SkillOpt；先 2-task raw-vs-efm smoke。

做成了什么：
- Vendoring：benchmarks/tau2/upstream（官方 sierra-research/tau2-bench main，commit 8ebb749，见 VENDOR_INFO.txt；注意 PyPI 上的 tau2 是无关包，必须用官方 repo）。
- venv：envs/tau2（py3.12）装 tau2[gym]+openai；数据随源码自带，TAU2_DATA_DIR=upstream/data；LLM cache 默认关（无需 redis）。
- 适配器 benchmarks/tau2/efm_env.py：Tau2EFMEnv 包 tau2 gym AgentGymEnv（我方驱动 agent 对 tau2 user simulator），实现 research.efm.harness.env.EFMEnv。agent system prompt 由 domain policy+tools 合成；tau2 内部 LLM（user simulator + nl-assertion judge）经 litellm 路由到本地 OpenAI 兼容端点、thinking 关闭、无需 OpenAI key（patch evaluator_nl_assertions 的模块级 model 名）。
- run_smoke.py：raw/efm 双臂，产出共享契约 + 调 research.efm.bench.eval。

验证（path）：
- 端点分工：agent=32B@8008、EFM target=4B@8007、EFM optimizer=32B@8006、user sim+nl judge=32B@8006。
- mock 域 create_task_1 + update_task_1，max-steps=10：raw 2/2 hard=1.0、efm 2/2 hard=1.0（reward 经 DB+action+communicate 检查真实算出，全 1.0）。efm 质量 consistency/completeness/efficiency=100/100/50、fallback 0%。
- trace 抽检：agent 发 create_task(user_1, Important Meeting) → EFM 把 tool 结果精炼成 grounded progress 信号 → done()。早期 reward=0 仅因 agent 多传 optional 参数 description 触发 tau2 严格匹配；prompt 加“只传必需参数”后 reward=1.0。证明集成正确，非 adapter bug。

发现/边界：
- tau2 内部 max_steps 计 orchestrator 轮（agent+env+user），比我方 per-action budget 细，适配器给 4×。
- raw 臂 observation 是 tau2 累积会话串；agent 仍自带 history，存在轻微重复（smoke 可接受，正式实验可改为只回传增量）。
- mock 任务偏易、量小，只证明通路；非研究结论。

下一步：Phase C appworld（装官方 appworld + 数据 + native 交互 env 适配器 + smoke）；之后才考虑放大 tau2（airline/retail/telecom）与自进化窗口。

## 2026-06-29 appworld on shared EFM harness (Phase C — green smoke)

目标：把 AppWorld 以 native 交互 env 接入共享 EFM harness，与 alfworld/tau2 共用 efm core，不依赖 SkillOpt；先 2-task raw-vs-efm smoke。

前置：确认用户此前并未装 AppWorld（仅 terminal_bench 下有 converter、无 venv）。装官方原版（PyPI appworld 0.1.3.post1，StonyBrookNLP / appworld.dev），envs/appworld（py3.11）+ openai；appworld install 解包 apps 代码（tests 落 ~/.appworld/tests），appworld download data --root benchmarks/appworld（193M）。注意 PyPI 上 tau2 是无关包，但 appworld 确为原版。

做成了什么：
- 适配器 benchmarks/appworld/efm_env.py：AppWorldEFMEnv 包 appworld.AppWorld（native 交互世界），实现 EFMEnv。agent system prompt 编码 AppWorld 协议（写 python 调 apis、api_docs 查 API、supervisor 取凭证、complete_task 收尾）；step 从 agent 文本抽取 ```python``` 代码块交 world.execute；done=task_completed；评测用 world.evaluate().to_dict()（success + passes/num_tests），code-based 确定性、无 LLM judge。
- run_smoke.py：raw/efm 双臂，从 split 取 task ids，产出共享契约 + 调 research.efm.bench.eval。

验证（path）：
- 端点：agent=32B@8008、EFM target=4B@8007、EFM optimizer=32B@8006（appworld 无需 user sim / judge）。
- train 前 2 任务（82e2fac_1/2），max-steps=10：raw 2/2 hard=0.0 soft=0.5、efm 2/2 hard=0.0 soft=0.5（各任务 2 个 test 过 1 个 = no-model-change 项）。efm 质量 consistency/completeness/efficiency=100/100/60、fallback 0%。
- trace 抽检（82e2fac_1，spotify QA）：agent 真实多步——login(password) →401、show_account_passwords 取凭证、用真实 password 重登成功；EFM 对真实执行输出给出正确类型 grounded 反馈（tool_error@401 → state_change@凭证表 → ambiguity@裸 success）。证明集成正确。
- hard=0 属预期：AppWorld 任务长程、冻结 agent + 小 step budget；smoke 只证通路，非研究结论。

发现/边界：
- AppWorld 有 max_interactions 上限（计 execute 次数，reset 列 apps 占 1 次）；适配器设为 3× per-step budget，使我方 max_steps 为绑定约束。
- 任务长程，需要更大 step budget / 更强 agent 才有非零 hard；放大时再调。

三 bench 现状：alfworld(经 SkillOpt) / tau2 / appworld 均跑通 raw-vs-efm smoke，共用 research/efm 核心；tau2、appworld 仅依赖各自 bench + research.efm，不 import skillopt。Phase A-C（scaffold+smoke）完成。

下一步（待用户指示）：放大单 bench（更多任务/步数、跑 raw vs handcrafted vs efm 对照）；或开自进化窗口（policy_update_enabled）做 v0→v1；或把 alfworld 也迁到共享 harness 以彻底统一。


## 2026-06-29 EFM long-run self-evolution (EFM_LONGRUN_20260629)

Question: can current EFM self-evolve over longer windows and improve environment feedback without major framework changes?

Artifacts:
- Report: /data5/ninghan/tlm/benchmarks/skillopt/outputs/efm_selfevo_longrun_report_20260629.md
- Main 5-window run: /data5/ninghan/tlm/benchmarks/skillopt/outputs/efm_selfevo_w24x5_mem_8006_20260628
- Continuation from v2 after prompt and validator fix: /data5/ninghan/tlm/benchmarks/skillopt/outputs/efm_continue_v2_fixed_w24x3_8008_20260629
- Held-out v0 eval: /data5/ninghan/tlm/benchmarks/skillopt/outputs/efm_heldout_v0_no_reflect_14_8006_20260629
- Held-out v2 eval: /data5/ninghan/tlm/benchmarks/skillopt/outputs/efm_heldout_v2_no_reflect_14_8008_20260629

Result:
- Main w24x5: 120 episodes, 5 update attempts, 2 accepted, final policy v2, hard/soft 53/120 = 0.4417.
- Continuation: 72 additional episodes from v2, 3 more update attempts, 0 accepted, hard/soft 19/72 = 0.2639.
- Total policy-update evidence: 8 attempts, 2 accepted. The accepted rules both emphasize grounded task-target absence across locations.
- Held-out isolated policy comparison on test:0120-0133: v0 3/14, v2 3/14. v2 rescued test:0124 but regressed test:0126.
- Feedback-quality comparison: consistency 0.9847 -> 0.9908; completeness 1.0000 -> 1.0000; efficiency 0.9985 -> 0.9969; fallback 0 -> 0.

Evidence boundary:
- This supports that the loop can perform non-smoke, multi-window self-evolution and accept some grounded feedback-policy improvements.
- It does not yet support a claim that evolved feedback improves downstream ALFWorld success. The current proposal distribution appears saturated around target-absence wording.
- The minimal code/prompt fix was useful as a guardrail: it rejects exploration-guidance rules that previously caused hallucinated entities, but it is not sufficient to create new useful policy dimensions.

Next decision: keep the framework, but change proposal pressure/evaluation toward diverse feedback dimensions and agent-consumption evidence before running larger batches.


## 2026-07-03 SkillOpt ALFWorld raw-preserving intention-aware feedback augmentation low-budget

Question: keep SkillOpt ALFWorld raw environment feedback, add one natural-language feedback augmentation sentence conditioned on the agent stated intention, and test local-Qwen SkillOpt behavior without EFM self-evolution.

Implementation:
- Modified benchmarks/skillopt/skillopt/envs/alfworld/rollout.py so target agents are asked to emit intention tags before action tags.
- When EFM is enabled, the next agent observation preserves raw env feedback and appends one natural-language augmentation sentence.
- Conversation traces persist intention, raw_env_feedback, feedback_augmentation, feedback_intention_status, and feedback_signal_type for SkillOpt reflection and analysis.
- EFM self-evolution was disabled in all augmentation experiments: env.feedback_policy_update_enabled=false and env.feedback_reflect_enabled=false; feedback_state.json confirmed policy.version=0 and no policy_updates.

Run setup:
- Repo: /data5/ninghan/tlm/benchmarks/skillopt
- Target model: qwen3-32b-awq-tool at http://127.0.0.1:8006/v1
- Optimizer and EFM model: qwen3-32b-awq-tool at http://127.0.0.1:8008/v1
- Training budget per arm: 1 epoch, train_size=16, batch_size=8, 2 update steps, selection env_num=12, max_steps=30, workers=4.
- Final and cross eval: valid_unseen, test_env_num=32, max_steps=30.

Artifacts:
- Raw SkillOpt train: /data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt32_raw_lowbudget_20260703_122629b
- Intent-aug SkillOpt train: /data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt32_intentaug_lowbudget_20260703_122629b
- Raw best eval: /data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt32_raw_lowbudget_20260703_122629b_test32_best
- Intent-aug best eval: /data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt32_intentaug_lowbudget_20260703_122629b_test32_best
- Initial plus raw cross eval: /data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt32_raw_initial_lowbudget_20260703_122629b_test32
- Rawbest plus intent-aug cross eval: /data5/ninghan/tlm/benchmarks/skillopt/outputs/skillopt32_intentaug_rawbest_lowbudget_20260703_122629b_test32

Results:
- Training selection: raw accepted step_0002 and reached best_score=0.6667; intent-aug rejected both candidate updates and kept initial_skill with best_score=0.6667.
- valid_unseen test32 2x2:
  - initial skill plus raw feedback: 24/32 = 0.7500 hard/soft
  - initial skill plus intent-aug feedback: 27/32 = 0.84375 hard/soft
  - raw-optimized skill plus raw feedback: 22/32 = 0.6875 hard/soft
  - raw-optimized skill plus intent-aug feedback: 29/32 = 0.90625 hard/soft
- Regression tests: from benchmarks/skillopt, pytest tests/test_efm.py tests/test_alfworld_worker_start.py passed: 22 passed.

Evidence boundary:
- This is a low-budget fixed test32 subset result, not a full SkillOpt paper reproduction.
- The 2x2 strongly suggests the feedback augmentation channel itself improved agent behavior on this subset; the raw skill update alone did not generalize on the same subset.
- Because intent-aug training kept the initial skill, the main observed gain is currently from online feedback consumption, not from better skill text evolution.

Next decision: run larger valid_unseen or full-test repetitions with fixed seeds and either freeze skill to isolate feedback-channel effect, or redesign reflection prompts so SkillOpt uses intention and feedback diagnostics to update skill text rather than relying mostly on online augmentation.

## 2026-07-04 SkillOpt ALFWorld intention-aug 4B follow-up

Context: target agent is Qwen/Qwen3.5-4B on port 8007; optimizer/EFM is qwen3-32b-awq-tool on port 8008. EFM policy self-update and trajectory reflection remain disabled for these runs. Split is ALFWorld valid_unseen test32, max_steps=30.

Baseline v1, old rollout prompt plus raw-preserving feedback augmentation:
- Train raw lowbudget: output `benchmarks/skillopt/outputs/skillopt4b_raw_lowbudget_20260704_4b_1059`; both step candidates rejected; best remains initial; selection best 0.6667.
- Train intent-aug lowbudget: output `benchmarks/skillopt/outputs/skillopt4b_intentaug_lowbudget_20260704_4b_1059`; step1 accepted, step2 rejected; best step1; selection best 0.5833.
- Test32:
  - initial+raw: 24/32 = 0.7500, mean turns 15.19, median 10.0.
  - initial+intent-aug: 27/32 = 0.8438, mean turns 13.19, median 7.5.
  - aug-best+raw: 23/32 = 0.7188, mean turns 15.06, median 8.0.
  - aug-best+intent-aug: 25/32 = 0.7812, mean turns 13.50, median 8.5.
- Diagnosis: augmentation helps base execution, but the accepted skill does not generalize on valid_unseen. Also most old-prompt trajectories did not contain explicit `<intention>` tags, so the method behaved more like raw-preserving feedback-summary augmentation than true stated-intention feedback.

Prompt/formatter v2:
- Code changes: rollout prompts request `<think>`, `<intention>`, `<action>`; `fmt_trajectory` exposes intention/raw_obs/feedback_aug/feedback_meta; analyst prompts mention intention diagnostics.
- Smoke: Qwen-4B begins emitting `intention`, and EFM produces fulfilled/unfulfilled statuses.
- Train output `benchmarks/skillopt/outputs/skillopt4b_intentaug_promptv2_lowbudget_20260704`: both candidates rejected; baseline selection 0.5833; step1 selection 0.5000; step2 selection 0.5000.
- Diagnosis: explicit intention signal exists, but analyst/ranking still produce generic search/visibility/backtrack edits.

Prompt/formatter v3:
- Additional prompt filter in analyst/merge/ranking to down-rank edits that restate existing ALFWorld basics and to allow selecting fewer/zero edits.
- Train output `benchmarks/skillopt/outputs/skillopt4b_intentaug_promptv3_lowbudget_20260704`: step1 rejected; step2 accepted as new best; selection best 0.6667.
- Accepted skill is still mostly search/location-tracking guidance, but it passes selection under strict-intention prompt.
- Test32 under promptv3 code:
  - initial+raw: 19/32 = 0.5938, mean turns 18.63, median 18.0.
  - initial+intent-aug: 23/32 = 0.7188, mean turns 16.59, median 11.5.
  - v3-best+raw: 23/32 = 0.7188, mean turns 16.09, median 12.0.
  - v3-best+intent-aug: 29/32 = 0.9062, mean turns 13.38, median 9.0.
- Interpretation: strict intention prompting hurts base 4B execution substantially, but with the evolved skill and augmentation together it reaches the best 4B test32 result so far. Next mechanism to try: avoid strict output-format prompting and infer intention from the target model free-text rationale before `<action>`, preserving old execution behavior while providing EFM/reflection an intention field.


## 2026-07-04 SkillOpt ALFWorld Qwen3.5-4B promptv4 soft-intention augmentation

Question: can raw-preserving intention-aware feedback augmentation improve base ALFWorld execution and/or SkillOpt skill evolution with the paper-aligned 4B target agent, while keeping EFM self-evolution disabled?

Setup:
- Repo: `/data5/ninghan/tlm/benchmarks/skillopt`
- Target agent: `Qwen/Qwen3.5-4B` at `http://127.0.0.1:8007/v1`
- Optimizer / EFM: `qwen3-32b-awq-tool` at `http://127.0.0.1:8008/v1`
- Train run: `outputs/skillopt4b_intentaug_promptv4_lowbudget_20260704`
- Eval split: `valid_unseen`, `test_env_num=32`, `max_steps=30`, `workers=4`
- EFM self-evolution disabled: `env.feedback_policy_update_enabled=false`, `env.feedback_reflect_enabled=false`
- v4 mechanism: keep original soft rollout prompt; extract explicit `<intention>`, else `<think>`, else free-text rationale before `<action>`; preserve raw feedback and append one natural-language augmentation sentence.

Training result:
- Baseline selection: `6/12 = 0.5000`
- Step 1: rollout `3/8 = 0.3750`; selected two generic failure edits about searched locations / visible-object pickup; selection `7/12 = 0.5833`; accepted as new best. Timing: total `2151.1s`, rollout `952.9s`, reflect `39.0s`, aggregate `43.6s`, select `6.5s`, evaluate `1109.0s`.
- Step 2: rollout `3/8 = 0.3750`; selected further generic navigation/progress edits; selection `6/12 = 0.5000`; rejected. Best remains step 1.

Valid-unseen test32 results:

| condition | hard | mean turns | median turns | success mean turns | notes |
|---|---:|---:|---:|---:|---|
| initial + raw | `20/32 = 0.6250` | `16.91` | `11.5` | `9.05` | raw baseline under v4 code |
| initial + intent-aug | `27/32 = 0.8438` | `13.50` | `10.0` | `10.44` | strong execution gain from augmentation alone |
| step1 best + raw | `23/32 = 0.7188` | `16.50` | `14.5` | `11.22` | accepted skill generalizes modestly under raw feedback |
| step1 best + intent-aug | `20/32 = 0.6250` | `17.09` | `11.5` | `9.35` | negative interaction between generic skill edits and augmentation |

Task-type split for intent-aug:
- `initial + intent-aug`: look_at_obj_in_light `16/18 = 0.889`, pick_and_place `11/14 = 0.786`
- `step1 best + intent-aug`: look_at_obj_in_light `13/18 = 0.722`, pick_and_place `7/14 = 0.500`

Feedback diagnostics:
- `initial + intent-aug`: 432 steps, 362 extracted intentions; statuses `unfulfilled=252`, `fulfilled=74`, `unclear=106`.
- `step1 best + intent-aug`: 547 steps, 478 extracted intentions; statuses `unfulfilled=366`, `fulfilled=73`, `unclear=108`.

Interpretation:
1. Feedback augmentation clearly helps base 4B execution on this 32-task valid-unseen subset: `+7/32` hard and lower mean turns compared with raw.
2. SkillOpt can accept an update under v4, and the accepted skill has modest raw-feedback transfer: `+3/32` hard over initial raw.
3. The accepted skill does not combine well with augmentation. It mostly adds generic strict navigation / avoid-revisit / immediate-pickup rules; under repeated `unfulfilled` feedback, the agent becomes more prone to over-searching and loops. The failure is especially visible on pick_and_place tasks.
4. Mechanism implication: the next useful change should not rewrite SkillOpt broadly. The narrow fix is to normalize the intention signal into short action-level expected outcomes before EFM refinement, so feedback distinguishes action execution/result mismatch from broad task-search failure and gives SkillOpt less incentive to learn generic search heuristics.


## 2026-07-05 Intra Activation Steering Prototype

目标：
围绕 Steering-Mediated Skill Self-Evolution 先建立最小 steering prototype：能从 positive/negative behavioral activations 得到 steering vector，能把 vector 注入模型层，能观察到模型行为变化，最好在一个受控任务上提升指标。

做成了什么：
- 新增 `research/steering/`：
  - `vectors.py`：padding-aware mean pooling；positive-minus-negative oriented PCA / mean-diff steering vector。
  - `hooks.py`：HF-style transformer layer activation hook，支持 `model.model.layers`、`model.language_model.layers`、`model.model.language_model.layers`、`model.transformer.h`。
  - `run_intra_action_steering.py`：真实 HF causal LM action-selection runner，内置 procedural action micro-benchmark，输出 alpha sweep、prediction jsonl 和 `steering_vector.pt`。
  - `run_tiny_intra_steering_smoke.py`：当 HF-loadable LLM 不可用时的 tiny-model steering smoke。
  - `README.md`：运行命令、产物和解释边界。
- 新增 `tests/research/steering/`，覆盖 PCA 方向、padding pooling、activation hook 行为。验证命令 `envs/skillopt-qwen35-vllm-cu128/bin/python -m pytest tests/research/steering -q` 通过：`3 passed`。

实验结果：
- 真实本地 LLM 路线当前被环境约束阻塞：
  - `models/Qwen3.5-4B` 是完整 checkpoint，但当前 HF `transformers` 不识别 `qwen3_5` 架构，`AutoModelForCausalLM` 失败。
  - `models/Qwen3-32B-AWQ` 是完整 AWQ checkpoint，但当前 env 缺 `auto-awq`，HF 不能加载。
  - `models/Qwen3-32B` 是 HF 支持的 `qwen3`，但目录只有 `model-00001-of-00017`，checkpoint 不完整。
  - lab-50 直接下载 `Qwen/Qwen2.5-0.5B-Instruct` 失败，原因是远端 `huggingface.co` `Network is unreachable`；升级权限重试后仍失败。
- tiny smoke 产物：`research/steering/runs/tiny_intra_smoke_20260705_v2/summary.json`。
  - baseline accuracy：`22/40 = 0.55`
  - best steering alpha：`1.0`
  - steered accuracy：`39/40 = 0.975`
  - delta accuracy：`+0.425`
  - prediction changes vs baseline：`17`

发现的问题：
- 目前已经证明工程闭环：intra positive/negative activations -> PCA steering vector -> layer hook -> behavior changes -> controlled metric improves。
- 但 tiny smoke 不是 LLM agent 证据；它只证明 steering pipeline 和注入机制可用。
- 下一步要拿真实 LLM 结果，需要先解决一个模型可加载入口：安装/准备 `auto-awq` 跑 Qwen3-32B-AWQ，或补齐/下载一个 HF-standard 小模型，或获得支持 `qwen3_5` 的 transformers/runtime。

下一步：
优先让 `run_intra_action_steering.py` 在一个真实 HF-loadable instruct model 上跑通 alpha sweep；成功标准是 `alpha=0` 到某个正 alpha 出现 action-selection accuracy 或 correct-minus-wrong margin 的稳定提升，并保留 per-case prediction changes。


## 2026-07-06 Qwen3-4B Intra Steering Alpha Sweep

目标：
用真实 HF-loadable `Qwen3-4B-Instruct-2507` 跑 `research/steering/run_intra_action_steering.py`，验证 intra steering vector 是否能改变模型行为，并观察是否提升 action-selection 性能。

做成了什么：
- 确认模型目录 `/data5/ninghan/tlm/models/Qwen3-4B-Instruct-2507` 完整：3/3 safetensors shard、`model.safetensors.index.json`、tokenizer 和 config 均存在；HF config/tokenizer 加载通过，模型类型 `qwen3`，36 layers，hidden size 2560。
- 初始 run：`research/steering/runs/intra_action_qwen3_4b_2507_layer24_20260706`。在原始标签下 baseline 为 `10/16`，各 alpha 未提升 accuracy。
- 调试发现 eval case 标签 bug：若干 `good_is_a=True` case 的 bad/good action 传参写反，例如 `kettle_heat`、`mug_sink`、`coffee_machine`。该 bug 使旧的 `10/16 -> 12/16` 结果不可作为有效提升证据。
- 增加测试并修复标签：`envs/skillopt-qwen35-vllm-cu128/bin/python -m pytest tests/research/steering -q` 通过 `5 passed`。
- 新增 `--prompt-style neutral`，用于去掉显式 anti-repeat/no-op 指导，测试是否原 prompt 过于简单。

修正标签后的结果：
- Guided prompt run：`research/steering/runs/intra_action_qwen3_4b_2507_pca_layer16_fixedlabels_20260706`
  - baseline accuracy：`16/16 = 1.0`
  - best alpha：`-10`
  - best accuracy：`16/16 = 1.0`
  - best margin：`25.67`，baseline margin：`25.45`
  - 强 steering 会改变行为并退化：例如 `alpha=-60/-70/-80/-90` 为 `6/16`
- Neutral prompt run：`research/steering/runs/intra_action_qwen3_4b_2507_pca_layer16_neutral_fixedlabels_20260706`
  - baseline accuracy：`16/16 = 1.0`
  - best alpha：`-20`
  - best accuracy：`16/16 = 1.0`
  - best margin：`28.38`，baseline margin：`26.91`
  - 说明 steering 能提高正确选项 logprob margin，但当前 16-case benchmark 已到 accuracy ceiling，无法证明 accuracy 提升。

发现的问题：
- 真实 Qwen3-4B 路径已跑通；现在缺的不是模型加载，而是更有区分度的 evaluation set。
- 当前 micro-benchmark 对 Qwen3-4B 太容易，修正标签后 baseline 已满分；因此只能证明 steering 改变 hidden-state behavior / margin，不能证明 accuracy gain。
- 强 alpha 可以显著改变模型决策，但过强会退化，说明 layer hook 生效，alpha 需要 gate 或 validation sweep。

下一步：
构造更难、更接近 rollout 的 steering eval：从 ALFWorld/GIAI2 真实失败 step 抽取 original action、candidate repair action 和环境反馈，避免人工 toy cases 的 ceiling；成功标准应从 `accuracy` 扩展为 `invalid/repeat/no-op 修复率 + correct-minus-wrong margin + held-out task success`。


## 2026-07-06 ALFWorld Trace Steering Prototype

目标：
把 steering prototype 从 toy action cases 推进到真实 ALFWorld 轨迹：从 `.efm.json` step trace 中抽 positive/negative behavior states，构造 steering vector，并在 held-out episode 的 offline repair action-selection 上评估。

做成了什么：
- 重构 `research/steering/`：
  - `schema.py`：`ChoiceCase` / `SteeringDataset` 数据结构。
  - `prompts.py`：behavior-state 与 A/B action-choice prompt。
  - `engine.py`：模型加载、hidden-state 表示抽取、alpha sweep、产物落盘。
  - `toy_cases.py`：原 toy 数据独立出来，旧 `run_intra_action_steering.py` 只保留兼容 CLI。
  - `alfworld_cases.py`：从 ALFWorld EFM traces 抽 positive/negative states 与 held-out repair cases。
  - `run_alfworld_steering.py`：ALFWorld trace steering CLI。
- 测试覆盖增加到 `8 passed`，包括 ALFWorld repeated-action 抽取、episode-level train/eval split、prompt style 和原 vector/hook 测试。

抽取规则：
- 输入：`benchmarks/skillopt/outputs/idea_alfworld_32b_efm_constfix_run2_20260627/feedback/*.efm.json`
- Positive states：非重复、`signal_type in {progress, state_change}`，且 action 不是低信息 `look/inventory`。
- Negative states：重复 action，或 `signal_type in {constraint_violated, ambiguity, invalid, no_progress, error}`。
- Eval case：在 held-out episode 中，用 negative step 的 action 作为 bad option，用同一 episode 后续第一个非重复 positive action 作为 repair option。
- Split：100 episodes 按 episode id 排序，70 train episodes 用于 vector，30 held-out episodes 用于 eval cases。

关键 run：
`research/steering/runs/alfworld_qwen3_4b_constfix_pca_l16_neutral_split70_20260706`

设置：
- Model：`models/Qwen3-4B-Instruct-2507`
- Layer：`16`
- Method：`pca`
- Prompt style：`neutral`
- Train states：`96 positive / 96 negative`
- Eval cases：`64`
- Alpha sweep：`-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120`

结果：
- Baseline `alpha=0`：`29/64 = 0.453125`
- Best `alpha=-40`：`39/64 = 0.609375`
- Delta：`+10/64 = +0.15625`
- Prediction changes vs baseline：`14`
- Changed-case decomposition：`12` fixed, `2` broken
- Mean correct-minus-wrong margin：baseline `-1.094` -> best `+1.274`

解释边界：
- 这已经是“真实 ALFWorld trace -> steering vector -> held-out offline action-selection gain”的证据，不再是 toy smoke。
- 但它仍不是 full ALFWorld rollout success gain：repair option 来自同一 held-out episode 的未来 progress action，相当于离线反事实选择任务，不等于 agent 真正在环境中 rollout 后会成功。
- 负 alpha 最优说明当前 PCA 向量的符号与“更好 repair choice”方向相反；这不影响 steering 生效性，但下一步需要在 validation split 上自动选 alpha/sign。

下一步：
把 offline choice eval 接到真正 rollout：在 ALFWorld agent 每一步对候选动作或 token generation 注入 selected steering vector，先做小 batch A/B：unsteered vs steered，指标看 repeat/no-op/constraint violation rate、episode success、turn count。

## 2026-07-06 — True ALFWorld HF Rollout Steering Smoke

Question: can the ALFWorld intra steering vector that improved offline action-choice accuracy be applied inside a real ALFWorld rollout with a locally loaded HF model, and does it change or improve task behavior?

Implementation:
- Added `research/steering/rollout_hf.py` with reusable HF chat generation, activation steering hook integration, response normalization, vector loading, and trace metrics.
- Added `research/steering/run_alfworld_hf_rollout.py`, a real ALFWorld runner that reuses SkillOpt's `build_alfworld_env` and TextWorld step/projection path but replaces the HTTP `chat_target` backend with local HF generation so hidden-state steering is possible.
- Added `tests/research/steering/test_rollout_hf.py`; steering test suite passes: `13 passed`.

Setup:
- Model: `models/Qwen3-4B-Instruct-2507` loaded via HF BF16 on GPU 2.
- Vector: `research/steering/runs/alfworld_qwen3_4b_constfix_pca_l16_neutral_split70_20260706/steering_vector.pt`, layer 16.
- Environment: real ALFWorld valid_unseen gamefiles from `benchmarks/skillopt/outputs/skillopt4b_promptv4_initial_raw_test32/results.jsonl`.
- Generation: greedy, `max_new_tokens=768`; shorter 128-token smoke truncated before `<action>` and caused fallback `look`, so 768 is the usable smoke setting.

Results:
- `research/steering/runs/alfworld_hf_rollout_smoke_1ep5step_t768_20260706`: baseline and steered both 0/1 success, but paired action changed on 4/5 steps with no invalid actions.
- `research/steering/runs/alfworld_hf_rollout_2ep10step_t768_a-40_20260706`: baseline 1/2 success, steered alpha=-40 0/2 success; paired actions changed on 15/18 comparable steps; invalid actions stayed 0, repeated actions increased from 1 to 7.
- `research/steering/runs/alfworld_hf_rollout_2ep10step_t768_a40_20260706`: steered alpha=+40 0/2 success; invalid actions 0, repeated actions 3.

Evidence boundary:
- Established: true activation steering is wired into real ALFWorld rollout and changes environment-facing valid actions under the same task/skill/model.
- Not established: task-level improvement in free-generation rollout. The offline choice-task best alpha (-40) does not directly transfer; in this small real rollout it hurts success and increases repetition.
- Next smallest useful step: tune rollout-specific steering application, especially token slice and alpha grid, instead of assuming the offline choice vector/alpha is deployable as-is.

## 2026-07-06 — Clean ALFWorld Multi-Layer Steering Prototype

Question: can we remove EFM/SkillOpt-skill confounds and run activation steering directly on official ALFWorld with HF Qwen3-4B, using NPM-style middle-to-late layers 17-19?

Implementation:
- Added `research/steering/clean_alfworld.py` for clean prompt construction, exact admissible-action projection, env-only trajectory labeling, and paired-result summaries.
- Added multi-layer hook support in `research/steering/hooks.py` via `MultiLayerActivationSteerer`.
- Added `research/steering/run_clean_alfworld_steering.py`, which uses official `alfworld.agents.environment.get_environment` and does not import `skillopt.envs.alfworld.rollout`; the only reused artifact is the ALFWorld config/data path.
- Added tests for clean prompt boundaries, env-only labeling, exact admissible-action fallback parsing, and multi-layer hook behavior. Verification: `envs/skillopt-qwen35-vllm-cu128/bin/python -m pytest tests/research/steering -q` -> `20 passed`.

Clean setting:
- Model: `models/Qwen3-4B-Instruct-2507` loaded via HF BF16.
- Layers: `17,18,19`.
- No EFM traces, no EFM runtime, no SkillOpt skill injection, no SkillOpt rollout wrapper.
- Prompt includes only task, recent history, current observation, admissible actions, and output format.
- Clean vector source: baseline clean rollout labels. Positive uses terminal success when available; because first clean smokes had no success, weak positive uses env-only first-time valid non-low-information actions. Negative uses invalid, repeated, low-information loops, and timeout-tail behavior.

Smoke evidence:
- Initial clean look-at-light smoke on first 2 valid_unseen tasks failed vector construction: baseline 0/2 success, positive=0, negative=16; invalid actions remained high even after parser fix.
- Pick-and-place smoke: `research/steering/runs/clean_alfworld_smoke_pickplace_2ep15step_a5_weakpos_20260706`.
  - Gamefiles: manifest offset 19-20, `pick_and_place_simple-Mug-None-Desk-308`.
  - Baseline: 0/2 success, mean_turns=15, invalid_actions=5, repeated_actions=0.
  - Clean vector artifact: `clean_steering_vectors.pt`, layers `[17,18,19]`, each vector shape `(2560,)`, 5 positive texts / 25 negative texts.
  - Alpha 5: 0/2 success, mean_turns=15, invalid_actions=4, repeated_actions=0, paired_action_changes=10/30.

Evidence boundary:
- Established: clean ALFWorld steering path works without EFM, without SkillOpt skill, and without SkillOpt rollout wrapper; multi-layer steering at Qwen3-4B layers 17-19 changes real environment-facing actions.
- Not established: task-level improvement. Clean prompting makes the baseline policy weak and often loops, so the current clean vector mostly proves behavioral controllability, not performance gain.
- Next smallest useful step: improve clean baseline action selection without adding learned skill, likely via stricter action-only decoding/reranking over admissible actions, then rerun alpha grid on 10-20 episodes.

## 2026-07-06 — SkillOpt Skill-Edit to Steering V1 Smoke

Question: can a text-space SkillOpt edit from a deliberately rough ALFWorld skill be distilled into an activation-space steering vector?

Implementation:
- Added rough initial skill: `benchmarks/skillopt/skillopt/envs/alfworld/skills/rough_v1.md`.
- Added paired old/new rollout parser: `research/steering/skill_edit_cases.py`.
- Added skill-edit vector runner: `research/steering/run_skill_edit_steering.py`.
- Added tests: `tests/research/steering/test_skill_edit_cases.py` and `tests/research/steering/test_skill_edit_runner.py`.
- Wrote design/plan:
  - `docs/superpowers/specs/2026-07-06-skill-edit-steering-v1-design.md`
  - `docs/superpowers/plans/2026-07-06-skill-edit-steering-v1.md`

SkillOpt update artifact:
- Run: `benchmarks/skillopt/outputs/skill_edit_v1_rough_train_20260706b`.
- Initial rough skill selection score: `4/6 = 0.6667`.
- Step 1 training rollout: `0/4`, giving failure signal.
- Optimizer proposed 2 edits, expanding skill length `290 -> 1127`.
- Candidate path: `benchmarks/skillopt/outputs/skill_edit_v1_rough_train_20260706b/steps/step_0001/candidate_skill.md`.
- Gate result: rejected, candidate selection `3/6 = 0.5000 <= current 4/6 = 0.6667`.
- Evidence boundary: this is a SkillOpt-generated text edit, not an accepted SkillOpt best-skill update.

HF paired rollout traces:
- Old rough skill: `research/steering/runs/skill_edit_v1_rough_hf_6ep_20260706`.
  - 6 look-at-light valid_unseen episodes, `0/6` success, `5` repeated actions, `0` invalid actions.
- Candidate skill: `research/steering/runs/skill_edit_v1_candidate_hf_6ep_20260706`.
  - Interrupted after usable partial traces; reconstructed `baseline/results_partial.jsonl`.
  - 4 completed paired episodes, `1/4` success, `2` repeated actions.

Vector extraction:
- Dataset: same task type `look_at_obj_in_light`.
- Old states: 64 selected from 72 candidates.
- New states: 45 selected from 45 candidates.
- Matched eval cases: 43 exact old/new action choices.
- Vector: layer 16, `mean_delta = mean(h_new) - mean(h_old)`, normalized.

Results:
- Edit vector: `research/steering/runs/skill_edit_v1_edit_mean_delta_l16_20260706`
  - Baseline alpha 0: `15/43 = 0.3488`.
  - Best alpha `-60`: `23/43 = 0.5349`, delta `+8/43 = +0.1860`.
  - Best margin: `+1.0407`.
  - Prediction changes vs baseline: `20`.
- Random vector: `research/steering/runs/skill_edit_v1_random_mean_delta_l16_20260706`
  - Best `23/43 = 0.5349`, best margin `+0.7789`.
- Unrelated ALFWorld vector: `research/steering/runs/skill_edit_v1_unrelated_mean_delta_l16_20260706`
  - Best `24/43 = 0.5581`, best margin `+0.1863`.

Interpretation boundary:
- Established: the old/new skill-edit delta produces a real activation steering signal on paired offline action-choice cases; it improves accuracy and margin over alpha 0.
- Not established: edit-vector specificity. Random and unrelated controls also improve accuracy on this small set, though the edit vector has the strongest margin among the tested vectors.
- Not established: accepted SkillOpt self-evolution. The first candidate was rejected by selection gate; the vector smoke uses a rejected but SkillOpt-generated candidate edit.
- Next step: build a harder and less steering-generic evaluation split, preferably with held-out same-class tasks where random/unrelated vectors are tuned only on a validation subset, then rerun vector controls and full rollout smoke.

## 2026-07-06 - Skill-edit steering v2 same-class prototype

- Scope: ALFWorld `look_at_obj_in_light` only; filtered split at `benchmarks/skillopt/data/alfworld_path_split_look_light_v2`.
- SkillOpt reproduction on same-class selection generated edits but gate rejected: current `1/3`, candidate `1/3`; artifacts in `benchmarks/skillopt/outputs/skill_edit_v2_skillopt_light_gate_20260706b`.
- Manual old/new validation under Qwen3.5-4B vLLM: old valid_unseen `4/18 = 0.2222`, manual new valid_unseen `17/18 = 0.9444`.
- Added NPM-style extraction code for inter-trajectory and intra step-group vectors; extraction run `research/steering/runs/npm_skill_memory_light_l16_mean_diff_20260706` produced `inter_traj_vector.pt` and `intra_step_vector.pt`.
- HF rollout steering boundary: local `models/Qwen3-4B-Instruct-2507` does not reproduce the vLLM manual-new upper bound. Inter vector worsened repetition; intra vector reduced repeated actions from 65 to 57 on 6 old-skill HF episodes but did not improve success.
- Full summary: `research/steering/skill_edit_v2_summary_20260706.md`.

## 2026-07-06 - Qwen3.5 HF same-model steering follow-up

- HF Qwen3.5 load is possible using isolated transformers main at `.tmp/transformers_main_20260706`; release transformers in the existing env does not recognize `qwen3_5`.
- Same-model positive-control holds on Qwen3.5 HF: old skill `1/6`, manual new skill `3/6` on the first six `look_at_obj_in_light` valid_unseen tasks.
- Re-extracted Qwen3.5 HF vectors in `research/steering/runs/qwen35_npm_skill_memory_light_l16_mean_diff_6ep_20260706`.
- All-token inter steering is destructive (`0/6`, repeated actions `136`). Last-token inter steering is the best current setting (`1/6`, repeated actions `22` vs old baseline `26`) but does not improve success. Intra vectors do not improve success.
- Updated summary: `research/steering/skill_edit_v2_summary_20260706.md`.


## 2026-07-07 SkillOpt ALFWorld GPT-5.5 optimizer reproduction smoke/light

Question: whether the OpenAI-compatible GPT-5.5 endpoint can reproduce SkillOpt-style ALFWorld skill updates with Qwen3.5-4B as the target agent, after previous Qwen optimizer runs often failed to pass the gate.

Setup: benchmark repo `/data5/ninghan/tlm/benchmarks/skillopt`; optimizer `gpt-5.5` via `https://newapi.metamind.work/v1` with `openai_compatible` auth and no proxy; target `Qwen/Qwen3.5-4B` on local vLLM port 8007; seed skill `skillopt/envs/alfworld/skills/rough_v1.md`; raw env feedback, EFM/feedback augmentation disabled.

Evidence:
- API smoke succeeded before training: `chat_optimizer` using `openai_chat` returned valid JSON from `gpt-5.5`.
- Full update-chain smoke: `benchmarks/skillopt/outputs/skillopt_gpt55_raw_light_smoke_20260707_tmux1`. GPT-5.5 produced 2 edits, merge/rank/update applied them, candidate skill evaluated, but tied selection and was rejected: baseline/candidate selection hard `0.5 -> 0.5`.
- Low-budget light SkillOpt run: `benchmarks/skillopt/outputs/skillopt_gpt55_raw_light_lowbudget_20260707_1`. On `data/alfworld_path_split_look_light_v2`, baseline selection hard `0.3333`; step1 accepted to `0.6667`; step5 accepted to `1.0000`; final summary `steps=5 accept=2 reject=1 skip=2`, best step 5.
- Held-out light test18: `benchmarks/skillopt/outputs/eval_gpt55_raw_light_initial_vs_best_test18_20260707_2`. Initial rough skill hard `5/18 = 0.2778`; GPT-5.5 SkillOpt best hard `15/18 = 0.8333`.

Boundary: this is a light-task subset reproduction, not yet the full mixed ALFWorld split. It establishes that the GPT-5.5 optimizer path can pass the SkillOpt gate and transfer to held-out light tasks with the Qwen3.5-4B target.

## 2026-07-07 SkillOpt ALFWorld GPT-5.5 raw full-split low-budget reproduction

Question: can SkillOpt be reproduced with the paper-aligned role split (GPT-5.5 optimizer, frozen Qwen3.5-4B target) on ALFWorld without any EFM/intent feedback augmentation?

Setup: benchmark repo `/data5/ninghan/tlm/benchmarks/skillopt`; optimizer `gpt-5.5` via OpenAI-compatible endpoint; target `Qwen/Qwen3.5-4B` on local vLLM port 8007; raw environment feedback only (`feedback_enabled=false`); seed skill `skillopt/envs/alfworld/skills/rough_v1.md`; split `data/alfworld_path_split` with train=39, val=18, test=134. Low-budget settings: one epoch, train_size=8, batch_size=1, minibatch_size=1, edit_budget=2, constant LR, no slow/meta skill. This is a low-budget reproduction, not the full paper default run.

Training artifact: `benchmarks/skillopt/outputs/skillopt_gpt55_raw_full_lowbudget_20260707_1`.
- Baseline selection hard: `4/6 = 0.6667`.
- Step 1 accepted new best: selection hard `5/6 = 0.8333`; later steps were rejected or skipped.
- Final summary: `steps=8 accept=1 reject=5 skip=2`, best step 1, total wall time 3636.9s, total tokens 848,227.
- Best skill learned generic ALFWorld rules, not selection-specific memorization: target-object discipline and clean-then-place sequencing.

Held-out test32 artifact: `benchmarks/skillopt/outputs/eval_gpt55_raw_full_initial_vs_best_test32_20260707_1`.
- Initial rough skill: hard `16/32 = 0.5000`, mean turns `21.22`, success mean turns `12.44`.
- GPT-5.5 SkillOpt best: hard `22/32 = 0.6875`, mean turns `17.47`, success mean turns `11.77`.
- Delta: `+18.75` hard-success points and `-3.75` mean turns on the same held-out test32.

Boundary: this validates the raw SkillOpt reproduction path under a low-budget subset. It should not be conflated with the separate EFM/intent-aug research line; that line was stopped after noticing the scope mix-up.


## 2026-07-07 SkillOpt ALFWorld GPT-5.5 raw full-split paper-ish gate run

Question: can the paper-aligned SkillOpt role split reproduce accepted skill self-evolution on the mixed ALFWorld split when the optimizer is GPT-5.5 and the frozen executor is Qwen3.5-4B, without EFM or intent feedback augmentation?

Setup: repo `/data5/ninghan/tlm/benchmarks/skillopt`; run artifact `benchmarks/skillopt/outputs/skillopt_gpt55_raw_full_paperish_gate_w2_20260707_1`; optimizer `gpt-5.5` via OpenAI-compatible endpoint; target `Qwen/Qwen3.5-4B` on vLLM `127.0.0.1:8007`; raw feedback only (`feedback_enabled=false`); split `data/alfworld_path_split`; seed skill `skillopt/envs/alfworld/skills/rough_v1.md`; train_size=16, batch_size=4, 2 epochs, 8 total update steps, workers=2, max_steps=30, `use_gate=true`, `use_slow_update=false`, `use_meta_skill=true`.

Training result:
- Selection baseline: `5/12 = 0.4167`.
- Accepted steps: step1 `0.5000`, step2 `0.5833`, step8 `0.6667`.
- Rejected steps: step3 tied current at `0.5833` and was rejected; steps4/5 had high train-rollout hard `0.75` but lower selection `0.5000` and were rejected; steps6/7 rejected. This confirms the selection gate was active.
- Best skill: `benchmarks/skillopt/outputs/skillopt_gpt55_raw_full_paperish_gate_w2_20260707_1/best_skill.md` (generic ALFWorld search/action rules, not split-id memorization).
- Meta-skill epoch2: current epoch memory rollout `11/20 = 0.55` vs previous `8/20 = 0.40`; meta result written to `meta_skill/epoch_02/meta_skill_result.json`.

Held-out test32 eval on `valid_unseen` / `split=test`, same 32 tasks and raw feedback:
- Initial rough skill: `benchmarks/skillopt/outputs/eval_gpt55_raw_initial_test32_20260707`, hard `15/32 = 0.4688`, mean turns `21.56`, success mean turns `12.00`.
- SkillOpt step8 best: `benchmarks/skillopt/outputs/eval_gpt55_raw_best_step8_test32_20260707`, hard `21/32 = 0.6562`, mean turns `16.22`, success mean turns `9.00`.
- Delta: `+6/32 = +18.75` hard-success points, `-5.34` mean turns, `-3.00` success mean turns.
- By task type: `look_at_obj_in_light` improved `7/18 -> 13/18` with mean turns `23.61 -> 14.28`; `pick_and_place` stayed `8/14 -> 8/14` with mean turns `18.93 -> 18.71`.

Sanity checks and caveats:
- Grep over config/log/eval artifacts found no EFM, intent augmentation, `feedback_summary`, `intention_status`, or `execution_diagnosis` markers.
- The log header prints `[slow update] acceptance=force-accept (unconditional)` even though `slow_update: False`; actual step records reject tied/worse candidates, so this appears to be a misleading stale log line rather than force-accept behavior.
- A 4-worker route was avoided after previous worker-boundary stalls; this stable reproduction used `ALFWORLD_WORKER_START_METHOD=spawn` and `workers=2`.

Boundary: this is still low-budget relative to a full paper-scale benchmark, but it is a successful raw SkillOpt reproduction on the mixed ALFWorld split: GPT-5.5 optimizer passes the gate, produces generic skill updates, and transfers to a held-out valid_unseen test32 subset with both higher success and fewer turns.

## 2026-07-08 SkillOpt accepted-edit steering vector extraction smoke

Question: whether atomic SkillOpt accepted edits can be represented as stable activation directions before any downstream steering/vector composition experiment.

Setup:
- Base run: `benchmarks/skillopt/outputs/skillopt_gpt55_raw_full_paperish_gate_w2_20260707_1`.
- Atomic edit pairs materialized under `research/steering/runs/skill_edit_suite_gpt55_20260708/skills`.
- Evaluated three prioritized edits with S-/S+ counterfactual rollouts:
  - `e_light_hold_then_lamp`: S-=`skill_v0001`, S+=S- plus the step2 light-task replacement.
  - `e_pickup_precise`: S-=`skill_v0002`, S+=S- plus the step8 precise pickup/hands-full rule.
  - `e_cool_fridge`: S-=`skill_v0002`, S+=S- plus the step8 cooling recipe.
- Target model: `models/Qwen3.5-4B`, HF rollout, layer 16, last-token hidden-state pooling.
- Case sources:
  - light: `research/steering/runs/skill_edit_suite_gpt55_20260708/cases/light_test32_first8.jsonl`
  - pickup: `research/steering/runs/skill_edit_suite_gpt55_20260708/cases/pick_place_test32_first8.jsonl`
  - cool: `research/steering/runs/skill_edit_suite_gpt55_20260708/cases/cool_valid_unseen_first8.jsonl`
- GPT-5.5 node labeler produced intra node pairs in `research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts_merged/llm_node_labels.jsonl`.

Rollout observations:
- `e_light_hold_then_lamp`: S- 3/8, mean turns 22.375; S+ 3/8, mean turns 20.625. Same success count but slightly fewer turns.
- `e_pickup_precise`: S- 4/8, mean turns 16.5; S+ 3/8, mean turns 17.125. No execution improvement in this case subset.
- `e_cool_fridge`: S- 1/8, mean turns 23.625; S+ 1/8, mean turns 23.875. One case was repaired by S+, one prior success regressed.

Vector extraction artifacts:
- Merged paired rollouts: `research/steering/runs/skill_edit_suite_gpt55_20260708/paired_rollouts_merged/paired_rollouts.jsonl`.
- Vectors and figures: `research/steering/runs/skill_edit_suite_gpt55_20260708/vectors_l16_last`.
- Instance vectors: 59 total = 24 inter + 35 intra.
- Intra labels: light 19 pairs, pickup 0 pairs, cool 16 pairs.
- Mean pairwise cosine within instance sets:
  - light inter -0.0108, light intra 0.0523
  - pickup inter 0.0415
  - cool inter 0.0432, cool intra 0.0121
- Pooled cosine highlights:
  - light inter vs light intra: -0.010
  - cool inter vs cool intra: 0.155
  - light inter vs cool inter: -0.208

Evidence boundary:
- The pipeline now works end-to-end for atomic SkillOpt edit -> paired rollout -> GPT-5.5 node labels -> inter/intra vectors -> PCA/cosine visualization.
- This first last-token/layer16 extraction does not show strong edit-level clustering. The result weakens the assumption that a natural-language skill edit automatically maps to a clean single direction under naive trajectory/node mean-difference pooling.
- Pickup generated no reliable GPT-5.5 intra pairs on the chosen pick-and-place subset, so it currently only has an inter trajectory vector.

Next technical decision:
- Improve extraction before steering intervention: try mean-token pooling and/or aligned decision-state prompts; restrict node pairs to same semantic phase; consider filtering inter pairs to cases where S+ changes outcome or key behavior.

### Correction: performance-delta hidden-state PCA for SkillOpt initial -> best

Follow-up to the 2026-07-08 skill-edit vector extraction smoke. The previous atomic-edit S-/S+ sampling used fixed first-8 task subsets and included many neutral/regressive pairs, so positive/negative hidden states were not expected to separate well. This was a sampling/design error for the representation-probe stage.

Corrected probe:
- Reused existing `eval_gpt55_raw_full_initial_vs_best_test32_20260707_1` artifacts; no new ALFWorld rollout.
- Positive examples: trajectories from the better skill (`best`) that succeeded.
- Negative examples: trajectories from the earlier skill (`initial`) that failed.
- Also included a stricter repaired-pair subset: same task id where `initial` failed and `best` succeeded.
- Model/layer: `models/Qwen3.5-4B`, layer 18, last-token hidden state over rendered trajectory text.
- Script: `research/steering/skill_edit/run_performance_delta_hidden_pca.py`.
- Output: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/initial_best_pos_neg_pca_layer18.png`.

Counts and separation:
- repaired_all: pos=8, neg=8, centroid cosine 0.7784
- repaired_light: pos=6, neg=6, centroid cosine 0.7579
- all_success_vs_failure: pos=22, neg=16, centroid cosine 0.7655
- light_success_vs_failure: pos=11, neg=12, centroid cosine 0.7449

Evidence boundary:
- This corrected sampling produces much clearer pos/neg PCA separation than the atomic-edit first-8 sampling, whose centroid cosines were around 0.98.
- It supports using performance-delta-selected trajectories as the first steering-vector extraction source before trying narrower atomic edit vectors.

### Intra visualization for corrected performance-delta repaired pairs

Follow-up after approving intra visualization for the corrected `initial fail -> best success` same-game pairs.

Setup:
- Built paired data from existing `eval_gpt55_raw_full_initial_vs_best_test32_20260707_1` artifacts.
- Pairs: 8 repaired same-game pairs, 6 `look_at_obj_in_light` and 2 `pick_and_place`.
- Pair file: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/intra/repaired_paired_rollouts.jsonl`.
- GPT-5.5 node labeler: `repaired_intra_node_labels.jsonl`.
- Labels: 32 intra node pairs, mean confidence 0.8738, min confidence 0.72.
- Hidden states: Qwen3.5-4B, layer 18, last-token pooling.

Artifacts:
- Intra pos/neg hidden-state PCA: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/intra/repaired_intra_pos_neg_hidden_pca_layer18.png`.
- Inter/intra delta vector PCA: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/intra/vectors_l18_last/figures/pca_skill_edit_instances.png`.
- Extracted vectors: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/intra/vectors_l18_last/vectors`.

Results:
- Intra hidden-state pos/neg centroid cosine: 0.9936, so node-level positive and negative states do not separate well under this rendering/pooling.
- Inter delta vectors for the 8 repaired full trajectories cluster much better: mean pairwise cosine 0.5876.
- Intra delta vectors are scattered: mean pairwise cosine 0.0287.

Interpretation:
- Corrected performance-delta sampling strongly improves inter trajectory-level separation.
- The current GPT-5.5 node-pair based intra representation is not yet clean. It may need better semantic alignment, stricter phase-specific node pairing, action-decision-only rendering, or different pooling/layer choices before using it for steering.

### Raw-trajectory-only inter hidden-state visualization

Question:
- Does inter positive/negative separation remain after removing explicit skill and outcome labels from the representation input?

Setup:
- Input contains only `Task`, followed by alternating rollout `Action` and environment `State` text.
- Explicitly excluded: skill text/version, plus/minus side, outcome, turns, reward, done, and valid.
- The old artifacts do not store the reset-time initial observation, so the sequence starts at the first action and its resulting state.
- Hidden states: Qwen3.5-4B, layer 18, last-token pooling.
- Positive trajectories: successful best-skill rollouts. Negative trajectories: failed initial-skill rollouts.

Results:
- Repaired exact-task pairs: pos=8, neg=8, centroid cosine=0.8997.
- Repaired light subset: pos=6, neg=6, centroid cosine=0.8934.
- All success vs failure: pos=22, neg=16, centroid cosine=0.8465.
- Light success vs failure: pos=11, neg=12, centroid cosine=0.8583.
- Figure: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/raw_trajectory_inter_all_l18/initial_best_raw_trajectory_inter_pca_layer18.png`.
- Metrics: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/raw_trajectory_inter_all_l18/initial_best_raw_trajectory_inter_pca_layer18_metrics.json`.

Evidence boundary:
- Separation remains without explicit labels and is visually clearer for the larger success/failure sets.
- This establishes that raw trajectory content contains a success/failure-related inter direction. It does not yet isolate the causal contribution of a specific skill edit.

### Inter ablation: add outcome, reward, and done

Setup:
- Same samples, Qwen3.5-4B, layer 18, and last-token pooling as the raw-trajectory-only run.
- Added overall `Outcome` and per-step `Reward` / `Done`.
- Still excluded skill text/version, plus/minus side, turns, valid, and task type.

Results:
- Repaired exact-task pairs: centroid cosine 0.8997 -> 0.8222.
- Repaired light subset: 0.8934 -> 0.8239.
- All success vs failure: 0.8465 -> 0.7921.
- Light success vs failure: 0.8583 -> 0.8091.
- Lower centroid cosine and the PCA plots both show stronger positive/negative separation.
- Figure: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/raw_plus_outcome_inter_l18/initial_best_raw_plus_outcome_inter_pca_layer18.png`.

Interpretation:
- Explicit outcome signals materially strengthen clustering.
- Because last-token pooling is close to the final `Done` field, this ablation measures result-state encoding more than a pure skill-induced behavioral direction.

Visualization update:
- Replotted both inter ablations from the saved hidden-state tensors; no model forward pass or representation values changed.
- Added compact 2x2 panels, larger typography, semantic subtitles, legends, grid lines, and per-panel centroid cosine.
- Raw-only polished figure: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/raw_trajectory_inter_all_l18/initial_best_raw_trajectory_inter_pca_polished_layer18.png`.
- Raw plus outcome-signals polished figure: `research/steering/runs/skill_edit_suite_gpt55_20260708/performance_delta_clusters/raw_plus_outcome_inter_l18/initial_best_raw_plus_outcome_inter_pca_polished_layer18.png`.

### Step-token mean INTRA: paper-style and atomic skill-edit contrasts

Question:
- What changes when both intra methods encode a raw trajectory and mean-pool only the target decision tokens?

Implementation:
- Shared extractor: `research/steering/skill_edit/intra_step_mean.py`.
- Dual-mode runner: `research/steering/skill_edit/run_intra_step_mean.py`.
- Representation: task plus raw action/state trajectory; current action token span is mean-pooled. This common span is available in both paper-style and atomic traces, while saved reasoning is not.
- Skill, side, outcome, reward, done, judge labels, and post-action feedback are excluded from the pooled current-step span.
- Span audits are saved beside each run.

Paper-style results:
- Source: 16 failed initial-skill test32 trajectories; 14 contained both effective and degenerate steps.
- Positive effective steps: 354; negative degenerate steps: 66.
- Positive/negative centroid cosine: 0.9673.
- 14 trajectory-level contrast vectors mean pairwise cosine: 0.3020.
- Within-task contrast cosine: light 0.3807; pick-and-place 0.0981.
- Artifact: `research/steering/runs/intra_step_mean_20260710/paper_style`.
- The strong separation shown by the paper was not reproduced on these Qwen3.5-4B SkillOpt traces under the available deterministic step rules.

Atomic skill-edit strict results:
- Strict filter: S+ improves success, or preserves success with fewer turns.
- Used 3 game/edit pairs and 8 GPT-5.5-aligned node pairs: 2 light games, 1 cool game.
- Positive/negative centroid cosine: 0.9410.
- Three game-level contrast vectors mean pairwise cosine: 0.0079.
- The two light vectors have cosine -0.0369; one cool vector is insufficient for a within-edit consistency metric.
- Artifact: `research/steering/runs/intra_step_mean_20260710/skill_edit_strict`.

Atomic skill-edit diagnostic results:
- Without requiring outcome/turn improvement: 9 game/edit pairs and 24 aligned node pairs.
- Positive/negative centroid cosine: 0.9760.
- Game-level contrast mean pairwise cosine: 0.0279.
- Within-edit cosine: light -0.0125; cool 0.0181.
- Artifact: `research/steering/runs/intra_step_mean_20260710/skill_edit_all_labeled`.

Evidence boundary:
- Correct step-token pooling improves the methodological validity but does not create a stable edit direction in the current sample.
- The strict skill-edit sample is too small for a positive clustering claim, while the larger diagnostic set shows near-zero consistency.
- Existing traces omit reset-time initial observations; the extractor records this limitation rather than fabricating them.
- The optimizer endpoint model-list check returned HTTP 401 for both historical credentials, so GPT-5.6 availability could not be verified and the existing GPT-5.5 labels were retained.

### Atomic skill text to INTRA steering alignment

Question:
- Do the behavioral `S+ step - S- step` vectors align specifically with the text edit that produced the paired rollouts?

Method:
- Added `research/steering/skill_edit/run_intra_text_alignment.py`.
- Behavioral direction: equal-weight mean of game-level intra deltas for each atomic edit.
- Text representations: (1) mean hidden state of the atomic edit text and (2) mean-hidden full-skill difference `plus.md - minus.md`.
- All representations use Qwen3.5-4B layer 18; alignment is cosine similarity.
- Cross-edit matrices test whether matching diagonal entries exceed mismatched off-diagonal entries.

Pair audit:
- Strict set has 3/2/3 aligned nodes in the three retained games, not one node per trajectory.
- Light pairs mostly follow the edit's pickup/carry/use-lamp chain.
- In the cool game, only the fridge-protocol pair is a direct edit primitive; search efficiency and final placement are upstream/downstream consequences.

Strict-set alignment:
- Light: 2 game vectors; within-edit and leave-one-game-out cosine both -0.0369.
- Cool: 1 game vector, so consistency cannot be estimated.
- Matching edit-text-to-behavior cosine: cool 0.0045; light 0.0295.
- Matching full-skill-delta-to-behavior cosine: cool -0.0003; light 0.0440.
- Edit-text diagonal advantage over mismatched entries: 0.0194.
- Full-skill-delta diagonal advantage: -0.0005.

All-labeled diagnostic alignment:
- Light: 4 game vectors, within-edit cosine -0.0125, leave-one-game-out cosine 0.0503.
- Cool: 5 game vectors, within-edit cosine 0.0181, leave-one-game-out cosine 0.0798.
- Matching edit-text-to-behavior cosine: cool 0.0795; light -0.0564.
- Matching full-skill-delta-to-behavior cosine: cool -0.0729; light -0.0591.
- Edit-text diagonal advantage: -0.0183; full-skill-delta diagonal advantage: -0.0911.
- Artifact: `research/steering/runs/intra_step_mean_20260710/skill_edit_all_labeled/alignment/text_steering_alignment.png`.

Interpretation:
- Positive/negative endpoint overlap alone does not invalidate difference vectors, but the current deltas also lack within-edit directional consistency and text-edit specificity.
- The present evidence does not support a text-skill-to-intra-vector correspondence. More strict repaired games and phase-pure node groups are required before intervention tests can be meaningfully interpreted.

## 2026-07-12 Phase 1 Skill Vectorization Fidelity harness and 6-episode smoke

Question:
- Can the causal behavior difference between no skill and the seed42 best SkillOpt skill be isolated at deterministic first-fork states with a 2x2 teacher-forced action-span contrast, then partially reproduced by activation steering?

Scope and implementation:
- Targeted the three test60 classes where seed42 best improved over initial at the full horizon: `look_at_obj_in_light` (5/10 -> 9/10), `pick_and_place` (6/10 -> 7/10), and `pick_two_obj_and_place` (5/10 -> 6/10).
- Added `research/steering/skill_edit/phase1_fidelity.py` and persisted the pre-action prompt in HF rollout traces.
- Greedy no-skill / best-skill pairs use the same local `models/Qwen3.5-4B`; only the first action fork is retained, and the pre-action prompt must match exactly.
- Each node uses the action-span context main effect: `0.5 * [(h(skill,y+) - h(no_skill,y+)) + (h(skill,y-) - h(no_skill,y-))]`, extracted for all 32 layers.
- Extraction and validation are split within each task type. Candidate middle layers are ranked by node-vector pairwise cosine, then checked at held-out fork states.
- Focused regression: `tests/research/steering -q` -> `17 passed`.

Smoke artifact:
- `research/steering/runs/phase1_fidelity_best42_6ep_20260712`.
- Six paired episodes, two per target class, 10-step collection horizon. No skill succeeded 0/6; best skill succeeded 1/6. The repaired case was `test:pick_and_place_simple:0008`, where no skill timed out and best finished in 7 steps.
- All 6 pairs produced a first fork and all 6 fork prompts matched exactly.
- Three stratified extraction nodes gave best middle-layer consistency at layer 10 (mean pairwise cosine 0.4375), followed by layer 14 (0.4103).
- Held-out fork certification at alpha <=2 recovered 0/3 target actions. At layer 10, alpha 3 recovered 1/3: the repaired mug-to-desk case changed from no-skill `look` to the exact best-skill action `go to desk 1`; the other two nodes preserved their no-skill actions and produced no unrelated action. Layer 11 alpha 8 also recovered that same node but changed another node, so it is less clean.
- Larger alpha values generally increased unrelated action changes and did not improve overall recovery.

Evidence boundary and next decision:
- The technical route is now end-to-end and produced one exact held-out fork recovery on the task-level repaired case. This is a proof-of-route, not evidence of a stable cross-task skill vector: extraction n=3 and validation n=3 are too small, and only one task class recovered.
- The next run should expand state-matched first-fork nodes within all three improved classes, retain stratified held-out certification, and compare global versus task-type vectors before adding secondary counterfactual probes.

## 2026-07-14 Clean SkillOpt ALFWorld reproduction from minimal initial skill

Question:
- Can upstream SkillOpt evolve a deliberately minimal ALFWorld skill into a skill text that improves Qwen3.5-4B success on a fully held-out `valid_unseen` set?

Protocol:
- Code base reset to upstream `microsoft/SkillOpt` commit `fc1f827` before setup.
- The only intended research-variable change is the minimal 209-character `skillopt/envs/alfworld/skills/rough_v1.md` initial skill.
- Upstream SkillOpt prompts, gradient, optimizer, gate, slow update, and update policy are unchanged.
- Two tested runtime bug fixes only: explicitly transmit Qwen `enable_thinking=false`, and resolve released relative ALFWorld gamefile paths against `ALFWORLD_DATA`; focused tests: `7 passed`.
- Optimizer: `gpt-5.5`; target: `Qwen/Qwen3.5-4B`; target temperature `0.7`; thinking disabled.
- Upstream critical settings: `num_epochs=4`, `batch_size=40`, `minibatch_size=8`, `merge_batch_size=8`, patch update, cosine edit budget 4 to 2, gate enabled, slow update enabled, meta-skill enabled.
- ALFWorld horizon: `max_steps=50` consistently for train, selection, and test.
- Runtime-only worker setting: spawn with `workers=3`, after the older 4-worker boundary had reproduced deadlocks.
- Data: released train manifest 39; full official `valid_seen` 140 for selection; full official `valid_unseen` 134 for final test.
- Core success criterion: `hard(best_skill, test134) - hard(rough_v1, test134) > 0`.

Preflight evidence:
- Both local Qwen endpoints listed `Qwen/Qwen3.5-4B`; authenticated target chat returned successfully.
- `gpt-5.5` optimizer chat returned `OK`.
- A clean-backend request without explicit `enable_thinking=false` spent its completion budget on reasoning and returned no content, directly motivating the one-line tested Qwen fix.
- The first path smoke failed before any model call because the released relative manifest path was interpreted from the repository cwd; the minimal resolver fix moved the second smoke into real three-episode model/environment execution.

Live run:
- tmux: `skillopt_clean_repro_seed42_20260714`
- output: `benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714`
- launch state: active, computing the 140-item minimal-skill selection baseline.
- Monitoring rule: inspect each completed selection round; stop and report on runtime/service failure or persistently very poor candidates with no improvement over baseline.

## 2026-07-17 Latent space and steering vectors from the clean SkillOpt repro (seed42)

Question:
- Do the seed42 SkillOpt skill improvements (rough_v1 41.43% -> step-1 52.14% -> step-2 62.14% on valid_seen140)
  correspond to identifiable directions in Qwen3.5-4B latent space, and can a reliable steering vector be
  extracted from (a) NPM-style success/failure contrasts and (b) skill-driven paired rollouts?

Data and infrastructure:
- Same 140 valid_seen tasks under three skills: selection_eval_baseline (v0000), step_0001 and step_0002
  selection evals from `benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714`.
  Pair categories v0000 vs step2: repaired 34, broken 5, both_success 53, both_fail 48.
- Behavior-only serialization (task + action/state trace, no skill text, no reward/done markers), exact
  action-token spans, mean pooling; all 32 layers trajectory-level, 8 layers step-level, fp16.
- Exact env replay of all 140 v0000 episodes recovered per-step prompts: 1188/1188 feedback matches, 0 errors,
  1213 states (steps 0..8).
- Prompt-conditioned contrast: each state forwarded under three prompts (bad = rough_v1, good = step-2
  candidate text, none), last-token hidden at all 32 layers. best_skill.md had already been overwritten by the
  accepted step-3 candidate (gate 0.65), so the step-2 candidate file was used deliberately to match the
  behavioral rollouts.
- Scripts: `research/steering/latent/` (README there documents the pipeline); run dir:
  `research/steering/runs/latent_skillopt_repro42_20260717` (manifest, reps/, prompt_deltas.pt, vectors/,
  analysis/, steered_eval/).
- Envs: `skillopt-qwen35-vllm` (transformers 5.12) is the only env that loads Qwen3.5-4B for HF forwards;
  `-cu128` (transformers 4.57) does not know `qwen3_5` and was used for env replay only.

Latent-space findings:
1. Length confound dominates naive inter contrasts: full-trajectory success/fail probes reach 0.96-0.97
   (best layers 13-18), but an n_steps-only baseline gives 0.95-0.97 because failures are 50-step timeouts.
   The clean PCA/t-SNE success/fail clusters encode outcome/length, not behavior quality. Cross-condition
   probe transfer 0.95-0.98 for the same reason. (analysis/inter_layer_sweep.png, inter_pca_tsne.png)
2. Early-window (first-5-step) outcome predictability is far weaker: 0.53-0.67 balanced accuracy.
3. Repaired-pair full-trajectory deltas look consistent (pairwise cos 0.52 @L14 vs shuffled null 0.22) but
   align at cos 0.89 with the success/fail direction - mostly the same outcome/length signal.
4. Under length control (early5 and matched-prefix k<=10 reps) repaired-specific delta consistency collapses
   to the shuffled-null level (~0.15-0.20): no task-specific "repair direction" survives.
   (analysis/controlled_delta_sweep.png)
5. The skill condition itself IS linearly identifiable from early behavior: task-grouped probe v0000-vs-step2
   on early5 reps reaches 0.83 (L10-22); v0000 occupies a distinct region of the combined map while
   step1/step2 overlap heavily. The skill shift is a global condition signature, not per-task directions.
   (analysis/combined_condition_map.png)
6. NPM-style intra (effective vs degenerate steps in failed v0000 rollouts, 1338 vs 2762): probe 0.97-0.98
   at L10-18; strong but plausibly partly lexical (repeated action text, "Nothing happens" context).
7. The prompt-conditioned skill contrast is the cleanest object: h(good)-h(bad) on identical states has
   cross-state pairwise cosine 0.98 (L0-2), 0.86 (L14), 0.76 (L18), 0.63 (L31); uniform across pair
   categories and step indices. At L14 the centroid shift is 2.3x the within-condition state spread and the
   three prompt conditions form fully separated clusters. (analysis/prompt_space_l14.png)
8. Vector-family geometry at mid depth: prompt(good-bad) vs behavioral early5 delta cos ~0.03-0.07 (near
   orthogonal; different token positions), prompt(good-bad) vs inter ~0.0, behavioral early5 vs inter
   -0.2..-0.5, prompt(good-bad) vs prompt(good-none) ~0.6, additivity gmb vs (gmn - bmn) = 1.0.
   Better-skill conditioning, induced behavior shift, and outcome/length are three distinct axes.
   (analysis/vector_alignment.png)

Steering vectors saved: `vectors/` - families v_prompt_good_minus_bad, v_prompt_good_minus_none,
v_behav_early5_all/repaired, v_inter_full at L{1,10,14,18,22} plus all-layer bundles and
gmb_raw_means.pt (per-layer raw mean deltas for alpha calibration; ||mean delta||: L14 2.3, L18 3.5).

Causal evaluation (HF greedy, skillopt-faithful system/user prompts, max_steps 35; matched task set
12 repaired + 6 both_fail unless noted):
- bad skill (rough_v1): 0/18. good skill (step-2 text): 7/18 (7/12 repaired) - the text-skill effect
  reproduces under HF greedy.
- bad + v_prompt_good_minus_bad L18 alpha 8: 1/18 (1/12 repaired). L14 alpha 2.5/5: 0/8 each.
  L18 alpha 12: 0/8 with generation collapse (invalid-action rate 0.97). Random unit vector at L18 alpha 8:
  0/8. Multi-layer injection of raw per-layer mean deltas (L6-22, alpha 1 or 1.5) destroys generation
  outright (gibberish) - per-layer deltas compound through depth and cannot be injected jointly.
- Graded behavior (action-set Jaccard vs the vllm good/bad reference rollouts; repeat rate):
  bad_base 0.263/0.561, repeat 0.62; steered L18 a8 0.267/0.409, repeat 0.47, invalid 0.07->0.23;
  random 0.223/0.540, repeat 0.59 (= baseline); good_base 0.538/0.372, repeat 0.40.
  The gmb L18 vector produces a specific, dose-sensitive shift: it pushes behavior away from the bad-skill
  pattern and reduces action perseveration (random vector does neither), but rarely lands on the good-skill
  behavior needed for task success, and it costs invalid actions.

Interpretation:
- For steering-vector research on this stack, the reliable extractable object is the global
  "better-skill conditioning" direction (prompt-conditioned contrast), not per-task repair directions and
  not the naive success/fail direction (length artifact).
- Single-layer injection of that direction is causally active and specific but under-dosed at safe alphas
  and destructive past ~2-3x the layer's natural delta norm; it transfers the "stop perseverating" component
  of the skill but not the full procedural content.

Evidence boundary:
- One model (Qwen3.5-4B), one SkillOpt run (seed42), 140 valid_seen tasks; causal arms are small
  (8-18 episodes, greedy only, max_steps 35 vs 50 in the original run).
- Inter/intra separations are descriptive; without the length/lexical controls they must not be read as
  quality directions.
- The prompt-conditioned vector is defined by two specific skill texts; generality across other skill pairs,
  steps, seeds, or sampling temperatures is untested.
- The 1/12 repaired flip at L18 alpha 8 with 0/12 baseline and 0/8 random control is directionally
  encouraging but not statistically meaningful on its own; the specific behavioral shift (Jaccard-away-from-bad,
  repeat-rate drop, random control flat) is the robust causal finding.

Next steps that would add the most information:
1. Token-slice and step-position ablations (steer only generated tokens vs all positions; only early steps).
2. Projection-based injection (remove the bad-skill component along v instead of adding a constant) and
   per-state adaptive alpha.
3. Repeat the contrast with the step-3 skill text (9835 bytes, gate 0.65) and across seeds to test
   generality of the conditioning direction.
4. Temperature-0.7 multi-sample causal arms to match the original rollout distribution.

## 2026-07-18 Clean SkillOpt reproduction final result (seed42)

Question and success criterion:
- Reproduce upstream SkillOpt on ALFWorld to obtain a high-quality skill text whose full valid_unseen134 hard score exceeds the minimal rough_v1 initial skill.
- Core criterion: hard(best_skill, valid_unseen134) - hard(rough_v1, valid_unseen134) > 0.

Formal setting:
- Upstream base: microsoft/SkillOpt fc1f827; only tested Qwen thinking=false and ALFWorld relative-path fixes, plus a runtime-only dual-shard scheduler.
- Optimizer gpt-5.5; target Qwen/Qwen3.5-4B; temperature 0.7; max_steps=50.
- rough_v1 initial skill; train39; full valid_seen140 selection; full valid_unseen134 test; 4 optimization steps.
- Runtime after baseline checkpoint: two independent ALFWorld managers, 8 workers each, pinned to Qwen endpoints 8007 and 8008. Search, prompts, scoring, sampling, data and optimizer settings were unchanged.
- Output: benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714.

Selection trajectory on full valid_seen140:
- rough_v1: 58/140 = 0.414286.
- Step 1: 73/140 = 0.521429, accepted.
- Step 2: 87/140 = 0.621429, accepted.
- Step 3: 91/140 = 0.650000, accepted.
- Step 4 patch candidate: 87/140 = 0.621429, rejected.
- Epoch-4 slow-update candidate: 95/140 = 0.678571 in final_selection_eval, promoted to formal best.

Final full valid_unseen134 result:
- rough_v1 baseline: 49/134 = 0.365672.
- promoted best skill: 100/134 = 0.746269.
- Absolute improvement: +51/134 = +0.380597 (+38.06 percentage points); relative solved-task increase 2.04x.
- Best per task: pick_and_place 19/24=0.7917; pick_two_obj_and_place 9/17=0.5294; look_at_obj_in_light 18/18=1.0000; heat 13/23=0.5652; cool 16/21=0.7619; clean 25/31=0.8065.
- Baseline per task: pick_and_place 15/24=0.6250; pick_two_obj_and_place 9/17=0.5294; look_at_obj_in_light 7/18=0.3889; heat 2/23=0.0870; cool 3/21=0.1429; clean 13/31=0.4194.

Artifact verification:
- best_skill.md, slow_update/epoch_04/candidate_skill.md and skills/skill_v0004.md are byte-identical (SHA256 88cc59c8962f7477...).
- Full test artifacts: test_eval_baseline/results.jsonl and summary.json; test_eval/results.jsonl and summary.json, each with exactly 134 items.
- Steering-relevant raw proposals, merged/ranked edits, edit apply reports, rejected step-4 candidate, slow-update comparison pairs and all skill versions remain preserved under the same output root.

Runtime recoveries and evidence boundary:
- Step-1 GPT-5.5 analyst 401 was caused by launcher key-name mapping; fixed by mapping OPTIMIZER_AZURE_OPENAI_API_KEY and reusing the completed rollout.
- Step-3 GPT endpoint connection failure was recovered by preserving and reusing the completed 39/39 rollout; no baseline or completed rollout was recomputed.
- Dual-shard scheduling changed throughput only and kept each ALFWorld manager at the proven 8-worker boundary.
- This establishes a large text-skill performance gain for one seed, one target model and ALFWorld. It does not establish cross-seed or cross-model generality.

Decision:
- Reproduction succeeds. Use the promoted epoch-4 slow-update best_skill.md as the high-quality skill-text foundation for subsequent steering-vector work.

## 2026-07-18 Round 2/3: gmn injection, gen-only steering, cross-text generality, temp-0.7 check

Follow-up to the 2026-07-17 entry; same run dir `research/steering/runs/latent_skillopt_repro42_20260717`.
Motivating user insights: (a) prefer skill-vs-none contrasts over good-vs-bad, (b) keep excluding verbal/length
confounds, (c) analysis must serve the final rollout numbers.

New machinery:
- `steered_eval.py`: `--gen-only` (steer only decode steps via seq_len==1 hook guard, prompt encoding
  untouched), `--temperature/--sample-tag` (seeded sampling arms).
- `prompt_forward.py`: `--good-skill-path/--out-name`; `analyze_generality.py` compares delta directions
  across skill texts. gmn_raw_means.pt added (gmn mean||delta||: L14 3.83, L18 5.87).
- GPUs 0/3 shared with another user's SFT job this round (their 2x ~10 GB; no interference observed).

Cross-text generality (strongest analysis result):
- gmb(step-2 text) vs gmb(step-3 text) on identical 1213 states: mid-layer (10-22) mean cosine 0.948;
  gmn: 0.970; bad-minus-none identity control 1.000 (forward determinism confirmed).
- The skill-conditioning direction is text-independent across the two evolved texts (5.8 KB vs 9.8 KB).
  Caveat: step-3 evolved from step-2 (related content); an unrelated handcrafted skill text is the missing
  control. (analysis/generality_step2_vs_step3.png, generality_metrics.json)

Causal arms (HF, max_steps 35; repaired = first 8/12 sorted; greedy unless noted):
| arm | successes | notes |
|---|---|---|
| bad_base | 0/12 rep, 0/6 bf | repeat 0.62 |
| good text | 7/12 rep, 0/6 bf | reference |
| none_base | 2/12 rep (0025, 0037) | repeat 0.40 - rough_v1 is actively harmful vs no skill (induces perseveration) |
| full-steer gmb L18 a8 | 1/12 (0025) | round 1 |
| gen-only gmb L18 a8 | 2/8 (0015, 0025) | invalid 0.15; jac_good 0.34 |
| gen-only gmb L18 a12 | 2/8 (0015, 0024) | works where full-position a12 collapsed; invalid 0.47 |
| gen-only random a8 | 0/8 | behavior ~= baseline -> flips are vector-specific |
| gen-only gmb L18 a16 | 0/8 | collapse (invalid 0.88) |
| none + gmn L18 a6 | 1/12 rep + 1/6 bf | no net gain (2/18 vs 2/18) |
| none + gmn L18 a10 | 2/8 (0015, 0025) | toward-good movement, invalid 0.41 |
| none + gmn L14 a7.5 | 0/8 | L14 collapses at 2x norm; L18 is the robust layer |
| bad t0.7 s1/s2 | 1/12, 2/12 | |
| gen-only a8 t0.7 s1/s2 | 1/12, 0/12 | steering gain not detectable under sampling (n=24) |

Findings:
1. Gen-only injection strictly dominates full-position injection at matched dose (2/8 vs 1/12 flips, lower
   invalid rate, larger behavioral movement) and extends the usable dose range (a12 functional vs collapsed).
   Interpretation: steering the prompt encoding corrupts the state/action representations the policy must
   read; steering only the generation stream conveys the conditioning signal at lower distortion.
2. Vector-specific flips concentrate on tasks the good text also repairs: val:0015 flips at both gen-only
   doses and under none+gmn a10; val:0024 at a12. val:0025 is a "restore-none-behavior" case (none_base also
   solves it; the bad skill breaks it). Random control: 0 flips, no behavioral movement.
3. none+gmn does not beat none_base on success despite moving behavior metrics; with rough_v1 being harmful,
   the practically relevant axis on this run is bad->good, and "remove the bad-skill damage" (perseveration)
   is what the vector reliably transfers.
4. Temperature 0.7 drowns the effect at this sample size; greedy is the operating regime for these
   single-vector interventions.

Evidence boundary:
- Repaired-subset ns remain 8-12 per arm; flip counts are 1-2. The specificity claim rests on the random
  control (0/8, flat behavior) and the good-task alignment of flips, not on statistical power.
- temp-0.7 null is n=24 over 2 seeds; a real sampling-regime test needs 5+ samples and both baselines.
- Generality tested only between two related evolved texts.

Next candidates: unrelated-text generality control; projection/ablation-style injection (remove bmn component
instead of adding); per-step adaptive alpha; more repaired tasks per arm for power.

## 2026-07-18 Round 4: step-ladder vectors, unit-edit vectors, semantics-to-vector alignment

Goal (user direction): move from coarse bad-vs-good to (1) step-level and unit-edit-level conditioning
vectors, and (2) a first viability test of the "skill-text semantic hidden -> steering vector" path.

Setup:
- `ladder_forward.py`: 600-state subsample of the 1213 replayed states; new variants forwarded: step-1
  candidate text (s1) and s1 + each single ranked edit of step 2 (via `skillopt.optimizer.skill.apply_edit`).
  Sequential application of the 3 ranked edits reproduces the step-2 candidate byte-for-byte
  (`ladder_apply_check.json`), so the unit decomposition is exact. s2/s3/bad/none reps reused.
- Skill-text self encodings (token-mean, all layers) for v0000/s1/s2/s3, the three unit texts, and the raw
  edit contents.
- `analyze_ladder.py` -> `analysis/ladder_metrics.json`, `analysis/ladder_geometry.png`.
- GPUs 0/3 shared with another user's ~66 GB SFT job; forwards ran fine at ~0.45 s/state.

Results (mid-layer 10-22 means):
1. Step ladder: each increment is internally consistent across states (d01 0.76, d12 0.64, d23 0.69) but
   increments are distinct directions (cos(d01,d12) 0.33, cos(d12,d23) 0.42, cos(d01,d23) 0.18) with norms
   d01 2.87 >> d23 1.30 > d12 0.96. SkillOpt iteration is a curve, not a ray: the big first step installs
   the dominant "skill presence/quality" direction (which is why cumulative deltas looked text-independent
   at cos 0.948), later steps add smaller, content-specific directions.
2. Unit edits (headline): the three single-edit conditioning vectors are near-orthogonal to each other
   (pairwise cos 0.35 / 0.17 / 0.09), individually moderately consistent across states (0.45 / 0.53 / 0.15;
   the weak one is also the smallest-norm edit), and compose almost perfectly linearly:
   cos(sum_i v(e_i), d12) = 0.97, norm ratio 1.07. Skill units add up linearly in activation space.
   This overturns the 2026-07-10 negative unit-level result, which used behavioral trajectory contrasts;
   with prompt-conditioned extraction, unit-level directions exist and are additive.
3. Semantics -> vector: zero alignment. Text-diff representations vs matched conditioning deltas:
   diag 0.002 vs offdiag -0.002; unit edit text reps vs their vectors 0.005. The direction a skill text
   INDUCES at the decision position is geometrically unrelated (in the naive sense) to the text's own
   pooled representation. A direct "encode skill text, use as steering vector" path is not viable;
   the path would require a learned text->vector mapping (many (edit, vector) pairs as supervision) or
   empirical extraction per unit (as done here, ~4.5 min per unit text on one GPU).

Evidence boundary:
- One step-2 edit set (3 units) from one run/seed; consistency numbers are geometry, not causality - no
  unit-vector injection arm has been run yet.
- Text reps use token-mean pooling; other pooling choices were not swept (the flat ~0.00 across all mid
  layers makes a pooling artifact unlikely to hide a large effect).
- 600-state subsample (seed 42) of the 1213 states.

Natural next causal test: inject v(e1) (search-frontier discipline edit, the most consistent unit) gen-only
@L18 with alpha ~ 2x its layer norm into the s1-skill agent on tasks where s1 fails and s2 succeeds, with a
mismatched-unit control (inject v(e0) on the same tasks).

## 2026-07-18 Round 5: unit-vector causal test and norm-calibrated multi-layer injection

Ops note: the seed42 SkillOpt run finished all 4 steps, so its two idle vllm servers (GPUs 1/2, 40 GB each)
were shut down; round-5 arms moved off the contended GPUs 0/3 (another user's SFT) onto now-free GPUs 1/2.

Arms (all gen-only; HF greedy; max_steps 35):
A. Unit causal chain - 16 tasks where step-1 fails and step-2 succeeds, agent runs with the step-1 skill text:
   - s1_base: 1/16 (val:0038 - a task HF-greedy s1 already solves)
   - + v(e1) "search frontier discipline" L18 alpha 2.2 (3.0x own norm): 2/16 (+val:0050)
   - + v(e0) "check visible surfaces early" L18 alpha 2.2 (3.9x own norm; mismatched-unit control): 4/16
     (+val:0001, 0015, 0050)
   - + v(d12) full step increment L18 alpha 3.3 (3.0x own norm): 1/16 (no gain)
   Behavior (jaccard_good / repeat): base 0.28/0.48; e1 0.32/0.47; e0 0.39/0.41; d12 0.30/0.51.
B. Norm-calibrated multi-layer (bad skill, 8 repaired tasks, direct comparison to single-layer L18 a8 = 2/8):
   - gmb at layers {14,18,22}, each scaled to its own mean-delta norm (2.34/3.51/6.22), global alpha 1.0:
     3/8 (val:0015, 0025, 0037 - 0037 is a flip single-layer never achieved), invalid rate 0.094
     (vs 0.235 for single-layer a8), repeat 0.42.
   - Same at alpha 1.5: 1/8, invalid 0.15 - the dose cliff reappears; the window is alpha ~1x per layer.

Findings:
1. NEW BEST INJECTION SCHEME: distributing the dose across three mid layers at each layer's natural delta
   norm (gen-only) reaches 3/8 repaired flips - 75% of the good-text effect on this subset (good 4/8,
   bad 0/8) - at near-baseline invalid-action cost. Spreading beats concentrating: single-layer needed
   2-3x overdrive (with 3x the invalid rate) to reach 2/8.
2. Unit vectors are causally active (both unit arms >= baseline in flips and behavioral movement) but the
   designed unit-specificity test FAILED: the mismatched unit e0 outperformed the matched e1 (4 vs 2).
   Honest caveats: both units are search-domain edits (weak contrast pair); absolute-alpha matching gave e0
   a higher multiple of its own norm (3.9x vs 3.0x); n=16 with 1-3 flip differences. Notably e0's content
   (prefer visible surfaces early) is plausibly the single most useful behavior for these pick/look tasks,
   so its stronger effect is semantically coherent - what failed is the assignment-specific prediction,
   not unit-level causal activity.
3. Whole-increment injection underperforms its parts: v(d12) (= e0+e1+e2 geometrically, cos 0.97) at 3x norm
   produced zero flips and the weakest behavioral movement. Injecting a sum direction dilutes the useful
   unit component; unit-level injection is the better interface.

Evidence boundary:
- Flip counts 1-4 on n=8-16; no statistical claims. The multi-layer 3/8 vs 2/8 vs 0/8 ordering is consistent
  with the behavioral metrics but needs replication on more tasks/seeds.
- The unit-specificity question remains open pending a domain-disjoint unit pair (e.g., a fridge/cool
  protocol edit vs a search edit) with per-norm-multiple-matched dosing.

Updated default recipe: gen-only, layers {14,18,22} at 1x each layer's mean-delta norm, greedy decoding.

## 2026-07-18 Round 6: domain-disjoint 2x2 unit specificity + mechanistic probes

Design:
- Unit S = step-2 edit e0 (prefer visible surfaces early; search domain). Unit P = step-3 ranked edit #1
  (use direct heat/cool/clean at appliances; transform-protocol domain), extracted on the same s1 base text
  and 600-state subsample (v_unit_P_l18: norm 0.36, consistency 0.28, cos to e0/e1 = 0.34/0.44).
- Task sets from s1 failures with s2-or-s3 success: T_search (16; pick/pick-two/look types) and
  T_protocol (19; heat/cool/clean types). Dose matched by own-norm multiple (3.9x): S alpha 2.2, P alpha 1.4.
  Agent = step-1 skill text, gen-only L18, greedy.

2x2 results (successes):
                 T_search   T_protocol
  s1_base          2/16        1/19
  + S (search)     5/16        1/19
  + P (protocol)   3/16        2/19
Partial double dissociation: S gives +3 in-domain with exactly zero cross-domain effect - the cleanest
unit-specificity evidence so far. P is weak everywhere (+1/+1); its norm is the smallest of all units and
protocol tasks need long correct suffixes, so a weak persistent bias plausibly under-delivers there.
Round-5 arms (e1, d12) cover subsets of these task sets and stay within noise.

Mechanistic probes (analysis/logit_lens.json, analysis/teacher_forced_shift.json):
- Teacher-forced likelihood shift at identical step-0 states (n=100, response = the good-skill rollout's
  actual step-0 answer, injection on response positions only): bad prompt 0 (ref); + gmb L18 a8
  -0.093/token (5% improved); + random same norm -0.094 (2%); + multi {14,18,22} a1 -0.075 (14%);
  good prompt +0.305 (100%). The vector does NOT mimic the good prompt's local next-token policy; its
  rollout effects must come from bias accumulating over the generated reasoning stream across steps.
  Multi-layer calibrated injection is the least off-manifold per token, matching its best rollout numbers.
- Logit-lens: unembedding projections of gmb/e0/P at L18 decode to semantically void multilingual
  fragments - the conditioning direction is not vocabulary-aligned; naive "what words does it write" is a
  null result, reported as such.

Evidence boundary:
- 2x2 cells are 16-19 tasks, differences of 1-3 flips; the S-unit dissociation pattern is directionally
  clean but not individually significant; needs replication (more tasks or seeds) before a paper claim.
- P's weakness confounds "protocol units are harder to vectorize" with "this particular edit is small";
  a stronger protocol unit (e.g., step-3 edit #0 or a slow-update block) is the next candidate.
- Teacher-forced test only probes step-0 single responses; a per-step version along full trajectories
  would localize where the likelihood benefit emerges.

## 2026-07-19 ScienceWorld powered steering adjustment

Protocol: all 30 `boil` variations with fixed train/val/test = 6/6/18, easy mode,
Qwen3.5-4B greedy, 20 steps. Extraction used 180 states from 18 train-only
trajectories (bad/full/parser skills). A ScienceWorld adapter bug was fixed:
valid actions were returned in unstable object order and truncated in-prompt;
the adapter now preserves verb-group order and sorts within groups.

Stable held-out text result: initial 0.0089 vs full SkillOpt text 0.0428
(+0.0339). The earlier n=3 estimate 0.0067 vs 0.1233 was inflated by action-order
noise and is retired.

Frozen steering results (n=18):
- Full good-minus-initial L14 prefill-last+gen: 0.0089 vs initial 0.0089;
  matched random 0.0100. It changed 10/18 trajectories but 0/18 scores.
- Boiling-component (full minus parser-only) L14 prefill-last+gen: 0.0022 vs
  parser-only 0.0011 and full text 0.0428. Four extra random controls span
  0.0011--0.0022, so the tiny gain is not direction-specific.
- Exact online current-state delta, state-conditioned kNN, step gating,
  early-only extraction, and shallow multi-layer injection did not beat the
  frozen textual/control boundary. The best multi-layer validation arm merely
  tied single-layer L14 at 0.0233, so it was not evaluated on test.

Conclusion: the vector is coherent and behaviorally causal, but static additive
last-token residual steering does not transfer the procedural text benefit.
Next architecture-level test should target prompt KV/attention state or a
learned distributed adapter, not more static alpha/layer sweeps. Full report:
`experiments/STEERING_SCIWORLD_ADJUSTMENT_20260719.md`; artifacts under
`outputs/steering_scienceworld_stable/`.
