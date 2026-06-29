# 学习真正促进 Agent 交互与自进化的环境反馈

## 1. 目标问题

目标问题不是如何再造一个环境，也不是如何再发明一种 prompt / skill updater，而是一个更基础的问题：
**在（长程）agent--environment 交互中，什么样的环境反馈，真的能让一个既定 updater 把 agent 变得更好？**

现有方法（例如 TextGrad、SkillOpt）已经证明：可以根据任务成败、轨迹和 critique，更新 prompt、skill 或其他 agent artifact。以 SkillOpt 为例，冻结 target model，把 Markdown skill 当作可训练状态；target 在环境中 rollout，optimizer LLM 读取环境反馈轨迹并优化 skill patch候选。

但这里隐含了一个假设：环境已有的反馈，比如：reward、observation、tool result、notification 和 verifier error，已经是足够好的学习信号。复杂环境里通常不是这样：

- 原始 observation 往往很长、噪声大，关键失败点被淹没；
- 最终 reward 只说明结果，不说明失败发生在哪一步、为何发生；
- 同一失败轨迹可能包含工具故障、偶发噪声、执行失误和真正缺失的 skill，不能一概拿去更新；


## 2. 方法

**核心：把“环境反馈”本身做成一个能在 agent–环境交互中自我进化的模块（EFM）。** 它持续学习如何从交互中提取、组织并表达更有利于 agent 解题（以及后续自进化）的环境反馈。

研究分两阶段。**第一阶段（当前）**：冻结 agent，且 agent 不带 skill，只让 EFM 学习——以干净地隔离“反馈模块自身进化”这一个变量。**第二阶段（后续）**：把成熟的 EFM 接入自进化 agent 框架，让更好的反馈同时驱动 agent artifact 的更新。

与最初设想不同，本研究让 EFM 拥有**自己的可训练状态与 update 闭环**（而非纯产出、完全交给外部 updater）。借鉴 SkillOpt，但把对象从 agent 换成 EFM：EFM 的 **skill = 不可变 constitution + 一个小而有版本的 policy（少量 scoped rule/example）**；每次只提一个**小 delta**（增/退一条 rule/example），绝不重写整份 skill。

EFM 的自我进化在两个时间尺度上**协同（co-evolution）**：

- **Step 级（在线、局部、快）**：每个 action 后，产出“当前 observation 最该让 agent 知道的、有据可查的那一个事实”（最佳局部表达）；并维护一份**回合内记忆（reflection）**，每隔若干步更新，用于改进本回合后续的反馈。该记忆是临时的、只对 EFM 可见，不直接改动 durable skill。

- **Trajectory 级（离线、全局、慢）**：一条 rollout 结束后，用**真实任务结果**做 credit assignment——定位对成败最关键的 **pivotal 步**，并刻画“局部最优反馈”与“放到整条任务看才最优的反馈（whole-picture）”之间的差距。

二者的耦合是关键，而**记忆是进化的一等单元**：trajectory 级会逐条评估 step 级写下的每条记忆（看其形成前后的步骤与该回合结果），判定它是否编码了可复用的反馈规律——**毕业**则沉淀为 durable skill 的一条 rule/example，否则丢弃。于是 step 级在线产生候选、trajectory 级用结果把真正有效者提炼进 skill，skill 再反过来塑造下一回合的在线反馈。

**门控（gate）以结果为准，而非以代理指标为准。** 一个关键发现：反馈的结构质量（一致性 / 完整性 / 简洁性）并不是任务表现的可靠代理——只优化结构质量可把“完整性”从 68% 提到 98%，任务成功率却几乎不动。因此一个候选 delta 只有在**留出的失败回合的 pivotal 步上、相对 whole-picture 目标更优，且不在其他步造成质量回退**时，才被接受、policy 版本 +1。

**三个层级**是同一模块在不同时间尺度上的自我优化方式（不是三类彼此独立的反馈）：

| 层级 | 时间尺度 / 可见信息 | 优化什么 | 进化依据 | 状态 |
|---|---|---|---|---|
| **Step** | 每步、在线、仅当前局部 | 当步最该反馈的那个事实 + 回合内记忆 | 是否帮 agent 当步更好决策、少走弯路 | 已实现 |
| **Trajectory** | 每 rollout、离线、全轨迹 + 结果 | pivotal 归因、local→global 差距、逐条记忆毕业评估 | 真实任务成败 | 已实现 |
| **Longitudinal** | 跨任务、跨版本 | 跨任务不变量、最稳健可迁移的反馈规律 | 多轮进化中的实际累积效果 | 待实现 |

Longitudinal 级是下一层，用于在多窗口、多任务、多版本之间识别最稳健可迁移的反馈规律，目前尚未实现。

## 3. 具体环境bench

| 环境 bench | 交互与原始反馈特点 | 原始反馈的主要问题 | 可学习的反馈形式 | 预期帮助 |
|---|---|---|---|---|
| GIAI2 / ARE | 多轮结构化工具调用、长 tool output、notification 与环境状态变化 | 关键证据埋在长日志中；后期失败可能源于早期状态误读 | 关键事件切片：首个决定性失败点 + 证据 + 状态变化 | 帮 agent 在交互中修正计划；帮 updater 学到状态同步等可迁移规则 |
| ALFWorld | 每一步都有 action、observation、reward、done | 最终 timeout 过粗；长 action-observation 序列掩盖失败转折 | 前置条件 / 状态估计 / 重复动作诊断 | 促使 agent 基于 observation 重规划，而非重复无效动作 |
| SpreadsheetBench | 代码/工具执行、stderr、文件产物、多个 testcase 与 verifier diff | `eval-mismatch` 太粗，完整日志和 workbook diff 又过长 | 最小反例：首个失败 case + 预期 invariant + 实际 artifact | 区分系统性 skill 缺陷与基础设施错误，产生可复用的生成/验证规则 |

需要保留的长程信息不应全部进入 feedback，而应以“可回溯的证据”形式挂在表格中的反馈之后。三个环境分别说明这一点：

- **GIAI2 / ARE：长程多工具环境。** agent 会处理长邮件正文、附件、base64 内容、大列表、日历/文件/网页状态，以及后续 notification。一个好的反馈不是复述整段日志，而是指出 notification 改变了什么状态、agent 从哪一步开始仍沿用旧假设。

- **ALFWorld：具身式 action--observation 环境。** 这里反馈天然带有局部因果结构：动作后会返回 observation、reward 和 done。应提取“目标—首次违反的前置条件—之后是否修复”的小片段，而不是只将最终 timeout 交给 updater。例如，agent 得到“容器尚未打开”的 observation 后仍重复放置物品，反馈可归因为“前置条件失败后未更新子目标”。这既能在本轮提醒 agent 先修复前置条件，也能支持将“每次 action 后检查 observation；失败后更新子目标”沉淀为跨任务 skill。

- **SpreadsheetBench：代码生成、执行与 verifier 环境。** 一次 rollout 同时包含模型输出、代码、运行错误、输出文件和多个 testcase 的校验。最有价值的是最小可复现反例，而非整个 workbook：首个失败 case、应满足的公式/值/格式 invariant、实际生成 artifact 与相关代码片段。比如“已有行写入了数值，但新增行没有生成 Total 公式”。这样 updater 能学习“处理新增和边界行后验证公式覆盖范围”，同时把纯执行故障、文件系统问题等不应固化为 skill 的信号排除出去。


## 4. 研究问题与验证方式

围绕第 1、2 部分，实验应同时回答三个问题：

1. **交互价值：** 相比直接使用原始 observation / tool output，step-level feedback 是否让 agent 在同一 rollout 内更好地理解状态、恢复失败并完成任务？
2. **进化价值：** 在固定 updater、task split 和更新预算下，trajectory-level / longitudinal-level feedback 是否比原始轨迹更能带来可靠的 agent 更新？
3. **自我改进：** 用交互与更新的真实后果作为监督后，feedback 模块是否逐渐提高对“哪些环境证据值得保留、何时给、以何种层级表达”的判断？

最小对比应保持 target、环境、任务划分、updater 和预算一致，只替换 feedback 表示：

| 对照 | 给 agent / updater 的信息 |
|---|---|
| Outcome-only | final reward、`fail_reason` 或最终 verifier 结论 |
| Raw trace | 完整 observation、tool output、conversation 和执行日志 |
| Handcrafted feedback | 固定模板或人工规则压缩的 feedback |
| EvoFeedback | 由模块产生、带环境证据的 step / trajectory / longitudinal feedback |

评价不应只看某一次任务是否成功，而应同时记录：

- **interaction success / recovery**：同一 rollout 中，收到反馈后是否避免重复错误、完成子目标或完成任务；
- **repair rate**：原本失败的训练类问题在更新后被修复的比例；
- **transfer gain**：更新在未见任务或邻近任务上的增益；
- **regression rate**：既有成功样本被更新破坏的比例；
- **feedback efficiency**：单位 feedback token、调用次数或 wall-clock 成本带来的交互与进化收益；
- **feedback fidelity**：反馈是否能追溯到关键事件与证据，且没有把工具故障或偶发噪声错误地固化为长期规则。

最终要学习的不是“如何生成一段看起来合理的 critique”，而是一个可被 agent 和 updater 实际利用、并能通过交互、repair、transfer 与 regression 共同检验的环境反馈机制。
