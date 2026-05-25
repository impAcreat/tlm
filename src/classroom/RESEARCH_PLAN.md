# 多智能体教学环境中的自进化 Agent Harness

## 核心问题

在一个多智能体教学环境中，如何让 teacher 通过可验证反馈，持续提升一组异质 student agents 的 harness，从而让下一届学生学得更快、更稳、更泛化？

这个问题的关键不是简单增加多个 agent 互相反馈，而是要分别证明三件事确实带来增益：

1. teacher 的教学策略有用；
2. student 的异质性有用；
3. 同伴经验共享有用。

## 研究主线

teacher 不应该只是 judge。它应该学习“什么类型的 harness 修改，对什么类型的 student，在什么类型的任务上有效”，并把这些教学经验迁移到后续学生和后续届次。

student 被修改的对象也不应该只是一句 prompt，而是一个分层 harness：

```text
StudentHarness =
  prompt policy
  skill library
  memory policy
  tool policy
  reflection policy
  optional steering/RL adapter
```

teacher 给出的建议应该是结构化 patch，而不是自然语言散文式 coaching：

```json
{
  "target_layer": "memory",
  "diagnosis": "student forgets failed search queries",
  "patch": "store failed query plus successful reformulation",
  "expected_effect": "improve retrieval robustness",
  "scope": "retrieval-heavy students on search tasks"
}
```

这样系统才是一个可实验、可消融、可归因的科研对象。

## 核心贡献

### 1. Harness-Level Teaching，而不是 Prompt Feedback

当前代码已经支持 prompt、skill、memory、tool policy 这些层的 patch。下一步应该把 patch 的语义显式化：

- `target_layer`：修改哪一层，例如 prompt、skill、memory、tool_policy、reflection_policy、steering_adapter；
- `diagnosis`：基于证据的失败模式诊断；
- `patch`：具体的 harness 修改；
- `expected_effect`：teacher 对修改效果的可检验预测；
- `cost`：patch 带来的 token、tool、memory 成本；
- `scope`：这个 patch 预计适用于哪些 student 类型和任务类型。

这样 teacher 的输出不只是“建议学生更仔细”，而是一次可记录、可执行、可验证的 harness 干预。

### 2. Teacher 也要做 Credit Assignment

teacher 的自进化不能只靠记录经验，而要看它给出的建议是否真的让 student 后续表现变好。

可以把 teacher reward 定义为：

```text
teacher_reward =
  post_advice_student_gain
  - advice_cost
  - overfitting_penalty
  + cross_student_transfer_gain
```

其中：

- `post_advice_student_gain`：student 接受建议后的直接提升；
- `advice_cost`：建议引入的复杂度、token 成本、工具调用成本；
- `overfitting_penalty`：建议是否只对当前题有效、对 held-out 任务无效；
- `cross_student_transfer_gain`：这个建议是否能迁移到其他 student。

这一步很关键。否则 teacher 只是一个评分器，不是一个 evolving teacher。

### 3. Student 异质性要可控

student 的 feature 不应该只是“谨慎”“激进”这类人格描述，而应该变成实验变量。

第一阶段可以先做 training-free 的异质性：

- `retrieval_heavy`：更依赖检索和历史经验；
- `planning_heavy`：更强调分解任务和先计划后执行；
- `memory_heavy`：更依赖长期 memory；
- `tool_aggressive`：更愿意使用工具和检查；
- `conservative_verifier`：更强调验证和保守输出。

这些差异可以先通过 prompt、tool budget、memory policy、reflection frequency 实现。后续再考虑 steering、RL 或轻量 LoRA adapter。

### 4. 同伴经验共享要通过可控机制实现

不建议一开始让 students 直接互相聊天，因为这容易引入噪声，也难做归因。

更好的机制是维护一个 `PeerExperienceBank`，记录每个 student 的：

- failure trace；
- successful trace；
- harness patch；
- patch effectiveness；
- task type；
- error type；
- student feature。

teacher 基于这个经验库做三类事情：

- 如果很多 students 都失败，生成 common failure stress test；
- 如果只有某一类 student 失败，生成 personalized remedial test；
- 如果不同 students 用不同方法解出同一题，提炼 transferable skill。

这样“同学共同进步”就不是模糊的互评，而是一个可控、可消融的算法模块。

## 当前代码对应关系

```text
src/classroom/
  agent/teacher.py          teacher 诊断、提出 patch、自我反思
  agent/student.py          student 解题、接收建议、更新 harness
  agent/types.py            StudentAttempt 和 TeachingAdvice 等记录
  harness/state.py          分层 harness 状态和 patch application
  interaction/classroom.py  多 student、多 generation 的主循环
  interaction/peer.py       PeerExperienceBank 的当前雏形
  interaction/workflow.py   PocketFlow 编排：attempt -> teach -> revise -> reflect
```

当前代码骨架方向是对的，已经有 teacher、student、peer bank、harness patch、teacher reflection。主要缺口在三个地方：

1. patch schema 还不够结构化；
2. teacher reward 和 credit assignment 还不够显式；
3. 还缺少系统性的 ablation 实验脚本。

## 前三步实现与实验规划

### 第一步：形式化 Harness Patch 和 Patch Outcome

实现目标：

- 扩展 `HarnessPatch`，加入 `target_layer`、`expected_effect`、`cost`、`scope`、`error_type` 等字段；
- 新增 `PatchOutcome`，记录 patch id、before score、after score、task type、student feature、delayed transfer gain；
- 统一 LLM teacher 和 rule-based teacher 的输出 schema；
- 保持对当前 prompt、skill、memory、tool_policy patch 的兼容。

实验目标：

- 用现有 HumanEval 小切片跑一轮；
- 每个 patch 都要能对应到 before/after 的可验证变化；
- 统计不同 harness layer 的 patch 成功率。

成功标准：

- 系统能回答：“这次修改改了哪一层？为什么改？改完有没有帮助？”

### 第二步：加入 Teacher Credit Assignment 和 Baseline

实现目标：

- 每次 advice 后计算 `teacher_reward`；
- teacher memory 不只存自然语言总结，还要存 patch family 的效果统计；
- 增加三种运行模式：
  - `no_teacher`：student 只做自我反思，不接收 teacher patch；
  - `fixed_teacher`：teacher 给建议，但自身不更新；
  - `evolving_teacher`：teacher 根据 patch outcome 更新自身 harness；
- 在 JSONL 和 transcript 中输出关键指标。

实验目标：

- 比较不同模式下的 learning curve：
  - before score；
  - after score；
  - gain per patch；
  - advice cost；
  - held-out HumanEval 表现。

成功标准：

- evolving teacher 应该优于 fixed teacher；
- fixed teacher 和 evolving teacher 都应该优于 no-teacher self-evolution；
- 这个优势要在相同 task budget 下成立。

### 第三步：让 Student 异质性和 Peer Sharing 可消融

实现目标：

- 把 `StudentFeature` 扩展成明确控制项，例如 tool budget、memory retrieval policy、reflection frequency、verification strictness、planning depth；
- 增加至少五种 deterministic student profile；
- 扩展 `PeerExperienceBank`，加入 task、error、student feature、patch effectiveness 元数据；
- 增加三种 peer mode：
  - `none`：teacher 只能看到当前 student；
  - `summary`：teacher 看到同伴失败统计；
  - `patch_bank`：teacher 可以复用已验证有效的 peer patch。

实验目标：

- 比较 single-student 和 multi-student classroom；
- 在相同总 attempt 数下比较不同 peer mode；
- 测量 peer-derived patch 是否能迁移到没有产生原始失败的 student。

成功标准：

- multi-student peer experience 应该提高 sample efficiency、final score 或 held-out transfer；
- 这个提升不能只是因为总尝试次数更多。

## 消融实验表

| 问题 | 对照组 A | 对照组 B | 期望观察 |
| --- | --- | --- | --- |
| teacher 是否有用？ | no teacher | fixed teacher | student 提升更快 |
| teacher 自进化是否有用？ | fixed teacher | evolving teacher | 后续 generation/cohort 更好 |
| harness-level patch 是否有用？ | prompt-only | multi-layer harness patch | held-out 提升更稳定 |
| student 异质性是否有用？ | 单一 student 类型 | 多 student 类型 | 暴露更丰富 failure distribution |
| peer sharing 是否有用？ | no peer bank | patch/effectiveness bank | 出现 cross-student transfer gain |

## 最小论文 Claim

这个方向最小但清晰的论文 claim 可以是：

> 在多智能体教学环境中，如果 teacher 的建议能够被结构化为 harness-level patch，并通过可验证 outcome 做 credit assignment，那么 self-evolving teacher 可以比固定 teacher 和无 teacher 的 self-evolution 更高效地提升异质 student agents；同时，peer experience bank 能提供更丰富的失败分布和可迁移 patch，从而提升后续学生的学习效率与泛化能力。

第一阶段不应该追求复杂 agent zoo，而应该先把闭环做扎实：

```text
structured patch
  -> verified outcome
  -> teacher credit assignment
  -> next-round teaching improvement
```
