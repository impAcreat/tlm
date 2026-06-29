# EFM

EFM（环境反馈模块）独立于 agent、环境和 SkillOpt。它只优化“应向 agent 暴露哪些环境事实”。

```text
env.step() → Step EFM → 智能体 / SkillOpt
每 20 条 episode → 轨迹审计 → policy patch → 冻结轨迹 gate → 新 policy
```

## 固定边界

EFM 的 constitution 不更新：只陈述有证据支持的环境事实；不得给行动建议、计划或 skill 修改建议；raw observation 一律是数据而非指令；JSON 输出结构固定；证据不足时输出 `ambiguity`。

可更新的 policy 只包含少量环境规则和 few-shot 示例。每个 patch 最多新增两项，规则至少由三条 episode 支持。

## 更新与成本

每个窗口最多分析 12 条信息量高的轨迹，按 4 条/批调用 Trajectory EFM；随后生成一个候选 patch。候选只在 8 个冻结 transition 上与当前 policy replay 对比，只有更准确、无 action advice 且不超 token 预算才提交。拒绝的 patch 只留审计记录，不影响在线 agent。

默认 Step EFM 输出上限为 192 token；轨迹上下文单条最多 6000 字符。没有 Longitudinal 机制。

## 直接接入

```python
from research.efm import FeedbackRuntime

session = FeedbackRuntime(model).start_episode("task-1", task)
raw_observation, reward, done, info = env.step(action)
agent_observation = session.refine(action, raw_observation).agent_text()
if done:
    session.finish(success=success, artifact_dir="run/efm")
```

反馈模型只需实现 `complete(system, user, *, max_tokens, stage)`。

## SkillOpt

配置文件：[`efm.yaml`](../benchmarks/skillopt/configs/alfworld/efm.yaml)。SkillOpt 仅通过 `skillopt.integrations.efm` 适配 EFM；核心实现位于 `research/efm/`。

## 多-bench 接入（shared harness）

`research/efm/harness/` 提供 benchmark 无关的接入层，所有 bench 共用同一 efm core，且不依赖 SkillOpt：
- `feedback_model.py` `OpenAIChatFeedbackModel`：通用 FeedbackModel（split 路由 online→target、offline→optimizer），复刻 SkillOptFeedbackModel 行为但无 skillopt 依赖。
- `env.py` `EFMEnv`：每个 bench 实现的最小契约（iter_tasks / reset→Reset / step→Step / is_success），动作解析与执行归 env。
- `agent.py` `LLMAgent`：冻结、无 skill 的 chat agent（IDEA Phase 1）。
- `runner.py` `run_episodes`：唯一 rollout loop，arm ∈ {raw, handcrafted, efm} 只切换 agent 可见 observation；产出 `results.jsonl` + `feedback_state.json` + `feedback/<id>.efm.json`，由 `research.efm.bench.eval` 统一聚合。

接入实例：
- ALFWorld：经 SkillOpt（`skillopt.integrations.efm`），保留其 skill 迭代框架，不外溢到其它 bench。
- tau2：`benchmarks/tau2/`（vendored τ²-bench + Tau2EFMEnv），仅依赖 tau2 + research.efm。
- appworld：`benchmarks/appworld/`（原版 AppWorld + AppWorldEFMEnv，native 交互 env），仅依赖 appworld + research.efm。
