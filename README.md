# TLM Research

TLM 是 learning-machine 研究项目的统一工作区，集中管理 benchmark、模型服务、数据、研究文档和实验记录。

## 项目入口

- 研究问题、方法路线、核心定义与评价指标：`IDEA.md`
- 实验的目标、结果、失败原因与下一步：`experiments/EXPERIMENT_LOG.md`
- 可复用的环境、命令、架构与 trace 说明：`docs/`
- 项目实验闭环：`.codex/skills/tlm-experiment/SKILL.md`

## 目录

- `benchmarks/giai2/`：GIAI2/ARE benchmark 与实验输出。
- `benchmarks/skillopt/`：SkillOpt 源码、ALFWorld 数据及复现实验输出。
- `benchmarks/tau2/`：τ²-bench（vendored 官方源码）+ EFM 适配，经共享 harness。
- `benchmarks/appworld/`：原版 AppWorld native 交互 env + EFM 适配，经共享 harness。
- `research/`：与具体 benchmark 解耦的可插拔研究模块；EFM 在 `research/efm/`；多-bench 共享接入层在 `research/efm/harness/`。
- `datasets/giai2/`：本地 GIAI2 datasets。
- `services/qwen/`：本地 Qwen/vLLM service wrapper、配置、日志与 smoke scripts。
- `models/`：本地模型权重。
- `envs/`：可复用运行环境配置。
- `docs/`：稳定的操作与技术说明。
- `experiments/`：按时间记录的实验结论。

## 当前科研计划

目标：实现与 `IDEA.md` 对应的 feedback prototype，并在 agent 交互与 agent 自进化两个场景中跑通可检查的闭环。

1. **实现 prototype**：明确输入 trace、输出 feedback、配置与记录格式；先在真实交互 trace 上完成最小 smoke。
2. **跑通 agent 交互**：确认环境反馈被 agent 接收，并能在后续状态、计划或动作中观察到变化；保留完整 trace。
3. **跑通 agent 自进化**：固定 updater 与任务划分，使 prototype feedback 能进入更新、复测和结果记录闭环。
4. **沉淀证据**：每一步记录配置、run 路径、trace、结果、失败分类和下一步；通过最小验证后再扩大实验。
