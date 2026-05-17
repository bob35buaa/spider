# E001 Plan: baseline/literature/code audit for collaborative retargeting

日期：2026-05-17

## Context

`workspace/core4d` E077-E081 证明了当前 CORE4D 动力学重定向 pipeline 对部分双人协作数据有正信号，但 E081 的最新结论仍是 partial positive：

- 加入腿/脚-箱 contact pair 后，`box025_person2` 的腿/箱几何干涉显著下降。
- 物体 lift / floor-contact 没有改善，大箱仍像扶/推/局部搬运，不是 strict carry。
- `box023_person2` guard 基本保持，说明 E081 物理修正可作为后续 baseline。

新方向需要先回答三个问题：

1. 当前 object state 到底如何控制：运动学 GT、free joint、object actuator、xfrc、还是机器人接触力主导？
2. CORE4D 双人任务如何建模另一端：虚拟力、mocap partner、双机器人、或面向未来 RL 的“真实人类另一端”假设？
3. SPIDER / DynaRetarget / 双人交互控制论文能给出哪些可落地实验，而不是继续盲调 reward？

本工作区实验编号从 E001 开始；E081 只作为 baseline 引用。

## Claims

| Claim | 最低证据 |
|-------|----------|
| C1 object/freejoint 控制口径可被明确解释 | 给出代码路径、关键变量、scene/scene_act 差异；回答“物体轨迹是否 GT、freejoint 是否全靠机器人力、CEM 是否采样物体控制”。 |
| C2 E081 baseline 验收口径可复用 | 列出 E081 关键结果路径、量化指标、视频/关键帧观察项，以及新实验必须对齐的 baseline 表。 |
| C3 论文启发转化为本项目可执行实验 | 对 SPIDER、DynaRetarget、It Takes Two 分别提炼算法假设、与当前 pipeline 的差距、可实现改动和风险。 |
| C4 产出 E002+ 实验路线图 | 至少 3 个候选实验，每个包含预期收益、实现成本、验收标准、是否需要远程并行。 |

## 改动

E001 是探索/审计实验，不先修改核心算法代码，不运行新的 CEM 训练。允许新增工作区文档和只读/评估辅助脚本。

### 1. 新建工作区骨架

**文件**：

- `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md`
- `workspace/core4d_collab_retarget/progress.md`
- `workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md`
- `workspace/core4d_collab_retarget/log/01_E001_baseline_literature_code_audit_results.md`（完成后写入）
- `workspace/core4d_collab_retarget/paper_notes/`

### 2. 代码审计

重点读取：

- `examples/run_mjwp.py`
- `spider/simulators/mjwp.py`
- `spider/config.py`
- E079-E081 override、train/eval 脚本
- `example_datasets/processed/core4d/.../scene.xml` 与 `scene_act.xml`

输出：object 控制模式说明、历史 E027b/E029-act/E071/E081 的关系图、freejoint 真实接触可行性判断。

### 3. 论文深读

使用 `deep-reading-analyst` + `pdf` 技能，优先读取：

- `paper/Pan 等 - 2026 - SPIDER Scalable Physics-Informed Dexterous Retargeting.pdf`
- `paper/Dhedin 等 - 2026 - DynaRetarget Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization.pdf`
- `paper/DynaRetarget_zh.md`
- `paper/Liu 等 - 2025 - It Takes Two Learning Interactive Whole-Body Control Between Humanoid Robots.pdf`

输出到 `paper_notes/`，并在 E001 log 中做 cross-source synthesis。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md` | 新方向 tracker，从 E001 开始 |
| 2 | `workspace/core4d_collab_retarget/progress.md` | E001 持续进展记录 |
| 3 | `workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md` | 本计划 |
| 4 | `workspace/core4d_collab_retarget/log/01_E001_baseline_literature_code_audit_results.md` | 完成分析后写入 |
| 5 | `workspace/core4d_collab_retarget/paper_notes/*.md` | 论文分析笔记 |

## 运行命令

E001 不训练。只运行静态审计、文本抽取和必要的 baseline eval/文件检查。若后续需要复跑 E081 eval 或抽帧，先创建 `workspace/core4d_collab_retarget/scripts/eval/` 下的脚本再运行。

## 成功标准

| 指标 | E081 baseline | E001 目标 |
|------|---------------|-----------|
| object 控制口径 | 分散在历史实验和代码中 | 一页内明确说明，并给出可验证代码引用 |
| baseline 验收 | E081 log 102 | 形成可复用 checklist，后续 E002+ 必须对齐 |
| 论文启发 | 尚未综合到新方向 | 产出跨论文对照表和本项目实验映射 |
| 下一步实验 | `workspace/exp_task.md` 中是想法 | 形成 E002+ 排序路线图，首个实验可直接写 plan |
