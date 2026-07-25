# MMHOI Collaborative Work 数据适配计划（v0）

> 历史计划：用户已将范围收敛为 `C_2/C_8`、首批 `C_2+box`。当前执行依据为
> [`03_v1_c2_c8_box_global_adaptation_plan.md`](03_v1_c2_c8_box_global_adaptation_plan.md)。

## Context

- 目标：将已初步打通的 CORE4D 原始 mocap → 重定向链路适配到 MMHOI。
- 范围：只处理 MMHOI 的 `Collaborative work` 双人协作子集。
- 原始数据：`/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI`。
- 论文：`paper/Kogashi 等 - 2025 - MMHOI Modeling Complex 3D Multi-Human Multi-Object Interactions.pdf`。
- Git 基线：`experiment/E161-surface-release-ablation`（启动时为 `67cef0b84d81107128a082f36804caa16a15251c`）。
- 工作分支：`experiment/MMHOI-data-adaptation`。

## Claims

1. 能从本地 MMHOI 文件而非论文概述，枚举 `Collaborative work` 的动作序列、样本数和物体类别。
2. 能把 MMHOI 原始字段与 CORE4D v3 的 S0–S6 输入/输出 contract 逐项对应，并显式列出不可直接复用的部分。
3. 能定义一个阶段化、可验证、可恢复、variant 不互相覆盖的 MMHOI 工作区和适配方案。
4. 第一阶段交付只做统计与方案，不把尚未实跑的 S1–S6 状态写成已通过。

## 本轮改动

- 初始化 `workspace/MMHOI/` 的 `plan/`、`log/`、`scripts/`、`results/`、tracker 和 progress 结构。
- 生成 `workspace/MMHOI/data_stat.md`。
- 生成 MMHOI 全局适配方案文档。
- 保存可复跑的数据 inventory/stat 脚本与机器可读统计快照（若原始格式允许）。

## 成功标准

- [x] 当前分支从指定基线创建，且原工作树改动未被覆盖。
- [x] 统计文档明确筛选口径、层级定义、去重规则和本地证据路径。
- [x] 动作序列、序列数、trial/sample 数、物体种类均可由脚本复算。
- [x] 适配方案覆盖 S0/S0b/S1/S2/S3/S4/S5/S6，每阶段列出输入、产物、验证 gate 和失败状态。
- [x] 明确双人身份、坐标系/单位、帧率、SMPL-X/人体参数、物体 mesh/pose/contact 的适配风险。
- [x] 文档不把计划项描述为已完成实验。
- [x] `git diff --check` 通过，文档内部链接有效。

## 执行/训练命令

本轮是数据审计与方案设计，不启动训练。统计命令必须固化到：

```text
workspace/MMHOI/scripts/data_inventory/
```

后续真正的重定向、CEM/RL 命令须在相应实验 plan 中版本化，并将正式产物写入：

```text
workspace/MMHOI/results/E###/
```
