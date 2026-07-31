# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md) ·
> [E180–E181 完整执行](progress_archive/E180_E181_20260731_full_backup.md)
>
> 本文件只保留最近完成工作的可靠结论与下一会话入口。

## 2026-07-31：E181 收口（ASSET_REJECTED）

- 计划/结果：`plan/199_E181_coacd_canonical_geometry_plan.md`、
  `log/245_E181_coacd_gate_b_asset_rejected.md`、
  `log/246_E181_rejected_candidate_visual_diagnostic.md`。
- Oracle `3/3 PASS`，CoACD `54/54 BUILD_PASS`；三物体均因冻结的全局
  cavity `≤0.1%` 得到 `0/18`，因此 E181 未生成 `C*`/`D_C` 或启动 Full。
- rejected 3D/2D 可视化 `3/3 APPROVED_DIAGNOSTIC`；它说明几何误差存在，
  但不能证明真实 P/R/G 与 downstream 不可用。

## 2026-07-31：E182 task-conditioned CoACD + paired Full（计划完成）

- 新计划：`plan/200_E182_task_conditioned_coacd_full_cem_plan.md`；Tracker 已新增
  Phase 45 planning-only 行。本轮只写计划，未实现 backend 或启动 GPU。
- `broader cavity≤0.1%` 降为 report-only；dev3 deterministic query tape 对
  同一点比较 `D_M/exact-C/D_C`，分别审计 P contact、R reward 和 G mask/rank。
- production geometry 只在 object-specific K8/16/32 中按
  task-error–runtime Pareto 冻结；只有 catastrophic launch floor 阻断 Full，
  轻微 preferred gate 或 throughput 超标只进入最终 verdict。
- heldout24 在 selection manifest 冻结前不参与 E182 选型，冻结后作为
  evaluation-only 正常跑 query audit 与 Full；禁止按其结果反向调参。
- Full authority 与 E178 同序 27 rows、`9/4/14`、`seed0,1024×32`；基线六门
  `16/27`、十二门 `10/27`，质量、逐门迁移、盲审和效率均 paired 报告。
- worker 为本地单卡 + A100 `3/6/7`；允许叠加已有任务但保留显存/UUID/owner
  双检查，不 kill/抢占。按 measured LPT 分配，历史速度下预期约 `12/5/5/5`。
- E181 尚未实现 grid-SDF runtime、query tape 和 sidecar；均已作为 E182 正式
  实现项列出，下一入口是用户确认后从 S0 authority/preflight 开始。
- 计划完整性审计 PASS：单 H1、10 个结构化 H2、Mermaid 含 accessibility
  metadata；Tracker 描述 62 字符；计划明确包含 report-only cavity、真实查询、
  K Pareto、四卡 Full、E178 paired 指标、效率拆分与失败协议；diff-check PASS。

## 2026-08-01：E182 计划口径修订

- 计划已按用户口径修订：删除全部 K4 rescue；K8 是最低预算，仍过慢时进入
  efficiency failure。heldout 文案改为 `selection-forbidden→evaluation-only`：
  case/E178 baseline 不保密，隔离的只是用于选型的 E182-dependent evidence。
- 修订审计 PASS：计划中 K4 仅以“明确不测试”出现；heldout 角色、冻结内容、
  失败后不反调参数和 full27 正常执行均已写明；单 H1/H2 结构、progress<100
  与 `git diff --check` 全部通过。
