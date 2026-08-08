# E190 — 38-case noPRG RL-ready export（PRG 侧 38-case 导出的对照版本）

## Context

下游 RL 已有一套冻结的 "PRG" 38-case 导出：`results/E180/rl_metric_separability/frozen_rl_labels.tsv`，
覆盖 box001(13)/box004(4)/box021(11)/box023(7)/box024(3)，分别打包在 E170（box021）、E172（box004）、
E173（box001/box023/box024）的 `s6_downstream/rl_export/`。用户希望为**同一组 38 个 case_id** 构建
noPRG 版 rl-ready 导出，以便下游 RL 能在完全相同的 case 上同时用 PRG / noPRG 两版动作做训练对比，
并且要求同时导出 partner（配对）动作。

noPRG CEM 证据已经存在，只是分散在四个不同实验里：box001/box004/box024 来自 E189（43 例 PRG vs noPRG
配对消融），box023 来自 E179（16 例配对消融），box021 的 9 个标准 case 来自 E168（PRG 概念提出之前的
基线跑法），box021 的 2 个 "bridge" case（`box021_20231018_029_p2`、`box021_20231011_035_p1`）从未跑过
PRG，PRG 侧本身就是用 E167/E167A 方法批准的（`USER_APPROVED_BRIDGE_OVERRIDE`，见
`exp_analysis_0726.md`）。

**用户已确认的两个边界**：
1. box021 直接复用 E167/E168 现有原始 CEM 结果，不新跑统一 CEM（接受方法学口径与 E179/E189 不完全一致）。
2. 不做新的人工视频 review；审批依据继承 PRG 侧已批准的 case_id 清单，审计上用各来源实验自身的自动化
   CEM 状态/门控指标。

E190 是一个纯打包/导出实验，**不跑任何新的物理仿真**，因此 `.claude/rules/experiment.md` §7 的场景快照
Safeguard-2 不适用。

## Parameters

- 目标：38 个 case_id（来自 `E180/frozen_rl_labels.tsv`），统一标记 `spider_method_id=E167A_zOnlyBody_noPRG`。
- 数据来源（按物体）：
  - box001/box004/box024 → `results/E189/s6_downstream/manifests/cem_full_manifest.tsv`
  - box023 → `results/E179/s6_downstream/manifests/cem_full_manifest.tsv`
  - box021（9 例）→ `results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv`
  - box021（2 例 bridge）→ `results/E167/holosoma_zonly/rl_export/s6_downstream/rl_export/rl_export_input.tsv`（`E167A` variant 行，重命名 case_id）
- S5 handoff 复用：box004→E172、box001/box023/box024→E173、box021→E168（`s5_handoff/rubber_hull/`）。
- Partner：原样复用 PRG 侧 partner_omnirt（E170/E172/E173/E167），不重新生成。

## Run command

```bash
python3 workspace/core4d/scripts/experiments/E190/build_noprg_source_manifest.py
python3 workspace/core4d/scripts/experiments/E190/export_38case_noprg_rl.py
python3 workspace/core4d/scripts/experiments/E190/build_noprg_partner_paired_export.py
python3 workspace/core4d/scripts/experiments/E190/audit_38case_noprg_rl.py
```

## Result

见 [267_E190_38case_noprg_rl_export_results.md](../log/267_E190_38case_noprg_rl_export_results.md)。
