# E197 分析计划：五类 box Full CEM 的 OmniRetarget 与 PRG 指标对比

_Core4D · Phase 60 · 2026-08-12 · 纯离线分析，不启动或重跑 CEM_

---

## 📋 Context

用户要求汇总 `box001`、`box004`、`box021`、`box023`、`box024` 所有进入
Full CEM 的 case，计算 OmniRetarget 与 PRG 的接触和穿透指标，并在
`workspace/core4d/analysis/` 交付 Markdown 与 XLSX。

冻结 Full 母集来自当前 production PRG authority：

| 来源 | 物体 | Case 数 | 说明 |
| --- | --- | ---: | --- |
| E173 | box001 | 28 | E173 Full manifest |
| E172 | box004 | 6 | E172 Full manifest |
| E170/E169 reuse | box021 | 28 | E170 统一 28-case metrics authority |
| E173 | box023 | 16 | E173 Full manifest |
| E173 | box024 | 9 | E173 Full manifest |
| 合计 | 五类 box | 87 | case_id 唯一 |

38-case E190 no-PRG/RL-ready 集是人工筛选后的下游子集，不作为本分析的
Full CEM 母集。

## 🎯 Claims 与成功标准

| Claim | 成功标准 |
| --- | --- |
| C0：母集完整 | 87/87 unique，按物体为 28/6/28/16/9 |
| C1：同口径 | 两种方法都直接使用 `eval.core.core_metrics.evaluate_sequence` |
| C2：四指标明确 | 3mm in-mask、raw in-mask、3mm 手物穿透、lower-body 穿透 |
| C3：同 case 可比 | 每条 OmniRetarget/PRG 共用 scene、contact mask、person index 和帧域 |
| C4：Euler 转换安全 | 从 compiled hinge axes 推导 convention；world-pose round-trip `<1e-4°` |
| C5：报告闭合 | Markdown、XLSX、逐 case TSV、汇总 TSV/JSON 均存在且 87/87 paired |
| C6：XLSX 可复核 | 聚合和 delta 使用公式；LibreOffice 重算后公式错误为 0 |

## 🔬 指标口径

| 用户表述 | Spider 公共字段 | 方向 |
| --- | --- | --- |
| 3mm in-mask 接触 | `hand_object_physics_contact_3mm_in_mask_frac` | 越高越好 |
| raw 接触 | `hand_object_physics_contact_in_mask_frac` | 越高越好 |
| 手物穿透 | `hand_object_physics_penetration_3mm_frame_frac` | 越低越好 |
| lower-body 穿透 | `leg_penetration_frac` | 越低越好 |

所有值均为 frame fraction。主 delta 定义为 `PRG − OmniRetarget`；报告另给
direction-aware improvement，接触沿用 delta，穿透取 delta 的相反数，因此正值
始终表示 PRG 改善。

## ⚙️ 实现与输入

新增单一离线 runner/report：

```text
workspace/core4d/scripts/eval/reports/gen_E197_full_cem_omnirt_vs_prg_metrics.py
```

输入 authority：

- `results/E172/s6_downstream/manifests/cem_full_manifest.tsv`
- `results/E173/s6_downstream/manifests/cem_full_manifest.tsv`
- `results/E170/s6_downstream/eval/full/e170_case_metrics.tsv`

PRG 对每条使用 authority 中的 `qpos_path/scene_xml/trajectory/contact_mask`；
OmniRetarget 使用同一 `trajectory`，先从 freejoint object qpos 转为相同 PRG
scene_act 的 6-DoF slide/hinge 参数，再调用公共 evaluator。转换 convention 从
compiled XML 的 object hinge axes 推导，不使用默认 `XYZ` 或不受约束的 sibling
metadata。

输出固定为：

```text
workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/
```

## 🚀 执行步骤

1. 冻结并审计 87-case authority、路径与 SHA
2. 对 87 条 OmniRetarget 与 87 条 PRG 统一重算四指标
3. 生成逐 case、按物体、总体 paired 汇总及 bootstrap CI
4. 生成 Markdown 报告和带公式的 XLSX 工作簿
5. LibreOffice 重算并扫描所有公式错误
6. 抽查 case-level 公式、集合一致性、指标范围和转换 round-trip
7. 更新 `progress.md`、tracker 与结果 log

## 🎯 下游 RL 宽口径过滤 gate（v3）

在同一 87-case population 上额外输出 `E197-omni-absolute-wide-v3` 候选预过滤。
该 gate **只读取 OmniRetarget** 的本 case 指标，PRG 对比结果不参与判定。

| Gate | 条件 |
| --- | --- |
| 3mm in-mask contact | OmniRetarget ≥0.01；box024 特例为 ≥0.0 |
| Raw in-mask contact | OmniRetarget ≥0.50 |
| 手物 >3mm 穿透 | OmniRetarget ≤0.80 |
| Lower-body 穿透 | OmniRetarget ≤0.30 |
| Foot slip max | OmniRetarget ≤1.90 m |
| Ankle jerk P95 | OmniRetarget ≤4000 m/s³ |
| Object speed max | 本轮仅报告/Viser 展示；用户未指定阈值，不参与 gate |

6 个指标对应 7 条规则（3mm 接触包含 box024 特例）同时满足才标记 `RL_CANDIDATE_OMNI_WIDE_GATE_PASS`。该标签只是数值预筛，
不替代人工 USE、partner、alignment、loader smoke 或既有 `RL_EXPORT_READY` contract。

## 👁️ OmniRetarget Viser review

新增独立 E197 OmniRetarget player，直接读取分析产物中的
`omni_converted_qpos/*.npz` 与对应 scene_act XML；不混入 PRG rollout。
播放器复用既有 `review_player` 的可移植 scene loader、MuJoCo body-transform
预计算与 Viser 播放控制模式，提供 case/object/gate 筛选、指标面板、逐帧拖动、播放、
碰撞体/网格显示与 CLI `--check` 完整性审计。

## ⚠️ 边界与限制

- 本轮是纯离线重评，不修改、不重跑 CEM，不改变现有 release/人工标签
- 比较的是 kinematic OmniRetarget reference 与动态 PRG CEM rollout，回答结果指标差异，
  不等同于单独识别 PRG 的 P/R/G 因果贡献
- 单 seed CEM，bootstrap 仅对 case 重采样，不估计 optimizer seed 方差
- 五物体 case 数不均；总体 micro average 会由 box001/box021 主导，因此同时报告
  per-object 与 object-balanced macro average
