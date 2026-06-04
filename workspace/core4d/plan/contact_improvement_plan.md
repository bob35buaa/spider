# Contact Improvement Plan

日期：2026-06-02

关联背景：

- E109 24-case 扩展评测：`workspace/core4d/results/E109/expanded_24_work_cases/`
- E109 统一 replay 评测：`workspace/core4d/results/E109/unified_replay_eval/`
- 数据构建 v3：`workspace/core4d/scripts/data_construction_v3/`
- 当前主问题：Spider 相比 OmniRetarget 明显降低深穿透，但 3/5cm 近接触和 hand-object physics contact 下降。

## 0. 定位

本计划用于指导后续一整套“接触质量提升”实验。目标不是单独把接触比例刷高，而是实现：

```text
高真实接触 + 低深穿透 + 姿态稳定 + lower-body/object strict pass
```

E109 的核心发现：

- Spider CEM 的 `hand_geom_deep_penetration_2cm` 远低于 OmniRetarget；
- Spider CEM 的 `hand_object_physics_contact`、`eef_near_3/5cm` 低于 OmniRetarget；
- `hand_geom_near_12/15cm` 差距很小，说明 Spider 多数时间仍在物体附近；
- 因此当前问题更像“把压入式接触修成浅近场/少穿透，但没有补回无穿透贴合接触”。

核心假设：

> 当前 `ref_fk + safety` CEM stack 缺少 raw contact 时窗、接触面/接触点语义和浅接触维持目标，导致优化解偏向“安全但隔几厘米”。接触和穿透在当前方法中表现为 trade-off，但物理目标上并非不可兼得。

## 1. 总体实验序列

| 实验 | 目标 | 是否改 CEM | 主要产物 |
|---|---|---:|---|
| E110 | Contact metric audit：解释接触下降机制 | 否 | raw-contact PR/F1、SDF histogram、run length |
| E111 | 数据构建 v3 接触证据链补齐 | 否 | contact artifacts、manifest 字段、S6 evaluator |
| E112 | 小规模 contact-aware CEM ablation | 是 | Pareto 表、视频复核、S6 evidence |
| E113 | 成功配置扩展到 20/24-case work set | 是 | expanded contact-aware comparison |
| E114 | strict + contact alignment positive 进入 RL handoff | 可能 | RL export list、RL smoke/full train evidence |

若 E112 小规模 ablation 失败，不进入 E113；应回到 contact target geometry、collision geom、raw target 和 handbox proxy 诊断。

## 2. E110: Contact Metric Audit

### 2.1 目标

不重跑 CEM/RL，只补评测。回答：

1. Spider 接触差是漏接触、浅间隙，还是只是从穿透变成无穿透？
2. 哪些 object/case 最容易出现接触下降？
3. 当前 `eef_near_*`、`hand_geom_near_*`、`hand_object_physics_contact` 哪个最接近真实接触语义？

### 2.2 输入

- 24-case work set：
  `workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_method_metrics.tsv`
- 11-case strict set：
  `workspace/core4d/results/E109/unified_replay_eval/unified_method_metrics.tsv`
- S1 raw contact masks from data_construction_v3 or legacy imported seed.
- OmniRetarget qpos and Spider CEM qpos from E109 case bank.

### 2.3 输出指标

按 case、method、hand 计算：

| 指标 | 说明 | 方向 |
|---|---|:-:|
| `raw_contact_precision` | method contact 中有多少落在 raw contact active frame | ↑ |
| `raw_contact_recall` | raw contact active frame 中 method 有多少真正接触 | ↑ |
| `raw_contact_f1` | precision/recall harmonic mean | ↑ |
| `raw_contact_iou` | active-frame overlap | ↑ |
| `sdf_deep_pen_frac` | hand-object SDF < -2cm | ↓ |
| `sdf_shallow_pen_frac` | -2cm <= SDF < 0 | 诊断 |
| `sdf_near_0_2cm_frac` | 0 <= SDF < 2cm | ↑ |
| `sdf_near_2_5cm_frac` | 2cm <= SDF < 5cm | 诊断 |
| `continuous_contact_run_mean/max` | 连续接触稳定性 | ↑ |
| `contact_frame_obj_err` | 接触帧内物体追踪误差 | ↓ |

### 2.4 成功标准

- 对 11-case strict 和 24-case work set 都能输出完整表；
- 能按 object_key 汇总 box004、box021、box023、box026、bucket004；
- 对 E109 已有 unified replay 指标做一致性检查，已有指标 mismatch 必须为 0，或明确解释差异来源；
- 结论必须能区分三类失败：
  - `missed_contact`
  - `safe_gap_near_contact`
  - `penetration_removed_without_contact_recovery`

## 3. E111: 数据构建 v3 接触证据链补齐

### 3.1 S1 raw contact artifacts

在 `workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py` 中，将 raw contact 从 gate 证据升级为训练/评测证据。

每个 pass/review case 应输出：

```text
raw_contact_proxy.npz
  raw_contact_mask_3cm      # raw frame, per-person/per-hand
  raw_contact_mask_5cm
  trimmed_contact_mask_3cm  # spider/trimmed frame, per-hand
  trimmed_contact_mask_5cm
  raw_contact_centroid_world
  raw_contact_centroid_object_local
  raw_to_trimmed_frame_index
```

对应 TSV/JSON manifest 需记录：

- `contact_mask_npz`
- `contact_label`
- `contact_person_idx`
- `raw_frame_count`
- `trimmed_frame_count`
- `raw_to_trimmed_mapping_status`
- `left_active_frac`
- `right_active_frac`
- `both_active_frac`
- `left_longest_run_frac`
- `right_longest_run_frac`
- `both_longest_run_frac`
- `contact_target_status`

默认规则：

- 3cm/5cm 仍都输出；
- `--stage2b-contact-label` 只决定进入 Stage2b 的候选阈值，不删除另一套 contact artifact；
- raw contact pass 不等于 downstream positive。

### 3.2 S3/S5 manifest propagation

Stage2b manifest 和 CEM override manifest 必须把 contact artifacts 传下去。

涉及入口：

- `stages/s3_retarget/run_stage2b.py`
- `stages/s5_handoff/export_cem_overrides.py`
- `state/update_case_state_registry.py`

新增或补齐字段：

```text
contact_mask_npz
contact_mask_label
contact_mask_person_idx
contact_mask_time_axis
contact_target_npz
contact_target_source        # ref_fk / raw_surface / fingertip_surface / handbox_surface
contact_target_frame         # world / object_local
contact_target_time_axis
contact_route_diagnostic_ref
```

CEM override 生成规则：

- baseline `ref_fk` 仍保留；
- 若 contact mask 存在，override 应写入 `contact_hdmi_mask_source=core4d_3cm|core4d_5cm` 和对应 path；
- 若 route 是 `raw_surface_contact` / `fingertip_surface` / `handbox_surface`，override 应写入 external target path；
- 不允许用 `ref_fk` output 冒充 contact-aware target route。

### 3.3 S4 target gate diagnostics

在 `stages/s4_gate_visual_qc/run_target_gate.py` 增加 contact-target sanity diagnostics。

对每条 target gate pass row 记录：

- `ref_fk_to_raw_contact_gap_mean_m`
- `ref_fk_to_raw_contact_gap_p90_m`
- `ref_fk_face_mismatch_frac`
- `handbox_surface_gap_mean_m`
- `handbox_surface_gap_p90_m`
- `contact_target_risk_label`

默认不 hard reject `ref_fk`，只用于 risk tagging：

| risk | 条件建议 |
|---|---|
| `low` | gap p90 < 5cm 且 face mismatch 低 |
| `medium` | gap p90 5-12cm |
| `high` | gap p90 >= 12cm 或 face mismatch 明显 |

### 3.4 S6 contact alignment evaluator

新增入口：

```text
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/evaluate_contact_alignment.py
```

输入：

- S5 handoff manifest or S6 downstream evidence manifest
- method qpos path
- scene XML
- S1 contact artifacts

输出：

```text
s6_downstream/contact_alignment/
  contact_alignment_metrics.tsv
  contact_alignment_summary.json
  contact_alignment_summary.md
  contact_sdf_histograms.json
```

输出字段至少包含：

- case/method/variant identifiers
- raw-contact precision/recall/F1/IoU
- per-hand metrics
- SDF band fractions
- physics contact fraction
- deep penetration fraction
- continuous contact run metrics
- failure mode label

### 3.5 QA/reproducibility

更新：

- `qa/verify_reproducibility.py`
- `qa/run_smoke_suite.py`
- `qa/audit_pipeline_release.py`

检查：

- contact artifacts path 存在且非空；
- manifest 中 contact path 字段可解析；
- S6 contact alignment summary 可复现；
- bad fixture：manifest 指向缺失 contact mask 时应失败或明确 warning；
- smoke fixture 至少覆盖一条 raw-contact active frame 和一条 inactive frame。

## 4. E112: Contact-Aware CEM Ablation

### 4.1 case selection

选 4-6 个代表 case：

| 类型 | 建议 case | 用途 |
|---|---|---|
| box004 positive | `e091_box004_20231003_2_083_p2` 或 E096b P1/P2 | 防止破坏已有正例 |
| box021 strict positive | `d003_box021_20231011_035_p1` | clean template 后高接触正例 |
| box021 borderline | `d003_box021_20231011_035_p2` 或 `029_p2` | lower-body/contact 边界 |
| box026 strict positive | E106 strict positive 中选 1-2 条 | 大箱、接触/穿透 trade-off |
| bucket004 nonbox | `bucket004_20231002_022_p1` | 非 box proxy 接触语义 |

### 4.2 ablation variants

每个 case 至少跑：

| variant | 改动 |
|---|---|
| `baseline_ref_fk` | 当前 E106/E107/E108 ref-FK stack |
| `raw_mask_ref_fk` | baseline + raw contact mask |
| `hold_band` | raw mask + hold-contact near-field band |
| `sdf_zero_band` | raw mask + signed SDF near-zero band reward |
| `raw_surface_target` | raw/object-local contact target |
| `handbox_surface_target` | handbox-aware target projection |

所有 variant 必须保持 deep penetration penalty 或 safety gate 开启；不能为了接触率放开深穿透。

### 4.3 成功标准

相对 baseline：

- raw-contact recall 或 `hand_object_physics_contact` 提升 `>= 8pp`；
- `hand_geom_deep_penetration_2cm` 不增加超过 `3pp`；
- pelvis/fall 不退化；
- leg penetration 不超过 strict gate；
- object error 不超过当前 case 可接受范围；
- 视频复核无手背假接触、腿/身体补偿性穿透。

### 4.4 失败决策

若 `raw_mask_ref_fk` 提升接触但增加穿透：

- 优先加强 SDF zero-band/penetration hinge；
- 不直接提高 contact gain。

若所有 contact-aware variants 接触仍低：

- 回到 target geometry，检查 raw target 与 G1 hand geometry 是否不可达；
- 对该 case 标记 `contact_geometry_unreachable`，不要继续调 reward。

若接触提高但 lower-body 退化：

- 增加 lower-body clearance gate；
- 或把该 case 保留为 upper-work diagnostic，不进入 RL handoff。

## 5. E113/E114 扩展与 RL handoff

### 5.1 扩展到 20/24-case work set

只有 E112 至少一个配置满足成功标准后，才扩展到 E109 20/24-case work set。

输出：

- contact-aware method summary
- per-case Pareto decision
- strict positive list
- upper-work but contact-fail list
- contact-good but lower-body-fail list

### 5.2 RL handoff criteria

进入 RL handoff 的 case 必须同时满足：

- S4 target gate pass；
- visual QC pass；
- CEM downstream pass；
- lower-body strict pass；
- contact alignment pass；
- deep penetration below threshold；
- S6 evidence 完整。

不再只用 upper/object WORK 判断 RL-ready。

## 6. 汇报口径

后续对外表述应固定为：

- OmniRetarget 手物接触比例高，但包含大量穿透式接触；
- Spider 当前显著降低深穿透，但接触连续性不足；
- 新实验目标是把 Spider 从“安全近场”推进到“无穿透真实接触”；
- 最终指标必须同时报告 contact、penetration、pelvis/fall、lower-body interference、object tracking。

不要单独用 `eef_near_5cm`、`hand_object_physics_contact` 或 `contact_frac_either` 判定方法优劣。

## 7. 本计划不做

- 不把 `fingertip_aware` 设为默认 route；
- 不用旧 polluted scene 的 Box021/Box026 结论；
- 不把 raw contact pass 直接当 positive；
- 不把 OmniRetarget 的高穿透接触当作最终 GT；
- 不在没有 S6 contact alignment evidence 的情况下扩大 RL handoff。

## 8. 最小落地顺序

推荐实现顺序：

1. E110 evaluator：先解释接触下降；
2. E111 S6 contact alignment：先让 downstream evidence 能记录接触质量；
3. E111 S1/S3/S5 artifact propagation：让 CEM override 能吃 raw contact mask/target；
4. E112 小规模 ablation；
5. E113 扩量；
6. E114 RL handoff。

如果只做一件事，优先做 E110/E111 的 contact alignment evaluator。没有这个 evaluator，后续 reward 或 target 调整无法判断是“真实变好”还是“穿透换接触率”。
