# E163 — narrow surfaceBand RL-ready export results

日期：2026-06-15

计划文件：`workspace/core4d/plan/173_E163_narrow_surface_band_rl_export_plan.md`

## 1. 目标

把 E163 narrowSurfaceBand 三 case 结果导出为下游 RL-ready 数据。此步骤不重跑 CEM，只把已有
E163 full 结果接入 data_construction_v3 的 S5/S6 handoff contract。

## 2. 实现

新增：

| 文件 | 内容 |
|---|---|
| `workspace/core4d/scripts/experiments/E163/export_narrowSurfaceBand_rl_handoff.py` | 从 E163 variants/metrics 生成 S5 handoff、S6 evidence、RL export input，并执行 partner OmniRetarget |
| `workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh` | 固定入口；默认 `--execute-partner --allow-partner-failure` |

固定版本字段：

```text
source_exp_id = E163
spider_method_id = gateA_surfaceBandA2_postureRerankA_narrowSurfaceBandReleaseDecay
retarget_variant_id = omnirt_v1
target_variant_id = ref_fk
hand_collision_variant_id = rubber_hull
```

注意：`target_variant_id=ref_fk` 只表示 target route；E163 的 CEM/reward 方法写入 `spider_method_id`。

## 3. 运行

静态检查：

```bash
python3 -m py_compile workspace/core4d/scripts/experiments/E163/export_narrowSurfaceBand_rl_handoff.py
bash -n workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh
git diff --check -- workspace/core4d/plan/173_E163_narrow_surface_band_rl_export_plan.md \
  workspace/core4d/scripts/experiments/E163/export_narrowSurfaceBand_rl_handoff.py \
  workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh
```

预检：

```bash
python3 workspace/core4d/scripts/experiments/E163/export_narrowSurfaceBand_rl_handoff.py
```

结果：`rl_export_input.tsv` 3/3 `RL_EXPORT_READY`，partner manifest 3 rows `ready_to_execute`。

正式导出：

```bash
bash workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh
```

## 4. 产物

输出根目录：

```text
workspace/core4d/results/E163/narrow_surface_band_rl_export/
```

关键文件：

| 文件 | rows | 说明 |
|---|---:|---|
| `manifest/narrowSurfaceBand_source_rows.tsv` | 3 | E163 source snapshot |
| `s5_handoff/handoff_manifest.tsv` | 3 | S5 handoff |
| `s6_downstream/evidence/downstream_evidence_manifest.tsv` | 3 | S6 CEM evidence |
| `s6_downstream/rl_export/rl_export_input.tsv` | 3 | 下游 RL 唯一输入索引 |
| `s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv` | 3 | partner OmniRetarget manifest |
| `summary.json` / `summary.md` | 1 | 导出摘要 |

RL export 检查：

| case_id | decision | scene_act | trajectory | contact_mask | cem_result |
|---|---|---|---|---|---|
| `box023_person2` | `RL_EXPORT_READY` | exists | exists | exists | exists |
| `d003_box021_20231018_029_p2` | `RL_EXPORT_READY` | exists | exists | exists | exists |
| `e091_box004_20231003_2_083_p2` | `RL_EXPORT_READY` | exists | exists | exists | exists |

版本字段检查：

```text
source_exp_id/spider_method_id =
  E163 / gateA_surfaceBandA2_postureRerankA_narrowSurfaceBandReleaseDecay
```

## 5. Partner OmniRetarget

三条 partner motion 均已实际执行并通过：

| source | partner | status | trimmed |
|---|---|---|---|
| `box023_person2` | `box023_20231008_045_p1` | pass | `.../holosoma_rl_partner_omnirt_omnirt_v1_box023_20231008_045_p1/trimmed/20231008-045-person1-Box023_with_obj_original.npz` |
| `d003_box021_20231018_029_p2` | `box021_20231018_029_p1` | pass | `.../holosoma_rl_partner_omnirt_omnirt_v1_box021_20231018_029_p1/trimmed/20231018-029-person1-Box021_with_obj_original.npz` |
| `e091_box004_20231003_2_083_p2` | `box004_20231003_2_083_p1` | pass | `.../holosoma_rl_partner_omnirt_omnirt_v1_box004_20231003_2_083_p1/trimmed/20231003_2-083-person1-box004_with_obj_original.npz` |

Trim 结果：

```text
box023 partner: 178 -> 134 frames
box021 partner: 134 -> 71 frames
box004 partner: 121 -> 102 frames
```

## 6. 结论

E163 narrowSurfaceBand 三 case 已导出为下游 RL-ready 数据：

```text
RL_EXPORT_READY = 3/3
partner OmniRetarget pass = 3/3
```

下游应消费：

```text
workspace/core4d/results/E163/narrow_surface_band_rl_export/s6_downstream/rl_export/rl_export_input.tsv
```

该表已包含 `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`、`source_exp_id` 和
`spider_method_id`，可区分 E163 方法版本。
