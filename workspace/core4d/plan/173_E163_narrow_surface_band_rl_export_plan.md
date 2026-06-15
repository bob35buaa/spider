# E163 — narrow surfaceBand RL-ready export plan

日期：2026-06-15

## 0. 背景

E163 narrow symmetric surfaceBand 三 case probe 已通过：

```text
e163_rows=3
missing=0
pass=3/3
tracked=3/3
fall=0/3
```

用户要求把这次 E163 结果导出为下游 RL-ready 数据。这里不重跑 CEM，只把 E163 full 结果接入
data_construction_v3 的 S5/S6 handoff contract。

## 1. 范围

导出三条 E163 narrowSurfaceBand 结果：

| short case | source variant |
|---|---|
| `box023_person2` | `E163_box023_person2_narrowSurfaceBand` |
| `box021_029_p2` | `E163_box021_029_p2_narrowSurfaceBand` |
| `box004_083_p2` | `E163_box004_083_p2_narrowSurfaceBand` |

输出目录：

```text
workspace/core4d/results/E163/narrow_surface_band_rl_export/
```

## 2. Contract

固定版本字段：

```text
source_exp_id = E163
spider_method_id = gateA_surfaceBandA2_postureRerankA_narrowSurfaceBandReleaseDecay
retarget_variant_id = omnirt_v1
target_variant_id = ref_fk
hand_collision_variant_id = rubber_hull
```

`target_variant_id` 仍只表示 target route，不写入 CEM/reward 方法名。

## 3. 新增入口

| 类型 | 路径 |
|---|---|
| exporter | `workspace/core4d/scripts/experiments/E163/export_narrowSurfaceBand_rl_handoff.py` |
| launcher | `workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh` |

launcher 默认执行 partner OmniRetarget，并允许单个 partner case 失败后继续记录：

```bash
bash workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh
```

## 4. 成功标准

硬检查：

1. `manifest/narrowSurfaceBand_source_rows.tsv` 为 3 rows。
2. `s5_handoff/handoff_manifest.tsv` 为 3 rows，均 `HANDOFF_READY`。
3. `s6_downstream/evidence/downstream_evidence_manifest.tsv` 为 3 rows，均 `cem_status=pass`。
4. `s6_downstream/rl_export/rl_export_input.tsv` 为 3 rows，均 `RL_EXPORT_READY`。
5. 每行必须有并存在：
   - `scene_act`
   - `trajectory`
   - `contact_mask`
   - `cem_result_npz`
6. `source_exp_id/spider_method_id` 必须透传到 S5/S6/RL export。
7. partner OmniRetarget manifest 为 3 rows；若失败，必须显式记录 `partner_status/failure_mode`。

## 5. 记录

完成后写结果 log，并更新：

```text
workspace/core4d/progress.md
workspace/core4d/EXPERIMENT_TRACKER.md
workspace/core4d/log/INDEX.md
```
