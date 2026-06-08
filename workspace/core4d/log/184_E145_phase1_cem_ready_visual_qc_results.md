# E145 Phase 1 CEM-Ready + 可视化结果

时间：2026-06-05 18:59 CST

## 目标

按 `workspace/core4d/plan/154_E145_full_nonbox_template_release_to_rl_ready_plan.md` 推进 Phase 1：

- 使用 E144 的 82 个 non-box 5cm pass 候选。
- 固定默认路线为 `raw_mask_ref_fk`：`retarget_variant_id=omnirt_v1`，`target_variant_id=ref_fk`。
- 产出 CEM-ready manifest、3 卡 split、override、preflight 和可视化包。
- 不启动 CEM，不启动 RL。

## Template release

输出：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s2_templates/nonbox_template_review.tsv`

结果：

- 21/21 source templates 显式标记 `approve_clean`。
- 类别：bucket 9、desk 8、chair 4。
- policy：bucket 使用 `bucket_wall_proxy_aabb`；desk/chair 使用已写入 v3 pipeline 的 surface voxel multi-box proxy 方案。

## Stage2b

输出：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv`
- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s3_retarget/omnirt_v1/ref_fk/stage2b_continue_failures_omnirt_v1_ref_fk.tsv`
- `workspace/core4d/scripts/E145/run_stage2b_continue.py`

结果：

- total rows: 82
- `stage2b_status=pass`: 69
- `stage2b_status=fail`: 13

13 个 Stage2b fail case：

| case_id | target_task | failure_mode |
|---|---|---|
| `desk023_20231030_019_p2` | `dcv3_omnirt_v1_ref_fk_desk023_20231030_019_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `chair005_20231030_044_p2` | `dcv3_omnirt_v1_ref_fk_chair005_20231030_044_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `chair021_20231008_055_p2` | `dcv3_omnirt_v1_ref_fk_chair021_20231008_055_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `chair021_20231020_076_p1` | `dcv3_omnirt_v1_ref_fk_chair021_20231020_076_p1` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk020_20231018_090_p2` | `dcv3_omnirt_v1_ref_fk_desk020_20231018_090_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk021_20231023_037_p2` | `dcv3_omnirt_v1_ref_fk_desk021_20231023_037_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk023_20231023_117_p2` | `dcv3_omnirt_v1_ref_fk_desk023_20231023_117_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk021_20231011_011_p2` | `dcv3_omnirt_v1_ref_fk_desk021_20231011_011_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk020_20231018_091_p2` | `dcv3_omnirt_v1_ref_fk_desk020_20231018_091_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `chair021_20231011_058_p2` | `dcv3_omnirt_v1_ref_fk_chair021_20231011_058_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `chair021_20231011_059_p2` | `dcv3_omnirt_v1_ref_fk_chair021_20231011_059_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk020_20231018_090_p1` | `dcv3_omnirt_v1_ref_fk_desk020_20231018_090_p1` | `legacy_stage2b_pipeline_failed_before_verify_summary` |
| `desk023_20231020_117_p2` | `dcv3_omnirt_v1_ref_fk_desk023_20231020_117_p2` | `legacy_stage2b_pipeline_failed_before_verify_summary` |

## Target gate + 可视化

Target gate：

- manifest: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv`
- `target_gate_status=pass`: 69
- `target_gate_status=not_run`: 13

可视化渲染：

- render manifest: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render/visual_qc_render_manifest.tsv`
- render package root: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render/cases/`
- review montage root: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_review_montage/`
- `render_status=pass`: 69
- `render_status=skipped`: 13

Visual QC manifest：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc/visual_qc_manifest.tsv`
- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc/visual_qc_review.tsv`
- `visual_qc_status=pass`: 69
- `visual_qc_status=not_run`: 13

审查方式：

- 将 69 张 contact sheet 合成 12 页 review montage，并逐页进行本地视觉检查。
- 审查标准覆盖 object offset、object flyaway/jump、robot collapse、严重 lower-body/object entanglement、明显 template pose 错误。
- `visual_qc_review.tsv` 每条 pass row 都记录原始 video/sheet 路径和对应 montage page。
- 高风险但通过的 case 在 notes 中标出：`chair021_20231011_058_p1`、`chair021_20231008_056_p2`、`desk021_20231023_037_p1`。

实际观察：

- bucket 系列整体稳定；bucket004/bucket009/bucket010 有桶倾斜或弯腰抓取，但没有物体飞离、明显跳变或机器人 collapse。
- chair005/chair021 是高风险对象；chair021 存在翻转/侧放/坐靠姿态，但 sheet 上为连续物体姿态，没有模板错位或明显 proxy 大错。
- desk007/desk021/desk023 的桌体/支架结构保持一致，没有 E144 早期语义 proxy 坐标轴或拓扑错误的表现；个别 desk021 视角下有分离感，标为 high-risk pass。

## S5 handoff + CEM-ready

S5：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s5_handoff/handoff_manifest.tsv`
- `HANDOFF_READY`: 69
- `HANDOFF_PENDING_STAGE2B`: 13
- `HANDOFF_PENDING_TEMPLATE_OR_VARIANT`: 82 shared rows

CEM-ready：

- variants: `workspace/core4d/scripts/E145/variants.tsv`
- CEM-ready TSV: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/raw_mask_ref_fk_cem_ready.tsv`
- preflight: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/raw_mask_ref_fk_cem_ready_preflight.tsv`
- summary: `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/raw_mask_ref_fk_cem_ready_summary.md`
- overrides: `examples/config/override/core4d_E145_*_raw_mask_ref_fk.yaml`

结果：

- `variant_rows`: 69
- `preflight_ok=True`: 69
- category counts: bucket 33、chair 11、desk 25
- split counts: local-gpu0 23、remote-gpu0 23、remote-gpu1 23

## 验证

- `python3 -m py_compile workspace/core4d/scripts/E145/*.py`: pass
- `git diff --check`: pass
- 12 页 visual QC review montage 已生成并检查。
- 本轮没有启动 CEM。
- 本轮没有启动 RL。

## Phase 2 入口

下一阶段按 3 卡并行 full CEM：

- local-gpu0: 23 variants
- remote-gpu0: 23 variants
- remote-gpu1: 23 variants

Phase 2 完成后再做 full eval、S6 downstream evidence、`rl_export_input.tsv`；仍不进入 RL，除非后续明确下令。
