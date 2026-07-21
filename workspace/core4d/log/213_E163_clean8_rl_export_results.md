# E163 clean8 RL-ready export results

日期: 2026-06-18

## 目的

按用户要求，将 E163 narrowSurfaceBand clean8 的 8 个 case 全部导出为下游 RL-ready 输入，单独写入 `narrow_surface_band_rl_export/s6_downstream/rl_export_8case`，不覆盖原 3-case export。

## 命令

```bash
PYTHON_BIN=python3 bash workspace/core4d/scripts/launch/active/run_E163_narrowSurfaceBand_rl_export.sh \
  --case-set clean8 \
  --metrics-tsv workspace/core4d/results/E163/narrow_surface_band/eval/clean8/e163_method_metrics.tsv \
  --source-ref E163_narrowSurfaceBand_clean8 \
  --manifest-subdir manifest_8case \
  --handoff-subdir s5_handoff_8case \
  --evidence-subdir s6_downstream/evidence_8case \
  --rl-export-subdir s6_downstream/rl_export_8case
```

## 产物

| artifact | path | status |
|---|---|---|
| source manifest | `workspace/core4d/results/E163/narrow_surface_band_rl_export/manifest_8case/narrowSurfaceBand_source_rows.tsv` | 8 rows |
| S5 handoff | `workspace/core4d/results/E163/narrow_surface_band_rl_export/s5_handoff_8case/handoff_manifest.tsv` | 8 rows |
| S6 evidence | `workspace/core4d/results/E163/narrow_surface_band_rl_export/s6_downstream/evidence_8case/downstream_evidence_manifest.tsv` | 8 rows |
| RL input TSV | `workspace/core4d/results/E163/narrow_surface_band_rl_export/s6_downstream/rl_export_8case/rl_export_input.tsv` | 8/8 `RL_EXPORT_READY` |
| RL input JSON | `workspace/core4d/results/E163/narrow_surface_band_rl_export/s6_downstream/rl_export_8case/rl_export_input.json` | 8 records |
| partner manifest | `workspace/core4d/results/E163/narrow_surface_band_rl_export/s6_downstream/rl_export_8case/partner_omnirt/rl_partner_omnirt_manifest.tsv` | 6 pass, 2 missing_outputs |
| summary | `workspace/core4d/results/E163/narrow_surface_band_rl_export/summary.md` | `source=E163_narrowSurfaceBand_clean8`, `source_case_count=8` |

## 结果

- 主 RL export 完成：`rl_export_input.tsv` 为 8 rows，全部 `RL_EXPORT_READY`。
- partner OmniRetarget 辅助导出完成 6/8 pass。
- 两个 partner rows 为 `missing_outputs`：
  - `e091_box004_20231003_2_082_p1 -> box004_20231003_2_082_p2`: manifest 缺 `omniretarget_output_npz/trimmed_npz/trim_window_json`；运行时观测为 Holosoma robot retarget `CVXPY solve failed: infeasible`。
  - `e091_box026_20231023_139_p1 -> box026_20231023_139_p2`: manifest 缺 `converted_npz/omniretarget_output_npz/trimmed_npz/trim_window_json`；运行时观测为缺 generated object mesh `box026.obj`。

## 代码变更

`workspace/core4d/scripts/experiments/E163/export_narrowSurfaceBand_rl_handoff.py` 增加可配置 case set / metrics / source-ref / 子目录参数，因此 clean8 export 可写入独立目录；同时修正 summary/source_notes 不再写死 `three-case`。

## 结论

用户要求的 E163 8-case 下游 RL-ready 主输入已经完成。partner OmniRetarget 是辅助对照产物，目前 6/8 pass；若后续需要 8/8 partner 对照，需要分别处理 `box004_082_p2` 的 Holosoma infeasible 和 `box026.obj` 模型缺失问题。
