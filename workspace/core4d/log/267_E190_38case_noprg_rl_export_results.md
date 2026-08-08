# E190 结果 — 38-case noPRG RL-ready export

## 结果

- 合并导出：`results/E190/s6_downstream/rl_export/rl_export_input.tsv`，**38/38 RL_EXPORT_READY**，
  case_id 集合与 `results/E180/rl_metric_separability/frozen_rl_labels.tsv` 完全一致（diff 为空）。
- 按物体行数：box001=13、box004=4、box021=11、box023=7、box024=3（合计 38，与 PRG 侧完全对齐）。
- 按来源实验行数：E189=20（box001/box004/box024）、E179=7（box023）、E168=9（box021 标准 case）、
  E167=2（box021 bridge case）。
- Partner 配对导出：`paired_rl_export_input.tsv`，**38/38 PAIR_COMPLETE + RL_EXPORT_READY**，全部原样
  复用 PRG 侧（E170/E172/E173/E167）已有 partner_omnirt 制品，未新跑 OmniRetarget。
- 独立审计 `E190_38case_noprg_rl_export_audit.json`：`status=pass`
  - `case_id_parity`：PASS（38/38，双向集合相等）
  - `artifact_existence`：PASS（scene_act/trajectory/contact_mask/cem_result_npz/cem_video 全部存在）
  - `source_row_hash_integrity`：PASS（38 行 sha256 逐一重算比对一致）
  - `prg_negative_config_check`：PASS（逐行 `config_act.yaml` 核实 `leg_object_penalty_scale=0`、
    `leg_object_penalty_geom_names=[]`、`cem_leg_gate_enabled` 为 false/缺失、`scene_name` 无
    lowerbody_physics/PRG 补丁痕迹）
  - `partner_parity`：PASS（38/38 pair_status=PAIR_COMPLETE，partner 制品文件存在）
  - `rl_ready_counts`：PASS（合并 38/38，按物体逐一核对）

## 执行中发现的环境问题（非设计缺陷）

box021 的 9 个标准 case，其 `example_datasets/` 下的 scene_act/trajectory 中间文件（`E168` 时代产出，
`dcv3_omnirt_v1_ref_fk_box021_*` 目录）在本机曾经缺失（已被清理，`example_datasets/` 按项目约定为
gitignored/临时目录），CEM 结果本身（npz/video/config_act，位于 `results/E168/s6_downstream/cem/full/`）
始终完整。用户从远端机器重新同步了这批数据后问题解除，未修改任何导出逻辑。另有 E170 自身 box021
partner manifest 中记录的历史绝对路径（如 `/mnt/<uuid>/spider_workdirs/...`，源自原始跑图机器）已在
`e190_common.py::repo_path()` 中加入基于 `core4d/`、`example_datasets/` 标记的重映射逻辑解决。

## 已知的溯源/口径说明（写入审计 JSON 的 `known_provenance_notes`）

1. box021 的 2 个 bridge case 从未跑过 PRG——PRG 侧本身就是用 PRG 之前的 E167/E167A 方法批准
   （`USER_APPROVED_BRIDGE_OVERRIDE`，见 `exp_analysis_0726.md`）。它们的 noPRG 行是对同一份 E167
   证据的重新打标，不是一次新的消融。
2. box021 的 9 个标准 case 的 noPRG 证据来自 E168（早于 E169/E170 提出 PRG 概念），CEM 超参/评分口径
   与 E179/E189 的统一配对消融流程不完全一致；未针对 box021 重新跑一次统一口径的 noPRG CEM。
3. 全部 38 行未做新的人工视频 review；审批依据继承自 PRG 侧已批准的 case_id 清单（用户已确认的决策）。

## Conclusion

38-case noPRG rl-ready 导出（含 partner）已完成，可与既有 PRG 侧 38-case 导出在完全相同的 case_id 上
供下游 RL 做 PRG vs noPRG 训练对比。box021 的口径/溯源差异已在审计文件中显式记录，不影响其余 27 例
（box001/box004/box023/box024）的口径一致性。

## Files changed

- `workspace/core4d/scripts/experiments/E190/e190_common.py`（新增）
- `workspace/core4d/scripts/experiments/E190/build_noprg_source_manifest.py`（新增）
- `workspace/core4d/scripts/experiments/E190/export_38case_noprg_rl.py`（新增）
- `workspace/core4d/scripts/experiments/E190/build_noprg_partner_paired_export.py`（新增）
- `workspace/core4d/scripts/experiments/E190/audit_38case_noprg_rl.py`（新增）
- `workspace/core4d/results/E190/`（新增，gitignored，不入库）
