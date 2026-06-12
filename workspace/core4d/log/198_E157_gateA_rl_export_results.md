# E157 — E156 gateA 下游 RL export 结果

> 计划：`workspace/core4d/plan/166_E157_gateA_rl_export_plan.md`  
> 状态：**完成 S6 RL export；partner OmniRetarget 3/4 pass，1/4 显式失败**
> 输入：E156 clean8 benchmark 中 3 条 `+gateA` CEM 结果

## 0. 一句话结论

E157 已把 `box021_035_p2`、`box021_035_p1`、`box023_person2`、`box004_082_p1` 四条 E156 `+gateA` 结果整理成标准 S6 下游 RL 输入。

`rl_export_input.tsv` 中 4/4 都是 `RL_EXPORT_READY`，且都使用 `scene_act_E147_rubber_hull.xml`。partner OmniRetarget 实际执行后，`box021_20231011_035_p1`、`box021_20231011_035_p2` 和 `box023_20231008_045_p1` 产出 trimmed motion；`box004_20231003_2_082_p2` 在 Holosoma OmniRetarget 的 CVXPY 求解阶段报 `infeasible`，manifest 中保留为 `missing_outputs / partner_omnirt_outputs_missing`。

## 1. 输出路径

| 类型 | 路径 |
|---|---|
| result root | `workspace/core4d/results/E157/gateA_rl_export/` |
| source snapshot | `workspace/core4d/results/E157/gateA_rl_export/manifest/gateA_source_rows.tsv` |
| S5 handoff | `workspace/core4d/results/E157/gateA_rl_export/s5_handoff/handoff_manifest.tsv` |
| S6 evidence | `workspace/core4d/results/E157/gateA_rl_export/s6_downstream/evidence/downstream_evidence_manifest.tsv` |
| RL export | `workspace/core4d/results/E157/gateA_rl_export/s6_downstream/rl_export/rl_export_input.tsv` |
| partner manifest | `workspace/core4d/results/E157/gateA_rl_export/s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv` |
| summary | `workspace/core4d/results/E157/gateA_rl_export/summary.md` |

## 2. RL export rows

| short case | case_id | person | scene_act | decision |
|---|---|---|---|---|
| `box021_035_p2` | `d003_box021_20231011_035_p2` | `person2` | `scene_act_E147_rubber_hull.xml` | `RL_EXPORT_READY` |
| `box021_035_p1` | `d003_box021_20231011_035_p1` | `person1` | `scene_act_E147_rubber_hull.xml` | `RL_EXPORT_READY` |
| `box023_person2` | `box023_person2` | `person2` | `scene_act_E147_rubber_hull.xml` | `RL_EXPORT_READY` |
| `box004_082_p1` | `e091_box004_20231003_2_082_p1` | `person1` | `scene_act_E147_rubber_hull.xml` | `RL_EXPORT_READY` |

验证：每行 `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz` 均存在。

## 3. Partner OmniRetarget

| source | partner | status | trim frames | 备注 |
|---|---|---|---:|---|
| `d003_box021_20231011_035_p2` | `box021_20231011_035_p1` | `pass` | 127 | trimmed npz 已生成 |
| `d003_box021_20231011_035_p1` | `box021_20231011_035_p2` | `pass` | 127 | trimmed npz 已生成 |
| `box023_person2` | `box023_20231008_045_p1` | `pass` | 134 | trimmed npz 已生成 |
| `e091_box004_20231003_2_082_p1` | `box004_20231003_2_082_p2` | `missing_outputs` | - | OmniRetarget CVXPY infeasible |

失败细节：`box004_20231003_2_082_p2` 在 `robot_retarget.py` 中抛出 `RuntimeError: CVXPY solve failed: infeasible`。这不是导出脚本假成功；manifest 保留了 `failure_mode=partner_omnirt_outputs_missing`，后续 RL 若需要该 case 的 partner motion，需要单独修复或替换 partner retarget route。

## 4. Claims 验证

| Claim | 结果 | 裁定 |
|---|---|---|
| C1: gateA case 都进入 `RL_EXPORT_READY` | 4/4 ready | 成立 |
| C2: RL export 使用 rubber hand collision scene | 4/4 `scene_act_E147_rubber_hull.xml` | 成立 |
| C3: CEM evidence 只记录 E156 后验结果 | 通过 S6 evidence manifest 生成，不回写 S1-S5 | 成立 |
| C4: partner OmniRetarget 输出有显式 manifest | 4 rows，3 pass + 1 missing_outputs | 成立 |

## 5. 脚本

| 类型 | 路径 |
|---|---|
| builder | `workspace/core4d/scripts/experiments/E157/export_gateA_rl_handoff.py` |
| canonical launcher | `workspace/core4d/scripts/launch/active/run_E157_gateA_rl_export.sh` |
| compatibility wrapper | `workspace/core4d/scripts/run_E157_gateA_rl_export.sh` |

验证命令已通过：

```bash
python3 -m py_compile workspace/core4d/scripts/experiments/E157/export_gateA_rl_handoff.py
bash -n workspace/core4d/scripts/launch/active/run_E157_gateA_rl_export.sh
bash -n workspace/core4d/scripts/run_E157_gateA_rl_export.sh
git diff --check
```
