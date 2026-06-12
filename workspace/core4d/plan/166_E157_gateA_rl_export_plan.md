# E157 实验计划：E156 gateA 下游 RL handoff 导出

日期：2026-06-12  
状态：执行中

## Context

用户希望先用 E156 的 `+gateA` 结果做下游 RL 实验，指定 3 个 case：

- `box021_035_p2`
- `box021_035_p1`（后续追加）
- `box023`（按当前 clean benchmark 中的 `box023_person2` 解释）
- `box004_082_p1`

本实验不重跑 CEM；它把 E156 clean8 benchmark 中已完成的 `+gateA` 轨迹整理成 `data_construction_v3` S6 下游接口，并导出 partner 的临时 OmniRetarget motion，供后续 RL 消费。

## Claims

| Claim | 验证方式 |
|---|---|
| C1: gateA case 都能进入 `RL_EXPORT_READY` | `s6_downstream/rl_export/rl_export_input.tsv` 中全部 ready |
| C2: RL export 使用 rubber hand collision scene | `scene_act` 指向 `scene_act_E147_rubber_hull.xml` |
| C3: CEM evidence 只记录 E156 后验结果，不回写 S1-S5 | 通过 `record_downstream_evidence.py` 生成 S6 manifest |
| C4: partner OmniRetarget 输出有显式 manifest | `partner_omnirt/rl_partner_omnirt_manifest.tsv` 记录 pass 或 failure |

## 输入

| 输入 | 路径 |
|---|---|
| E156 variants | `workspace/core4d/scripts/experiments/E156/variants.tsv` |
| E156 metrics | `workspace/core4d/results/E156/clean8_gate_decay/eval/full/e156_method_metrics.tsv` |
| E156 gateA CEM | `workspace/core4d/results/E156/clean8_gate_decay/cem/full/` |
| contact mask | `workspace/core4d/results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz` |
| CORE4D raw | `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real` |
| SMPL-X | `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx` |

## 输出

全部正式输出写入：

```text
workspace/core4d/results/E157/gateA_rl_export/
```

核心文件：

- `manifest/gateA_source_rows.tsv`
- `s5_handoff/handoff_manifest.tsv`
- `s6_downstream/evidence/downstream_evidence_input.tsv`
- `s6_downstream/evidence/downstream_evidence_manifest.tsv`
- `s6_downstream/rl_export/rl_export_input.tsv`
- `s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv`
- `summary.md`

## 实现

新增 canonical 脚本：

```text
workspace/core4d/scripts/experiments/E157/export_gateA_rl_handoff.py
workspace/core4d/scripts/launch/active/run_E157_gateA_rl_export.sh
```

兼容 wrapper：

```text
workspace/core4d/scripts/run_E157_gateA_rl_export.sh
```

脚本逻辑：

1. 从 E156 `variants.tsv` 读取目标 `method_group=gateA` rows。
2. 从 E156 metrics join 对应 `+gateA` 指标，作为 downstream notes。
3. 写 S5 handoff seed，`hand_collision_variant_id=rubber_hull`，`scene_act` 使用 rubber sidecar。
4. 写 S6 evidence seed，`cem_status=pass`、`rl_status=not_run`。
5. 调用 `record_downstream_evidence.py` 生成 S6 evidence manifest。
6. 调用 `export_rl_inputs.py` 生成 RL export input。
7. 调用 `export_rl_partner_omnirt.py` 生成 partner OmniRetarget manifest；默认实际执行 partner OmniRetarget，若失败必须在 manifest/log 中显式记录。

## 验证

- `python3 -m py_compile workspace/core4d/scripts/experiments/E157/export_gateA_rl_handoff.py`
- `bash -n workspace/core4d/scripts/launch/active/run_E157_gateA_rl_export.sh`
- `bash -n workspace/core4d/scripts/run_E157_gateA_rl_export.sh`
- 生成后检查：
  - `rl_export_input.tsv` rows 与目标 case 数一致，且全部 `RL_EXPORT_READY`
  - partner manifest rows 与目标 case 数一致
  - required paths 存在：`scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`
  - 若 partner 执行成功，`trimmed_npz` 存在；若失败，failure_mode 必须非空

## 风险

- `box023` 的当前 clean benchmark case 是 `box023_person2`，其 raw identity 来自 E077：`20231008/045/person2/Box023`。
- `box004_082_p1` 的 partner 是 `box004_20231003_2_082_p2`。历史上该 partner 可能存在 preprocess/OmniRetarget 风险；本次不静默假设成功，以 partner manifest 结果为准。
