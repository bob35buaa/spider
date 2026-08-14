# E198 G1A2-arm RL export (box001 / box024 / box004)

**日期**: 2026-08-15 · **Phase**: 61 (E198 下游) · **状态**: ✅ 完成

## 目的

把 E198 的 **G1+A2 arm** CEM 结果导出为下游 RL 模仿训练输入，schema/流程与 E173 完全对齐
（复用同一套共享 S6 工具），并为每个导出 case 提供**完整 partner 信息**（同 sequence 对侧 person 的
Stage2b 复用）。

## 选择与参数

- **arm**: 固定 G1A2（object gravcomp G1 + A2 hand-gate；scene = rubberHull gravcomp sidecar，
  cem_result = `E198_*_G1A2.npz`，hand_collision_variant_id=rubber_hull）。
- **case 选择**：
  - box001 / box024 → `user_manual_review_filled.tsv` 中 `manual_use_decision==USE`（box001=16、box024=4）
  - box004 → 用户显式 4 例（`082_p1/p2`、`083_p1/p2`；不在 review 表，合成 USE / `USER_SELECTED`）
- **源实验根**（S5 handoff + Stage2b 复用）：box001/box024→E173，box004→E172。每个选中 case 在源
  handoff 中恰有 1 条 `HANDOFF_READY & target_gate=pass & visual_qc=pass` 行。
- 人工 USE 为 operational allowlist；numeric/gate-health 事实（`numeric_release_pass` 等）保留在
  `*_source_rows.tsv`，不 gate、不覆盖。

## Phase 0 — box001 partner Stage2b 补齐

box001 两个 USE 源例 `2_039_p1` / `2_041_p1` 的对侧 person 从未 retarget（E173 只有 p1）。本机 raw
mocap + `box001_person2` 模板 + OmniRetarget toolchain 齐备，就地补建，产物写入 **E198 自有目录，
未改动 E173**：

| partner | raw-contact | 结果 | 变体 |
|---|---|---|---|
| `box001_20231003_2_041_p2` | 5cm 合法 `raw_contact_pass` | ✅ pass | omnirt_v1 |
| `box001_20231003_2_039_p2` | 3cm fail / 5cm review（接触弱 active≈0.29）→ **人工 override** raw_contact_pass | v1 `omniretarget_infeasible` → **v2 rescue pass** | omnirt_v2 |

override 记录在 `s3_retarget/box001_partner_raw_contact_5cm.tsv` 的 `raw_contact_notes`。两 partner 均
`stage2b_status=pass`，4 npz + trim_window + `raw_contact_mask_3cm.npz` 在盘。

## 结果

| object | primary RL_EXPORT_READY | partner PAIR_COMPLETE | partner 变体 |
|---|---|---|---|
| box024 | 4/4 | 4/4 | v1=2, v2=2 |
| box004 | 4/4 | 4/4 | v1=3, v2=1 |
| box001 | 16/16 | 16/16 | v1=11, v2=5 |
| **合计** | **24/24** | **24/24** | |

- box001 的两个补建 partner 正确进入 paired：`2_039_p1→2_039_p2`(v2)、`2_041_p1→2_041_p2`(v1)，均
  `PAIR_COMPLETE`，partner artifacts 指向 E198 s3_retarget。

## 验证

- **schema 与 E173 对齐**：`rl_export_input.tsv` / `rl_partner_omnirt_manifest.tsv` /
  `paired_rl_export_input.tsv` 表头与 E173 共享工具参照（box024）**逐字段 MATCH**（3 物体全部）。
  注：E173 box001 曾用 bespoke inline partner builder（多 `v1_failure_status`/`rescue_of` 列），
  canonical 参照取共享工具产物。
- **内容抽验**（box024_026_p1）：`cem_result_npz`→`E198_*_G1A2.npz`✅；`scene_act`→gravcomp sidecar✅；
  `source_exp_id=E198`、`hand_collision=rubber_hull`、`cem_status=pass`、`RL_EXPORT_READY`；
  source audit 的 sha256 与重算一致✅。
- E173 s3_retarget manifest 未改动（Phase 0 全部写入 E198）。

## 改动文件

| 文件 | 说明 |
|---|---|
| `scripts/experiments/E198/export_E198_user_approved_rl.py` | 新增：参数化 G1A2 RL export wrapper（复用 E173 helper/field + 共享 S6 工具） |
| `scripts/experiments/E198/build_box001_partner_raw_contact.py` | 新增：box001 partner 2 行 raw-contact 子集 + 2_039_p2 override |
| `results/E198/s3_retarget/{omnirt_v1,omnirt_v2}/ref_fk/...` | Phase 0 partner Stage2b 产物 + manifest（results/ 不入 git） |
| `results/E198/s6_downstream/rl_export/{box001,box024,box004}_user_approved/` | RL export + partner_omnirt 产物 |
| `results/E198/s6_downstream/evidence/{obj}_user_approved/` | downstream evidence manifest |

## 结论 / 下一步

24 个 G1A2 case 全部 RL_EXPORT_READY 且 partner 完整，可交付下游 RL。注意 `2_039_p2` 为弱接触
override 补建的 partner（第二人上下文，非训练主体），已在 manifest/audit 标注。RL 训练成功与否需另行
评估，本导出仅表示输入就绪（`rl_status=not_run`）。
