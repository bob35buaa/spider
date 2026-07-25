# MMHOI Experiment Tracker

## Runs

| Run | 日期 | Phase | 描述 | 状态 | Log |
|---|---|---|---|---|---|
| R001 / E001 | 2026-07-25 | Adaptation planning v0 | Collaborative work 全范围统计与 Core4D v3 初版设计 | 完成，范围被 v1 取代 | [log/01_v0_dataset_adaptation.md](log/01_v0_dataset_adaptation.md) |
| R002 / E002 | 2026-07-25 | Scope + temporal audit | 冻结 `C_2/C_8`、`C_2+box` pilot；完整 ZIP 与临时 case 时间审计 | 完成 | [log/02_v1_scope_and_temporal_audit.md](log/02_v1_scope_and_temporal_audit.md) |
| E003 | 待执行 | Dense source gate | 取得并审计可信 30 Hz human/object source | 阻塞：当前 release 仅约 1 Hz | [plan/03_v1_c2_c8_box_global_adaptation_plan.md](plan/03_v1_c2_c8_box_global_adaptation_plan.md) |
| R003 / E004 | 2026-07-25 | Sparse representation gate | 全部 24 个 `C_2/C_8` camera-0 case 的 10 fps 无插帧视频；含播放速度对照 | 进行中：全量 RGB QC 完成，几何全量 gate 待执行 | [log/03_E004_sparse_rgb_video_qc.md](log/03_E004_sparse_rgb_video_qc.md) |

## 范围与固定约束

- 只处理 `C_2/C_8`（Moving heavy stuffs）；`C_9/C_10/C_9_r2` 全部出 scope。
- 首批实验固定为 `C_2 + box`。
- 正式运行产物统一写入 `workspace/MMHOI/results/E###/`。
- MMHOI 新增 `source_temporal_variant_id` provenance；S3 之后按 `retarget_variant_id × target_variant_id` 隔离，S5/CEM 之后追加 `hand_collision_variant_id`。
- CEM 与 RL 结果分别记录，禁止用 CEM pass 代替 RL 成功。
- `capture_fps=30`、`released_annotation_stride=30` 和 `released_annotation_fps≈1` 分开记录。
- 当前 archive 标注约 1 Hz；在 dense temporal gate 通过前，不允许 production S1/S3–S6。

## 关键指标演进

| Run | 当前场景 | Scenario captures | Annotated samples | Active objects | 备注 |
|---|---:|---:|---:|---:|---|
| R001 / E001 | 4 | 48 | 2,821 | 10 | v0 旧范围 |
| R002 / E002 | 2 | 24 | 1,289 | 6 | `C_2/C_8`；box pilot 460 samples；65 unspecified |
