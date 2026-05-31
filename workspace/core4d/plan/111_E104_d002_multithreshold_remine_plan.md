# E104 — D002 multi-threshold raw-contact remine for medium-box candidates

日期：2026-05-31
上游：E103 scene rebuild + validity reset；E091 data_construction_v2 medium-box manifest

## Context

E103 remine 发现 `reject_raw_contact_not_pass=61/80`，但进一步代码审查显示这不是单一失败模式：

- `raw_contact_fail=5`：确实跑过 D002，但 3cm raw geometry proxy 未过阈值。
- `not_run=56`：没有 D002 raw-contact evidence，被 E103 mining 保守拒绝。

当前 D002 脚本已经在 `raw_contact_proxy.npz` 内保存 `2cm/3cm/5cm` masks，但最终 score/decision/summary 只使用 `3cm`。因此 E104 先修 D002 数据门：对 medium-box 80 行全量重跑 D002，并同时输出 `3cm` 与 `5cm` 两套 raw-contact summary/candidate sets。

## Claims

### C1 — D002 多阈值判定可复现

成功标准：
- 修改 `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/scripts/score_raw_contact_candidates_v2.py`：
  - 支持从 D001 JSON 读取 stage0 rows；
  - 支持 `stage1-queue=selected-medium-box`，覆盖 `box004/box022/box026` 的 80 个 case-person；
  - 同一次 raw proxy 计算同时输出 `3cm` 与 `5cm` score/decision；
  - 保留 3cm 旧文件名作为 backward-compatible alias。

### C2 — 80 个 medium-box case 全量重跑

成功标准：
- 输出目录：`workspace/core4d/results/E104/d002_medium_box_multithreshold/`
- `3cm` 与 `5cm` summary 都覆盖 80 个 case-person。
- 已跑过的 11 条 pass/fail case 也重新计算，不复用旧 summary。

### C3 — 两套候选/拒绝分布并列输出

成功标准：
- 输出 `raw_contact_summary_v2_3cm.tsv/json` 与 `raw_contact_summary_v2_5cm.tsv/json`。
- 基于两套 D002 summary 生成 threshold-specific medium manifest 和 E103-style mining result：
  - `medium_box_manifest_3cm.tsv`
  - `medium_box_manifest_5cm.tsv`
  - `v2_candidates_3cm_with_fingertip.tsv`
  - `v2_candidates_5cm_with_fingertip.tsv`
  - 对应 rejected/summary/REVIEW。

### C4 — 可视化和复审

成功标准：
- 每个 sequence 输出 3cm/5cm raw-contact timeline PNG。
- 输出 aggregate decision/count/scatter plots。
- 用 high subagent 只读复审 3cm/5cm 输出与关键可视化。

## Non-goals

- 不在 E104 内跑 CEM。
- 不因为 5cm pass 就自动进入 dynamics；5cm 是宽松候选集，需要后续视觉/指尖语义审查。
- 不修改 E103 的 clean scene/inertial 结论。

## Decision Rules

- 3cm candidate set 是 conservative set。
- 5cm candidate set 是 recall-oriented set；若 5cm 新增大量 pass，后续必须通过 fingertip/visual/H2 review 再进 target regeneration。
- 如果 5cm 与 3cm 差异主要来自单手贴近而非双手 overlap，不能直接视为可搬运 case。
