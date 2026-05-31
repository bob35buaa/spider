# E104 — D002 multi-threshold medium-box remine results

日期：2026-05-31
计划：`workspace/core4d/plan/111_E104_d002_multithreshold_remine_plan.md`
上游：E103 scene rebuild + validity reset

## 目标

E103 中 `reject_raw_contact_not_pass=61/80` 混合了 `not_run` 与真正 `raw_contact_fail`。E104 全量重跑 medium-box D002，并同时输出 3cm / 5cm 两套 raw-contact candidate set。

## 代码改动

- `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/scripts/score_raw_contact_candidates_v2.py`
  - 新增 `stage1-queue=selected-medium-box`，覆盖 `box004/box022/box026` 80 个 case-person。
  - 新增 `--decision-thresholds-m 0.03,0.05`。
  - 同一次 raw proxy 计算输出 3cm/5cm summary 与 timeline。
  - 字段修正：`target_both_active_frac` 表示当前阈值；`target_both_active_frac_3cm` / `_5cm` 保留各自真实值。
- `workspace/core4d/scripts/E104/build_threshold_medium_manifests.py`
- `workspace/core4d/scripts/E102/mine_v2_with_fingertip.py`
  - 支持 `--manifest`。
  - mining score 优先读取 generic threshold fields。
- `workspace/core4d/scripts/E102/render_candidate_audit.py`
  - REVIEW decision 文案改为按候选数量动态输出。

## 结果路径

| artifact | path |
|---|---|
| D002 multithreshold output | `workspace/core4d/results/E104/d002_medium_box_multithreshold/` |
| 3cm manifest | `workspace/core4d/results/E104/medium_box_manifest_3cm.tsv` |
| 5cm manifest | `workspace/core4d/results/E104/medium_box_manifest_5cm.tsv` |
| 3cm candidates | `workspace/core4d/results/E104/v2_candidates_3cm_with_fingertip.tsv` |
| 5cm candidates | `workspace/core4d/results/E104/v2_candidates_5cm_with_fingertip.tsv` |
| comparison | `workspace/core4d/results/E104/d002_multithreshold_candidate_comparison.md` |
| candidate REVIEW | `workspace/core4d/results/E104/visuals/candidate_audit_3cm/REVIEW.md`, `workspace/core4d/results/E104/visuals/candidate_audit_5cm/REVIEW.md` |

## D002 raw-contact results

Both thresholds cover 80 case-person rows, 40 sequences, and 3 objects:

| threshold | pass | review | fail | object rows |
|---|---:|---:|---:|---|
| 3cm | 46 | 5 | 29 | box004=20, box022=8, box026=52 |
| 5cm | 48 | 5 | 27 | box004=20, box022=8, box026=52 |

可视化：

- 40 sequences × 2 thresholds = 80 timeline PNG。
- 5 aggregate plots × 2 thresholds = 10 PNG。
- Timeline PNG 均为 raw geometry proxy evidence，不是 dynamics/CEM 证据。

## Mining after D002 remine

Policy update after user review: `size_vs_box023_volume_ratio > 3.0` is no longer a hard Box026 holdout. The authoritative size gate is D001 `size_band=target_medium_between_box023_and_box025`.

| threshold | executable | rejected | executable object | delta |
|---|---:|---:|---|---|
| 3cm | 41 | 39 | box004=11, box026=30 | baseline conservative |
| 5cm | 43 | 37 | box004=13, box026=30 | +2 vs 3cm |

3cm route counts:

| route | count |
|---|---:|
| `candidate_executable` | 39 |
| `reject_raw_contact_not_pass` | 26 |
| `reject_box022_preflight_not_pass` | 8 |
| `existing_positive_not_new` | 3 |
| `candidate_legacy_risk_needs_visual` | 2 |
| `reject_preprocess_infeasible` | 2 |

5cm route counts:

| route | count |
|---|---:|
| `candidate_executable` | 41 |
| `reject_raw_contact_not_pass` | 24 |
| `reject_box022_preflight_not_pass` | 8 |
| `existing_positive_not_new` | 3 |
| `candidate_legacy_risk_needs_visual` | 2 |
| `reject_preprocess_infeasible` | 2 |

5cm compared with 3cm adds two candidate rows:

- `e091_box004_20231002_048_p2`
- `e091_box004_20231003_2_089_p2`

## Interpretation

- E103 的 “0 executable” 主要来自 D002 旧 coverage 不完整和后续过强 holdout；补跑 D002 且移除 Box026 ratio hard gate 后，3cm/5cm 分别出现 41/43 条候选。
- Box026 不再因为 `size_vs_box023_volume_ratio=3.229768 > 3.0` 被硬拒绝；其 D001 size band 是 `target_medium_between_box023_and_box025`。
- 两条 Box026 仍标为 `candidate_legacy_risk_needs_visual`，这是 review 标签，不是 hard reject。
- Box022 仍被 `reject_box022_preflight_not_pass=8` 拦住；D002 多阈值不能替代 Box022 fingertip preflight。
- 5cm 只比 3cm 多放出 2 条 box004；它是 recall-oriented set，不应自动覆盖 3cm conservative set。

## 可视化/复审

High subagent 对 E104 初版只读复审：PASS。用户确认移除 Box026 volume-ratio hard gate 后，candidate mining / REVIEW / comparison 已本地重跑；本次 addendum 未重新启动 subagent。

- 3cm/5cm summary 都覆盖 80 rows / 40 sequences。
- 40 个 sequence 均有 3cm/5cm timeline PNG，aggregate plot 成对存在、可读。
- 当前 3cm mining = 41 candidates，5cm mining = 43 candidates；候选对象为 box004 + box026。
- 发现并已修复两个风险：
  - REVIEW 旧 boilerplate 与候选数矛盾。
  - 5cm 文件中 `*_3cm` 字段承载 5cm 值的字段混淆。

## Box022 fingertip preflight threshold alignment addendum

用户指出 D002 raw-contact 已改为 3cm/5cm，但 Box022 fingertip preflight 仍使用 E099 `vote_case` 默认 2cm，这会导致 Box022 独立 gate 比 D002 更严。

已修复并重跑：

- `workspace/core4d/scripts/E102/run_box022_preflight.py`
  - 新增 `--contact-threshold-m`
  - 新增 `--contact-threshold-label`
  - 新增 `--min-contact-frames`
  - 输出 `contact_threshold_m/contact_threshold_label/min_contact_frames/L_contact_frac/R_contact_frac`
  - JSON 文件名带阈值后缀，避免 3cm/5cm 覆盖
- `workspace/core4d/scripts/E102/render_box022_preflight.py`
  - PNG 标题显示 threshold/min_frames
  - footer 不再写死 “never pass”

重跑结果：

| threshold | Box022 rows | REJECT | SOURCE_BLOCKED | max L/R close-contact frames |
|---|---:|---:|---:|---|
| 3cm | 10 | 8 | 2 | 0 / 0 |
| 5cm | 10 | 8 | 2 | 0 / 0 |

结论：

- `20` 是每只手至少 20 个 close-contact frame，不是 2cm/20cm。
- Box022 在 3cm/5cm 下仍完全没有 fingertip close-contact frames；因此 E104 mining route counts 不变。
- `reject_box022_preflight_not_pass=8` 现在已经是 3cm/5cm 对齐后的结论，不再依赖旧 2cm 默认值。

## Box026 volume-ratio holdout removal addendum

用户指出当前尺寸目标是“大于 Box023、小于 Box025”，因此不应使用 `size_vs_box023_volume_ratio > 3.0` 作为 hard reject。已修改：

- `workspace/core4d/scripts/E102/mine_v2_with_fingertip.py` 删除 Box026 `size_vs_box023_volume_ratio > 3.0` hard holdout。
- 重新生成：
  - `workspace/core4d/results/E104/v2_candidates_3cm_with_fingertip.tsv`
  - `workspace/core4d/results/E104/v2_candidates_5cm_with_fingertip.tsv`
  - `workspace/core4d/results/E104/v2_candidates_3cm_rejected.tsv`
  - `workspace/core4d/results/E104/v2_candidates_5cm_rejected.tsv`
  - `workspace/core4d/results/E104/visuals/candidate_audit_3cm/REVIEW.md`
  - `workspace/core4d/results/E104/visuals/candidate_audit_5cm/REVIEW.md`
  - `workspace/core4d/results/E104/d002_multithreshold_candidate_comparison.md`

新结果：

| threshold | candidates | candidate_executable | candidate_legacy_risk_needs_visual | rejected |
|---|---:|---:|---:|---:|
| 3cm | 41 | 39 | 2 | 39 |
| 5cm | 43 | 41 | 2 | 37 |

候选对象分布：

| threshold | box004 | box026 |
|---|---:|---:|
| 3cm | 11 | 30 |
| 5cm | 13 | 30 |

结论：Box026 现在是候选，需要后续 visual / dynamics gate 排序，而不是在 E104 mining 层被 volume-ratio hard reject。

## Claims 验证

| claim | 状态 | 说明 |
|---|---|---|
| C1 多阈值判定可复现 | PASS | 脚本支持 selected-medium-box + 3cm/5cm 输出 |
| C2 80 case 全量重跑 | PASS | 80 rows、40 sequences、3 objects |
| C3 两套候选/拒绝分布 | PASS | 3cm/5cm manifest + mining candidate/rejected 已生成；Box026 ratio hard gate 已移除后重跑 |
| C4 可视化和复审 | PASS/UPDATED | timeline/aggregate 已复审；candidate REVIEW 已按新候选重跑 |

## 决策

- E104 不跑 CEM；它只修复数据门与候选分布。
- 下一步可开 E105：从 3cm 的 41 条候选中按对象分层选择 typical/diversity，不再把 Box026 因 volume ratio 直接排除。
- Box026 仍需要 visual / dynamics gate 排序；但这是候选优先级问题，不是 E104 data gate hard reject。
