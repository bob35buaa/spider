# E188 Analysis Plan：Bucket 与历史 Box 的 Object Tracking 分层审计

_CORE4D Phase 51 · 2026-08-05 · status: COMPLETE · analysis-only_

## 1. Context

E188 的 15 条 `2→5 kg` 配对结果显示 object tracking 相对 E187 变差：
`track_obj_pos_err_cm_mean` 约回退 1.61 cm，`track_obj_ori_err_deg_mean` 约回退
1.81°。用户进一步指出 E187 视频中的 bucket 位置本身也可能与参考序列存在系统偏差。

本审计统一使用 `core4d-e154-physics-contact-v1` case metrics 中的：

- object position：`track_obj_pos_err_cm_mean`（cm，越低越好）；
- object orientation：`track_obj_ori_err_deg_mean`（degree，越低越好）。

只聚合已有正式评测，不重新计算轨迹、不重跑 CEM、不修改历史人工 authority。

## 2. Questions

1. E178、E187、E188 的 bucket object tracking 在全量、numeric PASS、人工通过三个层次分别如何？
2. bucket 与历史 box 的位置/朝向误差是否存在稳定差距？
3. E178→E187、E187→E188 的变化有多少来自同 case 配对，而不是 case composition？
4. 人工 USE 是否会过滤掉 object tracking 较差的 case，还是视觉审核对该指标不敏感？

## 3. Frozen Authority

### 3.1 Bucket

| Dataset | Metrics | Manual authority |
|---|---|---|
| E178 | `results/E178/s6_downstream/eval/full/e178_case_metrics.tsv` | 同目录 `user_manual_review_filled.tsv` |
| E187 | `results/E187/s6_downstream/eval/full/e187_case_metrics.tsv` | 同目录 `user_manual_review_filled.tsv` |
| E188 | `results/E188/s6_downstream/eval/full/e188_case_metrics.tsv` | 无独立人工审核；另报 E187-USE matched subset，不冒充 E188 manual PASS |

### 3.2 Historical box canonical set

为避免同一 case 被多个 ablation 重复计数，box 全量/numeric 层冻结为每个 object 的最终
production experiment：

| Object | Authority |
|---|---|
| box021 | E170 `e170_case_metrics.tsv` |
| box026 | E171 `e171_case_metrics.tsv`；box022 为 `DATA_NEGATIVE`，无正式 Full metrics，不进入均值 |
| box004 | E172 `e171_case_metrics.tsv`（历史文件名保留） |
| box001、box023、box024 | E173 `e173_case_metrics.tsv`；人工 authority 使用三个最终 per-object RL-export snapshot |

人工层只使用真实 filled/冻结标签：E170 metrics 内 user review、E172 filled review、E173
三个最终 `*_manual_review_snapshot.tsv`（23 USE）。E173 旧的 eval `user_manual_review_filled.tsv`
早于 box023 后续人工审查，不能作为最终 authority。E171 没有 filled review，因此 box026 不进入 manual-only 层，并在结果中
显式报告覆盖差异。

## 4. Strata

每个 experiment/object group 生成以下层：

1. `all_cases`：正式 metrics 的全部有限值 case；
2. `numeric_pass`：`numeric_release_pass=true`；
3. `manual_use_self`：该实验自身人工 `manual_use_decision=USE`；
4. `manual_use_parent_matched`：仅 E188 使用，按 case ID 匹配 E187 的人工 USE，明确标记为
   selection transfer，不代表 E188 已人工通过。

每层报告 `n`、position/orientation mean、median、standard deviation、bootstrap 95% CI。
总体 box 使用 case-weighted mean，同时提供逐 object 分解，防止大 object group 支配结论。

## 5. Paired Controls

- E178↔E187：所有共同 case；
- E187↔E188：15 条共同 case；
- E178↔E187↔E188：三者共同 case，固定 case composition 后比较三版本；
- 对每个 pair 报 mean delta（new-old，正数表示误差变差）、非退化比例和 bootstrap 95% CI。

## 6. Claims / Completion Criteria

| Claim | Criterion |
|---|---|
| C0 authority 完整 | 输入存在、case ID 唯一、关键字段有限，历史 box object 无重复 authority |
| C1 分层正确 | all/numeric/manual 选择逻辑可审计，E188 manual transfer 不误写为 self review |
| C2 组合混杂受控 | 同时给出 experiment-level、per-object、paired-common-case 三套结果 |
| C3 结论可复现 | TSV/JSON/Markdown 由固定本地脚本一键生成并带输入 SHA256 |
| C4 历史不可变 | 不修改任何 E168–E188 原始 metrics、manual review 或视频 summary |

## 7. Outputs

```text
workspace/core4d/results/E188/s6_downstream/eval/object_tracking_audit/
  stratified_summary.tsv
  per_object_summary.tsv
  paired_version_comparison.tsv
  paired_per_object_comparison.tsv
  case_level_audit.tsv
  summary.json
  report.md
```

生成脚本：

```text
workspace/core4d/scripts/eval/reports/gen_E188_object_tracking_audit.py
```

正式命令：

```bash
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E188_object_tracking_audit.py
```

## 8. Interpretation Boundaries

- bucket 与 box 是跨 object、跨实验的描述性比较，不直接等价于算法因果效应；
- E187→E188 的远程 11 条仍有设备混杂，需与 same-device local-4 分开解释；
- numeric PASS 包含 object tracking gate，本身存在 selection-on-metric，不能把 PASS 层较低误差
  解释成算法自然改善；
- manual USE 由视觉整体质量决定，并非专门针对 object tracking，因此可能保留明显 object 偏差。

## 9. Completion

C0–C4 全部 PASS。正式结果见
`log/263_E188_object_tracking_cross_object_audit_results.md` 与
`results/E188/s6_downstream/eval/object_tracking_audit/`。
