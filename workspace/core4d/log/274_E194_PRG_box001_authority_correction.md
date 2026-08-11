# E194 PRG box001 人审 authority 更正

_Core4D · Phase 57 correction · 2026-08-11 · 对 [log273](273_E194_G1_box001_box023_box021_expansion_results.md) 中 PRG 人审口径的勘误_

本次更正回答一个单一审计问题：E194 的 PRG→G1 box001 对比应如何恢复 E173 最终人工 USE/DNU 判定。它不重跑 CEM、不改变 rollout 或公共 evaluator，只更换 PRG 人审证据来源并重建派生表格。

## TL;DR

- E173 PRG box001 的最终人审 authority 不是 `eval/full/user_manual_review_filled.tsv`，而是 `s6_downstream/rl_export/box001_user_approved/box001_user_approved_source_rows.tsv`。
- 新 authority 中的 13 个唯一 case 均为 `USE`；E194 冻结的 28 个 box001 case 中不在该集合的 15 个 case 均为 E173 `DO_NOT_USE`。
- 该 authority 只适用于 box001；box023/box021 不作 DNU 外推，在 XLSX 中明确记为 `NOT_APPLICABLE_BOX001_AUTHORITY`。
- 本更正只替换 PRG 人审来源与由此派生的人工迁移统计。72-case 数值指标、12-gate 结果和 G1 推广范围结论不变。

## 1. Authority contract

| 项目 | 更正后口径 |
|---|---|
| 权威来源 | `workspace/core4d/results/E173/s6_downstream/rl_export/box001_user_approved/box001_user_approved_source_rows.tsv` |
| Source SHA256 | `d19af1aafeeb20f9030f4d51f4141bf05727c675ab84c50913eee5eb488f6590` |
| Authority rows | 13 rows / 13 unique case IDs / 全部 `object_key=box001` / 全部 `manual_use_decision=USE` / 全部 `source_exp_id=E173` |
| E194 box001 universe | 28 unique case IDs |
| 最终判定 | authority 成员=`USE`；28-case universe 内非成员=`DO_NOT_USE` |
| 非 box001 | box023/box021=`NOT_APPLICABLE`，不填充 PRG 人审决策 |

旧 `user_manual_review_filled.tsv` 的 `19 USE / 9 DNU` 交集、以及复制到两种 comparison 后的 `38 USE / 18 DNU` 均不是最终 authority 统计，全部作废。

## 2. XLSX 更正结果

`Paired Comparison` 保持 144 行、61 列。原来的 review status/quality/taxonomy/note/reviewer/time/video 字段已替换为可审计的 authority scope、source、source SHA256、status、authoritative decision、USE-set membership、membership rule、source experiment 和 row reference。

| Scope | 唯一 case | Paired rows | 决策 |
|---|---:|---:|---|
| box001 authority USE members | 13 | 26 | USE |
| box001 authority complement | 15 | 30 | DO_NOT_USE |
| box023 | 16 | 32 | NOT_APPLICABLE |
| box021 | 28 | 56 | NOT_APPLICABLE |

排除分析指定的 `box001_20231023_110_p1` 不在 approved USE 集合，因此其两条 paired row 均正确标为 `DO_NOT_USE`。

## 3. 更正后的 PRG→G1 人工迁移

G1 人工结果仍读取 E194 自己的 `full_g1_expansion/user_manual_review_filled.tsv`；这里只用新 authority 修正 PRG 一侧。排除 `box001_20231023_110_p1` 后，G1 已审 14/27：

| 迁移 | Case 数 |
|---|---:|
| PRG USE → G1 USE | 3 |
| PRG USE → G1 DO_NOT_USE | 3 |
| PRG DO_NOT_USE → G1 USE | 3 |
| PRG DO_NOT_USE → G1 DO_NOT_USE | 5 |

此前基于错误 PRG 来源得到的 `6 / 4 / 0 / 4` 迁移统计全部作废。更正后的人工子集仍不支持“全面提升”：USE 与 DNU 之间的净迁移为 0，而且存在 3 个 USE→DNU case；这一判断同时保留原有数值/gate mixed 证据。

## 4. 验证

| 检查 | 结果 |
|---|---:|
| Builder compile / build | PASS |
| Paired rows / columns | 144 / 61 |
| box001 membership逐行一致性 | 56/56 paired rows PASS |
| box023/box021 N/A一致性 | 88/88 paired rows PASS |
| LibreOffice formulas / errors | 11,184 / 0 |
| Gate decisions noPRG/PRG/G1 | 709 / 722 / 720（不变） |
| Strict-12 noPRG/PRG/G1 | 15 / 19 / 19（不变） |
| Delta渐变 | 14列均保留；绿色=改善、黄色=0、红色=回退 |

最终工作簿：

`workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/E194_noPRG_PRG_G1_comparison.xlsx`

Workbook SHA256：`90a504c6051261c539602e8ad0b39c3bd614c119f24d991dcf9fff589a9b7bbf`

## 5. 解释、限制与后续

观测事实是：更正后的 14-case 人审子集中，PRG↔G1 的 USE/DNU 跨边界迁移为双向各 3 个。解释上，这不支持 G1 在人工可用性上相对 PRG 的单向全面提升；但该子集由当前已审 case 构成，仍不是 27-case 的完整随机终审，因此也不能据此估计总体人工通过率或显著性。

后续若要给出完整的人审提升结论，应完成排除后其余 13 个 box001 case 的 G1 终审，再使用同一 E173 membership authority 复算 27-case 迁移矩阵。当前无需重跑数值 eval。

## Reproducibility Notes

```bash
.venv/bin/python -m py_compile workspace/core4d/scripts/eval/reports/build_E194_three_arm_workbook.py
.venv/bin/python workspace/core4d/scripts/eval/reports/build_E194_three_arm_workbook.py
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/E194_noPRG_PRG_G1_comparison.xlsx
```
