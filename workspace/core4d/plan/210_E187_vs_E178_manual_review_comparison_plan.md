# E187 vs E178 人工终审配对比较计划

_Core4D Phase 50 · 2026-08-03 · 只读 S6 downstream evidence analysis_

## Context

E187 已完成 keep22 Full、numeric paired evaluation 和 22/22 paired video。用户已在
`workspace/core4d/results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv`
填写人工标注，要求与 E178 的历史人工标注直接比较。

两份人工表的覆盖口径不同：E187 有22行，但其中3行的
`manual_use_decision=PENDING`；E178 历史表只包含23条已裁决 Full27 row，投影到keep22后
为18条，bucket004四条缺失。分析必须把缺失或PENDING保留为未决，禁止当成
`DO_NOT_USE`。

本阶段只读两份annotation authority、E178/E187冻结case metrics和keep22 case set；不
修改人工标签、Full artifact、numeric gate、reward/grid/P/G或C9治理状态。

## Claims

| Claim | 验证标准 |
|---|---|
| M1 authority | E187/E178 annotation SHA、row count、duplicate、case-set closure完整记录；原文件SHA不变 |
| M2 pending-safe | 缺失和`PENDING`统一作为未决；USE/DNU transition只在双方均最终裁决的交集计算 |
| M3 paired transition | 输出22-row逐case表、全体/逐object迁移矩阵、恢复/回归/保持列表 |
| M4 quality transition | 只在双方已裁决且quality合法的行上报告CLEAN/MINOR/MAJOR/UNUSABLE变化 |
| M5 numeric alignment | 分别报告E178/E187 numeric对人工USE recall、DNU rejection和错配case；不把numeric改写为人工结论 |
| M6 governance | summary明确E187 C9仍为technical FAIL / USER_WAIVED；三条PENDING使人工终审仍未完全闭合 |

## 输出

```text
workspace/core4d/results/E187/s6_downstream/eval/full/
├── e187_vs_e178_manual_review_comparison.tsv
├── manual_review_comparison_summary.json
└── manual_review_comparison_summary.md
```

实现入口：

```text
workspace/core4d/scripts/eval/reports/gen_E187_vs_E178_manual_review_comparison.py
workspace/core4d/scripts/eval/wrappers/eval_E187_vs_E178_manual_review.sh
```

## 统计口径

- Full authority：E187 `e187_case_metrics.tsv` 的22个 unique case IDs；
- `decisive={USE,DO_NOT_USE}`；`PENDING`或E178缺失行均为undecided；
- primary paired set：双方均decisive的case交集；
- transition：`USE→USE`、`USE→DNU`、`DNU→USE`、`DNU→DNU`；
- quality顺序仅用于描述：`UNUSABLE < MAJOR_DEFECT < MINOR_ACCEPTABLE < CLEAN`；
- exact McNemar只对`DNU→USE`与`USE→DNU`做双侧binomial检验；小样本结果必须同时报告原始计数，不能只报p值；
- numeric alignment只使用冻结的`numeric_release_pass`，不重新计算物理指标。

## 成功标准

- E187 authority 22/22；duplicate/extra/missing=0；
- E178 annotation对keep22的present/missing精确记录；
- comparison TSV 22行且每行保留两侧decision/quality/note/numeric和transition状态；
- summary JSON/Markdown计数与TSV逐行重算一致；
- 两份用户filled TSV分析前后SHA完全不变；
- 不把3条E187 PENDING或4条E178缺失伪装为终审完成；
- 新结果log记录人工和numeric结论的分歧、限制与下一步。

## Canonical command

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E187_vs_E178_manual_review.sh
```

## 可视化说明

本阶段不生成新视频，也不重新解释视频内容；人工标签来源是已完成的22条paired视频审查。
可视化完整性与实际spot-check沿用plan209/log259的22/22证据。本阶段新增的是标签表的
只读统计，不需要新的scene snapshot或GPU运行。

## 禁止项

- 不修改或补猜用户的三条PENDING；
- 不回写E178人工表；
- 不因人工结果重新挑case、reward、grid或collider；
- 不把人工USE直接写成RL成功或绕过S6 evidence治理；
- Claims闭合前不更新为人工终审完成。
