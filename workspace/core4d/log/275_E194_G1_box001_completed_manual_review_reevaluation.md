# E194 G1 box001 完整人审续评：专项改善明确，但不构成全面提升

_Core4D · Phase 57 follow-up · 2026-08-12 · 基于 [authority correction 274](274_E194_PRG_box001_authority_correction.md) 与已完成的 E194 G1 28-case 人审_

## Summary

- G1 box001 人审已闭合：28/28 reviewed，`USE=15 / DO_NOT_USE=13`；质量为 `CLEAN=10 / MINOR_ACCEPTABLE=5 / UNUSABLE=13`。
- 按先前约定排除 `box001_20231023_110_p1` 后，PRG→G1 人审迁移为 `USE→USE=8、USE→DNU=5、DNU→USE=7、DNU→DNU=7`。USE 率由 `13/27=48.1%` 到 `15/27=55.6%`，净 `+7.4 pp`，但 exact McNemar `p=0.774414`，不支持稳定的总体人工通过率提升。
- 数值专项收益仍强：object z `−1.225 cm`、3D position `−1.275 cm`、hand penetration `−0.080`、leg penetration `−0.048`；同时 object orientation `+2.516°`、raw contact `−0.074`，说明 G1 改善重力/穿透问题时引入了新的姿态与接触回退。
- 5 个 `PRG USE→G1 DNU` case 的 z 仍平均改善 `−1.185 cm`，但 object orientation 平均恶化 `+8.485°`、raw contact 下降 `−0.230`。因此单看 z 或 penetration 会误判这些人审回退。
- 最终结论：`NOT_COMPREHENSIVE_IMPROVEMENT`。不建议用 G1 全量替换 PRG；建议 `CASE_LEVEL_PRG_G1_SELECTION`，保留 5 个 PRG-only case，并采用 7 个 G1-only recovery。

## 1. Experiment Motivation

昨日的续评只有 14 个 G1 box001 标签，无法判断 G1 的人审可用率是否真正超过 PRG。现在 28 个 G1 case 已全部标注，本次分析回答三个问题：

1. G1 是否在人审 USE 率上全面优于 PRG；
2. 人审翻转是否与数值 tracking/contact/penetration 指标一致；
3. 应全量切换 G1，还是保留 case 级版本选择。

## 2. Experiment Setup

| 项目 | 设置 |
|---|---|
| PRG 人审 authority | E173 `box001_user_approved_source_rows.tsv`；13 个成员=USE，E194 box001 universe 内其余=DO_NOT_USE |
| G1 人审 authority | E194 `full_g1_expansion/user_manual_review_filled.tsv`；28/28 reviewed |
| 主分析范围 | 排除 `box001_20231023_110_p1` 后的 27 case |
| 敏感性范围 | 全部 28 case |
| 数值对照 | E194 `PRG_to_G1` 同 case paired metrics；14 metrics + 12 gates |
| Bootstrap | seed 194 起、每指标 20,000 次 paired-case resampling |
| 人审显著性 | 双侧 exact McNemar；discordant pair 为 USE→DNU 与 DNU→USE |

Source SHA256：

- PRG authority：`d19af1aafeeb20f9030f4d51f4141bf05727c675ab84c50913eee5eb488f6590`
- G1 review：`6ce833adada65b68221a8da62dd30740e36aff206baeaf6d1e365ca14f91548c`
- Paired metrics：`29e0181924fba401b906e9a2fcd9e24d68b388d3355652201159d5d3bb14ccd7`

## 3. Core Algorithm or Method

本次没有新算法、训练或 CEM rollout。方法是对同一 case 的两个版本做 evidence join：PRG decision 来自最终 RL-export membership authority，G1 decision/quality 来自完整人审 TSV，数值 delta 定义为 `G1−PRG`。低误差/低穿透指标的负 delta 表示改善；contact 指标的正 delta 表示改善。

## 4. Metrics

| 指标 | 方向 | 单位/聚合 | 作用 |
|---|---|---|---|
| Manual USE rate | 越高越好 | case 比例 | 最终人工可用性 |
| Manual migration | DNU→USE 有利；USE→DNU 不利 | paired case count | 检查净收益与 churn |
| Object z / 3D error | 越低越好 | case mean 后再跨 case mean，cm | G1 主目标 |
| Object orientation error | 越低越好 | degree | 检查姿态副作用 |
| 3mm/raw contact | 越高越好 | frame fraction | 检查接触保持 |
| Hand/leg penetration | 越低越好 | frame fraction | 检查碰撞安全 |
| 12 gates | pass 越多越好 | paired pass count | 结构化失败模式 |

## 5. Results

### 5.1 完整人审迁移

主分析 27 case：

| PRG \ G1 | G1 USE | G1 DNU | 合计 |
|---|---:|---:|---:|
| PRG USE | 8 | 5 | 13 |
| PRG DNU | 7 | 7 | 14 |
| 合计 | 15 | 12 | 27 |

| 汇总 | 结果 |
|---|---:|
| PRG USE | 13/27（48.1%） |
| G1 USE | 15/27（55.6%） |
| 净变化 | +2 case（+7.4 pp） |
| Agreement | 15/27（55.6%） |
| Churn | 12/27（44.4%） |
| Exact McNemar p | 0.774414 |
| PRG/G1 USE union | 20/27（74.1%，post-hoc upper bound） |

全 28 case 敏感性为 `8/5/7/8`，PRG/G1 USE=`13/15`，McNemar p 同为 `0.774414`。被排除 case 是 `DNU→DNU`，所以排除与否不改变方向或结论。

### 5.2 主数值指标

| 指标 | PRG | G1 | G1−PRG | Bootstrap 95% CI | Better/Worse/Tie |
|---|---:|---:|---:|---:|---:|
| Object z MAE (cm) | 4.887 | 3.663 | **−1.225** | [−1.551, −0.859] | 25/2/0 |
| Object 3D pos (cm) | 11.379 | 10.104 | **−1.275** | [−1.874, −0.698] | 23/4/0 |
| Object orientation (deg) | 5.944 | 8.460 | **+2.516** | [+0.362, +5.165] | 16/11/0 |
| Raw contact frac | 0.818 | 0.744 | **−0.074** | [−0.129, −0.023] | 7/17/3 |
| Hand penetration frac | 0.220 | 0.140 | **−0.080** | [−0.122, −0.037] | 19/7/1 |
| Leg penetration frac | 0.076 | 0.028 | **−0.048** | [−0.093, −0.012] | 9/1/17 |

### 5.3 关键 gate 变化

| Gate | PRG | G1 | Delta | P→F / F→P | Exact p |
|---|---:|---:|---:|---:|---:|
| contact | 27/27 | 25/27 | −7.4 pp | 2 / 0 | 0.500000 |
| hand penetration | 20/27 | 26/27 | **+22.2 pp** | 0 / 6 | **0.031250** |
| lower body | 18/27 | 24/27 | **+22.2 pp** | 0 / 6 | **0.031250** |
| hand orientation | 18/27 | 15/27 | −11.1 pp | 4 / 1 | 0.375000 |
| object orientation | 27/27 | 22/27 | −18.5 pp | 5 / 0 | 0.062500 |

### 5.4 人审翻转的数值特征

| 人审组 | n | Δz cm | Δ3D cm | Δobj ori deg | Δ3mm contact | Δraw contact | Δhand pen | Δleg pen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| USE→DNU | 5 | −1.185 | −0.003 | **+8.485** | **−0.107** | **−0.230** | −0.102 | −0.014 |
| DNU→USE | 7 | −1.076 | −1.366 | −0.271 | +0.027 | +0.016 | −0.008 | −0.083 |
| USE→USE | 8 | −1.397 | −0.881 | −0.116 | +0.154 | −0.045 | −0.126 | +0.000 |
| DNU→DNU | 7 | −1.204 | −2.544 | +4.046 | +0.034 | −0.086 | −0.085 | −0.094 |

5 个 G1 人审回退 case：

- `box001_20231003_1_039_p1`
- `box001_20231003_1_041_p1`
- `box001_20231003_2_041_p1`
- `box001_20231020_014_p2`
- `box001_20231023_107_p2`

7 个 G1 人审恢复 case：

- `box001_20231003_1_040_p1`
- `box001_20231003_1_041_p2`
- `box001_20231003_2_038_p2`
- `box001_20231020_011_p2`
- `box001_20231023_107_p1`
- `box001_20231023_109_p1`
- `box001_20231023_110_p2`

## 6. How to Read the Tables

所有连续指标 delta 都是 `G1−PRG`。误差和 penetration 越低越好，因此负值为改善；contact 越高越好，因此正值为改善。`Better/Worse/Tie` 已按各指标方向换算，而不是简单按 delta 正负计数。人审矩阵的非对角项代表版本 churn；McNemar 只比较两个非对角项是否明显不对称。

## 7. Interpretation

G1 的机制性收益是真实且稳定的：object z/3D 和 penetration/lower-body 都改善，且 z 有 25/27 case 改善。但人审结果表明这些收益不能代表整体动作质量。5 个 `USE→DNU` case 的核心特征是 object orientation 和 contact 大幅回退，即 G1 把原有重力/穿透问题转成了新的姿态/接触问题。

`DNU→USE=7` 略多于 `USE→DNU=5`，所以 G1 USE 数净增 2；但 discordant 差异很小、p=0.774，且 churn 达 44.4%。因此证据支持“G1 与 PRG 互补”，不支持“G1 单调支配 PRG”。

## 8. Conclusion and Discussion

最终判定：`NOT_COMPREHENSIVE_IMPROVEMENT`。

推荐策略：`CASE_LEVEL_PRG_G1_SELECTION`。

- 8 个双方 USE case 可优先采用 G1，以获得更好的 z/penetration；
- 7 个 G1-only recovery 采用 G1；
- 5 个 PRG-only case 保留 PRG，避免 orientation/contact 回退；
- 7 个双方 DNU case 不进入可用集合。

上述 post-hoc union 为 `20/27=74.1%`，明显高于任一单版本，但这是基于人工终审的选择上限，不等同于已经存在可自动部署的 selector。

## 9. Limitations and Caveats

- G1 的 `manual_failure_taxonomy` 在 28 行中全为空，`manual_review_note` 仅 2/28 非空（`011_p1`“感觉还ok”、`014_p1`“接触位置不对，摔倒”）；显式原因标注过稀，分组 failure mechanism 主要由数值指标与 quality label 推断。
- 人审是单 reviewer、单批次结果；McNemar 未显示总体 USE 率显著变化。
- case-level union 是 post-hoc 人工选择，不应被表述为自动算法收益。
- 本结论只适用于 E194 box001 冻结 case/variant/config，不外推到 box023、box021 或 RL policy 成功率。

## 10. Next Steps

1. 若要产出最终可用资产，按 20-case union 构建带明确 `selected_arm=PRG/G1` 的审核 manifest；不要覆盖原始 PRG/G1 evidence。
2. 为 5 个 G1 回退 case 补 failure taxonomy，重点区分 object orientation、grasp/contact loss 与其他视觉原因。
3. 若希望自动选择，使用这 27 个 paired 标签评估一个只做 ranking 的 soft selector，并进行 leave-one-session-out 验证；在稳健性成立前不设 hard gate。

## Reproducibility Notes

```bash
.venv/bin/python workspace/core4d/scripts/eval/reports/analyze_E194_box001_manual_review.py
.venv/bin/python workspace/core4d/scripts/eval/reports/build_E194_three_arm_workbook.py
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/E194_noPRG_PRG_G1_comparison.xlsx
```

| Artifact | Path / SHA256 |
|---|---|
| Case comparison TSV | `results/E194/s6_downstream/eval/full_g1_expansion/e194_box001_manual_review_case_comparison.tsv` / `d85f0d088d492aa3ff971f6628f897a3ac21cfd170819403a3a8557b2bc23012` |
| Summary JSON | `results/E194/s6_downstream/eval/full_g1_expansion/e194_box001_manual_review_summary.json` / `a8426186a2368f7822cabfcfbae31c67107e786d12cf92859e961aa765448a1a` |
| Updated XLSX | `results/E194/s6_downstream/eval/full_g1_expansion/E194_noPRG_PRG_G1_comparison.xlsx` / `16d501adc6ff12694c918b212cd5f819f002a945f3f09ef24e6d9321b9655805` |
