# E187 vs E178 22-case Paired Evaluation 结果

日期：2026-08-03
实验：E187
阶段：Evaluation
结论：Evaluation 22/22闭合；E187 12门 `6/22`，低于同case E178 `8/22`；主要回归为lower-body penetration

## Context

E187 已在 C9 technical `FAIL`、progression authority `USER_WAIVED` 的治理前提下完成
Full CEM 22/22。本阶段执行 [plan 208](../plan/208_E187_vs_E178_paired_evaluation_plan.md)：
只对 keep22 做同 `case_id` 的 E187 vs E178 paired evaluation，不修改 E178 历史结果，
不回调 E187 reward/grid/P/G/Full artifact。

## 执行合同

| 项目 | E178 | E187 |
|---|---:|---:|
| paired rows | 22 | 22 |
| CEM budget | `1024×32 seed0` | `1024×32 seed0` |
| metric standard | `core4d-e154-physics-contact-v1` | 同左 |
| numeric gates | fall/body-z/contact/release/hand-pen/lower-body + 6 tracking gates | 同左 |
| tracking thresholds | root/hand/object pos `20cm`；root/hand ori `20°`；object ori `10°` | 同左 |
| C9 technical / authority | N/A | `FAIL / USER_WAIVED` |

Canonical commands：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E187_vs_E178_full.sh preflight
bash workspace/core4d/scripts/eval/wrappers/eval_E187_vs_E178_full.sh run
python workspace/core4d/scripts/eval/reports/gen_E187_vs_E178_xlsx.py
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py workspace/core4d/results/E187/s6_downstream/eval/full/E187_vs_E178_paired_evaluation.xlsx 60
```

## Evaluation 完整性

| 项目 | 结果 |
|---|---:|
| manifest / expected | `22 / 22` |
| evaluated / paired | `22 / 22` |
| not ready / errors / missing baseline | `0 / 0 / 0` |
| Full / promoted canary | `19 / 3` |
| E187 numeric pass | `6/22` |
| E178 numeric pass（同keep22） | `8/22` |
| pass-count delta | **`-2`** |
| leg gate health diagnostic | `0/22` |

`leg_gate_health_pass` 是额外诊断，不属于冻结的12门 numeric decision；不能把它与
`numeric_release_pass`混为一谈。

## Paired 结果

### Numeric transition

| E178 → E187 | 数量 |
|---|---:|
| FAIL → FAIL | 11 |
| PASS → FAIL | **5** |
| FAIL → PASS | 3 |
| PASS → PASS | 3 |

按object：

| object | E178 pass | E187 pass | delta |
|---|---:|---:|---:|
| bucket003 | 1/5 | 0/5 | -1 |
| bucket004 | 2/4 | 3/4 | +1 |
| bucket007 | 5/13 | 3/13 | -2 |

### Gate pass counts

| gate | E178 | E187 | improved | regressed |
|---|---:|---:|---:|---:|
| fall | 21 | 21 | 0 | 0 |
| body_z | 21 | 19 | 0 | 2 |
| contact | 19 | 21 | 2 | 0 |
| release | 19 | 19 | 2 | 2 |
| hand_penetration | 18 | 16 | 2 | 4 |
| **lower_body** | **19** | **8** | **0** | **11** |
| root_pos | 13 | 15 | 2 | 0 |
| root_ori | 18 | 19 | 2 | 1 |
| hand_pos | 12 | 17 | 5 | 0 |
| hand_ori | 11 | 11 | 3 | 3 |
| object_pos | 22 | 22 | 0 | 0 |
| object_ori | 20 | 22 | 2 | 0 |

五条 PASS→FAIL case 全部新增 `lower_body` failure；其中
`bucket003_20231018_001_p2` 还新增 `hand_ori`。因此净pass下降的直接主因不是接触或
tracking，而是 CoACD compound / grid-gated continuation 下的下肢穿透回归。

### 关键metric均值

正的 improvement 始终表示 E187 更好；lower-is-better 指标使用 `E178-E187`。

| metric | E178 mean | E187 mean | improvement |
|---|---:|---:|---:|
| contact in mask | 0.6824 | 0.7405 | **+0.0581** |
| release false 3mm（n=20） | 0.1096 | 0.1386 | -0.0290 |
| hand penetration 3mm | 0.1674 | 0.2109 | -0.0435 |
| **leg penetration** | **0.0221** | **0.1739** | **-0.1519** |
| root position error (cm) | 20.5958 | 16.4628 | **+4.1330** |
| hand position error (cm) | 18.5964 | 15.8820 | **+2.7143** |
| object position error (cm) | 10.6621 | 10.5366 | +0.1254 |
| object orientation error (deg) | 5.7721 | 6.3810 | -0.6088 |
| foot slip max (m) | 0.7067 | 0.7993 | -0.0925 |

结论是明确trade-off：E187增加contact并改善root/hand位置跟踪，但lower-body、hand
penetration、release和foot slip回归；冻结的12门总结果不支持“E187优于E178”。

## Excel 交付物

工作簿 `E187_vs_E178_paired_evaluation.xlsx` 包含：

1. Overview；
2. E187 Metrics；
3. E178 Baseline；
4. Paired Comparison（E178/E187/delta/improvement公式）；
5. Object Summary；
6. Gate Transitions；
7. Best Improvements；
8. Worst Regressions；
9. Artifact Provenance。

格式采用Arial、冻结表头、筛选、条件着色；LibreOffice重算结果为
`status=success`、`total_errors=0`、公式扫描`3621`。

## 可视化 → 实际观察

使用 `video-frames` 对3类代表case各抽取 E178/E187 的2s中帧，共6张：

- improved `bucket004_20231002_021_p1`：两侧ref/sim均可读；E187 sim在中帧的物体
  位置与手部姿态相对E178有明显变化，但无画面崩坏；数值由5项失败变为12门全过；
- regressed `bucket003_20231018_001_p2`：两侧均是人体处于桶体内部/边缘的高风险
  构型；E187足部接触高亮与E178不同，与新增lower-body/hand-orientation失败方向一致；
- stable-pass `bucket004_20231003_1_012_p2`：两侧站姿和物体相对位置总体相近，E187
  sim保持直立。

这只是中帧可读性与明显姿态spot-check，不替代完整时序数值门或用户人工终审。

## Claims 验证

| Claim | 状态 | 证据 |
|---|---|---|
| E1 配对完整性 | PASS | 22/22一一配对；missing/duplicate/unexpected=0 |
| E2 指标可比性 | PASS | 同公共evaluator、metric standard与阈值；E178 artifact未改 |
| E3 结果闭合 | PASS | case/group/paired/summary全部生成；errors=0 |
| E4 Excel可审计 | PASS | 9 sheets，22行paired与完整provenance |
| E5 工作簿可靠 | PASS | Arial；公式重算；error=0 |
| E6 治理状态不漂移 | PASS | C9始终`FAIL / USER_WAIVED` |

## 结果路径与 SHA256

| 内容 | 路径 | SHA256 |
|---|---|---|
| evaluation manifest | `workspace/core4d/results/E187/s6_downstream/manifests/e187_full_evaluation_manifest.tsv` | `507c3a79766e61b49d5e298432bd0a766d9566f931ec9a2382f1a5b6e82c6fc7` |
| E187 case metrics | `workspace/core4d/results/E187/s6_downstream/eval/full/e187_case_metrics.tsv` | `77405ebd72a02c7135d3b28945ffcf65e1fa2c0e2394bbba687e3c699000cb06` |
| paired deltas | `workspace/core4d/results/E187/s6_downstream/eval/full/e187_vs_e178_paired_deltas.tsv` | `0ae0b10f370cd9c31d03ae8aea40ea9a7008e11de6214e5405b5ad4323685715` |
| group summary | `workspace/core4d/results/E187/s6_downstream/eval/full/e187_group_summary.tsv` | `5671d9d1572cf4df35d46b0bdb86ef4fa365464e5f4c4316c08178f4dec8f359` |
| summary | `workspace/core4d/results/E187/s6_downstream/eval/full/summary.json` | `038d9778d197f8cfc17ddff53614fa7e8ead8ae2e53be9667436f4313b8246e8` |
| Excel | `workspace/core4d/results/E187/s6_downstream/eval/full/E187_vs_E178_paired_evaluation.xlsx` | `92bf74311bc76173bba6a93a3e4d949f5ceb2e6b825efac1226bde9db93f8389` |
| visual QC | `workspace/core4d/results/E187/s6_downstream/eval/full/visual_qc/` | 6 readable JPEGs |

## 下一步

本阶段任务已完成。若要继续科学迭代，应新开实验解决lower-body penetration与foot-slip
trade-off；不得在E187 Full结果后回调已冻结的reward/grid或把C9 waiver改写为technical
PASS。若只做数据选择，应优先使用workbook的Gate Transitions和paired case sheet，由用户
决定是否对3个FAIL→PASS case或其它片段做人工终审。
