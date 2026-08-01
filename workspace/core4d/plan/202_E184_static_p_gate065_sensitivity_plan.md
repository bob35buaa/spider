# E184 实验计划：Full27 static-P 0.65 precision/recall sensitivity

_Core4D Phase 47 · 2026-08-01 · offline re-aggregation authorized_

## Context

E183在E178 Full27上完成60 candidates、540 candidate-case rows的static-P审计，使用
precision/recall各`>=0.70`。用户要求把TP覆盖放宽到65%，phantom也对应放宽，观察是否有
更多object-specific candidate变得可用。

本实验不重跑SDF、CoACD或Full CEM，只读取E183冻结confusion counts重新聚合。E183的0.70
结论与artifact保持不变，E184作为独立sensitivity结果。

## 冻结gate

对oracle contact count大于0的case：

```text
recall    = TP / (TP + missed)       >= 0.65
precision = TP / (TP + phantom)      >= 0.65
```

因此phantom不是统一绝对数，而是：

```text
phantom <= floor(TP * (1 - 0.65) / 0.65)
        = floor(TP * 0.5384615...)
```

例如TP=`18/19/22`时，phantom最多=`9/10/11`。这比0.70 gate对应的`7/8/9`放宽，且对
不同接触数量保持同一precision语义。

对oracle contact count等于0的case，TP率无定义，采用独立零阳性合同：

```text
ZERO_ORACLE_PASS iff phantom == 0
```

同时保留E183 legacy `recall=0`结果用于exact对照，不能把零阳性语义变化伪装成threshold
收益。Macro precision/recall只对oracle-positive cases计算；zero-oracle通过率单列。

## Claims

| Claim | 最低证据 |
|---|---|
| C0 source | E183 case table SHA exact=`28a3c8...949c`，540 unique rows，60 candidates |
| C1 threshold | protocol在re-aggregation前冻结`precision=recall=0.65`与phantom公式 |
| C2 count integrity | 原TP/phantom/missed/TN逐行不改；pose与oracle totals和E183 exact |
| C3 gate integrity | positive-case gate与zero-oracle gate分别验证；边界TP/phantom单测 |
| C4 comparison | 0.70与0.65逐object pooled/macro/coverage PASS计数及迁移矩阵完整 |
| C5 candidate evidence | 每物体排名、best candidate、case failure原因完整；bucket003/v9单列 |
| C6 visual | 0.70→0.65各object PASS数与case-coverage变化图生成并实际检查 |
| C7 isolation | SDF/CoACD/GPU/Full访问均为0，E183 artifacts SHA不变 |

## 实现与结果路径

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/experiments/E184/reaggregate_gate065.py` | 冻结protocol、重算gate、生成对照表/图/validation |
| `workspace/core4d/scripts/experiments/E184/test_reaggregate_gate065.py` | 边界、零阳性、closure与source tamper合同 |
| `workspace/core4d/scripts/eval/wrappers/eval_E184_gate065.sh` | CPU-only固化入口 |

```text
workspace/core4d/results/E184/static_p_gate065/
  protocol_manifest.json
  case_candidate_metrics_gate065.tsv
  candidate_summary_gate065.tsv
  aggregate.json
  threshold_comparison.png
  validation.json
```

## 执行命令

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E184_gate065.sh
```

## 成功标准与stop/go

- 科学闭合标准是540行与0.70对照完整，不要求bucket003必须出现PASS；
- 若bucket003在0.65下仍无pooled+macro+case coverage候选，Full继续禁止；
- 若只在zero-oracle新语义下增加PASS，必须标为metric-semantics change而非几何改善；
- 若出现新candidate，通过只是P gate sensitivity evidence，R/G与Full仍需另行批准；
- 不访问GPU，不重跑query或distance，不改E183结果。
