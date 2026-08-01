# E185 实验计划：Full27 static-P 0.60 precision/recall sensitivity

_Core4D Phase 48 · 2026-08-01 · offline re-aggregation authorized_

## Context

E184把E183 Full27 static-P gate从precision/recall各`>=0.70`放宽到各`>=0.65`，最终
zero-aware all-case候选仅从`3/60`增至`4/60`，新增candidate只属于bucket004。用户要求
继续检查`0.60`口径是否能解除bucket003或bucket007阻塞。

E185仍是纯离线sensitivity：直接读取E183冻结的540行confusion counts，不从E184派生计数，
不重跑SDF、CoACD、query或Full CEM，不访问GPU。E183/E184 artifact和结论保持不变。

## 冻结gate

对oracle contact count大于0的case：

```text
recall    = TP / (TP + missed)       >= 0.60
precision = TP / (TP + phantom)      >= 0.60
```

对应phantom上限：

```text
phantom <= floor(TP * (1 - 0.60) / 0.60)
        = floor(TP * 2 / 3)
```

例如TP=`18/19/22`时，phantom最多=`12/12/14`。oracle contact count为0时保持E184合同：

```text
ZERO_ORACLE_PASS iff phantom == 0
```

Macro precision/recall只在oracle-positive cases计算；同时保留E183 legacy 0.70和E184
zero-aware 0.65 exact对照。主要比较链为`0.70 → 0.65 → 0.60`，禁止把zero-oracle语义修正
混成阈值收益。

## Claims

| Claim | 最低证据 |
|---|---|
| C0 source | E183两张TSV SHA exact，540 unique rows、60 candidates、27 cases |
| C1 threshold | protocol在结果前冻结P/R=`0.60`、phantom公式和zero-oracle合同 |
| C2 count integrity | TP/phantom/missed/TN逐行不改，pose/oracle totals exact |
| C3 regression | E183 0.70与E184 zero-aware 0.65主要计数exact复现 |
| C4 comparison | 0.70/0.65/0.60逐object pooled、macro+、all-case和最佳coverage完整 |
| C5 candidate evidence | 新晋级candidate、hull数、边界case与remaining failure逐项列出 |
| C6 bucket decision | bucket003/v9、bucket004 K8效率首选、bucket007 14/14分别判定 |
| C7 visual | 三阈值对比图生成、实际查看并记录观察 |
| C8 isolation | SDF/CoACD/GPU/Full访问均为0，E183/E184 artifact不改 |

## 实现与结果路径

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/experiments/E185/reaggregate_gate060.py` | 冻结protocol、三阈值重聚合、表格/图/validation |
| `workspace/core4d/scripts/experiments/E185/test_reaggregate_gate060.py` | 0.60边界、zero-oracle、source tamper、0.65 regression |
| `workspace/core4d/scripts/eval/wrappers/eval_E185_gate060.sh` | CPU-only正式入口 |

```text
workspace/core4d/results/E185/static_p_gate060/
  protocol_manifest.json
  case_candidate_metrics_gate060.tsv
  candidate_summary_gate060.tsv
  aggregate.json
  threshold_comparison_070_065_060.png
  visual_manifest.json
  validation.json
```

## 执行命令

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E185_gate060.sh
```

## 成功标准与stop/go

- 科学闭合要求540行、60 candidates、0.70/0.65 regression与validator全部PASS；
- 若bucket003仍没有9/9 candidate，bucket003 Full继续禁止，不能用pooled/macro替代all-case；
- 若bucket007达到14/14，必须确认不是zero-oracle语义独自产生，并列出0.60新增case；
- 若新晋级candidate hull数更高但已有低hull候选通过，production仍优先低hull方案；
- 0.60只回答P gate sensitivity，不自动批准R/G或Full CEM；
- 不访问GPU，不重跑query/distance，不修改E183/E184结果。
