# E185 实验日志：static-P precision/recall 0.60 sensitivity

_Core4D Phase 48 · 2026-08-01 · plan
[203](../plan/203_E185_static_p_gate060_sensitivity_plan.md) · 前置
[252](252_E184_static_p_gate065_sensitivity_results.md)_

## 0. 一句话结论

复用E183冻结的540行confusion counts，把positive-case precision/recall继续放宽到`0.60`
后，zero-aware all-case候选从0.65的`4/60`增至`7/60`。新增3个全部属于bucket004；
bucket003和bucket007仍为`0/24`与`0/18`。因此60%扩大了bucket004可选集合，但依然不能
解除三bucket统一进入Full CEM的P阻塞。

正式离线流程总wall=`0.83s`，SDF query、CoACD build、GPU访问和Full CEM运行均为0。

## 1. 冻结口径与回归

对oracle-positive case：

```text
recall    = TP / (TP + missed)  >= 0.60
precision = TP / (TP + phantom) >= 0.60
phantom <= floor(TP * 0.40 / 0.60) = floor(2TP/3)
```

TP=`18/19/22`时phantom上限=`12/12/14`。oracle contact为0时继续使用：

```text
ZERO_ORACLE_PASS iff phantom == 0
```

Macro P/R只在oracle-positive cases计算；全case要求pooled、macro+和每个case均PASS。E185
直接从E183原始counts独立重算三阈值，E184 aggregate只用作0.70/0.65回归authority。

protocol在正式aggregate前冻结，SHA：

```text
2cc0378df68a45760caa01b8b737ec40c545ec70b9d74b2d21078fc08637db08
```

## 2. Authority与闭合

| 项目 | 结果 |
|---|---:|
| E183 case/candidate source | `540 / 60` rows，SHA exact |
| Cases / object分布 | `27`，bucket003/004/007=`9/4/14` |
| Unique poses / oracle contacts | `14542 / 2198` |
| E184 0.70/0.65 primary regression | 6项exact |
| E185 formal validation | **PASS** |

原TP/phantom/missed/TN没有变化，E183/E184 artifacts没有被修改。

## 3. 三阈值总体变化

| Gate | 0.70 | 0.65 | 0.60 |
|---|---:|---:|---:|
| Pooled P/R | 23 | 27 | 35 |
| Positive-only macro | 26 | 30 | 32 |
| Zero-aware all-case | **3** | **4** | **7** |

0.60在pooled层面多放行8个candidate，但最终all-case只多3个，说明大部分增量仍被个别
case失败挡住。不能用`35/60 pooled PASS`代替真正的`7/60 all-case PASS`。

## 4. 分物体结果

| Object | Candidates | Pooled 0.70→0.65→0.60 | Macro+ 0.70→0.65→0.60 | All-case 0.70→0.65→0.60 | 最佳coverage |
|---|---:|---:|---:|---:|---:|
| bucket003 | 24 | 0→1→3 | 0→2→2 | **0→0→0** | 5/9→5/9→5/9 |
| bucket004 | 18 | 10→10→14 | 9→11→12 | **3→4→7** | 4/4→4/4→4/4 |
| bucket007 | 18 | 13→16→18 | 17→17→18 | **0→0→0** | 13/14→13/14→13/14 |

### 4.1 bucket003：60%仍不改变5/9上限

最佳coverage仍由E181`t020_k16_v032`和E182-v9
`taskpreseg_v9_both_t005_k16_v256`并列5/9。v9 K16的pooled/macro+已PASS，但四个case仍失败：

| Failure类型 | TP/phantom/missed | P/R |
|---|---:|---:|
| dev严重phantom | `22/24/5` | `0.478/0.815` |
| 低recall | `37/5/61` | `0.881/0.378` |
| 完全missed | `0/0/15` | `0/0` |
| recall略低于0.60 | `51/6/36` | `0.895/0.586` |

其中dev case在0.60下phantom上限仅14，实际24；另两个case远离边界。六个v9全case仍
`0/6`。因此继续小幅降低统一阈值不能解决bucket003的几何失真。

### 4.2 bucket004：新增3个候选，包含一个K8

0.65→0.60新晋级：

| Candidate | Actual hulls | Case 0.65→0.60 | Pooled P/R | Macro+ P/R |
|---|---:|---:|---:|---:|
| `t010_k16_v064` | 16 | 2/4→4/4 | `0.709/0.950` | `0.699/0.941` |
| `t020_k08_v064` | **8** | 3/4→4/4 | `0.839/0.881` | `0.812/0.880` |
| `t020_k16_v064` | 16 | 3/4→4/4 | `0.780/0.926` | `0.763/0.917` |

共同边界case `bucket004_20231002_021_p2`的TP=`37`，0.65/0.60 phantom上限从`19→24`；
两个t020候选实际phantom=`22`、precision=`0.627`，属于明确的阈值晋级。t010 K16还额外
救回一个precision=`0.648`的case。

0.60共有7个bucket004全覆盖candidate，其中3个是8 hull。不过原首选`t005_k08_v064`
仍有更高pooled/macro+下界，且从0.70起就稳定4/4，通过效率与裕量考虑仍应保持首选。

### 4.3 bucket007：所有candidate pooled/macro+通过，仍无14/14

0.60下bucket007的18个candidate全部通过pooled，18个也全部通过macro+，但all-case仍为0。
最佳K8`t020_k08_v064`保持13/14；唯一失败case
`bucket007_20231003_2_021_p2`只有3个oracle contacts，candidate为
`TP0/phantom0/missed3`、recall=0。阈值从0.70降到0.60不会改变完全漏检。

这正好证明all-case gate的必要性：pooled/macro可以掩盖稀有但真实的接触漏检。

## 5. 可视化与实际观察

[0.70/0.65/0.60对比图](../results/E185/static_p_gate060/threshold_comparison_070_065_060.png)
包含三物体candidate PASS数和最佳case coverage四个panel，已实际查看。图例、柱高、标签和
coverage标注清晰，无裁切或渲染异常。

实际观察：

1. bucket003在pooled层面0→1→3，但all-case和5/9上限完全不动；
2. bucket004 all-case形成3→4→7的单调增量；
3. bucket007 pooled/macro+在0.60达到18/18，但all-case仍为0且coverage固定13/14；
4. 绿色0.60柱只在bucket004 all-case panel产生实质增量。

## 6. Artifact SHA与validator

统一根：`workspace/core4d/results/E185/static_p_gate060/`。

| Artifact | SHA-256 |
|---|---|
| `protocol_manifest.json` | `2cc0378df68a45760caa01b8b737ec40c545ec70b9d74b2d21078fc08637db08` |
| `case_candidate_metrics_gate060.tsv` | `23da6299cf1f817dba79ea8d12e7f883626d06ada9a0956510c11ba14bd4db44` |
| `candidate_summary_gate060.tsv` | `2a1b4428e9898462cbd7c46983803adc09d077d7493a4b6ece33896ff90f95c4` |
| `aggregate.json` | `8606f28768e5101a390f9597006d1f3791333cc5073111c422f09afce78cac8c` |
| `threshold_comparison_070_065_060.png` | `e732e5d305542b35ec44fa7727ba19239773de8d31519723668046e7d7b66dc6` |
| `validation.json` | `361c17d2e6c7f780a032d17cbc76db7a69cfbc7ec79c9853b57904d9114f0e7a` |

validator的source SHA、540/60 closure、E184 regression、threshold monotonicity、zero-oracle、
visual和isolation checks全部PASS。

## 7. 决策

```text
E185_GATE060_SENSITIVITY_COMPLETE
ALLCASE_070_065_060=3_4_7_OF_60
BUCKET003_ALLCASE_0_OF_24_BEST_5_OF_9
BUCKET004_ALLCASE_7_OF_18_KEEP_T005_K8_PRIMARY
BUCKET007_ALLCASE_0_OF_18_BEST_13_OF_14
E182_V9_ALLCASE_0_OF_6
FULL_CEM_NOT_STARTED
GPU_SDF_COACD_ACCESS_ZERO
```

60%进一步说明bucket003/bucket007的失败不是“0.65略严”造成的。若要求三bucket统一推进，
P gate仍不满足；若按object-specific推进，bucket004仍可用原8-hull首选进入后续R/G与效率
验证，但这需要独立批准，不能由本次离线P sensitivity自动触发。
