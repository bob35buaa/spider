# E184 实验日志：static-P precision/recall 0.65 sensitivity

_Core4D Phase 47 · 2026-08-01 · plan
[202](../plan/202_E184_static_p_gate065_sensitivity_plan.md) · 前置
[251](251_E183_full27_static_p_coverage_results.md)_

## 0. 一句话结论

复用E183冻结的`540`行confusion counts，把positive-case precision/recall门槛从`0.70`
同步放宽到`0.65`后，zero-aware全case候选只从`3/60`增至`4/60`。唯一新增的是bucket004
`t005_k16_v064`；bucket003和bucket007仍分别为`0/24`与`0/18`全覆盖。因此放宽口径能
吸收一个边界candidate，但不能解除bucket003/bucket007对统一Full CEM的阻塞。

本实验总wall约`0.84s`，SDF query、CoACD build、GPU访问和Full CEM运行均为0。

## 1. 冻结口径

对oracle-positive case：

```text
recall    = TP / (TP + missed)  >= 0.65
precision = TP / (TP + phantom) >= 0.65
phantom <= floor(TP * 0.35 / 0.65)
```

所以TP=`18/19/22`时允许phantom最多=`9/10/11`。对oracle contact为0的case单列：

```text
ZERO_ORACLE_PASS iff phantom == 0
```

Macro P/R只在oracle-positive cases上计算；E183原始的all-case macro和legacy零阳性失败语义
继续保留作exact对照。全casezero-aware PASS要求：pooled gate PASS、positive-only macro gate
PASS、且每个positive/zero case分别通过自身合同。

protocol在aggregate前冻结，SHA：

```text
0fc6e3d141fca109da0d3c7218624ef24cbb1efaa58b0eeaac013c3a1804c7e5
```

## 2. Authority与闭合

| 项目 | 结果 |
|---|---:|
| E183 source case table | `540` rows，SHA `28a3c8...949c` |
| E183 source candidate table | `60` rows，SHA `08b892...6093` |
| Cases / object分布 | `27`，bucket003/004/007=`9/4/14` |
| Unique poses / oracle contacts | `14542 / 2198` |
| 0.70 legacy regression | pooled/macro/all-case=`23/19/3` exact |
| E184 formal validation | **PASS** |

原TP/phantom/missed/TN逐行不变；本实验没有重新访问mesh、grid、query tape或候选构建。

## 3. 0.70到0.65总体变化

| Gate | 0.70 | 0.65 | 变化 |
|---|---:|---:|---:|
| Pooled P/R | 23 | 27 | +4 |
| Legacy macro（含zero rows） | 19 | 29 | +10 |
| Positive-only macro | 26 | 30 | +4 |
| Legacy all-case | 3 | 4 | +1 |
| Zero-aware all-case | **3** | **4** | **+1** |

`legacy macro 19→29`不能直接解释成纯阈值收益，因为bucket007的两个zero-oracle rows在旧定义
下被记成P/R=0。主要科学口径应看positive-only macro的`26→30`，以及最终zero-aware
all-case的`3→4`。零阳性语义本身没有额外产生全覆盖candidate。

## 4. 分物体结果

| Object | Candidates | Pooled 0.70→0.65 | Macro+ 0.70→0.65 | All-case 0.70→0.65 | 最佳coverage 0.70→0.65 |
|---|---:|---:|---:|---:|---:|
| bucket003 | 24 | 0→1 | 0→2 | **0→0** | 5/9→5/9 |
| bucket004 | 18 | 10→10 | 9→11 | **3→4** | 4/4→4/4 |
| bucket007 | 18 | 13→16 | 17→17 | **0→0** | 13/14→13/14 |

### 4.1 bucket003：阈值放宽不足以解决跨case覆盖

E182-v9 `taskpreseg_v9_both_t005_k16_v256`在0.65下首次同时通过pooled与macro+：

| 指标 | 值 |
|---|---:|
| Actual hulls | 16 |
| Case PASS | 3/9→5/9 |
| Pooled P/R | `0.772 / 0.656` |
| Macro+ P/R | `0.678 / 0.691` |
| All-case | **FAIL** |

四个失败case分别由严重phantom、低recall或完全missed主导：dev case仍为
`TP22/phantom24/missed5`，而0.65只允许phantom 11；另有一个case为`TP0/missed15`。
E181最佳`t020_k16_v032`仍为5/9，pooled precision=`0.645`，甚至略低于0.65。
六个v9 candidate全case仍为`0/6`，所以不能据此启动bucket003 Full。

### 4.2 bucket004：新增一个边界候选，但K8首选不变

唯一新晋级candidate是`t005_k16_v064`：

| 指标 | 值 |
|---|---:|
| Actual hulls | 16 |
| Case PASS | 3/4→4/4 |
| Pooled P/R | `0.830 / 0.919` |
| Macro+ P/R | `0.802 / 0.911` |
| 晋级边界case | `bucket004_20231002_021_p2` |
| 边界case TP/phantom/missed | `37/19/3` |
| 边界case P/R | `0.661 / 0.925` |

这个case在0.70下phantom上限15，0.65下上限19，因而是严格的阈值晋级。原首选
`t005_k08_v064`仍以8 hull稳定4/4通过，pooled=`0.874/0.876`、macro+=`0.859/0.853`；
考虑Full效率，K8仍优于新增K16。

### 4.3 bucket007：zero-aware后接近全覆盖，但仍有真实漏检

K8候选`t020_k08_v064`是唯一13/14：pooled=`0.977/0.976`、macro+=`0.895/0.868`。
它在两个zero-oracle case上phantom均为0，因此zero-aware合同均PASS；这解释了为何它从
E183 legacy视角的11/14变成13/14，但这是度量语义修正，不是几何或阈值改善。

剩余失败case `bucket007_20231003_2_021_p2`只有3个oracle contacts，candidate为
`TP0/phantom0/missed3`，recall=0。0.65不会改变完全漏检，因此bucket007仍无14/14候选。

## 5. 可视化检查

[0.70→0.65对比图](../results/E184/static_p_gate065/threshold_comparison.png)包含三物体
candidate PASS计数和最佳case coverage四个panel，已实际查看：标签、图例、柱高及数值均
清晰，无裁切或渲染异常。

视觉直接显示：

1. bucket003只有pooled/macro边缘增加，all-case与最佳coverage完全不动；
2. bucket004仅all-case增加1个；
3. bucket007 pooled增加3个，但macro+、all-case和最佳coverage均不动。

## 6. Validation与artifact SHA

统一根：`workspace/core4d/results/E184/static_p_gate065/`。

| Artifact | SHA-256 |
|---|---|
| `protocol_manifest.json` | `0fc6e3d141fca109da0d3c7218624ef24cbb1efaa58b0eeaac013c3a1804c7e5` |
| `case_candidate_metrics_gate065.tsv` | `7816dfb9088051a4015089dda2f66d1d7f9dfe19593290eac999788db52f5444` |
| `candidate_summary_gate065.tsv` | `ab9ecb8feaeb0c96d4e453b6a9a76668c8e75440797ea22e806f34e76e6cec82` |
| `aggregate.json` | `e448c960ac723c0bab0de8b6582ddf9d0f8dfda285f70011b939de1dd47f91d1` |
| `threshold_comparison.png` | `c388bfe75e6a6bc3c2af037b1327361fc01e84e0c257058a167e78fd3f95e3bb` |
| `validation.json` | `12bdf8ad9d790c96b88d87e5d20b7ae8d04491bc4861d023afd1224834cf42f4` |

所有validator checks均PASS：source SHA、540/60 closure、unique rows、0.70 regression、
threshold monotonicity、zero-oracle合同、visual存在和isolation闭合。

## 7. 决策

```text
E184_GATE065_SENSITIVITY_COMPLETE
BUCKET003_ALLCASE_0_OF_24
BUCKET004_ALLCASE_4_OF_18_KEEP_K8_PRIMARY
BUCKET007_ALLCASE_0_OF_18_BEST_13_OF_14
E182_V9_ALLCASE_0_OF_6
FULL_CEM_NOT_STARTED
GPU_SDF_COACD_ACCESS_ZERO
```

0.65可以作为第一版较宽松的static-P sensitivity口径，但不能被描述为CoACD问题已经解决。
若要求三bucket统一进入Full，P仍被bucket003和bucket007阻塞；若允许object-specific推进，
bucket004继续使用K8候选最合理。
