# E183 实验日志：Full27 object-specific CoACD static-P coverage

_Core4D Phase 46 · 2026-08-01 · plan
[201](../plan/201_E183_full27_static_p_coverage_audit_plan.md) · 前置
[250](250_E182_bucket003_double_plane_preseg_v9_results.md)_

## 0. 一句话结论

E178全部27 case的reference + E178-final static-P审计已闭合：`14542 poses`、
`60 candidates`、`540/540 candidate-case rows`，4 CPU worker score wall=`51.27s`，
GPU访问为0。结果证明CoACD不是普遍失效，但强烈object-specific：

- **bucket003：0个可用候选**，最佳仅`5/9 case PASS`，pooled precision=`0.645`；
- **bucket004：3个全4/4 PASS**，最佳是仅8 hull的`t005_k08_v064`；
- **bucket007：0个全14/14 PASS**，但最佳达到`11/14`且macro/pooled gate均PASS；
- E182-v9六个bucket003 candidate在full9上仍全部失败，最好只有`3/9`。

因此单case证据确实不够：它低估了bucket004 CoACD的可用性，同时无法定位bucket003的
系统性phantom和bucket007的少数严重case。由于完整27-case production链仍被bucket003
阻塞，本实验不启动Full CEM，也不根据full27结果回头修改plane/K/threshold。

## 1. Authority、候选与闭合

| 项目 | 结果 |
|---|---:|
| E178 authority | 27 unique cases，bucket003/004/007=`9/4/14` |
| Query | 全reference + E178-final，object-local P points |
| Pose / oracle-contact | `14542 / 2198` |
| E181标准CoACD | `54`（18/object） |
| E182-v9附加candidate | `6`（bucket003 only） |
| Candidate-case rows | `540/540`，missing/duplicate=`0/0` |
| v9 dev regression | `6/6 exact PASS` |
| Formal validation | **PASS** |

protocol在任何full27 score前冻结，SHA：

```text
7ed82ada0dc2675dac17e3cc435de45fae79495b86a5e8ed06661c8e807d172f
```

本实验有意把原heldout24纳入coverage evaluation；所有60个candidate及评分代码在访问结果
前已冻结。结果只能说明full27覆盖，不能用于调参后再次声称独立heldout验证。

## 2. 总体结果

| Gate层级 | PASS / 60 | 含义 |
|---|---:|---|
| Pooled precision/recall各≥0.70 | 23 | 所有同物体pose合并 |
| Macro precision/recall各≥0.70 | 19 | 每case等权后平均 |
| All-case coverage | **3** | pooled+macro通过且每个case均通过 |

### 2.1 分物体

| Object | Candidates | Pooled PASS | Macro PASS | All-case PASS | 最佳case覆盖 |
|---|---:|---:|---:|---:|---:|
| bucket003 | 24 | 0 | 0 | **0** | 5/9 |
| bucket004 | 18 | 10 | 9 | **3** | 4/4 |
| bucket007 | 18 | 13 | 10 | **0** | 11/14 |

## 3. Object-specific结果

### 3.1 bucket003：系统性precision失败

全24个candidate均未通过pooled或macro gate。综合最佳为E181
`t020_k16_v032`：

| 指标 | 值 |
|---|---:|
| actual hulls | 16 |
| case PASS | 5/9 |
| macro precision / recall | `0.653 / 0.865` |
| pooled precision / recall | `0.645 / 0.863` |

它在不同case上的precision从`0.294`到`0.874`变化很大；
`bucket003_20231018_005_p1`为`TP25/phantom60/missed6`。失败主因不是统一recall不足，
而是凹腔/rim envelope在部分姿态产生大量phantom。

E182-v9 full9结果：

| Candidate | Case PASS | Macro P/R | Pooled P/R |
|---|---:|---:|---:|
| t005/K16 | 3/9 | `0.678/0.691` | `0.772/0.656` |
| t005/K32 | 2/9 | `0.430/0.706` | `0.282/0.691` |
| t010/K16 | 1/9 | `0.551/0.698` | `0.554/0.624` |
| t010/K32 | 1/9 | `0.404/0.733` | `0.239/0.707` |
| t020/K16 | 1/9 | `0.528/0.748` | `0.535/0.769` |
| t020/K32(actual24) | 1/9 | `0.503/0.748` | `0.455/0.769` |

v9六行在原dev case的TP/phantom/missed与log250 exact，但扩到另外8 case后没有候选接近
全覆盖。这进一步支持停止继续手工plane切分。

### 3.2 bucket004：K8已经足够

三个全4/4通过候选：

| Candidate | Actual hulls | Macro P/R | Pooled P/R | TP/phantom/missed |
|---|---:|---:|---:|---:|
| **t005_k08_v064** | **8** | `0.859/0.853` | `0.874/0.876` | `368/53/52` |
| t010_k08_v064 | 8 | `0.853/0.843` | `0.875/0.867` | `364/52/56` |
| t020_k32_v064 | 19 | `0.815/0.902` | `0.765/0.924` | `388/119/32` |

首选`t005_k08_v064`的最差case仍为precision/recall=`0.767/0.752`，因此不是长case
pooled平均掩盖失败。K8优于更大hull budget，说明对bucket004继续提高max hulls没有必要，
也支持后续把效率作为production选择维度。

### 3.3 bucket007：大部分case可用，但存在两类失败

综合case覆盖最佳`t005_k16_v064`：case PASS=`11/14`，macro=`0.765/0.838`，
pooled=`0.746/0.996`。K8的`t020_k08_v064`同为`11/14`，macro=`0.767/0.744`，
pooled=`0.977/0.976`；若以后只在通过集合中做效率选择，K8是重要Pareto候选。

K16最佳的三个FAIL case：

| Case | Oracle contact | TP/phantom/missed | P/R | 解释 |
|---|---:|---:|---:|---|
| `20231003_1_021_p1` | 0 | `0/166/0` | `0/0` | 真实严重phantom，不是纯指标边界 |
| `20231003_2_023_p1` | 0 | `0/0/0` | `0/0` | candidate与oracle均无接触；recall无正例时定义为0 |
| `20231023_075_p2` | 31 | `31/219/0` | `0.124/1.0` | 真实严重phantom |

所以bucket007的all-case失败由“零阳性case度量约定”和“局部姿态严重过碰撞”共同造成，
不能简单把3个FAIL全部归咎于metric，也不能用pooled高分掩盖。

## 4. 效率

| Stage | Workers | Wall | 其他 |
|---|---:|---:|---|
| Query build | 4 CPU | `4.820s` | max RSS `2038.6MiB`，275MiB tape |
| Candidate score | 4 CPU | `51.267s` | candidate wall sum `187.215s`，max RSS `993.6MiB` |
| Visual + validation | CPU | `7.0s` | 6图，validator PASS |

实测证明static-P full27无需GPU；4-worker在不干预现有任务的情况下不到1分钟完成核心540行
评分。CoACD构建仍是CPU任务；Full CEM和R/G CEM query capture才需要GPU。

## 5. 可视化与实际观察

每个物体均生成一份3D hull/oracle/query图和一份2D接触时间线：

- bucket003：[3D](../results/E183/full27_static_p/visual/bucket003_representative_3d.png) ·
  [2D](../results/E183/full27_static_p/visual/bucket003_representative_2d.png)
- bucket004：[3D](../results/E183/full27_static_p/visual/bucket004_representative_3d.png) ·
  [2D](../results/E183/full27_static_p/visual/bucket004_representative_2d.png)
- bucket007：[3D](../results/E183/full27_static_p/visual/bucket007_representative_3d.png) ·
  [2D](../results/E183/full27_static_p/visual/bucket007_representative_2d.png)

实际观察：

1. bucket003在reference早段和E178-final后段均出现成段phantom，final接触切换附近另有少量
   missed，视觉与precision主导失败一致；
2. bucket004总体跟随oracle接触区间，reference段有零散missed、final段有离散phantom，
   但最差case仍稳定越过0.70；
3. bucket007零oracle worst case的candidate在两段轨迹都产生密集接触，166个phantom在
   时间线上清晰可见，确认是真实过碰撞。

## 6. Claims验证

| Claim | 结果 |
|---|---|
| C0 authority | **PASS** — 27 unique，9/4/14，输入SHA闭合 |
| C1 candidate immutability | **PASS** — 60/60，object隔离，pre-score protocol frozen |
| C2 static query closure | **PASS** — 27/27，14542 poses，non-finite=0 |
| C3 regression | **PASS** — v9六行TP/phantom/missed exact |
| C4 score closure | **PASS** — 540/540，missing/duplicate=0 |
| C5 metric integrity | **PASS** — confusion与pooled求和一致，macro独立报告 |
| C6 runtime | **PASS** — 4 CPU workers，GPU access=0，runtime/RSS完整 |
| C7 evidence | **PASS** — object ranking、case matrix、3D/2D与实际观察完整 |

## 7. 结果路径与SHA

统一根：`workspace/core4d/results/E183/full27_static_p/`（总计277MiB）。

| Artifact | SHA-256 |
|---|---|
| `protocol_manifest.json` | `7ed82ada0dc2675dac17e3cc435de45fae79495b86a5e8ed06661c8e807d172f` |
| `query_aggregate.json` | `be891a41b76e3a7cfba6e3a801dfe6faf2f98bd9c9a89b78707328808a3c61c8` |
| `score_runtime.json` | `2aa0e0f40d8f11341e49959197242cbbe7d89d0ac280ce7645be894de8459c42` |
| `case_candidate_metrics.tsv` | `28a3c8118fe10767856b5d6bcdb93430dc9ba629d7f4bb58eead9924168a949c` |
| `candidate_summary.tsv` | `08b8927ccc0513b8ff6dc20bf36f83526c9cf265472fd6bf441107aa21766093` |
| `aggregate.json` | `a665b7f4c2cf27e7790f922fe7e16a997435a2fdf532e88f60981d7ab4f70962` |
| `validation.json` | `1ea70d78137455a8a7e382ef07d302cad17f5942e65ae71cf444a1519349a5bb` |

## 8. 决策

```text
E183_FULL27_STATIC_P_AUDIT_COMPLETE
BUCKET003_BLOCKED_0_OF_24
BUCKET004_VIABLE_3_OF_18
BUCKET007_MIXED_0_COVERAGE_BUT_11_OF_14
FULL_CEM_NOT_STARTED
GPU_ACCESS_ZERO
```

后续如继续：bucket004可以把K8作为physics throughput canary；bucket007应先针对两个严重
phantom case诊断局部几何；bucket003需要不同方法自由度，而不是继续平面切分。完整27-case
Full仍需先解决bucket003，且任何新候选必须重新预注册，不能反选本轮full27结果。
