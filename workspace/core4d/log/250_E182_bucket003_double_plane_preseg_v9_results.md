# E182 阶段日志：bucket003 simultaneous double-plane pre-segmentation v9

_Core4D Phase 45 · 2026-08-01 · plan
[200](../plan/200_E182_task_conditioned_coacd_full_cem_plan.md) · 前置
[249](249_E182_bucket003_task_aware_preseg_v8_results.md)_

## 0. 一句话结论

用户批准后，v9 严格按冻结方案同时使用两条既有 task-derived plane，构建
`threshold {5,10,20mm} × K {16,32}=6` 个 10-segment candidate。3/3 middle
CoACD、3/3 composite bases 和 6/6 candidates 均成功，但冻结 882-pose static P gate
仍为 **`0/6 PASS`**。

最接近 recall floor 的 `t010/K32` 为
`TP19/phantom14/missed8, precision=0.576, recall=0.704`，phantom 比上限8多6；
precision 最接近 floor 的 `t020/K32` 为
`TP15/phantom7/missed12, precision=0.682, recall=0.556`，TP 比下限19少4。
仍没有 `TP≥19 AND phantom≤8` 的第三工作点。

因此执行预注册动作：

```text
STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES
```

不进入完整 P/R/G、真实 MuJoCo replay、3D/2D launch-floor review、grid-SDF、production
freeze、heldout24 或 Full CEM。Full 未启动不是算力问题，而是碰撞体 P gate 未通过。

## 1. 冻结协议与实现审计

### 1.1 Topology 与 family

| 项目 | 冻结值 |
|---|---|
| source | 原 x2×y4 segment3，8-segment parent |
| planes | `-0.14311002844145554m`, `-0.11878629238288715m` |
| split | 同一 source 与3个闭区间 box 的 exact manifold intersection |
| final topology | 7 unchanged + 3 children = 10 segments |
| transition isolation | TP pose177→region0；phantom pose203→region1；phantom pose202→region2 |
| K | `16,32`；K8结构不可行，K4禁止 |
| threshold | `5,10,20mm` |
| fresh CoACD | 仅 middle segment4，三个 threshold 各1次 |
| reuse | 21个 v4 unchanged manifests + 6个 v8 outer manifests |
| static P authority | 882 poses，27 oracle-contact poses |
| hard gate | precision/recall均`≥0.70`，等价于`TP≥19 AND phantom≤8` |

三个 exact children 均 watertight、winding-consistent、positive-volume；体积为
`0.005048199531/0.001739451098/0.007296357166m³`，总 closure delta=
`3.584199e-11m³ ≤ 1e-8m³`。

### 1.2 Pre-freeze 可复现性

正式 protocol 创建前完成：

- exact-C / CoACD v1–v5 / v6 / v7 / v8 / visual / v9 functional matrix
  `55/55 PASS`；
- v9 contracts `13/13 PASS`，覆盖 protocol contamination、source/parent binding、
  segment/base/candidate/static resume、tamper rejection 与 non-isolated CoACD rejection；
- ruff check/format、compileall、repo `diff --check` 全 GREEN；
- v9 result root 在 freeze 前为空；
- v9 builder SHA=`0022ef4e4e71ed761641dc60c35b720c5b54874b96e3b4be11642f65c82f7a7e`；
- v8 builder/protocol/build/static/visual 与 exact-C、v5 evaluator authority SHA保持冻结值；
- outer child vertex/triangle/bounds/volume canonical fingerprint exact，最大 triangle-coordinate
  delta=`6.94e-18m`。

protocol SHA：

```text
5ea0245bd0cbe7808af3c3d0875d18067b6d1b33f35aaac3d73b546439acde62
```

## 2. 构建结果

### 2.1 Fresh middle CoACD 与 composite bases

| Threshold | Middle hulls | Request≤4 | CoACD wall(s) | Composite hulls |
|---:|---:|---:|---:|---:|
| 5mm | 4 | PASS | 5.549 | 40 |
| 10mm | 4 | PASS | 3.308 | 36 |
| 20mm | 2 | PASS | 0.428 | 24 |

5mm 与10mm 的 CoACD native log 报告 soft-cap4 下 residual concavity 仍高于 threshold；
这是冻结协议要求保留的近似质量 warning，不是 build failure，也没有据此提高 cap。

每份 composite manifest 都严格包含：

- 7个 `REUSED_V4_UNCHANGED_SEGMENT`；
- 2个 `REUSED_V8_OUTER_SEGMENT`；
- 1个 `V9_FRESH_MIDDLE_SEGMENT`；
- 10个连续、非空 segments，禁止跨 segment merge。

### 2.2 Final candidates

| Candidate | Base | K | Actual | Merges | Max vertices | Build wall(s) |
|---|---:|---:|---:|---:|---:|---:|
| `taskpreseg_v9_both_t005_k16_v256` | 40 | 16 | 16 | 24 | 74 | 1.363 |
| `taskpreseg_v9_both_t005_k32_v256` | 40 | 32 | 32 | 8 | 54 | 0.683 |
| `taskpreseg_v9_both_t010_k16_v256` | 36 | 16 | 16 | 20 | 69 | 0.916 |
| `taskpreseg_v9_both_t010_k32_v256` | 36 | 32 | 32 | 4 | 55 | 0.324 |
| `taskpreseg_v9_both_t020_k16_v256` | 24 | 16 | 16 | 8 | 67 | 0.265 |
| `taskpreseg_v9_both_t020_k32_v256` | 24 | 32 | 24 | 0 | 32 | 0.040 |

6/6 candidate 均满足 actual hulls`≤K`、10-segment coverage、final part vertices`≤256`、
ordered asset SHA 和 no-cross-segment merge 合同。

## 3. Static P 结果

| Candidate | t(mm) | K | Actual | TP/phantom/missed | Precision | Recall | P score | Wall(s) | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `taskpreseg_v9_both_t005_k16_v256` | 5 | 16 | 16 | 22/24/5 | 0.478 | 0.815 | 1.739 | 1.010 | FAIL |
| `taskpreseg_v9_both_t005_k32_v256` | 5 | 32 | 32 | 22/21/5 | 0.512 | 0.815 | 1.628 | 0.997 | FAIL |
| `taskpreseg_v9_both_t010_k16_v256` | 10 | 16 | 16 | 19/17/8 | 0.528 | 0.704 | 1.574 | 0.989 | FAIL |
| `taskpreseg_v9_both_t010_k32_v256` | 10 | 32 | 32 | 19/14/8 | 0.576 | 0.704 | 1.414 | 0.972 | FAIL |
| `taskpreseg_v9_both_t020_k16_v256` | 20 | 16 | 16 | 15/10/12 | 0.600 | 0.556 | 1.481 | 0.980 | FAIL |
| `taskpreseg_v9_both_t020_k32_v256` | 20 | 32 | 24 | 15/7/12 | 0.682 | 0.556 | 1.481 | 1.021 | FAIL |

aggregate=`0/6`，`selected_candidate_ids=[]`，`full_prg_eligible=false`。

### 3.1 与 v8 的直接解释

- t005 保留22个 TP，但需要21–24个 phantom，是 high-recall/low-precision 端；
- t010 恰好达到 TP19，却仍有14–17个 phantom；
- t020/K32 已把 phantom 降到7，但只保留15个 TP；
- 同一 threshold 下，K32通常减少 phantom，却不改变 TP/recall signature；
- simultaneous double-plane 没有把 v8 的 high-recall 与 high-precision 工作点组合成新工作点，
  只在原 trade-off 上产生小幅 phantom 改善。

这说明“把三个 transition query 分到不同 pre-segment”只是必要的结构分离，不足以保证
最终 convex-hull union 在 task query 上同时保真。每个 segment 内仍由 convex hull
外包，reducer 也只能在段内改变 envelope；局部 rim/thin-wall 的 false-positive 与
missed-contact trade-off 没有被两条平面消除。

Full CEM 无法修复这个前置问题：它只会在错误的 collision proxy 上优化轨迹。跳过 P gate
再看 Full 结果会把“优化器适应了错误碰撞体”误当成“碰撞体正确”。

## 4. Claims 验证

| Claim | 结果 |
|---|---|
| V9-C1 protocol pre-score freeze | **PASS** — root预先为空，parent/source/reuse/code SHA与6-row family闭合 |
| V9-C2 exact topology | **PASS** — 3/3 children闭合，volume delta=`3.58e-11m³`，三个transition各一region |
| V9-C3 minimal fresh build | **PASS** — 3/3 middle isolated CoACD；21+6 reuse exact，无outer重建 |
| V9-C4 candidate closure | **PASS** — 6/6 BUILD_PASS，10 segments，actual`≤K`，vertices`≤256` |
| V9-C5 third P point | **FAIL** — `0/6`，没有`TP≥19 AND phantom≤8` |
| V9-C6 contamination guard | **PASS** — heldout/grid/GPU/Full访问均为0；v8 parent SHA保持exact |

## 5. 可视化与物理 replay

### 5.1 状态

本轮没有生成新的 v9 3D/2D 图或真实 MuJoCo replay。原因不是 viewer 不可用，而是 v9
预注册 stop/go 明确规定：只有至少一行通过 static P，才进入完整 P/R/G、physics replay
与3D/2D review。`0/6` 后继续该阶段会越过 launch floor。

已有 v8 只读 diagnostic 已确认同类失败集中在 bucket rim/thin-wall 邻域；v9 数值结果
没有授权把新视觉结果用于重新挑 plane/K/threshold。这里将“未运行”作为 protocol-driven
stop 记录，而不写成视觉 PASS。

## 6. 结果路径与 SHA

统一结果根：

```text
workspace/core4d/results/E182/s2_task_query_eval/
  attempt2_task_aware_preseg_v9_double_plane/
```

| Artifact | SHA-256 |
|---|---|
| `attempt2_v9_protocol_manifest.json` | `5ea0245bd0cbe7808af3c3d0875d18067b6d1b33f35aaac3d73b546439acde62` |
| `segments/manifest.json` | `96e1b393db8e650ce6cf20526f6dab7bcb3a142d96bbc314efde7271921051cc` |
| `base_summary.json` | `e9ae98e0c96ca4803c91de0298c2b76246ff0b1c7897569652f42919aa3f6049` |
| `build_summary.json` | `2d900e7d39b09c3003d09831acfb3f4fecf4a225a9b7f49da4508bcf7e7623b3` |
| `static_p_screen/static_p_aggregate.json` | `87c401182e082c54de4dec5b0b4c4797c0ef2b5c552bb24b60b417d84b5a642c` |

结果根共175个文件；protocol/segment/base/build/static validators均已二次运行 PASS。

## 7. 停止项与后续决策

已冻结停止：

- 不移动或增加 plane；
- 不降低0.70 P floor；
- 不追加 K4/K8 或事后 threshold；
- 不根据 static/Full 结果重挑 collision geometry；
- 不进入完整 P/R/G、grid-SDF、production freeze、heldout 或 Full CEM；
- 不启动本地 GPU 或远程 Ada GPU，既有进程未被操作。

v5–v9 已排除 global threshold、per-segment threshold、局部 merge、single-plane 与
simultaneous double-plane pre-segmentation。若继续 E182，必须由用户批准新的方法自由度，
例如 rim-aware 非平面 primitive 或 MuJoCo SDF/plugin 高保真局部碰撞；不能继续给 CoACD
平面切分加手工规则。

当前权威状态：

```text
V9_STOPPED_COMPLETE_NEGATIVE_RESULT
P_LAUNCH_FLOOR_NOT_CLOSED
FULL_NOT_AUTHORIZED
HELDOUT_NOT_ACCESSED
```
