# E182 阶段日志：bucket003 P decomposition v5–v7 结果

_Core4D Phase 45 · 2026-08-01 · plan
[200](../plan/200_E182_task_conditioned_coacd_full_cem_plan.md)_

## 0. 一句话结论

bucket003 的 CoACD attempt2 已完成三层 task-conditioned 搜索，但仍没有通过
P contact precision/recall 双 `0.70` launch floor 的候选：v5 全局 threshold 为
`0/9`、v6 per-segment threshold hybrid 为每个 K `0/6561`、v7 segment3 四-part
全部 Bell(4)=15 分区在 K16/K32 均为 `0/15`。v7 证明 partial merge 只产生
`TP18/phantom8/missed9` 与 `TP19/phantom10/missed8` 两个离散工作点；因此当前状态为
**`P_LAUNCH_FLOOR_NOT_CLOSED / FULL_NOT_AUTHORIZED`**。heldout24、grid-SDF、production
freeze 和 Full CEM 均未启动。

## 1. 范围与冻结边界

- object/case 固定为 `bucket003/bucket003_20231018_001_p1`；
- P authority 固定为 reference + E178-final 共 `882` 个静态姿态、oracle contact=`27`；
- P contact 使用18个真实 physics robot geom的 radius-adjusted original-mesh SDF；
- `K=8/16/32`，不测试 K4；P precision/recall launch floor 各为 `≥0.70`；
- 所有搜索仅使用 dev evidence，heldout=`NOT_ACCESSED_DEV3_ONLY`；
- Full 固定 E178 的 27 case 与 `seed0, 1024×32`，但本阶段没有 launch 授权；
- 未来 Full 资源仍为本机单卡 + `spider-remote` RTX 6000 Ada GPU0/1，允许与既有
  compute 叠加，禁止 kill、暂停、抢占或修改既有进程；不使用 A100。

## 2. v5：global-budget CoACD candidate screen

v5 从每 threshold 的 isolated CoACD base 出发，用 deterministic within-segment convex
merge 把 global hull count 压到 K。9个候选全部 BUILD_PASS：

| Threshold | K8 | K16 | K32 |
|---|---:|---:|---:|
| 5 mm | 8 | 16 | 32 |
| 10 mm | 8 | 16 | 29 |
| 20 mm | 8 | 16 | 18 |

完整正式 P/R/G screen 为 `0/9` launch-floor PASS，但失败完全来自 P；D/R/G 保持健康：

| 指标 | 范围/结果 |
|---|---|
| point sign disagreement | `3.86e-6–6.62e-6` |
| deep mismatch | 约 `0` |
| SDF p90 | `2.0–4.4mm` |
| R normalized p90 | `0.1%` |
| R Spearman | `1.0` |
| G mask/fallback/top-k | 无分歧 |

P 呈显著 precision/recall trade-off：

| Family | Precision | Recall | 结论 |
|---|---:|---:|---|
| t005 K8 | 0.765 | 0.481 | recall fail |
| t005 K16/K32 | 0.733 | 0.407 | recall fail |
| t010 all | 0.636 | 0.259 | both fail |
| t020 K8 | 0.655 | 0.704 | precision fail |
| t020 K16/K32 | 0.692 | 0.667 | both narrowly fail |

正式 aggregate SHA=`544c2b491bc520930e242b8f8534633661d21ace29826abf3e45b339e62ba184`，
candidate TSV SHA=`52475b13f370055fb966d6722fd4c079c0ce3e95f6d4e31f4a8fd3029abac7a9`。

## 3. v6：per-segment threshold hybrid

v6 对 K8/K16/K32 分别枚举 `3^8=6561` 个 segment threshold assignment，并要求：

- actual hulls `≤K`；
- P precision/recall 均 `≥0.70`；
- segment contact OR 必须逐项 exact 重现9个 v5 parent 的正式 P confusion counts。

结果为三个 K 各 `0/6561` passing；9/9 parent reconstruction exact。最优 near miss：

| K | TP | Phantom | Missed | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 8 | 19 | 10 | 8 | 0.655 | 0.704 |
| 16 | 18 | 8 | 9 | 0.692 | 0.667 |
| 32 | 18 | 8 | 9 | 0.692 | 0.667 |

只有 segment3 和 segment7 会产生 contact，segment3 决定主要 trade-off。v6 search
SHA=`df1e4557f9dea9fad54c7f045049000e6d4f709cd4bd999922fc8483b5d860e4`。

## 4. v7：segment3 四-part exhaustive partition

### 4.1 冻结协议

v7 只搜索 K16/K32；K8 在8个 segment 至少各1 hull时没有 partial-merge预算。对 t020
segment3 的四个 unchanged base parts 枚举全部 Bell(4)=15 个 canonical partitions；
每个 block 使用其 source vertices 的 convex hull，其他7个 segment 固定为 v6 near-miss
context。

- pre-freeze functional closure=`41/41 PASS`；
- core/v6/v7 source SHA=`a31eb7e4…fd93/b9f6a2ba…8316/fc2fdb61…609e`；
- protocol SHA=
  `6e8cf951f793b9bc7a4ced02414b3ae7fe3c1258fff64555a94ff34e48feea37`；
- freeze 前 v7 root 为空，heldout未访问。

### 4.2 正式结果与审计

正式搜索自然完成，K16/K32 均 `0/15` passing，selected=`0`。结果 SHA=
`3ee38ec25936beeda7aa8be6676e79e723d3b1a5da3527e0a853227b81a3002a`。

30-row artifact audit PASS：

- `30/30 actual_hulls≤K`；
- 每行 confusion 总数=`882`，oracle contacts=`27`；
- protocol/result/source SHA exact；
- K16/K32 对相同 partition 的 P 指标完全相同，说明 hull budget不是当前瓶颈；
- 15个 partition 只产生两个接触签名：

| Partition behavior | 数量/每K | TP | Phantom | Missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| precision-side | 10 | 18 | 8 | 9 | 0.6923 | 0.6667 |
| recall-side | 5 | 19 | 10 | 8 | 0.6552 | 0.7037 |

不存在目标的 `TP≥19` 且 `phantom≤8` 工作点。

### 4.3 精确 transition

从代表 precision-side partition `[[0],[1,2,3]]` 到 recall-side 全合并
`[[0,1,2,3]]` 是严格单调新增3个 E178-final contact、无移除：

| E178-final pose | Oracle | 变化 | object-local触发点 (m) | precision C | recall C | oracle M |
|---:|---|---|---|---:|---:|---:|
| 177 | contact | TP +1 | `(-0.1667, 0.2025, 0.1968)` | +1.49mm | −0.11mm | +1.23mm* |
| 202 | free | phantom +1 | `(-0.1180, 0.2010, 0.2010)` | +1.91mm | −0.19mm | +2.78mm |
| 203 | free | phantom +1 | `(-0.1196, 0.2059, 0.2020)` | +2.05mm | −0.88mm | +3.50mm |

`*` pose177 的 oracle contact 由同姿态另一采样点触发；表中 M 是新 C contact 点自身的
clearance。三点全部位于 bucket 顶部 rim 的同一局部。全合并每救回1个 TP，同时增加2个
phantom，无法靠 K 或同一四-part partition 排序打破。

## 5. 三维/二维可视化与实际观察

post-search renderer 固定 K16 两种 signature，重建完整 hybrid union并逐项复算882姿态；
输出明确 `selection_eligible=false/construction_eligible=false`。visual manifest SHA=
`fd84a28f82bc3f4b34d6e0a4a19921e6781d620e89e7a753979c884f4ceb5218`。

实际查看四张原分辨率图：

- precision-side 与 recall-side 的洋红 phantom 均集中在顶部 rim/端部唇缘的手部点；
- 不是远离物体的 body/leg 点，也不是随机单帧噪声；
- 3D整体轮廓肉眼合理，但临界 rim 处 C 比 M 向外跨约5mm；
- XY/XZ/YZ 投影显示两种 partition 的误接触位于同一上缘局部；
- timeline 在 reference 与 E178-final 都出现多个离散 phantom 窗；
- 全合并只增加接触窗，没有消除已有 phantom。

结果路径：

```text
workspace/core4d/results/E182/s2_task_query_eval/
└── attempt2_segmented_x2y4_v7_segment3_partition/
    ├── v7_search_protocol_manifest.json
    ├── segment3_partition_search_results.json
    └── visual_diagnostic/
        ├── K16_precision_side_task_pose_3d.png
        ├── K16_precision_side_task_pose_2d.png
        ├── K16_recall_side_task_pose_3d.png
        ├── K16_recall_side_task_pose_2d.png
        └── visual_manifest.json
```

## 6. Claims 状态

| Claim | 当前证据 | 状态 |
|---|---|---|
| C0 authority | E178 full27/dev3/heldout24 authority保持冻结 | PASS |
| C2 real-query tape | Gate1 ref/final/CEM P/R/G query可恢复 | PASS |
| C3 task-conditioned fidelity | bucket003 v5–v7 无 P launch-floor candidate | FAIL/PENDING NEW METHOD |
| C4 Pareto selection | bucket003 K8/16/32均不可选 | BLOCKED BY C3 |
| C1/C5–C9 | production C/D_C、canary、Full、paired eval未执行 | NOT AUTHORIZED |

本阶段没有修改 E178/E181 历史结果，没有读取 heldout24，没有生成 `C_prod/D_C`，也没有
启动、暂停或终止任何本地/远程 GPU 任务。

## 7. 三次失败复盘与下一步

这三层负结果改变了实质自由度，而不是原样重复：

1. v5：全局 threshold + global hull budget；
2. v6：per-segment threshold assignment；
3. v7：segment3 base parts 的全部 partial-merge topology。

它们共同否定的是“继续在当前 x2×y4 segment + t005/t010/t020 CoACD parts 上做 threshold/
merge 组合就能闭合 bucket003 P”，不是否定 canonical `C→P, D_C→R/G` 架构。根据 plan 200
的三次失败协议，下一步不再增加同类 merge sweep，需向用户汇报并改变 decomposition：

**推荐路线：task-aware rim-preserving pre-segmentation。** 用冻结 dev P query/contact atlas
自动定位顶部 rim 的高风险带，将 rim/lip 与侧壁分成独立 manifold regions，再对每个 region
分别运行 CoACD；禁止人工编辑 hull。第一目标不是直接 Full，而是先证明出现第三个 P 工作点：
`TP≥19, phantom≤8`，随后仍需完整 P/R/G screen、MuJoCo replay、可视化和 runtime canary。

备选路线是改用另一种 convex decomposition（如 V-HACD）做同一 dev audit；但当前误差高度
局部且与 task contact atlas 对齐，优先级低于 task-aware pre-segmentation。

无论选择哪条路线，P launch floor 未闭合前继续禁止 S3 grid-SDF、production freeze、
heldout24 与 Full CEM；不得通过降低 `0.70` 门槛或按 Full 结果反向挑碰撞体救援。
