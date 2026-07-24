# E176 结果日志：个位数非 box proxy 与 CEM 吞吐 canary

_Core4D Phase 39 · 2026-07-24 · plan [192](../plan/192_E176_lowgeom_nonbox_proxy_plan.md)_

## 0. 一句话结论

E176 将 5 个 bucket 和 desk007 的 collision proxy 压到 **6–9 geoms**
且保持主要空腔/桌下开口，39/39 scene contract、39/39 ref-FK contact
fidelity 与 6/6 canary runtime 均通过；但最终 group-batched canary 的
`median plan time ≤3s` 仅 **4/6**，bucket004=`3.0079s`、
bucket010=`3.0572s`，因此 **Full 39 没有启动**。

## 1. 背景与改动

E175 已修复两个严重接入缺陷：

1. robot–object 物理 pair 必须覆盖全部 `object_collision*`；
2. PRG object SDF 必须查询全部 collision geom 的 union。

但 E175 surface-voxel proxy 每个物体有 `41–167 geoms`、
`738–3006 pairs`，Full 预计 52–66 小时。E176 按用户要求允许降低
几何精度，bucket/desk 统一改为 adaptive coarse surface voxel，并增加
两个数学等价的执行优化：

- tick-local exact tuple cache；
- E176 显式开启、其他实验默认关闭的 per-geom group batching。

reward、gate 阈值、seed、E174 39-case authority 和 Full 预算
`1024×32` 均未改变。

## 2. Proxy 与静态门

| Object | Geoms | Robot–object pairs | Mesh→proxy p90 | Ref-contact→proxy p90 |
|---|---:|---:|---:|---:|
| bucket003 | 7 | 126 | 6.62 cm | 4.24 cm |
| bucket004 | 6 | 108 | 2.79 cm | 7.26 cm |
| bucket007 | 7 | 126 | 5.71 cm | 7.28 cm |
| bucket009 | 6 | 108 | 5.06 cm | 6.29 cm |
| bucket010 | 7 | 126 | 2.08 cm | 7.68 cm |
| desk007 | 9 | 162 | 5.56 cm | 4.52 cm |

- 39/39 sidecar compile、pair matrix、authority parity PASS。
- 6/6 object overlay review approved；5 个 bucket 的中心空腔和 desk 桌下
  开口均保留，没有退化成实心 AABB。
- ref-FK contact fidelity 39/39 PASS；6 个 object-grouped p90 均
  `≤8cm`。
- 运行时 6/6 config 确认 `object_collision_sdf_mode=union`、
  `object_collision_sdf_batch_groups=true`，实际 geom 数与 manifest
  一致。

## 3. 三轮吞吐证据

固定 A100 GPUs `2,3,6,7`，相同 6-case、`64 samples × 4 iterations`：

| Object | Prod1 no-cache | Prod2 exact-cache | Prod4 group-batch | Prod1→Prod4 |
|---|---:|---:|---:|---:|
| bucket003 | 3.7189s | 3.0756s | **2.9865s** | -19.69% |
| bucket004 | 3.7202s | 3.1359s | **3.0079s** | -19.15% |
| bucket007 | 3.5832s | 3.0912s | **2.9659s** | -17.23% |
| bucket009 | 3.6031s | 3.0876s | **2.9798s** | -17.30% |
| bucket010 | 3.6694s | 3.1308s | **3.0572s** | -16.68% |
| desk007 | 3.6163s | 3.0745s | **2.9629s** | -18.07% |
| Throughput gate | 0/6 | 0/6 | **4/6** | — |
| Runtime gate | 6/6 | 6/6 | **6/6** | — |

Prod3 singleton composition 在启动后发现 frozen config 中
surface-band 与 hand gate 本来就是相同 `[lh,rh]` tuple，cache 已命中，
所以该优化实际零命中；任务立即停止且未作为结果轮次。

## 4. 原因分析

1. **E175 的首要慢因确实是高密度 collision proxy。** 将对象压到
   6–9 geoms 后，canary 从 E175 约 5.5s 降到 E176 no-cache
   3.58–3.72s。
2. **single-digit 后的主要可消除开销是同 tick 重复 union-SDF。**
   exact tuple cache 使六条再快约 14–18%；group batching 相对 cache
   再快约 2–4%，且 sphere/capsule/mesh 数值回归与 legacy 逐 world
   完全相等。
3. **剩余 0–57ms 超门不能再归因于 object geom 数量。**
   bucket004 只有 6 geoms 却微幅失败，desk007 有 9 geoms 反而通过；
   bucket007/bucket010 同为 7 geoms 也一过一败。剩余时间主要是固定
   CEM/MJWP 开销、group reduction/kernel 调度与运行波动，而不是继续
   减少一两个 proxy box 就能稳定解决。
4. bucket004 仅超门 7.9ms（0.26%），bucket010 超 57.2ms（1.91%）；
   但计划冻结的是逐 case `≤3.0s`，不能事后用四舍五入或放宽阈值改判。

## 5. Claims

| Claim | 结果 |
|---|---|
| C1 每物体 1–9 geoms | ✅ 6–9 |
| C2 compiled pair=`18×N`、最大162 | ✅ 39/39 |
| C3 PRG union runtime authority | ✅ 6/6 canary |
| C4 bucket 不填实 | ✅ 5/5 overlay/cross-section |
| C5 desk 不桥接 | ✅ desk007 桌下开口保留 |
| C6 ref contact grouped p90≤8cm | ✅ 6/6 objects，39/39 cases可评 |
| C7 median plan time≤3s | ❌ 4/6 |
| C8 E174 authority 39=39 | ✅ |
| C9 exact tuple cache 数值等价 | ✅ |
| C10 group batching 数值等价 | ✅ sphere/capsule/mesh |

## 6. 错误与处置

| 事件 | 处置 |
|---|---|
| A100 SSH/rsync 间歇 timeout | 有限重试恢复；每轮远端 SHA 1808/1808 |
| Prod3 singleton 假设错误 | 核对 effective config 后立即停，无残留 CEM |
| 同名 canary pull 使用 append 模式不合适 | 改为 `--partial --timeout=30`，prod2/prod4 回收通过 |
| Prod4 throughput 仅4/6 | 按三次失败协议停止，不改阈值、不启动Full |

## 7. 可视化

E176 proxy overlay 6/6 已人工复核：

- 5 个 bucket 保留中心空腔、底部薄层和粗顶沿；
- desk007 保留桌面、支撑结构与桌下通道；
- bucket003/007 存在允许的 coarse phantom 外扩，是“个位数 geom、精度
  可降低”的显式取舍。

Canary 配置为 `save_video=false`，吞吐轮不生成视频；视觉证据沿用本实验
启动前冻结的 object overlay/cross-section，而不是用缺失视频替代审核。

## 8. 决策与下一步

- E176 的 **低 geom proxy、完整 physics pair、PRG multi-geom 与
  runtime correctness 均可保留**。
- 性能硬门未达 6/6；按三次失败协议，当前停止继续微调。
- **Full 39 仍锁定且从未启动。** 后续只有两种明确决策：
  1. 保持逐 case `≤3.0s`：需要新的 kernel/fused 实现实验；
  2. 用户显式修改吞吐准入策略后，再重新评估是否放行 Full。

## 9. 结果路径

```text
workspace/core4d/results/E176/
├── s2_proxy/                         # lowgeom proxy + visual review
├── s5_handoff/                       # 39-case handoff/contact fidelity
├── s6_downstream/manifests/          # 39 full + 6 canary authority
├── s6_downstream/cem/canary/
│   ├── canary_runtime_gate.json      # PASS 6/6
│   └── canary_throughput_gate.json   # FAIL 4/6
└── baselines/
    ├── no_cache_prod1/
    ├── cache1_prod2/
    └── batch1_prod4/                 # gate/log/manifest/execution快照
```

主要实现：

```text
spider/config.py
spider/simulators/mjwp.py
workspace/core4d/scripts/experiments/E176/
workspace/core4d/scripts/launch/active/{run,pull,watch}_E176_remote_a100.sh
```
