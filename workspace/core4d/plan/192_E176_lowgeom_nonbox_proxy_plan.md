# E176 实验计划：个位数非 box proxy 与 multi-geom CEM 加速

_Core4D Phase 39 · 2026-07-24 · 继承 E175 physics/PRG 修复，压缩 object proxy 复杂度_

前置证据：[E174 结果](../log/234_E174_bucket_desk_move2_nonbox_results.md) ·
[E175 计划](191_E175_nonbox_proxy_fidelity_diagnostic_plan.md)

## Context

E175 修复了两个真实接入缺陷：robot–object physics pair 和 PRG SDF
都必须覆盖全部 `object_collision*`。但 E175 production proxy 使用
surface voxel，单物体产生 `41–167` 个 box，进而形成
`738–3006` 个显式 robot–object pair。

E175 Full 在 A100、`1024×32` 下实测每 2 sim steps 需要
`120–150s`；39-case 预计 2.5–3 天。用户明确否决该复杂度，允许降低
几何精度，要求每个物体 collision geom 为个位数。E175 Full 已停止，
隔离 root 与部分日志保留，不作为完成结果。

E176 不改变 E174 的 39-case authority、轨迹、contact mask、reward、
PRG 阈值、seed 或 CEM budget，只替换 object proxy 表达并重新生成
physics pair。

首轮 single-digit canary 的早期记录若仍高于 `3s/record`，允许加入一个
**数学等价、tick-local 的 union-SDF memoization**：同一 reward tick
内，相同 `robot geom id tuple` 对相同 object-box union 的查询只计算
一次。该优化不得跨 tick 缓存（geom pose 每 tick 会变），不得合并不同
robot geom 集合的近似值，也不得改变任何 SDF/reward/gate 数值。
若一个多-geom tuple 的全部 singleton SDF 已在同 tick 精确计算，则可
用这些 singleton 的逐 world `min` 精确合成该 tuple（集合 union 的
min 运算恒等式）。它只消除
`robot penalty ↔ safety gate`、`leg penalty ↔ leg gate` 的重复查询。

若 exact memoization canary 仍略高于3秒，则按三次失败协议重新审视
kernel 组织：E176 可显式启用 `object_collision_sdf_batch_groups`，把
本 tick 所需的上身/腿/手 geom 并为一个有序唯一列表，一次计算
`world×robot_geom` 的逐-geom union SDF，再对各原始集合的列取 `min`。
这必须与逐集合 `_geom_box_union_sdf_min` 逐 world 数值一致；默认关闭，
仅 E176 override 开启，避免静默改变其他实验的性能/内存特征。

## Claims

| Claim | 可验证标准 |
|---|---|
| C1 复杂度受控 | 每个物体 `1–9` 个 box geom |
| C2 physics 完整 | compiled pair 精确为 `18×geom_count`，最大 162 |
| C3 PRG 完整 | effective config 为 union，运行时 geom 名称/ID/SHA 一致 |
| C4 bucket 不填实 | 中心线中段不落入任一 proxy box；底板仅占底部薄层 |
| C5 desk 不桥接 | 桌面下中心区域不落入 proxy；四腿间隙保留 |
| C6 接触覆盖可接受 | ref-FK contact target 到 proxy 的分组 p90 ≤ 8 cm |
| C7 性能恢复 | 64×4 canary median plan time：每对象 ≤ 3 s/record |
| C8 authority 不漂移 | E176 Full case set 与 E174 精确 39=39 |
| C9 等价加速 | cache on/off 的 reward/info 张量逐字段一致，且每 tick 相同 geom tuple 只执行一次 union SDF |
| C10 分组批处理等价 | batched per-geom 后各原集合 min 与逐集合 union SDF 逐 world 一致 |

## Proxy 设计

### Bucket / Desk：统一 adaptive coarse surface voxel

Bucket 和 desk 继续使用完全相同的 mesh-surface voxel 算法，不手写
物体语义：

1. 对 `target_cells=3..9` 执行 E144/E175 同源的
   occupied-surface greedy merge。
2. 仅保留 box 数 `≤9` 的候选。
3. 在可行候选中最小化
   `max(mesh→proxy p90, proxy→mesh p90)`，并把选定 cells 按 object
   冻结，后续 builder 对选择结果做回归断言。
4. 沿用每个 box 的 12% pitch inward shrink，减少相邻粗 voxel 的
   phantom overlap。

真实 mesh 离线扫描得到的冻结结果：

| Object | target_cells | boxes |
|---|---:|---:|
| bucket003 | 4 | 7 |
| bucket004 | 4 | 6 |
| bucket007 | 3 | 7 |
| bucket009 | 4 | 6 |
| bucket010 | 9 | 7 |
| desk007 | 5 | 9 |

选定候选的 mesh→proxy p90 为 `2.1–6.7cm`，proxy→mesh p90 为
`6.3–13.7cm`。所有物体的 mesh AABB 中心均未落入 proxy box，说明
粗化后仍保留主要空腔/间隙。两类 proxy 都只输出 MuJoCo `box`，命名为
`object_collision` + `object_collision_coarse_001...`。

## Scope

### In scope

- E174 同一 39 rows。
- 5 个 bucket + desk007 的 E176 sidecar。
- E175 已修复的 18×N physics pair 和 PRG union SDF。
- single-digit 仍未过吞吐门时，启用 tick-local exact union-SDF memoization。
- 离线 geometry/contact fidelity、compiled contract、6-case A100 canary。
- canary throughput gate 通过后才允许 Full 39。

### Out of scope

- 覆盖或删除 E174/E175 scene、manifest、NPZ、日志。
- 修改 reward、gate 阈值、seed、CEM budget。
- 为提高 fidelity 把任一物体增至 10 个或更多 geom。
- 在低复杂度 canary 前续跑已停止的 E175 Full。

## Artifacts

```text
workspace/core4d/results/E176/
  s0_environment/
  s2_proxy/
  s5_handoff/
  s6_downstream/manifests/
  s6_downstream/cem/{canary,full}/
  s6_downstream/eval/
```

Sidecar：

```text
scene_act_E176_coarse9_multiGeom.xml
```

主要实现：

```text
workspace/core4d/scripts/experiments/E176/build_lowgeom_production.py
workspace/core4d/scripts/experiments/E176/run_cem_queue.py
workspace/core4d/scripts/experiments/E176/validate_cem_runtime.py
workspace/core4d/scripts/launch/active/run_E176_remote_a100.sh
workspace/core4d/scripts/launch/active/pull_E176_remote_a100_results.sh
workspace/core4d/scripts/launch/active/watch_E176_remote_a100.sh
```

## Gates

### Local build gate

- E174 authority 39/39 exact。
- 39/39 sidecar 可编译。
- object geom count `≤9`。
- compiled pair count `≤162` 且无 missing/duplicate/extra。
- bucket cavity 5/5 pass；desk under-table center pass。
- box-only union SDF tests pass。
- memoization 数值等价与 call-count regression pass。
- object-only mesh/proxy overlay 6/6 objects生成并复核。

### Canary gate

固定复用 E175 的 6 条：

```text
bucket003_20231018_001_p1
bucket004_20231002_021_p1
bucket007_20231020_055_p1
bucket009_20231002_056_p2
bucket010_20231003_2_055_p2
desk007_20231030_034_p2
```

预算 `64 samples × 4 iterations`。必须满足：

- 6/6 `run_complete_pending_eval`
- 6/6 runtime hard gate
- 无 OOM/import/device/scene error
- 每对象 median plan time ≤ 3s/record
- effective geom count ≤9、pair count ≤162

只有上述 gate 全过才允许同 39 rows 的 `1024×32` Full。

## 执行顺序

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E176/build_lowgeom_production.py \
  --preflight --require-review
.venv/bin/python workspace/core4d/scripts/experiments/E176/test_lowgeom_proxy.py

A100_POLICY_GPUS="2,3,6,7" E176_FIXED_GPUS=1 E176_PREP_ONLY=1 \
  bash workspace/core4d/scripts/launch/active/run_E176_remote_a100.sh canary
A100_POLICY_GPUS="2,3,6,7" E176_FIXED_GPUS=1 \
  bash workspace/core4d/scripts/launch/active/run_E176_remote_a100.sh canary
bash workspace/core4d/scripts/launch/active/watch_E176_remote_a100.sh canary
```

Full 命令只在 canary runtime + throughput gate 均通过后执行。
