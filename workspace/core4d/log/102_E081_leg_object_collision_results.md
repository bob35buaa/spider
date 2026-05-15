# E081 Results: leg/foot-object collision 派生 scene 验证

日期：2026-05-16

对应计划：`workspace/core4d/plan/86_E081_leg_object_collision_eval_plan.md`

## 结论摘要

E081 按用户要求没有直接修改原始 `scene_act.xml`，而是新建了两个派生任务：

- `box025_person2_legobj`
- `box023_person2_legobj`

每个派生 `scene_act.xml` 都在原有 hand-object / object-floor pairs 之外新增 16 个腿/脚-`object_collision` contact pair。运行安排为本地 RTX5090 跑 `box025_p2_legobj`，远程 `spider-remote` GPU1 跑 `box023_p2_legobj`。

核心结论：

- C1 通过：原始 `box025_person2/scene_act.xml` 和 `box023_person2/scene_act.xml` 未修改；只新增 `*_legobj` 派生 task。
- C2 通过：腿/脚-箱 contact 在 MuJoCo 中生效，eval 能统计 `leg_object_contact_count`。
- box025_p2：加碰撞后腿/箱几何穿入显著下降，case-window interference 从 E080 baseline `28.9%` 降到 `7.5%`，最小 adjusted SDF 从 `-4.6cm` 改善到 `-1.2cm`；物体误差也略改善。但箱体 lift/floor-contact 没有改善，case-window object bottom mean 仍约 `-7.5cm`，因此仍是 partial positive / near-usable，不是 strict success。
- box023_p2 guard：没有被明显破坏。case-window obj mean `16.2cm -> 16.4cm`、contact `69.3% -> 66.7%`，稳定性几乎不变；新增腿/箱接触很少，case-window interference `2.7%`。
- 性能：加入腿/脚-箱 contact pair 后 planning 明显变慢。本地 box025 每个 CEM tick 约 `7.7-8.1s/2 sim steps`，远程 box023 全程约 `20min`。

## 实验配置

| Variant | Source task | Derived task | Split | GPU | 角色 |
|---|---|---|---|---|---|
| `E081_box025_p2_legobj` | `box025_person2` | `box025_person2_legobj` | local | RTX5090 GPU0 | main |
| `E081_box023_p2_legobj` | `box023_person2` | `box023_person2_legobj` | remote | `spider-remote` GPU1 | guard |

新增 contact pairs：

```text
left/right hip_collision, thigh_collision, shin_collision, linkage_brace_collision
lf0-lf3, rf0-rf3
```

均配对到 `object_collision`，`solref="0.008 1" friction="1 1" condim="3"`。

## 结果路径

| 类型 | 路径 |
|------|------|
| 汇总 | `workspace/core4d/results/E081/comparison.csv` |
| 聚合 | `workspace/core4d/results/E081/aggregate_summary.json` |
| NPZ/MP4 | `workspace/core4d/results/E081/E081_box0{23,25}_p2_legobj.{npz,mp4}` |
| 原始 timeseries | `workspace/core4d/results/E081/timeseries_E081_*_legobj.csv` |
| leg/object timeseries | `workspace/core4d/results/E081/legobj_timeseries_E081_*_legobj.csv` |
| eval summary | `workspace/core4d/results/E081/eval_summary_E081_*_legobj.{json,csv}` |
| 关键帧 | `workspace/core4d/results/E081/keyframes/E081_*_legobj/` |
| scene snapshot | `workspace/core4d/results/E081/scene_snapshot/` |
| logs | `logs/E081/E081_box025_p2_legobj.log`, `logs/E081/E081_box023_p2_legobj.log`, `logs/E081/remote_gpu1.log` |

## 量化结果

### E081 结果

| Variant | CaseWin | Leg/Lift Proxy | Fixed post2 | Obj mean/max m | Hand contact % | Leg intf % | Leg contact % | Floor contact % | Bottom mean m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `E081_box025_p2_legobj` | True | False | False | `0.143/0.271` | `89.0` | `7.5` | `7.5` | `59.5` | `-0.075` |
| `E081_box023_p2_legobj` | True | True | True | `0.164/0.317` | `66.7` | `2.7` | `2.7` | `34.7` | `0.144` |

`Leg/Lift Proxy` 是 E081 新增的保守 proxy：case-window 三阈值通过，同时腿/箱 interference `<=5%`，且 sim object bottom mean 不比 ref 低超过 `5cm`。它不是最终成功判据，只用于本轮诊断。

### 与 baseline 对比

| Pair | Obj mean/max m | Hand contact % | Leg intf % | Leg contact % | Floor contact % | Bottom mean m | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| E080 `box025_p2` | `0.146/0.289` | `90.8` | `28.9` | `0.0` | `60.1` | `-0.073` | 原 scene 无腿-箱物理 contact，几何穿入较多 |
| E081 `box025_p2_legobj` | `0.143/0.271` | `89.0` | `7.5` | `7.5` | `59.5` | `-0.075` | 穿入显著减少，物体误差略改善，但没有更好 lift |
| E079 `box023_p2` | `0.162/0.317` | `69.3` | `0.0` | `0.0` | `32.0` | `0.143` | guard baseline 本身腿/箱无穿入 |
| E081 `box023_p2_legobj` | `0.164/0.317` | `66.7` | `2.7` | `2.7` | `34.7` | `0.144` | guard 基本保持，略多腿/箱接触 |

聚合：

```json
{
  "num_results": 2,
  "num_main_results": 1,
  "num_main_case_window_success": 1,
  "main_case_window_success_pct": 100.0,
  "num_main_legobj_strict_proxy_success": 0,
  "main_legobj_strict_proxy_success_pct": 0.0,
  "guard_results": ["E081_box023_p2_legobj"]
}
```

## 可视化观察

### `E081_box025_p2_legobj`

- `f100`：整体仍接近 E080 p2 的扶/搬箱姿态；右脚/小腿与箱体保持更明显的分离趋势，但脚尖附近仍非常接近箱底/侧面。
- `f125`：sim 与 ref 大体一致，手在箱侧；右脚在箱体近处，视觉上仍有“用腿清障/贴近箱子”的感觉，但不再是 E080 中那种明显穿入。
- `f160`：后段姿态仍像扶箱/搬箱，箱体高度仍偏低，接近地面；这与 bottom mean 没改善一致。

视觉结论：E081 对 box025_p2 的腿/箱穿入有实质改善，但没有把它从 partial positive 推到 strict success。当前主要剩余问题是箱体 lift/floor-contact，而不是腿穿模。

### `E081_box023_p2_legobj`

- `f100`：仍是比较可信的抱/搬小箱姿态，手与箱体关系清楚。
- `f125`：右脚靠近箱底，eval 也记录到少量腿/箱 contact；但接触比例低，不像主要支撑来源。
- `f160`：进入放下阶段，sim 保持稳定，视觉上没有因为新增碰撞而明显崩坏。

视觉结论：box023_p2 guard 基本保住。新增腿/脚-箱碰撞没有破坏该高质量 case，但会带来轻微腿/箱接触和手接触下降。

## Claims 验证

| Claim | 结果 | 说明 |
|------|------|------|
| C1 新 scene 不污染原始数据 | 通过 | `git diff -- box025_person2/scene_act.xml box023_person2/scene_act.xml` 为空；新增的是 `*_legobj` 派生目录。 |
| C2 腿/脚-箱 contact 生效 | 通过 | E081 eval 记录到 box025 case-window leg contact `7.5%`，box023 `2.7%`。 |
| C3 box025_p2 加碰撞后更可信 | 部分通过 | 腿/箱穿入显著下降，物体误差略改善；但 lift/floor-contact 未改善，仍不是 strict success。 |
| C4 box023_p2 guard 不被破坏 | 通过 | 物体误差和稳定性基本持平，手接触略降但仍通过 case-window / numeric。 |

## 分析

### 1. 加腿/脚碰撞是必要的物理修正

E080 的问题不是 MuJoCo 不支持腿-箱碰撞，而是当前 scene 没建 pair。E081 证明新增 pair 后，几何穿入能明显下降：box025 p2 的 case-window interference 从 `28.9%` 到 `7.5%`。

### 2. box025 的主要剩余问题转向 lift/floor-contact

加碰撞没有显著提升 object bottom proxy：E080 `-0.073m`，E081 `-0.075m`。这说明 box025_p2 的核心瓶颈现在更像“没有真正抬起/箱体仍大量触地”，不是单纯腿穿模。

### 3. box023 guard 说明该改动不会普遍破坏已成功 case

box023 baseline 本来几乎没有腿/箱干涉；E081 只引入 `2.7%` 的 case-window 腿接触，主指标基本持平。这说明腿/脚-箱碰撞可以作为默认物理合理性组件继续测试，但需警惕性能成本。

## 下一步建议

1. 将 E081 eval 的 `leg_box_interference`、`leg_object_contact`、`object_floor_contact`、`object_bottom_proxy` 纳入后续统一评估。
2. 对 box025_p2 下一步不应继续只调腿碰撞；应转向 lift/floor-contact 相关判据和 reward，或承认单 G1 对大箱只能达到扶/推/partial carry。
3. 若要扩大到 E079 其他 case，优先用 `_legobj` 派生 scene 批量重跑 2-3 个视觉正例，确认性能和 guard 稳定后再推广。
