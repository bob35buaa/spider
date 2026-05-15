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

## 脚本路径与运行命令

### 指标计算脚本

E081 的量化指标由以下脚本计算：

| 用途 | 路径 | 说明 |
|------|------|------|
| E081 主 eval | `workspace/core4d/scripts/eval/eval_E081.py` | 计算 `Leg intf`、`Leg contact`、`near_2cm`、`object_floor_contact`、`object_bottom_proxy`、case-window success 与 aggregate summary |
| E080 baseline eval | `workspace/core4d/scripts/eval/eval_E080.py` | E081 log 中 E080 baseline 对比的来源之一；E081 后续以 `eval_E081.py` 的 leg/object 指标为准 |
| 派生 scene 生成 | `workspace/core4d/scripts/E081/create_legobj_cases.py` | 从原始 task 复制数据，新建 `*_legobj` 派生 case，并只在派生 `scene_act.xml` 加腿/脚-箱 contact pair |
| override 生成 | `workspace/core4d/scripts/E081/generate_e081_overrides.py` | 根据 `variants.tsv` 生成 E081 Hydra override，并复制对应 3cm contact mask |
| variant 配置 | `workspace/core4d/scripts/E081/variants.tsv` | 定义 source task、derived task、mask 来源、person idx、本地/远程 split 和 main/guard 角色 |

E081 新增指标的核心计算位置：

```text
workspace/core4d/scripts/eval/eval_E081.py
  - 逐帧记录 leg_object_contact_count / object_floor_contact_count / object_bottom_proxy_m: 约 L196-L205
  - 汇总 Leg intf / near_2cm / Leg contact / Floor contact / Bottom proxy: 约 L235-L251
  - Leg/Lift Proxy 判据: 约 L286-L294
```

### 实验运行脚本

| 用途 | 路径 |
|------|------|
| 预处理入口 | `workspace/core4d/scripts/run_E081_preprocess.sh` |
| CEM 本地/远程训练入口 | `workspace/core4d/scripts/train/train_E081.sh` |
| 远程 GPU1 启动入口 | `workspace/core4d/scripts/run_E081_remote.sh` |
| 远程结果回收入口 | `workspace/core4d/scripts/pull_E081_remote_results.sh` |

本轮实际/可复现命令：

```bash
# 1. 生成 *_legobj 派生 case 与 E081 overrides
bash workspace/core4d/scripts/run_E081_preprocess.sh

# 2. 本地 RTX5090 跑 local split: E081_box025_p2_legobj
bash workspace/core4d/scripts/train/train_E081.sh local 0

# 3. 远程 spider-remote 仅用 GPU1 跑 remote split: E081_box023_p2_legobj
bash workspace/core4d/scripts/run_E081_remote.sh

# 4. 远程监控
ssh spider-remote "tmux capture-pane -t E081 -p | tail -40"

# 5. 远程完成后回收结果并在本地重跑合并 eval
REMOTE_HOST=spider-remote REMOTE_REPO=/home/xiayb/pHRI_workspace/spider \
  bash workspace/core4d/scripts/pull_E081_remote_results.sh

# 6. 只重跑 E081 全量 eval，不重新跑 CEM
bash workspace/core4d/scripts/train/train_E081.sh eval
```

单个 variant 调试命令：

```bash
# 本地单独跑 box025 p2 leg-object 派生 case
bash workspace/core4d/scripts/train/train_E081.sh single 0 E081_box025_p2_legobj

# 本地单独跑 box023 p2 leg-object 派生 case
bash workspace/core4d/scripts/train/train_E081.sh single 0 E081_box023_p2_legobj
```

## 量化结果

### 指标定义

E081 新增的腿/脚-箱指标分两类：几何干涉与 MuJoCo 真实接触。

`Leg intf` / `leg_box_interference_frames_pct`
: 几何干涉比例。它不依赖 MuJoCo 是否真的产生 contact force，而是离线计算所有腿/脚 geom 到 `object_collision` box 的 adjusted signed distance：

```text
leg_box_sdf_min_m = min(所有腿/脚 geom 到 object_collision box 的 adjusted signed distance)
Leg intf % = leg_box_sdf_min_m < 0 的帧比例
```

解释：

- `leg_box_sdf_min_m < 0`：腿/脚几何体穿入箱体碰撞盒，数值越负穿入越深。
- `leg_box_sdf_min_m = 0`：刚好贴边。
- `leg_box_sdf_min_m > 0`：腿/脚和箱体之间有间隙。

纳入统计的腿/脚 geom：

```text
left/right_hip_collision
left/right_thigh_collision
left/right_shin_collision
left/right_linkage_brace_collision
lf0-lf3
rf0-rf3
```

`Leg contact` / `leg_object_contact_frames_pct`
: MuJoCo 真实腿/脚-箱接触比例。它统计 `data.contact` 里是否存在上述腿/脚 geom 与 `object_collision` 的 contact：

```text
leg_object_contact_count = 当前帧 MuJoCo contact list 中腿/脚-箱 contact 数
Leg contact % = leg_object_contact_count > 0 的帧比例
```

注意：E080 原始 scene 没有腿/脚-箱 contact pair，所以即使视觉或几何上腿穿进箱子，MuJoCo 也不会产生腿/脚-箱接触力，`Leg contact` 会是 `0%`。E081 新建 `*_legobj` 派生 scene 后，腿/脚-箱 contact pair 生效，因此 `Leg contact` 才能反映真实物理接触。

`near_2cm`
: 几何接近比例，不要求穿入：

```text
leg_box_near_2cm_frames_pct = leg_box_sdf_min_m < 0.02 的帧比例
```

它用于观察腿/脚是否长期贴近箱体，即使尚未穿入。

`object_floor_contact_frames_pct`
: 箱子与地面的 MuJoCo 接触比例：

```text
object_floor_contact_frames_pct = object_floor_contact_count > 0 的帧比例
```

`object_bottom_proxy_m`
: 箱底高度的粗 proxy：

```text
object_bottom_proxy_m = object_z - object_half_z
```

它用于辅助判断箱子是否真的离地。`>0` 更接近抬起，`≈0` 接近触地，`<0` 通常说明箱体仍在地面附近或碰撞盒/姿态 proxy 有偏差。该指标需要和 `object_floor_contact_frames_pct` 一起看。

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

### 0. 为什么没有改 reward 也能减少腿/箱干涉

E081 从算法层面看没有新增显式腿部避障 reward，也没有改变 CEM 优化器主体。它沿用 E080/E079 口径：

- 默认 base 仍是 `core4d_e074a_box023`；
- `contact_hdmi_mask_source` 仍是 `core4d_3cm`；
- `hold_contact_rew_scale: 0.0`；
- `hold_contact_start_eval_time/end_eval_time: 0.0`，即没有 box023-specific hand-crafted hold window；
- palm normal、person idx、mask 逻辑与对应 baseline 保持一致。

核心差异只有 scene 物理模型：

```text
E080 box025_p2: task = box025_person2
E081 box025_p2: task = box025_person2_legobj
```

`*_legobj` 派生 task 只在 `scene_act.xml` 中新增腿/脚 geom 到 `object_collision` 的 MuJoCo contact pair。也就是说，E081 不是通过“指标惩罚腿碰箱”做到改善，而是把原来不参与物理约束的腿/脚-箱关系纳入前向动力学。

机制可以理解为：

1. E080 原 scene 没有腿/脚-箱 contact pair，腿/脚几何上穿进箱子时，MuJoCo 不会产生腿/脚-箱接触力；CEM 的 rollout 也不会因为这件事在动力学里付出代价。
2. E081 加 pair 后，类似控制序列一旦让腿/脚靠近或进入箱体，MuJoCo contact solver 会产生法向接触力和摩擦约束，改变机器人、箱体的后续状态。
3. 这些接触力会间接影响已有 objective：物体轨迹误差、手-箱接触、姿态/稳定性、控制 reference 等。即使 objective 没有显式写 `leg_box_sdf` penalty，穿箱方案在 rollout 中也更容易导致物体/身体状态变差。
4. CEM 在同一套 reward 下筛选 elite samples 时，会自然偏向那些既能维持手-箱/物体轨迹、又少触发腿/箱物理冲突的控制序列。

因此，E081 的改善是“物理可行性约束改变了 rollout 分布和 elite selection”，不是“算法显式学会了避开箱子”。这也解释了为什么 box025_p2 的 `Leg intf` 明显下降，但 `object_bottom_proxy` / `floor-contact` 没有改善：腿穿模被物理修正压下去了，箱子没真正抬起来这个瓶颈仍然存在。

另外，`Leg intf` 没有降到 0 是合理的。MuJoCo contact 是软约束，且 E081 使用 `solref="0.008 1"`；接触求解允许小幅 penetration 来产生接触力，所以仍会看到 `7.5%` 的干涉帧和最小约 `-1.2cm` 的 adjusted SDF。

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
