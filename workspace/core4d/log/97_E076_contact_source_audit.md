# E076 诊断: CORE4D contact source audit 与 box023 右手接触解释修正

## 状态

本诊断修正 E075 后续讨论里的一个过强推断：

> “ref 已进入 putdown，右手不一定应该继续强接触；mask 让右手 reward 继续生效。”

这个说法只能描述 **robot retarget / MuJoCo ref 几何口径**，不能直接解释为 CORE4D 真实动作或 raw mocap 中 person1 右手无接触。CORE4D 是双人协作搬运，物体可以由 person2 支撑；同时 raw 发布层也没有人工 hand-contact 真值。因此后续必须区分四层证据：

1. 真实物理接触：当前本地数据无法直接观测。
2. CORE4D raw SMPL-X + object mesh 几何接触：可计算 proxy，但受 SMPL-X 拟合、mesh pose、阈值影响。
3. OmniRetarget/G1 kinematic ref 的 hand-object 几何：当前 SPIDER 主要使用这一层。
4. MJWP reward mask：当前由 G1 ref 几何估计，不是 CORE4D/HDMI 原始标签。

## 数据与代码证据

### 1. Raw CORE4D 没有逐帧右手接触真值

官方 raw sequence 包含：

- `person1_poses.npz` / `person2_poses.npz`: `vertices (T,10475,3)`, `joints (T,127,3)`, pose/betas/hand pose。
- `smooth_objposes.npy`: `(T,4,4)` object pose。
- `object_metadata.json`。

没有“person1 right hand contact = 0/1”的人工标签，也没有 raw marker/C3D/BVH 可直接回查真实手指接触。

官方 benchmark contact 是几何生成：

- `/home/ubuntu/Workspace/CORE4D-Instructions/benchmarks/motion_forecasting/MDM_InterDiff/interdiff/data/prepare_hho.py`
- `/home/ubuntu/Workspace/CORE4D-Instructions/benchmarks/motion_forecasting/MDM_InterDiff/interdiff/data/prepare_behave.py`

`ContactLabelGenerator.get_contact_labels(..., thres=0.02)` 使用 SMPL/person mesh 和 object surface point 的距离，阈值 2cm。官方可视化里的 runtime contact 使用 `/home/ubuntu/Workspace/CORE4D-Instructions/benchmarks/data_processing/utils/visualization.py`，阈值 3cm。

结论：raw 层能做的是 **SMPL-X 几何接触 proxy**，不能当成真实物理接触真值。

### 2. `box023_person1` 的源序列与时间对齐

本地 case `box023_person1` 对应 Holosoma/CORE4D 源序列：

```text
raw:       /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions/20231008/045
object:    Box023
action:    move2_obs0
T raw:     178
person1:   vertices/joints, T=178
person2:   vertices/joints, T=178
obj pose:  smooth_objposes.npy, T=178
```

Holosoma retarget data：

```text
converted: 178 frames
retargeted: 178 frames
trimmed: 136 frames
SPIDER trajectory_kinematic: 136 frames
```

`trimmed` 与 `retargeted[42:178]` 精确一致，因此：

```text
SPIDER 30Hz ref frame k = raw frame k + 42
```

E075 视频/评估帧是 50Hz；所以用户指出的 `f115-f130` 不是 raw/ref 第 115-130 帧，而是约 `2.30-2.60s`：

```text
eval frame f -> time = f / 50
ref frame ~= round(time * 30)
raw frame ~= 42 + ref frame
E075 f115-f130 -> raw frames 111-120
```

这一点很重要：直接把视频 f120 当 raw f120 或 SPIDER ref f120 会错位。

## Raw 几何接触核验

核验方式：

- 读取 raw `20231008/045` 的 `person1/person2` SMPL-X vertices 和 `smooth_objposes.npy`。
- 加载 `object_models/box/box023_m.obj`，采样 20k object surface points。
- 对每个 eval frame 映射到 raw frame 后，用 KDTree 计算 hand proxy 到 object surface 的最近距离。
- hand proxy：
  - SMPL-X fingertip ids 来自官方 `vertex_ids.py`。
  - broad hand vertex ranges 用作手部区域近似：left `4700:5500`，right `7500:8150`。

注意：broad hand range 是 proxy，不是官方 hand segmentation；但它比单点 fingertip 更适合判断“手表面是否接近物体”。

### E075 f115-f130 对应 raw 111-120

距离单位为 cm；`n03` 是 hand-range 中距离 object surface 小于 3cm 的顶点数。

| eval f | raw f | p1 L min | p1 R min | p1 L n03 | p1 R n03 | p2 L min | p2 R min | p2 L n03 | p2 R n03 |
|--------|-------|----------|----------|----------|----------|----------|----------|----------|----------|
| 115 | 111 | 0.56 | 2.05 | 135 | 28 | 0.22 | 0.14 | 183 | 262 |
| 116 | 112 | 0.36 | 2.11 | 156 | 26 | 0.10 | 0.14 | 163 | 234 |
| 118 | 113 | 0.74 | 2.19 | 163 | 18 | 0.15 | 0.07 | 187 | 222 |
| 120 | 114 | 0.72 | 2.22 | 168 | 18 | 0.09 | 0.11 | 173 | 251 |
| 123 | 116 | 0.77 | 2.53 | 144 | 9 | 0.55 | 0.11 | 145 | 246 |
| 126 | 118 | 1.05 | 1.93 | 198 | 35 | 0.09 | 0.10 | 180 | 273 |
| 128 | 119 | 0.72 | 1.75 | 203 | 46 | 0.03 | 0.05 | 207 | 298 |
| 130 | 120 | 0.81 | 1.79 | 201 | 41 | 0.05 | 0.06 | 198 | 277 |

窗口聚合：

| eval window | raw window | person | hand | min dist mean/min/max | frames <2cm | frames <3cm | frames <5cm |
|-------------|------------|--------|------|-----------------------|-------------|-------------|-------------|
| 100-114 | 102-110 | p1 | L | 0.82 / 0.56 / 1.36 | 15/15 | 15/15 | 15/15 |
| 100-114 | 102-110 | p1 | R | 2.38 / 1.89 / 3.26 | 7/15 | 13/15 | 15/15 |
| 100-114 | 102-110 | p2 | L | 0.10 / 0.05 / 0.15 | 15/15 | 15/15 | 15/15 |
| 100-114 | 102-110 | p2 | R | 0.12 / 0.06 / 0.19 | 15/15 | 15/15 | 15/15 |
| 115-130 | 111-120 | p1 | L | 0.74 / 0.42 / 1.08 | 16/16 | 16/16 | 16/16 |
| 115-130 | 111-120 | p1 | R | 2.12 / 1.77 / 2.58 | 5/16 | 16/16 | 16/16 |
| 115-130 | 111-120 | p2 | L | 0.17 / 0.03 / 0.53 | 16/16 | 16/16 | 16/16 |
| 115-130 | 111-120 | p2 | R | 0.11 / 0.04 / 0.17 | 16/16 | 16/16 | 16/16 |
| 131-145 | 121-129 | p1 | L | 13.55 / 0.90 / 38.84 | 4/15 | 5/15 | 5/15 |
| 131-145 | 121-129 | p1 | R | 11.15 / 1.94 / 30.45 | 2/15 | 4/15 | 5/15 |
| 131-145 | 121-129 | p2 | L | 25.06 / 0.78 / 71.67 | 2/15 | 2/15 | 4/15 |
| 131-145 | 121-129 | p2 | R | 15.13 / 0.11 / 41.63 | 4/15 | 5/15 | 5/15 |

## 解释修正

### 1. “person1 右手无接触”不是当前证据支持的结论

在 E075 f115-f130 对应 raw 111-120：

- person1 左手强接触：min distance 约 0.4-1.1cm。
- person1 右手是 **边界接触**：
  - 2cm 阈值下仅 5/16 帧触发。
  - 3cm 阈值下 16/16 帧触发，且每帧有 9-46 个 broad hand-range vertices <3cm。
- person2 双手强接触：min distance 约 0.03-0.55cm，左右手每帧都有大量 vertices <3cm。

因此不能说 raw/SMPL-X 里 person1 右手“没有接触”。更准确的说法是：

> raw SMPL-X 几何显示 person1 右手在 putdown transition 附近处于 2cm/3cm 阈值边界；person2 双手则持续强接触。若使用官方 2cm contact-label 口径，person1 右手可能被标成间歇接触；若使用 3cm 可视化口径，则会被标成持续接触。

这也解释了用户提出的物理疑问：即使 person1 右手局部松开，object 也可以由 person1 左手和 person2 双手支撑，不会必然掉落。

### 2. 当前 SPIDER/MJWP 的 contact mask 不是 raw/HDMI contact label

SPIDER processed `trajectory_kinematic.npz` 只有单人 G1 + object：

```text
qpos:        (136,43)
ctrl:        (136,29)
contact:     (136,2), 全帧 [1,1]
contact_pos: (136,2,3)
```

转换器 `spider/process_datasets/core4d.py` 默认 `contact_detection_mode="one"`，直接把两只手 contact 置 1。这个字段只是下游 guidance 的启用信号，不是 raw contact 真值。

E039/E075 的 reward mask 又由 G1 kinematic ref 的 wrist/body-origin 到 box SDF 估计，而且当前仍是 scalar `(T,)`，不是 HDMI 风格 per-EEF `(T,2)`。

所以 E075 quick audit 里的：

```text
f115-f130 estimated mask right = 100%
f115-f130 MuJoCo ref geom right contact = 31.3%
```

只能说明 **G1 retargeted ref / MuJoCo geom / mask 口径不一致**，不能推出 CORE4D raw person1 右手真实无接触。

### 3. 更可能的问题位置

当前证据把问题从“raw 右手无接触”改写为：

1. **raw person1 右手是边界接触，partner 强接触**
   真实双人动作里，person1 右手是否承担主支撑不能只看单人右手。

2. **retarget 到单 G1 后丢掉了 partner，并且 G1 hand/box 几何变弱**
   `box023_person1` 只保留 person1；若 raw 中 object 的稳定由 person2 双手强支撑共同实现，单 G1 必须补偿这个缺失。

3. **current mask 把接触问题进一步简化错了**
   `contact=(1,1)` 全帧 + scalar SDF mask 会把“弱/边界右手接触”和“强左手接触”混在一起，给 CEM 的信号不是 raw 的 per-hand、per-person 接触结构。

4. **f120-f125 右腿前跨可能仍与 mask/reward 有关，但机制要重说**
   不是“右手本来应该无接触却被强接触”，而是：
   - raw 的 person1 右手是边界接触；
   - retargeted G1 右手在 MuJoCo geom 里接触弱；
   - reward/mask 却把双手 contact guidance 打开；
   - 同时 partner 支撑在当前单机器人任务里不存在；
   - CEM 可能用右腿前跨/重心调整去补一个单 G1 物理上更难承担的 support/contact 需求。

## 对下一步实验的影响

下一步不应继续调 `hold_contact_start_eval_time` / `hold_contact_end_eval_time` 这种 hand-crafted window。

更合理的 E076 主线应先做 contact source alignment：

1. 从 raw `20231008/045` 生成可复现的 per-person/per-hand contact proxy：
   - 2cm 官方 label 口径；
   - 3cm visualization 口径；
   - 输出 p1/p2 L/R contact timeline。
2. 把 raw contact proxy 映射到 30Hz `trajectory_kinematic`，再映射到 MJWP eval horizon。
3. 比较三条 timeline：
   - raw p1/p2 hand contact proxy；
   - G1 retargeted MuJoCo geom contact；
   - 当前 MJWP estimated mask。
4. 决定 reward mask 使用策略：
   - 至少改成 per-EEF `(T,2)`；
   - 不再 scalar broadcast；
   - 对 raw 边界接触用 soft weight，而不是硬开/硬关；
   - 对 person2 强支撑缺失，需要明确是忽略、等效为 object/support prior，还是引入 partner proxy，而不能假装单人数据已经包含全部支撑。

## 结论

用户的质疑成立。此前把 robot ref 右手 MuJoCo contact 低解释成“ref 右手不一定应该继续强接触”是不充分的。现在的证据显示：

- raw 层没有人工 hand-contact 真值，只能用 SMPL-X 几何 proxy。
- 对 E075 f115-f130，raw SMPL-X 中 person1 右手不是明显无接触，而是 2cm 阈值边界、3cm 阈值持续接触。
- person2 双手在同一窗口强接触，双人协作支撑必须纳入解释。
- 当前 SPIDER processed contact 全 1，E039/E075 mask 是从单 G1 ref 估计，不是 CORE4D/HDMI label。

因此下一步应把 E076 定义为 **contact source alignment / per-hand mask 修复**，而不是继续围绕手写 hold/release 时间窗调参。
