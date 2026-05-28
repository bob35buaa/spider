# E084 追加诊断：contact mask、raw 接触点与 G1 动态目标语义核查

日期：2026-05-28

对应结果：`workspace/core4d/log/106_E084_box021_constraint_groups_results.md`

## 结论

这次诊断修正了一个重要表述：之前说“左手目标基本在箱体下沿/底面附近”不准确。按 MuJoCo object body frame 正确解释后，`d003_box021_20231018_029_p2` 的 box 底面是 `-y`，顶面是 `+y`，不是 local `-z`。E084 的 G1 ref 动态目标里：

- 左手 reward point 主要在 `-x / -z` 侧面，投影到箱体表面的垂直比例均值 `0.836`，不是底面；
- 右手 reward point 垂直比例均值 `0.870`，也是偏上侧；
- 视觉里“手往下够”的现象不能简单归因为 contact target 在底面。

但更大的问题被确认了：**contact mask 本身没有明显错人/错帧，raw CORE4D 几何接触也很强；真正偏掉的是“用 G1 retargeted wrist + 固定 5cm offset 反推接触目标”这一层。** raw 接触点和 G1 动态目标在 object-local 表面上平均相差约 `27cm`，并且主要沿箱体 `x` 轴在相反侧。

因此 E084/E083 的失败不是单纯数据预处理 mask 错，也不是“左手目标在底面”这么简单，而是当前方法把 raw 灵巧手/手指接触语义压缩成 G1 wrist pseudo point 后，接触位置语义严重失真。

## 诊断产物

| 类型 | 路径 |
|---|---|
| 脚本 | `workspace/core4d/scripts/eval/audit_E084_contact_target_semantics.py` |
| summary | `workspace/core4d/results/E084/contact_target_audit/summary.json` |
| G1 target CSV | `workspace/core4d/results/E084/contact_target_audit/g1_contact_targets.csv` |
| raw contact CSV | `workspace/core4d/results/E084/contact_target_audit/raw_contact_points.csv` |
| paired 对比 CSV | `workspace/core4d/results/E084/contact_target_audit/paired_g1_vs_raw.csv` |
| scatter 图 | `workspace/core4d/results/E084/contact_target_audit/left_local_scatter.png`, `right_local_scatter.png` |
| 垂直位置曲线 | `workspace/core4d/results/E084/contact_target_audit/vertical_fraction_timeseries.png` |

## 坐标系修正

raw mesh local 不能直接和 MuJoCo scene 的 object-local 比。

MuJoCo 编译 `box021_m.obj` 时给 `object_visual` 生成了固定局部变换：

```text
object_visual geom_pos  = [0.00093, 0.00014, -0.00111]
object_visual geom_quat = [0.70520, -0.00969, 0.00085, 0.70894]
```

raw mesh AABB 半尺寸：

```text
[0.1955, 0.1515, 0.2500]
```

MuJoCo body frame 下的视觉 mesh 半尺寸约为：

```text
[0.1509, 0.1984, 0.2515]
```

当前 collision box 半尺寸：

```text
[0.1596, 0.2089, 0.2647]
```

所以本诊断先把 raw contact surface points 从 raw mesh local 变换到 MuJoCo object body frame，再和 G1 target 比。没有这一步会把 `z` 侧面误读成“底面/顶面”。

## Contact Mask 核查

E084 override 的 `contact_hdmi_mask_time_axis=auto` 在 `run_mjwp.py` 中选择：

```text
key: eval_contact_mask_3cm
source length: 125
resized length: 200
person_idx: 1 = person2
```

active 范围：

| 项 | Left | Right |
|---|---:|---:|
| source active | `68.0%` | `73.6%` |
| resized active | `68.5%` | `74.0%` |
| qpos ref active frame | `41-188` | `41-188` |
| raw frame used | `75-129` | `75-129` |

结论：mask 的人物和时序没有发现明显问题。它只提供“这只手此时应该接触”的二值门控，不提供接触点位置。

## Raw CORE4D 接触点

person2 raw 3cm 几何接触很强，不像 mask 虚假触发：

| 手 | broad hand min dist mean | fingertip min dist mean | raw surface 主分布 | surface vertical fraction |
|---|---:|---:|---|---:|
| Left | `1.8mm` | `7.6mm` | `-z` 35f, `+y` 15f, `+x` 1f | mean `0.077` |
| Right | `2.2mm` | `9.6mm` | `+z` 50f, `+y` 5f | mean `0.865` |

这里的 vertical fraction 是箱体表面从底到顶的比例，`0=底部`，`1=顶部`。

解释：

- raw left 确实偏低侧面/下沿，尤其 broad hand region；但这不是“手接触地面”，而是手/指在箱体侧面靠下的位置。
- raw right 是偏上侧面。
- 用户从视频看到“指尖接触箱子侧面中部”可能和视角/手掌区域有关；按当前 SMPL-X + object mesh proxy，left 接触更靠下，right 更靠上。这个 proxy 不是人工接触真值，但距离和顶点数都说明接触信号不是随机噪声。

## G1 动态目标核查

当前 `contact_hdmi_dynamic_target=True` 时，目标点不是 raw fingertip，也不是 raw 接触 centroid，而是：

```text
target(t, hand) = object_local( G1_ref wrist_yaw_link + [0.05, 0, 0] )
```

物理接触几何则是：

```text
contact site: wrist_yaw_link + [0.08, 0, 0]
lh/rh sphere center: wrist_yaw_link + [0.10, 0, 0], radius=0.05
```

G1 ref active target：

| 手 | reward point local mean | 主 face | surface vertical fraction | target-to-surface dist |
|---|---|---|---:|---:|
| Left | `[-0.245, +0.204, -0.276]` | `-x` 117f, `-z` 20f | mean `0.836` | mean `11.9cm` |
| Right | `[-0.226, +0.251, +0.239]` | `+z` 87f, `-x` 52f, `+y` 9f | mean `0.870` | mean `11.6cm` |

关键差异：

| 手 | G1 surface vs raw surface mean delta | 主要偏差 |
|---|---:|---|
| Left | `27.2cm` | `x` 方向约 `-26.3cm`，且 raw low side vs G1 upper side |
| Right | `26.8cm` | `x` 方向约 `-25.8cm`，高度相近但仍在不同侧 |

这说明 G1 ref wrist pseudo point 已经不代表 raw hand/fingertip 接触点。contact mask 可以完全正确，target 仍然错误。

## Sim 结果补查

把 E083/E084 sim 实际手部 reward point 也放到同一口径下检查：

| Variant | surface vertical frac L/R | hand bottom min | mean target dist L/R |
|---|---:|---:|---:|
| E083 | `0.598 / 0.857` | L `-2.4mm` | `9.7 / 6.2cm` |
| E084A | `0.682 / 0.813` | L/R `>9cm` | `7.2 / 7.8cm` |
| E084B | `0.355 / 0.448` | L/R `-1.3 / -0.4cm` | `20.9 / 34.8cm` |
| E084C | `0.669 / 0.826` | L/R `>7cm` | `7.2 / 7.2cm` |

解释：

- A/C 的手不是在追地面，hand-floor penalty 也确实把手撑地压住了。
- B 的 upright/ctrl 组合没有追上目标，实际手部会低/撑地，是它失败模式的一部分。
- A/C 仍失败，是因为它们追的是已经偏离 raw 接触语义的 G1 pseudo target；追得越好，不等于更像 raw hand-object 支撑。

## 根因判断

1. **不是主要的 mask 预处理问题。**  
   mask 选择了正确 person2、时序映射合理，raw 几何接触距离也很小。

2. **raw 数据本身有不理想但真实的语义复杂度。**  
   person2 left 在 raw proxy 中偏低侧面，right 偏上侧面；这不是标准“对侧中部夹持”。如果这个 case 要作为正例，后续要承认它是低-高非对称握姿，而不是假设双手都在中部侧面。

3. **当前方法的问题最大。**  
   `wrist_yaw_link + fixed eef_offset` 不是 fingertip/contact centroid。OmniRetarget 后 G1 wrist pseudo point 与 raw 接触表面相差约 `27cm`，而 reward 又完全依赖这个点，所以 CEM 收到的是错误 contact geometry。

4. **“头/躯干压箱”不是因为左手目标在底面。**  
   更合理的解释是：CEM 在错误/不可达接触目标、强 object tracking、有限 hand geometry 和无站立/支撑先验之间折中，找到蹲抱/压箱局部解。E083/E084 的碰撞与 penalty 只能限制手撑地或深穿模，不能把错误目标变成可承重搬运。

## 下一步建议

E085 的第一阶段应把 contact target 生成从 G1 wrist pseudo point 中解耦出来：

1. 用 raw SMPL-X hand/object surface centroid 或 fingertip-nearest surface 生成 per-frame object-local contact target，先低通/聚类，避免单帧噪声。
2. target 必须在 MuJoCo object body frame 中生成，包含 `object_visual` 的固定 mesh transform。
3. 对 G1 hand sphere，reward 点应区分“目标表面点”和“geom center 应在表面外 radius 处”，不要继续用同一个 `[0.05,0,0]` 同时代表 palm surface、finger contact 和 reward point。
4. 对 `d003_box021_20231018_029_p2`，先做 kinematic feasibility：能否让 G1 双手分别到 raw left 低侧面、raw right 高侧面，同时 pelvis/torso/head 安全。如果 IK 都不稳，就不要进 CEM。
5. 若继续走 COLA/support-body 路线，应让 support body 负责物体承重/稳定，手部 reward 只做语义接触闭合；否则错误 contact target 会继续把 CEM 拉向压箱局部解。
