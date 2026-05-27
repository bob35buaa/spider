# E082 趴倒 / 上半身碰撞诊断

日期：2026-05-27

## 问题

用户观察到 E082 的 sim 在弯腰搬箱时经常直接趴倒，手会接触地面；`20231018_029_p2` 里头部甚至栽进箱子内部。用户提出是否应像 E081 的腿/脚一样，加入 head/torso 与物体的碰撞检测。

本日志对比：

- E082 三个 D003 Box021 失败 case；
- E081 的 `box023_person2_legobj` guard，这是此前“同类动作里表现较好”的对照。

诊断脚本：

```text
workspace/core4d/scripts/eval/diagnose_E082_body_fall.py
```

输出：

```text
workspace/core4d/results/E082/diagnostics/body_fall_vs_box023.csv
logs/E082/diagnose_body_fall_vs_box023.log
```

## 关键事实

当前 `scene_act.xml` 里已经有 `head_collision` 和 `torso_collision` geom，但 E081/E082 只给以下 geoms 配了 `object_collision` pair：

```text
legs/feet: hip, thigh, shin, linkage_brace, lf0-lf3, rf0-rf3
hands: lh, rh
```

没有：

```text
head_collision - object_collision
torso_collision - object_collision
pelvis_collision - object_collision
shoulder/elbow collision - object_collision
```

因此，`20231018_029_p2` 里头进入箱子不是偶发的 contact solver 漏检，而是当前 scene 明确允许头/躯干穿过 `object_collision`。同时，scene 里有 `floor-lh/rh` pair，所以手撑地是物理有效的 contact 通道。

## 诊断结果

| Case | object half xyz | head-object pair | torso-object pair | head penetration | torso penetration | hand-floor contact | pelvis<45cm | sim contact | obj mean |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| E082 `20231018_029_p2` | `0.1596/0.2089/0.2647` | False | False | `76.0%` | `85.3%` | L `27.1%`, R `79.1%` | f100 | `14.7%` | `0.680m` |
| E082 `20231011_035_p2` | `0.1596/0.2089/0.2647` | False | False | `17.2%` | `51.1%` | L `0.0%`, R `37.9%` | f100 | `37.4%` | `0.457m` |
| E082 `20231020_019_p1` | `0.1596/0.2089/0.2647` | False | False | `32.4%` | `58.8%` | L `75.0%`, R `2.7%` | f100 | `69.6%` | `0.849m` |
| E081 `box023_p2` | `0.1531/0.1568/0.1766` | False | False | `0.0%` | `0.0%` | L `0.0%`, R `0.0%` | never | `66.7%` | `0.164m` |

最小 signed distance 也说明 ref 本身并不要求头/躯干穿箱：

| Case | sim head min | sim torso min | ref head min | ref torso min |
|---|---:|---:|---:|---:|
| E082 `20231018_029_p2` | `-18.7cm` | `-25.0cm` | `+17.6cm` | `+7.8cm` |
| E082 `20231011_035_p2` | `-10.8cm` | `-25.0cm` | `+30.1cm` | `+10.0cm` |
| E082 `20231020_019_p1` | `-6.0cm` | `-20.7cm` | `+12.1cm` | `+7.9cm` |
| E081 `box023_p2` | `+8.3cm` | `+13.9cm` | `+8.8cm` | `+11.5cm` |

时序上，穿箱/手撑地发生在 f100 的稳定性阈值之前：

| Case | first head penetration | first torso penetration | first hand-floor contact | first pelvis<45cm |
|---|---:|---:|---:|---:|
| `20231018_029_p2` | f37 | f40 | L f43 / R f40 | f100 |
| `20231011_035_p2` | f51 | f50 | R f60 | f100 |
| `20231020_019_p1` | f58 | f80 | L f45 | f100 |
| `box023_p2` | none | none | none | none |

## 解释

这不是单纯的“没有接触到箱子所以顺着惯性继续往下倒”。

更准确地说，E082 的 CEM 在弯腰/接触阶段找到了一个未被惩罚的局部最优：

1. contact mask 在 f115-f130 是打开的，三个 E082 case 左右手都是 `100%/100%` active，所以不是 mask 没开。
2. Box021 比 box023 更高更深，`object_collision` z half `0.2647m`，box023 是 `0.1766m`；弯腰 reach 更难，CEM 更容易把上身压进箱体。
3. 当前没有 head/torso-object pair，所以头和躯干不会被物理阻挡。
4. 当前有 hand-floor pair，所以手撑地是可行的物理支撑；但 reward 没有惩罚“手撑地替代手托箱”。
5. `stability_penalty_scale=0.0`、`task_body_rew_scale=0.0`，局部 frame body reward 对这种 head/torso 穿箱和手撑地没有足够约束。
6. E082 的 robot ctrl 大偏离在 f100 立刻出现，`case_window_robot_ctrl_linf_max≈2.05-2.20`，而 box023 guard 只有约 `0.62`，且 first large ctrl 延后到 f140。说明在 Box021 接触阶段，CEM 为了追物体/接触目标主动牺牲了全身姿态。

box023 做得好的原因不是 scene 里有 head/torso pair；它同样没有。区别是 box023 的物体更小，person2 ref/contact 质量更好，CEM 不需要借助 head/torso 穿箱或手撑地也能达到较好 contact/object objective。

## 是否应该加 head/torso collision

应该加，但它是必要条件，不是充分条件。

建议下一步 E083 不只加 head/torso，而是做两个层次：

### E083A: upper-body-object collision pairs

在派生 scene 里新增 object pairs：

```text
head_collision - object_collision
torso_collision - object_collision
pelvis_collision - object_collision
left/right_shoulder_yaw_collision - object_collision
left/right_elbow_yaw_collision - object_collision
```

同时把 eval 指标扩展为：

```text
head_object_penetration_pct
torso_object_penetration_pct
upperbody_object_penetration_pct
hand_floor_contact_pct
first_head_object_penetration_frame
first_hand_floor_contact_frame
```

只加 pair 的预期：

- 能阻止明显的头/躯干穿箱；
- 但 CEM 可能改成用胸/肩推箱、被箱子顶翻、或更早摔倒；
- 因此 E083A 应先跑 1 个最典型失败 case 加 1 个 box023 guard，不能直接全量推广。

### E083B: upper-body / hand-floor penalty

如果 E083A 只是把穿箱变成撞箱/摔倒，应加 reward penalty：

```text
upperbody_object_penalty: head/torso/pelvis/shoulder/elbow SDF < margin
hand_floor_penalty: lh/rh 与 floor contact，尤其在 contact mask active 时
stability_penalty: pelvis/torso upright guard，避免 f100 直接趴倒
stronger ctrl_ref_guard: 接触阶段减少 robot ctrl Linf≈2rad 的大偏离
```

这一步才有机会真正减少“趴下搬箱”的局部最优。

## 决策建议

下一轮不要继续复跑 E082 原配置。推荐 E083：

1. `E083A_upperbody_pair`: `20231018_029_p2` + `box023_p2 guard`，只加 upper-body-object pairs 和诊断指标。
2. 若 `20231018_029_p2` 仍倒伏，进入 `E083B_upperbody_penalty`: 加 upper-body penetration penalty + hand-floor penalty + stability/ctrl guard。
3. 成功标准不再只看 object/contact，必须同时要求：
   - head/torso penetration `0%`；
   - hand-floor contact `≈0%`；
   - pelvis z min `>=0.55m`；
   - case-window object mean 明显低于 E082；
   - box023 guard 不退化。
