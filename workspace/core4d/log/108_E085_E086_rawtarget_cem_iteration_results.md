# E085/E086 raw contact target 修复与 CEM 失败迭代结果

日期：2026-05-28

对应计划：

- `workspace/core4d/plan/91_E085_raw_contact_target_repair_plan.md`
- `workspace/core4d/plan/92_E086_rawtarget_failure_iteration_plan.md`

## 结论

这轮检查把“左手目标为什么看起来在箱体下沿/底面附近”拆清楚了：

1. **不是 contact mask 错人或错帧造成的。** `d003_box021_20231018_029_p2` 自动使用 `eval_contact_mask_3cm`，`125 -> 200`，person2 left/right active 为 `68.5%/74.0%`，raw active frame 主要是 `75-129`。
2. **raw SMPL-X/object proxy 里，person2 左手接触确实偏低侧面。** main left 的多种候选都很低：`broad_projected` vfrac mean/median `0.057/0.060`，`tip_best_projected` `0.052/0.044`，`tip_mean_projected` `0.063/0.068`；right 则是高侧面，vfrac mean 约 `0.895`。
3. **旧方法确实有 target 构造错误。** E084 之前的动态 target 是 G1 `wrist_yaw_link + [0.05,0,0]`，不是 raw fingertip/contact centroid；它和 raw surface target 平均差约 `27cm`。
4. **E085 又修掉了一个投影 bug，但修完后 left 仍然低。** 旧 projection 对 visual target 在 collision box 内部的点按最大归一化轴选面，会把一部分更靠近 `+y` 侧面的点误投到 `-z` 面；改为 inside 用 nearest-face 后，main left face counts 从旧的 `+x:1/+y:15/-z:35` 变为 `+x:1/+y:35/-z:15`，vfrac 从约 `0.037` 升到 `0.057`，但仍是低侧接触。

所以更准确的判断是：**预处理有 bug，已修；但修完后这个 case 的 raw left contact 对 G1 仍然是低位/难满足目标。后续 CEM 失败主因不是 mask，而是“直接用手部 raw target 承担物体支撑”这条方法路线会诱导头/躯干/手臂压箱的局部解。**

## 本轮改动

| 模块 | 改动 |
|---|---|
| `examples/run_mjwp.py` | `contact_hdmi_target_source=external`，可从 `.npz` 读取 per-frame object-local target；支持 `spider_contact_target_object_local` / `eval_contact_target_object_local` 并 resize 到 `qpos_ref` 长度 |
| `workspace/core4d/scripts/E085/generate_raw_contact_targets.py` | 从 raw CORE4D hand/object surface 生成 object-local contact target；修正 object visual transform 和 inside nearest-face projection |
| `spider/config.py` | 增加 external target 配置，以及 E086 hand-object deep penetration penalty 配置 |
| `spider/simulators/mjwp.py` | 增加 `hand_object_deep_penalty` reward 项，用于只惩罚手部深穿透 |
| `workspace/core4d/scripts/E086/make_vfrac_floor_target.py` | 生成 left minimum vertical fraction target，用于验证“低 target 是否是主因” |

静态检查和 smoke：

- E085/E086 Python 脚本、`examples/run_mjwp.py`、`spider/config.py`、`spider/simulators/mjwp.py` 均通过 `py_compile`。
- E085/E086 shell 脚本均通过 `bash -n`。
- E085 main/guard external target smoke 通过；E086A/E086B 4-step smoke 通过。

## Target 诊断

### Contact mask 与 raw 接触

| 检查项 | 结果 |
|---|---|
| mask key | `eval_contact_mask_3cm` |
| person | `person2` |
| resized length | `125 -> 200` |
| active L/R | `68.5% / 74.0%` |
| raw active frame | `75-129` |
| fingertip 近物体数量 | left active frame 平均 `4.55/5` 个 fingertips 在 `3cm` 内 |
| fingertip min distance | mean `0.76cm` |

main case target 统计：

| hand | target 候选 | vfrac mean/median | 解释 |
|---|---|---:|---|
| left | `broad_projected` | `0.057 / 0.060` | 低侧面 |
| left | `tip_best_projected` | `0.052 / 0.044` | 指尖最近点也低 |
| left | `tip_mean_projected` | `0.063 / 0.068` | 指尖均值仍低 |
| left | `high_close_projected` | `0.103 / 0.107` | 人为选更高近点也只是到中低位 |
| right | `broad_projected` | `0.895 / 0.927` | 高侧面 |

这里的 vfrac 是箱体表面从底到顶的比例，`0=底部`，`1=顶部`。这解释了为什么用户视频里感觉“指尖在侧面中部”，但 raw proxy 统计仍显示 left 低：视频视角、灵巧手 mesh、SMPL-X proxy 与实际 contact label 不完全等价；按当前可计算几何，left 低不是 mask 单独造成的。

### E085 raw target

| variant | hand | raw frames | face counts | vfrac mean/min/max | old G1 target delta |
|---|---|---:|---|---:|---:|
| E085A main | left | `75-129` | `+x:1,+y:35,-z:15` | `0.057/0.005/0.105` | `27.45cm` |
| E085A main | right | `75-129` | `+z:49,+y:5,+x:1` | `0.895/0.532/0.968` | `27.00cm` |
| E085A guard | left | active | mainly `+z` | mean `0.114` | `24.64cm` |
| E085A guard | right | active | `+z/-y` | mean `0.128` | `27.82cm` |

## CEM 结果

### E085 raw-target gate

| Variant | Case | Contact | Obj mean/max | Pelvis min | Head pen | Upper pen | LH/RH object pen | LH/RH floor | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| E085A main | box021 029 p2 | `82.95%` | `0.665/1.068m` | `0.659m` | `18.60%` | `53.49%` | `73.64/54.26%` | `0/0%` | fail |
| E085A guard | box023 p2 | `88.00%` | `0.226/0.431m` | `0.660m` | `2.00%` | `2.00%` | `77.33/27.33%` | `0/0%` | fail |

观察：

- main：contact 数字高、手没有撑地、pelvis 没有直接倒地，但 sim 通过低头、肩/肘/手部深穿和压箱来满足接触/object reward；first head penetration frame 是 `35`。
- guard：整体姿态接近可看，object error 明显好于 main，但 hand-object penetration 很高；按 gate 的 head penetration `<1%` 与 physical proxy 仍失败。

### E086 迭代

| Variant | 目的 | Contact | Obj mean/max | Pelvis min | Head pen | Upper pen | LH/RH object pen | LH/RH floor | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| E086A strict | 加强 hand deep + upperbody penalty，contact gain `5 -> 3` | `76.74%` | `0.698/1.134m` | `0.624m` | `58.91%` | `72.87%` | `35.66/65.12%` | `0/10.85%` | 更差 |
| E086B vfloor | left target 提到 min vfrac `0.2` | `75.19%` | `0.710/1.170m` | `0.613m` | `62.79%` | `68.99%` | `58.91/62.79%` | `0/0%` | 更差 |

观察：

- E086A 的 penalty 没把解带回手部真实承重，反而让姿态更扭，右手出现 `10.85%` floor contact。
- E086B 把 left target 抬高后仍然趴/压箱，说明问题不只是“left target 太低”这一维；CEM 仍会用头/胸/肩/肘在大箱子上找支撑。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1: E085 失败主要是穿透漏洞，堵住即可 | 不成立。E086A 加强 deep/upperbody penalty 后 head/upper penetration 反而升高，object error 也变差 |
| C2: 低位 raw target 是唯一主因 | 不成立。E086B 抬高 left target 后仍压箱，说明 target height 是因素但不是充分解释 |
| C3: 需要 support-body/COLA seed route | 支持。A/B 都失败后，不应继续手调 target/penalty；应切换到 support body / 6-DoF connector 语义 |

## 根因判断

1. **数据预处理不是完全没问题，但不是最终瓶颈。** E085 已修正 raw target 的 object visual transform 与 collision projection 问题；mask/person/time 也没有明显错误。
2. **raw contact label 本身对 G1 不友好。** left 低、right 高的非对称姿态可能来自双人协作时灵巧手/手指的局部接触；直接要求 G1 hand sphere 到这些点，会让机器人过度弯腰。
3. **当前 CEM objective 缺“物体由 support body 承重”的结构。** 当 object tracking/contact reward 很强，而手部几何又无法自然抓住箱体时，优化器最容易找到上半身压箱、手/肘/肩穿箱的局部解。
4. **继续局部调 penalty 或 vfrac 不划算。** E084/E085/E086 已经覆盖 safety penalty、raw target、hand deep penalty、target height 四类小修，失败模式没有转为真实搬运。

## 下一步计划

下一步应进入 E086C/E087：对齐 COLA support body 思路，而不是继续 support proxy anchor sweep。

建议实验组：

1. **Support-body kinematic seed。** 为 box021 main 构造一个跟随物体的 support body，并用 6-DoF connector 表达 support body 与物体的相对关系；先验证没有 CEM 时能否在可视化里维持 object pose 与安全 body pose。
2. **Support-body + low hand gain CEM。** object/support 负责承重和姿态稳定，hand contact reward 降权为语义闭合；成功标准优先看 head/torso penetration、object floor、姿态，而不是 contact 数字。
3. **Guard regression。** 用 `box023_p2` guard 验证 support route 不破坏已知可看 case；若 guard 退化，说明 support connector 或 support body target 设计不对。

停止条件：

- 如果 support body 仍需要人工 anchor 才能跑，则说明当前实现仍是 support proxy，不是 COLA-style support body；应回到 6-DoF connector 设计，而不是继续 anchor sweep。

## 结果路径

| 类型 | 路径 |
|---|---|
| E085 raw target | `workspace/core4d/results/E085/raw_targets/` |
| E085 target selection audit | `workspace/core4d/results/E085/target_selection_audit/` |
| E085 CEM results | `workspace/core4d/results/E085/E085A_rawtarget_main.npz`, `workspace/core4d/results/E085/E085A_rawtarget_guard.npz` |
| E085 videos | `workspace/core4d/results/E085/E085A_rawtarget_main.mp4`, `workspace/core4d/results/E085/E085A_rawtarget_guard.mp4` |
| E085 sheets | `workspace/core4d/results/E085/keyframes/contact_sheets/` |
| E086 CEM results | `workspace/core4d/results/E086/E086A_rawtarget_strict_main.npz`, `workspace/core4d/results/E086/E086B_rawtarget_vfloor_main.npz` |
| E086 videos | `workspace/core4d/results/E086/E086A_rawtarget_strict_main.mp4`, `workspace/core4d/results/E086/E086B_rawtarget_vfloor_main.mp4` |
| E086 vfloor target | `workspace/core4d/results/E086/vfrac_floor_targets/E086B_vfrac_floor_main/raw_contact_targets.npz` |
| E085 metrics | `workspace/core4d/results/E085/comparison.csv`, `workspace/core4d/results/E085/e085_gate_summary.json` |
| E086 metrics | `workspace/core4d/results/E086/comparison.csv`, `workspace/core4d/results/E086/e086_gate_summary.json` |

