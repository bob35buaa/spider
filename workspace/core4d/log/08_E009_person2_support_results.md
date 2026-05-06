# E009: 引入 Person2 力支撑 — 多方案探索结果

## 状态: 全部失败(但提供了关键几何洞察)

## 实验结果对比

所有变体使用统一指标:**box 底面 4 角最低点 z**(真实"离地高度",非物体质心 z)。

| Run | 配置 | obj 质心 z_max (m) | **box 底面 z_max (m)** | pelvis_err | 状态 |
|-----|------|-------------------|---------------------|-----------|------|
| Ref | 真值参考 | 0.533 | **0.150** | — | — |
| E006f | freejoint, no actuator | 0.307 | ≈0 | 0.06 | 推开箱子 |
| E008a | act, residual=0, gain=20 | 0.375 | ≈0 | 0.64 | 引导期 OK,撤除即落 |
| **E009a** | residual=0.3, gain=20 | 0.385 | ≈0 | **0.115** ✓ | 失败 |
| **E009a2** | residual=0.5, gain=30, pos_rew=5 | 0.363 | ≈0 | 0.116 ✓ | 失败 |
| **E009a3** | residual=0.7, gain=100, pos_rew=5 | 0.418 | ≈0 | 0.284 | pelvis 退化 |
| **E009b** | mass=0.05, freejoint | 0.342 | ≈0 | 0.257 | 失败 |
| **E009c** | gravcomp=1, freejoint | 0.381 | **0.008** | 0.278 | 离地仅 8mm |

**关键诊断**:箱子从未真正离开地面(底面 z ≤ 0.008m vs 参考 0.150m)。所谓的"质心 z 提升"全部来自**箱子轻微旋转**(被推时翘起一边),不是真实抬起。

## 根因(几何分析)

在参考中提取双人手部位置,frame 47(峰值):
```
obj 中心: (0.124, -0.346, 0.533),box 半边长 (0.305, 0.305, 0.446)
box 表面 -x face: x=-0.181, +x face: x=+0.429
p1 双手 x ≈ -0.33 (在 -x face 外侧 0.15m,即从 -x 端夹箱)
p2 双手 x ≈ +0.66 (在 +x face 外侧 0.23m,即从 +x 端夹箱)
```

**关键发现**:CORE4D 的真实姿态是**两人各自从 box 两端(±x 方向)对夹**,不是从前面/上面。

E006-E009 所有变体的 contact pair 配置(`lh_1/2/3` ↔ `object_collision`)都让 G1 的双手从前面接近箱子的 **-y 面或顶面**。这与参考的"侧夹"姿态在几何上**完全错位**。

CEM 优化的"从前侧+顶部接触"方案物理上不可能持续抬起 0.5kg 箱子(只能产生水平推力或顶面摩擦),所以箱子要么被推走(E006f),要么被 actuator 暂时拽起再落下(E008a/E009a)。

## Claims 验证

| Claim | 阈值 | 任意变体最佳 | 通过? |
|-------|------|------------|------|
| C-A | residual=0.3, obj_z>0.45 | 0.385 | ❌ |
| C-B | mass=0.05, obj_z>0.45 | 0.342 | ❌ |
| C-C | gravcomp, obj_z>0.45 | 0.381 | ❌ |
| C-视频 | 双手在箱两侧 | 双手在箱前 | ❌ |
| C-pelvis | err<0.20 | E009a/a2 通过 | 部分 ✓ |

## 关键结论

**G1 单 agent + 现有接触几何 不可能抬起 box025**,无论如何调:
- residual force(虚拟伙伴恒力):指标轻改善但箱子不离地
- mass 减少:同上
- gravcomp(完美抗重力):同上(说明问题不是力大小,是**接触点错位**)

**真正的解法必须改变 contact pair 的几何**:
1. CORE4D 真实姿态需要双手从 ±x 方向夹箱
2. G1 的 wrist_yaw_link 需要伸到 box ±x 面外侧
3. 当前 contact pair 让 lh_1/2/3 接触 object 任意面 — CEM 找到的局部最优是"从 -y 推"(因为 G1 站在 -y 侧)

## 下一步:E010 — Side-Grip 几何重构

**根本性方案**:
1. 强制 G1 双手到达 box ±x 端外侧位置(用 IK 重生成 ref qpos,或加 hand-target reward)
2. 添加显式 mocap p2 hands 在 +x 端,与 box 真实接触
3. 这样:p1 从 -x 端夹,mocap_p2 从 +x 端夹 → 物理夹持成立

但是,**G1 的臂展是 0.5m,box 长 0.61m**。p1 站在 box 中央对面,要让双手都到 -x 端,需要扭腰侧身。**几何上很难实现自然姿态**。

更现实的方案:**承认 G1 几何不适合搬大箱**,选择以下之一:
- **(A) 替换为更小的物体**:CORE4D 的 bucket(直径 0.3m)、chair 等更适合 G1
- **(B) 仅训练 IL/RL,不强求物理重定向**:用运动学 ref + 在 RL 中加 HDMI lost-contact
- **(C) 换更大的机器人**:Atlas 等臂展更大的人形

## 结果路径

| 产出 | 路径 |
|------|------|
| 计划 | `workspace/core4d/plan/08_E009_person2_support_plan.md` |
| 配置 | `examples/config/override/core4d_box025_e009{a,a2,a3,b,c}.yaml` |
| 场景 | `scene_forearm_lowmass.xml`, `scene_forearm_gravcomp.xml` |
| 评估脚本 | `workspace/core4d/scripts/eval/eval_e009_lift.py` |
| 视频 | `visualization_mjwp.mp4`, `visualization_mjwp_act.mp4`(每次 run 覆盖) |

## 复盘:为什么 E009 失败

这次实验本质上**回答了一个被忽略的问题**:G1 的几何能否物理上夹住 box025?
答案:**不能从前侧/顶部夹住,必须从 ±x 端**。

而 CORE4D 数据本来就是"两人从 ±x 端协作搬运" — 这是数据集本身的协作语义。
我之前 E001-E009 都假设 "G1 单人能完成 person1 的所有动作",但这违反了协作场景的根本前提。

**教训**:
1. **数据集语义先于优化**: 协作数据要求协作建模,单人优化不能解决双人物理
2. **几何约束先于参数**: G1 臂展 0.5m + box 长 0.61m,根本无法单人对夹
3. **看 ref 的 hand-vs-box 几何关系**比看 obj_z 数字更重要

## 用户决策点

需要选择 E010 方向:
- **(A)** 切换到更小物体(bucket/chair)继续 SPIDER 物理重定向
- **(B)** 放弃物理重定向,使用 holosoma 现有运动学 ref + HDMI 风格 RL 训练
- **(C)** 实施真正的双 agent 联合重定向(2 个 G1 + 共享 box,nu=58)
