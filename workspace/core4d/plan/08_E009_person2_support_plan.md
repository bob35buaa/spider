# E009: 引入 Person2 力支撑 — 多方案探索

## Context

### 上游结论(E001-E008)

经过 8 个实验,**纯 SPIDER Layer 1 物理重定向不能解决 G1 单人搬大箱子任务**:
- E006f 视频确认:机器人推开箱子(obj_z_world max=0.307m,xy 偏移 1.5m+)
- E008a 衰减 PD 失败:引导期 OK,撤除即失败(obj_z max=0.375m,pelvis 退化 4 倍)
- 几何硬限:G1 手部无夹持能力 + freejoint 大箱子 = CEM 不可解

### 用户洞察:CORE4D 是双人协作场景

**CORE4D 数据本来就有 person2** — 我们之前一直在 person1 的"单人搬箱"假设下做物理优化,这是在解决一个**根本不存在的任务**。

实际任务是"两人协作搬箱":
- person1 在箱子的一端(x≈-0.45)
- person2 在箱子的另一端(x≈+0.72) — **已确认 person2 数据存在**
- 真实物理:两人**同时**把箱子两端抬起,两端的支撑力平衡

### Person2 数据可用性(已验证)

Source: `/home/ubuntu/Workspace/holosoma/workspace/v2/results/retarget_replace_batch_trimmed/20231011-048-person2-Box025_with_obj_original.npz`
- shape: (124, 43),fps=30,与 person1 同长度同帧率
- 格式与 person1 完全一致(qpos[0:7] pelvis,qpos[7:36] joints,qpos[36:43] object freejoint)
- person1 和 person2 共享同一物体轨迹(差 < 0.027m,可视为同一参考物)
- 已在 spider 中预处理:`example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person2/scene{,_act}.xml` 存在

### Person2 双手轨迹(已计算并保存)

```
frame 0  (站立): L_hand=(0.70, -0.25, 0.73), R_hand=(0.72, 0.07, 0.72)
frame 47 (峰值): L_hand=(0.67, -0.55, 0.74), R_hand=(0.64, -0.13, 0.73)
frame 60 (走中): L_hand=(?, ?, ?)
obj at frame 47: (0.13, -0.35, 0.54)
```

person2 的双手在箱子**正后方** — 这正是"第二人帮抬另一端"的几何位置。

存档:`/tmp/p2_hand_traj.npz`(临时),后续移到 `workspace/core4d/data/p2_hand_box025.npz`

## 方案设计:三条路径,从最低到最高代码改动量

### 方案 A:Partial Residual Force(最简单)

**思路**:`residual_gain_ratio = 0.3 ~ 0.5` — 让 actuator 在最后迭代仍保留 30~50% 引导力,等价于"person2 持续提供恒定支撑力"。

**实现**:
- 直接改 `core4d_box025_forearm_act.yaml`:`residual_gain_ratio: 0.0 → 0.3`
- 不改任何代码

**物理解释**:
- E008a 的失败是"actuator 撤除后箱子掉" — 因为 person1 单人不够
- 如果"持续保留 30% actuator 力",等于"person2 总是在那里抬一半"
- 输出的轨迹箱子是被支撑的,机器人(person1)的手臂运动是物理优化过的
- 这不是"纯物理",但 reference motion 不要求纯物理 — 只要求姿态合理

**优势**:零代码改动,1 次实验
**劣势**:不显式建模 person2,actuator 力在所有方向都有(不像真实接触)

### 方案 B:Mass Reduction(超低质量,p2 承担 90%重量)

**思路**:把箱子质量 0.5kg → 0.05kg,等价于"person2 承担 90% 重量,person1 只需 10%"。

**实现**:改 `scene_forearm.xml` 中 object inertial mass="0.5" → "0.05"。

**物理解释**:
- person1 的手只需要提供 0.05 × 9.8 = 0.49N 的支撑力即可
- 单点接触 + 摩擦力(friction=2)即可保持
- 等价于 "person2 用稳定支撑承担 90%,person1 帮着扶" 的真实场景

**优势**:简单,纯物理可解
**劣势**:质量太轻可能让 CEM 找到"用一根手指顶住"的退化解,不像协作搬运

### 方案 C:Mocap Person2 Hands(显式第二人物理协作)

**思路**:在场景中加入两个 **mocap body**(由数据驱动,不参与优化)代表 person2 的双手。它们的位置每帧由 p2_hand_traj.npz 强制设置,与 box 产生真实接触和摩擦。

**实现**:
1. 改 scene_forearm.xml 加 2 个 mocap body(geom: 0.05m sphere,代表 person2 hand)
2. 加 contact pair: p2_lh ↔ object_collision, p2_rh ↔ object_collision
3. 在 mjwp.py 的 step 循环里,每帧根据 sim_t 把 mocap 位置插值更新
4. 用 sim_dt=0.0167 时间轴,每个 step 设 mocap_pos[mocap_lh] = interpolate(p2_lh_traj, t)

**物理解释**:
- person2 的双手是"被动 mocap" — 不被 CEM 优化,但参与物理接触
- person1 (G1) 由 SPIDER 优化 + 与箱子产生真实接触
- 箱子接收两个来源的接触力:person1 的手 + person2 的手
- 这是**最忠实**于 CORE4D 协作语义的方案

**优势**:物理最真实,reference 轨迹直接体现协作语义
**劣势**:需要改 mjwp.py 的 step 循环(注入 mocap 更新),改动量最大

### 实施顺序

按改动量 / 风险递增:**A → B → C**

A 需要 30 秒(改 1 个数字),B 需要 1 分钟(改 XML),C 需要 30 分钟(改 mjwp.py)

## Claims

| ID | Claim | 验证(npz qpos + 视频) | 阈值 |
|----|-------|----------------------|------|
| C-A | 方案 A: residual=0.3,obj_z_world max > 0.45 | qpos[:,38]+0.31 | ≥ 0.45 |
| C-B | 方案 B: mass=0.05,obj_z_world max > 0.45 | qpos[:,38] | ≥ 0.45 |
| C-C | 方案 C: mocap p2,obj_z_world max > 0.45 | qpos[:,38] | ≥ 0.45 |
| C-视频 | 视频中机器人确实把手放在箱子侧面,箱子被抬起 | 看 f060/f100 关键帧 | 视觉确认 |
| C-pelvis | pelvis 跟踪不退化 | qpos[:,0:3] vs ref | mean err < 0.20m |

## 训练命令

```bash
# A: residual=0.3
bash workspace/core4d/scripts/retarget/retarget_core4d_e009a.sh

# B: low mass
bash workspace/core4d/scripts/retarget/retarget_core4d_e009b.sh

# C: mocap p2
bash workspace/core4d/scripts/retarget/retarget_core4d_e009c.sh

# 客观验证(每个 run 完后)
uv run python workspace/core4d/scripts/eval/eval_e009_lift.py <variant>
```

## 风险

| 风险 | 缓解 |
|------|------|
| 方案 C 改 mjwp.py 引入 bug,影响其他场景 | 用 try/except + 仅当配置开关打开时才生效 |
| 方案 A 的 actuator 力在所有方向都有,可能导致箱子 xy 跟错位置 | 接受;如果 z 起来但 xy 错,降低 residual_gain_ratio |
| 方案 B 太轻让 CEM 钻空子 | 看视频判断;如果是单指顶住就视为 fail |

## 关键警惕点

1. **不要再相信 reward 字段** — 必须读 qpos[:, 38] (act 场景需要 + 0.31 偏移)
2. **必须看视频** — 即使指标过了也要确认双手在箱子侧
3. **Pelvis 退化是失败信号** — E008a 教训
