# E008: 真实搬箱失败诊断 — 视频驱动的根因分析

## Context

### 背景:E006f 是「指标繁荣下的实际失败」

**视频(`visualization_mjwp.mp4`)关键帧观察**:
| 时间 | ref(参考) | sim(SPIDER E006f) |
|------|-----------|-------------------|
| 0s | 人在箱前直立 | 机器人在箱前直立 |
| 2s | 人弯腰双手抱住箱侧 | 机器人弓步,**只一只手按在箱顶** |
| 2.6s | 人**抱起箱子**(箱体倾斜离地) | 机器人站直,**箱子已被推到地上、远离机器人** |
| 4s | 人继续抱箱前行 | 机器人独自动作,箱子在远处地面 |
| 5.2s | 人放下箱子 | 机器人独自直立,箱子在远处 |

**直接读取 trajectory_mjwp.npz 的客观验证**:
```
sim obj z:  min=0.304, max=0.307, mean=0.305    ← 箱子从未离地
ref obj z:  min=0.304, max=0.533, mean=0.375    ← 参考抬起到 0.533m (峰值在第47/124帧)
sim obj xy err: max=1.541m at t=90, mean=0.789m  ← 箱子被推飞,xy 偏离 0.79m
```

### 错误的乐观结论

E006f log 报告的 "obj_z_max=0.610m, obj_pos_err=0.324" — **来源是 reward/metric 字段的误读**。
Reward 字段记录的是参考轨迹的统计或归一化误差,**不等于 sim 中物体的真实状态**。

**真实结论**:E006f 没有真正搬起箱子。**唯一发生的事是**:机器人手碰到箱子顶面 → 由于箱顶接触+重力+无抓握约束 → 箱子被沿水平方向推开,然后机器人继续走自己的运动学轨迹直到结束。

### Phase 0:为什么 E006f 没真正搬起 — 三个根因假设

#### 假设 H1: 接触几何对箱子施加的是"推力"而非"夹持力"

E006 的 3-box 前臂碰撞几何让机器人"伸长的手臂"能碰到箱体侧面。但:
- 箱体是 0.61×0.61×0.89m 的长方体
- 机器人手宽<10cm,夹持力臂只能"抓"住箱顶薄薄一层
- **关键问题**:CEM 优化的是"减小 obj_pos_err",但没有约束"两手必须从两侧对夹"
- 结果:CEM 找到了"用一只手按住箱顶 + 另一只手在空气中挥"这种**最小代价的姿态**,而不是"弯腰双手对夹"

视频证实:f100 帧(2s)时,sim 机器人左手在箱顶,右手悬空,身体几乎直立 — 这是 ref 中"弯腰夹箱"的退化版本。

#### 假设 H2: 物体奖励压不过 base_pos 奖励

E006f 配置:
```
base_pos_rew_scale: 3.0   # pelvis xyz 跟踪
base_rot_rew_scale: 1.0   # pelvis rot 跟踪
pos_rew_scale: 3.0         # object pos 跟踪
rot_rew_scale: 1.0         # object rot 跟踪
contact_rew_scale: 1.0     # 接触
```

base_pos = pos = 3.0,**但 base_pos_rew 来自 pelvis 单点**,而 pos_rew 是物体单点 + 受机器人动力学链传递。CEM 在"被动跟随 pelvis 参考"和"主动驱动手抱起箱子"之间,会优先选前者(物理上更省力)。

视频证实:f100 中机器人 pelvis 高度跟参考类似,但**没有像参考那样弯腰** — 因为参考的 pelvis_z 下沉是抱箱时的副产品,在 SPIDER 优化里无法通过 base_pos_rew 强制还原。

#### 假设 H3: 「自由箱子 + 推接触」是物理不稳定的优化目标

scene_forearm.xml 用的是 **freejoint 物体**(7DOF 自由)。在 freejoint 下,只有"两手对箱产生封闭接触力链"才能稳定支起 0.5kg 箱子。

CEM 是非梯度优化,在 32 次迭代中很难找到"双手同时锁住两侧 + 提升"这种**精确接触序列**。容易陷入局部最优:"单手碰一下箱顶 → obj_z 不变,但身体姿态匹配 → reward 还过得去"。

对照 HDMI/OmniRetarget:
- HDMI 场景里 box 也是 freejoint,但他们靠**RL 训练时的 Lost-Contact termination + interaction reward** 强制接触持续(那是 Layer 2 的事)
- SPIDER 等价 Layer 1 工具是 **scene_act**(物体加 6DOF actuator + 衰减 PD 引导),它**可以**让物体被"虚拟手"先举起,然后过渡到接触主导

**但** E006 系列我们直接用的是 `scene_forearm`(无 actuator),**没有用 scene_forearm_act 的衰减 PD 引导**。这是关键遗漏。

### Phase 1: 解决路径(从最低风险到最高改动量)

#### 路径 A:启用 scene_forearm_act + 衰减 PD 引导(最直接)

**理由**:E003 在普通 sphere-hand 场景下用衰减 PD 让物体 z_max 维持(虽然最后归零归零失败),但与 E006 的"前臂接触几何"组合起来从未实验过。

把 `scene_forearm_act` + 合理的 `init_pos_actuator_gain` (~10-20) + `guidance_decay_ratio=0.85` + `residual_gain_ratio=0` 配合 E006f 的 reward 设置,期望:
- 前 80% 迭代:虚拟物体 actuator 把箱子拽到参考位置(包括抬起)
- 同时:CEM 学习用前臂接触提供托举力
- 最后 1 次:actuator 增益归零 → 物体只受真实接触力 + 重力

**已有 forearm_act yaml 但 residual_gain_ratio=1.0 没归零** — 这违反衰减 PD 的核心思想,需要改回 0.0。

#### 路径 B:加 contact-mass reward (HDMI 风格)

接触奖励现在是基于 *是否在 contact_pair 内有 contact*。HDMI 用 `exp(-||p_eef - p_target||/σ) * min(exp((F-F_thres)/σ), 1)` — **接触位置 + 接触力**。
力项让 CEM 不会满足于"轻轻碰一下",而是寻找"产生足够支撑力"的姿态。

但这要改 mjwp.py reward 计算,改动量大,留作 E009/E010 的 Path Y 子任务。

#### 路径 C:加 dual-hand 对称约束

reward 里加"左右手到物体侧面距离的对称性"。改动量小,但对动力学不利场景帮助有限。

### Phase 2:E008 实验设计(直接选路径 A)

## Claims

| ID | Claim | 验证方式 | 成功阈值 |
|----|-------|---------|---------|
| C1 | scene_forearm_act + 修正 residual_gain_ratio=0 + 衰减 PD,sim obj_z_max > 0.40m | 读 `trajectory_mjwp_act.npz` 的 `qpos[:, 38]` | obj_z 峰值 ≥ 0.40m(参考为 0.533) |
| C2 | sim 箱子 xy 误差不爆炸(不被推飞) | `qpos[:, 36:38]` 与 ref 比对 | xy err mean < 0.20m |
| C3 | 视频里能看到双手在箱子两侧 | f100 关键帧检查 | 视觉确认 |
| C4 | pelvis 跟踪不退化 | qpos[:, 0:3] vs ref | pelvis xy err mean < 0.15m |

C1+C2 同时满足 → 真正搬起;只 C1 不满足 → 路径 A 失败,需要 Path B(reward 重写)。

## 改动

### 改动 1:修正 forearm_act yaml — `residual_gain_ratio: 0.0`

理由:`residual_gain_ratio=1.0` 等于不衰减,actuator 全程接管物体,等于"不做接触优化"。改回 `0.0` 让最后一次迭代必须靠真实接触维持物体。

### 改动 2:E008a 用现有 forearm_act 配置基线

```bash
uv run examples/run_mjwp.py +override=core4d_box025_forearm_act task=box025_person1 data_id=0 viewer=none
```

读取 `trajectory_mjwp_act.npz`,验证 obj z/xy。

### 改动 3:E008b 调 actuator 增益(若 E008a obj 仍归零)

E008a 失败时可能是衰减 PD 太弱。提高 `init_pos_actuator_gain` 50→80,decay 0.85→0.92(更慢衰减)。

### 改动 4:E008c 提高 contact_rew_scale + 双手对称项

若动力学失败,加大 contact_rew_scale 1→3,增加配置 `dual_hand_symmetry_scale`(若需新代码,推到 E009)。

## 实验顺序

1. **E008a**: 修正 yaml → 跑 forearm_act → 读 npz 客观指标 → 看视频
2. **E008b** (条件):若 E008a obj_z<0.40,提高 actuator 增益重跑
3. **E008c** (条件):若 E008b 仍失败,提交结果 + 转向路径 B

## 训练命令

```bash
# E008a
bash workspace/core4d/scripts/retarget/retarget_core4d_forearm_act.sh

# 客观验证
uv run python -c "
import numpy as np
d = np.load('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp_act.npz')
q = d['qpos'].reshape(-1, d['qpos'].shape[-1])
dk = np.load('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_kinematic.npz')
print(f'sim obj z: max={q[:,38].max():.3f} mean={q[:,38].mean():.3f}')
print(f'ref obj z: max={dk[chr(34)+chr(113)+chr(112)+chr(111)+chr(115)+chr(34)][:,38].max():.3f}')
"
```

## 关键警惕点

1. **scene_forearm_act 的 qpos 维度是 42 不是 43** — 6DOF actuator 而非 freejoint。reading code 必须知道 `qpos[:, 36:39]` 是 obj xyz,不是 freejoint base。
2. **不要再相信 reward 字段** — 这次必须用 `qpos` 客观状态作为成功判定。
3. **视频是 ground truth** — 每个变体都必须看视频确认。
