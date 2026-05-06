# E008: 真实搬箱失败诊断 — 结果

## 状态: 失败(但提供了关键诊断信息)

## E006f 更正 — 视频驱动的真相

之前 E006f 报告 "obj_z_max=0.610m, obj_pos_err=0.324, 物体首次被搬起" 是**对 reward/metric 字段的误读**。

### 视频客观证据 (`visualization_mjwp.mp4`)

| 时间 | ref(参考) | sim(SPIDER E006f) |
|------|-----------|-------------------|
| 0s | 人在箱前直立 | 机器人在箱前直立 |
| 2s | 人弯腰双手抱住箱侧 | 机器人弓步,**只一只手按在箱顶** |
| 2.6s | 人**抱起箱子**(箱体倾斜离地) | 机器人站直,**箱子被推到地上、远离机器人** |
| 4s | 人继续抱箱前行 | 机器人独自动作,箱子在远处地面 |
| 5.2s | 人放下箱子 | 机器人独自直立,箱子在远处 |

### qpos 客观验证 (`trajectory_mjwp.npz`)

```
sim obj z:  min=0.304, max=0.307, mean=0.305    ← 箱子从未离地
ref obj z:  min=0.304, max=0.533, mean=0.375    ← 参考抬起到 0.533m
sim obj xy err: max=1.541m at t=90, mean=0.789m  ← 箱子被推飞
```

**结论**:E006f 唯一发生的事是机器人手碰到箱子顶面 → 由于无抓握约束 → 箱子被推开,然后机器人继续走自己的运动学轨迹。

## E008a 实验结果

### 改动

修正 `core4d_box025_forearm_act.yaml`:
- `residual_gain_ratio: 1.0 → 0.0`(让最后迭代真实归零,而非保留全增益)
- `guidance_decay_ratio: 0.99 → 0.85`(更明显的衰减曲线)
- `init_pos_actuator_gain: 50 → 20, bias: 10 → 20`(参考 E003 的合理值)

### 客观指标

| 指标 | E006f (无 act) | E008a (forearm_act + decay) | ref |
|------|---------------|---------------------------|-----|
| obj z max (world) | 0.307m | **0.375m** | 0.533m |
| obj z mean | 0.305m | 0.308m | 0.375m |
| obj xy 偏移 | x:1.5m+ (被推飞) | x≤6cm, y≤21cm | — |
| pelvis err mean | 0.06m | **0.635m** | — |
| Final obj_pos_err (reward) | 0.324 | 0.7306 | — |

### 视频观察 (`visualization_mjwp_act.mp4`)

| 时间 | sim |
|------|-----|
| f000 (0s) | 机器人箱前直立 |
| f060 (1.2s) | 机器人弯腰,手靠近箱子侧面,箱子仍在地面 |
| f120 (2.4s) | 机器人深度弯腰,双手在箱子附近,箱子未抬起 |
| f180 (3.6s) | 衰减 PD 归零后箱子滑回参考位置 |
| f263 (5.2s) | 机器人独立直立,箱子在地面 |

## Claims 验证

| Claim | 阈值 | 实测 | 通过? |
|-------|------|------|------|
| C1: obj_z_max > 0.40m | 0.40m | 0.375m | **❌** (差 0.025m) |
| C2: xy err < 0.20m | 0.20m | y最大 0.21m,x 0.06m | ⚠️ 边缘 |
| C3: 视频双手在箱两侧 | 视觉 | 双手都在箱前/侧 | 部分 |
| C4: pelvis err < 0.15m | 0.15m | 0.635m | **❌** (退化4倍) |

## 关键发现

### 发现 1:`residual_gain_ratio=0` 对 G1 大箱子任务过于严格

衰减 PD 在引导期能让 actuator 拽住箱子,但**最后 1 次迭代归零后**,机器人无法独立维持 0.5kg 箱子离地。原因:
- G1 的前臂/手几何只能提供"推力",不能提供"夹持力"
- CEM 在 32 次迭代内无法搜到"双手精确对夹 + 持续支撑"的姿态序列
- 一旦 actuator 撤除,箱子立刻在重力 + 接触不平衡下滑落

### 发现 2:用 actuator 后 pelvis 跟踪退化(意外副作用)

E006f pelvis err = 0.06m,E008a 退化到 0.635m(>10倍)。这是因为:
- act scene 启动时,actuator 把箱子拽到参考位置,机器人需要"够到"已经在参考位置的箱子
- 如果机器人的初始姿态来不及跟上,就会被"撕开"两端 — 被参考 pelvis 拉一边,被参考接触位置拉另一边
- 当对象奖励权重 = base 奖励权重时,机器人选择优先匹配接触,放弃 pelvis 跟踪

### 发现 3:G1 几何约束是物理硬限,不是参数调优问题

经过 E006a-f(6 个变体)+ E007a/b(物理对齐失败) + E008a(衰减引导失败),反复证明:
- 调 reward 权重:无效(E006c-f 已扫描过)
- 调 PD 增益:无效(E006c kp=100 振荡)
- 改变物理参数:破坏接触时机(E007)
- 衰减 PD 引导:引导期 OK,撤除即失败(E008a)

**根本约束**:G1 + freejoint 大箱子 + 单纯 CEM 优化 = 不可解。需要外部约束:
- (a) actuator 残余力(放弃"纯物理")
- (b) 把任务推到 Layer 2(RL + Lost-Contact 强制)
- (c) 改变物体表征(weld-joint 假设抓握成功,只验证臂部运动)

## 下一步:方向调整

经过 E001-E008 完整探索,**纯 SPIDER Layer 1 物理重定向不能解决 CORE4D 大箱子搬运任务**。两条工程路径供选择:

### 路径 1:Layer 1 提供「物理可行手臂运动」,Layer 2 解决接触维持

- 保留 E006f 配置(机器人手臂运动是物理可行的,只是箱子被推开)
- 用 E005 的 hybrid 导出脚本:机器人 qpos 来自 SPIDER + 物体来自参考
- 在 Holosoma RL 项目内加 HDMI 风格 reward(local contact + Lost-Contact termination)
- 优势:坦诚承认 Layer 1 限制,不再调虚假参数

### 路径 2:Layer 1 输出「带 actuator 残余力的接触辅助轨迹」

- `residual_gain_ratio = 0.3 ~ 0.5`(不完全归零)
- 物体跟踪好(actuator 始终提供部分支撑)
- 不是纯物理,但作为 RL reference 仍有效(reference 不要求纯物理,只要求姿态合理)
- 需要明确文档化"输出含 5kg→0.5kg 假设 + 残余 actuator 力"

**两条路径不矛盾**:可以先走路径 1 验证 RL 训练流程,再回来探索路径 2 是否能改善 RL 效果。

## 结果路径

| 产出 | 路径 |
|------|------|
| 计划 | `workspace/core4d/plan/07_E008_real_lift_diagnosis_plan.md` |
| 配置 | `examples/config/override/core4d_box025_forearm_act.yaml` |
| 脚本 | `workspace/core4d/scripts/retarget/retarget_core4d_forearm_act.sh` |
| 轨迹 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp_act.npz` |
| 视频 | 同上目录 `visualization_mjwp_act.mp4` |

## 运行命令

```bash
# E008a
uv run examples/run_mjwp.py +override=core4d_box025_forearm_act \
    task=box025_person1 data_id=0 viewer=none

# 客观验证(避开 reward 字段)
uv run python -c "
import numpy as np
d = np.load('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp_act.npz')
q = d['qpos'].reshape(-1, d['qpos'].shape[-1])
init_z = 0.310  # act scene body initial z
print(f'sim obj z (world): max={q[:,38].max()+init_z:.3f}, mean={q[:,38].mean()+init_z:.3f}')
"
```

## 复盘:为什么我之前漏掉了真相

1. **没有看视频** — E006 报告基于 reward 字段(看似改善 61%),没用视频确认
2. **没有读 qpos** — E006 标注"obj_z_max=0.610"是 metric 字段,不是 trajectory.qpos 客观状态
3. **过早乐观** — 看到 reward 减小就标注"突破",而不是验证 sim 中物体是否真的离地

**教训**(写入 progress.md):
- 任何"搬起/移动物体"的 claim,**必须**用 `qpos[:, obj_idx]` 实测,**必须**看视频确认
- reward 字段反映优化目标,不反映物理状态
- 视频比指标可信,qpos 比 reward 可信
