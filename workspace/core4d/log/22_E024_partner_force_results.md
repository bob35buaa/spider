# E024: Partner-Force Assisted Retargeting — 结果

## 状态: 失败 (partner force 不解决接触问题, CEM 不会主动伸手碰物体)

## 核心发现

1. **Partner force (50-90% gravity comp) 不让 G1 主动接触物体** — CEM 优化 body tracking, 手方向是副产品
2. **90% force 下物体漂起 (lift=84%) 是失重碰撞, 非搬运** — 序列末尾箱子因极低等效重力 (0.5kg) 被偶发碰撞推飘
3. **视频确认: 所有 partner force 配置中 G1 手始终在体侧, 未伸向物体**
4. **pelvis tracking 依然良好 (0.15-0.21m)** — body motion 本身 OK, 问题在手-物体交互

## 实验矩阵

| Run | partner_force | pos_rew | pelvis_err | obj_lift | lift% | sustained(>5cm) | stable% | 判定 |
|-----|--------------|---------|-----------|---------|-------|-----------------|---------|------|
| E024-a1 | 50% | 3.0 | 0.154 | 0.081m | 36.1% | 14fr (0.5s) | 100% | 碰撞推移 |
| E024-a2 | 70% | 3.0 | 0.211 | 0.024m | 10.9% | 0fr | 100% | 无效 |
| E024-a3 | 90% | 3.0 | 0.182 | 0.187m | 83.9% | 19fr (0.6s) | 100% | **失重飘移 (非搬运)** |

## Claims 验证 (严格)

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: obj_z > init+0.05m 持续 ≥30帧 | max=19帧 (90% force, 序列末尾) | **FAIL** ❌ |
| C2: 手到 obj surface < 0.05m 持续 ≥20帧 | 视频确认手从未碰到物体 | **FAIL** ❌ |
| C3: obj 运动方向与 ref 一致 | 90% 的 lift 在 ref 下降阶段发生 (反相关) | **FAIL** ❌ |
| C4: pelvis_z ≥ 0.50m ≥95% | 全部 100% | **PASS** ✅ |
| C5: 视频确认协作搬运 | 手在体侧未碰箱子, 箱子是失重飘起 | **FAIL** ❌ |

## 可视化观察

### 90% partner force, t=80% (frame ~200)
- **ref**: G1 弯腰, 手在箱顶面, 箱子在下方
- **sim**: G1 直立站在箱旁, 手在体侧, 未伸向箱子
- **箱子**: 仍在原位 (z=0.30), 未被搬起

### 90% partner force, t=95% (frame ~250)
- **ref**: G1 抬手 (搬运结束后的松手动作)
- **sim**: G1 抬手 (body tracking OK!), 但箱子突然漂起到 0.48m
- **原因**: 90% gravity comp → 箱子等效 0.5kg → 之前某个碰撞的余震让它飘起
- **判定**: 失重飘移, 非有意搬运

## 根因分析: 为什么 SPIDER CEM 不产生接触

```
SPIDER CEM 优化目标:
  reward = body_pos_reward + body_rot_reward + joint_reward + obj_pos_reward + contact_reward

问题: 所有 reward 都是在关节空间采样后被动计算的

CEM 优化循环:
  1. 在关节空间随机采样 2048 个动作序列
  2. rollout 每个序列得到仿真结果
  3. 计算 reward, 选取 top-K
  4. 更新分布 (mean, std)

关键限制:
  - 采样发生在关节空间 → 不直接控制手的 cartesian 位置
  - 短 horizon (1.6s, ~48 steps) → 无法规划"先伸手→再抓→再抬"的序列
  - contact_reward 只奖励"手碰到物体后", 不引导"手去接近物体"
  - 2048 个随机采样中, 产生有效接触的概率极低 (高维, 精确空间小)
```

**类比**: 让一个蒙眼的人随机摆动胳膊 1.6 秒, 看能不能碰到面前的箱子并抬起来 — 概率极低。

## Phase 5 (E020-E024) 总结论

| 实验 | 目标 | 结果 | 判定 |
|------|------|------|------|
| E020 | 多 case 诊断 | body 100% stable, pelvis_err 0.65-0.79m | 行走是瓶颈 |
| E021 | Anchor (去行走) | pelvis_err ↓47-72%, 但物体不动 | body tracking 改善 |
| E022 | Anchor + obj_rew | desk005 早期碰撞 37%, 其余无效 | 偶发碰撞 |
| E023 | Full anchor (+ yaw) | pelvis ↓48%, 但物体超臂展 | 协作数据结构限制 |
| E024 | Partner force | 90% → 失重飘移, 手未碰物体 | CEM 不产生接触 |

**共同结论**: SPIDER CEM 能做 body tracking, 不能做 contact-rich manipulation。
问题不是 reward 设计、不是 anchor、不是 partner force — 而是 **CEM 关节空间采样无法发现精确接触策略**。

## 后续方向选择

### 方向 1: 修改 SPIDER reward 引导手接近物体 (在 SPIDER 框架内)

**思路**: 当前 contact_rew 只在"已接触"时给奖励。如果加一个 **hand-approach reward** (手到物体距离的负值), CEM 会倾向于让手靠近物体。

```python
hand_approach_rew = exp(-sigma * hand_to_obj_surface_dist)
```

**优点**: 最小改动, 在 SPIDER 框架内
**风险**: 即使手到了物体旁, CEM 可能仍无法产生"压/推"的力方向 — 需要接触 + 正确方向的力

### 方向 2: Task-space 手位置跟踪 (强制手到参考位置)

**思路**: 参考轨迹中手的 world position 是已知的 (FK 计算)。加一个 task-space reward 直接跟踪手的 world position, 类似 DynaRetarget 的 hand_pos reward:

```python
hand_pos_rew = exp(-sigma * |hand_xpos_sim - hand_xpos_ref|)
```

这等价于 E018 的 task_body_rew, 但权重给手更高。如果手 xpos 跟踪到参考位置, 手就自然在物体旁边 (因为参考中手就在物体上)。

**优点**: 有 DynaRetarget 论文支持 (hand_pos=5.0), 已有 task_body_rew 基础设施
**风险**: 手到了正确位置不等于能产生接触力 — 手在物体旁 vs 手压在物体上

### 方向 3: Partner spring + hand approach 组合

**思路**: 
- Partner force = spring (方案 B), 物体跟着参考走 (被弹簧拉向 ref pos)
- Hand approach reward: 引导手靠近移动中的物体
- 当手碰到物体时, spring 可以减弱 (让 G1 接管)

**优点**: 物体运动由 spring 保证 (不依赖 G1), G1 只需要"搭上手"
**缺点**: 与 connect 约束类似, 物体运动主要由 spring 驱动, G1 的接触贡献有限

### 方向 4: 接受 SPIDER 做 body-only, 接触交给 RL

**思路**: SPIDER 的价值 = 生成物理合规的 body motion reference (pelvis 0.15-0.26m)。物体搬运是 RL 的任务 — SPIDER 不适合做这个。

**优点**: 诚实, 不纠结于 SPIDER 的结构性限制
**缺点**: 放弃了 "SPIDER 做接触重定向" 的研究方向

### 方向 5: SBTO (DynaRetarget 全序列优化)

E019 已初步实验 (不稳定), 但 DynaRetarget 论文证明 SBTO 在同数据集同机器人上将 SPIDER 37.9% → 74.6% success。区别是 SBTO 优化整条轨迹 (非 receding horizon), 可能解决"无法规划长接触序列"的问题。

**优点**: 有论文结果支持, 是 SPIDER 的自然升级
**缺点**: 实现复杂, E019 已暴露不稳定问题

### 建议优先级

1. **方向 2 (hand_pos tracking)** — 最快验证, 已有基础设施, 直接测试"手到位后能否产生接触"
2. **方向 1 (hand approach reward)** — 如果方向 2 手到位但仍无接触力, 加 approach 梯度
3. **方向 3 (spring + approach)** — 如果单纯 reward 不够, 用 spring 保证物体运动
4. **方向 5 (SBTO)** — 如果以上都失败, 尝试根本性算法升级

## 结果路径

| 产出 | 路径 |
|------|------|
| box025 pf50/70/90 | `results/E024_partner_force/box025/pf{50,70,90}.{npz,mp4}` |
| 关键帧 | `results/E024_partner_force/keyframes/box025_pf90_t{0,1,2}.png` |
| 脚本 | `scripts/retarget/retarget_e022_anchored_objrew.sh` (复用) |
| 计划 | `plan/24_E024_partner_force_plan.md` |
| 代码改动 | `spider/config.py` (+partner_force fields), `spider/simulators/mjwp.py` (+_apply_partner_force) |
| 配置 | `examples/config/override/core4d_box025_partner_force.yaml` |
