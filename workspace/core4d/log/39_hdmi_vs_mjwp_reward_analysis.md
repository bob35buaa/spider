# HDMI vs MJWP Reward 结构对比分析

## HDMI Reward 结构 (spider/simulators/hdmi.py:1009-1122)

```python
total = tracking + object_tracking

tracking = 0.5 * (
    rew_upper_pos +   # 上半身 body pos (local frame, σ=0.5)
    rew_upper_ori +   # 上半身 body ori (local frame, σ=1.0)
    rew_lower_pos +   # 下半身 body pos (local frame, σ=0.5)
    rew_lower_ori +   # 下半身 body ori (local frame, σ=1.0)
    rew_root_pos +    # pelvis global pos (σ=0.5)
    rew_root_ori +    # pelvis global ori (σ=0.5)
    rew_joint         # 关节角 tracking (σ=0.25)
)  # = 0.5 * 7 terms = 3.5 max

object_tracking = (
    rew_obj_pos +     # 物体全局位置 (σ=0.5)
    rew_obj_ori +     # 物体全局朝向 (σ=0.5)
    rew_contact       # EEF→物体表面距离 + 接触mask (gain=5.0, σ=0.3)
)  # = 3 terms, contact max = 5.0
```

### HDMI 关键特点:

1. **Local frame body tracking**: upper/lower body 的 pos/ori 都在 **root (pelvis) 的局部坐标系** 中计算
   - `_pos_tracking_local()`: 将 body pos 转换到 pelvis 局部坐标系后计算误差
   - 这意味着即使 pelvis 位移/旋转偏离 ref，只要身体各部分**相对于 pelvis 的姿态正确**，tracking reward 就高
   - **这天然保护了步态稳定性!** CEM 可以选择"pelvis 位置稍偏但身体姿态正确"的方案

2. **上下半身分离**: upper_body (肩/肘/腕) 和 lower_body (髋/膝/踝) 分开计算 reward
   - 每个有独立的 sigma
   - CEM 可以分别优化上下半身

3. **EEF contact 用接触 mask 门控**: `rew_contact = pos_rew * force_factor * in_range + (1 - in_range)`
   - 只在 ref 标记为"接触"的帧才计算接触 reward
   - 非接触帧 reward = 1.0 (不惩罚)
   - **不会在不该接触的时候强制手靠近物体**

4. **contact_target_offset + contact_eef_offset**: 接触目标是物体表面的特定点（预计算的 offset），不是简单的"手到物体中心距离"

5. **Reward 全部是 exp(-error/σ) 形式**: 范围 [0, 1]，加法组合
   - tracking 最大 3.5, object 最大 ~7 (含 contact gain=5)
   - 但 tracking 的 7 个 term vs object 的 3 个 term → body tracking 占多数

## MJWP Reward 结构 (spider/simulators/mjwp.py:335-479)

```python
reward = qpos_rew + qvel_rew + contact_rew + task_body_rew + task_obj_rew + interact_rew + hand_approach_rew

qpos_rew = -||diff_qpos * weight||_2   # 负的 L2 距离 (无上界!)
qvel_rew = -scale * ||qvel_diff||_2    # 负的 L2

hand_approach_rew = scale * exp(-σ * min_hand_dist)  # 指数衰减
```

### MJWP 关键特点:

1. **Global frame qpos tracking**: 直接在关节空间计算 `qpos_diff`
   - pelvis pos (3) + pelvis rot (3) + joints (29) 都在同一个 L2 norm 中
   - **没有 local frame 转换** → pelvis 位移误差和关节角误差混在一起
   - CEM 无法分别优化"身体姿态"和"位置跟踪"

2. **Linear weighting 乘法**: `base_pos_rew_scale * qpos_diff[:3]` 然后整体取 L2
   - 权重是乘在 diff 上再取 norm，不是独立的 reward term
   - 单个大误差会主导整个 reward（L2 的特性）

3. **hand_approach 是独立 additive term**: `+ scale * exp(-σ * dist)`
   - 和 qpos_rew (负的 L2) 直接相加
   - 当 hand_approach 给出 +3.0 的正值，它可以"覆盖" qpos_rew 的 -5.0 惩罚
   - **CEM 可能选择"姿态很差但手很近"的方案，因为总 reward 更高**

4. **没有接触 mask**: hand_approach 在**每一帧都激活**
   - 即使 ref 中该帧不应该接触物体，手也被拉向物体
   - 导致不该前倾的时候也前倾

## 核心差异总结

| 维度 | HDMI | MJWP |
|------|------|------|
| **Body tracking 坐标系** | Local (相对 pelvis) | Global (关节空间 L2) |
| **Reward 形式** | exp(-err/σ) ∈ [0,1] 加法 | -L2 (无下界) + exp 混合 |
| **接触控制** | 接触 mask 门控 (只在该接触时给 reward) | 全程激活 (每帧都拉手向物体) |
| **接触目标** | 物体表面特定点 (offset) | 最近表面距离 (bbox 近似) |
| **上下半身** | 分离 (独立 sigma) | 混合 (同一个 L2 norm) |

## 为什么 HDMI 不摔

1. **Local frame tracking 保护了步态**: CEM 可以选择"pelvis 稍偏但身体平衡"的方案，因为 local body tracking 不会因 pelvis 偏移而惩罚
2. **接触 mask 避免了不必要的前倾**: 只在 ref 标记为接触的帧才引导手靠近物体
3. **Bounded reward 避免了极端方案**: exp 形式的 reward 范围 [0,1]，不会出现"手很近给 +3 覆盖姿态差 -5"的情况

## 我们 MJWP 摔倒的根因

1. **hand_approach 全程激活** → 在机器人还在走路/不该接触的时候就拉手向物体 → 重心偏移
2. **Global qpos L2** → pelvis 位移误差被 base_pos_rew_scale 放大，但和关节角在同一个 L2 中竞争，CEM 无法分离优化
3. **hand_approach 的正 reward 覆盖了 qpos 的负 reward** → CEM 选择"姿态差但手近"的方案

## 建议的修复方向

### 方案 A: 添加接触 mask (最小改动)
- 用 ref 的 hand-surface 距离判断哪些帧应该激活 hand_approach
- 非接触帧 hand_approach_rew = 0 (不拉手)
- 预计可以避免"不该前倾时前倾"的问题

### 方案 B: 改为 local frame body tracking (中等改动)
- 将 body tracking 改为相对 pelvis 的 local frame
- 分离上下半身 reward
- 这是 HDMI 成功的核心设计

### 方案 C: 完整移植 HDMI reward 到 MJWP (大改动)
- 将 HDMI 的 reward 结构完整移植
- 包括 local tracking + contact mask + bounded exp reward
