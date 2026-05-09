# Reward System Documentation — HDMI vs SPIDER/MJWP

## 1. HDMI Reward (spider/simulators/hdmi.py:1076-1122)

### Total Reward 公式

```
total = tracking + object_tracking
```

### 1.1 Tracking (body following)

```python
tracking = W_TRACK * (
    rew_upper_pos + rew_upper_ori    # 上半身 pos/ori in pelvis-yaw-local frame
    + rew_lower_pos + rew_lower_ori  # 下半身 pos/ori in pelvis-yaw-local frame
    + rew_root_pos + rew_root_ori    # pelvis global pos/ori
    + rew_joint                      # joint angle tracking
)
# W_TRACK = 0.5, max = 0.5 * 7 = 3.5
```

每个子项公式:
```
rew_X_pos = exp(-‖pos_sim_local - pos_ref_local‖₂ / σ_pos)    ∈ [0, 1]
rew_X_ori = exp(-‖axis_angle(q_ref⁻¹ · q_sim)‖₂ / σ_ori)     ∈ [0, 1]
rew_joint = exp(-mean(|jt_sim - jt_ref|) / σ_jt)              ∈ [0, 1]
```

### 1.2 Object Tracking

```python
rew_obj_pos = exp(-‖obj_pos_sim - obj_pos_ref‖₂ / σ_obj_pos)   # σ=0.5
rew_obj_ori = exp(-‖axis_angle(obj_q_ref⁻¹ · obj_q_sim)‖₂ / σ_obj_ori)  # σ=0.5
```

### 1.3 Contact Reward (mask-gated)

```python
# Step 1: compute target contact point (in object frame)
target_pos = obj_pos + quat_apply(obj_quat, contact_target_offset)  # per-EEF

# Step 2: compute EEF contact point
contact_eef = eef_pos + quat_apply(eef_quat, contact_eef_offset)   # per-EEF

# Step 3: distance + exp kernel
eef_dist = ‖target_pos - contact_eef‖₂        # (N, 2) — left/right wrist
pos_rew = exp(-eef_dist / eef_pos_sigma)       # σ=0.3m

# Step 4: force factor (constant scalar)
force_factor = exp(-frc_thres / frc_sigma)     # exp(-10/40) ≈ 0.778

# Step 5: mask gating
in_range = object_contact[t]                   # (2,) bool→float, from ref NPZ
gain = 5.0

# Step 6: combine
contact_rew = pos_rew * force_factor           # (N, 2)
rew_contact = mean_over_EEFs(
    contact_rew * in_range * gain + (1 - in_range)
)
```

**关键设计**:
- `in_range=1` (ref 标记为接触帧): reward ∈ [0, gain * force_factor] = [0, 3.89]
- `in_range=0` (非接触帧): reward = **1.0** (constant, CEM 无梯度)
- `contact_target_offset` = 物体表面的具体接触点 (预定义, 如把手位置)
- `contact_eef_offset` = EEF 上的接触探测点 (如手掌中心)

### 1.4 HDMI Reward Budget 总结

| Component | Range | Max | 占比 |
|-----------|-------|-----|------|
| tracking | [0, 3.5] | 3.5 | 47% |
| rew_obj_pos | [0, 1] | 1.0 | 13% |
| rew_obj_ori | [0, 1] | 1.0 | 13% |
| rew_contact (contact frame) | [0, 3.89] | 3.89 | 52% |
| rew_contact (non-contact) | 1.0 | 1.0 | 13% |
| **Total (contact frame)** | — | **9.39** | — |
| **Total (non-contact frame)** | — | **6.5** | — |

---

## 2. SPIDER/MJWP Reward (spider/simulators/mjwp.py:458-700)

### Total Reward 公式

```python
reward = qpos_rew + qvel_rew + contact_rew + task_body_rew + task_obj_rew
         + interact_rew + hand_approach_rew + contact_mask_rew
         + stability_penalty
```

### 2.1 Local-Frame Tracking (E035, 当 use_local_frame_reward=True)

完全对齐 HDMI 的 tracking 设计:

```python
local_frame_rew = W_TRACK * (
    upper_pos_rew + upper_ori_rew     # 上半身 local
    + lower_pos_rew + lower_ori_rew   # 下半身 local
    + root_pos_rew + root_ori_rew     # pelvis global
    + joint_rew                       # joint angle
)
# W_TRACK = 0.5, max = 0.5 * 7 = 3.5 (与 HDMI 相同)
qpos_rew = local_frame_rew  # 替代原有的 qpos_rew
```

每个子项公式 (与 HDMI 完全相同):
```
upper_pos_rew = mean_i(exp(-‖Δpos_i_local‖ / σ_pos))    σ=0.5
upper_ori_rew = mean_i(exp(-‖Δori_i_local‖ / σ_ori))    σ=1.0
root_pos_rew = exp(-‖Δroot_pos‖ / σ_root)               σ=0.5
root_ori_rew = exp(-‖Δroot_ori‖ / σ_root)               σ=0.5
joint_rew = exp(-mean(|jt_err|) / σ_jt)                 σ=0.25
```

### 2.2 Object Tracking (E036, task_obj_rew)

```python
# 负值惩罚式 (与 HDMI 的 exp-kernel 不同!)
# nq_obj == 6 (contact guidance mode: 3 pos + 3 euler)
pos_err = ‖obj_pos_sim - obj_pos_ref‖²         # squared L2
task_obj_rew -= task_obj_pos_rew_scale * pos_err   # scale=1.0

rot_err = ‖obj_euler_sim - obj_euler_ref‖²
task_obj_rew -= task_obj_rot_rew_scale * rot_err   # scale=1.0
```

**与 HDMI 的差异**: HDMI 用 exp kernel (bounded [0,1])，我们用负 squared error (unbounded negative)。

### 2.3 Contact Mask Reward (E037)

```python
# Step 1: surface distance (box-SDF)
delta = |hand_pos - obj_pos|                     # per-axis abs diff
surface_dist = max(delta - half_extents, 0)      # clamp to surface
dist_per_hand = ‖surface_dist‖₂                  # (N, K_hands)
min_dist = min_over_hands(dist_per_hand)         # (N,) — best hand

# Step 2: mask + reward
mask = approach_mask_val                         # 0 or 1, from ref precomputation
gain = contact_mask_rew_scale                    # e.g. 3.5
sigma = contact_mask_rew_sigma                   # e.g. 0.15
baseline = contact_mask_rew_baseline             # e.g. 0.0

proximity = gain * exp(-min_dist / sigma)
contact_mask_rew = mask * proximity + (1 - mask) * baseline
```

**Approach mask 预计算** (run_mjwp.py:479-497):
```python
for t in range(T):
    # FK on ref qpos
    mj_kinematics(model, data_ref)
    obj_pos = data_ref.xpos[obj_body_id]
    for hid in hand_body_ids:
        hand_pos = data_ref.xpos[hid]
        surf_dist = box_sdf(hand_pos, obj_pos, half_extents)
        if surf_dist < hand_approach_contact_threshold:  # 0.3m
            approach_mask[t] = 1.0
            break
```

### 2.4 SPIDER Reward Budget (E037c S3 config)

| Component | Range | Max | 当前配置 |
|-----------|-------|-----|---------|
| local_frame_rew (tracking) | [0, 3.5] | 3.5 | W=0.5 |
| task_obj_rew | (-∞, 0] | 0 | scale=1.0 |
| contact_mask_rew (mask=1) | [0, gain] | 3.5 | gain=3.5, σ=0.15 |
| contact_mask_rew (mask=0) | baseline | 0.0 | baseline=0.0 |
| hand_approach_rew | 0 | 0 | **disabled** |
| stability_penalty | 0 | 0 | **disabled** |
| **Total (contact frame)** | — | **7.0** | — |
| **Total (non-contact frame)** | — | **3.5** | — |

---

## 3. 关键差异对比

| 设计点 | HDMI | SPIDER/MJWP |
|--------|------|-------------|
| **Contact 目标点** | 物体表面的具体偏移 (`contact_target_offset`) | 物体 bounding box 表面 (SDF) |
| **距离度量** | EEF 到目标点的精确 L2 | 手到 box surface 的近似 SDF |
| **Mask 来源** | NPZ 中的 `object_contact` 标注 | ref FK 预计算 (手距 <0.3m) |
| **Per-EEF vs Per-Hand** | 分别计算左/右手, 取 mean | 取 min (best hand) |
| **非接触帧处理** | reward = 1.0 (constant) | reward = baseline (0 或 1) |
| **Object tracking** | exp kernel [0,1] | squared error ≤0 (penalty) |
| **力反馈** | force_factor=0.778 (constant) | 无 |

---

## 4. "Contact 无法超越 Ref" 的论断 — 严格分析

### 4.1 论断内容

"当 MPKPE=1.4cm 时，contact 质量由 ref 动作决定，contact reward 无法超越 ref 的上限。"

### 4.2 论断的逻辑链

```
前提 1: MPKPE = 1.4cm → sim 的每个 body 位置与 ref 偏差 < 2cm
前提 2: 手是 body 之一 → sim 手位置与 ref 手位置偏差 < 2cm
前提 3: 物体由 PD controller 精确跟踪 ref → obj_pos_err < 1cm
推论:   hand-to-object 距离 ≈ ref_hand-to-ref_object 距离 ± 3cm
结论:   如果 ref 中手距物体 20cm, sim 中手距物体 ≈ 17-23cm, 不可能 < 10cm
```

### 4.3 这个论断能否站住脚?

**部分成立, 但有重要前提条件和反例:**

#### ✅ 成立的情况 (当前 E036/E037 系列)

当 `contact_mask_rew` 的 gain (3.5) ≤ tracking max (3.5) 时:
- CEM 不会为了 contact reward 而牺牲 body tracking
- 因此 hand 位置被 tracking reward "锁死"在 ref 手的 ±2cm 内
- Contact 质量确实被 ref 限制

**数学证明**: 假设 CEM 要把 sim 手移动 Δ=10cm 靠近物体 (偏离 ref):
- tracking 损失: 手偏离 10cm → exp(-0.1/0.5)=0.82, 损失 ≈ 0.5*(1-0.82)=0.09
  - 但这影响整个 upper body tracking (多个 body), 实际损失更大
- contact 收益: 距离从 20cm→10cm → gain*(exp(-0.1/0.15) - exp(-0.2/0.15)) = 3.5*(0.51 - 0.26) = 0.88

看起来收益 > 损失? 但实际上 CEM 在 **horizon 上累加 reward**, 且 tracking 影响全部 7 个子项 × 所有帧, 而 contact 只影响 mask=1 的帧。在 horizon=24 步 × 7 tracking terms 的规模下, 单帧的 0.88 contact gain 远不足以抵消全局 tracking 的退化。

#### ❌ 不成立的情况

1. **如果 gain >> tracking** (e.g. gain=10):
   - CEM 会牺牲 body tracking 来获得 contact
   - 手会偏离 ref, 主动伸向物体
   - 但这正是 E035 的失败模式 (hand_approach=5.0 > tracking=3.5 → MPKPE=48cm)

2. **如果使用 HDMI 的精确 contact_target_offset**:
   - HDMI 的 contact 目标是物体表面的具体点 (把手位置)
   - 不是 box SDF 的最近表面点
   - 如果这个目标点恰好与 ref 手的自然位置不同, contact reward 可以引导手到不同位置
   - 但这需要额外的数据标注 (contact offset per task)

3. **如果手的 body tracking 有方向性偏差**:
   - 如果 sim 手系统性地偏离物体方向 (而不是随机误差)
   - 那么即使 MPKPE=1.4cm (均值), 某些帧手可能偏离物体 >5cm
   - Contact reward 可以纠正这个方向性偏差

### 4.4 实验验证

| 配置 | MPKPE | Contact<10cm | 是否超越 ref? |
|------|-------|-------------|--------------|
| E036 (无 contact) | 1.4cm | 56% (box025) | baseline |
| E037 S3 (gain=3.5) | 1.4cm | 57% (box025) | +1% 微改善 |
| E035 (hand_approach=5) | 48cm | 95% (desk005) | ✅ 超越, 但 tracking 崩溃 |

**E035 证明 contact CAN 超越 ref** — 代价是 tracking 崩溃 (MPKPE 48cm)。
**E037 证明 在保持 tracking 的前提下, contact 确实无法显著超越 ref** (+1%)。

### 4.5 修正后的结论

> **在 gain ≤ tracking max 的约束下, contact reward 无法显著改善 contact quality。
> 要超越 ref 的 contact 限制, 必须允许 body tracking 退化 — 这是一个根本性 tradeoff。**
>
> 当前 SPIDER pipeline 的定位是: MPKPE < 5cm 优先 > contact quality。
> 因此 contact 改善应通过改善 ref 本身 (更好的 IK / motion capture), 而非 reward 调参。

### 4.6 可能打破限制的方法

1. **修改 ref 的 IK** — 让 ref 中手的位置更贴近物体 (preprocess 阶段)
2. **Wrist-specific tracking boost** — 提高 wrist body 在 local_frame_rew 中的权重, 间接增强 hand→ref hand 精度
3. **Hybrid approach** — 低 gain contact reward (不破坏 tracking) + 适当宽松的 wrist tracking sigma, 让 CEM 在 wrist 方向有更多自由度
4. **接受当前水平** — MPKPE=1.4cm 已远超 HDMI(7.7cm) 和 DynaRetarget(3.6cm), contact 由数据质量决定
