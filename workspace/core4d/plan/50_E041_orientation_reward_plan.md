# E041: Hand Orientation Reward — 解决手背接触问题

## Context

E039b~E040 的 contact reward 都存在 "手背接触物体" 的不自然行为。根因: position-only reward 没有方向约束, CEM 通过旋转手腕让 contact_point 靠近 target, 导致手背 (而非手掌) 朝向物体。

分析 G1 wrist frame 在 ref 接触帧中的朝向:
- Left wrist: **-y 轴**指向物体 (dot = -0.61 ~ -0.87)
- Right wrist: **+y 轴**指向物体 (dot = +0.67 ~ +0.96)

## Claims

1. **C1**: Contact<10cm ≥ 60% (box025, bucket010) — 不低于 E040
2. **C2**: MPKPE < 3cm
3. **C3**: Stability > 90%
4. **C4**: 手掌朝向物体 (视频验证, 无手背接触)

## 核心设计

### Orientation Reward 公式

```python
# Palm normal in wrist local frame
palm_normal_left = [0, -1, 0]   # left hand: -y points toward object
palm_normal_right = [0, +1, 0]  # right hand: +y points toward object

# Transform to world frame
palm_normal_world = quat_apply(eef_quat, palm_normal_local)

# Direction from EEF contact point toward target on object
direction_to_target = normalize(target_world - contact_point)

# Orientation reward: palm should face toward target
dot = sum(palm_normal_world * direction_to_target, dim=-1)  # [-1, 1]
ori_rew = clamp(dot, min=0.0)  # [0, 1], only reward when palm faces target
```

### 与 position reward 结合

```python
# Position: 距离近
pos_rew = exp(-dist / sigma)

# Orientation: 手掌朝向正确
ori_rew = clamp(dot(palm_normal_world, dir_to_target), min=0)

# Combined: 两者乘积 — 只有距离近 AND 方向对才给高 reward
combined_rew = pos_rew * ori_rew
# 或加权组合:
combined_rew = w_pos * pos_rew + w_ori * ori_rew
```

选择 **乘积** 而非加权和: 只有同时满足距离+方向才给 reward, 避免 CEM 只优化一个维度。

### 最终 reward

```python
contact_hdmi_rew = (combined_stack * mask * gain + (1 - mask)).mean(dim=1)
```

## 改动文件

### 1. `spider/config.py` — 2个新字段

```python
contact_hdmi_ori_weight: float = 0.0  # 0=disabled; weight for orientation term
contact_hdmi_palm_normal_left: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
contact_hdmi_palm_normal_right: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
```

### 2. `spider/simulators/mjwp.py` — 修改 contact_hdmi_rew 块

在 per-EEF 循环内, 在计算 `pos_rew` 后添加:
```python
if config.contact_hdmi_ori_weight > 0.0:
    palm_normal = palm_normals[ei]  # (3,) local
    palm_world = _lf_quat_apply(eef_quat, palm_normal.unsqueeze(0).expand(N, -1))  # (N, 3)
    dir_to_target = target_world - contact_point  # (N, 3)
    dir_to_target = dir_to_target / (dir_to_target.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (palm_world * dir_to_target).sum(dim=-1)  # (N,)
    ori_rew = torch.clamp(dot, min=0.0)
    # Combine: pos * ori (multiplicative gating)
    pos_rew = pos_rew * ori_rew
```

### 3. `examples/config/override/core4d_e041.yaml`

基于 E040 config + orientation:
```yaml
contact_hdmi_ori_weight: 1.0
contact_hdmi_palm_normal_left: [0.0, -1.0, 0.0]
contact_hdmi_palm_normal_right: [0.0, 1.0, 0.0]
```

## 运行命令

```bash
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e041 task=box025_person1 video_output_path=workspace/core4d/results/E041/E041_box025.mp4
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e041 task=bucket010_person1 video_output_path=workspace/core4d/results/E041/E041_bucket010.mp4
```

## 风险

1. **乘积可能过于严格**: pos_rew * ori_rew, 如果方向略偏 (dot=0.5), pos_rew 直接减半 → CEM 可能放弃接触
2. **Palm normal 对称问题**: left=-y, right=+y 是从 box025 ref 推断的, 其他物体/姿态可能不同
3. **Distance=0 时方向无意义**: 当 contact_point ≈ target_world 时, dir_to_target 接近零向量 → 需要 epsilon 防止 NaN
