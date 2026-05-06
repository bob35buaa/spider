# E023: Anchored + Rotation Alignment — 解决旋转漂移

## Context

E022 结果: Anchor 解决了行走, 但 bucket010 (69°旋转) 和 chair022 (107°旋转) 仍失败。
desk005 的 lift=37% 只是早期碰撞, 非持续接触 — Claim 口径需收紧。
box025 归结为臂展限制 (0.5m < 0.61m), 本轮排除。

**本实验核心**: 扩展 anchor 方案, 加入 pelvis yaw rotation 对齐。
每帧将 ref pelvis 的 yaw 旋转也对齐到初始方向, 物体做相同反向旋转。
使机器人始终"面对"物体, 而非随参考旋转后手指向错误方向。

## 目标 Case

| Case | 参考旋转 | 物体尺寸 | 期望效果 |
|------|---------|---------|---------|
| bucket010 | 69° | 0.40×0.74m | 消除旋转 → 手始终面向桶 |
| chair022 | 107° | 0.57×0.86m | 消除旋转 → 手始终面向椅 |
| desk005 | 9° | 0.40×0.74m | 旋转本就小, 作为对照 |

## Claims (严格标准)

| Claim | 定义 | 阈值 |
|-------|------|------|
| C1: pelvis_err 进一步降低 | anchored+rot < anchored-only | ↓20% 至少 bucket010/chair022 |
| C2: 手到物体表面距离 | sim 中手到 obj surface < 0.10m 的连续帧 | ≥ 15 帧 (0.5s), 至少 1/3 case |
| C3: 物体持续位移 | obj_z > init+0.03m 持续帧 | ≥ 30 帧 (1s), 至少 1/3 case |
| C4: 稳定性 | pelvis_z ≥ 0.50m | 所有 case ≥ 95% 帧 |

## 方法: Full Anchor (XY + Yaw)

```python
for t in range(T):
    # 1. Remove xy translation (same as E021)
    delta_xy = qpos[t, :2] - qpos[0, :2]
    anchored[t, 0:2] -= delta_xy
    anchored[t, 36:38] -= delta_xy  # object xy

    # 2. Remove yaw rotation
    # Extract yaw from pelvis quaternion
    yaw_t = extract_yaw(qpos[t, 3:7])
    yaw_0 = extract_yaw(qpos[0, 3:7])
    delta_yaw = yaw_t - yaw_0

    # Rotate pelvis quaternion to remove yaw
    anchored[t, 3:7] = remove_yaw(qpos[t, 3:7], delta_yaw)

    # Rotate object position and orientation by -delta_yaw around z-axis
    # (object pos relative to pelvis stays the same in body frame)
    obj_rel = qpos[t, 36:38] - qpos[t, 0:2]  # in world frame
    obj_rel_rotated = rotate_2d(obj_rel, -delta_yaw)
    anchored[t, 36:38] = anchored[t, 0:2] + obj_rel_rotated

    # Rotate object quaternion
    anchored[t, 39:43] = rotate_quat_yaw(qpos[t, 39:43], -delta_yaw)
```

## 执行计划

```
Step 1: 实现 anchor_pelvis_full.py (xy + yaw)
Step 2: 生成 trajectory_kinematic_fullanchor.npz (3 cases)
Step 3: 跑 body-only baseline (验证 pelvis_err 降低)
Step 4: 跑 obj_rew=3.0 (验证物体交互)
Step 5: 提取 metrics + 关键帧 + 手-物体距离时序
Step 6: 写 log
```

## Results 目录

```
workspace/core4d/results/E023_fullanchor/
├── bucket010/{bodyonly,withobj}.{npz,mp4}
├── chair022/{bodyonly,withobj}.{npz,mp4}
├── desk005/{bodyonly,withobj}.{npz,mp4}
├── keyframes/
└── metrics_summary.csv
```
