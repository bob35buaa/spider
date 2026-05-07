# E029: Position Actuator + Kinematic Override 探索 — 结果

## 状态: 方向明确但未解决 (需要 debug xfrc torque 坐标系)

## 尝试路径

### 路径 A: contact_guidance PD actuator (scene_act.xml)

| 配置 | 结果 | 问题 |
|------|------|------|
| decay=0.0, kp=100 | 物体翻倒 | decay=0 → 第 2 步 gains 归零 (0^1=0) |
| decay=1.0, residual=1.0, kp=200 | NaN | gains 太高 → 发散 |
| decay=1.0, residual=1.0, kp=30/rot=5 | pos_err=1.32 | gains 太低 → 跟踪差 |
| decay=1.0, residual=1.0, kp=80 + high joint damping | pos_err=1.12 | joint damping 反而阻碍 actuator |

**根因**: 在 `scene_act.xml` 模式下, CEM 采样全部 35 维 ctrl (包括 6 个 object actuator)。CEM 给 object actuator 发随机 ctrl → 干扰 ref target → 跟踪差。

### 路径 B: 直接 qpos override (kinematic object)

| 配置 | 结果 | 问题 |
|------|------|------|
| kp=-1 (pre-step override) | pos_err=0.60 | physics step 覆盖了 override |
| kp=-1 (post-step override) | pos_err=0.61 | metrics 可能在 MPC rollout 中计算, 非 commit step |

**根因**: override 只在 `step_env` 的真实 commit step 中生效。CEM rollout (2048/4096 个并行 world) 中物体仍然自由, CEM 的 reward 评估看到的是自由物体 → 优化方向不对。

## 关键澄清: CEM 架构中物体控制

| 模式 | Scene | nu | CEM 采样物体? | 物体怎么动 |
|------|-------|-----|-------------|----------|
| **scene.xml (freejoint)** | freejoint | **29** | **否** — 只采样 robot | 纯物理 (重力/碰撞/xfrc_applied) |
| scene_act.xml (6 joints) | 6 actuators | 35 | **是** — 采样全部 35 维 | actuator PD + CEM 随机 ctrl |

**结论**: 在 `scene.xml` 模式下 (E028 使用的), **CEM 不采样物体** — 物体只受物理力驱动。E028 翻转不是 CEM 干扰物体, 而是 **xfrc_applied torque 数值不稳定**。

所以问题归结为: **如何让 xfrc_applied 的 torque 分量 (3:6) 稳定地控制 freejoint 的 orientation?**

## xfrc_applied torque 不稳定原因分析

已尝试:
- kp_rot = kp * 0.1 (= 3.0 for kp=30): NaN
- kp_rot = 1.0: NaN  
- angular velocity damping (kd=0.5): NaN or pos_err 恶化

**可能的原因**:
1. **坐标系错误**: xfrc_applied 的 torque 是世界坐标系, quaternion error 的 axis-angle 也应该在世界系 — 但 MuJoCo 可能在某些配置下使用 body 坐标系?
2. **MuJoCo Warp batch 特殊性**: `data_wp.xfrc_applied` 在 batch 环境中的写入时机和生效方式可能与单环境不同
3. **graph capture**: `wp.capture_launch(env.graph)` 是 CUDA graph, 在 graph 内修改 xfrc_applied 可能被缓存覆盖
4. **Quaternion 符号翻转**: `torch.sign(qe_w)` 在 qe_w≈0 时抖动 → torque 方向翻转 → 能量注入

## 下一步: Debug xfrc torque

优先调查方向:
1. **验证坐标系**: 在单个 env (非 batch) 中用纯 MuJoCo (非 Warp) 测试 torque PD 是否稳定
2. **验证写入时机**: 确认 xfrc_applied 在 graph capture 之前写入是否生效
3. **简化测试**: 物体静止 (不移动, 只纠正 orientation) → 排除 position spring 干扰
4. **小角度限制**: clamp axis_angle 到 [-0.1, 0.1] 防止大 torque
5. **使用 MuJoCo 内建 joint damping**: 在 freejoint 上能否设置 orientation damping? (freejoint 不支持 damping 属性 — 可能需要 equality constraint)

## 结果路径

| 产出 | 路径 |
|------|------|
| E029 quasi-kinematic (xfrc kp=100) | `workspace/core4d/results/E029_quasikin/{case}/` |
| E029 contact_guidance actuator | `workspace/core4d/results/E029_act/bucket010/` |
| scene_act.xml (4 cases) | `example_datasets/.../scene_act.xml` |
| trajectory_kinematic_act.npz (4 cases) | `example_datasets/.../{task}/0/trajectory_kinematic_act.npz` |
| 生成脚本 | `workspace/core4d/scripts/convert/generate_scene_act.py` |
| 配置 | `examples/config/override/core4d_bucket010_e029{act,v2}.yaml` |
