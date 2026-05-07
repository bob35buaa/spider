# E030: xfrc_applied Orientation Torque Debug — 计划

## Context

E028/E029 证明 position spring 可以驱动物体沿 ref 移动, 但物体翻转 (orientation 不受控)。
xfrc_applied torque [3:6] 之前尝试均导致 NaN。根因待定:
- 坐标系问题? torque 是世界系还是 body 系?
- quaternion 符号翻转? qe_w≈0 时方向反转→能量注入
- 增益过高? 大 axis-angle → 大 torque → 发散
- MuJoCo Warp batch 特殊性? CUDA graph capture 缓存?

## Claims (可验证)

1. **C1**: 在纯 MuJoCo (CPU, 单环境) 中, xfrc_applied torque PD 可以稳定控制 freejoint orientation (静止物体→保持 upright)
2. **C2**: 如果 C1 成立, 移植到 MuJoCo Warp batch 后同样稳定
3. **C3**: Position + Orientation spring 组合在 4 个 case 中物体不翻转 (视频验证)

## 成功标准

- C1: 纯 MuJoCo 单环境测试, 物体倾斜后 <2s 恢复 upright, 无 NaN
- C2: Warp batch (N=2048) 中同样稳定, 无 NaN
- C3: 4 case 视频确认物体未翻转, pos_err < 0.15m

## 实验步骤

### Step 1: 纯 MuJoCo CPU 测试 (隔离 Warp/CUDA)

写独立测试脚本 `workspace/core4d/scripts/debug/test_torque_cpu.py`:
- 加载 box025 scene.xml (freejoint object)
- 初始倾斜物体 30°
- 每步 apply xfrc torque PD: `torque = -kp * axis_angle_error - kd * angular_vel`
- axis_angle_error = `quat_sub(quat_ref, quat_current)` (使用 spider.math)
- 验证: 物体恢复 upright, 无 NaN
- 扫参: kp=1,5,10,50; kd=auto(2√(I*kp))

关键点:
- MuJoCo `xfrc_applied` torque 是 **世界坐标系** (官方文档: "Cartesian force and torque applied to the body's center of mass, in world frame")
- `quat_sub` 返回世界系 axis-angle, 直接可用
- 需要 clamp axis-angle magnitude 防止大 torque

### Step 2: 移植到 MJWP Warp batch

如果 Step 1 成功, 将相同逻辑加入 `_apply_partner_force`:
- 从 qpos 取 object quaternion (MuJoCo freejoint: qpos[3:7] = wxyz)
- 从 qvel 取 object angular velocity (qvel[3:6])
- 使用 spider.math.quat_sub 计算 axis-angle error
- clamp axis-angle magnitude to [-0.5, 0.5] (防止大 torque)
- 写入 xfrc_applied[:, obj_body_id, 3:6]
- 新 config 字段: `partner_force_spring_kp_rot`, `partner_force_spring_kd_rot`

### Step 3: 4 Case 全覆盖运行 + 视频验证

- box025, bucket010, desk005, chair022
- kp=30 (position) + kp_rot=best_from_step1
- 保存视频, 提取关键帧验证物体未翻转
- 计算 pos_err, rot_err, pelvis_stability

## 改动范围

| 文件 | 改动 |
|------|------|
| `workspace/core4d/scripts/debug/test_torque_cpu.py` | 新文件: 纯 MuJoCo 测试 |
| `spider/simulators/mjwp.py` | `_apply_partner_force`: 添加 orientation torque |
| `spider/config.py` | 添加 `partner_force_spring_kp_rot`, `kd_rot` |
| `examples/config/override/core4d_*_e030.yaml` | 4 case 配置 |

## 风险

- xfrc torque 在世界系, 但 freejoint quaternion 可能有 sign ambiguity → clamp + sign correction
- 惯性矩阵 I 不是标量 (各轴不同) → 先用标量近似, 后续再用完整 I
- MuJoCo Warp batch 可能有 xfrc_applied 写入时机问题 → Step 2 验证
