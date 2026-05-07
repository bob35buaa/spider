# E027b: Object PD Override — 物体沿 Ref 6DOF 运动 + Robot CEM

## Context

E025-E027 reassessment 指出核心缺失：**partner 的横向搬运动作**。partner_force 只给竖直恒力，物体原地不动，机器人"碰到"的是不该在那里的静止物体。

后续 E028-E030 尝试用 position spring 驱动物体 XY 位置，但 orientation 翻转无法解决（xfrc torque 正反馈 / weld 被碰撞压倒）。

**E027b 的核心想法**：使用 `scene_act.xml` 的 6 个 position actuator（3 slide + 3 hinge）直接 PD 跟踪 ref，**在 step_env 中覆盖 CEM 产生的 ctrl**，使物体确定性地沿 ref 运动。Robot CEM 正常优化 body tracking + hand approach。

## Claims

1. **C1**: 物体沿 ref 6DOF 运动（pos_err < 0.05m, quat_err < 0.3 rad）— 非翻转
2. **C2**: 机器人保持稳定（pelvis_z ≥ 0.60m, ≥80% frames）
3. **C3**: 手-物体产生有意义接触（hand_approach_rew 有效，hand_dist < 5cm ≥30% frames）
4. **C4**: 视频呈现"机器人配合搬运中的物体"的效果（视觉验证）

## 成功标准

- C1+C2+C4 同时通过 ≥ 2/4 cases
- 视频中物体明确沿 ref 轨迹移动且不翻转

## 技术方案

### 关键改动

1. **`step_env` 中 object ctrl 覆盖** (`spider/simulators/mjwp.py`):
   - 在 `wp.copy(env.data_wp.ctrl, ...)` 之后，读取当前 object qpos（6DOF Euler）
   - 查表 ref 位置 → 计算 PD ctrl → 覆盖 ctrl 的 obj actuator 通道
   - 这样 CEM 对 obj actuator 的采样被每步覆盖，不影响

2. **Config 新字段** (`spider/config.py`):
   - `object_pd_override: bool = False`
   - `object_pd_kp_pos: float = 500.0` (强跟踪)
   - `object_pd_kp_rot: float = 100.0`

3. **Ref 数据适配**:
   - `scene_act.xml` 用 Euler angles 不用 quaternion → ref qpos[36:42] 已是 3pos+3euler
   - 需要验证 trajectory_kinematic.npz 对 scene_act 的 qpos 格式
   - 可能需要从 freejoint(pos3+quat4) 转换为 slide3+hinge3(euler)

4. **Config YAML** (`examples/config/override/core4d_e027b.yaml`):
   - scene_name: scene_act
   - object_pd_override: true
   - hand_approach_rew_scale: 5.0
   - partner_force_scale: 0 (不需要了)
   - fullanchor: true (去行走)

### nu 布局 (scene_act.xml)

- nu=35: robot[0:29] + object[29:35]
- CEM 应该只采样 robot 部分 → 需确认 config.nu 和 CEM 行为
- 方案：config.nu=35 但 CEM noise 只加到 [0:29]，或者设 config.nu=29 + 单独处理 obj ctrl

### 最简实现路径

不改 CEM nu 维度。让 CEM 照常采样 35 维 ctrl，但在 step_env 中**每步覆盖后 6 维**为 PD 目标。CEM 对 obj 维度的采样被覆盖后等效为噪声，不影响最终行为。

## 改动清单

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +object_pd_override, object_pd_kp_pos, object_pd_kp_rot |
| `spider/simulators/mjwp.py` | step_env: 在 ctrl copy 后覆盖 obj actuator ctrl |
| `examples/config/override/core4d_e027b.yaml` | 新配置 |
| `workspace/core4d/scripts/convert/generate_ref_act.py` | 将 freejoint ref → scene_act 格式 (quat→euler) |

## 训练命令

```bash
# Step 1: 生成 scene_act 格式的 ref (quat → euler 转换)
uv run workspace/core4d/scripts/convert/generate_ref_act.py

# Step 2: 运行 4 cases
for TASK in box025_person1 bucket010_person1 chair022_person1 desk005_person2; do
    MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e027b task=$TASK
done
```

## 风险

1. **Euler gimbal lock**: hinge 3 轴可能有万向节锁 → 如果 ref 有大旋转可能出问题
2. **PD 力与机器人碰撞**: 强 PD 可能推机器人 → 但这正是我们想要的（物体在动，机器人要适应）
3. **ref 格式转换**: freejoint(7DOF) → 6DOF(3slide+3euler) 需要小心
