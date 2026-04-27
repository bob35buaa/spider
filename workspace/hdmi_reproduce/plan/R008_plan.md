# R008 计划：Contact Guidance 使箱子可被搬运

## Context

R007 达到 rew_mean=5.33 (HF: 5.90, 90%)，tracking=2.20 接近 HF。但视频显示机器人在 ~2.5s（弯腰拾箱）时摔倒——因为 suitcase 是纯 freejoint 无执行器，机器人无法抓取/搬运。

MJWP 通过 **contact guidance** 解决此问题：
1. 将 object freejoint 替换为 6 个 slide/hinge 关节 + PD 执行器
2. 每次优化迭代设递减增益 `gain * decay^iter`（最后一步增益=0）
3. CEM 采样时 object ctrl 不加噪声，直接跟踪参考位置
4. 物体被"引导"到参考轨迹附近，机器人只需维持平衡

### 核心证据
- t=2s: robot/suitcase 都正常 (pelvis_z=0.79)
- t=2.5s: pelvis_z 从 0.79→0.60 (弯腰拾箱导致失衡)
- t=3s: pelvis_z=0.32 (摔倒), tracking 从 2.8→2.1
- t=4s: 完全趴在箱子上 (见 frame_t4s.png)
- 现有 `scene.xml` 的 suitcase 有 freejoint 无 actuator → 纯被动物体
- MJWP 的 `generate_xml.py:_add_object_xyzrpy_actuators` 仅支持 `right_object`/`left_object` 命名

## Claims

1. **给 suitcase 加 6 个 PD 执行器后，机器人不会在拾箱时摔倒**
   - 因为 suitcase 被引导到参考位置，机器人弯腰时有物体支撑
2. **rew_mean > 5.5 (接近 HF 5.90)**
   - obj_track 应从 3.13 提升（物体更稳定跟踪参考）
3. **tracking 维持 > 2.0**
   - 机器人不再摔倒，body tracking 全程保持

## 改动

### 方案：在 `setup_env` 中用 MuJoCo Spec API 给 suitcase 加执行器

不修改 `generate_xml.py`（它仅支持 right_object/left_object）。在 `hdmi.py:setup_env` 中用 `mujoco.MjSpec` 动态修改模型：

**Step 1**: 加载 scene XML 为 MjSpec
**Step 2**: 找到 suitcase body，删除 freejoint，添加 6 个 slide/hinge joints + position actuators
**Step 3**: 从 spec 编译新 model_cpu
**Step 4**: ctrl_ref 扩展为 nu=35 (29 robot + 6 object)，object ctrl = qpos_ref 的 suitcase xyz+rpy
**Step 5**: CEM 噪声中 object 维度设为 0 (不扰动物体)
**Step 6**: 每次 optimize iteration 设递减 object actuator gains

### 代码修改

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/simulators/hdmi.py` | `setup_env`: 用 MjSpec 给 suitcase 加 6 actuators |
| 2 | `spider/simulators/hdmi.py` | `get_reference`: ctrl_ref 扩到 35 维，加 object xyz+rpy |
| 3 | `spider/simulators/hdmi.py` | `step_env`: 不变（ctrl 已包含 object dims） |
| 4 | `examples/run_hdmi.py` | 加 contact guidance 增益衰减逻辑 (仿 run_mjwp.py:392-433) |
| 5 | `examples/config/hdmi.yaml` | 加 contact_guidance 相关参数 |
| 6 | `spider/config.py` | 确保 humanoid_object + contact_guidance 路径正确处理 noise_scale 中 object dims=0 |

### 关键实现细节

**MjSpec 修改模型** (在 `setup_env` 中):
```python
spec = mujoco.MjSpec.from_file(scene_xml_path)
# 找到 suitcase body
suitcase_body = spec.find_body("suitcase/suitcase")
# 删除 freejoint
for joint in suitcase_body.joints:
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
        joint.delete()
# 加 6 个 slide/hinge joints
for suffix, jtype, axis in [("pos_x", "slide", [1,0,0]), ...]:
    j = suitcase_body.add_joint(name=f"object_{suffix}", type=jtype, axis=axis)
# 加 6 个 position actuators
for suffix in ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]:
    spec.add_actuator(name=f"object_{suffix}", joint=f"object_{suffix}", ...)
# 编译
model_cpu = spec.compile()
```

**ctrl_ref 扩展** (在 `get_reference` 中):
```python
# 从 qpos_ref 提取 suitcase xyz + 转换 quat→rpy
obj_qadr = model.jnt_qposadr[obj_jnt_id]
obj_pos = qpos_ref[:, obj_qadr:obj_qadr+3]  # xyz
obj_quat = qpos_ref[:, obj_qadr+3:obj_qadr+7]  # wxyz
obj_rpy = quat_to_euler(obj_quat)  # 转换为 roll-pitch-yaw
obj_ctrl = torch.cat([obj_pos, obj_rpy], dim=-1)  # (T, 6)
ctrl_ref = torch.cat([ctrl_ref_robot, obj_ctrl], dim=-1)  # (T, 35)
```

**增益衰减** (在 `run_hdmi.py` 控制循环中):
```python
# 在每个 optimize iteration 前设增益
for i in range(max_iterations):
    gain = init_gain * decay_ratio ** i
    if i == max_iterations - 1: gain = 0  # 最后一步释放
    # 写入 model_wp actuator gains for object dims
```

实际上更简单的做法：用 `config.env_params_list` 传入每个 iteration 的 kp/kd，通过 `load_env_params` 更新。这是 MJWP 已有的机制。

### config 参数 (hdmi.yaml)

```yaml
contact_guidance: true
guidance_decay_ratio: 0.8
init_pos_actuator_gain: 10.0
init_pos_actuator_bias: 10.0
init_rot_actuator_gain: 0.3
init_rot_actuator_bias: 0.3
object_pos_actuator_names: [object_pos_x, object_pos_y, object_pos_z]
object_rot_actuator_names: [object_rot_x, object_rot_y, object_rot_z]
improvement_threshold: 0.0  # contact_guidance 要求
```

## 成功标准

| 指标 | R007 | R008 目标 | HF |
|------|------|----------|-----|
| rew_mean | 5.33 | **> 5.5** | 5.90 |
| tracking | 2.20 | **> 2.0** | 2.19 |
| obj_track | 3.13 | **> 3.5** | 3.71 |
| 视频 t=4s | 摔倒 | **站立搬箱** | 站立搬箱 |

## 验证步骤

1. 修改 hdmi.py — MjSpec 加 suitcase actuators
2. 修改 get_reference — ctrl_ref 扩到 35 维
3. 修改 run_hdmi.py — 加增益衰减逻辑
4. 修改 hdmi.yaml — contact_guidance 参数
5. test_hdmi_fast.py (N=64) 快速验证:
   - 确认 nu=35, ctrl_ref 形状正确
   - open-loop reward 合理
6. 完整运行 (N=1024, 250 steps)
7. **用 imageio 提取 t=2s,3s,4s 帧验证机器人不摔倒**
8. 分析 trajectory — rew/tracking/obj_track Q1-Q4
9. 记录 log + 更新 tracker

## 风险

| 风险 | 缓解 |
|------|------|
| MjSpec API 不支持 mujoco_warp | 先在 CPU 编译验证，再 put_model |
| quat→rpy 转换 gimbal lock | 用 scipy 或自己实现，注意角度范围 |
| guidance 增益太高导致不自然运动 | 参考 humanoid_object_act.yaml 的值 |
| nq 从 43 变为 42 (freejoint 7→joints 6) | 更新所有 qpos 索引和 reward 配置 |
