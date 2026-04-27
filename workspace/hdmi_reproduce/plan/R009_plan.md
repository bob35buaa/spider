# R009 计划：修复输出路径、参考可视化、腕关节反向

## Context

R008 达到 rew=5.63 (HF 95%)，但有三个需要修复的问题：
1. 输出路径被 `process_config` 覆盖，结果保存到 HF 数据目录（覆盖原始数据）
2. 参考视频中 suitcase 脱离手部 — `_quat_to_euler` 转换误差
3. 仿真视频 4-5s 右手出现反关节 — 腕关节 CEM 噪声未抑制

### 参数对比分析 (原始 run_hdmi.sh vs 我们的配置)

原始脚本: `joint_noise_scale=0.2` + 默认 `temperature=0.1` + `improvement_threshold=0.02`
我们 R008: `noise_scale=0.05` + `temperature=0.3` + `improvement_threshold=0.0` + `contact_guidance=true`

差异是合理的：原始脚本为 OLD code (CPU MuJoCo + normalized actions) 设计，ctrl 经过 `(ref-default)/scaling` 归一化，noise=0.2 等效 0.01-0.11 rad。我们的 Warp 后端直接写 raw radians，noise=0.05 rad 等效。contact_guidance 是 Warp 后端必需的（suitcase 无原生抓取机制）。

### 根因分析

**Issue 1: 输出路径**
- `spider/config.py:525` 的 `process_config` 无条件覆盖 `config.output_dir`
- 覆盖为 `example_datasets/processed/hdmi/.../1/`
- 用户传入的 `output_dir=workspace/...` 被丢弃

**Issue 2: 参考 suitcase 脱离**
- R008 用 contact guidance 模型 (nq=42, suitcase 用 xyz+rpy 6 joints)
- `get_reference` 中 `_quat_to_euler` 将 motion data 的 quat(wxyz) 转为 euler(rpy)
- 自定义实现可能有 Euler 约定/精度问题，导致 suitcase 朝向错误
- 解决：用 scipy 的 `Rotation.from_quat().as_euler("xyz")` 替换

**Issue 3: 腕关节反向**
- `get_noise_scale` 的 `humanoid_object` 分支对所有 29 个 DOF 统一加 `noise_scale*=0.05`
- 6 个腕关节 (左/右 × roll/pitch/yaw, actuator indices 19-21, 25-28) 没有被抑制
- HF 的 23-DOF action space 根本不包含腕关节 — 说明腕关节不应被 CEM 扰动
- ctrl_ref 中腕关节值来自 motion dataset（有值），但 CEM 的噪声让它们偏离

## Claims

1. **输出路径修复后 trajectory 保存到用户指定的 output_dir**
2. **参考视频中 suitcase 正确跟随手部** (用 scipy euler 转换)
3. **腕关节噪声归零后，4-5s 不再出现反关节动作**
4. **rew_mean 维持 > 5.5** (修复不影响 reward)

## 改动

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 1 | `examples/run_hdmi.py` | 在 `process_config` 后恢复 `output_dir` | 保存用户指定的路径 |
| 2 | `spider/simulators/hdmi.py` | `get_reference`: 用 scipy 替换 `_quat_to_euler` | 修复 suitcase 朝向 |
| 3 | `examples/run_hdmi.py` | 在 contact guidance noise 归零中，同时归零腕关节 | 6 个 wrist actuators noise=0 |

### Fix 1: 输出路径 (run_hdmi.py)

```python
# 在 process_config 之前保存用户 output_dir
user_output_dir = config.output_dir
config = process_config(config)
# 恢复用户指定的 output_dir (process_config 会覆盖为 HF 数据路径)
if user_output_dir and user_output_dir != "outputs/hdmi":
    config.output_dir = user_output_dir
    os.makedirs(config.output_dir, exist_ok=True)
```

### Fix 2: 参考 suitcase 朝向 (hdmi.py get_reference)

将 `_quat_to_euler(obj_quat)` 替换为 scipy:
```python
from scipy.spatial.transform import Rotation
# obj_quat: (T, 4) wxyz → scipy 需要 xyzw
q_np = obj_quat.numpy()
q_xyzw = np.stack([q_np[:,1], q_np[:,2], q_np[:,3], q_np[:,0]], axis=-1)
rpy = Rotation.from_quat(q_xyzw).as_euler("xyz")
obj_rpy = torch.from_numpy(rpy).float()
```

### Fix 3: 腕关节噪声归零 (run_hdmi.py)

在 contact guidance noise 归零部分，增加腕关节归零:
```python
# 腕关节 actuator 名称
wrist_names = ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
               "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
for ai in range(env.model_cpu.nu):
    aname = mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
    if aname and any(w in aname for w in wrist_names):
        config.noise_scale[:, :, ai] *= 0.0
```

## 成功标准

| 指标 | R008 | R009 目标 |
|------|------|----------|
| output_dir | 覆盖到 HF 路径 | 保存到用户指定路径 ✅ |
| ref 视频 suitcase | 脱离手部 | 正确跟随 ✅ |
| sim 视频 t=4-5s | 右手反关节 | 自然动作 ✅ |
| rew_mean | 5.63 | > 5.5 |
| tracking | 2.52 | > 2.0 |

## 验证步骤

1. 实现三处修复
2. 运行 `max_sim_steps=100 output_dir=workspace/hdmi_reproduce/results/R009`
3. 验证文件保存到 R009 目录 (不是 HF 路径)
4. 用 imageio 提取 t=2s,3s,4s 帧:
   - ref 侧 suitcase 是否在手中
   - sim 侧右手 t=4s 是否自然
5. 完整 250 步运行 + 指标分析
