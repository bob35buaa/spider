# E027d: HDMI Physics + Contact Guidance Debug — 根因定位

## 状态: 调试中 — 找到根因但尚未完全修复

## 核心思路

E027c 发现 contact_guidance 在 MJWP 上不工作（物体不动）。E027d 尝试移植 HDMI 的 physics 参数（physics_dt=0.002 + decimation），并深入调试根因。

## 运行命令

```bash
# 生成 scene_act.xml (kp=0, armature=0.01)
uv run workspace/core4d/scripts/convert/generate_scene_act.py

# 运行 (HDMI physics + high gains)
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e027d task=desk005_person2 \
    init_pos_actuator_gain=500 init_pos_actuator_bias=500 \
    init_rot_actuator_gain=50 init_rot_actuator_bias=50
```

## 关键发现

### 1. CUDA Graph Gain 更新确认有效 ✓

通过最小测试验证：`wp.copy` 到 `model_wp.actuator_gainprm` 在 `wp.capture_launch(graph)` 后**确实生效**。之前的假设（gains 不更新）是错误的。

```
kp=0 graph capture → wp.copy kp=100 → 1000 steps → qpos=0.000 ✓
```

### 2. 根因 1: Object Ctrl 未被重置为 Ref

`contact_len=0`（CORE4D 的 scene_act.xml 没有 contact sites）导致 `run_mjwp.py` line 656 的 contact delta 代码块被跳过。Object ctrl 通道不会被重置为 ref qpos → CEM 从上次优化结果出发 → 物体 ctrl 被优化偏离。

**修复**: 在 optimize 前无条件重置 object ctrl 为 ref qpos（已实现，`run_mjwp.py` ~line 657）。

### 3. 根因 2: noise_scale 为 0

`get_noise_scale()` 中 `noise_scale[:, :, object_ids] *= 0.0`（config.py line 440）。Contact guidance 设计上让 object actuator **不参与 CEM 噪声采样**。所以 CEM 不会改变 object ctrl — 物体应该完全由 PD actuator 驱动。

### 4. 当前状态: ctrl 重置 + noise=0 + gains 正确，但物体仍不动

已修复 ctrl 重置问题，但 48 步测试后物体位移仍为 0.001m。可能原因:
- rollout 内部 `save_state`/`load_state` 在每次 CEM 采样后重置了物体 qpos
- MJWP 的 `step_env` 中有其他代码覆盖了 ctrl
- scene_act.xml 的 joint damping=100 太高，PD 响应太慢

## 技术改动

| 文件 | 改动 |
|------|------|
| `spider/simulators/mjwp.py` | 移除 debug log |
| `examples/run_mjwp.py` | +object ctrl 重置为 ref (line ~657) |
| `examples/config/override/core4d_e027d.yaml` | 新: HDMI physics + contact_guidance |
| `workspace/core4d/scripts/convert/generate_scene_act.py` | armature=0.01, kp=0 |

## 下一步

1. 在 rollout 内部添加 debug 验证物体 qpos 是否在 physics step 后变化
2. 检查 `save_state`/`load_state` 是否重置了 CEM 迭代间的物体位置
3. 如果 rollout 内物体确实移动但 commit 时重置了 → 需要修改 state save/load 逻辑

## 结果路径

| 产出 | 路径 |
|------|------|
| 配置 | `examples/config/override/core4d_e027d.yaml` |
| CUDA graph 测试 | inline (run_mjwp.py E027d debug session) |
