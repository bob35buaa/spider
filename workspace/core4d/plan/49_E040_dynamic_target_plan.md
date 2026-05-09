# E040: Dynamic Per-Frame Contact Target — 解决 "手粘连物体" 问题

## Context

E039b/E039c 发现 contact reward 虽然大幅提升 Contact<10cm (box025: 85%, bucket010: 76%), 但引入了严重的 **"手粘连物体"** 问题:

- **固定 target offset** 假设手始终在物体同一个点 (适合 HDMI suitcase 把手)
- CORE4D 任务中人围绕物体活动, 手在物体表面的接触位置随时变化
- 固定 target → 手被拉回固定点 → 手腕反关节 + 身体扭转

**E040 方案**: 用 ref 中每帧手相对物体的实际位置作为该帧的 target, 实现 "物体坐标系下的 EEF tracking"。

## Claims

1. **C1**: Contact<10cm ≥ 70% (box025, bucket010)
2. **C2**: MPKPE < 3cm (body tracking 不退化)
3. **C3**: Stability > 90%
4. **C4**: 无 "手粘连/反关节" 现象 (视频验证)

## 核心思路

### 从固定 target 到动态 target

```python
# E039 (固定): 所有帧用同一个 target_offset
target_world = obj_pos + quat_apply(obj_quat, FIXED_OFFSET)

# E040 (动态): 每帧用 ref 中手实际在物体表面的位置
# 预计算: 将 ref 中手的位置转换到物体局部坐标系
target_offset[t, ei] = obj_mat[t].T @ (hand_pos_ref[t] - obj_pos_ref[t])
# 运行时: 按当前帧索引使用
target_world = obj_pos_sim + quat_apply(obj_quat_sim, target_offset[t])
```

本质: **从 "手应该在物体的固定位置" 变成 "手应该在 ref 中手所在的物体位置"**

### Mask gate 保留

仍使用 E039b 的 rotated-SDF mask (threshold=0.15m) 来区分接触/非接触帧。非接触帧不激励手靠近物体。

## 改动文件

### 1. `spider/config.py` — 新增 1 个字段

```python
# E040: dynamic per-frame contact target
contact_hdmi_dynamic_target: bool = False  # True=use per-frame ref-derived target
```

当 `contact_hdmi_dynamic_target=True` 时, `contact_hdmi_target_left/right` 不再使用, 改用预计算的 per-frame target。

### 2. `examples/run_mjwp.py` — 预计算 per-frame target (~20行)

在 E039b rotated-SDF mask 块之后, 添加 per-frame target 预计算:

```python
# E040: precompute per-frame contact target from ref FK
contact_target_per_frame = None
if config.contact_hdmi_dynamic_target and config.hand_approach_body_ids:
    obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id != -1:
        T = qpos_ref.shape[0]
        n_eef = len(config.hand_approach_body_ids)
        target_np = np.zeros((T, n_eef, 3), dtype=np.float32)
        for t in range(T):
            mj_data_ref.qpos[:] = qpos_ref[t].detach().cpu().numpy()
            mujoco.mj_forward(mj_model, mj_data_ref)
            obj_pos = mj_data_ref.xpos[obj_body_id]
            obj_mat = mj_data_ref.xmat[obj_body_id].reshape(3, 3)
            for ei, hid in enumerate(config.hand_approach_body_ids):
                hand_pos = mj_data_ref.xpos[hid]
                # Hand position in object local frame
                target_np[t, ei] = obj_mat.T @ (hand_pos - obj_pos)
        contact_target_per_frame = torch.tensor(target_np, device=config.device)
        loguru.logger.info("E040 dynamic target: shape={}", tuple(contact_target_per_frame.shape))
```

然后将其加入 `ref_data` tuple (第 9 个元素)。

### 3. `spider/simulators/mjwp.py` — 修改 contact_hdmi_rew 块

- `get_reward` 中解包 ref_data: 长度为 9 时取出 `contact_target_per_frame`
- 当 dynamic target 存在时, 用 `targets = [contact_target[ei] for ei ...]` (per-frame) 替代固定 `config.contact_hdmi_target_left/right`

```python
# E040: use per-frame dynamic target when available
if contact_target_dynamic is not None:
    targets = [contact_target_dynamic[ei] for ei in range(n_eef)]
else:
    targets = [torch.tensor(config.contact_hdmi_target_left, ...), ...]
```

### 4. `examples/config/override/core4d_e040.yaml` — 新配置

基于 E039 config, 添加:
```yaml
contact_hdmi_dynamic_target: true
# gain 可能需要降低 (动态 target 更精确, gain=5.0 可能太强)
contact_hdmi_gain: 5.0
contact_hdmi_threshold: 0.15
```

## ref_data tuple 演进

| 位置 | 内容 | 版本 |
|------|------|------|
| 0 | qpos_ref | 原始 |
| 1 | qvel_ref | 原始 |
| 2 | ctrl_ref | 原始 |
| 3 | contact_ref | 原始 |
| 4 | contact_pos_ref | 原始 |
| 5 | body_xpos_ref | E018 |
| 6 | approach_mask_t | E034 |
| 7 | body_xquat_ref_t | E035 |
| **8** | **contact_target_per_frame** | **E040** |

## 预计算依赖

E040 的 per-frame target 预计算需要:
- `mj_forward` 已在 E035 local-frame 预计算中使用 (每帧都做 FK)
- 可以**复用** E035 的 FK 循环, 在同一个 for-t 循环中同时计算 body FK + contact target
- 但为简单起见, 先独立预计算, 后续可合并优化

## 风险

1. **动态 target 过于精确**: 如果 ref 中手偶然远离物体 (如过渡帧), 该帧的 target 可能指向不合理位置。通过 mask gate (threshold=0.15m) 过滤这些帧。
2. **gain=5.0 可能仍太强**: 动态 target + gain=5.0 可能使 contact reward 占据 CEM 优化空间。如有 stability 问题需降低。
3. **本质退化为 hand tracking**: 如果 body tracking + contact target 都强制手到 ref 位置, 实际上等价于提高 hand body 的 tracking 权重。需要确认: (a) 这是否足够恢复 contact; (b) 是否比直接加 hand tracking 更自然。

## 运行命令

```bash
# box025
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e040 task=box025_person1 video_output_path=workspace/core4d/results/E040/E040_box025.mp4

# bucket010
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e040 task=bucket010_person1 video_output_path=workspace/core4d/results/E040/E040_bucket010.mp4

# desk005
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e040 task=desk005_person2 video_output_path=workspace/core4d/results/E040/E040_desk005.mp4
```

## 评估

```bash
uv run workspace/core4d/scripts/eval/eval_comprehensive.py <task> workspace/core4d/results/E040/E040_<case>.npz
```

## 成功标准

| Metric | 阈值 |
|--------|------|
| Contact<10cm (box025) | ≥70% |
| Contact<10cm (bucket010) | ≥70% |
| MPKPE | <3cm |
| Stability | >90% |
| 视觉自然度 | 无反关节/粘连 |
