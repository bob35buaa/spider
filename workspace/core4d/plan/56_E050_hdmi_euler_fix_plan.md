# E050: HDMI Euler Convention Bug Fix + 重跑

## Context

E048/E049 发现 HDMI 物体追踪严重失败 (ObjPos 24-91cm, ObjRot 53-66°)。
根因: `spider/simulators/hdmi.py` 的 `get_reference()` 使用 extrinsic `as_euler("xyz")`
将 body quaternion 转为 hinge joint angles, 但 MuJoCo 串联 hinge joints 需要
intrinsic `as_euler("XYZ")`。

对于 HDMI suitcase (方向接近 identity, 仅 -88° 绕 Z), 两种 convention 差异仅 2-6°, 不影响。
但 CORE4D 物体有 ~90° X+Z 旋转, 差异达 119-278°, 导致物体参考完全错误。

## Bug 位置

| 行号 | 当前代码 | 修正为 |
|------|---------|--------|
| 659 | `spt.Rotation.from_quat(q_xyzw).as_euler("xyz")` | `.as_euler("XYZ")` |
| 1303 | `R.from_quat(q_xyzw).as_euler("xyz")` | `.as_euler("XYZ")` |
| 136-154 | `_quat_to_euler()` 函数 (torch手写版) | 需验证是否等价于 intrinsic XYZ |

## Claims

- C1: 修复后 box023 Object Rot Error < 15° (当前 66.4°)
- C2: 修复后 box025 Object Pos Error < 30cm (当前 90.9cm)
- C3: 修复后 suitcase (R013) 结果不退化 (euler diff 仅 2-6°, 修复应无害)
- C4: 修复后 HDMI body tracking 优势保持 (Joint Angle < 10°)

## 实现

### 代码修改 (~3 行)

```python
# hdmi.py line 659
rpy = spt.Rotation.from_quat(q_xyzw).as_euler("XYZ")  # was "xyz"

# hdmi.py line 1303
rpy = R.from_quat(q_xyzw).as_euler("XYZ")  # was "xyz"
```

还需检查 `_quat_to_euler()` (line 136-154) 的 torch 实现是否一致。

### 实验矩阵

| 实验 | Case | 目的 | GPU |
|------|------|------|-----|
| E050a | box023 HDMI (修复后) | 验证 C1+C4 | 本地 |
| E050b | box025 HDMI (修复后) | 验证 C2 | 本地 |
| E050c | suitcase HDMI (修复后) | 验证 C3 不退化 | 本地 |

### 运行命令

```bash
# E050a
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=false save_info=true \
    output_dir=workspace/core4d/results/E050/E050a_box023 use_torch_compile=false

# E050b
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box025 +data_id=0 viewer=none save_video=false save_info=true \
    output_dir=workspace/core4d/results/E050/E050b_box025 use_torch_compile=false

# E050c (suitcase regression check)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_suitcase +data_id=1 viewer=none save_video=false save_info=true \
    output_dir=workspace/core4d/results/E050/E050c_suitcase use_torch_compile=false
```

### 评估

正确评估 = sim vs kinematic ground truth (不用双通道自比):
```bash
python3 eval_hdmi_vs_groundtruth.py <case> <trajectory_hdmi.npz> <trajectory_kinematic.npz>
```

### 验证 (渲染)

使用 CORE4D freejoint scene 渲染 (避免 euler 转换):
```bash
MUJOCO_GL=egl uv run workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
    --scene example_datasets/processed/core4d/.../scene.xml \
    --kin trajectory_kinematic.npz --phys trajectory_hdmi.npz \
    --output comparison.mp4 --euler XYZ
```

## 关键文件

| 文件 | 改动 |
|------|------|
| `spider/simulators/hdmi.py` line 659, 1303 | "xyz" → "XYZ" |
| `spider/simulators/hdmi.py` line 136-154 | 验证 `_quat_to_euler` |
