# R013: 修复非接触阶段腕关节抖动

## Context

R012 (rew=6.56, 111% HF) 在非接触阶段 (t=0-2s) 存在明显前臂抖动。
`left_wrist_yaw` qpos 振荡 ±44°, 而 ctrl 仅偏离 ±0.3° — 147x 放大。

根因: HDMI RL 策略不控制腕关节 (action_scaling 注释掉), HF 用 nu=23,
但 R012 对腕关节施加 30% MPC 噪声, 激发了欠阻尼 PD 共振 (ζ=0.20)。

## Claims

1. 腕关节 qpos 抖动 (前 2s) < 0.5° (R012: 3.98°)
2. rew_mean >= 6.0 (R012: 6.56)
3. tracking >= 2.5 (R012: 2.93)
4. obj_track >= 3.5 (R012: 3.63)
5. 视频无可见手臂抖动

## 改动

| Fix | 文件 | 改动 |
|-----|------|------|
| 1 | run_hdmi.py:117 | wrist noise 0.3 → 0.0 (匹配 HDMI/HF 行为) |
| 2 | hdmi.py:576 | 添加 wrist jnt_damping=5.0 (临界阻尼修复) |

## 运行命令

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python examples/run_hdmi.py \
  task=move_suitcase +data_id=1 viewer=none save_video=false save_info=true \
  output_dir=workspace/hdmi_reproduce/results/R013 use_torch_compile=false
```
