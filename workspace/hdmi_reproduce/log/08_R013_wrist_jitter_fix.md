# R013/R013b: 腕关节抖动修复

## Context

R012 (rew=6.56, 111% HF) 在非接触阶段 (t=0-2s) 存在明显前臂抖动。
视频对比中 sim 侧的手腕/前臂持续震颤，而 ref 侧稳定。

## 根因分析

三个独立问题叠加导致腕关节振荡:

1. **MPC 噪声激发共振**: R012 对腕关节施加 30% 噪声 (noise_scale *= 0.3)，但 HDMI RL 策略完全不控制腕关节 (hdmi-base.yaml 中 wrist action_scaling 被注释)，HF 也不含腕关节 (nu=23)。噪声激发了欠阻尼 PD 共振。

2. **Isaac config 零增益覆盖 bug (新发现)**: hdmi.py 的增益覆盖循环用 Isaac 的 stiffness/damping 字典覆盖 scene XML 的 actuator gains，但 Isaac config 对腕关节没有条目 (wrist 不在 stiffness dict 中) → 默认返回 kp=0, kd=0 → 将 scene XML 的有效 Kp=14-17 替换为 0。**腕关节变为完全无控制状态**。

3. **PD 本身欠阻尼**: 即便有 scene XML 的 Kp=14-17, Kd=0.9-1.1，阻尼比 ζ=0.20 远低于临界阻尼 (ζ=1.0)。外力 (重力、躯干运动惯性) 也会激发衰减振荡。

## 改动

| Fix | 文件 | 改动 |
|-----|------|------|
| 1 | run_hdmi.py:118 | wrist noise 0.3 → 0.0 (匹配 HDMI/HF) |
| 2 | hdmi.py:~565 | 增益覆盖循环: if kp==0 and kd==0 → skip (保留 scene XML 默认) |
| 3 | hdmi.py:~580 | 添加 dof_damping=5.0 for wrist joints (ζ≈1.1 临界阻尼) |

## R013b 结果 (threshold=0.001)

### 指标对比

| 指标 | R012 | R013b | HF ref | 变化 |
|------|------|-------|--------|------|
| rew_mean | 6.56 | 6.50 | 5.90 | -0.9% |
| tracking | 2.93 | 2.83 | 2.19 | -3.4% |
| obj_track | 3.63 | 3.67 | 3.71 | +1.1% |
| opt_steps | 32.0 | 6.56 | 3.12 | -80% |
| 运行时间 | ~110min | ~23min | — | -79% |

### 腕关节抖动 (前 2s, frame-diff std)

| 关节 | R012 | R013b | 下降 |
|------|------|-------|------|
| left_wrist_roll | 1.07° | 0.02° | 98.1% |
| left_wrist_pitch | 4.01° | 0.05° | 98.7% |
| left_wrist_yaw | 1.59° | 0.03° | 98.0% |
| right_wrist_roll | 0.67° | 0.01° | 98.2% |
| right_wrist_pitch | 2.61° | 0.07° | 97.3% |
| right_wrist_yaw | 1.07° | 0.02° | 98.2% |

### Per-quarter

| Quarter | R012 rew | R013b rew | R012 tracking | R013b tracking |
|---------|----------|-----------|---------------|----------------|
| Q1 (0-2.5s) | 6.01 | 5.55 | 3.16 | 2.73 |
| Q2 (2.5-5s) | 7.23 | 7.72 | 2.60 | 2.86 |
| Q3 (5-7.5s) | 7.07 | 7.11 | 2.84 | 2.86 |
| Q4 (7.5-10s) | 5.94 | 5.65 | 3.12 | 2.88 |

## Claims 验证

1. ✅ 腕关节 qpos 抖动 (前 2s) 0.02-0.07° < 0.5° (R012: 1.0-4.0°)
2. ✅ rew_mean = 6.50 >= 6.0
3. ✅ tracking = 2.83 >= 2.5
4. ✅ obj_track = 3.67 >= 3.5
5. ✅ 视频中非接触阶段前臂平滑，无可见抖动

## 视频

- R013b: `workspace/hdmi_reproduce/results/R013b/R013b_comparison.mp4`
- R012 对比: `workspace/hdmi_reproduce/results/R012/R012_comparison.mp4`

## R013 结果 (threshold=0.0, full 32 iter)

### 指标对比

| 指标 | R012 | **R013** | R013b | HF ref |
|------|------|----------|-------|--------|
| rew_mean | 6.56 | **6.83** | 6.50 | 5.90 |
| tracking | 2.93 | **3.07** | 2.83 | 2.19 |
| obj_track | 3.63 | **3.75** | 3.67 | 3.71 |
| wrist jitter | 1.0-4.0° | **0.003-0.012°** | 0.02-0.07° | — |
| 运行时间 | ~110min | 107min | ~23min | — |

R013 **超越 R012**，成为新 SOTA: rew=6.83 (HF 的 116%), tracking=3.07 (HF 的 140%)。

### Pelvis 稳定性验证

R013b (threshold=0.001) 在 t=1.2-1.8s pelvis_z 下沉 7cm (0.784→0.709)，有摔倒倾向。
R013 (threshold=0.0) pelvis_z 全程稳定 0.784±0.001，与 R012 完全一致。

| t(s) | R012 | R013 | R013b |
|------|------|------|-------|
| 1.4 | 0.784 | 0.784 | 0.733 |
| 1.8 | 0.784 | 0.784 | 0.709 |

原因: threshold=0.001 在站立阶段平均只跑 1-3 轮 CEM，扰动得不到纠正。

### 性能提升原因

Fix 2 (跳过零增益覆盖) 修复了腕关节增益被错误覆盖为零的 bug。
恢复 scene XML 的 Kp=14-17 + 新增 dof_damping=5.0 → 腕关节从「无控制」变为「临界阻尼」,
整体物理响应更合理，CEM 搜索空间也更高效。

## 运行命令

### R013 (threshold=0.0, GPU 0)

```bash
# 运行实验
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python -u examples/run_hdmi.py \
  task=move_suitcase +data_id=1 viewer=none save_video=false save_info=true \
  output_dir=workspace/hdmi_reproduce/results/R013 use_torch_compile=false \
  2>&1 | tee workspace/hdmi_reproduce/results/R013/run.log
```

### R013b (threshold=0.001, GPU 1)

```bash
# 运行实验 (较快版本)
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python -u examples/run_hdmi.py \
  task=move_suitcase +data_id=1 viewer=none save_video=false save_info=true \
  output_dir=workspace/hdmi_reproduce/results/R013b use_torch_compile=false \
  improvement_threshold=0.001 \
  2>&1 | tee workspace/hdmi_reproduce/results/R013b/run.log
```

### 渲染对比视频

```bash
# R013 对比视频
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
  --scene "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml" \
  --kin workspace/hdmi_reproduce/results/R013/trajectory_kinematic.npz \
  --phys workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz \
  --output workspace/hdmi_reproduce/results/R013/R013_comparison.mp4 \
  --fps 10

# R013b 对比视频
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
  --scene "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml" \
  --kin workspace/hdmi_reproduce/results/R013b/trajectory_kinematic.npz \
  --phys workspace/hdmi_reproduce/results/R013b/trajectory_hdmi.npz \
  --output workspace/hdmi_reproduce/results/R013b/R013b_comparison.mp4 \
  --fps 10
```

### 抽帧检查

```bash
# 使用 ffmpeg 抽取关键时间点帧 (t=0s, 1s, 2s, 4s, 8s)
ffmpeg -ss 00:00:00 -i workspace/hdmi_reproduce/results/R013/R013_comparison.mp4 -frames:v 1 /tmp/R013_t0.jpg
ffmpeg -ss 00:00:01 -i workspace/hdmi_reproduce/results/R013/R013_comparison.mp4 -frames:v 1 /tmp/R013_t1.jpg
ffmpeg -ss 00:00:02 -i workspace/hdmi_reproduce/results/R013/R013_comparison.mp4 -frames:v 1 /tmp/R013_t2.jpg
ffmpeg -ss 00:00:04 -i workspace/hdmi_reproduce/results/R013/R013_comparison.mp4 -frames:v 1 /tmp/R013_t4.jpg
ffmpeg -ss 00:00:08 -i workspace/hdmi_reproduce/results/R013/R013_comparison.mp4 -frames:v 1 /tmp/R013_t8.jpg
```

### 量化分析

使用 `workspace/hdmi_reproduce/scripts/` 下的脚本:

```bash
SCENE="example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml"

# 多 run 指标对比 (含腕关节抖动、pelvis 稳定性)
python workspace/hdmi_reproduce/scripts/eval_metrics.py --scene "$SCENE" \
  workspace/hdmi_reproduce/results/R012/trajectory_hdmi.npz \
  workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz \
  workspace/hdmi_reproduce/results/R013b/trajectory_hdmi.npz

# 两个 run 之间的详细对比 (含 delta、reduction 百分比)
python workspace/hdmi_reproduce/scripts/compare_runs.py --scene "$SCENE" \
  workspace/hdmi_reproduce/results/R012/trajectory_hdmi.npz \
  workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz
```

## 附注

- R013b 的 threshold=0.001 虽然节省 80% 运行时间，但站立阶段优化不足导致摔倒倾向
- 如需加速，建议设 min_iterations (如 5) 而非仅靠 improvement_threshold
