# R002 实验计划：修复 HDMI MPC 初始化对齐

## Context

R001 首次运行 HDMI Warp batch stepping，机器人 0.2 秒内倒地。根因：sim 初始状态与 ref 不对齐。

### 根因分析

MJWP 用 `qpos_ref[0]` 初始化 sim（sim/ref 起点一致），HDMI 用 `hdmi_env.reset()`（默认站姿），MPC 第一步就产生巨大误差把机器人拉倒。

### 关键 insight

对齐 sim 初始状态到 ref 第一帧，机器人应能站立并跟踪参考轨迹。

## Claims

| Claim | 最低证据 |
|-------|---------|
| get_reference 输出的 trajectory_kinematic.npz 与 HF 原始版本一致 | 两个文件 qpos 逐元素差 < 1e-3 |
| 修复初始化后，输出的 trajectory_hdmi.npz 与 HF 版本跟踪误差在同一量级 | pelvis Z 均值 > 0.5m，qpos_dist < 0.5m |

## 改动

### 0. 恢复 HF 数据 + 备份 R001

```bash
# 备份 R001
cp example_datasets/processed/hdmi/.../1/trajectory_*.npz workspace/hdmi_reproduce/results/R001/

# 从 HuggingFace 下载 hdmi 数据
python /home/xiayb/pHRI_workspace/holosoma/tools/download_hf_data.py \
  --repo_id retarget/retarget_example --allow_patterns "processed/hdmi/*" \
  --output_dir example_datasets
```

### 1. 输出路径隔离

**文件**: `examples/run_hdmi.py`

main() 中 process_config 之后覆盖 output_dir 到实验目录。

### 2. sim/ref 初始化对齐（核心）

**文件**: `spider/simulators/hdmi.py` setup_env()

在创建 Warp 数据前，先获取 qpos_ref[0]，用它设置 mj_data.qpos，再 mjwarp.put_data。

### 3. process_config hdmi 分支

**文件**: `spider/config.py` process_config()

hdmi simulator 不走 mjwp 的 model_path 读取逻辑。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `examples/run_hdmi.py` | output_dir → workspace/hdmi_reproduce/results/R002 |
| 2 | `spider/simulators/hdmi.py` | setup_env 用 qpos_ref[0] 初始化 |
| 3 | `spider/config.py` | process_config 添加 hdmi 分支 |

## 验证步骤

1. 对比 get_reference 输出的 trajectory_kinematic.npz 与 HF 下载的原始版本（逐元素 diff）
2. 对比输出的 trajectory_hdmi.npz 与 HF 下载的 trajectory_hdmi.npz（pelvis Z、qpos_dist）
3. 用 render_trajectory_video.py 离线渲染对比视频

## 成功标准

| 指标 | R001 | R002 目标 | HF 参考 |
|------|------|----------|---------|
| kinematic npz 一致性 | 被覆盖 | **diff < 1e-3** | baseline |
| Pelvis Z (mean) | 0.008m | **> 0.5m** | 0.70m |
| qpos_dist (mean) | 1.35m | **< 0.5m** | ~0.15m |
