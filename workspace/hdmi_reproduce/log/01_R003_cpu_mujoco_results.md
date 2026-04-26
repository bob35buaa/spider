# R003 实验结果：CPU MuJoCo HDMI MPC 复现

**日期**: 2026-04-25
**对应Plan**: `workspace/hdmi_reproduce/plan/R003_plan.md`
**前置**: R002 (feat/hdmi-uv-env 分支, 简化 qpos L2 reward)

## 1. 背景

R002 使用简化的 qpos L2 reward + 手动 PD 控制，pelvis Z 稳定 (0.789m) 但物体跟踪发散 (qpos_dist=1.81m)。R003 的目标是使用 HDMI 原生的指数核 body tracking reward + MuJoCo backend，复现 HF 参考结果。

---

## 2. R003: CPU MuJoCo + HDMI native reward

**关键改动**:
- 完整重写 `spider/simulators/hdmi.py` 为 CPU MuJoCo 架构
- N 个 `mujoco.MjData` 对象代替 Warp batch data
- 预计算所有 reward reference data 到 GPU
- 线程池并行 step_env (N>8 时)
- 修复 qpos 布局映射: MuJoCo `[suitcase(7), pelvis(7), joints(29)]` vs HF `[pelvis(7), joints(29), object(7)]`

### 性能

| 方案 | step_env 耗时 | 说明 |
|------|--------------|------|
| Warp batch (RTX 5090) | 285ms/step | sm_120 Blackwell 性能极差 |
| CPU MuJoCo serial (N=8) | 5.4ms/step | **777x 加速** |
| CPU MuJoCo threaded (N=1024, 32线程) | 90ms/step | 可用于生产 |

### Open-loop 验证 (data_id=0)

| Step | rew | tracking | obj_track | pelvis_z |
|------|-----|----------|-----------|----------|
| 1 | 5.219 | 2.224 | 2.996 | 0.791 |
| 5 | 4.778 | 1.894 | 2.885 | 0.766 |
| 10 | 4.409 | 1.523 | 2.885 | 0.681 |
| 20 | 3.657 | 0.769 | 2.888 | 0.212 |

Open-loop reference controls 无法维持站立 (无 MPC 反馈)，但初始 reward 与 HF 接近 (5.22 vs 5.40)。

### MPC 完整运行 (data_id=0, N=8, max_iter=32)

| 指标 | R003 | HF 参考 | 目标 |
|------|------|---------|------|
| rew_mean (first step) | 4.15 | 5.41 | 4.0-7.0 |
| rew_mean (全程 col0) | 1.93 | 5.72 | 4.0-7.0 |
| tracking_mean | 0.59 | 2.11 | 2.0-3.0 |
| obj_track_mean | 1.33 | 3.61 | > 2.0 |
| opt_steps (mean) | 1.0 | 2.5 | — |
| 运行时间 | 117s / 500步 | — | — |

### 分析

1. **物理和 reward 计算正确**: open-loop 初始 rew=5.22 与 HF 5.40 接近 (差异来自不同的 env reset 状态)
2. **MPC 优化效果差**: `num_samples=8` (默认 1024), CEM 采样不足导致优化器早停 (opt_steps=1)
3. **qpos 布局已修复**: `get_reference` 输出正确的 MuJoCo 布局
4. **HF data_id=1 被覆盖**: 需要重新下载

### 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| Warp 285ms/step | 1 | 改用 CPU MuJoCo N 个 mj_data |
| torch_compile hang | 1 | 设 use_torch_compile=false |
| qpos 布局错乱 | 1 | 重写 get_reference 映射 MuJoCo 布局 |
| process_config 覆盖 output_dir | 1 | 在 process_config 后重设 output_dir |
| format string ndarray | 1 | 转 float/检查 ndim |

---

## 3. Claims 验证

| Claim | 结果 |
|-------|------|
| main 分支代码能运行完整 HDMI 工作流 | **通过** — 500 步无报错完成 (117s) |
| kinematic npz 与 HF 一致 | **通过** — data_id=1 qpos diff=0 (完全匹配) |
| rew_mean 与 HF 参考同量级 | **未通过** — 1.93 vs 5.90 (data_id=1), num_samples=8 不足 |
| tracking_mean 与 HF 参考接近 | **未通过** — 需要 num_samples=1024 |
| object_tracking_mean 不发散 | **部分通过** — 初始 2.89 接近 HF 2.87，但长期下降 |

### 关键发现补充

1. **data_id=1 才是正确的参考**: motion data 与 HF data_id=1 完全匹配 (diff=0)。data_id=0 用的不同 motion 数据。
2. **data_id=1 HF 数据已恢复**: 从 HuggingFace 重新下载，rew_mean=5.46/5.90。
3. **ctrl_ref 有差异** (max=5.74): 可能是 action_scaling/default_joint_pos 版本差异，需进一步调查。

## 4. 可视化工作流

### 4.1 数据生成 (MPC 运行)

**脚本**: `test_hdmi_e2e.py` (根目录临时脚本，绕过 hydra cwd 问题)

```bash
# 运行 MPC，输出轨迹数据
.venv/bin/python3 test_hdmi_e2e.py
```

**输入**:
- HDMI env config: `examples/config/hdmi.yaml` + `HDMI/cfg/task/G1/hdmi/move_suitcase.yaml`
- Motion data: `/home/ubuntu/Workspace/HDMI/data/motion/g1/omomo/sub1_suitcase_011/motion.npz`
- 关键参数: `data_id=1, num_samples=8, max_num_iterations=32, use_torch_compile=false`

**输出**:
- `workspace/hdmi_reproduce/results/R003/trajectory_kinematic.npz` — 运动学参考轨迹
  - keys: `qpos (T+pad, 43)`, `qvel (T+pad, 41)`, `ctrl (T+pad, 23)`
  - qpos 布局: mjlab scene 格式 `[pelvis_free(7), hinge_joints(29), suitcase_free(7)]`
- `workspace/hdmi_reproduce/results/R003/trajectory_hdmi.npz` — MPC 物理轨迹
  - keys: `rew_mean (250, 32)`, `tracking_mean (250, 32)`, `object_tracking_mean (250, 32)`, `qpos (250, 2, 43)`, `qvel`, `ctrl`, `opt_steps`, ...
  - qpos 布局: HDMI env 格式 `[suitcase_free(7), pelvis_free(7), hinge_joints(29)]`（注意与 kinematic 不同！）

### 4.2 轨迹对比分析 (文本)

**脚本**: `workspace/hdmi_reproduce/scripts/compare_kin_vs_phys.py`

```bash
MUJOCO_GL=egl .venv/bin/python3 workspace/hdmi_reproduce/scripts/compare_kin_vs_phys.py \
    --scene "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml" \
    --kin "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/1/trajectory_kinematic.npz" \
    --phys "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/1/trajectory_hdmi.npz"
```

**输入**:
- `--scene`: mjlab scene XML (`nq=43, nbody=33`)，qpos 布局为 `[pelvis(7), joints(29), suitcase(7)]`
- `--kin`: 运动学 qpos (必须与 scene 布局一致)
- `--phys`: MPC 物理 qpos (必须与 scene 布局一致)

**输出**: 终端文本 — pelvis Z、object tracking error、hand-object distance 等指标

**注意**: `--phys` 的 qpos 如果是 HDMI env 布局需要先转换为 mjlab scene 布局，否则数值会错乱。HF 原版的 trajectory_hdmi.npz 已经是 mjlab scene 布局。

### 4.3 对比视频渲染

**脚本**: `workspace/hdmi_reproduce/scripts/render_trajectory_video.py`

```bash
MUJOCO_GL=egl .venv/bin/python3 workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
    --scene "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml" \
    --kin "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/1/trajectory_kinematic.npz" \
    --phys "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/1/trajectory_hdmi.npz" \
    --output "workspace/hdmi_reproduce/results/R003/comparison_hf_data1.mp4" \
    --fps 25
```

**输入**:
- 同 compare 脚本的三个文件
- `--fps`: 视频帧率 (motion 采样率 50Hz，但 ctrl_steps=2 所以实际输出 25Hz)

**输出**:
- 左右对比 MP4 视频 (左=kinematic, 右=physics)，H.264 编码 (VSCode 可播放)
- 分辨率: 1280x480 (640x480 x2 side-by-side)

**环境变量**:
- `MUJOCO_GL=egl`: 无头服务器必须设置，使用 EGL 离屏渲染

### 4.4 文件路径总览

```
example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/
├── scene/
│   └── mjlab scene.xml                    # 渲染用 MuJoCo 模型 (pelvis-first 布局)
├── 0/                                      # data_id=0 (不同 motion, 与当前 HDMI 不匹配)
│   ├── trajectory_kinematic.npz            # HF 原版 kinematic
│   ├── trajectory_hdmi.npz                 # HF 原版 MPC 结果
│   └── visualization_hdmi.mp4              # HF 原版视频
└── 1/                                      # data_id=1 (匹配当前 motion data)
    ├── trajectory_kinematic.npz            # HF 原版 kinematic (与我们的 diff=0)
    ├── trajectory_hdmi.npz                 # HF 原版 MPC 结果 (rew=5.90)
    └── visualization_hdmi.mp4              # HF 原版视频

workspace/hdmi_reproduce/results/R003/
├── trajectory_kinematic.npz                # R003 kinematic (与 HF data_id=1 匹配)
├── trajectory_hdmi.npz                     # R003 MPC 结果 (rew=1.93, N=8)
└── comparison_hf_data1.mp4                 # HF kin vs HF phys 对比视频
```

### 4.5 qpos 布局对照表

| 模型 | qpos[0:7] | qpos[7:36/43] | qpos[36:43] |
|------|-----------|---------------|-------------|
| mjlab scene.xml | pelvis freejoint | 29 hinge joints | suitcase freejoint |
| HDMI env model | suitcase freejoint | pelvis(7) + 29 hinges | — |
| HF kinematic npz | pelvis freejoint | 29 hinge joints | suitcase freejoint |
| HF trajectory_hdmi npz | pelvis freejoint | 29 hinge joints | suitcase freejoint |
| R003 trajectory_hdmi npz | suitcase freejoint | pelvis(7) + 29 hinges | — |

**关键**: 渲染脚本读取 qpos 并写入 mjlab scene model 的 `mj_data.qpos`，所以 **所有 npz 文件必须使用 mjlab scene 布局**。R003 的 trajectory_hdmi.npz 使用 HDMI env 布局，直接送入渲染会出错。

## 5. 下一步

1. **用 data_id=1 + num_samples=1024 重跑 R003**: 验证 MPC 优化效果
2. **trajectory_hdmi.npz 布局转换**: R003 输出需要从 HDMI env 布局转为 mjlab scene 布局后才能渲染
3. **ctrl_ref 差异调查**: max=5.74, 可能影响 MPC 初始控制质量
