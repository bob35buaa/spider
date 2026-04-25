# HDMI Workflow 复现指南

本文档记录 SPIDER + HDMI（Humanoid-Object Interaction）完整复现流程，包括环境搭建、运行方法、输入输出说明、代码架构以及当前实验结果。

> 相关文档：`docs/hdmi_setup_guide.md`（环境搭建 & 问题排查详细记录）

---

## 1. 概述

SPIDER 的 HDMI workflow 将人类动作捕捉数据转化为人形机器人与物体交互的物理可行动作。核心思路是：

1. 从 HDMI 的 motion dataset 获取参考轨迹（kinematic reference）
2. 使用 MPC（Model Predictive Control）+ 采样优化，在 1024 个并行 MuJoCo 世界中 rollout
3. 选择最优轨迹，控制机器人跟踪参考动作

**关键创新**：使用 MuJoCo Warp（mjwarp）在 GPU 上并行运行 1024 个物理仿真世界，实现高效的采样优化。

---

## 2. 环境搭建

### 2.1 前置条件

- Python 3.12+（spider 的 uv venv）
- CUDA GPU（RTX 6000 Ada 测试通过）
- MuJoCo, MuJoCo Warp, Warp, PyTorch 2.7.0

### 2.2 安装步骤（方案 C：spider uv venv + MuJoCo 后端）

```bash
# 1. Spider 基础环境
cd /home/xiayb/pHRI_workspace/spider
uv sync

# 2. 安装 HDMI 依赖
uv pip install --python .venv/bin/python3 \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python3 torchrl==0.7.0 tensordict==0.7.0 einops
uv pip install --python .venv/bin/python3 --no-deps -e /path/to/IsaacLab/source/isaaclab
uv pip install --python .venv/bin/python3 toml prettytable
uv pip install --python .venv/bin/python3 --no-deps -e /path/to/HDMI

# 3. 创建 omni.log stub（isaaclab.utils.math 需要）
mkdir -p .venv/lib/python3.12/site-packages/omni
cat > .venv/lib/python3.12/site-packages/omni/__init__.py << 'EOF'
EOF
cat > .venv/lib/python3.12/site-packages/omni/log.py << 'EOF'
import logging
_logger = logging.getLogger("omni.log")
def warn(msg, *args, **kwargs): _logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs): _logger.error(msg, *args, **kwargs)
def info(msg, *args, **kwargs): _logger.info(msg, *args, **kwargs)
EOF
```

> **注意**：不要用 `uv run`，它会根据 pyproject.toml 重新 sync 覆盖手动安装的包。直接用 `.venv/bin/python3`。

### 2.3 Git 分支

| 仓库 | 分支 | 说明 |
|------|------|------|
| Spider (`bob35buaa/spider`) | `feat/hdmi-uv-env` | 当前工作分支，包含 Warp batch stepping |
| HDMI (`bob35buaa/HDMI`) | `feat/spider-mujoco-v2` | MuJoCo 后端兼容修改 |

```bash
# 确保 HDMI 在正确分支
cd /path/to/HDMI
git checkout feat/spider-mujoco-v2

# HDMI 分支push方法
git push myfork feat/spider-mujoco-v2
# spider push
git push origin feat/hdmi-uv-env
```

---

## 3. 运行方法

### 3.1 基本命令

```bash
cd /home/xiayb/pHRI_workspace/spider

# 有 GUI 显示（需要 DISPLAY 环境变量）
.venv/bin/python3 examples/run_hdmi.py \
  task=move_suitcase \
  joint_noise_scale=0.2 \
  knot_dt=0.2 \
  ctrl_dt=0.04 \
  horizon=0.8 \
  +data_id=1 \
  viewer="mujoco-rerun" \
  rerun_spawn=true \
  +save_rerun=true \
  +save_metrics=false \
  max_sim_steps=-1

# 无 GUI（服务器/headless）
.venv/bin/python3 examples/run_hdmi.py \
  task=move_suitcase \
  joint_noise_scale=0.2 \
  knot_dt=0.2 \
  ctrl_dt=0.04 \
  horizon=0.8 \
  +data_id=1 \
  viewer="none" \
  save_video=false \
  max_sim_steps=10
```

### 3.2 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | `move_suitcase` | HDMI 任务名（`move_suitcase`, `move_largebox` 等） |
| `+data_id` | 0 | 数据序号（对应 motion data 的不同序列） |
| `sim_dt` | 0.02 | MuJoCo 仿真步长 (s) |
| `ctrl_dt` | 0.04 | MPC 控制周期 (s)，每个控制周期执行 2 个 sim_step |
| `knot_dt` | 0.20 | MPC 轨迹节点间距 (s) |
| `horizon` | 0.80 | MPC 预测范围 (s)，对应 40 个 sim steps |
| `num_samples` | 1024 | GPU 并行采样数 |
| `max_num_iterations` | 32 | 每个 MPC step 最大优化迭代数 |
| `joint_noise_scale` | 0.2 | 关节噪声尺度（控制探索幅度） |
| `max_sim_steps` | -1 | 最大仿真步数（-1 表示跟随参考轨迹长度） |
| `viewer` | `mujoco-rerun` | 可视化后端（`none`, `mujoco`, `rerun`, `mujoco-rerun`） |
| `save_video` | true | 是否保存渲染视频（headless 需设为 false） |

### 3.3 可用任务

目前 HDMI 提供以下任务的 MuJoCo scene XML：

```
g1_29dof_nohand-suitcase.xml
g1_29dof_nohand-box.xml
g1_29dof_nohand-ball.xml
g1_29dof_nohand-bread_box.xml
g1_29dof_nohand-door.xml
g1_29dof_nohand-foam.xml
g1_29dof_nohand-foldchair.xml
g1_29dof_nohand-plasticbox.xml
g1_29dof_nohand-stool.xml
g1_29dof_nohand-trash_bin.xml
g1_29dof_nohand-wood_board.xml
...
```

---

## 4. 输入输出说明

### 4.1 输入

#### Motion 数据（参考轨迹来源）

```
HDMI/data/motion/g1/omomo/sub1_suitcase_011/
```

由 HDMI 的 `RobotObjectTracking` command manager 加载，包含人体动作捕捉数据转换后的机器人关节轨迹和物体运动轨迹。

#### HDMI 配置文件

```
HDMI/cfg/task/base/hdmi-base.yaml     # 基础配置
HDMI/cfg/task/G1/hdmi/move_suitcase.yaml  # 任务特定配置
```

关键配置项：
- `robot.name: g1` — 机器人型号
- `command.object_asset_name: suitcase` — 交互物体
- `reward` — 跟踪奖励权重

#### MuJoCo Scene XML

```
HDMI/active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand-suitcase.xml
```

包含机器人和物体的组合 MuJoCo 模型，定义了物理属性、碰撞体、关节约束等。

#### SPIDER 配置

```
spider/examples/config/hdmi.yaml
```

定义 MPC 优化参数（采样数、迭代次数、时间步长等）。

### 4.2 输出

输出路径：`example_datasets/processed/hdmi/unitree_g1/humanoid_object/{task}/{data_id}/`

#### `trajectory_kinematic.npz` — 运动学参考轨迹

| 字段 | 形状 | 说明 |
|------|------|------|
| `qpos` | `(T, 43)` | 参考位置：root_pos(3) + root_quat(4) + joint_pos(29) + obj_pos(3) + obj_quat(4) |
| `qvel` | `(T, 41)` | 参考速度：root_lin_vel(3) + root_ang_vel(3) + joint_vel(29) + obj_lin_vel(3) + obj_ang_vel(3) |
| `ctrl` | `(T, 23)` | 参考控制（23 个被控关节的 action） |

其中 T = max_sim_steps + horizon_steps + ctrl_steps（最后几帧为 padding）。

#### `trajectory_hdmi.npz` — MPC 优化结果

**轨迹数据**（每个 MPC 控制周期记录 ctrl_steps 个 sim step）：

| 字段 | 形状 | 说明 |
|------|------|------|
| `qpos` | `(N_ctrl, ctrl_steps, 43)` | 实际关节位置 |
| `qvel` | `(N_ctrl, ctrl_steps, 41)` | 实际关节速度 |
| `time` | `(N_ctrl, ctrl_steps)` | 仿真时间 |
| `ctrl` | `(N_ctrl, ctrl_steps, 23)` | 实际执行的控制信号 |
| `ctrl_ref` | `(N_ctrl, ctrl_steps, 23)` | 参考控制信号 |

**优化诊断**（每个 MPC 步 × 每次迭代）：

| 字段 | 形状 | 说明 |
|------|------|------|
| `qpos_dist_{mean,max,min,median}` | `(N_ctrl, max_iter)` | 位置跟踪误差 |
| `qvel_dist_{mean,max,min,median}` | `(N_ctrl, max_iter)` | 速度跟踪误差 |
| `rew_{mean,max,min,median}` | `(N_ctrl, max_iter)` | 总奖励 |
| `improvement` | `(N_ctrl, max_iter)` | 每次迭代的改进量 |
| `opt_steps` | `(N_ctrl, 1)` | 实际优化迭代次数 |
| `sim_step` | `(N_ctrl,)` | 对应的仿真步 |
| `trace_cost` | `(N_ctrl, max_iter, 6)` | trace 成本（手脚+物体位置） |

#### `visualization_hdmi.mp4` — 渲染视频（需 `save_video=true` 且有 DISPLAY）

---

## 5. 代码架构

### 5.1 核心文件

```
spider/
├── examples/
│   ├── run_hdmi.py              # HDMI 入口脚本
│   └── config/hdmi.yaml         # HDMI 默认配置
├── spider/
│   ├── simulators/
│   │   ├── hdmi.py              # HDMI 模拟器（Warp batch stepping）
│   │   └── mjwp.py              # MJWP 模拟器（参考实现）
│   ├── optimizers/sampling.py   # 采样优化器（MPC）
│   └── config.py                # Config dataclass

HDMI/active_adaptation/
├── envs/
│   ├── locomotion.py            # SimpleEnv（MuJoCo 场景初始化）
│   ├── mujoco.py                # MJArticulation, MJScene, MJSim, MJRigidObject
│   ├── base.py                  # 基础环境（TorchRL）
│   └── mdp/
│       ├── base.py              # Command, Observation, Reward 基类
│       ├── action.py            # JointPosition ActionManager
│       ├── commands/hdmi/       # RobotObjectTracking command
│       └── observations/        # 观测模块
├── assets_mjcf/
│   └── g1_29dof_nohand/         # G1 机器人 MJCF + scene XML
└── cfg/task/                    # Hydra 配置
```

### 5.2 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_hdmi.py                              │
│                                                                 │
│  1. setup_env()                                                 │
│     ├── 创建 HDMI SimpleEnv（单 env，CPU MuJoCo）              │
│     ├── 提取 PD 增益、关节映射、action scaling                  │
│     ├── mjwarp.put_model() → GPU Warp model                    │
│     ├── mjwarp.put_data(nworld=1024) → 1024 个并行世界          │
│     └── _compile_step() → CUDA graph                           │
│                                                                 │
│  2. get_reference()                                             │
│     ├── 从 HDMI command manager 读取 motion dataset             │
│     └── 输出 qpos_ref, qvel_ref, ctrl_ref                      │
│                                                                 │
│  3. MPC 循环（每 ctrl_dt = 0.04s）                              │
│     ├── optimize()                                              │
│     │   ├── save_state() → wp.copy(data_wp → data_wp_prev)     │
│     │   ├── rollout 1024 samples × horizon steps               │
│     │   │   ├── step_env()                                     │
│     │   │   │   ├── action → PD torque (batch)                 │
│     │   │   │   ├── wp.copy(ctrl → data_wp.ctrl)               │
│     │   │   │   └── wp.capture_launch(graph) × decimation      │
│     │   │   └── get_reward() → qpos/qvel tracking              │
│     │   ├── load_state() → wp.copy(data_wp_prev → data_wp)     │
│     │   └── 选择最优采样，更新 ctrls                            │
│     ├── step_env() × ctrl_steps（执行最优控制）                 │
│     ├── sync_env() → broadcast world 0 → all worlds            │
│     └── receding horizon update（滑动窗口）                     │
│                                                                 │
│  4. 保存 trajectory_hdmi.npz, video                            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 PD 控制转换

HDMI 的 action 空间是关节位置偏移量。Warp batch stepping 需要将 action 转换为 MuJoCo 的 ctrl（扭矩）：

```python
# action → joint position target
joint_pos_target = default_joint_pos + action * action_scaling

# PD controller（batch: 1024 × 29 joints）
torque = kp * (target - current_pos) + kd * (0 - current_vel)

# 写入 Warp 数据（映射 isaac → mjc 关节顺序）
wp.copy(data_wp.ctrl, wp.from_torch(mj_ctrl))
wp.capture_launch(graph)  # 执行一步物理仿真
```

### 5.4 奖励计算

直接从 Warp 批量数据计算跟踪奖励（绕过 HDMI 的单 env 奖励系统）：

```python
qpos_sim = wp.to_torch(data_wp.qpos)     # (1024, 43)
qpos_ref = reference_at_current_step      # (43,) → broadcast

# 考虑四元数差异的 qpos 距离
qpos_diff = _diff_qpos(config, qpos_sim, qpos_ref)  # 处理 root_quat 和 obj_quat
qpos_weight = _weight_diff_qpos(config)               # base_pos/rot, joint, obj_pos/rot 权重

reward = -||qpos_diff * weight||₂ - vel_rew_scale * ||qvel_sim - qvel_ref||₂
```

---

## 6. 实验结果

### 6.1 运行结果（move_suitcase, data_id=1, max_sim_steps=10）

```
Realtime rate: 0.00, plan time: 42.18s, sim_steps: 2/10, opt_steps: 32
Realtime rate: 0.00, plan time: 40.59s, sim_steps: 4/10, opt_steps: 32
Realtime rate: 0.00, plan time: 40.11s, sim_steps: 6/10, opt_steps: 32
Realtime rate: 0.01, plan time:  4.07s, sim_steps: 8/10, opt_steps:  3
Realtime rate: 0.00, plan time: 40.28s, sim_steps:10/10, opt_steps: 32
Total time: 167.24s
```

| 指标 | 数值 | 说明 |
|------|------|------|
| 每步优化时间 | ~40s（32 iter） | GPU: RTX 6000 Ada, 1024 samples |
| 实时率 | ~0.01x | 远低于实时，主要瓶颈在 Warp 内核编译和 step |
| 位置跟踪误差 | 1.31–1.53 | 加权 L2 范数（含 root, joints, object） |
| 奖励 | -1.33 ~ -1.55 | 负的跟踪误差 |
| 优化改进 | 0.05–0.29 | 每步总改进量 |
| 收敛步数 | 3–32 | 大部分步跑满 32 次迭代 |

### 6.2 输出数据维度

| 维度 | 数值 | 组成 |
|------|------|------|
| nq（位置 DOF） | 43 | root_pos(3) + root_quat(4) + joints(29) + obj_pos(3) + obj_quat(4) |
| nv（速度 DOF） | 41 | root_lin_vel(3) + root_ang_vel(3) + joints(29) + obj_vel(6) |
| nu（action 维度）| 23 | 被控关节子集（不含 wrist pitch/roll/yaw 等 6 个关节） |
| decimation | 10 | 每个 sim_dt=0.02s 内执行 10 次 physics_dt=0.002s |

### 6.3 已知问题 & 待改进

| 问题 | 状态 | 说明 |
|------|------|------|
| 性能较慢 | 待优化 | 每步 ~40s，主要是 Warp step graph 执行慢。可尝试减少 decimation 或 num_samples |
| 跟踪误差较大 | 待调优 | 当前使用简单的 qpos/qvel 跟踪奖励，未使用 HDMI 的 body keypoint 跟踪 |
| headless 渲染 | 限制 | 无 DISPLAY 时无法 save_video，需 EGL 或 osmesa 后端 |
| 四元数归一化 | 潜在风险 | Warp step 后四元数可能漂移，未做归一化 |
| 未使用 contact reward | 待实现 | HDMI 的 eef_contact 奖励未在 Warp batch reward 中实现 |
| torch.compile | 未启用 | `use_torch_compile=true` 在 hdmi.yaml 中设置但 get_reward 未编译 |

---

## 7. HDMI 侧代码修改清单

以下修改在 HDMI 仓库 `feat/spider-mujoco-v2` 分支上：

### `active_adaptation/envs/mujoco.py`

| 修改 | 说明 |
|------|------|
| 添加 `import os` | viewer 条件创建需要 |
| body/joint 名称过滤 | Isaac body names 与 MuJoCo 不完全匹配时，过滤到交集 |
| IsaacLab 兼容属性 | `body_link_pos_w`, `root_link_pos_w`, `soft_joint_pos_limits` 等别名 |
| `write_root_link_pose_to_sim` | 写入根部位姿到仿真 |
| `write_root_com_velocity_to_sim` | 写入根部速度到仿真 |
| `MJRigidObject` 类 | 自由体刚体对象（suitcase） |
| `MJRigidObjectData` 类 | 刚体数据结构 |
| `MjFilteredContactSensor` 类 | 接触传感器 stub |
| `MjContactData.force_matrix_w` | 接触力矩阵字段 |
| `MJScene.rigid_objects` | 刚体对象字典 |
| Viewer 条件创建 | 无 DISPLAY 时跳过 viewer 创建 |
| `create_*_marker` 安全检查 | viewer 为 None 时返回 None |

### `active_adaptation/envs/locomotion.py`

| 修改 | 说明 |
|------|------|
| Combined XML 加载 | 自动查找 `{robot}-{object}.xml` 格式的组合模型 |
| 注册 MJRigidObject | 从 scene XML 中找到物体 free body 并注册 |
| 注册 MjFilteredContactSensor | 为 EEF-object 接触创建 sensor |

### `active_adaptation/assets_mjcf/__init__.py`

| 修改 | 说明 |
|------|------|
| 路径 `g1_23dof` → `g1_29dof_nohand` | 匹配实际目录名 |
| Key `g1_29dof` → `g1` | 匹配 HDMI config 中的 `robot.name: g1` |

### `active_adaptation/envs/mdp/base.py`（已在 v2 分支）

| 修改 | 说明 |
|------|------|
| 条件导入 `isaacsim/carb/omni` | 仅在 `get_backend() == "isaac"` 时导入 |

### `active_adaptation/envs/mdp/commands/base.py`（已在 v2 分支）

| 修改 | 说明 |
|------|------|
| 条件导入 `carb/omni` | 同上 |

### `active_adaptation/envs/mdp/observations/__init__.py`（已在 v2 分支）

| 修改 | 说明 |
|------|------|
| `from . import amp` 用 try/except | amp.py 导入 isaaclab.assets 触发 isaacsim |

---

## 8. Spider 侧代码修改清单

在 `feat/hdmi-uv-env` 分支上：

### `spider/simulators/hdmi.py`（完整重写）

| 组件 | 说明 |
|------|------|
| `HDMIWarpEnv` dataclass | 封装 Warp batch simulation 状态 |
| `setup_env()` | 创建 HDMI env → 提取 PD 参数 → mjwarp.put_model/data → 编译 graph |
| `step_env()` | action → PD torque（batch）→ wp.copy → graph launch × decimation |
| `get_reward()` | 从 wp_data 读 qpos/qvel，计算加权跟踪奖励 |
| `save_state()` / `load_state()` | `_copy_state(data_wp ↔ data_wp_prev)` |
| `sync_env()` | `_broadcast_state` + 同步回 CPU mj_data |
| `get_reference()` | 从 HDMI command manager 提取参考 qpos/qvel/ctrl |
| `get_trace()` | 从 wp_data.xpos 提取手脚物体位置 |
| `copy_sample_state()` | 采样间状态复制（用于 resampling） |

### `examples/run_hdmi.py`

| 修改 | 说明 |
|------|------|
| 使用 `HDMIWarpEnv` 接口 | 通过 `env.hdmi_env` 访问 HDMI 环境属性 |
| 参考数据移至 GPU | `qpos_ref_dev`, `qvel_ref_dev`, `ctrl_ref_dev` |
| 从 wp_data 读取状态 | `wp.to_torch(env.data_wp.qpos)[0]` 替代 `env.sim.data` |
| 设置 reward 参考 | `env._current_qpos_ref` 在每步更新 |

### `spider/config.py`

| 修改 | 说明 |
|------|------|
| 添加 `num_resamples: int = 0` | HDMI YAML 中定义的字段 |
| 添加 `resample_ratio: float = 0.2` | 同上 |
