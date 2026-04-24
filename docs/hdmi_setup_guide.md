# HDMI Workflow: Setup, Troubleshooting & Development Guide

本文档记录了 SPIDER 集成 HDMI (Humanoid-Object Interaction) 的完整流程，包括环境搭建、遇到的问题与解决方案、Git 管理、以及下一步开发方向。

---

## 1. 背景

SPIDER 的 HDMI workflow 将人类动作（来自视频/mocap）转化为人形机器人+物体交互的动作。HDMI 项目（LeCAR-Lab）提供了基于 IsaacSim/IsaacLab 的人形机器人 RL 训练环境，同时也有 MuJoCo 后端用于轻量级仿真。

### 关键路径

```
pHRI_workspace/
├── spider/                          # SPIDER retargeting framework (bob35buaa/spider)
├── Loco-Manipulation-projects/
│   └── HDMI/                        # HDMI project (fork: bob35buaa/HDMI)
├── IsaacLab/                        # IsaacLab v2.3.0 (clone)
└── IsaacLab_v2.2/                   # IsaacLab v2.2.0 (clone)
```

### 运行命令

```bash
python examples/run_hdmi.py \
  task=move_suitcase joint_noise_scale=0.2 knot_dt=0.2 ctrl_dt=0.04 \
  horizon=0.8 +data_id=1 viewer="mujoco-rerun" rerun_spawn=true \
  +save_rerun=true +save_metrics=false max_sim_steps=-1
```

---

## 2. 环境方案对比

我们尝试了三种环境方案：

| 方案 | 环境 | Python | 结果 |
|------|------|--------|------|
| A: hdmi conda + 旧 IsaacLab | `conda activate hdmi` | 3.10 | 大量 isaaclab/omniverse 依赖缺失，需创建 6+ stub 模块 |
| B: spider uv venv + IsaacLab | `.venv` (uv) | 3.12 | IsaacSim 4.5.0 不支持 Python 3.12，无法安装 |
| **C: spider uv venv + isaaclab (no-deps)** | `.venv` (uv) | 3.12 | **最干净**，只需 3 处 HDMI 改动 + 1 个 omni.log stub |

### 方案 C 详情（推荐用于 MuJoCo 后端开发）

```bash
# Spider 的 uv venv 已有 mujoco, warp, torch
# 额外安装：
uv pip install --python .venv/bin/python3 torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python3 torchrl==0.7.0 tensordict==0.7.0 einops
uv pip install --python .venv/bin/python3 --no-deps -e /path/to/IsaacLab/source/isaaclab
uv pip install --python .venv/bin/python3 toml prettytable
uv pip install --python .venv/bin/python3 --no-deps -e /path/to/HDMI

# 创建 omni.log stub（isaaclab.utils.math 需要）
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

**注意**：不要用 `uv run`，它会根据 pyproject.toml 重新 sync 覆盖手动安装的包。直接用 `.venv/bin/python3`。

### Isaac 后端为何不可行

IsaacSim 4.5.0 通过 pip 安装时，`isaacsim.core` 模块不可用——它需要完整的 Omniverse runtime 初始化。即使通过 `isaaclab.sh -p` 启动也不行（pip 安装的 IsaacSim 缺少 core extensions）。需要通过 Omniverse Launcher 安装的完整 IsaacSim 才能用 Isaac 后端。

---

## 3. Git 管理

### Spider 仓库

- **远程**: `bob35buaa/spider`
- **分支**:
  - `main` — 上游代码
  - `feat/hdmi-mujoco-backend` — 第一轮修复（已推送，基于 hdmi conda 环境的方案）
  - `feat/hdmi-uv-env` — 第二轮修复（当前工作分支，基于 spider uv venv 的方案）

### HDMI 仓库

- **上游**: `LeCAR-Lab/HDMI`（只读）
- **Fork**: `bob35buaa/HDMI`
- **远程配置**:
  ```
  origin   https://github.com/LeCAR-Lab/HDMI (fetch/push)
  myfork   https://github.com/bob35buaa/HDMI.git (fetch/push)
  ```
- **分支**:
  - `main` — 上游代码
  - `feat/spider-mujoco-compat` — 第一轮修复（已推送）
  - `feat/spider-mujoco-v2` — 第二轮修复（最小改动，只 3 个文件）

### 推送命令

```bash
# Spider
git push https://<YOUR_TOKEN>@github.com/bob35buaa/spider.git <branch>

# HDMI
cd /path/to/HDMI
git push myfork <branch>
```

---

## 4. 代码修改清单

### 4.1 Spider 侧修改（`feat/hdmi-uv-env` 分支）

#### `spider/config.py`

添加 HDMI YAML 中定义但 Config 缺失的字段：

```python
# 在 terminate_resample 之后添加
num_resamples: int = 0
resample_ratio: float = 0.2
```

#### `spider/simulators/hdmi.py`

| 改动 | 原因 |
|------|------|
| `active_adaptation.set_backend("mujoco")` | 避免 Isaac 后端的 isaacsim/omniverse 依赖 |
| `from active_adaptation.envs.locomotion import SimpleEnv` | 绕过 `envs/__init__.py` 触发的 isaaclab 导入链 |
| `cfg.num_envs = 1` | MuJoCo 后端单 env，SPIDER 通过 Warp 做 batch |
| `cfg.randomization = {}` | Randomization 类使用 PhysX API，MuJoCo 不可用 |
| 创建 `env.sim.wp_data` | `save_state`/`load_state` 需要 Warp data |
| `nconmax`/`njmax` fallback | `MJSim` 没有 `cfg` 属性，用 `config.nconmax_per_env` 兜底 |

#### `examples/run_hdmi.py`

| 改动 | 原因 |
|------|------|
| 添加 `get_terminate` 导入和参数 | `make_rollout_fn` 需要 10 个参数，原来漏了 `get_terminate` |
| `sim_data` fallback 到 `env.sim.mj_data` | `MJSim` 没有 Warp 的 `.data` 属性 |
| `ctrls.to(config.device)` | 参考轨迹在 CPU 但采样在 CUDA |

### 4.2 HDMI 侧修改（`feat/spider-mujoco-v2` 分支）

#### 最小改动（方案 C，3 个文件）

| 文件 | 改动 |
|------|------|
| `envs/mdp/base.py` | `import isaacsim/carb/omni` 放到 `if get_backend() == "isaac"` 下 |
| `envs/mdp/commands/base.py` | 同上，guard `import carb/omni` |
| `envs/mdp/observations/__init__.py` | `from . import amp` 用 try/except 包裹（amp.py 导入 isaaclab.assets 触发 isaacsim） |

#### 完整改动（方案 A，需要更多修改）

如果不装 isaaclab，还需要额外改动（已在 `feat/spider-mujoco-compat` 分支）：

| 文件 | 改动 |
|------|------|
| `assets_mjcf/__init__.py` | 路径 `g1_23dof` → `g1_29dof_nohand`，key `g1_29dof` → `g1` |
| `assets_mjcf/g1_29dof_nohand/g1_29dof_nohand.json` | 新增 robot config JSON |
| `envs/mujoco.py` | 大量改动：headless viewer、MJRigidObject、MjFilteredContactSensor、IsaacLab 兼容别名、body/joint 过滤、write_root_link_pose_to_sim 等 |
| `envs/locomotion.py` | 加载 combined XML、注册 rigid object 和 contact sensor |

---

## 5. 已验证的进展

使用方案 C（spider uv venv + MuJoCo 后端），以下阶段已通过：

| 阶段 | 状态 |
|------|------|
| HDMI 模块导入 | ✅ |
| Hydra config 加载 | ✅ |
| SimpleEnv 创建（MJScene + MJSim） | ✅ |
| Robot articulation 初始化 | ✅ |
| Suitcase rigid object 注册 | ✅ |
| Contact sensor 注册 | ✅ |
| RobotObjectTracking command 初始化 | ✅ |
| env.reset() | ✅ |
| Warp data 创建 (wp_data, data_wp_prev) | ✅ |
| Optimizer 初始化 | ✅ |
| **optimize() 进入 rollout()** | ✅ |
| **step_env() batch stepping** | ❌ 阻塞 |

---

## 6. 当前阻塞：Batch Stepping

### 问题描述

SPIDER 的优化器在每个 MPC step 需要并行 rollout 1024 个 sample（`num_samples=1024`）。MJWP 模拟器通过 MuJoCo Warp 在 GPU 上并行运行 1024 个 world。

但 HDMI 的 `step_env` 当前实现调用 `env.step()`，它是 HDMI 的 torchrl env，batch_size 固定为 1（单个 MuJoCo 环境）。当 optimizer 传入 `ctrl.shape = [1024, 23]` 时，TensorDict 报 batch dimension mismatch。

```
RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([1])
and value.shape=torch.Size([1024, 23])
```

### 根因

SPIDER MJWP 的 `step_env` 直接操作 Warp data：

```python
# mjwp.py step_env
wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl_mujoco))
wp.capture_launch(env.graph)
```

而 HDMI 的 `step_env` 走 HDMI 的 env.step()：

```python
# hdmi.py step_env (current)
tensordict = TensorDict({"action": ctrl}, batch_size=[env.num_envs])
env.apply_action(tensordict, substep)
env.scene.write_data_to_sim()
env.sim.step()
```

### 解决方案

需要改写 `spider/simulators/hdmi.py` 中的 `step_env`，使其直接操作 Warp data 而不是走 HDMI 的 env.step()。参考 MJWP 的实现：

```python
def step_env(config: Config, env, ctrl: torch.Tensor):
    """Step all worlds with provided controls of shape (N, nu)."""
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).repeat(config.num_samples, 1)
    with wp.ScopedDevice(env.sim.device):
        wp.copy(env.sim.wp_data.ctrl, wp.from_torch(ctrl.to(torch.float32)))
        # 需要创建 warp graph 或直接调用 mjwarp.step
        mjwarp.step(env.sim.mj_model, env.sim.wp_data, ...)
```

关键步骤：
1. 在 `setup_env` 中创建 Warp model 和 graph（类似 MJWP 的 `MJWPEnv.__init__`）
2. `step_env` 直接操作 `wp_data.ctrl` 并 launch graph
3. `save_state`/`load_state` 用 `_copy_state` 操作 wp_data
4. `get_reward` 从 wp_data 读取状态，调用 HDMI 的 reward 计算
5. `sync_env` 将最优 sample 的状态同步回 HDMI env（用于 viewer 和下一步 MPC）

### 需要参考的代码

- `spider/simulators/mjwp.py` — MJWP 的完整 Warp batch simulation 实现
- `spider/simulators/mjwp.py:MJWPEnv.__init__` — Warp model/graph 创建
- `spider/simulators/mjwp.py:step_env` — Warp batch stepping
- `spider/simulators/mjwp.py:save_state`/`load_state` — Warp state 管理

---

## 7. 环境安装备忘

### IsaacLab v2.2.0 安装（hdmi conda env）

```bash
WORKSPACE_DIR=/home/xiayb/pHRI_workspace
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.2.0 $WORKSPACE_DIR/IsaacLab_v2.2

conda activate hdmi
cd $WORKSPACE_DIR/IsaacLab_v2.2

pip install 'setuptools<81'
echo 'setuptools<81' > build-constraints.txt
export PIP_BUILD_CONSTRAINT="$(realpath build-constraints.txt)"
sed -i 's/flatdict==4.0.1/flatdict==4.1.0/' source/isaaclab/setup.py
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export OMNI_KIT_ACCEPT_EULA=1
./isaaclab.sh --install
unset PIP_BUILD_CONSTRAINT
```

**注意**：IsaacLab v2.2.0 和 v2.3.0 都需要 `isaacsim.core`，pip 安装的 IsaacSim 4.5.0 不提供此模块。Isaac 后端需要完整的 Omniverse 安装。

### Spider 安装到 hdmi env

```bash
conda activate hdmi
pip install --no-deps -e /path/to/spider
```

### HDMI 安装到 spider venv

```bash
uv pip install --python .venv/bin/python3 --no-deps -e /path/to/HDMI
```

---

## 8. 文件索引

| 文件 | 作用 |
|------|------|
| `examples/run_hdmi.py` | HDMI workflow 入口脚本 |
| `examples/config/hdmi.yaml` | HDMI 默认配置 |
| `spider/simulators/hdmi.py` | HDMI 模拟器适配层（setup_env, step_env, get_reward 等） |
| `spider/simulators/mjwp.py` | MJWP 模拟器（batch stepping 参考实现） |
| `spider/config.py` | Config dataclass |
| `spider/optimizers/sampling.py` | 采样优化器（make_rollout_fn, optimize 等） |
| HDMI `active_adaptation/envs/mujoco.py` | MuJoCo 后端核心（MJArticulation, MJScene, MJSim） |
| HDMI `active_adaptation/envs/locomotion.py` | SimpleEnv（setup_scene, reset 等） |
| HDMI `active_adaptation/envs/mdp/` | 观测、奖励、命令、终止条件等模块 |
