# R005 计划：重写 HDMI 后端为 MuJoCo Warp GPU，对齐 MJWP 工作流

## Context

**大背景**：SPIDER 的核心工作流是 `run_mjwp.py` + MuJoCo Warp (GPU batch stepping)，已验证能工作。HDMI 需要用 MuJoCo 后端复现 HF 结果。之前的 R003/R004 用 CPU MuJoCo 架构（N 个 MjData + thread pool），这是一个临时绕行方案，与 MJWP 的 GPU 架构完全不同。

**核心思路**：将 HDMI 后端重写为与 MJWP 相同的 MuJoCo Warp GPU 架构，直接复用 `mjwp.py` 的 save/load/sync/step 模式。之前 285ms/step 的性能问题是 `active_adaptation` 的 Warp 包装造成的，不是 `mujoco_warp` 本身的问题。

**关键发现**：
- `mjlab scene.xml` 已定义 29 个 `<general biastype="affine">` 位置模式 PD 执行器（Kp/Kd 内建于 XML）
- `ctrl` = 关节位置目标（弧度），PD 控制由 MuJoCo 引擎内部计算（GPU 上）
- nq=43, nv=41, nu=29（2 freejoint + 29 hinge，与 MJWP 灵巧手场景结构一致）
- `mujoco_warp` 3.7.0.1 已安装可用

**与灵巧手 (MJWP) 的差异**：
| 维度 | MJWP 灵巧手 | HDMI 全身人形 |
|------|-----------|-------------|
| embodiment_type | bimanual/left/right | humanoid_object |
| 自由度 | ~22 手指+6 腕 | 29 全身关节 |
| 物体交互 | 手指抓取 | 全身搬运 |
| Reward | qpos/qvel L2 距离 | body-space exp-kernel tracking |
| 参考数据来源 | 外部 npz 文件 | HDMI command_manager 内置 |
| 求解器参数 | iter=20, ls=50 | iter=5, ls=10 |

## MJWP vs HDMI 当前架构差异（需消除的 gap）

| 维度 | MJWP (目标架构) | HDMI 当前 (CPU) | 改动 |
|------|---------------|----------------|------|
| **Backend** | `mujoco_warp` GPU, CUDA graph | CPU MuJoCo, ThreadPool | 重写为 mjwarp |
| **Env 类型** | `MJWPEnv` dataclass | `SimpleEnv` + monkey-patch | 用 dataclass |
| **step_env** | `wp.copy(data_wp.ctrl, ...)` + `wp.capture_launch(graph)` | Python PD + `mj_step` per world | 直接写 ctrl 到 warp |
| **ctrl 含义** | 关节位置目标（弧度） | 归一化 actions | 改为关节位置目标 |
| **save/load** | Warp bulk copy (data_wp ↔ data_wp_prev) | numpy copy 4 fields | 用 warp copy |
| **sync_env** | `_broadcast_state` 16 fields | copy 4 fields | 用 warp broadcast |
| **get_reward** | 用传入的 ref slice (qpos L2) | 用 precomputed + _ref_step (body tracking) | 保持 body tracking，改读 warp 数据 |

## Plan

### Phase 1: 重写 `spider/simulators/hdmi.py` 为 MuJoCo Warp GPU 架构

核心改动：**仿照 `mjwp.py` 模式，用 `mjlab scene.xml` 创建 MuJoCo Warp 环境**。

#### 1.1 新的数据结构

```python
@dataclass
class HDMIEnv:
    # HDMI 原生 env (用于提取参考数据和 reward config)
    hdmi_env: SimpleEnv
    # MuJoCo Warp GPU 环境 (用于 MPC 批量仿真)
    model_cpu: mujoco.MjModel      # 从 mjlab scene.xml 加载
    data_cpu: mujoco.MjData
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data            # nworld=N
    data_wp_prev: mjwarp.Data       # save/load buffer
    graph: wp.ScopedCapture.Graph   # CUDA graph
    device: str
    num_worlds: int
    # Reward 配置
    rcfg: dict                      # body/joint indices for reward
    precomputed_ref: dict           # precomputed reward reference (GPU)
    ref_step: int                   # current reference step counter
```

#### 1.2 setup_env 改动

```python
def setup_env(config, ref_data):
    # 1. 创建 HDMI env (num_envs=1) 仅用于提取参考数据和 reward 配置
    hdmi_env = _create_hdmi_env(config)

    # 2. 加载 mjlab scene.xml (与 MJWP 的 setup_mj_model 对齐)
    model_cpu = mujoco.MjModel.from_xml_path(scene_xml_path)
    model_cpu.opt.timestep = config.sim_dt
    model_cpu.opt.iterations = 5     # humanoid_object 求解器参数
    model_cpu.opt.ls_iterations = 10
    model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # 3. 用参考数据初始化 CPU data
    data_cpu = mujoco.MjData(model_cpu)
    data_cpu.qpos[:] = qpos_ref[0]
    data_cpu.ctrl[:] = ctrl_ref[0]  # 关节位置目标
    mujoco.mj_step(model_cpu, data_cpu)

    # 4. 创建 GPU Warp 环境 (与 MJWP 完全对齐)
    model_wp = mjwarp.put_model(model_cpu)
    data_wp = mjwarp.put_data(model_cpu, data_cpu, nworld=N, ...)
    data_wp_prev = mjwarp.put_data(model_cpu, data_cpu, nworld=N, ...)
    graph = _compile_step(model_wp, data_wp)

    return HDMIEnv(...)
```

#### 1.3 step_env — 直接复用 MJWP 模式

```python
def step_env(config, env, ctrl):
    # ctrl: (N, 29) 关节位置目标（弧度）— 不是归一化 actions
    with wp.ScopedDevice(env.device):
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))
        wp.capture_launch(env.graph)
    env.ref_step += 1
```

#### 1.4 save/load/sync/copy — 直接复用 MJWP 的 _copy_state / _broadcast_state

从 `mjwp.py` 复制 `_copy_state` 和 `_broadcast_state` 函数，适配 HDMIEnv 字段名。

#### 1.5 get_reward — 保持 body tracking reward，改读 Warp 数据

```python
def get_reward(config, env, ref):
    # 从 Warp 读取 xpos/xquat (GPU tensor，无 CPU 同步)
    xpos = wp.to_torch(env.data_wp.xpos)   # (N, nbody, 3) 已在 GPU
    xquat = wp.to_torch(env.data_wp.xquat) # (N, nbody, 4) 已在 GPU
    # ... 现有 body tracking reward 计算（已经是 GPU tensor 操作）
```

**关键改进**：当前版本需要 `np.stack([d.xpos for d in env._mj_datas])` → numpy → GPU，改后直接从 Warp 读取 GPU tensor，零拷贝。

#### 1.6 get_reference — ctrl_ref 改为关节位置目标

当前 `ctrl_ref = (ref_joint_pos - default) / scaling`（归一化 actions）。
改为 `ctrl_ref = ref_joint_pos`（关节位置目标，直接写入 MuJoCo ctrl）。

**注意**：需要确认 scene XML 的执行器顺序与 HDMI motion dataset 的关节顺序映射。已确认执行器名 = `robot/{joint_name}`，与 motion dataset 的 joint_names 一致（可能需要前缀处理）。

### Phase 2: 快速验证

1. 用 N=64, max_iter=8, max_sim_steps=10 快速测试 MuJoCo Warp 能否运行
2. 验证 step 性能（预期 < 1ms/step，远快于 CPU 的 90ms/step at N=1024）
3. 验证 open-loop reward ≈ 5.2（与之前 CPU 版本一致）

### Phase 3: 完整运行 + 对比

N=1024, max_iter=32, max_sim_steps=250
输出到 `workspace/hdmi_reproduce/results/R005/`

### Phase 4: 如果 reward 仍低，CEM 参数调优

（保留作为 fallback，但预计 GPU 完整状态 save/load 会解决大部分问题）

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/simulators/hdmi.py` | **重写**：CPU MuJoCo → MuJoCo Warp GPU。复用 mjwp.py 的 _copy_state/_broadcast_state 模式 |
| 2 | `test_hdmi_fast.py` | 适配新接口 |
| 3 | `examples/run_hdmi.py` | 适配新接口（HDMIEnv） |
| 4 | `examples/config/hdmi.yaml` | 添加 nconmax_per_env, njmax_per_env 等 Warp 参数 |

## 从 MJWP 复用的代码

| 函数 | 来源 | 复用方式 |
|------|------|---------|
| `_copy_state` | `mjwp.py` | 直接复用（30+ warp fields 拷贝） |
| `_broadcast_state` | `mjwp.py` | 直接复用（16 fields broadcast） |
| `_compile_step` | `mjwp.py` | 直接复用（CUDA graph capture） |
| `setup_mj_model` solver 参数 | `mjwp.py` | 适配 humanoid_object 分支 |
| `step_env` 的 ctrl → warp → graph 模式 | `mjwp.py` | 直接复用 |

## 不从 MJWP 复用的代码

| 函数 | 原因 |
|------|------|
| `get_reward` | HDMI 用 body tracking exp-kernel，不是 qpos L2 |
| `get_reference` | HDMI 从 command_manager 提取，不是外部 npz |
| `precompute_reward_reference` | HDMI 特有的 reward 预计算 |
| `get_trace` | 不同的 body names |

## 成功标准

| 指标 | R004 (CPU) | R005 目标 | HF 参考 |
|------|-----------|----------|---------|
| rew_mean | 0.06 | **> 4.0** | 5.90 |
| step 性能 (N=1024) | 90ms/step | **< 5ms/step** | — |
| tracking_mean | N/A | **> 1.5** | 2.55 |
| obj_track_mean | N/A | **> 2.0** | 2.85 |

## 风险

| 风险 | 缓解 |
|------|------|
| RTX 5090 sm_120 对 mujoco_warp 也慢 | mujoco_warp 3.7.0 已支持 Blackwell; 与 active_adaptation 的 warp 不同 |
| nconmax/njmax 需要调优 | 参考 MJWP 默认值，按需增大 |
| 执行器顺序不匹配 motion dataset | 已验证执行器名=joint名，需确认 index 映射 |
| Warp xquat 是 wxyz 还是 xyzw | 检查 mujoco_warp 文档/代码确认 |
