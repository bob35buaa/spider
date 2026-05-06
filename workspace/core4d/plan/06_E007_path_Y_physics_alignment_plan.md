# R007 (V7.0) 实验计划：路径 Y — 抄 run_hdmi 设计模式到 run_mjwp，物理参数对齐 Holosoma RL

## Context

### 背景：为什么需要这次实验

E006 取得了关键突破——通过移植 HDMI 风格的 3-box 前臂碰撞 + 接触奖励 + 高物体权重，G1 机器人首次成功在 SPIDER MJWP 重定向中**物理抬起箱子至 0.61m**（vs E002 基线 0.305m，物体落地不动）。但 E006 暴露了一个**更深层的工程问题**：

> SPIDER 重定向阶段的物理仿真参数（sim_dt=0.0167, MuJoCo 默认 PD 增益顺序）与下游 Holosoma RL 训练的物理仿真参数（fps=200/physics_dt=0.005, decimation=4, IsaacSim breadth-first PD 顺序）**不一致**。

这意味着即使重定向阶段机器人能搬起物体，导出到 Holosoma RL 训练时，**同样的关节轨迹会因为物理参数不一致产生不同的行为**——接触力、碰撞响应、PD 跟踪误差都会改变。这是 sim2real 链条最常见的失败点。

### 关键讨论与决策路径

#### 第一阶段澄清：HDMI vs OmniRetarget(Holosoma) 是否本质相同

通过深度分析两篇论文（HDMI: Weng 2025; OmniRetarget: Yang 2025），明确：

- **任务定义层面**：完全相同（IsaacSim + PPO + DeepMimic + RSI，policy 输入输出空间一致）
- **数据生成层面**：HDMI 用软约束优化（reference 不干净），OmniRetarget 用 interaction mesh + 硬约束（reference 干净）
- **RL 训练层面**：HDMI 用 12 项 reward + interaction reward + Lost-Contact termination；OmniRetarget 用 5 项 minimal reward
- **核心 insight**：两者的 reward 差异不是设计偏好，而是**对应不同质量的 reference 数据**——HDMI 用 interaction reward "救" 不完美 reference，OmniRetarget 用干净数据降低 reward engineering

#### 第二阶段澄清：run_hdmi vs run_mjwp 的真正差异

| 维度 | run_mjwp.py | run_hdmi.py |
|------|-------------|-------------|
| 优化算法 | CEM 采样 MPC | **完全相同** |
| 物理后端 | mjwarp GPU batch | **完全相同** |
| Reference 来源 | 自己的 NPZ | HDMI 的 command_manager (借 IsaacLab SimpleEnv) |
| Scene XML | 自己写的 | mjlab 风格（带 `robot/`、`{obj}/` 前缀） |
| **物理时间步** | sim_dt=0.0167（直接用 ref_dt） | **physics_dt=0.002 + decimation=8**（与 IsaacLab 对齐） |
| **PD 增益** | XML 默认值 | **Isaac breadth-first 顺序覆盖** |
| **Reward 函数** | site→site 距离 | **local-frame contact offset reward** |
| 额外依赖 | 无 | active_adaptation (HDMI 包) |

**关键洞察**：两者都不是 RL 训练，**都是 Layer 1 重定向**。run_hdmi 的三个特殊设计（物理步长 / PD 顺序 / contact offset reward）**全部是为了与下游 IsaacLab 训练物理对齐**。

#### 第三阶段澄清：为什么选路径 Y（而不是路径 X）

**路径 X（直接用 run_hdmi 适配 Holosoma）的致命缺陷**：

1. 🔴 强制装 HDMI 项目（active_adaptation）作为依赖——拖入大量与 spider 主线无关的 IsaacLab 代码
2. 🔴 要重写 60% 的 run_hdmi.py（替换 SimpleEnv→NPZ、mjlab→自有 scene、HDMI yaml→CORE4D yaml）
3. 🔴 改完之后实质上等同于 run_mjwp.py——剥完依赖后剩下的代码与 mjwp 路径几乎相同
4. 🟠 调试链条长：bug 可能出在 HDMI 包、mjlab 解析、Holosoma 数据格式三处任意一个
5. 🟠 mjlab scene 格式与 SPIDER 其他数据集（Gigahand、OakInk）不一致，不利于代码统一

**路径 Y（基于 run_mjwp 抄设计模式）的优势**：

1. 🟢 **零新增外部依赖**——spider 项目保持干净
2. 🟢 **改动局部、可控**：每次只动一个点（先 sim_dt，再 PD 顺序，再 reward），可独立 ablate
3. 🟢 **修改 200-400 行 vs 重写 60% × 2000 行**——工程量差一个数量级
4. 🟢 **与其他数据集保持一致格式**——改动会扩散受益到 Gigahand、OakInk 等
5. 🟢 调试路径短：bug 一定在新加代码里

**抄什么不抄什么**（精确边界）：

| 抄过来 | 不抄 |
|---|---|
| ✅ physics_dt=Holosoma值 + decimation 设置（hdmi.py:540-547 模式） | ❌ active_adaptation 依赖 |
| ✅ 关节顺序映射工具（Isaac breadth-first ↔ MuJoCo depth-first） | ❌ HDMI 的 SimpleEnv 创建 |
| ✅ Local-frame contact offset reward（hdmi.py:1096-1117 数学公式） | ❌ HDMI 的 dataset/command_manager 接口 |
| ✅ 接触判定 gate (c_t,i) 的设计 | ❌ mjlab 的 `robot/`/`{obj}/` 前缀格式 |

### 当前根因分析

E006 之后剩余的根因（按优先级）：

1. **物理参数不对齐**：重定向 sim_dt=0.0167 ≠ Holosoma RL physics_dt=0.005——同一个轨迹在两个 sim 里物理结果不同（这是本次实验主要目标）
2. **PD 增益顺序未验证**：MuJoCo 的关节顺序与 IsaacSim breadth-first 顺序不同，若不显式映射，G1 各关节的 PD 增益会错位（c7fc76c commit 已部分修复 Holosoma 方向，但需要再次验证 mjwp 路径是否完全对齐）
3. **接触奖励缺少局部坐标信息**：CORE4D NPZ 只有手 site 的世界坐标 contact_pos，没有"贴到物体的哪个局部点"信息，导致接触不稳定（E006 物体抬起后回落）
4. **物体抬起不持续**：E006 物体只在 step 4-6 抬起后回落——可能是 contact reward 不够精细（缺少 local-frame 信息）

### 关键 insight

**物理一致性是 sim2real 的隐性必要条件**。即使 SPIDER 重定向得到漂亮的轨迹，如果与下游 Holosoma 训练的物理参数不一致，导出到 Holosoma 后会有 **"重定向能搬，训练不能搬"** 的退化。本次实验的核心假设：**对齐物理参数后，E006 的物体抬起效果在 Holosoma RL 训练里能保持/增强**。

## Claims（可验证声明）

| Claim | 最低证据 |
|-------|---------|
| C1: physics_dt=0.005+decimation=4 不破坏 E006 的物体抬起效果 | obj z_max ≥ 0.55m, obj_pos_err ≤ 0.40m |
| C2: 显式 PD 顺序映射不引入回归 | pelvis_err ≤ 0.15m, joint_err ≤ 0.08rad |
| C3: 加入 local-frame contact offset reward 改善接触持续性 | 物体 z>0.40m 的帧数比例 ≥ 30% (E006: 27%) |
| C4: 物理参数对齐后导出到 Holosoma 数据格式仍兼容 | E005 风格的导出脚本能直接处理新输出 |
| C5: 与 E006 baseline 比，新管线在 obj_pos_err 上不劣化超 10% | err 差值 < 0.04m |

## 改动清单

### 1. 添加可配置的 physics_dt 与 decimation 支持

**文件**: `spider/config.py`, `spider/simulators/mjwp.py`, `examples/run_mjwp.py`

**目的**：让 mjwp 路径支持"小 physics_dt + decimation"模式，与 Holosoma 对齐。

```python
# spider/config.py 新增字段
@dataclass
class Config:
    # 物理仿真细节（路径 Y）
    physics_dt: float = -1.0  # < 0 表示使用 sim_dt 作为 physics_dt（向后兼容）
    sim_decimation: int = 1   # 一个 sim_dt 内跑几次 mjwarp.step
```

```python
# spider/simulators/mjwp.py::setup_env 修改
if config.physics_dt > 0:
    decimation = int(round(config.sim_dt / config.physics_dt))
    assert abs(config.sim_dt - decimation * config.physics_dt) < 1e-5
    model_cpu.opt.timestep = config.physics_dt
    config.sim_decimation = decimation
else:
    model_cpu.opt.timestep = config.sim_dt  # 旧行为
    config.sim_decimation = 1
```

```python
# step_env 修改：内部循环 decimation 次
def step_env(config, env, ctrl_mujoco):
    ...
    for _ in range(config.sim_decimation):
        mjwarp.step(env.model_wp, env.data_wp)
```

**设计考量**：
- `physics_dt < 0` 作为 sentinel 保持向后兼容——所有现有数据集（Gigahand、OakInk、E001-E006）行为不变
- 仅在新配置（`core4d_box025_holosoma_aligned.yaml`）里设置 `physics_dt=0.005, sim_decimation=4`
- 对应 Holosoma fps=200 + control_decimation=4

### 2. 显式 PD 增益顺序映射工具

**文件**: `spider/mujoco_utils.py`（新增函数）, `spider/simulators/mjwp.py`

**目的**：把 Holosoma yaml 里的 PD 增益（IsaacSim breadth-first 顺序）正确映射到 MuJoCo actuator 顺序。

```python
# spider/mujoco_utils.py 新增
def apply_isaac_pd_to_mjwp(model_wp, isaac_joint_names, isaac_kp, isaac_kd, mj_actuator_names):
    """
    Isaac breadth-first 顺序 → MuJoCo actuator 顺序的 kp/kd 覆盖。

    Args:
        model_wp: mjwarp Model
        isaac_joint_names: list[str] (breadth-first order)
        isaac_kp/kd: numpy array (same order as isaac_joint_names)
        mj_actuator_names: list[str] from mj_model.actuator(i).name
    """
    for i, mj_name in enumerate(mj_actuator_names):
        if mj_name not in isaac_joint_names:
            continue  # object actuator etc.
        isaac_idx = isaac_joint_names.index(mj_name)
        # 写入 model_wp.actuator_gainprm[i, 0] = isaac_kp[isaac_idx]
        # 写入 model_wp.actuator_biasprm[i, 1] = -isaac_kd[isaac_idx]
```

**设计考量**：
- 工具函数化，便于未来给其他数据集复用
- 名字匹配（不依赖顺序假设）确保鲁棒性
- 从 Holosoma 的 `config_values/g1.py` 提取 PD 增益和 joint_names

### 3. Local-frame contact offset reward

**文件**: `spider/process_datasets/core4d.py`, `spider/simulators/mjwp.py`

**目的**：从 CORE4D SMPL-X + 物体姿态反算"手贴物体的局部偏移"，加入 reward。

```python
# spider/process_datasets/core4d.py 修改 - 在生成 NPZ 时计算 contact_offset
def compute_contact_offsets(qpos, obj_pos, obj_quat, hand_site_xpos):
    """
    返回 contact_target_offset (T, 2, 3): 在物体局部坐标系中的接触目标点
    返回 contact_eef_offset   (T, 2, 3): 在手 link 局部坐标系中的接触点（默认 0）
    """
    T = qpos.shape[0]
    contact_target_offset = np.zeros((T, 2, 3))
    for t in range(T):
        # 手在世界系的位置
        hand_world = hand_site_xpos[t]  # (2, 3)
        # 转到物体局部坐标系
        obj_pos_t = obj_pos[t]
        obj_quat_t = obj_quat[t]
        for h in range(2):
            local = quat_inv_apply(obj_quat_t, hand_world[h] - obj_pos_t)
            contact_target_offset[t, h] = local
    return contact_target_offset
```

```python
# spider/simulators/mjwp.py::get_reward 新增
if config.use_local_contact_reward:
    eef_pos = ...  # wrist_yaw_link 世界坐标
    eef_quat = ...
    obj_pos = ...
    obj_quat = ...
    target = obj_pos + quat_apply(obj_quat, contact_target_offset)
    eef_target = eef_pos + quat_apply(eef_quat, contact_eef_offset)
    R_local_contact = exp(-||target - eef_target|| / sigma) * contact_gate
```

**设计考量**：
- offset 在数据预处理时一次性计算并存入 NPZ，不增加运行时开销
- `use_local_contact_reward` 默认 False，向后兼容
- sigma 参考 HDMI 论文 Eq.1 用 0.3

### 4. 配置文件

**文件**: `examples/config/override/core4d_box025_holosoma_aligned.yaml`（新增）

```yaml
# @package _global_
dataset_name: core4d
task: box025_person1
data_id: 0
robot_type: unitree_g1
embodiment_type: humanoid_object
ref_dt: 0.0333333
trace_dt: 0.0333333

# === 路径 Y 关键改动 ===
sim_dt: 0.02              # = control_dt (Holosoma control_decimation*physics_dt)
physics_dt: 0.005         # = Holosoma fps=200
sim_decimation: 4         # = Holosoma control_decimation

scene_name: scene_forearm
num_samples: 2048
terminate_resample: true

# 继承 E006 的 reward 配置
contact_rew_scale: 1.0
pos_rew_scale: 3.0
rot_rew_scale: 1.0
base_pos_rew_scale: 3.0
base_rot_rew_scale: 1.0

# 路径 Y 新 reward
use_local_contact_reward: true
local_contact_sigma: 0.3
local_contact_rew_scale: 5.0  # 参考 HDMI 论文 weight=5.0

# Holosoma PD 对齐
apply_holosoma_pd: true
```

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/config.py` | +`physics_dt`, +`sim_decimation`, +`use_local_contact_reward`, +`local_contact_sigma`, +`local_contact_rew_scale`, +`apply_holosoma_pd` |
| 2 | `spider/simulators/mjwp.py::setup_env` | physics_dt 设置 + decimation 计算 |
| 3 | `spider/simulators/mjwp.py::step_env` | 内层循环 decimation 次 mjwarp.step |
| 4 | `spider/simulators/mjwp.py::get_reward` | local-frame contact offset reward 项 |
| 5 | `spider/mujoco_utils.py` | +`apply_isaac_pd_to_mjwp()` 工具 |
| 6 | `spider/process_datasets/core4d.py` | +`compute_contact_offsets()`，写入 NPZ |
| 7 | `examples/run_mjwp.py` | 调用 PD 映射工具（条件性） |
| 8 | `examples/config/override/core4d_box025_holosoma_aligned.yaml` | 新配置 |
| 9 | `workspace/core4d/scripts/retarget/retarget_core4d_holosoma_aligned.sh` | 新脚本 |

## Reward 权重对比

| 类别 | E006 (前次) | **E007 (本次)** | 备注 |
|------|------------|-----------------|------|
| Body pos/rot tracking | 3.0 / 1.0 | 3.0 / 1.0 | 不变 |
| Object pos/rot tracking | 3.0 / 1.0 | 3.0 / 1.0 | 不变 |
| Contact reward (site距离) | 1.0 | 1.0 | 不变（保留作为粗粒度信号） |
| **Local contact offset reward** | 无 | **5.0** | **新增 — HDMI 风格** |
| Joint regularization | 0.003 | 0.003 | 不变 |

## 训练命令

```bash
# E007a: 仅 physics_dt 对齐（C1 验证）
bash workspace/core4d/scripts/retarget/retarget_core4d_holosoma_aligned.sh \
    box025_person1 ablation=physics_dt_only

# E007b: physics_dt + PD 顺序映射（C1+C2）
bash workspace/core4d/scripts/retarget/retarget_core4d_holosoma_aligned.sh \
    box025_person1 ablation=physics_dt_pd

# E007c: 完整路径 Y（C1+C2+C3）
bash workspace/core4d/scripts/retarget/retarget_core4d_holosoma_aligned.sh \
    box025_person1 ablation=full
```

## 成功标准

| 指标 | E006 (前次) | **E007 目标** | Claim |
|------|-------------|---------------|-------|
| obj_pos_err | 0.324m | ≤ 0.40m（不劣化超 10%） | C5 |
| obj z_max | 0.610m | ≥ 0.55m | C1 |
| obj z>0.40m 帧占比 | ~27% (3/11) | ≥ 30% | C3 |
| pelvis z_min | 0.584m | ≥ 0.50m | C2 |
| joint_err | 0.073rad | ≤ 0.08rad | C2 |
| Holosoma 导出兼容 | OK | **OK**（脚本无错） | C4 |

## 特别需要注意的点（"踩坑预警"）

### 1. physics_dt 改变引入数值不稳定性

- mjwarp 在小 dt + 复杂接触下可能数值发散
- **缓解**：开启 `iterations=5, ls_iterations=10, integrator=IMPLICITFAST`（参考 hdmi.py:543-547 的稳定性设置）
- **验证**：先单跑一帧确认无 NaN，再上 CEM

### 2. PD 增益顺序映射的"反向 bug"

- Holosoma yaml 里关节名可能没有 `joint` 后缀（如 `left_hip_pitch` vs MuJoCo 的 `left_hip_pitch_joint`）
- **缓解**：apply_isaac_pd_to_mjwp 内做名字归一化（去除 `_joint` 后缀做匹配）
- **验证**：打印每个映射前后的 (mj_name, isaac_kp, isaac_kd) 对照表

### 3. CEM 优化时间增加

- decimation=4 意味着每个 rollout 步要跑 4 次 mjwarp.step → CEM 总时间约 ×4
- E006 单次重定向 ~50s → E007 预计 ~200s
- **缓解**：先减 num_samples (2048→1024) 验证流程，再恢复

### 4. local contact offset 的"参考偏差"

- CORE4D 是人手贴物体，G1 是球形/前臂贴物体——直接用人手 offset 可能让 G1 关节不可达
- **缓解**：offset 用 G1 重定向后的手 site 位置（而不是人手位置）反算物体局部坐标
- **风险**：如果重定向本身没贴到物体（E002-E004 那种），offset 会被记录为悬空点

### 5. 与 E006 改动的"叠加效应"

- E006 已经有：3-box 前臂碰撞 + 0.5kg 物体 + 高物体权重
- E007 再加：physics_dt + PD 顺序 + local contact reward
- **风险**：多个改动同时叠加，单点失败时难定位
- **缓解**：严格按 ablation 顺序（E007a → b → c）单点引入

### 6. Holosoma PD 值的来源

- Holosoma 的 G1 配置在 `holosoma/src/holosoma/.../config_values/g1.py`
- 必须**直接读取**而不是手抄数字（防数据陈旧）
- **建议**：写 utility 直接 `import` Holosoma 配置或读 yaml

### 7. 不要破坏 E001-E006 的回归

- 所有改动用 sentinel 默认值（physics_dt < 0、apply_holosoma_pd=False、use_local_contact_reward=False）
- E001-E006 重新跑一遍确认结果一致（pos_err 误差 < 0.01m）

### 8. 时间预算

- 实施：1.5 天（代码 + 配置 + 脚本）
- E007a/b/c 各跑 ~5min（含 build），总 ~30min
- Ablation 分析 + log：0.5 天
- **总计：2 天**

## 失败兜底

如果 E007c 在 obj_z_max 上劣化 > 15%，说明 physics_dt 改变破坏了 E006 的接触机制。预案：
- 退到 E007b（仅 PD 对齐，physics_dt 保持 0.0167）
- 把"物理参数对齐"挪到 Holosoma 项目侧（在 Holosoma RL 训练时单独验证）
- 主线继续 E006 配置 + Holosoma 导出

## 与"路径 X"的明确边界

本次实验**不做**以下事情：
- ❌ 安装 active_adaptation 或 HDMI 项目
- ❌ 修改 run_hdmi.py 或 simulators/hdmi.py
- ❌ 引入 mjlab scene 格式
- ❌ 改 spider 主流程的算法逻辑
- ❌ 集成 Holosoma RL 训练循环（那是 Layer 2 的事）

仅限 Layer 1 (重定向)，仅限 mjwp 路径，仅限 CORE4D 数据集。
