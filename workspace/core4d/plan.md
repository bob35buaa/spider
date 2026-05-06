# 研究计划：基于SPIDER动力学重定向的人机协作运动生成

## Context

**问题**: OmniRetarget 运动学重定向在 CORE4D 人-人-物协作数据上失败 — G1 机器人（1.32m）比人（~1.7m）矮，手只能触及箱子底部；无物理约束导致脚滑、穿模，40 次 RL 实验中机器人从未真正抬起物体。

**方案**: 用 SPIDER 的采样 MPC 物理重定向替代纯运动学重定向。SPIDER 已在 HDMI/OMOMO 场景验证（R013: rew=6.83，超 HuggingFace 基线 116%），需要适配到 CORE4D 的两人协作场景。

**目标**: 建立 CORE4D → SPIDER → Holosoma RL 的完整数据流，使动力学重定向后的运动数据可直接用于 Holosoma 项目的 RL 训练，替代当前运动学重定向数据。

---

## Phase 0: 数据管线搭建

**目标**: 将 holosoma 已重定向的 CORE4D 数据转换为 SPIDER 可消费的格式。

### 0.1 创建 `spider/process_datasets/core4d.py`

**输入**: holosoma 重定向数据 (`qpos(T,43), fps=30`)
- 路径: `/home/ubuntu/Workspace/holosoma/workspace/v2/results/retarget_replace_batch_trimmed/*.npz`
- 格式: `qpos[0:7]=pelvis_freejoint, qpos[7:36]=29_joints, qpos[36:43]=object_freejoint`

**输出**: SPIDER `trajectory_kinematic.npz`
- `qpos(T,43)` — 直接复制
- `qvel(T,41)` — 通过 `mujoco.mj_differentiatePos` 计算
- `ctrl(T,29)` — `qpos[:, 7:36]`（关节 PD 目标）
- `contact(T,2)` — 双手接触标记（协作搬运默认 `ones`）
- `contact_pos(T,2,3)` — 手部接触位 site 的世界坐标

**参考模板**: `spider/process_datasets/gmr.py`（相同模式：加载 qpos → FK → 计算 qvel/ctrl/contact → 保存 NPZ）

**关键细节**:
- fps=30 对应 `ref_dt=0.0333333`（与 `humanoid_object.yaml` 一致）
- contact site: 使用 G1 场景中 `left_rubber_hand_link` / `right_rubber_hand_link` 上的 site
- 输出目录结构: `example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}/{data_id}/`

### 0.2 场景 XML 生成

每个 CORE4D 物体需要一个 MuJoCo 场景 XML（G1 + 物体）：

**标准场景** (`scene.xml`): G1 freejoint + 物体 freejoint → `nq=43, nv=41, nu=29`
**接触引导场景** (`scene_act.xml`): G1 freejoint + 物体 6DOF PD 执行器 → `nq=42, nv=41, nu=35`

步骤:
1. 解压 CORE4D `object_models.zip` → 获取 `.obj` 网格
2. 用 `spider/preprocess/decompose.py` (CoACD) 做凸分解
3. 基于 `spider/assets/robots/unitree_g1/scene.xml` 插入物体 body/geom/joint
4. 参考 HDMI 场景 XML (`example_datasets/processed/hdmi/.../mjlab scene.xml`) 的结构

**需要生成的物体场景** (首批 5 类):
| 物体 | 网格 | 质量(估) | 接触几何 |
|------|------|----------|---------|
| Box025 | `box/box025_m.obj` | ~5kg | 长方体 61×61×89cm |
| bucket005 | `bucket/bucket005_m.obj` | ~2kg | 圆柱+把手 |
| bucket010 | `bucket/bucket010_m.obj` | ~2kg | 同上 |
| chair022 | `chair/chair022_m.obj` | ~5kg | 非规则 |
| desk005 | `desk/desk005_m.obj` | ~10kg | 大平面 |

### 0.3 Hydra 配置

已有 `examples/config/override/holosoma_box025.yaml` 可直接使用，只需:
- 将 `dataset_name` 改为 `core4d`
- 创建对应的 `core4d_box025.yaml` / `core4d_box025_act.yaml`（带 contact guidance）

**验证**: 加载生成的 NPZ + 场景 XML，在 MuJoCo viewer 中播放运动学轨迹，确认与 holosoma 可视化一致。

### 关键文件
- 新建: `spider/process_datasets/core4d.py`
- 新建: `examples/config/override/core4d_box025.yaml`, `core4d_box025_act.yaml`
- 参考: `spider/process_datasets/gmr.py`
- 参考: `spider/preprocess/generate_xml.py`（XML 生成逻辑）
- 参考: `spider/preprocess/decompose.py`（凸分解）

---

## Phase 1: 单人单物体动力学重定向

**目标**: 在 Box025 person1 上验证 SPIDER 物理重定向的效果。

### 1.1 基线运行（无接触引导）

```bash
uv run examples/run_mjwp.py +override=core4d_box025 task=box025_person1 data_id=0 viewer=rerun
```

预期: 机器人身体跟踪良好，物体因 freejoint 无约束而漂移。

### 1.2 接触引导运行

```bash
uv run examples/run_mjwp.py +override=core4d_box025_act task=box025_person1 data_id=0 viewer=rerun
```

预期: 物体跟踪显著改善（衰减 PD 增益引导物体，然后由物理接力）。

### 1.3 与运动学基线对比

| 指标 | 运动学基线 | SPIDER 无引导 | SPIDER 有引导 | 目标 |
|------|-----------|-------------|-------------|------|
| obj_pos_err (m) | 参考 | < 基线 | < 0.05 | < 0.05 |
| foot_slide (cm) | 有 | < 2cm | < 2cm | < 2cm |
| penetration | 有 | 无 | 无 | 无 |
| 手-物距离 (cm) | ~5cm | 待测 | < 5cm | < 3cm |

**可验证声明**: "SPIDER 物理优化使物体位置误差降低 >50%，同时消除脚滑和穿模"

### 关键参数调优空间
- `num_samples`: 2048（从 humanoid_object 继承）
- `horizon`: 0.8s
- `contact_guidance decay`: 0.8-0.9
- `init_pos_actuator_gain`: 10-20（参考 HDMI R011: pos=20, rot=0.3）
- `temperature`: 0.3
- `terminate_resample`: 打开（bad samples 被替换）

---

## Phase 2: 多物体泛化验证

**目标**: 在 9 个 CORE4D 案例上批量验证，确认方法泛化性。

### 2.1 批处理

对所有 9 个 holosoma 已重定向的案例:
1. 生成各自的场景 XML（含对应物体网格）
2. 转换为 `trajectory_kinematic.npz`
3. 运行 SPIDER MJWP 优化
4. 记录指标

### 2.2 物体特定调优

不同物体可能需要不同参数：
- **Box**: 双手侧面夹持 → 标准 contact guidance
- **Bucket**: 提手抓握 → 接触几何不同
- **Chair**: 大型非对称 → 可能需要更大 horizon
- **Desk**: 最重 → 可能需要更强 PD 增益

### 2.3 成功标准

>70% 的案例达到 obj_pos_err < 0.1m，无灾难性失败（机器人摔倒）。

---

## Phase 3: 双人协作重定向

**目标**: 处理 CORE4D 的核心特征 — 两人协作搬运同一物体。

### 3.1 方案 A: 独立重定向（先做）

分别对 person1 和 person2 进行 SPIDER 重定向：
- 每人各自的场景中包含物体（物体轨迹使用 CORE4D 地面真值）
- 分别优化，最后验证两个机器人的手是否在物体上保持兼容接触

优点: 无需修改 SPIDER 代码，直接复用 Phase 1/2 管线
缺点: 两人的物理动力学不耦合（各自的物体轨迹来自参考而非交互仿真）

**新指标**: 双人接触一致性 — 两机器人手到物体表面距离同时 < 阈值的帧比例

### 3.2 方案 B: 联合双机器人重定向（研究贡献）

一个 MuJoCo 场景包含两个 G1 + 一个共享物体:
- `nq = 36 + 36 + 7 = 79`, `nv = 35 + 35 + 6 = 76`, `nu = 29 + 29 = 58`
- SPIDER 优化器同时控制 58 个执行器
- 物体动力学由两个机器人的接触力共同决定

**所需代码修改**:
1. `spider/simulators/mjwp.py`: 在 `_diff_qpos` 中添加 `humanoid_object_multi` embodiment 分支
2. `spider/config.py`: 添加 `num_agents` 字段，调整 `noise_scale` 维度
3. 场景 XML: 两个 G1 body（前缀 `robot_A/`、`robot_B/`）+ 共享物体
4. 奖励函数: 分别跟踪两个机器人的身体+联合跟踪物体

**风险**: 58 维动作空间可能需要更多采样（4096+）或分解优化（交替优化 A/B）

### 3.3 建议路径

Phase 3.1 → 验证独立重定向可行 → Phase 3.2 尝试联合优化 → 对比两种方案

---

## Phase 4: Holosoma RL 训练数据导出

**目标**: 将 SPIDER 物理重定向输出转换为 Holosoma RL 训练格式，替换当前运动学重定向数据。

### 4.1 SPIDER 输出格式

SPIDER MJWP 优化器输出 `trajectory_mjwp.npz`:
- `qpos(N_steps, ctrl_substeps, 43)` — 物理优化后轨迹
- `qvel(N_steps, ctrl_substeps, 41)`
- `ctrl(N_steps, ctrl_substeps, 29)`

### 4.2 Holosoma RL 训练输入格式

Holosoma `MotionLoader`（`src/holosoma/holosoma/managers/command/terms/wbt.py`）直接读取 `.npz` 文件:

| Key | Shape | 说明 |
|-----|-------|------|
| `fps` | `(1,)` | 固定 50 |
| `body_pos_w` | `(T, 52, 3)` | 所有 52 个 body 的世界坐标（51 G1 links + 1 物体 link） |
| `body_quat_w` | `(T, 52, 4)` | wxyz 四元数（加载时转 xyzw） |
| `body_lin_vel_w` | `(T, 52, 3)` | body 线速度 |
| `body_ang_vel_w` | `(T, 52, 3)` | body 角速度 |
| `joint_pos` | `(T, 36)` | 前 7 列 = pelvis freejoint，后 29 列 = 关节（加载时跳过前 7） |
| `joint_vel` | `(T, 35)` | 前 6 列 = pelvis vel，后 29 列 = 关节速度（加载时跳过前 6） |
| `body_names` | `(52,)` | 52 个 body name（world, pelvis, ..., {object}_link） |
| `joint_names` | `(29,)` | 29 个关节名 |
| `object_pos_w` | `(T, 3)` | 物体世界位置 |
| `object_quat_w` | `(T, 4)` | 物体四元数（wxyz） |
| `object_lin_vel_w` | `(T, 3)` | 物体线速度 |
| `object_ang_vel_w` | `(T, 3)` | 物体角速度 |
| `partner_hand_pos_w` | `(T, 2, 3)` | 协作伙伴双手位置（可选） |
| `partner_hand_quat_w` | `(T, 2, 4)` | 协作伙伴双手朝向（可选） |

**重要**: 此格式与 `converted_for_rl_trimmed/` 中现有文件完全一致，可直接替换。

### 4.3 转换脚本: `workspace/core4d/scripts/export_spider_to_holosoma.py`

1. 加载 SPIDER `trajectory_mjwp.npz`，展平 `(N_steps × ctrl_substeps)` 时间维度
2. 加载场景 XML，对每帧设置 `mj_data.qpos` 并 `mj_forward`:
   - 提取全部 52 个 body 的 `xpos`/`xquat`（含所有 ankle sphere、contour link 等）
   - 提取物体 body 的 `xpos`/`xquat`
3. 组装 `joint_pos(T, 36)`: `qpos[:, :7]`（pelvis）+ `qpos[:, 7:36]`（29 关节）
4. 组装 `joint_vel(T, 35)`: `qvel[:, :6]`（pelvis vel）+ `qvel[:, 6:35]`（29 关节 vel）
5. 重采样到 50fps（从 SPIDER 的 sim_dt 到 Holosoma 标准）
6. 计算 body 速度（有限差分或从 qvel 推导）
7. 保存 `.npz`，格式与 `converted_for_rl_trimmed/` 一致

**body_names 映射**: 直接使用场景 XML 中 MuJoCo model 的 body 名称列表。参考已有的 `converted_for_rl_trimmed/` 文件中的 52 个 body_names（从 `world` 到 `{object}_link`）。

**partner_hand 处理**: 对于双人协作场景：
- person1 的数据中，`partner_hand_pos_w` = person2 的 SPIDER 重定向结果中的双手位置
- person2 的数据中，`partner_hand_pos_w` = person1 的 SPIDER 重定向结果中的双手位置
- 参考 holosoma 的 `add_partner_hands_to_motion.py` 脚本

### 4.4 Holosoma RL 训练验证

训练命令（直接替换 motion_file 路径）:
```bash
# 在 holosoma 项目中
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-w-object-v7-0-gated \
    simulator:isaacsim \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="SPIDER_OUTPUT.npz" \
    --robot.object.object_urdf_path="OBJECT.urdf"
```

**验证标准**:
- RL 训练启动无报错（数据格式兼容）
- 参考运动回放正确（物体位置、机器人姿态与 SPIDER 输出一致）
- 对比实验: SPIDER 物理重定向数据 vs 原始运动学重定向数据，训练相同 epoch 后:
  - object carry ratio 提升
  - object z_mean > 0.35m（实际抬起物体，而非地面滑行）
  - hand-object contact rate 提升

---

## 实验工作区结构

```
workspace/core4d/
├── EXPERIMENT_TRACKER.md       # 实验跟踪器
├── research_background.md      # 研究背景（已有）
├── progress.md                 # 会话进度
├── plan/
│   ├── 01_E001_data_pipeline_plan.md
│   ├── 02_E002_single_agent_plan.md
│   └── ...
├── log/
│   ├── 01_E001_data_pipeline_results.md
│   └── ...
└── scripts/
    ├── convert_core4d.sh       # 批量数据转换
    └── run_spider_batch.sh     # 批量 SPIDER 优化
```

---

## 实验计划映射

| 实验 | Phase | Claim | 成功标准 |
|------|-------|-------|---------|
| E001 | 0 | holosoma 数据可无损转换为 SPIDER 格式 | MuJoCo 播放与 holosoma 一致 |
| E002 | 1.1 | SPIDER 无引导可物理改善运动 | 无脚滑、无穿模 |
| E003 | 1.2 | 接触引导改善物体跟踪 >30% | obj_pos_err(引导) < 0.7 × obj_pos_err(无引导) |
| E004 | 2 | 方法泛化到 4+ 物体类别 | >70% 案例 obj_pos_err < 0.1m |
| E005 | 3.1 | 独立重定向维持双人接触一致性 | contact_consistency > 60% |
| E006 | 3.2 | 联合优化优于独立重定向 | contact_consistency 提升 >15% |
| E007 | 4 | SPIDER 输出可直接用于 Holosoma RL 训练 | 训练启动无报错，参考运动回放正确 |
| E008 | 4 | 物理重定向数据改善 RL 训练效果 | object z_mean > 0.35m（vs 运动学基线 z_mean ≈ 0.30m） |

---

## 立即行动（第一个实验 E001）

1. 创建 `workspace/core4d/` 实验工作区目录结构
2. 解压 CORE4D `object_models.zip`
3. 编写 `spider/process_datasets/core4d.py`（以 `gmr.py` 为模板）
4. 为 Box025 生成场景 XML
5. 转换 Box025 person1 数据为 `trajectory_kinematic.npz`
6. 在 MuJoCo viewer 中验证
7. 创建 Hydra 配置 `core4d_box025.yaml` / `core4d_box025_act.yaml`

---

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|---------|
| 物体碰撞网格太粗 → 手穿过物体 | 中 | 接触引导失效 | CoACD 凸分解 + 调整 mesh resolution |
| G1 身高限制 → 无法达到正确抓握高度 | 高 | 手在物体下方 | 接受适应性抓握；以手-物距离为辅助指标 |
| 58维联合优化收敛慢 | 中 | Phase 3.2 失败 | 增采样至 4096；交替 Gibbs 优化（冻结 A 优化 B） |
| holosoma 运动学参考噪声大 | 低 | SPIDER 跟踪困难 | Savitzky-Golay 平滑；修剪序列首尾 |
