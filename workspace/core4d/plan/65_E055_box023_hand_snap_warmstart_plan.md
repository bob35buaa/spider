# E055 实验计划：box023_person1 Hand-Snap Warmstart (Path B 首验证)

> 前置阅读：`workspace/core4d/log/64_E054_case_tier_analysis_results.md`、`workspace/core4d/report/NEXT_PLAN_E054+.md` §Path B

## Context

### 前置结论 (E054)

- **box023_person1** 被 E054 选为 Path B 首验证 case：
  - Tier 1（dim_max=0.39 m，单臂可达），obj_z 抬升 53.2 cm
  - 1 个 intent window 持续 52 帧（≈1.7 s @ 30 fps）
  - `dominant_hand=both`，hand_symmetry_ratio=0.96，L_mean=28 cm / R_mean=29 cm
  - hand_obj_dist_min = 27 cm — **mocap retarget 后双手与物体之间始终有 27 cm 几何间隙**
- 数据：`example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/0/trajectory_kinematic.npz`
  - `qpos`: (136, 43)，nq = 7 (pelvis freejoint) + 29 (G1) + 7 (object freejoint)
  - 物体 z 从 ~0.14 m 抬到峰值，搬运在中段 ~52 帧

### 根因分析

53 个 CEM 实验失败的根本原因不是 reward / sigma / collision margin，而是**初始 qpos 已经离物体表面 27 cm**：CEM 在 1024×32 采样里靠控制噪声"偶然"碰到物体的概率几乎为零。`trajectory_kinematic.npz` 是 OmniRetarget 的输出，本质上是 mocap 关节角的回放，没有任何机制把 G1 的手"放"到物体表面。

### 关键 insight

**Path B 不依赖 mocap 接触为 ground truth**，依赖更弱的两个命题：
1. 物理必要性："物体被抬起 ≡ 必有外力支撑" — E054 v3 detector 已用 `lifted` 通道捕获 intent window
2. 运动学统计："手长时间停在物体附近 ≡ 接触意图" — E054 `slow_rel` 通道辅助

在 intent window 内**主动用 IK 把双手姿态投影到物体表面**（物体 mesh + 5 mm 法向 offset），用物理几何替代不可信的手 mocap。窗口外用线性插值过渡到 ref qpos。这条 **修正后的 ref qpos** 同时作为：
- CEM 初始 mean trajectory（warmstart）
- CEM body tracking reward 的新 ref（让 reward 也"信"修正后的轨迹）

## Claims

| ID | Claim | 最低证据 |
|----|-------|---------|
| C1 | 在 intent 窗口（frames ~50-101）的每一帧，IK 解出的双手 palm 到物体表面距离 ≤ 1 cm（包含 5 mm offset） | `snap_diagnostics.csv` 的 `palm_to_surface_dist` 列 ≤ 0.01 m，对所有窗口帧 |
| C2 | snap 后 G1 关节限位全部满足，且 base/leg/torso qpos 与原 ref 相比 ≤ 1 cm 偏移（只动手臂链） | csv 中 `pelvis_xyz_diff < 0.01` 且 `leg_qpos_diff < 0.01` |
| C3 | 视觉合格：rerun/mp4 中能看到双手贴在 box 上而非穿模 | 至少 5 帧关键帧目检通过（front/side 两视角） |
| C4 | warmstart trajectory 在 MJWP 单步 mj_forward 下不爆炸（无 NaN，no contact penetration > 5 cm） | rollout 5 帧后 `qpos[2]` (pelvis_z) 在 [0.5, 1.0] m 内 |
| C5 | 整套流水线可复现：本地脚本一键跑 + 落到 EXPERIMENT_TRACKER | `run_E055_snap.sh` 存在并产出 npz + 视频 + csv |

> **本计划只验证 Path B 的"snap" 这一步是否在 box023 上几何可行**。是否能改善 CEM contact / stability 是 E055-CEM (后续) 的事，不在本计划 scope 内。

## 改动

### 1. 新建 `spider/preprocess/hand_snap_ik.py` — 单臂链 IK 投影到物体表面

**文件**: `spider/preprocess/hand_snap_ik.py` (~250 行)

**核心逻辑**：

```python
def snap_hands_to_object(
    scene_xml: str,
    qpos_ref: np.ndarray,           # (T, nq) ref qpos including object freejoint
    intent_window: tuple[int, int], # (t_start, t_end) inclusive
    dominant_hand: str,             # "L" / "R" / "both"
    object_body_name: str = "object",
    surface_offset: float = 0.005,  # 5 mm normal offset
    approach_blend_frames: int = 10,
) -> tuple[np.ndarray, dict]:
    """
    For each frame in intent_window, find closest point on object mesh surface
    to current ref palm position, push it 5mm along outward normal, then solve
    arm-only IK to that target. Outside window, blend linearly back to ref qpos
    over `approach_blend_frames`.

    Returns:
        qpos_snap: (T, nq) modified qpos (only arm joints differ from ref)
        diag: dict with per-frame diagnostics (palm_to_surface, ik_residual, ...)
    """
```

**实现细节**：
1. 解析 G1 关节索引：左臂 7 dof = qpos[22:29]，右臂 7 dof = qpos[29:36]（基于 scene.xml 的关节顺序，需用 mujoco 解析验证）
2. 用 MuJoCo `mj_kinematics` 在 ref qpos 下计算 `left_palm` / `right_palm` 站点世界坐标（site 已在 scene.xml: `pos="0.08 0 0"` 相对 wrist_yaw_link）
3. 物体表面最近点：用 trimesh 加载 `object_models/box/box023_m.obj`，对当前 object pose 变换 mesh，调用 `trimesh.proximity.closest_point()`
4. IK 用 MuJoCo 自带 differentiable Jacobian + Levenberg-Marquardt（参考 `spider/preprocess/ik.py` 的 mocap-equality-constraint 路径，但简化为单臂、无碰撞）
5. 边界帧（intent 窗口前 `approach_blend_frames` 帧）：线性插值 ref → snap，避免 qpos 跳变

### 2. 新建 `workspace/core4d/scripts/E055/snap_box023.py` — 调用 hand_snap_ik 生成 box023 warmstart

**文件**: `workspace/core4d/scripts/E055/snap_box023.py` (~80 行)

读 case_tier csv 拿 intent window 起止帧 + dominant_hand，调 `snap_hands_to_object`，存 npz：

```python
# 输出
out_dir = "workspace/core4d/results/E055/box023_person1/"
out_dir/warmstart_qpos.npz   # qpos_snap, qpos_ref, intent_window, snap_mask
out_dir/snap_diagnostics.csv # 每帧 palm_to_surface, ik_residual, joint_in_limits
```

### 3. 新建 `workspace/core4d/scripts/E055/visualize_snap.py` — 离线渲染对比视频

**文件**: `workspace/core4d/scripts/E055/visualize_snap.py` (~60 行)

并排渲染 `qpos_ref` vs `qpos_snap` 的 MuJoCo 离屏视频（front + side 双视角），用 imageio 合成 mp4。每帧标注 frame_idx、in_intent、palm_to_surface_dist。

### 4. 新建 `workspace/core4d/scripts/E055/extract_snap_keyframes.sh`

复用 `extract_case_keyframes.sh` 模式，从 visualize_snap 产出的 mp4 提取 5 个关键帧（特别是 intent window 起/中/止帧）做目检。

### 5. 新建 `workspace/core4d/scripts/run_E055_snap.sh`

一键脚本：snap → 可视化 → 提取关键帧 → 打印 csv 关键统计。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/preprocess/hand_snap_ik.py` | 新建 — 单臂 IK 投影到物体表面，按 dominant_hand 切换 L/R/both |
| 2 | `workspace/core4d/scripts/E055/snap_box023.py` | 新建 — box023 warmstart 生成 |
| 3 | `workspace/core4d/scripts/E055/visualize_snap.py` | 新建 — ref vs snap 并排渲染 |
| 4 | `workspace/core4d/scripts/E055/extract_snap_keyframes.sh` | 新建 — 关键帧提取 |
| 5 | `workspace/core4d/scripts/run_E055_snap.sh` | 新建 — 一键流水线 |
| 6 | `workspace/core4d/EXPERIMENT_TRACKER.md` | 添加 E055 行 + scripts 引用 |
| 7 | `workspace/core4d/log/65_E055_box023_hand_snap_results.md` | 实验完成后写 log |

**核心代码（spider/）改动严格限制在新文件 `hand_snap_ik.py`，不动 `ik.py` / `run_mjwp.py` / `config.py`** — Path B 的 CEM 接入留给 E056+。

## Reward 权重

不适用 — 本实验只生成 warmstart 轨迹，不跑 CEM。

## 训练命令

```bash
# 一键跑完 snap + 可视化 + 关键帧
bash workspace/core4d/scripts/run_E055_snap.sh

# 等价于：
# 1. python workspace/core4d/scripts/E055/snap_box023.py
# 2. python workspace/core4d/scripts/E055/visualize_snap.py
# 3. bash workspace/core4d/scripts/E055/extract_snap_keyframes.sh
```

预期 < 2 分钟（无 CEM，纯 IK + 渲染）。

## 成功标准

| 指标 | 前次 (E054) | 本次目标 (E055) |
|------|-------------|----------------|
| 双手 palm 到 box023 表面距离（intent window 内均值） | **27 cm** (mocap 原始) | **≤ 1 cm** (snap 后) |
| 视频：双手在搬运相位是否贴在 box 上 | ❌ 总有 27 cm 间隙 | ✅ 双手贴住 box 两侧（无穿模） |
| Pelvis / leg qpos 是否被改动 | — | ≤ 1 cm 偏移（只改手臂链） |
| 关节限位 | — | 100% 满足 |

## 风险与回退

| 风险 | 概率 | 应对 |
|------|------|------|
| IK 把手解到物体内部（穿模） | 中 | 5 mm 法向 offset；穿模检测 → 回退到表面外 1 cm |
| IK 违反关节限位 | 中 | 退化到"投影到最近的可达点"，diag 标记 `geometrically_suboptimal=True` |
| 物体 mesh 加载失败（trimesh / 路径） | 低 | 直接读 scene.xml 中 `<mesh file="...">` 路径 |
| ref qpos 中物体 freejoint 与 mesh 坐标系不一致 | 中 | 用 MuJoCo 自身 mj_forward 拿 `body("object").xpos/xquat`，避免手算 |
| box023 物体 mesh AABB 17×18×20 cm，IK 失败时手仍在 27 cm 外 | 低 | 视频复查 + diag csv 量化报告，不强行成功 |

## 后续衔接 (E055-CEM 之后)

如果本计划成功（C1-C5 全过）：
- E055-CEM：将 `warmstart_qpos.npz` 传入 `examples/run_mjwp.py` 作为 CEM 初始 mean，对比有/无 warmstart 的 contact / stability / 视频
- E056：在 bucket001_person1 上验证**单手 snap**（dominant_hand="L"）

如果失败（C1 < 80%）：
- 检查 IK Jacobian 收敛性，可能需要换 ikpy / placo / mink
- 退化到 mocap-equality-constraint 路径（复用 ik.py 框架）
