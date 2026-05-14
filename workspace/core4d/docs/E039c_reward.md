# E039c Reward 完整公式与解释

> 实验定位: E039c = E039 yaml + `contact_hdmi_threshold` 0.30 → 0.15 (mask gate 收紧). 是 Phase 11 在 config bug fix (E039b) 后, 用更严格 mask 重新跑出的结果. 视频暴露 "手粘连物体 + 手腕反关节" 问题, 引出 E040 的 dynamic target.
>
> Code 位置: `spider/simulators/mjwp.py::get_reward()` (主 reward), `examples/run_mjwp.py::746-775` (mask 预计算).

---

## 0. Total Reward — 一行总览

```
R_total(t, n) = R_local + R_obj + R_contact_hdmi
              + (R_qvel + R_contact_site + R_task_body + R_interact + R_hand_approach + R_contact_mask + R_stability)
                ⎵_____________________________________________________________________________________⎵
                                                  全部 = 0  (E039c yaml 关闭)
```

**E039c 实际激活的 3 项**:
- `R_local` (local-frame body tracking, E035 移植 HDMI 核心) — **正向, max ≈ 3.5**
- `R_obj` (object position + orientation tracking) — **负向 cost, ≤ 0**
- `R_contact_hdmi` (HDMI-aligned predefined target, E039) — **正向, max = gain × n_eef = 5.0** 但接触帧才 >1

剩余 7 项在 E039c yaml 中 scale 为 0, 不参与计算.

下面逐项展开.

---

## 1. `R_local` — Local-frame Body Tracking (E035 移植)

### 1.1 物理意义

HDMI 不摔的真正核心 trick: body tracking error 在 **pelvis yaw-only 坐标系** 中算, 不是 world frame. 这样 CEM 可以选择"pelvis 位置/朝向稍偏 ref 但保持平衡"的方案. World-frame tracking 强行让 pelvis 跟 ref → 必摔.

### 1.2 公式

定义 `Q_yaw(q)` 为只保留 q 的 yaw 分量的单位四元数 (绕 z 轴), `q^*` 为共轭, `R(q)·v` 为四元数旋转向量, `‖·‖` 为 L2 norm.

#### a) Body local-frame 跟踪 (upper + lower)

对 body 集合 `B ∈ {upper, lower}`:

```
local_pos_b(t,n)   = R(Q_yaw(q_pelvis_sim))^* · (xpos_b_sim - xpos_pelvis_sim)
local_pos_b_ref(t) = R(Q_yaw(q_pelvis_ref))^* · (xpos_b_ref - xpos_pelvis_ref)

err_pos_b   = ‖local_pos_b - local_pos_b_ref‖₂
R_pos_B     = mean_{b∈B} exp(-err_pos_b / σ_pos)        ∈ [0, 1]

q_local_b      = Q_yaw(q_pelvis_sim)^* · q_b_sim
q_local_b_ref  = Q_yaw(q_pelvis_ref)^* · q_b_ref
err_ori_b      = ‖axis_angle(q_local_b_ref^* · q_local_b)‖₂   # rad
R_ori_B        = mean_{b∈B} exp(-err_ori_b / σ_ori)            ∈ [0, 1]
```

#### b) Root global tracking (pelvis pos + ori)

```
err_root_pos = ‖xpos_pelvis_sim - xpos_pelvis_ref‖₂
R_root_pos   = exp(-err_root_pos / σ_root)              ∈ [0, 1]

err_root_ori = ‖axis_angle(q_pelvis_ref^* · q_pelvis_sim)‖₂
R_root_ori   = exp(-err_root_ori / σ_root)              ∈ [0, 1]
```

#### c) Joint angle tracking

排除 base (前 7 维) + object (后 7 维) 后的所有 joint:

```
err_joint = mean_j |qpos_sim[j] - qpos_ref[j]|        # rad, mean over joints
R_joint   = exp(-err_joint / σ_joint)                 ∈ [0, 1]
```

#### d) 加权汇总 (max = W × 7)

```
R_local = W_track × ( R_pos_upper + R_ori_upper
                    + R_pos_lower + R_ori_lower
                    + R_root_pos  + R_root_ori
                    + R_joint )
```

### 1.3 E039c 数值

| 符号 | yaml 字段 | 值 | 说明 |
|---|---|---|---|
| `W_track` | `local_frame_w_track` | **0.5** | 整体权重 |
| `σ_pos` | `local_frame_pos_sigma` | **0.5 m** | upper/lower body 位置带宽 |
| `σ_ori` | `local_frame_ori_sigma` | **1.0 rad** | upper/lower body 朝向带宽 |
| `σ_root` | `local_frame_root_sigma` | **0.5** | root pos+ori 共用 (m / rad) |
| `σ_joint` | `local_frame_joint_sigma` | **0.25 rad** | 关节角带宽 |
| upper bodies | `local_frame_upper_ids` | 默认 (~17) | 躯干 + 双臂 |
| lower bodies | `local_frame_lower_ids` | 默认 (~12) | 双腿 |

**Reward 范围**: `R_local ∈ [0, 0.5 × 7] = [0, 3.5]`

### 1.4 几个微妙点

1. `Q_yaw(q)` 提取: 用 q 的 4 个分量直接构造 `(w, 0, 0, z) / norm`, 有效消去 pitch/roll. `_lf_yaw_quat()` 的实现.
2. **CEM 的关键自由度**: pelvis 的 pitch / roll 不被 R_local 直接惩罚, 由 R_root_ori 全局约束 (但 σ_root=0.5 比较松). 这就是允许 "pelvis 稍倾" 但仍维持 body-relative 跟踪的原因.
3. E044 sweep 验证过给 wrist 加额外权重, 全面退化, 所以 E039c 用 `local_frame_wrist_weight=1.0` (即不加权).

---

## 2. `R_obj` — Object Position + Orientation Tracking (E036 启用)

### 2.1 物理意义

E036 的核心改动之一. 直接让 CEM 优化"物体姿态 vs ref"的 L2 误差. **本质是负 cost**, 不与正向 reward 抢上限. 配合 `contact_guidance: true` (object 用 PD 沿 ref 走), 让 sim 不需要"主动搬", body tracking 跟好就行.

### 2.2 公式

E039c 用 freejoint object → `nq_obj = 7` (xyz + wxyz quat). `task_obj_use_exp=False` (E065 才引入 exp 形式), 所以走 unbounded L2:

```
err_pos² = ‖obj_pos_sim - obj_pos_ref‖²₂                       # m²
R_obj_pos = - scale_pos × err_pos²                              ≤ 0

err_rot² = ‖quat_sub(obj_quat_sim, obj_quat_ref)‖²₂             # quat 差的平方和
R_obj_rot = - scale_rot × err_rot²                              ≤ 0

R_obj = R_obj_pos + R_obj_rot                                   ≤ 0
```

`quat_sub(q1, q2)` 是 `spider.math` 中的 quat 差 (定义为 q2^* · q1 后取 axis-angle 的 vec 分量), 不是直接 q1 - q2.

### 2.3 E039c 数值

| 符号 | yaml 字段 | 值 |
|---|---|---|
| `scale_pos` | `task_obj_pos_rew_scale` | **1.0** |
| `scale_rot` | `task_obj_rot_rew_scale` | **1.0** |
| `task_obj_use_exp` | (E065 后引入) | **False** (默认) |

**Reward 范围**: `R_obj ∈ (-∞, 0]`. 实际数量级: 物体偏 1cm pos err 贡献 -1e-4, 偏 10cm 贡献 -0.01. 由于有 contact_guidance 把 object 拉得很准 (E036 实测 obj_pos_err ≈ 0.87cm), 这一项数值上 ≈ -1e-4 量级, 主要起"惩罚漂移"作用.

---

## 3. `R_contact_hdmi` — HDMI-aligned Contact Reward (E039 核心)

### 3.1 物理意义

E039 引入. 完全对齐 HDMI 的 `rew_contact` 设计:
1. **预定义接触点** `contact_target_offset` (per-task YAML, 在物体局部坐标系中)
2. **精确 quat_apply** 把 target 旋转到 world: `target_world = obj_pos + R(obj_quat) · target_offset`
3. **EEF 探测点** 不是 wrist body 中心, 而是 wrist + 5cm 的 palm 中心: `contact_point = eef_pos + R(eef_quat) · [0.05, 0, 0]`
4. **per-EEF** 分别算左右手 reward 然后 mean (不是 min)
5. **HDMI baseline 公式**: 接触帧给 gain × proximity, 非接触帧给 1.0 baseline

### 3.2 完整公式

对每个 EEF `i ∈ {left, right}`:

```
target_world_i(t,n) = obj_pos(t,n) + R(obj_quat(t,n)) · target_offset_i

contact_point_i(t,n) = eef_pos_i(t,n) + R(eef_quat_i(t,n)) · eef_offset

dist_i(t,n) = ‖target_world_i - contact_point_i‖₂                            # m

pos_rew_i(t,n) = exp(-dist_i / σ_contact)                                    ∈ [0, 1]
```

E039c **没开 orientation reward** (`contact_hdmi_ori_weight = 0` 默认), 所以 `pos_rew_i` 不再修改.

汇总到 contact reward:

```
rew_stack(t,n) = stack([pos_rew_left, pos_rew_right])                        # shape (n_eef,)

mask(t) = approach_mask_t  ∈ {0, 1}                                          # rotated SDF 预计算

R_contact_hdmi(t,n) = mean_i [ rew_stack_i × mask × gain + (1 - mask) × 1.0 ]
```

> ⚠️ **公式陷阱**: HDMI baseline 是 `(1 - mask) × 1.0`, 不是 `× gain`. 接触帧 reward 范围 [0, gain], 非接触帧 reward = 1.0. 这意味着 **接触帧 max (5.0) >> non-contact baseline (1.0)** → CEM 在 mask=1 帧有强激励把 dist 缩到 0.

### 3.3 mask `approach_mask_t` 的定义 (E039b rotated SDF)

mask 在 episode 开始前预计算 (run_mjwp.py:746-775), per-timestep, 共享 left/right (一只手满足就 mask=1):

```python
# 对每帧 t, 用 ref kinematics:
obj_pos  = mj_data_ref.xpos[obj_body_id]
obj_mat  = mj_data_ref.xmat[obj_body_id].reshape(3,3)              # world ← obj 旋转
half_ext = config.hand_approach_obj_half_extents                   # box 半边长

for hid in [left_wrist, right_wrist]:
    hand_pos = mj_data_ref.xpos[hid]
    local    = obj_mat.T @ (hand_pos - obj_pos)                    # 转到物体局部坐标
    clamped  = clip(local, -half_ext, +half_ext)                   # 投影到 box 表面
    surf_dist = ‖local - clamped‖₂                                  # box-SDF 距离
    if surf_dist < threshold:
        mask[t] = 1.0; break                                       # 一手满足即开
```

**关键差异 vs E034 旧 mask**: E034 用 `delta = |hand - obj|` (axis-aligned, 不旋转!) 当物体有旋转时严重错误. E039b 改成 rotated SDF: `obj_mat.T @ (hand - obj)` 正确转到物体局部坐标.

### 3.4 E039c 数值

| 符号 | yaml 字段 | 值 | 说明 |
|---|---|---|---|
| `gain` | `contact_hdmi_gain` | **5.0** | 接触帧 max reward |
| `σ_contact` | `contact_hdmi_sigma` | **0.3 m** | 接触距离 exp 带宽 (HDMI default) |
| `target_offset_left` | `contact_hdmi_target_left` | **[0.243, 0.270, -0.486]** (box025) | 物体局部坐标系下左手接触点 |
| `target_offset_right` | `contact_hdmi_target_right` | **[-0.243, 0.259, -0.489]** (box025) | 同上, 右手 |
| `eef_offset` | `contact_hdmi_eef_offset` | **[0.05, 0.0, 0.0]** | wrist → palm (HDMI move_suitcase 一致) |
| `threshold` | `contact_hdmi_threshold` | **0.15 m** ⭐ | E039c 关键改动: 从 E039 的 0.30 降到 0.15 |
| `dynamic_target` | `contact_hdmi_dynamic_target` | **False** | E040 才开. E039c 仍用固定 target |
| `ori_weight` | `contact_hdmi_ori_weight` | **0.0** | E041 才开. E039c 无 orientation 项 |

**Reward 范围 (per-frame)**:
- 接触帧 (mask=1): `R_contact_hdmi ∈ [0, gain] = [0, 5.0]`. dist=0 时 = 5.0 (因为 mean over 2 EEFs, 两手都贴 target 时各 5.0 mean = 5.0).
- 非接触帧 (mask=0): `R_contact_hdmi = 1.0` (constant baseline).

### 3.5 E039c 各 case 的 mask 激活率 (threshold=0.15 后)

| Case | E034 旧 axis-aligned mask | E039b rotated SDF (thr=0.30) | **E039c rotated SDF (thr=0.15)** |
|---|---|---|---|
| box025 | 100% | 100% | **64%** |
| desk005 | 100% | 100% | **72%** |
| bucket010 | 99% | 95% | **67%** |

threshold=0.15 合理过滤了 30-40% 的 ref "非接触" 帧 (即使站在物旁但不算接触).

---

## 4. 关闭的 7 项 (yaml scale=0)

下面 7 项在 E039c yaml 中显式禁用 (scale=0 或 disable flag). 列出来是为了完整说明 reward 框架, 也方便后续复盘.

| 项 | 公式简介 | 关闭原因 (E039c) |
|---|---|---|
| `R_qvel = -vel_rew_scale × ‖qvel_sim - qvel_ref‖` | 关节速度跟踪 | E039c `vel_rew_scale=0` |
| `R_contact_site` (sum-of-distance over contact sites) | DexMachina 风格 site 接触 | `contact_site_ids = []` |
| `R_task_body = -scale × Σ_b w_b ‖body_sim - body_ref‖²` | E018 task-space body | E039c `task_body_rew_scale=0` |
| `R_interact` (relative offsets, Harmanoid Eq.15) | partner_force 模式 | E039c `partner_force_scale=0` |
| `R_hand_approach = scale × exp(-min_dist/σ)` | E025 旧 hand-to-surface reward | **E036 起关闭** (E039c 沿用), max=5 会压制 R_local |
| `R_contact_mask = mask × scale × exp(-min_dist/σ_m) + (1-mask) × baseline` | E037 旧 mask reward (axis-aligned) | E039c `contact_mask_rew_scale=0`, 被 R_contact_hdmi 取代 |
| `R_stability = -scale × ReLU(threshold - pelvis_z)` | E034 反应式补救 | **E036 起关闭** (E039c 沿用), 不需要因为 R_local 已稳 |

注: `R_qpos` 在 `use_local_frame_reward=True` 时被 `R_local` **替换** (mjwp.py:659 `qpos_rew = local_frame_rew`), 所以最终 reward sum 中的 `qpos_rew` 项就是 `R_local`.

---

## 5. 数值大小对比 — 哪一项主导 CEM?

近似估算 (典型接触帧, body tracking 较好):

| 项 | 公式量纲 | E039c 典型范围 | 主导地位 |
|---|---|---|---|
| `R_local` | 7 项 × W=0.5 × exp 项 ∈ [0, 1] | **0.5 ~ 3.5** | ★★★ 主导 (max=3.5) |
| `R_obj` | -L2 m² | **-0.001 ~ 0** (有 contact_guidance) | 几乎不影响 |
| `R_contact_hdmi` | 接触帧 [0, gain]; 非接触 1.0 | **1.0 (mask=0) ~ 5.0 (mask=1, dist=0)** | ★★ 接触帧主导 |

**问题**: 在 mask=1 的帧, `R_contact_hdmi` 的 max (5.0) > `R_local` 的 max (3.5). CEM 在接触帧倾向于"先满足 contact 再 tracking" → 解释了 E039c 视频里"手粘连物体 + 手腕反关节 + 身体后仰扭转":
1. 固定 target 假设手在物体上同一个点
2. 但 ref 中机器人围着物体走, 手在物体表面位置随时变
3. CEM 强行把手粘在固定 target → 手腕反关节
4. body tracking 要求身体跟 ref 走 → 手被粘 → 手臂过度伸展 → 身体后仰

→ 引出 E040 (dynamic target) 把固定 target 换成 per-frame ref-derived target, 解决粘连但引入新问题"手背接触" (因为 position-only 无方向约束) → 引出 E041c (additive orientation reward).

---

## 6. E039c 完整 yaml 摘录 (核心字段)

```yaml
# E039 yaml + threshold 改 0.15 = E039c
use_local_frame_reward: true                    # E035
local_frame_w_track: 0.5
local_frame_pos_sigma: 0.5
local_frame_ori_sigma: 1.0
local_frame_root_sigma: 0.5
local_frame_joint_sigma: 0.25

hand_approach_rew_scale: 0.0                    # E036 关
stability_penalty_scale: 0.0                    # E036 关
task_obj_pos_rew_scale: 1.0                     # E036 开
task_obj_rot_rew_scale: 1.0
contact_mask_rew_scale: 0.0                     # E037 旧 mask 关

contact_hdmi_gain: 5.0                          # E039
contact_hdmi_sigma: 0.3
contact_hdmi_eef_offset: [0.05, 0.0, 0.0]
contact_hdmi_target_left: [0.243, 0.270, -0.486]   # box025
contact_hdmi_target_right: [-0.243, 0.259, -0.489]
contact_hdmi_threshold: 0.15                    # ⭐ E039c 关键 (E039 = 0.30)
contact_hdmi_dynamic_target: false              # E040 才开
contact_hdmi_ori_weight: 0.0                    # E041 才开

contact_guidance: true                          # 物体 PD override
guidance_decay_ratio: 1.0
init_pos_actuator_gain: 500.0
init_rot_actuator_gain: 50.0
```

---

## 7. E039c 的实测数据 (来自 log/48)

| Case | mask active | MPKPE | Stability | Contact<10cm |
|---|---|---|---|---|
| box025 | 64% | 1.4 cm | 100% | **82%** |
| bucket010 | 67% | 1.3 cm | 100% | **68%** |
| desk005 | 72% | 2.1 cm | **17%** ❌ | 91% (但摔倒) |

box025/bucket010 contact 接近 ref 上限 (69%/76%), 但视频暴露**手粘连物体 + 手腕反关节**. desk005 因 contact target 在桌面下方横梁, gain=5 把手向下拉 ~20cm, 机器人前倾摔倒.

→ 直接 motivation 出 E040 dynamic target (解决固定 target 不适合自由交互的问题).

---

## 8. 与 HDMI 原版的逐项 diff

| 维度 | HDMI (`hdmi.py:1095-1117`) | E039c | 是否对齐 |
|---|---|---|---|
| 目标点定义 | `contact_target_offset` per-task YAML | 同 | ✅ |
| 目标点计算 | `obj_pos + quat_apply(obj_quat, offset)` | 同 | ✅ |
| EEF 探测点 | `eef_pos + quat_apply(eef_quat, [0.05,0,0])` | 同 | ✅ |
| 距离度量 | `‖target - contact_point‖₂` | 同 | ✅ |
| Per-EEF | 分别算左右然后 mean | 同 | ✅ |
| Pos kernel | `exp(-dist/σ)`, σ=0.3 | 同 | ✅ |
| **Mask 来源** | NPZ 中预标注 per-EEF `object_contact` (bool) | rotated-SDF 距 box <0.15m, **左右共享** | ⚠️ 近似 (粒度更粗) |
| **Mask threshold** | (NPZ 自带, 不是几何 threshold) | 0.15 m (rotated SDF) | — |
| 非接触帧 baseline | 1.0 (constant) | 1.0 (constant) | ✅ |
| Gain | 5.0 | 5.0 | ✅ |
| **Force factor** | `exp(-10/40) ≈ 0.778` 常数乘子 | **省略** | ⚠️ 等效缩放 gain, 不影响 CEM 方向 |
| 优化器 | CEM (`make_optimize_fn`, 同 SPIDER) | 同 | ✅ |

**关键差异**: 只有 mask 粒度不同 (HDMI 可以 left=1 right=0, E039c 共享一个标量 mask). 其他完全对齐.

→ HDMI 在 move_suitcase 上 work 但 E039c 在 box025/bucket010 上有 "粘连" 问题, **不是 reward 设计差异**, 而是任务结构差异:
- HDMI move_suitcase: 把手位置固定 + wrist 零噪声 (`run_hdmi.py:112-118`) + 凸出几何引导手掌 → 固定 target 是合理假设
- CORE4D box025: 平面物体 + 人围着走 + wrist 全噪声 → 固定 target 不成立

这个判断在 E041 报告 §3 被进一步验证 (HDMI 工作流也是 CEM, 不是 RL).

---

## 9. 索引

| 资源 | 路径 |
|---|---|
| Config | `examples/config/override/core4d_e039.yaml` (E039c 用同一 yaml + CLI override `contact_hdmi_threshold=0.15`) |
| Reward 主体 | `spider/simulators/mjwp.py::get_reward()` lines 540-949 |
| `R_contact_hdmi` 计算 | `spider/simulators/mjwp.py` lines 824-928 |
| Mask 预计算 (rotated SDF) | `examples/run_mjwp.py` lines 746-775 |
| Local-frame helper | `spider/simulators/mjwp.py::_local_pos_tracking, _local_ori_tracking, _lf_yaw_quat` |
| 实验 log | `workspace/core4d/log/48_E039b_config_bug_fix_results.md` (E039c 在 §"E039c" 段) |
| 上下文报告 | `workspace/core4d/report/E041c-report.md` §4 |
