# CORE4D 评估指标汇总（12-Gate + 运动健康/诊断指标）

Metric Standard ID: `core4d-e154-physics-contact-v1`（E154 起冻结，E194 扩展为 12-gate）。

本文汇总当前评测体系的全部指标：**12 个 pass/fail gate**（6 物理 + 6 tracking）以及一批
连续型 **运动健康 / 平滑度 / 诊断指标**（`obj_speed_max`、各类 `jerk_p95`、脚滑等）。
每条给出：定义、代码位置、公式、含义、**是否依赖 GT（reference）**。

> 旧版 `docs/eval_metrics.md` 基于 `eval_e035_comprehensive.py`（E035 时期），已被本体系取代，仅作历史参考。

---

## 0. 评测入口与数据链路

| 环节 | 位置 |
|------|------|
| 逐帧 MuJoCo replay 指标（连续量） | `scripts/eval/core/core_metrics.py::evaluate_sequence(...)` |
| 运动健康 / 平滑度 / 脚部 | `scripts/eval/core/motion_health.py` |
| 12-gate 阈值 + 判定（权威） | `scripts/eval/runners/eval_E194_G1_expansion.py`：`GATE_THRESHOLDS` + `apply_12gates()` |
| 物理 6-gate 子集（早期 runner 复用） | `scripts/eval/runners/eval_E172_box004.py` / `eval_E173_boxes.py`：`apply_gates()` |
| 标准字段清单 / 契约说明 | `scripts/eval/core/METRICS_STANDARD.md` |

**GT（reference）来源**（两类，缺失则相关指标为 NaN / gate 不适用）：
- **固定 kinematic 参考**：`<case_dir>/0/trajectory_kinematic.npz`（重定向后的 G1 运动学轨迹，非某次 run 漂移的内部 ref）。tracking 6-gate 与 body_z gate 依赖它。
- **接触 mask**：`results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz` 中 `spider_contact_mask_3cm`，形状 `(N, 2 persons, 2 hands)`。contact / release gate 依赖它。

`fps` 统一由 `motion_health.fps_from_npz()` 从 npz 的 `time` 字段推 `1/median(diff(time))`，缺失回落默认值。

---

## 1. 十二门（12-Gate）

案例通过 = 全部 12 门 PASS（`numeric_release_pass_12gate`）。判定见 `apply_12gates()`（eval_E194_G1_expansion.py:170）。方向：门内容满足即 PASS。

### 1.1 物理 6-gate

| Gate | 底层指标 | 阈值 | PASS 条件 | 依赖 GT | 代码 |
|------|----------|------|-----------|:------:|------|
| `fall` | `fall_flag` | pelvis_z ≥ 0.45 m | `not fall_flag` | 否 | core_metrics.py:1368 |
| `body_z` | `body_z_err_p95_m` | ≤ 0.20 m | p95(z 误差) ≤ 0.20 | **是**（kin 参考） | eval_E194_G1_expansion.py:117 |
| `contact` | `hand_object_physics_contact_in_mask_frac` | ≥ 0.50 | mask 内接触帧占比 ≥ 0.50 | **是**（contact mask） | core_metrics.py:1114 |
| `release` | `hand_object_release_false_contact_3mm_frac` | ≤ 0.30（或 N/A） | 释放窗内假接触 ≤ 0.30 | **是**（contact mask） | core_metrics.py:1146 |
| `hand_penetration` | `hand_object_physics_penetration_3mm_frame_frac` | ≤ 0.30 | 手-物穿透帧占比 ≤ 0.30 | 否 | core_metrics.py:1381 |
| `lower_body` | `leg_penetration_frac` | ≤ 0.10 | 腿-物穿透帧占比 ≤ 0.10 | 否 | core_metrics.py:1420 |

**定义与公式**

- **`fall`** — 整段最低骨盆高度是否跌破阈值。
  $$\text{fall\_flag} = \big(\min_t z_{\text{pelvis},t} < 0.45\big),\qquad \text{PASS} = \neg\,\text{fall\_flag}$$
  仅用 sim 自身骨盆 z，无参考。

- **`body_z`** — sim 与固定 kinematic 参考在监测 body（`left/right_ankle_roll_link`、`left/right_wrist_yaw_link`）上的竖直误差 p95。逐帧 FK 后
  $$z\_err_{b,t}=|z^{\text{sim}}_{b,t}-z^{\text{ref}}_{b,t}|,\qquad \text{body\_z\_err\_p95\_m}=P_{95}(z\_err),\qquad \text{PASS}\le 0.20$$

- **`contact`** — 在参考 mask 标记"应接触"的帧里，sim 实际发生 MuJoCo 手-物接触的比例（`rp` = raw physical contact 布尔，`m` = mask）。
  $$\text{contact\_in\_mask}=\frac{|\{t\in m:\text{sim 有手物接触}\}|}{|m|},\qquad \text{PASS}\ge 0.50$$
  注：这里用**原始物理接触**（不是 clean 3mm），因为下游 RL 主要看接触门是否激活，对穿透深度不敏感（见 METRICS_STANDARD.md「RL-Safe Contact Gate」）。

- **`release`** — 参考最后一次接触之后的"释放窗"内，sim 仍误接触（clean 3mm）的比例；若无释放窗则该门 **N/A（自动 PASS）**。释放窗判定见 `release_window_info()`（eval_E194_G1_expansion.py:129）。
  $$\text{release\_false\_3mm}=\frac{|\{t\in\text{release}:c_{3mm,t}\}|}{|\text{release}|},\qquad \text{PASS}\le 0.30\ \lor\ \text{N/A}$$

- **`hand_penetration`** — 手-物 MuJoCo 接触深度超过 3mm 的帧占比。
  $$\text{PASS}=\Big(\text{frac}\big(\text{contact.dist}<-0.003\big)\le 0.30\Big)$$

- **`lower_body`** — 腿到物体的表面有向距离 < 0（穿透）的帧占比。
  $$\text{leg\_penetration\_frac}=\text{frac}(d_{\text{leg-obj}}<0),\qquad \text{PASS}\le 0.10$$

### 1.2 Tracking 6-gate（**全部依赖 GT**）

来源：SPIDER Table-4 风格，逐帧对固定 kinematic 参考 `trajectory_kinematic.npz` 做 MuJoCo FK 后求误差均值。实现 `_table4_tracking_metrics()`（core_metrics.py:619）。

| Gate | 底层指标 | 阈值 | 依赖 GT | 代码 |
|------|----------|------|:------:|------|
| `root_pos` | `track_root_pos_err_cm_mean` | ≤ 20 cm | 是 | core_metrics.py:719 |
| `root_ori` | `track_root_ori_err_deg_mean` | ≤ 20 deg | 是 | core_metrics.py:721 |
| `hand_pos` | `track_eef_pos_err_cm_mean` | ≤ 20 cm | 是 | core_metrics.py:715 |
| `hand_ori` | `track_eef_ori_err_deg_mean` | ≤ 20 deg | 是 | core_metrics.py:717 |
| `object_pos` | `track_obj_pos_err_cm_mean` | ≤ 20 cm | 是 | core_metrics.py:723 |
| `object_ori` | `track_obj_ori_err_deg_mean` | ≤ 10 deg | 是 | core_metrics.py:727 |

**公式**（$\bar{}$ 为对帧取均值；位置转 cm，朝向为四元数测地角 deg）

$$\text{root\_pos}=100\cdot\overline{\lVert p^{\text{sim}}_{\text{pelvis}}-p^{\text{ref}}_{\text{pelvis}}\rVert}\qquad \text{root\_ori}=\overline{\angle(q^{\text{sim}}_{\text{pelvis}},q^{\text{ref}}_{\text{pelvis}})}$$

$$\text{hand\_pos}=100\cdot\overline{\tfrac{1}{2}\!\!\sum_{e\in\{L,R\}}\!\!\lVert p^{\text{sim}}_e-p^{\text{ref}}_e\rVert}\qquad \text{hand\_ori}=\overline{\tfrac{1}{2}\!\!\sum_{e}\angle(q^{\text{sim}}_e,q^{\text{ref}}_e)}$$

$$\text{object\_pos}=100\cdot\overline{\lVert p^{\text{sim}}_{\text{obj}}-p^{\text{ref}}_{\text{obj}}\rVert}\qquad \text{object\_ori}=\overline{\angle(q^{\text{sim}}_{\text{obj}},q^{\text{ref}}_{\text{obj}})}$$

- 末端执行器 $e$ = `left_wrist_yaw_link` / `right_wrist_yaw_link`。
- 关节误差另有 `track_joint_err_deg_mean` = `deg(mean(|qpos[7:nq_robot] − ref|))`（非 gate，仅诊断）。
- 参考 object 位姿可来自 scene qpos 布局（nq 一致）或机器人前缀后的 world freejoint 位姿（nq=43）。

---

## 2. 运动健康 / 平滑度 / 脚部指标（连续量，**均不依赖 GT**）

全部只用 sim 自身轨迹（qpos + FK），无参考。速度/加速度/加加速度采用**前向差分 × fps 的对应幂**。代码：`motion_health.py`。

### 2.1 qpos 层（全关节向量）— `qpos_kinematic_health()`（motion_health.py:87）

仅取真实机器人 qpos 通道（npz `qpos` 为 `(T,2,nq)`，channel 0 = sim）。

| 指标 | 公式 | 含义 |
|------|------|------|
| `qpos_speed_l2_p95` | $P_{95}\big(\lVert\Delta q_t\rVert_2\cdot \text{fps}\big)$ | 关节角速度幅度 p95 |
| `qpos_accel_l2_p95` | $P_{95}\big(\lVert\Delta^2 q_t\rVert_2\cdot \text{fps}^2\big)$ | 关节角加速度 p95 |
| `qpos_jerk_l2_p95` | $P_{95}\big(\lVert\Delta^3 q_t\rVert_2\cdot \text{fps}^3\big)$ | 关节 jerk p95（平滑度） |

### 2.2 body 层（FK 世界坐标）— `body_motion_health()`（motion_health.py:162）

监测 body 集见 `TRACK_BODY_NAMES / ANKLE_BODY_NAMES / WRIST_BODY_NAMES`；object = scene 中名为 `object` 的 body 质心。逐帧灌 qpos、`mj_forward` 取 `xpos`。差分公式 $v_t=\lVert p_{t+1}-p_t\rVert_2\cdot\text{fps}$。

| 指标 | 公式/取法 | 含义 | 代码 |
|------|-----------|------|------|
| **`obj_speed_max`** | $\max_t \lVert\Delta p^{\text{obj}}_t\rVert_2\cdot\text{fps}$ | 物体质心平动速度峰值 (m/s)，抖动/飞出探测 | motion_health.py:235,253 |
| `trackbody_speed_max` | $\max$ 躯干 body 速度 | 躯干最大速度 | :229,238 |
| `ankle_speed_max` | $\max$ 踝速度 | 踝最大速度 | :230,239 |
| `wrist_speed_max` | $\max$ 腕速度 | 腕最大速度 | :240 |
| `trackbody_acc_max` | $\max \lVert\Delta^2 p\rVert\cdot\text{fps}^2$ | 躯干最大加速度 | :231,245 |
| `ankle_acc_max` | 同上（踝） | 踝最大加速度 | :232,246 |
| `trackbody_jerk_p95` | $P_{95}(\lVert\Delta^3 p\rVert\cdot\text{fps}^3)$ | 躯干 jerk p95（平滑度） | :233,247 |
| `ankle_jerk_p95` | 同上（踝） | 踝 jerk p95 | :234,250 |

> 说明：`obj_speed_max` 只含平动（质心位移），不含旋转。

### 2.3 脚部 — `foot_motion_metrics()`（motion_health.py:142）

"接地"判定：每只脚取其 z 的 5 百分位作地面基准 `ground_z`，`grounded = ankle_z ≤ ground_z + 0.05m`。对每段连续接地区间（≥2 帧）：

| 指标 | 公式 | 含义 | 依赖 GT |
|------|------|------|:------:|
| `foot_slip_max_m` | $\max_{\text{段}}\max_t\lVert xy_t - xy_{\text{start}}\rVert$ | 接地脚水平滑移峰值 (m)，脚滑 | 否 |
| `foot_ground_dev_max_m` | $\max_{\text{段}}\max_t\lvert z_t-\text{ground\_z}\rvert$ | 接地期竖直偏离峰值 | 否 |
| `foot_grounded_frame_frac` | $\text{mean}_t(\exists\text{脚接地})$ | 有脚接地的帧占比 | 否 |

---

## 3. 接触/穿透诊断指标（部分依赖 mask）

`evaluate_sequence` 还输出一批接触/穿透诊断量（非 gate，用于质量-穿透权衡分析）。清洁阈值 `PHYSICS_CONTACT_THRESHOLDS_M = (0.003, 0.005)`（core_metrics.py:252）。

| 指标 | 公式 | 依赖 GT | 代码 |
|------|------|:------:|------|
| `hand_object_physics_contact_frac` | frac(任一手有 MuJoCo 手物接触) | 否 | :1377 |
| `hand_object_physics_contact_3mm_frac` | frac(clean 接触, dist ≥ −3mm) | 否 | :1379 |
| `hand_object_physics_penetration_3mm_frame_frac` | frac(dist < −3mm) | 否 | :1381 |
| `hand_object_physics_penetration_5mm_frame_frac` | frac(dist < −5mm) | 否 | :1384 |
| `hand_geom_penetration_2mm/5mm_frac` | frac(手-物几何面距 < −2/−5mm) | 否 | :1374,1375 |
| `leg_near_2cm_frac` | frac(腿-物面距 < 2cm) | 否 | :1419 |
| `leg_object_physics_contact_frac` | frac(腿-物 MuJoCo 接触) | 否 | :1421 |
| `hand_object_physics_contact_{3,5}mm_in_mask_frac` | mask 内 clean 接触占比 | **是** | :1116,1117 |
| `hand_object_{approach,release}_false_contact_3mm_frac` | 接近/释放窗假接触 | **是** | :1141,1146 |
| `ref_contact_frac` | mask 中标接触的帧占比（参考量本身） | **是** | :1102 |

**恒等式**（同阈值下）：`contact_{3,5}mm_frac + penetration_{3,5}mm_frame_frac == physics_contact_frac`（见 METRICS_STANDARD.md）。

---

## 4. GT 依赖速查

| 类别 | 指标 | 依赖 |
|------|------|------|
| **需 kinematic 参考** | 6 个 tracking gate、`body_z`、以及所有 `track_*` 诊断 | `trajectory_kinematic.npz` |
| **需 contact mask** | `contact`、`release` gate、所有 `*_in_mask_*` / `*false_contact*` / `ref_contact_frac` | `raw_contact_mask_3cm.npz` |
| **不需要参考（纯物理/自监督）** | `fall`、`hand_penetration`、`lower_body`、`obj_speed_max`、所有 `*_speed/acc/jerk*`、`qpos_*`、脚滑、几何/物理穿透（非 mask 版） | sim 自身轨迹 |

---

## 5. 阈值来源与冻结说明

- 12 门阈值由 `GATE_THRESHOLDS`（eval_E194_G1_expansion.py:47）与物理子集常量（eval_E172_box004.py:33-39）冻结：`BODY_Z_MAX=0.20`、`CONTACT_MIN=0.50`、`RELEASE_MAX=0.30`、`HAND_PEN_MAX=0.30`、`LEG_PEN_MAX=0.10`、tracking 位置/朝向 20/20/20/20/20/**10**。
- `fall_pelvis_z_m=0.45`、清洁接触阈 3mm/5mm 等在 `core_metrics.py` 的 config dataclass（:28,:252）。
- 新增 evaluator 必须 `import` `eval.core.core_metrics` 的字段组，不得本地重定义指标（见 METRICS_STANDARD.md），以保证跨实验可比。
