# CORE4D 评估指标体系

统一的评估标准，综合 SPIDER (Pan et al., 2026)、DynaRetarget (Dhedin et al., 2026)、OmniRetarget (Yang et al., 2025) 三篇论文的评测方法。

## 指标总览

| 类别 | 指标 | 单位 | 方向 | 来源 |
|------|------|------|------|------|
| Body Tracking | MPKPE | cm | ↓ | DynaRetarget Table V |
| Body Tracking | Joint Error | deg | ↓ | SPIDER Table 4 |
| Body Tracking | EEF Pos Error | cm | ↓ | SPIDER Table 4 |
| Body Tracking | EEF Ori Error | deg | ↓ | SPIDER Table 4 |
| Root Tracking | Root Pos Error | cm | ↓ | SPIDER Table 4 |
| Root Tracking | Root Ori Error | deg | ↓ | SPIDER Table 4 |
| Object Tracking | Obj Pos Error | cm | ↓ | SPIDER Table 4, DynaRetarget Table V |
| Object Tracking | Obj Ori Error | deg | ↓ | SPIDER Table 4, DynaRetarget Table V |
| Physical Plausibility | Stability (>0.60m) | % | ↑ | 本任务特定 |
| Physical Plausibility | Penetration Duration | % | ↓ | OmniRetarget Table II |
| Physical Plausibility | Foot Skating Duration | % | ↓ | OmniRetarget Table II |
| Interaction Quality | Contact Preservation | % | ↑ | OmniRetarget Table II |
| Interaction Quality | Sustained Contact | s | ↑ | 本任务特定 |
| Smoothness | Joint Acceleration | rad/s² | ↓ | 本任务特定 |

---

## A. Body Tracking 指标

### A1. MPKPE — Mean Per-Keypoint Position Error

**来源**: DynaRetarget Table V

**定义**: 所有机器人 body 的全局位置误差均值

$$\text{MPKPE} = \frac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \| \mathbf{p}^{\text{sim}}_{k,t} - \mathbf{p}^{\text{ref}}_{k,t} \|_2$$

- $\mathbf{p}^{\text{sim}}_{k,t}$: sim 中 body $k$ 在时刻 $t$ 的 xpos (FK 结果)
- $\mathbf{p}^{\text{ref}}_{k,t}$: ref 中 body $k$ 在时刻 $t$ 的 xpos
- $K$: 机器人 body 数量 (G1: bodies 1-30, K=30)
- $T$: 总帧数

**核心实现**:
```python
# workspace/core4d/scripts/eval/eval_e035_comprehensive.py
for bid in robot_body_ids:  # [1, 2, ..., 30]
    err = np.linalg.norm(data_sim.xpos[bid] - data_ref.xpos[bid])
    kp_errors.append(err)
mpkpe_t = np.mean(kp_errors)  # per-frame mean
```

### A2. Joint Error — 关节角误差

**来源**: SPIDER Table 4

**定义**: 所有 robot 关节角的绝对误差均值

$$\text{JointErr} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} | q^{\text{sim}}_{j,t} - q^{\text{ref}}_{j,t} |$$

- $q_{j,t}$: 关节 $j$ 在时刻 $t$ 的 qpos 值 (rad)
- $J$: 关节数量 (G1: qpos[7:36], J=29)
- 结果转换为 degrees

**核心实现**:
```python
j_sim = qpos_sim[t, 7:36]  # skip 7 freejoint DOF
j_ref = qpos_ref[t, 7:36]
joint_err_t = np.mean(np.abs(j_sim - j_ref))  # rad, convert to deg
```

### A3. EEF Pos Error — 末端执行器位置误差

**来源**: SPIDER Table 4 (Pos. Err.)

**定义**: 左右手腕 (wrist) 全局位置误差均值

$$\text{EEF\_Pos} = \frac{1}{T \cdot 2} \sum_{t=1}^{T} \sum_{e \in \{\text{L,R}\}} \| \mathbf{p}^{\text{sim}}_{e,t} - \mathbf{p}^{\text{ref}}_{e,t} \|_2$$

- $e$: left_wrist_yaw_link (id=23), right_wrist_yaw_link (id=30)

### A4. EEF Ori Error — 末端执行器朝向误差

**来源**: SPIDER Table 4 (Ori. Err.)

**定义**: 左右手腕四元数差的角度

$$\text{EEF\_Ori} = \frac{1}{T \cdot 2} \sum_{t=1}^{T} \sum_{e} \angle(\mathbf{q}^{\text{sim}}_{e,t}, \mathbf{q}^{\text{ref}}_{e,t})$$

**核心实现**:
```python
from scipy.spatial.transform import Rotation as R
r_sim = R.from_quat(xquat_sim_xyzw)
r_ref = R.from_quat(xquat_ref_xyzw)
angle_deg = np.degrees((r_sim.inv() * r_ref).magnitude())
```

---

## B. Root Tracking 指标

### B1. Root Pos Error — 骨盆全局位置误差

$$\text{RootPos} = \frac{1}{T} \sum_{t=1}^{T} \| \mathbf{p}^{\text{sim}}_{\text{pelvis},t} - \mathbf{p}^{\text{ref}}_{\text{pelvis},t} \|_2$$

### B2. Root Ori Error — 骨盆全局朝向误差

$$\text{RootOri} = \frac{1}{T} \sum_{t=1}^{T} \angle(\mathbf{q}^{\text{sim}}_{\text{pelvis},t}, \mathbf{q}^{\text{ref}}_{\text{pelvis},t})$$

---

## C. Object Tracking 指标

### C1. Obj Pos Error — 物体位置误差

**来源**: SPIDER Table 4, DynaRetarget Table V

$$\text{ObjPos} = \frac{1}{T} \sum_{t=1}^{T} \| \mathbf{p}^{\text{sim}}_{\text{obj},t} - \mathbf{p}^{\text{ref}}_{\text{obj},t} \|_2$$

### C2. Obj Ori Error — 物体朝向误差

$$\text{ObjOri} = \frac{1}{T} \sum_{t=1}^{T} \angle(\mathbf{q}^{\text{sim}}_{\text{obj},t}, \mathbf{q}^{\text{ref}}_{\text{obj},t})$$

**注意**: ref 使用 freejoint (scene.xml, nq=43) 做 FK; sim 使用 scene_act.xml (nq=42) 做 FK。需要分别加载对应模型。

---

## D. Physical Plausibility 指标

### D1. Stability — 骨盆高度稳定性

**来源**: 本任务特定 (参考 DynaRetarget 的 success rate 概念)

**定义**: pelvis xpos z > 阈值的帧比例。报告多阈值:

| 阈值 | 含义 |
|------|------|
| >0.70m | 正常站立 (G1 站立高度 ~0.78m) |
| >0.65m | 轻微蹲低 |
| >0.60m | 明显蹲低但未摔倒 |
| >0.55m | 严重前倾/摔倒边缘 |

**主要报告指标**: `>0.60m` 作为 "stable" 的判定阈值

附加: **最长不稳定段** (longest consecutive stretch with pelvis_z < 0.60m)

### D2. Penetration Duration — 穿透时间比

**来源**: OmniRetarget Table II

**定义**: 机器人 body 与地面/物体发生穿透的时间比例

$$\text{PenDur} = \frac{|\{t : \exists \text{body} \in \text{robot}, z_{\text{body},t} < 0 \text{ or } d_{\text{obj}} < 0 \}|}{T}$$

**简化实现** (检测脚穿透地面):
```python
foot_z = min(data_sim.xpos[left_ankle_id, 2], data_sim.xpos[right_ankle_id, 2])
penetration = foot_z < -0.01  # below ground with margin
```

### D3. Foot Skating — 脚滑时间比

**来源**: OmniRetarget Table II

**定义**: stance foot (接地脚) 有显著水平速度的时间比例

$$\text{SkDur} = \frac{|\{t : \text{foot on ground} \wedge \|\mathbf{v}^{xy}_{\text{foot},t}\| > v_{\text{thresh}} \}|}{|\{t : \text{foot on ground}\}|}$$

- $v_{\text{thresh}} = 0.1$ m/s (10 cm/s)
- "foot on ground": foot_z < 0.05m

**核心实现**:
```python
foot_z = data_sim.xpos[foot_id, 2]
if foot_z < 0.05:  # on ground
    foot_vel_xy = (foot_pos_t - foot_pos_t_prev) / dt
    skating = np.linalg.norm(foot_vel_xy[:2]) > 0.10
```

---

## E. Interaction Quality 指标

### E1. Contact Preservation — 接触保持率

**来源**: OmniRetarget Table II

**定义**: 在 ref 标记为"应该接触"的帧中，sim 实际接触的比例

$$\text{ContactPres} = \frac{|\{t : t \in T_{\text{desired}} \wedge d^{\text{surf}}_{t} < \delta \}|}{|T_{\text{desired}}|}$$

- $T_{\text{desired}}$: ref 中 hand-object 距离 < 0.15m 的帧集合 (标记为"应该接触")
- $d^{\text{surf}}_t$: sim 中 hand-to-object-surface 距离
- $\delta$: 接触阈值 (我们使用 0.10m)

**与 OmniRetarget 的区别**: OmniRetarget 用预标注的 contact label; 我们从 ref FK 推断 desired contact 帧。

### E2. Sustained Contact — 最长连续接触

**定义**: 连续 hand-surface 距离 < 0.10m 的最长时段

$$\text{SusCont} = \max_{\text{runs}} |\{t_{\text{start}}:t_{\text{end}} : \forall t \in [t_s, t_e], d^{\text{surf}}_t < 0.10 \}|$$

### E3. Mean Surface Distance — 平均手-物体表面距离

$$\text{MeanSurf} = \frac{1}{T} \sum_{t=1}^{T} \min_{e \in \{L,R\}} d^{\text{surf}}_{e,t}$$

**Surface distance 计算** (axis-aligned bbox approximation):
```python
obj_pos = data_sim.xpos[obj_id]
obj_mat = data_sim.xmat[obj_id].reshape(3, 3)
local = obj_mat.T @ (hand_pos - obj_pos)  # hand in object frame
clamped = np.clip(local, -half_ext, half_ext)
surf_dist = np.linalg.norm(local - clamped)
```

---

## F. Smoothness 指标

### F1. Joint Acceleration — 关节加速度

**来源**: DynaRetarget Table IV 中的 Action Rate regularization

**定义**: 关节角二阶差分的绝对值均值 (proxy for jerk)

$$\text{JointAcc} = \frac{1}{T-2} \sum_{t=2}^{T} \frac{1}{J} \sum_{j=1}^{J} \left| \frac{q_{j,t} - 2q_{j,t-1} + q_{j,t-2}}{\Delta t^2} \right|$$

---

## 基线对比表

| 指标 | HDMI R013 (suitcase) | E032a (desk005) | E035 (desk005) | SPIDER论文(OMOMO) | DynaRetarget |
|------|---------------------|-----------------|----------------|-----------------|-------------|
| **MPKPE (cm)** | **7.72** | 31.35 | 47.92 | — | 3.57 |
| **Joint Err (deg)** | **3.22** | 11.71 | 10.26 | 0.83 | — |
| **EEF Pos (cm)** | **7.88** | 38.48 | 48.36 | 0.20 | — |
| **EEF Ori (deg)** | **5.46** | — | 55.19 | 0.17 | — |
| **Root Pos (cm)** | **7.17** | 26.49 | 47.43 | — | — |
| **Root Ori (deg)** | **2.30** | — | 18.71 | — | — |
| **Obj Pos (cm)** | **5.39** | 16.37 | 22.55 | 0.18 | 8.81 |
| **Obj Ori (deg)** | **4.28** | 6.14 | 14.01 | 0.06 | 6.3 |
| **Stability >0.60m** | 84.8% | 77.6% | **100%** | — | — |
| **Pelvis z min** | 0.300m | 0.223m | **0.657m** | — | — |

**注**: SPIDER/DynaRetarget 论文数值极低因为他们评的是 short-horizon 精确操控 (手部dexterous)，非 full-body locomotion+manipulation。HDMI R013 是最相关的同任务基线。

---

## 评测脚本

主脚本: `workspace/core4d/scripts/eval/eval_e035_comprehensive.py`

```bash
# 用法 (internal ref from saved npz):
uv run workspace/core4d/scripts/eval/eval_e035_comprehensive.py <sim.npz> "" <task>

# 用法 (external kinematic ref):
uv run workspace/core4d/scripts/eval/eval_e035_comprehensive.py <sim.npz> <ref.npz> <task>
```

需要: sim npz 格式 `qpos: (T, 2, nq)` channel 0=sim, 1=ref; 或 `qpos: (T, nq)` + 外部 ref。

---

## 评测标准 (通过/不通过)

基于 HDMI R013 baseline + OmniRetarget 标准:

| 指标 | 合格线 | 优秀线 | 依据 |
|------|-------|-------|------|
| MPKPE | < 15 cm | < 8 cm | HDMI baseline = 7.7cm |
| Joint Err | < 8 deg | < 4 deg | HDMI baseline = 3.2deg |
| Obj Pos | < 15 cm | < 6 cm | HDMI baseline = 5.4cm |
| Stability >0.60m | > 90% | > 95% | 无摔倒 |
| Contact Preservation | > 80% | > 95% | OmniRetarget = 96% |
| Penetration Duration | < 5% | < 1% | OmniRetarget = 0% |
| Foot Skating Duration | < 10% | < 2% | OmniRetarget = 0% |

**当前 E035 desk005 状态**:
- MPKPE: 47.9cm ❌ (合格线 15cm)
- Joint Err: 10.3deg ❌ (合格线 8deg)
- Stability: 100% ✅
- Contact: 94% ✅
