# E068 Results — MJWP init drift 诊断修正: 真问题在 first committed CEM ctrl

**日期**: 2026-05-14
**实验域 (exp_name)**: `core4d`
**对应Plan**: `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
**前置**: `workspace/core4d/log/87_R4_HDMI_diagnosis_init_pose_bug.md`

## 1. 背景

log 87 发现 MJWP box023 结果在 t=0.033s 已经偏离 ref 约 22 deg yaw，而 HDMI 同 case 基本对齐。E068 的目标是隔离这个 drift 是否来自 `setup_env()` 的 init `mj_step()` / `mjwarp.put_data()`，还是来自第一轮 MPC commit 的控制输入。

## 2. 诊断结果

### 2.1 Init path 对比

结果路径：

| 类型 | 路径 |
|------|------|
| Init 诊断 JSON | `workspace/core4d/results/E068/init_drift_diagnosis.json` |
| Init 诊断 CSV | `workspace/core4d/results/E068/init_drift_diagnosis.csv` |
| 运行日志 | `logs/E068/diagnose_init_drift.log` |

| 路径 | yaw err | quat err | qpos max diff | 结论 |
|------|---------|----------|---------------|------|
| `ref_qpos0` | 0.000 deg | 0.000 deg | 0.00000 | ref 正常 |
| `cpu_forward` | 0.000 deg | 0.000 deg | 0.00000 | 直接 `mj_forward` 完全对齐 |
| `cpu_after_one_mj_step` | 0.220 deg | 0.225 deg | 0.00439 | 单个 init `mj_step` 只造成极小 drift |
| `warp_manual_put_data_from_cpu_step` | 0.220 deg | 0.225 deg | 0.00439 | `mjwarp.put_data` 本身未放大 drift |
| `warp_after_current_setup_env` | N/A | N/A | N/A | 本机无 CUDA，无法 capture graph |

**结论**: log 87 的 22 deg yaw drift 不是 init `mj_step()` 单独造成。`mj_forward` 完全对齐，`mj_step` 只偏 0.22 deg，`put_data` 不放大偏差。

### 2.2 真实结果 first committed step 对比

结果路径：

| 类型 | 路径 |
|------|------|
| First-step qpos trace | `workspace/core4d/results/E068/first_commit_trace.csv` |
| First-step ctrl delta | `workspace/core4d/results/E068/first_ctrl_delta.csv` |
| 脚本 | `workspace/core4d/scripts/debug/analyze_E068_first_commit.py` |

E062/E063/E067 真实 `.npz` 的前 0.17s 复核：

| Source | t=0.017s yaw err | t=0.033s yaw err | t=0.100s yaw err | 结论 |
|--------|------------------|------------------|------------------|------|
| E062 box023 | 12.34 deg | 22.00 deg | 50.54 deg | 复现 log 87 |
| E063 box023 | 12.34 deg | 22.00 deg | 50.54 deg | stability 改动不影响初始 drift |
| E067N box023 | 12.35 deg | 21.99 deg | 50.79 deg | narrow partition 也同源起漂 |

第一帧 committed ctrl 与 `ctrl_ref` 对比：

| Source | robot ctrl max diff | object ctrl max diff | 最大偏差 actuator |
|--------|---------------------|----------------------|-------------------|
| E062 | 1.560 rad | 0.010 | `left_hip_roll_joint` |
| E063 | 1.559 rad | 0.010 | `left_hip_roll_joint` |
| E065A | 1.559 rad | 0.010 | `left_hip_roll_joint` |
| E066A | 1.559 rad | 0.010 | `left_hip_roll_joint` |
| E067N | 1.559 rad | 0.010 | `left_hip_roll_joint` |

关键 actuator 表（E062 t=0.017s）：

| actuator | q0/ref | committed ctrl | delta |
|----------|--------|----------------|-------|
| `left_hip_roll_joint` | +0.002 | -1.558 | -1.560 |
| `right_shoulder_yaw_joint` | -0.762 | +0.685 | +1.447 |
| `left_elbow_joint` | +0.966 | -0.188 | -1.155 |
| `left_hip_pitch_joint` | +0.142 | -0.834 | -0.976 |

**结论**: 真实 drift 出现在第一轮优化后的 commit step。CEM 在 t=0 就把 robot actuator target 推离 ref 1.5 rad 级别，而 object ctrl 几乎仍跟 ref。此前 "init pose bug" 判断过窄，应修正为 **first MPC tick 没有 ref-control warmup / optimizer immediate override**。

## 3. 可视化

已有视频/关键帧：

| Source | 路径 |
|--------|------|
| E062 video | `workspace/core4d/results/E062/E062_box023_sphere_autopalm.mp4` |
| E063 video | `workspace/core4d/results/E063/E063_box023.mp4` |
| E067N video | `workspace/core4d/results/E067/E067N_box023.mp4` |
| E062 keyframe | `workspace/core4d/results/E062/keyframes/box023_t0.5s.jpg` |
| E063 keyframe | `workspace/core4d/results/E063/keyframes/E063_box023_kf3_t1.67s.jpg` |
| E067N keyframe | `workspace/core4d/results/E067/keyframes/E067N_box023_kf3_t0.80s.jpg` |

**实际观察**:

- E062 t=0.5s: ref 是双脚接地弯腰接近 box；sim 已经明显侧转，单脚/斜身接近，box 旁出现不自然 contact marker。
- E063 t=1.67s: ref 在搬箱前进；sim 是长步 lunge，后脚大幅后伸，姿态不是稳定搬运。
- E067N t=0.80s: sim 变成手倒立/倒栽姿态，右腿高举，验证 narrow body partition 让 CEM 找到更极端局部最优。

## 4. Claims 验证

| Claim | 结果 |
|-------|------|
| C1: 当前 MJWP init drift 可复现 | **部分通过** — 真实 `.npz` 可复现 12/22/50 deg 早期 drift；但不是 init path 单独造成 |
| C2: `mj_forward` init 与 ref 对齐 | **通过** — yaw/quat/qpos diff 全 0 |
| C3: drift 来源可定位 | **通过** — `mj_step` 仅 0.22 deg，first committed ctrl robot diff 1.56 rad |
| C4: 最小修复候选明确 | **通过** — 下一步应验证 `warmup_steps` / first tick ref ctrl，不应先改 `mjwp_init_mode=forward` |
| C5: box025 regression | **未执行** — E068 是诊断，不做修复验证 |
| C6: box023 pre-contact lunge 改善 | **未执行** — 下一步 E069 验证 |

## 5. 结论修正

log 87 的核心现象成立：MJWP 在最早期就偏离 ref，且 HDMI 不偏。但根因从 "init pose bug" 修正为：

> 第一轮 MPC 没有 warmup，CEM 在 t=0 立即提交远离 ref 的 robot ctrl。由于初始站姿还未 settle，1.5 rad 级关节 target 造成 pelvis yaw 快速旋转，随后进入 lunge / handstand basin。

这也解释了为什么 E065/E066/E067 的 reward/dynamics/partition 改动都无法消除最早期 drift：它们没有阻止第一个 MPC tick 直接覆盖 `ctrl_ref`。

## 6. 下一步

进入 E069: **first-tick ref-control warmup 验证**。

推荐两个 box023-only 变体，不改代码，只用已有 `warmup_steps`：

| 变体 | 配置 | 目标 |
|------|------|------|
| E069-W02 | `warmup_steps: 0.20` | 前 0.2s 强制 ref ctrl，验证 yaw drift 是否被压住 |
| E069-W05 | `warmup_steps: 0.50` | 更长 settle，验证 lunge 是否延后/消失 |

判据：

- 若 W02/W05 的前 0.2s yaw err < 5 deg 且 B1 < 0.10m，确认 first tick CEM override 是主因。
- 若 warmup 结束后马上 lunge，下一步应做 "CEM delta clamp / trust region around ref ctrl"，而不是 reward weight 调参。
- 若 warmup 无效，再回到 actuator force trace / contact impulse trace。

## 7. 改动文件

| 类型 | 路径 |
|------|------|
| Plan | `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md` |
| Init 诊断脚本 | `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py` |
| First commit 分析脚本 | `workspace/core4d/scripts/debug/analyze_E068_first_commit.py` |
| 入口脚本 | `workspace/core4d/scripts/train/train_E068.sh` |
| 结果 | `workspace/core4d/results/E068/init_drift_diagnosis.{json,csv}` |
| 结果 | `workspace/core4d/results/E068/first_commit_trace.csv` |
| 结果 | `workspace/core4d/results/E068/first_ctrl_delta.csv` |
