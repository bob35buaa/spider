# E167 实验计划：Holosoma z-only body tracking 对齐（7 case × 3 arms）

日期：2026-06-20
分支：`experiment/E161-surface-release-ablation`（exp_name: core4d，沿用，不新建分支）
状态：planning only。本轮只调研和写计划，不改训练代码、不启动 CEM/RL。

---

## Context

用户指出 `CORE4D_E163_8CASE_E165D_DOWNSTREAM_ANALYSIS_CN.md` §7 的关键差异：

- SUGAR refiner 的 `ee_body_pos` 是踝+腕 **3D norm**，阈值 `0.30m`。
- Holosoma WBT V4.3 的 `bad_motion_body_pos` 使用 `BadTrackingZOnly`，踝+腕只检查 **z 轴误差**，阈值 `0.25m`。
- Holosoma 的 object position 仍是 3D norm，因此本计划只对 body/root-style tracking 口径做 z-only 对齐，不改 object tracking。

E166 的 A/B1/B2 调研结论显示，旧方案没有严格 z-only：

| E166 组件 | 代码证据 | 是否限制 xy |
|---|---|---|
| `A` 的 `foot_slip` | `spider/optimizers/sampling.py:114-123` 对 grounded ankle 的 `pos[..., :2]` 算 XY speed | **是** |
| `A` 的 ankle extra weight | `spider/simulators/mjwp.py:794-813` 复用 `_local_pos_tracking`，3D local body position | **是，隐含 3D** |
| `A` 的 `foot_ground` | `sampling.py:125-133` 只算 z deviation | 否，仅 z |
| `B1` smooth | `sampling.py:68-85` 对 ankle body pos 的 accel/jerk 做 3D norm | **是，隐含 3D** |
| `B2` postSmooth | `spider/postprocess/smooth_handoff.py` 默认平滑整条 `qpos/qvel/ctrl` 等浮点数组 | **是，非轴选择** |

所以 E167 不能直接复用 E166 A/B1/B2。E167 的核心假设是：**上游 SPIDER 不应为了对齐 Holosoma 而惩罚/平滑 lateral XY body tracking；只把 z 轴 body executability 做成 handoff 前约束与平滑目标。**

---

## Scope

### Cases

做 clean8 中除 `box026_139_p1` 外的 7 个 case：

| case | 来源/备注 |
|---|---|
| `box023_person2` | pathology/self-collision 风险 case，仍纳入以验证 z-only 是否无法解决接触病因 |
| `box021_029_p2` | E166 A_B2 downstream 成功 case，用于防回归 |
| `box021_035_p1` | E166 A_B2 downstream 回归 case，重点看 z-only 是否避免旧 B2 过约束 |
| `box021_035_p2` | E166 三 case 主战场，脚/抖动双坏 |
| `box004_082_p1` | E166 三 case 主战场，动力学最抖 |
| `box004_083_p1` | remaining4 中未被 A_B2 救回 |
| `box004_083_p2` | 原三 case，脚坏但较平滑，用来判别 A_z 是否独立有效 |

明确排除：

- `box026_139_p1`：按用户要求不补、不跑。

### Arms

本轮只做三组用户指定实验：

| arm | 名称 | 新 CEM | 新 downstream | z-only 定义 |
|---|---|:--:|:--:|---|
| `E167A` | `A_zOnlyBody` | yes | yes | 只惩罚 Holosoma body set 的 z 误差/着地 z，不启用 foot-slip XY，不启用 3D ankle extra-weight |
| `E167A+B1` | `A_zOnlyBody + B1_zOnlyCemSmooth` | yes | yes | 在 A_z 基础上，只对 body z 序列算 accel/jerk |
| `E167A+B2` | `A_zOnlyBody + B2_zOnlyHandoffSmooth` | no extra CEM; reuse `E167A` source | yes | 对 handoff/SUGAR reference 的 z 通道做后处理，必须证明 xy 完全不变 |

对照不重跑：

- E163 narrowSurfaceBand baseline。
- E166 A / A_B2_postSmooth 作为历史对照。

---

## Claims

| Claim | 最低证据 |
|---|---|
| C0-axis-audit | E167 三个 arm 的 manifest/config/report 明确显示无 `foot_slip_enabled`、无 `local_frame_ankle_weight != 1` 的 3D ankle tracking、无 3D smooth norm；B2_z 输出的 monitored body/root xy 与输入逐元素一致或最大差 < `1e-6` |
| C1-z-gate | E167A 相对 E163/E166A 降低 Holosoma-style `bad_motion_body_pos_z` 峰值/越阈帧；7 case 中至少 2 个 case 的 z-only hard-gate 状态改善 |
| C2-B1-zsmooth | E167A+B1 降低 monitored body z accel/jerk P95，且不显著改变 xy drift 指标；接触/穿透相对 E167A 不退化超过 5% |
| C3-B2-zsmooth | E167A+B2 在不改 xy 的前提下降低 z jerk/accel；若 body-z postprocess 与 joint/root reference 不一致超过阈值，B2_z 判为不可推广而不是强行训练 |
| C4-contact-no-regression | 三个 arm 的 raw contact、clean3 contact、3mm penetration 不比 E163 baseline 退化超过 5%；否则不推广 |
| C5-downstream-diagnosis | downstream failed windows 必须分解为 z-exceed / xy-only / object / anchor；若 SUGAR 仍因 3D `ee_body_pos` 的 xy-only 失败，不算 Holosoma-z 失败 |
| C6-performance | 以 Holosoma-z offline gate 为主指标，E167A+B2 或 E167A+B1 至少在 7 case 中 2 个 case 给出可解释改善；SUGAR 3D success 只作为保守副指标 |

---

## Design

### 1. E167A：z-only body/foot constraint

新增默认关闭 config，第一版 body set 与 Holosoma/SUGAR `ee_body_pos` 对齐：

```python
e167_body_z_enabled: bool = False
e167_body_z_names: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
e167_body_z_weight: float = 0.0
e167_body_z_threshold_m: float = 0.25
e167_ground_z_enabled: bool = False
e167_ground_z_weight: float = 0.0
e167_ground_contact_height_m: float = 0.05
```

实现原则：

- `mjwp.py` 输出 monitored body sim/ref z 序列到 rollout info。
- `sampling.py` 聚合 z error、z over-threshold、grounded ankle z deviation。
- 不使用 `pos[..., :2]`，不计算 XY speed。
- 不使用 E166 的 `local_frame_ankle_weight=2.0`，因为它是 3D local position reward。
- 不修改 object 3D tracking；Holosoma 也保留 object 3D norm。

### 2. E167A+B1：z-only CEM smooth

扩展 E166 `cem_smooth_*` 为 axis-aware：

```python
cem_smooth_axis: str = "xyz"  # existing behavior
```

E167 设置：

```yaml
cem_smooth_enabled: true
cem_smooth_axis: z
cem_smooth_body_names:
  - left_ankle_roll_link
  - right_ankle_roll_link
  - left_wrist_yaw_link
  - right_wrist_yaw_link
```

实现原则：

- accel/jerk 只对 `pos[..., 2]` 做差分。
- report 同时输出 `z_accel_p95/z_jerk_p95` 和 `xy_speed/xy_drift`，后者只诊断不进 reward。
- 与 E166 B1 的 3D norm 分开命名，避免误用旧结果。

### 3. E167A+B2：z-only handoff postprocess

旧 `smooth_handoff.py` 不能原样复用，因为它默认平滑整条 `qpos/qvel/ctrl`，会改变 XY / 关节 / 接触相关量。E167 B2 需要新模式：

1. 优先做 **SUGAR reference 层 body-z postprocess**：对 `robot_50hz.npz` 中 monitored `body_pos_w[..., 2]` 及必要 root z 做 Savitzky-Golay；保留 `[..., 0:2]`、object、contact labels、joint pos 不变。
2. 生成 `zonly_smooth_report.json`，强制写：
   - `xy_max_abs_delta = 0`（或 < `1e-6`）
   - `body_z_jerk_delta`
   - `joint_body_consistency_error`（post body z 与 joint/root FK 参考的一致性诊断）
3. 若 consistency error 超过预设阈值（初始 `0.05m`），不启动 B2_z downstream，先回到 CEM-side B1_z。

这使 `E167A+B2` 的解释变成：“只改变消费端 body-z reference 的时间平滑，不碰 lateral reference”。它与 E166 B2 的 qpos-wide smoothing 不是同一个处理，结果不能混用。

---

## Execution Plan

### Phase 0 — implementation preflight（不训练）

要先写脚本但只跑 smoke/preflight：

| 文件 | 目的 |
|---|---|
| `workspace/core4d/scripts/experiments/E167/build_zonly_manifest.py` | 生成 7 case × 3 arms manifest；检查无 box026 |
| `workspace/core4d/scripts/eval/runners/eval_E167_zonly_axis_audit.py` | 读取 config/report，验证无 XY penalty 与 B2_z xy invariance |
| `workspace/core4d/scripts/eval/runners/eval_E167_holosoma_zgate.py` | 计算 Holosoma-style z-only gate、SUGAR 3D gate、xy-only failure split |
| `workspace/core4d/scripts/eval/wrappers/eval_E167_zonly.sh` | eval wrapper |

Phase 0 出口：

- manifest rows = 21，cases = 7，arms = 3，`box026` rows = 0。
- axis audit pass。
- smoke CEM 只允许短跑 1 case × 1 arm，不能启动 full。

### Phase 1 — SPIDER/CEM full（14 new CEM + 7 postprocess）

计算量：

- `E167A`: 7 new CEM。
- `E167A+B1`: 7 new CEM。
- `E167A+B2`: 复用 `E167A` CEM，7 CPU/SUGAR-reference z-only postprocess。

推荐 GPU split：

| worker | cases |
|---|---|
| local GPU0 | `box021_035_p2`, `box021_035_p1` |
| remote GPU0 | `box004_082_p1`, `box004_083_p1`, `box004_083_p2` |
| remote GPU1 | `box023_person2`, `box021_029_p2` |

训练前脚本必须调用 scene snapshot：

```bash
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E167 \
  box023_person2 box021_029_p2 box021_035_p1 box021_035_p2 \
  box004_082_p1 box004_083_p1 box004_083_p2
```

### Phase 2 — export + downstream

Primary downstream interpretation is Holosoma-aligned z-only. Since current automated refiner infrastructure is SUGAR and SUGAR still has 3D `ee_body_pos`, every SUGAR failure must be split:

- `z_exceed`: would fail Holosoma-z too.
- `xy_only`: SUGAR 3D fail, Holosoma-z would not fail.
- `object/anchor`: separate, not fixed by E167.

Planned scripts:

| 文件 | 目的 |
|---|---|
| `workspace/core4d/scripts/experiments/E167/export_zonly_rl_handoff.py` | Export E167 arms into SUGAR/Holosoma-compatible input folders |
| `workspace/core4d/scripts/launch/active/run_E167_sugar_export.sh` | Export wrapper |
| `workspace/core4d/scripts/launch/active/run_E167_remote.sh` | SPIDER remote CEM full |
| `workspace/core4d/scripts/launch/active/pull_E167_remote_results.sh` | Pull SPIDER/SUGAR results |
| SUGAR `scripts/sugar_rl/launch_core4d_e167_refiner.sh` | 21 downstream jobs, if approved |
| SUGAR `scripts/sugar_rl/summarize_core4d_e167_zsplit_failed_windows.py` | success + z/xy/object failure split |

### Phase 3 — report/log/tracker

结果 log 必须包含：

- 7 × 3 completion table。
- Axis audit table：证明 E167 没有限制 XY。
- SPIDER metrics：contact/penetration、z-gate、xy diagnostic、jerk/accel。
- Downstream table：SUGAR 3D success、Holosoma-z offline pass、failure split。
- 与 E163 baseline、E166 A_B2 的 per-case delta。
- 可视化观察：至少对每个 arm 抽 1 个改善 case + 1 个回归 case 做视频/关键帧分析。

---

## Success Criteria

最低完成标准：

1. 21 rows manifest 完整，`box026` 不出现。
2. `E167A` 和 `E167A+B1` full CEM 14/14 完成；`E167A+B2` z-only postprocess 7/7 完成。
3. Axis audit 通过：无 XY penalty，B2_z xy invariance pass。
4. Contact no-regression 通过或明确标记 fail case。
5. Holosoma-z gate 至少 2/7 case 改善；若 SUGAR 3D 不改善但失败主要为 `xy_only`，结论写为“符合 Holosoma 对齐但不适合 SUGAR 3D”。
6. 若任一 arm 在 `box021_029_p2` 或 `box021_035_p1` 这类历史可成功 case 上出现明显 collapse，暂停推广并分析 B2_z consistency / contact regression。

---

## Planned Commands

这些命令本轮不执行，实施阶段创建脚本后再跑：

```bash
# preflight only
python workspace/core4d/scripts/experiments/E167/build_zonly_manifest.py
bash workspace/core4d/scripts/eval/wrappers/eval_E167_zonly.sh preflight

# SPIDER CEM full, after user approval
bash workspace/core4d/scripts/launch/active/run_E167_remote.sh full
bash workspace/core4d/scripts/launch/active/pull_E167_remote_results.sh full

# export/downstream, after CEM gate passes
bash workspace/core4d/scripts/launch/active/run_E167_sugar_export.sh
bash /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/launch_core4d_e167_refiner.sh all
bash /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/pull_core4d_e167_refiner_remote.sh
python /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/summarize_core4d_e167_zsplit_failed_windows.py
```

---

## Risks

| Risk | Mitigation |
|---|---|
| B2_z body reference smoothing breaks joint/body consistency | preflight computes consistency error; above threshold do not train B2_z |
| SUGAR 3D termination hides Holosoma-z gains | report z/xy split; treat SUGAR success as secondary |
| Removing XY foot-slip reintroduces lateral skating/contact loss | keep XY diagnostics and contact no-regression as hard report criteria, but do not optimize XY |
| box023 pathology is not z-only body tracking | keep as negative-control case; do not overfit E167 to it |
| Dirty E166 worktree complicates diff review | keep E167 changes in new files/namespaces where possible; do not refactor E166 in implementation |

