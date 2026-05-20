# E029 实验计划：Bucket001 stability / posture-valid contact control

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e029-stability-control`

## Context

E024 已经证明 `bucket001_p1` 与 `bucket001_p2` 不是同一种失败：

| Case | E024 结论 | 下一步含义 |
|---|---|---|
| `bucket001_p1` | 三个 stability/root/contact variants 全部 fall，pelvis min `0.066-0.156m`，5cm contact `0%` | 不能继续复跑 E024 的线性 height penalty / root sigma / contact gain sweep；需要更硬的 upright/root terminal gate 与 posture-valid contact |
| `bucket001_p2` | no-fall 可修，pelvis min `0.713-0.726m`，contact `77.53-79.78%`，但 deep pen `59.60-64.65%` | stability 可作为 guard；penetration 已由 E028 证明不能靠硬 barrier 第一版解决 |

E027 将 `bucket001_p1/p2` 标为 `usable_with_caveat`，不是弃用对象。当前不能把 p1 的 fall 直接归因到数据差；除非 E029 新增 independent reference/posture evidence，否则它仍是 P1 diagnostic algorithmic failure。

E028 对 `bucket001_p2` 的 contact gate 能把 deep penetration 降到 `0%`，但 contact 从 `77.53%` 降到 `2.81%`。这说明 E029 不把 `bucket001_p2` 当 penetration success，而只用它验证新增 stability/contact gate 不破坏已修复的 upright behavior。

## Claims

| Claim | 最低证据 |
|---|---|
| C1: `bucket001_p1` 的 fall 需要 hard posture feasibility，而不是 E024 同类参数微调 | 至少 3 个机制不同的 p1 full variants；如果仍失败，要输出 pelvis/root/foot/reference diagnostic，证明失败不是简单 gain 不足 |
| C2: hard upright / score-cap 能阻止 CEM 选择趴地姿态 | 至少 1 个 p1 variant no-fall 且 `full_pelvis_z_min_m >=0.45m`；目标 `>=0.55m` |
| C3: posture-valid contact gate 不能靠切断接触伪造稳定 | p1 no-fall variant 必须 contact `>20%` 才算有效信号；contact 仍 `0%` 时只算 stability-only partial |
| C4: object-side support proxy 不回退 | 所有 p1/p2 variants object pos `<=10cm`；guard `bucket001_p2` object `<=8cm` |
| C5: guard 不回退 | `bucket001_p2` 保持 no-fall、pelvis `>=0.55m`、contact `>=50%`；`box025_p2` strict-pass guard 不因全局 posture/contact gate 退化 |

## Scope

主目标：

- `bucket001_p1`

Guards：

- `bucket001_p2`：E024 已能 no-fall，用来验证新增 E029 knobs 不破坏 stability。
- `box025_p2`：E018b strict-pass guard，用来验证 posture/contact gate 不对已通过 case 产生明显接触回退。

不纳入 E029：

- `bucket001_p2` penetration repair：E028 已证明第一版 hard barrier/contact gate 会让 contact collapse，后续需 E028b/E030 surface target。
- `box021_p1/p2`：仍按用户要求暂不优化；只保留 P1 caveat。
- `desk021_p1`：E027 已 `discard_from_success_denominator`。

## 改动

### 1. Core stability / posture knobs

E029 预计需要核心代码改动，因此从 E028 结果分支切出新分支。

| 文件 | 改动 |
|---|---|
| `spider/config.py` | 新增默认关闭的 `upright_barrier_*`、`upright_score_cap_*`、`root_tilt_penalty_*`、`foot_support_*`、`posture_contact_gate_*` |
| `spider/simulators/mjwp.py` | 在 reward 中实现 pelvis height squared barrier、fall score-cap、root tilt penalty、foot support penalty、posture-valid contact gate，并输出 info fields |
| `workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py` | 生成 E029 manifest + Hydra overrides，复用 E018b/E024 source variants |
| `workspace/core4d_collab_retarget/scripts/E029/variants.tsv` | 固定不超过 6 个 full variants |
| `workspace/core4d_collab_retarget/scripts/train/train_E029.sh` | 本地/远程/smoke/eval entrypoint |
| `workspace/core4d_collab_retarget/scripts/run_E029_remote.sh` | 远程 2-GPU 队列 |
| `workspace/core4d_collab_retarget/scripts/eval/eval_E029.py` | 汇总 pelvis/root/contact/object/penetration/guard 指标 |

所有新增 knobs 默认关闭，旧 E024/E028 行为必须保持不变。

### 2. Hard posture feasibility

E024 的 `stability_penalty_scale` 是线性 height penalty：

```text
reward += -scale * clamp(threshold - pelvis_z, min=0)
```

E029 第一版只做小而可解释的升级：

| 机制 | 作用 | 风险 |
|---|---|---|
| `upright_barrier` | 对 `pelvis_z < threshold` 给 squared / margin-normalized barrier，比 E024 线性 penalty 更硬 | 可能让 robot 远离 bucket，contact 仍为 `0%` |
| `upright_score_cap` | 如果 frame 出现 `pelvis_z < fall_threshold`，给 sample 加大幅 cap penalty，近似 reject fall trajectory | 可能让 CEM 停在无接触但站立的局部解 |
| `root_tilt_penalty` | 惩罚 pelvis/root body z-axis 偏离世界 z，减少 crouch / horizontal posture shortcut | root body orientation 口径需验证，避免误惩罚正常弯腰 |
| `foot_support` | 惩罚双脚离地或 foot sliding 过大，避免 p1 用低 pelvis/无支撑姿态换 object tracking | 如果过强，可能把 robot 固定在原地，object/contact 不动 |
| `posture_contact_gate` | 当 pelvis/root/foot 姿态不合法时关闭 contact reward；姿态合法后再追 contact | 可能进一步降低 contact，需要与 no-fall 分开解释 |

第一版不做复杂 footstep planner，但会实现默认关闭的轻量 foot-support reward/penalty：基于 MuJoCo `left_foot` / `right_foot` sites 的高度和相邻帧近似速度，惩罚双脚同时离地、低 pelvis 时无支撑、以及明显 foot sliding。复杂可规划 foot support / stepping target 推到 E029b。

## Variants

控制 full variants 不超过 6 个：

| Variant | Source | Case | Queue | 机制 | 关键参数 |
|---|---|---|---|---|---|
| `E029_bucket001_p1_upright_barrier_t055` | `E024_bucket001_p1_root025_gain2_stab_t065` | `bucket001_p1` | local | `upright_barrier` | pelvis threshold `0.55m`, root sigma `0.25`, contact gain `2` |
| `E029_bucket001_p1_posture_gate_t055` | `E024_bucket001_p1_root025_gain2_stab_t065` | `bucket001_p1` | remote_gpu0 | `upright_barrier + posture_contact_gate` | contact reward only when pelvis/root/foot valid |
| `E029_bucket001_p1_scorecap_t045` | `E024_bucket001_p1_root025_gain2_stab_t065` | `bucket001_p1` | remote_gpu1 | `upright_score_cap + foot_support` | cap when pelvis `<0.45m` |
| `E029_bucket001_p1_tilt_gate_t055` | `E024_bucket001_p1_root025_gain2_stab_t065` | `bucket001_p1` | remote_gpu0 | `upright_barrier + root_tilt_penalty + foot_support` | root tilt max `55deg`, no posture gate |
| `E029_bucket001_p2_guard_posture_gate` | `E024_bucket001_p2_root025_gain2_stab_t065` | `bucket001_p2` | remote_gpu1 | guard | preserve E024 p2 no-fall setting |
| `E029_box025_p2_guard_posture_gate` | `E018b_box025_p2_canonical_t02` | `box025_p2` | local_after_main | guard | verify pass case contact does not collapse |

执行顺序：

1. 本地先跑 `E029_bucket001_p1_upright_barrier_t055`，快速判断 hard upright 是否能改变 p1 fall。
2. 远程 GPU0 跑 `posture_gate` 后接 `tilt_gate`。
3. 远程 GPU1 跑 `scorecap` 后接 `bucket001_p2` guard。
4. `box025_p2` guard 在本地主 variant 完成后本地补跑；如果 p1 主 variant 已暴露 core wiring 错误，先修 smoke。

## 本地 / 远程并行

E029 有 `>=3` 个独立 full variants，满足远程并行触发条件。计划使用本地 1 卡 + 远程 2 卡：

```bash
# 生成 variants / overrides
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py

# smoke：本地确认所有 overrides 和新 knobs 可解析
RUN_TIMEOUT_SECONDS=600 RUN_STALL_TIMEOUT_SECONDS=180 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh smoke 0

# full：本地关键 case + guard
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh local 0

# full：远程 2 卡，需先 git push
git push -u origin exp/core4d-collab-retarget-e029-stability-control
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git fetch && git switch exp/core4d-collab-retarget-e029-stability-control && git pull"
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/run_E029_remote.sh

# 结果回收后本地统一评估
bash workspace/core4d_collab_retarget/scripts/pull_E029_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E029.py --all
```

如果远程主 worktree 仍有用户/旧实验 dirty 文件，不清理、不覆盖，改用 E028 同样的独立 worktree 方案。

## Evaluation

`eval_E029.py` 必须输出：

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/34_E029_bucket001_stability_control_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E029/variants.tsv` |
| Manifest | `workspace/core4d_collab_retarget/results/E029/manifest.tsv` |
| Results | `workspace/core4d_collab_retarget/results/E029/*.npz`, `online_video/*.mp4` |
| Comparison | `workspace/core4d_collab_retarget/results/E029/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E029/aggregate_summary.json` |
| Baseline delta | `workspace/core4d_collab_retarget/results/E029/baseline_delta.csv` |
| Log | `workspace/core4d_collab_retarget/log/29_E029_bucket001_stability_control_results.md` |

核心指标：

- `full_pelvis_z_min_m`
- `case_window_pelvis_z_min_m`
- `first_pelvis_z_lt_45cm_frame`
- `E018b_robot_fall_detected`
- `paper_omniretarget_contact_preservation_5cm_pct`
- `paper_omniretarget_robot_object_deep_penetration_duration_pct`
- `paper_omniretarget_robot_object_max_penetration_cm`
- `paper_object_Epos_case_m`
- `paper_object_Erot_case_deg`
- `paper_omniretarget_foot_skating_duration_pct`
- `paper_pelvis_pos_error_case_cm`

E029 p1 stability pass：

```text
no fall
full_pelvis_z_min_m >= 0.45
object_Epos <= 10cm
```

E029 p1 useful signal：

```text
stability pass
contact_5cm > 20%
object_Epos <= 10cm
```

E029 p1 strict target：

```text
contact_5cm >= 50%  (stretch >=70%)
robot_object_deep_penetration <=20%  (target <=15%)
robot_object_max_penetration <=5cm
object_Epos <=10cm
no fall
```

## 成功标准

| 指标 | 目标 |
|---|---|
| target coverage | `bucket001_p1` 至少 4 个 full E029 results |
| p1 stability pass | 至少 `1/4` p1 variants no-fall 且 pelvis min `>=0.45m` |
| p1 useful signal | 至少 `1/4` p1 variants contact `>20%` 且 object `<=10cm` |
| p1 strict target | stretch：至少 1 个 p1 variant contact `>=50%`、deep pen `<=20%`、max pen `<=5cm` |
| p2 guard | no-fall、pelvis `>=0.55m`、contact `>=50%`、object `<=8cm` |
| box025 guard | no-fall、object `<=8cm`、contact 不低于 baseline `-20pp`，deep pen 不回退 |

## 停止条件

| 情况 | 动作 |
|---|---|
| smoke 中新增 knobs 解析失败或旧 default 行为改变 | 先修 wiring，不跑 full |
| p1 4 个机制 variants 全部 pelvis `<0.25m` 或 contact 仍 `0%` | 停止 stability reward sweep，转 reference/support timing 或 lower-body control infeasible audit |
| hard upright 让 object pos `>15cm` 且 contact 仍低 | 不继续加 scale，改查 support/contact target 是否把 robot 拉离 object |
| p2 guard 从 no-fall 回退到 fall | 暂停 E029 结论，修 posture gate 默认/条件逻辑 |
| box025 guard contact collapse | 不把 posture gate 并入通用 reward；后续只做 case-specific diagnostic |

## Git 策略

E029 至少分三次提交：

1. `plan(core4d_collab): start E029 bucket001 stability plan`
2. `feat(core4d_collab): add E029 posture stability controls`
3. `log(core4d_collab): record E029 stability results`

如果实现中需要 foot-support planner 或 scene XML 几何修改，另起 E029b，不混入本实验。
