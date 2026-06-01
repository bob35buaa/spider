# E107 Box021 selected-4 full CEM 结果

计划：`workspace/core4d/plan/115_E107_box021_selected4_full_cem_plan.md`

核心产物：

- `workspace/core4d/results/E107/cem/full/full_eval_summary.md`
- `workspace/core4d/results/E107/cem/full/full_eval_summary.csv`
- `workspace/core4d/results/E107/cem/full/full_eval_summary.json`
- `workspace/core4d/results/E107/cem/full/E107C*.npz`
- `workspace/core4d/results/E107/cem/full/E107C*_full.mp4`
- `workspace/core4d/results/E107/cem/full/legobj_timeseries_E107C*.csv`

## 实验设置

E107 Phase 2 使用 `workspace/core4d/results/E107/selected_case_to_cem.json` 中的 4 个 selected case，路线固定为 `ref_fk_clean`。所有 case 在 CEM 前均通过 clean task preflight 与 medium subagent replay 审查：

- 4/4 clean task 校验通过；
- scene/scene_act 无 `29.632` robot inertial 污染；
- leg/foot + upper-body object collision pairs 完整；
- pre-CEM replay 4/4 `PASS_WITH_NOTES`；
- override 中 `contact_hdmi_target_source=ref_fk`，不使用外部 raw target/mask。

首次启动时发现 E107 clean reconstruction 只保存了 `qpos`，导致 CEM 读取 `trajectory_kinematic.npz` 时缺 `qvel`。已修复 `workspace/core4d/scripts/E107/build_box021_clean_gate.py`，重建 clean target 时复制旧 D003 trajectory NPZ 的全部数组：`qpos/qvel/ctrl/contact/contact_pos`。

## 执行过程

原计划为本地 1 卡 + 远程 2 卡并行。实际执行时远程 GPU0 被已有 R134 训练进程占用，因此采用动态调度，避免等待队列重复执行：

| 变体 | source task | 原计划分配 | 实际执行 |
|---|---|---|---|
| `E107C01_box021_20231011_034_p1_ref_fk_clean` | `d003_box021_20231011_034_p1` | local-gpu0 | local GPU0 |
| `E107C02_box021_20231011_035_p1_ref_fk_clean` | `d003_box021_20231011_035_p1` | remote-gpu0 | C01 完成后接到 local GPU0 |
| `E107C03_box021_20231011_035_p2_ref_fk_clean` | `d003_box021_20231011_035_p2` | remote-gpu1 | remote GPU1 |
| `E107C04_box021_20231018_029_p2_ref_fk_clean` | `d003_box021_20231018_029_p2` | remote-gpu0 | C03 完成后接到 remote GPU1 |

所有 4 条 full CEM 均完成，产出 4 个 root NPZ、4 个 outdir trajectory NPZ、4 个 MP4。`ffprobe` 验证视频均可解码：

| 变体 | 视频帧数 | 时长 |
|---|---:|---:|
| `E107C01_box021_20231011_034_p1_ref_fk_clean` | 286 | 5.72s |
| `E107C02_box021_20231011_035_p1_ref_fk_clean` | 258 | 5.16s |
| `E107C03_box021_20231011_035_p2_ref_fk_clean` | 266 | 5.32s |
| `E107C04_box021_20231018_029_p2_ref_fk_clean` | 150 | 3.00s |

## 结果

`full_eval_summary.md` 使用 E105 上半身安全阈值，并补 E026/E081 下半身 strict proxy：`leg_box_interference_frac <= 5%`。按当前 E107 口径，源序列时长和 pelvis tilt 不作为 pass/fail gate；tilt 只作为诊断数值保留。Replay gate 只拦截 `pelvis_low` 和 `lie_on_box`。

| 变体 | 接触率 | 物体均值误差 | 物体最大误差 | pelvis 最低 | replay | 上半身状态 | 腿部干涉 | lower strict | RL strict |
|---|---:|---:|---:|---:|---|---|---:|---|---|
| `E107C01_box021_20231011_034_p1_ref_fk_clean` | 83.2% | 0.010m | 0.026m | 0.655m | FAIL | FAIL | 18.9% | FAIL | NO |
| `E107C02_box021_20231011_035_p1_ref_fk_clean` | 77.5% | 0.008m | 0.019m | 0.639m | PASS | WORK | 3.1% | PASS | YES |
| `E107C03_box021_20231011_035_p2_ref_fk_clean` | 75.9% | 0.008m | 0.021m | 0.589m | PASS | WORK | 8.3% | FAIL | NO |
| `E107C04_box021_20231018_029_p2_ref_fk_clean` | 69.3% | 0.010m | 0.029m | 0.662m | PASS | WORK | 6.7% | FAIL | NO |

上半身穿箱已经不是这 4 条 full CEM 的主要问题：4/4 的 head penetration、upper-body penetration、hand-floor `<5cm` 比例均为 `0.0%`。

Replay/姿态诊断：

- `E107C01`：pelvis 末段高度通过；tilt `78.9deg` 只作为诊断数值保留，但 `lie_on_box=31.5%` 仍触发 replay fail。
- `E107C04`：pelvis 末段高度通过；tilt `82.7deg` 只作为诊断数值保留，因此在当前 E107 口径下 replay 通过。

下半身失败：

- `E107C01`：leg interference `18.9%`，最差 geom 为 `right_thigh_collision`，min SDF `-0.008m`。
- `E107C03`：leg interference `8.3%`，最差 geom 为 `right_thigh_collision`，min SDF `-0.007m`。
- `E107C04`：leg interference `6.7%`，最差 geom 为 `left_thigh_collision`，min SDF `-0.004m`。

## 结论

E107 Phase 2 找到 1 条 strict positive：

- `E107C02_box021_20231011_035_p1_ref_fk_clean`

这条 case 是唯一同时通过 replay/上半身安全 gate 和 lower-body strict gate 的结果：接触率 `77.5%`，物体均值误差 `0.008m`，物体最大误差 `0.019m`，腿部干涉 `3.1%`。

另外 3 条可作为诊断样本，但不建议进入 RL positive handoff：

- `E107C01`：接触率高、物体误差低，但 `lie_on_box` 触发 replay fail，且腿部干涉明显。
- `E107C03`：上半身/replay 为 WORK，但 lower-body strict fail。
- `E107C04`：移除 tilt/时长 gate 后上半身/replay 为 WORK，但 lower-body strict fail。

因此，E107 在 E103 template 修复后更新了 Box021 结论：clean Box021 并非完全不可用，但 strict-positive 产率较低（当前 selected CEM sample 为 `1/4`；若不继续跑剩余 case，则在 13 条 E107 clean `cem_ready` 中暂时只有 `1/13`）。下一步可以把 `E107C02` 作为窄正例交给 RL，也可以继续扩跑剩余 clean `cem_ready` rows 以寻找更多 Box021 positive。
