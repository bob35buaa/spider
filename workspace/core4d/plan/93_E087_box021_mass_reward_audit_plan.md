# E087 Plan: Box021 质量与 reward 分项诊断

日期：2026-05-28

## Context

用户指出 COLA-style support body 未必优于当前几乎 GT 的物体轨迹，而且历史上 6-DoF joint connection 也试过不行。因此 E087 先不继续 support-body 路线，改查两个更直接的问题：

1. **物体是否太重。** 当前 `d003_box021_20231018_029_p2_upperobj_e083` 的 object inertial mass 是 `29.632kg`，而较可看的 `box023_p2` guard 和 `box025_p2` 相关 scene 都是 `5.0kg`。这可能让单 G1 + 当前 CEM 更倾向用头/胸/肩压箱，而不是手部承重。
2. **reward 分项是否把 CEM 拉向错误局部解。** E085/E086 的 `.npz` 只保存 qpos/qvel/ctrl，不保存每步 reward term；需要离线 replay 或复算 `qpos_rew/contact_hdmi/task_obj/upperbody penalty/hand penalty/object lift` 等贡献，确认哪个项占主导。

## Claims

| Claim | 验证方式 | 成功/失败判断 |
|---|---|---|
| C1 物体质量是 Box021 main 失败的关键因素 | 派生 mass-scaled scene，先跑 `5kg` 与 `10kg` main | 若 head/upper penetration、object floor、obj error 明显改善，说明 29.6kg 是主因之一 |
| C2 当前 reward 正在用 contact/object 项压过 safety/body 项 | 对 E085/E086 结果离线复算 reward breakdown | 若 contact/task_obj 正贡献远大于 upperbody/hand penalty，且高 reward 帧对应压箱，则 reward 结构有问题 |
| C3 reward 调整能在 29.6kg 或低质量下改善 | 基于 breakdown 做 2-3 组 reward 调参 | 至少一组满足 contact `>=50%`、head pen `<5%`、upper pen `<15%`、object mean 不显著恶化 |

## 实验设计

### E087A: mass audit + mass sweep

先做只读 audit：

- 统计活跃 CORE4D scene 的 object mass/inertia/碰撞盒尺寸；
- 对比 `box021=29.632kg`、`box023=5kg`、`box025=5kg`；
- 检查 mass 是否来自所有 box021 派生 scene，而不是 E083/E085 创建过程引入。

再创建派生 task：

- `d003_box021_20231018_029_p2_upperobj_e083_m5_e087`
- `d003_box021_20231018_029_p2_upperobj_e083_m10_e087`

派生规则：

- 复制 E083 upper-body-object scene；
- 只改 object `<inertial mass>`；
- `diaginertia` 按 `new_mass / old_mass` 等比例缩放；
- 保持碰撞 pair、object visual/collision、trajectory 不变。

### E087B: reward breakdown replay

新增离线脚本，对已有 E085/E086 结果和后续 E087 结果复算 per-frame reward term：

- 输入：variant、override、task、npz；
- 加载同一 config、ref、mask、external target；
- 用 CPU MuJoCo 对保存的 sim qpos/qvel 做 `mj_forward`；
- 复算：
  - `qpos_rew` 或 local-frame tracking 分项；
  - `task_obj_rew`；
  - `contact_hdmi_rew`；
  - `ctrl_ref_guard_rew`；
  - `robot_object_penalty`；
  - `hand_floor_penalty`；
  - `hand_object_deep_penalty`；
  - `object_lift_rew`；
  - `object_floor_penalty`；
  - `stability_penalty`；
- 输出 CSV + summary JSON，按 full/case-window 分别统计 mean/sum/min/max。

目标不是完全重现 MJWarp CEM 每个 sample 的内部 reward，而是对最终 rollout 用同一公式做可解释归因。

### E087C: reward tuning groups

在看完 E087B 后执行；预设 3 组候选，只先跑 main gate：

| 组 | 思路 | 预期 |
|---|---|---|
| Mass-only | E085 raw target + mass `5kg/10kg`，reward 不改 | 隔离质量影响 |
| Safety-dominant | 降 contact/object gain，增强 upperbody/hand deep/stability；优先避免头胸压箱 | 若 contact 下降但姿态安全，说明旧 reward 太激进 |
| Object-light + bounded object | 低质量 + HDMI-style bounded object reward，避免 unbounded tracking 拉着身体压箱 | 在低质量下找可用 seed |

初始建议参数：

- `contact_hdmi_gain`: `5.0 -> 2.0 或 3.0`
- `task_obj_use_exp: true`
- `task_obj_pos_rew_scale`: `0.5` 或更低
- `robot_object_penalty_scale`: `8-12`
- `hand_object_deep_penalty_scale`: `10`
- `stability_penalty_scale`: `1-2`
- 暂不再提高 vfrac；target height 已在 E086B 验证不是唯一主因。

## 成功标准

main gate：

- `case_window_sim_contact_frames_pct >= 50%`
- `case_window_obj_err_mean_m <= E085 + 0.10m`，即不超过约 `0.765m`
- `case_window_sim_head_collision_object_penetration_pct < 5%`
- `case_window_sim_upperbody_object_penetration_pct < 15%`
- `case_window_sim_lh_floor_contact_pct <= 5%`
- `case_window_sim_rh_floor_contact_pct <= 5%`
- 视频中不能再以头/胸/肩/肘压箱作为主要支撑。

guard 规则：

- 只有 main 至少有一组显著改善时，才跑 `box023_p2` guard；
- mass-only 不跑 guard，因为 guard 物体本来是 `5kg`。

## 执行顺序

1. 实现并运行 E087A mass audit。
2. 实现并运行 E087B reward breakdown，对 E085 main/guard、E086A/B 做离线复算。
3. 根据 breakdown 生成 E087 overrides。
4. 先 smoke E087 mass variants。
5. 本地跑最关键 `5kg mass-only`；远程并行跑 `10kg mass-only` 与一组 reward-tuned variant。
6. 回收、评估、生成 keyframe sheets、写 log、更新 tracker。

## 结果路径

| 类型 | 路径 |
|---|---|
| mass audit | `workspace/core4d/results/E087/mass_audit/` |
| reward breakdown | `workspace/core4d/results/E087/reward_breakdown/` |
| derived scenes | `example_datasets/processed/core4d/unitree_g1/humanoid_object/*_e087/` |
| CEM results | `workspace/core4d/results/E087/` |
| scripts | `workspace/core4d/scripts/E087/`, `workspace/core4d/scripts/eval/eval_E087.py` |

