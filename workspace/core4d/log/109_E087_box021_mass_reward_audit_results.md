# E087 Box021 质量与 reward 分项诊断结果

日期：2026-05-28

对应计划：`workspace/core4d/plan/93_E087_box021_mass_reward_audit_plan.md`

## 结论

这轮验证支持用户的判断：**继续转 COLA-style support body 不是当前最直接的下一步，Box021 失败更应该先看物体质量和 reward 结构。**

结果分两层：

1. **物体质量确实异常。** D003 Box021 系列 object mass 是 `29.632kg`，而较可看的 `box023_p2` guard 和 `box025_p2` 是 `5.0kg`。这不是 E083/E085 派生过程引入，而是 D003 Box021 scene 本身继承的 mass/inertia。
2. **但质量不是唯一主因。** 把 main case 降到 `5kg/10kg` 后仍然失败；`10kg` 比 `5kg` 略好，但三组都继续出现头/上身压箱。说明当前 raw hand target + reward 结构仍会把 CEM 拉到“身体压箱拿 contact”的局部解。

reward breakdown 的核心发现更直接：E085 main case-window 平均 `contact_hdmi_rew=2.747`、`qpos_rew=1.965`、`task_obj=0.383`，而 `robot_object_penalty=-0.030`。也就是说，旧 reward 中 safety penalty 的量级小到几乎不能影响 CEM 的 elite selection。

E087C 把 contact/object 降权、upperbody penalty 加强后，`robot_object_penalty` 已到 `-0.634`，但 head/upper penetration 仍 `89.1%`。这说明**单纯线性 scalar penalty 调权还不够**，后续要么引入更硬的 constraint/gating，要么在 CEM 采样阶段直接过滤 upperbody penetration，而不是继续小幅调权。

## 改动与产物

| 类型 | 路径 |
|---|---|
| 计划 | `workspace/core4d/plan/93_E087_box021_mass_reward_audit_plan.md` |
| mass audit | `workspace/core4d/results/E087/mass_audit/` |
| reward breakdown | `workspace/core4d/results/E087/reward_breakdown/` |
| 质量派生 task | `d003_box021_20231018_029_p2_upperobj_e083_m5_e087`, `d003_box021_20231018_029_p2_upperobj_e083_m10_e087` |
| overrides | `examples/config/override/core4d_E087A_m5_rawtarget_main.yaml`, `core4d_E087B_m10_rawtarget_main.yaml`, `core4d_E087C_m5_safe_main.yaml` |
| train/eval | `workspace/core4d/scripts/train/train_E087.sh`, `workspace/core4d/scripts/eval/eval_E087.py` |
| remote/pull | `workspace/core4d/scripts/run_E087_remote.sh`, `workspace/core4d/scripts/pull_E087_remote_results.sh` |
| 可视化 sheets | `workspace/core4d/results/E087/keyframes/contact_sheets/` |

## 质量审计

| Scene | Mass | Collision half extents | Density proxy |
|---|---:|---:|---:|
| `d003_box021_20231018_029_p2_upperobj_e083` | `29.632kg` | `0.1596/0.2089/0.2647m` | `419.7kg/m^3` |
| `box023_person2_upperobj_e083` | `5.0kg` | `0.1531/0.1568/0.1766m` | `147.4kg/m^3` |
| `box025_person2` | `5.0kg` | `0.3768/0.3778/0.4692m` | `9.36kg/m^3` |

补充：

- 当前仓库中 D003 Box021 派生 scene 基本都是 `29.632kg`。
- 早期 `box021_person1` 是 `5.0kg`，说明 mass 不是物体类别不可变常量，而是不同数据转换/scene 版本带来的差异。
- E087 派生 scene 只改 object `<inertial mass>`，并按 `new_mass / old_mass` 等比例缩放 `diaginertia`，不改轨迹、碰撞盒和 collision pair。

## Reward Breakdown

case-window 平均 reward term：

| Variant | qpos/local | contact | task_obj | upperbody penalty | hand deep | object lift | object floor | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E085 main | `1.965` | `2.747` | `0.383` | `-0.030` | `0.000` | `0.007` | `0.000` | `4.987` |
| E085 guard | `2.493` | `2.866` | `0.537` | `-0.002` | `0.000` | `0.330` | `-0.446` | `5.705` |
| E086A strict | `1.689` | `1.283` | `0.392` | `-0.218` | `-0.024` | `0.006` | `0.000` | `3.072` |
| E086B vfloor | `1.825` | `1.751` | `0.428` | `-0.040` | `0.000` | `0.006` | `0.000` | `3.899` |
| E087A 5kg | `1.625` | `2.438` | `0.401` | `-0.080` | `0.000` | `0.007` | `0.000` | `4.251` |
| E087B 10kg | `1.790` | `2.411` | `0.427` | `-0.052` | `0.000` | `0.007` | `0.000` | `4.404` |
| E087C 5kg safe | `1.743` | `0.853` | `0.179` | `-0.634` | `0.000` | `0.004` | `0.000` | `2.039` |

解释：

- E085/E087 mass-only 中，contact + local tracking 是主要正项；upperbody penalty 只有百分之一到十分之一量级。
- E086A/E087C 加强 safety 后 penalty 量级上来了，但仍没有带来可接受姿态，说明当前 penalty 作为 soft reward 仍无法当成 hard physical constraint。
- `object_lift_rew` 在 main 几乎为零，说明它没有实际提供“抬起箱子”的有效梯度；`object_floor_penalty` 在 main 也基本为零，原因是当前计算相对 ref bottom，且 sim/ref 都处在接近地面的错误区间时约束不敏感。

## CEM 结果

| Variant | Mass | Reward | Contact | Obj mean/max | Pelvis min | Head pen | Upper pen | LH/RH obj pen | LH/RH floor | Gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| E087A | `5kg` | E085 raw target | `82.95%` | `0.782/1.297m` | `0.574m` | `89.15%` | `89.15%` | `76.74/81.40%` | `0/0%` | fail |
| E087B | `10kg` | E085 raw target | `82.95%` | `0.735/1.218m` | `0.629m` | `69.77%` | `75.97%` | `74.42/52.71%` | `0/0%` | fail |
| E087C | `5kg` | safety tuned | `65.89%` | `0.774/1.282m` | `0.618m` | `89.15%` | `89.15%` | `54.26/60.47%` | `0/1.55%` | fail |

对照 E085 main：

- E085 main `29.632kg`：contact `82.95%`，obj mean/max `0.665/1.068m`，head/upper `18.60/53.49%`。
- 降到 `10kg` 后 object error 变差、head 更差，但 pelvis 更稳；
- 降到 `5kg` 后 head/upper 最差，视觉上更容易把箱子压/推歪；
- `5kg+safety` 降低 contact 和手部穿透，但没有降低 head/upper penetration。

## 可视化观察

sheet：`workspace/core4d/results/E087/keyframes/contact_sheets/E087_all_cases_sheet.jpg`

- E087A 5kg：从中段开始明显弯腰趴到箱体上，后段上身/头部长期压在箱体边缘，箱体姿态漂移更明显。
- E087B 10kg：比 5kg 略稳，箱体不如 5kg 那么轻易被推歪，但仍是头/胸/肩压箱，不是双手承重搬运。
- E087C 5kg safety：接触动作更保守，手部穿透有所下降，但机器人仍把上身放到箱体上，说明当前 safety reward 没有形成硬约束。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 物体质量是 Box021 main 失败关键因素 | 部分成立。`29.6kg` 明显异常，但降质量没有直接成功；`10kg` 略好于 `5kg`，说明质量影响动力学，但不是唯一主因 |
| C2 reward 中 contact/object 正项压过 safety/body 项 | 成立。E085 main 中 `contact+qpos+task_obj ≈ 5.10`，upperbody penalty 仅 `-0.03` |
| C3 reward 调整可改善 | 本轮不成立。E087C 强化 safety 后 hand penetration/contact 有变化，但 head/upper penetration 没降 |

## 下一步建议

不建议继续做小幅 reward weight sweep。更合理的 E088 方向：

1. **Hard gate / elite filtering**：CEM 每轮直接过滤 head/torso/pelvis/shoulder/elbow penetration 超阈值的 samples，而不是只加 soft penalty。
2. **改 object lift/floor reward 口径**：当前相对 ref bottom 的 lift/floor shaping 对 main 无效，应改成绝对 bottom clearance 或目标高度区间。
3. **保留 mass sweep，但采用 `10kg` 而不是 `5kg`。** `10kg` 比 `5kg` 在视觉和指标上更稳；`5kg` 会让箱体过容易被上身推/压着漂。
4. **若继续 reward 调参，优先做 hard-safety + low-contact**：`contact_hdmi_gain <= 1`、`task_obj <= 0.1`、upperbody penetration hard gate，而不是继续把 `robot_object_penalty_scale` 从 12 调到 20/30。

## 结果路径

| 类型 | 路径 |
|---|---|
| comparison | `workspace/core4d/results/E087/comparison.csv` |
| gate summary | `workspace/core4d/results/E087/e087_gate_summary.json` |
| videos | `workspace/core4d/results/E087/E087A_m5_rawtarget_main.mp4`, `E087B_m10_rawtarget_main.mp4`, `E087C_m5_safe_main.mp4` |
| keyframe sheets | `workspace/core4d/results/E087/keyframes/contact_sheets/` |
| reward breakdown summaries | `workspace/core4d/results/E087/reward_breakdown/*/summary.json` |
| mass audit | `workspace/core4d/results/E087/mass_audit/summary.json` |

