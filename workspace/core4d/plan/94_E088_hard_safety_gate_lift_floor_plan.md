# E088 Plan: Hard Safety Gate 与绝对物体离地奖励

日期：2026-05-28

## Context

E087 按用户建议检查了两个方向：物体质量与 reward 分项。

关键结论：

1. `d003_box021_20231018_029_p2` 相关 scene 的物体质量是 `29.632kg`，明显高于较可看的 `box023_p2` / `box025_p2` 的 `5kg`。但把 Box021 main 改成 `5kg/10kg` 后仍然失败，所以质量异常是因素，但不是唯一主因。
2. reward breakdown 显示 E085 main 的主要正项是 `contact_hdmi_rew=2.747`、`qpos_rew=1.965`、`task_obj=0.383`，而 upper-body/object penalty 只有 `-0.030`。E087C 把 safety penalty 提到 `-0.634` 后，head/upper penetration 仍为 `89.1%/89.1%`。
3. `object_lift_rew` 与 `object_floor_penalty` 在 main 上几乎没有贡献，说明当前相对 ref bottom 的 lift/floor 设计没有真正告诉 CEM “箱子必须离地、不能贴地滑动或被身体压着走”。

因此 E088 不继续做小幅 reward weight sweep，而是做两类机制改动：

- **CEM hard safety gate / elite filtering**：违反 head/torso/pelvis/shoulder/elbow 与 object collision 穿透阈值的 sample 不允许成为 elite。
- **绝对 object bottom clearance / height interval reward**：用世界坐标下箱体底面离地高度提供有效 lift/floor 信号，不再只相对 ref bottom。

## 概念解释

### 1. CEM elite filtering / hard gate 是什么

当前 CEM 流程是：

1. 对当前 control 均值采样很多条 trajectory；
2. rollout 每条 trajectory，得到一个 scalar reward；
3. 选 reward 最高的一小部分 sample 作为 elite；
4. 用 elite 的 control 均值和方差更新下一轮分布。

问题是：如果一个 sample 用头、胸、肩、肘穿进箱子，可以拿到很高的 contact/qpos/object reward，那么它仍会进入 elite。即使 reward 里有 soft penalty，只要 penalty 量级不够或被其他正项抵消，这个错误 sample 仍会支配下一轮 CEM。

Hard gate 的含义是：在 top-k 之前先计算每个 sample 的 safety 指标。如果 head/torso/pelvis/shoulder/elbow 对 object 的 SDF 小于阈值，或违规帧比例超过阈值，就把这个 sample 标记为 invalid。invalid sample 不能进入 elite，也不能参与 control 均值/方差更新。

这和 “加大 penalty” 的区别：

| 机制 | 作用位置 | 问题 |
|---|---|---|
| soft penalty | 加到 reward 标量里 | 高 contact/qpos reward 仍可能抵消 penalty |
| hard gate | top-k / elite selection 之前 | 违规 sample 直接不能更新 CEM 分布 |

实现时需要 fallback，否则如果一轮采样全都 invalid，optimizer 会卡死。fallback 规则建议是：先记录 `fallback_used=True`，然后从 invalid sample 里按 violation 最小、reward 最高的顺序选少量样本继续更新，同时把这轮标记为不可接受。

### 2. 绝对 bottom clearance / 高度区间是什么意思

当前代码里的 `object_lift_rew` 近似在比较：

```text
abs(sim_object_bottom_z - ref_object_bottom_z)
```

这在 Box021 main 上没贡献，原因是如果 ref bottom 本来就接近地面，或者 sim/ref 都在低位错误区间，reward 只会说“你和 ref 一样低”，不会说“箱子必须离开地面”。`object_floor_penalty` 也是围绕 ref bottom 做 margin，因此在 main 上几乎为 0。

E088 要改成绝对世界坐标口径：

```text
object_bottom_z = object_collision_center_z - object_collision_half_z
bottom_clearance = object_bottom_z - floor_z
```

然后给一个绝对区间，例如：

```text
bottom_clearance >= 0.04m
bottom_clearance <= 0.18m
```

低于下界时明确扣分，落在区间内给 bounded reward，高于上界可选轻微扣分，避免 CEM 通过物体 actuator 或碰撞把箱子弹飞。

这个 reward 的目标不是追踪 ref 物体高度，而是让 CEM 避免“箱子贴地、身体压箱、手只是蹭到物体”的局部解。

## Claims

| Claim | 验证方式 | 判断标准 |
|---|---|---|
| C1 hard gate 比 soft penalty 更能压住头/躯干/肩肘穿箱 | 在 Box021 main 上跑 gate-only，与 E087B/E087C 对比 | head pen `<5%`，upper pen `<15%`，且 `gate_fallback_used` 不长期为 True |
| C2 绝对 clearance reward 能让 object lift/floor 分项真正贡献优化 | reward breakdown 复算新增分项 | case-window 中 `object_clearance_rew/penalty` 非零，且 object bottom 不再长期低于阈值 |
| C3 `10kg + hard gate + low-contact + clearance` 是下一步 CEM/CEM-to-RL 更合理 seed | 与 E087B `10kg raw` 和 E087C `5kg safe` 对比 | 在安全指标显著改善的前提下，contact 保持 `>=40%-50%`，object mean 不明显恶化 |

## 实现计划

### A. 新增配置

在 `spider/config.py` 增加默认关闭的配置，确保旧实验行为不变。

Hard gate：

| 字段 | 建议默认值 | 含义 |
|---|---:|---|
| `cem_safety_gate_enabled` | `False` | 是否启用 CEM sample 级 hard gate |
| `cem_safety_gate_mode` | `"elite_filter"` | 先只实现 elite filtering |
| `cem_safety_gate_geom_names` | upper-body geoms | head/torso/pelvis/shoulder/elbow collision geoms |
| `cem_safety_gate_geom_ids` | `[]` | config processing 里解析 |
| `cem_safety_gate_min_sdf_m` | `-0.005` | 允许最多 5mm 数值穿透；后续可改 `0.0` |
| `cem_safety_gate_max_violation_pct` | `0.0` | 默认不允许违规帧进入 elite |
| `cem_safety_gate_min_valid_frac` | `0.02` | 低于该比例触发 fallback |
| `cem_safety_gate_fallback` | `"least_violation"` | 全部或几乎全部 invalid 时的降级策略 |

Object clearance：

| 字段 | 建议默认值 | 含义 |
|---|---:|---|
| `object_clearance_rew_scale` | `0.0` | 默认关闭 |
| `object_clearance_penalty_scale` | `0.0` | 默认关闭 |
| `object_clearance_floor_z` | `0.0` | 地面高度 |
| `object_clearance_min_m` | `0.04` | 底面至少离地 4cm |
| `object_clearance_max_m` | `0.18` | 允许的上界，避免飞箱 |
| `object_clearance_sigma` | `0.04` | bounded reward 的尺度 |
| `object_clearance_gate_source` | `"contact_mask"` | 只在 ref 接触/搬运窗口或指定 time window 生效 |

### B. 提取 upper-body/object SDF 计算

现在 `spider/simulators/mjwp.py` 的 reward 内部已有 `geom_box_sdf_min()`，但它是局部函数，只服务 soft penalty。E088 需要把它提成可复用函数，至少支持：

- 输入一组 robot geom ids；
- 使用 `object_collision` box 的 pose 和 half extents；
- 输出每个 sample 当前 frame 的 `min_sdf`；
- 输出每个 sample 的 `violation = min_sdf < threshold`。

新增 info：

| info | shape | 用途 |
|---|---|---|
| `cem_gate_min_sdf` | `(N,)` | 每帧每 sample 的 upper-body/object 最小 SDF |
| `cem_gate_violation` | `(N,)` | 当前帧是否违规 |
| `cem_gate_violation_depth` | `(N,)` | `threshold - min_sdf` 的正部 |

### C. 在 rollout 内聚合 sample 级 gate 指标

`spider/optimizers/sampling.py` 的 rollout 已经把每步 reward info stack 成 `(T,N)` 再求 mean。E088 需要保留 gate 的 sample 级聚合，不能只求 mean。

建议在 rollout 末尾额外输出：

| rollout_info | 计算 |
|---|---|
| `sample_gate_min_sdf` | `cem_gate_min_sdf` 在 horizon 上取 min |
| `sample_gate_violation_pct` | `cem_gate_violation` 在 horizon 上取 mean |
| `sample_gate_violation_depth_mean` | violation depth 在 horizon 上取 mean |
| `sample_gate_valid_mask` | `min_sdf >= threshold` 且 `violation_pct <= max_pct` |

`sampling_fast.py` 的 select-best 路径也要同步，否则 fast path 和 normal path 行为会不一致。

### D. 修改 elite selection

需要修改 `spider/optimizers/sampling.py` 与 `spider/optimizers/sampling_fast.py`。

当前 `_compute_weights_impl()` 只看 `rews`，并且会把 `inf` 当作异常替换成最小 reward。E088 不建议只把 invalid reward 设为 `-inf`，因为这会让 top-k、elite_std、trace top-k 的行为不够透明。

建议新增显式 mask 路径：

```text
valid_mask = rollout_info["sample_gate_valid_mask"]
valid_count = valid_mask.sum()

if valid_count >= max(1, min_valid_frac * num_samples):
    elite_pool = valid samples
else:
    fallback_used = True
    elite_pool = sort by violation_depth asc, violation_pct asc, reward desc

top-k 和 softmax 只在 elite_pool 内做。
```

同时记录：

| info | 含义 |
|---|---|
| `cem_gate_valid_frac` | 本轮有效 sample 比例 |
| `cem_gate_fallback_used` | 是否 fallback |
| `cem_gate_selected_valid_frac` | 被用于 elite 的样本中 valid 比例 |
| `cem_gate_min_sdf_mean/min` | 诊断 gate 严格程度 |
| `cem_gate_violation_pct_mean/max` | 诊断违规频率 |

### E. 新增绝对 object clearance reward

在 `get_reward()` 里新增默认关闭的分项，不替代旧 `object_lift_rew`，先并行存在：

```text
obj_bottom = object_collision_center_z - object_half_z
clearance = obj_bottom - object_clearance_floor_z
below = clamp(min_m - clearance, min=0)
above = clamp(clearance - max_m, min=0)
```

建议形式：

```text
object_clearance_rew =
  scale * exp(-abs(clearance - target_mid) / sigma) * window_gate

object_clearance_penalty =
  -penalty_scale * (below + 0.25 * above) * window_gate
```

其中 `window_gate` 先用 contact mask 的 max，必要时加 time window。这样 pre-contact 不会强迫箱子离地，搬运窗口内才要求 bottom clearance。

### F. 评估与可视化扩展

在 E088 eval 中补充：

- head/torso/pelvis/shoulder/elbow 分 geom penetration 统计；
- object bottom clearance mean/min/max；
- object bottom below threshold 帧比例；
- gate 每轮 valid/fallback 曲线；
- reward breakdown 中新增 clearance 分项；
- contact sheet 标注每个视频帧的 head pen、upper pen、bottom clearance。

## 实验矩阵

只针对 main case：

```text
d003_box021_20231018_029_p2
```

先用 E087 中更稳的 `10kg` 派生 scene，不再优先使用 `5kg`。

| 组 | 配置 | 目的 |
|---|---|---|
| E088A | `10kg + raw target + hard gate only` | 隔离 hard gate 是否能压住穿箱 |
| E088B | `10kg + hard gate + low contact/object` | 避免 contact/qpos 正项重新把身体拉去压箱 |
| E088C | `10kg + hard gate + low contact/object + absolute clearance` | 验证 lift/floor reward 是否能让箱体离地行为进入优化 |
| E088D 可选 | `29.632kg + E088C` | 如果 E088C 在 10kg 有明显改善，再检查原始质量是否仍完全不可行 |

初始参数建议：

| 参数 | E088A | E088B | E088C |
|---|---:|---:|---:|
| mass | `10kg` | `10kg` | `10kg` |
| `cem_safety_gate_min_sdf_m` | `-0.005` | `-0.005` | `-0.005` |
| `cem_safety_gate_max_violation_pct` | `0.0` | `0.0` | `0.0` |
| `contact_hdmi_gain` | 沿 E087B | `1.0` | `1.0` |
| `task_obj_pos/rot_rew_scale` | 沿 E087B | `0.1/0.1` | `0.1/0.1` |
| `object_clearance_min_m` | off | off | `0.04` |
| `object_clearance_max_m` | off | off | `0.18` |
| `object_clearance_penalty_scale` | off | off | `10.0` |
| `object_clearance_rew_scale` | off | off | `1.0` |

## 成功标准

main gate：

- `case_window_sim_head_collision_object_penetration_pct < 5%`
- `case_window_sim_upperbody_object_penetration_pct < 15%`
- `case_window_sim_lh_floor_contact_pct <= 5%`
- `case_window_sim_rh_floor_contact_pct <= 5%`
- `case_window_sim_contact_frames_pct >= 40%`，理想为 `>=50%`
- `case_window_obj_err_mean_m <= 0.85m`，不能靠完全弃箱换安全
- `object_bottom_below_4cm_pct` 明显低于 E087B/E087C
- `cem_gate_valid_frac` 不能长期接近 0；fallback 不能成为主要更新路径
- 视频中不能出现头/胸/肩/肘作为主要物体支撑。

guard 规则：

- 只有 E088A/B/C 中至少一组 main 的 head/upper 指标显著改善，才跑 `box023_p2` guard；
- guard 主要验证 hard gate 不破坏已知较可看的 case；
- 如果 main 全部 gate valid frac 接近 0，不跑 guard，先回到 target/pose 可行性诊断。

## 失败解释规则

| 现象 | 解释 | 下一步 |
|---|---|---|
| `valid_frac` 长期接近 0，fallback 频繁 | raw target / reference pose 对 G1 在不穿箱条件下不可达 | 做 kinematic IK / target feasibility，而不是继续调 reward |
| head/upper 降了，但 contact 也崩到 `<20%` | 旧 raw hand target 需要身体穿透才能满足 | 改 contact target 或引入 reachability-aware target，不接 CEM |
| clearance 提高，但手和箱子不接触 | 物体 actuator/PD 或碰撞被其他项驱动，不是真实搬运 | 降 object tracking，强化 semantic contact 或判失败 |
| safety 好、contact 还在、object error 稍差 | 可作为后续 CEM-to-RL 或更长 horizon seed | 再跑 guard 与多 case |

## 执行顺序

1. 写 E088 plan 并冻结 E087 结论。
2. 实现 config 默认关闭字段。
3. 把 upper-body/object SDF 提成可复用函数，并在 reward info 中输出 gate 指标。
4. 修改 `sampling.py` / `sampling_fast.py` 的 top-k/elite selection，加入 explicit valid mask 和 fallback。
5. 增加 object clearance reward/penalty，默认关闭。
6. 生成 E088 overrides 和 train/eval 脚本。
7. 先跑 4-step smoke，检查 gate stats、Hydra compose、旧 config backward compatibility。
8. 本地先跑 E088A；若 `valid_frac` 不为 0，再远程并行跑 E088B/E088C。
9. 回收结果，做 reward breakdown、gate 曲线、contact sheet 和视频检查。
10. 写 E088 log，更新 tracker。

## 预期产物

| 类型 | 路径 |
|---|---|
| plan | `workspace/core4d/plan/94_E088_hard_safety_gate_lift_floor_plan.md` |
| overrides | `examples/config/override/core4d_E088*.yaml` |
| train script | `workspace/core4d/scripts/train/train_E088.sh` |
| eval script | `workspace/core4d/scripts/eval/eval_E088.py` |
| results | `workspace/core4d/results/E088/` |
| log | `workspace/core4d/log/110_E088_hard_safety_gate_lift_floor_results.md` |
