# E084 Plan: Box021 constraint groups after E083A

日期：2026-05-28

## Context

E083A 已完整验证：

- upper-body-object collision pairs 对 `box023_p2` guard 安全；
- 但 3 个 Box021 main 仍全部失败；
- 失败模式从 E082 的“深穿箱/倒伏”变成 E083 的“趴箱/浅穿/手撑地/箱子贴地”。

因此下一步不再继续只改 scene contact pairs，而是进入 reward/constraint 设计。E084 规划为 3 组对照实验，先用最典型失败 `20231018_029_p2` 做 main，再配 `box023_p2` guard。每组 2 个 case，共 6 个 full CEM run。

如果某组在 `20231018_029_p2` 上明显改善且 guard 不退化，再进入 E085 扩展到 3 个 Box021 main。

## 实验矩阵

| Group | 名称 | Main | Guard | GPU 分配 |
|---|---|---|---|---|
| E084A | safety penalty | `20231018_029_p2` | `box023_p2` | local GPU0 串行 |
| E084B | upright / ctrl trust | `20231018_029_p2` | `box023_p2` | remote GPU0 串行 |
| E084C | semantic hand contact + lift | `20231018_029_p2` | `box023_p2` | remote GPU1 串行 |

所有组都复用 E083 的 `*_upperobj_e083` 派生 scene，不再新增 scene contact pair。

运行原则：

- 本地和远程 3 卡并行；
- 远程按 `experiment-planning-zh/remote-execution.md`，使用 tmux，不 kill 其他实验；
- 每个 group 的 main/guard 在同一 GPU 上串行，便于对比和避免结果覆盖。

## Group A: safety penalty

### Hypothesis

E083A 失败是因为 CEM 没有直接惩罚错误物理接触。只要强惩罚 upper-body-object penetration、hand-floor contact 和 pelvis collapse，CEM 会避免“趴箱/手撑地”局部解；若 hand-object contact 随之消失，则说明当前 contact objective 不足以找到可行搬运姿态。

### 改动

新增配置字段：

```text
upperbody_object_penalty_scale
upperbody_object_penalty_margin
upperbody_object_penalty_geoms
hand_floor_penalty_scale
hand_floor_penalty_margin
```

在 `spider/simulators/mjwp.py` reward 中基于 geom/world pose 计算：

- head/torso/pelvis/shoulder/elbow 到 object AABB 的 adjusted SDF；
- SDF < margin 时给负 reward；
- lh/rh 接近 floor 或低于 floor margin 时给负 reward；
- 恢复 `stability_penalty_scale`，建议初值 `1.0`、threshold `0.60`。

建议初值：

```yaml
upperbody_object_penalty_scale: 5.0
upperbody_object_penalty_margin: 0.03
hand_floor_penalty_scale: 3.0
hand_floor_penalty_margin: 0.03
stability_penalty_scale: 1.0
stability_penalty_threshold: 0.60
```

### 成功标准

- main: head/torso penetration 明显低于 E083，hand-floor 接近 0，pelvis min >= `0.55m`；
- main: object/contact 不允许完全崩掉，sim contact 不低于 E083 的一半；
- guard: `box023_p2` 不退化，upperbody penetration 仍为 0。

## Group B: upright / ctrl trust

### Hypothesis

E083A 的根因不只是缺惩罚，而是 contact/object objective 太强，CEM 通过大幅 robot ctrl 偏离牺牲全身姿态。增强姿态/控制 trust region、降低接触牵引，可以阻止“低髋趴箱”。

### 改动

不新增复杂 SDF penalty，主要调已有机制：

```yaml
ctrl_ref_guard_scale: 1.0        # E083 为 0.5
ctrl_ref_guard_sigma: 0.15       # E083 为 0.25
ctrl_ref_guard_start_eval_time: 0.6
ctrl_ref_guard_end_eval_time: 3.0
stability_penalty_scale: 1.0
stability_penalty_threshold: 0.60
task_body_rew_scale: 2.0
task_body_names:
  - pelvis
  - torso_link
  - head_link
  - left_ankle_roll_link
  - right_ankle_roll_link
task_obj_use_exp: true
task_obj_pos_rew_scale: 0.5
task_obj_rot_rew_scale: 0.3
contact_hdmi_gain: 3.0
```

具体 body 名需要在实现时用 MuJoCo name audit 确认，若 `torso_link/head_link` 名称不同，以实际 model body name 为准。

### 成功标准

- main: pelvis min、torso/head posture 明显改善；
- main: robot ctrl Linf max 明显低于 E083；
- 允许 contact 暂时下降，但不能出现远离箱体/完全不接触；
- guard 不退化。

## Group C: semantic hand contact + lift

### Hypothesis

E083A 中 sim contact 高但语义错误：身体压箱、箱子贴地。需要把 reward 从“接近物体”改成“手部主导接触 + 箱体离地/不贴地”，否则 CEM 会继续用错误接触满足数值目标。

### 改动

新增或组合以下机制：

```text
semantic_hand_contact_gate
object_lift_rew_scale
object_lift_margin
object_floor_penalty_scale
```

建议 reward 形式：

- contact_hdmi reward 只在 upperbody safe、hand-floor safe 时生效；
- 对 object bottom 低于 ref bottom - margin 给 penalty；
- 或用 `exp(-abs(object_bottom - ref_bottom)/sigma)` 约束 lift trajectory；
- upperbody penetration 不一定强罚到 Group A 的程度，但作为 contact gate。

建议初值：

```yaml
contact_hdmi_gain: 5.0
semantic_contact_gate_upperbody_margin: 0.02
semantic_contact_gate_hand_floor_margin: 0.02
object_lift_rew_scale: 2.0
object_lift_sigma: 0.05
object_floor_penalty_scale: 2.0
```

### 成功标准

- main: object floor-contact 明显低于 E083；
- main: object bottom gap vs ref 改善至少 5cm；
- main: hand-object contact 不能靠上身/头部接触替代；
- guard 不退化。

## 评估

E084 eval 必须继承并扩展 E083 指标：

- `upperbody_object_penetration_pct`
- `head/torso/pelvis penetration`
- `hand_floor_contact_pct`
- `leg_box_interference_pct`
- `object_floor_contact_pct`
- `object_bottom_gap_vs_ref`
- `robot_ctrl_linf_max`
- `case_window_obj_err_mean/max`
- `semantic_visual_label`

所有结果必须生成：

```text
workspace/core4d/results/E084/comparison.csv
workspace/core4d/results/E084/aggregate_summary.json
workspace/core4d/results/E084/keyframes/contact_sheets/
logs/E084/
```

并使用 subagent high 做视觉复核。

## 决策树

| 结果 | 下一步 |
|---|---|
| A 成功，B/C 不成功 | E085 扩展 A 到 3 个 Box021 main + guard |
| B 成功，A/C 不成功 | 说明主要是姿态/控制 trust 问题，E085 扩展 B |
| C 成功，A/B 不成功 | 说明主要是接触语义/lift 目标缺失，E085 扩展 C |
| A/B/C 都只改善 guard 或都失败 | 停止 Box021 CEM 小调参，转数据可行性/IK/原始 OmniRetarget seed 路线 |
| A 消除穿模但 contact 归零 | 需要组合 A+C，不能单独扩大 |
| B 稳住但不接触 | 需要组合 B+C，降低 contact trade-off |

## 风险

| 风险 | 处理 |
|---|---|
| 新 reward 代码实现成本高 | 先做 A/B，C 可作为第二阶段 |
| penalty 让 CEM 完全不碰箱 | 记录为有效排除错误局部解，但不是成功；再组合 hand semantic |
| guard 退化 | 回退该组，不能推广 |
| 指标再次误判 | 视觉 sheet + subagent high 是硬要求 |
