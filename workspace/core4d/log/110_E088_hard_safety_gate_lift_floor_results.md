# E088 Hard Safety Gate 与绝对物体离地奖励结果

日期：2026-05-28
对应计划：`workspace/core4d/plan/94_E088_hard_safety_gate_lift_floor_plan.md`

## 结论

E088 已完成 `d003_box021_20231018_029_p2` 的 `10kg` 派生 scene 三组 full CEM。三组均未通过 gate，`accepted_variants=[]`，不建议接后续 RL。

1. Hard gate 机制生效，但在 Box021 main 上可行样本很少，CEM 大量依赖 fallback 更新。E088A valid frac mean 只有 `0.0029`，fallback `90.7%`；E088B valid frac 为 `0`，fallback `92.0%`；E088C valid frac 提到 `0.280`，但 fallback 仍有 `61.6%`。
2. 绝对 clearance reward 确实产生了优化信号，但被 CEM 找到了错误局部解。E088C 的 bottom clearance mean 达到 `5.5cm`，低于 `4cm` 的帧降到 `30.2%`，reward breakdown 中 `object_clearance_rew=0.440`、`object_clearance_penalty=-0.165`；但视觉是箱子倾倒/竖起、机器人侧倒，不是真实搬运。
3. A/B/C 都没有同时满足接触、物体误差、上半身安全和手不撑地。A 降低了 head penetration 但 pelvis 崩；B pelvis 稳但 head/upper penetration 高；C 物体误差最低但通过翻箱/摔倒实现。

## 实现与检查

| 文件 | 变更 |
|---|---|
| `spider/config.py` | 新增默认关闭的 `cem_safety_gate_*` 与 `object_clearance_*` 配置，并解析 upper-body geoms |
| `spider/simulators/mjwp.py` | 提取 object box SDF helper；输出 `cem_gate_*`、`sample_gate_*` 源信息；新增绝对 clearance reward/penalty |
| `spider/optimizers/sampling.py` | normal CEM 在 elite selection 前应用 valid mask；valid 不足时用 least-violation fallback |
| `spider/optimizers/sampling_fast.py` | fast CEM 同步 hard gate / fallback 逻辑 |
| `examples/run_mjwp.py` | 修复 info 聚合只保留首 tick keys 的问题，避免 gate/clearance 指标被 warmup tick 丢掉 |
| `workspace/core4d/scripts/E088/*` | 生成 override、远程运行、回收和评估脚本 |
| `workspace/core4d/scripts/E087/reward_breakdown.py` | 补充 `object_clearance_rew/object_clearance_penalty` 分项 |

已通过检查：

- `py_compile`: `examples/run_mjwp.py`、`spider/config.py`、`spider/simulators/mjwp.py`、`spider/optimizers/sampling.py`、`spider/optimizers/sampling_fast.py`、E088 eval/generator、reward breakdown。
- `bash -n`: E088 train/remote/pull 脚本。
- 本地 smoke：三组 4-step 均通过；E088C 24-step 进入 CEM 后 npz 含 `cem_gate_*`、`sample_gate_*`、`object_clearance_*` keys。

## 运行方式

远程执行：

| 资源 | 内容 |
|---|---|
| host | `spider-remote` |
| remote repo | `/home/xiayb/pHRI_workspace/spider` |
| tmux | `E088_gate_clearance` |
| GPU0 | `E088A_m10_gate_main` |
| GPU1 | `E088B_m10_gate_low_main`，完成后串行 `E088C_m10_gate_clearance_main` |

结果已回收到本地并完成合并评估。

## 主指标

| Variant | 设计 | contact % | obj mean/max m | pelvis min m | head pen % | upper pen % | LH/RH floor % | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E087B_m10_rawtarget_main | 对照，10kg raw target | 82.95 | 0.735 / 1.218 | 0.629 | 69.77 | 75.97 | 0 / 0 | fail |
| E088A_m10_gate_main | hard gate only | 82.17 | 0.695 / 1.148 | 0.181 | 27.91 | 55.04 | 10.85 / 0 | fail |
| E088B_m10_gate_low_main | hard gate + low contact/object | 70.54 | 0.722 / 1.198 | 0.643 | 71.32 | 75.19 | 0 / 0 | fail |
| E088C_m10_gate_clearance_main | B + absolute clearance | 73.64 | 0.612 / 0.985 | 0.173 | 19.38 | 80.62 | 40.31 / 0 | fail |

Gate 规则：

```text
contact >= 40%
object mean <= 0.85m
head penetration < 5%
upperbody penetration < 15%
left/right hand floor <= 5%
cem_gate_valid_frac > 0
```

E088 三组均未通过，`workspace/core4d/results/E088/e088_gate_summary.json`：

```json
{
  "accepted_variants": [],
  "gate_rule": "contact>=40, obj_mean<=0.85m, head<5%, upper<15%, hand-floor<=5%, gate_valid_frac>0",
  "num_gate_accept": 0,
  "num_results": 3
}
```

## Gate 统计

| Variant | valid frac mean/min/max | fallback used % | sample min SDF min m | violation pct mean |
|---|---:|---:|---:|---:|
| E088A | 0.0029 / 0 / 0.4336 | 90.71 | -0.04917 | 0.4285 |
| E088B | 0 / 0 / 0 | 92.00 | -0.04913 | 0.7735 |
| E088C | 0.2797 / 0 / 1.0000 | 61.63 | -0.04918 | 0.2909 |

解释：

- E088A/B 的 hard gate 基本把 CEM 的有效样本空间筛没了；优化主要走 fallback，不是稳定地从 valid elite 学到分布。
- E088C 因为 clearance 改变了物体和身体状态，有更多 sample 在 gate 上暂时有效，但最终仍落到翻箱/倒地局部解。

## Object Lift / Floor / Clearance 口径

旧的 `object_lift_rew` 与 `object_floor_penalty` 是相对参考物体 bottom 的项：

```text
object_lift_rew = scale * exp(-abs(sim_object_bottom_z - ref_object_bottom_z) / sigma)
object_floor_penalty = -scale * max((ref_object_bottom_z - margin) - sim_object_bottom_z, 0)
```

因此如果 ref 的箱底本来接近地面，或 sim/ref 都处于低位错误区间，这两个项不会表达“必须离地”，只会表达“和 ref 一样低”。这正是 E087/E088 reward breakdown 中它们贡献很小的原因。

E088C 新增的 `object_clearance_rew/object_clearance_penalty` 是绝对世界坐标口径：

```text
clearance = object_bottom_z - floor_z
target interval = [0.04m, 0.18m]
```

主结果：

| Variant | clearance mean/min m | below 4cm % | clearance rew mean | clearance penalty mean |
|---|---:|---:|---:|---:|
| E088A | -0.0439 / -0.0684 | 100.00 | 0 | 0 |
| E088B | -0.0588 / -0.0662 | 100.00 | 0 | 0 |
| E088C | 0.0551 / -0.0666 | 30.23 | 0.4397 | -0.1654 |

这个结果支持两个判断：

1. 绝对 clearance 的信号是有效的，能让物体 bottom 从长期低于地面附近变成部分离地。
2. 单独加入 clearance 不够，因为 CEM 可以通过翻箱、身体压箱、手撑地来满足高度区间，而不是形成真实手部承重。

## Reward Breakdown

case window 为 frame `21-149`。

| Variant | qpos | contact | task_obj | robot_obj_penalty | object_lift | object_floor | clearance rew | clearance penalty | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E088A | 1.345 | 1.844 | 0.361 | -0.036 | 0.0067 | 0.000 | 0.000 | 0.000 | 3.462 |
| E088B | 1.924 | 0.519 | 0.116 | -0.050 | 0.0067 | 0.000 | 0.000 | 0.000 | 2.439 |
| E088C | 1.334 | 0.367 | 0.072 | -0.043 | 0.0063 | 0.000 | 0.440 | -0.165 | 1.956 |

旧 `object_lift_rew` 在三组都只有约 `0.006-0.007` 的均值，`object_floor_penalty` 为 `0`，验证了 plan 94 的诊断：旧 lift/floor 不是有效的离地约束。

## 可视化复核

关键帧总览：

- `workspace/core4d/results/E088/keyframes/E088_all_variants_sheet.jpg`

视频：

- `workspace/core4d/results/E088/E088A_m10_gate_main.mp4`
- `workspace/core4d/results/E088/E088B_m10_gate_low_main.mp4`
- `workspace/core4d/results/E088/E088C_m10_gate_clearance_main.mp4`

观察：

- E088A: 中段机器人趴在箱体上方，head/upper penetration 虽低于 E087B，但 pelvis collapse，左手后段有撑地。
- E088B: pelvis 更稳定，但机器人深度弯腰，头颈/躯干仍压到箱体区域，上半身安全没有解决。
- E088C: 箱体被抬高主要来自倾倒/竖起，机器人侧倒且左手触地；这不是可接 RL 的搬运 seed。

## 诊断

E088 说明当前瓶颈不是简单的物体质量或单个 reward 权重。

1. Hard gate 把错误样本从 elite 里过滤掉后，Box021 main 的可行样本空间非常稀薄。A/B 长期 fallback，说明 CEM 在当前 seed/target/noise 下很难自然采到“手承重且上半身不穿箱”的轨迹。
2. 绝对 clearance 解决了旧 lift/floor 不工作的口径问题，但没有约束物体姿态、手部承重和身体安全之间的因果关系，所以会产生翻箱局部解。
3. `object_lift_rew/object_floor_penalty` 的失败是设计口径问题，不是实现没跑到；它们围绕 `ref_bottom`，而不是围绕地面绝对 clearance。

## 下一步建议

不建议继续在 E088C 上直接接 RL。更合理的下一步是先做两类短实验：

1. **Gate feasibility audit**：不用 full CEM，直接统计 ref、kinematic replay、small-noise samples 在 head/torso/pelvis/shoulder/elbow SDF gate 下的有效率，按 geom 和时间窗口拆开，确认是 gate 过严、target 不可达，还是当前 CEM seed 分布太差。
2. **Anti-tip clearance gate**：保留绝对 clearance，但同时加入 object tilt/orientation hard gate、hand-floor hard gate、upperbody hard gate，并把 clearance 只限于接触窗口。目标是证明“离地”不能靠翻箱/摔倒达成。

只有当这两类实验能产生低穿透、低翻箱、手不撑地的 seed，才值得继续做 CEM-to-RL。

## 结果路径

| 类型 | 路径 |
|---|---|
| metrics | `workspace/core4d/results/E088/comparison.csv` |
| gate summary | `workspace/core4d/results/E088/e088_gate_summary.json` |
| aggregate | `workspace/core4d/results/E088/aggregate_summary.json` |
| keyframes | `workspace/core4d/results/E088/keyframes/` |
| reward breakdown | `workspace/core4d/results/E088/reward_breakdown/` |
| logs | `logs/E088/` |
