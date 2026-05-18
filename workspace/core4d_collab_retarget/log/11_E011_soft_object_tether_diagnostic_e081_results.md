# E011 结果：soft object tether 诊断 E081 所需外部 coupling

日期：2026-05-18

## 状态

E011 plan、代码接入、脚本、smoke、本地 full、远程 full、回收、关键帧检查和 9-variant 显式重评均已完成。结论：E011 不是 work，没有任何 main 变体达到 E081 transport gate；但它很好地解释了 E006 视频里“只旋转、不平移”的失败模式。

最关键的诊断结论是：COM-level spring 可以把 E006 的 off-COM 旋转捷径压下去，并显著恢复水平平移；但即使用 `kp=100` 或 `gravity_scale=1`，object error 仍远高于 E081，或者手端 contact 掉出门槛。这说明剩余瓶颈不是“物体能不能被外力推走”，而是 robot-side hand/support pose 与 object 姿态/高度没有形成真实闭环。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E011_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E011.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E011.py --all
bash workspace/core4d_collab_retarget/scripts/run_E011_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E011.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E011_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E011.py \
  E011_box025_p2_com_xyz_k20 \
  E011_box025_p2_com_xyz_k50 \
  E011_box025_p2_com_xyz_k100 \
  E011_box025_p2_com_xyz_k50_g1 \
  E011_box025_p2_com_xyz_k50_rot1 \
  E011_box025_p2_ypos_k20_vmax2_com_k25 \
  E011_box025_p2_ypos_k20_vmax2_com_k50 \
  E011_box023_p2_com_xyz_k50 \
  E011_box023_p2_com_xyz_k100
```

最终结果以上面 9 个 full NPZ 显式重评为准。main NPZ 大小约 `1.09-1.11MB`，guard NPZ 大小约 `1.20MB`，不是 4-step smoke。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/11_E011_soft_object_tether_diagnostic_e081_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E011/` |
| Logs | `logs/core4d_collab_retarget/E011/` |
| Comparison | `workspace/core4d_collab_retarget/results/E011/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E011/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E011/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E011/keyframes/` |
| Visual montage | `workspace/core4d_collab_retarget/results/E011/keyframes/e011_visual_montage.jpg` |

## Full 汇总

最终 full aggregate：

```json
{
  "num_results": 9,
  "num_main_results": 7,
  "num_guard_results": 2,
  "num_freejoint_parity_ok": 9,
  "num_partner_force_metrics_present": 9,
  "num_support_proxy_metrics_present": 2,
  "num_main_proxy_timebase_ok": 2,
  "num_main_proxy_support_tracking_ok": 2,
  "num_main_reaches_E081_transport": 0,
  "num_main_beats_or_matches_E081_majority": 4,
  "num_main_improves_E008_best": 1,
  "num_guard_stable": 1,
  "diagnostic_classes": {
    "insufficient_coupling": 7,
    "robot_side_blocked": 2
  }
}
```

关键指标：

| Variant | Role | 诊断 | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | force mean/max (N) | 结论 |
|---------|------|------|------------------|--------|---------|-----------|----------|---------|--------------------|------|
| `E011_box025_p2_com_xyz_k20` | main | insufficient | `0.533 / 1.035` | `86.7` | `58.4` | `1.2` | `0.617` | `4.9` | `26.3 / 35.1` | 去掉旋转捷径，但 coupling 太弱 |
| `E011_box025_p2_com_xyz_k50` | main | insufficient | `0.451 / 0.851` | `85.0` | `62.4` | `0.0` | `0.832` | `2.7` | `30.1 / 43.4` | 平移明显恢复，obj error 仍高 |
| `E011_box025_p2_com_xyz_k100` | main | robot-side blocked | `0.340 / 0.673` | `86.1` | `61.3` | `0.0` | `0.905` | `3.5` | `32.6 / 53.1` | E011 best，仍未达 E081 |
| `E011_box025_p2_com_xyz_k50_g1` | main | insufficient | `0.350 / 0.694` | `75.1` | `22.5` | `0.0` | `0.971` | `11.1` | `49.1 / 58.7` | 竖直支撑强，手端脱开 |
| `E011_box025_p2_com_xyz_k50_rot1` | main | insufficient | `0.427 / 0.835` | `83.2` | `56.6` | `0.0` | `0.763` | `2.8` | `28.9 / 42.7` | 弱 rot tether 不解决 error |
| `E011_box025_p2_ypos_k20_vmax2_com_k25` | main | insufficient | `0.420 / 0.792` | `87.9` | `69.4` | `0.0` | `0.686` | `24.8` | `7.9 / 19.7` | E008+弱 COM 反而增旋转 |
| `E011_box025_p2_ypos_k20_vmax2_com_k50` | main | insufficient | `0.378 / 0.718` | `83.8` | `65.9` | `4.0` | `0.811` | `13.9` | `11.9 / 27.8` | xy 改善但 obj 仍差 |
| `E011_box023_p2_com_xyz_k50` | guard | insufficient | `0.681 / 1.235` | `55.3` | `51.3` | `0.0` | `0.774` | `10.7` | `43.0 / 70.2` | stable guard，但无 transport |
| `E011_box023_p2_com_xyz_k100` | guard | robot-side blocked | `0.351 / 0.604` | `57.3` | `33.3` | `18.7` | `0.963` | `9.2` | `42.6 / 83.9` | tracking 改善但摔倒/干涉 |

E081 main baseline：obj `0.143/0.271m`、hand `89.0%`、floor `59.5%`、leg interference `7.5%`。E011 best main `com_xyz_k100` 仍比 E081 obj mean 高 `0.197m`、obj max 高 `0.402m`，因此不能判 work。

## 可视化观察

已检查 E011 关键帧拼图：

`workspace/core4d_collab_retarget/results/E011/keyframes/e011_visual_montage.jpg`

实际观察：

- COM-only `k20/k50/k100` 与 E006 相比，sim 侧箱体不再主要绕一端翻转；随着 kp 增大，箱体水平位移明显接近 ref，视觉上从“原地旋转/贴地”转为“被整体带走”。
- `k100` 后期仍能看到 sim 侧机器人与箱体相对姿态不完全对齐，箱体位置比 ref 更偏，量化上对应 obj `0.340/0.673m`，不是 E081 级同步搬运。
- `k50_g1` 后期箱体明显离地并走得更远，但机器人手端没有稳定跟住；这与 hand contact 只有 `75.1%`、floor 只有 `22.5%` 一致，属于外部竖直支撑强、robot-side 闭环弱。
- `E008 best + COM` 两条在中后期又出现明显姿态偏差：`k25` 旋转达到 `24.8deg`，`k50` 虽 rot 回到 `13.9deg`，但 object error 仍不达标。
- guard `box023_k50` 姿态稳定但无有效协作搬运；`box023_k100` 的后期关键帧显示机器人姿态崩坏/摔倒，与 pelvis min `0.113m`、leg interference `18.7%` 一致。

## 与 E006 失败现象的关系

用户指出 E006 视频像“物体不平移，只旋转”。E006 后验指标确认：参考 object xy 位移约 `1.57m`、旋转约 `2deg`，而 E006 main 实际 xy 只有 `0.21-0.50m`、旋转 `20.6-47.6deg`。

E011 给出更直接的对照：

1. 只要把外力从 off-COM support point 改成 COM-level spring，旋转立刻被压低：`k20/k50/k100` rot 只有 `4.9/2.7/3.5deg`。
2. xy ratio 随 COM kp 明显上升：`0.617 -> 0.832 -> 0.905`。这说明 E006 的“只旋转、不平移”不是物体完全推不动，而是 off-COM wrench + 地面/手端弱闭环诱导了旋转捷径。
3. 即使 `k100` 已能恢复平移并略优于 E008 best，obj error 仍停在 `0.340/0.673m`。因此下一步不能只继续加外力或 stiffness，必须让机器人手端和 partner 端构成更真实的双点/双手闭合。
4. `g1` 进一步说明竖直支撑可以把 floor contact 降到 `22.5%`，但如果 robot hand 没有跟住，视觉和指标都会变成“外部支撑把物体带走”，不是协作搬运。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 COM-level soft tether 能消除 E006 的 off-COM 旋转捷径 | ✅ 通过 | COM-only k20/k50/k100 rot `4.9/2.7/3.5deg`，远低于 E006 `20.6-47.6deg` |
| C2 中等/强 kp 可把 main 拉近 E081 | ⚠️ 部分通过 | k100 达 xy `0.905`、obj `0.340/0.673m`，优于 E008 best 但离 E081 `0.143/0.271m` 仍远 |
| C3 若 robot-side 不参与，tether 成功也不能判 work | ✅ 通过 | g1 xy `0.971`、floor `22.5%`，但 hand `75.1%`，未达 transport gate |
| C4 记录外力代价 | ✅ 通过 | 9/9 有 partner force metrics；main force mean 约 `7.9-49.1N`，均未爆炸 |
| C5 true-freejoint parity 不破坏 | ✅ 通过 | 9/9 `nu=29`、`nq_obj=7`、object actuator empty，contact guidance off |

## 结论

E011 不是最终算法，但它把 E006 的失败原因分清楚了：

- E006 的“只旋转、不平移”主要来自 off-COM connector wrench 与弱 robot-side 闭环的耦合；不是单纯 kp 太小。
- COM-level 外部 coupling 能恢复平移，且 effort 并不夸张；`k100` 是当前 best diagnostic。
- 但只靠外部 coupling 不能达到 E081 级同步搬运；强竖直支撑甚至会降低 hand contact，变成外部支撑主导。
- `E008 + COM` 的局部修补路线失败：k25 增加旋转，k50 只改善 xy 不改善 object error。

下一步不应继续扫 COM kp / gravity scale。E012 应转向结构化 robot-side + partner-side 闭合：优先做双点/双手 virtual partner constraint 或 robot-side hand/support pose shaping，让物体两端高度、相对姿态和手端接触同时受控，而不是继续单点/单 COM 外力。
