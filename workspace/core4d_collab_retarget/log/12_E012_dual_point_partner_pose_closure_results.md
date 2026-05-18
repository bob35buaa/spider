# E012 结果：dual-point partner pose closure 诊断

日期：2026-05-18

## 状态

E012 plan、dual-point 虚拟协作力实现、smoke、本地 2 条 full、远程 6 条 full、结果回收、8 条显式重评和关键帧检查均已完成。结论：E012 不是 work；6 条 main 没有任何一条达到 E081 transport gate，也没有任何一条满足 `pose_closure_helped`，2 条 guard 均不稳定。

最重要的诊断结论是：dual-point force 没有把 E011 的 COM spring 推进到 E081 级闭环，反而重新打开了 off-COM torque shortcut。弱 coupling 推不动，强 coupling / 大力臂 / 更强 object reward 会让箱体大角度旋转，表面上改善一部分 object/xy/floor 指标，但不是协作搬运。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E012_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E012.py --all
bash workspace/core4d_collab_retarget/scripts/run_E012_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E012_remote_results.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh eval \
  E012_box025_p2_dualy_x20_k100_g05 \
  E012_box025_p2_dualy_x20_k100_g05_obj3 \
  E012_box025_p2_dualy_x20_k50_g05 \
  E012_box025_p2_dualy_x30_k100_g05 \
  E012_box025_p2_dualy_x20_k100_g08 \
  E012_box025_p2_dualy_x20_k150_g05 \
  E012_box023_p2_dualx_y10_k50_g05 \
  E012_box023_p2_dualx_y10_k100_g05
```

最终结果以上面 8 条 full NPZ 显式重评为准。所有 full NPZ 大小约 `1.1-1.2MB`，不是 4-step smoke 的 `20KB` 占位文件。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/12_E012_dual_point_partner_pose_closure_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E012/` |
| Logs | `logs/core4d_collab_retarget/E012/` |
| Comparison | `workspace/core4d_collab_retarget/results/E012/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E012/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E012/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E012/keyframes/` |
| Visual montage | `workspace/core4d_collab_retarget/results/E012/keyframes/e012_visual_montage.jpg` |

## Full 汇总

最终 full aggregate：

```json
{
  "num_results": 8,
  "num_main_results": 6,
  "num_guard_results": 2,
  "num_dual_points_config_ok": 8,
  "num_freejoint_parity_ok": 8,
  "num_partner_force_metrics_present": 8,
  "num_main_reaches_E081_transport": 0,
  "num_main_pose_closure_helped": 0,
  "num_guard_stable": 0,
  "diagnostic_classes": {
    "guard_unstable": 2,
    "insufficient_coupling": 2,
    "rotation_shortcut": 4
  }
}
```

关键指标：

| Variant | Role | 诊断 | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | force mean/max (N) | torque max (Nm) | 结论 |
|---------|------|------|------------------|--------|---------|-----------|----------|---------|--------------------|-----------------|------|
| `E012_box025_p2_dualy_x20_k50_g05` | main | insufficient | `0.466 / 0.899` | `86.1` | `63.0` | `4.0` | `0.815` | `9.0` | `30.2 / 72.7` | `30.0` | 姿态正常些，但 coupling 明显不足 |
| `E012_box025_p2_dualy_x20_k100_g05` | main | insufficient | `0.362 / 0.660` | `80.9` | `64.7` | `1.2` | `0.788` | `20.6` | `33.9 / 128.8` | `30.0` | max 小幅优于 E011，但 mean/xy/rot 变差 |
| `E012_box025_p2_dualy_x20_k100_g05_obj3` | main | rotation shortcut | `0.360 / 0.643` | `85.0` | `64.2` | `3.5` | `0.539` | `147.3` | `33.7 / 139.1` | `30.0` | object reward 加强后走大旋转捷径 |
| `E012_box025_p2_dualy_x30_k100_g05` | main | rotation shortcut | `0.376 / 0.760` | `90.2` | `54.9` | `4.6` | `0.734` | `142.4` | `37.7 / 121.1` | `30.0` | 力臂变大后明显转动 |
| `E012_box025_p2_dualy_x20_k100_g08` | main | rotation shortcut | `0.363 / 0.729` | `90.8` | `52.6` | `4.0` | `1.050` | `138.4` | `42.4 / 96.2` | `30.0` | 竖直支撑改善 floor/xy，但姿态失效 |
| `E012_box025_p2_dualy_x20_k150_g05` | main | rotation shortcut | `0.306 / 0.606` | `83.2` | `59.5` | `0.0` | `0.980` | `97.8` | `36.1 / 120.0` | `30.0` | obj 最好但靠大角度旋转，不是 work |
| `E012_box023_p2_dualx_y10_k50_g05` | guard | guard unstable | `0.672 / 1.175` | `64.7` | `78.7` | `22.0` | `0.737` | `172.9` | `45.8 / 102.6` | `19.8` | 摔倒/腿箱干涉严重 |
| `E012_box023_p2_dualx_y10_k100_g05` | guard | guard unstable | `0.682 / 1.273` | `79.3` | `75.3` | `1.3` | `0.981` | `27.7` | `63.4 / 132.9` | `22.6` | 仍不稳定且 object error 很高 |

E011 best `E011_box025_p2_com_xyz_k100` 是 obj `0.340/0.673m`、xy `0.905`、rot `3.5deg`。E012 best-by-error `k150` 虽然 obj 改到 `0.306/0.606m`，但 rot 变成 `97.8deg`，所以不能视为对 E011 的有效改进。E081 main baseline 是 obj `0.143/0.271m`、hand `89.0%`、floor `59.5%`、leg interference `7.5%`；E012 全部 main 均未接近该级别。

## 可视化观察

已检查关键帧拼图：

`workspace/core4d_collab_retarget/results/E012/keyframes/e012_visual_montage.jpg`

实际观察：

- `x20_k50` 的 sim 箱体姿态相对正常，但后期位置明显落后于 ref，符合 obj `0.466/0.899m`、coupling 不足。
- `x20_k100` 比 `k50` 更能推走箱体，但中后期箱体开始偏转，最终 rot `20.6deg` 已超出 E012 gate。
- `obj3`、`x30_k100`、`g08`、`k150` 的后期关键帧都能看到 sim 箱体姿态明显不同于 ref；量化 rot `97-147deg` 说明它们是在用 torque shortcut 解释轨迹。
- `g08` 和 `k150` 的 floor/xy 看起来更好，但视觉上不是箱体朝向稳定地被双手协作搬运，而是箱体被外部点力扭动后到达部分位置。
- 两个 guard 都明显失稳：`guard_k50` 后期倒地，`guard_k100` 也出现跪倒/倒地趋势；这与 `num_guard_stable=0`、pelvis min 和 leg interference 指标一致。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 dual-point spring 比 single off-COM support point 更少旋转捷径 | ❌ 不通过 | 4/6 main 为 `rotation_shortcut`，rot `97.8-147.3deg` |
| C2 dual-point closure 可显著降低 E011 k100 的 object error | ❌ 不通过 | 只有 `k150` object error 明显低于 E011，但伴随 `97.8deg` 旋转；`pose_closure_helped=0` |
| C3 若 robot-side 仍不闭合，dual-point 成功也会暴露为 hand contact 低 | ⚠️ 部分验证 | 多数 hand 不低，但仍失败，说明问题不是单纯 contact 掉线，而是手端/物体姿态闭环错误 |
| C4 effort 必须合理 | ⚠️ 部分不通过 | force mean/max 未爆炸，但 main 多数 torque max 打到 `30Nm` clamp |
| C5 true-freejoint parity 不破坏 | ✅ 通过 | 8/8 `nu=29`、`nq_obj=7`、object actuator empty、contact guidance off |

## 结论

E012 没有证明 dual-point partner-side pose closure 是有效方向。它暴露出的规律是：

1. 双点弱 coupling (`k50`) 不足以追上 E081 级 object trajectory。
2. 增大 kp、gravity、力臂或 object reward 会改善某些 transport 表面指标，但同时诱发大角度旋转。
3. 这种失败与 E006 的“只旋转、不平移”同源：偏离 COM 的外部点力在 robot-side 闭环不足时会形成 `r x F` torque shortcut。
4. Guard 结果进一步说明该策略不稳，不能作为通用 virtual partner 支持。

下一步不应继续简单扫 dual-point `kp`、point gap 或 object reward。更合理的 E013 方向是限制外部力矩通道：以 E011 COM spring 做主平移 coupling，只允许很弱的姿态/高度 correction；或者转向 robot-side hand/support pose shaping，让手端相对位姿先闭合，再考虑 partner-side 约束。
