# E015 结果：COLA A+B dynamic support + PD command

日期：2026-05-19

## 状态

E015 plan、dynamic support scene/data generator、overrides、4-step smoke、setup commit、full 本地+远程、远程回收、全量 eval 和视频关键帧检查均已完成。结论：E015 工程 wiring 成立，但算法结果未通过 E013/E014 soft target，也明显差于 E014。

E015 的关键失败模式是 dynamic support target lag + PD saturation。默认 main `m2_kp500` 没有数值崩溃，但 support target lag 达 `0.127/0.195m`，force/torque 打到 clamp，object 出现 `50deg` 旋转，obj `0.311/0.418m`。降低 mass 或提高 kp 都没有改善，反而出现 NaN 轨迹和黑帧。Guard 稳定，但 hand contact 不达 guard target。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E015_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E015.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E015.py --all
git commit -m "exp(core4d_collab_retarget): set up E015 dynamic support"
git push origin exp/core4d-collab-retarget
bash workspace/core4d_collab_retarget/scripts/train/train_E015.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E015_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E015_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E015.py --all
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/15_E015_cola_ab_dynamic_support_pd_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E015/` |
| Logs | `logs/core4d_collab_retarget/E015/` |
| Comparison | `workspace/core4d_collab_retarget/results/E015/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E015/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E015/*.mp4` |
| Montage | `workspace/core4d_collab_retarget/results/E015/keyframes/E015_visual_montage.jpg` |
| Visual eval | `workspace/core4d_collab_retarget/results/E015/visual_eval.md` |
| Scene snapshots | `workspace/core4d_collab_retarget/results/E015/scene_snapshot/` |

Full NPZ 均已覆盖 smoke：main 三条约 `1.3MB`，guard `1.4MB`。

## Full 汇总

最终 aggregate：

```json
{
  "num_results": 4,
  "num_main_results": 3,
  "num_guard_results": 1,
  "num_freejoint_parity_ok": 4,
  "num_support_dynamic_scene_ok": 4,
  "num_object_last_ok": 4,
  "num_no_direct_wrench": 4,
  "num_pd_metrics_present": 4,
  "num_numerical_instability": 2,
  "num_main_soft_target_pass": 0,
  "num_main_effort_reasonable": 0,
  "num_main_full_success": 0,
  "num_main_lag_free": 0,
  "num_guard_stable": 1
}
```

关键指标：

| Variant | Role | obj mean/max (m) | hand % | floor % | leg % | xy ratio | rot deg | force mean/max N | support target gap mean/max m | 诊断 |
|---------|------|------------------|--------|---------|-------|----------|---------|------------------|-------------------------------|------|
| `E015_box025_p2_m2_kp500` | main | `0.311 / 0.418` | `63.6` | `43.4` | `2.3` | `1.197` | `50.3` | `79.6 / 250.0` | `0.127 / 0.195` | dynamic support lag + clamp |
| `E015_box025_p2_m1_kp500` | main | `nan / nan` | `67.1` | `81.5` | `4.6` | `nan` | `120.3` | `52.8 / 250.0` | `nan / nan` | numerical instability |
| `E015_box025_p2_m2_kp1000` | main | `nan / nan` | `71.7` | `73.4` | `0.0` | `nan` | `120.3` | `61.5 / 250.0` | `nan / nan` | numerical instability |
| `E015_box023_p2_m2_kp500` | guard | `0.097 / 0.239` | `47.3` | `42.0` | `5.3` | `0.964` | `11.4` | `50.2 / 154.2` | `0.073 / 0.162` | guard stable, robot-side contact gap |

NaN warning summary:

| Variant | warning count | max bad samples |
|---------|---------------|-----------------|
| `m2_kp500` | `494` | `208/1024` |
| `m1_kp500` | `3578` | `1024/1024` |
| `m2_kp1000` | `3530` | `1024/1024` |
| `box023_guard` | `0` | `0/1024` |

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 保持 true-freejoint object，不回退 `scene_act` / object actuator | ✅ 通过 | 4/4 `E015_freejoint_parity_ok=true`，`nq_obj=7`，`nu=29`，object actuator empty |
| C2 support 是 dynamic 6-DoF body，不是 mocap oracle | ✅ 通过 | 4/4 `E015_support_dynamic_scene_ok=true`；support 非 mocap，6 scalar joints，q/d `36/35` |
| C3 PD 通过 `qfrc_applied`，无 direct object wrench / kinematic override | ✅ 通过 | 4/4 `E015_no_direct_wrench=true`，`support_proxy_mode=dynamic_weld` |
| C4 dynamic support + soft weld 仍过 E013 soft target | ❌ 失败 | main `0/3` 过 target；best main obj `0.311/0.418m`，差于 E014 `0.056/0.087m` |
| C5 partner effort 合理 | ❌ 失败 | main `0/3` effort reasonable；default force max `250N`、torque max `80Nm` 打 clamp |
| C6 guard 稳定 | ✅ 部分通过 | guard pelvis min `0.682m`、floor/leg 过门；但 hand `47.3% < 61.7%`，不通过 guard soft target |

## 可视化观察

见 `visual_eval.md` 和 `keyframes/E015_visual_montage.jpg`。

- default main 视觉上有明显支撑滞后和大角度箱体旋转，未复现 E014 的紧约束跟踪。
- m1/kp1000 后段关键帧为黑帧，和 NaN 轨迹一致。
- guard 姿态稳定，但手端接触弱，不能作为 COLA A+B 成立证据。

## 结论与下一步

E015 没有证明完整 COLA A+B 成立。相反，它说明从 E014 kinematic support 升级到 dynamic support 后，当前 3 slide + 3 hinge PD 参数会引入显著 support target lag、force/torque saturation 和数值不稳。

按预设判定，不能跳 E016。下一步应触发 E015b，聚焦 effort/PD tuning 和数值稳定性，而不是扩大结构路线：

- 降低 `rot_kp` / 提高 torque clamp 前先分离旋转 DOF，避免 `120deg` NaN/rotation shortcut。
- 增大 support mass/inertia 或降低 `pos_kp`，用 target lag 与 force clamp 命中率决定方向。
- 增加 support joint damping/friction 或 PD force ramp，减少前期 CEM sample 爆炸。
- E015b 的成功门仍沿用 E013 soft target + effort 合理；若只减少 NaN 但仍差于 E014，则停止 A 动态化路线，保留 E014 B-only 作为 pipeline candidate。
