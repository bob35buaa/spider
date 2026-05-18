# E014 结果：COLA-B kinematic support + soft weld/equality

日期：2026-05-19

## 状态

E014 plan、soft-weld scene generator、support mocap quaternion runtime、overrides、smoke、6 条 full、显式 eval、远程回收和视频关键帧检查均已完成。结论：E014 的 COLA-B 位置约束成立。6/6 变体都保持 true-freejoint parity、非 COM object oracle、无 direct wrench，并全部通过 E013 soft target；4/4 main lag-free 且 push-vs-carry 通过，2/2 guard 稳定。

这说明 E011/E012 的主要残差确实来自 spring/force coupling 范式，而不是 true-freejoint object 数据不可达。把 coupling 换成 object-local support anchor 的 soft equality 后，main obj error 从 E011 best `0.340/0.673m` 降到 `0.056-0.082 / 0.085-0.142m`，并且没有 E012 的大角度 rotation shortcut。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E014_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E014.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E014.py --all
git commit -m "exp(core4d_collab_retarget): set up E014 soft weld"
git push
bash workspace/core4d_collab_retarget/scripts/run_E014_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E014.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E014_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E014.py --all
```

视频关键帧按 `video-frames` skill / ffmpeg 从 E014 full 视频抽取，并生成 contact sheet。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/14_E014_cola_b_kinematic_weld_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E014/` |
| Logs | `logs/core4d_collab_retarget/E014/` |
| Comparison | `workspace/core4d_collab_retarget/results/E014/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E014/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E014/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E014/keyframes/` |
| video-frames sheets | `workspace/core4d_collab_retarget/results/E014/keyframes_skill/*_sheet.jpg` |
| Scene snapshots | `workspace/core4d_collab_retarget/results/E014/scene_snapshot/` |

Full NPZ 已确认覆盖 smoke：main 4 条均 `1.2MB`，guard 2 条均 `1.4MB`。

## Full 汇总

最终 aggregate：

```json
{
  "num_results": 6,
  "num_main_results": 4,
  "num_guard_results": 2,
  "num_freejoint_parity_ok": 6,
  "num_anchor_not_com_oracle": 6,
  "num_no_direct_wrench": 6,
  "num_support_proxy_metrics_present": 6,
  "num_main_soft_target_pass": 4,
  "num_main_lag_free": 4,
  "num_main_push_vs_carry_ok": 4,
  "num_guard_stable": 2,
  "diagnostic_classes": {
    "soft_target_pass": 6
  }
}
```

关键指标：

| Variant | Role | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | support gap mean/max (m) | 结论 |
|---------|------|------------------|--------|---------|-----------|----------|---------|--------------------------|------|
| `E014_box025_p2_jointB_t02` | main | `0.056 / 0.087` | `86.7` | `51.4` | `0.0` | `0.999` | `2.2` | `0.060 / 0.098` | 过 soft target，best main |
| `E014_box025_p2_jointB_t05` | main | `0.082 / 0.142` | `87.3` | `49.7` | `0.0` | `0.995` | `1.6` | `0.083 / 0.139` | 过 soft target，较软但仍稳定 |
| `E014_box025_p2_jointB_t02_g08` | main | `0.057 / 0.088` | `85.5` | `51.4` | `0.0` | `0.998` | `2.0` | `0.060 / 0.098` | 过 soft target，gravity marker 不影响 |
| `E014_box025_p2_jointB_t02_hc1` | main | `0.057 / 0.085` | `90.8` | `51.4` | `0.0` | `0.997` | `1.7` | `0.060 / 0.098` | 过 soft target，hand contact 最高 |
| `E014_box023_p2_jointB_t02` | guard | `0.043 / 0.080` | `74.7` | `33.3` | `0.0` | `0.999` | `3.7` | `0.057 / 0.105` | guard stable |
| `E014_box023_p2_jointB_t02_hc1` | guard | `0.042 / 0.080` | `68.7` | `33.3` | `0.0` | `0.999` | `3.6` | `0.057 / 0.105` | guard stable |

E014 对 E013 soft target：

- Main target：obj `<=0.193/0.371m`、hand `>=73.6%`、floor `<=69.5%`、leg `<=12.5%`、lag-free obj mean `<0.28m`。
- Guard target：obj `<=0.214/0.417m`、hand `>=61.7%`、floor `<=44.7%`、leg `<=7.7%`、pelvis min `>=0.55m`。
- E014 全部满足上述目标。

## 可视化观察

已检查以下 contact sheet：

- `workspace/core4d_collab_retarget/results/E014/keyframes_skill/E014_box025_p2_jointB_t02_sheet.jpg`
- `workspace/core4d_collab_retarget/results/E014/keyframes_skill/E014_box025_p2_jointB_t05_sheet.jpg`
- `workspace/core4d_collab_retarget/results/E014/keyframes_skill/E014_box025_p2_jointB_t02_hc1_sheet.jpg`
- `workspace/core4d_collab_retarget/results/E014/keyframes_skill/E014_box023_p2_jointB_t02_sheet.jpg`
- `workspace/core4d_collab_retarget/results/E014/keyframes_skill/E014_box023_p2_jointB_t02_hc1_sheet.jpg`

实际观察：

- main 中 sim object 与 ref object 在全程关键帧基本重合，没有 E012 的大角度旋转捷径；蓝色 support anchor 位于 object 侧面 offset，而不是 COM。
- main 的机器人手端接触持续且视觉上仍像单人接管 partner-side 支撑后的搬运，不是 floor pushing；这与 hand `85.5-90.8%`、floor `49.7-51.4%`、leg `0%` 一致。
- `t05` 比 `t02` 稍软，视觉上 object 仍稳定；指标上的 support gap 和 obj error 也略大但仍远低于 target。
- guard 两条都稳定，无 E012 guard 摔倒/跪倒；object/ref 基本一致，pelvis min `0.689-0.691m` 支持稳定判断。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 E014 保持 true-freejoint object，不回退到 `scene_act`/object actuator | ✅ 通过 | 6/6 `E014_freejoint_parity_ok=true`；`contact_guidance=false`、`nu=29`、`nq_obj=7`、object actuator empty |
| C2 E014 不是 E013/旧 `scene_weld` COM object oracle | ✅ 通过 | 6/6 `E014_anchor_not_com_oracle=true`；scene 使用 `support_weld_anchor`，relpose 为 `[0,0.38,0.30]` 或 `[0.16,0,0.10]`，没有 `object_target` |
| C3 不使用 direct wrench/force spring | ✅ 通过 | 6/6 `E014_no_direct_wrench=true`；`support_proxy_mode=mocap_pad`，partner force scale/spring 均为 0 |
| C4 位置约束能消除 E011/E012 的 xy lag | ✅ 通过 | 4/4 main obj mean `<0.082m`，xy ratio `0.995-0.999`，全部 lag-free |
| C5 push-vs-carry 通过，未靠 floor/leg/rotation shortcut | ✅ 通过 | main floor `49.7-51.4%`，leg `0%`，rot `1.6-2.2deg` |
| C6 guard 稳定 | ✅ 通过 | 2/2 guard stable，pelvis min `0.689-0.691m`，leg `0%` |

## 结论

E014 证明 COLA-B 的 kinematic support + soft equality 是当前路线的有效范式。E014b stiffness sweep 不触发：E014 已全量通过 E013 soft target，且 `t02` / `t05` / `g08` / `hc1` 都不是边缘通过。

下一步不应继续 force/spring sweep，也不需要 E016 回退。按总路线，应进入 E015 或 pipeline 对接验证：把 kinematic support anchor 升级为 dynamic support body + PD command，或先将 E014 B-only 配置接入正式 pipeline 作为 work candidate。
