# E013 结果：true-freejoint object oracle

日期：2026-05-19

## 状态

E013 plan、freejoint object kinematic oracle 实现、override 生成、4-step smoke、2 条 full、显式 eval、视频关键帧检查和 E014 soft target 生成均已完成。结论：E013 是有效 oracle 诊断，不是物理 work；它证明在 true-freejoint scene 中，如果 object 被 oracle 直接按 ref 放置，object tracking 本身可以远优于 E081 数字，因此后续 E014 的 object soft target 不需要因 freejoint oracle 而大幅放宽。

最重要的结果是：main oracle obj `0.011/0.038m`，guard oracle obj `0.017/0.064m`，都远低于 E081 main `0.143/0.271m` / guard `0.164/0.317m`。剩余差异主要是 robot-side contact/body tracking：main hand contact 只有 `78.6%`，比 E081 低 `10.4pp`，但 floor、leg interference 和 pelvis 稳定性都健康。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E013_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E013.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E013.py --all
bash workspace/core4d_collab_retarget/scripts/train/train_E013.sh full 0
```

视频关键帧按 `video-frames` skill 的 `frame.sh` 从 E013 full 视频抽取，并额外生成 contact sheet：

```bash
bash /home/ubuntu/.codex/skills/video-frames/scripts/frame.sh <video> --index <frame> --out <frame.jpg>
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/13_E013_true_freejoint_object_oracle_plan.md` |
| Execution plan | `workspace/core4d_collab_retarget/docs/03_agent_execution_plan_E013_E016.md` |
| Results | `workspace/core4d_collab_retarget/results/E013/` |
| Logs | `logs/core4d_collab_retarget/E013/` |
| Comparison | `workspace/core4d_collab_retarget/results/E013/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E013/aggregate_summary.json` |
| E014 soft targets | `workspace/core4d_collab_retarget/results/E013/e014_soft_targets.json` |
| Videos | `workspace/core4d_collab_retarget/results/E013/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E013/keyframes/` |
| video-frames sheets | `workspace/core4d_collab_retarget/results/E013/keyframes_skill/*_sheet.jpg` |
| Scene snapshots | `workspace/core4d_collab_retarget/results/E013/scene_snapshot/` |

Full NPZ 已确认覆盖 smoke：main `1.1MB`，guard `1.2MB`。

## Full 汇总

最终 aggregate：

```json
{
  "num_results": 2,
  "num_main_results": 1,
  "num_guard_results": 1,
  "num_freejoint_oracle_config_ok": 2,
  "num_near_e081_obj_oracle": 2,
  "num_guard_stable": 1
}
```

关键指标：

| Variant | Role | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | pelvis min | 结论 |
|---------|------|------------------|--------|---------|-----------|----------|---------|------------|------|
| `E013_box025_p2_obj_oracle` | main | `0.011 / 0.038` | `78.6` | `57.8` | `0.0` | `1.000` | `1.99` | `0.765m` | object oracle 极强，robot-side hand contact 仍低于 E081 |
| `E013_box023_p2_obj_oracle` | guard | `0.017 / 0.064` | `72.7` | `34.7` | `0.0` | `1.000` | `3.05` | `0.688m` | guard 稳定，object oracle 正常 |

对 E081 delta：

| Variant | obj mean Δ | obj max Δ | hand Δ | floor Δ | leg Δ |
|---------|------------|-----------|--------|---------|-------|
| main vs E081 main | `-0.132m` | `-0.233m` | `-10.4pp` | `-1.7pp` | `-7.5pp` |
| guard vs E081 guard | `-0.147m` | `-0.253m` | `+6.0pp` | `0.0pp` | `-2.7pp` |

## E014 soft targets

生成文件：`workspace/core4d_collab_retarget/results/E013/e014_soft_targets.json`

Main target:

- obj mean `<=0.193m`
- obj max `<=0.371m`
- hand contact `>=73.6%`
- floor contact `<=69.5%`
- leg interference `<=12.5%`
- lag-free diagnostic：obj mean `<0.28m`
- push-vs-carry：floor `<=70%`、leg `<=15%`

Guard target:

- obj mean `<=0.214m`
- obj max `<=0.417m`
- hand contact `>=61.7%`
- floor contact `<=44.7%`
- leg interference `<=7.7%`

## 可视化观察

已检查 E013 video-frame contact sheets：

- `workspace/core4d_collab_retarget/results/E013/keyframes_skill/main_sheet.jpg`
- `workspace/core4d_collab_retarget/results/E013/keyframes_skill/guard_sheet.jpg`

实际观察：

- main 中 sim object 与 ref object 在全程关键帧几乎重合，没有 E012 `g08/k150` 那种大角度旋转捷径；这与 obj `0.011/0.038m`、xy ratio `1.000`、rot `1.99deg` 一致。
- main 的机器人并未形成 E081 级持续手部协作接触，后段能看到手端相对 object 不够稳定；这与 hand contact `78.6%` 和 `first_sim_min_hand_sdf_gt_10cm_frame=203` 一致。
- guard 中 object 同样跟 ref 基本重合，机器人姿态稳定，没有 E012 guard 倒地/跪倒；pelvis min `0.688m` 支持稳定判断。
- guard 的 hand contact 高于 E081 guard，但仍不是最终算法成功，因为 object 是 oracle 强制放置。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 E013 是 true-freejoint scene，不回退到 `scene_act` object actuator | ✅ 通过 | 2/2 `E013_freejoint_oracle_config_ok=true`；`contact_guidance=false`、`scene_name=scene`、`nu=29`、`nq_obj=7`、object actuator empty |
| C2 object oracle 能把 object trajectory 基本压到 ref | ✅ 通过 | main obj `0.011/0.038m`，guard obj `0.017/0.064m`，均远低于 E081 |
| C3 oracle 后的 hand/floor/leg/pelvis 指标能给 E014 soft target 提供依据 | ✅ 通过 | 已生成 `e014_soft_targets.json`，main/guard 指标完整 |
| C4 oracle 不被误判为物理 work | ✅ 通过 | log 明确标记为 cheating upper bound；aggregate 只记录 oracle config/near-E081，不写 work 成功 |
| C5 脚本/评估口径可复现 | ✅ 通过 | plan/scripts/overrides/eval/full NPZ/videos/keyframes/scene snapshots 均落盘 |

## 结论

E013 证明：E011/E012 的 `0.30-0.34m` object error 不是 true-freejoint 数据/评估本身不可达，而是 spring force 范式的 xy lag 与 robot-side 闭环共同造成的。Oracle object 放置可以把 object error 降到厘米级，同时保持 floor/leg 干净。

因此 E014 应按计划进入 COLA 特征 B：kinematic support body + soft weld/equality 位置约束。E014 的关键判据应是能否在非-oracle、可解释的 partner-object 位置约束下把 main obj mean 压到 `<=0.193m`，并保持 hand `>=73.6%`、floor `<=69.5%`、leg `<=12.5%`。若 E014 best 仍 `>=0.28m`，说明位置约束也没有消除 lag，需要按总计划进入 E014b/E015，而不是跳回 force sweep。
