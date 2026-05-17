# E008 结果：support proxy 高/不限速与 E081 搬运验收

日期：2026-05-18

## 状态

E008 plan、脚本、预授权、smoke 和 full 已完成。结论：E007 的 `support_proxy_max_xy_speed=0.8` 的确是关键瓶颈之一；取消或提高限速后，proxy target 能完整跟随 reference support point，object 从“几乎不平移、主要旋转”改善为“明显平移、旋转受控”。但 E008 仍未达到 E081 搬运验收：最佳 main `E008_box025_p2_ypos_k20_vmax2` 的 object xy ratio 为 `0.724`，接近 `0.75` 门槛，但 object mean/max 仍为 `0.363/0.684m`，明显差于 E081 `0.143/0.271m`。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E008_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E008_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E008_remote_results.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E008.py --all
bash workspace/core4d_collab_retarget/scripts/run_E008_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/train/train_E008.sh one 0 E008_box025_p2_ypos_k20_vmax2
bash workspace/core4d_collab_retarget/scripts/pull_E008_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E008.py \
  E008_box025_p2_yneg_k20_vmax0 \
  E008_box025_p2_yneg_k40_vmax0 \
  E008_box025_p2_ypos_k20_vmax0 \
  E008_box025_p2_yneg_k20_vmax2 \
  E008_box025_p2_ypos_k20_vmax2 \
  E008_box023_p2_xpos_k10_vmax0
```

`E008_box023_p2_xneg_k10_vmax0` 的 full 在远程 `96/272` 处卡住，GPU 利用率为 0 且 CPU 100%，已终止。由于 smoke 阶段存在同名 4-step NPZ，最终 full eval 显式列出 6 个完整结果，避免把 smoke 当 full。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/08_E008_support_proxy_speed_unclamped_e081_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E008/` |
| Logs | `logs/core4d_collab_retarget/E008/` |
| Comparison | `workspace/core4d_collab_retarget/results/E008/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E008/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E008/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E008/keyframes/` |

## Smoke

4-step smoke 只验证 wiring，不作为任务结论。结果：

```json
{
  "num_results": 7,
  "num_freejoint_parity_ok": 7,
  "num_support_proxy_metrics_present": 7,
  "num_main_proxy_support_tracking_ok": 5
}
```

## Full 汇总

最终 full aggregate：

```json
{
  "num_results": 6,
  "num_main_results": 5,
  "num_guard_results": 1,
  "num_freejoint_parity_ok": 6,
  "num_support_proxy_metrics_present": 6,
  "num_main_proxy_support_tracking_ok": 5,
  "num_main_reaches_E081_transport_proxy": 0,
  "num_main_beats_or_matches_E081_majority": 0,
  "num_guard_stable_proxy": 1
}
```

关键指标：

| Variant | Role | vmax/kp | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | proxy ratio | gap mean (m) | force mean/max (N) | 结论 |
|---------|------|---------|------------------|--------|---------|-----------|----------|---------|-------------|--------------|--------------------|------|
| `E008_box025_p2_yneg_k20_vmax0` | main | `0 / 20` | `0.485 / 0.827` | `43.4` | `69.9` | `22.0` | `0.711` | `12.7` | `1.000` | `0.476` | `27.8 / 97.4` | 平移显著改善，但手接触低、腿干涉高 |
| `E008_box025_p2_yneg_k40_vmax0` | main | `0 / 40` | `0.476 / 0.870` | `55.5` | `84.4` | `16.2` | `0.650` | `43.2` | `1.000` | `0.390` | `35.9 / 81.1` | gap 降低但旋转/地面更差 |
| `E008_box025_p2_ypos_k20_vmax0` | main | `0 / 20` | `0.418 / 0.749` | `75.7` | `65.9` | `0.0` | `0.636` | `17.9` | `1.000` | `0.356` | `28.1 / 49.8` | 稳定且无腿干涉，但运输不足 |
| `E008_box025_p2_yneg_k20_vmax2` | main | `2 / 20` | `0.561 / 1.015` | `42.8` | `83.2` | `16.2` | `0.508` | `36.7` | `0.992` | `0.476` | `30.3 / 58.9` | 高限速仍落后，失败 |
| `E008_box025_p2_ypos_k20_vmax2` | main | `2 / 20` | `0.363 / 0.684` | `78.0` | `64.7` | `2.9` | `0.724` | `13.6` | `0.998` | `0.304` | `27.7 / 46.7` | 本轮最好，接近但未过 E081 |
| `E008_box023_p2_xpos_k10_vmax0` | guard | `0 / 10` | `0.830 / 1.449` | `77.3` | `90.7` | `0.0` | `0.183` | `73.1` | `1.000` | `0.802` | `32.0 / 53.1` | guard stable，但不运输 |

## 对比 E007 / E081

- E007 首个 full：object xy ratio `0.291`，rot `41.6deg`，proxy ratio `0.693`，obj `0.625/1.205m`。
- E008 最好：object xy ratio `0.724`，rot `13.6deg`，proxy ratio `0.998`，obj `0.363/0.684m`。
- E081 main baseline：obj `0.143/0.271m`，hand `89.0%`，floor `59.5%`，leg intf `7.5%`。

E008 显著缩小了与 E081 的差距，但还不是 work。最接近的 `ypos_k20_vmax2` 仍比 E081 obj mean 高 `0.22m`，hand contact 低约 `11pp`，xy ratio 差 `0.026`，因此不能判定为基本搬运达标。

## 可视化观察

检查 `E008_box025_p2_ypos_k20_vmax2` 关键帧：

- 物体不再像 E006/E007 那样主要原地旋转，已经跟随参考方向发生明显水平平移。
- 但 sim 物体始终相对 ref 滞后，后半段尤其明显；f204 时 sim box 已移动到机器人前侧，但仍未到 ref 位置。
- 机器人手端没有形成稳定托举/夹持。视觉上手经常贴近箱体侧面，但不像 E081 ref 那样持续支撑；这与 hand contact `78.0%`、connector gap `0.304m` 一致。
- 箱体仍偏地面支撑，floor contact `64.7%` 只略好于 E081+margin 口径，语义上仍更像“虚拟协作者推/拉 + 机器人跟随”，不是干净双端搬运。
- `ypos_k20_vmax0` 更稳定且腿干涉为 0，但运输比例更低；`yneg` 方向平移较大时容易引入腿/箱干涉或旋转。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 高/不限速后 proxy 能完整跟随 reference support point | ✅ 通过 | 5/5 main proxy ratio `>=0.992`，final gap `<=0.012m` |
| C2 若 E007 失败来自限速，object transport 应显著改善 | ✅ 部分通过 | xy ratio `0.291 -> 0.724`，rot `41.6deg -> 13.6deg` |
| C3 proxy 完整平移但 object 仍不达标时，force-only off-COM route 仍不足 | ✅ 通过 | proxy gate 全过，但 `num_main_reaches_E081_transport_proxy=0` |
| C4 eval 对齐 E081 而非 E005 | ✅ 通过 | 输出 `E008_vs_E081_*`、majority score、E081 transport gate |
| C5 true-freejoint parity 不破坏 | ✅ 通过 | 6/6 full parity ok，`nu=29`、`nq_obj=7`、no object actuator |

## 结论

E008 是本方向第一次把 true-freejoint object 推到接近“基本运输”的区域。它说明 E006/E007 的“物体不平移只旋转”不是 support proxy 范式完全失败，而是先被时间基准/速度限制截断；速度门槛修正后，object 可以平移到参考的 `72%` 左右，并把旋转压到 `15deg` 以下。

剩余瓶颈已经转移：

1. robot hand/support 端没有闭环，手部接触仍低于 E081；
2. connector gap 仍大，虚拟协作者和真实 robot 手端没有形成刚体两端协作；
3. yneg 方向会引入腿/箱干涉，ypos 方向更干净；
4. guard `box023` stable 但不 transport，xneg guard 还出现远程卡住。

## 下一步

E009 不应继续只扫 `support_proxy_max_xy_speed` 或 `kp`。建议围绕 E008 best `ypos_k20_vmax2` 做 robot-side 闭环：

1. `ypos_k20_vmax2 + hold_contact`：把 hold-contact window 从旧的 `1.8-2.5s` 扩到搬运 case window，测试能否把 hand contact 从 `78%` 推到 `>=80-85%`；
2. `ypos_k20_vmax2 + lower/penalize leg interference`：保持 ypost 方向，避免 yneg 的腿干涉；
3. 若 hold-contact 不能改善 object error，则进入 XML-level mocap/contact pad，让 partner support 通过 contact/soft constraint 传力，而不是继续直接 object wrench。
