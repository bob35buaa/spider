# E006 结果：COLA-style support-body proxy 迁移

日期：2026-05-18

## 状态

E006 setup、smoke、local full 和 remote full 均已完成。7 个 full 结果全部保持 true-freejoint parity，且全部写入 support proxy 诊断字段。核心结论：首版“虚拟 support body/controller + soft connector wrench”成功接入了 MJWarp 重定向管线，但没有改善 `box025_p2` main 搬运，也没有超过 E005 direct support-site；guard 中一个方向能显著降低 object/floor 指标但机器人摔倒，另一个方向稳定但不运输。

## 实现摘要

E006 没有直接在 XML 里新增 freejoint support body，因为当前 MJWarp reward/eval/保存路径大量假设 object 是末尾 `nq_obj=7`。本轮实现的是 COLA support body 的安全近似：

- object 仍是 true-freejoint passive body；
- robot action 仍是 `nu=29`；
- support proxy 不进入 `qpos/qvel/ctrl`；
- setup 时从 reference object support-site 预计算 proxy pose/velocity；
- step 时用 proxy-to-object support point 的 spring-damper connector 生成 wrench：`F, r x F`；
- NPZ 记录 `support_proxy_force/torque/pos/vel` 与 `support_point_pos/vel`，用于 partner effort 和 connector gap 分析。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E006_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E006.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/run_E006_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E006.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E006_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E006.py --all
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Results | `workspace/core4d_collab_retarget/results/E006/` |
| Logs | `logs/core4d_collab_retarget/E006/` |
| Overrides | `examples/config/override/core4d_collab_E006_*.yaml` |
| Variants | `workspace/core4d_collab_retarget/scripts/E006/variants.tsv` |
| Eval comparison | `workspace/core4d_collab_retarget/results/E006/comparison.csv` |
| Keyframe montage | `workspace/core4d_collab_retarget/results/E006/e006_full_keyframe_montage.jpg` |

## Smoke

4-step smoke 只验证 wiring，不作为任务结论。结果：

```json
{
  "num_results": 7,
  "num_freejoint_parity_ok": 7,
  "num_support_proxy_metrics_present": 7
}
```

`E006_box025_p2_yneg_k20_v1` smoke 中 support force mean/max 约 `22.9/23.8N`，connector gap mean 约 `0.044m`，说明新增诊断字段能正常记录。

## Full 汇总

最终 full aggregate：

```json
{
  "num_results": 7,
  "num_main_results": 5,
  "num_guard_results": 2,
  "num_freejoint_parity_ok": 7,
  "num_support_proxy_metrics_present": 7,
  "num_main_beats_E005_support_site_proxy": 0,
  "num_main_useful_proxy": 0,
  "num_guard_stable_proxy": 1
}
```

关键指标：

| Variant | Role | kp / vscale | obj mean/max (m) | hand % | floor % | leg obj % | pelvis min | force mean/max (N) | gap mean (m) | end-height diff (m) | 结论 |
|---------|------|-------------|------------------|--------|---------|-----------|------------|--------------------|--------------|---------------------|------|
| `E006_box025_p2_yneg_k20_v1` | main | `20 / 1.0` | `0.769 / 1.514` | `52.6` | `93.1` | `0.0` | `0.765` | `22.6 / 34.7` | `0.193` | `0.740` | 不运输，手参与低 |
| `E006_box025_p2_yneg_k40_v1` | main | `40 / 1.0` | `0.733 / 1.474` | `69.4` | `92.5` | `0.6` | `0.762` | `22.6 / 41.3` | `0.145` | `0.749` | yneg 最好但仍失败 |
| `E006_box025_p2_yneg_k20_v05` | main | `20 / 0.5` | `0.802 / 1.549` | `60.7` | `95.4` | `0.0` | `0.783` | `23.0 / 32.6` | `0.164` | `0.745` | 速度减半更差 |
| `E006_box025_p2_ypos_k20_v1` | main | `20 / 1.0` | `0.704 / 1.410` | `78.0` | `91.3` | `0.0` | `0.775` | `25.6 / 32.9` | `0.195` | `0.757` | main 最好但仍不及 E005 COM |
| `E006_box025_p2_yneg_k20_v1_hc` | main | `20 / 1.0 + HC` | `0.767 / 1.503` | `64.7` | `93.6` | `0.0` | `0.770` | `22.9 / 34.6` | `0.189` | `0.743` | HC 提高手接触但不改善运输 |
| `E006_box023_p2_xneg_k10_v1` | guard | `10 / 1.0` | `0.662 / 1.151` | `66.0` | `60.0` | `4.7` | `0.069` | `28.1 / 43.5` | `0.430` | `0.040` | tracking/floor 好但摔倒 |
| `E006_box023_p2_xpos_k10_v1` | guard | `10 / 1.0` | `0.847 / 1.441` | `71.3` | `90.0` | `0.0` | `0.656` | `29.5 / 49.8` | `0.531` | `0.032` | 稳定但不运输 |

## 对比 E005 / E003

- E005 main 最好 COM：`E005_box025_p2_com_s20 = 0.678 / 1.325m`，floor `71.7%`。E006 main 最好 `ypos_k20_v1 = 0.704 / 1.410m`，floor `91.3%`，明显更依赖地面。
- E005 support-site：`yneg_s20 = 0.760 / 1.545m`，floor `96.5%`；`ypos_s20 = 0.710 / 1.422m`，floor `94.2%`。E006 对 support-site 有小幅口径改善，但未达到预设“mean 改善 `>=0.10m` 或 floor 降低 `>=15pp`”。
- E003 best passive physics：`box025_m1_f4 = 0.400 / 0.795m`。E006 全部远差于 E003 best。
- Guard：`xneg` 能把 `box023` obj mean 降到 `0.662m`、floor `60%`，但 pelvis min `0.069m`，属于摔倒换 tracking；`xpos` 稳定但 obj mean `0.847m`，接近 E002/E005 失败水平。

## 可视化

关键帧拼图：

`workspace/core4d_collab_retarget/results/E006/e006_full_keyframe_montage.jpg`

实际观察（结合关键帧与量化指标）：

- `box025` 五个 main 变体没有形成双端托举搬运；case window floor contact 仍为 `91-95%`，视觉判断应按“贴地/地面支撑主导”处理。
- `yneg_k40` 和 `ypos_k20` 比 `yneg_k20` 稍好，但 support proxy connector gap 仍约 `0.15-0.20m`，说明 proxy 与 object support point 没有形成紧耦合协作端。
- `box025` end-height diff 约 `0.74-0.76m`，这不是水平双端搬运，而是物体两端高度严重失衡。
- `box023_xneg` 的 floor/object 指标看起来更好，但 pelvis 低到 `0.069m`，可视化应判为机器人摔倒/跪倒后物体被 proxy/地面主导。
- `box023_xpos` 姿态稳定，但 object 仍明显不跟随，不构成 transport。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 support proxy 可以不破坏 true-freejoint parity | ✅ 通过 | 7/7 `nu=29`、`nq_obj=7`、`contact_guidance=false`、object actuator empty |
| C2 proxy connector 更接近 COLA 语义并可记录 effort | ✅ 工程通过 | 7/7 NPZ 有 force/torque/proxy/support point；eval 输出 force/gap/end-height |
| C3 proxy 不能只靠虚拟人替机器人搬 | ❌ 行为未达标 | main hand contact 最高 `78%`，低于 `80%`，object/floor 指标失败 |
| C4 main 至少超过 E005 direct support-site | ❌ 未通过 | `num_main_beats_E005_support_site_proxy=0`，best main `0.704/1.410m` |
| C5 guard 不应被 proxy 破坏 | ⚠️ 部分通过 | `xpos` 稳定但失败；`xneg` tracking/floor 好但 pelvis min `0.069m` |

## 结论

E006 证明 support-body proxy 的工程接口可行：它能在不改变 `nq/nv/nu` 的情况下向 true-freejoint object 施加可解释的 partner-side connector wrench，并补齐 partner effort / connector gap / end-height 指标。

但作为搬运算法，它当前失败。失败不是因为 support force 爆炸：main force mean 只有 `22-26N`，max `33-41N`，而是因为该 proxy 仍然没有建立“机器人手端 + partner 端 + object”的稳定闭环。`box025` 仍高度依赖地面，手参与不足，端点高度差巨大；`box023` 的好 tracking 方向以机器人摔倒为代价。

下一步不建议继续只扫 `support_proxy_connector_kp` 或 velocity scale。E007 应转向更结构化的接触/约束路线：

1. XML-level mocap/contact pad：在 support side 加 mocap contact pad/hand geom，通过 MuJoCo contact 而不是直接 object wrench 传力；
2. 或 robot-side 先行：强化 robot 手端接触/支撑 reward，让机器人真正承担一端，再加入轻量 partner effort penalty；
3. 若继续 support proxy，必须把 partner effort/gap 加入 reward 约束，避免 proxy target 与 object support point 长期脱钩。
