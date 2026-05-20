# E024 结果：bucket001 stability repair

日期：2026-05-20

## 初始目标

按照 `plan/27_E024_bucket001_stability_repair_plan.md`，E024 只处理 E020 中剩余的 `algo_stability` case：`bucket001_p1` 和 `bucket001_p2`。`box021_*` 已按用户要求暂不优化，E024 也不修改 object-side support proxy，只测试 robot-side stability/root/contact-gain 参数是否能修复 fall，同时保持 object transport 和 collision artifact guard。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/27_E024_bucket001_stability_repair_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E024/variants.tsv` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E024.sh`, `workspace/core4d_collab_retarget/scripts/run_E024_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E024.py` |
| Results | `workspace/core4d_collab_retarget/results/E024/` |
| Comparison | `workspace/core4d_collab_retarget/results/E024/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E024/aggregate_summary.json` |
| Online videos | `workspace/core4d_collab_retarget/results/E024/online_video/` |
| Video-frame skill output | `workspace/core4d_collab_retarget/results/E024/video_frames_skill/` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E024/scene_snapshot/manifest.txt` |

## 执行状态

- [x] Preprocess：8 个 E024 variants 生成，support proxy fields 与 E018b bucket001 rows 保持不变。
- [x] Smoke：8/8 variants 4-step smoke 通过；smoke 仅验证 wiring。
- [x] Full：5 个关键 full variants 完成，覆盖 p1 的 main/fallback/isolate 和 p2 的 main/fallback。
- [x] Eval：`eval_E024.py` 对 5 个 full variants 完成，`num_results=5`。
- [x] Visual artifacts：5/5 online MP4 到位，`video-frames` skill 已抽取 4 张代表帧。

未继续 full 跑 baseline replay 与 p2 height-only isolate：E018b 已给出 baseline failure，且 p1 的三个 planned repair variants 已全失败；p2 main/fallback 已证明 stability 可修但 artifact 仍失败，按计划决策规则转入 E025 collision penalty，不继续重复 stability sweep。

## 通俗解释：E024 做了什么、为什么没过

E024 查的是 `bucket001_p1/p2` 的“机器人自己站不住”问题。前面的 E018b 已经说明：物体本身能被 support proxy 托住，object tracking 没有明显坏掉；真正麻烦的是机器人姿态。可以把 E024 理解成：物体那边先不动，只尝试让机器人别趴下、别为了接触把身体拉到很低。

本轮主要调了三类东西：

| 调整 | 通俗含义 | 目的 |
|---|---|---|
| `stability_penalty_scale` / `threshold` | 骨盆太低就惩罚 | 逼机器人保持站姿，不要摔到桶附近 |
| `local_frame_root_sigma` 变小 | root/pelvis 更紧地跟参考动作 | 防止身体姿态漂太远 |
| `contact_hdmi_gain` 降低 | 降低“为了碰到物体而硬拉手”的力量 | 避免接触 reward 把机器人拉倒或拉进物体 |

计划里一共生成 8 个 variants，但 full 只跑了 5 个关键版本。原因是 baseline 在 E018b 已经明确失败，p1 的三个修复版本也已经全部失败；p2 的 main/fallback 已经证明“能站住但穿透严重”。继续把剩下 baseline 或 p2 height-only isolate 跑完，不会改变当前决策。

分 case 看结果更清楚：

| Case | E024 看到的现象 | 说明 |
|---|---|---|
| `bucket001_p1` | 三个修复版本都还是摔，pelvis 最低只有 `0.066-0.156m`，hand-object contact 仍是 `0%` | 这不是简单加一点站立惩罚、收紧 root、降低 contact gain 就能修的。p1 更像是 reference/support timing 或 lower-body 控制本身有问题 |
| `bucket001_p2` | 两个主版本都不摔了，pelvis 最低 `0.713-0.726m`，contact 也很高 | 稳定性问题被修了一部分；但高 contact 是“手伸进桶里”的接触，deep penetration 仍有 `59.60-64.65%`，max penetration 超过 `8cm` |

所以 E024 的结论不是“完全没用”。它把 bucket001 分成了两种失败模式：

1. `bucket001_p1`：站立修复失败，机器人还是趴下，而且完全没有有效手物接触。下一步不能继续小幅 sweep 这些参数，需要更强的 upright/root terminal gate、lower-body regularizer，或者重新检查 p1 的 reference/support timing。
2. `bucket001_p2`：站立修复成功，但接触质量很差。它能站住、能碰到物体，但主要是穿透式接触，所以应该转入 collision/penetration penalty，而不是继续调 stability。

没有出现的损失也很重要：5 个 full variants 的 object tracking 都通过，说明 E024 没有破坏 object-side support proxy。失败主要在 robot-side：p1 是 stability/contact 都没起来，p2 是 contact 很高但穿透太重。

## 量化结果

| Variant | Case | Pelvis min | No fall | Contact 5cm | Deep pen | Max pen | Epos | Erot | Artifact guard | E024 success |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `E024_bucket001_p1_root03_gain3_stab_t065` | p1 | `0.1343m` | false | `0.00%` | `0.00%` | `0.00cm` | `0.0333m` | `2.65deg` | true | false |
| `E024_bucket001_p1_root025_gain2_stab_t065` | p1 | `0.1556m` | false | `0.00%` | `0.00%` | `0.00cm` | `0.0333m` | `2.65deg` | true | false |
| `E024_bucket001_p1_stab_s1_t055` | p1 | `0.0657m` | false | `0.00%` | `0.00%` | `0.00cm` | `0.0333m` | `2.65deg` | true | false |
| `E024_bucket001_p2_root03_gain3_stab_t065` | p2 | `0.7128m` | true | `88.76%` | `64.65%` | `8.31cm` | `0.0314m` | `6.11deg` | false | false |
| `E024_bucket001_p2_root025_gain2_stab_t065` | p2 | `0.7257m` | true | `92.70%` | `59.60%` | `8.08cm` | `0.0305m` | `5.65deg` | false | false |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `5` |
| `num_stability_pass` | `2` |
| `num_pelvis_target_pass` | `2` |
| `num_contact_guard_pass` | `2` |
| `num_object_no_regression_pass` | `5` |
| `num_artifact_guard_pass` | `3` |
| `num_E024_success` | `0` |
| `best_pelvis_min_m` | `0.7257m` (`E024_bucket001_p2_root025_gain2_stab_t065`) |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 bucket001 fall 修复 | 不通过：p1 三个 full variants 均 fall；p2 两个 variants no-fall |
| C2 robot posture 改善 | 部分通过：p2 pelvis min 达 `0.71-0.73m`，p1 仍只有 `0.066-0.156m` |
| C3 object-side 不回退 | 通过：5/5 object Epos `<0.10m`、Erot `<25deg`、transport pass |
| C4 artifact 不转移 | 不通过：p2 deep penetration 仍 `59.60-64.65%`，max penetration `8.08-8.31cm` |
| C5 E024 可复现 | 通过：scripts、manifest、scene snapshot、NPZ、MP4、comparison、aggregate 均落盘 |

## 可视化观察

`video-frames` skill 已抽取：

- `E024_bucket001_p1_root03_gain3_stab_t065_t0340.jpg`
- `E024_bucket001_p1_stab_s1_t055_t0340.jpg`
- `E024_bucket001_p2_root03_gain3_stab_t065_t0320.jpg`
- `E024_bucket001_p2_root025_gain2_stab_t065_t0448.jpg`

量化指标和在线 MP4 一致：p1 variants 的 object tracking 保持住，但机器人进入低 pelvis / fall 姿态，手物 contact 为 `0%`；p2 variants 姿态稳定且 contact 很高，但 high contact 来自明显的 hand-object penetration，artifact guard 失败。

## 结论

E024 是负结果，但把 bucket001 的失败模式分清了：

1. `bucket001_p1` 不是简单的 `stability_penalty_scale`、`local_frame_root_sigma`、`contact_hdmi_gain` 参数问题。三个 planned repair variants 都摔倒，且 contact 仍为 `0%`。下一步应换策略：更强 upright/root terminal gate、lower-body control regularizer，或重新检查 p1 reference/support timing。
2. `bucket001_p2` 的 fall 可以修复：两个 variants 都 no-fall，pelvis min `>0.71m`，contact `88.76-92.70%`。
3. 但 p2 的接触是穿透式接触：deep penetration `59.60-64.65%`、max penetration `8cm+`。降低 contact gain / root sigma 只小幅改善，不足以通过 artifact guard。
4. object-side support proxy 没有回退：5/5 object tracking pass。因此 E024 后续不应继续调 anchor/support proxy。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 远程 E024 未稳定接入，SSH 多次 reset / timeout | 多次 | 改成本地顺序跑关键 full variants；保留远程给 E025 |
| p2 fallback 完成后仍 artifact fail | 1 | 不继续同类 stability/contact-gain sweep，转入 E025 explicit collision penalty |

## 下一步

- `bucket001_p2`：并入 E025-style robot-object collision penalty，重点压低 hand-object deep penetration。
- `bucket001_p1`：另开后续 stability/control 实验，不再重复 E024 的 root/contact gain sweep。
- 当前 E025 已开始 full：本地跑 `E025_box023_p1_hc2_gain8_sigma20_ori_nf`，远程 tmux `E025` 跑其余 variants。
