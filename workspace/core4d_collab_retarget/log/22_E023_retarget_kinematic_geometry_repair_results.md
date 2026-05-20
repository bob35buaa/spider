# E023 结果：retarget kinematic geometry repair

日期：2026-05-20

## 初始目标

按照 `plan/26_E023_retarget_kinematic_geometry_repair_plan.md`，处理 E020 的 `retarget_kinematic` case：`box025_p1` 和 `bucket007_p2`。E020 S2 显示这两个 kinematic reference 本身已有约 `66%` 的 leg/object interference，因此 E023 不做 reward sweep，而是测试 lower-body collision geometry / leg-object pair 修复是否能把 ref-side interference 降到 `<15%`，并保持 E018b object-side support proxy 不回退。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/26_E023_retarget_kinematic_geometry_repair_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E023/variants.tsv` |
| Asset scripts | `workspace/core4d_collab_retarget/scripts/E023/` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E023.sh`, `workspace/core4d_collab_retarget/scripts/run_E023_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E023.py` |
| Results | `workspace/core4d_collab_retarget/results/E023/` |
| Comparison | `workspace/core4d_collab_retarget/results/E023/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E023/aggregate_summary.json` |
| Unified eval | `workspace/core4d_collab_retarget/results/E023/eval_unified/` |
| Online videos | `workspace/core4d_collab_retarget/results/E023/online_video/` |
| Video-frame skill output | `workspace/core4d_collab_retarget/results/E023/video_frames_skill/` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E023/scene_snapshot/manifest.txt` |

## 执行状态

- [x] E023 task copies：6 个 task copy 生成，未修改 E018b 原始 task。
- [x] XML patch：
  - `baseline_replay`: no patch。
  - `legpair_off`: 删除 16 个 lower-body/object contact pairs，diagnostic only。
  - `lowerbody_proxy_min`: shrink 16 个 lower-body collision geoms，strict candidate。
- [x] Overrides：6 个 `examples/config/override/core4d_collab_E023_*.yaml` 生成。
- [x] Smoke：6/6 variants 4-step smoke 通过；smoke 仅验证 wiring。
- [x] Full：本地 `box025_p1_lowerbody_proxy_min` + 远程 5 variants 全部完成，6/6 NPZ 与 6/6 MP4 到位。
- [x] Eval：`eval_E023.py --all` 完成，`num_results=6`。
- [x] Unified eval：`eval_unified/` CSV/JSON bundle 已生成。
- [x] Visual artifacts：box025 videos `248` frames / `4.96s`；bucket007 videos `190` frames / `3.80s`；`video-frames` skill 已抽取代表帧。

## Geometry patch 范围与副作用

先用通俗语言说：E023 要查的是“参考动作本身是不是有问题”。在 `box025_p1` 和 `bucket007_p2` 里，物体已经能跟得比较稳，但人的腿部碰撞体和物体经常重叠，像是机器人的大腿/小腿/脚的碰撞外壳太粗，或者腿和物体的碰撞规则不合适。E023 不是继续调 reward，而是复制一份任务，在复制出来的 XML 里动腿部碰撞设置，看能不能把这种腿-物体重叠降下来。

E023 对每个 case 都做了 3 个版本：

| 版本 | 做了什么 | 能不能算真正修复 |
|---|---|---|
| `baseline_replay` | 什么都不改，只重跑一遍作为对照 | 不能，只是 baseline |
| `legpair_off` | 直接关掉腿/脚和物体之间的 16 个碰撞 pair | 不能，只是诊断；因为它可能让腿直接穿过物体 |
| `lowerbody_proxy_min` | 把腿/脚的碰撞外壳缩小 | 是本轮真正想验证的修复候选 |

所以，“每个 case 都缩小了吗”的答案是：两个 case 都各自有一个 `lowerbody_proxy_min` 版本做了缩小；但 6 个总 variants 里只有 2 个真的 shrink。`baseline_replay` 没改，`legpair_off` 只是关碰撞 pair，也没有缩小几何。

缩小的内容也很固定：对 `box025_p1` 和 `bucket007_p2` 都缩同一组 16 个腿部/脚部碰撞体，包括左右 hip、thigh、shin、linkage brace，以及左右脚的 `lf0-lf3`、`rf0-rf3`。腿部主体半径缩到 `0.005`，脚部 proxy 缩到 `0.001`。没有改手、物体、object support proxy、object 初始姿态、contact mask，也没有打开 direct object control。

结果可以这样理解：

| Case | 真正 shrink 后发生了什么 | 有没有明显代价 |
|---|---|---|
| `box025_p1` | 腿/物体重叠明显下降：`66.53% -> 25.40%`，说明“腿部碰撞外壳太粗”确实是问题的一部分 | 没有明显 object/fall 回退；object 仍稳定，没摔倒。contact 略升 `59.64% -> 63.57%`，但 deep penetration 小幅变差 `6.74% -> 7.30%`，整体仍没达标 |
| `bucket007_p2` | 也有改善，但不够：full window `66.32% -> 45.26%`，case window `57.89% -> 31.58%` | object 仍稳定，没摔倒；但 contact 小幅变差 `29.03% -> 27.96%`，max penetration `6.54cm -> 6.82cm`，所以 shrink 对这个 case 没带来实际收益 |

`legpair_off` 的结果更像反例：它把腿/脚和物体的碰撞关系关掉，看起来有时会让 contact 数字变好，但这是不可靠的。比如 `box025_p1_legpair_off` contact 到了 `72.86%`，但 robot-object deep penetration 也升到 `32.02%`，说明它不是解决了接触，而是让身体更容易穿进去。因此 E023 不把 `legpair_off` 当成可接受方案。

最终结论：E023 证明 lower-body geometry 确实是根因之一，尤其对 `box025_p1` 很明显；但简单把腿/脚碰撞体统一缩小还不够，两个 case 都没有达到 `<15%` 的 geometry repair 目标。它没有破坏 object tracking，也没有引入 fall；主要问题是修得不够细，`bucket007_p2` 还出现 contact/penetration 小幅变差。后续应该做更细的 per-geom collision 建模或训练期 penetration penalty，而不是简单关碰撞 pair。

## 量化结果

| Variant | Patch | Full ref leg intf | Case ref leg intf | Sim leg intf | Contact 5cm | Deep pen | Max pen | Epos | Erot | Fall | E023 success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `box025_p1_baseline_replay` | none | `66.53%` | `73.03%` | `12.92%` | `59.64%` | `6.74%` | `4.03cm` | `0.0705m` | `5.00deg` | false | false |
| `bucket007_p2_baseline_replay` | none | `66.32%` | `57.89%` | `5.26%` | `29.03%` | `21.71%` | `6.54cm` | `0.0483m` | `4.08deg` | false | false |
| `box025_p1_legpair_off` | legpair_off | `66.53%` | `73.03%` | `45.51%` | **`72.86%`** | `32.02%` | `4.84cm` | `0.0695m` | `4.76deg` | false | false |
| `bucket007_p2_legpair_off` | legpair_off | `66.32%` | `57.89%` | `14.47%` | `15.05%` | `19.74%` | `6.55cm` | `0.0439m` | `2.89deg` | false | false |
| `box025_p1_lowerbody_proxy_min` | lowerbody_proxy_min | **`25.40%`** | **`32.02%`** | `12.92%` | `63.57%` | `7.30%` | `2.24cm` | `0.0704m` | `5.05deg` | false | false |
| `bucket007_p2_lowerbody_proxy_min` | lowerbody_proxy_min | `45.26%` | `31.58%` | `9.87%` | `27.96%` | `21.71%` | `6.82cm` | `0.0445m` | `3.46deg` | false | false |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `6` |
| `num_ref_geometry_repair_pass` | `0` |
| `num_contact_goal_pass` | `1` |
| `num_object_no_regression_pass` | `6` |
| `num_artifact_guard_pass` | `2` |
| `num_E023_success` | `0` |
| `best_ref_leg_interference_pct` | `25.40%` (`box025_p1_lowerbody_proxy_min`) |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 ref geometry repair 降到 `<15%` | 不通过：best full ref interference 仍 `25.40%`，case-window best `31.58-32.02%` |
| C2 object-side support proxy 不回退 | 通过：6/6 `E023_support_proxy_unchanged=true`，6/6 object no-regression pass |
| C3 physical rollout 完整 retarget | 不通过：只有 `box025_p1_legpair_off` contact `>=70%`，但 deep penetration `32.02%`，不是有效 pass |
| C4 true-freejoint invariant 保持 | 通过：E018b evaluator config gate 保持 true-freejoint / no direct wrench |
| C5 E023 可复现 | 通过：scripts、manifest、scene snapshot、NPZ、MP4、comparison、aggregate、unified eval 均落盘 |

## 可视化观察

`video-frames` skill 已抽取：

- `E023_box025_p1_baseline_replay_f115.jpg`
- `E023_box025_p1_lowerbody_proxy_min_f115.jpg`
- `E023_bucket007_p2_baseline_replay_f100.jpg`
- `E023_bucket007_p2_lowerbody_proxy_min_f100.jpg`

视觉观察与指标一致：E023 没有破坏 object-side transport，也没有出现 fall；但 lower-body/object geometry patch 只是降低了一部分 ref interference，并没有形成高质量 hand-object retarget。`legpair_off` 的 box025 contact 表面上达到 `72.86%`，但同时 deep penetration 升到 `32.02%`、sim leg interference 升到 `45.51%`，说明删除 contact pairs 只是绕开约束，不是可接受的修复。

## 结论

E023 没有达到 strict success，但进一步缩小了根因：

1. `lowerbody_proxy_min` 能降低 ref-side leg/object interference，`box025_p1` 从 `66.53% -> 25.40%`，`bucket007_p2` 从 `66.32% -> 45.26%`，证明 lower-body proxy geometry 是一部分原因。
2. 但 shrink lower-body proxy 仍不足以达到 `<15%`，也没有把 contact preservation 推到 `>=70%`。
3. `legpair_off` 不是解：它不改变 ref interference，且 box025 虽然 contact 变高，但 deep penetration / sim leg artifact 明显变坏。
4. 两个 case 的 object tracking 都稳定，说明 E023 失败仍是 robot-side geometry/contact-control 问题，不是 E018b support proxy 回退。

后续不应继续重复删除 contact pairs。`box025_p1` 和 `bucket007_p2` 应转入更细的 lower-body collision geometry建模或 E025-style contact/collision penalty：需要保留真实 contact pairs，同时用训练期 penalty 抑制 hand/leg penetration。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 远程状态查询出现一次 SSH key-exchange reset | 1 | 等待 30s 后重试成功；E023 tmux 和结果不受影响 |
| 4-step smoke 的 ref interference 指标容易误导 | 1 | 在 progress 中明确 caveat：smoke 只验证 wiring，full rollout 后才判断 ref geometry repair |

## 下一步

- E024：按已写计划处理 `bucket001_p1/p2` stability，保持 support proxy 不变，优先 root/stability/contact-gain sweep。
- E025：把 `box023_p1`、E020 `algo_contact` 四个 case，以及 E023 未通过的 contact/collision artifact 纳入 robot-side contact closure + explicit penetration penalty 的统一实现。
