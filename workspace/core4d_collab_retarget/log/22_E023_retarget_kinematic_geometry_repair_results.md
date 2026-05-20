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

E023 修的是 E020 `retarget_kinematic` 分线暴露出的 ref-side lower-body/object geometry 问题。`box025_p1` 和 `bucket007_p2` 在 E018b object-side support proxy 已稳定的前提下，kinematic reference 本身就有很高的腿/物体干涉：full ref leg/object interference 分别为 `66.53%` 和 `66.32%`。因此 E023 没有继续做 reward sweep，而是复制 E018b derived task 后只 patch copied task 的 `scene*.xml`，用来判断干涉是否来自 lower-body collision proxy / lower-body-object contact pair 建模。

E023 一共对两个 target case 各生成 3 个 variants；不是每个 variant 都 shrink：

| Case | Variant | Patch mode | 是否 shrink lower-body geometry | 作用 |
|---|---|---|---|---|
| `box025_p1` | `baseline_replay` | `none` | 否 | E023 plumbing/control |
| `box025_p1` | `legpair_off` | `legpair_off` | 否 | 删除 lower-body/object contact pairs，仅诊断 |
| `box025_p1` | `lowerbody_proxy_min` | `lowerbody_proxy_min` | 是 | strict candidate |
| `bucket007_p2` | `baseline_replay` | `none` | 否 | E023 plumbing/control |
| `bucket007_p2` | `legpair_off` | `legpair_off` | 否 | 删除 lower-body/object contact pairs，仅诊断 |
| `bucket007_p2` | `lowerbody_proxy_min` | `lowerbody_proxy_min` | 是 | strict candidate |

具体实现上，`lowerbody_proxy_min` 对两个 case 的 copied task 都 shrink 同一组 `16` 个 lower-body/foot collision geoms，而不是只 shrink 每个 case 的 top offending geoms。被 shrink 的 geoms 是：`left_hip_collision`、`right_hip_collision`、`left_thigh_collision`、`right_thigh_collision`、`left_shin_collision`、`right_shin_collision`、`left_linkage_brace_collision`、`right_linkage_brace_collision`、`lf0-lf3`、`rf0-rf3`。其中非 foot lower-body geom 的 `size[0]` 设为 `0.005`，foot sphere/proxy 的 `size[0]` 设为 `0.001`。patch 后用 MuJoCo 加载 XML 校验模型维度仍为 `nq=43/nv=41/nu=29`，同时不修改 hand geoms、object geoms、object qpos、support proxy anchor、contact masks 或 true-freejoint/object-action 配置。

`legpair_off` 是另一条诊断分支：它不 shrink geometry，而是删除同一组 `16` 个 lower-body geom 与 object 的 contact pairs，例如 thigh/shin/linkage/foot 对 object 的 pair。这个分支只能判断“接触 pair 是否参与 artifact”，不能作为最终修复，因为它可能直接绕开真实碰撞约束。

副作用分 case 看：

| Case | Patch | Ref geometry 变化 | Contact / penetration 变化 | Object / fall 变化 | 判断 |
|---|---|---|---|---|---|
| `box025_p1` | `lowerbody_proxy_min` | full ref intf `66.53% -> 25.40%`，case-window `73.03% -> 32.02%` | contact `59.64% -> 63.57%`；deep pen `6.74% -> 7.30%` 小幅变差；max pen `4.03cm -> 2.24cm` 变好 | Epos `0.0705m -> 0.0704m`，Erot `5.00deg -> 5.05deg`，no fall，object no-regression pass | 有效降低 ref interference，且没有明显 object/fall 回退；但仍未达 `<15%` / contact `>=70%` |
| `bucket007_p2` | `lowerbody_proxy_min` | full ref intf `66.32% -> 45.26%`，case-window `57.89% -> 31.58%` | contact `29.03% -> 27.96%` 小幅变差；deep pen `21.71% -> 21.71%` 不变；max pen `6.54cm -> 6.82cm` 变差 | Epos `0.0483m -> 0.0445m`，Erot `4.08deg -> 3.46deg`，no fall，object no-regression pass | case-window geometry 有改善，但 full ref interference 仍高，contact/penetration 没有收益 |
| `box025_p1` | `legpair_off` | full/case ref intf 不变，仍 `66.53%/73.03%` | contact `59.64% -> 72.86%` 表面过线；但 robot-object deep pen `6.74% -> 32.02%`，case-window sim leg intf `12.92% -> 45.51%` | object 稳定，no fall | 明显副作用，说明删除 pair 是绕开约束，不是修复 |
| `bucket007_p2` | `legpair_off` | full/case ref intf 不变，仍 `66.32%/57.89%` | contact `29.03% -> 15.05%` 变差；robot-object deep pen `21.71% -> 19.74%` 小幅改善但仍 max pen `6.55cm`；sim leg intf `5.26% -> 14.47%` 变差 | object 稳定，no fall | 对 contact 有负面影响，也不是可接受修复 |

因此，“每个 case 都缩小了吗”的准确答案是：E023 的两个目标 case 都各自有一个 `lowerbody_proxy_min` shrink variant，并且该 variant 对两个 case 都 shrink 了同一组 16 个 lower-body/foot geoms；但 baseline 与 `legpair_off` variants 没有 shrink。带来的损失不是 object-side support proxy 或 fall 回退，6/6 object no-regression pass、6/6 no fall；主要损失/不足在于 shrink 后 geometry interference 仍未降到 `<15%`，`bucket007_p2` contact/penetration 还略有变差，而 `legpair_off` 会引入明显 penetration/sim leg artifact。

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
