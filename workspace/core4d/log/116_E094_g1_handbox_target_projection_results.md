# E094 结果：承接 E093 的 contact target semantic repair；adaptive_support 只让 box004 保持成功，Box026 暴露姿态失败

日期：2026-05-29

对应计划：`workspace/core4d/plan/100_E094_g1_handbox_target_projection_plan.md`

## 0. 先澄清：E093 的下一步 vs E094 之后的下一步

E093 的下一步建议是：

> 先做 target semantic repair / G1-handbox-aware target projection，不要直接 rerun CEM/RL。

也就是说，E093 要求先修 `raw contact`、`wrist_yaw_link + 5cm`、support face、handbox proxy 之间的语义错位。**E094 做的就是这一步**。

E094 完成后才产生新的结论：

- target semantic repair 对 box004/box023 guard 没破坏；
- Box026 的 support/inside/object tracking 被修好，但 full-body 姿态仍崩；
- 所以 **E094 之后** 才建议做 posture / valid-contact / anti-fall CEM gate。

因此，posture gate 不是 E093 原始下一步，而是 E094 验证后暴露出的下一层问题。

## 1. E094 到底做了什么

E094 做了四件事：

1. **生成 G1-handbox-aware external contact target。**
   从 E093 的 raw contact / wrist5 / handbox 诊断出发，把明显 wrong-face / inside / low-support 的 target 修到更合理的 support patch。

2. **做 projection 可视化和 kinematic gate。**
   覆盖 5 条诊断 case：`box023_p2`、`box004_083_p2`、`box021_d003_029_p2`、`box026_039_p2`、`box026_135_p2`。

3. **用最终 external target 跑三条 E092 full CEM。**
   C1 box004 本地跑，C2/C3 Box026 远程两卡跑。

4. **修正 MuJoCo 视频相机并补 corrected autocam 可视化。**
   旧 CEM 视频缺 `front` 相机时 fallback 到 pelvis-attached `track`，导致只能看到上半身；E094 中已修成 `video_camera=auto` full-body free camera，并离线重渲三条 CEM 视频。

## 2. `adaptive_support` 是什么

`adaptive_support` 是 E094 最终采用的 **离线 per-frame target 生成策略**。它不是 CEM 运行时在线策略，也不是 RL policy。

输入：

- `old target`：原 MJWP reward 追的 `ref_fk wrist_yaw_link + 5cm EEF offset`；
- `raw target`：从 CORE4D raw hand/object surface 接触算出的接触点；
- `support patch`：把 raw target 投到物体当前姿态下 world-up/support face 后得到的 object-local 表面点；
- `handbox closest`：G1 handbox 到 support patch 的最近点，只用于诊断，不直接作为 reward target。

首轮试过但拒绝的候选：

| candidate | 规则 | 拒绝原因 |
|---|---|---|
| `handbox_compensated` | `support_patch + (wrist5 - closest_on_handbox)` | 把 guard / fail case 的 reward target 拉动过大，部分 inside 风险升高 |
| direct `support_patch` | raw-active 帧全量投到 support face | 太激进，会无差别改变 box004/box023 positive guard |

最终候选 `adaptive_support` 的规则：

```text
if 当前帧没有 raw-active contact:
    保留 old target
elif old target 不 inside，且已经在 support face 附近:
    保留 old target
elif old target 到 raw target 的距离 <= 0.30m:
    保留 old target
else:
    使用 support_patch 替代 old target
```

这个策略的目标是：**能不动就不动，只修明显错面/inside/远离 raw 的帧**。所以它能保住 box004/box023 guard，同时对 Box026 这种 low-support / inside 风险 case 做修复。

## 3. 关键路径

最终采用的 `adaptive_support` 产物在这个目录：

`workspace/core4d/results/E094/handbox_target_projection/`

其中：

| 内容 | 路径 |
|---|---|
| projection summary | `workspace/core4d/results/E094/handbox_target_projection/projection_summary.md` |
| per-frame CSV | `workspace/core4d/results/E094/handbox_target_projection/per_frame_projection.csv` |
| external target NPZ | `workspace/core4d/results/E094/handbox_target_projection/targets/` |
| object-local 可视化 | `workspace/core4d/results/E094/handbox_target_projection/visuals/object_local/` |
| timeline 可视化 | `workspace/core4d/results/E094/handbox_target_projection/visuals/timeline/` |
| MuJoCo keyframes | `workspace/core4d/results/E094/handbox_target_projection/visuals/mujoco/keyframes/` |
| MuJoCo videos | `workspace/core4d/results/E094/handbox_target_projection/visuals/mujoco/videos/` |
| projection high review | `workspace/core4d/results/E094/handbox_target_projection/visual_review/high_subagent_projection_review.md` |

被拒绝的中间候选保留在：

- `workspace/core4d/results/E094/handbox_target_projection_compensated_initial/`
- `workspace/core4d/results/E094/handbox_target_projection_support_patch_initial/`

CEM 结果：

| 内容 | 路径 |
|---|---|
| full CEM summary | `workspace/core4d/results/E094/cem/full/full_eval_summary.md` |
| full CEM NPZ/MP4 | `workspace/core4d/results/E094/cem/full/` |
| corrected autocam videos | `workspace/core4d/results/E094/cem/full/*_full_autocam.mp4` |
| corrected autocam keyframes | `workspace/core4d/results/E094/cem/full/keyframes_autocam/` |
| CEM high visual review | `workspace/core4d/results/E094/cem/full/visual_review/high_subagent_cem_autocam_review.md` |
| train logs | `logs/E094/cem/full/` |

## 4. Projection Gate 结果

| case | hand | old support | patch support | reward inside | reward delta p90 | gate |
|---|---|---:|---:|---:|---:|---|
| `box023_p2` | L/R | `100.0/58.1%` | `100.0/97.8%` | `0.0/0.0%` | `0.000/0.000m` | PASS |
| `box004_083_p2` | L/R | `42.9/24.8%` | `100.0/93.3%` | `0.0/0.0%` | `0.000/0.000m` | PASS |
| `box021_d003_029_p2` | L/R | `9.3/77.3%` | `77.3/85.3%` | `0.0/0.0%` | `0.332/0.267m` | PASS |
| `box026_039_p2` | L/R | `19.5/15.4%` | `100.0/95.9%` | `0.0/0.0%` | `0.522/0.362m` | REVIEW |
| `box026_135_p2` | L/R | `62.2/41.5%` | `100.0/100.0%` | `0.0/0.0%` | `0.486/0.631m` | REVIEW |

解释：

- `box023_p2` 和 `box004_083_p2`：reward target 几乎没移动，说明 guard 没被破坏。
- `box021_d003_029_p2`：inside 被清掉，support 明显改善，但 target 位移已经到 20-30cm 级别。
- `box026_039_p2` / `box026_135_p2`：support/inside 指标被修好，但依赖 36-63cm 的大位移，因此 projection gate 只能给 `REVIEW`，不能当作低风险修复。

## 5. Full CEM 设置

三条 CEM case 来自 E092 三 case：

| case | variant | source | split |
|---|---|---|---|
| C1 box004 | `E094P1_box004_083_p2_hbproj` | `e091_box004_20231003_2_083_p2` | local GPU0 |
| C2 Box026 039 | `E094P2_box026_039_p2_hbproj` | `e091_box026_20231018_039_p2` | remote GPU0 |
| C3 Box026 135 | `E094P3_box026_135_p2_hbproj` | `e091_box026_20231020_135_p2` | remote GPU1 |

执行脚本：

```bash
python workspace/core4d/scripts/E094/build_cem_tasks.py --force
bash workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh local full 0
bash workspace/core4d/scripts/run_E094_remote.sh full
bash workspace/core4d/scripts/pull_E094_remote_results.sh full
```

备注：本地 P1 wrapper 在完成 eval 后打印过一次 shell EOF。原因是该 bash 进程运行期间我修改了同一个脚本文件，bash 后续读文件时撞到变更窗口；当前 `bash -n workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh` 通过，P1 结果文件已完整保存。

## 6. Full CEM 指标

| variant | case | T | contact | obj_mean | obj_max | pelvis | head | upper | LH floor | RH floor | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E094P1_box004_083_p2_hbproj` | C1 | 105 | 61.0% | 0.007m | 0.018m | 0.658m | 0.0% | 0.0% | 0.0% | 0.0% | WORK |
| `E094P2_box026_039_p2_hbproj` | C2 | 123 | 80.5% | 0.002m | 0.017m | 0.440m | 0.0% | 0.0% | 0.0% | 0.0% | FAIL |
| `E094P3_box026_135_p2_hbproj` | C3 | 82 | 41.5% | 0.010m | 0.041m | 0.171m | 0.0% | 0.0% | 0.0% | 22.0% | FAIL |

判断：

- C1 是有效 positive result：object tracking、contact、安全姿态都过 gate。
- C2 的 object tracking 和 contact 很好，但 pelvis 只有 `0.440m`，说明不是可用站立搬箱动作。
- C3 更差，pelvis `0.171m`，并出现 RH floor `22.0%`。

## 7. 可视化观察

### 7.1 相机问题

旧 CEM 视频相机确实有问题：

- `render_image()` 先请求名为 `front` 的 MuJoCo camera；
- E091/E094 scene XML 只有 `track` / `track2`，没有 `front`；
- 旧代码 exception fallback 到 camera id 0，也就是 pelvis-attached `track`；
- 结果就是容易只看到机器人上半身。

已修复：

- `spider/viewers/__init__.py` 增加 full-body auto free camera；
- `spider/config.py` / `examples/config/default.yaml` 增加 `video_camera` 和 auto camera 参数；
- `train_E094_handbox_proj_cem.sh` 显式传 `video_camera=auto`；
- 已用 `rerender_cem_autocam.py` 从保存的 rollout NPZ 离线重渲，不需要重跑 CEM。

Corrected autocam 视频：

| case | video | frames |
|---|---|---:|
| C1 | `workspace/core4d/results/E094/cem/full/E094P1_box004_083_p2_hbproj_full_autocam.mp4` | `1440x480`, 210 |
| C2 | `workspace/core4d/results/E094/cem/full/E094P2_box026_039_p2_hbproj_full_autocam.mp4` | `1440x480`, 246 |
| C3 | `workspace/core4d/results/E094/cem/full/E094P3_box026_135_p2_hbproj_full_autocam.mp4` | `1440x480`, 164 |

### 7.2 CEM autocam 观察

人工抽查和 high subagent 复核一致：

- C1 `f0092`: ref/sim 都完整可见；sim 姿态接近 ref，双手在箱体附近，站姿稳定。视觉支持 `WORK`。
- C2 `f0136` / `f0245`: ref 仍是站姿搬箱；sim 机器人明显趴到 Box026 上方/箱体边缘，低髋局部解清楚。object tracking 和 contact 很好，但动作不可用。
- C3 `f0090` / `f0163`: sim 机器人倒地，箱子被掀翻/立起；末段右手/手臂接近地面。视觉与 pelvis `0.171m`、RH floor `22.0%` 一致。

high review：

`workspace/core4d/results/E094/cem/full/visual_review/high_subagent_cem_autocam_review.md`

核心判断：autocam 视角已能看完整机器人和箱子；C1 视觉支持 `WORK`；C2/C3 是真实低髋/趴箱/倒地/箱体翻转，不是相机误判。

## 8. Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| H1: 对 Box026/D003，把 raw target 投到 support face 可降低 wrong-face / inside / low-support | PASS for kinematic gate | Box026 patch support 到 `95.9-100%`，reward inside `0%`；D003 right inside 清零。 |
| H2: projection 不破坏 box004/box023 positive guards | PASS | box004/box023 reward delta p90 均 `0`，inside `0%`；box004 full CEM 仍 `WORK`。 |
| H3: 三条 E092 case 可用同一 external-target CEM 脚本验证 | PASS execution / FAIL for Box026 success | 三条 full CEM 完成；C1 `WORK`，C2/C3 因姿态崩溃 `FAIL`。 |

## 9. 分析

E094 验证了一个重要分层：

1. **contact target semantic repair 层**：E094 做到了。Box026 的 support/inside/object tracking 确实被修好。
2. **full-body dynamic feasibility 层**：Box026 仍失败。CEM 找到的是“低髋/趴箱/倒地也能把箱子轨迹跟好”的局部解。

这解释了为什么 C2 的指标看起来矛盾：

- contact `80.5%` 很高；
- obj_mean `0.002m` 极好；
- head/upper/floor 都 `0%`；
- 但 pelvis `0.440m`，视觉上趴在箱子上。

也就是说，当前 eval 的 `head/upper penetration` 只能排除穿模，不能排除“身体压箱但不穿模”的无效策略。Box026 的问题已经从 contact target 语义，推进到 full-body dynamics / posture validity。

## 10. 决策

| case | 决策 | 原因 |
|---|---|---|
| C1 box004 | 可作为 Holosoma RL 候选 | full CEM `WORK`，视觉也支持 |
| C2 box026_039 | 不进入 RL | 低髋/趴箱，虽然 object/contact 很好 |
| C3 box026_135 | 不进入 RL | 倒地、箱体翻转、RH floor 风险 |

特别说明：

- C2/C3 不进入 RL，不是因为 E093 的 target repair 没做；E094 已经完成 target repair，并证明 target/object/contact 层有效。
- 它们不进 RL 的原因是 full-body 动态姿态失败。直接把这些序列送进 Holosoma RL，会把低髋/趴箱/倒地策略作为 seed 或 reference。

## 11. E094 之后的下一步

这里是 **E094 之后** 的下一步，不是 E093 原始下一步：

1. **如果目标是尽快推进可用正样本：**
   - 对 C1 box004 开 Holosoma RL 计划；
   - 输入用 `workspace/core4d/results/E094/cem/full/E094P1_box004_083_p2_hbproj.npz` 或其转换后的动作序列；
   - RL 计划需要明确 Holosoma 入口、obs/reward、checkpoint、视频可视化和 failure gate。

2. **如果目标是继续攻 Box026：**
   - 开 E095：Box026 posture / valid-contact CEM gate；
   - 在 E094 external target 基础上加 pelvis/upright/torso-pitch/body-on-object/hand-floor hard gate 或 elite filter；
   - 先跑 C2 代表 case，确认能否保留 object tracking 同时避免趴箱，再决定是否扩到 C3。

3. **需要补的 eval 指标：**
   - pelvis trajectory，而不是只看 pelvis min；
   - torso pitch / root orientation；
   - body-on-object support 或 torso-on-box contact；
   - object flip / tilt；
   - valid hand contact vs body contact。
