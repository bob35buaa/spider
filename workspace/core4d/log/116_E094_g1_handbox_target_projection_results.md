# E094 结果：adaptive-support target projection 保住 box004，但 Box026 仍进入低髋/趴箱局部解

日期：2026-05-29

对应计划：`workspace/core4d/plan/100_E094_g1_handbox_target_projection_plan.md`

## 结论

E094 完成了 G1-handbox-aware target projection、full-body MuJoCo 可视化、三条 E092 case 的 full CEM、统一评估和 corrected autocam 视频复核。

核心结论：

1. **相机问题确认并修复。** CEM 视频旧逻辑找不到 `front` 相机时 fallback 到第 0 个 pelvis-attached `track` 相机，导致只看到上半身。已改为 missing `front` / `video_camera=auto` 时用 full-body free camera；本次所有 CEM 都补了 `*_autocam.mp4`。
2. **projection 对 positive guard 不破坏。** box004 / box023 的 reward target 基本不移动，inside 保持 `0%`；box004 full CEM 达到 `WORK`。
3. **projection 对 Box026 的物体/contact 指标有效，但不解决姿态局部解。** C2/C3 的 object tracking 很好，contact 也不低，但机器人选择低髋、趴箱、倒地、箱子翻转等动态解。
4. **C1 可进入下一阶段 Holosoma RL 候选；C2/C3 不应进 RL。** C2/C3 的失败不是相机误判，而是真实动态失败。直接拿这些序列做 RL 会把低髋/倒地策略传给 RL。
5. **下一步不应继续只调 contact target。** Box026 需要 CEM 内部姿态/稳定性约束或 elite gate，例如 pelvis/upright/torso pitch/hand-floor/valid-contact 联合过滤，再比较是否能保留 object tracking。

## 运行命令

```bash
python workspace/core4d/scripts/E094/build_handbox_target_projection.py --force
python workspace/core4d/scripts/E094/render_projection_mujoco.py --video-frames 48
python workspace/core4d/scripts/E094/build_cem_tasks.py --force

bash workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh local full 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider_e094_run && tmux new-session -d -s E094_hbproj_full_0529 "bash workspace/core4d/scripts/run_E094_remote.sh full"'
bash workspace/core4d/scripts/pull_E094_remote_results.sh full

RESULTS=workspace/core4d/results/E094/cem/full \
VARIANTS_FILE=workspace/core4d/scripts/E094/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E094_cem.py --stage full

MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/E094/rerender_cem_autocam.py --stage full --overwrite
```

备注：本地 P1 wrapper 在完成 eval 后打印过一次 shell EOF，这是因为该 bash 进程运行期间脚本文件被修改；当前 `bash -n workspace/core4d/scripts/train/train_E094_handbox_proj_cem.sh` 通过，P1 结果文件已完整保存。

## 结果路径

| 类型 | 路径 |
|---|---|
| plan | `workspace/core4d/plan/100_E094_g1_handbox_target_projection_plan.md` |
| projection root | `workspace/core4d/results/E094/handbox_target_projection/` |
| projection summary | `workspace/core4d/results/E094/handbox_target_projection/projection_summary.md` |
| projection MuJoCo videos | `workspace/core4d/results/E094/handbox_target_projection/visuals/mujoco/videos/` |
| CEM results | `workspace/core4d/results/E094/cem/full/` |
| CEM logs | `logs/E094/cem/full/` |
| corrected CEM videos | `workspace/core4d/results/E094/cem/full/*_full_autocam.mp4` |
| corrected CEM keyframes | `workspace/core4d/results/E094/cem/full/keyframes_autocam/` |
| projection visual review | `workspace/core4d/results/E094/handbox_target_projection/visual_review/high_subagent_projection_review.md` |
| CEM visual review | `workspace/core4d/results/E094/cem/full/visual_review/high_subagent_cem_autocam_review.md` |

## Projection Gate

| case | hand | old support | patch support | reward inside | reward delta p90 | gate |
|---|---|---:|---:|---:|---:|---|
| `box023_p2` | L/R | `100.0/58.1%` | `100.0/97.8%` | `0.0/0.0%` | `0.000/0.000m` | PASS |
| `box004_083_p2` | L/R | `42.9/24.8%` | `100.0/93.3%` | `0.0/0.0%` | `0.000/0.000m` | PASS |
| `box021_d003_029_p2` | L/R | `9.3/77.3%` | `77.3/85.3%` | `0.0/0.0%` | `0.332/0.267m` | PASS |
| `box026_039_p2` | L/R | `19.5/15.4%` | `100.0/95.9%` | `0.0/0.0%` | `0.522/0.362m` | REVIEW |
| `box026_135_p2` | L/R | `62.2/41.5%` | `100.0/100.0%` | `0.0/0.0%` | `0.486/0.631m` | REVIEW |

Interpretation:

- `adaptive_support` 是保守策略：旧 target 不 inside、已在 support face、或离 raw 不太远时保留旧 target；只有明显 wrong-face/inside/远离 raw 才投影到 support patch。
- Box026 的 projection 数值上修复了 support/inside，但代价是 `36-63cm` p90 target 位移，因此 full CEM 前已标记 high-risk。

## Full CEM 指标

| variant | case | T | contact | obj_mean | obj_max | pelvis | head | upper | LH floor | RH floor | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E094P1_box004_083_p2_hbproj` | C1 | 105 | 61.0% | 0.007m | 0.018m | 0.658m | 0.0% | 0.0% | 0.0% | 0.0% | WORK |
| `E094P2_box026_039_p2_hbproj` | C2 | 123 | 80.5% | 0.002m | 0.017m | 0.440m | 0.0% | 0.0% | 0.0% | 0.0% | FAIL |
| `E094P3_box026_135_p2_hbproj` | C3 | 82 | 41.5% | 0.010m | 0.041m | 0.171m | 0.0% | 0.0% | 0.0% | 22.0% | FAIL |

## 可视化观察

Corrected autocam 视频：

| variant | video | frames |
|---|---|---:|
| C1 | `workspace/core4d/results/E094/cem/full/E094P1_box004_083_p2_hbproj_full_autocam.mp4` | `1440x480`, 210 |
| C2 | `workspace/core4d/results/E094/cem/full/E094P2_box026_039_p2_hbproj_full_autocam.mp4` | `1440x480`, 246 |
| C3 | `workspace/core4d/results/E094/cem/full/E094P3_box026_135_p2_hbproj_full_autocam.mp4` | `1440x480`, 164 |

人工抽查：

- C1 `f0092`: ref/sim 都完整可见；sim 姿态接近 ref，双手在箱体附近，站姿稳定。视觉支持 `WORK`。
- C2 `f0136` / `f0245`: ref 仍是站姿搬箱；sim 机器人明显趴到 Box026 上方/箱体边缘，低髋局部解清楚。object tracking 和 contact 之所以好，是因为 CEM 找到了用身体/低姿态压住箱子的动态方式。
- C3 `f0090` / `f0163`: sim 机器人倒地，箱子被掀翻/立起；末段右手/手臂接近地面。视觉与 pelvis `0.171m`、RH floor `22.0%` 一致。

## Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| H1: 对 Box026/D003，把 raw target 投到 support face 可降低 wrong-face / inside / low-support | PASS for kinematic gate | Box026 patch support 到 `95.9-100%`，reward inside `0%`；D003 right inside 清零。 |
| H2: projection 不破坏 box004/box023 positive guards | PASS | box004/box023 reward delta p90 均 `0`，inside `0%`；box004 full CEM 仍 `WORK`。 |
| H3: 三条 E092 case 可用同一 external-target CEM 脚本验证 | PASS execution / FAIL for Box026 success | 三条 full CEM 完成；C1 `WORK`，C2/C3 因姿态崩溃 `FAIL`。 |

## 分析

E094 把 E093 的问题拆成两层：

1. target 语义/面分配：`wrist+5cm` wrong-face、inside、low-support；
2. full-body dynamics：CEM 是否能在不倒、不趴、不撑地的姿态下利用这些 target。

Projection 解决了第一层的一部分，尤其是 Box026 的 support/inside 指标和 object tracking。但 C2/C3 说明第二层仍未解决：CEM 可以通过低髋或倒地把物体轨迹跟得很好，且当前 eval 的 head/upper penetration 指标不一定能捕获“趴箱但不穿模”的无效姿态。

因此，Box026 的下一步重点应从 contact target 转到 dynamic feasibility 约束：

- CEM elite gate：reject pelvis below threshold / torso pitch too large / body COM too low / hand-floor contact；
- upright/stability reward：在 contact-active window 中约束 pelvis height、torso orientation、foot support；
- contact validity：区分 hand contact 与 body-on-box contact，避免 object tracking 由躯干或倒地姿态完成；
- 保留 C1 box004 和 box023 作为 positive guard，避免姿态 gate 过强破坏已 work pattern。

## 下一步建议

1. **C1 进入 Holosoma RL 候选。** 输入应使用 `E094P1_box004_083_p2_hbproj.npz` 或其转换出的动作序列；RL 计划需另开，明确 Holosoma 入口、obs/reward、checkpoint 和可视化评估。
2. **C2/C3 暂不进入 RL。** 它们的 full CEM 是失败动态解；直接 RL 会学习低髋/倒地/趴箱策略。
3. **开 E095：Box026 posture/valid-contact CEM gate。** 在 E094 external target 基础上加姿态/有效接触 hard gate 或 elite filter，先只跑 C2 代表 case，再决定是否扩到 C3。
4. **补 eval 指标：body-on-object support / torso pitch / pelvis trajectory。** 当前 head/upper penetration `0%` 不能排除趴箱接触，应把视觉观察转成量化指标。
