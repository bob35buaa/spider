# E094 CEM full autocam 视觉复核

复核对象：

- `workspace/core4d/results/E094/cem/full/E094P1_box004_083_p2_hbproj_full_autocam.mp4`
- `workspace/core4d/results/E094/cem/full/E094P2_box026_039_p2_hbproj_full_autocam.mp4`
- `workspace/core4d/results/E094/cem/full/E094P3_box026_135_p2_hbproj_full_autocam.mp4`
- `workspace/core4d/results/E094/cem/full/keyframes_autocam/<variant>/f*.jpg`
- `workspace/core4d/results/E094/cem/full/full_eval_summary.md`

## 结论

autocam corrected 视角有效：三个 case 的 ref/sim 画面都能看到完整机器人、完整箱子和地面接触关系，不再是只能看到上半身的裁切。因此当前 keyframes/video 足以支持视觉复核。

逐 case 结论与 metrics 一致：C1 `box004_083` 视觉上支持 `WORK`；C2 `box026_039` 是物体/contact 指标很好但机器人趴箱、低髋的失败；C3 `box026_135` 是更严重的低髋/倒地，并伴随箱子翻转和右手触地风险。建议只让 C1 进入后续 RL 候选；C2/C3 不应作为通过样本直接进入 RL，应先修正 CEM/筛选目标中的姿态稳定约束，否则 RL 会从“物体轨迹正确但人体姿态崩坏”的坏先验开始。

## 逐 case 观察

### C1: `E094P1_box004_083_p2_hbproj`

- 关键帧：`f0000` 到 `f0209`，视频 210 帧，约 4.20s。
- autocam 覆盖完整机器人和 box004；sim 侧能看到脚、髋、躯干、手和箱子全程关系。
- 机器人在搬运/接触阶段保持站立姿态，后段有弯腰和侧向步态，但没有视觉上的倒地、趴箱、头/上身穿箱或手撑地。
- 箱子位置和 ref 对齐较好，接触关系稳定；视觉上支持 `WORK` 判定。

### C2: `E094P2_box026_039_p2_hbproj`

- 关键帧：`f0000` 到 `f0245`，视频 246 帧，约 4.92s。
- autocam 覆盖完整机器人和 Box026；失败不是视角误判。
- ref 侧是机器人站立搬运箱子；sim 侧中段开始机器人上身压到箱子上，后段基本趴在箱子上完成物体位置跟随。
- 箱子整体没有明显翻倒，失败主因是低髋/趴箱/姿态崩坏，而不是物体轨迹本身失控。

### C3: `E094P3_box026_135_p2_hbproj`

- 关键帧：`f0000` 到 `f0163`，视频 164 帧，约 3.28s。
- autocam 能清楚看到完整机器人、箱子、足端和手端。
- sim 侧早中段出现深弯腰和重心过低，随后机器人倒向箱体/地面；后段箱子明显被翻起或侧翻，机器人已倒地或接近仰倒。
- 失败模式是低髋、倒地、局部手触地风险和箱体翻转叠加，比 C2 更严重。

## Metrics 对照

| case | status | contact | obj_mean / obj_max | pelvis_min | hand floor | visual check |
|---|---:|---:|---:|---:|---:|---|
| C1 box004_083 | WORK | 61.0% | 0.007m / 0.018m | 0.658m | LH 0.0%, RH 0.0% | 站立完成，箱体跟随好，无明显崩坏 |
| C2 box026_039 | FAIL | 80.5% | 0.002m / 0.017m | 0.440m | LH 0.0%, RH 0.0% | contact 和物体误差很好，但机器人趴箱、低髋 |
| C3 box026_135 | FAIL | 41.5% | 0.010m / 0.041m | 0.171m | LH 0.0%, RH 22.0% | 倒地、箱子翻转、右手触地风险 |

metrics 与视觉观察总体一致。C2 是最典型的“target projection 让物体误差/contact 看起来很好，但姿态失败”的样本：80.5% contact 和 2mm mean object error 并不代表机器人动作可用，`pelvis_min=0.440m` 和视觉趴箱更能解释 FAIL。C3 的 `pelvis_min=0.171m`、RH floor 22.0%、object max 4.1cm 与视觉倒地/箱体翻转一致。

## 风险/建议

- 不建议把 C2/C3 直接推进 RL 作为成功初始化。当前失败说明目标投影已经能把箱体和接触点拟合得很好，但缺少足够强的全身姿态、支撑和防跌倒约束；直接 RL 可能会学习到“靠趴箱/倒地维持物体误差”的局部策略。
- C1 可以作为 RL 候选或正样本继续推进，但仍建议保留 autocam 视觉 gate，避免只依赖 obj/contact 数值。
- C2/C3 下一步优先修 CEM gate/reward：提高 pelvis/stability 约束权重或硬门槛；加入 torso-on-box / pelvis-low / fall 状态过滤；对 box026 增加箱体翻转/姿态偏差 gate；把 hand-floor 和 body-floor 事件纳入通过条件。
- 对 target projection 路线，建议把通过标准从“object error + contact”扩展为“object/contact + full-body feasibility”。当 object/contact 已经很好但姿态崩掉时，下一步应先做 CEM 目标和 safety gate 修正，再进入 RL；RL 只适合作为通过这些姿态 gate 后的 refinement，而不是用来修补明显倒地的 CEM 种子。
