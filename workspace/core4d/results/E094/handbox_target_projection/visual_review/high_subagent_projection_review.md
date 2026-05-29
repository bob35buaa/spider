# E094 handbox-aware target projection 可视化复核

## Verdict

总体建议：可以进入一轮 **受控 full CEM**，但不建议把所有 case 同等置信度放入主线。

- `box023_p2`、`box004_083_p2`：positive guards 未被 projection 明显破坏。reward target 基本没有移动，inside 维持 0%，support patch 明显改善支撑，是低风险 CEM candidate。
- `box021_d003_029_p2`：projection 比旧 target 合理，尤其修正了旧 target 支撑不足/右手 inside 的问题；可进入 CEM，但建议标记为 D003 修复验证 case。
- `box026_039_p2`、`box026_135_p2`：new reward/support patch 解决了 low-support / inside 指标，但依赖 0.36-0.63m p90 级别的大位移，视觉上 target 从旧接触/原始接触轨迹被搬到箱体边/角面，属于高风险 candidate；可以进入探索性 CEM，但不应作为默认可信修复直接推广。

## 逐 case 观察

### `box023_p2`

- 数值：left/right patch support 为 100.0% / 97.8%，reward inside 均为 0.0%，reward delta p90 为 0.000m。
- object-local：support patch 和 reward target 都落在 box 外表面附近，没有看到穿入 box 的趋势。right hand 的旧 support 从 58.1% 提升到 97.8%，主要是贴回正支撑面。
- timeline：reward target move 基本为 0，inside/reward inside 全程为负；positive guard 没被 projection 改坏。
- MuJoCo：关键帧中 marker 可见，动作语义仍像围绕箱体抓取/接触，没有出现明显脱离物体的异常目标。

结论：建议进入 full CEM，低风险。

### `box004_083_p2`

- 数值：left/right patch support 为 100.0% / 93.3%，reward inside 均为 0.0%，reward delta p90 为 0.000m。
- object-local：projection 没有把 reward target 大幅搬离原轨迹；patch 主要把 wrist/support 点贴到可支撑面，未见明显 inside。
- timeline：reward delta 除个别尖峰外整体为 0；inside/reward inside 保持为负。old support 原本有长段缺失，patch 后支撑覆盖明显更完整。
- MuJoCo：物体、手、marker 都在画面内；少数帧手掌遮挡 marker，但不影响判断目标大体位置。

结论：positive guard 未被破坏，建议进入 full CEM，低风险。

### `box021_d003_029_p2`

- 数值：left patch support 从 9.3% 提升到 77.3%，right 从 77.3% 提升到 85.3%；right old inside 为 33.3%，projection 后 reward inside 为 0.0%。reward delta p90 left/right 为 0.332m / 0.267m。
- object-local：旧 target 有明显贴面不稳定和右手 inside 风险；projection 后 target 更集中在箱体上沿/侧面，视觉上比旧 wrist5 target 更像可接触的外表面目标。
- timeline：support patch 覆盖比 old 明显稳定，但 reward delta 是间歇性较大改动，不是纯小幅投影。inside 指标修复清楚。
- MuJoCo：关键帧中 marker 在箱体上表面/边缘附近可见，语义上更像手到箱体外表面，而不是旧 target 的穿入或漂移。

结论：比旧 target 合理。建议进入 full CEM，风险中等；应作为 D003 修复验证 case 单独跟踪。

### `box026_039_p2`

- 数值：left/right patch support 为 100.0% / 95.9%，reward inside 均为 0.0%；但 reward delta p90 为 0.522m / 0.362m，触发 REVIEW。
- object-local：new patch 把目标统一压到 `+y` 为主的支撑面，解决了原始 low-support；但旧/raw 轨迹与新 reward 位置距离很大，尤其 left hand 大段从远离箱体的轨迹被投到箱体边缘。
- timeline：几乎整个有效接触段都有 0.4-0.5m 左右的 left reward delta，right 也长期在 0.25-0.36m；这说明修复依赖强投影，而不是局部校正。
- MuJoCo：marker 可见，且大多贴在箱体上表面/侧边；不过当前视角较远，难以确认 marker 是否恰好在可抓取接触点，尤其边角和遮挡帧。

结论：support/inside 指标被修复，但位移过大。只建议作为高风险探索性 CEM candidate，不建议直接纳入低风险主线。

### `box026_135_p2`

- 数值：left/right patch support 均为 100.0%，reward inside 均为 0.0%；reward delta p90 为 0.486m / 0.631m，right hand 位移尤其大。
- object-local：right hand 的旧/raw 轨迹和新 target 分离很明显；projection 后虽然贴到支撑面，但更像把目标迁移到箱体固定外表面，而非保留原动作语义。
- timeline：right reward delta 长时间约 0.6m，handbox gap 也接近 0.55m；left 在多个区间约 0.48m。inside 风险被清除，但代价很高。
- MuJoCo：关键帧完整，marker 基本可见；视觉上 target 在箱体上表面/边缘附近，动作阶段可读，但远景不足以排除“过度投影导致奖励目标不再对应手部意图”的问题。

结论：解决了 inside/support，但位移过大且语义风险最高。若进入 full CEM，必须标记高风险，并与更保守投影阈值版本对照。

## MuJoCo 视角与 marker 可见性

- 5 个 keyframes 图均为 6 帧拼图，人物和物体主体完整，没有明显裁切导致无法判断的 case。
- marker 在大多数关键帧可见；颜色图例为 raw orange/green、old red/blue、patch yellow/cyan、reward purple、handbox green。
- 局限：当前是单一远景视角，部分 marker 被手掌、箱体边或身体遮挡；对 `box026_*` 这种边角投影和大位移 case，建议补充局部近景/多视角 keyframes，尤其需要看 target 是否贴在可抓取面而不是视觉上被箱体轮廓遮住。

## 风险

- 最大风险是 `box026_039_p2` 和 `box026_135_p2` 的 reward target 过度迁移：指标通过来自大位移 projection，可能改变原始接触语义，让 CEM 优化到“碰投影点”而非复现真实手-物交互。
- `box021_d003_029_p2` 虽然合理性提升明显，但 reward delta 已接近 30cm 级，应作为修复验证而不是 guard。
- MuJoCo 可视化目前能做完整性检查，但不足以精判接触合法性；高风险 case 需要更近视角或 per-hand 局部渲染。

## 进入 CEM 建议

建议进入 full CEM：

- 低风险主线：`box023_p2`、`box004_083_p2`
- 中风险验证：`box021_d003_029_p2`

建议进入探索性 full CEM，但标记高风险并单独汇总：

- `box026_039_p2`
- `box026_135_p2`

不建议把 `box026_*` 的结果直接作为通过标准。若资源有限，优先跑 `box026_039_p2` 一个高风险代表；`box026_135_p2` 因 right reward delta p90 达 0.631m，应排在高风险队列最后，或先生成更保守 projection 版本再比较。
