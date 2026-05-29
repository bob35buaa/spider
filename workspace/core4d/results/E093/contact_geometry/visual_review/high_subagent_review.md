# E093 High-Reasoning Visual Review

Reviewer: subagent `019e720f-83fb-7193-a7a2-3924ab7d8527`

## 总体结论

1. `raw contact` 与 `wrist+5cm` 大多不是同面/同区域。14/14 行 `wrist5->raw mean` 都超过 20cm；object-local overlay 显示 `wrist+5cm` 经常贴到边、角、相邻面，甚至另一侧。
2. `sphere` 会导致接触位置不准。它不是唯一问题，但会把手部接触简化成球面距离，无法表达 palm/handbox 的平面接触和朝向；所有 case 的 `sphere p90 |gap|` 都超过 10cm，box025/box026 达到 45-61cm。
3. `handbox proxy` 通常比 `sphere` 和 `3-box` 更接近 raw，但只是相对更近，不是根治。summary 中 handbox 是 14/14 行的 best proxy；但 handbox p90 仍普遍超过 10cm，box025/box026 仍在 0.53-0.59m 量级。
4. box004/box023 容易 work 的原因不是 raw/wrist 完全正确，而是偏差较小且仍在 G1 可达的外侧/上沿区域；box021 D003/box026 失败则更像接触语义错位：inside、低 support、错面、过大 offset 共同让优化只能在“追 reward”和“避免穿模/摔倒”之间冲突。

## 逐 case 观察

| case | 观察 |
|---|---|
| `box023_p2` | L/R `wrist5->raw mean` 都约 0.282m，不是同一区域；但 wrist5 inside 为 0，support 尚可，MuJoCo 中手仍在箱体附近外侧/上沿，偏差没有把姿态推崩。handbox p90 比 sphere 低，尤其 R: 0.173m vs 0.212m。 |
| `box025_p2` | raw/wrist 偏移很大：L 0.578m，R 0.641m。wrist5 support 100% 只是落在某个支撑面，不代表 raw 同区域；sphere/3-box/handbox 都是约 0.5m 级 gap。 |
| `box004_083_p2` | L/R 偏移 0.283m / 0.212m。raw face 是两侧 `-y/+y`，wrist5 混到 `+x/-z/-x` 等，R support 只有 24.8%。尺寸较小、机器人能把手送到箱附近，E092 C1 能 work 更像偏差仍可被动力学/可达性容忍。 |
| `box021_person1` | 老 control 明显好于 D003 fail：L wrist support 98.9%，R 75%，inside 低；handbox p90 L 0.110m，是全表最接近的一行。说明 box021 失败不是物体名必然失败，而是具体 retarget/contact placement 出问题。 |
| `box021_d003_029_p2` | 典型失败形态：L wrist support 只有 9.3%，R wrist inside 33.3%。raw/wrist face 明显错配，timeline 中偏差长期在 25-33cm。D003 contact target 已经把 G1 推到错面/内嵌风险区域。 |
| `box026_039_p2` | raw 两手都在 `-y`，wrist5 大量跑到 `+y/-z`，几乎是错面/边缘目标；wrist support L/R 仅 19.5% / 15.4%。失败根因偏结构性：大平面物体 + 错面 target + 低 support。 |
| `box026_135_p2` | raw 两手主要 `-z`，wrist5 分到 `+y/-x`，不是同面。R inside 12.2%，对应 E092 C3 的 inside-risk；handbox 仍略优，但只是跟着错误区域整体移动，不能恢复 raw 接触语义。 |

## 判断

E093 支持这个结论：当前 reward 使用的 `wrist+5cm` / sphere 类 target 与 raw contact 的几何语义不一致。handbox/3-box 只能缓解 proxy 形状误差，不能解决 raw target 与 retargeted hand region 错面的问题。下一步若要提升 box021 D003/box026，应优先修正上游 contact target / face assignment，而不是继续调 sphere 半径或 reward 权重。
