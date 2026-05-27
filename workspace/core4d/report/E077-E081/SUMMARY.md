# E077-E081 组会汇报摘要

## Summary

- 主问题：把 CORE4D contact 从手写/标量窗口推进到可泛化的 3cm per-EEF mask，并验证 CEM 在多 case 上是否仍可用。
- 结论一：数据与 mask pipeline 已打通。E077 生成 `raw/spider/eval=(178/136/227,2,2)` mask，并构造 `box023_person2`；E079 扩展到 10+ case，`11/12` 可预处理运行。
- 结论二：per-EEF mask 是必要修正，但不能单独修好坏 reference。E078 中 `box023_p1` 仍失败，`box023_p2` 明显更稳定，说明数据/retarget 可行性是主瓶颈。
- 结论三：E079 case-specific window 主成功率为 `6/10=60%`，高于 fixed post2 的 `2/10=20%`，但三阈值会产生 false positive。
- 结论四：E080/E081 说明“物理合理性指标”必须进入评估。给 box025_p2 加腿/脚-箱碰撞后，腿箱干涉从 `28.9%` 降到 `7.5%`，但箱体 lift 未改善，仍不是 strict success。
- 推荐方向：停止围绕 p1/手写窗口局部调参；转向 data-quality audit、case-specific 语义评估、leg-object/lift-aware 指标。

## 1. Experiment Motivation

E075/E078 前后的核心疑问是：`box023_p1` 的失败到底来自 contact mask 设计，还是 reference/retarget 本身不可执行。E077-E081 围绕这个问题逐步验证：先修 contact 表达，再做多 case 泛化，最后补上大物体和腿/箱物理约束。

## 2. Experiment Setup

| 实验 | 作用 | 主要输出 |
|---|---|---|
| E077 | 生成 CORE4D 3cm per-person/per-hand mask；构造 `box023_person2` | mask `.npz/.csv/.json`；p2 SPIDER case |
| E078 | `box023_p1/p2` 接入 3cm per-EEF mask 跑 CEM | p1 无改善；p2 positive guard |
| E079 | 10+ case 泛化验证 | 10 main + 1 guard，case-window 成功 `6/10` |
| E080 | `box025_p1/p2` 边界/负控复查 | p1 false positive；p2 partial positive |
| E081 | 派生 `_legobj` scene 加腿/脚-箱 contact pair | box025_p2 干涉下降，lift 未改善；box023_p2 guard 保持 |

## 3. Core Algorithm or Method

- Contact：用 CORE4D visualization-style `3cm` 几何距离生成 `(T, person, hand)` mask；这是 contact proxy，不是真实物理接触标注。
- Reward 接入：`contact_hdmi` 支持 per-EEF `(T,2)` mask；E079 起关闭 hand-crafted hold window，主变量改为 `core4d_3cm` mask。
- 评估口径：从 fixed `post2=2.0-3.6s` 改为 case-specific contact/intent window。
- 物理修正：E081 不改原始 scene，只创建 `*_legobj` 派生 task，新增 16 个腿/脚 geom 到 `object_collision` 的 contact pair。

## 4. Metrics

- `CaseWin`：主量化成功；要求 `pelvis_z_min >= 0.55m`、`sim contact >= 50%`、`obj_err_mean <= 0.20m`，越高/低方向分别按指标定义。
- `Fixed post2`：历史诊断窗口，只适合 `box023` 后半段问题，不再作为多 case 主结论。
- `Leg intf`：腿/脚-箱几何穿入比例，越低越好。
- `Bottom mean` / floor contact：箱底高度与触地 proxy，用于判断是否真的 lift；E081 表明它比单纯 contact 更关键。
- Visual label：人工视频复核标签，用于修正三阈值 false positive。

## 5. Results

| 实验 | 观测结果 | 对结论的含义 |
|---|---|---|
| E077 | p2 case 构造通过；但 p1/p2 retarget object qpos 最大差 `6.2cm` | p2 可单人使用；p1/p2 不能直接合成双机器人，需 common-scale alignment |
| E078 | p1 per-EEF mask 无实质改善；p2 右脚 step/hip ctrl 明显更接近 ref | 失败不主要是 scalar mask；p2 ref/contact 质量更好 |
| E079 | pipeline `11/12` 可运行；main `CaseWin=6/10`，fixed `2/10` | 数据 pipeline 可泛化，但成功判据仍不足 |
| E080 | box025 p1/p2 `CaseWin=2/2`，fixed `0/2`；p1 false positive，p2 partial positive | 三阈值会高估大物体/贴靠式动作 |
| E081 | box025_p2 `Leg intf 28.9% -> 7.5%`，`Bottom mean -0.073m -> -0.075m`；box023_p2 guard 基本保持 | 加腿/箱碰撞有效减少穿模，但 box025 瓶颈转为 lift/floor-contact |

## 6. How to Read the Table

上表只保留组会要点：`观测结果` 是日志/CSV 中的测量值，`含义` 是基于这些结果的解释。`CaseWin` 可比较多 case，但不能单独代表视觉可用；凡是 p1/p2 或 box025 结论，都需要结合 visual label 和 leg/lift 指标读。

## 7. Interpretation

E077-E081 支持一个更清晰的判断：当前系统的主要瓶颈已经不是“有没有 contact mask 接口”，而是 reference/retarget 可执行性与评估语义。`box023_p2`、`bucket007_p2`、`bucket005_s2_p2` 说明 CEM/reward stack 在高质量输入上可以工作；`box023_p1`、`bucket005_s2_p1`、`desk021_p1` 说明三阈值会误判；`box025_p2` 说明大物体还需要 lift-aware 评估。

## 8. Conclusion and Discussion

短结论：3cm per-EEF mask + case-specific window 是正确方向，E077 pipeline 已可用于多 case；但后续不应继续靠手写窗口或局部 reward 修 p1。真正要推进泛化，需要把数据质量筛选、trim/ref feasibility、腿/箱碰撞、object lift/floor-contact 纳入统一评估与训练选择。

## 9. Limitations and Caveats

- 多数 variant 是单次运行，缺少多 seed 稳定性。
- 3cm mask 是几何 proxy，不是标注接触真值。
- visual label 来自人工复核，主观但必要；现有自动三阈值会产生 false positive。
- p1/p2 retarget 有 scale 差异，不能直接拼双人场景。
- box025 的 partial positive 不能等同 strict carry success，因为 lift/floor-contact 仍失败。

## 10. Next Steps

1. 把 E081 的 `leg_box_interference`、`leg_object_contact`、`object_floor_contact`、`object_bottom_proxy` 合入 E079/E080 统一评估。
2. 对 `box021_p1`、`bucket007_p2`、`bucket005_s2_p2`、`box023_p2` 批量跑 `_legobj` 派生 scene，验证正例是否稳定。
3. 做 data-quality audit：trim 前摇、foot support、pelvis height、object-human relative trajectory，优先筛掉不可执行 ref。
4. 对 box025_p2 转向 lift/floor-contact reward 或将其定位为 partial carry / push-assist，而非 strict lift success。

## Reproducibility Notes

- 代码版本：`0dc1e17`
- 主日志：`log/98_E077_3cm_contact_mask_and_person2_results.md` 到 `log/102_E081_leg_object_collision_results.md`
- 主结果：`results/E078/comparison.csv`、`results/E079/comparison.csv`、`results/E080/comparison.csv`、`results/E081/comparison.csv`
- E077 证据：`results/E077/contact_masks/box023/audit_summary_3cm.json`、`results/E077/box023_person2_verify_summary.json`
- E079-E081 聚合：`results/E079/aggregate_summary.json`、`results/E080/aggregate_summary.json`、`results/E081/aggregate_summary.json`
