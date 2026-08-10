# E195 更紧 hand gate Full 结果：候选层合规未转化为穿透改善

_Core4D · Phase 58 · 2026-08-10 · 对应 [plan221](../plan/221_E195_stricter_hand_gate_plan.md)_

## Summary

- E195 在 E192 A2 的 15 个同 case、同 seed、同预算配对上，只把 hand gate 从 `-0.010 / 0.05 / -0.015 m` 收紧到 `-0.008 / 0.05 / -0.012 m`
- Full CEM 按本机 GPU0 七例、远程 RTX 6000 Ada GPU0/1 各四例完成 `15/15`，失败 `0`；统一评测、15 条 self 视频、15 条 paired 视频和 14 条必审视觉记录均闭合
- box024 的 3 mm 穿透由 `0.3082` 升到 `0.3433`（恶化 `+3.51pp`），仅 `3/9` case 改善；固定 12 mm 口径也仅 `1/9` 改善，C1/C3 FAIL
- 接触 stop-loss 没有触发：box024 3 mm 接触只下降 `4.26pp`、总体接触为 `0.8078`，视觉未确认承重断联，C2 PASS
- 安全门出现预注册回退：box004 有 3 个 case×gate PASS→FAIL，box024 单例 leg 增量最高 `+7.64pp`，C5 FAIL

> **最终判决：`SAFETY_REGRESSION`。不升级 A3，不继续沿 hand-gate-only 方向机械收紧。**

## 1. Experiment Motivation

E192 A2 证明候选层 hand hard floor 可以按配置执行，但 box024 的最终物理穿透改善不稳定。E195 检验一个更直接的问题：在其余条件全部冻结时，把允许穿透的起点和绝对硬地板再各收紧 2–3 mm，能否把 A2 的不稳定收益变成 box024 多数 case 上的一致改善。

该结果服务于是否继续投入 hand-gate-only 路线的决策。E195 的直接基线是 E192 A2，不是 E172/E173 的历史 A0；因此本报告只讨论“相对 A2 再收紧”的增量效果，不把 box024 与 box004 的差异解释为纯尺寸因果。

## 2. Experiment Setup

| 项目 | E192 A2 基线 | E195 A3 |
|---|---|---|
| Case | box024×9 + box004×6 | 完全相同的 15 case |
| hand min SDF | `-0.010 m` | `-0.008 m` |
| max violation | `0.05` | `0.05` |
| hard floor | `-0.015 m` | `-0.012 m` |
| CEM | `1024 samples × 32 iterations` | 相同 |
| seed | `0` | `0` |
| 其余配置 | PRG on、`rubber_hull`、`ref_fk`、no-gravcomp、kp `500/50` | 全部冻结 |

调度固定为 local GPU0 `7/7`、Ada GPU0 `4/4`、Ada GPU1 `4/4`，与 E192 最终完成映射一致。本轮在已有程序上叠加运行，没有等待空卡、停止其他程序或动态重分配。02:54 启动，local 于 05:21 完成，远程两卡于 06:21 完成；自动回收、评测与渲染于 06:23 闭合。

主要执行与评测入口均已固化：

- 三卡入口：`scripts/launch/active/run_E195_hybrid_3gpu.sh`
- 远程回收：`scripts/launch/active/pull_E195_remote_Ada6000_results.sh`
- 统一评测：`scripts/eval/wrappers/eval_E195_stricter_hand_gate.sh`
- paired 报告：`scripts/eval/reports/gen_E195_comparison.py`
- 离线渲染：`scripts/launch/active/run_E195_render_all.sh`

## 3. Core Algorithm or Method

E195 没有更改 CEM 目标函数或 rollout 动力学，只改变候选筛选的 hand SDF 策略包。对 100 帧候选：

- 最多 5 帧可以低于 `-0.008 m`
- 任意一帧都不得低于 `-0.012 m`
- 无满足条件候选时仍沿用既有 fallback 机制

评测链分成两层：候选层检查 selected candidate 是否满足 hard floor；rollout 层用独立物理接触/穿透、12 门和视觉证据判断最终轨迹。这个分层很关键，因为 E195 的核心负结果正是“候选合规，但 rollout 没有单调改善”。

## 4. Metrics

| 指标 | 方向/门槛 | 用途 |
|---|---|---|
| 3 mm penetration frame fraction | 越低越好；box024 至少再降 `0.03` 且 `≥6/9` 改善 | C1 主效应 |
| 3 mm in-mask contact | E195 不得比 E192 低超过 `0.05` | C2 接触 stop-loss |
| overall in-mask contact | `≥0.75` | C2 总体承重覆盖 |
| fixed 12 mm frame fraction | 越低越好；box024 `≥6/9` 改善 | C3 浅层物理穿透 |
| non-fallback hard-floor violation | 必须为 `0` | C3 候选层执行正确性 |
| hand valid / fallback | valid `≥0.60`；fallback `≤E192+0.15` | C4 可行域健康 |
| 12 门 PASS→FAIL 与 leg delta | box004 不新增；box024 单例 leg `≤+0.05` | C5 安全回退 |
| paired improvement | `E192 - E195`，越大越好 | C6 效应与 bootstrap CI |

连续指标先按 case 计算，再在物体内取均值。C6 使用 case-level paired bootstrap、seed 0、10,000 次重采样并报告 95% CI。C6 的 PASS 只表示诊断完整，不代表效应方向有利。

## 5. Results

### 5.1 执行与产物闭合

| 项目 | 结果 |
|---|---:|
| Full manifest | `15/15 run_complete_pending_eval` |
| 运行失败 | `0` |
| qpos finite / 必要 diagnostics | `15/15` |
| 评测 | E192 `15/15` + E195 `15/15`，errors `0` |
| metrics / paired deltas | `30` / `15` 行 |
| E195 self / E192-E195 paired 视频 | `15` / `15` |
| 必审视觉 case | `14/14 reviewed` |

### 5.2 物体内均值

| Object | Metric | E192 A2 | E195 A3 | E195−E192 |
|---|---|---:|---:|---:|
| box024 | 3 mm penetration | 0.3082 | 0.3433 | **+0.0351** |
| box024 | 3 mm contact | 0.3864 | 0.3439 | −0.0426 |
| box024 | overall contact | 0.8037 | 0.8078 | +0.0041 |
| box024 | fixed 12 mm | 0.0295 | 0.0390 | **+0.0094** |
| box024 | leg penetration | 0.1050 | 0.0922 | −0.0128 |
| box004 | 3 mm penetration | 0.1070 | 0.1072 | +0.0002 |
| box004 | 3 mm contact | 0.4403 | 0.3779 | −0.0624 |
| box004 | overall contact | 0.6199 | 0.5474 | −0.0725 |
| box004 | fixed 12 mm | 0.0046 | 0.0129 | **+0.0083** |
| box004 | leg penetration | 0.0771 | 0.1287 | **+0.0516** |

更紧的候选 gate 没有在任何一个物体上降低平均 fixed-12mm 穿透。box024 的 3 mm penetration 不仅未达到预期下降，反而恶化 `3.51pp`；9 条中只有 `026_p2`、`028_p1`、`030_p1` 三条改善。

case 级异质性很大：

- `box024_028_p2`：3 mm penetration `+18.52pp`
- `box024_031_p1`：3 mm penetration `+17.65pp`
- `box024_028_p1`：penetration 改善 `−14.58pp`，但 leg penetration 增加 `+7.64pp`
- `box004_083_p1`：penetration 改善 `−2.94pp`，同时 3 mm contact 下降 `25.40pp`、leg penetration 增加 `39.22pp`
- `box004_086_p2`：release gate 由 PASS 变 FAIL，视觉出现明显末段塌姿

### 5.3 候选 gate 健康与实际效应

| Object | E195 hand valid | E192 fallback | E195 fallback | 上限 | C4 |
|---|---:|---:|---:|---:|---|
| box024 | 0.9435 | 0.2987 | 0.3152 | 0.4487 | PASS |
| box004 | 0.8853 | 0.1949 | 0.3391 | 0.3449 | PASS |

所有 E195 non-fallback selected ticks 的 hard-floor violation 为 `0`，说明 A3 策略包确实执行。与此同时，box004 fallback 增加 `14.42pp`，距离预注册上限只剩约 `0.58pp`；可行域虽未正式塌缩，但已接近门槛。

paired improvement 定义为 `E192−E195`：

| Object/effect | Estimate | 95% CI |
|---|---:|---:|
| box024 | −0.0351 | [−0.1000, 0.0325] |
| box004 | −0.0002 | [−0.0519, 0.0379] |
| signed DiD | −0.0349 | [−0.1125, 0.0492] |

三个区间都跨过 0。点估计方向表明 A3 相对 A2 对 box024 更不利，但单 seed、9/6 case 的样本不足以把物体差异解释为稳定的尺寸效应。

### 5.4 Claims 判定

| Claim | 状态 | 证据 |
|---|---|---|
| C1 | FAIL | box024 penetration `0.3082→0.3433`；仅 `3/9` 改善 |
| C2 | PASS | 3 mm contact 损失 `0.0426<0.05`；overall contact `0.8078`；无视觉确认断联 |
| C3 | FAIL | fixed12 仅 `1/9` 改善；hard-floor violation `0` |
| C4 | PASS | 两物体 valid/fallback 均过预注册门 |
| C5 | FAIL | box004 3 个 PASS→FAIL；box024 max leg delta `+0.0764` |
| C6 | PASS | 两物体 paired effect、CI 与 signed DiD 已完整报告 |
| C7 | PASS | 15/15 评测、30 条视频、14/14 必审视觉闭合 |

box004 的 3 个 PASS→FAIL 是 `083_p1` 的 contact 与 lower-body，以及 `086_p2` 的 release。按预注册优先级，C5 失败直接把完整结果判为 `SAFETY_REGRESSION`。

## 6. How to Read the Figures

每张四阶段图的顺序是左上 grasp、右上 lift、左下 carry、右下 place；每个阶段内部 E192 A2 在左、E195 A3 在右。图像没有数值坐标，作用是辨别手是否离开、箱体是否滑动/倾斜，以及 lower-body 是否替代手部承担任务。

![box024 028 p2 A2/A3 four-phase](../results/E195/s6_downstream/render/keyframes/box024_20231011_028_p2_4phase.png)
_Figure 1：`028_p2` 的 A3 在 carry/place 明显改变身体与箱体朝向；双手仍可见接触，但 penetration、hand/object gates 同时回退。_

![box024 031 p2 A2/A3 four-phase](../results/E195/s6_downstream/render/keyframes/box024_20231011_031_p2_4phase.png)
_Figure 2：`031_p2` 两臂都采用深蹲贴箱，A3 没有可确认 hand-away；说明 contact stop-loss 未触发不等于姿态干净。_

![box004 083 p1 A2/A3 four-phase](../results/E195/s6_downstream/render/keyframes/box004_20231003_2_083_p1_4phase.png)
_Figure 3：`083_p1` 的 A3 身体/下肢介入更强，与 leg `+39.22pp` 以及 contact/lower-body PASS→FAIL 一致。_

![box004 086 p2 A2/A3 four-phase](../results/E195/s6_downstream/render/keyframes/box004_20231003_2_086_p2_4phase.png)
_Figure 4：`086_p2` 的 A3 在 place 形成明显地面附近塌姿和物体位移，对应 release PASS→FAIL。_

完整逐例判读见 [`visual_review.tsv`](../results/E195/s6_downstream/render/keyframes/visual_review.tsv)。14 条必审 case 均覆盖四阶段；box024 `9/9` 未确认承重接触丢失。

## 7. Interpretation

### 观察事实

1. A3 候选层 hard floor 的 non-fallback 违规为零，因此不能把负结果归因于阈值未接线或配置未生效。
2. rollout 层的 3 mm 与 fixed-12mm 穿透都没有随候选 gate 单调变好；box024 反而由 0.3082 升到 0.3433。
3. C2 通过说明主要失败机制不是简单“把手拿开”：overall contact 略升，四阶段视觉也未确认 box024 承重断联。
4. 安全代价集中在 case 级重排，而非所有指标一起崩溃：一些 case 降低手穿透时增加 lower-body 介入，另一些 case 则改变身体/箱体朝向后手穿透更严重。
5. box004 fallback 接近上限，说明继续收紧很可能先压缩小箱可行候选，而不是获得稳定物理收益。

### 机制推断

最符合证据的解释是候选层约束与最终 rollout 物理之间存在传导缺口。CEM 可以通过身体朝向、箱体倾斜、lower-body 接近以及 fallback 候选重新分配代价；因此“selected candidate 不越过 hard floor”不等价于“最终动作在物理评测中更少穿透”。

E192 A2 已经表现出这种非单调性，E195 A3 把它进一步放大：候选规则更紧，但 fixed12 在两个物体上都变差，且 box024 只有 `1/9` case 改善。这使“阈值还不够紧”成为不再受数据支持的解释。继续把阈值推向零，预期更可能增加 fallback 与姿态代偿，而不是恢复稳定的接触拓扑。

这一段是机制推断，不是新因果实验。现有设计同时改变 min-SDF 与 hard floor，且只有一个 seed，不能识别究竟是哪一项或哪个优化阶段触发了重排。

## 8. Conclusion and Discussion

E195 回答了“沿 E192 再收紧一步是否有效”：没有。A3 既没有降低 box024 平均穿透，也没有改善固定 12 mm 的 case 一致性；虽然候选 gate health 与接触 stop-loss 仍过线，但新增跨门和 worst-case lower-body 回退。

最终决策为 `SAFETY_REGRESSION`，A3 不升级为 production 默认。更重要的是，E192→E195 的连续证据不支持继续做 A4 式 hand-gate-only 收紧；后续应优先改变抓握拓扑或支撑机制，而不是继续压阈值。

## 9. Limitations and Caveats

- 只有 seed 0，box024/box004 分别只有 9/6 条；bootstrap CI 较宽
- box024 与 box004 同时混有尺寸、几何、动作和抓取拓扑差异，signed DiD 不是尺寸因果效应
- A3 同时收紧 min-SDF 与 hard floor，不能分解单参数贡献
- 四阶段视觉适合确认 gross hand-away 与姿态重排，但不保证捕捉阶段间的瞬时滑动；未见断联应理解为“未确认”，不是逐帧接触测量
- E192/E195 复用相同 worker 映射降低了设备差异，但 GPU/CUDA 数值非确定性仍可能影响个别边界 case

## 10. Next Steps

1. 停止继续收紧 hand-gate-only 参数，不启动 A4 阈值臂。
2. 优先执行 E193 的零 GPU 抓握拓扑审计，检验对置接触不足是否解释 A2/A3 的非单调传导。
3. 若需要新的 Full，对受控机制臂做小范围验证：保留 A2/A3 已知安全边界，单独改变抓握拓扑或采用 E194 已证实稳定的 G1 gravcomp；不要把两者与新 gate 阈值叠加在同一首轮。
4. 在新 GPU 实验前，先用 E192/E195 的 30 条已有 rollout 做 phase-level selected-SDF、fallback、箱体倾斜和 lower-body 代偿相关审计，缩小待验证机制。

## Reproducibility Notes

| Evidence | Path |
|---|---|
| 冻结计划 | `plan/221_E195_stricter_hand_gate_plan.md` |
| Full manifest | `results/E195/s6_downstream/manifests/cem_full_manifest.tsv` |
| 配置审计 | `results/E195/preflight/config_audit.json` |
| 30 行统一指标 | `results/E195/s6_downstream/eval/full/e195_case_metrics.tsv` |
| 15 行 paired delta | `results/E195/s6_downstream/eval/full/e195_paired_deltas.tsv` |
| Claims 与判决 | `results/E195/s6_downstream/eval/full/e195_claims.json` |
| Self / paired 视频 | `results/E195/s6_downstream/render/full/` · `results/E195/s6_downstream/render/paired_e192_e195/` |
| 视觉 authority | `results/E195/s6_downstream/render/keyframes/visual_review.tsv` |
| 自动收尾日志 | `logs/E195/monitor/full_finalize.log` |

本报告不覆盖 RL、Holosoma 或 production 导出；E195 只对 CEM hand-gate A3 的 15-case paired Full 作结论。
