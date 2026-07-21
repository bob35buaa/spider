# E168 已有 CEM case 全面评测

日期：2026-07-17

状态：production 仍在运行；本报告覆盖评测快照时产物齐全的 `20/40` 条，不是最终 40 条结论。

## 结论

用户观察到的重定向质量问题得到量化确认。20 条已有结果中只有 `2/20` 通过当前全部 numeric hard gates，其余 `18/20` 至少命中一项失败；当前不得批量进入 RL export。

通过 numeric gates 的两条为：

- `box004_20231003_2_086_p1`
- `box021_20231011_036_p2`

两条的 8-frame `ref|sim` sheet 均未见明显跌倒、物体飞离或大尺度穿插，但 production 尚未结束，仍不在本轮作最终 release。

## 评测口径

评测入口：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E168_e167a_metrics.sh
```

- 核心标准：`core4d-e154-physics-contact-v1`，直接 import `eval.core.core_metrics`。
- tracking：无 fall，terminal pelvis-z error `<=0.08m`。
- z-only：左右腕踝四个 body 相对固定 kinematic trajectory 的 peak z error `<=0.25m`。
- contact：raw 3cm mask 内 physics contact `>=0.25`。
- release：有 trailing release window 时，3mm false contact `<=0.30`。
- penetration：>3mm physics penetration frame fraction `<=0.25`。
- lower body：leg/object interference `<=0.05`。
- 新 case 没有同 case E163 baseline，relative delta 显式为 unavailable，不构造伪 baseline。

`box021_20231020_020_p2` 的 reference contact 持续到末帧，没有 trailing release window。该 row 的 release gate 记为 `NOT_APPLICABLE_NO_RELEASE_WINDOW`，不把 NaN 填为 0，也不误判失败；它仍因 lower-body gate 失败。

## Gate 结果

| gate | pass | fail/N/A | 结论 |
|---|---:|---:|---|
| numeric all | 2/20 | 18 | 当前总体通过率 10% |
| tracking | 20/20 | 0 | 单独的 pelvis/fall gate 无法识别本批主要质量问题 |
| fixed-reference z-only | 14/20 | 6 | 6 条腕踝 z 偏差超过 0.25m |
| raw contact | 19/20 | 1 | `033_p1` 仅 0.222 |
| release | 19/19 applicable | 1 N/A | 适用 case 全通过；不是本批主因 |
| >3mm penetration | 15/20 | 5 | 最差 `034_p2=0.517`、`038_p2=0.462` |
| lower body | 6/20 | 14 | 最大失败源，70% rows 超过 0.05 |
| fall | 20/20 no-fall | 0 | 差视频不等于跌倒视频 |

失败模式可叠加，因此各失败数之和大于 18。

## 最差样本

| 维度 | case | 值 |
|---|---|---:|
| body-z peak | `box021_20231018_029_p1` | 0.319m |
| EEF position mean | `box021_20231018_030_p1` | 42.20cm |
| object position mean | `box021_20231018_028_p1` | 23.08cm |
| raw contact | `box021_20231018_033_p1` | 0.222 |
| >3mm penetration | `box021_20231011_034_p2` | 0.517 |
| lower-body interference | `box021_20231018_033_p2` | 0.598 |

分组结果：box004 为 `1/2` numeric pass，box021 为 `1/16`，首批 bucket004 为 `0/2`。v1 为 `2/17`，v2 rescue 为 `0/3`；v2 三条本来就是 v1 infeasible 的困难样本，不能据此做 v1/v2 公平优劣结论。

## 方法学审计

E167 legacy zgate 把 NPZ `(control_tick, 2 sim_steps, nq)` 中第二个 simulation substep 当作 reference，因此测到的只是同一 control tick 内两个 sim step 的差异。本批按该 legacy 口径会得到 `20/20` z pass；改用 manifest 固定 kinematic trajectory 后是 `14/20`，暴露出 6 条原先会被隐藏的 z-only 失败。

同时，SUGAR-style 3D peak `<=0.30m` 的诊断为 `0/20` pass，且 `14/20` 属于 z 通过但 3D 失败的 XY-only failure。E168 本身是 z-only contract，所以该 3D 指标不作为 release hard gate，但它解释了“视频看起来差而 z-only 仍可能过”的现象。

## 可视化观察

已对 20 条 full MP4 生成并逐页检查 8-frame `ref|sim` sheets，另放大检查量化最差样本和两条 numeric-pass 样本：

- `028_p1`：后段 sim 上肢/手臂明显偏离 reference，和 EEF 31.41cm、object 23.08cm、3D peak 1.493m 一致。
- `030_p1`：搬运/放置阶段末端与物体相对关系明显漂移，和 EEF mean 42.20cm、lower-body 0.398 一致。
- `033_p1/p2`：多帧可见机器人下肢处于物体上方或与物体空间关系异常，分别对应 lower-body 0.302/0.598；`033_p1` 同时 raw contact 不足。
- `034_p2`：稀疏 sheet 不足以稳定看清逐帧深穿透，但 physics metric 为 0.517，不能因静态帧外观尚可而放行。
- 两条 numeric-pass 与新增两条 bucket sheet 整体动作连贯；但 bucket 两条 lower-body 分别为 0.093/0.193，仍按 hard gate 拒绝。

视觉证据：

```text
workspace/core4d/results/E168/s6_downstream/cem/eval/e167a_aligned_available/visual_review/sheets/
workspace/core4d/results/E168/s6_downstream/cem/eval/e167a_aligned_available/visual_review/pages/
```

## 产物

- 汇总：`workspace/core4d/results/E168/s6_downstream/cem/eval/e167a_aligned_available/summary.md`
- 逐 case：`e168_case_metrics.tsv`
- 指标统计：`e168_metric_summary.tsv`
- 分组统计：`e168_group_summary.tsv`
- 最差排名：`e168_worst_case_rankings.tsv`
- 未就绪：`e168_not_ready.tsv`
- 评测快照：`evaluated_manifest_snapshot.tsv`
- Excel 完整报表：`E168_E167A_aligned_available_case_metrics.xlsx`（9 sheets；完整指标 `20×182`）

Excel 由 `workspace/core4d/scripts/eval/reports/gen_E168_available_metrics_xlsx.py` 生成。首屏“门控总览”使用公式从逐 case 数据动态汇总；LibreOffice 强制重算与全工作簿扫描结果为 `0` formula errors。

## Claims 与下一步

- “已有差视频可由 E167A-aligned metrics 识别”：支持，lower-body、fixed-reference z、penetration、EEF/object tracking 与视觉退化一致。
- “当前已有 case 可批量 release”：不支持，仅 `2/20` numeric pass。
- “E167 legacy zgate 足以审查 E168”：不支持，它会把本批 6 条 fixed-reference z 失败全部漏掉。

production watcher 继续回收剩余 rows。后续每次新增完整产物可重跑同一 wrapper；40/40 后再冻结最终 release 表，并只对 numeric + visual 均通过的 source 进入 partner/RL export。
