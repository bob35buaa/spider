# E001-E025 协作重定向阶段汇报摘要

## Summary

- 核心问题：E081 的好结果是否来自真实 robot-object 物理搬运，还是依赖 object actuator / support proxy；若转成 true-freejoint，哪种 partner-side coupling 有效。
- 主要结论：纯 robot reward、freejoint mass/friction、force/spring/contact-pad 路线都不能稳定搬运；关键正结果来自 **E014/E018/E018b: canonical support proxy + soft equality**。
- 最强结果：E018b 在 13 case 上 object/transport gate `13/13`，mean Epos `5.45cm`、Erot `5.22deg`、carry progress `1.004`；但 strict full-retargeting 仅 `1/13`，瓶颈转到 robot-side stability/contact/penetration。
- 评测亮点：E019 建成 paper-aligned unified eval；N=12 跨方法对比中 spider physical smoothness `13428` vs Holosoma kinematic `36048 rad/s^2`，降低 `62.7%`。
- 后续决策：不要继续扫 anchor / soft reward；保留 object-side support proxy，转向 hard no-penetration、contact timing、lower-body geometry/control。

## 1. Experiment Motivation

本阶段目标是把 `workspace/core4d` E081 的 actuator-guided object 结果，推进到更可信的 collaborative retargeting：object 尽量 true-freejoint，partner-side support 由可解释机制提供，同时保持 robot contact、稳定性和无穿透。

## 2. Experiment Setup

- 工作区：`workspace/core4d_collab_retarget`
- 覆盖实验：E001-E025；完全失败的参数 sweep 不单独展开，只保留其排除结论。
- 主要数据：CORE4D box/bucket/desk 交互 case；E016/E018b 扩到 13 case。
- 代码版本：`0dc1e17`
- 主要证据：`log/01-25*.md`、`results/E*/comparison.csv`、`results/E*/aggregate_summary.json`

## 3. Core Algorithm or Method

- E001-E012：诊断 true-freejoint object 下，单 robot reward、mass/friction、virtual force、contact pad 是否足够。结论是不足。
- E013：object oracle 上界，证明 true-freejoint eval 下 object target 本身可达。
- E014/E018/E018b：引入 object-local support anchor + soft equality / canonical support proxy，提供 partner-side coupling。
- E019-E025：统一评测、13-case 归因、mask/geometry/stability/contact-penetration 局部修复。

## 4. Metrics

| 指标 | 方向 | 含义 |
|---|---:|---|
| Object Epos / case-window obj err | ↓ | object trajectory tracking error |
| Object Erot | ↓ | object orientation error |
| Carry progress ratio | →1 | 是否完成参考平移 |
| 5cm contact preservation | ↑ | robot hand/object 接触保持 |
| Deep penetration duration | ↓ | robot-object 穿透 artifact |
| Pelvis min z / fall gate | ↑ | robot 稳定性 |
| Strict generalization pass | ↑ | object、contact、penetration、stability 同时过线 |
| Smoothness | ↓ | motion jerk / acceleration proxy |

## 5. Results

| 实验 | 关键结果 | 结论 |
|---|---|---|
| E002-E012 | true-freejoint robot-only best 仍在 `0.30-0.70m` obj err 区间；E011 best `0.340/0.673m`，E012 有 rotation shortcut | force/spring/contact-pad 不足，需结构性 partner coupling |
| E013 oracle | main obj `0.011/0.038m`，guard `0.017/0.064m` | object target 可达，瓶颈不是 freejoint eval 本身 |
| E014 soft equality | 6/6 soft target pass；main obj `0.056-0.082m`，hand `85.5-90.8%`，leg `0%` | 第一条真正有效的 object-side coupling |
| E016 13-case quick | object/transport `13/13`，mean Epos `0.050m`；contact ok `3/13`，strict `0/13` | object-side 泛化成立，完整成功不足 |
| E018 GT gate | 2/2 GT anchor pass；mean Epos `0.049m`、Erot `2.10deg` | canonical anchor 复现手工 anchor |
| E018b 13-case full | object/transport `13/13`；strict `1/13`；fall `4/13`；contact ok `5/13` | anchor/object-side 成立，robot-side 是主瓶颈 |
| E019 unified eval | SPIDER T4 + OmniRetarget penetration + Tab.5 N=12；smoothness `13428` vs `36048` | 形成可写论文/组会的跨方法指标亮点 |
| E020 attribution | root cause: stability 4、contact 4、retarget geometry 2、mask 1、raw data 1、pass 1 | 失败被拆成可执行分线 |
| E021 RL export | 13/13 base conversion ok，13/13 load verification ok，12 个 partner 文件 | 可作为 Holosoma RL 数据输入 |
| E022-E025 | mask overclaim `54.41% -> 0-0.44%`；bucket001 p2 pelvis `0.439 -> 0.726m`；bucket007 deep pen `51.01% -> 35.57%` | 没有 strict success，但定位了后续机制方向 |

## 6. How to Read the Tables

总表见 `E001_E025_key_metrics.xlsx`。`variant_metrics` 是可量化 variant 级指标；`stage_summary` 是适合 PPT 的阶段摘要；`local_deltas` 只列 E022-E025 的局部正向变化。

## 7. Interpretation

阶段性结论很明确：**object-side support proxy 已经解决到可泛化 object tracking 的程度，但完整协作重定向仍被 robot-side 接触、稳定、穿透 artifact 限制**。E014/E018b 是主线成果；E022-E025 不是成功算法，但把失败原因从“泛化失败”拆成了具体机制缺口。

## 8. Conclusion and Discussion

组会主线建议：从“E081 依赖 object guidance”讲起，展示 E014/E018b 如何把 object transport 做到 13/13，再强调 strict pass 低不是 anchor 问题，而是 robot-side physics/contact 问题。卖点是 E019 的统一评测和 smoothness 优势；下一阶段不做 reward sweep，做 hard constraints / timing / geometry。

## 9. Limitations and Caveats

- 多数实验是单 seed / 单组 CEM，不适合做统计显著性声明。
- E014/E018b 的 support proxy 是 kinematic/soft equality，不等同于完整 dynamic partner。
- strict pass 低，说明当前不能宣称 full collaborative retargeting 已解决。
- E021 只做 load-level RL export 验证，未验证 downstream RL 训练收益。

## 10. Next Steps

1. Box023 contact timing diagnosis：验证 contact mask / EEF target / support proxy 是否错相。
2. Hard no-penetration：SDF barrier、sample rejection、surface projection，替代 soft penalty。
3. Lower-body geometry/control：真实 collision geometry + lower-body regularization。
4. Bucket001 p1 stability：upright/root terminal、foot support、staged contact objective。

## Reproducibility Notes

- Tracker: `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md`
- Main logs: `log/13_E013_true_freejoint_object_oracle_results.md`, `log/14_E014_cola_b_kinematic_weld_results.md`, `log/19_E018b_canonical_support_proxy_13case_results.md`, `log/20a_E019_unified_eval_framework_results.md`, `log/20_E020_failure_attribution_audit_results.md`, `log/25_E022_E025_optimization_stage_summary.md`
- Metrics: `results/E013|E014|E016|E018|E018b|E022|E023|E024|E025/{comparison.csv,aggregate_summary.json}`
