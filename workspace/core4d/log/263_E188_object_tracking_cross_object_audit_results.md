# E188 Analysis Report：Bucket 与历史 Box 的 Object Tracking 分层审计

_CORE4D Phase 51 · 2026-08-05 · batch-summary / analysis-only_

## Summary

- **观察得到确认但需拆开表述**：E187 bucket 的 object position 确实存在肉眼可见量级的偏差，22条全量均值为 `10.54 cm`，其中 `11/22 >10 cm`；但它与 E178 全量 `10.56 cm` 基本相同，也优于 canonical box 全量 `13.61 cm`。
- **E188 是明确的系统性退化**：同一15条 case 上，E187→E188 position `9.82→11.43 cm`（`+1.606 cm`），orientation `5.80→7.61°`（`+1.805°`），两项 bootstrap 95% CI 均不跨0，且两项均为 `0/15` 非退化，即15条逐 case 全部变差。
- **设备变化不能解释该现象**：same-device local-4 也恶化 `+1.300 cm/+2.103°`；cross-device11 为 `+1.717 cm/+1.697°`，两个分层方向一致。
- **人工通过不等于紧 object tracking**：E187 人工 USE 14条均值仍为 `9.61 cm/5.94°`；canonical box 最终人工 USE 45条为 `12.35 cm/6.38°`。人工审核在判断整体可用性，但没有形成严格 object-pose gate。
- **决策**：保留 E188 “5kg 不升级”的结论，并将 object tracking 系统性退化列为新的高优先级诊断项；下一步应做逐帧 reference/sim object pose overlay，区分恒定 frame/mesh-origin 偏置与动力学漂移。

## 1. Experiment Motivation

E188 的配对视频和数值结果显示 object tracking 较 E187 变差；同时 E187 bucket 在视觉上已经存在物体位置相对参考序列的偏差。本审计回答三个决策问题：

1. E187 的偏差是相对 E178 新出现，还是历史上普遍存在？
2. E188 的恶化是否只是 case composition 或 GPU 设备变化造成？
3. numeric PASS 与人工 USE 是否有效过滤 object tracking 较差的 case？

## 2. Experiment Setup

本阶段不重跑 CEM，只聚合已有正式 `core4d-e154-physics-contact-v1` metrics。

### Bucket authority

| Dataset | Cases | Objects | Manual authority |
|---|---:|---|---|
| E178 | 27 | bucket003/004/007 | filled review，USE=12 |
| E187 | 22 | bucket003/004/007 | final filled review，USE=14 |
| E188 | 15 | bucket003/007 | 无独立人工审核；另报 E187-USE matched 10条 |

### Canonical box authority

为避免 ablation 重复计数，每个 box 只采用最终 production experiment：E170 box021（28）、E171 box026（12）、E172 box004（6）、E173 box001/023/024（53），合计99条。box022 为 `DATA_NEGATIVE`，无正式 Full metrics，不进入均值。

人工层使用最终 authority：E170 metrics 中 human review、E172 filled review、E173 三个最终 per-object RL-export snapshot。最终 box artificial USE 为45条；E171 box026 没有 filled review，因此只进入 all/numeric 层。

## 3. Core Algorithm or Method

Not applicable。本阶段是固定指标的离线分层与配对聚合，不改变 SPIDER、reward、scene、CEM 配置或历史标签。

聚合层：

- `all_cases`：正式 metrics 全量；
- `numeric_pass`：12门 `numeric_release_pass=true`；
- `manual_use_self`：该实验自身最终人工 `USE`；
- `manual_use_parent_matched`：仅 E188，按 case ID 匹配 E187 的人工 USE，不代表 E188 已人工通过。

均值 CI 使用 case bootstrap 10,000次、确定性 seed。paired delta 定义为 `new-old`，正数表示误差变差。

## 4. Metrics

| Metric | Unit | Direction | Current numeric gate |
|---|---|---|---|
| `track_obj_pos_err_cm_mean` | cm | lower is better | `≤20 cm` |
| `track_obj_ori_err_deg_mean` | degree | lower is better | `≤10°` |

`>10 cm` 与 `>5°` 在报告中作为“较容易肉眼感知”的描述阈值，不是新增 release gate。

## 5. Results

### 5.1 Requested stratification

| Dataset | Stratum | n | Position mean | Orientation mean |
|---|---|---:|---:|---:|
| E178 | all | 27 | 10.56 cm | 5.63° |
| E178 | numeric PASS | 10 | 9.04 cm | 4.61° |
| E178 | manual USE | 12 | 8.10 cm | 4.57° |
| E187 | all | 22 | 10.54 cm | 6.38° |
| E187 | numeric PASS | 6 | 10.80 cm | 6.45° |
| E187 | manual USE | 14 | 9.61 cm | 5.94° |
| E188 | all | 15 | 11.43 cm | 7.61° |
| E188 | numeric PASS | 3 | 7.93 cm | 4.99° |
| E188 | E187-manual-USE matched | 10 | 10.63 cm | 6.89° |
| Canonical box | all | 99 | 13.61 cm | 6.26° |
| Canonical box | numeric PASS | 48 | 12.06 cm | 6.08° |
| Canonical box | manual USE | 45 | 12.35 cm | 6.38° |

E188 的3条 numeric PASS 均值较好属于明显的 selection-on-metric：numeric PASS 本身包含 object position/orientation gates，不能据此反推 E188 全量 tracking 改善。

### 5.2 Common-case paired comparison

| Scope | Pair | n | Position old→new | Δ 95% CI | Orientation old→new | Δ 95% CI |
|---|---|---:|---:|---:|---:|---:|
| 22 common | E178→E187 | 22 | 10.66→10.54 (`-0.125`) | [-0.576,+0.275] | 5.77→6.38 (`+0.609`) | [-0.193,+1.355] |
| 15 triple-common | E178→E187 | 15 | 9.81→9.82 (`+0.018`) | [-0.292,+0.349] | 5.63→5.80 (`+0.173`) | [-0.682,+0.857] |
| 15 triple-common | E187→E188 | 15 | 9.82→11.43 (`+1.606`) | [+1.111,+2.276] | 5.80→7.61 (`+1.805`) | [+1.266,+2.408] |

E187→E188 的 position/orientation 非退化数都是 `0/15`。因此不是少数 outlier 拉高均值，而是全部 case 同方向变化。

### 5.3 Device and object decomposition

| Subgroup | n | Position Δ | Orientation Δ |
|---|---:|---:|---:|
| same-device local-4 | 4 | +1.300 cm | +2.103° |
| cross-device11 | 11 | +1.717 cm | +1.697° |
| bucket003 E187→E188 | 2 | +2.158 cm | +3.991° |
| bucket007 E187→E188 | 13 | +1.521 cm | +1.469° |

两个设备分层、两个 object 分层都同方向退化。bucket003 的退化幅度最大，但 bucket007 的13条也一致恶化，说明结论不是由2条 bucket003 单独驱动。

E178→E187 的 per-object 结果则不同：bucket007 position/orientation 基本不变（`-0.005 cm/+0.010°`）；bucket003 position改善 `-0.641 cm`，但 orientation 恶化 `+2.707°`。因此 E187 相对 E178 的主要变化是 bucket003 orientation，而不是全局 position 漂移。

### 5.4 Visible-offset prevalence

| Dataset | Position >10 cm | Position >20 cm | Orientation >5° | Orientation >10° |
|---|---:|---:|---:|---:|
| E178 | 14/27 | 0/27 | 12/27 | 2/27 |
| E187 | 11/22 | 0/22 | 14/22 | 0/22 |
| E188 | 7/15 | 1/15 | 13/15 | 3/15 |
| Canonical box | 77/99 | 10/99 | 71/99 | 7/99 |

E187 没有越过正式 object gate，却有一半 case 超过10 cm、近三分之二超过5°，解释了“视频里看起来有偏差但 numeric object gate 仍 PASS”的现象。

## 6. How to Read the Tables

- experiment-level 表用于回答最终产物集的总体质量；不同 experiment case 数不同，不能单独用于因果结论。
- common-case paired 表固定 case composition，是判断 E178→E187→E188 版本变化的主证据。
- device 分层用于检查 E187 Ada→E188 A100/5090 的混杂；same-device local-4 是质量变化的最强直接证据。
- manual USE 是整体视觉可用性 authority，不是 object-pose 专项审核。

## 7. Interpretation

### Observed

1. E187 position error 约10.5 cm，偏差量级足以肉眼观察到，但相对 E178 没有系统性新增；历史 box 的 position error 反而更高。
2. E188 相对 E187 的 position 和 orientation 在15条 case 上全部恶化，且 same-device/cross-device、bucket003/007 四个分层方向一致。
3. E188 全量 orientation `7.61°` 比 canonical box `6.26°` 更差；position `11.43 cm` 仍好于 box `13.61 cm`。因此不能用一个“object tracking 总体更差”概括两个维度。
4. 人工 USE 虽降低 bucket 均值，但仍允许 E187 `9.61 cm/5.94°`、box `12.35 cm/6.38°` 的偏差，说明现有人工审核不是紧 object alignment gate。

### Inferred

E188 的全 case 同方向变化、同设备仍退化，支持“5kg inertial intervention 改变了 CEM 解并牺牲 object tracking”的解释；但当前证据还不能区分：

- 物体动力学导致的随时间漂移；
- CEM 在接触/下肢/object tracking 之间重新权衡；
- reference/sim object frame、mesh origin 或初始 pose 中已有恒定偏置。

这些机制需要逐帧 pose 曲线和 frame overlay 才能区分，不能仅凭 sequence mean 下结论。

## 8. Conclusion and Discussion

用户的核心观察成立：E187 的物体位置确有明显偏差，而 E188 又在此基础上系统性恶化。但比较结果同时修正了两个可能的过度推断：

- E187 position 并没有比 E178 或历史 box 更差；它是当前评测门对约10 cm偏差较宽松的问题。
- E188 不是只在均值上偶然变差，而是15/15 position、15/15 orientation 都逐 case 退化。

因此 E188 的5kg配置继续不应升级；object tracking 应从“C7均值未超过2 cm/2°所以非劣”升级为独立风险项，因为 aggregate claim 恰好掩盖了逐 case 100%同方向退化。

## 9. Limitations and Caveats

- box 与 bucket 是跨 object、跨 case、跨实验的描述性比较，不是受控因果对照。
- canonical box manual-only 不含无 filled review 的 box026；人工层 object coverage 与 all/numeric 不完全相同。
- E188 没有自身人工 review，E187-USE matched 只表示 case selection transfer。
- bootstrap 单位是 case；E188 仍是单 seed。
- sequence mean 无法区分恒定坐标偏置、短时峰值和随时间累积漂移。
- `>10 cm`/`>5°` 是描述阈值，不应在本审计内 post-hoc 替换正式 release gate。

## 10. Next Steps

1. 为 E178/E187/E188 的15条 common case 输出逐帧 object translation/orientation error 曲线，并分解初始偏置、均值、terminal drift。
2. 选 bucket003 两条和 bucket007 至少三条，渲染 reference/sim object COM、坐标轴和 mesh-origin overlay，判断偏差来自 frame mapping 还是动力学跟踪。
3. 独立人工审核 E188 15条 object alignment；不要直接继承 E187 USE 标签。
4. 在机制明确前，不修改正式20 cm/10°门；若确认视觉可用性要求更紧，另开 threshold sensitivity audit。

## Reproducibility Notes

- Plan：`workspace/core4d/plan/213_E188_object_tracking_cross_object_audit_plan.md`
- Script：`workspace/core4d/scripts/eval/reports/gen_E188_object_tracking_audit.py`
- Results：`workspace/core4d/results/E188/s6_downstream/eval/object_tracking_audit/`
- Command：`.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E188_object_tracking_audit.py`
- Git HEAD at audit：`bb5fc62`
- Metric standard：`core4d-e154-physics-contact-v1`
- `summary.json` SHA256：`662c026826c90307f420a74d04823b1fbbced40d34a261bac892bb728a1b2e6c`
- `stratified_summary.tsv` SHA256：`84493cce375be078331325a7d9d234726b2fb051a763eb6079bbdec6f29e1e12`
- `paired_version_comparison.tsv` SHA256：`b2219f99616eee0dcac7d867ca5f85979257978b099b2522ca4bd1684b37e808`
- `report.md` SHA256：`7c618c98b3c9b07e0c41effe6786802643cf34c06073fb0c38b30b8f6e6c8b0b`

本轮为纯分析且暴露新的高优先级 object tracking 风险，不 commit/push，不自动进入 RL。
