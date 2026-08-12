# E197 结果：五类 box Full CEM 的 OmniRetarget 与 PRG 七指标对比

_Core4D Phase 60 · 2026-08-12 · 纯离线统一重评 · 87-case paired analysis_

---

## 📋 结论

- Full CEM 母集完整：`87/87` unique，box001/004/021/023/024=`28/6/28/16/9`
- 统一评测：OmniRetarget `87/87`，PRG `87/87`；四项 physics fraction 均在 `[0,1]`，三项 motion-health 均 finite
- Euler 转换审计：最大 world-pose orientation round-trip=`2.96e-6°`，position=`1.45e-13 cm`
- PRG 相对 OmniRetarget 的 object-balanced macro：3mm in-mask 接触 `+31.7pp`，raw in-mask 接触 `-18.1pp`，手物 >3mm 穿透 improvement `+33.0pp`，lower-body 穿透 improvement `+2.7pp`
- 结论是明确 trade-off：PRG 提高“干净的 3mm 接触”并降低手物穿透，但 raw 接触在五个物体均下降；lower-body 仅部分物体改善

## 🔬 方法

使用 `eval.core.core_metrics.evaluate_sequence`，同一 case 共用 PRG scene_act、OmniRetarget trajectory、3cm raw contact mask、person index 与帧域。OmniRetarget freejoint object pose 转换的 Euler convention 从 compiled object hinge axes 推导；未使用默认 `XYZ` fallback。主 delta 为 `PRG − OmniRetarget`；穿透 improvement 取 delta 相反数。

## 📊 关键汇总

| 聚合 | 指标 | OmniRetarget | PRG | Improvement | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Object-balanced macro | 3mm in-mask 接触 | 11.2% | 42.9% | +31.7pp | [+27.0, +36.4]pp |
| Object-balanced macro | Raw in-mask 接触 | 91.8% | 73.7% | −18.1pp | [−22.3, −13.7]pp |
| Object-balanced macro | 手物 >3mm 穿透 | 54.2% | 21.2% | +33.0pp | [+28.8, +37.4]pp |
| Object-balanced macro | Lower-body 穿透 | 9.1% | 6.4% | +2.7pp | [+0.6, +4.8]pp |

## ⚠️ 限制

本轮是单 seed 的离线重评，bootstrap 只对 case 维度重采样；不改变已有 numeric gate、人工 USE/DNU 或 RL export 决策，也不能把 OmniRetarget reference 与 PRG rollout 的差异解释为 P/R/G 单组件因果效应。

## 🔗 产物

- [Markdown 报告](../analysis/E197_full_cem_omnirt_vs_prg_metrics/E197_full_cem_omnirt_vs_prg_metrics.md)
- [XLSX 工作簿](../analysis/E197_full_cem_omnirt_vs_prg_metrics/E197_full_cem_omnirt_vs_prg_metrics.xlsx)
- `e197_case_comparison.tsv`
- `e197_summary_by_object.tsv`
- `e197_method_metrics.tsv`
- `e197_input_audit.tsv`

## 🎯 下游 RL 宽口径预筛

追加三项 motion-health 指标：`foot_slip_max_m`、`obj_speed_max`、`ankle_jerk_p95`。
当前宽口径版本为 `E197-omni-absolute-wide-v4`，**只评估 OmniRetarget**：
3mm in-mask 接触默认 `≥0.01`，box024 特例为 `≥0.0`；raw in-mask 接触 `≥0.50`、手物穿透 `≤0.80`、
lower-body 穿透 `≤0.30`、foot slip `≤1.90 m`、ankle jerk P95 `≤4000 m/s³`。
`obj_speed_max` 继续报告和可视化，但没有用户指定阈值，因此不参与 gate；PRG
不参与 gate 判定，仅保留在 paired comparison 中。

OmniRetarget 通过 `52/87`，按 box001/004/021/023/024 分布为 `9/3/22/12/6`。
这是数值候选预筛，不是 `RL_EXPORT_READY`；人工 USE、partner、alignment、
loader smoke 等既有 gate 仍然必须通过。通过行的 `omnirt_failure_modes` 均为空。

报告与 XLSX 的 delta 均定义为 `PRG − OmniRetarget`。四项 physics fraction 使用
百分比/pp；`foot_slip_max_m`、`obj_speed_max`、`ankle_jerk_p95` 分别使用
m、m/s、m/s³，不按百分比展示。

## ✅ 验证

评测脚本 `py_compile` 通过；LibreOffice 重算工作簿 `1,610` 个公式，`0` 个错误；
逐 case pass/failure-mode 抽查与 decision count 一致；最终 XLSX SHA256 已写入
`e197_summary.json`；
`git diff --check` 通过。
