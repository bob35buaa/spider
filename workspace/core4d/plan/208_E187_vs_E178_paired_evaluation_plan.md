# E187 vs E178 22-case Paired Evaluation 计划

## Context

- E187 已按用户授权在 C9 technical `FAIL`、progression authority `USER_WAIVED` 的前提下完成 Full CEM，22/22 manifests、finite、视频与 SHA 检查均 PASS。
- E187 keep22 是 E178 Full authority 的精确 `case_id` 子集；本阶段只做同 case 的 paired evaluation，不改变 E178 历史结果，不回调 E187 reward/grid/P/G/Full artifact。
- E178 的 `eval_E178_lowgeom.py`、公共 `eval_E176_lowgeom.py` 与 `gen_E178_bucket_prg_xlsx.py` 提供冻结的指标和工作簿口径；E187 必须直接复用公共 evaluator 逻辑，不能以 dynamic import 绕过合同。
- E178 tracking gates 保持：root position/orientation、hand position/orientation、object position均不超过20，object orientation不超过10（位置单位cm、角度单位deg）。

## Claims

| Claim | 验证方式 | 判定标准 |
|---|---|---|
| E1 配对完整性 | E187 eval manifest 与 E178 metrics 按 `case_id` 连接 | 22/22 一一配对；missing=0；duplicate=0；unexpected=0 |
| E2 指标可比性 | 两侧使用同一公共 low-geometry evaluator 与 gate 定义 | metric standard、gate阈值和计算字段一致；不得改写 E178 artifact |
| E3 结果闭合 | 生成 E187 case metrics、group summary、paired delta、summary JSON | evaluator 22/22 完成；所有关键数值 finite；每行保留 artifact provenance |
| E4 Excel 可审计 | 生成 E187 vs E178 paired workbook | 22行配对；E178/E187/Delta、gate transitions、object summary、best/worst、provenance齐全 |
| E5 工作簿可靠 | LibreOffice 重算并扫描 | 公式错误0；Arial；冻结表头；筛选；单位/方向说明明确 |
| E6 治理状态不漂移 | summary、log、tracker 保留 waiver 语义 | C9 technical=`FAIL`；progression authority=`USER_WAIVED` |

## 改动

1. 在 `scripts/experiments/E187/` 增加只读构建器，将 E187 Full 22-row authority 映射到 evaluator manifest，并锁定 E178 baseline 的22个 `case_id`。
2. 在 `scripts/eval/runners/` 增加 E187 paired evaluator，直接调用公共 low-geometry evaluator实现，输出 E187 metrics、group summary、E178 paired baseline、paired deltas 与 summary JSON。
3. 在 `scripts/eval/wrappers/` 固化 canonical evaluation 命令。
4. 在 `scripts/eval/reports/` 增加 Excel generator，输出 Overview、Paired Comparison、E187 Metrics、E178 Baseline、Object Summary、Gate Transitions、Worst Regressions、Best Improvements、Artifact Provenance。
5. 新建结果 log，并更新 `log/INDEX.md`、`EXPERIMENT_TRACKER.md` 与 `progress.md`。

## 成功标准

- E187 Evaluation manifest 22行，且 case set 与 E178 keep22 baseline 完全相等。
- E187 evaluator 22/22 成功，无缺失、重复、非有限关键指标或 artifact provenance 缺口。
- paired delta 的方向明确：对“越低越好”指标使用 `E178-E187` 作为 improvement；对“越高越好”指标使用 `E187-E178`。
- Excel 中计算列使用公式；经 `recalc.py` 后 `total_errors=0`。
- 评测结果只陈述配对证据，不把 C9 waiver 改写为 technical PASS。

## Canonical commands

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E187_vs_E178_full.sh preflight
bash workspace/core4d/scripts/eval/wrappers/eval_E187_vs_E178_full.sh run
python workspace/core4d/scripts/eval/reports/gen_E187_vs_E178_xlsx.py
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py workspace/core4d/results/E187/s6_downstream/eval/full/E187_vs_E178_paired_evaluation.xlsx 60
```

## 禁止项

- 不修改 E178 metrics、manifest、workbook 或任何历史 log。
- 不修改 E187 `log/256`、`log/257`，不覆盖 Full manifests/results/config/videos。
- 不因 Evaluation 结果回选 reward/grid/P/G/Full case set。
- Claims 未逐项闭合前不宣称 E187 整体完成。
