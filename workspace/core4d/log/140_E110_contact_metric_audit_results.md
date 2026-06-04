# E110 Contact Metric Audit 结果

计划：`workspace/core4d/plan/119_E110_contact_metric_audit_plan.md`

总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

## 目标

E110 不重跑 CEM/RL，只对 E109 24-case work set 的统一 replay 指标做接触诊断聚合。目标是把“Spider 接触低于 OmniRetarget”拆成 aggregate SDF band 和 per-case delta，判断主要是：

- 漏接触；
- 安全浅间隙；
- 深穿透移除但接触未恢复；
- 或少数 object/case outlier。

## 结果路径

| 产物 | 路径 |
|---|---|
| eval 脚本 | `workspace/core4d/scripts/eval/eval_E110_contact_metric_audit.sh` |
| evaluator | `workspace/core4d/scripts/eval_omni_vs_spider/contact_metric_audit.py` |
| 输出目录 | `workspace/core4d/results/E110/contact_metric_audit/` |
| band metrics | `workspace/core4d/results/E110/contact_metric_audit/contact_band_metrics.tsv` |
| case delta | `workspace/core4d/results/E110/contact_metric_audit/contact_case_delta.tsv` |
| method summary | `workspace/core4d/results/E110/contact_metric_audit/contact_method_summary.tsv` |
| object summary | `workspace/core4d/results/E110/contact_metric_audit/contact_object_summary.tsv` |
| summary md/json | `workspace/core4d/results/E110/contact_metric_audit/contact_metric_audit_summary.{md,json}` |

## 执行

固定命令：

```bash
bash workspace/core4d/scripts/eval/eval_E110_contact_metric_audit.sh
```

执行结果：

| 项目 | 数值 |
|---|---:|
| method rows | 48 |
| paired cases | 24 |
| raw-contact coverage | missing_artifacts |
| safety regression rows | 8 |
| exit code | 0 |

静态验证：

```bash
python3 -m py_compile workspace/core4d/scripts/eval_omni_vs_spider/contact_metric_audit.py
bash -n workspace/core4d/scripts/eval/eval_E110_contact_metric_audit.sh
```

两项均通过。

## 关键指标

方法均值：

| method | physics contact | eef5 | hand5 | hand12 | hand pen | hand deep | shallow pen | far>5 | far>12 | leg pen | pelvis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniRetarget | 0.544317 | 0.468198 | 0.646299 | 0.670979 | 0.546984 | 0.293993 | 0.252991 | 0.353701 | 0.329021 | 0.113383 | 0.694415 |
| Spider CEM | 0.430861 | 0.350827 | 0.618059 | 0.656726 | 0.435194 | 0.030734 | 0.404461 | 0.381941 | 0.343274 | 0.100266 | 0.678494 |

解释：

- Spider 的 deep penetration 从 `29.4%` 降到 `3.1%`；
- physics contact 从 `54.4%` 降到 `43.1%`；
- hand12 只从 `67.1%` 降到 `65.7%`；
- shallow penetration 从 `25.3%` 升到 `40.4%`。
- far>5 从 `35.4%` 升到 `38.2%`，far>12 从 `32.9%` 升到 `34.3%`。

这支持 E109 后验判断：Spider 主要把 OmniRetarget 的深穿透式接触转成浅穿透/近场接触，但没有稳定补回无穿透 physics contact。

## Failure labels

| label | count |
|---|---:|
| `penetration_removed_contact_not_recovered` | 18 |
| `safe_gap_near_contact` | 2 |
| `contact_preserved_or_improved` | 3 |
| `neutral` | 1 |

其中 `8` 行另有 `safety_regression_flag=true`，作为二级标记记录，不覆盖主接触 failure label。

Top physics-contact regressions：

| case | object | delta physics | delta eef5 | delta deep pen | label |
|---|---|---:|---:|---:|---|
| `e091_box026_20231020_141_p2` | box026 | -0.369048 | -0.297619 | -0.095238 | `safe_gap_near_contact` |
| `d003_box021_20231018_029_p2` | box021 | -0.253333 | -0.093333 | -0.386667 | `safe_gap_near_contact` |
| `e091_box026_20231020_138_p2` | box026 | -0.238636 | -0.318182 | -0.147727 | `penetration_removed_contact_not_recovered` |
| `e091_box026_20231023_139_p2` | box026 | -0.234043 | -0.198582 | -0.205674 | `penetration_removed_contact_not_recovered` |
| `e091_box026_20231023_137_p1` | box026 | -0.222222 | 0.033333 | -0.055556 | `penetration_removed_contact_not_recovered` |

Object summary 要点：

- box004、box021、box023、box026 均出现 physics contact 下降；
- bucket004 是例外，2 case 平均 physics contact 反而上升；
- box026 是最主要的回归来源，15 case 中大量为 `penetration_removed_contact_not_recovered`。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1: 接触下降可拆为 SDF band | 通过：输出 penetration/deep/shallow/near/far bands |
| C2: Spider 主要是 deep penetration 下降、接触未完全恢复 | 通过：deep -26.3pp，physics contact -11.3pp，hand12 仅 -1.4pp |
| C3: physics contact 下降是否集中可量化 | 通过：per-object summary 和 top regression 输出；box026 是主贡献 |
| C4: raw-contact PR/F1 不伪造 | 通过：raw-contact coverage 记录为 `missing_artifacts`，precision/recall/F1 留空 |
| C5: 可作为 E111 前置原型 | 通过：脚本按 case/method rows 输入，输出 TSV/JSON/MD，后续可接 S1 raw artifacts |
| C6: 审阅发现的 schema/标签问题已修正 | 通过：保留 `near_0_2cm/near_2_5cm` 空列并说明 E109 缺 2cm 阈值；新增 `far>5`、`safety_regression_flag` 和机器可读 top regressions |

## 可视化

E110 不生成新 trajectory 或视频；它只对 E109 已有 replay 指标做 aggregate audit。因此本轮没有新的 viewer/MP4 可视化产物。后续 E111/S6 contact alignment evaluator 若输出 per-frame 曲线或视频帧，应按可视化规则补 visual observation。

## 结论

E110 支持当前主假设：

> 当前 Spider ref-FK CEM/safety stack 主要解决了深穿透，但没有把接触推到“无穿透贴合”。更准确地说，质量从深穿透转向浅穿透/稍远间隙，物理接触和 EEF 近场没有同步恢复；box026 贡献最大，bucket004 是例外。

下一步应进入 E111：补 data_construction_v3 的 S1/S3/S5/S6 contact artifacts 和 per-frame contact alignment evaluator。没有 raw-contact PR/F1 和 run-length 前，不应直接进入 contact reward sweep 或 full CEM 扩量。
