# Spider vs OmniRetarget fair-eval scripts

本目录用于 E109 及后续 Spider vs OmniRetarget 公平评测。

设计原则：

- 不把任一方法输出当 GT。
- 历史 `qpos_ref` / self-ref 指标只作为诊断列，不作为公平胜负。
- 依赖阈值的指标必须显式写出阈值；当前 replay 阈值为 `3cm/5cm/8cm/10cm/12cm/15cm`。
- `results/` 不提交 git；脚本和文档提交 git。

## 当前推荐输出

### 20-case xlsx-only 汇报表

最新 PPT/汇报优先使用这个表：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/build_filtered_20_work_xlsx.py
```

口径：

```text
24-case work 表
- bucket004_20231003_1_012_p1
- e091_box026_20231020_134_p1
- e091_box026_20231020_141_p2
- e091_box026_20231023_139_p2
= 20-case filtered work 表
```

注意：用户原始删除列表里 `e091_box026_20231020_134_p1` 出现两次；脚本会在 `排除说明` sheet 里记录去重和实际删除情况。

输出是 xlsx-only：

```text
workspace/core4d/results/E109/filtered_20_work_cases_xlsx/
  filtered_20_omni_vs_spider_work_eval.xlsx
```

主要 sheet：

- `20case逐case对比`：OmniRetarget vs Spider 逐 case 指标；Spider 指标单元格有红/绿标记。
- `PPT方法汇总20`：半页 PPT 用窄版汇总。
- `方法汇总20`：完整方法均值。
- `颜色规则`：红/绿判定阈值。
- `排除说明`：删除 case 记录。
- `完整method_metrics`：完整 replay 指标。
- `历史对齐校验`：Spider 历史 summary 对齐结果。

### 24-case work 扩展表

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/build_expanded_24_work_eval.py
```

这个入口不覆盖 11-case strict 主表。它单独输出：

```text
11 条 ref_fk strict 主表 case
13 条去重后的 ref_fk upper-WORK/non-strict case
```

用途：汇报“更多 Spider CEM work 候选”和失败模式背景。13 条 non-strict case 不能直接当 RL-ready positive。

输出：

```text
workspace/core4d/results/E109/expanded_24_work_cases/
  expanded_24_omni_vs_spider_work_eval.xlsx
  expanded_24_omni_vs_spider_work_eval.md
  expanded_24_case_comparison.tsv
  expanded_24_method_metrics.tsv
  expanded_24_history_validation.tsv
  run_summary.json
```

### 11-case strict 主表

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/unified_replay_eval.py
```

这是最严格的主评测入口。它只纳入 `workspace/core4d/data_construction_v3/existing_cases.tsv` 中 `cem_status=pass` 且 `target_variant_id != adaptive` 的 case，把 OmniRetarget 和 Spider CEM qpos 都重新做 MuJoCo replay，重算 hand/body/leg/object 指标，并把 Spider 新指标和历史 summary 中已有指标逐项对齐校验。

输出：

```text
workspace/core4d/results/E109/unified_replay_eval/
  unified_omni_vs_spider_comparison.md
  unified_case_comparison_full.md
  unified_method_metrics_full.md
  history_validation_full.md
  unified_omni_vs_spider_comparison.xlsx
  unified_case_comparison.tsv
  unified_method_metrics.tsv
  history_validation.tsv
  unified_method_summary.tsv
```

## 手-物指标口径

PPT sheet 里的 `手5cm/8cm/10cm/12cm/15cm` 是 `eef_near_*`：

```text
left/right_wrist_yaw_link + wrist局部[0.05, 0, 0] 的 EEF proxy
到物体 collision SDF 的距离 < 阈值
左右手分别算比例，最终取 max(left_frac, right_frac)
```

它不是 fingertip。这个口径是为了和历史 `contact_frac_either` 对齐；历史字段名像 either/union，但代码复核后实际是 `max(L,R)`。

逐 case 表和 method metrics 还包含更几何/物理的手指标：

- `hand_geom_near_*`：机器人 hand geom `lh/rh` 到物体 SDF 的近接触比例。
- `hand_geom_penetration_frac`：hand geom SDF < 0 的比例。
- `hand_geom_deep_penetration_2cm_frac`：hand geom SDF < -2cm 的比例。
- `hand_object_physics_contact_frac`：MuJoCo contact pair 中 hand geom 与 object geom 发生真实物理 contact 的比例。

PPT 汇总里当前额外放：

```text
手geom12cm↑
手物理接触↑
```

## 颜色规则

`20case逐case对比` 和 `24case逐case对比` 只给 Spider 对应指标单元格上色：

- 绿色：Spider 明显好于 OmniRetarget。
- 红色：Spider 明显差于 OmniRetarget。

默认“明显”阈值：

| 指标族 | 方向 | 阈值 |
|---|---|---:|
| pelvis 高度 | 越高越好 | 3cm |
| 手近接触比例 | 越高越好 | 5 个百分点 |
| hand/body/leg 穿透比例 | 越低越好 | 2 个百分点 |
| hand-object physics contact | 越高越好 | 5 个百分点 |
| 物体 XY 位移 | 越高越好 | 5cm |
| fall | False 优于 True | 布尔差异 |

## 历史/调试入口

完整 OmniRetarget vs Spider 指标对比表入口：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/build_existing_cases_comparison.py
```

输出：

```text
workspace/core4d/results/E109/omni_vs_spider_existing_cases/
  omni_vs_spider_comparison.md
  omni_vs_spider_case_metrics_full.md
  omni_vs_spider_comparison.xlsx
  omni_vs_spider_case_metrics.tsv
  omni_vs_spider_method_summary.tsv
  metric_definitions.tsv
  coverage_warnings.tsv
```

E109 历史复现实验入口：

```bash
bash workspace/core4d/scripts/eval_omni_vs_spider/run_E109_fair_eval.sh
```

核心输出：

- `case_bank.tsv`：统一 case/method/evidence 索引。
- `method_case_metrics.tsv`：逐 case 指标；`fair_metric_scope=diagnostic_only` 的行只作背景。
- `method_summary.tsv` / `object_summary.tsv` / `threshold_summary.tsv`：方法、物体、阈值聚合。
- `warnings.tsv`：self-ref、method-ref、缺少 raw-GT 等 caveat。
