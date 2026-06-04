# E110 Plan: contact metric audit for Spider vs OmniRetarget

日期：2026-06-02

关联总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

## 0. Context

E109 统一 replay 评测显示：Spider CEM 相比 OmniRetarget 大幅降低 `hand_geom_deep_penetration_2cm`，但 `eef_near_3/5cm` 与 `hand_object_physics_contact` 下降；同时 `hand_geom_near_12/15cm` 差距很小。当前需要在不重跑 CEM/RL 的前提下解释接触下降来自：

- 漏接触；
- 安全浅间隙；
- 仅把 OmniRetarget 的穿透式接触移除但未补回无穿透贴合；
- 或少数 object/case outlier。

E110 是评测/诊断实验，只消费 E109 已有 qpos/scene 和 unified replay outputs，不启动 full CEM，不使用远程 GPU。

## 1. Claims

| Claim | 验证方式 |
|---|---|
| C1: E109 的接触下降可以被分解为 SDF band，而不是只看单个 contact fraction | 对 24-case method metrics 输出 `deep_pen / shallow_pen / near_0_2cm / near_2_5cm / far_gt5cm` |
| C2: Spider 的主要变化应表现为 deep penetration 大幅下降，同时物理接触/近场接触是否被恢复要分开判断 | 方法汇总和 Spider-Omni delta 表验证 |
| C3: physics contact 下降是否集中在少数 object/case 可量化 | 输出 per-object summary 和 top regressions |
| C4: raw-contact PR/F1 目前覆盖不足应被显式记录，不伪造指标 | `raw_contact_coverage_status` 必须在 summary 中说明 `missing_artifacts` 或具体覆盖数 |
| C5: E110 可复用为 E111 S6 contact alignment evaluator 的前置原型 | 脚本按 case/method rows 输入，输出 TSV/JSON/MD，后续可接 raw mask |

## 2. Implementation

新增：

```text
workspace/core4d/scripts/eval_omni_vs_spider/contact_metric_audit.py
workspace/core4d/scripts/eval/eval_E110_contact_metric_audit.sh
```

默认输入：

```text
workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_method_metrics.tsv
workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_case_comparison.tsv
```

默认输出：

```text
workspace/core4d/results/E110/contact_metric_audit/
  contact_band_metrics.tsv
  contact_case_delta.tsv
  contact_object_summary.tsv
  contact_method_summary.tsv
  contact_metric_audit_summary.json
  contact_metric_audit_summary.md
```

实现口径：

- 第一版使用 E109 unified replay 已有 aggregate fractions 推导 SDF band：
  - `deep_pen = hand_geom_deep_penetration_2cm`
  - `shallow_pen = hand_geom_penetration - deep_pen`
  - `near_0_2cm = hand_geom_near_2cm - hand_geom_penetration` 如果无 2cm 列则留空；
  - `near_0_3cm = hand_geom_near_3cm - hand_geom_penetration`
  - `near_3_5cm = hand_geom_near_5cm - hand_geom_near_3cm`
  - `near_5_12cm = hand_geom_near_12cm - hand_geom_near_5cm`
  - `far_gt5cm = 1 - hand_geom_near_5cm`
  - `far_gt12cm = 1 - hand_geom_near_12cm`
- 对 physics contact、EEF near、hand geom near/penetration、leg/body safety 做 Spider-Omni delta。
- 输出 failure label：
  - `penetration_removed_contact_not_recovered`
  - `safe_gap_near_contact`
  - `contact_regression_with_safety_regression`
  - `contact_preserved_or_improved`
  - `neutral`
- 同时输出 `safety_regression_flag` / `secondary_flags`，避免主 failure label 的单标签顺序掩盖 lower-body/pelvis 退化。
- raw-contact PR/F1 第一版只记录 coverage；不在缺 raw artifacts 时填假 precision/recall。

## 3. Eval command

固化本地评估脚本：

```bash
bash workspace/core4d/scripts/eval/eval_E110_contact_metric_audit.sh
```

该脚本只运行 CPU/Python 评测，不占 GPU，不触发远程多卡执行。

## 4. Success criteria

| 标准 | 阈值 |
|---|---|
| 运行成功 | eval 脚本 exit 0 |
| 覆盖 24-case work set | method rows = 48，case deltas = 24 |
| 输出完整 | 6 个核心输出文件均存在且非空 |
| 诊断有效 | summary 中列出 method/object/case 三层结论和 top regressions |
| 不伪造 raw metrics | raw-contact PR/F1 在缺 artifacts 时为空，并记录 coverage warning |

## 5. 本轮不做

- 不重跑 OmniRetarget/SPIDER CEM。
- 不修改 `data_construction_v3` S1/S3/S5/S6 代码；这些进入 E111。
- 不把 aggregate band 误称为逐帧 run-length；run-length 等逐帧指标留给 E111/S6 evaluator。
- 不把 OmniRetarget 高接触率解释为好接触，必须同时报告 penetration。
