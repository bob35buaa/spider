# E031 results: Best dynamic assembly + full evaluation ledger

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e031-full-eval`

## 目标

E031 按 `plan/31_post_E026_next_stage_optimization_plan.md` 和 `plan/36_E031_best_dynamic_assembly_full_eval_plan.md` 完成 E026-style 账本组装：把 E027 data-quality caveat、E028-E030 diagnostic/negative evidence、E018b/E022-E025 conservative best-positive pool 合并到统一 full-eval 输出中。

E031 不是新 CEM rollout 实验，不触发远程 GPU wrapper；它的作用是防止把 E028-E030 的 negative/guard 结果误选成 best dynamic，并确认 post-E026 阶段是否带来新的 strict gain。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/36_E031_best_dynamic_assembly_full_eval_plan.md` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py` |
| Results root | `workspace/core4d_collab_retarget/results/E031_full_eval/` |
| Coverage | `workspace/core4d_collab_retarget/results/E031_full_eval/coverage.json` |
| Method metrics | `workspace/core4d_collab_retarget/results/E031_full_eval/method_case_metrics.csv` |
| Best selection | `workspace/core4d_collab_retarget/results/E031_full_eval/best_dynamic_selection.csv` |
| Rejected diagnostics | `workspace/core4d_collab_retarget/results/E031_full_eval/rejected_diagnostic_candidates.csv` |
| Data caveats | `workspace/core4d_collab_retarget/results/E031_full_eval/data_quality_caveats.csv`, `workspace/core4d_collab_retarget/results/E031_full_eval/data_quality_caveats.md` |
| Summaries | `workspace/core4d_collab_retarget/results/E031_full_eval/summary_9case.md`, `workspace/core4d_collab_retarget/results/E031_full_eval/summary_13case.md` |
| Visual / metric audit | `workspace/core4d_collab_retarget/results/E031_full_eval/visual_metric_audit.md` |
| Aggregate | `workspace/core4d_collab_retarget/results/E031_full_eval/aggregate_summary.json` |

## 执行命令

```bash
.venv/bin/python -m py_compile \
  workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py

.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py --all
```

执行输出摘要：

| Metric | Value |
|---|---:|
| Normalized rows | `94` |
| Best-selection rows | `13` |
| Rejected diagnostic candidates | `18` |
| Retarget-questionable cases | `2` |
| Success-denominator discards | `1` |

实现中额外修正了 `strict_success` 的 method-specific 归因：E028/E029/E030 只读各自实验字段，避免 E030 从源 E018b 字段继承 stale success；`spider_E081_full_rerun` 显式读取 `E081_success_legobj_strict_proxy`，保持 E026 baseline strict proxy 口径。

## Coverage

| Source | Rows | 用途 |
|---|---:|---|
| Holosoma / OmniRetarget kinematic | `12` | kinematic baseline；`desk021_p1` SOCP infeasible |
| E081 legacy | `2` | 原始 two-case scene-actuator baseline |
| E026 E081 full rerun | `13` | paper-metrics E081 baseline |
| E018b | `13` | conservative best-positive baseline |
| E022 | `4` | best-positive candidate pool |
| E023 | `6` | best-positive candidate pool |
| E024 | `5` | best-positive candidate pool |
| E025 | `8` | best-positive candidate pool |
| E027 quality audit | `13` | P0/P1 caveat and discard protocol |
| E028 | `6` | diagnostic/rejected only |
| E029 | `6` | diagnostic/rejected only |
| E030 | `6` | diagnostic/rejected only |

Known missing is intentional: E027 has no rollout candidates; Holosoma misses `desk021_p1`; E081 legacy only has two original cases, so E026 full rerun is the comparable E081 baseline.

## P0 / P1 汇总

P0 9case：

| Method | Obj Pos | Contact 5cm | Deep Pen | Falls | Strict |
|---|---:|---:|---:|---:|---:|
| OmniRetarget kinematic | `0.00cm` | `-` | `-` | `0` | `0/9` |
| E081 full rerun | `21.27cm` | `42.17%` | `16.25%` | `1` | `4/9` |
| E018b | `5.14cm` | `63.61%` | `35.83%` | `1` | `1/9` |
| E031 conservative best | `4.97cm` | `65.81%` | `30.52%` | `0` | `1/9` |

P1 13case：

| Method | Obj Pos | Contact 5cm | Deep Pen | Falls | Strict |
|---|---:|---:|---:|---:|---:|
| OmniRetarget kinematic | `0.00cm` | `-` | `-` | `0` | `0/12` |
| E081 full rerun | `27.10cm` | `36.23%` | `12.39%` | `4` | `4/13` |
| E018b | `5.45cm` | `54.35%` | `30.04%` | `4` | `1/13` |
| E031 conservative best | `5.33cm` | `55.87%` | `26.37%` | `3` | `1/13` |

结论：E031 账本没有新增 strict gain。E018b/E022-E025 的 conservative best 仍保持 E026 数字：P0 strict `1/9`，P1 strict `1/13`。

## Selection / rejection audit

Best-positive pool 固定为：

- `spider_E018b`
- `spider_E022`
- `spider_E023`
- `spider_E024`
- `spider_E025`

E028/E029/E030 全部作为 diagnostic/rejected evidence，不参与 `best_dynamic_selection.csv`。拒绝表共 `18` rows：

| Source | Rows | 主要拒绝原因 |
|---|---:|---|
| E028 | `6` | hard barrier 导致 contact collapse、fall 或 high-contact high-penetration |
| E029 | `6` | `bucket001_p1` stability-only contact `0%`；`bucket001_p2` high penetration；`box025_p2` 仅 guard/non-regression，不替换 E018b strict baseline |
| E030 | `6` | target geometry/surface-control 全负；`box025_p1` fall/contact collapse；`bucket005_s2_p1` high-contact high-penetration；`box025_p2` clean guard contact regression |

关键防误选点：

- `E029_box025_p2_guard_posture_gate` 是 guard pass，但 E018b `box025_p2` 已是 strict baseline，因此不替换。
- `E030_bucket005_s2_p1_leg_guard_surface` contact `99.47%`，但 deep pen `94.31%`、max pen `5.69cm`，明确是 high-contact high-penetration 坏解。
- `E030_bucket007_p2_tinygeom_surface_gate` contact `57.71%` 且 deep pen `0%`，但仍有 retarget/geometry caveat 与 E030 failure label，不进入 positive pool。

## Data-quality caveats

E031 沿用 E027 多证据数据协议，不因为 E028-E030 算法负结果新增弃用 case。

| Case | Label | Decision |
|---|---|---|
| `desk021_p1` | `discard_from_success_denominator` | 唯一 success-denominator discard；P1 caveat 保留 |
| `box025_p1` | `retarget_questionable` | 保留 P0/P1 分母；标记 OmniRetarget/geometry caveat |
| `bucket007_p2` | `retarget_questionable` | 保留 P0/P1 分母；标记 OmniRetarget/geometry caveat |
| 其余 case | `usable_algorithmic_failure` 或 `usable_with_caveat` | 不弃用 |

## 可视化 / 指标一致性

E031 复用 E026-E030 已归档 online videos/keyframes，并在 `visual_metric_audit.md` 写入逐 case evidence path。实际观察结论：

- Conservative best rows 中只有 `box025_p2` 达到 strict；其他 case 虽 object tracking 好，但仍有 low contact、deep penetration 或 fall。
- E028/E030 的 low-contact rows 不能按 object tracking 误判为成功。
- E029/E030 的 high-contact high-penetration rows 必须优先按 penetration guard 拒绝。
- `desk021_p1` 的弃用来自 E027 raw / OmniRetarget / visual / metric 多证据，不来自单个 dynamic variant 的失败。

## Claims 验证

| Claim | 结果 | 说明 |
|---|---|---|
| C1: 复现并扩展 E026 full-eval schema | 通过 | 输出 coverage、method metrics、selection、P0/P1 summary、aggregate，覆盖 E018b/E022-E025/E028-E030/E081/Holosoma |
| C2: best-positive selection 保守 | 通过 | selection 只从 E018b/E022-E025 选；E028-E030 全部进 rejected diagnostics |
| C3: E027 data-quality audit 进入分母说明 | 通过 | `data_quality_caveats.csv/md` 标出唯一 discard `desk021_p1` 与 `box025_p1/bucket007_p2` caveat |
| C4: 不虚构 post-E026 strict 改善 | 通过 | P0 strict `1/9`、P1 strict `1/13`，明确无新增 strict gain |
| C5: visual audit 防止 metric-only positive | 通过 | `visual_metric_audit.md` 列出 high-penetration、contact-collapse、fall、guard-only rejected evidence |

## 结论

E031 完成 post-E026 阶段的保守 full-eval 账本。E027-E030 提供了重要诊断和拒绝证据，但没有任何新 variant 可以合法进入 best-positive pool。当前可报告 best dynamic 仍是 E018b/E022-E025 conservative assembly：object tracking 强、fall 比 E018b 少，但 strict success 仍停在 P0 `1/9`、P1 `1/13`。

下一机制线不应做 selection trick，也不应继续 case-specific XML shrink、普通 contact gain sweep 或单纯提高 barrier scale。应转入 runtime surface target / CEM candidate-level rejection-projection：显式定义 object surface/contact normal target，并在 candidate 层拒绝或投影 high-penetration、leg/body artifact、fall candidate。
