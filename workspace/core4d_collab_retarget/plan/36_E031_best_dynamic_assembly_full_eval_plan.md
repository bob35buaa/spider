# E031 实验计划：Best Dynamic Assembly + Full Evaluation

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e031-full-eval`

## Context

E027-E030 已按 Post-E026 计划完成：

- E027 完成 13case data / retarget quality audit；唯一 `discard_from_success_denominator=True` 的 case 是 `desk021_p1`。
- E028 hard no-penetration 是负结果：部分 case 能降低 penetration，但通常伴随 contact collapse；不应作为 best-positive。
- E029 posture/stability 是局部正结果：`bucket001_p1` no-fall/stability 改善，但 contact 仍 `0%`；`box025_p2` guard pass，可作为 non-regression evidence。
- E030 lower-body geometry / surface-control 是负结果：target geometry success `0/2`、diagnostic surface signal `0/2`、clean guard pass `0/1`，但进一步确认不能继续 case-specific XML shrink / 普通 contact gain sweep。

因此 E031 的角色不是新增 runtime surface target 机制，而是按 `plan/31_post_E026_next_stage_optimization_plan.md` 组装 E026-style full eval 账本：把 E027-E030 的 evidence 纳入统一候选池/拒绝理由/caveat table，形成下一版可报告的 P0/P1 结论，并明确下一机制实验应转 runtime surface target / CEM candidate-level rejection-projection。

## Claims

| Claim | 最低证据 |
|---|---|
| C1: E031 能复现并扩展 E026 full eval schema | 输出 `method_case_metrics.csv`、`best_dynamic_selection.csv`、`summary_9case.md`、`summary_13case.md`、coverage/aggregate，且覆盖 E018b/E022-E025/E028-E030/E081/Holosoma |
| C2: best-positive selection 必须保守，不把 E028/E030 负结果或 high-contact high-penetration 坏解选入 | E031 selection 中 positive pool 只包含 E018b/E022-E025；E028-E030 进入 diagnostic/rejected table，标出 reject reason |
| C3: E027 data-quality audit 必须进入 P0/P1 分母说明 | 输出 `data_quality_caveats.csv/md`，明确 `desk021_p1` 是唯一 success-denominator discard，`box025_p1/bucket007_p2` 只是 retarget caveat |
| C4: E031 不应虚构 post-E026 strict 改善 | 若 strict 仍为 P0 `1/9`、P1 `1/13`，log 需明确写为“账本确认无新增 strict gain”，并把下一步机制线转 surface target / candidate rejection |
| C5: visual audit 必须防止 metric-only positive | 复用 E026/E028-E030 keyframes 与 logs，逐 case 标注 negative/guard evidence，不允许 high-contact high-penetration 被误判为成功 |

## Scope

### Candidate groups

| Group | 是否进入 best-positive pool | 用途 |
|---|---|---|
| Holosoma / OmniRetarget kinematic | 否 | kinematic reference / smoothness / contact28 对照 |
| E081 legacy + E026 E081 full rerun | 否 | scene-actuator baseline 与 paper-metrics rerun 对照 |
| E018b canonical | 是 | object-side canonical support baseline |
| E022-E025 validated variants | 是 | E026 原 best dynamic pool |
| E027 | 否 | data quality / timing audit，无 full rollout candidate |
| E028 | 否 | diagnostic / rejected candidates；hard barrier/contact collapse/penetration 反例 |
| E029 | 仅 diagnostic；`box025_p2` guard 不替换 strict baseline | stability / guard evidence |
| E030 | 否 | diagnostic / rejected candidates；geometry/surface-control 负结果 |

### P0 / P1 denominator

- P0 仍使用 E026 9case：`box023_p1`、`box023_p2`、`box025_p1`、`box025_p2`、`bucket001_p2`、`bucket005_s2_p1`、`bucket005_s2_p2`、`bucket007_p1`、`bucket007_p2`。
- P0 继续排除 `box021_p1`、`box021_p2`、`bucket001_p1`、`desk021_p1`。
- P1 仍保留 13case 全表。
- `desk021_p1` 是唯一 `discard_from_success_denominator=True` case，仍必须在 P1 / caveat table 中报告。
- E028-E030 不新增 discard case；算法负结果不能反推数据差。

## 计划改动

### 1. E031 full-eval script

新增：

| 文件 | 作用 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py` | 派生 E026 full eval，扩展 source list、diagnostic/rejected rows、E027 caveat table |

实现策略：

- 复制/派生 `eval_E026_full_eval.py`，避免破坏 E026 已归档结果。
- 在 `normalize_row()` 增加 E028/E029/E030 的 strict/success key：
  - `E028_strict_success`
  - `E029_success` / `E029_guard_pass` / `E029_p1_strict_target`
  - `E030_success` / `E030_target_geometry_success` / `E030_clean_guard_pass`
- `best_positive_methods` 固定为 `spider_E018b`、`spider_E022`、`spider_E023`、`spider_E024`、`spider_E025`。
- `diagnostic_methods` 增加 `spider_E028`、`spider_E029`、`spider_E030`，只输出到 all metrics / rejected table，不参与 best-positive selection。
- 读取 E027 `case_quality_audit.csv`，将 `quality_label`、`discard_from_p0`、`discard_from_success_denominator`、evidence/rationale 合并到 selection 和 summary。

### 2. Outputs

输出目录：

```text
workspace/core4d_collab_retarget/results/E031_full_eval/
```

核心产物：

| 文件 | 内容 |
|---|---|
| `INDEX.md` | E031 输出索引 |
| `coverage.json` | 各 source CSV 覆盖、missing case、已知缺失原因 |
| `method_case_metrics.csv` | 所有 normalized method / case rows，包括 E028-E030 diagnostic rows |
| `best_dynamic_selection.csv` | 保守 best-positive selection，只从 E018b/E022-E025 中选 |
| `rejected_diagnostic_candidates.csv` | E028-E030 负结果/guard regression/high-penetration 的 reject reason |
| `data_quality_caveats.csv` | E027 quality labels / discard flags / evidence |
| `summary_9case.md` | P0 9case summary |
| `summary_13case.md` | P1 13case summary |
| `visual_metric_audit.md` | 视觉/指标一致性、E028-E030 负结果防误选说明 |
| `aggregate_summary.json` | P0/P1 strict/contact/deep/object/fall aggregate |

### 3. Log / tracker

新增：

| 文件 | 内容 |
|---|---|
| `workspace/core4d_collab_retarget/log/31_E031_full_eval_results.md` | E031 执行命令、P0/P1 指标、selection/rejected/caveat、下一步机制线 |

更新：

- `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md`
- `workspace/core4d_collab_retarget/progress.md`

## 成功标准

| 指标 | 最低目标 |
|---|---|
| source coverage | E018b/E022-E025/E028-E030/E026_E081_full/Holosoma rows 均被读取；missing reason 显式 |
| selection determinism | 每个 case 的 selected variant、score、candidate list 可复现 |
| P0 strict | 如无新 positive candidate，预期仍为 `1/9`；必须如实记录 |
| P1 strict | 预期仍为 `1/13`；`desk021_p1` 保留 caveat/discard 说明 |
| rejection audit | E028-E030 的所有 variants 有 reject reason，不被误选 |
| visual audit | 对 high-contact high-penetration / contact collapse / fall / guard regression 有具体说明 |
| git | plan、script、results、log、tracker/progress 分阶段 commit 并 push |

## 停止条件

- 如果 E031 发现 E028/E029/E030 某个 diagnostic row 按 deterministic score 会替换 E026 best-positive，但 violates strict/guard rules，则必须保守拒绝并记录，不修改 scoring 让它“看起来变好”。
- 如果 E031 aggregate 严格保持 P0 `1/9`，不再做 E031b selection trick；下一实验转 runtime surface target / candidate-level rejection-projection。
- 如果 E027 caveat table 与 E031 P0/P1 分母冲突，以 E027 多证据协议为准，并在 log 写明。

## 执行命令

```bash
.venv/bin/python -m py_compile \
  workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py

.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E031_full_eval.py --all
```

E031 是离线评估/账本组装，不需要远程 GPU；但 log 需要明确说明本实验不触发 remote wrapper 条件。

## Git strategy

1. Plan commit: `plan(core4d_collab): start E031 full eval assembly`
2. Implementation/results commit: `log(core4d_collab): record E031 full eval assembly`
