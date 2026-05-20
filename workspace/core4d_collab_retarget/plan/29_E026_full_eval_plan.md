# E026 实验计划：全面评估 OmniRetarget / E081 / E018b / E022-E025

日期：2026-05-20

## Context

`task_full_eval.md` 要求基于 E019 unified eval 框架，全面评估当前工作的 kinematic 与 dynamic 结果，并输出两版 case 集合：

- P0 9case：排除 `desk021_p1`、`box021_p1`、`box021_p2`、`bucket001_p1`
- P1 13case full：完整并集，缺失/数据问题必须显式标注

已有证据：

- E019 已实现 SPIDER T4 / OmniRetarget T2 / DynaRetarget T5 / CORE4D 自定义指标框架。
- E018b 有 13/13 dynamic canonical support proxy 结果。
- E022-E025 有 post-E020 optimization variants，可按 case 挑最好结果。
- E081 旧 baseline 目前只有 `box025_p2` 与 `box023_p2` 两个 case，且 comparison schema 未接入 E019 `paper_metrics`。
- OmniRetarget / holosoma kinematic 代码映射为 12/13，`desk021_p1` SOCP infeasible；当前本机 holosoma 数据在 `/home/ubuntu/Workspace/holosoma`，adapter 还只写了旧 `/mnt/...` 路径，需要补 fallback 后重评。

## Claims

| Claim | 最低证据 |
|---|---|
| C1 kinematic 数据来源可追溯 | 文档记录 adapter、holosoma batch、trimmed NPZ、companion object NPZ 路径；本机 `eval_holosoma_kinematic.py --all` 产出 comparison |
| C2 28cm contact preservation 口径被验证 | 写明 28cm 来自 holosoma v1 eval 代码而非论文正文；输出多阈值 sweep（5/10/15/20/28/35/50cm）证明阈值敏感性 |
| C3 P0 9case 表完整 | OmniRetarget、E018b、E022-E025 best-by-case 至少在 9case 表中有 coverage；E081 若仍只 N=2，表中明确 coverage fail，不能伪装完整 |
| C4 P1 13case 表完整审计 | 13case 表包含每个方法的 present/missing、metric means、缺失原因 |
| C5 E022-E025 best-by-case 选择可复现 | 脚本输出 `best_dynamic_selection.csv`，列出每个 case 选中的 exp/variant 和评分依据 |
| C6 视觉与指标一致性检查完成 | 输出 `visual_metric_audit.md`，至少覆盖 E018b 13case online video 结论与 E022-E025 关键 case 视频/日志，对照 contact/fall/penetration 指标 |
| C7 E081 full rerun 状态诚实 | 若 13case CEM 未实际完成，必须在 E026 coverage 中标为 blocker/weak coverage，并提供可运行的 rerun plan/script，而不是把 N=2 当 full baseline |

## 改动

### 1. 修 holosoma 本机路径 fallback

**文件**：`workspace/core4d_collab_retarget/scripts/eval/adapters/kinematic_to_common.py`

把 `HOLOSOMA_RESULT_DIRS` / `HOLOSOMA_DEMO_DIRS` 扩展为 `/mnt/...` 与 `/home/ubuntu/Workspace/holosoma/...` 双路径，保证当前机器可重建 `results/holosoma_v2_kinematic/`。

### 2. E026 full eval 汇总脚本

**文件**：`workspace/core4d_collab_retarget/scripts/eval/eval_E026_full_eval.py`

职责：

- 读取 E018b、E022-E025、E081、holosoma comparison.csv。
- 标准化 case 名称到 13case short name。
- 生成 P0 9case 与 P1 13case 两版 summary。
- 生成 OmniRetarget 28cm 多阈值 sweep。
- 按 case 选择 E022-E025 best dynamic variant，并保留 E018b baseline 对比。
- 生成 coverage / missing reason / visual audit。

### 3. 输出目录

**路径**：`workspace/core4d_collab_retarget/results/E026_full_eval/`

预期产物：

| 文件 | 内容 |
|---|---|
| `coverage.json` | 各方法 case coverage、缺失原因 |
| `method_case_metrics.csv` | 标准化 per-case 指标长表 |
| `summary_9case.md` | P0 9case 方法均值与结论 |
| `summary_13case.md` | P1 13case 方法均值与缺失说明 |
| `best_dynamic_selection.csv` | E018b/E022-E025 by-case best 选择 |
| `omni_threshold_sweep.csv` / `.md` | 5/10/15/20/28/35/50cm preservation sweep |
| `visual_metric_audit.md` | 视频/视觉与指标一致性检查 |

## 需要修改的文件

| # | 文件 | 改动 |
|---|---|---|
| 1 | `plan/29_E026_full_eval_plan.md` | 本计划 |
| 2 | `progress.md` | 记录 E026 执行状态 |
| 3 | `scripts/eval/adapters/kinematic_to_common.py` | holosoma fallback 路径 |
| 4 | `scripts/eval/eval_E026_full_eval.py` | E026 汇总评估 |
| 5 | `log/26_E026_full_eval_results.md` | 最终结果日志 |
| 6 | `EXPERIMENT_TRACKER.md` | 完成后登记 E026 |

## 评估命令

```bash
# 1. 重建 holosoma kinematic comparison
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py --all

# 2. 可选：重评已有 dynamic results，确保 paper fields 最新
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_E018b.py --all

# 3. 生成 E026 全面评估
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E026_full_eval.py --all
```

E081 13case CEM 由于当前只 N=2，先在 coverage 中显式标注；如资源允许再补 E081 extended rerun，不让该缺口阻塞其他方法的 E026 P0/P1 汇总。

## 成功标准

| 指标 | 目标 |
|---|---|
| P0 summary | 9case 表生成，缺失项仅 E081 coverage 可显式 fail |
| P1 summary | 13case 表生成，`desk021_p1` OmniRetarget 缺失原因明确 |
| Omni threshold sweep | 12 个已成功 kin case上输出 7 个半径阈值 |
| Best selector | E022-E025 variants 可复现排序，每个优化过的 case 有选择依据 |
| Visual audit | 至少覆盖 fall/contact/penetration 三类指标与视频一致性 |
| Log/tracker | 写入 E026 log、更新 progress 和 tracker |
