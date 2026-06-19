# E166 remaining4 A/A_B2 + SUGAR results

日期：2026-06-19

## Scope

本轮完成 clean8 中排除已有三 case 和 `box026_139_p1` 后的 remaining4：

- `box023_person2`
- `box021_029_p2`
- `box021_035_p1`
- `box004_083_p1`

重定向版本：

- `A`: CEM foot constraints。
- `A_B2_postSmooth`: 先跑 `A`，再用 B2 同款 CPU 后平滑。

SUGAR-RL 只跑 `A_B2_postSmooth`。`AplusB`/`A+B1`/`A_B1` 未进入 remaining4 SUGAR 路径，`box026` 未进入队列。

## Artifacts

SPIDER/CEM:

- Plan: `workspace/core4d/plan/182_E166_remaining4_A_A_B2_queue_plan.md`
- Manifest: `workspace/core4d/scripts/experiments/E166/remaining4_variants.tsv`
- Results root: `workspace/core4d/results/E166/foot_smooth_retarget/`
- Eval xlsx: `workspace/core4d/results/E166/foot_smooth_retarget/eval/remaining4/E166_foot_smooth_vs_E163_three_case_eval.xlsx`
- RL export: `workspace/core4d/results/E166/foot_smooth_retarget/rl_export_remaining4/`

SUGAR:

- Data root: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/data/Core4D_E166_A_B2_postSmooth_*`
- Output root: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e166_remaining4_A_B2_refiner_rl`
- Summary: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e166_remaining4_A_B2_refiner_rl/comparison/e166_remaining4_A_B2_staggered_summary.md`

## Completion Audit

| item | result |
|---|---:|
| remaining4 manifest rows | 12 |
| manifest arms | `baseline`, `A`, `A_B2_postSmooth` |
| manifest `box026` / old arm rows | 0 |
| A CEM full npz/mp4/outdir | 4/4 |
| A_B2 postprocess npz + smooth report | 4/4 |
| eval rows | 12/12, missing 0 |
| xlsx sheets | `主表`, `逐case`, `SmoothFoot健康度`, `E166产物检查`, `说明`, `原始metrics` |
| RL export | `RL_EXPORT_READY=4/4` |
| SUGAR data folders | 4/4 |
| SUGAR final ckpt + success CSV | 4/4 |
| active E166/AplusB/box026 processes | 0 |

Final artifact check:

- `model_5999.pt`: 4/4, each 14,957,429 bytes.
- `success_vs_phase.csv`: 4/4, each 65 lines.
- `failed_windows.csv`: present for `box023_person2` and `box021_035_p1`, absent for the two cases without window-level failure CSV.

## SUGAR Results

| case | baseline | A_B2 success | delta | phases | main failure |
|---|---:|---:|---:|---:|---|
| `box023_person2` | 0.000 | 0.000 | +0.000 | 64 | `ee_body_pos` 55/64 |
| `box021_029_p2` | 0.672 | 1.000 | +0.328 | 64 | none |
| `box021_035_p1` | 0.672 | 0.000 | -0.672 | 64 | `ee_body_pos` 43/64 |
| `box004_083_p1` | 0.000 | 0.000 | +0.000 | 64 | no completed rollout; no failed_windows CSV |

## Interpretation

流程目标已完成：remaining4 的 `A` 和 `A_B2_postSmooth` 都已生成，`A_B2_postSmooth` 已导出并完成 SUGAR-RL + staggered eval，远程结果已回收到本地。

效果上不能把 `A_B2_postSmooth` 当作 clean8 通用提升：`box021_029_p2` 从 baseline `0.672` 提升到 `1.000`，但 `box021_035_p1` 从 `0.672` 掉到 `0.000`，主要失败仍是 `ee_body_pos`。`box023_person2` 和 `box004_083_p1` 保持 `0.000`，说明前者 pathology/接触自碰撞类问题、后者该 clean8 子例未被这条后平滑路径解决。

## Claims

| claim | conclusion | evidence |
|---|---|---|
| C-rem4-A | pass | full A CEM 4/4 root npz/mp4/outdir |
| C-rem4-A_B2 | pass | postprocess npz + smooth report 4/4 |
| C-rem4-RL | pass for pipeline, mixed for performance | SUGAR final/eval 4/4; success only 1/4 |
| C-queue | pass | watcher waited for prior three case A/A_B2, then started remaining4; final tmux sessions exited |

## Next

Do not resume `AplusB` by default. For downstream performance, prioritize diagnosing why `box021_035_p1` regressed under `A_B2_postSmooth`, especially the `ee_body_pos` failed windows, before treating A_B2 as a general replacement.
