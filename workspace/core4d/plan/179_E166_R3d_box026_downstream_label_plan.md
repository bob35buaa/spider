# E166-R3d 计划：补 box026 downstream staggered 标签

日期：2026-06-18
前置结果：`log/216_E166_R3c_two_stage_redline_results.md`
状态：**补验证标签，不启动 E166 CEM/RL GPU 优化**。

## Context

E166 Phase0 红线审计已经计算 8/8 Tier-1 handoff 指标，但 downstream staggered 成功率只有 7/8 标签；缺失项是 `box026_139_p1`。当前 two-stage redline 在 7 标签上拟合 PR=1.0/1.0，但 leave-one 仅 3/7，说明阈值贴合单个 case，不能作为放行 E166 A/B/C GPU 的硬门。

本轮只补 `box026_139_p1` 的 SUGAR refiner 训练与 staggered eval 标签，用于重新运行 E166-R3c/R3d 审计。它不是 E166 的算法消融，不改变 plan 177 的硬门：若补标签后稳健性仍不足，继续不启动 E166 CEM/RL。

## Claims

| # | Claim | 最低证据 |
|---|---|---|
| C-R3d-1 | `box026_139_p1` 可以用 E163 spider_e163 同配方生成 downstream staggered 标签 | local SUGAR train 产出 latest checkpoint；eval 写出 `success_vs_phase.csv` |
| C-R3d-2 | 第 8 个标签能检验 two-stage redline 是否只是 n=7 过拟合 | 更新 E166 runner 指向 box026 eval；重新输出 8-label PR 与 leave-one |
| C-R3d-3 | 仍不绕过 E166 Phase0 硬门 | 只有 8-label 稳健性通过时，才另开 E166 GPU launch；否则记录 blocked/replan |

## 改动

1. 在 SUGAR repo 增加固定入口 `scripts/sugar_rl/launch_core4d_e163_box026_refiner.sh`：
   - `preflight` 检查 data/USD/env。
   - `train` 使用 E163 4case 相同配方：`NUM_ENVS=4096`、`MAX_ITERATIONS=6000`。
   - `eval` 运行 latest checkpoint 的 full eval 与 `eval_staggered_phase_mw30_latest_checkpoint`，写 `analysis/success_vs_phase.csv`。
   - `status` 汇报 checkpoint、eval、success_vs_phase 是否存在。
2. 在 spider 增加 canonical launch wrapper：
   - `workspace/core4d/scripts/launch/active/run_E166_R3d_box026_sugar_local.sh`
   - 该 wrapper 只转发 SUGAR 单 case 脚本，便于 tracker/log 复现。
3. 训练/eval 完成后，更新 `eval_E166_redline_predictive.py` 中 `box026_139_p1` 的 `eval_rel`，重新运行：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
```

## 成功标准

| 阶段 | 标准 |
|---|---|
| preflight | box026 SUGAR data 三个文件和 `Box026/obj_aligned.usd` 存在 |
| train | `model_*.pt` checkpoint 至少一个，latest checkpoint 可被 eval 脚本识别 |
| eval | `analysis/success_vs_phase.csv` 存在，至少包含 staggered phase rows |
| E166-R3d | `redline_case_metrics.tsv` 中 `box026_139_p1` 的 `staggered_success` 非空；summary 明确 8-label gate 是否通过 |

## 资源策略

当前只有一个缺失 case，使用本地 GPU0 即可；不占用远程 2-GPU，避免为单 case 增加同步/回收复杂度。若本地 GPU 被占满，允许改用远程 GPU1，但必须另写 pull 脚本并不 kill 现有任务。

## 决策

在 E166-R3d 结果出来前，仍不运行 `run_E166_remote.sh`、不启动 9 条 CEM 或 12 条 SUGAR RL 消融。
