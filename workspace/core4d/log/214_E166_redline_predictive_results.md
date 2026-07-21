# E166-R3 Phase0 离线红线预测力结果

日期：2026-06-18
计划：`workspace/core4d/plan/177_E166_foot_smooth_retarget_plan.md`
产物：`workspace/core4d/results/E166/redline_predictive/`
脚本：
- `workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py`
- `workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh`

## 结论

**C-R3 未通过，按 plan 风险预案暂停 Phase 1/2/3 GPU 实验。**

离线 Tier-1 指标已对 8/8 handoff 计算完成；但 downstream staggered 标签当前只有 7/8（`box026_139_p1` 尚无训练结果），因此相关性和 PR 只在 7 个有标签 case 上计算。

关键结果：
- `jerk/acc` 相对 staggered 的最高 `|Spearman rho|` 只有 `0.342`（trackbody acc）/ `0.505`（ankle acc），未达到计划门槛 `>=0.6`。
- Isaac contact 对照几乎不预测 downstream：`isaac_both_contact_frac` 与 `isaac_contact_iou` 的 `|rho|=0.027`。
- Tier-1 多维 OR 红线在 7 个有标签 case 上可标出 `staggered<0.10` 的 4 个失败 case，`precision=1.0, recall=1.0`，但阈值来自 n=7 标定，且 `foot_ground_dev` 当前接近常数，不能替代 C-R3 通过。

最重要的解释：`box023_person2` 是“平滑但失败”的例外（接触/自碰撞病），它把纯平滑指标与 downstream success 的 rank correlation 拉低。去掉 box023 后，敏感性检查显示 `trackbody_jerk_p95 rho=-0.725`、`ankle_acc_max rho=-0.783`、`obj_speed_max rho=-0.667`，但这属于事后分层，不能用来让原 C-R3 过门。

## Claim 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C-R3：jerk/acc 峰值对 downstream 的预测力强于接触且 `|rho|>=0.6` | ❌ 未通过 | 平滑/速度指标确实强于接触，但最高 `|rho|=0.505`，低于 `0.6` |
| C-R3b：足-滑/足-地不一致有预测力 | ⚠️ 未充分支持 | `foot_slip rho=-0.198`；`foot_ground_dev rho=0.036`，当前 foot-ground 定义太弱 |
| C-C：Tier-1 多维红线能区分 `staggered<0.10` | ⚠️ 形式通过但需谨慎 | 标定集 n=7 上 `precision=1.0, recall=1.0`，但存在过拟合风险，不能单独作为开 GPU 依据 |

## 主要表格

详见 `redline_summary.md`。核心数值：

| case | staggered | accMax | jerkP95 | footSlip | objV | Isaac both | Isaac IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| box021_029_p2 | 0.672 | 55.8 | 1210 | 1.016 | 1.41 | 0.051 | 0.066 |
| box004_083_p2 | 0.484 | 60.4 | 907 | 0.713 | 1.41 | 0.000 | 0.000 |
| box021_035_p1 | 0.234 | 272.5 | 3241 | 1.671 | 4.28 | 0.164 | 0.172 |
| box004_082_p1 | 0.094 | 361.6 | 2956 | 1.371 | 5.70 | 0.017 | 0.029 |
| box021_035_p2 | 0.016 | 151.3 | 3217 | 1.752 | 4.35 | 0.276 | 0.339 |
| box004_083_p1 | 0.016 | 141.9 | 3773 | 1.033 | 2.40 | 0.000 | 0.000 |
| box023_person2 | 0.000 | 93.4 | 1069 | 0.961 | 1.47 | 0.023 | 0.046 |
| box026_139_p1 | n/a | 173.8 | 2787 | 1.548 | 3.93 | 0.216 | 0.294 |

## 运行命令与验证

```bash
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py
bash -n workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
git diff --check -- workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh workspace/core4d/progress.md
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
```

生成文件：
- `redline_case_metrics.tsv`
- `redline_correlations.tsv`
- `redline_threshold_grid.tsv`
- `redline_thresholds.json`
- `redline_summary.md`
- `redline_scatter.png`
- `tier1_pr.png`

## 下一步

不进入 Phase 1/2/3 GPU。先重审假设：
1. 把 E166 的前置红线改成 **两阶段/分病因**：先用 contact/self-collision/box023 类指标分出“非平滑主因”，再在剩余可比 case 上验证脚/平滑。
2. 改进 foot-ground/foot-slip 指标定义；当前 foot-ground 几乎常数，不能证明 C-R3b。
3. 如仍要推进 A/B 消融，需要把它从“C-R3 已通过后的确认实验”降级为“探索性验证”，并重新写 plan 成功标准。
